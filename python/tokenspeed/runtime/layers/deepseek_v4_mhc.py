# Copyright (c) 2026 LightSeek Foundation
#
# Portions copyright the vLLM project contributors under Apache-2.0.

from __future__ import annotations

from functools import cache

import torch
import triton
import triton.language as tl

from tokenspeed.runtime.utils import ceil_div

try:
    from tokenspeed_kernel.thirdparty import deep_gemm
except Exception:
    deep_gemm = None  # type: ignore[assignment]


@cache
def _compute_num_split(block_k: int, k: int | None, grid_size: int) -> int:
    device_props = torch.cuda.get_device_properties(0)
    split_k = device_props.multi_processor_count // grid_size
    if k is not None:
        num_block_k = ceil_div(k, block_k)
        split_k = min(split_k, num_block_k // 4)
    return max(split_k, 1)


@triton.jit
def _hc_prenorm_gemm_triton_kernel(
    a_ptr,
    w_ptr,
    mul_ptr,
    sqrsum_ptr,
    num_tokens,
    stride_at,
    stride_ak,
    stride_wn,
    stride_wk,
    k_size: tl.constexpr,
    n_size: tl.constexpr,
    n_splits: tl.constexpr,
    block_t: tl.constexpr,
    block_k: tl.constexpr,
    block_n: tl.constexpr,
):
    """Split-K ``A @ W.T`` fused with A's per-row sum of squares.

    Emits the same two partial buffers ``tf32_hc_prenorm_gemm`` does:
    ``mul[split, token, n]`` and ``sqrsum[split, token]``, both contiguous and
    both reduced over ``split`` by the mix kernel. Fusing the two lets the
    residual be read once; it is a [T, 16384] x [16384, 24] contraction for
    V4-Flash, so the read of A dominates.
    """
    pid_t = tl.program_id(0)
    split_id = tl.program_id(1)

    offs_t = pid_t * block_t + tl.arange(0, block_t)
    offs_n = tl.arange(0, block_n)
    mask_t = offs_t < num_tokens
    mask_n = offs_n < n_size

    k_per_split = tl.cdiv(k_size, n_splits)
    k_begin = split_id * k_per_split
    k_end = tl.minimum(k_begin + k_per_split, k_size)

    acc = tl.zeros((block_t, block_n), dtype=tl.float32)
    sqr = tl.zeros((block_t,), dtype=tl.float32)

    for k0 in range(k_begin, k_end, block_k):
        offs_k = k0 + tl.arange(0, block_k)
        mask_k = offs_k < k_end
        a = tl.load(
            a_ptr + offs_t[:, None] * stride_at + offs_k[None, :] * stride_ak,
            mask=mask_t[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        w = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.dot(a, tl.trans(w))
        sqr += tl.sum(a * a, axis=1)

    mul_offset = (
        split_id * num_tokens * n_size + offs_t[:, None] * n_size + offs_n[None, :]
    )
    tl.store(mul_ptr + mul_offset, acc, mask=mask_t[:, None] & mask_n[None, :])
    tl.store(sqrsum_ptr + split_id * num_tokens + offs_t, sqr, mask=mask_t)


def hc_prenorm_gemm_triton(
    a: torch.Tensor,
    w: torch.Tensor,
    mul_out: torch.Tensor,
    sqrsum_out: torch.Tensor,
    n_splits: int,
) -> None:
    """Portable ``tf32_hc_prenorm_gemm`` replacement.

    Args:
        a: ``[num_tokens, k]`` activations, any float dtype.
        w: ``[n, k]`` weights, contracted along ``k``.
        mul_out: ``[n_splits, num_tokens, n]`` float32, written in place.
        sqrsum_out: ``[n_splits, num_tokens]`` float32, written in place.
        n_splits: number of K-splits; each output slice holds one partial.
    """
    num_tokens, k_size = a.shape
    n_size = w.shape[0]
    block_t = 64
    block_k = 64
    block_n = max(16, triton.next_power_of_2(n_size))
    grid = (ceil_div(num_tokens, block_t), n_splits)
    _hc_prenorm_gemm_triton_kernel[grid](
        a,
        w,
        mul_out,
        sqrsum_out,
        num_tokens,
        a.stride(0),
        a.stride(1),
        w.stride(0),
        w.stride(1),
        k_size=k_size,
        n_size=n_size,
        n_splits=n_splits,
        block_t=block_t,
        block_k=block_k,
        block_n=block_n,
        num_warps=4,
    )


@triton.jit
def _reduce_split_partials(
    gemm_out_mul,
    token_id,
    first_mix_id: tl.constexpr,
    width,
    num_tokens,
    hc_mult3: tl.constexpr,
    n_splits: tl.constexpr,
    block_split: tl.constexpr,
    block_w: tl.constexpr,
):
    """Sum ``width`` consecutive mix partials across all K-splits.

    The partials for one token are ``n_splits`` rows of ``hc_mult3`` floats
    strided by ``num_tokens * hc_mult3``, so the whole reduction is one 2D tile
    load. Walking it as scalars instead costs ``n_splits * width`` dependent
    loads per token, which is what dominated this kernel at decode widths.
    """
    splits = tl.arange(0, block_split)
    cols = tl.arange(0, block_w)
    offsets = (
        splits[:, None] * (num_tokens * hc_mult3)
        + token_id * hc_mult3
        + first_mix_id
        + cols[None, :]
    )
    live = (splits < n_splits)[:, None] & (cols < width)[None, :]
    tile = tl.load(gemm_out_mul + offsets, mask=live, other=0.0)
    return tl.sum(tile, axis=0)


@triton.jit
def _mhc_pre_mix_triton_kernel(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    pre_mix,
    post_mix,
    comb_mix,
    hidden_size: tl.constexpr,
    rms_eps: tl.constexpr,
    hc_eps: tl.constexpr,
    sinkhorn_iters: tl.constexpr,
    n_splits: tl.constexpr,
    hc_mult: tl.constexpr,
    hc_mult2: tl.constexpr,
    hc_mult3: tl.constexpr,
    block_comb: tl.constexpr,
    block_gate: tl.constexpr,
    block_split: tl.constexpr,
    num_tokens,
):
    token_id = tl.program_id(0)

    splits = tl.arange(0, block_split)
    rms_sum = tl.sum(
        tl.load(
            gemm_out_sqrsum + splits * num_tokens + token_id,
            mask=splits < n_splits,
            other=0.0,
        ),
        axis=0,
    )
    rms = tl.rsqrt(rms_sum / (hc_mult * hidden_size) + rms_eps)

    # The pre and post gates share a tile: both are sigmoids of the same
    # affine form, differing only in which scale they use and how the result
    # is finished.
    gate_offsets = tl.arange(0, block_gate)
    gate_active = gate_offsets < hc_mult * 2
    is_pre = gate_offsets < hc_mult
    gate_mix = _reduce_split_partials(
        gemm_out_mul,
        token_id,
        0,
        hc_mult * 2,
        num_tokens,
        hc_mult3,
        n_splits,
        block_split,
        block_gate,
    )
    gate_scale = tl.where(is_pre, tl.load(hc_scale), tl.load(hc_scale + 1))
    gate = tl.sigmoid(
        gate_mix * rms * gate_scale
        + tl.load(hc_base + gate_offsets, mask=gate_active, other=0.0)
    )
    tl.store(
        pre_mix + token_id * hc_mult + gate_offsets,
        gate + hc_eps,
        mask=is_pre,
    )
    tl.store(
        post_mix + token_id * hc_mult + gate_offsets - hc_mult,
        gate * 2.0,
        mask=gate_active & (gate_offsets >= hc_mult),
    )

    comb_offsets = tl.arange(0, block_comb)
    comb_mask = comb_offsets < hc_mult2
    comb_scale = tl.load(hc_scale + 2)
    comb_mix_values = _reduce_split_partials(
        gemm_out_mul,
        token_id,
        hc_mult * 2,
        hc_mult2,
        num_tokens,
        hc_mult3,
        n_splits,
        block_split,
        block_comb,
    )
    comb_values = comb_mix_values * rms * comb_scale + tl.load(
        hc_base + hc_mult * 2 + comb_offsets, mask=comb_mask, other=0.0
    )
    rows = comb_offsets // hc_mult
    cols = comb_offsets - rows * hc_mult
    active = comb_mask

    for row_id in tl.static_range(0, hc_mult):
        row_values = tl.where((rows == row_id) & active, comb_values, -float("inf"))
        row_max = tl.max(row_values, axis=0)
        comb_values = tl.where(
            (rows == row_id) & active, tl.exp(comb_values - row_max), comb_values
        )
    for row_id in tl.static_range(0, hc_mult):
        row_sum = tl.sum(tl.where((rows == row_id) & active, comb_values, 0.0), axis=0)
        comb_values = tl.where(
            (rows == row_id) & active, comb_values / row_sum + hc_eps, comb_values
        )
    for col_id in tl.static_range(0, hc_mult):
        col_sum = tl.sum(tl.where((cols == col_id) & active, comb_values, 0.0), axis=0)
        comb_values = tl.where(
            (cols == col_id) & active,
            comb_values / (col_sum + hc_eps),
            comb_values,
        )

    for _ in tl.static_range(1, sinkhorn_iters):
        for row_id in tl.static_range(0, hc_mult):
            row_sum = tl.sum(
                tl.where((rows == row_id) & active, comb_values, 0.0), axis=0
            )
            comb_values = tl.where(
                (rows == row_id) & active,
                comb_values / (row_sum + hc_eps),
                comb_values,
            )
        for col_id in tl.static_range(0, hc_mult):
            col_sum = tl.sum(
                tl.where((cols == col_id) & active, comb_values, 0.0), axis=0
            )
            comb_values = tl.where(
                (cols == col_id) & active,
                comb_values / (col_sum + hc_eps),
                comb_values,
            )

    tl.store(
        comb_mix + token_id * hc_mult2 + comb_offsets,
        comb_values,
        mask=comb_mask,
    )


@triton.jit
def _mhc_pre_layer_triton_kernel(
    pre_mix,
    residual,
    layer_input,
    hidden_size: tl.constexpr,
    hc_mult: tl.constexpr,
    block_h: tl.constexpr,
):
    token_id = tl.program_id(0)
    hidden_block_id = tl.program_id(1)

    hidden_offsets = hidden_block_id * block_h + tl.arange(0, block_h)
    hidden_mask = hidden_offsets < hidden_size
    layer_acc = tl.zeros((block_h,), tl.float32)
    for hc_id in tl.static_range(0, hc_mult):
        pre = tl.load(pre_mix + token_id * hc_mult + hc_id).to(tl.float32)
        residual_offsets = (
            token_id * hc_mult * hidden_size + hc_id * hidden_size + hidden_offsets
        )
        residual_values = tl.load(
            residual + residual_offsets, mask=hidden_mask, other=0.0
        ).to(tl.float32)
        layer_acc += pre * residual_values
    tl.store(
        layer_input + token_id * hidden_size + hidden_offsets,
        layer_acc,
        mask=hidden_mask,
    )


@triton.jit
def _mhc_post_triton_kernel(
    comb,
    residual,
    post,
    hidden_states,
    out,
    hidden_size: tl.constexpr,
    hc_mult: tl.constexpr,
    block_h: tl.constexpr,
):
    token_id = tl.program_id(0)
    hidden_block_id = tl.program_id(1)
    hidden_offsets = hidden_block_id * block_h + tl.arange(0, block_h)
    hidden_mask = hidden_offsets < hidden_size
    hidden_values = tl.load(
        hidden_states + token_id * hidden_size + hidden_offsets,
        mask=hidden_mask,
        other=0.0,
    ).to(tl.float32)

    for out_hc in tl.static_range(0, hc_mult):
        acc = tl.load(post + token_id * hc_mult + out_hc).to(tl.float32) * hidden_values
        for in_hc in tl.static_range(0, hc_mult):
            comb_value = tl.load(
                comb + token_id * hc_mult * hc_mult + in_hc * hc_mult + out_hc
            ).to(tl.float32)
            residual_values = tl.load(
                residual
                + token_id * hc_mult * hidden_size
                + in_hc * hidden_size
                + hidden_offsets,
                mask=hidden_mask,
                other=0.0,
            ).to(tl.float32)
            acc += comb_value * residual_values
        tl.store(
            out
            + token_id * hc_mult * hidden_size
            + out_hc * hidden_size
            + hidden_offsets,
            acc,
            mask=hidden_mask,
        )


@triton.jit
def _mhc_post_hc4_triton_kernel(
    comb,
    residual,
    post,
    hidden_states,
    out,
    hidden_size: tl.constexpr,
    block_h: tl.constexpr,
):
    token_id = tl.program_id(0)
    hidden_block_id = tl.program_id(1)
    hidden_offsets = hidden_block_id * block_h + tl.arange(0, block_h)
    hidden_mask = hidden_offsets < hidden_size
    token_hidden_offset = token_id * hidden_size
    token_residual_offset = token_id * 4 * hidden_size

    hidden_values = tl.load(
        hidden_states + token_hidden_offset + hidden_offsets,
        mask=hidden_mask,
        other=0.0,
    ).to(tl.float32)

    post_base = token_id * 4
    acc0 = tl.load(post + post_base + 0).to(tl.float32) * hidden_values
    acc1 = tl.load(post + post_base + 1).to(tl.float32) * hidden_values
    acc2 = tl.load(post + post_base + 2).to(tl.float32) * hidden_values
    acc3 = tl.load(post + post_base + 3).to(tl.float32) * hidden_values

    comb_base = token_id * 16
    for in_hc in tl.static_range(0, 4):
        residual_values = tl.load(
            residual + token_residual_offset + in_hc * hidden_size + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        comb_row = comb_base + in_hc * 4
        acc0 += tl.load(comb + comb_row + 0).to(tl.float32) * residual_values
        acc1 += tl.load(comb + comb_row + 1).to(tl.float32) * residual_values
        acc2 += tl.load(comb + comb_row + 2).to(tl.float32) * residual_values
        acc3 += tl.load(comb + comb_row + 3).to(tl.float32) * residual_values

    tl.store(
        out + token_residual_offset + hidden_offsets,
        acc0,
        mask=hidden_mask,
    )
    tl.store(
        out + token_residual_offset + hidden_size + hidden_offsets,
        acc1,
        mask=hidden_mask,
    )
    tl.store(
        out + token_residual_offset + hidden_size * 2 + hidden_offsets,
        acc2,
        mask=hidden_mask,
    )
    tl.store(
        out + token_residual_offset + hidden_size * 3 + hidden_offsets,
        acc3,
        mask=hidden_mask,
    )


def mhc_fused_hc(
    x_prev: torch.Tensor,
    residual_prev: torch.Tensor,
    post_prev: torch.Tensor,
    comb_prev: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused post_mapping(prev) + pre_mapping(curr).

    Returns (residual_cur, layer_input, post_cur, comb_cur).
    """
    residual_cur = mhc_post(x_prev, residual_prev, post_prev, comb_prev)
    layer_input, post_cur, comb_cur = mhc_pre(
        residual_cur,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_eps,
        sinkhorn_iters,
    )
    return residual_cur, layer_input, post_cur, comb_cur


def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if residual.dtype != torch.bfloat16 or fn.dtype != torch.float32:
        raise RuntimeError("fast mHC requires bf16 residual and fp32 weights")
    if not residual.is_cuda:
        raise RuntimeError("fast mHC requires CUDA tensors")

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    hc_hidden_size = hc_mult * hidden_size
    outer_shape = residual.shape[:-2]
    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    if num_tokens == 0:
        return (
            residual.new_empty(*outer_shape, hidden_size),
            torch.empty(
                *outer_shape,
                hc_mult,
                1,
                dtype=torch.float32,
                device=residual.device,
            ),
            torch.empty(
                *outer_shape,
                hc_mult,
                hc_mult,
                dtype=torch.float32,
                device=residual.device,
            ),
        )

    block_k = 64
    block_m = 64
    n_splits = _compute_num_split(
        block_k, hc_hidden_size, ceil_div(num_tokens, block_m)
    )

    post_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    pre_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        num_tokens, hc_mult2, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )
    gemm_out_mul = torch.empty(
        n_splits, num_tokens, hc_mult3, dtype=torch.float32, device=residual.device
    )
    gemm_out_sqrsum = torch.empty(
        n_splits, num_tokens, dtype=torch.float32, device=residual.device
    )

    residual_2d = residual_flat.view(num_tokens, hc_hidden_size)
    if deep_gemm is None:
        hc_prenorm_gemm_triton(
            residual_2d,
            fn,
            gemm_out_mul,
            gemm_out_sqrsum,
            n_splits,
        )
    else:
        deep_gemm.tf32_hc_prenorm_gemm(
            residual_2d,
            fn,
            gemm_out_mul,
            gemm_out_sqrsum,
            n_splits,
        )
    block_h = 1024
    block_comb = triton.next_power_of_2(hc_mult2)
    block_gate = triton.next_power_of_2(hc_mult * 2)
    block_split = triton.next_power_of_2(n_splits)
    _mhc_pre_mix_triton_kernel[(num_tokens,)](
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        pre_mix,
        post_mix,
        comb_mix,
        hidden_size=hidden_size,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        n_splits=n_splits,
        hc_mult=hc_mult,
        hc_mult2=hc_mult2,
        hc_mult3=hc_mult3,
        block_comb=block_comb,
        block_gate=block_gate,
        block_split=block_split,
        num_tokens=num_tokens,
        num_warps=1,
    )
    _mhc_pre_layer_triton_kernel[(num_tokens, triton.cdiv(hidden_size, block_h))](
        pre_mix,
        residual_flat,
        layer_input,
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        block_h=block_h,
        num_warps=4,
    )

    return (
        layer_input.view(*outer_shape, hidden_size),
        post_mix.view(*outer_shape, hc_mult, 1),
        comb_mix.view(*outer_shape, hc_mult, hc_mult),
    )


def mhc_post(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    if not hidden_states.is_cuda:
        raise RuntimeError("fast mHC requires CUDA tensors")
    if residual.numel() == 0:
        return torch.empty_like(residual)
    out = torch.empty_like(residual)
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    residual_flat = residual.view(-1, hc_mult, hidden_size)
    hidden_states_flat = hidden_states.view(-1, hidden_size)
    post_flat = post.view(-1, hc_mult)
    comb_flat = comb.view(-1, hc_mult, hc_mult)
    num_tokens = residual_flat.shape[0]
    if hc_mult == 4:
        block_h = 256
        _mhc_post_hc4_triton_kernel[(num_tokens, triton.cdiv(hidden_size, block_h))](
            comb_flat,
            residual_flat,
            post_flat,
            hidden_states_flat,
            out,
            hidden_size=hidden_size,
            block_h=block_h,
            num_warps=4,
        )
        return out

    block_h = 1024
    _mhc_post_triton_kernel[(num_tokens, triton.cdiv(hidden_size, block_h))](
        comb_flat,
        residual_flat,
        post_flat,
        hidden_states_flat,
        out,
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        block_h=block_h,
        num_warps=4,
    )
    return out

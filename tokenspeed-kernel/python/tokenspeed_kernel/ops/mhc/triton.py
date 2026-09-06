# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

from functools import cache

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.mhc.triton_prefill import (
    _mhc_prefill_config_hc4,
    mhc_pre_mix_hc4,
    mhc_prefill_project_hc4,
)
from tokenspeed_kernel.platform import CapabilityRequirement, pdl_enabled
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


@cache
def _compute_num_split(
    device: torch.device, block_k: int, k: int | None, grid_size: int
) -> int:
    device_props = torch.cuda.get_device_properties(device)
    split_k = device_props.multi_processor_count // grid_size
    if k is not None:
        num_block_k = triton.cdiv(k, block_k)
        split_k = min(split_k, num_block_k // 4)
    return max(split_k, 1)


def _pre_reduce_apply_is_supported(pre_reduce_apply_impl, n_splits: int) -> bool:
    if pre_reduce_apply_impl is None:
        return False
    supported = getattr(pre_reduce_apply_impl, "supported_n_splits", None)
    return supported is None or n_splits in supported


def _pre_reduce_apply_fuses_norm(
    pre_reduce_apply_impl, use_pre_reduce_apply: bool, has_norm_weight: bool
) -> bool:
    return bool(
        use_pre_reduce_apply
        and has_norm_weight
        and getattr(pre_reduce_apply_impl, "supports_fused_norm", False)
    )


@triton.jit
def _mhc_prenorm_gemm_triton_kernel(
    x,
    fn,
    out_mul,
    out_sqrsum,
    num_tokens,
    K: tl.constexpr,
    N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    split_id = tl.program_id(0)
    token_block = tl.program_id(1)
    n_block = tl.program_id(2)
    offs_m = token_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    split_start = split_id * SPLIT_K
    dot_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    square_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k_start in range(0, SPLIT_K, BLOCK_K):
        offs_k = split_start + k_start + tl.arange(0, BLOCK_K)
        x_values = tl.load(
            x + offs_m[:, None] * K + offs_k[None, :],
            mask=(offs_m[:, None] < num_tokens) & (offs_k[None, :] < K),
            other=0.0,
        )
        fn_values = tl.load(
            fn + offs_k[:, None] + offs_n[None, :] * K,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        dot_acc = tl.dot(
            x_values.to(tl.float32),
            fn_values,
            dot_acc,
            input_precision="ieee",
        )
        x_fp32 = x_values.to(tl.float32)
        square_acc += tl.sum(x_fp32 * x_fp32, axis=1)

    tl.store(
        out_mul + split_id * num_tokens * N + offs_m[:, None] * N + offs_n[None, :],
        dot_acc,
        mask=(offs_m[:, None] < num_tokens) & (offs_n[None, :] < N),
    )
    tl.store(
        out_sqrsum + split_id * num_tokens + offs_m,
        square_acc,
        mask=(offs_m < num_tokens) & (n_block == 0),
    )


def _mhc_prenorm_gemm_triton(
    x: torch.Tensor,
    fn: torch.Tensor,
    out_mul: torch.Tensor,
    out_sqrsum: torch.Tensor,
    n_splits: int,
) -> None:
    num_tokens, k = x.shape
    n = fn.shape[0]
    block_k = 64
    split_k = triton.cdiv(triton.cdiv(k, n_splits), block_k) * block_k
    _mhc_prenorm_gemm_triton_kernel[
        (n_splits, triton.cdiv(num_tokens, 16), triton.cdiv(n, 32))
    ](
        x,
        fn,
        out_mul,
        out_sqrsum,
        num_tokens,
        K=k,
        N=n,
        SPLIT_K=split_k,
        BLOCK_M=16,
        BLOCK_N=32,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=1,
    )


@triton.jit
def _load_reduced_mix(
    gemm_out_mul,
    token_id,
    mix_id: tl.constexpr,
    num_tokens,
    hc_mult3: tl.constexpr,
    n_splits: tl.constexpr,
):
    value = tl.full((), 0.0, tl.float32)
    for split_id in tl.static_range(0, n_splits):
        offset = split_id * num_tokens * hc_mult3 + token_id * hc_mult3 + mix_id
        value += tl.load(gemm_out_mul + offset)
    return value


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
    num_tokens,
):
    token_id = tl.program_id(0)

    rms_sum = tl.full((), 0.0, tl.float32)
    for split_id in tl.static_range(0, n_splits):
        rms_sum += tl.load(gemm_out_sqrsum + split_id * num_tokens + token_id)
    rms = tl.rsqrt(rms_sum / (hc_mult * hidden_size) + rms_eps)

    pre_scale = tl.load(hc_scale)
    for hc_id in tl.static_range(0, hc_mult):
        mix = _load_reduced_mix(
            gemm_out_mul,
            token_id,
            hc_id,
            num_tokens,
            hc_mult3,
            n_splits,
        )
        pre = tl.sigmoid(mix * rms * pre_scale + tl.load(hc_base + hc_id)) + hc_eps
        tl.store(pre_mix + token_id * hc_mult + hc_id, pre)

    post_scale = tl.load(hc_scale + 1)
    for hc_id in tl.static_range(0, hc_mult):
        mix = _load_reduced_mix(
            gemm_out_mul,
            token_id,
            hc_mult + hc_id,
            num_tokens,
            hc_mult3,
            n_splits,
        )
        post = (
            tl.sigmoid(mix * rms * post_scale + tl.load(hc_base + hc_mult + hc_id))
            * 2.0
        )
        tl.store(post_mix + token_id * hc_mult + hc_id, post)

    comb_offsets = tl.arange(0, block_comb)
    comb_mask = comb_offsets < hc_mult2
    comb_scale = tl.load(hc_scale + 2)
    comb_mix_values = tl.zeros((block_comb,), tl.float32)
    for split_id in tl.static_range(0, n_splits):
        split_base = split_id * num_tokens * hc_mult3 + token_id * hc_mult3
        comb_mix_values += tl.load(
            gemm_out_mul + split_base + hc_mult * 2 + comb_offsets,
            mask=comb_mask,
            other=0.0,
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
    acc0 = tl.load(post + post_base).to(tl.float32) * hidden_values
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
        acc0 += tl.load(comb + comb_row).to(tl.float32) * residual_values
        acc1 += tl.load(comb + comb_row + 1).to(tl.float32) * residual_values
        acc2 += tl.load(comb + comb_row + 2).to(tl.float32) * residual_values
        acc3 += tl.load(comb + comb_row + 3).to(tl.float32) * residual_values

    tl.store(out + token_residual_offset + hidden_offsets, acc0, mask=hidden_mask)
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


def _mhc_pre_impl(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    prenorm_gemm,
    norm_weight: torch.Tensor | None,
    norm_eps: float | None,
    pre_mix_impl=None,
    pre_reduce_apply_impl=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (norm_weight is None) != (norm_eps is None):
        raise ValueError("norm_weight and norm_eps must be provided together")
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

    n_splits = _compute_num_split(
        residual.device,
        64,
        hc_hidden_size,
        triton.cdiv(num_tokens, 64),
    )
    post_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    use_pre_reduce_apply = _pre_reduce_apply_is_supported(
        pre_reduce_apply_impl, n_splits
    )
    pre_mix = (
        None
        if use_pre_reduce_apply
        else torch.empty(
            num_tokens, hc_mult, dtype=torch.float32, device=residual.device
        )
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
    prenorm_gemm(
        residual_2d,
        fn,
        gemm_out_mul,
        gemm_out_sqrsum,
        n_splits,
    )
    block_h = 1024
    fused_norm = _pre_reduce_apply_fuses_norm(
        pre_reduce_apply_impl,
        use_pre_reduce_apply,
        norm_weight is not None,
    )
    if use_pre_reduce_apply:
        fused_norm_kwargs = {}
        if getattr(pre_reduce_apply_impl, "supports_fused_norm", False):
            fused_norm_kwargs = {
                "norm_weight": norm_weight if fused_norm else None,
                "norm_eps": norm_eps if fused_norm else 0.0,
                "block_size": 512,
                "enable_pdl": pdl_enabled(),
            }
        pre_reduce_apply_impl(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_flat,
            layer_input,
            post_mix,
            comb_mix,
            hidden_size,
            rms_eps,
            hc_eps,
            sinkhorn_iters,
            n_splits,
            num_tokens,
            **fused_norm_kwargs,
        )
    else:
        if pre_mix_impl is None:
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
                block_comb=triton.next_power_of_2(hc_mult2),
                num_tokens=num_tokens,
                num_warps=1,
            )
        else:
            pre_mix_impl(
                gemm_out_mul,
                gemm_out_sqrsum,
                hc_scale,
                hc_base,
                pre_mix,
                post_mix,
                comb_mix,
                hidden_size,
                rms_eps,
                hc_eps,
                sinkhorn_iters,
                n_splits,
                num_tokens,
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

    if norm_weight is not None and not fused_norm:
        if norm_eps is None:
            raise ValueError("norm_eps is required when norm_weight is provided")
        layer_input = torch.nn.functional.rms_norm(
            layer_input, (hidden_size,), norm_weight, norm_eps
        )

    return (
        layer_input.view(*outer_shape, hidden_size),
        post_mix.view(*outer_shape, hc_mult, 1),
        comb_mix.view(*outer_shape, hc_mult, hc_mult),
    )


def _tiled_mhc_pre_hc4(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the shared branch's tiled four-stream prefill projection."""
    outer_shape = residual.shape[:-2]
    hidden_size = residual.shape[-1]
    residual_flat = residual.view(-1, 4, hidden_size)
    num_tokens = residual_flat.shape[0]
    if num_tokens == 0:
        return (
            residual.new_empty(*outer_shape, hidden_size),
            torch.empty(
                *outer_shape, 4, 1, dtype=torch.float32, device=residual.device
            ),
            torch.empty(
                *outer_shape, 4, 4, dtype=torch.float32, device=residual.device
            ),
        )

    n_splits, block_m, block_k = _mhc_prefill_config_hc4(num_tokens)
    n_splits = min(n_splits, triton.cdiv(4 * hidden_size, block_k))
    projection = torch.empty(
        n_splits, num_tokens, 24, dtype=torch.float32, device=residual.device
    )
    square_sum = torch.empty(
        n_splits, num_tokens, dtype=torch.float32, device=residual.device
    )
    pre_mix = torch.empty(num_tokens, 4, dtype=torch.float32, device=residual.device)
    post_mix = torch.empty_like(pre_mix)
    comb_mix = torch.empty(num_tokens, 16, dtype=torch.float32, device=residual.device)
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )

    mhc_prefill_project_hc4(
        residual_flat,
        fn,
        projection,
        square_sum,
        n_splits=n_splits,
        block_m=block_m,
        block_k=block_k,
    )
    mhc_pre_mix_hc4(
        projection,
        square_sum,
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
        num_tokens=num_tokens,
    )
    block_h = 1024
    _mhc_pre_layer_triton_kernel[(num_tokens, triton.cdiv(hidden_size, block_h))](
        pre_mix,
        residual_flat,
        layer_input,
        hidden_size=hidden_size,
        hc_mult=4,
        block_h=block_h,
        num_warps=4,
    )
    return (
        layer_input.view(*outer_shape, hidden_size),
        post_mix.view(*outer_shape, 4, 1),
        comb_mix.view(*outer_shape, 4, 4),
    )


@register_kernel(
    "mhc",
    "pre",
    name="triton_mhc_pre",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=frozenset(
        {
            format_signature(
                residual=dense_tensor_format(torch.bfloat16),
                fn=dense_tensor_format(torch.float32),
                hc_scale=dense_tensor_format(torch.float32),
                hc_base=dense_tensor_format(torch.float32),
            )
        }
    ),
    priority=Priority.PORTABLE,
    tags={"portability"},
)
def triton_mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_weight: torch.Tensor | None,
    norm_eps: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the portable Triton mHC pre-mapping."""
    num_tokens = residual.numel() // (residual.shape[-2] * residual.shape[-1])
    if residual.shape[-2] == 4 and num_tokens > 256:
        layer_input, post_mix, comb_mix = _tiled_mhc_pre_hc4(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_eps,
            sinkhorn_iters,
        )
        if norm_weight is not None:
            if norm_eps is None:
                raise ValueError("norm_eps is required when norm_weight is provided")
            layer_input = torch.nn.functional.rms_norm(
                layer_input, (residual.shape[-1],), norm_weight, norm_eps
            )
        return layer_input, post_mix, comb_mix
    return _mhc_pre_impl(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_eps,
        sinkhorn_iters,
        _mhc_prenorm_gemm_triton,
        norm_weight,
        norm_eps,
    )


@register_kernel(
    "mhc",
    "post",
    name="triton_mhc_post",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=frozenset(
        {
            format_signature(
                hidden_states=dense_tensor_format(torch.bfloat16),
                residual=dense_tensor_format(torch.bfloat16),
                post=dense_tensor_format(torch.float32),
                comb=dense_tensor_format(torch.float32),
            )
        }
    ),
    priority=Priority.PORTABLE,
    tags={"portability"},
)
def triton_mhc_post(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Run the portable Triton mHC post-mapping."""
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

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

"""Activation quantization: per-tensor FP8 and dynamic MXFP4 with
CDNA4-swizzled e8m0 block scales (optionally fused with routed gather)."""

from __future__ import annotations

from typing import Any, Optional

import torch
from tokenspeed_kernel_amd._triton import tl, triton
from tokenspeed_kernel_amd.ops.gfx950.moe._common import (
    RaggedTensorMetadata,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused._common import (
    _ALIGN_K_SCALE_SWIZZLE,
    _NON_K_PRESHUFFLE_BLOCK_SIZE,
    _as_int32,
    _make_dummy,
    _ragged_scale_block_offs,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.scale_layout import (
    MXFP4_BLOCK,
    empty_swizzled_cdna4_mxfp4_scale,
)

_MXFP4_QUANT_TILED_MIN_ROWS = 128


_MXFP4_QUANT_TILED_BLOCK_M = 32


_MXFP4_QUANT_TILED_BLOCK_K_SCALE = 32


@triton.jit
def _mxfp4_quantize_block(x):
    max_normal: tl.constexpr = 6
    min_normal: tl.constexpr = 1
    amax = tl.max(tl.abs(x), axis=0)
    amax = amax.to(tl.int32, bitcast=True)
    amax = (amax + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    amax = amax.to(tl.float32, bitcast=True)
    scale_e8m0_unbiased = tl.log2(amax).floor() - 2
    scale_e8m0_unbiased = tl.clamp(scale_e8m0_unbiased, min=-127, max=127)
    scale_byte = scale_e8m0_unbiased.to(tl.uint8) + 127
    qx = x * tl.exp2(-scale_e8m0_unbiased)
    qx = qx.to(tl.uint32, bitcast=True)

    sign = qx & 0x80000000
    qx = qx ^ sign
    qx_fp32 = qx.to(tl.float32, bitcast=True)
    saturate_mask = qx_fp32 >= max_normal
    denormal_mask = (not saturate_mask) & (qx_fp32 < min_normal)
    normal_mask = not (saturate_mask | denormal_mask)

    denorm_exp: tl.constexpr = (127 - 1) + (23 - 1) + 1
    denorm_mask_int: tl.constexpr = denorm_exp << 23
    denorm_mask_float: tl.constexpr = tl.cast(denorm_mask_int, tl.float32, bitcast=True)
    denormal_x = qx_fp32 + denorm_mask_float
    denormal_x = denormal_x.to(tl.uint32, bitcast=True)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(tl.uint8)

    normal_x = qx
    mant_odd = (normal_x >> (23 - 1)) & 1
    normal_x += 0xC11FFFFF
    normal_x += mant_odd
    normal_x = normal_x >> (23 - 1)
    normal_x = normal_x.to(tl.uint8)

    e2m1 = tl.full(x.shape, 0x7, dtype=tl.uint8)
    e2m1 = tl.where(normal_mask, normal_x, e2m1)
    e2m1 = tl.where(denormal_mask, denormal_x, e2m1)
    sign_lp = sign >> (23 + 8 - 1 - 2)
    sign_lp = sign_lp.to(tl.uint8)
    e2m1 = e2m1 | sign_lp
    e2m1 = tl.reshape(e2m1, [16, 2])
    evens, odds = tl.split(e2m1)
    return evens | (odds << 4), scale_byte


@triton.jit
def _mxfp4_quantize_blocks(x):
    max_normal: tl.constexpr = 6
    min_normal: tl.constexpr = 1
    amax = tl.max(tl.abs(x), axis=2)
    amax = amax.to(tl.int32, bitcast=True)
    amax = (amax + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    amax = amax.to(tl.float32, bitcast=True)
    scale_e8m0_unbiased = tl.log2(amax).floor() - 2
    scale_e8m0_unbiased = tl.clamp(scale_e8m0_unbiased, min=-127, max=127)
    scale_byte = scale_e8m0_unbiased.to(tl.uint8) + 127
    qx = x * tl.expand_dims(tl.exp2(-scale_e8m0_unbiased), 2)
    qx = qx.to(tl.uint32, bitcast=True)

    sign = qx & 0x80000000
    qx = qx ^ sign
    qx_fp32 = qx.to(tl.float32, bitcast=True)
    saturate_mask = qx_fp32 >= max_normal
    denormal_mask = (not saturate_mask) & (qx_fp32 < min_normal)
    normal_mask = not (saturate_mask | denormal_mask)

    denorm_exp: tl.constexpr = (127 - 1) + (23 - 1) + 1
    denorm_mask_int: tl.constexpr = denorm_exp << 23
    denorm_mask_float: tl.constexpr = tl.cast(denorm_mask_int, tl.float32, bitcast=True)
    denormal_x = qx_fp32 + denorm_mask_float
    denormal_x = denormal_x.to(tl.uint32, bitcast=True)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(tl.uint8)

    normal_x = qx
    mant_odd = (normal_x >> (23 - 1)) & 1
    normal_x += 0xC11FFFFF
    normal_x += mant_odd
    normal_x = normal_x >> (23 - 1)
    normal_x = normal_x.to(tl.uint8)

    e2m1 = tl.full(x.shape, 0x7, dtype=tl.uint8)
    e2m1 = tl.where(normal_mask, normal_x, e2m1)
    e2m1 = tl.where(denormal_mask, denormal_x, e2m1)
    sign_lp = sign >> (23 + 8 - 1 - 2)
    sign_lp = sign_lp.to(tl.uint8)
    e2m1 = e2m1 | sign_lp
    e2m1 = tl.reshape(e2m1, [x.shape[0], x.shape[1], 16, 2])
    evens, odds = tl.split(e2m1)
    return evens | (odds << 4), scale_byte


@triton.jit
def _mxfp4_quantize_cdna4_scale_kernel(
    x_ptr,
    gather_ptr,
    slice_offs_ptr,
    scale_block_offs_ptr,
    out_ptr,
    scale_ptr,
    x_row_stride,
    out_row_stride,
    scale_stride_kswizzled,
    scale_stride_mblock,
    M: tl.constexpr,
    K_SCALE: tl.constexpr,
    HAS_GATHER: tl.constexpr,
    HAS_PADDED_SCALE_ROWS: tl.constexpr,
    N_EXPERTS: tl.constexpr,
    EXPERT_SEARCH_STEPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    M_SWIZZLE: tl.constexpr,
    K_SWIZZLE: tl.constexpr,
):
    out_m = tl.program_id(0)
    k_group = tl.program_id(1)
    row_in_range = out_m < M
    if HAS_PADDED_SCALE_ROWS:
        # Under HIP/CUDA-graph replay the grid is captured for the padded row
        # count M, but the route kernel only writes gather indices + ragged
        # metadata for gates that map to a valid expert (mask=valid). The final
        # slice_offs entry is that valid row count; rows at/after it are padding
        # whose gather index and scale are never consumed downstream. Their
        # metadata is uninitialised, so the expert binary search overshoots and
        # produces an out-of-bounds scale store. Skip them.
        n_valid_rows = tl.load(slice_offs_ptr + N_EXPERTS)
        row_in_range = row_in_range & (out_m < n_valid_rows)
    valid = row_in_range & (k_group < K_SCALE)
    src_m = out_m
    if HAS_GATHER:
        src_m = tl.load(gather_ptr + out_m, mask=row_in_range, other=0)

    offs_k = k_group * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(
        x_ptr + src_m * x_row_stride + offs_k,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    packed, scale_byte = _mxfp4_quantize_block(x)
    scale_byte = tl.where(valid, scale_byte, 0)

    pack_idx = tl.arange(0, 16)
    tl.store(
        out_ptr + out_m * out_row_stride + k_group * 16 + pack_idx,
        packed,
        mask=valid,
    )

    scale_m = out_m
    if HAS_PADDED_SCALE_ROWS:
        search_m = tl.minimum(out_m, M - 1)
        lo = tl.full((), 0, tl.int32)
        hi = tl.full((), N_EXPERTS, tl.int32)
        for _ in range(EXPERT_SEARCH_STEPS):
            mid = (lo + hi) // 2
            end = tl.load(slice_offs_ptr + mid + 1)
            go_left = search_m < end
            hi = tl.where(go_left, mid, hi)
            lo = tl.where(go_left, lo, mid + 1)
        # Clamp to a valid expert index. For in-range rows the search never
        # exceeds N_EXPERTS-1; the clamp only guards padding rows (masked off
        # above) against an out-of-bounds slice_offs/scale_block_offs read.
        expert = tl.minimum(lo, N_EXPERTS - 1)
        compact_base = tl.load(slice_offs_ptr + expert)
        scale_block_base = tl.load(scale_block_offs_ptr + expert)
        scale_m = scale_block_base * M_SWIZZLE + (out_m - compact_base)

    m_in_block = scale_m % M_SWIZZLE
    m_hi = m_in_block // 16
    m_lo = m_in_block % 16
    k_block = k_group // K_SWIZZLE
    k_in_block = k_group % K_SWIZZLE
    k_hi = k_in_block // 4
    k_lo = k_in_block % 4
    swizzled_k = (((k_block * 4 + k_lo) * 16 + m_lo) * 2 + k_hi) * 2 + m_hi
    m_block = scale_m // M_SWIZZLE
    tl.store(
        scale_ptr + swizzled_k * scale_stride_kswizzled + m_block * scale_stride_mblock,
        scale_byte,
        mask=valid,
    )


@triton.jit
def _mxfp4_quantize_cdna4_scale_tiled_kernel(
    x_ptr,
    gather_ptr,
    slice_offs_ptr,
    scale_block_offs_ptr,
    out_ptr,
    scale_ptr,
    x_row_stride,
    out_row_stride,
    scale_stride_kswizzled,
    scale_stride_mblock,
    M: tl.constexpr,
    K_SCALE: tl.constexpr,
    HAS_GATHER: tl.constexpr,
    HAS_PADDED_SCALE_ROWS: tl.constexpr,
    N_EXPERTS: tl.constexpr,
    EXPERT_SEARCH_STEPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    M_SWIZZLE: tl.constexpr,
    K_SWIZZLE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K_SCALE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_ks = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ks = pid_ks * BLOCK_K_SCALE + tl.arange(0, BLOCK_K_SCALE)
    offs_block = tl.arange(0, BLOCK_SIZE)
    valid_m = offs_m < M
    if HAS_PADDED_SCALE_ROWS:
        # See scalar kernel: skip padded rows beyond the valid routed-row count
        # (slice_offs[N_EXPERTS]); their gather index / metadata are
        # uninitialised under graph-replay and drive out-of-bounds stores.
        n_valid_rows = tl.load(slice_offs_ptr + N_EXPERTS)
        valid_m = valid_m & (offs_m < n_valid_rows)
    valid_ks = offs_ks < K_SCALE

    src_m = offs_m
    if HAS_GATHER:
        src_m = tl.load(gather_ptr + offs_m, mask=valid_m, other=0)

    src_m_e = tl.expand_dims(tl.expand_dims(src_m, 1), 2)
    offs_ks_e = tl.expand_dims(tl.expand_dims(offs_ks, 0), 2)
    offs_block_e = tl.expand_dims(tl.expand_dims(offs_block, 0), 0)
    valid = tl.expand_dims(tl.expand_dims(valid_m, 1), 2) & tl.expand_dims(
        tl.expand_dims(valid_ks, 0), 2
    )
    x = tl.load(
        x_ptr + src_m_e * x_row_stride + offs_ks_e * BLOCK_SIZE + offs_block_e,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    packed, scale_byte = _mxfp4_quantize_blocks(x)

    pack_idx = tl.arange(0, 16)
    out_m_e = tl.expand_dims(tl.expand_dims(offs_m, 1), 2)
    out_ks_e = tl.expand_dims(tl.expand_dims(offs_ks, 0), 2)
    pack_idx_e = tl.expand_dims(tl.expand_dims(pack_idx, 0), 0)
    out_mask = tl.expand_dims(tl.expand_dims(valid_m, 1), 2) & tl.expand_dims(
        tl.expand_dims(valid_ks, 0), 2
    )
    tl.store(
        out_ptr + out_m_e * out_row_stride + out_ks_e * 16 + pack_idx_e,
        packed,
        mask=out_mask,
    )

    scale_m = offs_m
    if HAS_PADDED_SCALE_ROWS:
        search_m = tl.minimum(offs_m, M - 1)
        lo = tl.full((BLOCK_M,), 0, tl.int32)
        hi = tl.full((BLOCK_M,), N_EXPERTS, tl.int32)
        for _ in range(EXPERT_SEARCH_STEPS):
            mid = (lo + hi) // 2
            end = tl.load(slice_offs_ptr + mid + 1)
            go_left = search_m < end
            hi = tl.where(go_left, mid, hi)
            lo = tl.where(go_left, lo, mid + 1)
        # Clamp to a valid expert index (guards padding rows masked off above).
        expert = tl.minimum(lo, N_EXPERTS - 1)
        compact_base = tl.load(slice_offs_ptr + expert)
        scale_block_base = tl.load(scale_block_offs_ptr + expert)
        scale_m = scale_block_base * M_SWIZZLE + (offs_m - compact_base)

    m_in_block = scale_m % M_SWIZZLE
    m_hi = m_in_block // 16
    m_lo = m_in_block % 16
    k_block = offs_ks // K_SWIZZLE
    k_in_block = offs_ks % K_SWIZZLE
    k_hi = k_in_block // 4
    k_lo = k_in_block % 4
    swizzled_k = (
        ((k_block * 4 + k_lo) * 16 + tl.expand_dims(m_lo, 1)) * 2 + k_hi
    ) * 2 + tl.expand_dims(m_hi, 1)
    m_block = scale_m // M_SWIZZLE
    scale_mask = tl.expand_dims(valid_m, 1) & tl.expand_dims(valid_ks, 0)
    tl.store(
        scale_ptr
        + swizzled_k * scale_stride_kswizzled
        + tl.expand_dims(m_block, 1) * scale_stride_mblock,
        scale_byte,
        mask=scale_mask,
    )


def _as_gather_tensor(gather_indx: Any | None) -> torch.Tensor | None:
    if gather_indx is None:
        return None
    return gather_indx.src_indx if hasattr(gather_indx, "src_indx") else gather_indx


def _quantize_mxfp4_activation(
    activations: torch.Tensor,
    gather_indx: Any | None = None,
    ragged_metadata: Any | None = None,
    *,
    _force_scalar: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if activations.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(
            "MXFP4 activation quantization requires bf16/fp16 input, "
            f"got {activations.dtype}"
        )
    if activations.ndim != 2:
        raise ValueError(
            "MXFP4 activation quantization expects a rank-2 tensor, "
            f"got shape={tuple(activations.shape)}"
        )
    if activations.shape[-1] % MXFP4_BLOCK != 0:
        raise ValueError(
            "MXFP4 activation quantization requires the last dimension to be "
            f"divisible by {MXFP4_BLOCK}, got {activations.shape[-1]}"
        )

    x = activations.contiguous()
    gather_tensor = _as_gather_tensor(gather_indx)
    rows = int(gather_tensor.shape[0]) if gather_tensor is not None else int(x.shape[0])
    k = int(x.shape[1])
    k_scale = k // MXFP4_BLOCK
    out = torch.empty((rows, k // 2), dtype=torch.uint8, device=x.device)
    if ragged_metadata is not None:
        n_slices = int(ragged_metadata.slice_sizes.shape[0])
        scale_rows = (
            RaggedTensorMetadata.n_blocks(
                n_slices,
                rows,
                _NON_K_PRESHUFFLE_BLOCK_SIZE,
            )
            * _NON_K_PRESHUFFLE_BLOCK_SIZE
        )
        slice_offs = _as_int32(ragged_metadata.slice_offs)
        scale_block_offs = _as_int32(_ragged_scale_block_offs(ragged_metadata))
        expert_search_steps = (n_slices + 1).bit_length()
    else:
        n_slices = 0
        scale_rows = rows
        slice_offs = _make_dummy(x.device, torch.int32)
        scale_block_offs = _make_dummy(x.device, torch.int32)
        expert_search_steps = 0
    scale = empty_swizzled_cdna4_mxfp4_scale(scale_rows, k_scale, device=x.device)
    if rows == 0:
        return out, scale

    k_scale_pad = (
        (k_scale + _ALIGN_K_SCALE_SWIZZLE - 1)
        // _ALIGN_K_SCALE_SWIZZLE
        * _ALIGN_K_SCALE_SWIZZLE
    )
    rows_pad = (
        (rows + _NON_K_PRESHUFFLE_BLOCK_SIZE - 1)
        // _NON_K_PRESHUFFLE_BLOCK_SIZE
        * _NON_K_PRESHUFFLE_BLOCK_SIZE
    )
    gather = (
        _as_int32(gather_tensor).contiguous()
        if gather_tensor is not None
        else _make_dummy(x.device, torch.int32)
    )
    if not _force_scalar and rows >= _MXFP4_QUANT_TILED_MIN_ROWS:
        block_m = _MXFP4_QUANT_TILED_BLOCK_M
        block_k_scale = _MXFP4_QUANT_TILED_BLOCK_K_SCALE
        grid = (
            triton.cdiv(rows_pad, block_m),
            triton.cdiv(k_scale_pad, block_k_scale),
        )
        _mxfp4_quantize_cdna4_scale_tiled_kernel[grid](
            x,
            gather,
            slice_offs,
            scale_block_offs,
            out,
            scale,
            x.stride(0),
            out.stride(0),
            scale.stride(0),
            scale.stride(1),
            M=rows,
            K_SCALE=k_scale,
            HAS_GATHER=gather_tensor is not None,
            HAS_PADDED_SCALE_ROWS=ragged_metadata is not None,
            N_EXPERTS=n_slices,
            EXPERT_SEARCH_STEPS=expert_search_steps,
            BLOCK_SIZE=MXFP4_BLOCK,
            M_SWIZZLE=_NON_K_PRESHUFFLE_BLOCK_SIZE,
            K_SWIZZLE=_ALIGN_K_SCALE_SWIZZLE,
            BLOCK_M=block_m,
            BLOCK_K_SCALE=block_k_scale,
            num_warps=4,
        )
    else:
        grid_rows = rows if rows * k_scale_pad >= 256 else rows_pad
        _mxfp4_quantize_cdna4_scale_kernel[(grid_rows, k_scale_pad)](
            x,
            gather,
            slice_offs,
            scale_block_offs,
            out,
            scale,
            x.stride(0),
            out.stride(0),
            scale.stride(0),
            scale.stride(1),
            M=rows,
            K_SCALE=k_scale,
            HAS_GATHER=gather_tensor is not None,
            HAS_PADDED_SCALE_ROWS=ragged_metadata is not None,
            N_EXPERTS=n_slices,
            EXPERT_SEARCH_STEPS=expert_search_steps,
            BLOCK_SIZE=MXFP4_BLOCK,
            M_SWIZZLE=_NON_K_PRESHUFFLE_BLOCK_SIZE,
            K_SWIZZLE=_ALIGN_K_SCALE_SWIZZLE,
            num_warps=1,
        )
    return out, scale


@triton.jit
def _fp8_quantize_kernel(
    x_ptr,
    out_ptr,
    scale,
    M,
    N,
    x_row_stride,
    out_row_stride,
    BLOCK_N: tl.constexpr,
    EVEN_N: tl.constexpr,
    FP8_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HAS_SCALE_TENSOR: tl.constexpr,
):
    pid = tl.program_id(0)
    m_idx = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_idx < M
    n_idx = tl.arange(0, BLOCK_N)

    if EVEN_N:
        load_mask = m_mask[:, None]
    else:
        load_mask = m_mask[:, None] & (n_idx[None, :] < N)

    x_off = m_idx[:, None] * x_row_stride + n_idx[None, :]
    x = tl.load(x_ptr + x_off, mask=load_mask)

    x = x.to(tl.float32)
    if HAS_SCALE:
        if HAS_SCALE_TENSOR:
            scale = tl.load(scale)
        x = x * (1.0 / scale)
    x_fp8 = x.to(FP8_DTYPE)

    out_off = m_idx[:, None] * out_row_stride + n_idx[None, :]
    tl.store(out_ptr + out_off, x_fp8, mask=load_mask)


@triton.jit
def _dynamic_fp8_quantize_kernel(x_ptr, out_ptr, scale_ptr, N: tl.constexpr):
    block: tl.constexpr = triton.next_power_of_2(N)
    offsets = tl.arange(0, block)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x)), 1.0e-12)
    scale = amax / 448.0
    tl.store(scale_ptr, scale)
    tl.store(out_ptr + offsets, (x / scale).to(tl.float8e4nv), mask=mask)


def _dynamic_fp8_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale = torch.empty(1, dtype=torch.float32, device=x.device)
    _dynamic_fp8_quantize_kernel[(1,)](
        x,
        out,
        scale,
        N=x.numel(),
        num_warps=8,
    )
    return out, scale


def _flatten_to_2d(x: torch.Tensor):
    assert x.stride(-1) == 1, f"expected stride-1 inner dim, got stride={x.stride(-1)}"
    N = x.shape[-1]
    if x.ndim == 1:
        return 1, N, N
    M = x.numel() // N
    row_stride = x.stride(-2)
    # Validate that every leading dim packs onto the next.
    for d in range(x.ndim - 2):
        expected = x.shape[d + 1] * x.stride(d + 1)
        if x.stride(d) != expected:
            raise ValueError(
                f"cannot flatten dim {d}: stride={x.stride(d)} but expected "
                f"shape[{d+1}]*stride[{d+1}]={expected}. Tensor shape={tuple(x.shape)}, "
                f"stride={tuple(x.stride())}."
            )
    return M, N, row_stride


def fp8_quantize(
    x: torch.Tensor,
    scale: float | torch.Tensor | None = None,
    out: Optional[torch.Tensor] = None,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    assert x.dtype in (
        torch.bfloat16,
        torch.float16,
    ), f"fp8_quantize input must be bf16/fp16, got {x.dtype}"
    assert fp8_dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float8_e4m3fnuz,
    ), f"fp8_quantize unsupported fp8 dtype: {fp8_dtype}"
    has_scale = scale is not None
    has_scale_tensor = isinstance(scale, torch.Tensor)
    if has_scale_tensor:
        assert scale.numel() == 1, "scale tensor must be scalar"
        scale = scale.contiguous()

    M, N, x_row_stride = _flatten_to_2d(x)

    if out is None:
        out = torch.empty(x.shape, dtype=fp8_dtype, device=x.device)
    else:
        assert out.shape == x.shape and out.dtype == fp8_dtype
    out_M, _, out_row_stride = _flatten_to_2d(out)
    assert out_M == M

    if fp8_dtype is torch.float8_e4m3fn:
        fp8_dtype_const = tl.float8e4nv
    elif fp8_dtype is torch.float8_e5m2:
        fp8_dtype_const = tl.float8e5
    else:
        fp8_dtype_const = tl.float8e4b8

    if M <= 2048:
        block_m = 4
    elif M <= 16384:
        block_m = 16
    else:
        block_m = 32
    num_warps = 4
    num_stages = 2

    grid = (triton.cdiv(M, block_m),)

    block_n = max(1, triton.next_power_of_2(N))
    even_n = block_n == N

    _fp8_quantize_kernel[grid](
        x,
        out,
        1.0 if scale is None else scale,
        M,
        N,
        x_row_stride,
        out_row_stride,
        BLOCK_N=block_n,
        EVEN_N=even_n,
        FP8_DTYPE=fp8_dtype_const,
        BLOCK_M=block_m,
        HAS_SCALE=has_scale,
        HAS_SCALE_TENSOR=has_scale_tensor,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out

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

"""Gluon MXFP4/MXFP8 quantization helpers for staged MXFP4-weight MoE."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton


@gluon.jit
def _mxfp4_quantize_tile(out):
    max_normal: gl.constexpr = 6.0
    min_normal: gl.constexpr = 1.0
    BLOCK_M: gl.constexpr = out.shape[0]
    OUT_BLOCK_N: gl.constexpr = out.shape[1]
    Q_GROUPS: gl.constexpr = OUT_BLOCK_N // 32
    gl.static_assert(OUT_BLOCK_N % 32 == 0)

    vals = out.to(gl.bfloat16).to(gl.float32).reshape((BLOCK_M, Q_GROUPS, 32))
    raw_abs = vals.to(gl.uint32, bitcast=True) & 0x7FFFFFFF
    abs_vals = raw_abs.to(gl.float32, bitcast=True)
    amax = gl.max(abs_vals, axis=2, keep_dims=True)
    amax_bits = amax.to(gl.uint32, bitcast=True)
    rounded_bits = (amax_bits + 0x200000) & 0x7F800000
    exp_biased = (rounded_bits >> 23).to(gl.int32)
    scale_i = gl.minimum(gl.maximum(exp_biased - 2, 0), 254)
    scale_byte = scale_i.to(gl.uint8).reshape((BLOCK_M, Q_GROUPS))

    inv_scale_bits = ((254 - scale_i) << 23).to(gl.uint32)
    inv_scale = inv_scale_bits.to(gl.float32, bitcast=True)
    qx = vals * inv_scale
    qx_bits = qx.to(gl.uint32, bitcast=True)

    sign = qx_bits & 0x80000000
    qx_mag = qx_bits ^ sign
    qx_fp32 = qx_mag.to(gl.float32, bitcast=True)
    saturate_mask = qx_fp32 >= max_normal
    denormal_mask = (not saturate_mask) & (qx_fp32 < min_normal)
    normal_mask = not (saturate_mask | denormal_mask)

    denorm_mask_int: gl.constexpr = ((127 - 1) + (23 - 1) + 1) << 23
    denorm_mask_float: gl.constexpr = gl.cast(
        denorm_mask_int,
        gl.float32,
        bitcast=True,
    )
    denormal_x = qx_fp32 + denorm_mask_float
    denormal_x = denormal_x.to(gl.uint32, bitcast=True)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(gl.uint8)

    normal_x = qx_mag
    mant_odd = (normal_x >> (23 - 1)) & 1
    normal_x += 0xC11FFFFF
    normal_x += mant_odd
    normal_x = normal_x >> (23 - 1)
    normal_x = normal_x.to(gl.uint8)

    e2m1 = gl.full(vals.shape, 0x7, gl.uint8, layout=vals.type.layout)
    e2m1 = gl.where(normal_mask, normal_x, e2m1)
    e2m1 = gl.where(denormal_mask, denormal_x, e2m1)
    sign_lp = (sign >> (23 + 8 - 1 - 2)).to(gl.uint8)
    e2m1 = e2m1 | sign_lp
    e2m1 = e2m1.reshape((BLOCK_M, Q_GROUPS, 16, 2))
    evens, odds = gl.split(e2m1)
    packed = evens | (odds << 4)
    return packed, scale_byte


@gluon.jit
def _mxfp8_quantize_tile(out):
    """Quantize each contiguous 32-value group to E4M3 + UE8M0."""

    BLOCK_M: gl.constexpr = out.shape[0]
    OUT_BLOCK_N: gl.constexpr = out.shape[1]
    Q_GROUPS: gl.constexpr = OUT_BLOCK_N // 32
    gl.static_assert(OUT_BLOCK_N % 32 == 0)

    vals = out.to(gl.bfloat16).to(gl.float32).reshape((BLOCK_M, Q_GROUPS, 32))
    amax = gl.max(gl.abs(vals), axis=2, keep_dims=True)
    safe_amax = gl.where(amax > 0.0, amax, 448.0 * (2.0**-127))
    scale_exp = gl.ceil(gl.log2(safe_amax / 448.0))
    scale_exp = gl.minimum(gl.maximum(scale_exp, -127.0), 127.0)
    scale_exp = gl.where(amax > 0.0, scale_exp, -127.0)
    quantized = (vals * gl.exp2(-scale_exp)).to(gl.float8e4nv)
    scale_byte = (scale_exp + 127.0).to(gl.uint8)
    return quantized.reshape((BLOCK_M, OUT_BLOCK_N)), scale_byte.reshape(
        (BLOCK_M, Q_GROUPS)
    )


@gluon.jit
def _mxfp4_store_cdna4_scale(
    scale_ptr,
    scale_byte,
    scale_m,
    scale_k,
    stride_kswizzled,
    stride_mblock,
    mask,
    M_SWIZZLE: gl.constexpr,
    K_SWIZZLE: gl.constexpr,
):
    m_in_block = scale_m % M_SWIZZLE
    m_hi = m_in_block // 16
    m_lo = m_in_block % 16
    k_block = scale_k // K_SWIZZLE
    k_in_block = scale_k % K_SWIZZLE
    k_hi = k_in_block // 4
    k_lo = k_in_block % 4
    swizzled_k = (((k_block * 4 + k_lo) * 16 + m_lo) * 2 + k_hi) * 2 + m_hi
    m_block = scale_m // M_SWIZZLE
    gl.store(
        scale_ptr
        + swizzled_k.to(gl.int64) * stride_kswizzled
        + m_block.to(gl.int64) * stride_mblock,
        scale_byte,
        mask=mask,
    )


@gluon.jit
def _mxfp8_quantize_sorted_kernel(
    x_ptr,
    sorted_ids_ptr,
    num_valid_ids_ptr,
    out_ptr,
    scale_ptr,
    M,
    K,
    EM,
    stride_xm,
    stride_xk,
    stride_om,
    stride_ok,
    scale_stride_kswizzled,
    scale_stride_mblock,
    BLOCK_M: gl.constexpr,
    BLOCK_K: gl.constexpr,
):
    """Gather routed rows and quantize them directly into sorted order."""

    layout: gl.constexpr = gl.BlockedLayout(
        [1, 16],
        [8, 8],
        [4, 1],
        [1, 0],
    )
    m_layout: gl.constexpr = gl.SliceLayout(1, layout)
    k_layout: gl.constexpr = gl.SliceLayout(0, layout)

    pid = gl.program_id(axis=0)
    num_pid_k = gl.cdiv(K, BLOCK_K)
    pid_m = pid // num_pid_k
    pid_k = pid % num_pid_k
    valid_extent = gl.load(num_valid_ids_ptr)
    if pid_m * BLOCK_M >= valid_extent:
        return

    rows = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=m_layout)
    cols = pid_k * BLOCK_K + gl.arange(0, BLOCK_K, layout=k_layout)
    packed_ids = gl.load(
        sorted_ids_ptr + rows,
        mask=rows < valid_extent,
        other=M,
    )
    source_rows = packed_ids & 0xFFFFFF
    valid_rows = (rows < valid_extent) & (source_rows < M)
    values = gl.load(
        x_ptr
        + source_rows[:, None].to(gl.int64) * stride_xm
        + cols[None, :].to(gl.int64) * stride_xk,
        mask=valid_rows[:, None] & (cols[None, :] < K),
        other=0.0,
    )
    quantized, scale_byte = _mxfp8_quantize_tile(values)
    gl.store(
        out_ptr
        + rows[:, None].to(gl.int64) * stride_om
        + cols[None, :].to(gl.int64) * stride_ok,
        quantized,
        mask=(rows[:, None] < valid_extent) & (cols[None, :] < K),
    )

    scale_layout: gl.constexpr = scale_byte.type.layout
    scale_rows = pid_m * BLOCK_M + gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(1, scale_layout)
    )
    scale_cols = pid_k * (BLOCK_K // 32) + gl.arange(
        0, BLOCK_K // 32, layout=gl.SliceLayout(0, scale_layout)
    )
    _mxfp4_store_cdna4_scale(
        scale_ptr,
        scale_byte,
        scale_rows[:, None],
        scale_cols[None, :],
        scale_stride_kswizzled,
        scale_stride_mblock,
        (scale_rows[:, None] < valid_extent) & (scale_cols[None, :] < K // 32),
        M_SWIZZLE=32,
        K_SWIZZLE=8,
    )


def quantize_mxfp8_sorted_routes(
    hidden_states: torch.Tensor,
    sorted_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather BF16 routes and emit sorted MXFP8 data and CDNA4 scales."""

    if hidden_states.ndim != 2 or hidden_states.dtype != torch.bfloat16:
        raise TypeError("sorted MXFP8 quantization requires a rank-2 BF16 input")
    if sorted_ids.ndim != 1 or sorted_ids.dtype != torch.int32:
        raise TypeError("sorted MXFP8 quantization requires rank-1 int32 route IDs")
    if num_valid_ids.ndim != 1 or num_valid_ids.dtype != torch.int32:
        raise TypeError("sorted MXFP8 quantization requires int32 valid metadata")
    if hidden_states.shape[1] % 256:
        raise ValueError("sorted MXFP8 quantization requires K divisible by 256")

    rows = int(sorted_ids.shape[0])
    k = int(hidden_states.shape[1])
    scale_cols = k // 32
    scale_rows = triton.cdiv(rows, 32) * 32
    output = torch.empty(
        (rows, k), dtype=torch.float8_e4m3fn, device=hidden_states.device
    )
    scales = torch.empty(
        (scale_rows, scale_cols), dtype=torch.uint8, device=hidden_states.device
    )
    if rows == 0:
        return output, scales

    block_m = 32
    block_k = 256
    grid = (triton.cdiv(rows, block_m) * triton.cdiv(k, block_k),)
    _mxfp8_quantize_sorted_kernel[grid](
        hidden_states,
        sorted_ids,
        num_valid_ids,
        output,
        scales,
        int(hidden_states.shape[0]),
        k,
        rows,
        hidden_states.stride(0),
        hidden_states.stride(1),
        output.stride(0),
        output.stride(1),
        1,
        scale_cols * 32,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return output, scales

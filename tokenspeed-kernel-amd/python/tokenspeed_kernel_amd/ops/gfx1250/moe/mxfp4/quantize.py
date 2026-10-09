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

"""Dynamic MXFP4 activation quantization with row-major e8m0 block scales."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import tl, triton

MXFP4_BLOCK = 32

# TDM descriptor row strides must stay 16-byte aligned.
_SCALE_ROW_ALIGN = 16

_TILED_MIN_ROWS = 128


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


def _quantize_launch_metadata(grid, kernel, args):
    """Report the activation read and the packed value and scale writes."""
    values = args["M"] * args["K_SCALE"] * MXFP4_BLOCK
    return {
        "name": kernel.name,
        "bytes": values * args["x_ptr"].element_size()
        + values // 2
        + values // MXFP4_BLOCK,
    }


@triton.jit(launch_metadata=_quantize_launch_metadata, do_not_specialize=["M"])
def triton_quantize_mxfp4_activation_gfx1250(
    x_ptr,
    out_ptr,
    scale_ptr,
    x_row_stride,
    out_row_stride,
    scale_row_stride,
    M,
    K_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K_SCALE: tl.constexpr,
):
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ks = tl.program_id(1) * BLOCK_K_SCALE + tl.arange(0, BLOCK_K_SCALE)
    offs_m = offs_m.to(tl.int64)
    valid = (offs_m < M)[:, None] & (offs_ks < K_SCALE)[None, :]

    offs_v = tl.arange(0, 32)
    x = tl.load(
        x_ptr
        + offs_m[:, None, None] * x_row_stride
        + offs_ks[None, :, None] * 32
        + offs_v[None, None, :],
        mask=valid[:, :, None],
        other=0.0,
    ).to(tl.float32)
    packed, scale_byte = _mxfp4_quantize_blocks(x)

    offs_p = tl.arange(0, 16)
    tl.store(
        out_ptr
        + offs_m[:, None, None] * out_row_stride
        + offs_ks[None, :, None] * 16
        + offs_p[None, None, :],
        packed,
        mask=valid[:, :, None],
    )
    tl.store(
        scale_ptr + offs_m[:, None] * scale_row_stride + offs_ks[None, :],
        scale_byte,
        mask=valid,
    )


def launch_triton_quantize_mxfp4_activation_gfx1250(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize rows of ``x`` to MXFP4 with one e8m0 scale per 32 values.

    Args:
        x: BF16/FP16 activations shaped ``(M, K)`` with ``K % 32 == 0``.

    Returns:
        ``(packed, scale)``: ``packed`` is ``(M, K // 2)`` uint8 with element
        ``2i`` in the low nibble of byte ``i``; ``scale`` is a ``(M, K // 32)``
        uint8 view whose row stride is padded to 16 bytes.
    """
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"MXFP4 quantization expects bf16/fp16 input, got {x.dtype}")
    if x.ndim != 2 or x.shape[1] % MXFP4_BLOCK != 0:
        raise ValueError(
            "MXFP4 quantization expects (M, K) input with K divisible by "
            f"{MXFP4_BLOCK}, got {tuple(x.shape)}"
        )
    if x.stride(1) != 1:
        x = x.contiguous()
    rows, k = map(int, x.shape)
    k_scale = k // MXFP4_BLOCK
    k_scale_padded = triton.cdiv(k_scale, _SCALE_ROW_ALIGN) * _SCALE_ROW_ALIGN
    packed = torch.empty((rows, k // 2), dtype=torch.uint8, device=x.device)
    scale = torch.empty((rows, k_scale_padded), dtype=torch.uint8, device=x.device)
    scale = scale[:, :k_scale]
    if rows == 0:
        return packed, scale

    if rows >= _TILED_MIN_ROWS:
        block_m, block_k_scale, num_warps = 32, 32, 4
    else:
        block_m, block_k_scale, num_warps = 4, 8, 1
    grid = (triton.cdiv(rows, block_m), triton.cdiv(k_scale, block_k_scale))
    triton_quantize_mxfp4_activation_gfx1250[grid](
        x,
        packed,
        scale,
        x.stride(0),
        packed.stride(0),
        scale.stride(0),
        rows,
        K_SCALE=k_scale,
        BLOCK_M=block_m,
        BLOCK_K_SCALE=block_k_scale,
        num_warps=num_warps,
    )
    return packed, scale

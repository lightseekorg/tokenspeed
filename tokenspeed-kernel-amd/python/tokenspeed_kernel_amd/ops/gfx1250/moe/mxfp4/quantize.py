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
from tokenspeed_kernel_amd._triton import gl, gluon, tl, triton

MXFP4_BLOCK = 32
_BLOCK_VALUES = gl.constexpr(MXFP4_BLOCK)

# TDM descriptor row strides must stay 16-byte aligned.
_SCALE_ROW_ALIGN = 16

_TILED_MIN_ROWS = 128


@triton.jit
def _mxfp4_quantize_blocks(x):
    """Quantize FP32 [M, B, 32] blocks to packed E2M1 [M, B, 16] and e8m0 [M, B].

    Adding ``0x200000`` and masking with ``0xFF800000`` rounds the block amax to
    a power of two, up once its mantissa reaches 0.75, and the scale is that
    power over 4, so scaled values stay below 7. Scaled values round to nearest
    even on their FP32 bits: below 1, adding ``2 ** 22`` makes the FP32 add
    round to a multiple of 0.5 and subtracting the same bits leaves the code.
    From 1 up to 6, ``0x1FFFFF`` plus the odd kept-mantissa bit rounds at that
    bit, the wrapping ``0xC1000000`` removes the exponent bias difference of
    126, and ``>> 22`` keeps the code. 6 and above saturate to ``0x7``. The even
    element of each pair goes in the low nibble.
    """
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


@gluon.constexpr_function
def _mxfp4_tile_split(block_m, block_n, num_warps):
    """Spread a ``[block_m, block_n]`` tile over the CTA without replication.

    Each lane holds ``values`` consecutive elements of one 32-value block;
    small tiles split a block across ``lanes_v`` lanes rather than leaving
    warps holding copies of the same rows.
    """
    blocks = block_n // MXFP4_BLOCK
    per_thread = max(1, block_m * block_n // (32 * num_warps))
    values = min(MXFP4_BLOCK, max(2, 1 << (per_thread - 1).bit_length()))
    lanes_v = MXFP4_BLOCK // values
    lanes_b = min(blocks, 32 // lanes_v)
    lanes_m = 32 // (lanes_v * lanes_b)
    warps_m = min(num_warps, max(1, block_m // lanes_m))
    return values, lanes_m, lanes_b, lanes_v, warps_m, num_warps // warps_m


@gluon.constexpr_function
def _mxfp4_tile_layout(block_m, block_n, num_warps, packed):
    """3D ``[M, blocks, values]`` layout; ``packed`` halves the value axis."""
    values, lanes_m, lanes_b, lanes_v, warps_m, warps_b = _mxfp4_tile_split(
        block_m, block_n, num_warps
    )
    return gl.BlockedLayout(
        [1, 1, values // 2 if packed else values],
        [lanes_m, lanes_b, lanes_v],
        [warps_m, warps_b, 1],
        [2, 1, 0],
    )


@gluon.constexpr_function
def _mxfp4_row_layout(block_m, block_n, num_warps):
    values, lanes_m, lanes_b, lanes_v, warps_m, warps_b = _mxfp4_tile_split(
        block_m, block_n, num_warps
    )
    return gl.BlockedLayout(
        [1, values], [lanes_m, lanes_b * lanes_v], [warps_m, warps_b], [1, 0]
    )


@gluon.jit
def store_mxfp4_tile_gfx1250(
    out,
    Y,
    stride_y_m,
    YMxScale,
    stride_y_mx_m,
    row_start,
    valid_rows,
    col_start,
    valid_cols,
    NUM_WARPS: gl.constexpr,
):
    """Quantize an FP32 ``[BLOCK_M, BLOCK_N]`` tile to MXFP4 and store it.

    The values and scales match ``_mxfp4_quantize_blocks`` on the same FP32
    values. ``row_start``/``col_start`` locate the tile in value units and
    only rows below ``valid_rows`` and 32-value blocks below ``valid_cols`` are
    written.
    """
    VALUES: gl.constexpr = _BLOCK_VALUES
    BLOCK_M: gl.constexpr = out.shape[0]
    BLOCK_N: gl.constexpr = out.shape[1]
    BLOCKS: gl.constexpr = BLOCK_N // VALUES
    gl.static_assert(BLOCK_N % VALUES == 0)
    ROW_LAYOUT: gl.constexpr = _mxfp4_row_layout(BLOCK_M, BLOCK_N, NUM_WARPS)
    X_LAYOUT: gl.constexpr = _mxfp4_tile_layout(BLOCK_M, BLOCK_N, NUM_WARPS, False)
    P_LAYOUT: gl.constexpr = _mxfp4_tile_layout(BLOCK_M, BLOCK_N, NUM_WARPS, True)

    x = gl.convert_layout(out, ROW_LAYOUT)
    x = gl.convert_layout(gl.reshape(x, [BLOCK_M, BLOCKS, VALUES]), X_LAYOUT)

    max_normal: gl.constexpr = 6
    min_normal: gl.constexpr = 1
    amax = gl.max(gl.abs(x), axis=2)
    amax = amax.to(gl.int32, bitcast=True)
    amax = (amax + 0x200000).to(gl.uint32, bitcast=True) & 0xFF800000
    amax = amax.to(gl.float32, bitcast=True)
    scale_e8m0_unbiased = gl.floor(gl.log2(amax)) - 2
    scale_e8m0_unbiased = gl.clamp(scale_e8m0_unbiased, -127, 127)
    scale_byte = scale_e8m0_unbiased.to(gl.uint8) + 127
    qx = x * gl.expand_dims(gl.exp2(-scale_e8m0_unbiased), 2)
    qx = qx.to(gl.uint32, bitcast=True)

    sign = qx & 0x80000000
    qx = qx ^ sign
    qx_fp32 = qx.to(gl.float32, bitcast=True)
    saturate_mask = qx_fp32 >= max_normal
    denormal_mask = (not saturate_mask) & (qx_fp32 < min_normal)
    normal_mask = not (saturate_mask | denormal_mask)

    # 2 ** 22, the FP32 whose ulp is 0.5.
    denorm_mask_int: gl.constexpr = ((127 - 1) + (23 - 1) + 1) << 23
    denormal_x = qx_fp32 + 4194304.0
    denormal_x = denormal_x.to(gl.uint32, bitcast=True)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(gl.uint8)

    normal_x = qx
    mant_odd = (normal_x >> (23 - 1)) & 1
    normal_x += 0xC11FFFFF
    normal_x += mant_odd
    normal_x = normal_x >> (23 - 1)
    normal_x = normal_x.to(gl.uint8)

    e2m1 = gl.full(x.shape, 0x7, gl.uint8, X_LAYOUT)
    e2m1 = gl.where(normal_mask, normal_x, e2m1)
    e2m1 = gl.where(denormal_mask, denormal_x, e2m1)
    sign_lp = sign >> (23 + 8 - 1 - 2)
    sign_lp = sign_lp.to(gl.uint8)
    e2m1 = e2m1 | sign_lp
    e2m1 = gl.reshape(e2m1, [BLOCK_M, BLOCKS, VALUES // 2, 2])
    evens, odds = gl.split(e2m1)
    packed = gl.convert_layout(evens | (odds << 4), P_LAYOUT)

    row_start = row_start.to(gl.int64)
    S_LAYOUT: gl.constexpr = gl.SliceLayout(2, X_LAYOUT)
    offs_m = gl.arange(0, BLOCK_M, gl.SliceLayout(1, S_LAYOUT))
    offs_b = gl.arange(0, BLOCKS, gl.SliceLayout(0, S_LAYOUT))
    mask = (offs_m < valid_rows)[:, None] & (col_start + offs_b * VALUES < valid_cols)[
        None, :
    ]
    gl.amd.gfx1250.buffer_store(
        scale_byte,
        YMxScale + row_start * stride_y_mx_m + col_start // VALUES,
        offs_m[:, None] * stride_y_mx_m + offs_b[None, :],
        mask=mask,
    )

    PS_LAYOUT: gl.constexpr = gl.SliceLayout(2, P_LAYOUT)
    offs_pm = gl.arange(0, BLOCK_M, gl.SliceLayout(1, PS_LAYOUT))
    offs_pb = gl.arange(0, BLOCKS, gl.SliceLayout(0, PS_LAYOUT))
    offs_pv = gl.arange(0, VALUES // 2, gl.SliceLayout(0, gl.SliceLayout(1, P_LAYOUT)))
    packed_mask = (offs_pm < valid_rows)[:, None, None] & (
        col_start + offs_pb * VALUES < valid_cols
    )[None, :, None]
    gl.amd.gfx1250.buffer_store(
        packed,
        Y + row_start * stride_y_m + col_start // 2,
        offs_pm[:, None, None] * stride_y_m
        + (offs_pb * (VALUES // 2))[None, :, None]
        + offs_pv[None, None, :],
        mask=packed_mask,
    )


@gluon.jit
def quantize_mxfp4_rows_tile_gfx1250(
    x_ptr,
    stride_x,
    q_ptr,
    stride_q,
    q_scale_ptr,
    stride_q_scale,
    M,
    tile,
    K: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Quantize tile ``tile`` of ``[M, K]`` rows, ``BLOCK_M x BLOCK_K`` each,
    with the layout of ``launch_triton_quantize_mxfp4_activation_gfx1250``."""
    COL_TILES: gl.constexpr = (K + BLOCK_K - 1) // BLOCK_K
    row_start = (tile // COL_TILES) * BLOCK_M
    col_start = (tile % COL_TILES) * BLOCK_K
    layout: gl.constexpr = _mxfp4_row_layout(BLOCK_M, BLOCK_K, NUM_WARPS)
    rows = row_start + gl.arange(0, BLOCK_M, gl.SliceLayout(1, layout))
    cols = col_start + gl.arange(0, BLOCK_K, gl.SliceLayout(0, layout))
    x = gl.load(
        x_ptr + rows[:, None] * stride_x + cols[None, :],
        mask=(rows < M)[:, None] & (cols < K)[None, :],
        other=0.0,
    )
    store_mxfp4_tile_gfx1250(
        x.to(gl.float32),
        q_ptr,
        stride_q,
        q_scale_ptr,
        stride_q_scale,
        row_start,
        M - row_start,
        col_start,
        K,
        NUM_WARPS,
    )


def empty_mxfp4_activation_scale(
    rows: int, k: int, device: torch.device, row_align: int = _SCALE_ROW_ALIGN
) -> torch.Tensor:
    """Allocate ``(rows, k // 32)`` e8m0 scales, rows padded to ``row_align``."""
    k_scale = k // MXFP4_BLOCK
    k_scale_padded = triton.cdiv(k_scale, row_align) * row_align
    scale = torch.empty((rows, k_scale_padded), dtype=torch.uint8, device=device)
    return scale[:, :k_scale]


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
    packed = torch.empty((rows, k // 2), dtype=torch.uint8, device=x.device)
    scale = empty_mxfp4_activation_scale(rows, k, x.device)
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

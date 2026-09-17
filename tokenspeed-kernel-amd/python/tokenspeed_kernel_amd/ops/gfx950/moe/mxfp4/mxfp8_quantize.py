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

"""Token-major E4M3 quantization and route-sorted group32 E8M0 scale words."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton


@gluon.jit
def _downcast_mxfp8(values, exponent):
    # E8M0 255 means NaN to scaled_downcast, whereas this quantizer's infinite
    # scale maps finite inputs to signed zero and infinite inputs to NaN.
    infinite_scale = exponent == 255
    values = gl.where(infinite_scale[:, None], values * 0.0, values)
    scale = gl.where(infinite_scale, 127, exponent).to(gl.uint8)
    return gl.amd.cdna4.scaled_downcast(values, scale[:, None], "e4m3", axis=1)


@gluon.jit(do_not_specialize=("ROWS",))
def _quantize_mxfp8_kernel(
    x,
    q,
    scales,
    ROWS,
    K: gl.constexpr,
    STRIDE: gl.constexpr,
    INNER: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([1, 32], [64, 1], [1, 1], [1, 0])
    group = gl.program_id(0) * 64 + gl.arange(0, 64, layout=gl.SliceLayout(1, layout))
    column = gl.arange(0, 32, layout=gl.SliceLayout(0, layout))
    row = group // (K // 32)
    base_k = (group % (K // 32)) * 32
    values = gl.load(
        x
        + row[:, None].to(gl.int64) * STRIDE
        + (base_k[:, None] + column[None, :]).to(gl.int64) * INNER,
        mask=row[:, None] < ROWS,
        other=0.0,
    ).to(gl.float32)
    # Sequential fmax ignores NaNs; the floor also covers an all-NaN group.
    magnitudes = gl.where(values == values, gl.abs(values), 0.0)
    amax = gl.maximum(gl.max(magnitudes, 1), 1.0e-10)
    bits = (amax * (1.0 / 448.0)).to(gl.uint32, bitcast=True)
    exponent = (bits >> 23) & 255
    exponent += ((exponent < 255) & ((bits & 0x7FFFFF) != 0)).to(gl.uint32)
    packed = _downcast_mxfp8(values, exponent)
    gl.store(
        q + group[:, None].to(gl.int64) * 32 + column[None, :],
        packed,
        mask=row[:, None] < ROWS,
    )
    gl.store(scales + group, exponent.to(gl.uint8), mask=row < ROWS)


@gluon.jit(do_not_specialize=("M",))
def _sort_mxfp8_scales(
    scales,
    sorted_ids,
    valid_ids,
    output,
    M,
    TOPK: gl.constexpr,
    K_GROUPS: gl.constexpr,
    K_PAD: gl.constexpr,
    K_BLOCK: gl.constexpr,
    SLOT_MAJOR: gl.constexpr,
):
    base_m = gl.program_id(0) * 32
    valid_rows = gl.load(valid_ids)
    if base_m < valid_rows:
        output += base_m.to(gl.int64) * K_PAD
        layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
        row = base_m + gl.arange(0, 32, layout=gl.SliceLayout(1, layout))
        k = gl.arange(0, K_BLOCK, layout=gl.SliceLayout(0, layout))
        encoded = gl.load(sorted_ids + row, mask=row < valid_rows, other=M).to(
            gl.uint32
        )
        token = encoded & 0xFFFFFF
        slot = encoded >> 24
        source = token * TOPK + slot if SLOT_MAJOR else token
        valid = (row < valid_rows) & (token < M) & (slot < TOPK)
        value = gl.load(
            scales + source[:, None].to(gl.int64) * K_GROUPS + k[None, :],
            mask=valid[:, None] & (k[None, :] < K_GROUPS),
            other=127,
        )
        offset = (
            (k[None, :] // 8) * 256
            + (k[None, :] % 4) * 64
            + (row[:, None] % 16) * 4
            + ((k[None, :] % 8) // 4) * 2
            + (row[:, None] % 32) // 16
        )
        gl.store(
            output + offset,
            value,
            mask=(row[:, None] < valid_rows) & (k[None, :] < K_GROUPS),
        )


@gluon.jit(do_not_specialize=("ROWS", "M", "VALUE_BLOCKS"))
def _quantize_sorted_mxfp8(
    x,
    q,
    sorted_ids,
    valid_ids,
    output,
    ROWS,
    M,
    VALUE_BLOCKS,
    TOPK: gl.constexpr,
    K: gl.constexpr,
    STRIDE: gl.constexpr,
    INNER: gl.constexpr,
    SLOT_MAJOR: gl.constexpr,
):
    # Two lanes own a group32 maximum, with 16 contiguous values per lane.
    layout: gl.constexpr = gl.BlockedLayout([1, 16], [32, 2], [4, 1], [1, 0])
    group = gl.arange(0, 128, layout=gl.SliceLayout(1, layout))
    column = gl.arange(0, 32, layout=gl.SliceLayout(0, layout))
    pid = gl.program_id(0)
    valid_rows = gl.load(valid_ids)
    values_role = pid < VALUE_BLOCKS
    if not values_role:
        # Compare row indices without multiplying the valid-prefix extent.
        if (pid - VALUE_BLOCKS) * 128 // (K // 32) >= valid_rows:
            return
    group += gl.where(values_role, pid, pid - VALUE_BLOCKS) * 128
    row, kg = group // (K // 32), group % (K // 32)
    if values_role:
        source = row.to(gl.int64)
        live = row < ROWS
    else:
        encoded = gl.load(sorted_ids + row, mask=row < valid_rows, other=M).to(
            gl.uint32
        )
        token, slot = encoded & 0xFFFFFF, encoded >> 24
        source = token.to(gl.int64) * TOPK + slot if SLOT_MAJOR else token.to(gl.int64)
        live = (row < valid_rows) & (token < M) & (slot < TOPK)
    values = gl.load(
        x
        + source[:, None].to(gl.int64) * STRIDE
        + (kg[:, None] * 32 + column[None, :]).to(gl.int64) * INNER,
        mask=live[:, None],
        other=0.0,
    ).to(gl.float32)
    magnitudes = gl.where(values == values, gl.abs(values), 0.0)
    amax = gl.maximum(gl.max(magnitudes, 1), 1.0e-10)
    bits = (amax * (1.0 / 448.0)).to(gl.uint32, bitcast=True)
    exponent = (bits >> 23) & 255
    exponent += ((exponent < 255) & ((bits & 0x7FFFFF) != 0)).to(gl.uint32)
    if values_role:
        packed = _downcast_mxfp8(values, exponent)
        gl.store(
            q + group[:, None].to(gl.int64) * 32 + column[None, :],
            packed,
            mask=live[:, None],
        )
    else:
        # Sorted routes have distinct scale bytes. Only the value-role CTAs
        # write q, including tokens with duplicate or entirely remote routes.
        offset = (
            (row.to(gl.int64) // 32) * (K // 32) * 32
            + (kg // 8) * 256
            + (kg % 4) * 64
            + (row % 16) * 4
            + ((kg % 8) // 4) * 2
            + (row % 32) // 16
        )
        gl.store(
            output + offset,
            gl.where(live, exponent, 127).to(gl.uint8),
            mask=row < valid_rows,
        )


def quantize_mxfp8(
    x: torch.Tensor,
    sorted_ids: torch.Tensor,
    valid_ids: torch.Tensor,
    *,
    tokens: int,
    topk: int,
    slot_major: bool,
    block_m: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return token-major FP8 values and route-sorted packed group32 scales."""
    rows, k = x.shape
    if k <= 0 or k % 256 or x.dtype != torch.bfloat16:
        raise ValueError("MXFP8 quantization requires BF16 rows and K divisible by 256")
    if rows * (k // 32) >= 2**31:
        raise ValueError("MXFP8 quantization exceeds the int32 group-index range")
    if block_m not in (32, 128):
        raise ValueError("MXFP8 scale blocks must contain 32 or 128 rows")
    if block_m == 32:
        value_blocks = triton.cdiv(rows * (k // 32), 128)
        sorted_rows = min(sorted_ids.numel(), tokens * topk * block_m)
        scale_blocks = triton.cdiv(sorted_rows * (k // 32), 128)
        # Include masked lanes in the final CTA, not only valid value groups.
        if max(value_blocks, scale_blocks) * 128 > 2**31:
            raise ValueError(
                "MXFP8 sorted quantization exceeds the int32 group-index range"
            )
    q = torch.empty((rows, k), dtype=torch.float8_e4m3fn, device=x.device)
    if block_m == 128:
        scales = torch.empty((rows, k // 32), dtype=torch.uint8, device=x.device)
    sorted_scales = torch.empty(
        (triton.cdiv(sorted_ids.numel(), 32) * 32, k // 32),
        dtype=torch.uint8,
        device=x.device,
    )
    if rows == 0:
        return q, sorted_scales
    if block_m == 32:
        _quantize_sorted_mxfp8[(value_blocks + scale_blocks,)](
            x,
            q,
            sorted_ids,
            valid_ids,
            sorted_scales,
            rows,
            tokens,
            value_blocks,
            topk,
            k,
            x.stride(0),
            x.stride(1),
            slot_major,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return q, sorted_scales
    _quantize_mxfp8_kernel[(triton.cdiv(rows * (k // 32), 64),)](
        x,
        q,
        scales,
        rows,
        k,
        x.stride(0),
        x.stride(1),
        num_warps=1,
        enable_fp_fusion=False,
    )
    _sort_mxfp8_scales[(triton.cdiv(sorted_ids.numel(), 32),)](
        scales,
        sorted_ids,
        valid_ids,
        sorted_scales,
        tokens,
        topk,
        k // 32,
        k // 32,
        triton.next_power_of_2(k // 32),
        slot_major,
        num_warps=4,
    )
    return q, sorted_scales

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


"""Package-prefill activation-scale gather for the gfx950 MXFP4-weight package.

The MXFP4 quantizer emits CDNA4-swizzled, token-order activation scales, but the
package stage kernels need those same bytes in sorted-route row order. This
module copies directly between the two CDNA4 layouts (token order -> sorted
route order) with no intermediate unswizzle.

It is package-specific glue -- it understands the CDNA4 scale layout and the
sorted-route packing produced by :mod:`moe_sorting` -- so it lives next to the
stage kernels that consume it.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import tl, triton

# MXFP4 microblock size and the CDNA4 scale-swizzle alignment are defined once in
# mxfp4_cdna4_scale_layout, shared with the weight-scale (B) preshuffle in the
# preprocessor. Local aliases preserve the names used throughout this module.
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.scale_layout import (
    CDNA4_SCALE_K_BLOCK as _ALIGN_K_SCALE_SWIZZLE,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.scale_layout import (
    CDNA4_SCALE_N_BLOCK as _NON_K_PRESHUFFLE_BLOCK_SIZE,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.scale_layout import (
    MXFP4_BLOCK as _MXFP4_BLOCK,
)


@triton.jit
def _gather_package_cdna4_scale_kernel(
    src_scale,
    sorted_ids,
    dst_scale,
    source_rows,
    num_sorted_ids,
    K_SCALE: tl.constexpr,
    K_SCALE_PAD: tl.constexpr,
    TOPK: tl.constexpr,
    FLATTEN_TOPK: tl.constexpr,
):
    """Gather row-major activation scales into a sorted-route CDNA4 swizzle.

    ``sorted_ids`` packs ``(topk_id << 24) | token_id`` per slot. One program
    owns one 32-row destination block: it loads each row's scales with
    contiguous vector loads, permutes the tile into the CDNA4 order
    ``(((k_block * 4 + k_lo) * 16 + m_lo) * 2 + k_hi) * 2 + m_hi`` in
    registers, and stores the block contiguously.
    """
    mblock = tl.program_id(0)
    rows = mblock * 32 + tl.arange(0, 32)
    in_bounds = rows < num_sorted_ids
    packed = tl.load(sorted_ids + rows, mask=in_bounds, other=source_rows)
    token = packed & 0xFFFFFF
    if FLATTEN_TOPK:
        src_row = token * TOPK + (packed >> 24)
    else:
        src_row = token
    valid = in_bounds & (src_row < source_rows)
    k = tl.arange(0, K_SCALE_PAD)
    value = tl.load(
        src_scale + src_row[:, None] * K_SCALE + k[None, :],
        mask=valid[:, None] & (k[None, :] < K_SCALE),
        other=127,
    )
    # Rows split as (m_hi, m_lo) and columns as (k_block, k_hi, k_lo).
    value = tl.reshape(value, (2, 16, K_SCALE_PAD // 8, 2, 4))
    value = tl.permute(value, (2, 4, 1, 3, 0))
    value = tl.reshape(value, (32 * K_SCALE_PAD,))
    offset = tl.arange(0, 32 * K_SCALE_PAD)
    dst_row = mblock * 32 + (offset // 4) % 16 + (offset % 2) * 16
    mask = (offset < 32 * K_SCALE) & (dst_row < num_sorted_ids)
    tl.store(dst_scale + mblock * 32 * K_SCALE + offset, value, mask=mask)


def gather_package_cdna4_scale(
    scale: torch.Tensor,
    sorted_ids: torch.Tensor,
    *,
    source_rows: int,
    cols: int,
    top_k: int,
    flatten_topk: bool,
) -> torch.Tensor:
    """Gather row-major activation scales into sorted-route CDNA4 order.

    Args:
        scale: row-major ``(source_rows, cols // 32)`` uint8 scale, as written
            by ``_quantize_mxfp4_activation(..., swizzle_scale=False)``.
        sorted_ids: sorted-route slots (``(topk_id << 24) | token_id``).
        source_rows: number of valid source rows (token or token*topk extent).
        cols: activation column count (K), must divide 32.
        top_k: experts per token.
        flatten_topk: if True, source rows are flattened ``token * TOPK + slot``.

    Returns:
        ``(rows_pad, K // 32)`` uint8 scale in the CDNA4-swizzled sorted-route
        order; slots past ``sorted_ids`` or with out-of-range tokens read 127.
    """
    if cols % _MXFP4_BLOCK != 0:
        raise ValueError(f"package prefill scale columns must divide by 32: {cols}")
    k_scale = cols // _MXFP4_BLOCK
    if (
        scale.dtype != torch.uint8
        or scale.ndim != 2
        or scale.shape[1] != k_scale
        or not scale.is_contiguous()
    ):
        raise ValueError(
            "package prefill requires a contiguous row-major uint8 activation "
            f"scale with {k_scale} columns"
        )
    if k_scale % _ALIGN_K_SCALE_SWIZZLE != 0:
        raise ValueError(
            "package prefill currently requires K/32 divisible by "
            f"{_ALIGN_K_SCALE_SWIZZLE}, got {k_scale}"
        )
    sorted_rows = int(sorted_ids.shape[0])
    rows_pad = (
        (sorted_rows + _NON_K_PRESHUFFLE_BLOCK_SIZE - 1)
        // _NON_K_PRESHUFFLE_BLOCK_SIZE
        * _NON_K_PRESHUFFLE_BLOCK_SIZE
    )
    out = torch.empty((rows_pad, k_scale), dtype=torch.uint8, device=scale.device)
    if sorted_rows == 0:
        return out
    _gather_package_cdna4_scale_kernel[(rows_pad // _NON_K_PRESHUFFLE_BLOCK_SIZE,)](
        scale,
        sorted_ids,
        out,
        source_rows,
        sorted_rows,
        K_SCALE=k_scale,
        K_SCALE_PAD=triton.next_power_of_2(k_scale),
        TOPK=top_k,
        FLATTEN_TOPK=flatten_topk,
        # Eight waves keep more of the 4 KiB block permutes in flight (K3 TP8
        # prefill: 26 -> 17 us per layer versus four).
        num_warps=8,
    )
    return out
    _gather_package_cdna4_scale_kernel[(rows_pad // _NON_K_PRESHUFFLE_BLOCK_SIZE,)](
        scale,
        sorted_ids,
        out,
        source_rows,
        scale.stride(1),
        k_scale * _NON_K_PRESHUFFLE_BLOCK_SIZE,
        sorted_rows,
        K_SCALE=k_scale,
        TOPK=top_k,
        FLATTEN_TOPK=flatten_topk,
        K_GROUPS=triton.next_power_of_2(k_scale // 2),
        num_warps=4,
    )
    return out

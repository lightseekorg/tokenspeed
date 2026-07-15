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


"""Package-prefill activation-scale gather for the gfx950 A4W4 MoE package.

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
import triton
import triton.language as tl

# MXFP4 microblock size and the CDNA4 scale-swizzle alignment. These are fixed
# properties of the CDNA4 MXFP4 scale layout, kept local so the package is
# self-contained.
_MXFP4_BLOCK = 32
_NON_K_PRESHUFFLE_BLOCK_SIZE = 32
_ALIGN_K_SCALE_SWIZZLE = 8


@triton.jit
def _gather_package_cdna4_scale_kernel(
    src_scale,
    sorted_ids,
    dst_scale,
    source_rows,
    src_mblock_stride,
    dst_mblock_stride,
    num_sorted_ids,
    K_SCALE: tl.constexpr,
    TOPK: tl.constexpr,
    FLATTEN_TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Gather a CDNA4-swizzled activation scale by sorted-route rows.

    Copies directly between the token-order and sorted-route CDNA4 layouts;
    ``sorted_ids`` packs ``(topk_id << 24) | token_id`` per slot.
    """
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    dst_row = linear // K_SCALE
    k_scale = linear % K_SCALE
    in_bounds = dst_row < num_sorted_ids

    packed = tl.load(sorted_ids + dst_row, mask=in_bounds, other=source_rows)
    token = packed & 0xFFFFFF
    if FLATTEN_TOPK:
        slot = packed >> 24
        src_row = token * TOPK + slot
    else:
        src_row = token
    valid = in_bounds & (src_row < source_rows)

    src_m_in = src_row % 32
    src_m_hi = src_m_in // 16
    src_m_lo = src_m_in % 16
    k_block = k_scale // 8
    k_hi = (k_scale % 8) // 4
    k_lo = k_scale % 4
    src_swizzled_k = (((k_block * 4 + k_lo) * 16 + src_m_lo) * 2 + k_hi) * 2 + src_m_hi
    src_off = src_swizzled_k + (src_row // 32) * src_mblock_stride

    dst_m_in = dst_row % 32
    dst_m_hi = dst_m_in // 16
    dst_m_lo = dst_m_in % 16
    dst_swizzled_k = (((k_block * 4 + k_lo) * 16 + dst_m_lo) * 2 + k_hi) * 2 + dst_m_hi
    dst_off = dst_swizzled_k + (dst_row // 32) * dst_mblock_stride

    value = tl.load(src_scale + src_off, mask=valid, other=127)
    tl.store(dst_scale + dst_off, value, mask=in_bounds)


def gather_package_cdna4_scale(
    scale: torch.Tensor,
    sorted_ids: torch.Tensor,
    *,
    source_rows: int,
    cols: int,
    top_k: int,
    flatten_topk: bool,
) -> torch.Tensor:
    """Remap a CDNA4-swizzled activation scale into sorted-route row order.

    Args:
        scale: rank-2 uint8 CDNA4-swizzled activation scale.
        sorted_ids: sorted-route slots (``(topk_id << 24) | token_id``).
        source_rows: number of valid source rows (token or token*topk extent).
        cols: activation column count (K), must divide 32.
        top_k: experts per token.
        flatten_topk: if True, source rows are flattened ``token * TOPK + slot``.

    Returns:
        ``(rows_pad, K // 32)`` uint8 scale in sorted-route order.
    """
    if scale.dtype != torch.uint8 or scale.ndim != 2:
        raise ValueError(
            "package prefill requires a rank-2 uint8 gdot128 activation scale"
        )
    if cols % _MXFP4_BLOCK != 0:
        raise ValueError(f"package prefill scale columns must divide by 32: {cols}")
    k_scale = cols // _MXFP4_BLOCK
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
    block = 256
    _gather_package_cdna4_scale_kernel[(triton.cdiv(sorted_rows * k_scale, block),)](
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
        BLOCK=block,
        num_warps=4,
    )
    return out

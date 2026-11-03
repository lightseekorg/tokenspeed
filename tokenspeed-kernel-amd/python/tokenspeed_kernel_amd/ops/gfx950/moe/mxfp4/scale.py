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
    src_mblock_stride,
    dst_mblock_stride,
    num_sorted_ids,
    K_BLOCKS: tl.constexpr,
    TOPK: tl.constexpr,
    FLATTEN_TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Gather a CDNA4-swizzled activation scale by sorted-route rows.

    Copies directly between the token-order and sorted-route CDNA4 layouts;
    ``sorted_ids`` packs ``(topk_id << 24) | token_id`` per slot. In both
    layouts the byte for ``(row, k)`` within a 32-row panel and 8-scale K block
    sits at ``(k % 4) * 64 + (row % 16) * 4 + (k % 8 // 4) * 2 + row % 32 // 16``,
    so each destination dword holds rows ``m`` and ``m + 16`` at scales ``k``
    and ``k + 4``. One lane writes one destination dword from one source dword
    per row: the row's two scales are bytes ``row % 32 // 16`` and ``2 + that``.
    Padded slots, and rows past ``num_sorted_ids`` up to the panel boundary,
    receive 127 (2^0).
    """
    src_words = src_scale.to(tl.pointer_type(tl.int32), bitcast=True)
    dst_words = dst_scale.to(tl.pointer_type(tl.int32), bitcast=True)
    word = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m_lo = word % 16
    k_lo = (word // 16) % 4
    k_block = (word // 64) % K_BLOCKS
    panel = word // (64 * K_BLOCKS)
    in_range = panel < tl.cdiv(num_sorted_ids, 32)
    k_part = k_block * 256 + k_lo * 64
    result = tl.zeros([BLOCK], dtype=tl.uint32)
    for m_hi in tl.static_range(2):
        dst_row = panel * 32 + m_hi * 16 + m_lo
        in_bounds = in_range & (dst_row < num_sorted_ids)
        packed = tl.load(sorted_ids + dst_row, mask=in_bounds, other=source_rows)
        token = packed & 0xFFFFFF
        if FLATTEN_TOPK:
            src_row = token * TOPK + (packed >> 24)
        else:
            src_row = token
        valid = in_bounds & (src_row < source_rows)
        src_word = tl.load(
            src_words
            + ((src_row // 32) * src_mblock_stride + k_part + (src_row % 16) * 4) // 4,
            mask=valid,
            other=0x7F7F7F7F,
        ).to(tl.uint32, bitcast=True)
        shift = ((src_row % 32) // 16 * 8).to(tl.uint32)
        result |= ((src_word >> shift) & 0xFF) << (m_hi * 8)
        result |= ((src_word >> (shift + 16)) & 0xFF) << (16 + m_hi * 8)
    tl.store(
        dst_words + (panel * dst_mblock_stride + k_part + m_lo * 4) // 4,
        result.to(tl.int32, bitcast=True),
        mask=in_range,
    )


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
    if scale.stride(0) != 1 or scale.stride(1) % 4 or scale.data_ptr() % 4:
        raise ValueError(
            "gather_package_cdna4_scale reads 4-byte words and needs stride(0) == 1, "
            f"stride(1) % 4 == 0 and a 4-byte aligned pointer; got strides "
            f"{tuple(scale.stride())} and pointer offset {scale.data_ptr() % 4}"
        )
    # One lane per destination dword: 32 rows x 8 scales per 256-byte block.
    block = 512
    _gather_package_cdna4_scale_kernel[(triton.cdiv(rows_pad * k_scale // 4, block),)](
        scale,
        sorted_ids,
        out,
        source_rows,
        scale.stride(1),
        k_scale * _NON_K_PRESHUFFLE_BLOCK_SIZE,
        sorted_rows,
        K_BLOCKS=k_scale // _ALIGN_K_SCALE_SWIZZLE,
        TOPK=top_k,
        FLATTEN_TOPK=flatten_topk,
        BLOCK=block,
        num_warps=4,
    )
    return out

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

"""Row-wise FP32 top-k for GFX1250 CSA2 selection.

One program per row finds the k-th largest value with four 8-bit radix
histogram passes, then writes the selected columns in ascending order. Ties at
the threshold are taken in column order.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import tl, triton

__all__ = ["launch_triton_dsv41_index_topk_select_gfx1250"]

_BLOCK = 4096
_NUM_WARPS = 16
_RADIX_BINS = 256
_RADIX_PASSES = 4


@triton.jit
def _ordered_key(values):
    """Map FP32 values to uint32 keys with the same order; -0.0 maps to +0.0."""
    bits = values.to(tl.uint32, bitcast=True)
    bits = tl.where(bits == 0x80000000, 0, bits)
    return tl.where(bits >= 0x80000000, ~bits, bits | 0x80000000)


def _select_launch_metadata(grid, kernel, args):
    """Report the radix passes and the final pass over each row, plus the
    selected values and columns."""
    rows = grid[0]
    width = args["width"]
    k = args["k"]
    return {
        "name": kernel.name,
        "bytes": rows * width * 4 * (_RADIX_PASSES + 1) + rows * k * (4 + 8),
    }


@triton.jit(
    launch_metadata=_select_launch_metadata,
    do_not_specialize=["width", "k", "stride", "out_stride"],
)
def triton_dsv41_index_topk_select_gfx1250(
    scores_ptr,
    values_ptr,
    columns_ptr,
    width,
    k,
    stride,
    out_stride,
    BLOCK: tl.constexpr,
    RADIX_BINS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    scores_ptr += row * stride
    offsets = tl.arange(0, BLOCK)
    bins = tl.arange(0, RADIX_BINS)
    threshold = tl.full((), 0, tl.uint32)
    remaining = k
    for shift in tl.static_range(24, -1, -8):
        hist = tl.zeros((RADIX_BINS,), dtype=tl.int32)
        for start in range(0, width, BLOCK):
            columns = start + offsets
            live = columns < width
            key = _ordered_key(tl.load(scores_ptr + columns, mask=live, other=0.0))
            if shift < 24:
                live = live & ((key >> (shift + 8)) == (threshold >> (shift + 8)))
            digits = ((key >> shift) & (RADIX_BINS - 1)).to(tl.int32)
            hist += tl.histogram(digits, RADIX_BINS, mask=live)
        at_least = tl.sum(hist, axis=0) - tl.cumsum(hist, axis=0) + hist
        digit = tl.sum((at_least >= remaining).to(tl.int32), axis=0) - 1
        remaining -= tl.sum(tl.where(bins == digit, at_least - hist, 0), axis=0)
        threshold |= digit.to(tl.uint32) << shift

    taken = 0
    ties = 0
    values_ptr += row * out_stride
    columns_ptr += row * out_stride
    for start in range(0, width, BLOCK):
        columns = start + offsets
        live = columns < width
        values = tl.load(scores_ptr + columns, mask=live, other=0.0)
        key = _ordered_key(values)
        equal = live & (key == threshold)
        tie_rank = ties + tl.cumsum(equal.to(tl.int32), axis=0)
        selected = (live & (key > threshold)) | (equal & (tie_rank <= remaining))
        slot = taken + tl.cumsum(selected.to(tl.int32), axis=0) - 1
        tl.store(values_ptr + slot, values, mask=selected)
        tl.store(columns_ptr + slot, columns.to(tl.int64), mask=selected)
        taken += tl.sum(selected.to(tl.int32), axis=0)
        ties += tl.sum(equal.to(tl.int32), axis=0)


def launch_triton_dsv41_index_topk_select_gfx1250(
    scores: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the ``k`` largest entries of each row of FP32 ``scores``.

    Args:
        scores: FP32 ``[rows, width]`` scores with unit column stride.
        k: Number of entries per row, ``1 <= k <= width``.

    Returns:
        ``(values, columns)`` shaped ``[rows, k]`` like ``torch.topk(...,
        sorted=False)``, with each row's columns ascending.
    """
    if scores.ndim != 2 or scores.dtype != torch.float32 or scores.stride(1) != 1:
        raise ValueError("scores must be FP32 [rows, width] with unit column stride")
    rows, width = scores.shape
    if not 1 <= k <= width:
        raise ValueError(f"k must be in [1, {width}], got {k}")
    values = torch.empty((rows, k), device=scores.device, dtype=torch.float32)
    columns = torch.empty((rows, k), device=scores.device, dtype=torch.int64)
    if rows:
        triton_dsv41_index_topk_select_gfx1250[(rows,)](
            scores,
            values,
            columns,
            width,
            k,
            scores.stride(0),
            k,
            BLOCK=_BLOCK,
            RADIX_BINS=_RADIX_BINS,
            num_warps=_NUM_WARPS,
        )
    return values, columns

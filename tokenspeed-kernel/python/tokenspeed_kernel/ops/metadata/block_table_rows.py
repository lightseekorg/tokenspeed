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

"""Rebuild block-table rows on the device from the span each row uploads.

A sliding-window group's row is mostly null holes (0) before the window and
-1 padding after the row's pages. The host uploads only the span between
them and this op writes the full rows back, so the device tables are the
same as uploading every entry.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

__all__ = ["BLOCK_TABLE_ROW_FIELDS", "expand_block_table_rows"]

#: Columns of one ``rows`` entry: where the row starts in ``out``, its width,
#: its span ``[start, end)`` and where the span's entries start in ``values``.
BLOCK_TABLE_ROW_FIELDS = 5
_BLOCK = 1024


@triton.jit(do_not_specialize_on_alignment=["Values"])
def _expand_block_table_rows_kernel(Rows, Values, Out, BLOCK: tl.constexpr):
    row = Rows + tl.program_id(0) * 5
    # The fields are int32 (the public op bounds them); offsets go int64.
    out = Out + tl.load(row).to(tl.int64)
    width = tl.load(row + 1)
    start = tl.load(row + 2)
    end = tl.load(row + 3)
    values = Values + tl.load(row + 4).to(tl.int64)
    # A row's programs stride over its tiles, so the grid follows the widest
    # rows' share of the table rather than the widest row.
    for tile in range(tl.program_id(1), tl.cdiv(width, BLOCK), tl.num_programs(1)):
        col = tile * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(values + (col - start), (col >= start) & (col < end), other=0)
        tl.store(out + col, tl.where(col < end, value, -1), col < width)


def _expand_block_table_rows_torch(
    rows: torch.Tensor, values: torch.Tensor, out: torch.Tensor
) -> None:
    """The kernel in tensor ops, for devices without it."""
    offset, width, start, end, source = rows.to(torch.int64).unbind(1)
    row = torch.repeat_interleave(
        torch.arange(rows.shape[0], device=out.device),
        width,
        output_size=out.numel(),
    )
    col = torch.arange(out.numel(), device=out.device) - offset[row]
    start, end = start[row], end[row]
    window = (col >= start) & (col < end)
    index = torch.where(window, source[row] + col - start, 0)
    value = values[index] if values.numel() else torch.zeros_like(out)
    out.copy_(torch.where(window, value, torch.where(col < start, 0, -1)))


def expand_block_table_rows(
    rows: torch.Tensor, values: torch.Tensor, out: torch.Tensor
) -> None:
    """Write block-table rows from their spans.

    Entry ``(offset, width, start, end, source)`` of ``rows`` fills
    ``out[offset : offset + width]``: null holes (0) before column ``start``,
    ``values[source : source + end - start]`` in ``[start, end)``, and
    padding (-1) from ``end`` on.

    Args:
        rows: ``[R, BLOCK_TABLE_ROW_FIELDS]`` contiguous int32 row entries,
            with ``0 <= start <= end <= width``. The rows tile ``out`` in
            order: each ``offset`` is the sum of the widths before it.
        values: 1-D int32 span entries on ``rows.device``.
        out: Contiguous 1-D int32 destination on ``rows.device``.

    Returns:
        None; ``out`` is written in place, every entry exactly once.
    """
    if (
        rows.dtype != torch.int32
        or rows.ndim != 2
        or rows.shape[1] != BLOCK_TABLE_ROW_FIELDS
        or not rows.is_contiguous()
    ):
        raise ValueError(f"rows must be contiguous int32 [R, {BLOCK_TABLE_ROW_FIELDS}]")
    if values.dtype != torch.int32 or values.ndim != 1:
        raise ValueError("values must be 1-D int32")
    if out.dtype != torch.int32 or out.ndim != 1 or not out.is_contiguous():
        raise ValueError("out must be contiguous 1-D int32")
    if rows.device != out.device or values.device != out.device:
        raise ValueError("rows, values and out must share a device")
    # The int32 row fields address out and values.
    if max(out.numel(), values.numel()) >= 1 << 31:
        raise ValueError("block tables beyond 2**31 entries need int64 row fields")
    if not rows.shape[0] or not out.numel():
        return
    if not out.is_cuda:
        _expand_block_table_rows_torch(rows, values, out)
        return
    tiles = triton.cdiv(out.numel(), _BLOCK)
    _expand_block_table_rows_kernel[
        (rows.shape[0], max(1, min(64, triton.cdiv(tiles, rows.shape[0]))))
    ](
        rows,
        # An empty span list still needs a valid pointer; no load reads it.
        values if values.numel() else rows,
        out,
        BLOCK=_BLOCK,
    )

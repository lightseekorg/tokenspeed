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

"""Expand scheduler pages into kernel pages in one launch."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(
    do_not_specialize=[
        "logical_cols",
        "value_cols",
        "table_stride",
        "out_stride",
        "out_cols",
        "ratio",
    ]
)
def _expand_page_table_kernel(
    table,
    out,
    logical_cols,
    value_cols,
    table_stride,
    out_stride,
    out_cols,
    ratio,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    inside = col < out_cols
    logical = col // ratio
    # Columns past the expanded region stay zero, as the torch spelling left them.
    valid = inside & (col < value_cols) & (logical < logical_cols)
    page = tl.load(table + row * table_stride + logical, mask=valid, other=0)
    value = tl.where(valid, tl.maximum(page, 0) * ratio + col % ratio, 0)
    tl.store(out + row * out_stride + col, value, mask=inside)


def expand_page_table(
    page_table: torch.Tensor,
    out: torch.Tensor,
    *,
    ratio: int,
    max_kernel_pages: int,
) -> torch.Tensor:
    """Write ``page * ratio + offset`` for every kernel page of every request.

    The torch spelling allocates an intermediate ``[rows, cols, ratio]`` tensor
    and runs zero/arange/clamp/mul/add/copy; each is launch-bound at decode
    sizes. One program block covers a tile of the output, so the whole table
    expands in a single launch with no intermediate.

    Args:
        page_table: Scheduler page ids ``[rows, logical_cols]``; negative
            entries are table holes and clamp to the null page.
        out: Destination ``[>=rows, >=max_kernel_pages]``, same dtype and
            device; its full width is written (expanded values then zeros), so
            a stale tail cannot survive.
        ratio: Kernel pages per scheduler page.
        max_kernel_pages: Kernel-page columns the caller will read.

    Returns:
        The ``[rows, max_kernel_pages]`` view of ``out``.
    """
    if page_table.ndim != 2 or out.ndim != 2:
        raise ValueError("page_table and out must both be 2-D")
    if out.dtype != page_table.dtype or out.device != page_table.device:
        raise ValueError("out must match page_table dtype and device")
    # Unit-stride arithmetic below; the torch spelling handled any stride.
    if page_table.stride(1) != 1 or out.stride(1) != 1:
        raise ValueError("page_table and out rows must be unit-stride")
    if out.data_ptr() == page_table.data_ptr():
        raise ValueError("out must not alias page_table")
    rows, logical_cols = page_table.shape
    if out.shape[0] < rows or out.shape[1] < max_kernel_pages:
        raise ValueError(
            f"out shape {tuple(out.shape)} cannot hold ({rows}, {max_kernel_pages})"
        )
    if ratio < 1:
        raise ValueError(f"ratio must be positive, got {ratio}")
    if rows == 0 or out.shape[1] == 0:
        return out[:rows, :max_kernel_pages]

    out_cols = out.shape[1]
    # Fixed so a new table width cannot trigger a recompile.
    block = 256
    _expand_page_table_kernel[(rows, triton.cdiv(out_cols, block))](
        page_table,
        out,
        logical_cols,
        max_kernel_pages,
        page_table.stride(0),
        out.stride(0),
        out_cols,
        ratio,
        BLOCK=block,
    )
    return out[:rows, :max_kernel_pages]

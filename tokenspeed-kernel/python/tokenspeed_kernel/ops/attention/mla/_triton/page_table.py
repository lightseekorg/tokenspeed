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
from tokenspeed_kernel._triton import tl, triton


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


@triton.jit
def resolve_group_slot(
    Table,
    position,
    request,
    table_rows,
    table_cols,
    row_stride,
    col_stride,
    ROWS: tl.constexpr,
    STRIDE: tl.constexpr,
    first_page,
    page_count,
):
    """The shared raw-position -> group-slot mapping, with explicit bounds."""
    logical = position // STRIDE
    column = logical // ROWS
    valid = (
        (position >= 0)
        & (request >= 0)
        & (request < table_rows)
        & (column >= 0)
        & (column < table_cols)
    )
    page = tl.load(
        Table + request * row_stride + column * col_stride, valid, other=-1
    ).to(tl.int64)
    valid &= (page >= first_page) & (page < page_count)
    return tl.where(valid, page * ROWS + logical % ROWS, -1)


@triton.jit(do_not_specialize=["N"])
def _group_slots_kernel(
    P,
    R,
    Table,
    Out,
    N,
    W: tl.constexpr,
    PS0: tl.constexpr,
    PS1: tl.constexpr,
    RS0: tl.constexpr,
    RS1: tl.constexpr,
    TR,
    TC,
    TS0,
    TS1,
    ROWS: tl.constexpr,
    STRIDE: tl.constexpr,
    first_page,
    page_count,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pos = tl.load(P + i // W * PS0 + i % W * PS1, i < N, other=-1)
    req = tl.load(R + i // W * RS0 + i % W * RS1, i < N, other=-1)
    slot = resolve_group_slot(
        Table, pos, req, TR, TC, TS0, TS1, ROWS, STRIDE, first_page, page_count
    )
    tl.store(Out + i, slot, i < N)


def bounded_group_slots(
    positions: torch.Tensor,
    requests: torch.Tensor,
    table: torch.Tensor,
    rows_per_page: int,
    entry_stride: int,
    first_page: int,
    page_count: int,
) -> torch.Tensor:
    """Map equal-shaped 1-D/2-D positions and requests into int64 group slots.

    Supports strided inputs, including broadcast request rows. Table entries
    outside [first_page, page_count), invalid coordinates, and negative positions
    resolve to -1. The output has positions.shape; no cache bytes are accessed.
    """
    if positions.shape != requests.shape or positions.ndim not in (1, 2):
        raise ValueError("positions and requests require equal 1-D/2-D shapes")
    if rows_per_page <= 0 or entry_stride <= 0:
        raise ValueError("group row geometry must be positive")
    if not positions.is_cuda:
        logical = positions.to(torch.int64) // entry_stride
        col = logical // rows_per_page
        if not table.numel():
            return torch.full_like(positions, -1, dtype=torch.int64)
        page = table[
            requests.clamp(0, table.shape[0] - 1), col.clamp(0, table.shape[1] - 1)
        ].to(torch.int64)
        valid = (
            (positions >= 0)
            & (requests >= 0)
            & (requests < table.shape[0])
            & (col < table.shape[1])
            & (page >= first_page)
            & (page < page_count)
        )
        return (page * rows_per_page + logical % rows_per_page).masked_fill(~valid, -1)
    out = torch.empty(positions.shape, dtype=torch.int64, device=positions.device)
    if positions.numel():
        p = positions[:, None] if positions.ndim == 1 else positions
        r = requests[:, None] if requests.ndim == 1 else requests
        _group_slots_kernel[(triton.cdiv(positions.numel(), 256),)](
            p,
            r,
            table,
            out,
            positions.numel(),
            p.shape[1],
            *p.stride(),
            *r.stride(),
            *table.shape,
            *table.stride(),
            rows_per_page,
            entry_stride,
            first_page,
            page_count,
            BLOCK=256,
        )
    return out


@triton.jit(
    do_not_specialize=[
        "live_rows",
        "source_cols",
        "source_stride",
        "out_cols",
        "out_stride",
    ]
)
def _copy_page_table_kernel(
    Source,
    Out,
    live_rows,
    source_cols,
    source_stride,
    out_cols,
    out_stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(
        Source + row * source_stride + col,
        (row < live_rows) & (col < source_cols),
        other=0,
    )
    tl.store(Out + row * out_stride + col, value, col < out_cols)


def copy_page_table(source: torch.Tensor, out: torch.Tensor, live_rows: int) -> None:
    """Copy live rows into a persistent table, clearing padding and unused columns.

    source/out are int32 matrices with unit column stride. out may have a larger
    capacity; its entire extent is overwritten so stale cache pages cannot leak
    into a subsequent batch. The source is never changed and may not alias out.
    """
    if source.dtype != torch.int32 or out.dtype != torch.int32:
        raise ValueError("page tables must be int32")
    if source.stride(1) != 1 or out.stride(1) != 1:
        raise ValueError("page table columns must be contiguous")
    if (
        not 0 <= live_rows <= min(source.shape[0], out.shape[0])
        or source.shape[1] > out.shape[1]
    ):
        raise ValueError("page table copy exceeds capacity")
    if not out.is_cuda:
        out.zero_()
        out[:live_rows, : source.shape[1]].copy_(source[:live_rows])
        return
    if out.numel():
        _copy_page_table_kernel[(out.shape[0], triton.cdiv(out.shape[1], 256))](
            source,
            out,
            live_rows,
            source.shape[1],
            source.stride(0),
            out.shape[1],
            out.stride(0),
            BLOCK=256,
        )

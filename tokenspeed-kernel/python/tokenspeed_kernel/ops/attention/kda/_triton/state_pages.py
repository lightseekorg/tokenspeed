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

"""State pages a speculative verify reads and its commit writes.

A state group's block table holds two slots per request. Slot 0 is the input
checkpoint: a claimed prefix hit, the block the request's previous forward
wrote before the scheduler rotated it, or a fresh zeroed block when the
request starts from the zero state; null (``0``) once that block has been
consumed. Slot 1 is the live block the request's forwards write. A forward
reads slot 0 when it is non-null and slot 1 otherwise, and always writes
slot 1 -- verify reads the committed state from the input slot and the commit
writes the accepted state to the live slot.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import pdl_enabled

STATE_INPUT_SLOT = 0
STATE_LIVE_SLOT = 1
STATE_TABLE_SLOTS = 2


@triton.jit
def _state_input_page(table, row, table_stride, live):
    """Slot 0 when it holds a block, else slot 1 (in-place evolution)."""
    checkpoint = tl.load(table + row * table_stride, mask=live, other=0)
    working = tl.load(table + row * table_stride + 1, mask=live, other=0)
    return tl.where(checkpoint > 0, checkpoint, working)


@triton.jit(do_not_specialize=["bs", "table_stride"])
def _verify_state_blocks_kernel(
    table,
    pages_out,
    live_out,
    bs,
    table_stride,
    BLOCK: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = row < bs
    page = _state_input_page(table, row, table_stride, live)
    tl.store(pages_out + row, page.to(tl.int32), mask=live)
    if live_out is not None:
        working = tl.load(table + row * table_stride + 1, mask=live, other=0)
        tl.store(live_out + row, working.to(tl.int32), mask=live)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def _torch_verify_state_blocks(table, batch_size, pages_out, live_out):
    """Portable spelling, kept for non-CUDA callers (the backends' CPU tests)."""
    rows = table[:batch_size]
    checkpoint = rows[:, STATE_INPUT_SLOT]
    working = rows[:, STATE_LIVE_SLOT]
    pages_out[:batch_size].copy_(
        torch.where(checkpoint > 0, checkpoint, working).to(torch.int32)
    )
    if live_out is not None:
        live_out[:batch_size].copy_(working.to(torch.int32))


def _check_state_table(table: torch.Tensor, batch_size: int) -> None:
    if table.stride(1) != 1:
        raise ValueError("state block tables need unit-stride rows")
    if table.shape[1] < STATE_TABLE_SLOTS:
        raise ValueError(
            f"a state block table holds {STATE_TABLE_SLOTS} slots per request, "
            f"got {table.shape[1]}"
        )
    if batch_size > table.shape[0]:
        raise ValueError("batch_size exceeds the table rows")


def verify_state_blocks(
    table: torch.Tensor,
    *,
    batch_size: int,
    pages_out: torch.Tensor,
    live_out: torch.Tensor | None,
) -> None:
    """Resolve the committed-state page each request's verify reads.

    Args:
        table: Group block table ``[>=batch_size, 2]``, dense inner.
        batch_size: Live requests.
        pages_out: INT32 destination ``[>=batch_size]``.
        live_out: None, or an INT32 ``[>=batch_size]`` destination for each
            request's live page (slot 1), the page the post-verify commit
            writes (:func:`verify_commit_resolve`). The commit then reads it
            instead of this forward's block table.
    """
    enable_pdl = pdl_enabled()
    _check_state_table(table, batch_size)
    if pages_out.stride(0) != 1:
        raise ValueError("verify state blocks need unit-stride rows")
    if pages_out.dtype != torch.int32:
        raise ValueError("pages_out must be INT32")
    if pages_out.numel() < batch_size:
        raise ValueError("outputs cannot hold batch_size rows")
    if batch_size == 0:
        return
    if live_out is not None and (
        live_out.dtype != torch.int32
        or live_out.stride(0) != 1
        or live_out.numel() < batch_size
    ):
        raise ValueError("live_out must be unit-stride INT32 of >= batch_size rows")
    if not table.is_cuda:
        return _torch_verify_state_blocks(table, batch_size, pages_out, live_out)

    block = 256
    _verify_state_blocks_kernel[(triton.cdiv(batch_size, block),)](
        table,
        pages_out,
        live_out,
        batch_size,
        table.stride(0),
        BLOCK=block,
        ENABLE_PDL=enable_pdl,
        **({"launch_pdl": True} if enable_pdl else {}),
    )


@triton.jit(
    do_not_specialize=["bs", "draft_tokens", "table_stride", "out_row", "out_stride"]
)
def _commit_state_pages_kernel(
    accepted,
    table,
    pages_out,
    steps_out,
    bs,
    draft_tokens,
    table_stride,
    out_row,
    out_stride,
    BLOCK: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = row < bs
    # A live round always advances state by the target token, so it commits
    # at least one. A stopped request keeps none, but its row is padding
    # (live page -1), so the commit writes nothing for it.
    steps = tl.minimum(
        tl.maximum(tl.load(accepted + row, mask=live, other=1).to(tl.int64), 1),
        draft_tokens,
    )
    page = tl.load(table + row * table_stride + 1, mask=live, other=0)
    tl.store(pages_out + out_row * out_stride + row, page.to(tl.int32), mask=live)
    tl.store(steps_out + row, steps.to(tl.int32), mask=live)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def _torch_commit_state_pages(
    accepted_length, table, batch_size, draft_tokens, pages_out, out_row, steps_out
):
    """Portable spelling, kept for non-CUDA callers (the backends' CPU tests)."""
    steps = accepted_length[:batch_size].to(torch.int64).clamp(min=1, max=draft_tokens)
    pages_out[out_row, :batch_size].copy_(
        table[:batch_size, STATE_LIVE_SLOT].to(torch.int32)
    )
    steps_out[:batch_size].copy_(steps.to(torch.int32))


def commit_state_pages(
    accepted_length: torch.Tensor,
    table: torch.Tensor,
    *,
    batch_size: int,
    draft_tokens: int,
    pages_out: torch.Tensor,
    out_row: int,
    steps_out: torch.Tensor,
) -> None:
    """Resolve where each request's accepted state is written back: slot 1.

    Writing row ``out_row`` of a preallocated ``[groups, batch]`` buffer lets
    every group's resolve land in one stacked tensor without a later stack.

    Args:
        accepted_length: Draft matches per request ``[>=batch_size]``.
        table: Group block table ``[>=batch_size, 2]``, dense inner.
        batch_size: Live requests.
        draft_tokens: Speculative window; a round commits at most this many.
        pages_out: INT32 ``[groups, >=batch_size]`` destination.
        out_row: Which group row to fill.
        steps_out: INT32 destination ``[>=batch_size]`` for the clamped steps.
    """
    enable_pdl = pdl_enabled()
    _check_state_table(table, batch_size)
    if (
        accepted_length.stride(0) != 1
        or pages_out.stride(1) != 1
        or steps_out.stride(0) != 1
    ):
        raise ValueError("commit state pages need unit-stride rows")
    if draft_tokens < 1:
        raise ValueError(f"draft_tokens must be positive, got {draft_tokens}")
    if pages_out.dtype != torch.int32 or steps_out.dtype != torch.int32:
        raise ValueError("pages_out and steps_out must be INT32")
    if not 0 <= out_row < pages_out.shape[0]:
        raise ValueError(f"out_row {out_row} outside {pages_out.shape[0]} groups")
    if pages_out.shape[1] < batch_size or steps_out.numel() < batch_size:
        raise ValueError("outputs cannot hold batch_size rows")
    if batch_size > accepted_length.shape[0]:
        raise ValueError("batch_size exceeds the accepted rows")
    if batch_size == 0:
        return
    if not table.is_cuda:
        return _torch_commit_state_pages(
            accepted_length,
            table,
            batch_size,
            draft_tokens,
            pages_out,
            out_row,
            steps_out,
        )

    block = 256
    _commit_state_pages_kernel[(triton.cdiv(batch_size, block),)](
        accepted_length,
        table,
        pages_out,
        steps_out,
        batch_size,
        draft_tokens,
        table.stride(0),
        out_row,
        pages_out.stride(0),
        BLOCK=block,
        ENABLE_PDL=enable_pdl,
        **({"launch_pdl": True} if enable_pdl else {}),
    )


@triton.jit(do_not_specialize=["bs", "verify_width", "live_stride", "read_stride"])
def _verify_commit_resolve_kernel(
    accepted,
    live_pages,
    live_stride,
    read_pages,
    read_stride,
    group_indices,
    steps_out,
    src_rows,
    dst_rows,
    read_rows,
    bs,
    verify_width,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = request < bs
    layer = tl.program_id(1)
    # A live round always advances state by the target token, so it commits
    # at least one (see _commit_state_pages_kernel).
    steps = tl.minimum(
        tl.maximum(tl.load(accepted + request, mask=live, other=1).to(tl.int32), 1),
        verify_width,
    )
    if layer == 0:
        tl.store(steps_out + request, steps, mask=live)
    group = tl.load(group_indices + layer).to(tl.int64)
    page = tl.load(live_pages + group * live_stride + request, mask=live, other=0)
    read = tl.load(read_pages + group * read_stride + request, mask=live, other=0)
    out = layer.to(tl.int64) * bs + request
    tl.store(src_rows + out, request * (verify_width + 1) + steps, mask=live)
    tl.store(dst_rows + out, tl.where(page > 0, page, -1), mask=live)
    tl.store(read_rows + out, read, mask=live)


def verify_commit_resolve(
    accepted_length: torch.Tensor,
    live_pages: torch.Tensor,
    read_pages: torch.Tensor,
    group_indices: torch.Tensor,
    *,
    verify_width: int,
    steps_out: torch.Tensor,
    src_rows: torch.Tensor,
    dst_rows: torch.Tensor,
    read_rows: torch.Tensor,
) -> None:
    """Every row a chain verify's state commit needs, in one launch.

    For each request it clamps the accepted count to ``[1, verify_width]``;
    for each layer and request it emits the verify-scratch row of the last
    accepted token, the live page the commit writes (-1 for none) and the
    committed page the replay reads. It replaces :func:`commit_state_pages`
    per group, ``state_verify_commit_rows`` and the gather of every layer's
    read page.

    Args:
        accepted_length: ``[bs]`` accepted tokens of each request.
        live_pages: INT32 ``[groups, >= bs]`` live page of each request per
            state group (``verify_state_blocks(live_out=...)``).
        read_pages: INT32 ``[groups, >= bs]`` page each request's verify
            read per state group.
        group_indices: ``[layers]`` state group of each layer.
        verify_width: Verify tokens per request.
        steps_out: INT32 ``[bs]`` output: the clamped accepted counts.
        src_rows, dst_rows, read_rows: INT32 ``[layers * bs]`` outputs,
            layer major.
    """
    bs = accepted_length.shape[0]
    layers = group_indices.shape[0]
    for name, t in (("live_pages", live_pages), ("read_pages", read_pages)):
        if t.dtype != torch.int32 or t.ndim != 2 or t.stride(1) != 1:
            raise ValueError(f"{name} must be INT32 [groups, >= bs] with unit rows")
    for name, t in (
        ("steps_out", steps_out),
        ("src_rows", src_rows),
        ("dst_rows", dst_rows),
        ("read_rows", read_rows),
    ):
        if t.dtype != torch.int32 or not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous INT32")
    if bs == 0:
        return
    if not accepted_length.is_cuda:
        # Portable spelling, kept for non-CUDA callers (the backends' CPU tests).
        steps = accepted_length.to(torch.int64).clamp(min=1, max=verify_width)
        group = group_indices.to(torch.int64)
        live = live_pages[:, :bs].index_select(0, group)
        steps_out.copy_(steps.to(torch.int32))
        base = torch.arange(bs, dtype=torch.int64) * (verify_width + 1) + steps
        src_rows.copy_(base.repeat(group.numel()).to(torch.int32))
        dst_rows.copy_(torch.where(live > 0, live, -1).reshape(-1))
        read_rows.copy_(read_pages[:, :bs].index_select(0, group).reshape(-1))
        return
    block = 256
    _verify_commit_resolve_kernel[(triton.cdiv(bs, block), layers)](
        accepted_length,
        live_pages,
        live_pages.stride(0),
        read_pages,
        read_pages.stride(0),
        group_indices,
        steps_out,
        src_rows,
        dst_rows,
        read_rows,
        bs,
        verify_width,
        BLOCK=block,
    )


@triton.jit(
    do_not_specialize=["bs", "st0", "st1", "st2", "st3", "pages_stride", "live_stride"],
    do_not_specialize_on_alignment=["t0", "t1", "t2", "t3"],
)
def _verify_state_blocks_groups_kernel(
    t0,
    t1,
    t2,
    t3,
    st0,
    st1,
    st2,
    st3,
    pages_out,
    pages_stride,
    live_out,
    live_stride,
    bs,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = row < bs
    g = tl.program_id(1)
    table = tl.where(g == 0, t0, tl.where(g == 1, t1, tl.where(g == 2, t2, t3)))
    stride = tl.where(g == 0, st0, tl.where(g == 1, st1, tl.where(g == 2, st2, st3)))
    page = _state_input_page(table, row, stride, live)
    tl.store(pages_out + g * pages_stride + row, page.to(tl.int32), mask=live)
    working = tl.load(table + row * stride + 1, mask=live, other=0)
    tl.store(live_out + g * live_stride + row, working.to(tl.int32), mask=live)


def verify_state_blocks_groups(
    tables,
    *,
    batch_size: int,
    pages_out: torch.Tensor,
    live_out: torch.Tensor,
) -> None:
    """:func:`verify_state_blocks` (with ``live_out``) of one to four state
    groups in one launch.

    Args:
        tables: Each group's INT32 block table ``[>=batch_size, 2]``.
        batch_size: Live requests.
        pages_out, live_out: INT32 ``[groups, >=batch_size]`` with unit
            columns: row ``g`` receives group ``g``'s pages.
    """
    if not 1 <= len(tables) <= 4:
        raise ValueError(f"1 to 4 state groups per launch, got {len(tables)}")
    for table in tables:
        _check_state_table(table, batch_size)
        if table.dtype != torch.int32:
            raise ValueError("state tables must be INT32")
    for name, out in (("pages_out", pages_out), ("live_out", live_out)):
        if (
            out.dtype != torch.int32
            or out.stride(1) != 1
            or out.shape[0] < len(tables)
            or out.shape[1] < batch_size
        ):
            raise ValueError(f"{name} must be INT32 [groups, >= batch_size]")
    if batch_size == 0:
        return
    if not tables[0].is_cuda:
        for g, table in enumerate(tables):
            verify_state_blocks(
                table,
                batch_size=batch_size,
                pages_out=pages_out[g],
                live_out=live_out[g],
            )
        return
    pad = 4 - len(tables)
    block = 256
    _verify_state_blocks_groups_kernel[(triton.cdiv(batch_size, block), len(tables))](
        *tables,
        *([tables[0]] * pad),
        *(table.stride(0) for table in tables),
        *([0] * pad),
        pages_out,
        pages_out.stride(0),
        live_out,
        live_out.stride(0),
        batch_size,
        BLOCK=block,
    )

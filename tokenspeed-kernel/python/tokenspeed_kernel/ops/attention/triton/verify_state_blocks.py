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

"""Committed-state pages for a speculative verify and its commit."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton


@triton.jit(
    do_not_specialize=["bs", "num_slots", "draft_tokens", "granularity", "table_stride"]
)
def _verify_state_blocks_kernel(
    seq_lens,
    table,
    pages_out,
    committed_out,
    bs,
    num_slots,
    draft_tokens,
    granularity,
    table_stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = row < bs
    committed = tl.maximum(
        tl.load(seq_lens + row, mask=live, other=0).to(tl.int64) - draft_tokens, 0
    )
    slot = tl.minimum(tl.maximum(committed - 1, 0) // granularity, num_slots - 1)
    page = tl.load(table + row * table_stride + slot, mask=live, other=0)
    # A request with no committed history has no state page to read.
    tl.store(pages_out + row, tl.where(committed > 0, page, -1).to(tl.int32), mask=live)
    tl.store(committed_out + row, committed, mask=live)


def _torch_verify_state_blocks(
    seq_lens, table, batch_size, draft_tokens, granularity, pages_out, committed_out
):
    """Portable spelling, kept for non-CUDA callers (the backends' CPU tests)."""
    committed = (seq_lens[:batch_size].to(torch.int64) - draft_tokens).clamp_min(0)
    slots = torch.div(
        (committed - 1).clamp_min(0), granularity, rounding_mode="floor"
    ).clamp(min=0, max=table.shape[1] - 1)
    pages = table[:batch_size].gather(1, slots.unsqueeze(1)).squeeze(1)
    pages_out[:batch_size].copy_(
        torch.where(committed > 0, pages, torch.full_like(pages, -1)).to(torch.int32)
    )
    committed_out[:batch_size].copy_(committed)


def verify_state_blocks(
    seq_lens: torch.Tensor,
    table: torch.Tensor,
    *,
    batch_size: int,
    draft_tokens: int,
    granularity: int,
    pages_out: torch.Tensor,
    committed_out: torch.Tensor,
) -> None:
    """Resolve each request's committed-state page and committed length.

    Verify reads state at the last committed position, which the torch spelling
    derives with a dozen elementwise launches (cast, sub, clamp, sub, clamp,
    div, compare, clamp, gather, full, where, cast) over ``[batch_size]``
    integers. The whole derivation is row-local.

    Args:
        seq_lens: Integer sequence lengths ``[>=batch_size]``.
        table: Group page table ``[>=batch_size, num_slots]``, dense inner.
        batch_size: Live requests.
        draft_tokens: Speculative window subtracted to reach the committed tip.
        granularity: Positions per checkpoint slot.
        pages_out: INT32 destination ``[>=batch_size]``; -1 where a request has
            no committed history.
        committed_out: INT64 destination ``[>=batch_size]``.
    """
    if (
        seq_lens.stride(0) != 1
        or table.stride(1) != 1
        or pages_out.stride(0) != 1
        or committed_out.stride(0) != 1
    ):
        raise ValueError("verify state blocks need unit-stride rows")
    if granularity < 1:
        raise ValueError(f"granularity must be positive, got {granularity}")
    if batch_size > table.shape[0] or batch_size > seq_lens.shape[0]:
        raise ValueError("batch_size exceeds the seq_lens or table rows")
    if table.shape[1] < 1:
        raise ValueError("a table without slots has no page to resolve")
    if pages_out.dtype != torch.int32 or committed_out.dtype != torch.int64:
        raise ValueError("pages_out must be INT32 and committed_out INT64")
    if pages_out.numel() < batch_size or committed_out.numel() < batch_size:
        raise ValueError("outputs cannot hold batch_size rows")
    if batch_size == 0:
        return
    if not table.is_cuda:
        return _torch_verify_state_blocks(
            seq_lens,
            table,
            batch_size,
            draft_tokens,
            granularity,
            pages_out,
            committed_out,
        )

    block = 256
    _verify_state_blocks_kernel[(triton.cdiv(batch_size, block),)](
        seq_lens,
        table,
        pages_out,
        committed_out,
        batch_size,
        table.shape[1],
        draft_tokens,
        granularity,
        table.stride(0),
        BLOCK=block,
    )


@triton.jit(
    do_not_specialize=[
        "bs",
        "num_slots",
        "draft_tokens",
        "granularity",
        "table_stride",
        "out_row",
        "out_stride",
    ]
)
def _commit_state_pages_kernel(
    accepted,
    committed,
    table,
    pages_out,
    steps_out,
    bs,
    num_slots,
    draft_tokens,
    granularity,
    table_stride,
    out_row,
    out_stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = row < bs
    # The target token always advances state, so a round commits at least one.
    steps = tl.minimum(
        tl.maximum(tl.load(accepted + row, mask=live, other=1).to(tl.int64), 1),
        draft_tokens,
    )
    last = tl.load(committed + row, mask=live, other=0) + steps - 1
    slot = tl.minimum(tl.maximum(last, 0) // granularity, num_slots - 1)
    page = tl.load(table + row * table_stride + slot, mask=live, other=0)
    tl.store(pages_out + out_row * out_stride + row, page.to(tl.int32), mask=live)
    tl.store(steps_out + row, steps.to(tl.int32), mask=live)


def _torch_commit_state_pages(
    accepted_length,
    committed,
    table,
    batch_size,
    draft_tokens,
    granularity,
    pages_out,
    out_row,
    steps_out,
):
    """Portable spelling, kept for non-CUDA callers (the backends' CPU tests)."""
    steps = accepted_length[:batch_size].to(torch.int64).clamp(min=1, max=draft_tokens)
    last = committed[:batch_size] + steps - 1
    slots = torch.div(last.clamp_min(0), granularity, rounding_mode="floor").clamp(
        min=0, max=table.shape[1] - 1
    )
    pages = table[:batch_size].gather(1, slots.unsqueeze(1)).squeeze(1)
    pages_out[out_row, :batch_size].copy_(pages.to(torch.int32))
    steps_out[:batch_size].copy_(steps.to(torch.int32))


def commit_state_pages(
    accepted_length: torch.Tensor,
    committed: torch.Tensor,
    table: torch.Tensor,
    *,
    batch_size: int,
    draft_tokens: int,
    granularity: int,
    pages_out: torch.Tensor,
    out_row: int,
    steps_out: torch.Tensor,
) -> None:
    """Resolve where each request's accepted state is written back.

    The commit's torch spelling took a dozen elementwise launches to turn accept
    lengths into per-group write pages, then stacked the groups. Writing row
    ``out_row`` of a preallocated ``[groups, batch]`` buffer skips the stack too.

    Args:
        accepted_length: Draft matches per request ``[>=batch_size]``.
        committed: Committed lengths from the verify resolve ``[>=batch_size]``.
        table: Group page table ``[>=batch_size, num_slots]``, dense inner.
        batch_size: Live requests.
        draft_tokens: Speculative window; a round commits at most this many.
        granularity: Positions per checkpoint slot.
        pages_out: INT32 ``[groups, >=batch_size]`` destination.
        out_row: Which group row to fill.
        steps_out: INT32 destination ``[>=batch_size]`` for the clamped steps.
    """
    if (
        accepted_length.stride(0) != 1
        or committed.stride(0) != 1
        or table.stride(1) != 1
        or pages_out.stride(1) != 1
        or steps_out.stride(0) != 1
    ):
        raise ValueError("commit state pages need unit-stride rows")
    if granularity < 1:
        raise ValueError(f"granularity must be positive, got {granularity}")
    if pages_out.dtype != torch.int32 or steps_out.dtype != torch.int32:
        raise ValueError("pages_out and steps_out must be INT32")
    if not 0 <= out_row < pages_out.shape[0]:
        raise ValueError(f"out_row {out_row} outside {pages_out.shape[0]} groups")
    if pages_out.shape[1] < batch_size or steps_out.numel() < batch_size:
        raise ValueError("outputs cannot hold batch_size rows")
    if batch_size > table.shape[0] or batch_size > committed.shape[0]:
        raise ValueError("batch_size exceeds the committed or table rows")
    if table.shape[1] < 1:
        raise ValueError("a table without slots has no page to resolve")
    if batch_size == 0:
        return
    if not table.is_cuda:
        return _torch_commit_state_pages(
            accepted_length,
            committed,
            table,
            batch_size,
            draft_tokens,
            granularity,
            pages_out,
            out_row,
            steps_out,
        )

    block = 256
    _commit_state_pages_kernel[(triton.cdiv(batch_size, block),)](
        accepted_length,
        committed,
        table,
        pages_out,
        steps_out,
        batch_size,
        table.shape[1],
        draft_tokens,
        granularity,
        table.stride(0),
        out_row,
        pages_out.stride(0),
        BLOCK=block,
    )

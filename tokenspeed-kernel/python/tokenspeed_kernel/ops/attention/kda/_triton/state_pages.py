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
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def _torch_verify_state_blocks(table, batch_size, pages_out):
    """Portable spelling, kept for non-CUDA callers (the backends' CPU tests)."""
    rows = table[:batch_size]
    checkpoint = rows[:, STATE_INPUT_SLOT]
    working = rows[:, STATE_LIVE_SLOT]
    pages_out[:batch_size].copy_(
        torch.where(checkpoint > 0, checkpoint, working).to(torch.int32)
    )


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
) -> None:
    """Resolve the committed-state page each request's verify reads.

    Args:
        table: Group block table ``[>=batch_size, 2]``, dense inner.
        batch_size: Live requests.
        pages_out: INT32 destination ``[>=batch_size]``.
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
    if not table.is_cuda:
        return _torch_verify_state_blocks(table, batch_size, pages_out)

    block = 256
    _verify_state_blocks_kernel[(triton.cdiv(batch_size, block),)](
        table,
        pages_out,
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
    # The target token always advances state, so a round commits at least one.
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

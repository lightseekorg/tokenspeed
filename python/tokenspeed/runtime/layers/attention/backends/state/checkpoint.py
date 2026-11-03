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

"""Two-slot state block tables shared by recurrent and PLE state consumers.

A state group's block table holds two slots per request:

* slot 0, the **input checkpoint**: the claimed prefix hit, the block the
  request's previous forward wrote before the scheduler rotated it, or a
  fresh zeroed block when the request starts from the zero state; null
  (``0``) once that block has been consumed;
* slot 1, the **live block**: the block this request's forwards write.

A forward reads slot 0 when it is non-null and slot 1 otherwise (in-place
evolution), and always writes slot 1. A published checkpoint is never written:
it only ever appears in slot 0, and slot 0 is never the write target. A
request without history reads no block at all, whatever its row holds.

A padding row (``-1`` in both slots: a graph padding row, or a request the
device stopped while a forward behind it was already dispatched) reads the
zero state and writes nothing; every state kernel skips negative blocks.
The debug checks below cannot tell such a row from one corrupted to
``(-1, -1)``, so they skip it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    STATE_CHECKPOINT_SLOT,
    STATE_INPUT_SLOT,
    STATE_LIVE_SLOT,
    STATE_TABLE_SLOTS,
)

__all__ = [
    "STATE_CHECKPOINT_SLOT",
    "STATE_INPUT_SLOT",
    "STATE_LIVE_SLOT",
    "STATE_TABLE_SLOTS",
    "InForwardStateCheckpoints",
    "compute_state_block_indices",
    "in_forward_state_checkpoints",
    "scan_checkpoint_plan",
]


@dataclass(frozen=True)
class InForwardStateCheckpoints:
    """The in-forward State checkpoints one extend forward writes.

    The scheduler gives an extend row a checkpoint when its chunk spans the
    page boundary before a long prompt's replay tail: besides its final state
    (slot 1), every State group of that row stores the state after the
    chunk's first ``offsets[i]`` tokens into the row's third block (slot 2),
    which the scheduler publishes as a prefix-cache checkpoint.

    Args:
        rows: Host int64 batch rows that write a checkpoint.
        offsets: Host int64 tokens of each such row's chunk before its
            checkpoint (a positive multiple of the prefix page).
        token_index: Device int64 packed-input index of the first token after
            each checkpoint (the row's first token plus its offset).
        blocks_by_group: Per State group, the device int32 receiving block of
            each checkpoint.
    """

    rows: torch.Tensor
    offsets: torch.Tensor
    token_index: torch.Tensor
    blocks_by_group: dict[str, torch.Tensor]


def in_forward_state_checkpoints(
    checkpoint_lens_cpu: torch.Tensor | None,
    extend_seq_lens_cpu: torch.Tensor,
    num_extends: int,
    block_tables: Mapping[str, torch.Tensor],
    group_ids: Sequence[str],
) -> InForwardStateCheckpoints | None:
    """Resolve this forward's in-forward State checkpoints, if any.

    Args:
        checkpoint_lens_cpu: Host ``[>= num_extends]`` tokens of each extend
            row's chunk before its checkpoint (0 for none), or ``None`` when
            the forward path delivers none.
        extend_seq_lens_cpu: Host ``[>= num_extends]`` new tokens per extend
            row, which pack the input in row order.
        num_extends: Leading extend rows.
        block_tables: The forward's raw per-group block tables.
        group_ids: The State groups the caller writes.

    Returns:
        The checkpoints, or ``None`` when no row writes one.

    Raises:
        RuntimeError: A State table carries checkpoint blocks but the forward
            path dropped their positions (the scheduler would publish blocks
            nobody wrote), a checkpointed row's table lacks one, or a
            checkpoint lies at or past the end of its chunk.
    """
    rows = torch.empty(0, dtype=torch.int64)
    if checkpoint_lens_cpu is not None:
        lens = checkpoint_lens_cpu[:num_extends].to(torch.int64)
        rows = torch.nonzero(lens > 0).flatten()
    if rows.numel() == 0:
        # The scheduler appends a third entry only for a row it gave a
        # checkpoint, so one without positions means the forward path lost
        # them.
        for group_id in group_ids:
            if block_tables[group_id].shape[1] > STATE_CHECKPOINT_SLOT:
                raise RuntimeError(
                    f"state paging: {group_id!r} rows carry in-forward checkpoint "
                    "blocks, but this forward path delivered no checkpoint positions"
                )
        return None
    offsets = lens[rows]
    seq_lens = extend_seq_lens_cpu[:num_extends].to(torch.int64)
    past_end = offsets >= seq_lens[rows]
    if bool(past_end.any()):
        row = int(rows[past_end][0])
        raise RuntimeError(
            f"state paging: extend row {row}'s in-forward checkpoint lies at or "
            f"past the end of its chunk (offset {int(lens[row])}, chunk "
            f"{int(seq_lens[row])} tokens)"
        )
    starts = torch.cumsum(seq_lens, 0) - seq_lens
    blocks_by_group: dict[str, torch.Tensor] = {}
    device = None
    for group_id in group_ids:
        table = block_tables[group_id]
        if table.shape[1] <= STATE_CHECKPOINT_SLOT:
            raise RuntimeError(
                f"state paging: {group_id!r} rows lack the in-forward checkpoint block"
            )
        device = table.device
        blocks_by_group[group_id] = (
            table[rows.to(table.device), STATE_CHECKPOINT_SLOT]
            .to(torch.int32)
            .contiguous()
        )
    return InForwardStateCheckpoints(
        rows=rows,
        offsets=offsets,
        token_index=(starts[rows] + offsets).to(device),
        blocks_by_group=blocks_by_group,
    )


def scan_checkpoint_plan(
    checkpoints: InForwardStateCheckpoints, scan_seq_lens_cpu: torch.Tensor
) -> tuple[int, torch.Tensor, torch.Tensor]:
    """Where a chunked scan stores this forward's in-forward checkpoints.

    A scan stores states at one interval for every sequence it covers, the
    ``j``-th of sequence ``i`` (0-based) after ``(j + 1) * interval`` of its
    tokens at row ``cu_starts[i] + j``. The interval is the checkpoint
    offsets' greatest common divisor, so each checkpoint is one of those rows
    and the scan stores as few others as one interval allows.

    Args:
        checkpoints: This forward's checkpoints.
        scan_seq_lens_cpu: Host ``[bs]`` tokens of each scanned sequence, in
            batch-row order.

    Returns:
        ``(interval, host int64 [bs + 1] cumulative state counts, host int64
        buffer row of each checkpoint)``.
    """
    interval = math.gcd(*(int(x) for x in checkpoints.offsets.tolist()))
    counts = scan_seq_lens_cpu.to(torch.int64) // interval
    cu_starts = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0)))
    rows = cu_starts[checkpoints.rows] + checkpoints.offsets // interval - 1
    return interval, cu_starts, rows


def compute_state_block_indices(
    rows: torch.Tensor,
    seq_lens_before: torch.Tensor,
    *,
    validate: bool,
    group_id: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve each request's (input, output) state blocks for one forward.

    Args:
        rows: ``[bs, 2]`` int32 block-id table of one state group (``-1``
            pad, ``0`` null). Wider rows are accepted and read at slots 0/1.
        seq_lens_before: Per-request token count before this forward; ``0``
            means the request starts from the zero state and reads no block.
        validate: Run the host-synchronizing write-side checks.
        group_id: State group the table belongs to; only used to attribute
            validation errors (multi-group KDA runs this once per group).

    Returns:
        ``(state_in, state_out)`` int32 block ids per request; ``state_in``
        is ``0`` for a request without history.
    """
    bs = seq_lens_before.shape[0]
    rows = rows[:bs]
    if rows.shape[1] < STATE_TABLE_SLOTS:
        raise ValueError(
            f"state paging: {group_id!r} table has {rows.shape[1]} slots; a state "
            f"group holds {STATE_TABLE_SLOTS} per request (input checkpoint, live)"
        )
    checkpoint = rows[:, STATE_INPUT_SLOT]
    live = rows[:, STATE_LIVE_SLOT]
    has_history = seq_lens_before > 0
    state_in = torch.where(
        has_history,
        torch.where(checkpoint > 0, checkpoint, live),
        torch.zeros_like(live),
    )
    # The kernels take unit-stride block vectors; a table column is not one,
    # and contiguous() leaves a one-row column at the table's stride, which
    # the kernels would compile per table width.
    state_out = torch.empty_like(live, memory_format=torch.contiguous_format).copy_(
        live
    )

    if validate:
        real = (checkpoint != -1) | (live != -1)
        if bool((state_in[has_history & real] <= 0).any()):
            raise ValueError(
                "state paging: the input block is a pad (-1) or hole (0) for a "
                "request with history; reading it would silently resume from "
                f"the zero state ({group_id!r} table)"
            )
        if bool((state_out[real] <= 0).any()):
            raise ValueError(
                "state paging: the live block is a hole (0); the request's "
                f"working state block must be present in the {group_id!r} table"
            )
        # A claimed checkpoint is read-only; the forward must write elsewhere.
        if bool((checkpoint[checkpoint > 0] == live[checkpoint > 0]).any()):
            raise ValueError(
                "state paging: a request's input checkpoint and live block are "
                "the same block; writing it would corrupt every branch sharing "
                f"the checkpoint ({group_id!r} table)"
            )
        if torch.unique(state_out[real]).numel() != int(real.sum()):
            raise ValueError(
                f"state out pages must be unique per batch ({group_id!r} "
                "table): two requests writing one working state page would "
                "silently clobber each other"
            )
    return state_in.to(torch.int32), state_out.to(torch.int32)

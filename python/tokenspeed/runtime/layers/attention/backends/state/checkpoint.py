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
"""

from __future__ import annotations

import torch

from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    STATE_INPUT_SLOT,
    STATE_LIVE_SLOT,
    STATE_TABLE_SLOTS,
)

__all__ = [
    "STATE_INPUT_SLOT",
    "STATE_LIVE_SLOT",
    "STATE_TABLE_SLOTS",
    "compute_state_block_indices",
]


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
    # The kernels take unit-stride block vectors; a table column is not one.
    state_out = live.contiguous()

    if validate:
        if bool((state_in[has_history] <= 0).any()):
            raise ValueError(
                "state paging: the input block is a pad (-1) or hole (0) for a "
                "request with history; reading it would silently resume from "
                f"the zero state ({group_id!r} table)"
            )
        if bool((state_out <= 0).any()):
            raise ValueError(
                "state paging: the live block is a pad (-1) or hole (0); the "
                f"request's working state block must be present in the {group_id!r} "
                "table"
            )
        # A claimed checkpoint is read-only; the forward must write elsewhere.
        if bool((checkpoint[checkpoint > 0] == live[checkpoint > 0]).any()):
            raise ValueError(
                "state paging: a request's input checkpoint and live block are "
                "the same block; writing it would corrupt every branch sharing "
                f"the checkpoint ({group_id!r} table)"
            )
        if torch.unique(state_out).numel() != state_out.numel():
            raise ValueError(
                f"state out pages must be unique per batch ({group_id!r} "
                "table): two requests writing one working state page would "
                "silently clobber each other"
            )
    return state_in.to(torch.int32), state_out.to(torch.int32)

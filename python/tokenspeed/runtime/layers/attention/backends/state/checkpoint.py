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

"""Checkpoint addressing shared by recurrent and PLE state consumers."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class _StateBlockIndexPlan:
    checkpoint_granularity: int
    before: torch.Tensor
    after: torch.Tensor
    has_history: torch.Tensor
    in_slots: torch.Tensor
    out_slots: torch.Tensor


def _compute_state_block_index_plan(
    checkpoint_granularity: int,
    seq_lens_before: torch.Tensor,
    seq_lens_after: torch.Tensor,
) -> _StateBlockIndexPlan:
    before = seq_lens_before
    after = seq_lens_after
    in_slots = torch.div(
        before - 1, checkpoint_granularity, rounding_mode="floor"
    ).clamp_(min=0)
    out_slots = torch.div(after - 1, checkpoint_granularity, rounding_mode="floor")
    return _StateBlockIndexPlan(
        checkpoint_granularity=checkpoint_granularity,
        before=before,
        after=after,
        has_history=before > 0,
        in_slots=in_slots,
        out_slots=out_slots,
    )


def _gather_state_block_indices(
    rows: torch.Tensor,
    plan: _StateBlockIndexPlan,
    *,
    out_slots_safe: torch.Tensor | None,
    validate: bool,
    group_id: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    bs = plan.before.shape[0]
    rows = rows[:bs]
    max_slots = rows.shape[1]
    if out_slots_safe is None:
        out_slots_safe = plan.out_slots.clamp(min=0, max=max_slots - 1)

    state_in = rows.gather(1, plan.in_slots.unsqueeze(1)).squeeze(1)
    state_in = torch.where(plan.has_history, state_in, torch.zeros_like(state_in))
    state_out = rows.gather(1, out_slots_safe.unsqueeze(1)).squeeze(1)

    if validate:
        if bool((plan.after <= 0).any()):
            raise ValueError(
                "state paging: seq_lens_after must be >= 1 for every request"
            )
        if bool((plan.out_slots >= max_slots).any()):
            raise ValueError(
                "state paging: out page slot exceeds table width "
                f"{max_slots} (checkpoint_granularity="
                f"{plan.checkpoint_granularity})"
            )
        if bool((state_in[plan.has_history] <= 0).any()):
            raise ValueError(
                "state paging: in page is a pad (-1) or hole (0) for a "
                "request with history; reading it would silently resume "
                f"from the zero state ({group_id!r} table)"
            )
        if bool((state_out <= 0).any()):
            raise ValueError(
                "state paging: out page is a pad (-1) or hole (0); the "
                "request's working state page must be present in the "
                f"{group_id!r} table"
            )
        # A step that crosses a page boundary or resumes from a prefix hit
        # reads a page that is, or becomes, a read-only snapshot. It must write
        # a different page.
        # in == out is legal only for in-place evolution inside one page.
        crossing = plan.has_history & (plan.in_slots != out_slots_safe)
        if bool((state_in[crossing] == state_out[crossing]).any()):
            raise ValueError(
                "state paging: a boundary-crossing or prefix-resuming step "
                "resolves the same page for input and output; the input "
                "page is a read-only prefix snapshot and writing it would "
                f"corrupt every branch sharing it ({group_id!r} table)"
            )
        # The <= 0 raise above guarantees every state_out entry is positive.
        if torch.unique(state_out).numel() != state_out.numel():
            raise ValueError(
                f"state out pages must be unique per batch ({group_id!r} "
                "table): two requests writing one working state page would "
                "silently clobber each other"
            )
    return state_in.to(torch.int32), state_out.to(torch.int32)


def compute_state_block_indices(
    rows: torch.Tensor,
    checkpoint_granularity: int,
    seq_lens_before: torch.Tensor,
    seq_lens_after: torch.Tensor,
    *,
    validate: bool,
    group_id: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dual-index state pages: in = slot of position n-1 (0/null when no
    history), out = slot of the step's last position. rows: [bs, max_slots]
    int32 page ids (-1 pad, 0 hole). Within a slot in == out (in-place
    evolution); crossing a checkpoint boundary reads the old slot and writes
    the new one; resuming from a prefix hit reads the claimed snapshot slot
    and writes the fresh working slot.

    Args:
        rows: ``[bs, max_slots]`` int32 page-id table of one state group.
        checkpoint_granularity: Token span of a state-block slot (``g``),
            independent of the prefix identity granularity.
        seq_lens_before: Per-request token count before this forward.
        seq_lens_after: Per-request token count after this forward.
        validate: Run the host-synchronizing write-side checks.
        group_id: State group the table belongs to; only used to attribute
            validation errors (multi-group KDA runs this once per group).

    Returns:
        ``(state_in, state_out)`` int32 page ids per request.
    """
    plan = _compute_state_block_index_plan(
        checkpoint_granularity, seq_lens_before, seq_lens_after
    )
    return _gather_state_block_indices(
        rows, plan, out_slots_safe=None, validate=validate, group_id=group_id
    )


def verified_state_block_slots(
    committed: torch.Tensor,
    accepted_steps: torch.Tensor,
    checkpoint_granularity: int,
) -> torch.Tensor:
    """Return destination slots for an already-clamped accepted prefix.

    Args:
        committed: Token counts before verification, one per live request.
        accepted_steps: Accepted target steps, clamped to the verify width.
        checkpoint_granularity: Logical tokens represented by one checkpoint.

    Returns:
        Int64 checkpoint slot indices, shared by groups with equal granularity.
    """
    bs = accepted_steps.shape[0]
    last = committed[:bs] + accepted_steps - 1
    return torch.div(last, checkpoint_granularity, rounding_mode="floor")


def gather_verified_state_blocks(
    rows: torch.Tensor, slots: torch.Tensor
) -> torch.Tensor:
    """Resolve accepted checkpoint slots in one group's raw block table.

    Args:
        rows: Raw scheduler block table for one checkpoint group.
        slots: Accepted checkpoint slot indices, one per live request.

    Returns:
        Int64 destination block ids; negative padding ids become null blocks.
    """
    bs = slots.shape[0]
    slots = slots.clamp(min=0, max=rows.shape[1] - 1)
    return (
        rows[:bs].gather(1, slots.unsqueeze(1)).squeeze(1).to(torch.int64).clamp_min(0)
    )

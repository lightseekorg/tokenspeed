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

"""Explicit collective reconstruction of bounded MLA prefill history."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from tokenspeed.runtime.distributed.comm_ops import token_all_gather
from tokenspeed.runtime.layers.attention.dcp.comm import gather_owned_rows
from tokenspeed.runtime.layers.attention.dcp.placement import (
    CachePlacement,
    cyclic_slot_owner,
    resolve_cache_slots,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.kv_cache.mla import MLATokenToKVPool
    from tokenspeed.runtime.layers.paged_attention import PagedAttention


def gather_mla_history(
    pool: MLATokenToKVPool,
    layer: PagedAttention,
    loc: torch.Tensor,
    *,
    dst_dtype: torch.dtype,
    placement: CachePlacement,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct only the requested virtual rows, in their original order.

    All ranks supply identical loc tensors. Each row has exactly one owner;
    nonowners contribute zero, including when their dummy page contains NaNs.
    The caller bounds loc to its prefill history chunk, not the entire arena.
    """
    slots, owned = resolve_cache_slots(loc, placement)
    nope, rope = pool.get_mla_kv_buffer(layer, slots, dst_dtype)
    values = gather_owned_rows(torch.cat((nope, rope), dim=-1), owned, placement.group)
    return (
        values[..., : pool.kv_lora_rank].contiguous(),
        values[..., pool.kv_lora_rank :].contiguous(),
    )


@dataclass(frozen=True)
class HistoryGatherPlan:
    """How one request group's history rows split over the page owners.

    Built once per forward and group from the virtual slots of every history
    row (prefix plus this chunk, request-major, position order) and the
    host-side owned-row counts; reused by every layer's KV and index-K
    gathers. Rank ``k``'s rows are the ``owned_rows_per_rank[k]`` rows whose
    owner is ``k``, in position order, so the all-gathered buffer is
    rank-major and ``order`` maps it back to position order.

    Attributes:
        virtual_slots: ``[rows]`` int64 virtual cache slots in position order.
        local_slots: ``[rows]`` int64 this rank's physical slots (0 where
            another rank owns the row).
        order: ``[rows]`` int64 permutation: position of the i-th row of the
            rank-major gathered buffer.
        owned_rows_per_rank: Host counts of rows each owner contributes.
        group: The owner group the gather runs over (a singleton without
            page sharding).
        rank: This rank's position in ``group``.
    """

    virtual_slots: torch.Tensor
    local_slots: torch.Tensor
    order: torch.Tensor
    owned_rows_per_rank: tuple[int, ...]
    group: tuple[int, ...]
    rank: int

    @property
    def rows(self) -> int:
        return sum(self.owned_rows_per_rank)

    @property
    def local_fetch_slots(self) -> torch.Tensor:
        """Physical slots of the rows this rank contributes, position order."""
        first = sum(self.owned_rows_per_rank[: self.rank])
        mine = self.order[first : first + self.owned_rows_per_rank[self.rank]]
        return self.local_slots.index_select(0, mine)


def plan_history_gather(
    virtual_slots: torch.Tensor,
    *,
    placement: CachePlacement | None,
    owned_rows_per_rank: Sequence[int],
) -> HistoryGatherPlan:
    """Plan the owner-split gather of a request group's history rows.

    Args:
        virtual_slots: ``[rows]`` int64 virtual slots of every history row,
            position order, identical on every rank.
        placement: The group's page ownership, or ``None`` when this rank
            holds every row (the gather is then a local slot gather).
        owned_rows_per_rank: Host counts of rows each owner contributes
            (``owned_history_rows`` summed over the group's requests); one
            entry without a placement.

    Returns:
        The plan; no device synchronization.
    """
    counts = tuple(int(count) for count in owned_rows_per_rank)
    if placement is None:
        if len(counts) != 1:
            raise ValueError("an unsharded history gather has one owner")
        group, rank = (0,), 0
    else:
        group, rank = placement.group, placement.rank
        if len(counts) != len(group):
            raise ValueError(
                f"owned_rows_per_rank names {len(counts)} owners for a group of "
                f"{len(group)}"
            )
    if sum(counts) != virtual_slots.numel():
        raise ValueError(
            f"owned rows sum to {sum(counts)} but the history has "
            f"{virtual_slots.numel()} rows"
        )
    local_slots, _owned = resolve_cache_slots(virtual_slots, placement)
    owner = cyclic_slot_owner(virtual_slots, placement)
    order = torch.argsort(owner, stable=True)
    return HistoryGatherPlan(
        virtual_slots=virtual_slots,
        local_slots=local_slots,
        order=order,
        owned_rows_per_rank=counts,
        group=group,
        rank=rank,
    )


def gather_history_rows(
    plan: HistoryGatherPlan,
    local_rows: torch.Tensor,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    """Assemble a request group's history rows in position order.

    Every rank contributes the rows it owns (``plan.local_fetch_slots`` of its
    own cache plane, as 2-D rows); one all-gather with per-rank counts puts
    them rank-major, and the plan's order scatters them into position order.
    Pure data movement: a row's bytes are the owner's bytes wherever the
    gather lands them. Without page sharding the rows are already local and
    only the scatter runs.

    Args:
        plan: The group's gather plan.
        local_rows: ``[owned_rows_per_rank[rank], width]`` rows this rank
            contributes, position order.
        out: ``[>= rows, width]`` destination; the leading ``rows`` are written.

    Returns:
        ``out[:rows]``.
    """
    rows = plan.rows
    if (
        local_rows.dim() != 2
        or local_rows.shape[0] != plan.owned_rows_per_rank[plan.rank]
    ):
        raise ValueError(
            f"rank {plan.rank} contributes {plan.owned_rows_per_rank[plan.rank]} "
            f"rows, got {tuple(local_rows.shape)}"
        )
    if out.dim() != 2 or out.shape[0] < rows or out.shape[1] != local_rows.shape[1]:
        raise ValueError(
            f"history destination {tuple(out.shape)} does not hold {rows} rows of "
            f"{local_rows.shape[1]}"
        )
    if len(plan.group) == 1:
        gathered = local_rows
    else:
        gathered = token_all_gather(
            local_rows.contiguous(), plan.group, list(plan.owned_rows_per_rank)
        )
    target = out[:rows]
    target.index_copy_(0, plan.order, gathered)
    return target


def history_gather_workspace_rows(max_model_len: int) -> int:
    """Rows the gathered-history workspace holds: one request's whole history."""
    if max_model_len <= 0:
        raise ValueError("history workspace needs a positive max_model_len")
    return int(max_model_len)

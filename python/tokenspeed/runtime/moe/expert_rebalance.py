# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
#
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

"""Online expert rebalancing's data-plane values: the slot move plan and specs.

The control plane (``ts-control``'s ``eplb`` module) decides when a rebalance
runs and derives the placement; the device moves the weights with the
``SlotMoves`` it hands over and reports the ``ExpertRebalanceSpecs`` the
control plane rebalances with.

The move plan is a pure function of ``(old_row, new_row)`` and the rank
topology, so it is identical on every rank: sends mirror receives by
construction and the P2P batch pairs without any further agreement.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

__all__ = [
    "ExpertRebalanceSpecs",
    "SlotMoves",
    "plan_slot_moves",
]

# -------------------------------- the move plan -------------------------------------


@dataclass(frozen=True)
class SlotMoves:
    """One rank's part of one layer's weight redistribution.

    Slots are local indices in ``[0, num_local)``; ranks are EP ranks.

    Attributes:
        recv: ``(dst_slot, src_rank)`` P2P receives into the staging buffer,
            ordered by the logical expert id so both ends of every pair issue
            their ops in the same order.
        send: ``(src_slot, dst_rank)`` P2P sends from the live slot, ordered
            by logical expert id, then destination rank.
        local_copy: ``(dst_slot, src_slot)`` the expert moved between two of
            this rank's slots; copied live -> staging before any live slot is
            overwritten (cycles), then staging -> live.
        free_rider: ``(dst_slot, src_slot)`` an earlier local slot receives
            the same expert in this update; copied live -> live after the
            source landed, in increasing ``dst_slot`` order (``src < dst``).
    """

    recv: tuple[tuple[int, int], ...]
    send: tuple[tuple[int, int], ...]
    local_copy: tuple[tuple[int, int], ...]
    free_rider: tuple[tuple[int, int], ...]

    @property
    def is_empty(self) -> bool:
        return not (self.recv or self.send or self.local_copy or self.free_rider)

    @property
    def num_changed_slots(self) -> int:
        """Local slots that receive a new expert (staging destinations and free-riders)."""
        return len(self.recv) + len(self.local_copy) + len(self.free_rider)


def _spread(elements: Sequence[int], over: Sequence[int]) -> dict[int, int]:
    """Assign ``elements`` to ``over`` in contiguous, evenly sized runs.

    ``len(elements) // len(over)`` per target, the remainder going to the
    first targets -- the reference engine's chunking, so destinations spread
    evenly over the sources and both ends derive the same assignment.
    """
    if not elements:
        return {}
    if not over:
        raise ValueError("nothing to spread the destinations over")
    short, long_runs = divmod(len(elements), len(over))
    assignment: dict[int, int] = {}
    start = 0
    for index, target in enumerate(over):
        end = start + short + (1 if index < long_runs else 0)
        for element in elements[start:end]:
            assignment[element] = target
        start = end
    return assignment


def plan_slot_moves(
    old_row: torch.Tensor | Sequence[int],
    new_row: torch.Tensor | Sequence[int],
    *,
    ep_rank: int,
    ep_rank_nodes: Sequence[int],
    num_local: int,
) -> SlotMoves:
    """Plan how ``ep_rank`` turns its slots of ``old_row`` into ``new_row``.

    For every local destination slot whose logical expert changes: nothing if
    unchanged; a same-GPU copy when another local slot held the expert; a
    free-ride on an earlier local slot that receives it in this update;
    otherwise a P2P receive from one source rank -- a same-node source when
    the node has one, else any source -- with the destinations of each source
    spread evenly. Sends mirror the receives: a rank sends each expert it
    holds to exactly the destinations that chose it. Node identity comes from
    ``ep_rank_nodes``, never from a rank-divisibility assumption.

    Args:
        old_row: ``[num_physical]`` logical id of every slot before the move.
        new_row: ``[num_physical]`` logical id of every slot after the move.
        ep_rank: The rank whose moves to plan.
        ep_rank_nodes: The node of every EP rank (its length is the EP size).
        num_local: Slots per rank.

    Returns:
        This rank's ``SlotMoves``; identical on every rank for the same rows.
    """
    old = [
        int(v)
        for v in (old_row.tolist() if isinstance(old_row, torch.Tensor) else old_row)
    ]
    new = [
        int(v)
        for v in (new_row.tolist() if isinstance(new_row, torch.Tensor) else new_row)
    ]
    ep_size = len(ep_rank_nodes)
    num_physical = len(old)
    if len(new) != num_physical:
        raise ValueError("old_row and new_row must have the same length")
    if num_local <= 0 or num_physical != ep_size * num_local:
        raise ValueError(
            f"{num_physical} slots are not {ep_size} ranks x {num_local} local slots"
        )
    if not 0 <= ep_rank < ep_size:
        raise ValueError(f"ep_rank={ep_rank} is outside the {ep_size} EP ranks")
    nodes = [int(node) for node in ep_rank_nodes]
    first = ep_rank * num_local

    # One pass over the row: per logical expert, the ranks holding it before
    # the move (in rank order) and the ranks that receive it from elsewhere.
    src_ranks_of: dict[int, list[int]] = {}
    for p, logical in enumerate(old):
        ranks = src_ranks_of.setdefault(logical, [])
        if not ranks or ranks[-1] != p // num_local:
            ranks.append(p // num_local)
    dst_ranks_of: dict[int, list[int]] = {}
    for p, logical in enumerate(new):
        rank = p // num_local
        ranks = dst_ranks_of.setdefault(logical, [])
        if rank not in src_ranks_of.get(logical, ()) and (
            not ranks or ranks[-1] != rank
        ):
            ranks.append(rank)
    # The slots of this rank: which hold each expert now, and the first slot
    # receiving each expert in this update (free-ride source).
    held_slot_of: dict[int, int] = {}
    for slot in reversed(range(num_local)):
        held_slot_of[old[first + slot]] = slot

    def source_for(logical: int, dst_rank: int) -> int:
        """The rank ``dst_rank`` receives ``logical`` from (identical on both ends)."""
        src_ranks = src_ranks_of[logical]
        dst_ranks = dst_ranks_of[logical]
        dst_node = nodes[dst_rank]
        same_node_sources = [r for r in src_ranks if nodes[r] == dst_node]
        if same_node_sources:
            same_node_dsts = [r for r in dst_ranks if nodes[r] == dst_node]
            return _spread(same_node_dsts, same_node_sources)[dst_rank]
        src_nodes = {nodes[r] for r in src_ranks}
        cross_node_dsts = [r for r in dst_ranks if nodes[r] not in src_nodes]
        return _spread(cross_node_dsts, src_ranks)[dst_rank]

    recv: list[tuple[int, int, int]] = []  # (logical, dst_slot, src_rank)
    local_copy: list[tuple[int, int]] = []
    free_rider: list[tuple[int, int]] = []
    first_receiving_slot: dict[int, int] = {}
    for dst_slot in range(num_local):
        logical = new[first + dst_slot]
        if old[first + dst_slot] == logical:
            continue
        held = held_slot_of.get(logical)
        if held is not None:
            local_copy.append((dst_slot, held))
            continue
        earlier = first_receiving_slot.setdefault(logical, dst_slot)
        if earlier != dst_slot:
            free_rider.append((dst_slot, earlier))
            continue
        recv.append((logical, dst_slot, source_for(logical, ep_rank)))

    send: list[tuple[int, int, int]] = []  # (logical, src_slot, dst_rank)
    for logical, src_slot in sorted(held_slot_of.items(), key=lambda kv: kv[1]):
        for dst_rank in dst_ranks_of.get(logical, ()):
            if source_for(logical, dst_rank) == ep_rank:
                send.append((logical, src_slot, dst_rank))

    recv.sort()
    send.sort()
    return SlotMoves(
        recv=tuple((dst_slot, src_rank) for _, dst_slot, src_rank in recv),
        send=tuple((src_slot, dst_rank) for _, src_slot, dst_rank in send),
        local_copy=tuple(local_copy),
        free_rider=tuple(free_rider),
    )


@dataclass(frozen=True)
class ExpertRebalanceSpecs:
    """Plain values the control plane rebalances with (``DeviceSpecs`` carries them).

    Attributes:
        num_layers: MoE layers (rows of the placement tables).
        num_logical_experts: Routed experts ``E``.
        num_physical_experts: Slots per layer ``P = E + R``.
        ep_size: EP ranks the slots are spread over.
        ep_rank: This rank's position in its EP group.
        ep_rank_nodes: The node of every EP rank.
        all_to_all_ep: Each rank counted only its own tokens' routes, so the
            load must be summed over the EP group; otherwise every rank
            counted every token and the counters already are the group load.
        num_groups: The model's expert groups, or None.
        num_nodes: Nodes the job spans (the hierarchical algorithm's tier).
    """

    num_layers: int
    num_logical_experts: int
    num_physical_experts: int
    ep_size: int
    ep_rank: int
    ep_rank_nodes: tuple[int, ...]
    all_to_all_ep: bool
    num_groups: int | None
    num_nodes: int

    @property
    def num_local_physical_experts(self) -> int:
        return self.num_physical_experts // self.ep_size

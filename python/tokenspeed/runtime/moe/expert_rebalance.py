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

"""Online expert rebalancing: the slot move plan and the control-plane state machine.

A rebalance is a sequence of *internal control ops* the scheduler completes
through the same attention-DP same-round gate as the RL weight ops
(``RequestHandler._rendezvous_replica_flush``), so every rank runs each step
in the same round and the device call may block the control thread:

1. ``EplbSnapshot``: read the route counters since the previous snapshot
   (``DeviceHandle.snapshot_expert_load``), reduce them over the EP group when
   each rank counted only its own tokens, and hand the logical load to the
   controller. EP rank 0 derives the new placement in a spawned CPU worker
   process (``compute_placement_maps``), so the Python greedy loop never
   contends for this process's GIL with the forward thread.
2. ``EplbCommit``, ``COMMIT_DELAY_FORWARDS`` forwards later: rank 0 waits for
   its result, the map is broadcast over the EP group, and every rank plans
   the same slot moves (``plan_slot_moves``) from the old and new rows.
3. ``EplbApplyChunk(layer_ids)``, one per round: the weights move
   (``DeviceHandle.apply_expert_placement``) and the layer's routing tables
   switch in place after its slots landed.

``ExpertRebalanceController`` owns the phase machine
(IDLE -> COMPUTING -> APPLYING -> IDLE), the forward counter that triggers a
snapshot, the background future and the chunk cursor. It is self-contained:
no loop reference, no collective, no device access; ``engine/eplb_hooks.py``
glues it to the request handler and the device handle.

The move plan is a pure function of ``(old_row, new_row)`` and the rank
topology, so it is identical on every rank: sends mirror receives by
construction and the P2P batch pairs without any further agreement.
"""

from __future__ import annotations

import logging
import multiprocessing
from collections.abc import Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from enum import Enum

import torch

from tokenspeed.runtime.moe.eplb_algorithms import EplbAlgorithm
from tokenspeed.runtime.moe.expert_location import load_balancedness, logical_count_of
from tokenspeed.runtime.moe.placement_maps import compute_placement_maps

__all__ = [
    "COMMIT_DELAY_FORWARDS",
    "EplbApplyChunk",
    "EplbCommit",
    "EplbSnapshot",
    "ExpertRebalanceController",
    "ExpertRebalanceSpecs",
    "PlacementComputeWorker",
    "RebalancePhase",
    "SlotMoves",
    "chunk_layer_ids",
    "expected_balancedness",
    "plan_slot_moves",
]

logger = logging.getLogger(__name__)

# Forwards between the snapshot and the commit: the placement computation (a
# Python greedy loop of a few seconds on EP rank 0) overlaps these forwards
# instead of stalling the commit round. A slow computation is still waited
# for at the commit (the reference engine's 200-step overlap).
COMMIT_DELAY_FORWARDS = 200


def _worker_ready() -> bool:
    """Warm-up task: forces the compute worker to spawn and import at startup."""
    return True


class PlacementComputeWorker:
    """One spawned CPU process that derives placements off the serving process.

    A Python-bound thread would contend for the GIL with the forward thread
    (eager launches, metadata refresh) for the seconds the greedy packing
    takes; a spawned process does not, and it never inherits the CUDA
    context. The worker is started here, at construction, so the first
    rebalance does not pay the child's torch import inside the snapshot
    round; inputs and results cross as small CPU tensors.
    """

    def __init__(self) -> None:
        self._pool = ProcessPoolExecutor(
            max_workers=1, mp_context=multiprocessing.get_context("spawn")
        )
        self._ready: Future = self._pool.submit(_worker_ready)

    def submit(self, logical_count: torch.Tensor, **kwargs) -> Future:
        """Schedule ``compute_placement_maps(logical_count, **kwargs)`` in the worker."""
        if logical_count.device.type != "cpu":
            raise ValueError("the placement worker takes host tensors only")
        return self._pool.submit(compute_placement_maps, logical_count, **kwargs)

    def wait_ready(self, timeout: float | None = None) -> None:
        """Block until the worker process is up (tests and startup checks)."""
        self._ready.result(timeout=timeout)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


# -------------------------------- internal op kinds ---------------------------------


@dataclass(frozen=True)
class EplbSnapshot:
    """Read and reset the route counters; start the placement computation."""


@dataclass(frozen=True)
class EplbCommit:
    """Agree on the new placement and plan every layer's slot moves."""


@dataclass(frozen=True)
class EplbApplyChunk:
    """Move the weights of ``layer_ids`` and switch their routing tables."""

    layer_ids: tuple[int, ...]


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


def chunk_layer_ids(num_layers: int, layers_per_chunk: int) -> list[tuple[int, ...]]:
    """Split ``range(num_layers)`` into ``ceil(num_layers / layers_per_chunk)`` runs."""
    if not 1 <= layers_per_chunk <= num_layers:
        raise ValueError(
            f"layers_per_chunk={layers_per_chunk} must be in [1, {num_layers}]"
        )
    return [
        tuple(range(start, min(start + layers_per_chunk, num_layers)))
        for start in range(0, num_layers, layers_per_chunk)
    ]


def expected_balancedness(
    logical_count: torch.Tensor, physical_to_logical_map: torch.Tensor, ep_size: int
) -> torch.Tensor:
    """Per-layer balancedness a placement would reach for ``logical_count``.

    Each expert's load is split evenly over its replicas, the static
    assumption the EPLB algorithm balances under.
    """
    map_long = physical_to_logical_map.to(torch.int64)
    replicas = torch.zeros_like(logical_count, dtype=torch.int64).scatter_add_(
        1, map_long, torch.ones_like(map_long)
    )
    per_slot = logical_count.double().gather(1, map_long) / replicas.gather(
        1, map_long
    ).double().clamp_min(1)
    return load_balancedness(per_slot, ep_size)


# -------------------------------- the controller ------------------------------------


class RebalancePhase(Enum):
    IDLE = "idle"
    # A snapshot op is queued or taken; the placement is being derived.
    COMPUTING = "computing"
    # The commit and chunk ops are queued; chunks complete one per round.
    APPLYING = "applying"


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


class ExpertRebalanceController:
    """Decides when a rebalance starts and what each of its ops carries.

    Self-contained: it counts the forwards this rank submitted, holds the
    phase, the background computation and the chunk cursor, and returns the
    ops to enqueue. It never touches the device or a collective; the hooks
    complete the ops with the handle and feed the results back here.

    EP rank 0 of each EP group computes the placement in a spawned CPU
    worker process and the hooks broadcast it at the commit, so no
    rank-agreement argument about the algorithm's tie-breaking is needed.
    """

    def __init__(
        self,
        specs: ExpertRebalanceSpecs,
        *,
        rebalance_num_iterations: int,
        layers_per_chunk: int,
        algorithm: EplbAlgorithm,
        commit_delay_forwards: int,
        compute_worker: PlacementComputeWorker | None,
    ) -> None:
        """
        Args:
            specs: The placement geometry and this rank's position.
            rebalance_num_iterations: Forwards between snapshots (``N``).
            layers_per_chunk: Layers switched per round (``L``).
            algorithm: The EPLB algorithm variant.
            commit_delay_forwards: Forwards between a snapshot and its commit
                (``COMMIT_DELAY_FORWARDS`` in serving).
            compute_worker: Where EP rank 0 derives the placement (a spawned
                ``PlacementComputeWorker`` in serving); None on every other
                rank, which takes the broadcast map instead.
        """
        if rebalance_num_iterations <= 0:
            raise ValueError("rebalance_num_iterations must be positive")
        if commit_delay_forwards < 0:
            raise ValueError("commit_delay_forwards must be non-negative")
        if (compute_worker is None) == (specs.ep_rank == 0):
            raise ValueError(
                "EP rank 0 derives the placement and needs a compute worker; "
                "every other rank takes the broadcast map and must not have one"
            )
        self.specs = specs
        self._interval = rebalance_num_iterations
        self._commit_delay = commit_delay_forwards
        self._chunks = chunk_layer_ids(specs.num_layers, layers_per_chunk)
        self._algorithm = algorithm
        self._compute_worker = compute_worker
        self.phase = RebalancePhase.IDLE
        # Forwards this rank submitted (real or DP-idle), rank-identical.
        self.forwards = 0
        self._next_snapshot_at = rebalance_num_iterations
        self._commit_at: int | None = None
        self._future: Future | None = None
        # The placement the snapshot was counted under and the committed one.
        self._old_map: torch.Tensor | None = None
        self._new_map: torch.Tensor | None = None
        self._logical_count: torch.Tensor | None = None
        self._moves_by_layer: dict[int, SlotMoves] = {}
        # Chunks committed but not yet applied, in order.
        self._pending_chunks: list[tuple[int, ...]] = []
        self.rebalances_completed = 0

    # -------------------------------- round driving -------------------------------

    @property
    def commit_delay_forwards(self) -> int:
        return self._commit_delay

    @property
    def is_idle(self) -> bool:
        return self.phase is RebalancePhase.IDLE

    @property
    def is_applying(self) -> bool:
        return self.phase is RebalancePhase.APPLYING

    def note_round(self, *, forwarded: bool) -> list:
        """Count a round's forward and return the ops to enqueue this round.

        A snapshot is due ``rebalance_num_iterations`` forwards after the
        previous one (deferred while a rebalance is still in progress); the
        commit and every chunk op are enqueued together
        ``commit_delay_forwards`` forwards after the snapshot completed.
        """
        if forwarded:
            self.forwards += 1
        if self.phase is RebalancePhase.IDLE:
            if self.forwards >= self._next_snapshot_at:
                return [self._begin_snapshot()]
            return []
        if (
            self.phase is RebalancePhase.COMPUTING
            and self._commit_at is not None
            and self.forwards >= self._commit_at
        ):
            self._commit_at = None
            return [EplbCommit(), *(EplbApplyChunk(ids) for ids in self._chunks)]
        return []

    def request_snapshot(self) -> EplbSnapshot | None:
        """A manual trigger: the snapshot op, or None while a rebalance is in progress."""
        if self.phase is not RebalancePhase.IDLE:
            return None
        return self._begin_snapshot()

    def _begin_snapshot(self) -> EplbSnapshot:
        self.phase = RebalancePhase.COMPUTING
        self._next_snapshot_at = self.forwards + self._interval
        return EplbSnapshot()

    # -------------------------------- op results ----------------------------------

    def on_snapshot(
        self, physical_count: torch.Tensor, physical_to_logical_map: torch.Tensor
    ) -> None:
        """Take the group-wide load; start the computation; schedule the commit.

        Args:
            physical_count: ``[layers, physical]`` int64 host routes per slot
                of the whole EP group.
            physical_to_logical_map: The placement they were counted under.
        """
        if self.phase is not RebalancePhase.COMPUTING or self._commit_at is not None:
            raise RuntimeError(f"unexpected expert load snapshot in phase {self.phase}")
        specs = self.specs
        expected = (specs.num_layers, specs.num_physical_experts)
        if tuple(physical_count.shape) != expected:
            raise ValueError(
                f"expert load has shape {tuple(physical_count.shape)}, expected {expected}"
            )
        self._old_map = physical_to_logical_map.to(device="cpu", dtype=torch.int32)
        self._logical_count = logical_count_of(
            physical_count.cpu(), self._old_map, specs.num_logical_experts
        )
        before = load_balancedness(physical_count.cpu(), specs.ep_size)
        logger.info(
            f"Expert rebalance: snapshot of {int(physical_count.sum())} routes over "
            f"{specs.num_layers} layers; balancedness min {before.min():.3f} "
            f"mean {before.mean():.3f}; commit in {self._commit_delay} forwards"
        )
        if self._compute_worker is not None:
            self._future = self._compute_worker.submit(
                self._logical_count,
                num_physical_experts=specs.num_physical_experts,
                ep_size=specs.ep_size,
                num_groups=specs.num_groups,
                num_nodes=specs.num_nodes,
                algorithm=self._algorithm,
            )
        self._commit_at = self.forwards + self._commit_delay

    def placement_for_commit(self) -> torch.Tensor:
        """The ``[layers, physical]`` int32 map to broadcast from EP rank 0.

        Rank 0 waits for its computation here (a slow computation stalls the
        commit round, never a deadlock); every other rank returns the buffer
        the broadcast fills.
        """
        if self.phase is not RebalancePhase.COMPUTING or self._old_map is None:
            raise RuntimeError(
                f"unexpected expert placement commit in phase {self.phase}"
            )
        specs = self.specs
        if self._future is None:
            return torch.empty(
                (specs.num_layers, specs.num_physical_experts), dtype=torch.int32
            )
        physical_to_logical_map, _ = self._future.result()
        self._future = None
        return physical_to_logical_map.contiguous()

    def on_commit(self, new_map: torch.Tensor) -> None:
        """Plan every layer's moves from the agreed placement; enter APPLYING."""
        if self.phase is not RebalancePhase.COMPUTING or self._old_map is None:
            raise RuntimeError(
                f"unexpected expert placement commit in phase {self.phase}"
            )
        specs = self.specs
        new_map = new_map.to(device="cpu", dtype=torch.int32)
        if tuple(new_map.shape) != tuple(self._old_map.shape):
            raise ValueError(
                f"committed placement has shape {tuple(new_map.shape)}, expected "
                f"{tuple(self._old_map.shape)}"
            )
        self._moves_by_layer = {
            layer_id: plan_slot_moves(
                self._old_map[layer_id],
                new_map[layer_id],
                ep_rank=specs.ep_rank,
                ep_rank_nodes=specs.ep_rank_nodes,
                num_local=specs.num_local_physical_experts,
            )
            for layer_id in range(specs.num_layers)
        }
        self._new_map = new_map
        self._pending_chunks = list(self._chunks)
        self.phase = RebalancePhase.APPLYING
        if self._logical_count is not None:
            after = expected_balancedness(self._logical_count, new_map, specs.ep_size)
            moved = sum(m.num_changed_slots for m in self._moves_by_layer.values())
            logger.info(
                f"Expert rebalance: committed placement moves {moved} local slots; "
                f"expected balancedness min {after.min():.3f} mean {after.mean():.3f}"
            )

    def chunk_payload(
        self, layer_ids: Sequence[int]
    ) -> tuple[torch.Tensor, dict[int, SlotMoves]]:
        """The new rows and moves of the next chunk, which must be ``layer_ids``."""
        if self.phase is not RebalancePhase.APPLYING or self._new_map is None:
            raise RuntimeError(
                f"unexpected expert placement chunk in phase {self.phase}"
            )
        layer_ids = tuple(int(layer_id) for layer_id in layer_ids)
        if not self._pending_chunks or self._pending_chunks[0] != layer_ids:
            raise RuntimeError(
                f"chunk {layer_ids} is out of order; expected "
                f"{self._pending_chunks[0] if self._pending_chunks else None}"
            )
        return (
            self._new_map[list(layer_ids)],
            {layer_id: self._moves_by_layer[layer_id] for layer_id in layer_ids},
        )

    def on_chunk_applied(self, layer_ids: Sequence[int]) -> None:
        """Advance the chunk cursor; the last chunk completes the rebalance."""
        layer_ids = tuple(int(layer_id) for layer_id in layer_ids)
        if not self._pending_chunks or self._pending_chunks[0] != layer_ids:
            raise RuntimeError(f"chunk {layer_ids} was not the pending chunk")
        self._pending_chunks.pop(0)
        if self._pending_chunks:
            return
        self.phase = RebalancePhase.IDLE
        self._old_map = None
        self._new_map = None
        self._logical_count = None
        self._moves_by_layer = {}
        self.rebalances_completed += 1
        logger.info(f"Expert rebalance {self.rebalances_completed} applied")

    @property
    def compute_worker(self) -> PlacementComputeWorker | None:
        """The spawned worker (EP rank 0 only), for startup checks and tests."""
        return self._compute_worker

    def shutdown(self) -> None:
        if self._compute_worker is not None:
            self._compute_worker.shutdown()

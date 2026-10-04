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

"""Loop-side glue of the online expert rebalance (``--enable-eplb``).

``ExpertRebalanceController`` (``moe/expert_rebalance.py``) decides; these
hooks act with the loop's collaborators: the request handler's internal-op
FIFO, the ``DeviceHandle``'s two named operations, and the EP gloo group for
the two agreements a rebalance needs -- the group-wide load at the snapshot
and the committed placement -- which are legal here because every internal
op completes inside the attention-DP same-round gate, so every EP rank is in
the op at once (``docs/design/event-loop.md``).

The loop carries one line, ``note_round(forwarded=...)``, after its forward
submission; the request handler calls ``complete`` for a gate-agreed internal
op and ``begin_manual_rebalance`` for ``POST /rebalance_experts``.
"""

from __future__ import annotations

import hashlib
import logging

import torch

from tokenspeed.runtime.moe import eplb_algorithms
from tokenspeed.runtime.moe.expert_rebalance import (
    COMMIT_DELAY_FORWARDS,
    EplbApplyChunk,
    EplbCommit,
    EplbSnapshot,
    ExpertRebalanceController,
    ExpertRebalanceSpecs,
    PlacementComputeWorker,
)

__all__ = [
    "EplbHooks",
    "make_expert_rebalance_controller",
    "rebalance_release_refusal",
]

logger = logging.getLogger(__name__)


def make_expert_rebalance_controller(
    server_args, specs: ExpertRebalanceSpecs | None
) -> ExpertRebalanceController | None:
    """The loop's rebalance controller from the server args and the device specs.

    None unless the server started with ``--enable-eplb`` (the device side
    then reports the placement geometry in ``DeviceSpecs.expert_rebalance``).
    EP rank 0 gets the spawned CPU worker that derives placements; it starts
    here, during startup, so the first rebalance pays no child import.
    """
    if specs is None:
        if server_args.enable_eplb:
            raise RuntimeError(
                "--enable-eplb is set but the device side built no expert "
                "placement updater"
            )
        return None
    if not server_args.enable_eplb:
        raise RuntimeError(
            "the device side built an expert placement updater without --enable-eplb"
        )
    return ExpertRebalanceController(
        specs,
        rebalance_num_iterations=server_args.eplb_rebalance_num_iterations,
        layers_per_chunk=server_args.eplb_rebalance_layers_per_chunk,
        algorithm=eplb_algorithms.compute_algorithm(
            raw_algorithm=server_args.eplb_algorithm,
            num_groups=specs.num_groups,
            num_nodes=specs.num_nodes,
        ),
        commit_delay_forwards=COMMIT_DELAY_FORWARDS,
        compute_worker=PlacementComputeWorker() if specs.ep_rank == 0 else None,
    )


def rebalance_release_refusal(controller: ExpertRebalanceController | None):
    """Why a memory-saver release must be refused right now, or None.

    Internal ops must not run against unmapped weights: a release is refused
    while a rebalance is in progress (its chunk ops would land after the
    weights were released, and its commit cannot be reached without forwards).
    """

    def refusal() -> str | None:
        if controller is None or controller.is_idle:
            return None
        return (
            f"an expert rebalance is in progress ({controller.phase.value}); "
            "retry the release once it has been applied"
        )

    return refusal


class EplbHooks:
    """Glue hooks: controller + request handler FIFO + device named ops + EP gloo group.

    Args:
        controller: The rebalance state machine, or None when the server runs
            without ``--enable-eplb`` (every entry point is then a no-op and
            no completer is installed, so the manual trigger is refused).
        request_handler: Owns the internal-op FIFO and the gate.
        device: The ``DeviceHandle``; injected, never reached through the loop.
        ep_cpu_group: The EP group's gloo process group, or None at EP size 1.
        ep_group_ranks: Global ranks of the EP group in EP-rank order (the
            broadcast source is the first).
    """

    def __init__(
        self,
        controller: ExpertRebalanceController | None,
        request_handler,
        device,
        *,
        ep_cpu_group,
        ep_group_ranks: tuple[int, ...],
    ) -> None:
        self._controller = controller
        self._request_handler = request_handler
        self._device = device
        self._ep_cpu_group = ep_cpu_group if len(ep_group_ranks) > 1 else None
        self._ep_src_rank = ep_group_ranks[0] if ep_group_ranks else 0
        # Preallocated: the agreement check runs on the control thread,
        # where allocating on the bound device is what Principle 1 forbids.
        self._checksum_buf = torch.zeros(2, dtype=torch.int64, device="cpu")
        if controller is not None:
            request_handler.set_internal_op_completer(self)

    # -------------------------------- loop entry point ----------------------------

    def note_round(self, *, forwarded: bool) -> None:
        """Count this round's forward; enqueue the ops the controller makes due."""
        if self._controller is None:
            return
        for op in self._controller.note_round(forwarded=forwarded):
            self._request_handler.enqueue_internal_op(op)

    def close(self) -> None:
        """Stop the controller's compute worker at engine shutdown."""
        if self._controller is not None:
            self._controller.shutdown()

    # -------------------------------- request handler entry points ---------------

    def complete(self, op) -> None:
        """Complete a gate-agreed internal op through the handle's named operations."""
        controller = self._controller
        if controller is None:
            raise RuntimeError("expert rebalancing is not enabled")
        if isinstance(op, EplbSnapshot):
            self._snapshot(controller)
        elif isinstance(op, EplbCommit):
            self._commit(controller)
        elif isinstance(op, EplbApplyChunk):
            rows, moves = controller.chunk_payload(op.layer_ids)
            self._device.apply_expert_placement(op.layer_ids, rows, moves)
            controller.on_chunk_applied(op.layer_ids)
        else:
            raise TypeError(f"{type(op).__name__} is not an internal control op")

    def begin_manual_rebalance(self) -> tuple[bool, str]:
        """``POST /rebalance_experts``: take the snapshot now, if idle."""
        controller = self._controller
        if controller is None:
            return (
                False,
                "expert rebalancing needs the server to start with --enable-eplb",
            )
        op = controller.request_snapshot()
        if op is None:
            return False, (
                f"an expert rebalance is already in progress ({controller.phase.value})"
            )
        self.complete(op)
        return True, (
            f"expert load snapshot taken; the placement commits after "
            f"{controller.commit_delay_forwards} forwards"
        )

    # -------------------------------- the ops -------------------------------------

    def _snapshot(self, controller: ExpertRebalanceController) -> None:
        snapshot = self._device.snapshot_expert_load()
        physical = snapshot.physical_count
        if self._ep_cpu_group is not None:
            if controller.specs.all_to_all_ep:
                # Each rank counted its own tokens: the layer's load is the sum.
                torch.distributed.all_reduce(
                    physical,
                    op=torch.distributed.ReduceOp.SUM,
                    group=self._ep_cpu_group,
                )
            else:
                # Every rank routed every token through identical tables, so
                # the counters already are the group load -- and must agree.
                self._assert_ranks_agree(physical)
        controller.on_snapshot(physical, snapshot.physical_to_logical_map)

    def _assert_ranks_agree(self, physical: torch.Tensor) -> None:
        """Fail loudly if replicated-input EP ranks counted different routes.

        A divergence means the ranks routed differently within one forward,
        so some routes were computed twice or never: a correctness bug, not a
        statistic to average away.
        """
        digest = hashlib.blake2b(
            physical.contiguous().numpy().tobytes(), digest_size=8
        ).digest()
        checksum = int.from_bytes(digest, "little", signed=True)
        buf = self._checksum_buf
        buf[0] = checksum
        buf[1] = -checksum
        torch.distributed.all_reduce(
            buf, op=torch.distributed.ReduceOp.MAX, group=self._ep_cpu_group
        )
        if int(buf[0]) != -int(buf[1]):
            raise RuntimeError(
                "expert load counters differ across EP ranks under "
                "replicated-input EP: the ranks routed tokens differently within "
                "a forward, so the MoE output is wrong; this is a routing bug"
            )

    def _commit(self, controller: ExpertRebalanceController) -> None:
        placement = controller.placement_for_commit()
        if self._ep_cpu_group is not None:
            torch.distributed.broadcast(
                placement, src=self._ep_src_rank, group=self._ep_cpu_group
            )
        controller.on_commit(placement)

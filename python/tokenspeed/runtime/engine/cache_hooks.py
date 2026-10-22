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

"""L2 cache-op submission and rank-synchronized completion tracking.

Owns everything between an execution plan's cache ops and the scheduler
events their completions eventually produce: count what the plan puts in
flight (``DeviceHandle.execute`` submits the transfers themselves, on the
data plane, from the same plan), poll
completions (control-side event queries), and agree across every
cache-owning rank in the replica (attention TP, then CP, then PP) on
which completions EVERY rank has seen (the C++ scheduler is mirrored, so an
event may only advance once all ranks hold it). Only the GPU copy fences
these events; L3 backup has independent tickets and completion agreement.
Every rank enters every replica-group gather, including with an empty
intermediate intersection, so peers remain in collective order.
``poll_ready_events`` returns events for the event loop to apply —
feedback into the scheduler stays an explicit ``advance_scheduler`` call
in the loop body.

Depends only on the device handle and static parallel-layout config, not on
live event-loop state. ``device=None`` (the engine can emit no cache op, see
``cache_hooks_armed``) makes every method a cheap no-op.
"""

from __future__ import annotations

import logging
from collections import OrderedDict

import torch
import torch.distributed as dist

from tokenspeed.runtime.cache.transfer.ops import RestoreOp, SnapshotOp
from tokenspeed.runtime.engine.scheduler_utils import (
    cache_event_from_payload,
    cache_event_key,
    cache_event_to_payload,
    cache_sync_debug_enabled,
    is_device_cache_op,
    pop_common_cache_event_payloads,
)

logger = logging.getLogger(__name__)


def cache_hooks_armed(*, enable_kvstore: bool, max_retracted_requests: int) -> bool:
    """Whether the engine can emit cache ops, so the hooks must poll for them.

    Armed hooks poll completions and run a gloo all-reduce while a cache op is open, so
    they are armed only for an engine that can emit cache ops: one with the
    Host L2 tier (write-backs, load-backs) or with a retraction snapshot pool
    (the scheduler retracts by snapshot store and resumes by restore).

    Args:
        enable_kvstore: The Host L2 tier is configured.
        max_retracted_requests: The retraction snapshot pool's slot-state
            rows; positive when a pool is configured.
    """
    return bool(enable_kvstore) or max_retracted_requests > 0


class CacheOpHooks:
    """Tracks in-flight cache ops for one scheduler event loop."""

    def __init__(
        self,
        device,
        *,
        speculative_algorithm: str | None,
        attn_tp_rank: int,
        attn_tp_size: int,
        attn_tp_cpu_group,
        pp_size: int,
        pp_cpu_group,
        global_rank: int,
    ) -> None:
        self._device = device
        # Replica-wide skip/intersect is the general path, not DFLASH-only.
        # The argument stays required so EventLoop wiring remains explicit.
        self._speculative_algorithm = speculative_algorithm
        self._attn_tp_rank = attn_tp_rank
        self._attn_tp_size = attn_tp_size
        self._attn_tp_cpu_group = attn_tp_cpu_group
        self._global_rank = global_rank
        self._pending_payloads: OrderedDict[tuple[str, int], dict] = OrderedDict()
        # All ranks submit identical cache plans (the C++ scheduler is
        # mirrored), so a local in-flight counter mirrors across ranks: if it's
        # 0 here, no rank has anything pending. Lets us skip the replica
        # collectives in poll_ready_events entirely when nothing is in flight
        # — but only after every cache-owning rank agrees the replica is idle.
        self._num_inflight = 0
        # Uncommitted tickets: equal on every rank (same plans, same intersections).
        self._num_open = 0
        replica_groups = []
        if attn_tp_size > 1 and attn_tp_cpu_group is not None:
            replica_groups.append((attn_tp_size, attn_tp_cpu_group))
        if pp_size > 1 and pp_cpu_group is not None:
            replica_groups.append((pp_size, pp_cpu_group))
        self._replica_groups = replica_groups

    def count_ops(self, cache_ops) -> int:
        """Count the prepared Device-copy operations submitted this round.

        The loop passes this same snapshot to DeviceHandle.execute. Storage
        jobs have independent retirement facts and never consume these GPU
        completion tickets. Returns the snapshot-store count for preemptions.
        """
        if self._device is None:
            return 0
        cache_ops = tuple(cache_ops)
        if not all(is_device_cache_op(op) for op in cache_ops):
            raise TypeError(
                "cache completion accounting requires Device-copy operations"
            )
        num_snapshot_ops = 0
        for op in cache_ops:
            if isinstance(op, (SnapshotOp, RestoreOp)):
                self._num_inflight += 1
                self._num_open += 1
                if isinstance(op, SnapshotOp):
                    num_snapshot_ops += 1
            else:
                self._num_inflight += len(op.op_ids)
                self._num_open += len(op.op_ids)
        return num_snapshot_ops

    def poll_ready_events(self) -> list:
        """Poll completed cache ops and return their rank-synchronized
        scheduler events. Returns an empty list when there is nothing ready.
        """
        if self._device is None:
            return []
        if self._num_open == 0:
            # No rank has a ticket in flight or a completion to commit.
            return []
        cache_results = self._device.poll_cache_results()
        self._num_inflight -= len(cache_results)
        for event in cache_results:
            payload = cache_event_to_payload(event)
            self._pending_payloads[cache_event_key(payload)] = payload

        # CUDA copy completions are async and not lock-step
        # across TP/CP/PP, so local state (_num_inflight / _pending_payloads)
        # diverges transiently. A rank-local skip would let some ranks gather
        # while others return, deadlocking the group. Agree on work before
        # gathering completion payloads on each replica group.
        local_has_work = bool(self._num_inflight != 0 or self._pending_payloads)
        if self._replica_groups:
            if not self._converge_poll_state(local_has_work):
                return []
        elif not local_has_work:
            return []

        ready_payloads = self._pop_ready_payloads()
        self._num_open -= len(ready_payloads)
        if not ready_payloads:
            return []
        logger.debug(
            f"[cache_poll] got {len(ready_payloads)!s} synchronized results",
        )
        events = []
        for payload in ready_payloads:
            e = cache_event_from_payload(payload)
            logger.debug(
                f"[cache_poll] event: op_id={e.op_id!s} type={type(e).__name__!s}",
            )
            events.append(e)
        return events

    def _converge_poll_state(self, local_has_work: bool) -> bool:
        """All owners poll until every local cache transfer is accounted for."""
        flag = torch.tensor([int(local_has_work)], dtype=torch.int32)
        for _size, group in self._replica_groups:
            dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=group)
        return bool(flag[0].item())

    def _pop_ready_payloads(self) -> list[dict]:
        """Intersect pending completions across every replica group.

        Attention TP, then CP, then PP, matching cache flush. Every
        rank enters every ``all_gather_object``: an empty intermediate
        intersection still gathers ``[]`` so a peer that is ready on a
        later group is not left unmatched.
        """
        ready_payloads = list(self._pending_payloads.values())
        for size, group in self._replica_groups:
            gathered_payloads = [None] * size
            dist.all_gather_object(
                gathered_payloads,
                ready_payloads,
                group=group,
            )
            ready_payloads = pop_common_cache_event_payloads(gathered_payloads)
            if self._attn_tp_rank == 0 and cache_sync_debug_enabled():
                pending_ops = [
                    [(payload["kind"], payload["op_id"]) for payload in rank_payloads]
                    for rank_payloads in gathered_payloads
                ]
                if len({tuple(rank_ops) for rank_ops in pending_ops}) > 1:
                    logger.info(
                        f"[cache_sync] rank={self._global_rank!s} pending_ops="
                        f"{pending_ops!s} ready_ops="
                        f"{[(payload['kind'], payload['op_id']) for payload in ready_payloads]!s}",
                    )

        for payload in ready_payloads:
            self._pending_payloads.pop(cache_event_key(payload), None)
        return ready_payloads

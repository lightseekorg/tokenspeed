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
events their completions eventually produce: submit ops through the device
handle (the transfers launch on the data plane; see
``DeviceHandle.submit_cache_plan``), count what is in flight, poll
completions (control-side event queries), and agree across attn-tp ranks on
which completions EVERY rank has seen (the C++ scheduler is mirrored, so an
event may only advance once all ranks hold it). ``poll_ready_events`` returns
events for the event loop to apply — feedback into the scheduler stays an
explicit ``advance_scheduler`` call in the loop body.

Depends only on the device handle and static parallel-layout config, not on
live event-loop state. ``device=None`` (kvstore disabled) makes every method
a cheap no-op.
"""

from __future__ import annotations

import logging
from collections import OrderedDict

import torch
import torch.distributed as dist
from tokenspeed_scheduler import Cache

from tokenspeed.runtime.engine.scheduler_utils import (
    cache_event_from_payload,
    cache_event_key,
    cache_event_to_payload,
    cache_sync_debug_enabled,
    pop_common_cache_event_payloads,
)

logger = logging.getLogger(__name__)


class L2CacheHooks:
    """Tracks in-flight L2 cache ops for one scheduler event loop."""

    def __init__(
        self,
        device,
        *,
        speculative_algorithm: str | None,
        attn_tp_rank: int,
        attn_tp_size: int,
        attn_tp_cpu_group,
        global_rank: int,
    ) -> None:
        self._device = device
        self._speculative_algorithm = speculative_algorithm
        self._attn_tp_rank = attn_tp_rank
        self._attn_tp_size = attn_tp_size
        self._attn_tp_cpu_group = attn_tp_cpu_group
        self._global_rank = global_rank
        self._pending_payloads: OrderedDict[tuple[str, int], dict] = OrderedDict()
        # All ranks submit identical cache plans (the C++ scheduler is
        # mirrored), so a local in-flight counter mirrors across ranks: if it's
        # 0 here, no rank has anything pending. Lets us skip the TP collective
        # in poll_ready_events entirely when nothing is in flight.
        self._num_inflight = 0

    def submit(self, execution_plan) -> None:
        """Queue the plan's cache ops on the data plane; count them in flight."""
        if self._device is None:
            return
        self._device.submit_cache_plan(execution_plan)
        for op in execution_plan.cache:
            if isinstance(op, (Cache.WriteBackOp, Cache.LoadBackOp)):
                self._num_inflight += len(op.op_ids)
            else:
                raise TypeError(f"unsupported cache op kind: {type(op).__name__}")

    def poll_ready_events(self) -> list:
        """Poll completed L2 cache ops and return their rank-synchronized
        scheduler events. Returns an empty list when there is nothing ready.
        """
        if self._device is None:
            return []
        cache_results = self._device.poll_cache_results()
        self._num_inflight -= len(cache_results)
        for event in cache_results:
            payload = cache_event_to_payload(event)
            self._pending_payloads[cache_event_key(payload)] = payload

        # The gather below is a collective, but cache-op completion is async and
        # not lock-step across ranks, so local state (_num_inflight /
        # _pending_payloads) diverges transiently. A rank-local skip would let
        # some ranks gather while others return, deadlocking the group. Agree on
        # the skip via a cheap single-int all_reduce.
        # NOTE: For non-DFLASH algorithms, cache ops are deterministic across
        # ranks, so the local short-circuit is safe and avoids collective overhead.
        local_has_work = bool(self._num_inflight != 0 or self._pending_payloads)
        if self._speculative_algorithm in ("DFLASH", "DSPARK"):
            if not self._group_has_work(local_has_work):
                return []
        else:
            if not local_has_work:
                return []

        ready_payloads = self._pop_ready_payloads()
        if not ready_payloads:
            return []
        logger.debug(
            "[cache_poll] got %s synchronized results",
            len(ready_payloads),
        )
        events = []
        for payload in ready_payloads:
            e = cache_event_from_payload(payload)
            logger.debug(
                "[cache_poll] event: op_id=%s type=%s",
                e.op_id,
                type(e).__name__,
            )
            events.append(e)
        return events

    def _group_has_work(self, local_has_work: bool) -> bool:
        """Whether ANY attn-tp rank has cache work this step (unanimous via a
        single-int MAX all_reduce, far cheaper than the payload gather it
        guards). Deciding from rank-local state alone deadlocks the group; see
        poll_ready_events.

        Args:
            local_has_work: This rank's view of whether any cache op is in
                flight or any polled payload awaits commit.

        Returns:
            ``True`` if any rank has work (all must gather); ``False`` only when
            every rank is idle.
        """
        if self._attn_tp_size == 1:
            return local_has_work
        flag = torch.tensor([1 if local_has_work else 0], dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=self._attn_tp_cpu_group)
        return bool(flag.item())

    def _pop_ready_payloads(self) -> list[dict]:
        local_payloads = list(self._pending_payloads.values())
        if self._attn_tp_size == 1:
            ready_payloads = local_payloads
        else:
            gathered_payloads = [None] * self._attn_tp_size
            dist.all_gather_object(
                gathered_payloads,
                local_payloads,
                group=self._attn_tp_cpu_group,
            )
            ready_payloads = pop_common_cache_event_payloads(gathered_payloads)
            if self._attn_tp_rank == 0 and cache_sync_debug_enabled():
                pending_ops = [
                    [(payload["kind"], payload["op_id"]) for payload in rank_payloads]
                    for rank_payloads in gathered_payloads
                ]
                if len({tuple(rank_ops) for rank_ops in pending_ops}) > 1:
                    logger.info(
                        "[cache_sync] rank=%s pending_ops=%s ready_ops=%s",
                        self._global_rank,
                        pending_ops,
                        [
                            (payload["kind"], payload["op_id"])
                            for payload in ready_payloads
                        ],
                    )

        for payload in ready_payloads:
            self._pending_payloads.pop(cache_event_key(payload), None)
        return ready_payloads

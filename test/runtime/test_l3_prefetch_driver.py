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

"""The runtime's side of an L3 prefetch, driven against the real C++ scheduler.

No engine and no store: the scheduler is built through ``make_config`` as the
event loop builds it, its ``Cache.PrefetchOp`` is adapted by
``cache_ops_from_plan`` as ``DeviceHandle.execute`` adapts it, the executor's
lane is stood in for by a device fake reporting what it landed, and
``L3CacheHooks.converge_prefetches`` completes the op so its one
``PrefetchDoneEvent`` reaches the scheduler through ``CacheOpHooks`` -- after
which the request is admitted as an ordinary Host hit of the landed prefix.
"""

from __future__ import annotations

import pytest

ts = pytest.importorskip("tokenspeed_scheduler")

from tokenspeed.runtime.cache.transfer.ops import PrefetchOp  # noqa: E402
from tokenspeed.runtime.engine.cache_hooks import CacheOpHooks  # noqa: E402
from tokenspeed.runtime.engine.l3_cache_hooks import L3CacheHooks  # noqa: E402
from tokenspeed.runtime.engine.scheduler_utils import (  # noqa: E402
    advance_scheduler,
    cache_ops_from_plan,
    make_config,
)

PAGE = 16
PROMPT = list(range(40))  # two hash-complete pages and an 8-token tail


class _Device:
    """The DeviceHandle surface the two hooks use: the executor's prefetch
    lane as a table of outcomes, and the ACKs it yields once completed."""

    def __init__(self) -> None:
        self.progress: dict[int, tuple[bool, int]] = {}
        self.results: list = []

    def l3_prefetch_progress(self):
        return dict(self.progress)

    def complete_l3_prefetch(self, op_id, landed_pages) -> None:
        self.progress.pop(op_id)
        self.results.append(ts.Cache.PrefetchDoneEvent(op_id, landed_pages))

    def poll_cache_results(self) -> list:
        results, self.results = self.results, []
        return results

    def consume_l3_backup_poll_failure(self) -> bool:
        return False


def _scheduler() -> ts.Scheduler:
    groups = [
        ts.CacheGroupConfig(
            group_id="history",
            block_granularity=PAGE,
            total_pages=64 + 1,
            retention=ts.CacheRetention.FullHistory,
            family=ts.CacheGroupFamily.History,
        )
    ]
    return ts.Scheduler(
        make_config(
            num_device_pages=64 + 1,
            max_scheduled_tokens=256,
            max_batch_size=4,
            prefix_granularity=PAGE,
            num_host_pages=32 + 1,
            disable_l2_cache=False,
            enable_l3_storage=True,
            role="fused",
            num_snapshot_pages=1,
            max_retracted_requests=0,
            l3_prefetch_min_pages=1,
            cache_groups=groups,
        )
    )


def _cache_hooks(device) -> CacheOpHooks:
    return CacheOpHooks(
        device,
        speculative_algorithm=None,
        attn_tp_rank=0,
        attn_tp_size=1,
        attn_tp_cpu_group=None,
        pp_size=1,
        pp_cpu_group=None,
        global_rank=0,
    )


def _l3_hooks(scheduler, device) -> L3CacheHooks:
    return L3CacheHooks(
        scheduler,
        device,
        attn_tp_size=1,
        attn_tp_cpu_group=None,
        pp_size=1,
        pp_cpu_group=None,
    )


@pytest.mark.parametrize("landed", [2, 1])
def test_an_l3_hit_is_prefetched_before_admission_and_admitted_as_a_host_hit(
    landed: int,
) -> None:
    scheduler = _scheduler()
    device = _Device()
    cache_hooks = _cache_hooks(device)
    l3_hooks = _l3_hooks(scheduler, device)
    # The submit-time probe found both prefix pages in L3.
    groups, hashes, offsets = scheduler.expand_prefix_keys(
        scheduler.prefix_hashes_for_tokens(PROMPT)
    )
    scheduler.register_storage_keys(list(groups), list(hashes), list(offsets))
    spec = ts.RequestSpec()
    spec.request_id = "a"
    spec.tokens = list(PROMPT)
    spec.max_new_tokens = 8
    scheduler.submit_requests([spec])

    # The admission round emits the prefetch and holds the request: no forward,
    # no Device page, a waiting request.
    plan = scheduler.next_execution_plan()
    cache_hooks.count_plan_ops(plan)
    assert [op.request_ids for op in plan.forward if op.request_ids] == []
    (op,) = cache_ops_from_plan(plan)
    assert isinstance(op, PrefetchOp)
    assert (op.request_id, op.first_page, op.num_pages) == ("a", 0, 2)
    assert [(row.page_index, row.content_hash) for row in op.rows] == [
        (0, hashes[0]),
        (1, hashes[1]),
    ]
    assert scheduler.waiting_size() == 1 and scheduler.active_lcm_blocks() == 0
    # Nothing is acknowledged before the lane lands and the replica converges.
    assert cache_hooks.poll_ready_events() == []

    # The lane lands ``landed`` pages; the hooks converge and complete the op,
    # and its one ACK reaches the scheduler through the cache poll.
    l3_hooks.converge_prefetches()
    assert device.progress == {}
    device.progress = {op.op_id: (True, landed)}
    l3_hooks.converge_prefetches()
    events = cache_hooks.poll_ready_events()
    assert [(type(e).__name__, e.op_id, e.landed_pages) for e in events] == [
        ("PrefetchDoneEvent", op.op_id, landed)
    ]
    advance_scheduler(scheduler, events)
    # The one ticket was the one ACK: nothing further comes out of the poll.
    assert cache_hooks.poll_ready_events() == []

    # Admitted as a Host hit of exactly the landed prefix: the first chunk
    # starts there and the L2 load-back brings those pages, nothing more.
    plan = scheduler.next_execution_plan()
    (forward,) = [f for f in plan.forward if f.request_ids]
    assert list(forward.request_ids) == ["a"]
    assert list(forward.extend_prefix_lens) == [landed * PAGE]
    assert list(forward.input_lengths) == [len(PROMPT) - landed * PAGE]
    loads = [c for c in plan.cache if isinstance(c, ts.Cache.LoadBackOp)]
    assert len(loads) == 1 and len(loads[0].src_pages[0]) == landed
    assert scheduler.waiting_size() == 0

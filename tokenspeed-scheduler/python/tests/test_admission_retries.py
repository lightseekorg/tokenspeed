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

"""Admission retries reuse prompt identity while observing current cache contents."""

import pytest
import tokenspeed_scheduler as ts


@pytest.mark.parametrize("disable_prefix_cache", [False, True])
@pytest.mark.parametrize("enable_kv_events", [False, True])
def test_capacity_blocked_request_observes_prefix_published_while_waiting(
    disable_prefix_cache, enable_kv_events
):
    cfg = ts.SchedulerConfig()
    cfg.role = ts.SchedulerConfig.Role.D
    cfg.prefix_granularity = 64
    cfg.max_scheduled_tokens = 256
    cfg.max_batch_size = 8
    cfg.num_device_pages = 5
    cfg.num_snapshot_pages = 1
    cfg.disable_l2_cache = True
    cfg.disable_prefix_cache = disable_prefix_cache
    cfg.enable_kv_cache_events = enable_kv_events
    cfg.cache_groups = [
        ts.CacheGroupConfig(
            group_id="history",
            block_granularity=64,
            total_pages=5,
            attn_kind=ts.AttnKind.Full,
            transfer_policy=ts.CacheTransferPolicy.FullSuffix,
        )
    ]
    scheduler = ts.Scheduler(cfg)
    resident = ts.RequestSpec()
    resident.request_id = "resident"
    resident.tokens = list(range(128))
    resident.max_new_tokens = 4
    scheduler.submit_requests([resident])
    scheduler.advance(
        ts.ExecutionEvent().add_event(ts.PD.BootstrappedEvent("resident"))
    )
    plan = scheduler.next_execution_plan()
    assert list(plan.remote_prefill.request_ids) == ["resident"]

    waiting = ts.RequestSpec()
    waiting.request_id = "waiting"
    waiting.tokens = list(range(129))
    waiting.max_new_tokens = 4
    scheduler.submit_requests([waiting])
    scheduler.advance(ts.ExecutionEvent().add_event(ts.PD.BootstrappedEvent("waiting")))
    for _ in range(4):
        plan = scheduler.next_execution_plan()
        assert plan.remote_prefill is None
        assert scheduler.waiting_size() == 1
        assert scheduler.drain_kv_events() == []

    scheduler.advance(
        ts.ExecutionEvent().add_event(ts.PD.RemotePrefillDoneEvent("resident", 999))
    )
    finish = ts.ForwardEvent.Finish()
    finish.request_id = "resident"
    scheduler.advance(ts.ExecutionEvent().add_event(finish))
    published = scheduler.drain_kv_events()
    plan = scheduler.next_execution_plan()
    assert list(plan.remote_prefill.request_ids) == ["waiting"]
    assert list(plan.remote_prefill.extend_prefix_lens) == [
        0 if disable_prefix_cache else 128
    ]
    assert bool(published) == (enable_kv_events and not disable_prefix_cache)

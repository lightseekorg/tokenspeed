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

"""D capacity retraction must readmit through local recovery without L2."""

import pytest
import tokenspeed_scheduler as ts


@pytest.mark.parametrize("queries", [1, 4])
def test_d_retraction_recovers_locally_and_releases_history(queries):
    cfg = ts.SchedulerConfig()
    cfg.role = ts.SchedulerConfig.Role.D
    cfg.prefix_granularity = 64
    cfg.num_device_pages = 81
    cfg.max_batch_size = 2
    cfg.max_scheduled_tokens = 64
    cfg.decode_input_tokens = queries
    cfg.disable_l2_cache = True
    cfg.disable_prefix_cache = True
    cfg.cache_groups = [
        ts.CacheGroupConfig(
            group_id="history",
            block_granularity=64,
            total_pages=81,
            retention=ts.CacheRetention.FullHistory,
            family=ts.CacheGroupFamily.History,
            transfer_policy=ts.CacheTransferPolicy.FullSuffix,
        )
    ]
    scheduler = ts.Scheduler(cfg)

    def submit(rid, length, budget):
        spec = ts.RequestSpec()
        spec.request_id = rid
        spec.tokens = list(range(length))
        spec.max_new_tokens = budget
        scheduler.submit_requests([spec])
        scheduler.advance(ts.ExecutionEvent().add_event(ts.PD.BootstrappedEvent(rid)))

    def event(
        kind, rid, *, tokens=None, reserve_num_tokens_in_next_schedule_event=None
    ):
        e = kind()
        e.request_id = rid
        if tokens is not None:
            e.tokens = tokens
        if reserve_num_tokens_in_next_schedule_event is not None:
            e.reserve_num_tokens_in_next_schedule_event = (
                reserve_num_tokens_in_next_schedule_event
            )
        scheduler.advance(ts.ExecutionEvent().add_event(e))

    submit("victim", 65, 4200)
    initial = scheduler.next_execution_plan()
    assert list(initial.remote_prefill.request_ids) == ["victim"]
    scheduler.advance(
        ts.ExecutionEvent().add_event(ts.PD.RemotePrefillDoneEvent("victim", 100))
    )
    assert scheduler.next_execution_plan().forward[0].num_extends() == 0
    event(ts.ForwardEvent.ExtendResult, "victim", tokens=[101])
    submit("blocker", 5057, 16)
    # Force a cache growth miss after draining the previous forward. This
    # exercises chooseVictim/retractVictim, not a synthetic Retract event.
    event(
        ts.ForwardEvent.UpdateReserveNumTokens,
        "victim",
        reserve_num_tokens_in_next_schedule_event=5120,
    )
    retracted = scheduler.next_execution_plan()
    assert list(retracted.remote_prefill.request_ids) == ["blocker"]
    assert all(not list(batch.request_ids) for batch in retracted.forward)
    assert not list(retracted.cache)
    event(ts.ForwardEvent.Abort, "blocker")

    prefixes = []
    for _ in range(8):
        plan = scheduler.next_execution_plan()
        assert plan.remote_prefill is None
        assert len(plan.forward) == 1
        batch = plan.forward[0]
        assert list(batch.request_ids) == ["victim"]
        if batch.num_extends() == 0:
            event(ts.ForwardEvent.ExtendResult, "victim", tokens=[102])
            break
        assert batch.num_extends() == 1
        prefixes.extend(list(batch.extend_prefix_lens))
        assert sum(batch.input_lengths) <= 64
        complete = batch.extend_prefix_lens[0] + batch.input_lengths[0] >= 67
        event(ts.ForwardEvent.ExtendResult, "victim", tokens=[102] if complete else [])
    else:
        pytest.fail("recovery did not return to decode")
    assert prefixes[0] == 0 and len(prefixes) >= 2
    event(ts.ForwardEvent.Finish, "victim")
    scheduler.next_execution_plan()
    assert scheduler.active_lcm_blocks() == 0

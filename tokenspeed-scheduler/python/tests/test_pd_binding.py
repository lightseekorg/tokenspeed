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

"""Tests for PD Python bindings."""

from tokenspeed_scheduler import (
    PD,
    CacheGroupConfig,
    CacheGroupFamily,
    CacheRetention,
    CacheTransferPolicy,
    ExecutionEvent,
    ForwardEvent,
    RequestSpec,
    Scheduler,
    SchedulerConfig,
)


def make_scheduler() -> Scheduler:
    cfg = SchedulerConfig()
    cfg.prefix_granularity = 16
    cfg.max_scheduled_tokens = 32
    cfg.max_batch_size = 4
    cfg.num_device_pages = 1024
    cfg.cache_groups = [
        CacheGroupConfig(
            group_id="full_attention",
            block_granularity=cfg.prefix_granularity,
            total_pages=cfg.num_device_pages,
            retention=CacheRetention.FullHistory,
            family=CacheGroupFamily.History,
        )
    ]
    return Scheduler(cfg)


def make_spec(request_id: str, tokens: list[int]) -> RequestSpec:
    spec = RequestSpec()
    spec.request_id = request_id
    spec.tokens = tokens
    return spec


def test_pd_event_fields_are_bound():
    """PD event objects require request_id constructor arg and expose it as read-only."""
    event = PD.BootstrappedEvent("req-0")

    assert event.request_id == "req-0"


def test_execution_event_accepts_pd_events():
    """ExecutionEvent.add_event accepts PD events and returns self for chaining."""
    execution_event = ExecutionEvent()
    event = PD.SucceededEvent("req-0")

    assert execution_event.add_event(event) is execution_event


def test_execution_plan_exposes_forward():
    scheduler = make_scheduler()
    scheduler.submit_requests([make_spec("r0", [1, 2, 3, 4])])

    plan = scheduler.next_execution_plan()

    assert len(plan.forward) == 1
    assert plan.forward[0].request_ids == ["r0"]


def test_pd_counters_follow_request_state():
    cfg = SchedulerConfig()
    cfg.role = SchedulerConfig.Role.D
    cfg.prefix_granularity = 16
    cfg.max_scheduled_tokens = 32
    cfg.max_batch_size = 4
    cfg.num_device_pages = 64
    cfg.disable_l2_cache = True
    cfg.cache_groups = [
        CacheGroupConfig(
            group_id="history",
            block_granularity=16,
            total_pages=64,
            transfer_policy=CacheTransferPolicy.FullSuffix,
            retention=CacheRetention.FullHistory,
            family=CacheGroupFamily.History,
        )
    ]
    scheduler = Scheduler(cfg)
    scheduler.submit_requests([make_spec("remote", [1, 2, 3, 4])])
    assert scheduler.bootstrapping_size() == 1
    assert scheduler.remote_prefilling_size() == scheduler.pd_transfer_size() == 0
    scheduler.advance(ExecutionEvent().add_event(PD.BootstrappedEvent("remote")))
    assert scheduler.bootstrapping_size() == 0
    scheduler.next_execution_plan()
    assert scheduler.remote_prefilling_size() == scheduler.pd_transfer_size() == 1
    scheduler.advance(
        ExecutionEvent().add_event(PD.RemotePrefillDoneEvent("remote", 5))
    )
    assert scheduler.remote_prefilling_size() == scheduler.pd_transfer_size() == 0
    finish = ForwardEvent.Finish()
    finish.request_id = "remote"
    scheduler.advance(ExecutionEvent().add_event(finish))
    assert scheduler.active_lcm_blocks() == 0


def test_prefill_role_reserves_the_decode_window_on_the_completing_chunk():
    """The P role never decodes, but the chunk that completes a prompt drafts
    the first candidate window, so it reserves ``decode_input_tokens`` exactly
    like a decoding role; intermediate chunks hold only their own tokens."""
    cfg = SchedulerConfig()
    cfg.role = SchedulerConfig.Role.P
    cfg.prefix_granularity = 2
    cfg.max_scheduled_tokens = 4
    cfg.max_batch_size = 1
    cfg.num_device_pages = 17
    cfg.disable_l2_cache = True
    cfg.decode_input_tokens = 3
    cfg.cache_groups = [
        CacheGroupConfig(
            group_id="history",
            block_granularity=2,
            total_pages=17,
            transfer_policy=CacheTransferPolicy.FullSuffix,
            retention=CacheRetention.FullHistory,
            family=CacheGroupFamily.History,
        )
    ]
    scheduler = Scheduler(cfg)
    scheduler.submit_requests([make_spec("chunked", list(range(8)))])
    scheduler.advance(ExecutionEvent().add_event(PD.BootstrappedEvent("chunked")))

    def held_pages(batch) -> int:
        return len(
            [page for page in dict(batch.block_tables)["history"][0] if page > 0]
        )

    first_chunk = scheduler.next_execution_plan().forward[0]
    assert list(first_chunk.input_lengths) == [4]
    assert held_pages(first_chunk) == 2, "tokens 0..3 only, no reserve yet"

    completing_chunk = scheduler.next_execution_plan().forward[0]
    assert list(completing_chunk.input_lengths) == [4]
    assert held_pages(completing_chunk) == 6, "tokens 0..7 plus a 3-token window"

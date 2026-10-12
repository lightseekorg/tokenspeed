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

import pytest
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
    cfg.num_snapshot_pages = 1  # never retracts
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
    cfg.num_snapshot_pages = 1  # never retracts
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
    cfg.num_snapshot_pages = 1  # the P role never retracts
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


@pytest.mark.parametrize("decode_input_tokens", [1, 4])
@pytest.mark.parametrize("max_new_tokens", [0, 32])
def test_remote_prefill_crosses_partial_prefix_promotion(
    decode_input_tokens, max_new_tokens
):
    cfg = SchedulerConfig()
    cfg.role = SchedulerConfig.Role.D
    cfg.prefix_granularity = 128
    cfg.max_scheduled_tokens = 1024
    cfg.max_batch_size = 2
    cfg.num_device_pages = 129
    cfg.num_snapshot_pages = 1  # never retracts
    cfg.decode_input_tokens = decode_input_tokens
    cfg.overlap_schedule_depth = 0
    cfg.disable_l2_cache = True
    cfg.enable_l3_storage = False
    cfg.disable_prefix_cache = False
    cfg.cache_groups = [
        CacheGroupConfig(
            group_id="history",
            block_granularity=64,
            total_pages=257,
            cache_blocks_per_lcm_block=2,
            family=CacheGroupFamily.History,
            retention=CacheRetention.FullHistory,
            transfer_policy=CacheTransferPolicy.FullSuffix,
        ),
        CacheGroupConfig(
            group_id="window",
            block_granularity=32,
            total_pages=129,
            cache_blocks_per_lcm_block=1,
            family=CacheGroupFamily.History,
            retention=CacheRetention.SlidingWindow,
            sliding_window_tokens=513,
            transfer_policy=CacheTransferPolicy.FullSuffix,
        ),
    ]
    scheduler = Scheduler(cfg)
    tokens = list(range(769))
    for request_id, prompt in (
        ("cold", tokens),
        ("warm", tokens),
        ("partial", tokens[:640] + list(range(10000, 10129))),
    ):
        request = make_spec(request_id, prompt)
        request.max_new_tokens = max_new_tokens
        scheduler.submit_requests([request])
        scheduler.advance(ExecutionEvent().add_event(PD.BootstrappedEvent(request_id)))
        remote = scheduler.next_execution_plan().remote_prefill
        assert remote is not None
        prefix = remote.extend_prefix_lens[0]
        assert prefix == (768 if request_id == "warm" else 0)
        assert prefix + remote.input_lengths[0] == len(prompt)
        tables = dict(remote.block_tables)
        assert all(page > 0 for page in tables["history"][0][:13])
        assert len(tables["history"][0]) >= 13
        assert all(page > 0 for page in tables["window"][0][8:25])
        assert len(tables["window"][0]) >= 25
        scheduler.advance(
            ExecutionEvent().add_event(PD.RemotePrefillDoneEvent(request_id, 9001))
        )
        assert scheduler.next_execution_plan().forward
        finish = ForwardEvent.Finish()
        finish.request_id = request_id
        scheduler.advance(ExecutionEvent().add_event(finish))
        scheduler.next_execution_plan()
        assert scheduler.active_lcm_blocks() == scheduler.pd_transfer_size() == 0
    scheduler.clear_cache()
    assert scheduler.empty_lcm_blocks() == 128

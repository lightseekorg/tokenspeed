from __future__ import annotations

from collections import deque

import pytest
from tokenspeed_scheduler import (
    CacheGroupConfig,
    CacheGroupFamily,
    CacheRetention,
    ExecutionEvent,
    ForwardEvent,
    RequestSpec,
    Scheduler,
    SchedulerConfig,
)


def _make_spec(request_id: str, tokens: list[int]) -> RequestSpec:
    spec = RequestSpec()
    spec.request_id = request_id
    spec.tokens = tokens
    return spec


def _advance_tokens(scheduler: Scheduler, request_id: str, tokens: list[int]) -> None:
    event = ForwardEvent.ExtendResult()
    event.request_id = request_id
    event.tokens = tokens
    execution_event = ExecutionEvent()
    execution_event.add_event(event)
    scheduler.advance(execution_event)


def _send_reserve(scheduler: Scheduler, request_id: str, n: int = 0) -> None:
    event = ForwardEvent.UpdateReserveNumTokens()
    event.request_id = request_id
    event.reserve_num_tokens_in_next_schedule_event = n
    execution_event = ExecutionEvent()
    execution_event.add_event(event)
    scheduler.advance(execution_event)


def _base_config(num_device_pages: int = 64) -> SchedulerConfig:
    cfg = SchedulerConfig()
    cfg.prefix_granularity = 64
    cfg.max_scheduled_tokens = 4096
    cfg.max_batch_size = 8
    cfg.num_device_pages = num_device_pages
    cfg.disable_l2_cache = True
    return cfg


def _request_ids_in_plan(plan) -> set[str]:
    out = set()
    for op in plan.forward:
        out.update(op.request_ids)
    return out


def _overlap_admission_scheduler(verify_width: int) -> Scheduler:
    committed_tokens = 3
    reservation_end = committed_tokens - 1 + verify_width
    # One additional verify window stays protected while the scheduler runs
    # one step ahead of the Device.
    protected_pages = verify_width
    total_pages = reservation_end + 1 + protected_pages
    cfg = _base_config(num_device_pages=total_pages)
    cfg.prefix_granularity = 1
    cfg.decode_input_tokens = verify_width
    cfg.overlap_schedule_depth = 1
    # Cache group page 0 is reserved by the allocator.
    cfg.cache_groups = [
        CacheGroupConfig(
            group_id="overlap.history",
            block_granularity=1,
            total_pages=total_pages,
            retention=CacheRetention.FullHistory,
            family=CacheGroupFamily.History,
        )
    ]
    scheduler = Scheduler(cfg)
    scheduler.submit_requests([_make_spec("r", [1, 2])])
    assert _request_ids_in_plan(scheduler.next_execution_plan()) == {"r"}
    _advance_tokens(scheduler, "r", [3])
    return scheduler


@pytest.mark.parametrize("verify_width", [1, 2, 4, 8])
def test_overlap_decode_admission_uses_runtime_verify_width(verify_width: int):
    scheduler = _overlap_admission_scheduler(verify_width)
    assert _request_ids_in_plan(scheduler.next_execution_plan()) == {"r"}
    assert scheduler.cache_group_available_pages("overlap.history") == verify_width


@pytest.mark.parametrize("overlap_depth", [0, 1])
@pytest.mark.parametrize("accepted_tokens", [1, 3, 4])
@pytest.mark.parametrize("mixed", [False, True])
def test_speculative_reservations_do_not_accumulate_across_prefill_interruptions(
    overlap_depth: int, accepted_tokens: int, mixed: bool
):
    cfg = _base_config(num_device_pages=4096)
    cfg.prefix_granularity = 64
    cfg.decode_input_tokens = 4
    cfg.overlap_schedule_depth = overlap_depth
    cfg.enable_mixed_prefill_decode = mixed
    cfg.disable_prefix_cache = True
    cfg.cache_groups = [
        CacheGroupConfig(
            group_id="history",
            block_granularity=64,
            total_pages=4096,
        ),
        CacheGroupConfig(
            group_id="compressor",
            block_granularity=4,
            total_pages=4096,
            retention=CacheRetention.SlidingWindow,
            family=CacheGroupFamily.History,
            sliding_window_tokens=8,
        ),
        CacheGroupConfig(
            group_id="state",
            block_granularity=64,
            total_pages=4096,
            family=CacheGroupFamily.State,
        ),
    ]
    scheduler = Scheduler(cfg)
    scheduler.submit_requests([_make_spec("long", list(range(16)))])
    pending = deque()
    for step in range(100):
        if step % 5 == 3:
            scheduler.submit_requests([_make_spec(f"short-{step}", list(range(16)))])
        for op in scheduler.next_execution_plan().forward:
            if "long" in op.request_ids and op.num_extends() == 0:
                row = op.request_ids.index("long")
                table = op.block_tables["compressor"][row]
                committed = scheduler.request_token_size("long") - 1
                reservation_end = (
                    committed + (overlap_depth + 1) * cfg.decode_input_tokens
                )
                assert len(table) <= (reservation_end + 3) // 4
                assert table[committed // 4] > 0
            pending.append(op)
        while len(pending) > overlap_depth:
            op = pending.popleft()
            events = ExecutionEvent()
            for index, rid in enumerate(op.request_ids):
                result = ForwardEvent.ExtendResult()
                result.request_id = rid
                result.tokens = [1] * (
                    accepted_tokens if index >= op.num_extends() else 1
                )
                events.add_event(result)
                if rid != "long":
                    finish = ForwardEvent.Finish()
                    finish.request_id = rid
                    events.add_event(finish)
                elif index >= op.num_extends():
                    reserve = ForwardEvent.UpdateReserveNumTokens()
                    reserve.request_id = rid
                    reserve.reserve_num_tokens_in_next_schedule_event = accepted_tokens
                    events.add_event(reserve)
            scheduler.advance(events)


def test_overlap_schedule_depth_defaults_to_zero_and_rejects_deeper_pipeline():
    assert SchedulerConfig().overlap_schedule_depth == 0
    cfg = _base_config()
    cfg.cache_groups = [
        CacheGroupConfig(
            group_id="history",
            block_granularity=cfg.prefix_granularity,
            total_pages=cfg.num_device_pages,
        )
    ]
    for invalid_depth in (-1, 2):
        cfg.overlap_schedule_depth = invalid_depth
        with pytest.raises(ValueError, match="overlap_schedule_depth"):
            Scheduler(cfg)

    cfg.overlap_schedule_depth = 1
    cfg.decode_input_tokens = 0
    with pytest.raises(ValueError, match="decode_input_tokens"):
        Scheduler(cfg)

    cfg.overlap_schedule_depth = 0
    cfg.decode_input_tokens = -1
    with pytest.raises(ValueError, match="decode_input_tokens"):
        Scheduler(cfg)


def test_sliding_release_before_admit_prevents_oom():
    cfg = _base_config(num_device_pages=8)
    cfg.prefix_granularity = 2
    cfg.max_scheduled_tokens = 1024
    cfg.cache_groups = [
        CacheGroupConfig(
            group_id="swa.test",
            block_granularity=2,
            total_pages=8,
            retention=CacheRetention.SlidingWindow,
            sliding_window_tokens=4,
        )
    ]
    scheduler = Scheduler(cfg)

    scheduler.submit_requests([_make_spec("r0", list(range(8)))])
    scheduler.next_execution_plan()
    scheduler.next_execution_plan()

    for step in range(40):
        _send_reserve(scheduler, "r0", 1)
        plan = scheduler.next_execution_plan()
        assert "r0" in _request_ids_in_plan(plan)
        _advance_tokens(scheduler, "r0", [10_000 + step])


def test_batch_admission_debits_simulated_free_pages():
    cfg = _base_config(num_device_pages=12)
    cfg.prefix_granularity = 2
    cfg.max_batch_size = 4
    cfg.max_scheduled_tokens = 512
    cfg.cache_groups = [
        CacheGroupConfig(
            group_id=f"swa.g{i}",
            block_granularity=2,
            total_pages=12,
            retention=CacheRetention.SlidingWindow,
            sliding_window_tokens=4,
        )
        for i in range(2)
    ]

    scheduler = Scheduler(cfg)
    scheduler.submit_requests(
        [_make_spec("r0", list(range(8))), _make_spec("r1", list(range(8)))]
    )

    plan = scheduler.next_execution_plan()
    admitted = _request_ids_in_plan(plan)
    assert len(admitted & {"r0", "r1"}) <= 1


def test_group_tables_use_each_groups_block_granularity():
    cfg = _base_config(num_device_pages=17)
    cfg.prefix_granularity = 8
    cfg.cache_groups = [
        CacheGroupConfig(
            group_id="history",
            block_granularity=8,
            total_pages=17,
            retention=CacheRetention.FullHistory,
            family=CacheGroupFamily.History,
        ),
        CacheGroupConfig(
            group_id="swa",
            block_granularity=2,
            total_pages=65,
            cache_blocks_per_lcm_block=4,
            retention=CacheRetention.SlidingWindow,
            sliding_window_tokens=4,
            family=CacheGroupFamily.History,
        ),
    ]
    scheduler = Scheduler(cfg)
    scheduler.submit_requests([_make_spec("r", list(range(8)))])

    plan = scheduler.next_execution_plan()
    operation = next(op for op in plan.forward if "r" in op.request_ids)
    tables = dict(operation.block_tables)

    # The first round covers eight prompt tokens plus one decode-reserve token.
    assert len(tables["history"][0]) == 2
    assert len(tables["swa"][0]) == 5


def _hybrid_chunked_scheduler(num_usable_pages: int) -> Scheduler:
    """Fused role, P=4, 8-token chunks: one full-history group beside one
    sliding-window group (window 4), both one page per LCM block."""
    cfg = _base_config(num_device_pages=num_usable_pages + 1)
    cfg.prefix_granularity = 4
    cfg.max_scheduled_tokens = 8
    cfg.decode_input_tokens = 1
    cfg.cache_groups = [
        CacheGroupConfig(
            group_id="history",
            block_granularity=4,
            total_pages=cfg.num_device_pages,
            retention=CacheRetention.FullHistory,
            family=CacheGroupFamily.History,
        ),
        CacheGroupConfig(
            group_id="swa",
            block_granularity=4,
            total_pages=cfg.num_device_pages,
            retention=CacheRetention.SlidingWindow,
            sliding_window_tokens=4,
            family=CacheGroupFamily.History,
        ),
    ]
    return Scheduler(cfg)


def test_first_chunk_prepays_prompt_headroom_only_in_full_history_groups():
    # A 32-token prompt with max_new_tokens=8 admits its first 8-token chunk on
    # a decoding role with 24 unscheduled prompt tokens + 8 tokens of decode
    # headroom prepaid. The full-history group must hold that: 8 + 32 tokens
    # -> 10 pages. The sliding-window group recycles slid-out pages, so with no
    # other request holding pages the rest of the prompt costs it nothing: it
    # holds only the chunk, 2 pages.
    # 12 pages fit a 16-page pool; had the headroom been broadcast to both
    # groups (20 pages) the request would have sat in the waiting queue.
    scheduler = _hybrid_chunked_scheduler(num_usable_pages=16)
    request = _make_spec("long", list(range(32)))
    request.max_new_tokens = 8
    scheduler.submit_requests([request])

    plan = scheduler.next_execution_plan()

    operation = next(op for op in plan.forward if "long" in op.request_ids)
    assert operation.input_lengths == [8]
    tables = dict(operation.block_tables)
    assert len(tables["history"][0]) == 10
    assert len(tables["swa"][0]) == 2
    assert scheduler.cache_group_available_pages("history") == 4
    assert scheduler.waiting_size() == 0
    assert scheduler.prefilling_size() == 1

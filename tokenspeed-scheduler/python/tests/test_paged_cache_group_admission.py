from __future__ import annotations

import pytest
from tokenspeed_scheduler import (
    ExecutionEvent,
    ForwardEvent,
    PagedCacheGroupConfig,
    PagedCacheGroupFamily,
    PagedCacheRetention,
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
    cfg.block_size = 64
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


def _overlap_admission_scheduler(
    verify_width: int, *, exact_capacity: bool
) -> Scheduler:
    committed_tokens = 3
    reservation_end = committed_tokens - 1 + verify_width
    total_pages = reservation_end + (1 if exact_capacity else 0)
    cfg = _base_config(num_device_pages=total_pages)
    cfg.block_size = 1
    cfg.decode_input_tokens = verify_width
    cfg.overlap_schedule_depth = 1
    # Paged-cache group page 0 is reserved by the allocator.
    cfg.paged_cache_groups = [
        PagedCacheGroupConfig(
            group_id="overlap.history",
            rows_per_page=1,
            entry_stride_tokens=1,
            total_pages=total_pages,
            retention=PagedCacheRetention.FullHistory,
            family=PagedCacheGroupFamily.History,
        )
    ]
    scheduler = Scheduler(cfg)
    scheduler.submit_requests([_make_spec("r", [1, 2])])
    initial_request_ids = _request_ids_in_plan(scheduler.next_execution_plan())
    if exact_capacity:
        assert initial_request_ids == {"r"}
        _advance_tokens(scheduler, "r", [3])
    else:
        assert initial_request_ids == set()
    return scheduler


@pytest.mark.parametrize("verify_width", [1, 2, 4, 8])
def test_overlap_decode_admission_uses_runtime_verify_width(verify_width: int):
    scheduler = _overlap_admission_scheduler(verify_width, exact_capacity=True)
    assert _request_ids_in_plan(scheduler.next_execution_plan()) == {"r"}
    assert scheduler.paged_cache_group_available_pages("overlap.history") == 0
    assert scheduler.paged_cache_group_failed_alloc_count("overlap.history") == 0


@pytest.mark.parametrize("verify_width", [1, 2, 4, 8])
def test_overlap_admission_rejects_one_page_short(verify_width: int):
    scheduler = _overlap_admission_scheduler(verify_width, exact_capacity=False)
    assert scheduler.paged_cache_group_available_pages("overlap.history") == (
        1 + verify_width
    )
    assert scheduler.paged_cache_group_failed_alloc_count("overlap.history") == 0


def test_overlap_schedule_depth_defaults_to_zero_and_rejects_deeper_pipeline():
    assert SchedulerConfig().overlap_schedule_depth == 0
    cfg = _base_config()
    cfg.paged_cache_groups = [
        PagedCacheGroupConfig(
            group_id="history",
            rows_per_page=cfg.block_size,
            entry_stride_tokens=1,
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
    cfg.block_size = 2
    cfg.max_scheduled_tokens = 1024
    cfg.paged_cache_groups = [
        PagedCacheGroupConfig(
            group_id="swa.test",
            rows_per_page=2,
            entry_stride_tokens=1,
            total_pages=8,
            retention=PagedCacheRetention.SlidingWindow,
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

    assert scheduler.paged_cache_group_failed_alloc_count("swa.test") == 0


def test_batch_admission_debits_simulated_free_pages():
    cfg = _base_config(num_device_pages=4)
    cfg.block_size = 2
    cfg.max_batch_size = 4
    cfg.max_scheduled_tokens = 512
    cfg.paged_cache_groups = [
        PagedCacheGroupConfig(
            group_id=f"swa.g{i}",
            rows_per_page=2,
            entry_stride_tokens=1,
            total_pages=4,
            retention=PagedCacheRetention.SlidingWindow,
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
    for gid in ("swa.g0", "swa.g1"):
        assert scheduler.paged_cache_group_failed_alloc_count(gid) == 0

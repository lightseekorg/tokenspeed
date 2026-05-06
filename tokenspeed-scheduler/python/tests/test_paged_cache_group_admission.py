from tokenspeed_scheduler import (
    ExecutionEvent,
    ForwardEvent,
    PagedCacheGroupConfig,
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
    cfg.page_size = 64
    cfg.max_scheduled_tokens = 4096
    cfg.max_batch_size = 8
    cfg.num_device_pages = num_device_pages
    cfg.disable_l2_cache = True
    return cfg


def _compressed_group(total_pages: int) -> PagedCacheGroupConfig:
    return PagedCacheGroupConfig(
        group_id="compressed.4",
        rows_per_page=64,
        entry_stride_tokens=4,
        total_pages=total_pages,
        retention=PagedCacheRetention.FullHistory,
    )


def _request_ids_in_plan(plan) -> set[str]:
    out = set()
    for op in plan.forward:
        out.update(op.request_ids)
    return out


def test_full_history_admission_denies_instead_of_throwing():
    cfg = _base_config()
    cfg.max_batch_size = 4
    cfg.paged_cache_groups = [_compressed_group(total_pages=2)]

    scheduler = Scheduler(cfg)
    scheduler.submit_requests([_make_spec("r0", list(range(256)))])
    plan = scheduler.next_execution_plan()
    assert "r0" in _request_ids_in_plan(plan)
    assert len(scheduler.get_request_paged_cache_page_ids("r0", "compressed.4")) == 1

    scheduler.submit_requests([_make_spec("r1", list(range(256)))])
    plan2 = scheduler.next_execution_plan()
    assert "r1" not in _request_ids_in_plan(plan2)
    assert scheduler.paged_cache_group_failed_alloc_count("compressed.4") == 0


def test_full_history_stride_admission_accounts_partial_entries():
    cfg = _base_config()
    cfg.max_scheduled_tokens = 512
    cfg.max_batch_size = 4
    cfg.paged_cache_groups = [_compressed_group(total_pages=4)]

    scheduler = Scheduler(cfg)
    scheduler.submit_requests([_make_spec("short", [1])])
    plan = scheduler.next_execution_plan()
    assert "short" in _request_ids_in_plan(plan)
    assert len(scheduler.get_request_paged_cache_page_ids("short", "compressed.4")) == 1

    scheduler.submit_requests([_make_spec("boundary", list(range(257)))])
    plan2 = scheduler.next_execution_plan()
    assert "boundary" in _request_ids_in_plan(plan2)
    assert (
        len(scheduler.get_request_paged_cache_page_ids("boundary", "compressed.4")) == 2
    )


def test_sliding_release_before_admit_prevents_oom():
    cfg = _base_config(num_device_pages=256)
    cfg.page_size = 16
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
        _send_reserve(scheduler, "r0", 0)
        plan = scheduler.next_execution_plan()
        assert "r0" in _request_ids_in_plan(plan)
        _advance_tokens(scheduler, "r0", [10_000 + step])

    assert scheduler.paged_cache_group_failed_alloc_count("swa.test") == 0


def test_batch_admission_debits_simulated_free_pages():
    cfg = _base_config(num_device_pages=128)
    cfg.page_size = 16
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

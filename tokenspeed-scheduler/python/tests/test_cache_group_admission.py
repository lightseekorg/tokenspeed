from __future__ import annotations

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
            rows_per_page=1,
            entry_stride_tokens=1,
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


def test_overlap_schedule_depth_defaults_to_zero_and_rejects_deeper_pipeline():
    assert SchedulerConfig().overlap_schedule_depth == 0
    cfg = _base_config()
    cfg.cache_groups = [
        CacheGroupConfig(
            group_id="history",
            rows_per_page=cfg.prefix_granularity,
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
    cfg.prefix_granularity = 2
    cfg.max_scheduled_tokens = 1024
    cfg.cache_groups = [
        CacheGroupConfig(
            group_id="swa.test",
            rows_per_page=2,
            entry_stride_tokens=1,
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
            rows_per_page=2,
            entry_stride_tokens=1,
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


def test_v4_long_prompt_does_not_reserve_full_headroom_in_sliding_groups():
    num_lcm_blocks = 12_383
    cfg = _base_config(num_device_pages=num_lcm_blocks + 1)
    cfg.prefix_granularity = 256
    cfg.max_scheduled_tokens = 8192
    cfg.max_batch_size = 8
    cfg.decode_input_tokens = 6
    cfg.overlap_schedule_depth = 1
    cfg.disable_prefix_cache = False
    cfg.prefix_replay_tokens = 128

    def group(
        group_id: str,
        rows_per_page: int,
        entry_stride_tokens: int,
        packing: int,
        retention: CacheRetention,
        family: CacheGroupFamily,
        sliding_window_tokens: int | None,
    ) -> CacheGroupConfig:
        return CacheGroupConfig(
            group_id=group_id,
            rows_per_page=rows_per_page,
            entry_stride_tokens=entry_stride_tokens,
            total_pages=1 + num_lcm_blocks * packing,
            cache_blocks_per_lcm_block=packing,
            retention=retention,
            family=family,
            sliding_window_tokens=sliding_window_tokens,
        )

    cfg.cache_groups = [
        group(
            "v4.swa_kv",
            64,
            1,
            1,
            CacheRetention.SlidingWindow,
            CacheGroupFamily.State,
            128,
        ),
        group(
            "v4.c4a.compressed_kv",
            64,
            4,
            1,
            CacheRetention.FullHistory,
            CacheGroupFamily.History,
            None,
        ),
        group(
            "v4.c4a.compressor_state",
            4,
            1,
            2,
            CacheRetention.SlidingWindow,
            CacheGroupFamily.State,
            10,
        ),
        group(
            "v4.c4a.indexer_compressor_state",
            4,
            1,
            8,
            CacheRetention.SlidingWindow,
            CacheGroupFamily.State,
            10,
        ),
        group(
            "v4.c128a.compressed_kv",
            2,
            128,
            32,
            CacheRetention.FullHistory,
            CacheGroupFamily.History,
            None,
        ),
        group(
            "v4.c128a.compressor_state",
            8,
            1,
            2,
            CacheRetention.SlidingWindow,
            CacheGroupFamily.State,
            128,
        ),
    ]
    scheduler = Scheduler(cfg)
    request = _make_spec("v4-long", list(range(54_645)))
    request.max_new_tokens = 500
    scheduler.submit_requests([request])

    plan = scheduler.next_execution_plan()

    operation = next(op for op in plan.forward if "v4-long" in op.request_ids)
    assert operation.input_lengths == [8192]
    assert scheduler.waiting_size() == 0
    assert scheduler.prefilling_size() == 1


def test_group_tables_use_each_groups_block_granularity():
    cfg = _base_config(num_device_pages=17)
    cfg.prefix_granularity = 8
    cfg.cache_groups = [
        CacheGroupConfig(
            group_id="history",
            rows_per_page=8,
            entry_stride_tokens=1,
            total_pages=17,
            retention=CacheRetention.FullHistory,
            family=CacheGroupFamily.History,
        ),
        CacheGroupConfig(
            group_id="state",
            rows_per_page=2,
            entry_stride_tokens=1,
            total_pages=65,
            cache_blocks_per_lcm_block=4,
            retention=CacheRetention.SlidingWindow,
            sliding_window_tokens=4,
            family=CacheGroupFamily.State,
        ),
    ]
    scheduler = Scheduler(cfg)
    scheduler.submit_requests([_make_spec("r", list(range(8)))])

    plan = scheduler.next_execution_plan()
    operation = next(op for op in plan.forward if "r" in op.request_ids)
    tables = dict(operation.block_tables)

    # The first round covers eight prompt tokens plus one decode-reserve token.
    assert len(tables["history"][0]) == 2
    assert len(tables["state"][0]) == 5


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
            rows_per_page=4,
            entry_stride_tokens=1,
            total_pages=cfg.num_device_pages,
            retention=CacheRetention.FullHistory,
            family=CacheGroupFamily.History,
        ),
        CacheGroupConfig(
            group_id="swa",
            rows_per_page=4,
            entry_stride_tokens=1,
            total_pages=cfg.num_device_pages,
            retention=CacheRetention.SlidingWindow,
            sliding_window_tokens=4,
            family=CacheGroupFamily.State,
        ),
    ]
    return Scheduler(cfg)


def test_first_chunk_prepays_prompt_headroom_only_in_full_history_groups():
    # A 32-token prompt with max_new_tokens=8 admits its first 8-token chunk on
    # a decoding role with 24 unscheduled prompt tokens + 8 tokens of decode
    # headroom prepaid. The full-history group must hold that: 8 + 32 tokens
    # -> 10 pages. The sliding-window group recycles slid-out pages, so the
    # rest of the prompt costs it nothing: it holds only the chunk, 2 pages.
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

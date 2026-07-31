import pytest
from tokenspeed_scheduler import (
    ExecutionEvent,
    ForwardEvent,
    PagedCacheGroupConfig,
    PagedCacheGroupFamily,
    PagedCacheRetention,
    PrefixCacheAdjunctSpec,
    RequestSpec,
    Scheduler,
    SchedulerConfig,
)

COMPRESSED_GROUP_ID = "v4.c4a.compressed_kv"


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


def _compressed_group(total_pages: int) -> PagedCacheGroupConfig:
    return PagedCacheGroupConfig(
        group_id=COMPRESSED_GROUP_ID,
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


def _overlap_admission_scheduler(
    verify_width: int, *, exact_capacity: bool
) -> Scheduler:
    committed_tokens = 3
    reservation_end = committed_tokens + 2 * verify_width
    cfg = _base_config(num_device_pages=256)
    cfg.decode_input_tokens = verify_width
    cfg.overlap_schedule_depth = 1
    # Paged-cache group page 0 is reserved by the allocator.
    cfg.paged_cache_groups = [
        PagedCacheGroupConfig(
            group_id="overlap.history",
            rows_per_page=1,
            entry_stride_tokens=1,
            total_pages=reservation_end + (1 if exact_capacity else 0),
            retention=PagedCacheRetention.FullHistory,
            family=PagedCacheGroupFamily.History,
        )
    ]
    scheduler = Scheduler(cfg)
    scheduler.submit_requests([_make_spec("r", [1, 2])])
    assert _request_ids_in_plan(scheduler.next_execution_plan()) == {"r"}
    _advance_tokens(scheduler, "r", [3])
    return scheduler


@pytest.mark.parametrize("verify_width", [1, 2, 4, 8])
def test_overlap_decode_admission_uses_runtime_verify_width(verify_width: int):
    scheduler = _overlap_admission_scheduler(verify_width, exact_capacity=True)
    assert _request_ids_in_plan(scheduler.next_execution_plan()) == {"r"}
    assert scheduler.paged_cache_group_available_pages("overlap.history") == 0
    assert scheduler.paged_cache_group_failed_alloc_count("overlap.history") == 0


@pytest.mark.parametrize("verify_width", [1, 2, 4, 8])
def test_overlap_decode_admission_rejects_one_page_short(verify_width: int):
    scheduler = _overlap_admission_scheduler(verify_width, exact_capacity=False)
    assert _request_ids_in_plan(scheduler.next_execution_plan()) == set()
    assert scheduler.paged_cache_group_failed_alloc_count("overlap.history") == 0


def test_overlap_schedule_depth_defaults_to_zero_and_rejects_deeper_pipeline():
    assert SchedulerConfig().overlap_schedule_depth == 0
    cfg = _base_config()
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


def test_full_history_admission_denies_instead_of_throwing():
    cfg = _base_config()
    cfg.max_batch_size = 4
    cfg.paged_cache_groups = [_compressed_group(total_pages=2)]

    scheduler = Scheduler(cfg)
    scheduler.submit_requests([_make_spec("r0", list(range(256)))])
    plan = scheduler.next_execution_plan()
    assert "r0" in _request_ids_in_plan(plan)
    assert (
        len(scheduler.get_request_paged_cache_page_ids("r0", COMPRESSED_GROUP_ID)) == 1
    )

    scheduler.submit_requests([_make_spec("r1", list(range(256)))])
    plan2 = scheduler.next_execution_plan()
    assert "r1" not in _request_ids_in_plan(plan2)
    assert scheduler.paged_cache_group_failed_alloc_count(COMPRESSED_GROUP_ID) == 0


def test_full_history_stride_admission_accounts_partial_entries():
    cfg = _base_config()
    cfg.max_scheduled_tokens = 512
    cfg.max_batch_size = 4
    cfg.paged_cache_groups = [_compressed_group(total_pages=4)]

    scheduler = Scheduler(cfg)
    scheduler.submit_requests([_make_spec("short", [1])])
    plan = scheduler.next_execution_plan()
    assert "short" in _request_ids_in_plan(plan)
    assert (
        len(scheduler.get_request_paged_cache_page_ids("short", COMPRESSED_GROUP_ID))
        == 1
    )

    scheduler.submit_requests([_make_spec("boundary", list(range(257)))])
    plan2 = scheduler.next_execution_plan()
    assert "boundary" in _request_ids_in_plan(plan2)
    assert (
        len(scheduler.get_request_paged_cache_page_ids("boundary", COMPRESSED_GROUP_ID))
        == 2
    )


def test_sliding_release_before_admit_prevents_oom():
    cfg = _base_config(num_device_pages=256)
    cfg.block_size = 16
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
    cfg.block_size = 16
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


def _transport_state_checkpoint_scheduler(
    state_total_pages: int,
    *,
    max_scheduled_tokens: int = 64,
    requests: list[tuple[str, list[int]]] | None = None,
    decode_input_tokens: int = 1,
    enable_mixed_prefill_decode: bool = False,
    num_host_pages: int = 0,
) -> Scheduler:
    cfg = _base_config(num_device_pages=256)
    cfg.block_size = 16
    cfg.max_scheduled_tokens = max_scheduled_tokens
    cfg.max_batch_size = 2
    cfg.decode_input_tokens = decode_input_tokens
    cfg.enable_mixed_prefill_decode = enable_mixed_prefill_decode
    cfg.num_host_pages = num_host_pages
    cfg.paged_cache_groups = [
        PagedCacheGroupConfig(
            group_id="fh.test",
            rows_per_page=16,
            entry_stride_tokens=1,
            total_pages=256,
            retention=PagedCacheRetention.FullHistory,
            family=PagedCacheGroupFamily.History,
        ),
        PagedCacheGroupConfig(
            group_id="c4.test",
            rows_per_page=4,
            entry_stride_tokens=1,
            total_pages=state_total_pages,
            retention=PagedCacheRetention.SlidingWindow,
            sliding_window_tokens=8,
            family=PagedCacheGroupFamily.State,
        ),
    ]
    adjunct = PrefixCacheAdjunctSpec()
    adjunct.required_groups = ["fh.test"]
    cfg.prefix_cache_adjunct = adjunct

    scheduler = Scheduler(cfg)
    if requests is None:
        requests = [
            ("A", list(range(260))),
            ("B", list(range(1_000, 1_500))),
        ]
    scheduler.submit_requests(
        [_make_spec(request_id, tokens) for request_id, tokens in requests]
    )
    return scheduler


def _request_input_lengths(plan) -> dict[str, int]:
    return {
        request_id: input_length
        for op in plan.forward
        for request_id, input_length in zip(op.request_ids, op.input_lengths)
    }


def test_transport_state_terminal_checkpoint_credit_is_not_overcounted():
    scheduler = _transport_state_checkpoint_scheduler(state_total_pages=23)

    for _ in range(4):
        assert _request_ids_in_plan(scheduler.next_execution_plan()) == {"A"}

    split_plan = scheduler.next_execution_plan()
    assert _request_ids_in_plan(split_plan) == {"A", "B"}
    split_op = next(op for op in split_plan.forward if "B" in op.request_ids)
    assert list(split_op.request_ids) == ["A", "B"]
    assert list(split_op.input_lengths) == [4, 60]
    assert len(scheduler.get_request_paged_cache_page_ids("B", "c4.test")) == 15
    assert scheduler.get_request_paged_cache_base_logical_page("B", "c4.test") == 0

    # Publishing B's 48-token terminal checkpoint retains two owned state
    # pages as borrowed snapshot pages before ReleaseSkipped runs. Admission
    # must not count those pages as immediately available to the next acquire.
    next_plan = scheduler.next_execution_plan()
    assert _request_ids_in_plan(next_plan) == {"A"}
    assert len(scheduler.get_request_paged_cache_page_ids("B", "c4.test")) == 15
    assert scheduler.get_request_paged_cache_base_logical_page("B", "c4.test") == 0
    assert scheduler.paged_cache_group_failed_alloc_count("c4.test") == 0

    finish = ForwardEvent.Finish()
    finish.request_id = "A"
    execution_event = ExecutionEvent()
    execution_event.add_event(finish)
    scheduler.advance(execution_event)

    resumed_plan = scheduler.next_execution_plan()
    assert _request_ids_in_plan(resumed_plan) == {"B"}
    assert scheduler.paged_cache_group_failed_alloc_count("c4.test") == 0


def test_transport_state_terminal_checkpoint_exact_capacity_is_admitted():
    scheduler = _transport_state_checkpoint_scheduler(state_total_pages=24)

    for _ in range(4):
        assert _request_input_lengths(scheduler.next_execution_plan()) == {"A": 64}

    split_plan = scheduler.next_execution_plan()
    assert _request_input_lengths(split_plan) == {"A": 4, "B": 60}

    next_plan = scheduler.next_execution_plan()
    assert _request_input_lengths(next_plan) == {"B": 64}
    assert scheduler.paged_cache_group_available_pages("c4.test") == 0
    assert scheduler.paged_cache_group_failed_alloc_count("c4.test") == 0


def test_existing_terminal_state_snapshot_preserves_real_release_credit():
    scheduler = _transport_state_checkpoint_scheduler(
        state_total_pages=41,
        max_scheduled_tokens=136,
        enable_mixed_prefill_decode=True,
        requests=[
            ("A", list(range(68))),
            ("B", list(range(203))),
        ],
    )

    first_plan = scheduler.next_execution_plan()
    assert _request_input_lengths(first_plan) == {"A": 68, "B": 68}
    assert scheduler.paged_cache_group_available_pages("c4.test") == 6

    # A first publishes the depth-64 continuation snapshot. B reaches the
    # exact same terminal in this plan, so CommitChunk reuses that snapshot
    # and B's stale owned page remains real release credit.
    second_plan = scheduler.next_execution_plan()
    assert _request_input_lengths(second_plan) == {"B": 135, "A": 1}
    assert scheduler.paged_cache_group_available_pages("c4.test") == 0
    assert scheduler.paged_cache_group_failed_alloc_count("c4.test") == 0


def test_existing_terminal_state_snapshot_credit_is_branch_specific():
    scheduler = _transport_state_checkpoint_scheduler(
        state_total_pages=41,
        max_scheduled_tokens=136,
        enable_mixed_prefill_decode=True,
        requests=[
            ("A", list(range(68))),
            ("B", list(range(1_000, 1_203))),
        ],
    )

    first_plan = scheduler.next_execution_plan()
    assert _request_input_lengths(first_plan) == {"A": 68, "B": 68}

    # A has a continuation snapshot at the same depth, but B is on another
    # token branch and must still reserve the pages its own checkpoint retains.
    second_plan = scheduler.next_execution_plan()
    assert _request_input_lengths(second_plan) == {"A": 1}
    assert scheduler.paged_cache_group_failed_alloc_count("c4.test") == 0


# Cover both failed generic retract and available-host fallback paths.
@pytest.mark.parametrize("num_host_pages", [0, 256])
def test_transport_state_decode_checkpoint_credit_is_not_overcounted(
    num_host_pages: int,
):
    scheduler = _transport_state_checkpoint_scheduler(
        state_total_pages=20,
        max_scheduled_tokens=72,
        decode_input_tokens=64,
        enable_mixed_prefill_decode=True,
        num_host_pages=num_host_pages,
        requests=[
            ("A", list(range(68))),
            ("B", list(range(1_000, 1_004))),
        ],
    )

    first_plan = scheduler.next_execution_plan()
    assert _request_input_lengths(first_plan) == {"A": 68, "B": 4}
    assert scheduler.paged_cache_group_available_pages("c4.test") == 1

    # The terminal checkpoint retains one of the pages that ReleaseSkipped
    # would otherwise return, so A cannot safely fit at this capacity.
    second_plan = scheduler.next_execution_plan()
    assert _request_input_lengths(second_plan) == {}
    assert not second_plan.scheduler_aborts
    assert scheduler.paged_cache_group_available_pages("c4.test") == 1
    assert scheduler.paged_cache_group_failed_alloc_count("c4.test") == 0


def test_transport_state_decode_checkpoint_exact_capacity_is_admitted():
    scheduler = _transport_state_checkpoint_scheduler(
        state_total_pages=21,
        max_scheduled_tokens=72,
        decode_input_tokens=64,
        enable_mixed_prefill_decode=True,
        num_host_pages=256,
        requests=[
            ("A", list(range(68))),
            ("B", list(range(1_000, 1_004))),
        ],
    )

    first_plan = scheduler.next_execution_plan()
    assert _request_input_lengths(first_plan) == {"A": 68, "B": 4}
    assert scheduler.paged_cache_group_available_pages("c4.test") == 2

    second_plan = scheduler.next_execution_plan()
    assert _request_input_lengths(second_plan) == {"A": 64}
    assert scheduler.paged_cache_group_available_pages("c4.test") == 0
    assert scheduler.paged_cache_group_failed_alloc_count("c4.test") == 0

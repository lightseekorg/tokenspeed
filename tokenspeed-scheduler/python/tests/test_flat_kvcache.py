"""End-to-end test for the flat KV-cache path, driven through the nanobind
binding (no GPU).

Mirrors the C++ lifecycle test ``tests/cpp/test_flat_kvcache_lifecycle.cpp``:
a GPT-OSS-shaped config with two paged-cache groups (full-history + sliding
window) is driven Submit -> prefill -> decode, asserting the per-group flat
block tables (full-history keeps every page; sliding-window develops a null
hole; the two groups draw disjoint physical pages from the shared pool).

The nanobind extension is build-identical between the radix and flat builds --
``FlatForwardOp.flat_block_tables`` is always exposed -- but it is only
*populated* when the extension is built with ``TOKENSPEED_FLAT_KVCACHE=ON``.
On the default (radix) build it stays empty, so the whole module is
``skipif``-guarded on a behavioral probe and is a no-op there.
"""

from __future__ import annotations

import array
import gc

import pytest

ts = pytest.importorskip("tokenspeed_scheduler")


def _make_config() -> ts.SchedulerConfig:
    cfg = ts.SchedulerConfig()
    cfg.block_size = 2
    cfg.num_device_pages = 32
    cfg.num_host_pages = 32
    cfg.max_scheduled_tokens = 64
    cfg.max_batch_size = 8
    cfg.enable_l3_storage = False
    cfg.disable_l2_cache = True
    cfg.disable_prefix_cache = True

    full = ts.PagedCacheGroupConfig(
        group_id="full",
        rows_per_page=cfg.block_size,
        entry_stride_tokens=1,
        total_pages=cfg.num_device_pages,
        retention=ts.PagedCacheRetention.FullHistory,
        family=ts.PagedCacheGroupFamily.History,
    )
    swa = ts.PagedCacheGroupConfig(
        group_id="swa",
        rows_per_page=cfg.block_size,
        entry_stride_tokens=1,
        total_pages=cfg.num_device_pages,
        retention=ts.PagedCacheRetention.SlidingWindow,
        sliding_window_tokens=4,
        family=ts.PagedCacheGroupFamily.State,
    )
    cfg.paged_cache_groups = [full, swa]
    return cfg


def _make_state_only_config(*, disable_prefix_cache: bool) -> ts.SchedulerConfig:
    cfg = _make_config()
    total_blocks = 32
    cfg.num_device_pages = 0
    cfg.num_host_pages = 1
    cfg.disable_prefix_cache = disable_prefix_cache
    pool = ts.FlatBlockPoolConfig()
    pool.pool_id = "v4.swa"
    pool.total_blocks = total_blocks
    pool.bytes_per_block = 64
    cfg.flat_block_pools = [pool]
    state = ts.PagedCacheGroupConfig(
        group_id="v4.swa_kv",
        rows_per_page=cfg.block_size,
        entry_stride_tokens=1,
        total_pages=total_blocks,
        retention=ts.PagedCacheRetention.SlidingWindow,
        sliding_window_tokens=4,
        family=ts.PagedCacheGroupFamily.State,
        block_size=cfg.block_size,
        pool_id=pool.pool_id,
        prefix_role=ts.PagedCachePrefixRole.ContinuationState,
        table_layout=ts.PagedCacheTableLayout.BoundedWindow,
        required_producer_domain_mask=1,
        owner_mask=1,
    )
    cfg.paged_cache_groups = [state]
    return cfg


def _make_spec(
    request_id: str, num_pages: int, page_size: int = 2, start: int = 1
) -> ts.RequestSpec:
    # page_size must stay in sync with cfg.block_size: the token count below
    # (num_pages * page_size) is what determines how many pages get allocated.
    spec = ts.RequestSpec()
    spec.request_id = request_id
    spec.tokens = list(range(start, start + num_pages * page_size))
    return spec


def _advance_tokens(scheduler, request_id: str, tokens: list[int]) -> None:
    event = ts.ForwardEvent.ExtendResult()
    event.request_id = request_id
    event.tokens = tokens
    execution_event = ts.ExecutionEvent()
    execution_event.add_event(event)
    scheduler.advance(execution_event)


def _find_flat_op(plan) -> object:
    for op in plan.forward:
        tables = dict(op.flat_block_tables)
        if tables:
            return op
    return None


def _flat_build_available() -> bool:
    """Behavioral probe: the extension is flat-built iff a prefilled request's
    FlatForwardOp carries a non-empty ``flat_block_tables``. On the radix build
    the field is exposed but stays empty, so this returns False there *without
    raising* (an unexpected raise means a real regression and should surface as a
    collection error, not a silent skip)."""
    scheduler = ts.Scheduler(_make_config())
    scheduler.submit_requests([_make_spec("probe", num_pages=2)])
    plan = scheduler.next_execution_plan()
    return _find_flat_op(plan) is not None


pytestmark = pytest.mark.skipif(
    not _flat_build_available(),
    reason="extension not built with TOKENSPEED_FLAT_KVCACHE=ON",
)


def test_prefill_produces_per_group_block_tables():
    scheduler = ts.Scheduler(_make_config())
    scheduler.submit_requests([_make_spec("r1", num_pages=2)])

    plan = scheduler.next_execution_plan()
    op = _find_flat_op(plan)
    assert op is not None

    tables = dict(op.flat_block_tables)
    assert "full" in tables
    assert "swa" in tables

    full_row = list(tables["full"][0])
    assert full_row, "full-history row should be non-empty after prefill"
    assert all(page_id > 0 for page_id in full_row)


def test_copy_flat_block_table_to_returns_shape_and_writes_buffers():
    scheduler = ts.Scheduler(_make_config())
    scheduler.submit_requests([_make_spec("r1", num_pages=2)])

    op = _find_flat_op(scheduler.next_execution_plan())
    assert op is not None

    expected_table = list(op.flat_block_tables["full"][0])
    expected_base = op.flat_block_table_base_offsets["full"][0]
    table = array.array("i", [-1] * 32)
    bases = array.array("i", [-1] * 8)

    rows, cols = op.copy_flat_block_table_to("full", table, bases, 32)

    assert (rows, cols) == (1, len(expected_table))
    assert list(table[:cols]) == expected_table
    assert bases[0] == expected_base


def test_execution_plan_forward_is_a_borrowed_view_with_plan_keepalive():
    scheduler = ts.Scheduler(_make_config())
    scheduler.submit_requests([_make_spec("borrowed-forward", num_pages=2)])

    plan = scheduler.next_execution_plan()
    forward_op = plan.forward[0]

    # A copied operation produces a fresh nanobind wrapper on every property
    # read. A borrowed operation reuses the wrapper registered for the C++
    # address inside ExecutionPlan.
    assert plan.forward[0] is forward_op

    del plan
    gc.collect()

    # reference_internal on the element (not just the temporary list) keeps
    # its owning ExecutionPlan alive after callers retain plan.forward[0].
    assert list(forward_op.request_ids) == ["borrowed-forward"]


def test_prefix_disabled_state_only_config_starts_from_cold_root():
    scheduler = ts.Scheduler(_make_state_only_config(disable_prefix_cache=True))
    scheduler.submit_requests([_make_spec("state-only", num_pages=2)])

    op = _find_flat_op(scheduler.next_execution_plan())
    assert op is not None
    assert set(op.flat_block_tables) == {"v4.swa_kv"}
    assert list(op.extend_prefix_lens) == [0]
    assert list(op.flat_block_table_base_offsets["v4.swa_kv"]) == [0]
    assert all(page_id > 0 for page_id in op.flat_block_tables["v4.swa_kv"][0])


def test_prefix_enabled_state_only_config_requires_history_anchor():
    with pytest.raises(
        ValueError,
        match="continuation-state flat groups require at least one history anchor",
    ):
        ts.Scheduler(_make_state_only_config(disable_prefix_cache=False))


def test_decode_slides_swa_window_to_null_hole():
    scheduler = ts.Scheduler(_make_config())
    scheduler.submit_requests([_make_spec("r1", num_pages=2)])

    scheduler.next_execution_plan()  # prefill
    _advance_tokens(scheduler, "r1", [42])

    last_plan = None
    token = 43
    # sliding_window_tokens=4, page_size=2 => window spans 2 pages; ~4 decode
    # steps push total pages past 2, so the oldest page slides out and leaves a
    # null hole in the swa block table.
    for _ in range(4):
        last_plan = scheduler.next_execution_plan()
        assert _find_flat_op(last_plan) is not None
        _advance_tokens(scheduler, "r1", [token])
        token += 1

    op = _find_flat_op(last_plan)
    assert op is not None
    tables = dict(op.flat_block_tables)

    full_row = list(tables["full"][0])
    # page id 0 is the reserved null-block sentinel: >0 means a real page, 0
    # means a hole. The full-history group should never develop a hole.
    assert all(
        page_id > 0 for page_id in full_row
    ), "full row should keep history with no null/padding hole"

    swa_row = list(tables["swa"][0])
    assert (
        0 in swa_row
    ), "swa row should contain a null hole after the sliding window slides"


def test_two_groups_use_disjoint_pages():
    scheduler = ts.Scheduler(_make_config())
    scheduler.submit_requests([_make_spec("r1", num_pages=2)])

    plan = scheduler.next_execution_plan()
    op = _find_flat_op(plan)
    assert op is not None
    tables = dict(op.flat_block_tables)

    full_pages = {page_id for page_id in tables["full"][0] if page_id > 0}
    swa_pages = {page_id for page_id in tables["swa"][0] if page_id > 0}
    assert full_pages, "full row should hold at least one real page"
    assert swa_pages, "swa row should hold at least one real page"
    assert full_pages.isdisjoint(
        swa_pages
    ), "groups share a BlockPool but must not double-allocate physical pages"

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

import pytest

ts = pytest.importorskip("tokenspeed_scheduler")


def _make_config() -> ts.SchedulerConfig:
    cfg = ts.SchedulerConfig()
    cfg.page_size = 2
    cfg.num_device_pages = 32
    cfg.num_host_pages = 32
    cfg.max_scheduled_tokens = 64
    cfg.max_batch_size = 8
    cfg.enable_l3_storage = False
    cfg.disable_l2_cache = True
    cfg.disable_prefix_cache = True

    full = ts.PagedCacheGroupConfig(
        group_id="full",
        rows_per_page=cfg.page_size,
        entry_stride_tokens=1,
        total_pages=cfg.num_device_pages,
        retention=ts.PagedCacheRetention.FullHistory,
        family=ts.PagedCacheGroupFamily.History,
    )
    swa = ts.PagedCacheGroupConfig(
        group_id="swa",
        rows_per_page=cfg.page_size,
        entry_stride_tokens=1,
        total_pages=cfg.num_device_pages,
        retention=ts.PagedCacheRetention.SlidingWindow,
        sliding_window_tokens=4,
        family=ts.PagedCacheGroupFamily.State,
    )
    cfg.paged_cache_groups = [full, swa]
    return cfg


def _make_spec(
    request_id: str, num_pages: int, page_size: int = 2, start: int = 1
) -> ts.RequestSpec:
    # page_size must stay in sync with cfg.page_size: the token count below
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
    assert all(page_id > 0 for page_id in full_row), (
        "full row should keep history with no null/padding hole"
    )

    swa_row = list(tables["swa"][0])
    assert 0 in swa_row, (
        "swa row should contain a null hole after the sliding window slides"
    )


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
    assert full_pages.isdisjoint(swa_pages), (
        "groups share a BlockPool but must not double-allocate physical pages"
    )

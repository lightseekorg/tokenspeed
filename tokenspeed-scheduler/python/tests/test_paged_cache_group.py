from types import SimpleNamespace

import pytest
from tokenspeed_scheduler import (
    PagedCacheGroupAllocator,
    PagedCacheGroupConfig,
    PagedCacheGroupFamily,
    PagedCacheGroupTable,
    PagedCachePrefixRole,
    PagedCacheRetention,
    PagedCacheTableLayout,
)

from tokenspeed.runtime.configs.deepseek_v4_cache_spec import build_v4_cache_specs
from tokenspeed.runtime.engine.scheduler_utils import pool_to_paged_cache_groups


def _full_history_config(rows_per_page=64, entry_stride_tokens=4, total_pages=10):
    return PagedCacheGroupConfig(
        group_id="full",
        rows_per_page=rows_per_page,
        entry_stride_tokens=entry_stride_tokens,
        total_pages=total_pages,
        retention=PagedCacheRetention.FullHistory,
    )


def _sliding_config(rows_per_page=2, entry_stride_tokens=1, total_pages=8, window=4):
    return PagedCacheGroupConfig(
        group_id="swa",
        rows_per_page=rows_per_page,
        entry_stride_tokens=entry_stride_tokens,
        total_pages=total_pages,
        retention=PagedCacheRetention.SlidingWindow,
        sliding_window_tokens=window,
    )


_V4_SPECS = tuple(
    build_v4_cache_specs(
        SimpleNamespace(sliding_window=128),
        layer_ratio=(1, 4, 128),
    )
)


def _v4_native_groups() -> dict[str, PagedCacheGroupConfig]:
    pool = SimpleNamespace(
        paged_cache_group_specs=_V4_SPECS,
        paged_cache_group_page_counts={spec.group_id: 32 for spec in _V4_SPECS},
    )
    return {group.group_id: group for group in pool_to_paged_cache_groups(pool)}


@pytest.mark.parametrize("spec", _V4_SPECS, ids=lambda spec: spec.group_id)
def test_v4_production_spec_round_trips_native_geometry(spec) -> None:
    group = _v4_native_groups()[spec.group_id]

    group.validate_flat_block_geometry()
    assert (group.rows_per_page, group.entry_stride_tokens, group.block_size) == (
        spec.rows_per_page,
        spec.entry_stride_tokens,
        spec.block_size,
    )
    assert (group.total_pages, group.pool_id, group.owner_mask) == (
        32,
        spec.pool_id,
        spec.owner_mask,
    )
    assert (
        group.family
        == {
            "history": PagedCacheGroupFamily.History,
            "state": PagedCacheGroupFamily.State,
        }[spec.family]
    )
    assert (
        group.prefix_role
        == {
            "history_anchor": PagedCachePrefixRole.HistoryAnchor,
            "continuation_state": PagedCachePrefixRole.ContinuationState,
        }[spec.prefix_role]
    )
    assert (
        group.table_layout
        == {
            "absolute": PagedCacheTableLayout.Absolute,
            "bounded_window": PagedCacheTableLayout.BoundedWindow,
        }[spec.table_layout]
    )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("block_size", 8, "block_size must equal"),
        ("sliding_window_tokens", 126, "window must be page-aligned"),
        ("owner_mask", 0, "requires an owner mask"),
    ],
)
def test_v4_flat_group_rejects_invalid_geometry(
    field: str, value: int, error: str
) -> None:
    config = _v4_native_groups()["v4.c4a.compressor_state"]
    setattr(config, field, value)

    with pytest.raises(ValueError, match=error):
        config.validate_flat_block_geometry()


def test_full_history_release_skipped_is_noop():
    alloc = PagedCacheGroupAllocator(_full_history_config(total_pages=8))
    table = PagedCacheGroupTable(alloc)
    table.acquire(1024)
    before = list(table.page_ids())
    active_before = table.active_pages_count()

    assert table.release_skipped(1_000_000) == []
    assert list(table.page_ids()) == before
    assert table.active_pages_count() == active_before


def test_sliding_release_skipped_releases_below_threshold():
    alloc = PagedCacheGroupAllocator(_sliding_config())
    table = PagedCacheGroupTable(alloc)
    table.acquire(8)

    released = table.release_skipped(4)

    assert len(released) == 2
    # Compact: only live entries remain; column c == absolute page base + c.
    page_ids = list(table.page_ids())
    assert len(page_ids) == 2
    assert all(pid >= 0 for pid in page_ids)
    assert table.base_logical_page() == 2
    assert table.size() == 2
    assert table.active_pages_count() == 2
    assert table.release_skipped(4) == []
    assert table.base_logical_page() == 2


def test_acquire_throws_on_exhaustion_without_leaking_cursor():
    alloc = PagedCacheGroupAllocator(_sliding_config(total_pages=2))
    table = PagedCacheGroupTable(alloc)
    failed_before = alloc.failed_alloc_count()

    with pytest.raises(RuntimeError, match="failed to allocate"):
        table.acquire(4)

    assert alloc.failed_alloc_count() == failed_before + 1
    assert table.size() == 0
    assert table.raw_token_cursor() == 0


def test_stride_group_allocates_partial_entry_and_boundary_page():
    alloc = PagedCacheGroupAllocator(
        _full_history_config(rows_per_page=64, entry_stride_tokens=4, total_pages=4)
    )
    table = PagedCacheGroupTable(alloc)

    table.acquire(1)
    assert table.size() == 1
    assert table.raw_token_cursor() == 1

    table.acquire(256)
    assert table.size() == 1

    table.acquire(257)
    assert table.size() == 2


def test_chunked_prefill_release_lower_bound_uses_first_pos_not_target():
    alloc = PagedCacheGroupAllocator(_sliding_config(total_pages=16, window=4))
    table = PagedCacheGroupTable(alloc)
    table.acquire(10)

    chunk_start = 10
    chunk_end = 14
    new_lower = max(0, chunk_start - 4 + 1)
    old_lower = max(0, chunk_end - 4)

    released = table.release_skipped(new_lower)
    page_ids = list(table.page_ids())
    assert len(released) == 3
    # Compact view: 2 live entries (absolute pages 3 and 4), base = 3.
    assert table.base_logical_page() == 3
    assert len(page_ids) == 2
    assert page_ids[0] >= 0, "page covering position 7 must be retained"
    assert page_ids[1] >= 0, "page covering position 8-9 must be retained"

    buggy_alloc = PagedCacheGroupAllocator(
        PagedCacheGroupConfig(
            group_id="swa_buggy",
            rows_per_page=2,
            entry_stride_tokens=1,
            total_pages=16,
            retention=PagedCacheRetention.SlidingWindow,
            sliding_window_tokens=4,
        )
    )
    buggy_table = PagedCacheGroupTable(buggy_alloc)
    buggy_table.acquire(10)
    buggy_released = buggy_table.release_skipped(old_lower)
    # old_lower drops every page including the one chunk2 needs; compact view
    # leaves zero live entries with base advanced to 5.
    assert len(buggy_released) == 5
    assert list(buggy_table.page_ids()) == []
    assert buggy_table.base_logical_page() == 5

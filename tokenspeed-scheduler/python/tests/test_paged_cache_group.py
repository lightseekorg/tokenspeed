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


def _v4_compressor_state_config() -> PagedCacheGroupConfig:
    return PagedCacheGroupConfig(
        group_id="v4.c4a.compressor_state",
        rows_per_page=4,
        entry_stride_tokens=1,
        total_pages=32,
        retention=PagedCacheRetention.SlidingWindow,
        sliding_window_tokens=8,
        family=PagedCacheGroupFamily.State,
        block_size=4,
        pool_id="v4.c4.state",
        prefix_role=PagedCachePrefixRole.ContinuationState,
        table_layout=PagedCacheTableLayout.BoundedWindow,
        owner_mask=1,
    )


def test_v4_flat_group_accepts_native_geometry() -> None:
    config = _v4_compressor_state_config()

    config.validate_flat_block_geometry()
    assert (
        config.rows_per_page,
        config.entry_stride_tokens,
        config.block_size,
        config.pool_id,
        config.family,
        config.prefix_role,
        config.table_layout,
        config.owner_mask,
    ) == (
        4,
        1,
        4,
        "v4.c4.state",
        PagedCacheGroupFamily.State,
        PagedCachePrefixRole.ContinuationState,
        PagedCacheTableLayout.BoundedWindow,
        1,
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
    config = _v4_compressor_state_config()
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

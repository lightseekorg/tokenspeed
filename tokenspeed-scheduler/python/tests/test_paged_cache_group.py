import pytest
from tokenspeed_scheduler import (
    PagedCacheGroupAllocator,
    PagedCacheGroupConfig,
    PagedCacheGroupTable,
    PagedCacheRetention,
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
    page_ids = list(table.page_ids())
    assert page_ids[:2] == [-1, -1]
    assert page_ids[2] >= 0
    assert page_ids[3] >= 0
    assert table.release_skipped(4) == []


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
    assert page_ids[3] >= 0
    assert page_ids[4] >= 0

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
    assert len(buggy_released) == 5
    assert list(buggy_table.page_ids())[3] == -1

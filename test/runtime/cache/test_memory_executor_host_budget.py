import pytest

from tokenspeed.runtime.cache.executor.memory_executor import (
    _aligned_token_count,
    _auto_capped_host_size_tokens,
)


def test_host_budget_no_cap_when_request_fits():
    assert (
        _auto_capped_host_size_tokens(
            requested_tokens=1024,
            page_size=64,
            size_per_token=12,
            available_host_memory_bytes=1_000_000,
            host_parallel_count=4,
        )
        == 0
    )


def test_host_budget_cap_accounts_for_draft_pool_and_page_alignment():
    host_size_tokens = _auto_capped_host_size_tokens(
        requested_tokens=1024,
        page_size=64,
        size_per_token=12,
        available_host_memory_bytes=24_576,
        host_parallel_count=4,
    )

    assert _aligned_token_count(host_size_tokens, 64) == 512


def test_host_budget_raises_when_cgroup_budget_is_too_small():
    with pytest.raises(ValueError, match="Not enough host memory"):
        _auto_capped_host_size_tokens(
            requested_tokens=1024,
            page_size=64,
            size_per_token=12,
            available_host_memory_bytes=100,
            host_parallel_count=1,
        )

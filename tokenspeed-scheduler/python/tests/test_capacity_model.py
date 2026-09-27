"""Binding tests for the config-only capacity model.

The model is the one place the per-request cache working set is defined;
the Python recipes size a pool from it and the Scheduler bounds requests
against that pool with it. These tests keep the marshalling honest and
check the property that ties the two callers together: a pool sized from
``concurrent_group_pages`` admits every request the sized Scheduler's
``max_single_request_tokens`` accepts.
"""

from __future__ import annotations

import pytest

ts = pytest.importorskip("tokenspeed_scheduler")


def _sizing_config(
    *,
    role,
    prefix_granularity: int,
    max_scheduled_tokens: int,
    max_batch_size: int,
    decode_input_tokens: int,
    overlap_schedule_depth: int,
    groups: list,
) -> "ts.SchedulerConfig":
    """A config as the recipes hand it to the model: no page counts yet."""
    cfg = ts.SchedulerConfig()
    cfg.role = role
    cfg.prefix_granularity = prefix_granularity
    cfg.max_scheduled_tokens = max_scheduled_tokens
    cfg.max_batch_size = max_batch_size
    cfg.decode_input_tokens = decode_input_tokens
    cfg.overlap_schedule_depth = overlap_schedule_depth
    cfg.disable_l2_cache = True
    cfg.cache_groups = groups
    return cfg


def _full(group_id: str, block_granularity: int, packing: int) -> "ts.CacheGroupConfig":
    return ts.CacheGroupConfig(
        group_id=group_id,
        block_granularity=block_granularity,
        total_pages=0,
        retention=ts.CacheRetention.FullHistory,
        family=ts.CacheGroupFamily.History,
        cache_blocks_per_lcm_block=packing,
    )


def _sliding(
    group_id: str, block_granularity: int, window: int, packing: int
) -> "ts.CacheGroupConfig":
    return ts.CacheGroupConfig(
        group_id=group_id,
        block_granularity=block_granularity,
        total_pages=0,
        retention=ts.CacheRetention.SlidingWindow,
        sliding_window_tokens=window,
        family=ts.CacheGroupFamily.History,
        cache_blocks_per_lcm_block=packing,
    )


def _state(
    group_id: str, block_granularity: int, packing: int
) -> "ts.CacheGroupConfig":
    return ts.CacheGroupConfig(
        group_id=group_id,
        block_granularity=block_granularity,
        total_pages=0,
        retention=ts.CacheRetention.FullHistory,
        family=ts.CacheGroupFamily.State,
        cache_blocks_per_lcm_block=packing,
    )


def _sized(cfg: "ts.SchedulerConfig", usable_lcm_blocks: int) -> "ts.SchedulerConfig":
    """The same config once the arena exists: every group addresses every parent."""
    cfg.num_device_pages = usable_lcm_blocks + 1
    for group in cfg.cache_groups:
        group.total_pages = 1 + usable_lcm_blocks * group.cache_blocks_per_lcm_block
    return cfg


def test_reports_group_pages_in_config_order() -> None:
    cfg = _sizing_config(
        role=ts.SchedulerConfig.Role.Fused,
        prefix_granularity=64,
        max_scheduled_tokens=8192,
        max_batch_size=16,
        decode_input_tokens=4,
        overlap_schedule_depth=1,
        groups=[
            _full("full", 64, 12),
            _state("state", 64, 1),
            _sliding("swa", 64, 128, 3),
        ],
    )
    model = ts.CapacityModel(cfg)
    assert model.num_groups == 3
    pages = model.concurrent_group_pages(max_total_tokens=65536, max_context_len=4096)
    assert pages == [1024 + 16 * 2, 16 * 4, 16 * 4 + 2 + 128]
    assert model.single_request_group_pages(token_limit=4096)[1] == 4
    # ceil(1056 / 12) + ceil(64 / 1) + ceil(194 / 3)
    assert model.lcm_blocks_needed_for(pages) == 88 + 64 + 65
    # A model outlives the config it was built from.
    del cfg
    assert model.max_single_request_tokens(usable_lcm_blocks=100) > 0


def test_rejects_unsized_pool_only_in_the_scheduler() -> None:
    cfg = _sizing_config(
        role=ts.SchedulerConfig.Role.Fused,
        prefix_granularity=4,
        max_scheduled_tokens=8,
        max_batch_size=1,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
        groups=[_full("full", 4, 1)],
    )
    ts.CapacityModel(cfg)
    with pytest.raises(ValueError, match="null page and usable capacity"):
        ts.Scheduler(cfg)
    cfg.overlap_schedule_depth = 2
    with pytest.raises(ValueError, match="overlap_schedule_depth must be 0 or 1"):
        ts.CapacityModel(cfg)


@pytest.mark.parametrize("decode_input_tokens", [1, 3])
@pytest.mark.parametrize("overlap_schedule_depth", [0, 1])
@pytest.mark.parametrize("max_scheduled_tokens", [4, 9])
@pytest.mark.parametrize(
    "groups",
    [
        [_full("full", 4, 1)],
        [_full("full", 4, 2), _state("state", 2, 1)],
        [_full("full", 4, 3), _sliding("swa", 4, 5, 1), _sliding("tail", 2, 9, 2)],
    ],
    ids=["full", "full+state", "full+two-windows"],
)
def test_pool_sized_by_the_model_admits_what_the_scheduler_bounds(
    decode_input_tokens: int,
    overlap_schedule_depth: int,
    max_scheduled_tokens: int,
    groups: list,
) -> None:
    """Size a pool for one request of L tokens, then ask the sized Scheduler."""
    for token_limit in range(decode_input_tokens + 1, 40):
        cfg = _sizing_config(
            role=ts.SchedulerConfig.Role.Fused,
            prefix_granularity=4,
            max_scheduled_tokens=max_scheduled_tokens,
            max_batch_size=1,
            decode_input_tokens=decode_input_tokens,
            overlap_schedule_depth=overlap_schedule_depth,
            groups=groups,
        )
        model = ts.CapacityModel(cfg)
        pages = model.concurrent_group_pages(
            max_total_tokens=token_limit, max_context_len=token_limit
        )
        usable_lcm_blocks = model.lcm_blocks_needed_for(pages)
        scheduler = ts.Scheduler(_sized(cfg, usable_lcm_blocks))
        assert scheduler.max_single_request_tokens() >= token_limit, (
            token_limit,
            pages,
            usable_lcm_blocks,
        )

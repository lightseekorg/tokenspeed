from types import SimpleNamespace

import pytest

from tokenspeed.runtime.engine.scheduler_utils import scheduler_cache_geometry_from_pool


def _pool(*, num_lcm_blocks: int, prefix_granularity: int, token_capacity: int):
    """A pool whose arena published the contract, the only geometry source."""
    return SimpleNamespace(
        arena=SimpleNamespace(
            runtime_contract=SimpleNamespace(
                num_lcm_blocks=num_lcm_blocks,
                prefix_granularity=prefix_granularity,
                token_capacity=token_capacity,
            )
        )
    )


def test_lcm_scheduler_geometry_counts_parents_not_child_pages():
    # The scheduler pages are LCM parents; a group's child pages are its own
    # per-group count, not this number.
    geometry = scheduler_cache_geometry_from_pool(
        _pool(num_lcm_blocks=37, prefix_granularity=128, token_capacity=37 * 8 * 128)
    )

    assert geometry.prefix_granularity == 128
    # Parent 0 is the reserved null LCM block.
    assert geometry.num_device_pages == 38
    assert geometry.num_usable_pages == 37
    assert geometry.token_capacity == 37 * 8 * 128


def test_lcm_scheduler_geometry_uses_contract_token_capacity():
    # Capacity comes from the contract, never re-derived from parents times a
    # packing guess.
    geometry = scheduler_cache_geometry_from_pool(
        _pool(num_lcm_blocks=37, prefix_granularity=128, token_capacity=10_000)
    )

    assert geometry.num_usable_pages == 37
    assert geometry.token_capacity == 10_000


@pytest.mark.parametrize("grain", [None, 4096])
def test_state_checkpoint_granularity_survives_the_scheduler_bridge(grain):
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.scheduler_bridge import (
        cache_group_config,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import CacheGroupSpec

    spec = CacheGroupSpec(
        group_id="state",
        attn_kind="state",
        replayable=False,
        checkpoint_granularity=grain,
    )
    config = cache_group_config(spec, total_pages=257, cache_blocks_per_lcm_block=1)
    config.validate()
    assert config.checkpoint_granularity == (grain or 1)
    assert config.block_granularity == 0


@pytest.mark.parametrize("grain", [0, -1, True, 1.5])
def test_invalid_state_checkpoint_grain_is_rejected(grain):
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import CacheGroupSpec

    with pytest.raises(ValueError, match="checkpoint_granularity"):
        CacheGroupSpec(
            group_id="state",
            attn_kind="state",
            replayable=False,
            checkpoint_granularity=grain,
        )


def test_history_groups_do_not_accept_a_state_checkpoint_grain():
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import CacheGroupSpec

    with pytest.raises(ValueError, match="checkpoint_granularity"):
        CacheGroupSpec(
            group_id="kv",
            attn_kind="full_history",
            replayable=False,
            rows_per_page=128,
            entry_stride_tokens=1,
            checkpoint_granularity=4096,
        )

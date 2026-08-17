from types import SimpleNamespace

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

from types import SimpleNamespace

from tokenspeed.runtime.engine.scheduler_utils import scheduler_cache_geometry_from_pool


def test_lcm_scheduler_geometry_counts_parents_not_child_pages():
    pool = SimpleNamespace(runtime_contract=None, num_lcm_blocks=37)

    geometry = scheduler_cache_geometry_from_pool(
        pool,
        fallback_token_capacity=37 * 8 * 128,
        fallback_prefix_granularity=128,
    )

    assert geometry.prefix_granularity == 128
    assert geometry.num_device_pages == 38
    assert geometry.num_usable_pages == 37
    assert geometry.token_capacity == 37 * 8 * 128


def test_lcm_scheduler_geometry_uses_contract_token_capacity():
    contract = SimpleNamespace(
        num_lcm_blocks=37,
        prefix_granularity=128,
        token_capacity=10_000,
    )
    pool = SimpleNamespace(runtime_contract=contract, num_lcm_blocks=37)

    geometry = scheduler_cache_geometry_from_pool(
        pool,
        fallback_token_capacity=37 * 12 * 128,
        fallback_prefix_granularity=128,
    )

    assert geometry.num_usable_pages == 37
    assert geometry.token_capacity == 10_000


def test_ordinary_scheduler_geometry_adds_the_null_page():
    geometry = scheduler_cache_geometry_from_pool(
        SimpleNamespace(runtime_contract=None),
        fallback_token_capacity=4 * 64,
        fallback_prefix_granularity=64,
    )

    assert geometry.num_device_pages == 5
    assert geometry.num_usable_pages == 4
    assert geometry.token_capacity == 4 * 64

from types import SimpleNamespace

from tokenspeed.runtime.engine.scheduler_utils import scheduler_cache_geometry_from_pool


def test_lcm_scheduler_geometry_counts_parents_not_child_pages():
    pool = SimpleNamespace(runtime_contract=None, num_lcm_blocks=37)

    geometry = scheduler_cache_geometry_from_pool(
        pool,
        fallback_token_capacity=37 * 8 * 128,
        fallback_page_size=128,
    )

    assert geometry.page_size == 128
    assert geometry.num_device_pages == 38
    assert geometry.num_usable_pages == 37
    assert geometry.token_capacity == 37 * 8 * 128

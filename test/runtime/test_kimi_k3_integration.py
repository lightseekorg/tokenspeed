from __future__ import annotations

from test.runtime.conftest import (
    KIMI_GROUP_IDS,
    cache_metadata_for,
    make_kimi_pool,
)

import numpy as np
import pytest

from tokenspeed.runtime.engine.scheduler_utils import (
    pool_to_cache_groups,
    scheduler_cache_geometry_from_pool,
)


def test_scheduler_uses_lcm_parents_and_per_group_child_counts() -> None:
    pool = make_kimi_pool("cpu", usable_pages=2)

    geometry = scheduler_cache_geometry_from_pool(pool)
    assert geometry.num_usable_pages == 2
    assert geometry.num_device_pages == 3
    assert geometry.prefix_granularity == 128
    groups = pool_to_cache_groups(pool)
    assert {group.group_id for group in groups} == set(KIMI_GROUP_IDS)
    assert {group.group_id: group.total_pages for group in groups} == {
        "full_attention": 25,
        "linear_attention_0": 3,
        "linear_attention_1": 3,
        "linear_attention_2": 3,
    }


def test_metadata_validates_each_groups_own_page_range() -> None:
    pool = make_kimi_pool("cpu", usable_pages=2)
    valid = {
        "full_attention": np.array([[24]], dtype=np.int32),
        "linear_attention_0": np.array([[2]], dtype=np.int32),
    }
    metadata, forward_op = cache_metadata_for(pool.arena.runtime_contract, valid, "cpu")
    tables = metadata.tables(active_forward_op=forward_op)
    assert tables["full_attention"].item() == 24
    assert tables["linear_attention_0"].item() == 2

    with pytest.raises(ValueError, match="outside -1..2"):
        cache_metadata_for(
            pool.arena.runtime_contract,
            {"linear_attention_0": np.array([[3]], dtype=np.int32)},
            "cpu",
        )

from __future__ import annotations

from test.runtime.conftest import (
    KIMI_GROUP_IDS,
    flat_metadata_for,
    make_kimi_pool,
)
from types import SimpleNamespace

import numpy as np
import pytest

import tokenspeed.runtime.configs.paged_cache_spec as paged_cache_spec
from tokenspeed.runtime.configs.paged_cache_spec import (
    preflight_kimi_k3_flat_consumers,
)
from tokenspeed.runtime.engine.scheduler_utils import (
    pool_to_paged_cache_groups,
    scheduler_cache_geometry_from_pool,
)

_K3_ARCHITECTURE = "KimiK3ForConditionalGeneration"


def test_preflight_admits_flat_kimi_and_rejects_radix(monkeypatch) -> None:
    model = SimpleNamespace(hf_config=SimpleNamespace(architectures=[_K3_ARCHITECTURE]))
    monkeypatch.setattr(paged_cache_spec, "scheduler_ext_flat_kvcache", lambda: True)
    preflight_kimi_k3_flat_consumers(model)

    monkeypatch.setattr(paged_cache_spec, "scheduler_ext_flat_kvcache", lambda: False)
    with pytest.raises(RuntimeError, match="FlatKV-only"):
        preflight_kimi_k3_flat_consumers(model)


def test_scheduler_uses_lcm_parents_and_per_group_child_counts() -> None:
    pool = make_kimi_pool("cpu", usable_pages=2)

    geometry = scheduler_cache_geometry_from_pool(
        pool,
        fallback_token_capacity=pool.size,
        fallback_page_size=pool.page_size,
    )
    assert geometry.num_usable_pages == 2
    assert geometry.num_device_pages == 3
    assert geometry.page_size == 128
    groups = pool_to_paged_cache_groups(pool)
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
    metadata, forward_op = flat_metadata_for(pool.runtime_contract, valid, "cpu")
    assert metadata.max_page_ids == {
        "full_attention": 24,
        "linear_attention_0": 2,
        "linear_attention_1": 2,
        "linear_attention_2": 2,
    }
    assert (
        metadata.require_full_attention_table(active_forward_op=forward_op).item() == 24
    )

    with pytest.raises(ValueError, match="outside -1..2"):
        flat_metadata_for(
            pool.runtime_contract,
            {"linear_attention_0": np.array([[3]], dtype=np.int32)},
            "cpu",
        )

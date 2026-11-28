from __future__ import annotations

from test.runtime.conftest import (
    KIMI_GROUP_IDS,
    cache_metadata_for,
    make_kimi_pool,
    whole_row_spans,
    whole_row_windows,
)

import numpy as np
import pytest

from tokenspeed.runtime.engine.scheduler_utils import (
    pool_to_cache_groups,
    scheduler_cache_geometry_from_pool,
)
from tokenspeed.runtime.layers.attention.backends.cache_metadata import (
    CacheBatchMetadata,
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


class _ForwardTables:
    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self.arrays = arrays

    def block_table_windows(self) -> dict[str, tuple[np.ndarray, int]]:
        return whole_row_windows(self.arrays)

    def block_table_spans(self) -> dict[str, np.ndarray]:
        return whole_row_spans(self.arrays)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_metadata_scans_kv_only_when_asked_but_always_bounds_state_groups(
    device: str,
) -> None:
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("cuda device unavailable")

    contract = make_kimi_pool("cpu", usable_pages=2).arena.runtime_contract
    # KV groups (full_attention, 0..24) stay opt-in under validate_page_ids.
    kv_oob = {group_id: np.array([[1]], dtype=np.int32) for group_id in KIMI_GROUP_IDS}
    kv_oob["full_attention"] = np.array([[25]], dtype=np.int32)
    kv_oob_op = _ForwardTables(kv_oob)
    kw = dict(device=device, contract=contract, num_requests=1, host_tables=False)

    if device == "cuda":
        # Warm up packing kernel before enabling strict sync checks.
        CacheBatchMetadata.from_forward_op(kv_oob_op, validate_page_ids=False, **kw)

    previous = torch.cuda.get_sync_debug_mode() if device == "cuda" else None
    try:
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode("error")
        metadata = CacheBatchMetadata.from_forward_op(
            kv_oob_op, validate_page_ids=False, **kw
        )
        # State groups (linear_attention_0/1/2, 0..2) are always bounded on CPU
        # with zero GPU sync even when validate_page_ids=False.
        for state_group in (
            "linear_attention_0",
            "linear_attention_1",
            "linear_attention_2",
        ):
            for bad_id in (-2, 3):
                state_oob = {
                    group_id: np.array([[1]], dtype=np.int32)
                    for group_id in KIMI_GROUP_IDS
                }
                state_oob[state_group] = np.array([[bad_id]], dtype=np.int32)
                with pytest.raises(ValueError, match="outside -1..2"):
                    CacheBatchMetadata.from_forward_op(
                        _ForwardTables(state_oob), validate_page_ids=False, **kw
                    )
    finally:
        if device == "cuda":
            torch.cuda.set_sync_debug_mode(previous)

    assert metadata.tables(active_forward_op=kv_oob_op)["full_attention"].item() == 25

    with pytest.raises(ValueError, match="outside -1..24"):
        CacheBatchMetadata.from_forward_op(kv_oob_op, validate_page_ids=True, **kw)

    # The structural checks run either way.
    with pytest.raises(ValueError, match="rows"):
        CacheBatchMetadata.from_forward_op(
            kv_oob_op,
            device=device,
            contract=contract,
            num_requests=2,
            validate_page_ids=False,
            host_tables=False,
        )

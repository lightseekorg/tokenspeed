from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.configs.paged_cache_spec import PagedCacheGroupSpec
from tokenspeed.runtime.layers.attention.registry import (
    _validate_shared_cache_geometry,
)


def _pool(*, rows_per_page: int = 128):
    group_id = "full_attention"
    group = SimpleNamespace(
        group_id=group_id,
        cache_blocks_per_lcm_block=1,
        page_count=3,
    )
    plan = SimpleNamespace(
        logical_block_tokens=128,
        num_lcm_blocks=2,
        groups=(group,),
    )
    spec = PagedCacheGroupSpec(
        group_id=group_id,
        retention="full_history",
        rows_per_page=rows_per_page,
        entry_stride_tokens=1,
        sliding_window_tokens=None,
        family="history",
        cache_blocks_per_lcm_block=1,
    )
    return SimpleNamespace(
        runtime_contract=object(),
        plan=plan,
        paged_cache_group_specs=(spec,),
        buffer=torch.zeros(1),
    )


def test_shared_geometry_uses_current_cache_spec_fields() -> None:
    """The post-#930 spec has no legacy block_size attribute."""
    _validate_shared_cache_geometry(_pool(), _pool())


def test_shared_geometry_rejects_different_scheduler_semantics() -> None:
    with pytest.raises(RuntimeError, match="scheduler semantics"):
        _validate_shared_cache_geometry(_pool(), _pool(rows_per_page=64))

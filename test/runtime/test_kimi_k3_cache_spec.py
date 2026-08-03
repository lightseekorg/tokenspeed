from __future__ import annotations

from test.runtime.conftest import TP8_PAGE_SET_BYTES

import torch

from tokenspeed.runtime.configs.kimi_k3_cache_spec import (
    kimi_k3_lcm_blocks_needed,
    kimi_k3_token_capacity_for_lcm_pool,
    plan_kimi_k3_lcm_cache,
)
from tokenspeed.runtime.configs.kimi_k3_config import KimiLinearConfig


def test_lcm_reference_geometry_is_exact() -> None:
    plan = plan_kimi_k3_lcm_cache(
        KimiLinearConfig(),
        flat_kvcache_enabled=True,
        tp_size=8,
        mla_cache_dtype=torch.float8_e4m3fn,
        mla_quant_method=None,
        num_lcm_blocks=7,
    )

    assert plan.logical_block_tokens == 128
    assert plan.lcm_block_bytes == TP8_PAGE_SET_BYTES
    assert len(plan.planes) == 24
    assert {
        group.group_id: group.cache_blocks_per_lcm_block for group in plan.groups
    } == {
        "full_attention": 12,
        "linear_attention_0": 1,
        "linear_attention_1": 1,
        "linear_attention_2": 1,
    }
    fields_by_group = {
        group_id: [field for field in plan.fields if field.group_id == group_id]
        for group_id in (
            "full_attention",
            "linear_attention_0",
            "linear_attention_1",
            "linear_attention_2",
        )
    }
    assert len(fields_by_group["full_attention"]) == 24
    assert all(
        len(fields_by_group[group_id]) == 46
        for group_id in fields_by_group
        if group_id != "full_attention"
    )
    for group_id in (
        "linear_attention_0",
        "linear_attention_1",
        "linear_attention_2",
    ):
        assert {field.plane_id for field in fields_by_group[group_id]} == {
            f"slot.{slot}" for slot in range(23)
        }


def test_lcm_geometry_packs_two_kda_pages_at_tp16() -> None:
    """KDA state halves at TP16; two pages pack per MLA-sized plane."""
    plan = plan_kimi_k3_lcm_cache(
        KimiLinearConfig(),
        flat_kvcache_enabled=True,
        tp_size=16,
        mla_cache_dtype=torch.float8_e4m3fn,
        mla_quant_method=None,
        num_lcm_blocks=7,
    )

    assert {
        group.group_id: group.cache_blocks_per_lcm_block for group in plan.groups
    } == {
        "full_attention": 12,
        "linear_attention_0": 2,
        "linear_attention_1": 2,
        "linear_attention_2": 2,
    }
    # MLA planes still dominate, so the LCM block geometry matches TP8.
    assert plan.lcm_block_bytes == TP8_PAGE_SET_BYTES
    assert len(plan.planes) == 24


def test_lcm_parent_demand_uses_per_group_packing() -> None:
    plan = plan_kimi_k3_lcm_cache(
        KimiLinearConfig(),
        flat_kvcache_enabled=True,
        tp_size=8,
        mla_cache_dtype=torch.float8_e4m3fn,
        mla_quant_method=None,
        num_lcm_blocks=300,
    )
    sizing = dict(
        max_scheduled_tokens=8_192,
        max_live_requests=1,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )

    assert kimi_k3_lcm_blocks_needed(plan, token_capacity=131_072, **sizing) == 281
    assert (
        kimi_k3_token_capacity_for_lcm_pool(
            plan,
            num_lcm_blocks=281,
            upper_bound_tokens=131_072,
            **sizing,
        )
        == 131_072
    )
    assert (
        kimi_k3_token_capacity_for_lcm_pool(
            plan,
            num_lcm_blocks=280,
            upper_bound_tokens=131_072,
            **sizing,
        )
        < 131_072
    )

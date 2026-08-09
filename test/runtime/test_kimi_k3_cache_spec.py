from __future__ import annotations

import os
import sys

_TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_TEST_DIR))

from test.runtime.conftest import TP8_PAGE_SET_BYTES

import torch

from tokenspeed.runtime.configs.kimi_k3_config import KimiLinearConfig
from tokenspeed.runtime.layers.attention.kv_cache.recipes.kimi_k3 import (
    kimi_k3_lcm_blocks_needed,
    kimi_k3_token_capacity_for_cache_pool,
    solve_kimi_k3_cache_layout,
)


def _plan(num_lcm_blocks: int, *, tp_size: int = 8):
    layout = solve_kimi_k3_cache_layout(
        KimiLinearConfig(),
        tp_size=tp_size,
        mla_cache_dtype=torch.float8_e4m3fn,
        mla_quant_method=None,
    )
    return layout.with_num_lcm_blocks(num_lcm_blocks)


def test_lcm_reference_geometry_is_exact() -> None:
    plan = _plan(7)

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
    plan = _plan(7, tp_size=16)

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
    plan = _plan(300)
    sizing = dict(
        max_scheduled_tokens=8_192,
        max_live_requests=1,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )

    assert kimi_k3_lcm_blocks_needed(plan, token_capacity=131_072, **sizing) == 284
    assert (
        kimi_k3_token_capacity_for_cache_pool(
            plan,
            num_lcm_blocks=284,
            upper_bound_tokens=131_072,
            **sizing,
        )
        == 131_072
    )
    assert (
        kimi_k3_token_capacity_for_cache_pool(
            plan,
            num_lcm_blocks=283,
            upper_bound_tokens=131_072,
            **sizing,
        )
        < 131_072
    )


def test_k3_merged_solve_with_draft_shares_page_ids():
    """One big model: a draft MLA layer joins the K3 solve as continuation
    layer 93 in the full_attention group — same packing, same page-id
    space, one plan, one arena."""
    import torch

    from tokenspeed.runtime.configs.kimi_k3_config import KimiLinearConfig
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.ordinary import (
        mla_cache_fields,
    )

    draft_fields = mla_cache_fields(
        layer_group_ids=("full_attention",),
        logical_block_tokens=128,
        latent_width=576,
        element_size=torch.bfloat16.itemsize,
    )
    merged = solve_kimi_k3_cache_layout(
        KimiLinearConfig(),
        tp_size=8,
        mla_cache_dtype=torch.float8_e4m3fn,
        mla_quant_method=None,
        draft_fields=draft_fields,
    )
    # 24 target MLA planes + 1 draft continuation plane.
    assert len(merged.plane_bytes) == 25
    assert dict(merged.group_packing)["full_attention"] == 12
    plan = merged.with_num_lcm_blocks(7)
    draft_field = plan.field("layer.93.latent_kv")
    target_field = plan.field("layer.3.latent_kv")
    assert draft_field.group_id == target_field.group_id == "full_attention"
    assert draft_field.element_size == 2
    assert target_field.element_size == 1
    assert draft_field.page_stride_bytes == 2 * target_field.page_stride_bytes
    # One group -> one page-id space: same page_count by identity.
    assert plan.group("full_attention").page_count == 1 + 7 * 12


def test_k3_binding_utilization_baseline_and_draft_widening():
    """Binding-hole metric on real K3 geometry: full bindings use
    the whole parent; state bindings use 88.2%, dropping to ~84.7% when a
    1-layer draft plane widens the parent (naive join)."""
    import torch

    from tokenspeed.runtime.configs.kimi_k3_config import KimiLinearConfig
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.ordinary import (
        mla_cache_fields,
    )

    base = solve_kimi_k3_cache_layout(
        KimiLinearConfig(),
        tp_size=8,
        mla_cache_dtype=torch.float8_e4m3fn,
        mla_quant_method=None,
    ).with_num_lcm_blocks(10)
    report = base.capacity_report()
    assert abs(report["full_attention"]["binding_utilization"] - 1.0) < 1e-3
    for k in range(3):
        assert (
            abs(report[f"linear_attention_{k}"]["binding_utilization"] - 0.882) < 1e-3
        )

    draft_fields = mla_cache_fields(
        layer_group_ids=("full_attention",),
        logical_block_tokens=128,
        latent_width=576,
        element_size=1,
    )
    merged = solve_kimi_k3_cache_layout(
        KimiLinearConfig(),
        tp_size=8,
        mla_cache_dtype=torch.float8_e4m3fn,
        mla_quant_method=None,
        draft_fields=draft_fields,
    ).with_num_lcm_blocks(10)
    widened = merged.capacity_report()
    assert abs(widened["full_attention"]["binding_utilization"] - 1.0) < 1e-3
    assert abs(widened["linear_attention_0"]["binding_utilization"] - 0.847) < 1e-3

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

    assert plan.prefix_granularity == 128
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
    conv = next(
        field for field in plan.fields if field.field_id.endswith(".conv_state")
    )
    assert conv.shape[0] == 3 * 96 * 128 // 8


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
    conv = next(
        field for field in plan.fields if field.field_id.endswith(".conv_state")
    )
    assert conv.shape[0] == 3 * 96 * 128 // 16


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
    """One big model: five BF16 draft MLA layers join the K3 solve as
    continuation layers 93-97 in the full_attention group — same packing/page-id
    space, one plan, one arena."""
    import torch

    from tokenspeed.runtime.configs.kimi_k3_config import KimiLinearConfig
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.ordinary import (
        mla_cache_fields,
    )

    draft_fields = mla_cache_fields(
        layer_group_ids=("full_attention",) * 5,
        prefix_granularity=128,
        latent_width=576,
        element_size=torch.bfloat16.itemsize,
    )
    merged = solve_kimi_k3_cache_layout(
        KimiLinearConfig(),
        tp_size=8,
        mla_cache_dtype=torch.float8_e4m3fn,
        mla_quant_method=None,
        draft_fields=draft_fields,
        draft_layer_count=5,
    )
    # 24 target MLA planes + 5 draft continuation planes.
    assert len(merged.plane_bytes) == 29
    assert dict(merged.group_packing)["full_attention"] == 12
    plan = merged.with_num_lcm_blocks(7)
    target_field = plan.field("layer.3.latent_kv")
    assert target_field.element_size == 1
    target_plane_ids = {f"slot.{slot}" for slot in range(24)}
    for global_layer_id in range(93, 98):
        draft_field = plan.field(f"layer.{global_layer_id}.latent_kv")
        assert draft_field.group_id == target_field.group_id == "full_attention"
        assert draft_field.plane_id == f"slot.{global_layer_id}"
        assert draft_field.plane_id not in target_plane_ids
        assert draft_field.element_size == 2
        assert draft_field.page_stride_bytes == 2 * target_field.page_stride_bytes
    # One group -> one page-id space: same page_count by identity.
    assert plan.group("full_attention").page_count == 1 + 7 * 12
    assert merged.lcm_block_bytes == 30_081_024
    assert plan.arena_bytes == 8 * merged.lcm_block_bytes


def test_k3_binding_utilization_with_real_bf16_draft_geometry():
    """Binding-hole metric on real K3 geometry: full bindings use
    the whole parent; state bindings use 88.2%, dropping to ~62.2% when the
    five BF16 draft planes widen the parent."""
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
        layer_group_ids=("full_attention",) * 5,
        prefix_granularity=128,
        latent_width=576,
        element_size=torch.bfloat16.itemsize,
    )
    merged = solve_kimi_k3_cache_layout(
        KimiLinearConfig(),
        tp_size=8,
        mla_cache_dtype=torch.float8_e4m3fn,
        mla_quant_method=None,
        draft_fields=draft_fields,
        draft_layer_count=5,
    ).with_num_lcm_blocks(10)
    widened = merged.capacity_report()
    assert abs(widened["full_attention"]["binding_utilization"] - 1.0) < 1e-3
    assert abs(widened["linear_attention_0"]["binding_utilization"] - 0.6224) < 1e-3

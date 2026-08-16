from __future__ import annotations

import os
import sys

_TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_TEST_DIR))

from test.runtime.conftest import TP8_PAGE_SET_BYTES, kimi_tp8_layout

import torch


def _plan(num_lcm_blocks: int, *, tp_size: int = 8):
    return kimi_tp8_layout(tp_size=tp_size)[2].bind(num_lcm_blocks)


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
    recipe, _, layout = kimi_tp8_layout(max_bs=1, max_scheduled_tokens=8_192)

    # 131_072 tokens at this concurrency needs exactly 284 parents; the search
    # inverts that demand -- what 284 parents admit needs no more than 284,
    # and one parent fewer admits strictly less.
    assert recipe.parents_needed(layout, 131_072) == 284
    admitted = recipe.token_capacity(layout, 284)
    assert admitted >= 131_072
    assert recipe.parents_needed(layout, admitted) <= 284
    assert recipe.token_capacity(layout, 283) < admitted


def test_k3_merged_solve_with_draft_shares_page_ids():
    """One big model: five BF16 draft MLA layers join the K3 solve as
    continuation layers 93-97 in the full_attention group — same packing/page-id
    space, one plan, one arena."""
    _, _, merged = kimi_tp8_layout(draft_layers=5)
    # 24 target MLA planes + 5 draft continuation planes.
    assert len(merged.plane_bytes) == 29
    assert dict(merged.group_packing)["full_attention"] == 12
    plan = merged.bind(7)
    target_field = plan.field("layer.3.latent_kv")
    assert target_field.element_size == 1
    target_plane_ids = {f"slot.{slot}" for slot in range(24)}
    for draft_index, global_layer_id in enumerate(range(93, 98)):
        draft_field = plan.field(f"layer.{global_layer_id}.latent_kv")
        assert draft_field.group_id == target_field.group_id == "full_attention"
        # Planes number by tenancy, not by layer id: the draft layers are the
        # group's 25th..29th tenants, continuing the target's slot.0..23.
        assert draft_field.plane_id == f"slot.{24 + draft_index}"
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
    base = kimi_tp8_layout()[2].bind(10)
    report = base.capacity_report()
    assert abs(report["full_attention"]["binding_utilization"] - 1.0) < 1e-3
    for k in range(3):
        assert (
            abs(report[f"linear_attention_{k}"]["binding_utilization"] - 0.882) < 1e-3
        )

    merged = kimi_tp8_layout(draft_layers=5)[2]
    merged = merged.bind(10)
    widened = merged.capacity_report()
    assert abs(widened["full_attention"]["binding_utilization"] - 1.0) < 1e-3
    assert abs(widened["linear_attention_0"]["binding_utilization"] - 0.6224) < 1e-3

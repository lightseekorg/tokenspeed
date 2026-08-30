from __future__ import annotations

import os
import sys
from types import SimpleNamespace

_TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_TEST_DIR))

from test.runtime.conftest import TP8_PAGE_SET_BYTES, kimi_tp8_layout

import torch
from tokenspeed_kernel.ops.attention import kda_conv_state_layout

from tokenspeed.runtime.configs.kimi_k3_config import KimiLinearConfig
from tokenspeed.runtime.layers.attention.kv_cache.recipes.transfer import (
    _partition_for_field,
)


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
    expected_shape = (
        (3, 3 * 96 * 128 // 8)
        if kda_conv_state_layout() == "sequence_major"
        else (3 * 96 * 128 // 8, 3)
    )
    assert conv.shape == expected_shape


def test_conv_transfer_partitions_the_channel_axis() -> None:
    plan = _plan(1)
    conv = next(
        field for field in plan.fields if field.field_id.endswith(".conv_state")
    )
    model = SimpleNamespace(
        num_attention_layers=93,
        hf_text_config=KimiLinearConfig(),
    )

    partition = _partition_for_field(
        conv,
        model_config=model,
        draft_model_config=None,
        prefix_granularity=plan.prefix_granularity,
        inkling_layers=frozenset(),
    )

    assert partition is not None
    assert partition.axis == (1 if kda_conv_state_layout() == "sequence_major" else 0)
    assert partition.global_extent == 3 * 96 * 128
    assert partition.global_parts == (96 * 128, 96 * 128, 96 * 128)


def test_lcm_geometry_shrinks_with_the_kda_state_at_tp16() -> None:
    """KDA state halves at TP16; the plane and the parent halve with it."""
    plan = _plan(7, tp_size=16)

    assert {
        group.group_id: group.cache_blocks_per_lcm_block for group in plan.groups
    } == {
        "full_attention": 6,
        "linear_attention_0": 1,
        "linear_attention_1": 1,
        "linear_attention_2": 1,
    }
    # Half the TP8 state needs half the plane width, so the parent halves.
    assert plan.lcm_block_bytes == TP8_PAGE_SET_BYTES // 2
    assert len(plan.planes) == 24
    conv = next(
        field for field in plan.fields if field.field_id.endswith(".conv_state")
    )
    expected_shape = (
        (3, 3 * 96 * 128 // 16)
        if kda_conv_state_layout() == "sequence_major"
        else (3 * 96 * 128 // 16, 3)
    )
    assert conv.shape == expected_shape


def test_attention_dp_layouts_grow_the_mla_packing() -> None:
    """Attention-DP (tp < 8) grows the KDA state; the packing follows it.

    The original constant-12 packing made every tp < 8 boot fail on the
    exact-page-stride check: the state outgrew the plane and the planner
    widened it, breaking the latent kernel's implicit 73,728-byte stride.
    """
    latent_page_bytes = 128 * 576
    for tp_size, mla_packing in ((4, 23), (2, 45), (1, 89)):
        plan = _plan(7, tp_size=tp_size)

        assert {
            group.group_id: group.cache_blocks_per_lcm_block for group in plan.groups
        } == {
            "full_attention": mla_packing,
            "linear_attention_0": 1,
            "linear_attention_1": 1,
            "linear_attention_2": 1,
        }
        assert plan.lcm_block_bytes == 24 * mla_packing * latent_page_bytes
        assert len(plan.planes) == 24


def test_latent_stride_stays_exact_for_every_tp() -> None:
    """The MLA kernel indexes pages by an implicit payload-sized stride."""
    latent_page_bytes = 128 * 576
    for tp_size in (1, 2, 4, 8, 16):
        plan = _plan(7, tp_size=tp_size)
        latent = next(
            field for field in plan.fields if field.field_id.endswith(".latent_kv")
        )
        assert latent.page_stride_bytes == latent_page_bytes, tp_size


def test_bf16_mla_cache_reuses_the_same_packing_rule() -> None:
    """The packing is a ratio of byte counts, so the cache dtype scales it."""
    _, _, layout = kimi_tp8_layout(tp_size=1, kv_cache_dtype=torch.bfloat16)
    plan = layout.bind(7)

    latent_page_bytes = 128 * 576 * 2
    assert {
        group.group_id: group.cache_blocks_per_lcm_block for group in plan.groups
    } == {
        "full_attention": 45,
        "linear_attention_0": 1,
        "linear_attention_1": 1,
        "linear_attention_2": 1,
    }
    latent = next(
        field for field in plan.fields if field.field_id.endswith(".latent_kv")
    )
    assert latent.page_stride_bytes == latent_page_bytes


def test_speculative_verify_workspace_is_reserved_outside_the_arena(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "tokenspeed_kernel.ops.attention.kda_replay_commit_supported",
        lambda dtype, **kwargs: False,
    )
    recipe, _, layout = kimi_tp8_layout(
        draft_layers=5,
        max_bs=4,
        speculative_algorithm="DSPARK",
        speculative_num_draft_tokens=8,
    )

    # Four requests, each with one committed seed row and eight candidate rows,
    # across all 69 target KDA layers.
    expected_workspace_bytes = 2_022_174_720
    assert recipe.workspace_bytes() == expected_workspace_bytes

    setup = recipe.setup()
    assert setup.fixed_workspace_bytes == expected_workspace_bytes
    expected_parents = (
        recipe.cache_budget_bytes - expected_workspace_bytes
    ) // layout.lcm_block_bytes - 1
    assert setup.spec.memory_plan.num_lcm_blocks == expected_parents


def test_replay_verify_workspace_reserves_conv_rows_and_payloads(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "tokenspeed_kernel.ops.attention.kda_replay_commit_supported",
        lambda dtype, **kwargs: True,
    )
    monkeypatch.setattr(
        "tokenspeed_kernel.ops.attention.kda_batched_replay_uses_raw_gate",
        lambda dtype: False,
    )
    recipe, groups, layout = kimi_tp8_layout(
        draft_layers=5,
        max_bs=4,
        speculative_algorithm="DSPARK",
        speculative_num_draft_tokens=8,
    )
    conv_row_bytes = 4 * sum(
        field.payload_bytes
        for spec, fields in groups
        if spec.group_id != "full_attention"
        for field in fields
        if field.field_id.endswith(".conv_state")
    )
    # 69 KDA layers x (4 requests x 8 draft tokens) rows, each one bf16
    # qkv/f_a/beta row (4608 + 128 + 12 channels) plus an fp32 gate row
    # (12 x 128), per rank at TP8.
    expected_payload_bytes = 69 * 4 * 8 * ((4608 + 128 + 12) * 2 + 12 * 128 * 4)
    expected_workspace_bytes = conv_row_bytes + expected_payload_bytes

    setup = recipe.setup()

    assert recipe.workspace_bytes() == expected_workspace_bytes
    assert setup.fixed_workspace_bytes == expected_workspace_bytes
    expected_parents = (
        recipe.cache_budget_bytes - expected_workspace_bytes
    ) // layout.lcm_block_bytes - 1
    assert setup.spec.memory_plan.num_lcm_blocks == expected_parents


def test_non_speculative_kimi_reserves_no_verify_workspace() -> None:
    recipe, _, _ = kimi_tp8_layout(
        draft_layers=5,
        max_bs=4,
        speculative_algorithm=None,
        speculative_num_draft_tokens=8,
    )

    assert recipe.workspace_bytes() == 0


def test_lcm_parent_demand_uses_per_group_packing() -> None:
    recipe, _, layout = kimi_tp8_layout(max_bs=1, max_scheduled_tokens=8_192)

    # Non-overlap sparse state prefill needs one input and one output block per
    # KDA group; the next decode allocates its destination after completion.
    # The search inverts that demand -- what 92 parents admit needs no more,
    # and one parent fewer admits strictly less.
    assert recipe.parents_needed(layout, 131_072) == 92
    admitted = recipe.token_capacity(layout, 92)
    assert admitted >= 131_072
    assert recipe.parents_needed(layout, admitted) <= 92
    assert recipe.token_capacity(layout, 91) < admitted


def test_sparse_state_parent_demand_tracks_decode_and_overlap_width() -> None:
    baseline, _, layout = kimi_tp8_layout(
        max_bs=3,
        max_scheduled_tokens=8_192,
        decode_input_tokens=128,
        overlap_schedule_depth=0,
    )
    overlapped, _, _ = kimi_tp8_layout(
        max_bs=3,
        max_scheduled_tokens=8_192,
        decode_input_tokens=128,
        overlap_schedule_depth=1,
    )

    # KDA state uses the same two rolling pages with or without overlap. The
    # small full-attention protection term still fits in the same packed parent.
    assert (
        overlapped.parents_needed(layout, 131_072)
        - baseline.parents_needed(layout, 131_072)
        == 0
    )


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

from __future__ import annotations

import os
import sys
from test.ci_system.ci_register import register_cuda_ci

_TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_TEST_DIR))

from test.runtime.conftest import TP8_PAGE_SET_BYTES, kimi_tp8_layout

import pytest
import torch

register_cuda_ci(est_time=5, suite="runtime-1gpu")


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
        "full_attention": 16,
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
        len(fields_by_group[group_id]) == 138
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


def test_lcm_geometry_shrinks_with_the_kda_state_at_tp16() -> None:
    """KDA state halves at TP16; the plane and the parent halve with it."""
    plan = _plan(7, tp_size=16)

    assert {
        group.group_id: group.cache_blocks_per_lcm_block for group in plan.groups
    } == {
        "full_attention": 8,
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
    assert conv.shape[0] == 3 * 96 * 128 // 16


def test_attention_dp_layouts_grow_the_mla_packing() -> None:
    """Attention-DP (tp < 8) grows the KDA state; the packing follows it.

    The original constant-12 packing made every tp < 8 boot fail on the
    exact-page-stride check: the state outgrew the plane and the planner
    widened it, breaking the latent kernel's implicit 73,728-byte stride.
    """
    latent_page_bytes = 128 * 576
    for tp_size, mla_packing in ((4, 31), (2, 61), (1, 121)):
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


def test_bf16_base_keeps_fp32_retained_history_at_tp8() -> None:
    _, _, layout = kimi_tp8_layout(mamba_ssm_dtype="bfloat16")
    plan = layout.bind(7)
    assert plan.group("full_attention").cache_blocks_per_lcm_block == 10
    assert plan.lcm_block_bytes == 24 * 10 * 128 * 576
    fields = {field.field_id: field for field in plan.fields}
    assert fields["layer.0.recurrent_state"].dtype == "bfloat16"
    for name in ("replay_correction", "replay_key", "replay_decay"):
        assert fields[f"layer.0.{name}"].dtype == "float32"
    assert fields["layer.0.replay_length"].dtype == "int32"


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
        "full_attention": 61,
        "linear_attention_0": 1,
        "linear_attention_1": 1,
        "linear_attention_2": 1,
    }
    latent = next(
        field for field in plan.fields if field.field_id.endswith(".latent_kv")
    )
    assert latent.page_stride_bytes == latent_page_bytes


@pytest.mark.parametrize(
    "device,backend_name,kvp_size",
    [("cpu", "mla", 1), ("cuda", "tokenspeed_mla", 1), ("cuda", "tokenspeed_mla", 4)],
)
def test_speculative_verify_workspace_is_reserved_outside_the_arena(
    device,
    backend_name,
    kvp_size,
) -> None:
    from dataclasses import replace

    recipe, _, layout = kimi_tp8_layout(
        draft_layers=5,
        max_bs=4,
        speculative_algorithm="DSPARK",
        speculative_num_draft_tokens=8,
    )

    recipe.attn_config = replace(
        recipe.attn_config,
        device=device,
        kvp_size=kvp_size,
        kvp_group=tuple(range(kvp_size)),
        components=(
            replace(recipe.attn_config.components[0], backend_name=backend_name),
            *recipe.attn_config.components[1:],
        ),
    )

    # 69 layers, four requests and eight QKV candidates.
    expected_payload_bytes = 69 * 4 * 8 * 4608 * 2
    descriptor_bytes = 69 * (9 * 8 + 4) + 3 * 4 * 8
    page_and_step_bytes = (2 * 3 + 1) * 4 * 4
    record_metadata_bytes = 4 * 3 * 8 + 69 * 4 * 8 * 2 * 8
    expected_workspace_bytes = (
        expected_payload_bytes
        + descriptor_bytes
        + page_and_step_bytes
        + record_metadata_bytes
    )

    setup = recipe.setup()

    assert recipe.workspace_bytes() == expected_workspace_bytes
    assert setup.fixed_workspace_bytes == expected_workspace_bytes
    expected_parents = (
        recipe.cache_budget_bytes - expected_workspace_bytes
    ) // layout.lcm_block_bytes - 1
    assert setup.spec.memory_plan.num_lcm_blocks == expected_parents


def test_the_amd_kvp_prefill_reserves_its_gather_workspace(monkeypatch) -> None:
    """Kimi-K3.1 over KVP pages on AMD (a prefill over gathered histories,
    ``dsa_kvp_history_prefill``): the recipe plans the DSA history gather
    workspace beside the KDA verify staging, and alone on a prefill-only
    engine; off AMD or in a bitwise envelope the verify staging alone."""
    from dataclasses import replace
    from types import SimpleNamespace

    from tokenspeed.runtime.layers.attention.configs import dsa as dsa_configs
    from tokenspeed.runtime.layers.attention.configs.dsa import (
        dsa_history_gather_workspace_bytes,
    )
    from tokenspeed.runtime.utils.env import global_server_args_dict

    monkeypatch.setitem(global_server_args_dict, "numerics", "auto")
    recipe, _, _ = kimi_tp8_layout(
        draft_layers=5,
        max_bs=4,
        speculative_algorithm="EAGLE3",
        speculative_num_draft_tokens=4,
        dsa_index_k_format="fp8_scaled",
    )
    recipe.attn_config = replace(
        recipe.attn_config,
        device="cuda",
        kvp_size=8,
        kvp_group=tuple(range(8)),
        components=(
            replace(recipe.attn_config.components[0], backend_name="dsa"),
            *recipe.attn_config.components[1:],
        ),
    )

    def on(is_amd: bool) -> None:
        monkeypatch.setattr(
            dsa_configs, "current_platform", lambda: SimpleNamespace(is_amd=is_amd)
        )

    on(False)
    verify = recipe.workspace_bytes()
    assert verify > 0
    on(True)
    history = dsa_history_gather_workspace_bytes(
        recipe.attn_config, max_model_len=recipe.attn_config.context_len
    )
    assert history > 0
    assert recipe.workspace_bytes() == verify + history
    # A bitwise envelope keeps the sharded arm, which gathers no history.
    monkeypatch.setitem(global_server_args_dict, "numerics", "rl-bitwise")
    assert recipe.workspace_bytes() == verify
    monkeypatch.setitem(global_server_args_dict, "numerics", "auto")
    recipe.server_args.disaggregation_mode = "prefill"
    assert recipe.workspace_bytes() == history


def test_non_speculative_kimi_reserves_no_verify_workspace() -> None:
    """A prefill-only engine has no decode records, with or without a drafter."""
    recipe, _, _ = kimi_tp8_layout(
        draft_layers=5,
        max_bs=4,
        speculative_algorithm=None,
        speculative_num_draft_tokens=8,
    )

    recipe.server_args.disaggregation_mode = "prefill"
    assert recipe.workspace_bytes() == 0


def test_lcm_parent_demand_uses_per_group_packing() -> None:
    recipe, _, layout = kimi_tp8_layout(max_bs=1, max_scheduled_tokens=8_192)

    # State groups hold only an input checkpoint and a live page. The full
    # history uses ceil(1026 / 16) = 65 parents; three state groups use two
    # parents each. The inverse capacity search must agree at the boundary.
    assert recipe.parents_needed(layout, 131_072) == 71
    admitted = recipe.token_capacity(layout, 71)
    assert admitted >= 131_072
    assert recipe.parents_needed(layout, admitted) <= 71
    assert recipe.token_capacity(layout, 70) < admitted


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

    # State tables keep two slots regardless of decode width or overlap.
    # Additional full-attention protection still fits in the same packed parent.
    assert overlapped.parents_needed(layout, 131_072) == baseline.parents_needed(
        layout, 131_072
    )


def test_k3_merged_solve_with_draft_shares_page_ids():
    """One big model: five BF16 draft MLA layers join the K3 solve as
    continuation layers 93-97 in the full_attention group — same packing/page-id
    space, one plan, one arena."""
    _, _, merged = kimi_tp8_layout(draft_layers=5)
    # 24 target MLA planes + 5 draft continuation planes.
    assert len(merged.plane_bytes) == 29
    assert dict(merged.group_packing)["full_attention"] == 16
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
    assert plan.group("full_attention").page_count == 1 + 7 * 16
    assert merged.lcm_block_bytes == 40_108_032
    assert plan.arena_bytes == 8 * merged.lcm_block_bytes


def test_k3_binding_utilization_with_real_bf16_draft_geometry():
    """Binding-hole metric on real K3 geometry: full bindings use
    the whole parent; state bindings use 90.1%, dropping to ~63.5% when the
    five BF16 draft planes widen the parent."""
    base = kimi_tp8_layout()[2].bind(10)
    report = base.capacity_report()
    assert abs(report["full_attention"]["binding_utilization"] - 1.0) < 1e-3
    for k in range(3):
        assert (
            abs(report[f"linear_attention_{k}"]["binding_utilization"] - 0.901) < 1e-3
        )

    merged = kimi_tp8_layout(draft_layers=5)[2]
    merged = merged.bind(10)
    widened = merged.capacity_report()
    assert abs(widened["full_attention"]["binding_utilization"] - 1.0) < 1e-3
    assert abs(widened["linear_attention_0"]["binding_utilization"] - 0.6359) < 1e-3


@pytest.mark.parametrize("limit", [1, 16384])
@pytest.mark.parametrize("kvp_size", [1, 4])
def test_token_limit_retains_kda_working_set(limit, kvp_size):
    from dataclasses import replace

    recipe, _, layout = kimi_tp8_layout(max_bs=4)
    recipe.attn_config = replace(
        recipe.attn_config,
        device="cuda",
        kvp_size=kvp_size,
        kvp_group=tuple(range(kvp_size)),
        components=(
            replace(recipe.attn_config.components[0], backend_name="flashmla"),
            recipe.attn_config.components[1],
        ),
    )
    recipe.server_args.max_total_tokens = limit
    setup = recipe.setup()
    assert setup.spec.token_capacity == limit
    assert setup.spec.memory_plan.num_lcm_blocks == recipe.parents_needed(layout, limit)
    assert recipe.parents_needed(layout, limit) > (
        limit // (recipe._max_packing(layout) * layout.prefix_granularity)
    )


@pytest.mark.parametrize("width,capacity", [(1, 16), (4, 16), (20, 20)])
def test_retained_kimi_schema_and_prefill_workspace(width, capacity):
    from test.runtime.conftest import kimi_recipe_with_history

    recipe = kimi_recipe_with_history(
        speculative_num_draft_tokens=width, history_capacity=None
    )
    fields = [field for _, fields in recipe.groups() for field in fields]
    corrections = [
        field for field in fields if field.field_id.endswith(".replay_correction")
    ]
    assert len(corrections) == 69
    assert all(field.shape == (12, capacity, 128) for field in corrections)
    schema = recipe.groups()
    recipe.server_args.disaggregation_mode = "prefill"
    assert recipe.groups() == schema
    assert recipe.workspace_bytes() == 0


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires NVIDIA portable retained kernels",
)
@pytest.mark.parametrize("head_dim", [64, 128])
def test_retained_recipe_supports_portable_square_geometry(head_dim):
    from dataclasses import replace
    from test.runtime.conftest import kimi_recipe_with_history

    from tokenspeed.runtime.layers.attention.configs.linear_attn import LinearAttnConfig
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.kda import (
        kda_retained_supported,
    )

    recipe = kimi_recipe_with_history(history_capacity=None)
    linear = recipe.attn_config.component(LinearAttnConfig)
    config = replace(
        recipe.attn_config,
        components=tuple(
            (
                replace(component, head_k_dim=head_dim, head_v_dim=head_dim)
                if component is linear
                else component
            )
            for component in recipe.attn_config.components
        ),
    )
    assert kda_retained_supported(config)


def test_retained_recipe_rejects_nonsquare_recurrent_geometry():
    from dataclasses import replace
    from test.runtime.conftest import kimi_recipe_with_history

    from tokenspeed.runtime.layers.attention.configs.linear_attn import LinearAttnConfig
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.kda import (
        kda_retained_supported,
    )

    recipe = kimi_recipe_with_history(history_capacity=None)
    linear = recipe.attn_config.component(LinearAttnConfig)
    config = replace(
        recipe.attn_config,
        components=tuple(
            replace(component, head_k_dim=64) if component is linear else component
            for component in recipe.attn_config.components
        ),
    )
    assert not kda_retained_supported(config)

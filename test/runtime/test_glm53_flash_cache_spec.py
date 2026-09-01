"""GLM-5.3-Flash cache-recipe tests."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig
from tokenspeed.runtime.layers.attention.configs.linear_attn import (
    LinearAttnConfig,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.glm53_flash import (
    GLM53_FLASH_LOGICAL_BLOCK_TOKENS,
    Glm53FlashPoolOptions,
    Glm53FlashRecipe,
    glm53_flash_parents_needed,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import CacheLayout, pack
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    split_recurrent_state_groups,
)

_NUM_LAYERS = 45
_FULL_LAYER_IDS = tuple(range(3, _NUM_LAYERS, 4))
_LINEAR_LAYER_IDS = tuple(
    layer_id for layer_id in range(_NUM_LAYERS) if layer_id not in _FULL_LAYER_IDS
)
_TARGET_LAYER_TYPES = tuple(
    LINEAR_ATTENTION if layer_id in _LINEAR_LAYER_IDS else FULL_ATTENTION
    for layer_id in range(_NUM_LAYERS)
)


def _recipe(
    *,
    tp_size: int,
    mla_cache_dtype: torch.dtype,
    draft_layers: int = 0,
    linear_tp_size: int | None = None,
    pd_disaggregation_enabled: bool = False,
) -> Glm53FlashRecipe:
    linear_tp_size = linear_tp_size or tp_size
    dsa = DSAConfig(
        backend_name="dsa",
        num_attention_heads=64,
        num_kv_heads=64,
        head_dim=256,
        attn_tp_size=tp_size,
        layer_types=_TARGET_LAYER_TYPES,
        kv_lora_rank=512,
        qk_nope_head_dim=256,
        qk_rope_head_dim=0,
        v_head_dim=256,
        scaling=256**-0.5,
        kv_cache_dim=512,
        index_topk=2048,
        index_head_dim=128,
        index_n_heads=32,
        index_kpool=4,
    )
    linear = LinearAttnConfig(
        num_k_heads=64,
        num_v_heads=64,
        head_k_dim=128,
        head_v_dim=128,
        conv_kernel_size=4,
        layer_ids=_LINEAR_LAYER_IDS,
        tp_size=linear_tp_size,
    )
    attn_config = AttnConfig(
        device="cpu",
        dtype=torch.bfloat16,
        kv_cache_dtype=mla_cache_dtype,
        kv_cache_quant_method="none",
        prefix_granularity=GLM53_FLASH_LOGICAL_BLOCK_TOKENS,
        context_len=4096,
        max_bs=16,
        max_graph_bs=16,
        pd_disaggregation_enabled=pd_disaggregation_enabled,
        speculative_num_steps=2 if draft_layers else 0,
        speculative_num_draft_tokens=3 if draft_layers else 1,
        components=(dsa, linear),
    )
    draft_attn_config = (
        replace(
            attn_config,
            is_draft=True,
            components=(replace(dsa, layer_types=(FULL_ATTENTION,)),),
        )
        if draft_layers
        else None
    )
    return Glm53FlashRecipe(
        server_args=SimpleNamespace(
            max_total_tokens=None,
            chunked_prefill_size=8192,
            speculative_algorithm="MTP" if draft_layers else None,
            speculative_num_draft_tokens=3,
        ),
        model_config=SimpleNamespace(num_attention_layers=_NUM_LAYERS),
        attn_config=attn_config,
        draft_model_config=(
            SimpleNamespace(num_attention_layers=draft_layers) if draft_layers else None
        ),
        draft_attn_config=draft_attn_config,
        cache_budget_bytes=1 << 34,
        decode_input_tokens=4 if draft_layers else 1,
        overlap_schedule_depth=0,
    )


def _layout(
    *,
    tp_size: int,
    mla_cache_dtype: torch.dtype,
    draft_layers: int = 0,
    linear_tp_size: int | None = None,
) -> CacheLayout:
    recipe = _recipe(
        tp_size=tp_size,
        linear_tp_size=linear_tp_size,
        mla_cache_dtype=mla_cache_dtype,
        draft_layers=draft_layers,
    )
    groups = recipe.groups()
    return pack(
        groups,
        prefix_granularity=recipe.prefix_granularity,
        cache_blocks_per_lcm_block=recipe.packing(groups),
        alignment=recipe.alignment,
        max_padding_fraction=recipe.max_padding_fraction,
    )


def test_lcm_reference_geometry_is_exact() -> None:
    plan = _layout(
        tp_size=8,
        mla_cache_dtype=torch.float8_e4m3fn,
    ).bind(7)
    assert plan.prefix_granularity == 64
    assert plan.lcm_block_bytes == 7_031_808
    assert len(plan.planes) == 12
    assert {
        group.group_id: group.cache_blocks_per_lcm_block for group in plan.groups
    } == {
        "full_attention": 18,
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
    assert len(fields_by_group["full_attention"]) == 22
    assert (
        sum(
            field.field_id.endswith(".index_k")
            for field in fields_by_group["full_attention"]
        )
        == 11
    )
    assert len(fields_by_group["linear_attention_0"]) == 24
    assert len(fields_by_group["linear_attention_1"]) == 22
    assert len(fields_by_group["linear_attention_2"]) == 22
    assert {field.plane_id for field in fields_by_group["linear_attention_0"]} == {
        f"slot.{slot}" for slot in range(12)
    }


def test_mtp_layout_reserves_a_separate_index_plane() -> None:
    layout = _layout(
        tp_size=4,
        mla_cache_dtype=torch.float8_e4m3fn,
        draft_layers=1,
    )
    group_ids = tuple(split_recurrent_state_groups(_TARGET_LAYER_TYPES)) + (
        FULL_ATTENTION,
    )

    assert len(layout.plane_bytes) == 13
    Glm53FlashRecipe.check_layout(
        type("Recipe", (), {"group_ids": group_ids})(), layout
    )


def test_dsa_and_kda_components_own_independent_tp_geometry() -> None:
    recipe = _recipe(
        tp_size=4,
        linear_tp_size=8,
        mla_cache_dtype=torch.bfloat16,
    )
    groups = recipe.groups()
    full_packing = recipe.packing(groups)[FULL_ATTENTION]
    conv = next(
        field
        for spec, fields in groups
        if spec.group_id.startswith(LINEAR_ATTENTION)
        for field in fields
        if field.field_id.endswith(".conv_state")
    )

    assert full_packing == 18
    assert conv.shape == (3 * 64 * 128 // 8, 3)


def test_request_local_tail_workspace_reserves_rollback_slots() -> None:
    options = Glm53FlashPoolOptions(
        index_kpool=4,
        tail_extra_slots=3,
        index_head_dim=128,
        num_request_slots=10,
        dsa_layer_ids=(3, 7),
    )

    assert options.tail_width == 7
    assert options.workspace_bytes == 2 * 2 * 10 * 7 * 128 * 2


def test_disaggregated_serving_requires_private_tail_transfer_bridge() -> None:
    recipe = _recipe(
        tp_size=4,
        mla_cache_dtype=torch.bfloat16,
        pd_disaggregation_enabled=True,
    )
    with pytest.raises(NotImplementedError, match="request-local KPool tail"):
        recipe.groups()


def test_lcm_parent_demand_uses_per_group_packing() -> None:
    layout = _layout(
        tp_size=8,
        mla_cache_dtype=torch.float8_e4m3fn,
    )
    sizing = dict(
        max_scheduled_tokens=8_192,
        max_live_requests=1,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )

    assert glm53_flash_parents_needed(layout, token_capacity=131_072, **sizing) == 501


@pytest.mark.parametrize("budget_slack", [0, 100])
def test_lcm_parent_budget_preserves_heterogeneous_group_demand(
    budget_slack: int,
) -> None:
    layout = _layout(
        tp_size=4,
        mla_cache_dtype=torch.bfloat16,
        draft_layers=1,
    )
    token_limit = 131_072
    expected = glm53_flash_parents_needed(
        layout,
        token_capacity=token_limit,
        max_scheduled_tokens=8_192,
        max_live_requests=16,
        decode_input_tokens=2,
        overlap_schedule_depth=1,
    )
    budgeted = expected + budget_slack

    class Recipe:
        family = "GLM-5.3-Flash"
        cache_budget_bytes = (budgeted + 1) * layout.lcm_block_bytes

        def workspace_bytes(self) -> int:
            return 0

        def parents_needed(self, candidate_layout, token_capacity: int) -> int:
            assert candidate_layout is layout
            assert token_capacity == token_limit
            return expected

    recipe = Recipe()
    recipe.token_limit = token_limit

    assert Glm53FlashRecipe.num_lcm_blocks(recipe, layout) == expected

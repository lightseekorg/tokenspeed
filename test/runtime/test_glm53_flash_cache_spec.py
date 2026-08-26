"""GLM-5.3-Flash cache-recipe tests."""

from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.configs.glm53_flash_config import Glm53FlashTextConfig
from tokenspeed.runtime.layers.attention.kv_cache.recipes.glm53_flash import (
    GLM53_FLASH_LOGICAL_BLOCK_TOKENS,
    Glm53FlashPoolOptions,
    Glm53FlashRecipe,
    declare_glm53_flash_groups,
    glm53_flash_packing_counts,
    glm53_flash_parents_needed,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import CacheLayout, pack
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    split_recurrent_state_groups,
)


def _text_config() -> Glm53FlashTextConfig:
    full_attn_layers = list(range(3, 45, 4))
    full_attn_layer_set = set(full_attn_layers)
    return Glm53FlashTextConfig(
        linear_attn_config={
            "num_heads": 64,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
            "kda_layers": [
                layer for layer in range(45) if layer not in full_attn_layer_set
            ],
            "full_attn_layers": full_attn_layers,
        }
    )


def _layout(
    text_config: Glm53FlashTextConfig,
    *,
    tp_size: int,
    mla_cache_dtype: torch.dtype,
    draft_layers: int = 0,
) -> CacheLayout:
    groups = declare_glm53_flash_groups(
        text_config,
        tp_size=tp_size,
        mla_cache_dtype=mla_cache_dtype,
        draft_layers=draft_layers,
    )
    state_group_ids = [
        spec.group_id
        for spec, _ in groups
        if spec.group_id.startswith(LINEAR_ATTENTION)
    ]
    packing = glm53_flash_packing_counts(
        tp_size=tp_size,
        mla_element_size=mla_cache_dtype.itemsize,
        state_group_ids=state_group_ids,
    )
    return pack(
        groups,
        prefix_granularity=GLM53_FLASH_LOGICAL_BLOCK_TOKENS,
        cache_blocks_per_lcm_block=packing,
        alignment=256,
        max_padding_fraction=float("inf") if draft_layers else 0.25,
    )


def test_lcm_reference_geometry_is_exact() -> None:
    plan = _layout(
        _text_config(),
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
    text_config = _text_config()
    layout = _layout(
        text_config,
        tp_size=4,
        mla_cache_dtype=torch.float8_e4m3fn,
        draft_layers=1,
    )
    group_ids = tuple(
        split_recurrent_state_groups(text_config.paged_cache_layer_types)
    ) + (FULL_ATTENTION,)

    assert len(layout.plane_bytes) == 13
    Glm53FlashRecipe.check_layout(
        type("Recipe", (), {"group_ids": group_ids})(), layout
    )


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
    with pytest.raises(NotImplementedError, match="request-local KPool tail"):
        declare_glm53_flash_groups(
            _text_config(),
            tp_size=4,
            mla_cache_dtype=torch.bfloat16,
            pd_disaggregation_enabled=True,
        )


def test_lcm_parent_demand_uses_per_group_packing() -> None:
    layout = _layout(
        _text_config(),
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
        _text_config(),
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

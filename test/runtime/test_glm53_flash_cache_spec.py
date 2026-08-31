"""GLM-5.3-Flash cache-recipe tests."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.configs.glm53_flash_config import Glm53FlashTextConfig
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig
from tokenspeed.runtime.layers.attention.configs.linear_attn import LinearAttnConfig
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


def _composite_attn_config(
    text_config: Glm53FlashTextConfig, *, tp_size: int = 4
) -> AttnConfig:
    dsa = DSAConfig(
        backend_name="dsa",
        num_attention_heads=text_config.num_attention_heads,
        num_kv_heads=text_config.num_key_value_heads,
        head_dim=text_config.qk_nope_head_dim + text_config.qk_rope_head_dim,
        attn_tp_size=tp_size,
        layer_types=tuple(text_config.paged_cache_layer_types),
        kv_lora_rank=text_config.kv_lora_rank,
        qk_nope_head_dim=text_config.qk_nope_head_dim,
        qk_rope_head_dim=text_config.qk_rope_head_dim,
        v_head_dim=text_config.v_head_dim,
        scaling=(text_config.qk_nope_head_dim + text_config.qk_rope_head_dim) ** -0.5,
        kv_cache_dim=text_config.kv_lora_rank + text_config.qk_rope_head_dim,
        index_topk=text_config.index_topk,
        index_head_dim=text_config.index_head_dim,
        index_n_heads=text_config.index_n_heads,
        index_kpool=text_config.index_kpool,
    )
    kda = text_config.linear_attn_config
    linear = LinearAttnConfig(
        num_k_heads=kda["num_heads"],
        num_v_heads=kda["num_heads"],
        head_k_dim=kda["head_dim"],
        head_v_dim=kda["head_dim"],
        conv_kernel_size=kda["short_conv_kernel_size"],
        layer_ids=tuple(text_config.linear_layer_ids),
        tp_size=tp_size,
    )
    return AttnConfig(
        device="cpu",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.float8_e4m3fn,
        kv_cache_quant_method="none",
        prefix_granularity=GLM53_FLASH_LOGICAL_BLOCK_TOKENS,
        context_len=4096,
        max_bs=2,
        max_graph_bs=2,
        max_scheduled_tokens=128,
        speculative_num_draft_tokens=3,
        components=(dsa, linear),
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


def test_composite_config_builds_target_and_mtp_cache_plan() -> None:
    text_config = _text_config()
    attn_config = _composite_attn_config(text_config)
    dsa = attn_config.component(DSAConfig)
    assert dsa is not None
    draft_attn_config = replace(attn_config, components=(dsa,))
    recipe = Glm53FlashRecipe(
        server_args=SimpleNamespace(
            max_total_tokens=4096,
            chunked_prefill_size=128,
            speculative_algorithm="MTP",
            speculative_num_draft_tokens=3,
        ),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(text_config=text_config)
        ),
        attn_config=attn_config,
        draft_model_config=SimpleNamespace(num_attention_layers=1),
        draft_attn_config=draft_attn_config,
        cache_budget_bytes=1 << 34,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )

    setup = recipe.setup()
    plan = setup.spec.memory_plan

    assert setup.num_draft_layers == 1
    assert plan.group(FULL_ATTENTION).cache_blocks_per_lcm_block == 36
    assert plan.field("layer.0.conv_state").shape == (3 * 64 * 128 // 4, 3)
    assert plan.field("layer.0.recurrent_state").shape == (64 // 4, 128, 128)
    assert plan.field("layer.3.latent_kv").shape == (64, 1, 512)
    assert plan.field("layer.45.latent_kv").shape == (64, 1, 512)


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

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
from tokenspeed.runtime.layers.attention.kv_cache.recipes.base import CacheRecipe
from tokenspeed.runtime.layers.attention.kv_cache.recipes.glm53_flash import (
    Glm53FlashPoolOptions,
    Glm53FlashRecipe,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import CacheLayout, pack
from tokenspeed.runtime.layers.attention.kv_cache.recipes.scheduler_bridge import (
    capacity_model,
)
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
    index_k_format: str = "fp8_scaled",
    mamba_ssm_dtype: str = "float32",
) -> Glm53FlashRecipe:
    linear_tp_size = linear_tp_size or tp_size
    dsa = DSAConfig(
        backend_name="dsa",
        num_attention_heads=64,
        num_kv_heads=64,
        head_dim=256,
        attn_tp_size=tp_size,
        cache_layer_types=_TARGET_LAYER_TYPES,
        kv_lora_rank=512,
        qk_nope_head_dim=256,
        qk_rope_head_dim=0,
        v_head_dim=256,
        scaling=256**-0.5,
        kv_cache_dim=512,
        index_topk=2048,
        index_head_dim=128,
        index_n_heads=32,
        index_k_format=index_k_format,
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
        context_len=4096,
        max_bs=16,
        pd_disaggregation_enabled=pd_disaggregation_enabled,
        speculative_num_steps=2 if draft_layers else 0,
        speculative_num_draft_tokens=3 if draft_layers else 1,
        components=(dsa, linear),
    )
    draft_attn_config = (
        replace(
            attn_config,
            is_draft=True,
            components=(replace(dsa, cache_layer_types=(FULL_ATTENTION,)),),
        )
        if draft_layers
        else None
    )
    return Glm53FlashRecipe(
        server_args=SimpleNamespace(
            kv_offload_config=None,
            max_total_tokens=None,
            chunked_prefill_size=8192,
            disaggregation_mode="null",
            enable_prefix_caching=True,
            speculative_algorithm="MTP" if draft_layers else None,
            speculative_num_draft_tokens=3,
            mamba_ssm_dtype=mamba_ssm_dtype,
        ),
        model_config=SimpleNamespace(num_attention_layers=_NUM_LAYERS),
        attn_config=attn_config,
        draft_model_config=(
            SimpleNamespace(num_attention_layers=draft_layers) if draft_layers else None
        ),
        draft_attn_config=draft_attn_config,
        cache_budget_bytes=1 << 34,
        probe_batch_rows=None,
        decode_input_tokens=4 if draft_layers else 1,
        overlap_schedule_depth=0,
    )


def _layout(
    *,
    tp_size: int,
    mla_cache_dtype: torch.dtype,
    draft_layers: int = 0,
    linear_tp_size: int | None = None,
    mamba_ssm_dtype: str = "float32",
) -> CacheLayout:
    recipe = _recipe(
        tp_size=tp_size,
        linear_tp_size=linear_tp_size,
        mla_cache_dtype=mla_cache_dtype,
        draft_layers=draft_layers,
        mamba_ssm_dtype=mamba_ssm_dtype,
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
    assert plan.lcm_block_bytes == 7_077_888
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


def test_bf16_kda_state_halves_the_parent_at_tp8() -> None:
    """``--mamba-ssm-dtype bfloat16`` halves the KDA page, so the MLA packing
    that covers one page halves too and so does the parent."""
    plan = _layout(
        tp_size=8, mla_cache_dtype=torch.float8_e4m3fn, mamba_ssm_dtype="bfloat16"
    ).bind(7)

    assert {
        group.group_id: group.cache_blocks_per_lcm_block for group in plan.groups
    } == {
        "full_attention": 9,
        "linear_attention_0": 1,
        "linear_attention_1": 1,
        "linear_attention_2": 1,
    }
    assert plan.lcm_block_bytes == 7_077_888 // 2
    recurrent = next(
        field for field in plan.fields if field.field_id.endswith(".recurrent_state")
    )
    assert recurrent.dtype == "bfloat16"


def test_bf16_kda_state_rejects_tp8_with_a_bf16_mla_cache() -> None:
    """Half of the FP32-state packing is too narrow for TP8 with a BF16 MLA
    cache, so that shape raises instead of mis-packing the layout."""
    with pytest.raises(NotImplementedError, match="no verified LCM packing"):
        _layout(tp_size=8, mla_cache_dtype=torch.bfloat16, mamba_ssm_dtype="bfloat16")


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


@pytest.mark.parametrize("mamba_ssm_dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize(
    ("tp_size", "mla_cache_dtype", "draft_layers"),
    [
        (8, torch.float8_e4m3fn, 0),
        (4, torch.float8_e4m3fn, 0),
        (4, torch.bfloat16, 0),
        (4, torch.float8_e4m3fn, 1),
    ],
)
def test_kda_layers_share_one_page_stride(
    tp_size: int,
    mla_cache_dtype: torch.dtype,
    draft_layers: int,
    mamba_ssm_dtype: str,
) -> None:
    # Batched KDA replay addresses every layer's state pages with one stride.
    layout = _layout(
        tp_size=tp_size,
        mla_cache_dtype=mla_cache_dtype,
        draft_layers=draft_layers,
        mamba_ssm_dtype=mamba_ssm_dtype,
    )
    strides = {
        field.page_stride_bytes
        for field in layout.fields
        if field.group_id.startswith(LINEAR_ATTENTION)
    }
    assert len(strides) == 1


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


def test_pooled_index_rows_are_planned_for_the_fp8_plane_only() -> None:
    # The recipe sizes the pooled index-K rows as FP8 keys plus a scale; a
    # config naming the bf16 plane must be refused, not silently misbudgeted.
    recipe = _recipe(tp_size=4, mla_cache_dtype=torch.bfloat16, index_k_format="bf16")
    with pytest.raises(ValueError, match="fp8_scaled"):
        recipe.groups()


def test_lcm_parent_demand_uses_per_group_packing() -> None:
    recipe = _recipe(tp_size=8, mla_cache_dtype=torch.float8_e4m3fn)
    groups = recipe.groups()
    layout = pack(
        groups,
        prefix_granularity=recipe.prefix_granularity,
        cache_blocks_per_lcm_block=recipe.packing(groups),
        alignment=recipe.alignment,
        max_padding_fraction=recipe.max_padding_fraction,
    )
    packing = dict(layout.group_packing)
    token_capacity = 131_072

    # The scheduler's model sizes every group; the recipe folds each group's
    # pages into parents by its own packing rather than a flat product.
    model = capacity_model(
        recipe._group_specs,
        prefix_granularity=layout.prefix_granularity,
        virtual_packing=packing,
        limits=recipe.scheduler_limits,
    )
    pages = dict(
        zip(
            (spec.group_id for spec in recipe._group_specs),
            model.concurrent_group_pages(
                max_total_tokens=token_capacity, max_context_len=4096
            ),
        )
    )
    # Dense MLA history plus one unaligned tail page per live request; each
    # linear-attention state group holds its fixed per-request working set.
    assert pages[FULL_ATTENTION] == 131_072 // 64 + 16
    state_ids = [
        spec.group_id for spec in recipe._group_specs if spec.attn_kind == "state"
    ]
    assert state_ids and all(pages[gid] == 16 * 4 for gid in state_ids)
    expected = sum(-(-pages[gid] // packing[gid]) for gid in pages)
    assert expected == recipe.parents_needed(layout, token_capacity)
    assert expected != sum(pages.values()) // max(packing.values())


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
    expected = CacheRecipe.parents_needed(
        _recipe(tp_size=4, mla_cache_dtype=torch.bfloat16, draft_layers=1),
        layout,
        token_limit,
    )
    budgeted = expected + budget_slack

    recipe = _recipe(tp_size=4, mla_cache_dtype=torch.bfloat16, draft_layers=1)
    recipe.token_limit = token_limit
    assert recipe.select_num_lcm_blocks(layout, max_lcm_blocks=budgeted) == expected

from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.configs.paged_cache_spec import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    PagedCacheGroupSpec,
)
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.configs.msa import MSAConfig
from tokenspeed.runtime.layers.attention.kv_cache.factory import create_cache_pool
from tokenspeed.runtime.layers.attention.kv_cache.hybrid_mha import (
    HybridMHATokenToKVPool,
)
from tokenspeed.runtime.layers.attention.kv_cache.mha import MHATokenToKVPool
from tokenspeed.runtime.layers.attention.kv_cache.mla import MLATokenToKVPool
from tokenspeed.runtime.layers.attention.kv_cache.plan import CacheFieldSpec
from tokenspeed.runtime.layers.attention.kv_cache.recipes.ordinary import (
    build_hybrid_cache_setup,
)
from tokenspeed.runtime.layers.attention.kv_cache.setup import prepare_cache_setup
from tokenspeed.runtime.layers.attention.registry import (
    _validate_shared_cache_geometry,
)


def _mha_config() -> MHAConfig:
    return MHAConfig(
        device="cpu",
        backend_name="fa2",
        num_attention_heads=1,
        layer_types=(),
        kv_cache_mxfp8=False,
        num_kv_heads=1,
        attn_tp_size=1,
        head_dim=2,
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        context_len=1024,
        max_graph_bs=2,
        max_bs=2,
        page_size=64,
        kv_cache_quant_method="none",
        max_scheduled_tokens=128,
    )


def _mla_config() -> MLAConfig:
    return MLAConfig(
        device="cpu",
        backend_name="trtllm_mla",
        num_attention_heads=1,
        num_kv_heads=1,
        attn_tp_size=1,
        head_dim=8,
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        context_len=1024,
        max_graph_bs=2,
        max_bs=2,
        page_size=64,
        kv_cache_quant_method="none",
        kv_lora_rank=4,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=4,
        scaling=1.0,
        kv_cache_dim=6,
        max_scheduled_tokens=128,
    )


def _msa_config() -> MSAConfig:
    return MSAConfig(
        device="cpu",
        backend_name="msa",
        num_attention_heads=1,
        num_kv_heads=1,
        attn_tp_size=1,
        head_dim=2,
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        context_len=1024,
        max_graph_bs=2,
        max_bs=2,
        page_size=64,
        kv_cache_quant_method="none",
        compute_layer_types=("full_attention", "sparse_attention"),
        sparse_layer_ids=frozenset({1}),
        max_scheduled_tokens=128,
        index_head_dim=4,
        index_n_heads=1,
        index_block_size=64,
        index_topk_blocks=1,
        index_init_blocks=1,
        index_local_blocks=1,
    )


def _hybrid_setup_with_narrow_draft():
    return build_hybrid_cache_setup(
        family="inkling",
        server_args=SimpleNamespace(max_total_tokens=None),
        fields=(
            CacheFieldSpec("full_attention", "target.full", "shared", (256,), 1),
            CacheFieldSpec("state", "target.state", "shared", (128,), 1),
        ),
        layer_types=("full_attention",),
        group_ids=("full_attention",),
        state_dtypes={},
        layer_kv_head_counts=None,
        draft_fields=(
            CacheFieldSpec("full_attention", "draft.full", "shared", (256,), 1),
        ),
        draft_layer_types=("full_attention",),
        draft_group_ids=("full_attention",),
        draft_layer_kv_head_counts=None,
        cache_budget_bytes=2_048,
        fixed_workspace_bytes=0,
        logical_block_tokens=4,
        max_padding_fraction=1.0,
    )


def test_attention_configs_do_not_own_cache_setup() -> None:
    cache_setup_fields = {
        "conv_state_shape",
        "temporal_state_shape",
        "recurrent_state_shape",
        "conv_dtype",
        "ssm_dtype",
        "recurrent_dtype",
        "lcm_memory_plan",
        "layer_cache_group_ids",
        "token_capacity",
    }

    assert cache_setup_fields.isdisjoint(field.name for field in fields(MHAConfig))
    assert cache_setup_fields.isdisjoint(field.name for field in fields(MLAConfig))
    assert not hasattr(MHAConfig, "create_pool")
    assert not hasattr(MLAConfig, "create_pool")


def test_qwen_recipe_preserves_backend_kernel_page_size() -> None:
    text_config = SimpleNamespace(
        mamba2_cache_params=(
            (2, 2),
            (1, 2, 2),
            torch.bfloat16,
            torch.float32,
            (0,),
        )
    )
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(text_config=text_config),
    )
    attn_config = MHAConfig(
        device="cpu",
        backend_name="fa2",
        num_attention_heads=1,
        layer_types=(LINEAR_ATTENTION, FULL_ATTENTION),
        kv_cache_mxfp8=False,
        num_kv_heads=1,
        attn_tp_size=1,
        head_dim=2,
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        context_len=1024,
        max_graph_bs=2,
        max_bs=2,
        page_size=64,
        kv_cache_quant_method="none",
        max_scheduled_tokens=128,
    )
    server_args = SimpleNamespace(
        block_size=64,
        max_total_tokens=None,
        speculative_num_draft_tokens=0,
    )

    setup = prepare_cache_setup(
        family="qwen_gdn",
        server_args=server_args,
        model_config=model_config,
        attn_config=attn_config,
        draft_model_config=None,
        draft_attn_config=None,
        cache_budget_bytes=16_384,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )

    assert server_args.block_size == 64
    assert attn_config.page_size == 64
    assert setup.target.memory_plan.logical_block_tokens == 128
    assert setup.draft is None
    assert setup.target.layer_group_ids == (
        f"{LINEAR_ATTENTION}_0",
        FULL_ATTENTION,
    )
    assert setup.target.state_field_dtypes == {
        "layer.0.conv": torch.bfloat16,
        "layer.0.ssm": torch.float32,
    }
    assert not hasattr(attn_config, "lcm_memory_plan")
    pool = create_cache_pool(
        setup.target,
        attn_config,
        num_layers=2,
        rank=0,
        enable_memory_saver=False,
    )
    assert type(pool) is HybridMHATokenToKVPool
    assert pool.buffer is not None


def test_ordinary_mha_reserves_null_parent_within_cache_budget() -> None:
    model_config = SimpleNamespace(
        num_attention_layers=2,
        hf_config=SimpleNamespace(),
    )
    attn_config = _mha_config()
    server_args = SimpleNamespace(max_total_tokens=None)

    setup = prepare_cache_setup(
        family="mha",
        server_args=server_args,
        model_config=model_config,
        attn_config=attn_config,
        draft_model_config=None,
        draft_attn_config=None,
        cache_budget_bytes=16_384,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )

    assert setup.target.family == "mha"
    assert setup.target.memory_plan.logical_block_tokens == 64
    assert setup.target.memory_plan.num_lcm_blocks == 15
    assert setup.target.memory_plan.arena_bytes <= 16_384
    assert setup.target.token_capacity == 960
    assert setup.draft is None
    pool = create_cache_pool(
        setup.target,
        attn_config,
        num_layers=2,
        rank=0,
        enable_memory_saver=False,
    )
    assert type(pool) is MHATokenToKVPool
    assert pool.runtime_contract.token_capacity == setup.target.token_capacity
    with pytest.raises(TypeError, match="incompatible with MHAConfig"):
        create_cache_pool(
            replace(setup.target, family="kimi_k3"),
            attn_config,
            num_layers=2,
            rank=0,
            enable_memory_saver=False,
        )


def test_ordinary_mla_reserves_null_parent_within_cache_budget() -> None:
    model_config = SimpleNamespace(
        num_attention_layers=2,
        hf_config=SimpleNamespace(),
    )
    attn_config = _mla_config()
    server_args = SimpleNamespace(max_total_tokens=None)

    setup = prepare_cache_setup(
        family="mla",
        server_args=server_args,
        model_config=model_config,
        attn_config=attn_config,
        draft_model_config=None,
        draft_attn_config=None,
        cache_budget_bytes=24_576,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )

    assert setup.target.family == "mla"
    assert setup.target.memory_plan.logical_block_tokens == 64
    assert setup.target.memory_plan.num_lcm_blocks == 15
    assert setup.target.memory_plan.arena_bytes <= 24_576
    assert setup.target.token_capacity == 960
    assert setup.draft is None
    pool = create_cache_pool(
        setup.target,
        attn_config,
        num_layers=2,
        rank=0,
        enable_memory_saver=False,
    )
    assert type(pool) is MLATokenToKVPool
    assert pool.runtime_contract.token_capacity == setup.target.token_capacity


@pytest.mark.parametrize(
    ("family", "target_config"),
    (("mla", _mla_config), ("msa", _msa_config)),
)
def test_ordinary_recipe_uses_the_draft_attention_family(
    family: str,
    target_config,
) -> None:
    model_config = SimpleNamespace(num_attention_layers=2, hf_config=SimpleNamespace())
    draft_model_config = SimpleNamespace(
        num_attention_layers=1, hf_config=SimpleNamespace()
    )
    draft_attn_config = _mha_config()

    setup = prepare_cache_setup(
        family=family,
        server_args=SimpleNamespace(max_total_tokens=None),
        model_config=model_config,
        attn_config=target_config(),
        draft_model_config=draft_model_config,
        draft_attn_config=draft_attn_config,
        cache_budget_bytes=65_536,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )

    assert setup.draft is not None
    assert setup.draft.family == "mha"
    draft_pool = create_cache_pool(
        setup.draft,
        draft_attn_config,
        num_layers=1,
        rank=0,
        enable_memory_saver=False,
    )
    assert type(draft_pool) is MHATokenToKVPool
    assert draft_pool.runtime_contract.token_capacity == setup.draft.token_capacity


def test_shared_cache_geometry_uses_current_group_spec_fields() -> None:
    setup = _hybrid_setup_with_narrow_draft()
    assert setup.draft is not None
    target_group = setup.target.memory_plan.group("full_attention")
    spec = PagedCacheGroupSpec(
        group_id="full_attention",
        retention="full_history",
        rows_per_page=4,
        entry_stride_tokens=1,
        sliding_window_tokens=None,
        cache_blocks_per_lcm_block=target_group.cache_blocks_per_lcm_block,
    )
    target_pool = SimpleNamespace(
        runtime_contract=object(),
        plan=setup.target.memory_plan,
        paged_cache_group_specs=(spec,),
        buffer=torch.empty(1),
    )
    draft_pool = SimpleNamespace(
        runtime_contract=object(),
        plan=setup.draft.memory_plan,
        paged_cache_group_specs=(spec,),
        buffer=torch.empty(1),
    )

    _validate_shared_cache_geometry(target_pool, draft_pool)


def test_hybrid_draft_capacity_uses_its_own_group_packing() -> None:
    setup = _hybrid_setup_with_narrow_draft()

    assert setup.draft is not None
    draft_max_packing = max(
        group.cache_blocks_per_lcm_block for group in setup.draft.memory_plan.groups
    )
    assert setup.draft.token_capacity == (
        setup.draft.memory_plan.num_lcm_blocks * draft_max_packing * 4
    )
    assert setup.draft.token_capacity < setup.target.token_capacity

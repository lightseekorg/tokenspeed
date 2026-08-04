from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.configs.paged_cache_spec import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
)
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.kv_cache.factory import create_cache_pool
from tokenspeed.runtime.layers.attention.kv_cache.hybrid_mha import (
    HybridMHATokenToKVPool,
)
from tokenspeed.runtime.layers.attention.kv_cache.mha import MHATokenToKVPool
from tokenspeed.runtime.layers.attention.kv_cache.mla import MLATokenToKVPool
from tokenspeed.runtime.layers.attention.kv_cache.setup import prepare_cache_setup


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


def test_ordinary_mha_uses_cache_setup_with_legacy_capacity() -> None:
    model_config = SimpleNamespace(
        num_attention_layers=2,
        hf_config=SimpleNamespace(),
    )
    attn_config = MHAConfig(
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
    assert setup.target.memory_plan.num_lcm_blocks == 16
    assert setup.target.token_capacity == 1024
    assert setup.draft is None
    pool = create_cache_pool(
        setup.target,
        attn_config,
        num_layers=2,
        rank=0,
        enable_memory_saver=False,
    )
    assert type(pool) is MHATokenToKVPool
    with pytest.raises(TypeError, match="incompatible with MHAConfig"):
        create_cache_pool(
            replace(setup.target, family="kimi_k3"),
            attn_config,
            num_layers=2,
            rank=0,
            enable_memory_saver=False,
        )


def test_ordinary_mla_uses_cache_setup_with_legacy_capacity() -> None:
    model_config = SimpleNamespace(
        num_attention_layers=2,
        hf_config=SimpleNamespace(),
    )
    attn_config = MLAConfig(
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
    assert setup.target.memory_plan.num_lcm_blocks == 16
    assert setup.target.token_capacity == 1024
    assert setup.draft is None
    pool = create_cache_pool(
        setup.target,
        attn_config,
        num_layers=2,
        rank=0,
        enable_memory_saver=False,
    )
    assert type(pool) is MLATokenToKVPool

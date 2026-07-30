from dataclasses import fields
from types import SimpleNamespace
from unittest import mock

import torch

from tokenspeed.runtime.configs.paged_cache_spec import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
)
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.kv_cache.lcm_mha import (
    LcmMHATokenToKVPool,
)
from tokenspeed.runtime.layers.attention.lcm_setup import (
    create_lcm_pool,
    prepare_lcm_setup,
)

_FLAT_PROBE = "tokenspeed.runtime.configs.paged_cache_spec.scheduler_ext_flat_kvcache"


def test_attention_configs_do_not_own_lcm_setup() -> None:
    lcm_only_fields = {
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

    assert lcm_only_fields.isdisjoint(field.name for field in fields(MHAConfig))
    assert lcm_only_fields.isdisjoint(field.name for field in fields(MLAConfig))


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

    setup = prepare_lcm_setup(
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
    with mock.patch(_FLAT_PROBE, return_value=True):
        pool = create_lcm_pool(
            setup.target,
            attn_config,
            num_layers=2,
            rank=0,
            enable_memory_saver=False,
        )
    assert isinstance(pool, LcmMHATokenToKVPool)

from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.configs.kimi_k3_cache_spec import (
    kimi_k3_layer_group_ids,
    plan_kimi_k3_lcm_cache,
)
from tokenspeed.runtime.configs.kimi_k3_config import KimiLinearConfig
from tokenspeed.runtime.configs.paged_cache_spec import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
)
from tokenspeed.runtime.layers.attention.kv_cache.mla import MLATokenToKVPool


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_kimi_k3_pool_binds_mla_and_kda_to_one_lcm_backing() -> None:
    text_config = KimiLinearConfig()
    num_lcm_blocks = 2
    plan = plan_kimi_k3_lcm_cache(
        text_config,
        flat_kvcache_enabled=True,
        tp_size=8,
        mla_cache_dtype=torch.float8_e4m3fn,
        mla_quant_method=None,
        num_lcm_blocks=num_lcm_blocks,
    )
    group_ids = kimi_k3_layer_group_ids(text_config)
    layer_types = tuple(
        FULL_ATTENTION if group_id == FULL_ATTENTION else LINEAR_ATTENTION
        for group_id in group_ids
    )
    linear = text_config.linear_attn_config
    tp_size = 8
    conv_shape = (
        3 * linear["num_heads"] * linear["head_dim"] // tp_size,
        linear["short_conv_kernel_size"] - 1,
    )
    recurrent_shape = (
        linear["num_heads"] // tp_size,
        linear["head_dim"],
        linear["head_dim"],
    )
    pool = MLATokenToKVPool(
        size=num_lcm_blocks * 12 * plan.logical_block_tokens,
        model_dtype=torch.bfloat16,
        dtype=torch.float8_e4m3fn,
        quant_method=None,
        kv_lora_rank=text_config.kv_lora_rank,
        qk_rope_head_dim=text_config.qk_rope_head_dim,
        layer_num=text_config.num_hidden_layers,
        device="cuda",
        enable_memory_saver=False,
        max_batch_size=1,
        max_context_len=4096,
        page_size=plan.logical_block_tokens,
        rank=0,
        layer_types=layer_types,
        layer_cache_group_ids=group_ids,
        max_scheduled_tokens=1024,
        pd_disaggregation_enabled=True,
        conv_state_shape=conv_shape,
        recurrent_state_shape=recurrent_shape,
        conv_dtype=torch.bfloat16,
        recurrent_dtype=torch.float32,
        lcm_memory_plan=plan,
        token_capacity=1024,
    )

    assert pool.num_lcm_blocks == num_lcm_blocks
    assert pool.runtime_contract is not None
    assert pool.runtime_contract.token_capacity == 1024
    assert {
        spec.group_id: spec.transfer_policy for spec in pool.paged_cache_group_specs
    } == {
        FULL_ATTENTION: "full_suffix",
        f"{LINEAR_ATTENTION}_0": "latest_snapshot",
        f"{LINEAR_ATTENTION}_1": "latest_snapshot",
        f"{LINEAR_ATTENTION}_2": "latest_snapshot",
    }
    assert pool.runtime_contract.group_page_counts == {
        FULL_ATTENTION: num_lcm_blocks * 12 + 1,
        f"{LINEAR_ATTENTION}_0": num_lcm_blocks + 1,
        f"{LINEAR_ATTENTION}_1": num_lcm_blocks + 1,
        f"{LINEAR_ATTENTION}_2": num_lcm_blocks + 1,
    }
    full_layer = text_config.full_attention_layer_ids[0]
    state_layer = next(
        layer_id
        for layer_id, group_id in enumerate(group_ids)
        if group_id != FULL_ATTENTION
    )
    assert (
        pool.kv_buffer[full_layer].untyped_storage().data_ptr()
        == pool.lcm_pool.backing.untyped_storage().data_ptr()
    )
    conv, recurrent = pool.get_state_buffers(state_layer)
    assert tuple(conv.shape[1:]) == conv_shape
    assert tuple(recurrent.shape[1:]) == recurrent_shape
    assert (
        conv.untyped_storage().data_ptr()
        == recurrent.untyped_storage().data_ptr()
        == pool.lcm_pool.backing.untyped_storage().data_ptr()
    )

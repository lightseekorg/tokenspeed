from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.configs.kimi_k3_config import KimiLinearConfig
from tokenspeed.runtime.layers.attention.kv_cache.hybrid_kda import (
    HybridKDATokenToKVPool,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.kimi_k3 import (
    kimi_k3_layer_group_ids,
    solve_kimi_k3_cache_layout,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    build_paged_cache_group_specs,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_kimi_k3_pool_binds_mla_and_kda_to_one_lcm_backing() -> None:
    text_config = KimiLinearConfig()
    num_lcm_blocks = 2
    layout = solve_kimi_k3_cache_layout(
        text_config,
        tp_size=8,
        mla_cache_dtype=torch.float8_e4m3fn,
        mla_quant_method=None,
    )
    plan = layout.with_num_lcm_blocks(num_lcm_blocks)
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
    pool = HybridKDATokenToKVPool(
        size=num_lcm_blocks * 12 * plan.logical_block_tokens,
        model_dtype=torch.bfloat16,
        dtype=torch.float8_e4m3fn,
        quant_method=None,
        kv_lora_rank=text_config.kv_lora_rank,
        qk_rope_head_dim=text_config.qk_rope_head_dim,
        layer_num=text_config.num_hidden_layers,
        device="cuda",
        enable_memory_saver=False,
        page_size=plan.logical_block_tokens,
        rank=0,
        layer_types=layer_types,
        layer_group_ids=group_ids,
        pd_disaggregation_enabled=True,
        paged_cache_group_specs=build_paged_cache_group_specs(
            layer_types=layer_types,
            group_ids=group_ids,
            sliding_window_tokens=None,
            page_size=plan.logical_block_tokens,
            pd_disaggregation_enabled=True,
        ),
        state_field_dtypes={
            field_id: dtype
            for layer_id, layer_type in enumerate(layer_types)
            if layer_type == LINEAR_ATTENTION
            for field_id, dtype in (
                (f"layer.{layer_id}.conv_state", torch.bfloat16),
                (f"layer.{layer_id}.recurrent_state", torch.float32),
            )
        },
        memory_plan=plan,
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
        == pool.buffer.untyped_storage().data_ptr()
    )
    conv, recurrent = pool.get_state_buffers(state_layer)
    assert tuple(conv.shape[1:]) == conv_shape
    assert tuple(recurrent.shape[1:]) == recurrent_shape
    assert (
        conv.untyped_storage().data_ptr()
        == recurrent.untyped_storage().data_ptr()
        == pool.buffer.untyped_storage().data_ptr()
    )


def test_kimi_k3_bf16_draft_uses_typed_view_over_fp8_target_arena() -> None:
    from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
    from tokenspeed.runtime.layers.attention.kv_cache.factory import create_cache_pool
    from tokenspeed.runtime.layers.attention.kv_cache.mla import MLATokenToKVPool
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.ordinary import (
        mla_cache_fields,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import CachePoolSpec

    text_config = KimiLinearConfig()
    target_group_ids = kimi_k3_layer_group_ids(text_config)
    target_layer_types = tuple(
        FULL_ATTENTION if group_id == FULL_ATTENTION else LINEAR_ATTENTION
        for group_id in target_group_ids
    )
    num_draft_layers = 5
    draft_layer_types = (FULL_ATTENTION,) * num_draft_layers
    draft_fields = mla_cache_fields(
        layer_group_ids=draft_layer_types,
        logical_block_tokens=128,
        latent_width=576,
        element_size=torch.bfloat16.itemsize,
    )
    plan = solve_kimi_k3_cache_layout(
        text_config,
        tp_size=8,
        mla_cache_dtype=torch.float8_e4m3fn,
        mla_quant_method=None,
        draft_fields=draft_fields,
    ).with_num_lcm_blocks(1)
    merged_layer_types = target_layer_types + draft_layer_types
    merged_group_ids = target_group_ids + draft_layer_types
    state_dtypes = {
        f"layer.{layer_id}.conv_state": torch.bfloat16
        for layer_id, layer_type in enumerate(target_layer_types)
        if layer_type == LINEAR_ATTENTION
    } | {
        f"layer.{layer_id}.recurrent_state": torch.float32
        for layer_id, layer_type in enumerate(target_layer_types)
        if layer_type == LINEAR_ATTENTION
    }
    merged_spec = CachePoolSpec(
        family="kimi_k3",
        memory_plan=plan,
        layer_types=merged_layer_types,
        layer_group_ids=merged_group_ids,
        paged_cache_group_specs=build_paged_cache_group_specs(
            layer_types=merged_layer_types,
            group_ids=merged_group_ids,
            sliding_window_tokens=None,
            page_size=plan.logical_block_tokens,
            pd_disaggregation_enabled=False,
        ),
        state_field_dtypes=state_dtypes,
        token_capacity=128,
    )
    target_spec = merged_spec.layer_view(
        first_layer=0,
        num_layers=text_config.num_hidden_layers,
    )
    draft_spec = merged_spec.layer_view(
        first_layer=text_config.num_hidden_layers,
        num_layers=num_draft_layers,
        family="mla",
        publish_runtime_contract=False,
    )

    common_config = dict(
        device="cpu",
        backend_name="mla",
        num_attention_heads=64,
        num_kv_heads=64,
        attn_tp_size=8,
        head_dim=192,
        dtype=torch.bfloat16,
        context_len=1024,
        max_graph_bs=1,
        max_bs=1,
        page_size=plan.logical_block_tokens,
        kv_cache_quant_method="none",
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        scaling=1.0,
        kv_cache_dim=576,
        max_scheduled_tokens=128,
    )
    target_config = MLAConfig(
        kv_cache_dtype=torch.float8_e4m3fn,
        **common_config,
    )
    draft_config = MLAConfig(
        kv_cache_dtype=torch.bfloat16,
        is_draft=True,
        **common_config,
    )
    target_pool = create_cache_pool(
        target_spec,
        target_config,
        num_layers=text_config.num_hidden_layers,
        rank=0,
        enable_memory_saver=False,
    )
    draft_pool = create_cache_pool(
        draft_spec,
        draft_config,
        num_layers=num_draft_layers,
        rank=0,
        enable_memory_saver=False,
        field_layer_offset=text_config.num_hidden_layers,
        backing_pool=target_pool,
    )

    assert isinstance(target_pool, HybridKDATokenToKVPool)
    assert type(draft_pool) is MLATokenToKVPool
    assert draft_pool.buffer is target_pool.buffer
    assert draft_pool._fields is target_pool._fields
    assert draft_pool.runtime_contract is target_pool.runtime_contract
    target_cache = target_pool.get_key_buffer(text_config.full_attention_layer_ids[0])
    assert target_cache.dtype == torch.float8_e4m3fn
    assert draft_pool.get_key_buffer(0).dtype == torch.bfloat16
    assert (
        draft_pool.kv_buffer[0].data_ptr()
        == target_pool.field("layer.93.latent_kv", torch.bfloat16).data_ptr()
    )
    assert set(target_pool._fields) == {
        field.field_id for field in merged_spec.memory_plan.fields
    }

    target_layer = text_config.full_attention_layer_ids[0]
    target_before = target_pool.kv_buffer[target_layer].clone()
    draft_pool.kv_buffer[0][1].fill_(1)
    assert torch.equal(target_pool.kv_buffer[target_layer], target_before)
    draft_pool.clear_kv_buffers()
    assert torch.count_nonzero(draft_pool.kv_buffer[0])
    target_pool.clear_kv_buffers()
    assert not torch.count_nonzero(draft_pool.kv_buffer[0])

"""GLM-5.3-Flash cache-pool tests."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tokenspeed.runtime.configs.glm53_flash_config import Glm53FlashTextConfig
from tokenspeed.runtime.layers.attention.backends.cache_metadata import (
    CacheBatchMetadata,
)
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig
from tokenspeed.runtime.layers.attention.configs.linear_attn import (
    LinearAttnConfig,
)
from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
from tokenspeed.runtime.layers.attention.kv_cache.hybrid_glm53_flash import (
    HybridGlm53FlashTokenToKVPool,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.glm53_flash import (
    GLM53_FLASH_LOGICAL_BLOCK_TOKENS,
    Glm53FlashPoolOptions,
    declare_glm53_flash_groups,
    glm53_flash_packing_counts,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import pack
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    split_recurrent_state_groups,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_glm53_flash_pool_binds_paged_cache_and_request_local_tail() -> None:
    text_config = Glm53FlashTextConfig(
        num_hidden_layers=4,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "deepseek_sparse_attention",
        ],
        linear_attn_config={
            "num_heads": 64,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
            "kda_layers": [0, 1, 2],
            "full_attn_layers": [3],
        },
    )
    num_lcm_blocks = 2
    dsa = DSAConfig(
        backend_name="dsa",
        num_attention_heads=text_config.num_attention_heads,
        num_kv_heads=text_config.num_key_value_heads,
        head_dim=text_config.qk_head_dim,
        attn_tp_size=4,
        layer_types=tuple(text_config.paged_cache_layer_types),
        kv_lora_rank=text_config.kv_lora_rank,
        qk_nope_head_dim=text_config.qk_nope_head_dim,
        qk_rope_head_dim=text_config.qk_rope_head_dim,
        v_head_dim=text_config.v_head_dim,
        scaling=text_config.qk_head_dim**-0.5,
        kv_cache_dim=text_config.kv_lora_rank + text_config.qk_rope_head_dim,
        index_topk=text_config.index_topk,
        index_head_dim=text_config.index_head_dim,
        index_n_heads=text_config.index_n_heads,
        index_kpool=text_config.index_kpool,
    )
    linear = LinearAttnConfig(
        num_k_heads=text_config.linear_num_heads,
        num_v_heads=text_config.linear_num_heads,
        head_k_dim=text_config.linear_head_dim,
        head_v_dim=text_config.linear_head_dim,
        conv_kernel_size=text_config.linear_conv_kernel_dim,
        layer_ids=tuple(text_config.linear_layer_ids),
        tp_size=4,
    )
    attn_config = AttnConfig(
        device="cuda",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.float8_e4m3fn,
        kv_cache_quant_method="none",
        prefix_granularity=GLM53_FLASH_LOGICAL_BLOCK_TOKENS,
        context_len=4096,
        max_bs=8,
        components=(dsa, linear),
    )
    draft_attn_config = replace(
        attn_config,
        is_draft=True,
        components=(replace(dsa, layer_types=(FULL_ATTENTION,)),),
    )
    groups = declare_glm53_flash_groups(
        text_config.num_hidden_layers,
        attn_config=attn_config,
        draft_attn_config=draft_attn_config,
        draft_layers=1,
    )
    state_group_ids = [
        spec.group_id
        for spec, _ in groups
        if spec.group_id.startswith(LINEAR_ATTENTION)
    ]
    packing = glm53_flash_packing_counts(
        tp_size=4,
        mla_element_size=torch.float8_e4m3fn.itemsize,
        state_group_ids=state_group_ids,
    )
    plan = pack(
        groups,
        prefix_granularity=GLM53_FLASH_LOGICAL_BLOCK_TOKENS,
        cache_blocks_per_lcm_block=packing,
        alignment=256,
        # A merged target+draft pool deliberately accepts the extra full-attention
        # page slack; this matches Glm53FlashRecipe.max_padding_fraction.
        max_padding_fraction=float("inf"),
    ).bind(num_lcm_blocks)
    group_ids = tuple(
        split_recurrent_state_groups(text_config.paged_cache_layer_types)
    ) + (FULL_ATTENTION,)
    layer_types = tuple(
        FULL_ATTENTION if group_id == FULL_ATTENTION else LINEAR_ATTENTION
        for group_id in group_ids
    )
    arena = CacheArena(
        plan,
        "cuda",
        cache_group_specs=tuple(spec for spec, _ in groups),
        token_capacity=1024,
    )
    pool_options = Glm53FlashPoolOptions(
        index_kpool=text_config.index_kpool,
        tail_extra_slots=0,
        index_head_dim=text_config.index_head_dim,
        num_request_slots=10,
        dsa_layer_ids=(text_config.full_attention_layer_ids[0], 4),
    )
    pool = HybridGlm53FlashTokenToKVPool(
        arena=arena,
        model_dtype=torch.bfloat16,
        dtype=torch.float8_e4m3fn,
        quant_method=None,
        kv_lora_rank=text_config.kv_lora_rank,
        qk_rope_head_dim=text_config.qk_rope_head_dim,
        layer_num=len(layer_types),
        rank=0,
        pool_options=pool_options,
        layer_types=layer_types,
    )
    draft_pool = HybridGlm53FlashTokenToKVPool(
        arena=arena,
        model_dtype=torch.bfloat16,
        dtype=torch.float8_e4m3fn,
        quant_method=None,
        kv_lora_rank=text_config.kv_lora_rank,
        qk_rope_head_dim=text_config.qk_rope_head_dim,
        layer_num=1,
        rank=0,
        pool_options=pool_options,
        layer_types=(FULL_ATTENTION,),
        field_layer_offset=4,
    )

    assert pool.arena.plan.num_lcm_blocks == num_lcm_blocks
    assert pool.arena.runtime_contract is not None
    assert pool.arena.runtime_contract.token_capacity == 1024
    assert {
        spec.group_id: spec.transfer_policy for spec in pool.arena.cache_group_specs
    } == {
        FULL_ATTENTION: None,
        "linear_attention_0": None,
        "linear_attention_1": None,
        "linear_attention_2": None,
    }
    assert pool.arena.runtime_contract.group_page_counts == {
        group.group_id: group.page_count for group in plan.groups
    }
    arrays = {
        spec.group_id: np.zeros((1, 1), dtype=np.int32)
        for spec in pool.arena.runtime_contract.group_specs
    }
    forward_op = SimpleNamespace(block_tables_arrays=lambda: arrays)
    metadata = CacheBatchMetadata.from_forward_op(
        forward_op,
        device="cuda",
        contract=pool.arena.runtime_contract,
        num_requests=1,
    )
    tables = metadata.tables(active_forward_op=forward_op)
    assert tables[FULL_ATTENTION].shape == (1, 1)
    backing_ptr = pool.arena.buffer.untyped_storage().data_ptr()
    dsa_layer = text_config.full_attention_layer_ids[0]
    assert pool.kv_buffer[dsa_layer].untyped_storage().data_ptr() == backing_ptr

    index_k, tail_k, tail_gate = pool.get_kpool_buffers(dsa_layer)
    assert plan.field(f"layer.{dsa_layer}.index_k").group_id == FULL_ATTENTION
    assert tuple(index_k.shape[1:]) == (16, 132)
    assert tuple(tail_k.shape) == (10, 4, 128)
    assert tuple(tail_gate.shape) == (10, 4, 128)
    values, scales = pool.index_k_block_views(index_k)
    assert values.dtype == torch.float8_e4m3fn
    assert scales.dtype == torch.float32
    assert all(
        view.untyped_storage().data_ptr() == backing_ptr
        for view in (index_k, values, scales)
    )
    tail_ptr = tail_k.untyped_storage().data_ptr()
    assert tail_ptr == tail_gate.untyped_storage().data_ptr()
    assert tail_ptr != backing_ptr
    _, draft_tail_k, draft_tail_gate = draft_pool.get_kpool_buffers(0)
    assert draft_tail_k.untyped_storage().data_ptr() == tail_ptr
    assert draft_tail_gate.untyped_storage().data_ptr() == tail_ptr
    assert draft_tail_k.data_ptr() != tail_k.data_ptr()

    conv, recurrent = pool.get_state_buffers(0)
    assert tuple(conv.shape[1:]) == (3 * 64 * 128 // 4, 3)
    assert tuple(recurrent.shape[1:]) == (64 // 4, 128, 128)
    assert (
        conv.untyped_storage().data_ptr()
        == recurrent.untyped_storage().data_ptr()
        == backing_ptr
    )

    loc = torch.arange(600, device="cuda", dtype=torch.int32)
    cache_k_nope = torch.randn(
        600, 1, text_config.kv_lora_rank, device="cuda", dtype=torch.bfloat16
    )
    cache_k_rope = torch.empty(600, 1, 0, device="cuda", dtype=torch.bfloat16)
    layer = SimpleNamespace(layer_id=dsa_layer)
    pool.set_mla_kv_buffer(layer, loc, cache_k_nope, cache_k_rope)
    actual_nope, actual_rope = pool.get_mla_kv_buffer(
        layer, loc, dst_dtype=torch.bfloat16
    )
    expected_nope = cache_k_nope.to(torch.float8_e4m3fn).to(torch.bfloat16)
    assert torch.equal(actual_nope, expected_nope)
    assert actual_rope.numel() == 0

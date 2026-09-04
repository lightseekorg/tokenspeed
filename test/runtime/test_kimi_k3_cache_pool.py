from __future__ import annotations

import os
import sys

import pytest
import torch

# ``test/`` (for ``ci_system``) and the repo root (for ``test.runtime.*``
# absolute imports) both need to be importable when run_ci_suite executes this
# file as a standalone script.
_TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TEST_DIR)
sys.path.insert(0, os.path.dirname(_TEST_DIR))

from test.runtime.conftest import kimi_recipe, kimi_tp8_layout

from cache_pool_test_utils import make_arena
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")

from tokenspeed.runtime.configs.kimi_k3_config import KimiLinearConfig
from tokenspeed.runtime.layers.attention.kv_cache.hybrid_kda import (
    HybridKDATokenToKVPool,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
)


def test_kimi_k3_draft_mla_cache_retains_full_history() -> None:
    """DFlash2 SWA changes compute visibility, never draft KV retention."""
    num_draft_layers = 6
    recipe = kimi_recipe(draft_layers=num_draft_layers)

    assert recipe.group_ids[-num_draft_layers:] == (FULL_ATTENTION,) * num_draft_layers
    assert (
        recipe.layer_types[-num_draft_layers:] == (FULL_ATTENTION,) * num_draft_layers
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_kimi_k3_pool_binds_mla_and_kda_to_one_lcm_backing() -> None:
    text_config = KimiLinearConfig()
    num_lcm_blocks = 2
    recipe, groups, layout = kimi_tp8_layout(pd_enabled=True)
    plan = layout.bind(num_lcm_blocks)
    group_ids = recipe.target_group_ids
    layer_types = recipe.layer_types
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
    arena = make_arena(
        plan,
        cache_group_specs=tuple(spec for spec, _ in groups),
        token_capacity=1024,
    )
    pool = HybridKDATokenToKVPool(
        arena=arena,
        model_dtype=torch.bfloat16,
        dtype=torch.float8_e4m3fn,
        quant_method=None,
        kv_lora_rank=text_config.kv_lora_rank,
        qk_rope_head_dim=text_config.qk_rope_head_dim,
        layer_num=text_config.num_hidden_layers,
        rank=0,
        layer_types=layer_types,
    )

    assert pool.arena.plan.num_lcm_blocks == num_lcm_blocks
    assert pool.arena.runtime_contract is not None
    assert pool.arena.runtime_contract.token_capacity == 1024
    assert {
        spec.group_id: spec.transfer_policy for spec in pool.arena.cache_group_specs
    } == {
        FULL_ATTENTION: "full_suffix",
        f"{LINEAR_ATTENTION}_0": "latest_snapshot",
        f"{LINEAR_ATTENTION}_1": "latest_snapshot",
        f"{LINEAR_ATTENTION}_2": "latest_snapshot",
    }
    assert pool.arena.runtime_contract.group_page_counts == {
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
        == pool.arena.buffer.untyped_storage().data_ptr()
    )
    conv, recurrent = pool.get_state_buffers(state_layer)
    assert tuple(conv.shape[1:]) == conv_shape
    assert tuple(recurrent.shape[1:]) == recurrent_shape
    assert (
        conv.untyped_storage().data_ptr()
        == recurrent.untyped_storage().data_ptr()
        == pool.arena.buffer.untyped_storage().data_ptr()
    )


def test_kimi_k3_bf16_draft_uses_typed_view_over_fp8_target_arena() -> None:
    from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
    from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
    from tokenspeed.runtime.layers.attention.kv_cache.factory import (
        create_cache_arena,
        create_cache_pool,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.mla import MLATokenToKVPool
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import CachePoolSpec

    text_config = KimiLinearConfig()
    num_draft_layers = 5
    recipe, groups, layout = kimi_tp8_layout(draft_layers=num_draft_layers)
    plan = layout.bind(1)
    merged_spec = CachePoolSpec(
        family="kimi_k3",
        memory_plan=plan,
        layer_types=recipe.layer_types,
        cache_group_specs=tuple(spec for spec, _ in groups),
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
    )

    mla_spec = MLAConfig(
        backend_name="mla",
        num_attention_heads=64,
        num_kv_heads=64,
        attn_tp_size=8,
        head_dim=192,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        scaling=1.0,
        kv_cache_dim=576,
    )
    common_config = dict(
        device="cpu",
        dtype=torch.bfloat16,
        context_len=1024,
        max_bs=1,
        prefix_granularity=plan.prefix_granularity,
        kv_cache_quant_method="none",
        components=(mla_spec,),
    )
    target_config = AttnConfig(
        kv_cache_dtype=torch.float8_e4m3fn,
        **common_config,
    )
    draft_config = AttnConfig(
        kv_cache_dtype=torch.bfloat16,
        is_draft=True,
        **common_config,
    )
    arena = create_cache_arena(
        merged_spec, device=target_config.device, enable_memory_saver=False
    )
    target_pool = create_cache_pool(
        target_spec,
        target_config,
        arena,
        num_layers=text_config.num_hidden_layers,
        rank=0,
    )
    draft_pool = create_cache_pool(
        draft_spec,
        draft_config,
        arena,
        num_layers=num_draft_layers,
        rank=0,
        field_layer_offset=text_config.num_hidden_layers,
    )

    assert isinstance(target_pool, HybridKDATokenToKVPool)
    assert type(draft_pool) is MLATokenToKVPool
    assert draft_pool.arena is target_pool.arena
    assert draft_pool.arena.buffer is target_pool.arena.buffer
    assert draft_pool.arena.runtime_contract is target_pool.arena.runtime_contract
    target_cache = target_pool.get_key_buffer(text_config.full_attention_layer_ids[0])
    assert target_cache.dtype == torch.float8_e4m3fn
    assert draft_pool.get_key_buffer(0).dtype == torch.bfloat16
    assert (
        draft_pool.kv_buffer[0].data_ptr()
        == target_pool.arena.field("layer.93.latent_kv").data_ptr()
    )
    assert arena.field_ids() == {
        field.field_id for field in merged_spec.memory_plan.fields
    }
    full_attention_segments = set(arena.block_byte_segments(FULL_ATTENTION, [1]))
    for global_layer_id in range(93, 98):
        field_id = f"layer.{global_layer_id}.latent_kv"
        field = plan.field(field_id)
        assert (
            arena.field_block_byte_offset(field_id, 1),
            field.payload_bytes,
        ) in full_attention_segments

    target_layer = text_config.full_attention_layer_ids[0]
    target_before = target_pool.kv_buffer[target_layer].clone()
    draft_pool.kv_buffer[0][1].fill_(1)
    assert torch.equal(target_pool.kv_buffer[target_layer], target_before)
    # Both views name the one arena, so a clear through either zeros it.
    draft_pool.clear_kv_buffers()
    assert not torch.count_nonzero(draft_pool.kv_buffer[0])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

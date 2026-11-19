from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

# ``test/`` (for ``ci_system``) and the repo root (for ``test.runtime.*``
# absolute imports) both need to be importable when run_ci_suite executes this
# file as a standalone script.
_TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TEST_DIR)
sys.path.insert(0, os.path.dirname(_TEST_DIR))

from test.runtime.cache_pool_test_utils import make_arena
from test.runtime.conftest import kimi_recipe, kimi_tp8_layout

from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")

from tokenspeed.runtime.configs.kimi_k3_config import KimiLinearConfig
from tokenspeed.runtime.layers.attention.kv_cache.hybrid_kda import (
    HybridKDATokenToKVPool,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.kda import kda_state_fields
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    pack,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    CacheGroupSpec,
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
        FULL_ATTENTION: num_lcm_blocks * 16 + 1,
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
        max_resident_requests=1,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "history_capacity, initial_length", [(8, 2), (16, 0), (16, 10)]
)
@pytest.mark.parametrize("recurrent_state_dtype", [torch.float32, torch.bfloat16])
def test_kimi_k3_checkpoint_survives_copy_and_reuse(
    history_capacity: int, initial_length: int, recurrent_state_dtype: torch.dtype
) -> None:
    """Whole-block copies preserve the endpoint, including accepted history."""
    device = "cuda"
    heads, dim = 2, 8
    groups = (
        (
            CacheGroupSpec(
                group_id=FULL_ATTENTION,
                attn_kind="full_history",
                rows_per_page=128,
                entry_stride_tokens=1,
                replayable=False,
            ),
            (CacheFieldSpec("layer.0.latent_kv", "slot.0", (128, 1, 576), "bfloat16"),),
        ),
        (
            CacheGroupSpec(
                group_id=LINEAR_ATTENTION, attn_kind="state", replayable=False
            ),
            kda_state_fields(
                1,
                "slot.0",
                conv_shape=(3 * heads * dim, 3),
                recurrent_shape=(heads, dim, dim),
                recurrent_state_dtype=recurrent_state_dtype,
                history_capacity=history_capacity,
            ),
        ),
    )
    plan = pack(
        groups,
        prefix_granularity=128,
        cache_blocks_per_lcm_block={FULL_ATTENTION: 1, LINEAR_ATTENTION: 1},
        max_padding_fraction=float("inf"),
    ).bind(4)
    arena = make_arena(
        plan, device=device, cache_group_specs=tuple(spec for spec, _ in groups)
    )
    pool = HybridKDATokenToKVPool(
        arena=arena,
        model_dtype=torch.bfloat16,
        dtype=torch.bfloat16,
        quant_method=None,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        layer_num=2,
        rank=0,
        layer_types=(FULL_ATTENTION, LINEAR_ATTENTION),
    )
    names = (
        "conv_state",
        "recurrent_state",
        "replay_correction",
        "replay_key",
        "replay_decay",
        "replay_length",
    )
    fields = dict(
        zip(
            names,
            (*pool.get_state_buffers(1), *pool.get_retained_history_fields(1)),
            strict=True,
        )
    )
    generator = torch.Generator(device=device).manual_seed(8123)
    for name, tensor in fields.items():
        if name == "replay_length":
            tensor[1:4, 0] = torch.tensor([initial_length, 0, 1], device=device)
        else:
            values = torch.randn(
                tensor[1:4].shape,
                dtype=tensor.dtype,
                device=device,
                generator=generator,
            )
            tensor[1:4].copy_(
                values.sigmoid() if name == "replay_decay" else values / 8
            )
    if history_capacity:
        keys = fields["replay_key"][1:4]
        keys.copy_(torch.nn.functional.normalize(keys, dim=-1))
        # A zero-length checkpoint must ignore stale history bytes.
        for name in ("replay_correction", "replay_key", "replay_decay"):
            fields[name][2].fill_(float("nan"))

    def endpoint(block: int) -> torch.Tensor:
        value = fields["recurrent_state"][block].float().clone()
        length = int(fields["replay_length"][block, 0]) if history_capacity else 0
        for token in range(length):
            value = (
                value * fields["replay_decay"][block, :, token, None, :]
                + fields["replay_correction"][block, :, token, :, None]
                * fields["replay_key"][block, :, token, None, :]
            )
        return value

    # State blocks 1..3 occupy parents 0..2. MLA block 4 owns parent 3.
    latent = pool.get_key_buffer(0)[4]
    latent.fill_(0.25)
    before = {name: tensor.clone() for name, tensor in fields.items()}
    endpoints = (endpoint(1), endpoint(2))
    arena.copy_blocks({LINEAR_ATTENTION: np.asarray([[1, 2], [2, 3]], dtype=np.int64)})
    for name, tensor in fields.items():
        for destination, source in ((1, 1), (2, 1), (3, 2)):
            torch.testing.assert_close(
                tensor[destination],
                before[name][source],
                atol=0,
                rtol=0,
                equal_nan=True,
            )
    torch.testing.assert_close(endpoint(2), endpoints[0], atol=0, rtol=0)
    torch.testing.assert_close(endpoint(3), endpoints[1], atol=0, rtol=0)

    if history_capacity:
        # Exercise the storage contract through real pool views. Kernel writes
        # and accepted-length publication are integrated in later stages.
        expected = endpoints[0]
        for accepted in (2, 3):
            # The second round starts after two accepted rows, overwriting the
            # previous round's two rejected rows. It does not advance by four.
            start = int(fields["replay_length"][2, 0])
            candidates = {}
            for name in ("replay_correction", "replay_key", "replay_decay"):
                prefix = fields[name][2, :, :start].clone()
                values = torch.randn(
                    (heads, 4, dim), device=device, generator=generator
                )
                if name == "replay_key":
                    values = torch.nn.functional.normalize(values, dim=-1)
                elif name == "replay_decay":
                    values = values.sigmoid()
                candidates[name] = values
                # Poison rejected rows to detect accidental publication or gaps.
                values[:, accepted:].fill_(float("nan"))
                fields[name][2, :, start : start + 4].copy_(values)
                torch.testing.assert_close(
                    fields[name][2, :, :start], prefix, atol=0, rtol=0
                )
                torch.testing.assert_close(
                    fields[name][1], before[name][1], atol=0, rtol=0
                )
            # Filling the tail changes neither the accepted endpoint nor base.
            torch.testing.assert_close(endpoint(2), expected, atol=0, rtol=0)
            torch.testing.assert_close(
                fields["recurrent_state"][2],
                before["recurrent_state"][1],
                atol=0,
                rtol=0,
            )
            for token in range(accepted):
                expected = (
                    expected * candidates["replay_decay"][:, token, None, :]
                    + candidates["replay_correction"][:, token, :, None]
                    * candidates["replay_key"][:, token, None, :]
                )
            # Publication changes only the length, with no record-copy step.
            fields["replay_length"][2, 0] = start + accepted
            torch.testing.assert_close(endpoint(2), expected, atol=0, rtol=0)

    pool.zero_new_blocks({LINEAR_ATTENTION: np.asarray([2], dtype=np.int64)})
    for name, tensor in fields.items():
        torch.testing.assert_close(
            tensor[2], torch.zeros_like(tensor[2]), atol=0, rtol=0
        )
        torch.testing.assert_close(tensor[1], before[name][1], atol=0, rtol=0)
        torch.testing.assert_close(
            tensor[3], before[name][2], atol=0, rtol=0, equal_nan=True
        )
    torch.testing.assert_close(
        endpoint(2), torch.zeros_like(endpoints[0]), atol=0, rtol=0
    )
    torch.testing.assert_close(latent, torch.full_like(latent, 0.25), atol=0, rtol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

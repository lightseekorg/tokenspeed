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
@pytest.mark.parametrize("recurrent_state_dtype", [torch.float32, torch.bfloat16])
def test_kimi_k3_checkpoint_survives_copy_and_reuse(
    recurrent_state_dtype: torch.dtype,
) -> None:
    """Copy and zero a dense checkpoint without touching an adjacent MLA block."""
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
    fields = {
        name: pool.get_component(1, name) for name in ("conv_state", "recurrent_state")
    }
    for index, tensor in enumerate(fields.values(), 1):
        tensor[1].fill_(index)
        tensor[2].fill_(index + 3)

    # State blocks 1..3 occupy parents 0..2. MLA block 4 owns parent 3.
    latent = pool.get_key_buffer(0)[4]
    latent.fill_(0.25)
    before = {name: tensor.clone() for name, tensor in fields.items()}
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

    pool.zero_new_blocks({LINEAR_ATTENTION: np.asarray([2], dtype=np.int64)})
    for name, tensor in fields.items():
        torch.testing.assert_close(
            tensor[2], torch.zeros_like(tensor[2]), atol=0, rtol=0
        )
        torch.testing.assert_close(tensor[1], before[name][1], atol=0, rtol=0)
        torch.testing.assert_close(
            tensor[3], before[name][2], atol=0, rtol=0, equal_nan=True
        )
    torch.testing.assert_close(latent, torch.full_like(latent, 0.25), atol=0, rtol=0)


def test_kimi_k3_gqa_block_draft_kv_pages_join_the_full_attention_group() -> None:
    """A GQA block draft's K/V pages sit in the MLA group's page-id space at
    P=128, beside the target's latents, without changing the packing."""
    from tokenspeed.runtime.layers.attention.kv_cache.factory import (
        create_cache_arena,
        create_cache_pool,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.mha import MHATokenToKVPool
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import CachePoolSpec

    text_config = KimiLinearConfig()
    num_target_layers = text_config.num_hidden_layers
    num_draft_layers = 5
    _, _, target_only = kimi_tp8_layout()
    recipe, groups, layout = kimi_tp8_layout(
        draft_layers=num_draft_layers, draft_attn="mha"
    )

    assert layout.group_packing == target_only.group_packing
    # One plane per MLA layer plus a K and a V plane per draft layer; the KDA
    # state still rides inside the MLA planes.
    assert (
        len(layout.plane_bytes) == len(target_only.plane_bytes) + 2 * num_draft_layers
    )
    full_fields = {
        field.field_id: field
        for spec, fields in groups
        if spec.group_id == FULL_ATTENTION
        for field in fields
    }
    for layer_id in range(num_target_layers, num_target_layers + num_draft_layers):
        for kind in ("k", "v"):
            # P=128 pages of 16 KV heads / 8 ranks, head_dim 64.
            assert full_fields[f"layer.{layer_id}.{kind}"].shape == (128, 2, 64)

    plan = layout.bind(1)
    merged_spec = CachePoolSpec(
        family="kimi_k3",
        memory_plan=plan,
        layer_types=recipe.layer_types,
        cache_group_specs=tuple(spec for spec, _ in groups),
        token_capacity=128,
    )
    arena = create_cache_arena(merged_spec, device="cpu", enable_memory_saver=False)
    target_pool = create_cache_pool(
        merged_spec.layer_view(first_layer=0, num_layers=num_target_layers),
        recipe.attn_config,
        arena,
        num_layers=num_target_layers,
        rank=0,
    )
    draft_pool = create_cache_pool(
        merged_spec.layer_view(
            first_layer=num_target_layers, num_layers=num_draft_layers, family="mha"
        ),
        recipe.draft_attn_config,
        arena,
        num_layers=num_draft_layers,
        rank=0,
        field_layer_offset=num_target_layers,
    )

    assert type(draft_pool) is MHATokenToKVPool
    assert draft_pool.arena.buffer is target_pool.arena.buffer
    assert draft_pool.get_key_buffer(0).dtype == torch.bfloat16
    target_layer = text_config.full_attention_layer_ids[0]
    target_before = target_pool.kv_buffer[target_layer].clone()
    draft_pool.get_key_buffer(0).fill_(1)
    draft_pool.get_value_buffer(0).fill_(1)
    assert torch.equal(target_pool.kv_buffer[target_layer], target_before)


def _kvp8(recipe):
    """``recipe`` with its target page-sharded over eight KVP ranks."""
    from dataclasses import replace

    config = recipe.attn_config
    recipe.attn_config = replace(
        config,
        device="cuda",
        kvp_size=8,
        kvp_group=tuple(range(8)),
        components=(
            replace(config.components[0], backend_name="flashmla"),
            *config.components[1:],
        ),
    )
    return recipe


def _pool_spec(recipe):
    from test.runtime.conftest import kimi_layout

    from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import CachePoolSpec

    groups, layout = kimi_layout(recipe)
    return CachePoolSpec(
        family="kimi_k3",
        memory_plan=layout.bind(1),
        layer_types=recipe.layer_types,
        cache_group_specs=tuple(spec for spec, _ in groups),
        token_capacity=128,
    )


def test_kimi_k3_sliding_gqa_block_draft_owns_an_unsharded_window_group() -> None:
    """A GQA block draft whose every layer slides keeps its K/V pages in a
    window group of its own, inside the target's MLA planes, and KVP shards
    the MLA group but not that window."""
    from tokenspeed.runtime.layers.attention import registry
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.kimi_k3 import (
        DRAFT_SLIDING_ATTENTION,
    )

    num_target_layers = KimiLinearConfig().num_hidden_layers
    num_draft_layers = 5
    _, _, target = kimi_tp8_layout()
    recipe, groups, layout = kimi_tp8_layout(
        draft_layers=num_draft_layers, draft_attn="mha", draft_window=2048
    )

    window = next(
        spec for spec, _ in groups if spec.group_id == DRAFT_SLIDING_ATTENTION
    )
    assert window.attn_kind == "sliding_window"
    assert window.sliding_window_tokens == 2048
    assert not window.replayable
    field_ids = {
        spec.group_id: {field.field_id for field in fields} for spec, fields in groups
    }
    draft_field_ids = {
        f"layer.{layer_id}.{kind}"
        for layer_id in range(num_target_layers, num_target_layers + num_draft_layers)
        for kind in ("k", "v")
    }
    assert field_ids[DRAFT_SLIDING_ATTENTION] == draft_field_ids
    assert not field_ids[FULL_ATTENTION] & draft_field_ids
    # The draft adds no bytes: the planes and every target group's packing are
    # the draft-less target's, and each draft layer takes two MLA planes.
    assert layout.plane_bytes == target.plane_bytes
    assert layout.lcm_block_bytes == target.lcm_block_bytes
    packing = dict(layout.group_packing)
    draft_packing = packing.pop(DRAFT_SLIDING_ATTENTION)
    assert packing == dict(target.group_packing)
    draft_fields = next(
        fields for spec, fields in groups if spec.group_id == DRAFT_SLIDING_ATTENTION
    )
    assert [field.plane_id for field in draft_fields] == [
        f"slot.{index}" for index in range(2 * num_draft_layers)
    ]
    plane_bytes = dict(layout.plane_bytes)
    assert all(
        draft_packing * field.payload_bytes <= plane_bytes[field.plane_id]
        for field in draft_fields
    )

    spec = _pool_spec(_kvp8(recipe))
    shard_counts = {
        group.group_id: group.shard_count for group in spec.cache_group_specs
    }
    assert shard_counts[FULL_ATTENTION] == 8
    assert shard_counts[DRAFT_SLIDING_ATTENTION] == 1
    registry._validate_hybrid_kvp_cache(
        spec, kvp_size=8, num_target_layers=num_target_layers
    )
    registry._validate_unsharded_draft_storage(
        spec, num_target_layers=num_target_layers
    )

    # Without a window the draft stays in the MLA group, which KVP shards.
    shared_spec = _pool_spec(
        _kvp8(kimi_recipe(draft_layers=num_draft_layers, draft_attn="mha"))
    )
    with pytest.raises(ValueError, match="page-shards cache group 'full_attention'"):
        registry._validate_unsharded_draft_storage(
            shared_spec, num_target_layers=num_target_layers
        )


def test_kimi_k31_sliding_gqa_block_draft_pads_kv_pages_inside_mla_planes() -> None:
    """At attention TP 1 a draft K page is narrower than its share of an MLA
    plane: K and V are strided 4-D views of two planes with one page stride,
    and the parent is the draft-less Kimi-K3.1 target's."""
    from test.runtime.conftest import kimi_layout

    from tokenspeed.runtime.layers.attention.kv_cache.recipes.kimi_k3 import (
        DRAFT_SLIDING_ATTENTION,
    )

    num_draft_layers = 5
    target_recipe = kimi_recipe(
        tp_size=1, dsa_index_k_format="fp8_scaled", mamba_ssm_dtype="bfloat16"
    )
    _, target = kimi_layout(target_recipe)
    recipe = kimi_recipe(
        tp_size=1,
        dsa_index_k_format="fp8_scaled",
        mamba_ssm_dtype="bfloat16",
        draft_layers=num_draft_layers,
        draft_attn="mha",
        draft_window=2048,
    )
    _, layout = kimi_layout(recipe)
    assert layout.plane_bytes == target.plane_bytes
    assert layout.lcm_block_bytes == target.lcm_block_bytes

    assert len(layout.plane_bytes) == 48
    assert layout.lcm_block_bytes == 100_048_896

    plan = layout.bind(3)
    draft_packing = dict(layout.group_packing)[DRAFT_SLIDING_ATTENTION]
    assert draft_packing == 12
    draft_fields = [
        field for field in plan.fields if field.group_id == DRAFT_SLIDING_ATTENTION
    ]
    assert len(draft_fields) == 2 * num_draft_layers
    plane_bytes = dict(layout.plane_bytes)
    for index, field in enumerate(draft_fields):
        assert field.plane_id == f"slot.{index}"
        assert field.page_stride_bytes == plane_bytes[field.plane_id] // draft_packing
        assert field.payload_bytes == 262_144
        assert field.page_stride_bytes == 282_624

    arena = make_arena(plan)
    for field in draft_fields:
        view = arena.field(field.field_id)
        assert view.shape[0] >= 3 * draft_packing
        assert view.shape[1:] == field.shape
        assert view.stride(0) * view.element_size() == field.page_stride_bytes


def test_drafter_keeps_fused_kv_writes_for_an_unsharded_draft_group() -> None:
    """Under KVP the fused context-KV scatter stays on when the draft's own
    group is unsharded; it is asked before the layers' groups are bound."""
    from types import SimpleNamespace

    from tokenspeed.runtime.execution.drafter.dflash import DFlash

    def drafter(kvp_size: int, draft_group: str):
        specs = (
            SimpleNamespace(group_id="full_attention", shard_count=kvp_size),
            SimpleNamespace(group_id="draft_sliding_attention", shard_count=1),
        )
        pool = SimpleNamespace(
            arena=SimpleNamespace(cache_group_specs=specs),
            history_group_by_layer=lambda: dict.fromkeys(range(5), draft_group),
        )
        runner = SimpleNamespace(
            mapping=SimpleNamespace(attn=SimpleNamespace(kvp_size=kvp_size))
        )
        return SimpleNamespace(draft_model_runner=runner, token_to_kv_pool=pool)

    assert not DFlash._draft_cache_page_sharded(drafter(8, "draft_sliding_attention"))
    assert DFlash._draft_cache_page_sharded(drafter(8, "full_attention"))
    assert not DFlash._draft_cache_page_sharded(drafter(1, "full_attention"))


def test_kimi_k3_and_k31_sliding_mla_draft_uses_unsharded_group() -> None:
    """An all-sliding MLA block draft (DFlash2) owns DRAFT_SLIDING_ATTENTION,
    stays unsharded under target KVP=8, and gets contiguous latent planes
    with exact page stride across TP sizes and FP8/BF16 dtypes."""
    from dataclasses import replace
    from test.runtime.conftest import kimi_layout

    from tokenspeed.runtime.layers.attention.kv_cache.factory import (
        create_cache_arena,
        create_cache_pool,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.kimi_k3 import (
        DRAFT_SLIDING_ATTENTION,
        FULL_ATTENTION,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import (
        CachePoolSpec,
    )

    num_draft_layers = 5
    cases = [
        # Kimi-K3 at TP=8 with KVP=8, BF16 draft over FP8 target.
        dict(
            tp_size=8,
            kvp_size=8,
            dsa_index_k_format=None,
            kv_cache_dtype=torch.float8_e4m3fn,
        ),
        # Kimi-K3.1 at TP=8 with KVP=8, BF16 draft over FP8 target.
        dict(
            tp_size=8,
            kvp_size=8,
            dsa_index_k_format="fp8_scaled",
            kv_cache_dtype=torch.float8_e4m3fn,
        ),
        # TP=2 odd mla_packing rounds up so BF16 draft fits exact_page_stride.
        dict(
            tp_size=2,
            kvp_size=1,
            dsa_index_k_format="fp8_scaled",
            kv_cache_dtype=torch.float8_e4m3fn,
        ),
        # Matching BF16 target and draft keeps packing parity.
        dict(
            tp_size=8,
            kvp_size=8,
            dsa_index_k_format="fp8_scaled",
            kv_cache_dtype=torch.bfloat16,
        ),
    ]
    for case in cases:
        recipe = kimi_recipe(
            tp_size=case["tp_size"],
            dsa_index_k_format=case["dsa_index_k_format"],
            kv_cache_dtype=case["kv_cache_dtype"],
            mamba_ssm_dtype="bfloat16",
            draft_layers=num_draft_layers,
            draft_attn="mla",
            draft_window=1024,
            speculative_algorithm="DFLASH",
        )
        if case["kvp_size"] > 1:
            target_softmax = replace(
                recipe.attn_config.components[0],
                backend_name="tokenspeed_mla",
            )
            recipe.attn_config = replace(
                recipe.attn_config,
                kvp_size=case["kvp_size"],
                kvp_group=tuple(range(case["kvp_size"])),
                components=(target_softmax, recipe.attn_config.components[1]),
            )
        target_less = kimi_recipe(
            tp_size=case["tp_size"],
            dsa_index_k_format=case["dsa_index_k_format"],
            kv_cache_dtype=case["kv_cache_dtype"],
            mamba_ssm_dtype="bfloat16",
        )
        _, target_layout = kimi_layout(target_less)

        assert recipe.draft_window_tokens == 1024
        groups, layout = kimi_layout(recipe)
        specs = {spec.group_id: spec for spec, _ in groups}
        assert set(specs) >= {FULL_ATTENTION, DRAFT_SLIDING_ATTENTION}
        assert specs[FULL_ATTENTION].shard_count == case["kvp_size"]
        draft_spec = specs[DRAFT_SLIDING_ATTENTION]
        assert draft_spec.attn_kind == "sliding_window"
        assert draft_spec.sliding_window_tokens == 1024
        assert not draft_spec.replayable
        assert draft_spec.shard_count == 1
        assert draft_spec.block_granularity == layout.prefix_granularity

        num_target_mla = sum(
            group_id == FULL_ATTENTION for group_id in recipe.target_group_ids
        )
        expected_index_planes = (
            num_target_mla if case["dsa_index_k_format"] is not None else 0
        )
        assert recipe.num_index_k_planes == expected_index_planes
        assert len(layout.plane_bytes) == len(target_layout.plane_bytes)

        draft_fields = [
            field
            for field in layout.fields
            if field.group_id == DRAFT_SLIDING_ATTENTION
        ]
        assert len(draft_fields) == num_draft_layers
        packing = dict(layout.group_packing)
        plane_bytes = dict(layout.plane_bytes)
        for index, field in enumerate(draft_fields):
            field_spec = recipe.fields_for_layer(
                recipe.num_target_layers + index, DRAFT_SLIDING_ATTENTION, index
            )[0]
            assert field_spec.exact_page_stride
            assert field.plane_id == f"slot.{index}"
            assert field.page_stride_bytes == field.payload_bytes
            assert (
                packing[DRAFT_SLIDING_ATTENTION] * field.payload_bytes
                == plane_bytes[field.plane_id]
            )
            assert ".index_k" not in field.field_id

        recipe.check_layout(layout)
        plan = layout.bind(2)
        pool_spec = CachePoolSpec(
            family="kimi_k3",
            memory_plan=plan,
            layer_types=recipe.layer_types,
            cache_group_specs=tuple(spec for spec, _ in groups),
            token_capacity=2 * packing[FULL_ATTENTION] * layout.prefix_granularity,
        )
        arena = create_cache_arena(pool_spec, device="cpu", enable_memory_saver=False)
        draft_pool = create_cache_pool(
            pool_spec.layer_view(
                first_layer=recipe.num_target_layers,
                num_layers=num_draft_layers,
                family="mla",
            ),
            recipe.draft_attn_config,
            arena,
            num_layers=num_draft_layers,
            rank=0,
            field_layer_offset=recipe.num_target_layers,
        )
        for local_layer in range(num_draft_layers):
            layer_view = arena.field(
                f"layer.{recipe.num_target_layers + local_layer}.latent_kv"
            )
            assert layer_view.dtype == torch.bfloat16
            assert layer_view.shape[1:] == (1, 576)
            assert layer_view.shape[0] >= (
                2 * packing[DRAFT_SLIDING_ATTENTION] * layout.prefix_granularity
            )
            assert layer_view.is_contiguous()
            pool_buf = draft_pool.get_key_buffer(local_layer)
            assert pool_buf.dtype == torch.bfloat16
            assert pool_buf.shape[-1] == 576


def test_kimi_k3_sliding_mla_draft_requires_all_sliding_without_sink() -> None:
    """Mixed full/sliding layers, non-zero sink, non-block drafts, or DSA drafts
    keep draft_window_tokens as None instead of forming a sliding group."""
    from dataclasses import replace
    from types import SimpleNamespace

    from tokenspeed.runtime.layers.attention.configs.base import (
        get_dflash_sliding_window_tokens,
    )
    from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig
    from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.kimi_k3 import (
        KimiK3Recipe,
    )

    assert (
        get_dflash_sliding_window_tokens(
            SimpleNamespace(layer_types=["sliding_attention"], sliding_window=1),
            num_layers=1,
        )
        == 1
    )
    with pytest.raises(ValueError, match="sliding_window must be positive"):
        get_dflash_sliding_window_tokens(
            SimpleNamespace(layer_types=["sliding_attention"], sliding_window=0),
            num_layers=1,
        )

    base = kimi_recipe(
        tp_size=8,
        dsa_index_k_format="fp8_scaled",
        draft_layers=5,
        draft_attn="mla",
        draft_window=1024,
        speculative_algorithm="DFLASH",
    )
    assert base.draft_window_tokens == 1024

    def variant(*, draft_model_config=None, draft_attn_config=None) -> KimiK3Recipe:
        return KimiK3Recipe(
            server_args=base.server_args,
            model_config=base.model_config,
            attn_config=base.attn_config,
            draft_model_config=(
                base.draft_model_config
                if draft_model_config is None
                else draft_model_config
            ),
            draft_attn_config=(
                base.draft_attn_config
                if draft_attn_config is None
                else draft_attn_config
            ),
            cache_budget_bytes=base.cache_budget_bytes,
            probe_batch_rows=None,
            decode_input_tokens=base.decode_input_tokens,
            overlap_schedule_depth=base.overlap_schedule_depth,
        )

    mixed = variant(
        draft_model_config=SimpleNamespace(
            num_attention_layers=5,
            hf_config=SimpleNamespace(
                layer_types=["sliding_attention"] * 4 + ["full_attention"],
                sliding_window=1024,
            ),
        )
    )
    assert mixed.draft_window_tokens is None

    top_sink = variant(
        draft_model_config=SimpleNamespace(
            num_attention_layers=5,
            hf_config=SimpleNamespace(
                layer_types=["sliding_attention"] * 5,
                sliding_window=1024,
                sliding_window_sink=4,
            ),
        )
    )
    assert top_sink.draft_window_tokens is None

    export_sink = variant(
        draft_model_config=SimpleNamespace(
            num_attention_layers=5,
            hf_config=SimpleNamespace(
                layer_types=["sliding_attention"] * 5,
                sliding_window=1024,
                dflash_config={"context_sink_size": 2},
            ),
        )
    )
    assert export_sink.draft_window_tokens is None

    non_block = variant(
        draft_attn_config=replace(base.draft_attn_config, draft_block_decode=False)
    )
    assert non_block.draft_window_tokens is None

    target_dsa = base.attn_config.component(MLAConfig)
    assert isinstance(target_dsa, DSAConfig)
    dsa_draft = variant(
        draft_attn_config=replace(base.draft_attn_config, components=(target_dsa,))
    )
    assert dsa_draft.draft_window_tokens is None


def test_kimi_k3_sliding_mla_check_layout_allows_more_draft_than_target_planes() -> (
    None
):
    """When draft_layers exceeds target MLA layers, check_layout accounts for
    max(num_mla_layers, num_draft_planes) and rejects mismatched plane counts."""
    from test.runtime.conftest import kimi_layout

    from tokenspeed.runtime.configs.kimi_k3_config import KimiLinearConfig
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import CacheLayout

    # Toy Kimi-K3 config with 4 total layers -> 1 FULL_ATTENTION target layer
    # and 3 KDA layers, paired with 3 sliding MLA draft layers (3 > 1).
    small_text = KimiLinearConfig(
        num_hidden_layers=4,
        linear_attn_config={
            "full_attn_layers": [4],
            "kda_layers": [1, 2, 3],
            "num_heads": 96,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
        },
    )
    recipe = kimi_recipe(
        text_config=small_text,
        tp_size=8,
        draft_layers=3,
        draft_attn="mla",
        draft_window=1024,
        speculative_algorithm="DFLASH",
    )
    _, layout = kimi_layout(recipe)
    assert len(layout.plane_bytes) == 3
    recipe.check_layout(layout)

    bad_layout = CacheLayout(
        prefix_granularity=layout.prefix_granularity,
        lcm_block_bytes=layout.lcm_block_bytes,
        group_packing=layout.group_packing,
        plane_bytes=layout.plane_bytes[:-1],
        fields=layout.fields,
    )
    with pytest.raises(ValueError, match="slot planes"):
        recipe.check_layout(bad_layout)


def test_kimi_k31_sliding_mla_long_prefix_and_snapshot_contract() -> None:
    """Metadata and group contracts keep FULL_ATTENTION (shard_count=8) and
    DRAFT_SLIDING_ATTENTION (shard_count=1, sliding_window=1024) isolated
    across >1024-token prefix windows with leading null slots."""
    from dataclasses import replace
    from test.runtime.conftest import cache_metadata_for, kimi_layout

    from tokenspeed.runtime.layers.attention.kv_cache.factory import (
        create_cache_arena,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.kimi_k3 import (
        DRAFT_SLIDING_ATTENTION,
        FULL_ATTENTION,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import (
        CachePoolSpec,
    )

    recipe = kimi_recipe(
        tp_size=8,
        dsa_index_k_format="fp8_scaled",
        mamba_ssm_dtype="bfloat16",
        draft_layers=5,
        draft_attn="mla",
        draft_window=1024,
        speculative_algorithm="DFLASH",
    )
    target_softmax = replace(
        recipe.attn_config.components[0],
        backend_name="tokenspeed_mla",
    )
    recipe.attn_config = replace(
        recipe.attn_config,
        kvp_size=8,
        kvp_group=tuple(range(8)),
        components=(target_softmax, recipe.attn_config.components[1]),
    )
    groups, layout = kimi_layout(recipe)
    plan = layout.bind(4)
    pool_spec = CachePoolSpec(
        family="kimi_k3",
        memory_plan=plan,
        layer_types=recipe.layer_types,
        cache_group_specs=tuple(spec for spec, _ in groups),
        token_capacity=4 * dict(layout.group_packing)[FULL_ATTENTION] * 128,
    )
    arena = create_cache_arena(pool_spec, device="cpu", enable_memory_saver=False)
    contract = arena.runtime_contract

    # 16 pages = 2048 tokens: prefix pages before the 1024-token window expire
    # to slot 0 while the trailing 8 pages retain live page ids, preserving
    # absolute indexing across target full_attention and unsharded draft.
    full_row = list(range(1, 17))
    draft_row = [0] * 8 + list(range(1, 9))
    metadata, active = cache_metadata_for(
        contract,
        {FULL_ATTENTION: [full_row], DRAFT_SLIDING_ATTENTION: [draft_row]},
        device="cpu",
    )
    tables = dict(metadata.tables(active_forward_op=active))
    assert tables[FULL_ATTENTION].shape == (1, 16)
    assert tables[DRAFT_SLIDING_ATTENTION].tolist() == [draft_row]

    group_specs = {spec.group_id: spec for spec in contract.group_specs}
    assert group_specs[FULL_ATTENTION].shard_count == 8
    assert group_specs[DRAFT_SLIDING_ATTENTION].shard_count == 1
    assert group_specs[DRAFT_SLIDING_ATTENTION].sliding_window_tokens == 1024
    assert not group_specs[DRAFT_SLIDING_ATTENTION].replayable


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

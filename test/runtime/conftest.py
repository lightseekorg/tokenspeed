"""Shared helpers for Kimi-K3 paged cache runtime tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

TP8_PHYSICAL_PAGE_BYTES = 884_736
TP8_PHYSICAL_SLOTS = 24
TP8_PAGE_SET_BYTES = TP8_PHYSICAL_SLOTS * TP8_PHYSICAL_PAGE_BYTES
KIMI_GROUP_IDS = (
    "full_attention",
    "linear_attention_0",
    "linear_attention_1",
    "linear_attention_2",
)
KIMI_STATE_GROUPS = KIMI_GROUP_IDS[1:]
MLA_KV_LORA_RANK = 512
MLA_QK_ROPE_DIM = 64
MLA_LATENT_DIM = MLA_KV_LORA_RANK + MLA_QK_ROPE_DIM


def _poison(shape, device="cuda", dtype=torch.int32):
    return torch.full(shape, 987_654, device=device, dtype=dtype)


def kimi_recipe(
    *,
    text_config=None,
    tp_size: int = 8,
    draft_layers: int = 0,
    pd_enabled: bool = False,
    max_bs: int = 1,
    max_scheduled_tokens: int = 128,
    context_len: int = 4096,
    decode_input_tokens: int = 1,
    overlap_schedule_depth: int = 0,
    speculative_algorithm: str | None = None,
    speculative_num_draft_tokens: int = 1,
    kv_cache_dtype: torch.dtype = torch.float8_e4m3fn,
):
    """A Kimi-K3 recipe over the reference config, with tiny scheduler limits."""
    from types import SimpleNamespace

    from tokenspeed.runtime.configs.kimi_k3_config import KimiLinearConfig
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.kimi_k3 import (
        KimiK3Recipe,
    )

    text_config = text_config if text_config is not None else KimiLinearConfig()
    attn_config = SimpleNamespace(
        attn_tp_size=tp_size,
        dtype=torch.bfloat16,
        kv_cache_dtype=kv_cache_dtype,
        kv_cache_quant_method=None,
        kv_lora_rank=text_config.kv_lora_rank,
        qk_rope_head_dim=text_config.qk_rope_head_dim,
        prefix_granularity=128,
        max_bs=max_bs,
        # K3's per-group demand reads the scheduler's concurrency through
        # CacheRecipe.scheduler_limits, context length included.
        context_len=context_len,
        pd_disaggregation_enabled=pd_enabled,
    )
    # The real K3 draft is BF16 MLA over the FP8 target arena.
    draft_attn_config = (
        SimpleNamespace(**{**vars(attn_config), "kv_cache_dtype": torch.bfloat16})
        if draft_layers
        else None
    )
    return KimiK3Recipe(
        server_args=SimpleNamespace(
            max_total_tokens=None,
            chunked_prefill_size=max_scheduled_tokens,
            speculative_algorithm=speculative_algorithm,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        ),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(text_config=text_config)
        ),
        attn_config=attn_config,
        draft_model_config=(
            SimpleNamespace(num_attention_layers=draft_layers) if draft_layers else None
        ),
        draft_attn_config=draft_attn_config,
        cache_budget_bytes=1 << 34,
        decode_input_tokens=decode_input_tokens,
        overlap_schedule_depth=overlap_schedule_depth,
    )


def kimi_tp8_layout(**recipe_kwargs):
    """The capacity-independent K3 layout: group, then pack."""
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import pack

    recipe = kimi_recipe(**recipe_kwargs)
    groups = recipe.groups()
    layout = pack(
        groups,
        prefix_granularity=recipe.prefix_granularity,
        cache_blocks_per_lcm_block=recipe.packing(groups),
        alignment=recipe.alignment,
        max_padding_fraction=recipe.max_padding_fraction,
    )
    # Same order as CacheRecipe.setup: group, pack, check, then bind.
    recipe.check_layout(layout)
    return recipe, groups, layout


def kimi_tp8_plan(*, num_lcm_blocks: int = 7):
    return kimi_tp8_layout()[2].bind(num_lcm_blocks)


def _kimi_group_specs(group_ids, layer_types, plan):
    del group_ids, layer_types, plan
    return tuple(spec for spec, _ in kimi_recipe().groups())


def make_kimi_pool(device, usable_pages: int = 6, *, with_mla_dims: bool = True):
    from tokenspeed.runtime.configs.kimi_k3_config import KimiLinearConfig
    from tokenspeed.runtime.layers.attention.kv_cache.hybrid_kda import (
        HybridKDATokenToKVPool,
    )

    del with_mla_dims
    text_config = KimiLinearConfig()
    plan = kimi_tp8_plan(num_lcm_blocks=usable_pages)
    recipe = kimi_recipe()
    group_ids = recipe.target_group_ids
    layer_types = recipe.layer_types
    from cache_pool_test_utils import make_pool

    _, pool = make_pool(
        HybridKDATokenToKVPool,
        plan,
        device=device,
        model_dtype=torch.bfloat16,
        dtype=torch.float8_e4m3fn,
        quant_method=None,
        kv_lora_rank=MLA_KV_LORA_RANK,
        qk_rope_head_dim=MLA_QK_ROPE_DIM,
        layer_num=text_config.num_hidden_layers,
        rank=0,
        layer_types=layer_types,
        cache_group_specs=_kimi_group_specs(group_ids, layer_types, plan),
    )
    return pool


@pytest.fixture(scope="module")
def kimi_pool():
    return make_kimi_pool("cpu", usable_pages=1)


def layer_for_group(pool, group_id: str) -> int:
    return next(
        layer_id
        for layer_id, candidate in sorted(pool.state_group_by_layer.items())
        if candidate == group_id
    )


def mla_layer_id(pool) -> int:
    return next(
        layer_id
        for layer_id in range(pool.layer_num)
        if layer_id not in pool.state_group_by_layer
    )


def kda_layer_id(pool) -> int:
    return min(pool.state_group_by_layer)


def cache_metadata_for(contract, tables, device, *, filler_page: int = 1):
    from tokenspeed.runtime.layers.attention.backends.cache_metadata import (
        CacheBatchMetadata,
    )

    bs = np.asarray(next(iter(tables.values()))).shape[0]
    arrays = {}
    for spec in contract.group_specs:
        if spec.group_id in tables:
            arrays[spec.group_id] = np.asarray(tables[spec.group_id], dtype=np.int32)
        else:
            arrays[spec.group_id] = np.full((bs, 1), filler_page, dtype=np.int32)
    forward_op = SimpleNamespace(block_tables_arrays=lambda: arrays)
    metadata = CacheBatchMetadata.from_forward_op(
        forward_op, device=device, contract=contract, num_requests=bs
    )
    return metadata, forward_op


def full_attention_metadata_for(pool, full_table_np, device):
    return cache_metadata_for(
        pool.arena.runtime_contract,
        {"full_attention": np.asarray(full_table_np, dtype=np.int32)},
        device,
    )

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


def kimi_tp8_plan(*, num_lcm_blocks: int = 7):
    from tokenspeed.runtime.configs.kimi_k3_cache_spec import (
        plan_kimi_k3_lcm_cache,
    )
    from tokenspeed.runtime.configs.kimi_k3_config import KimiLinearConfig

    return plan_kimi_k3_lcm_cache(
        KimiLinearConfig(),
        tp_size=8,
        mla_cache_dtype=torch.float8_e4m3fn,
        mla_quant_method=None,
        num_lcm_blocks=num_lcm_blocks,
    )


def make_kimi_pool(device, usable_pages: int = 6, *, with_mla_dims: bool = True):
    from tokenspeed.runtime.configs.kimi_k3_cache_spec import (
        kimi_k3_layer_group_ids,
    )
    from tokenspeed.runtime.configs.kimi_k3_config import KimiLinearConfig
    from tokenspeed.runtime.configs.paged_cache_spec import (
        FULL_ATTENTION,
        LINEAR_ATTENTION,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.lcm_mla import (
        LcmMLATokenToKVPool,
    )

    del with_mla_dims
    text_config = KimiLinearConfig()
    plan = kimi_tp8_plan(num_lcm_blocks=usable_pages)
    group_ids = kimi_k3_layer_group_ids(text_config)
    layer_types = tuple(
        FULL_ATTENTION if group_id == FULL_ATTENTION else LINEAR_ATTENTION
        for group_id in group_ids
    )
    return LcmMLATokenToKVPool(
        size=usable_pages * 12 * plan.logical_block_tokens,
        model_dtype=torch.bfloat16,
        dtype=torch.float8_e4m3fn,
        quant_method=None,
        kv_lora_rank=MLA_KV_LORA_RANK,
        qk_rope_head_dim=MLA_QK_ROPE_DIM,
        layer_num=text_config.num_hidden_layers,
        device=device,
        enable_memory_saver=False,
        max_batch_size=1,
        max_context_len=131_072,
        page_size=plan.logical_block_tokens,
        rank=0,
        layer_types=layer_types,
        layer_group_ids=group_ids,
        max_scheduled_tokens=8192,
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
    )


@pytest.fixture(scope="module")
def kimi_pool():
    return make_kimi_pool("cpu", usable_pages=1)


def layer_for_group(pool, group_id: str) -> int:
    return next(
        layer_id
        for layer_id, candidate in pool._group_ids_by_layer.items()
        if candidate == group_id
    )


def mla_layer_id(pool) -> int:
    return layer_for_group(pool, "full_attention")


def kda_layer_id(pool) -> int:
    return next(
        layer_id
        for layer_id, group_id in pool._group_ids_by_layer.items()
        if group_id != "full_attention"
    )


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
        pool.runtime_contract,
        {"full_attention": np.asarray(full_table_np, dtype=np.int32)},
        device,
    )

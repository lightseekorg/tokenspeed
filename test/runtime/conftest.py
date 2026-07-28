"""Shared helpers and fixtures for the Kimi-K3 FlatKV runtime tests.

Centralizes the reduced-Kimi-TP8 plan builder, contract-pool constructors,
layer/group lookups, operation-bound flat metadata builders, and the small
synthetic hybrid plan shared by the pool and scheduler-geometry tests.

Imports of tokenspeed modules stay lazy (inside the helpers) so collecting
unrelated tests in this directory does not pull in the runtime stack.
"""

from __future__ import annotations

import importlib
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _poison(shape, device="cuda", dtype=torch.int32):
    """Sentinel-filled tensor for inputs the flat path must never consume.

    Args:
        shape: tensor shape.
        device: tensor device.
        dtype: tensor dtype.

    Returns:
        A tensor of ``shape`` filled with the 987_654 poison sentinel.
    """
    return torch.full(shape, 987_654, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Kimi-K3 TP8 reference geometry
# ---------------------------------------------------------------------------

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


def kimi_tp8_plan(*, budget: int = TP8_PAGE_SET_BYTES * 8, flat: bool = True):
    """Full Kimi-K3 TP8 flat cache plan (fp8 MLA latent, block size 1536).

    Args:
        budget: cache budget in bytes handed to the planner.
        flat: value for ``flat_kvcache_enabled`` (False exercises rejection).

    Returns:
        The ``FlatHybridCachePlan`` produced by ``plan_kimi_k3_flat_cache``.
    """
    cache_spec = importlib.import_module(
        "tokenspeed.runtime.configs.kimi_k3_cache_spec"
    )
    config_module = importlib.import_module("tokenspeed.runtime.configs.kimi_k3_config")
    return cache_spec.plan_kimi_k3_flat_cache(
        config_module.KimiLinearConfig(),
        flat_kvcache_enabled=flat,
        tp_size=8,
        mla_cache_dtype=torch.float8_e4m3fn,
        mla_quant_method=None,
        preferred_block_size=128,
        kernel_alignment=128,
        cache_budget_bytes=budget,
    )


def reduced_kimi_plan(usable_pages: int = 1):
    """TP8 plan shrunk to ``usable_pages`` so tests allocate a tiny pool."""
    plan = kimi_tp8_plan()
    total = (usable_pages + 1) * len(plan.physical_slots) * plan.physical_page_bytes
    diagnostics = replace(
        plan.diagnostics,
        usable_pages=usable_pages,
        total_allocated_bytes=total,
        theoretical_capacity_tokens=tuple(
            (group.group_id, usable_pages * group.block_size) for group in plan.groups
        ),
    )
    return replace(
        plan,
        usable_pages=usable_pages,
        token_capacity=usable_pages * plan.block_size,
        diagnostics=diagnostics,
    )


def make_kimi_pool(device, usable_pages: int = 6, *, with_mla_dims: bool = True):
    """Reduced-plan ``FlatHybridCachePool``.

    Args:
        device: pool device ("cpu" or "cuda").
        usable_pages: usable page count for the reduced plan.
        with_mla_dims: bind the MLA read geometry (kv_lora_rank 512, rope 64).
    """
    module = importlib.import_module(
        "tokenspeed.runtime.layers.attention.kv_cache.flat_hybrid"
    )
    kwargs = (
        dict(mla_kv_lora_rank=MLA_KV_LORA_RANK, mla_qk_rope_head_dim=MLA_QK_ROPE_DIM)
        if with_mla_dims
        else {}
    )
    return module.FlatHybridCachePool(
        plan=reduced_kimi_plan(usable_pages), device=device, **kwargs
    )


@pytest.fixture(scope="module")
def kimi_pool():
    """One-page CPU Kimi-K3 contract pool shared across a test module."""
    return make_kimi_pool("cpu", usable_pages=1, with_mla_dims=False)


def layer_for_group(pool, group_id: str) -> int:
    """First layer bound to ``group_id`` in the pool's plan."""
    return next(
        binding.layer_id
        for binding in pool.plan.layer_bindings
        if binding.group_id == group_id
    )


def mla_layer_id(pool) -> int:
    """First layer bound to the ``full_attention`` group."""
    return layer_for_group(pool, "full_attention")


def kda_layer_id(pool) -> int:
    """First layer bound to any state (non-full-attention) group."""
    return next(
        binding.layer_id
        for binding in pool.plan.layer_bindings
        if binding.group_id != "full_attention"
    )


def flat_metadata_for(contract, tables, device, *, filler_page: int = 1):
    """Operation-bound ``FlatCacheBatchMetadata`` from per-group numpy tables.

    Args:
        contract: a ``FlatPagedCacheRuntimeContract``.
        tables: mapping of group_id -> 2D int32 array; groups absent from the
            mapping get a one-column table filled with ``filler_page``.
        device: metadata device.
        filler_page: page id used for the filled groups.

    Returns:
        ``(metadata, forward_op)`` where the metadata is bound to forward_op.
    """
    from tokenspeed.runtime.layers.attention.backends.flat_cache_metadata import (
        FlatCacheBatchMetadata,
    )

    bs = np.asarray(next(iter(tables.values()))).shape[0]
    arrays = {}
    for spec in contract.group_specs:
        if spec.group_id in tables:
            arrays[spec.group_id] = np.asarray(tables[spec.group_id], dtype=np.int32)
        else:
            arrays[spec.group_id] = np.full((bs, 1), filler_page, dtype=np.int32)
    forward_op = SimpleNamespace(flat_block_tables_arrays=lambda: arrays)
    metadata = FlatCacheBatchMetadata.from_forward_op(
        forward_op, device=device, contract=contract, num_requests=bs
    )
    return metadata, forward_op


def full_attention_metadata_for(pool, full_table_np, device):
    """Metadata whose ``full_attention`` table is ``full_table_np`` (state
    groups filled with page 1). Returns ``(metadata, forward_op)``."""
    return flat_metadata_for(
        pool.runtime_contract,
        {"full_attention": np.asarray(full_table_np, dtype=np.int32)},
        device,
    )


# ---------------------------------------------------------------------------
# Small synthetic hybrid plan (pool + scheduler-geometry tests)
# ---------------------------------------------------------------------------


def make_synthetic_hybrid_plan(
    *,
    history_retention: str = "full_history",
    sliding_window_tokens: int | None = None,
    state_layer_ids: tuple[int, ...] = (2, 3),
    conv_shape: tuple[int, ...] = (2,),
):
    """Two history layers + N state layers planned over two physical slots.

    Args:
        history_retention: retention policy of the history layers.
        sliding_window_tokens: window size when retention is sliding_window.
        state_layer_ids: layer ids of the state layers.
        conv_shape: shape of the conv_state component.
    """
    hybrid_cache_plan = importlib.import_module(
        "tokenspeed.runtime.configs.hybrid_cache_plan"
    )
    history = hybrid_cache_plan.CacheComponentSpec(
        name="kv",
        shape=(1, 4),
        dtype=torch.float8_e4m3fn,
        bytes_per_token=4,
        constant_bytes=0,
        alignment=1,
    )
    conv = hybrid_cache_plan.CacheComponentSpec(
        name="conv_state",
        shape=conv_shape,
        dtype=torch.bfloat16,
        bytes_per_token=0,
        constant_bytes=torch.bfloat16.itemsize * torch.Size(conv_shape).numel(),
        alignment=2,
    )
    recurrent = hybrid_cache_plan.CacheComponentSpec(
        name="recurrent_state",
        shape=(2,),
        dtype=torch.float32,
        bytes_per_token=0,
        constant_bytes=8,
        alignment=4,
    )
    specs = tuple(
        hybrid_cache_plan.LayerCacheSpec(
            layer_id=layer_id,
            family="history",
            retention=history_retention,
            transfer_policy="full_suffix",
            group_id_prefix="history",
            group_order=0,
            compatibility_key="history",
            preferred_block_size=4,
            kernel_alignment=4,
            components=(history,),
            sliding_window_tokens=sliding_window_tokens,
        )
        for layer_id in range(2)
    ) + tuple(
        hybrid_cache_plan.LayerCacheSpec(
            layer_id=layer_id,
            family="state",
            retention="full_history",
            transfer_policy="latest_snapshot",
            group_id_prefix="state",
            group_order=1,
            compatibility_key="state",
            preferred_block_size=4,
            kernel_alignment=4,
            components=(conv, recurrent),
        )
        for layer_id in state_layer_ids
    )
    return hybrid_cache_plan.plan_flat_hybrid_cache(
        specs,
        cache_budget_bytes=4 * 2 * 16,
        minimum_usable_pages=3,
    )

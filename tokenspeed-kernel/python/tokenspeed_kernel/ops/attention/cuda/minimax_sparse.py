# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_kernel.thirdparty.cuda.minimax_m3_fused import (
    fused_qknorm_rope_kv_insert,
)


@register_kernel(
    "attention",
    "minimax_sparse_qknorm_rope",
    name="cuda_minimax_sparse_qknorm_rope",
    solution="cuda",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(9, 0),
        vendors=frozenset({"nvidia"}),
    ),
    signatures=format_signatures(("qkv",), "dense", {torch.float16, torch.bfloat16}),
    traits={
        "head_dim": frozenset({128}),
        "index_head_dim": frozenset({128}),
        "rotary_dim": frozenset({8, 16, 32, 64, 128}),
        "cache_dtype": frozenset(
            {None, torch.float16, torch.bfloat16, torch.float8_e4m3fn}
        ),
    },
    priority=Priority.SPECIALIZED,
    tags={"latency"},
)
def cuda_minimax_sparse_qknorm_rope(
    *,
    qkv: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    index_q_norm_weight: torch.Tensor,
    index_k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    num_index_heads: int,
    rotary_dim: int,
    eps: float,
    q_out: torch.Tensor,
    index_q_out: torch.Tensor,
    cache_insert=None,
    enable_pdl: bool = False,
) -> None:
    slot_mapping = k_cache = v_cache = index_k_cache = None
    kv_cache_dtype = "auto"
    if cache_insert is not None:
        slot_mapping = cache_insert.slot_mapping
        k_cache = cache_insert.k_cache
        v_cache = cache_insert.v_cache
        index_k_cache = cache_insert.index_k_cache
        if v_cache.dtype != k_cache.dtype:
            raise TypeError("K and V caches must have the same dtype")
        if index_k_cache.dtype != qkv.dtype:
            raise TypeError(
                "MiniMax sparse index cache must match the projection dtype"
            )
        if k_cache.dtype == torch.float8_e4m3fn:
            k_cache = k_cache.view(torch.uint8)
            v_cache = v_cache.view(torch.uint8)
            kv_cache_dtype = "fp8_e4m3"

    fused_qknorm_rope_kv_insert(
        qkv,
        q_norm_weight,
        k_norm_weight,
        cos_sin_cache,
        positions,
        num_heads,
        num_kv_heads,
        rotary_dim,
        eps,
        index_q_norm_weight=index_q_norm_weight,
        index_k_norm_weight=index_k_norm_weight,
        num_index_heads=num_index_heads,
        slot_mapping=slot_mapping,
        k_cache=k_cache,
        v_cache=v_cache,
        index_cache=index_k_cache,
        q_out=q_out,
        index_q_out=index_q_out,
        kv_cache_dtype=kv_cache_dtype,
        enable_pdl=enable_pdl,
    )

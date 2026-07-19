# Copyright (c) 2026 LightSeek Foundation
# SPDX-License-Identifier: MIT

"""Registered Triton kernels for indexed block-sparse attention (MSA)."""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from tokenspeed_kernel.thirdparty.triton.minimax_m3 import (
    minimax_m3_msa_indexer as _triton_indexer,
)
from tokenspeed_kernel.thirdparty.triton.minimax_m3 import (
    minimax_m3_msa_sparse_attention as _triton_sparse_attention,
)

_BF16_INDEX_SIGNATURE = format_signature(
    index_q=dense_tensor_format(torch.bfloat16),
    index_k=dense_tensor_format(torch.bfloat16),
    index_k_cache=dense_tensor_format(torch.bfloat16),
)
_BF16_ATTN_SIGNATURE = format_signature(
    q=dense_tensor_format(torch.bfloat16),
    k_cache=dense_tensor_format(torch.bfloat16),
    v_cache=dense_tensor_format(torch.bfloat16),
)
_MSA_TRAITS = {
    "head_dim": frozenset({128}),
    "page_size": frozenset({128}),
    "topk": frozenset({16}),
    "decode": frozenset({False, True}),
}


@register_kernel(
    "attention",
    "msa_indexer",
    name="triton_msa_indexer",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=frozenset({_BF16_INDEX_SIGNATURE}),
    traits=_MSA_TRAITS,
    priority=Priority.PERFORMANT,
    tags={"portability", "determinism"},
)
def triton_msa_indexer(*args, **kwargs) -> torch.Tensor:
    return _triton_indexer(*args, **kwargs)


@register_kernel(
    "attention",
    "msa_sparse_attention",
    name="triton_msa_sparse_attention",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=frozenset({_BF16_ATTN_SIGNATURE}),
    traits=_MSA_TRAITS,
    priority=Priority.PERFORMANT,
    tags={"portability"},
)
def triton_msa_sparse_attention(*args, **kwargs) -> torch.Tensor:
    return _triton_sparse_attention(*args, **kwargs)


__all__ = [
    "triton_msa_indexer",
    "triton_msa_sparse_attention",
]

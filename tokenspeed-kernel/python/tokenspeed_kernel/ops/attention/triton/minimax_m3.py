# Copyright (c) 2026 LightSeek Foundation
# SPDX-License-Identifier: MIT

"""Registered Triton kernels for MiniMax-M3 MSA."""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
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
_FP8_INDEX_SIGNATURE = format_signature(
    index_q=dense_tensor_format(torch.bfloat16),
    index_k=dense_tensor_format(torch.bfloat16),
    index_k_cache=dense_tensor_format(torch.float8_e4m3fn),
)
_FP8_ATTN_SIGNATURE = format_signature(
    q=dense_tensor_format(torch.bfloat16),
    k_cache=dense_tensor_format(torch.float8_e4m3fn),
    v_cache=dense_tensor_format(torch.float8_e4m3fn),
)
_MSA_TRAITS = {
    "head_dim": frozenset({128}),
    "page_size": frozenset({128}),
    "topk": frozenset({16}),
    "decode": frozenset({False, True}),
}


@register_kernel(
    "attention",
    "minimax_m3_msa_indexer",
    name="triton_minimax_m3_msa_indexer",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=frozenset({_BF16_INDEX_SIGNATURE}),
    traits=_MSA_TRAITS,
    priority=Priority.PERFORMANT,
    tags={"portability", "determinism"},
)
def triton_minimax_m3_msa_indexer(*args, **kwargs) -> torch.Tensor:
    return _triton_indexer(*args, **kwargs)


@register_kernel(
    "attention",
    "minimax_m3_msa_indexer",
    name="triton_minimax_m3_msa_fp8_indexer",
    solution="triton",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(10, 0),
        vendors=frozenset({"nvidia"}),
    ),
    signatures=frozenset({_FP8_INDEX_SIGNATURE}),
    traits=_MSA_TRAITS,
    priority=Priority.PERFORMANT,
    tags={"determinism", "fp8"},
)
def triton_minimax_m3_msa_fp8_indexer(*args, **kwargs) -> torch.Tensor:
    return _triton_indexer(*args, **kwargs)


@register_kernel(
    "attention",
    "minimax_m3_msa_sparse_attention",
    name="triton_minimax_m3_msa_sparse_attention",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=frozenset({_BF16_ATTN_SIGNATURE}),
    traits=_MSA_TRAITS,
    priority=Priority.PERFORMANT,
    tags={"portability"},
)
def triton_minimax_m3_msa_sparse_attention(*args, **kwargs) -> torch.Tensor:
    return _triton_sparse_attention(*args, **kwargs)


@register_kernel(
    "attention",
    "minimax_m3_msa_sparse_attention",
    name="triton_minimax_m3_msa_fp8_sparse_attention",
    solution="triton",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(10, 0),
        vendors=frozenset({"nvidia"}),
    ),
    signatures=frozenset({_FP8_ATTN_SIGNATURE}),
    traits=_MSA_TRAITS,
    priority=Priority.PERFORMANT,
    tags={"fp8"},
)
def triton_minimax_m3_msa_fp8_sparse_attention(*args, **kwargs) -> torch.Tensor:
    return _triton_sparse_attention(*args, **kwargs)


__all__ = [
    "triton_minimax_m3_msa_indexer",
    "triton_minimax_m3_msa_fp8_indexer",
    "triton_minimax_m3_msa_sparse_attention",
    "triton_minimax_m3_msa_fp8_sparse_attention",
]

# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import torch
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.registry import KernelRegistry, Priority
from tokenspeed_kernel.selection import (
    NoKernelFoundError,
    select_kernel,
    spec_matches_traits,
)
from tokenspeed_kernel.signature import (
    MXFP8_BLOCK_SCALE,
    dense_tensor_format,
    format_signature,
    tensor_format,
)

AttentionResult = torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]


# One UE8M0 scale per 32 consecutive head_dim elements (MXFP8).
MXFP8_ATTENTION_BLOCK_SCALE = MXFP8_BLOCK_SCALE


def _attention_format_signature(**roles: torch.Tensor):
    return format_signature(
        **{role: dense_tensor_format(tensor.dtype) for role, tensor in roles.items()}
    )


def _mxfp8_attention_format_signature(**roles: torch.Tensor):
    return format_signature(
        **{
            role: tensor_format(
                "mxfp8", tensor.dtype, scale=MXFP8_ATTENTION_BLOCK_SCALE
            )
            for role, tensor in roles.items()
        }
    )


def _blockscaled_signature_and_scales(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    q_scale: torch.Tensor | None,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
):
    """Pick dense vs MXFP8 signature and build the scale kwargs splat.

    q_scale selects the block-scaled path; k_scale/v_scale must accompany it.
    Returns (signature, scale_kwargs) for the paged-KV-cache entry points.
    """
    if q_scale is not None:
        assert (
            k_scale is not None and v_scale is not None
        ), "MXFP8 attention requires q_scale, k_scale, and v_scale together"
        signature = _mxfp8_attention_format_signature(
            q=q, k_cache=k_cache, v_cache=v_cache
        )
    else:
        signature = _attention_format_signature(q=q, k_cache=k_cache, v_cache=v_cache)
    return signature, dict(q_scale=q_scale, k_scale=k_scale, v_scale=v_scale)


LSE_LN = math.log2(math.e)


# ===-----------------------------------------------------------------------===#
# QSA Sparse Attention
# ===-----------------------------------------------------------------------===#


def qsa_sparse_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    selected_slots: torch.Tensor,
    *,
    scale: float,
    max_seqlen_q: int,
    metadata_capacity_rows: int | None,
    k_scale: float | torch.Tensor | None,
    v_scale: float | torch.Tensor | None,
    override: str | None,
    solution: str | None,
) -> torch.Tensor:
    """Attend to a per-query list of physical QSA KV-cache slots.

    Args:
        q: Query tensor shaped ``[tokens, query_heads, head_dim]``.
        k_cache: Flattened key cache shaped
            ``[cache_slots, kv_heads, head_dim]``.
        v_cache: Flattened value cache shaped
            ``[cache_slots, kv_heads, value_head_dim]``.
        selected_slots: Physical cache slots shaped ``[tokens, budget]``;
            non-positive values are ignored.
        scale: Softmax scale applied to query-key scores.
        max_seqlen_q: Number of uniformly packed query tokens per request. This
            is 1 for normal decode and ``spec_num_tokens`` for compact
            speculative decode.
        metadata_capacity_rows: Row capacity reserved by stateful fallback
            implementations. Pass ``None`` to use the actual query-row count;
            workspace-free kernels ignore it.
        k_scale: Optional scalar FP8 key descale.
        v_scale: Optional scalar FP8 value descale.
        override: Optional registered kernel name or solution override.
        solution: Optional kernel solution selected through normal capability
            and shape filtering.

    Returns:
        Attention output shaped
        ``[tokens, query_heads, value_head_dim]`` with the query dtype.

    The SM100 CuTe DSL implementation is preferred when its specialization
    matches. Other supported NVIDIA architectures use FlashInfer FA2 sparse
    attention as the registered fallback.
    """

    if q.ndim != 3 or k_cache.ndim != 3 or v_cache.ndim != 3:
        raise ValueError("QSA sparse attention expects rank-three Q/K/V tensors")
    if selected_slots.ndim != 2 or selected_slots.shape[0] != q.shape[0]:
        raise ValueError("QSA selected slots must have one row per query token")
    if max_seqlen_q < 1:
        raise ValueError("QSA max_seqlen_q must be positive")
    if q.shape[0] % max_seqlen_q:
        raise ValueError("QSA query rows must be divisible by max_seqlen_q")
    if q.shape[0] == 0:
        return q.new_empty((0, q.shape[1], v_cache.shape[-1]))
    traits = {
        "batch_size": q.shape[0] // max_seqlen_q,
        "q_len": max_seqlen_q,
        "head_dim": q.shape[-1],
        "value_head_dim": v_cache.shape[-1],
        "num_q_heads": q.shape[1],
        "num_kv_heads": k_cache.shape[1],
        "selected_width": selected_slots.shape[1],
    }
    signature = _attention_format_signature(q=q, k_cache=k_cache, v_cache=v_cache)
    kernel = select_kernel(
        "attention",
        "qsa_sparse_attention",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )
    return kernel(
        q,
        k_cache,
        v_cache,
        selected_slots,
        scale=scale,
        max_seqlen_q=max_seqlen_q,
        metadata_capacity_rows=metadata_capacity_rows,
        k_scale=k_scale,
        v_scale=v_scale,
    )


# Backend registration (side-effect imports)
# isort: off
import tokenspeed_kernel.ops.attention.qsa.triton  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.qsa.cute_dsl  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.qsa.flashinfer  # noqa: E402,F401

# isort: on


__all__ = [
    "qsa_sparse_attention",
]

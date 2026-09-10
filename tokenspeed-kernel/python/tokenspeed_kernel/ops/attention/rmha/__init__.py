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
from tokenspeed_kernel.platform import current_platform, pdl_enabled
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


# rel_mha: relative-distance-bias MHA; own family keeps model-specific args out of plain mha.


def rel_mha_plan(
    dtype: torch.dtype,
    head_dim: int,
    window_left: int = -1,
    return_lse: bool = False,
    solution: str | None = None,
) -> dict:
    """Build a relative-attention MHA execution plan.

    Args:
        dtype: Query/K/V dtype for prefill planning.
        head_dim: Attention head dimension.
        window_left: Exclusive left sliding-window size, or -1 for full-context
            attention.
        return_lse: Whether the selected path must return LSE values.
        solution: Optional kernel solution to restrict planning.

    Returns:
        Same "extend_mode" dict as mha_plan, planned over the rel_mha_prefill
        operator.
    """
    if dtype == torch.float8_e4m3fn:
        return {"extend_mode": "prewrite"}

    traits = {
        "head_dim": head_dim,
        "sliding_window": window_left >= 0,
        "return_lse": return_lse,
    }
    signature = format_signature(
        q=dense_tensor_format(dtype),
        k=dense_tensor_format(dtype),
        v=dense_tensor_format(dtype),
    )
    candidates = KernelRegistry.get().get_for_operator(
        "attention",
        "rel_mha_prefill",
        platform=current_platform(),
        format_signature=signature,
        solution=solution,
    )
    candidates = [spec for spec in candidates if spec_matches_traits(spec, traits)]
    extend_mode = (
        "postwrite"
        if any(spec.priority >= Priority.PERFORMANT for spec in candidates)
        else "prewrite"
    )
    return {"extend_mode": extend_mode}


def rel_mha_prefill(
    # attention inputs
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rel_logits: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: list[int],
    max_seqlen: int,
    # attention options
    window_left: int = -1,
    return_lse: bool = False,
    softmax_scale: float | None = None,
    tau: torch.Tensor | None = None,
    # dispatch options
    override: str | None = None,
    solution: str | None = None,
) -> AttentionResult:
    """Relative-attention MHA prefill from uncached KV.

    Args:
        q: Query tensor with shape [total_q, num_q_heads, head_dim].
        k: Key tensor with shape [total_kv, num_kv_heads, head_dim].
        v: Value tensor with shape [total_kv, num_kv_heads, head_dim].
        rel_logits: Learned relative bias logits with shape
            [total_q, num_q_heads, rel_extent]. rel_logits[t, h, d] is added
            to the pre-softmax logit of query row t against the key d
            positions behind it, for 0 <= d < rel_extent; other distances
            contribute zero bias.
        cu_seqlens: Cumulative sequence lengths with shape [batch + 1].
            KV cumulative sequence lengths are assumed to be identical.
        cu_seqlens_cpu: Host-side cumulative sequence lengths as a strict
            list[int]. Used for host-side launch metadata; must match cu_seqlens.
        max_seqlen: Maximum sequence length.
        window_left: Exclusive left sliding-window size. -1 means full attention.
        return_lse: Whether to also return natural-log log-sum-exp values with
            shape [total_q, num_q_heads].
        softmax_scale: Scale applied to QK logits before softmax. None uses the
            backend default 1/sqrt(head_dim).
        tau: Optional fp32 per-query-row multiplier on the total pre-softmax
            logits, ``tau * (softmax_scale * q@k^T + rel)``; shape matches
            q's row count and values must be positive.
        override: Optional kernel override name.
        solution: Optional kernel solution to force through normal selection.

    Attention is always causal within each sequence.
    """
    batch_size = cu_seqlens.shape[0] - 1

    traits = {
        "head_dim": q.shape[-1],
        "sliding_window": window_left >= 0,
        "return_lse": return_lse,
    }
    signature = _attention_format_signature(q=q, k=k, v=v)
    kernel = select_kernel(
        "attention",
        "rel_mha_prefill",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )

    shape_params = {
        "batch_size": batch_size,
        "total_q": q.shape[0],
        "total_kv": k.shape[0],
        "num_q_heads": q.shape[1],
        "num_kv_heads": k.shape[1],
        "head_dim": q.shape[-1],
        "rel_extent": rel_logits.shape[-1],
        "max_seqlen": max_seqlen,
    }
    ShapeCapture.get().record(
        "attention",
        "rel_mha_prefill",
        kernel.name,
        q.dtype,
        shape_params,
    )

    with kernel_scope(
        "attention",
        "rel_mha_prefill",
        q.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            q=q,
            k=k,
            v=v,
            rel_logits=rel_logits,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            max_seqlen=max_seqlen,
            window_left=window_left,
            return_lse=return_lse,
            softmax_scale=softmax_scale,
            tau=tau,
            enable_pdl=pdl_enabled(),
        )


def rel_mha_extend_with_kvcache(
    # attention inputs
    q: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    rel_logits: torch.Tensor,
    # attention options
    window_left: int = -1,
    return_lse: bool = False,
    softmax_scale: float | None = None,
    tau: torch.Tensor | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    # dispatch options
    override: str | None = None,
    solution: str | None = None,
) -> AttentionResult:
    """Relative-attention MHA extend with paged KV cache.

    Args:
        q: Query tensor with shape [total_q, num_q_heads, head_dim].
        cu_seqlens_q: Query cumulative sequence lengths with shape [batch + 1].
        cu_seqlens_kv: KV cumulative sequence lengths with shape [batch + 1].
        k_cache: Paged key cache with shape [num_pages, page_size, num_kv_heads, head_dim].
        v_cache: Paged value cache with shape [num_pages, page_size, num_kv_heads, head_dim].
        page_table: Page table with shape [batch, max_pages_per_seq].
        cache_seqlens: Visible KV lengths in the cache, shape [batch]. Query
            lengths are independent and may be smaller than KV lengths.
        max_seqlen_q: Maximum query length.
        max_seqlen_k: Maximum KV length.
        rel_logits: Learned relative bias logits with shape
            [total_q, num_q_heads, rel_extent]; rows are addressed by each
            request's batch-flattened query positions (cu_seqlens_q). The
            relative distance is computed against the query's absolute
            position in the cached sequence.
        window_left: Exclusive left sliding-window size. -1 means full attention.
        return_lse: Whether to also return natural-log log-sum-exp values with
            shape [total_q, num_q_heads].
        softmax_scale: Scale applied to QK logits before softmax. None uses the
            backend default 1/sqrt(head_dim).
        tau: Optional fp32 per-query-row multiplier on the total pre-softmax
            logits, ``tau * (softmax_scale * q@k^T + rel)``; shape matches
            q's row count and values must be positive.
        override: Optional kernel override name.
        solution: Optional kernel solution to force through normal selection.

    Each request's query tokens attend all visible cached KV tokens causally.

    ``q_scale``/``k_scale``/``v_scale`` select the MXFP8 block-scaled path:
    q/k_cache/v_cache must then be float8_e4m3fn; q_scale is flat per-token
    UE8M0 [total_q, num_q_heads, head_dim // 32], k_scale/v_scale use the
    paged interleaved layout [num_pages, num_kv_heads, page_size // 128,
    32, 4, 4].
    """
    signature, scale_kwargs = _blockscaled_signature_and_scales(
        q, k_cache, v_cache, q_scale, k_scale, v_scale
    )
    traits = {
        "head_dim": q.shape[-1],
        "page_size": k_cache.shape[1],
        "sliding_window": window_left >= 0,
        "return_lse": return_lse,
    }
    kernel = select_kernel(
        "attention",
        "rel_mha_extend_with_kvcache",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )

    shape_params = {
        "batch_size": cache_seqlens.shape[0],
        "total_q": q.shape[0],
        "num_pages": k_cache.shape[0],
        "page_size": k_cache.shape[1],
        "max_pages_per_seq": page_table.shape[1],
        "num_q_heads": q.shape[1],
        "num_kv_heads": k_cache.shape[2],
        "head_dim": q.shape[-1],
        "rel_extent": rel_logits.shape[-1],
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
    }
    ShapeCapture.get().record(
        "attention",
        "rel_mha_extend_with_kvcache",
        kernel.name,
        q.dtype,
        shape_params,
    )

    with kernel_scope(
        "attention",
        "rel_mha_extend_with_kvcache",
        q.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            q=q,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            rel_logits=rel_logits,
            window_left=window_left,
            return_lse=return_lse,
            softmax_scale=softmax_scale,
            tau=tau,
            enable_pdl=pdl_enabled(),
            **scale_kwargs,
        )


def rel_mha_decode_with_kvcache(
    # attention inputs
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    rel_logits: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int = 1,
    # attention options
    window_left: int = -1,
    softmax_scale: float | None = None,
    tau: torch.Tensor | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    # dispatch options
    override: str | None = None,
    solution: str | None = None,
) -> AttentionResult:
    """Relative-attention MHA decode with paged KV cache.

    Args:
        q: Query tensor with shape [batch * max_seqlen_q, num_q_heads, head_dim].
        k_cache: Paged key cache with shape [num_pages, page_size, num_kv_heads, head_dim].
        v_cache: Paged value cache with shape [num_pages, page_size, num_kv_heads, head_dim].
        page_table: Page table with shape [batch, max_pages_per_seq].
        cache_seqlens: Total visible KV lengths after appending current decode
            tokens, shape [batch].
        max_seqlen_k: Maximum KV length.
        rel_logits: Learned relative bias logits with shape
            [batch * max_seqlen_q, num_q_heads, rel_extent], one row per
            decode token.
        cu_seqlens_q: Query cumulative sequence lengths with shape [batch + 1]
            (arange(batch + 1) * max_seqlen_q). Required: decode runs the
            varlen path so each request's query rows map into rel_logits at
            their batch-flattened positions.
        max_seqlen_q: Number of uniformly packed query tokens per request. This
            is 1 for normal decode and `spec_num_tokens` for compact
            speculative decode.
        window_left: Exclusive left sliding-window size. -1 means full attention.
        softmax_scale: Scale applied to QK logits before softmax. None uses the
            backend default 1/sqrt(head_dim).
        tau: Optional fp32 per-query-row multiplier on the total pre-softmax
            logits, ``tau * (softmax_scale * q@k^T + rel)``; shape matches
            q's row count and values must be positive.
        override: Optional kernel override name.
        solution: Optional kernel solution to force through normal selection.

    ``q_scale``/``k_scale``/``v_scale`` select the MXFP8 block-scaled path:
    q/k_cache/v_cache must then be float8_e4m3fn; q_scale is flat per-token
    UE8M0 [batch, num_q_heads, head_dim // 32], k_scale/v_scale use the
    paged interleaved layout [num_pages, num_kv_heads, page_size // 128,
    32, 4, 4].

    Uniform ``max_seqlen_q > 1`` (spec verify) rides v2's native prediction
    dimension — unexpanded ``[batch]`` seqlens and ``[batch, W]`` table, one
    KV load per request. Non-uniform multi-query takes the fork varlen path.
    """
    blockscaled = q_scale is not None
    signature, scale_kwargs = _blockscaled_signature_and_scales(
        q, k_cache, v_cache, q_scale, k_scale, v_scale
    )
    traits = {
        "head_dim": q.shape[-1],
        "page_size": k_cache.shape[1],
        "sliding_window": window_left >= 0,
        "return_lse": False,
    }
    kernel = select_kernel(
        "attention",
        "rel_mha_decode_with_kvcache",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )

    shape_params = {
        "batch_size": cache_seqlens.shape[0],
        "total_q": q.shape[0],
        "num_pages": k_cache.shape[0],
        "page_size": k_cache.shape[1],
        "max_pages_per_seq": page_table.shape[1],
        "num_q_heads": q.shape[1],
        "num_kv_heads": k_cache.shape[2],
        "head_dim": q.shape[-1],
        "rel_extent": rel_logits.shape[-1],
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
    }
    ShapeCapture.get().record(
        "attention",
        "rel_mha_decode_with_kvcache",
        kernel.name,
        q.dtype,
        shape_params,
    )

    with kernel_scope(
        "attention",
        "rel_mha_decode_with_kvcache",
        q.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            max_seqlen_k=max_seqlen_k,
            rel_logits=rel_logits,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            window_left=window_left,
            softmax_scale=softmax_scale,
            tau=tau,
            enable_pdl=pdl_enabled(),
            **scale_kwargs,
        )


# Backend registration (side-effect imports)
# isort: off
import tokenspeed_kernel.ops.attention.rmha.triton  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.rmha.cuda  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.rmha.cute_dsl  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.rmha.gluon  # noqa: E402,F401

# isort: on

__all__ = [
    "rel_mha_plan",
    "rel_mha_prefill",
    "rel_mha_extend_with_kvcache",
    "rel_mha_decode_with_kvcache",
]

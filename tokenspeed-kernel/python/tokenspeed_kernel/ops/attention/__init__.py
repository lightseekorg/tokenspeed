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

# Backend registration (side-effect imports)
import tokenspeed_kernel.ops.attention.flash_attn  # noqa: F401
import tokenspeed_kernel.ops.attention.flashinfer  # noqa: F401
import tokenspeed_kernel.ops.attention.triton  # noqa: F401
import torch
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.selection import select_kernel

AttentionResult = torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]

__all__ = [
    "mha_prefill",
    "mha_prefill_with_kvcache",
    "mha_decode_with_kvcache",
]


def _requires_logit_cap(logit_cap: float) -> bool:
    return logit_cap != 0.0


def mha_prefill(
    # attention inputs
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    # attention options
    softmax_scale: float | None = None,
    is_causal: bool = True,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
    # dispatch options
    override: str | None = None,
) -> AttentionResult:
    """Ragged MHA prefill without KV cache.

    Args:
        q: Query tensor with shape [total_q, num_q_heads, head_dim].
        k: Key tensor with shape [total_kv, num_kv_heads, head_dim].
        v: Value tensor with shape [total_kv, num_kv_heads, head_dim].
        cu_seqlens_q: Query cumulative sequence lengths with shape [batch + 1].
            KV cumulative sequence lengths are assumed to be identical.
        max_seqlen_q: Maximum query length.
        max_seqlen_k: Maximum KV length.
        softmax_scale: Optional scale factor applied before softmax.
        is_causal: Whether to apply causal masking.
        window_left: Inclusive left sliding-window size. -1 means full attention.
        logit_cap: Optional soft cap applied to attention logits.
        sinks: Optional attention sink tensor.
        return_lse: Whether to also return log-sum-exp values.
        override: Optional kernel override name.

    Standard full-sequence prefill assumes query and KV sequence boundaries match.
    """
    # Select kernel
    traits = {
        "num_q_heads": q.shape[1],
        "num_kv_heads": k.shape[1],
        "head_dim": q.shape[-1],
        "is_causal": is_causal,
        "sliding_window": window_left >= 0,
        "support_logit_cap": _requires_logit_cap(logit_cap),
        "support_sinks": sinks is not None,
        "return_lse": return_lse,
    }
    kernel = select_kernel(
        "attention",
        "mha_prefill",
        q.dtype,
        traits=traits,
        override=override,
    )

    # Record shapes
    shape_params = {
        "batch_size": cu_seqlens_q.shape[0] - 1,
        "total_q": q.shape[0],
        "total_kv": k.shape[0],
        "num_q_heads": q.shape[1],
        "num_kv_heads": k.shape[1],
        "head_dim": q.shape[-1],
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
    }
    ShapeCapture.get().record(
        "attention",
        "mha_prefill",
        kernel.name,
        q.dtype,
        shape_params,
    )

    # Enter profiling scope
    with kernel_scope(
        "attention",
        "mha_prefill",
        q.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            is_causal=is_causal,
            window_left=window_left,
            logit_cap=logit_cap,
            sinks=sinks,
            return_lse=return_lse,
        )


def mha_prefill_with_kvcache(
    # attention inputs
    q: torch.Tensor,
    k: torch.Tensor | None,
    v: torch.Tensor | None,
    cu_seqlens_q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    # attention options
    softmax_scale: float | None = None,
    is_causal: bool = True,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
    # dispatch options
    override: str | None = None,
) -> AttentionResult:
    """Ragged MHA extend-prefill with paged KV cache.

    Args:
        q: Query tensor with shape [total_q, num_q_heads, head_dim].
        k: Optional new key tensor with shape [total_q, num_kv_heads, head_dim].
            When None, the appended KV is assumed to already be present in k_cache.
        v: Optional new value tensor with shape [total_q, num_kv_heads, head_dim].
            When None, the appended KV is assumed to already be present in v_cache.
        cu_seqlens_q: Query cumulative sequence lengths with shape [batch + 1].
        k_cache: Paged key cache with shape [num_pages, page_size, num_kv_heads, head_dim].
        v_cache: Paged value cache with shape [num_pages, page_size, num_kv_heads, head_dim].
        page_table: Page table with shape [batch, max_pages_per_seq].
        cache_seqlens: Total visible KV lengths after appending the new KV, shape [batch].
        max_seqlen_q: Maximum query length.
        max_seqlen_k: Maximum KV length.
        softmax_scale: Optional scale factor applied before softmax.
        is_causal: Whether to apply causal masking.
        window_left: Inclusive left sliding-window size. -1 means full attention.
        logit_cap: Optional soft cap applied to attention logits.
        sinks: Optional attention sink tensor.
        return_lse: Whether to also return log-sum-exp values.
        override: Optional kernel override name.
    """
    if (k is None) != (v is None):
        raise ValueError("k and v must both be provided or both be None")
    prewritten_kv = k is None

    # Select kernel
    traits = {
        "num_q_heads": q.shape[1],
        "num_kv_heads": (k.shape[1] if k is not None else k_cache.shape[2]),
        "head_dim": q.shape[-1],
        "page_size": k_cache.shape[1],
        "prewritten_kv": prewritten_kv,
        "is_causal": is_causal,
        "sliding_window": window_left >= 0,
        "support_logit_cap": _requires_logit_cap(logit_cap),
        "support_sinks": sinks is not None,
        "return_lse": return_lse,
    }
    kernel = select_kernel(
        "attention",
        "mha_prefill_with_kvcache",
        q.dtype,
        traits=traits,
        override=override,
    )

    # Record shapes
    shape_params = {
        "batch_size": cache_seqlens.shape[0],
        "total_q": q.shape[0],
        "num_pages": k_cache.shape[0],
        "page_size": k_cache.shape[1],
        "max_pages_per_seq": page_table.shape[1],
        "num_q_heads": q.shape[1],
        "num_kv_heads": k.shape[1] if k is not None else k_cache.shape[2],
        "head_dim": q.shape[-1],
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
    }
    ShapeCapture.get().record(
        "attention",
        "mha_prefill_with_kvcache",
        kernel.name,
        q.dtype,
        shape_params,
    )

    # Enter profiling scope
    with kernel_scope(
        "attention",
        "mha_prefill_with_kvcache",
        q.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            softmax_scale=softmax_scale,
            is_causal=is_causal,
            window_left=window_left,
            logit_cap=logit_cap,
            sinks=sinks,
            return_lse=return_lse,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
        )


def mha_decode_with_kvcache(
    # attention inputs
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    # attention options
    softmax_scale: float | None = None,
    is_causal: bool = True,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
    # dispatch options
    override: str | None = None,
) -> AttentionResult:
    """Single-token MHA decode with paged KV cache.

    Args:
        q: Query tensor with shape [batch, num_q_heads, head_dim].
        k_cache: Paged key cache with shape [num_pages, page_size, num_kv_heads, head_dim].
        v_cache: Paged value cache with shape [num_pages, page_size, num_kv_heads, head_dim].
        page_table: Page table with shape [batch, max_pages_per_seq].
        cache_seqlens: Total visible KV lengths after appending current decode tokens, shape [batch].
        max_seqlen_k: Maximum KV length.
        softmax_scale: Optional scale factor applied before softmax.
        is_causal: Whether to apply causal masking.
        window_left: Inclusive left sliding-window size. -1 means full attention.
        logit_cap: Optional soft cap applied to attention logits.
        sinks: Optional attention sink tensor.
        return_lse: Whether to also return log-sum-exp values.
        override: Optional kernel override name.
    """
    if q.shape[0] != cache_seqlens.shape[0]:
        raise ValueError(
            "mha_decode_with_kvcache assumes query length 1; "
            f"got q.shape[0]={q.shape[0]} and batch={cache_seqlens.shape[0]}"
        )

    # Select kernel
    traits = {
        "num_q_heads": q.shape[1],
        "num_kv_heads": k_cache.shape[2],
        "head_dim": q.shape[-1],
        "page_size": k_cache.shape[1],
        "is_causal": is_causal,
        "sliding_window": window_left >= 0,
        "support_logit_cap": _requires_logit_cap(logit_cap),
        "support_sinks": sinks is not None,
        "return_lse": return_lse,
        "query_len": 1,
    }
    kernel = select_kernel(
        "attention",
        "mha_decode_with_kvcache",
        q.dtype,
        traits=traits,
        override=override,
    )

    # Record shapes
    shape_params = {
        "batch_size": cache_seqlens.shape[0],
        "total_q": q.shape[0],
        "num_pages": k_cache.shape[0],
        "page_size": k_cache.shape[1],
        "max_pages_per_seq": page_table.shape[1],
        "num_q_heads": q.shape[1],
        "num_kv_heads": k_cache.shape[2],
        "head_dim": q.shape[-1],
        "max_seqlen_q": 1,
        "max_seqlen_k": max_seqlen_k,
    }
    ShapeCapture.get().record(
        "attention",
        "mha_decode_with_kvcache",
        kernel.name,
        q.dtype,
        shape_params,
    )

    # Enter profiling scope
    with kernel_scope(
        "attention",
        "mha_decode_with_kvcache",
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
            softmax_scale=softmax_scale,
            is_causal=is_causal,
            window_left=window_left,
            logit_cap=logit_cap,
            sinks=sinks,
            return_lse=return_lse,
            max_seqlen_k=max_seqlen_k,
        )

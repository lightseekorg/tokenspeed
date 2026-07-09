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

# Backend registration (side-effect imports)
import tokenspeed_kernel.ops.attention.cuda  # noqa: F401
import tokenspeed_kernel.ops.attention.deep_gemm  # noqa: F401
import tokenspeed_kernel.ops.attention.flash_attn  # noqa: F401
import tokenspeed_kernel.ops.attention.flash_mla  # noqa: F401
import tokenspeed_kernel.ops.attention.flashinfer  # noqa: F401
import tokenspeed_kernel.ops.attention.gluon  # noqa: F401
import tokenspeed_kernel.ops.attention.triton  # noqa: F401
import torch
from tokenspeed_kernel.ops.attention.gdn_utils import (
    GdnCheckpointLayout,
    GdnChunkPrefillResult,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.registry import KernelRegistry, Priority
from tokenspeed_kernel.selection import (
    NoKernelFoundError,
    select_kernel,
    spec_matches_traits,
)
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

AttentionResult = torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]


def _attention_format_signature(**roles: torch.Tensor):
    return format_signature(
        **{role: dense_tensor_format(tensor.dtype) for role, tensor in roles.items()}
    )


__all__ = [
    "mha_prefill",
    "mha_extend_with_kvcache",
    "mha_decode_with_kvcache",
    "gdn_chunk_prefill",
    "GdnCheckpointLayout",
    "GdnChunkPrefillResult",
    "mla_prefill",
    "mla_decode_with_kvcache",
    "dsa_prefill",
    "dsa_decode",
    "dsa_prefill_topk",
    "dsa_decode_topk",
    "dsa_plan",
    "attn_merge_state",
    "mha_plan",
]

LSE_LN = math.log2(math.e)


# ===-----------------------------------------------------------------------===#
# GDN Kernels
# ===-----------------------------------------------------------------------===#


def gdn_chunk_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    qk_l2norm: bool = False,
    output_final_state: bool = True,
    output_h: bool = False,
    override: str | None = None,
    solution: str | None = None,
) -> GdnChunkPrefillResult:
    """Run Gated Delta Net chunked prefill through kernel selection.

    Args:
        q: Query tensor shaped ``[1, total_tokens, num_q_heads, head_dim]``.
        k: Key tensor shaped ``[1, total_tokens, num_k_heads, head_dim]``.
        v: Value tensor shaped ``[1, total_tokens, num_v_heads, head_v_dim]``.
        g: Log-space forget gate shaped ``[1, total_tokens, num_v_heads]``.
        beta: Beta gate shaped ``[1, total_tokens, num_v_heads]``.
        scale: Attention scale. ``None`` lets the implementation use its default.
        initial_state: Recurrent state shaped ``[batch, num_v_heads, head_dim, head_v_dim]``.
        cu_seqlens: Cumulative sequence lengths for variable-length prefill.
        qk_l2norm: Whether the selected kernel should L2-normalize Q/K.
        output_final_state: Whether to return the final recurrent state.
        output_h: Whether to return intermediate recurrent checkpoints in the
            selected backend's native layout.
        override: Optional kernel override name.
        solution: Optional kernel solution to force through normal selection.

    Returns:
        ``GdnChunkPrefillResult`` with output, final state, and optional
        backend-native recurrent checkpoints.
    """
    head_dim = q.shape[-1]
    head_v_dim = v.shape[-1]
    num_q_heads = q.shape[-2]
    num_v_heads = v.shape[-2]
    traits = {
        "head_dim": head_dim,
        "head_v_dim": head_v_dim,
        "head_v_eq_head_k": head_v_dim == k.shape[-1],
        "num_v_gte_num_q": num_v_heads >= num_q_heads,
        "qk_l2norm": qk_l2norm,
        "output_h": output_h,
    }
    signature = _attention_format_signature(q=q, k=k, v=v)
    kernel = select_kernel(
        "attention",
        "gdn_chunk_prefill",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )

    shape_params = {
        "batch_size": cu_seqlens.shape[0] - 1,
        "total_tokens": q.shape[1] if q.dim() == 4 else q.shape[0],
        "num_q_heads": num_q_heads,
        "num_v_heads": num_v_heads,
        "head_dim": head_dim,
        "head_v_dim": head_v_dim,
    }
    ShapeCapture.get().record(
        "attention",
        "gdn_chunk_prefill",
        kernel.name,
        q.dtype,
        shape_params,
    )

    with kernel_scope(
        "attention",
        "gdn_chunk_prefill",
        q.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            qk_l2norm=qk_l2norm,
            output_final_state=output_final_state,
            output_h=output_h,
        )


# ===-----------------------------------------------------------------------===#
# MHA Kernels
# ===-----------------------------------------------------------------------===#


def mha_prefill(
    # attention inputs
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: list[int],
    max_seqlen: int,
    # attention options
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
    # dispatch options
    override: str | None = None,
    solution: str | None = None,
) -> AttentionResult:
    """MHA prefill from uncached KV.

    Args:
        q: Query tensor with shape [total_q, num_q_heads, head_dim].
        k: Key tensor with shape [total_kv, num_kv_heads, head_dim].
        v: Value tensor with shape [total_kv, num_kv_heads, head_dim].
        cu_seqlens: Cumulative sequence lengths with shape [batch + 1].
            KV cumulative sequence lengths are assumed to be identical.
        cu_seqlens_cpu: Host-side cumulative sequence lengths as a strict
            list[int]. Used for host-side launch metadata; must match cu_seqlens.
        max_seqlen: Maximum sequence length.
        window_left: Exclusive left sliding-window size. -1 means full attention.
        logit_cap: Optional soft cap applied to attention logits.
        sinks: Optional attention sink tensor.
        return_lse: Whether to also return natural-log log-sum-exp values with
            shape [total_q, num_q_heads].
        override: Optional kernel override name.
        solution: Optional kernel solution to force through normal selection.

    Standard full-sequence prefill assumes query and KV sequence boundaries match.
    """
    batch_size = cu_seqlens.shape[0] - 1

    # Select kernel
    traits = {
        "head_dim": q.shape[-1],
        "sliding_window": window_left >= 0,
        "support_logit_cap": logit_cap != 0.0,
        "support_sinks": sinks is not None,
        "return_lse": return_lse,
    }
    signature = _attention_format_signature(q=q, k=k, v=v)
    kernel = select_kernel(
        "attention",
        "mha_prefill",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )

    # Record shapes
    shape_params = {
        "batch_size": batch_size,
        "total_q": q.shape[0],
        "total_kv": k.shape[0],
        "num_q_heads": q.shape[1],
        "num_kv_heads": k.shape[1],
        "head_dim": q.shape[-1],
        "max_seqlen": max_seqlen,
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
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            max_seqlen=max_seqlen,
            window_left=window_left,
            logit_cap=logit_cap,
            sinks=sinks,
            return_lse=return_lse,
        )


def mha_extend_with_kvcache(
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
    # attention options
    is_causal: bool = False,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
    # dispatch options
    override: str | None = None,
    solution: str | None = None,
) -> AttentionResult:
    """MHA extend with paged KV cache.

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
        is_causal: Whether query tokens are a causal suffix of cached KV.
        window_left: Exclusive left sliding-window size. -1 means full attention.
        logit_cap: Optional soft cap applied to attention logits.
        sinks: Optional attention sink tensor.
        return_lse: Whether to also return natural-log log-sum-exp values with
            shape [total_q, num_q_heads].
        override: Optional kernel override name.
        solution: Optional kernel solution to force through normal selection.

    Each request's query tokens attend all visible cached KV tokens.
    """
    # Select kernel
    traits = {
        "head_dim": q.shape[-1],
        "page_size": k_cache.shape[1],
        "is_causal": is_causal,
        "sliding_window": window_left >= 0,
        "support_logit_cap": logit_cap != 0.0,
        "support_sinks": sinks is not None,
        "return_lse": return_lse,
    }
    signature = _attention_format_signature(q=q, k_cache=k_cache, v_cache=v_cache)
    kernel = select_kernel(
        "attention",
        "mha_extend_with_kvcache",
        signature,
        traits=traits,
        solution=solution,
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
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
    }
    ShapeCapture.get().record(
        "attention",
        "mha_extend_with_kvcache",
        kernel.name,
        q.dtype,
        shape_params,
    )

    # Enter profiling scope
    with kernel_scope(
        "attention",
        "mha_extend_with_kvcache",
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
    max_seqlen_q: int,
    # attention options
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
    # dispatch options
    override: str | None = None,
    solution: str | None = None,
) -> AttentionResult:
    """MHA decode with paged KV cache.

    Args:
        q: Query tensor with shape [batch * max_seqlen_q, num_q_heads, head_dim].
        k_cache: Paged key cache with shape [num_pages, page_size, num_kv_heads, head_dim].
        v_cache: Paged value cache with shape [num_pages, page_size, num_kv_heads, head_dim].
        page_table: Page table with shape [batch, max_pages_per_seq].
        cache_seqlens: Total visible KV lengths after appending current decode tokens, shape [batch].
        max_seqlen_k: Maximum KV length.
        max_seqlen_q: Number of uniformly packed query tokens per request. This
            is 1 for normal decode and `spec_num_tokens` for compact
            speculative decode.
        window_left: Exclusive left sliding-window size. -1 means full attention.
        logit_cap: Optional soft cap applied to attention logits.
        sinks: Optional attention sink tensor.
        return_lse: Whether to also return log-sum-exp values.
        override: Optional kernel override name.
        solution: Optional kernel solution to force through normal selection.
    """
    # Select kernel
    traits = {
        "head_dim": q.shape[-1],
        "page_size": k_cache.shape[1],
        "sliding_window": window_left >= 0,
        "support_logit_cap": logit_cap != 0.0,
        "support_sinks": sinks is not None,
        "return_lse": return_lse,
    }
    signature = _attention_format_signature(q=q, k_cache=k_cache, v_cache=v_cache)
    kernel = select_kernel(
        "attention",
        "mha_decode_with_kvcache",
        signature,
        traits=traits,
        solution=solution,
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
        "max_seqlen_q": max_seqlen_q,
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
            window_left=window_left,
            logit_cap=logit_cap,
            sinks=sinks,
            return_lse=return_lse,
            max_seqlen_k=max_seqlen_k,
            max_seqlen_q=max_seqlen_q,
        )


# ===-----------------------------------------------------------------------===#
# MLA Kernels
# ===-----------------------------------------------------------------------===#


def mla_prefill(
    # attention inputs
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    softmax_scale: float,
    # attention options
    seq_lens_kv: torch.Tensor | None = None,
    is_causal: bool = True,
    logit_cap: float = 0.0,
    return_lse: bool = False,
    out: torch.Tensor | None = None,
    # dispatch options
    override: str | None = None,
    solution: str | None = None,
) -> AttentionResult:
    """MLA prefill/cross-attention from explicit, non-cached Q/K/V tensors.

    This API is for the non-absorbed MLA path. Callers materialize full
    per-head K/V before calling this function, so the kernel contract is close
    to MHA ragged attention. It is used for both prompt/new-token causal
    prefill and prefix-cache replay chunks after the compressed MLA cache has
    been read and expanded by the model.

    Args:
        q: Query tensor with shape [total_q, num_q_heads, qk_head_dim], where
            qk_head_dim = qk_nope_head_dim + qk_rope_head_dim.
        k: Key tensor with shape [total_kv, num_kv_heads, qk_head_dim]. For
            DeepSeek MLA prefill today, num_kv_heads is normally num_q_heads
            after expanding the shared RoPE key part across heads.
        v: Value tensor with shape [total_kv, num_kv_heads, v_head_dim].
        cu_seqlens_q: Query cumulative sequence lengths with shape [batch + 1].
        cu_seqlens_kv: KV cumulative sequence lengths with shape [batch + 1].
            This is independent from cu_seqlens_q so prefix-cache chunks can use
            q_lens != kv_lens.
        max_seqlen_q: Maximum query length in the batch.
        max_seqlen_kv: Maximum KV length in the batch.
        softmax_scale: Scale applied to QK logits before softmax.
        seq_lens_kv: Optional per-request KV lengths with shape [batch]. Some
            backends need this in addition to cu_seqlens_kv.
        is_causal: Whether to apply a causal mask between Q and KV. Prefix-cache
            replay chunks should pass False because all prefix tokens precede all
            extend tokens.
        logit_cap: Optional soft cap applied to attention logits.
        return_lse: Whether to also return natural-log log-sum-exp values with
            shape [total_q, num_q_heads]. Required when partial attention states
            will be merged.
        out: Optional output tensor with shape [total_q, num_q_heads, v_head_dim].
        override: Optional kernel override name.
        solution: Optional kernel solution to force through normal selection.

    Returns:
        Attention output with shape [total_q, num_q_heads, v_head_dim], or
        (output, lse) when return_lse is True.
    """
    batch_size = cu_seqlens_q.shape[0] - 1
    traits = {
        "qk_head_dim": q.shape[-1],
        "v_head_dim": v.shape[-1],
        "is_causal": is_causal,
        "support_logit_cap": logit_cap != 0.0,
        "return_lse": return_lse,
    }
    signature = _attention_format_signature(q=q, k=k, v=v)
    kernel = select_kernel(
        "attention",
        "mla_prefill",
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
        "qk_head_dim": q.shape[-1],
        "v_head_dim": v.shape[-1],
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_kv": max_seqlen_kv,
    }
    ShapeCapture.get().record(
        "attention",
        "mla_prefill",
        kernel.name,
        q.dtype,
        shape_params,
    )

    with kernel_scope(
        "attention",
        "mla_prefill",
        q.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            softmax_scale=softmax_scale,
            seq_lens_kv=seq_lens_kv,
            is_causal=is_causal,
            logit_cap=logit_cap,
            return_lse=return_lse,
            out=out,
        )


def mla_decode_with_kvcache(
    # attention inputs
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    # MLA dimensions
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
    # attention options
    logit_cap: float = 0.0,
    return_lse: bool = False,
    out: torch.Tensor | None = None,
    # dispatch options
    override: str | None = None,
    solution: str | None = None,
) -> AttentionResult:
    """MLA absorbed decode over compressed paged MLA KV cache.

    This API is for the absorbed MLA decode path. The model has already
    transformed the non-RoPE query part into latent space using the key half of
    kv_b_proj, so Q and the compressed cache share the same q/k dimension:
    kv_lora_rank + qk_rope_head_dim. The kernel returns the attention-weighted
    latent value; the model applies the value half of kv_b_proj afterward.

    Args:
        q: Absorbed query with shape
            [batch, q_len, num_q_heads, kv_lora_rank + qk_rope_head_dim]. For
            plain decode q_len is 1; speculative/draft paths may pass q_len > 1.
        kv_cache: Paged compressed MLA cache with shape
            [num_pages, page_size, 1, kv_lora_rank + qk_rope_head_dim]. The first
            kv_lora_rank elements are latent KV; the final qk_rope_head_dim
            elements are the RoPE key part.
        page_table: Page table with shape [batch, max_pages_per_seq].
        cache_seqlens: Visible KV lengths in the cache, shape [batch]. These
            lengths include current decode tokens when they were prewritten.
        max_seqlen_k: Maximum visible KV length.
        qk_nope_head_dim: Original non-RoPE q/k head dim. Some backends need
            this for kernel specialization even though q stores the absorbed
            latent dimension.
        kv_lora_rank: MLA latent rank R. The output head dim is R.
        qk_rope_head_dim: RoPE q/k head dim.
        softmax_scale: Scale applied to QK logits before softmax.
        logit_cap: Optional soft cap applied to attention logits.
        return_lse: Whether to also return log-sum-exp values.
        out: Optional output tensor with shape [batch, q_len, num_q_heads,
            kv_lora_rank].
        override: Optional kernel override name.
        solution: Optional kernel solution to force through normal selection.

    Returns:
        Latent attention output with shape [batch, q_len, num_q_heads,
        kv_lora_rank], or (output, lse) when return_lse is True. The caller is
        responsible for applying the MLA value projection from latent rank to
        v_head_dim.
    """
    traits = {
        "page_size": kv_cache.shape[1],
        "q_len": q.shape[1],
        "qk_nope_head_dim": qk_nope_head_dim,
        "kv_lora_rank": kv_lora_rank,
        "qk_rope_head_dim": qk_rope_head_dim,
        "support_logit_cap": logit_cap != 0.0,
        "return_lse": return_lse,
    }
    signature = _attention_format_signature(q=q, kv_cache=kv_cache)
    kernel = select_kernel(
        "attention",
        "mla_decode_with_kvcache",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )

    shape_params = {
        "batch_size": q.shape[0],
        "q_len": q.shape[1],
        "num_q_heads": q.shape[2],
        "num_pages": kv_cache.shape[0],
        "page_size": kv_cache.shape[1],
        "max_pages_per_seq": page_table.shape[1],
        "qk_nope_head_dim": qk_nope_head_dim,
        "kv_lora_rank": kv_lora_rank,
        "qk_rope_head_dim": qk_rope_head_dim,
        "max_seqlen_k": max_seqlen_k,
    }
    ShapeCapture.get().record(
        "attention",
        "mla_decode_with_kvcache",
        kernel.name,
        q.dtype,
        shape_params,
    )

    with kernel_scope(
        "attention",
        "mla_decode_with_kvcache",
        q.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            q=q,
            kv_cache=kv_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            max_seqlen_k=max_seqlen_k,
            qk_nope_head_dim=qk_nope_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            softmax_scale=softmax_scale,
            logit_cap=logit_cap,
            return_lse=return_lse,
            out=out,
        )


# ===-----------------------------------------------------------------------===#
# DSA Kernels
# ===-----------------------------------------------------------------------===#


def dsa_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor | None,
    sparse_kv_cache: torch.Tensor | None,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor | None,
    max_seqlen_k: int,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
    page_size: int,
    q_len_per_req: int = 1,
    logit_cap: float = 0.0,
    k_scale: float = 1.0,
    return_lse: bool = False,
    out: torch.Tensor | None = None,
    override: str | None = None,
    solution: str | None = None,
) -> AttentionResult:
    """Sparse DSA decode over selected global KV slots.

    Args:
        q: Absorbed MLA query with shape [tokens, heads, R + D_rope] or
            [batch, q_len, heads, R + D_rope].
        kv_cache: Regular compressed MLA KV cache, flat [slots, dim] or paged.
        sparse_kv_cache: Packed sparse DSA KV cache, flat [slots, row_bytes] or
            paged.
        topk_slots: Global KV slot ids with shape [tokens, topk]. Invalid
            entries are -1.
        topk_lens: Valid selected-slot count per token, or None when the
            implementation relies on -1 padding.
        max_seqlen_k: Maximum dense visible context length for this batch.
        qk_nope_head_dim: Original non-RoPE q/k dimension.
        kv_lora_rank: MLA latent rank and output head dimension.
        qk_rope_head_dim: RoPE q/k dimension.
        softmax_scale: Scale applied to attention logits.
        page_size: KV cache page size.
        q_len_per_req: Query rows per request.
        logit_cap: Optional logit cap.
        k_scale: KV scale multiplier for FP8 backends.
        return_lse: Whether to return LSE in addition to output.
        out: Optional output buffer.
        override: Optional exact kernel override name.
        solution: Optional kernel solution to force through normal selection.

    Returns:
        Latent DSA attention output, or ``(out, lse)`` when ``return_lse=True``.
    """
    if q.dim() == 4:
        batch_size, q_len, num_heads, head_dim = q.shape
        tokens = batch_size * q_len
    else:
        tokens, num_heads, head_dim = q.shape
        q_len = int(q_len_per_req)
        batch_size = tokens // q_len

    traits = {
        "page_size": int(page_size),
        "q_len_per_req": int(q_len_per_req),
        "qk_nope_head_dim": int(qk_nope_head_dim),
        "kv_lora_rank": int(kv_lora_rank),
        "qk_rope_head_dim": int(qk_rope_head_dim),
        "topk": int(topk_slots.shape[-1]),
        "kv_cache_available": kv_cache is not None,
        "sparse_kv_cache_available": sparse_kv_cache is not None,
        "topk_layout": "global_slots",
        "support_logit_cap": logit_cap != 0.0,
        "return_lse": return_lse,
    }
    signature = _attention_format_signature(q=q)
    kernel = select_kernel(
        "attention",
        "dsa_decode",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )
    shape_params = {
        "batch_size": batch_size,
        "q_len": q_len,
        "tokens": tokens,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "topk": topk_slots.shape[-1],
        "page_size": int(page_size),
        "max_seqlen_k": int(max_seqlen_k),
    }
    ShapeCapture.get().record(
        "attention", "dsa_decode", kernel.name, q.dtype, shape_params
    )
    with kernel_scope(
        "attention", "dsa_decode", q.dtype, kernel_name=kernel.name, **shape_params
    ):
        return kernel(
            q=q,
            kv_cache=kv_cache,
            sparse_kv_cache=sparse_kv_cache,
            topk_slots=topk_slots,
            topk_lens=topk_lens,
            max_seqlen_k=max_seqlen_k,
            qk_nope_head_dim=qk_nope_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            softmax_scale=softmax_scale,
            page_size=page_size,
            q_len_per_req=q_len_per_req,
            logit_cap=logit_cap,
            k_scale=k_scale,
            return_lse=return_lse,
            out=out,
        )


def dsa_prefill(
    q: torch.Tensor,
    kv_cache: torch.Tensor | None,
    sparse_kv_cache: torch.Tensor | None,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor,
    max_seqlen_k: int,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
    page_size: int,
    logit_cap: float = 0.0,
    k_scale: float = 1.0,
    return_lse: bool = False,
    out: torch.Tensor | None = None,
    override: str | None = None,
    solution: str | None = None,
) -> AttentionResult:
    """Sparse DSA prefill over selected global KV slots."""
    if q.dim() == 4:
        batch_size, q_len, num_heads, head_dim = q.shape
        tokens = batch_size * q_len
    else:
        tokens, num_heads, head_dim = q.shape
        q_len = 1
        batch_size = tokens

    traits = {
        "page_size": int(page_size),
        "q_len_per_req": 1,
        "qk_nope_head_dim": int(qk_nope_head_dim),
        "kv_lora_rank": int(kv_lora_rank),
        "qk_rope_head_dim": int(qk_rope_head_dim),
        "topk": int(topk_slots.shape[-1]),
        "kv_cache_available": kv_cache is not None,
        "sparse_kv_cache_available": sparse_kv_cache is not None,
        "topk_layout": "global_slots",
        "support_logit_cap": logit_cap != 0.0,
        "return_lse": return_lse,
    }
    signature = _attention_format_signature(q=q)
    kernel = select_kernel(
        "attention",
        "dsa_prefill",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )
    shape_params = {
        "batch_size": batch_size,
        "q_len": q_len,
        "tokens": tokens,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "topk": topk_slots.shape[-1],
        "page_size": int(page_size),
        "max_seqlen_k": int(max_seqlen_k),
    }
    ShapeCapture.get().record(
        "attention", "dsa_prefill", kernel.name, q.dtype, shape_params
    )
    with kernel_scope(
        "attention", "dsa_prefill", q.dtype, kernel_name=kernel.name, **shape_params
    ):
        return kernel(
            q=q,
            kv_cache=kv_cache,
            sparse_kv_cache=sparse_kv_cache,
            topk_slots=topk_slots,
            topk_lens=topk_lens,
            max_seqlen_k=max_seqlen_k,
            qk_nope_head_dim=qk_nope_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            softmax_scale=softmax_scale,
            page_size=page_size,
            q_len_per_req=1,
            logit_cap=logit_cap,
            k_scale=k_scale,
            return_lse=return_lse,
            out=out,
        )


def dsa_prefill_topk(
    q: torch.Tensor,
    weights: torch.Tensor,
    kv_workspace_slots: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    *,
    topk: int,
    softmax_scale: float,
    index_k_cache: torch.Tensor | None = None,
    page_size: int | None = None,
    index_k_fp8: torch.Tensor | None = None,
    index_k_scale: torch.Tensor | None = None,
    max_logits_bytes: int | None = None,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
    override: str | None = None,
    solution: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute DSA prefill top-k over packed workspace rows.

    Args:
        q: BF16 indexer query with shape [tokens, index_heads, head_dim].
        weights: FP32 per-token/head weights with shape [tokens, index_heads].
        kv_workspace_slots: Global KV slot for each workspace row, shape
            [workspace_rows].
        row_starts: Inclusive workspace-row start per query token, shape [tokens].
        row_ends: Exclusive workspace-row end per query token, shape [tokens].
        topk: Number of workspace candidates to select.
        softmax_scale: Score scale, normally index_head_dim ** -0.5.
        index_k_cache: Packed FP8 index-K cache with scales (uint8). Used by
            Triton with kv_workspace_slots and by DeepGEMM to gather workspace
            rows internally.
        page_size: KV cache page size for index_k_cache.
        index_k_fp8: FP8 index-K rows in workspace-row order. Used by DeepGEMM.
        index_k_scale: FP8 index-K scales in workspace-row order. Used by DeepGEMM.
        max_logits_bytes: Optional temporary logits memory cap.
        out: Optional int32 output buffer with shape [tokens, topk].
        lens_out: Optional int32 output buffer with shape [tokens].
        override: Optional exact kernel override name.
        solution: Optional kernel solution to force through normal selection.

    Returns:
        Tuple of workspace row ids and valid counts. Returned indices are
        absolute row ids into kv_workspace_slots; invalid entries are -1.
    """
    if out is not None and out.shape != (q.shape[0], int(topk)):
        raise ValueError(
            f"out must have shape {(q.shape[0], int(topk))}, got {tuple(out.shape)}"
        )
    if lens_out is not None and lens_out.shape != (q.shape[0],):
        raise ValueError(
            f"lens_out must have shape {(q.shape[0],)}, got {tuple(lens_out.shape)}"
        )
    traits = {
        "head_dim": q.shape[-1],
        "topk": int(topk),
    }
    has_fp8 = index_k_cache is not None or (
        index_k_fp8 is not None and index_k_scale is not None
    )
    if has_fp8:
        traits["index_k_format"] = "fp8_scaled"
    signature = _attention_format_signature(q=q, weights=weights)
    kernel = select_kernel(
        "attention",
        "dsa_prefill_topk",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )
    shape_params = {
        "tokens": q.shape[0],
        "workspace_rows": kv_workspace_slots.numel(),
        "index_heads": q.shape[1],
        "head_dim": q.shape[-1],
        "topk": int(topk),
    }
    ShapeCapture.get().record(
        "attention", "dsa_prefill_topk", kernel.name, q.dtype, shape_params
    )
    with kernel_scope(
        "attention",
        "dsa_prefill_topk",
        q.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            q=q,
            weights=weights,
            kv_workspace_slots=kv_workspace_slots,
            row_starts=row_starts,
            row_ends=row_ends,
            topk=topk,
            softmax_scale=softmax_scale,
            index_k_cache=index_k_cache,
            page_size=page_size,
            index_k_fp8=index_k_fp8,
            index_k_scale=index_k_scale,
            max_logits_bytes=max_logits_bytes,
            out=out,
            lens_out=lens_out,
        )


def dsa_decode_topk(
    q: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    page_size: int,
    topk: int,
    softmax_scale: float,
    q_len_per_req: int = 1,
    index_k_cache: torch.Tensor | None = None,
    seq_lens_2d: torch.Tensor | None = None,
    plan: object | None = None,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
    override: str | None = None,
    solution: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute DSA decode top-k over a paged KV cache.

    Args:
        q: BF16 indexer query with shape [tokens, index_heads, head_dim].
        weights: FP32 per-token/head weights with shape [tokens, index_heads].
        seq_lens: Per-request full KV length, shape [num_reqs] (= tokens /
            q_len_per_req). Each query token's causal bound
            seq_lens[req] - (q_len_per_req - 1) + j is derived in-kernel.
        block_table: Paged KV block table with one row per request,
            shape [num_reqs, max_pages].
        page_size: Number of tokens per KV page.
        topk: Number of KV candidates to select.
        softmax_scale: Score scale, normally index_head_dim ** -0.5.
        q_len_per_req: Query rows per request (spec-verify next_n). Plain
            decode uses 1, where per-request is equivalent to per-token.
        index_k_cache: Packed FP8 index-K cache with scales (uint8). Used by
            both Triton and DeepGEMM.
        plan: Optional opaque backend-specific plan.
        out: Optional int32 output buffer with shape [tokens, topk].
        lens_out: Optional int32 output buffer with shape [tokens].
        override: Optional exact kernel override name.
        solution: Optional kernel solution to force through normal selection.

    Returns:
        Tuple of global KV slots and valid counts; invalid entries are -1.
    """
    if out is not None and out.shape != (q.shape[0], int(topk)):
        raise ValueError(
            f"out must have shape {(q.shape[0], int(topk))}, got {tuple(out.shape)}"
        )
    if lens_out is not None and lens_out.shape != (q.shape[0],):
        raise ValueError(
            f"lens_out must have shape {(q.shape[0],)}, got {tuple(lens_out.shape)}"
        )
    traits = {
        "head_dim": q.shape[-1],
        "topk": int(topk),
        "page_size": int(page_size),
        "q_len_per_req": int(q_len_per_req),
    }
    if index_k_cache is not None:
        traits["index_k_format"] = "fp8_scaled"
    signature = _attention_format_signature(q=q, weights=weights)
    kernel = select_kernel(
        "attention",
        "dsa_decode_topk",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )
    shape_params = {
        "tokens": q.shape[0],
        "max_pages": block_table.shape[1],
        "index_heads": q.shape[1],
        "head_dim": q.shape[-1],
        "page_size": int(page_size),
        "topk": int(topk),
        "q_len_per_req": int(q_len_per_req),
    }
    ShapeCapture.get().record(
        "attention", "dsa_decode_topk", kernel.name, q.dtype, shape_params
    )
    with kernel_scope(
        "attention",
        "dsa_decode_topk",
        q.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            q=q,
            weights=weights,
            seq_lens=seq_lens,
            block_table=block_table,
            page_size=page_size,
            topk=topk,
            softmax_scale=softmax_scale,
            q_len_per_req=q_len_per_req,
            index_k_cache=index_k_cache,
            seq_lens_2d=seq_lens_2d,
            plan=plan,
            out=out,
            lens_out=lens_out,
        )


def dsa_plan(
    *,
    page_size: int,
    seq_lens_2d: torch.Tensor,
    out: object | None = None,
    override: str | None = None,
    solution: str | None = None,
) -> object | None:
    """Build or refresh an opaque plan for DSA decode top-k.

    Args:
        page_size: KV cache page size.
        seq_lens_2d: Prebuilt [num_reqs, next_n] context_lens (last column =
            full per-request KV length), built once per forward by the caller.
        out: Optional previously allocated plan object to refresh in place.
        override: Optional exact kernel override name.
        solution: Optional kernel solution to force through normal selection.

    Returns:
        Opaque backend-owned plan object, or None when no selected backend needs
        an explicit plan.
    """
    if seq_lens_2d.dtype != torch.int32:
        seq_lens_2d = seq_lens_2d.to(torch.int32)
    traits = {
        "page_size": int(page_size),
    }
    signature = format_signature()
    try:
        kernel = select_kernel(
            "attention",
            "dsa_plan",
            signature,
            traits=traits,
            solution=solution,
            override=override,
        )
    except NoKernelFoundError:
        return None

    shape_params = {
        "batch_size": int(seq_lens_2d.shape[0]),
        "tokens": int(seq_lens_2d.numel()),
        "page_size": int(page_size),
    }
    ShapeCapture.get().record(
        "attention", "dsa_plan", kernel.name, seq_lens_2d.dtype, shape_params
    )
    with kernel_scope(
        "attention",
        "dsa_plan",
        seq_lens_2d.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            seq_lens_2d=seq_lens_2d,
            page_size=page_size,
            out=out,
        )


# ===-----------------------------------------------------------------------===#
# Attention Utility Kernels
# ===-----------------------------------------------------------------------===#


def attn_merge_state(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
    *,
    lse_scale_log2: float = LSE_LN,
    inplace: bool = False,
    enable_pdl: bool = False,
    override: str | None = None,
    solution: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge two partial attention states.

    Args:
        out_a: First partial output with shape [total_q, num_heads, head_dim].
        lse_a: First partial log-sum-exp with shape [total_q, num_heads].
        out_b: Second partial output with shape [total_q, num_heads, head_dim].
        lse_b: Second partial log-sum-exp with shape [total_q, num_heads].
        lse_scale_log2: Multiplier that converts input LSE to log2 domain.
        inplace: Whether to write the merged state back into ``out_a``/``lse_a``.
        enable_pdl: Whether the selected backend should enable PDL when supported.
        override: Optional kernel override name.
        solution: Optional kernel solution to force through normal selection.

    This is shared by MHA and MLA because the merge only depends on partial
    attention outputs and LSE values, not on how the K/V states were produced.
    """
    traits = {
        "head_dim": out_a.shape[-1],
    }
    signature = _attention_format_signature(out_a=out_a, out_b=out_b)
    kernel = select_kernel(
        "attention",
        "attn_merge_state",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )

    shape_params = {
        "total_q": out_a.shape[0],
        "num_heads": out_a.shape[1],
        "head_dim": out_a.shape[2],
    }
    ShapeCapture.get().record(
        "attention",
        "attn_merge_state",
        kernel.name,
        out_a.dtype,
        shape_params,
    )

    with kernel_scope(
        "attention",
        "attn_merge_state",
        out_a.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            out_a=out_a,
            lse_a=lse_a,
            out_b=out_b,
            lse_b=lse_b,
            lse_scale_log2=lse_scale_log2,
            inplace=inplace,
            enable_pdl=enable_pdl,
        )


def mha_plan(
    dtype: torch.dtype,
    head_dim: int,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
    solution: str | None = None,
) -> dict:
    """Build a dense MHA execution plan from registered kernel capabilities.

    Args:
        dtype: Query/K/V dtype for prefill planning.
        head_dim: Attention head dimension.
        window_left: Exclusive left sliding-window size, or -1 for full-context
            attention.
        logit_cap: Logit soft-cap value, or 0.0 when disabled.
        sinks: Attention sinks tensor when sinks are enabled.
        return_lse: Whether the selected path must return LSE values.
        solution: Optional kernel solution to restrict planning.

    Returns:
        A dict containing:
        - "extend_mode":
          "postwrite" means run prefill before writing KV cache;
          "prewrite" means write KV cache first and run cached extend.

    FP8 currently prefers "prewrite" because the cache write and downcast
    path is easier to fuse. Other dtypes use "postwrite" only when a
    matching prefill kernel with at least performant priority exists;
    otherwise they use "prewrite".
    """
    if dtype == torch.float8_e4m3fn:
        return {"extend_mode": "prewrite"}

    traits = {
        "head_dim": head_dim,
        "sliding_window": window_left >= 0,
        "support_logit_cap": logit_cap != 0.0,
        "support_sinks": sinks is not None,
        "return_lse": return_lse,
    }
    signature = format_signature(
        q=dense_tensor_format(dtype),
        k=dense_tensor_format(dtype),
        v=dense_tensor_format(dtype),
    )
    candidates = KernelRegistry.get().get_for_operator(
        "attention",
        "mha_prefill",
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

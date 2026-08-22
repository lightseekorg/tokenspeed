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

# Backend registration (side-effect imports)
import tokenspeed_kernel.ops.attention.cuda  # noqa: F401
import tokenspeed_kernel.ops.attention.deep_gemm  # noqa: F401
import tokenspeed_kernel.ops.attention.flash_attn  # noqa: F401
import tokenspeed_kernel.ops.attention.flash_mla  # noqa: F401
import tokenspeed_kernel.ops.attention.flashinfer  # noqa: F401
import tokenspeed_kernel.ops.attention.gluon  # noqa: F401
import tokenspeed_kernel.ops.attention.msa  # noqa: F401
import tokenspeed_kernel.ops.attention.triton  # noqa: F401
import torch
from tokenspeed_kernel.ops.attention.gdn_utils import (
    GdnCheckpointLayout,
    GdnChunkPrefillResult,
)
from tokenspeed_kernel.ops.attention.kda_utils import (
    KdaFusedDecodeResult,
    KdaPrefillResult,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.registry import KernelRegistry, Priority
from tokenspeed_kernel.selection import (
    NoKernelFoundError,
    select_kernel,
    spec_matches_traits,
)
from tokenspeed_kernel.signature import (
    MXFP8_BLOCK_SCALE,
    ScaleFormat,
    dense_tensor_format,
    format_signature,
    tensor_format,
)

AttentionResult = torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]


@dataclass(frozen=True)
class MLAQueryProjection:
    """Projected MLA query and an optional prepared absorb destination.

    When ``absorbed_query`` is present, ``query`` contains only the per-head
    NoPE channels and the RoPE tail of ``absorbed_query`` is populated. The
    absorb BMM owns the remaining latent prefix.
    """

    query: torch.Tensor
    absorbed_query: torch.Tensor | None


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


__all__ = [
    "mha_prefill",
    "mha_extend_with_kvcache",
    "mha_decode_with_kvcache",
    "rel_mha_prefill",
    "rel_mha_extend_with_kvcache",
    "rel_mha_decode_with_kvcache",
    "rel_mha_plan",
    "gdn_chunk_prefill",
    "gdn_decode_step",
    "gdn_decode_mtp",
    "gdn_replay_commit",
    "gdn_replay_commit_supported",
    "kda_paged_prefill",
    "kda_recurrent_layout",
    "kda_paged_decode",
    "try_kda_fused_paged_decode",
    "try_kda_fused_paged_verify",
    "KdaFusedDecodeResult",
    "try_kda_replay_commit",
    "resolve_kda_batched_replay_commit",
    "kda_replay_commit_supported",
    "KdaPrefillResult",
    "GdnCheckpointLayout",
    "GdnChunkPrefillResult",
    "MLAQueryProjection",
    "mla_prefill",
    "mla_use_absorbed_extend",
    "mla_extend_with_kvcache",
    "mla_normalize_project_query",
    "mla_project_value",
    "mla_project_value_prefers_contiguous_weight",
    "mla_decode_with_kvcache",
    "dsa_prefill",
    "dsa_decode",
    "dsa_prefill_topk",
    "dsa_decode_topk",
    "dsa_plan",
    "msa_decode_with_kvcache",
    "msa_extend_with_kvcache",
    "attn_merge_state",
    "mha_plan",
]

LSE_LN = math.log2(math.e)


def mla_project_value_prefers_contiguous_weight(
    *,
    dtype: torch.dtype,
    heads: int,
    latent_dim: int,
    value_dim: int,
    gated: bool = False,
    batch_size: int = 1,
) -> bool:
    """Whether the selected kernel wants a contiguous weight."""
    signature = format_signature(
        attention=dense_tensor_format(dtype),
        weight=dense_tensor_format(dtype),
        out=dense_tensor_format(dtype),
    )
    traits = {
        "batch_size": batch_size,
        "num_heads": heads,
        "latent_dim": latent_dim,
        "value_dim": value_dim,
        "gate_kind": "sigmoid" if gated else "none",
        "inputs_contiguous": True,
    }
    try:
        select_kernel("attention", "mla_project_value", signature, traits=traits)
    except NoKernelFoundError:
        return False
    return True


def mla_project_value(
    attention: torch.Tensor,
    weight: torch.Tensor,
    *,
    gate: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    override: str | None = None,
    solution: str | None = None,
) -> torch.Tensor:
    """Project per-head MLA values and optionally apply a sigmoid gate.

    The headwise projection accumulates in FP32 and is materialized in the
    input dtype before the optional gate, preserving the unfused numerical
    boundary.

    Args:
        attention: Absorbed values shaped ``[batch, heads, latent_dim]``.
        weight: Per-head projection shaped ``[heads, latent_dim, value_dim]``.
        gate: Optional raw sigmoid gate shaped ``[batch, heads * value_dim]``.
        out: Optional output with the same shape as ``gate`` when provided, or
            ``[batch, heads * value_dim]`` otherwise.
        override: Optional exact registered kernel name.
        solution: Optional registered solution name.

    Returns:
        Projected values shaped ``[batch, heads * value_dim]``.
    """
    if attention.ndim != 3 or attention.shape[0] < 1:
        raise ValueError("attention must have shape [batch, heads, latent_dim]")
    if weight.ndim != 3 or weight.shape[:2] != attention.shape[1:]:
        raise ValueError("weight must have shape [heads, latent_dim, value_dim]")
    if attention.dtype != weight.dtype or attention.device != weight.device:
        raise ValueError("attention and weight must match dtype and device")

    batch, heads, latent_dim = attention.shape
    value_dim = weight.shape[2]
    expected_output = (batch, heads * value_dim)
    if gate is not None and (
        tuple(gate.shape) != expected_output
        or gate.dtype != attention.dtype
        or gate.device != attention.device
    ):
        raise ValueError(f"gate must match attention and have shape {expected_output}")
    if out is None:
        out = attention.new_empty(expected_output)
    elif (
        tuple(out.shape) != expected_output
        or out.dtype != attention.dtype
        or out.device != attention.device
        or not out.is_contiguous()
    ):
        raise ValueError(f"out must be contiguous and have shape {expected_output}")

    signature = _attention_format_signature(
        attention=attention,
        weight=weight,
        out=out,
    )
    traits = {
        "batch_size": batch,
        "num_heads": heads,
        "latent_dim": latent_dim,
        "value_dim": value_dim,
        "gate_kind": "none" if gate is None else "sigmoid",
        "inputs_contiguous": (
            attention.is_contiguous()
            and weight.is_contiguous()
            and (gate is None or gate.is_contiguous())
            and out.is_contiguous()
        ),
    }
    try:
        kernel = select_kernel(
            "attention",
            "mla_project_value",
            signature,
            traits=traits,
            solution=solution,
            override=override,
        )
    except NoKernelFoundError:
        if override is not None or solution is not None:
            raise
        kernel = None

    if kernel is None and not traits["inputs_contiguous"]:
        try:
            candidate = select_kernel(
                "attention",
                "mla_project_value",
                signature,
                traits={**traits, "inputs_contiguous": True},
            )
        except NoKernelFoundError:
            candidate = None
        if candidate is not None:
            attention = attention.contiguous()
            weight = weight.contiguous()
            gate = None if gate is None else gate.contiguous()
            kernel = candidate

    if kernel is not None:
        shape_params = {
            "batch_size": batch,
            "num_heads": heads,
            "latent_dim": latent_dim,
            "value_dim": value_dim,
            "gate_kind": traits["gate_kind"],
        }
        ShapeCapture.get().record(
            "attention",
            "mla_project_value",
            kernel.name,
            attention.dtype,
            shape_params,
        )
        with kernel_scope(
            "attention",
            "mla_project_value",
            attention.dtype,
            kernel_name=kernel.name,
            **shape_params,
        ):
            return kernel(attention=attention, weight=weight, gate=gate, out=out)

    output_view = out.view(batch, heads, value_dim)
    if current_platform().is_nvidia:
        torch.bmm(
            attention.transpose(0, 1),
            weight,
            out=output_view.transpose(0, 1),
        )
    else:
        projected = torch.bmm(attention.transpose(0, 1).contiguous(), weight)
        output_view.copy_(projected.transpose(0, 1))
    if gate is not None:
        if out.is_cuda:
            from tokenspeed_kernel.ops.activation.triton import sigmoid_mul

            sigmoid_mul(out, gate)
        else:
            out.copy_(out.float() * torch.sigmoid(gate.float()))
    return out


def mla_normalize_project_query(
    query: torch.Tensor,
    kv: torch.Tensor,
    query_norm_weight: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    projection_weight: torch.Tensor,
    *,
    eps: float,
    prepare_absorbed_query: bool = False,
    qk_nope_head_dim: int | None = None,
    qk_rope_head_dim: int | None = None,
    override: str | None = None,
    solution: str | None = None,
) -> MLAQueryProjection:
    """Normalize MLA query/KV latents and project the normalized query.

    The query normalization is materialized in its input dtype before the
    projection. ``kv`` is normalized in place so callers can retain a view
    into a larger latent-cache tensor.

    Args:
        query: Query latent shaped ``[tokens, query_width]``.
        kv: KV latent shaped ``[tokens, kv_width]``; modified in place.
        query_norm_weight: Query RMSNorm weight shaped ``[query_width]``.
        kv_norm_weight: KV RMSNorm weight shaped ``[kv_width]``.
        projection_weight: Query projection weight shaped
            ``[output_width, query_width]``.
        eps: Positive RMSNorm epsilon.
        prepare_absorbed_query: Whether to prepare the per-head query layout
            consumed by MLA absorb decode when a compatible kernel is available.
        qk_nope_head_dim: Per-head NoPE width required when preparing an
            absorbed query.
        qk_rope_head_dim: Per-head RoPE width required when preparing an
            absorbed query.
        override: Optional exact registered kernel name.
        solution: Optional registered solution name.

    Returns:
        The projected query and an optional absorbed-query destination. The
        ordinary query is shaped ``[tokens, output_width]``. When an absorbed
        destination is returned, the query is instead shaped
        ``[tokens, heads, qk_nope_head_dim]``.
    """
    if query.ndim != 2 or query.shape[0] < 1:
        raise ValueError("query must have shape [tokens, query_width]")
    tokens, query_width = query.shape
    if kv.ndim != 2 or kv.shape[0] != tokens:
        raise ValueError("kv must have shape [tokens, kv_width]")
    kv_width = kv.shape[1]
    output_width = projection_weight.shape[0] if projection_weight.ndim == 2 else 0
    expected = (
        (query_norm_weight, (query_width,), "query_norm_weight"),
        (kv_norm_weight, (kv_width,), "kv_norm_weight"),
        (projection_weight, (output_width, query_width), "projection_weight"),
    )
    for tensor, shape, name in expected:
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}")
    for tensor, name in (
        (kv, "kv"),
        (query_norm_weight, "query_norm_weight"),
        (kv_norm_weight, "kv_norm_weight"),
        (projection_weight, "projection_weight"),
    ):
        if tensor.dtype != query.dtype or tensor.device != query.device:
            raise ValueError(f"{name} must match query dtype and device")
        if tensor.stride(-1) != 1:
            raise ValueError(f"{name} must have unit inner stride")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    num_heads = None
    if prepare_absorbed_query:
        if qk_nope_head_dim is None or qk_nope_head_dim <= 0:
            raise ValueError(
                "qk_nope_head_dim must be positive when preparing an absorbed query"
            )
        if qk_rope_head_dim is None or qk_rope_head_dim <= 0:
            raise ValueError(
                "qk_rope_head_dim must be positive when preparing an absorbed query"
            )
        head_width = qk_nope_head_dim + qk_rope_head_dim
        if output_width % head_width != 0:
            raise ValueError(
                f"projection width {output_width} is not divisible by head width {head_width}"
            )
        num_heads = output_width // head_width

    def select_for_layout(
        *, prefix_width: int, tail_width: int, required: bool = False
    ):
        tensor_roles = {
            "query": dense_tensor_format(query.dtype),
            "kv": dense_tensor_format(kv.dtype),
            "projection_weight": dense_tensor_format(projection_weight.dtype),
            "out": dense_tensor_format(query.dtype),
        }
        split_output = tail_width > 0
        if split_output:
            tensor_roles["tail_out"] = dense_tensor_format(query.dtype)
        signature = format_signature(**tensor_roles)
        traits = {
            "num_tokens": tokens,
            "query_width": query_width,
            "kv_width": kv_width,
            "output_width": output_width,
            "output_prefix_width": prefix_width,
            "output_tail_width": tail_width,
            "split_output": split_output,
            "inputs_contiguous": all(
                tensor.is_contiguous()
                for tensor in (
                    query,
                    kv,
                    query_norm_weight,
                    kv_norm_weight,
                    projection_weight,
                )
            ),
            "outputs_inner_contiguous": True,
        }
        try:
            return select_kernel(
                "attention",
                "mla_normalize_project_query",
                signature,
                traits=traits,
                solution=solution,
                override=override,
            )
        except NoKernelFoundError:
            if required:
                raise
            return None

    kernel = None
    split_selected = False
    if num_heads is not None:
        assert qk_nope_head_dim is not None and qk_rope_head_dim is not None
        kernel = select_for_layout(
            prefix_width=qk_nope_head_dim,
            tail_width=qk_rope_head_dim,
        )
        split_selected = kernel is not None
    if kernel is None:
        kernel = select_for_layout(
            prefix_width=output_width,
            tail_width=0,
            required=override is not None or solution is not None,
        )

    absorbed_query = None
    tail_out = None
    if split_selected:
        assert num_heads is not None
        assert qk_nope_head_dim is not None and qk_rope_head_dim is not None
        out = query.new_empty(tokens, num_heads, qk_nope_head_dim)
        absorbed_query = query.new_empty(tokens, num_heads, kv_width + qk_rope_head_dim)
        tail_out = absorbed_query[..., kv_width:]
    else:
        out = query.new_empty((tokens, output_width))

    if kernel is not None:
        shape_params = {
            "num_tokens": tokens,
            "query_width": query_width,
            "kv_width": kv_width,
            "output_width": output_width,
        }
        ShapeCapture.get().record(
            "attention",
            "mla_normalize_project_query",
            kernel.name,
            query.dtype,
            shape_params,
        )
        with kernel_scope(
            "attention",
            "mla_normalize_project_query",
            query.dtype,
            kernel_name=kernel.name,
            **shape_params,
        ):
            output = kernel(
                query=query,
                kv=kv,
                query_norm_weight=query_norm_weight,
                kv_norm_weight=kv_norm_weight,
                projection_weight=projection_weight,
                eps=eps,
                out=out,
                tail_out=tail_out,
            )
            return MLAQueryProjection(output, absorbed_query)

    projection_out = out
    if query.is_cuda and query.dtype == torch.bfloat16:
        if current_platform().is_amd:
            from tokenspeed_kernel.ops.layernorm.triton import (
                rmsnorm_fused_parallel,
            )
        else:
            from tokenspeed_kernel.ops.layernorm.cuda import rmsnorm_fused_parallel

        from tokenspeed_kernel.ops.gemm.triton_gemv import decode_gemv

        query_norm = torch.empty_like(query)
        rmsnorm_fused_parallel(
            input1=query,
            weight1=query_norm_weight,
            output1=query_norm,
            input2=kv,
            weight2=kv_norm_weight,
            output2=kv,
            eps=eps,
        )
        decode_gemv(query_norm, projection_weight, out=projection_out)
    else:
        query_fp32 = query.float()
        query_norm = query_fp32 * torch.rsqrt(
            query_fp32.square().mean(dim=-1, keepdim=True) + eps
        )
        query_norm = (query_norm * query_norm_weight.float()).to(query.dtype)
        kv_fp32 = kv.float()
        kv_norm = kv_fp32 * torch.rsqrt(
            kv_fp32.square().mean(dim=-1, keepdim=True) + eps
        )
        kv.copy_((kv_norm * kv_norm_weight.float()).to(kv.dtype))
        torch.mm(query_norm, projection_weight.t(), out=projection_out)
    return MLAQueryProjection(out, None)


def msa_decode_with_kvcache(
    q: torch.Tensor,
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    index_k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    *,
    topk: int,
    page_size: int,
    index_scale: float,
    attention_scale: float,
    init_blocks: int,
    local_blocks: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    k_scale: float | torch.Tensor | None = None,
    v_scale: float | torch.Tensor | None = None,
    score_out: torch.Tensor | None = None,
    enable_pdl: bool = False,
    override: str | None = None,
    solution: str | None = None,
) -> torch.Tensor:
    """Run MSA decode against paged K/V and index-key caches.

    Args:
        q: Main queries shaped ``[tokens, local_heads, head_dim]``.
        index_q: Index queries shaped ``[tokens, local_groups, index_dim]``.
        index_k: Index keys for the current tokens shaped
            ``[tokens, index_dim]``.
        k_cache: Paged key cache shaped
            ``[pages, local_kv_heads, page_size, head_dim]``.
        v_cache: Paged value cache with the same shape as ``k_cache``.
        index_k_cache: Per-layer index-key cache shaped
            ``[slots, index_dim]``.
        slot_mapping: Cache slot for each current token.
        page_table: Logical-to-physical page table.
        cache_seqlens: Visible sequence lengths after the current tokens.
        topk: Number of sparse blocks selected for each index query.
        page_size: Number of cache tokens in each indexed block.
        index_scale: Scale applied to index scores.
        attention_scale: Scale applied to main attention scores.
        init_blocks: Leading blocks forced into the selected set.
        local_blocks: Recent blocks forced into the selected set.
        max_seqlen_q: Uniform query-token count per request.
        max_seqlen_k: Maximum KV length addressable through ``page_table``.
        k_scale: Optional scalar descale for an FP8 ``k_cache``; keys were
            divided by this scale before quantization. None means 1.0.
        v_scale: Optional scalar descale for an FP8 ``v_cache``, with the
            same convention as ``k_scale``.
        score_out: Optional caller-owned index-score buffer, pre-filled with
            ``-inf`` and reused across layers; forwarded to the kernel to avoid
            a per-layer allocation + fill. Ignored by kernels that do not
            accept it or when its shape does not match.
        enable_pdl: Request Programmatic Dependent Launch (SM90+) for the
            indexer's index-key store; forwarded to the kernel.
        override: Optional kernel override name.
        solution: Optional kernel solution to force through normal selection.

    Returns:
        Attention output with the same shape and dtype as ``q``. The indexer
        stage also writes ``index_k`` into ``index_k_cache`` at
        ``slot_mapping``.
    """
    traits = {
        "head_dim": q.shape[-1],
        "index_head_dim": index_q.shape[-1],
        "page_size": page_size,
        "topk": topk,
    }
    signature = _attention_format_signature(
        q=q,
        index_q=index_q,
        index_k=index_k,
        k_cache=k_cache,
        v_cache=v_cache,
        index_k_cache=index_k_cache,
    )
    kernel = select_kernel(
        "attention",
        "msa_decode_with_kvcache",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )

    shape_params = {
        "batch_size": cache_seqlens.shape[0],
        "total_q": q.shape[0],
        "num_pages": k_cache.shape[0],
        "page_size": page_size,
        "max_pages_per_seq": page_table.shape[1],
        "num_q_heads": q.shape[1],
        "num_kv_heads": k_cache.shape[1],
        "head_dim": q.shape[-1],
        "index_head_dim": index_q.shape[-1],
        "topk": topk,
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
    }
    ShapeCapture.get().record(
        "attention",
        "msa_decode_with_kvcache",
        kernel.name,
        q.dtype,
        shape_params,
    )

    with kernel_scope(
        "attention",
        "msa_decode_with_kvcache",
        q.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            q=q,
            index_q=index_q,
            index_k=index_k,
            k_cache=k_cache,
            v_cache=v_cache,
            index_k_cache=index_k_cache,
            slot_mapping=slot_mapping,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            topk=topk,
            page_size=page_size,
            index_scale=index_scale,
            attention_scale=attention_scale,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            k_scale=k_scale,
            v_scale=v_scale,
            score_out=score_out,
            enable_pdl=enable_pdl,
        )


def msa_extend_with_kvcache(
    q: torch.Tensor,
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    index_k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    *,
    topk: int,
    page_size: int,
    index_scale: float,
    attention_scale: float,
    init_blocks: int,
    local_blocks: int,
    seq_lens_cpu: Sequence[int],
    k_scale: float | torch.Tensor | None = None,
    v_scale: float | torch.Tensor | None = None,
    query_lens_cpu: Sequence[int] | None = None,
    override: str | None = None,
    solution: str | None = None,
) -> torch.Tensor:
    """Run MSA extend against paged K/V and index-key caches.

    Args:
        q: Main queries shaped ``[total_q, local_heads, head_dim]``.
        index_q: Index queries shaped
            ``[total_q, local_groups, index_dim]``.
        index_k: Index keys for the current tokens shaped
            ``[total_q, index_dim]``.
        k_cache: Paged key cache shaped
            ``[pages, local_kv_heads, page_size, head_dim]``.
        v_cache: Paged value cache with the same shape as ``k_cache``.
        index_k_cache: Per-layer index-key cache shaped
            ``[slots, index_dim]``.
        slot_mapping: Cache slot for each current token.
        page_table: Logical-to-physical page table.
        cache_seqlens: Visible sequence lengths after the current tokens.
        cu_seqlens_q: Cumulative query lengths shaped ``[batch + 1]``.
        prefix_lens: Cached prefix length for each request.
        max_seqlen_q: Maximum query length in the batch.
        max_seqlen_k: Maximum visible KV length in the batch.
        topk: Number of sparse blocks selected for each index query.
        page_size: Number of cache tokens in each indexed block.
        index_scale: Scale applied to index scores.
        attention_scale: Scale applied to main attention scores.
        init_blocks: Leading blocks forced into the selected set.
        local_blocks: Recent blocks forced into the selected set.
        k_scale: Optional scalar descale for an FP8 ``k_cache``; keys were
            divided by this scale before quantization. None means 1.0.
        v_scale: Optional scalar descale for an FP8 ``v_cache``, with the
            same convention as ``k_scale``.
        query_lens_cpu: Optional host-side per-request new-token counts;
            with ``seq_lens_cpu`` this lets the indexer plan its fmha
            OnlyScore path without a device sync.
        seq_lens_cpu: Host-side per-request total sequence lengths.
        override: Optional kernel override name.
        solution: Optional kernel solution to force through normal selection.

    Returns:
        Attention output with the same shape and dtype as ``q``. The indexer
        stage also writes ``index_k`` into ``index_k_cache`` at
        ``slot_mapping``.
    """
    traits = {
        "head_dim": q.shape[-1],
        "index_head_dim": index_q.shape[-1],
        "page_size": page_size,
        "topk": topk,
    }
    signature = _attention_format_signature(
        q=q,
        index_q=index_q,
        index_k=index_k,
        k_cache=k_cache,
        v_cache=v_cache,
        index_k_cache=index_k_cache,
    )
    kernel = select_kernel(
        "attention",
        "msa_extend_with_kvcache",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )

    shape_params = {
        "batch_size": cache_seqlens.shape[0],
        "total_q": q.shape[0],
        "num_pages": k_cache.shape[0],
        "page_size": page_size,
        "max_pages_per_seq": page_table.shape[1],
        "num_q_heads": q.shape[1],
        "num_kv_heads": k_cache.shape[1],
        "head_dim": q.shape[-1],
        "index_head_dim": index_q.shape[-1],
        "topk": topk,
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
    }
    ShapeCapture.get().record(
        "attention",
        "msa_extend_with_kvcache",
        kernel.name,
        q.dtype,
        shape_params,
    )

    with kernel_scope(
        "attention",
        "msa_extend_with_kvcache",
        q.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            q=q,
            index_q=index_q,
            index_k=index_k,
            k_cache=k_cache,
            v_cache=v_cache,
            index_k_cache=index_k_cache,
            slot_mapping=slot_mapping,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            prefix_lens=prefix_lens,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            topk=topk,
            page_size=page_size,
            index_scale=index_scale,
            attention_scale=attention_scale,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            k_scale=k_scale,
            v_scale=v_scale,
            query_lens_cpu=query_lens_cpu,
            seq_lens_cpu=seq_lens_cpu,
        )


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
        initial_state: Recurrent state, K-last: ``[batch, num_v_heads,
            head_v_dim, head_dim]``. This matches flashinfer's native GDN
            decode/MTP layout (and the runtime's SSM state pool); backends
            whose own math is FLA-native (e.g. Triton) transpose internally.
        cu_seqlens: Cumulative sequence lengths for variable-length prefill.
        qk_l2norm: Whether the selected kernel should L2-normalize Q/K.
        output_final_state: Whether to return the final recurrent state.
        output_h: Whether to return intermediate recurrent checkpoints in the
            selected backend's native layout.
        override: Optional kernel override name.
        solution: Optional kernel solution to force through normal selection.

    Returns:
        ``GdnChunkPrefillResult`` with output, final state (K-last, same
        layout as ``initial_state``), and optional backend-native recurrent
        checkpoints (also K-last).
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


def gdn_decode_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: float | None = None,
    output_state_indices: torch.Tensor | None = None,
    use_qk_l2norm: bool = True,
    override: str | None = None,
    solution: str | None = None,
) -> torch.Tensor:
    """Run one single-token (T=1) GDN decode step through kernel selection.

    Args:
        q: Query tensor shaped ``[B, 1, num_q_heads, head_dim]``.
        k: Key tensor shaped ``[B, 1, num_q_heads, head_dim]``.
        v: Value tensor shaped ``[B, 1, num_v_heads, head_v_dim]``.
        A_log: Floating-point log decay parameter shaped ``[num_v_heads]``.
            Backends that require FP32 normalize it internally.
        a: Input-dependent decay shaped ``[B, 1, num_v_heads]``.
        dt_bias: Floating-point decay bias shaped ``[num_v_heads]``. Backends
            that require FP32 normalize it internally.
        b: Update-gate (beta) input shaped ``[B, 1, num_v_heads]``.
        initial_state: SSM state pool, K-last ``[pool_size, num_v_heads,
            head_v_dim, head_dim]`` (matches the runtime's SSM state pool).
        initial_state_indices: Per-batch read row, shaped ``[B]``. ``-1``
            marks CUDA-graph padding; handled internally, no caller clamp
            needed.
        scale: Attention scale. ``None`` lets the implementation use its default.
        output_state_indices: Per-batch write row, shaped ``[B]``. ``None``
            writes back to ``initial_state_indices`` (the common, non-flat
            pool case); pass distinct rows for flat dual-index state paging.
        use_qk_l2norm: Whether the selected kernel should L2-normalize Q/K.
        override: Optional kernel override name.
        solution: Optional kernel solution to force through normal selection.

    Returns:
        Decode output shaped ``[B, 1, num_v_heads, head_v_dim]`` (q.dtype).
    """
    head_dim = q.shape[-1]
    signature = _attention_format_signature(q=q, k=k, v=v)
    kernel = select_kernel(
        "attention",
        "gdn_decode_step",
        signature,
        traits={"head_dim": head_dim},
        solution=solution,
        override=override,
    )
    with kernel_scope(
        "attention",
        "gdn_decode_step",
        q.dtype,
        kernel_name=kernel.name,
        batch_size=q.shape[0],
        num_v_heads=v.shape[-2],
        head_dim=head_dim,
        head_v_dim=v.shape[-1],
    ):
        return kernel(
            q=q,
            k=k,
            v=v,
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            b=b,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            scale=scale,
            output_state_indices=output_state_indices,
            use_qk_l2norm=use_qk_l2norm,
        )


def gdn_decode_mtp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: float | None = None,
    disable_state_update: bool = True,
    use_qk_l2norm: bool = True,
    intermediate_states_buffer: torch.Tensor | None = None,
    output_state_indices: torch.Tensor | None = None,
    override: str | None = None,
    solution: str | None = None,
) -> torch.Tensor:
    """Run one multi-token (T>1) GDN MTP verify step through kernel selection.

    Args:
        q: Query tensor shaped ``[B, T, num_q_heads, head_dim]``.
        k: Key tensor shaped ``[B, T, num_q_heads, head_dim]``.
        v: Value tensor shaped ``[B, T, num_v_heads, head_v_dim]``.
        A_log: Floating-point log decay parameter shaped ``[num_v_heads]``.
            Backends that require FP32 normalize it internally.
        a: Input-dependent decay shaped ``[B, T, num_v_heads]``.
        dt_bias: Floating-point decay bias shaped ``[num_v_heads]``. Backends
            that require FP32 normalize it internally.
        b: Update-gate (beta) input shaped ``[B, T, num_v_heads]``.
        initial_state: SSM state pool, K-last ``[pool_size, num_v_heads,
            head_v_dim, head_dim]`` (matches the runtime's SSM state pool).
        initial_state_indices: Per-batch read row, shaped ``[B]``. When
            ``output_state_indices`` is not provided and
            ``disable_state_update=False``, the final state is written back to
            that same row. Padding handling is solution and state-dtype
            specific: the portable Triton and FlashInfer FP32 paths suppress
            state reads and writes for negative rows, while FlashInfer's BF16
            fast path redirects them to row 0 and requires the caller to
            reserve that row.
        scale: Attention scale. ``None`` lets the implementation use its default.
        disable_state_update: When True (default), never write back to
            ``initial_state_indices``.
        use_qk_l2norm: Whether the selected kernel should L2-normalize Q/K.
        intermediate_states_buffer: Optional batch-scoped ``[B, T,
            num_v_heads, head_v_dim, head_dim]`` (K-last, same dtype as
            ``initial_state``) buffer that receives every step's post-update
            state at ``buffer[i_n, step]``.
        output_state_indices: Optional per-token state-pool destinations shaped
            ``[B, T]`` with dtype ``torch.int32``. When provided, each
            post-update state ``h_{t+1}`` is written directly to
            ``initial_state[output_state_indices[i, t]]``. Negative entries
            are safe only when the selected solution skips the corresponding
            negative initial-state row; otherwise entries must be
            non-negative. This is mutually exclusive with
            ``intermediate_states_buffer`` and requires
            ``disable_state_update=False``.
        override: Optional kernel override name.
        solution: Optional kernel solution to force through normal selection.

    Returns:
        Decode output shaped ``[B, T, num_v_heads, head_v_dim]`` (q.dtype).
    """
    if output_state_indices is not None:
        if output_state_indices.shape != q.shape[:2]:
            raise ValueError(
                "output_state_indices must have shape "
                f"{tuple(q.shape[:2])}, got {tuple(output_state_indices.shape)}"
            )
        if output_state_indices.dtype != torch.int32:
            raise ValueError(
                "output_state_indices must have dtype torch.int32, got "
                f"{output_state_indices.dtype}"
            )
        if intermediate_states_buffer is not None:
            raise ValueError(
                "output_state_indices and intermediate_states_buffer are "
                "mutually exclusive"
            )
        if disable_state_update:
            raise ValueError("output_state_indices requires disable_state_update=False")

    head_dim = q.shape[-1]
    signature = _attention_format_signature(q=q, k=k, v=v)
    kernel = select_kernel(
        "attention",
        "gdn_decode_mtp",
        signature,
        traits={"head_dim": head_dim},
        solution=solution,
        override=override,
    )
    with kernel_scope(
        "attention",
        "gdn_decode_mtp",
        q.dtype,
        kernel_name=kernel.name,
        batch_size=q.shape[0],
        seq_len=q.shape[1],
        num_v_heads=v.shape[-2],
        head_dim=head_dim,
        head_v_dim=v.shape[-1],
    ):
        return kernel(
            q=q,
            k=k,
            v=v,
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            b=b,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            scale=scale,
            disable_state_update=disable_state_update,
            use_qk_l2norm=use_qk_l2norm,
            intermediate_states_buffer=intermediate_states_buffer,
            output_state_indices=output_state_indices,
        )


def gdn_replay_commit(
    payload: torch.Tensor,
    parameters: torch.Tensor,
    *,
    state_addresses: torch.Tensor,
    state_row_strides: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    accepted_length: torch.Tensor,
    draft_token_num: int,
    geometry: tuple[int, int, int, int],
    state_dtype: torch.dtype,
    override: str | None = None,
    solution: str | None = None,
) -> None:
    """Replay every GDN layer's accepted prefix in one kernel launch.

    K/V/a/b share one layer-major allocation. Recurrent slabs may remain
    physically disjoint: ``state_addresses`` and ``state_row_strides`` expose
    them as a layer-indexed table to the kernel. Each program decodes its
    layer, request, and value-head coordinates and writes only the final
    accepted state.

    Args:
        payload: Contiguous packed K/V/a/b storage shaped
            ``[L, token_capacity, H*K + HV*V + 2*HV]``. The first ``B*T``
            rows of each layer hold the current request-major verify window.
        parameters: FP32 A_log/dt_bias table shaped ``[L, 2, HV]``.
        state_addresses: uint64 base-address table shaped ``[L]`` for the
            K-last recurrent-state pools.
        state_row_strides: int64 row strides in elements, shaped ``[L]``.
        read_indices: Committed-state pages shaped ``[L, B]``.
        write_indices: Accepted-state destination pages shaped ``[L, B]``.
        accepted_length: Accepted verified-token count per request, shaped ``[B]``.
        draft_token_num: Number of verify positions per request (``T``).
        geometry: ``(num_k_heads, num_v_heads, head_k_dim, head_v_dim)``.
        state_dtype: Element dtype shared by all recurrent-state pools.
        override: Optional exact registered kernel name.
        solution: Optional registered solution name.
    """
    if payload.dim() != 3 or not payload.is_contiguous():
        raise ValueError("GDN layer replay payload must be contiguous [L, rows, width]")
    num_layers = payload.shape[0]
    batch_size = accepted_length.numel()
    if num_layers == 0 or batch_size == 0:
        return
    num_k_heads, num_v_heads, head_k_dim, head_v_dim = geometry
    if draft_token_num <= 0:
        raise ValueError("draft_token_num must be positive")
    if num_v_heads <= 0 or num_k_heads <= 0 or num_v_heads % num_k_heads:
        raise ValueError("num_v_heads must be divisible by num_k_heads")
    if head_k_dim <= 0 or head_v_dim <= 0:
        raise ValueError("GDN replay head dimensions must be positive")
    payload_width = (
        num_k_heads * head_k_dim + num_v_heads * head_v_dim + 2 * num_v_heads
    )
    if payload.shape[1] < batch_size * draft_token_num:
        raise ValueError("GDN replay payload has insufficient token capacity")
    if payload.shape[2] != payload_width:
        raise ValueError(
            f"GDN replay payload width must be {payload_width}, got {payload.shape[2]}"
        )
    if parameters.shape != (num_layers, 2, num_v_heads):
        raise ValueError(
            "GDN replay parameters must have shape "
            f"{(num_layers, 2, num_v_heads)}, got {tuple(parameters.shape)}"
        )
    if parameters.dtype != torch.float32 or not parameters.is_contiguous():
        raise ValueError("GDN replay parameters must be contiguous torch.float32")
    if state_addresses.shape != (num_layers,) or state_addresses.dtype != torch.uint64:
        raise ValueError(
            "state_addresses must be torch.uint64 with one entry per layer"
        )
    if (
        state_row_strides.shape != (num_layers,)
        or state_row_strides.dtype != torch.int64
    ):
        raise ValueError(
            "state_row_strides must be torch.int64 with one entry per layer"
        )
    if read_indices.shape != (num_layers, batch_size):
        raise ValueError(
            "read_indices must have shape "
            f"{(num_layers, batch_size)}, got {tuple(read_indices.shape)}"
        )
    if write_indices.shape != (num_layers, batch_size):
        raise ValueError(
            "write_indices must have shape "
            f"{(num_layers, batch_size)}, got {tuple(write_indices.shape)}"
        )
    if read_indices.dtype != torch.int32 or write_indices.dtype != torch.int32:
        raise ValueError("GDN replay page tables must have dtype torch.int32")
    if accepted_length.shape != (batch_size,) or accepted_length.dtype != torch.int32:
        raise ValueError("accepted_length must be a one-dimensional torch.int32 tensor")
    if state_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"unsupported GDN replay state dtype: {state_dtype}")
    tensors = (
        parameters,
        state_addresses,
        state_row_strides,
        read_indices,
        write_indices,
        accepted_length,
    )
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError(
            "GDN replay address, stride, index, and parameter tables must be contiguous"
        )
    if any(tensor.device != payload.device for tensor in tensors):
        raise ValueError("all GDN replay tensors must reside on the payload device")

    signature = _attention_format_signature(q=payload, k=payload, v=payload)
    kernel = select_kernel(
        "attention",
        "gdn_replay_commit",
        signature,
        traits={"flat_state": True},
        solution=solution,
        override=override,
    )
    with kernel_scope(
        "attention",
        "gdn_replay_commit",
        payload.dtype,
        kernel_name=kernel.name,
        batch_size=batch_size,
        seq_len=draft_token_num,
        num_layers=num_layers,
        num_v_heads=num_v_heads,
        head_dim=head_k_dim,
        head_v_dim=head_v_dim,
    ):
        kernel(
            payload=payload,
            parameters=parameters,
            state_addresses=state_addresses,
            state_row_strides=state_row_strides,
            read_indices=read_indices,
            write_indices=write_indices,
            accepted_length=accepted_length,
            draft_token_num=draft_token_num,
            num_k_heads=num_k_heads,
            num_v_heads=num_v_heads,
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
            state_dtype=state_dtype,
        )


def gdn_replay_commit_supported(
    dtype: torch.dtype = torch.bfloat16,
    *,
    solution: str | None = None,
) -> bool:
    """Whether ReplaySSM can replace per-draft GDN recurrent-state scratch.

    Args:
        dtype: Activation dtype used by the target verify pass.
        solution: Optional registered solution restriction.

    Returns:
        ``True`` when a compatible GDN replay kernel is registered for the
        current platform.
    """
    probe = torch.empty(0, dtype=dtype, device="meta")
    signature = _attention_format_signature(q=probe, k=probe, v=probe)
    try:
        select_kernel(
            "attention",
            "gdn_replay_commit",
            signature,
            traits={"flat_state": True},
            solution=solution,
        )
    except NoKernelFoundError:
        return False
    return True


def kda_recurrent_layout() -> str:
    """Return the recurrent state layout this platform's KDA kernels consume.

    Returns:
        ``"v_major"`` where the paged slab is ``[pages, HV, V, K]``, else
        ``"k_major"``. K equals V for the supported head geometry, so the two
        differ only in which axis is contiguous.
    """
    platform = current_platform()
    return "v_major" if platform.is_nvidia or platform.is_cdna4 else "k_major"


def kda_paged_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu=None,
    lower_bound: float | None = -5.0,
    override: str | None = None,
    solution: str | None = None,
    recurrent_layout: str | None = None,
) -> KdaPrefillResult:
    """Run packed KDA prefill through capability-based kernel selection.

    Args:
        q/k/g_raw: Packed tensors ``[1, total_tokens, heads, key_dim]``.
        v: Values ``[1, total_tokens, heads, value_dim]``.
        beta_logits: Raw beta logits ``[1, total_tokens, heads]``.
        A_log/dt_bias: FP32 gate parameters.
        initial_state: One backend-owned recurrent state per sequence.
        cu_seqlens: Device sequence boundaries ``[num_sequences + 1]``.
        cu_seqlens_cpu: Optional host copy of ``cu_seqlens`` (tuple, list, or
            CPU tensor) whose contents must equal ``cu_seqlens``.
            Host-planning kernels (CuteDSL) use it to skip the
            stream-synchronizing D2H boundary read; device-planning kernels
            ignore it.
        lower_bound: Optional safe lower bound for log decay.
        override: Optional exact kernel name.
        solution: Optional registered solution name.
        recurrent_layout: Layout of the backend-owned recurrent state; the
            platform default when omitted.

    Returns:
        Packed output and final state, in the caller's ``recurrent_layout``.
    """
    recurrent_layout = recurrent_layout or kda_recurrent_layout()
    if q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("KDA q must be [1, total_tokens, heads, key_dim]")
    if k.shape != q.shape or g_raw.shape != q.shape:
        raise ValueError("KDA q, k, and g_raw must have identical shapes")
    if v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError("KDA v must match q through the head dimension")
    if beta_logits.shape != q.shape[:-1]:
        raise ValueError("KDA beta logits must be [1, total_tokens, heads]")
    num_sequences = cu_seqlens.numel() - 1
    if initial_state.ndim != 4 or initial_state.shape[0] != num_sequences:
        raise ValueError("KDA initial_state must contain one row per sequence")
    if solution == "fla":
        solution = "triton"
    kernel = select_kernel(
        "attention",
        "kda_paged_prefill",
        _attention_format_signature(q=q, k=k, v=v),
        solution=solution,
        override=override,
    )
    spec = KernelRegistry.get().get_by_name(kernel.name)
    supported = None if spec is None else spec.traits.get("recurrent_layout")
    # Kernels that declare no layout consume the caller's state as it is.
    relayout = supported is not None and recurrent_layout not in supported
    if relayout:
        initial_state = initial_state.transpose(-1, -2).contiguous()
    # Forwarded only when set so registered kernels without the
    # host-boundary hint parameter keep working unchanged.
    hint = {} if cu_seqlens_cpu is None else {"cu_seqlens_cpu": cu_seqlens_cpu}
    result = kernel(
        q=q,
        k=k,
        v=v,
        g_raw=g_raw,
        beta_logits=beta_logits,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        lower_bound=lower_bound,
        **hint,
    )
    if relayout:
        # Hand the final state back in the caller's layout (a view; no copy).
        return KdaPrefillResult(result.out, result.final_state.transpose(-1, -2))
    return result


def kda_paged_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    lower_bound: float | None = -5.0,
    override: str | None = None,
    solution: str | None = None,
    recurrent_layout: str | None = None,
) -> torch.Tensor:
    """Run post-convolution KDA decode against an indexed state pool.

    Args:
        q/k/g_raw: Packed tensors ``[1, batch, heads, key_dim]``.
        v: Packed values ``[1, batch, heads, value_dim]``.
        beta_logits: Raw beta logits ``[1, batch, heads]``.
        A_log/dt_bias: FP32 gate parameters.
        state_pool: Backend-owned recurrent-state pool.
        read_indices/write_indices: Independent source/destination rows.
        cu_seqlens: Device boundaries ``[batch + 1]``.
        lower_bound: Optional safe lower bound for log decay.
        override: Optional exact kernel name.
        solution: Optional registered solution name.
        recurrent_layout: Layout of the state pool; the platform default
            when omitted.

    Returns:
        KDA output with the same shape as ``v``.
    """
    recurrent_layout = recurrent_layout or kda_recurrent_layout()
    if q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("KDA decode q must be [1, batch, heads, key_dim]")
    if k.shape != q.shape or g_raw.shape != q.shape:
        raise ValueError("KDA decode q, k, and g_raw must have identical shapes")
    if v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError("KDA decode v must match q through the head dimension")
    if beta_logits.shape != q.shape[:-1]:
        raise ValueError("KDA beta logits must be [1, total_tokens, heads]")
    num_sequences = read_indices.numel()
    if read_indices.ndim != 1 or write_indices.shape != (num_sequences,):
        raise ValueError("KDA decode requires one read/write index per sequence")
    if cu_seqlens.numel() != num_sequences + 1:
        raise ValueError("KDA decode cu_seqlens must contain one boundary per sequence")

    kernel = select_kernel(
        "attention",
        "kda_paged_decode",
        _attention_format_signature(q=q, k=k, v=v),
        traits={
            "indexed_state": True,
            "single_token": q.shape[1] == num_sequences,
            "recurrent_layout": recurrent_layout,
        },
        solution=solution,
        override=override,
    )
    return kernel(
        q=q,
        k=k,
        v=v,
        g_raw=g_raw,
        beta_logits=beta_logits,
        A_log=A_log,
        dt_bias=dt_bias,
        state_pool=state_pool,
        read_indices=read_indices,
        write_indices=write_indices,
        cu_seqlens=cu_seqlens,
        lower_bound=lower_bound,
    )


def try_kda_fused_paged_decode(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_states: torch.Tensor,
    f_a_out: torch.Tensor,
    f_b_weight: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    num_heads: int,
    head_dim: int,
    cu_seqlens: torch.Tensor,
    lower_bound: float | None = -5.0,
    output_gate: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float | None = None,
    recurrent_layout: str | None = None,
    override: str | None = None,
    solution: str | None = None,
) -> KdaFusedDecodeResult | None:
    """Try a registered pre-convolution KDA decode fusion.

    ``output_gate``, ``norm_weight``, and ``norm_eps`` request a fused gated
    RMSNorm epilogue. If the selected backend only supports the original core
    fusion, the returned result reports that the caller must apply the
    epilogue.

    Returns ``None`` only when no implementation supports the current
    platform. Otherwise, returns the output and whether output normalization
    was applied. Invalid inputs and execution failures remain visible.
    """
    recurrent_layout = recurrent_layout or kda_recurrent_layout()
    if (output_gate is None) != (norm_weight is None):
        raise ValueError("output_gate and norm_weight must be provided together")
    if output_gate is not None and norm_eps is None:
        raise ValueError("norm_eps is required with fused KDA output normalization")
    if recurrent_layout not in ("k_major", "v_major"):
        raise ValueError(f"unsupported KDA recurrent layout {recurrent_layout!r}")

    signature = _attention_format_signature(
        q=mixed_qkv,
        k=mixed_qkv,
        v=mixed_qkv,
    )
    try:
        kernel = select_kernel(
            "attention",
            "kda_fused_paged_decode",
            signature,
            traits={
                "paged_state": True,
                "fused_output_norm": output_gate is not None,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "conv_kernel_size": conv_weights.shape[-1],
                "recurrent_layout": recurrent_layout,
            },
            solution=solution,
            override=override,
        )
    except NoKernelFoundError:
        if output_gate is None:
            return None
        try:
            kernel = select_kernel(
                "attention",
                "kda_fused_paged_decode",
                signature,
                traits={
                    "paged_state": True,
                    "fused_output_norm": False,
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                    "conv_kernel_size": conv_weights.shape[-1],
                    "recurrent_layout": recurrent_layout,
                },
                solution=solution,
                override=override,
            )
        except NoKernelFoundError:
            return None

    selected_spec = KernelRegistry.get().get_by_name(kernel.name)
    output_norm_applied = (
        output_gate is not None
        and selected_spec is not None
        and spec_matches_traits(
            selected_spec,
            {"fused_output_norm": True},
            require_all_traits=True,
        )
    )

    out = kernel(
        mixed_qkv=mixed_qkv,
        conv_weights=conv_weights,
        conv_states=conv_states,
        f_a_out=f_a_out,
        f_b_weight=f_b_weight,
        beta_logits=beta_logits,
        A_log=A_log,
        dt_bias=dt_bias,
        state_pool=state_pool,
        read_indices=read_indices,
        write_indices=write_indices,
        num_heads=num_heads,
        head_dim=head_dim,
        cu_seqlens=cu_seqlens,
        lower_bound=lower_bound,
        output_gate=output_gate if output_norm_applied else None,
        norm_weight=norm_weight if output_norm_applied else None,
        norm_eps=norm_eps if output_norm_applied else None,
    )
    return KdaFusedDecodeResult(out=out, output_norm_applied=output_norm_applied)


def try_kda_fused_paged_verify(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_states: torch.Tensor,
    conv_scratch: torch.Tensor,
    f_a_out: torch.Tensor,
    f_b_weight: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    state_scratch: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    num_heads: int,
    head_dim: int,
    draft_token_num: int,
    lower_bound: float | None = -5.0,
    recurrent_layout: str | None = None,
    override: str | None = None,
    solution: str | None = None,
    store_states: bool = True,
) -> torch.Tensor | None:
    """Try a registered pre-convolution KDA target-verify fusion.

    Mirrors ``try_kda_fused_paged_decode`` for the speculative verify batch:
    per-position conv windows and recurrent states land in the verify
    scratches for partial-accept commit. ``store_states`` selects the
    rollback-tape variant and ``recurrent_layout`` defaults to the
    platform's state layout; which producer arrangement runs is the
    registry's choice. Returns ``None`` only when no implementation
    supports the current platform.
    """
    recurrent_layout = recurrent_layout or kda_recurrent_layout()
    if recurrent_layout not in ("k_major", "v_major"):
        raise ValueError(f"unsupported KDA recurrent layout {recurrent_layout!r}")
    signature = _attention_format_signature(
        q=mixed_qkv,
        k=mixed_qkv,
        v=mixed_qkv,
    )
    try:
        kernel = select_kernel(
            "attention",
            "kda_fused_paged_verify",
            signature,
            traits={
                "paged_state": True,
                "store_states": store_states,
                "recurrent_layout": recurrent_layout,
            },
            solution=solution,
            override=override,
        )
    except NoKernelFoundError:
        return None
    return kernel(
        mixed_qkv=mixed_qkv,
        conv_weights=conv_weights,
        conv_states=conv_states,
        conv_scratch=conv_scratch,
        f_a_out=f_a_out,
        f_b_weight=f_b_weight,
        beta_logits=beta_logits,
        A_log=A_log,
        dt_bias=dt_bias,
        state_pool=state_pool,
        state_scratch=state_scratch,
        read_indices=read_indices,
        write_indices=write_indices,
        num_heads=num_heads,
        head_dim=head_dim,
        draft_token_num=draft_token_num,
        lower_bound=lower_bound,
    )


def try_kda_replay_commit(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_states: torch.Tensor,
    conv_out: torch.Tensor,
    f_a_out: torch.Tensor,
    f_b_weight: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    state_out: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    accepted_length: torch.Tensor,
    num_heads: int,
    head_dim: int,
    draft_token_num: int,
    lower_bound: float | None = -5.0,
    override: str | None = None,
    solution: str | None = None,
    gate_scratch: torch.Tensor | None = None,
    recurrent_layout: str | None = None,
) -> bool:
    """Try a registered KDA speculative replay-commit.

    Replays the accepted prefix of a verified draft window from the committed
    page, so the caller never has to keep a recurrent state per draft
    position. Pass the SAME projections the verify pass consumed.
    ``gate_scratch`` is transient fp32 scratch for the hoisted gate
    (``[>= N*T, num_heads*head_dim]``); ``None`` falls back to a
    kernel-module buffer. ``recurrent_layout`` defaults to the platform's
    state layout.

    Returns:
        ``True`` when a kernel ran, ``False`` when none supports the current
        platform (the caller must then fall back to a scratch-based commit).
    """
    recurrent_layout = recurrent_layout or kda_recurrent_layout()
    signature = _attention_format_signature(
        q=mixed_qkv,
        k=mixed_qkv,
        v=mixed_qkv,
    )
    try:
        kernel = select_kernel(
            "attention",
            "kda_replay_commit",
            signature,
            traits={"flat_state": True, "recurrent_layout": recurrent_layout},
            solution=solution,
            override=override,
        )
    except NoKernelFoundError:
        return False
    kernel(
        mixed_qkv=mixed_qkv,
        conv_weights=conv_weights,
        conv_states=conv_states,
        conv_out=conv_out,
        f_a_out=f_a_out,
        f_b_weight=f_b_weight,
        beta_logits=beta_logits,
        A_log=A_log,
        dt_bias=dt_bias,
        state_pool=state_pool,
        state_out=state_out,
        read_indices=read_indices,
        write_indices=write_indices,
        accepted_length=accepted_length,
        num_heads=num_heads,
        head_dim=head_dim,
        draft_token_num=draft_token_num,
        lower_bound=lower_bound,
        gate_scratch=gate_scratch,
    )
    return True


def resolve_kda_batched_replay_commit(dtype: torch.dtype = torch.bfloat16):
    """Resolve the all-layer replay kernel once, or return ``None``.

    ``None`` for any dtype but bfloat16: the batched kernels dereference raw
    descriptor addresses as bf16 (an override resolves by name and skips the
    signature check), so other dtypes must use the per-layer commit, which
    reads its dtypes from the tensors.
    """
    if dtype is not torch.bfloat16:
        return None
    probe = torch.empty(0, dtype=dtype, device="meta")
    signature = _attention_format_signature(q=probe, k=probe, v=probe)
    try:
        return select_kernel(
            "attention",
            "kda_replay_commit",
            signature,
            traits={"flat_state": True, "batched_layers": True},
            override="triton_nvidia_kda_batched_replay_commit",
        )
    except NoKernelFoundError:
        return None


# ===-----------------------------------------------------------------------===#
# MHA Kernels
# ===-----------------------------------------------------------------------===#


def kda_replay_commit_supported(
    dtype: torch.dtype = torch.bfloat16,
    *,
    solution: str | None = None,
    recurrent_layout: str | None = None,
) -> bool:
    """Whether this platform can run the KDA speculative replay path.

    Lets a caller decide up front whether it can skip allocating a
    per-draft-position state scratch, before any verify batch has run. The
    eager replay path has no decomposed fallback, so it needs both the
    standalone commit kernel and the no-store fused verify it rides on.

    Args:
        dtype: activation dtype the verify batch will use.
        solution: restrict to one registered solution, as in ``select_kernel``.
        recurrent_layout: Layout of the committed state; the platform default
            when omitted. It must match what the caller stores, or the probe
            answers for kernels the backend will not select.

    Returns:
        ``True`` when both kernels are registered for the current platform.
    """
    recurrent_layout = recurrent_layout or kda_recurrent_layout()
    probe = torch.empty(0, dtype=dtype, device="meta")
    signature = _attention_format_signature(q=probe, k=probe, v=probe)
    try:
        select_kernel(
            "attention",
            "kda_replay_commit",
            signature,
            traits={"flat_state": True, "recurrent_layout": recurrent_layout},
            solution=solution,
        )
        select_kernel(
            "attention",
            "kda_fused_paged_verify",
            signature,
            traits={
                "paged_state": True,
                "store_states": False,
                "recurrent_layout": recurrent_layout,
            },
            solution=solution,
        )
    except NoKernelFoundError:
        return False
    return True


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
    softmax_scale: float | None = None,
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
        softmax_scale: Scale applied to QK logits before softmax. None uses the
            backend default 1/sqrt(head_dim).
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
            softmax_scale=softmax_scale,
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
    softmax_scale: float | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
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
        softmax_scale: Scale applied to QK logits before softmax. None uses the
            backend default 1/sqrt(head_dim).
        q_scale: MXFP8 block scales for q (UE8M0, one per 32 head_dim
            elements), shape [total_q, num_q_heads, head_dim // 32]. Providing
            it selects the block-scaled path; q/k_cache/v_cache must then be
            float8_e4m3fn.
        k_scale: MXFP8 block scales for k_cache in the kernel's paged layout
            (interleaved [num_pages, num_kv_heads, 32, 4, 4] atom at
            page_size 128).
        v_scale: MXFP8 block scales for v_cache, same layout as k_scale.
        override: Optional kernel override name.
        solution: Optional kernel solution to force through normal selection.

    Each request's query tokens attend all visible cached KV tokens.
    """
    signature, scale_kwargs = _blockscaled_signature_and_scales(
        q, k_cache, v_cache, q_scale, k_scale, v_scale
    )

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
            softmax_scale=softmax_scale,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            **scale_kwargs,
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
    softmax_scale: float | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
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
        softmax_scale: Scale applied to QK logits before softmax. None uses the
            backend default 1/sqrt(head_dim).
        q_scale: MXFP8 block scales for q (UE8M0, one per 32 head_dim
            elements), shape [batch * max_seqlen_q, num_q_heads, head_dim // 32].
            Providing it selects the block-scaled path; q/k_cache/v_cache must
            then be float8_e4m3fn.
        k_scale: MXFP8 block scales for k_cache in the kernel's paged layout
            (interleaved [num_pages, num_kv_heads, 32, 4, 4] atom at
            page_size 128).
        v_scale: MXFP8 block scales for v_cache, same layout as k_scale.
        override: Optional kernel override name.
        solution: Optional kernel solution to force through normal selection.
    """
    signature, scale_kwargs = _blockscaled_signature_and_scales(
        q, k_cache, v_cache, q_scale, k_scale, v_scale
    )

    # Select kernel
    traits = {
        "q_len": max_seqlen_q,
        "head_dim": q.shape[-1],
        "page_size": k_cache.shape[1],
        "sliding_window": window_left >= 0,
        "support_logit_cap": logit_cap != 0.0,
        "support_sinks": sinks is not None,
        "return_lse": return_lse,
    }
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
            softmax_scale=softmax_scale,
            max_seqlen_k=max_seqlen_k,
            max_seqlen_q=max_seqlen_q,
            **scale_kwargs,
        )


# rel_mha: relative-distance-bias MHA; own family keeps model-specific args out of plain mha.


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
    enable_pdl: bool = False,
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
        enable_pdl: Launch eligible kernels with Programmatic Dependent
            Launch (Hopper+).
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
            enable_pdl=enable_pdl,
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
    enable_pdl: bool = False,
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
        enable_pdl: Launch eligible kernels with Programmatic Dependent
            Launch (Hopper+).
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
            enable_pdl=enable_pdl,
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
    enable_pdl: bool = False,
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
        enable_pdl: Launch eligible kernels with Programmatic Dependent
            Launch (Hopper+).
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
            enable_pdl=enable_pdl,
            **scale_kwargs,
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


def mla_use_absorbed_extend(
    *,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    num_q_heads: int,
    page_size: int,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    max_seqlen_q: int | None = None,
    solution: str | None = None,
) -> bool:
    """Return whether a registered kernel supports absorbed MLA extend.

    Args:
        q_dtype: Absorbed query dtype.
        kv_dtype: Compressed KV-cache dtype.
        num_q_heads: Number of local query heads.
        page_size: Number of cache tokens per page.
        qk_nope_head_dim: Original non-RoPE query/key dimension.
        kv_lora_rank: Compressed MLA latent rank.
        qk_rope_head_dim: RoPE query/key dimension.
        max_seqlen_q: Optional maximum query length used to filter kernels whose
            registered shape domain is narrower than their operator API.
        solution: Optional kernel solution to restrict the query.

    Returns:
        Whether the current platform has a matching causal absorbed-extend
        implementation. Kernel registrations remain the source of truth for
        hardware, dtype, and shape support.
    """
    signature = format_signature(
        q=dense_tensor_format(q_dtype),
        kv_cache=dense_tensor_format(kv_dtype),
    )
    traits = {
        "num_q_heads": num_q_heads,
        "page_size": page_size,
        "qk_nope_head_dim": qk_nope_head_dim,
        "kv_lora_rank": kv_lora_rank,
        "qk_rope_head_dim": qk_rope_head_dim,
        "is_causal": True,
        "support_logit_cap": False,
        "return_lse": False,
    }
    if max_seqlen_q is not None:
        traits["max_seqlen_q"] = max_seqlen_q
    candidates = KernelRegistry.get().get_for_operator(
        "attention",
        "mla_extend_with_kvcache",
        platform=current_platform(),
        format_signature=signature,
        solution=solution,
    )
    return any(spec_matches_traits(spec, traits) for spec in candidates)


def mla_extend_with_kvcache(
    # attention inputs
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    # MLA dimensions
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
    # attention options
    is_causal: bool = True,
    logit_cap: float = 0.0,
    return_lse: bool = False,
    out: torch.Tensor | None = None,
    # dispatch options
    override: str | None = None,
    solution: str | None = None,
) -> AttentionResult:
    """MLA multi-token attention over a compressed paged KV cache.

    The model supplies packed absorbed queries and prewrites the current tokens
    to the compressed cache. A zero-length prefix represents initial prefill;
    a nonzero prefix represents cached extend.

    Args:
        q: Packed absorbed query shaped ``[total_q, num_q_heads,
            kv_lora_rank + qk_rope_head_dim]``.
        kv_cache: Compressed paged cache shaped ``[num_pages, page_size, 1,
            kv_lora_rank + qk_rope_head_dim]``.
        page_table: Page table shaped ``[batch, max_pages_per_seq]``.
        cache_seqlens: Total visible KV lengths, including the current query
            tokens, shaped ``[batch]``.
        cu_seqlens_q: Packed query boundaries shaped ``[batch + 1]``.
        cu_seqlens_kv: Packed total-KV boundaries shaped ``[batch + 1]``.
        max_seqlen_q: Maximum query length in the batch.
        max_seqlen_k: Maximum visible KV length in the batch.
        qk_nope_head_dim: Original non-RoPE query/key dimension.
        kv_lora_rank: Compressed MLA latent rank and output head dimension.
        qk_rope_head_dim: RoPE query/key dimension.
        softmax_scale: Scale applied to QK logits.
        is_causal: Whether each query chunk is a causal suffix of its cache.
        logit_cap: Optional soft cap applied to logits.
        return_lse: Whether to return natural-log log-sum-exp values.
        out: Optional output shaped ``[total_q, num_q_heads, kv_lora_rank]``.
        override: Optional exact kernel name.
        solution: Optional kernel solution backend.

    Returns:
        Latent attention output, or ``(output, lse)`` when supported and
        ``return_lse`` is true. The caller applies the MLA value projection.
    """
    batch_size = cache_seqlens.shape[0]
    traits = {
        "page_size": kv_cache.shape[1],
        "num_q_heads": q.shape[1],
        "max_seqlen_q": max_seqlen_q,
        "qk_nope_head_dim": qk_nope_head_dim,
        "kv_lora_rank": kv_lora_rank,
        "qk_rope_head_dim": qk_rope_head_dim,
        "is_causal": is_causal,
        "support_logit_cap": logit_cap != 0.0,
        "return_lse": return_lse,
    }
    signature = _attention_format_signature(q=q, kv_cache=kv_cache)
    kernel = select_kernel(
        "attention",
        "mla_extend_with_kvcache",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )

    shape_params = {
        "batch_size": batch_size,
        "total_q": q.shape[0],
        "num_q_heads": q.shape[1],
        "num_pages": kv_cache.shape[0],
        "page_size": kv_cache.shape[1],
        "max_pages_per_seq": page_table.shape[1],
        "qk_nope_head_dim": qk_nope_head_dim,
        "kv_lora_rank": kv_lora_rank,
        "qk_rope_head_dim": qk_rope_head_dim,
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
    }
    ShapeCapture.get().record(
        "attention",
        "mla_extend_with_kvcache",
        kernel.name,
        q.dtype,
        shape_params,
    )

    with kernel_scope(
        "attention",
        "mla_extend_with_kvcache",
        q.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            q=q,
            kv_cache=kv_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            qk_nope_head_dim=qk_nope_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            softmax_scale=softmax_scale,
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
    # optional projected-value epilogue
    value_weight: torch.Tensor | None = None,
    gate: torch.Tensor | None = None,
) -> AttentionResult:
    """MLA absorbed decode over compressed paged MLA KV cache.

    This API is for the absorbed MLA decode path. The model has already
    transformed the non-RoPE query part into latent space using the key half of
    kv_b_proj, so Q and the compressed cache share the same q/k dimension:
    kv_lora_rank + qk_rope_head_dim. The kernel returns the attention-weighted
    latent value. When ``value_weight`` is provided, a supporting kernel may
    instead apply the value projection and optional output gate while reducing
    the attention partials. Otherwise the API composes the latent decode and
    value projection.

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
            kv_lora_rank]. When ``value_weight`` is provided, this is required
            and has shape [batch, num_q_heads * value_head_dim].
        value_weight: Optional per-head value projection with shape
            [num_q_heads, kv_lora_rank, value_head_dim].
        gate: Optional raw sigmoid gate with shape
            [batch, num_q_heads * value_head_dim]. Requires ``value_weight``.
        override: Optional kernel override name.
        solution: Optional kernel solution to force through normal selection.

    Returns:
        Latent attention output with shape [batch, q_len, num_q_heads,
        kv_lora_rank], or (output, lse) when return_lse is True. When
        ``value_weight`` is provided, returns ``out`` containing the projected
        and optionally gated value.
    """
    if gate is not None and value_weight is None:
        raise ValueError("gate requires value_weight")

    projected_value = value_weight is not None
    if projected_value:
        if q.ndim != 4 or q.shape[1] != 1:
            raise ValueError(
                "projected MLA decode requires q shape [batch,1,heads,dim]"
            )
        if value_weight.ndim != 3 or value_weight.shape[:2] != (
            q.shape[2],
            kv_lora_rank,
        ):
            raise ValueError(
                "value_weight must have shape [heads,kv_lora_rank,value_head_dim]"
            )
        if out is None:
            raise ValueError("projected MLA decode requires out")
        expected_output = (q.shape[0], q.shape[2] * value_weight.shape[2])
        if out.shape != expected_output:
            raise ValueError(f"out must have shape {expected_output}")
        if (
            out.dtype != value_weight.dtype
            or out.device != q.device
            or not out.is_contiguous()
        ):
            raise ValueError(
                "out must match value_weight dtype and be contiguous and colocated with q"
            )
        if gate is not None and gate.shape != expected_output:
            raise ValueError(f"gate must have shape {expected_output}")
        if return_lse:
            raise ValueError("projected MLA decode does not support return_lse")

    traits = {
        "batch_size": q.shape[0],
        "page_size": kv_cache.shape[1],
        "q_len": q.shape[1],
        "num_q_heads": q.shape[2],
        "batch_size_div_64": q.shape[0] % 64 == 0,
        "qk_nope_head_dim": qk_nope_head_dim,
        "kv_lora_rank": kv_lora_rank,
        "qk_rope_head_dim": qk_rope_head_dim,
        "support_logit_cap": logit_cap != 0.0,
        "return_lse": return_lse,
    }
    if projected_value:
        traits.update(
            {
                "value_head_dim": value_weight.shape[2],
                "gate_kind": "none" if gate is None else "sigmoid",
            }
        )
        signature = _attention_format_signature(
            q=q,
            kv_cache=kv_cache,
            value_weight=value_weight,
            out=out,
        )
    else:
        signature = _attention_format_signature(q=q, kv_cache=kv_cache)
    dispatch_mode = (
        "mla_decode_projected_value" if projected_value else "mla_decode_with_kvcache"
    )
    try:
        kernel = select_kernel(
            "attention",
            dispatch_mode,
            signature,
            traits=traits,
            solution=solution,
            override=override,
        )
    except NoKernelFoundError:
        if projected_value:
            if override is not None or solution is not None:
                raise
            attention = mla_decode_with_kvcache(
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
            )
            return mla_project_value(
                attention.reshape(q.shape[0], q.shape[2], kv_lora_rank),
                value_weight,
                gate=gate,
                out=out,
            )
        if q.dtype == kv_cache.dtype:
            raise
        q = q.to(kv_cache.dtype)
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
    if projected_value:
        shape_params["value_head_dim"] = value_weight.shape[2]
    ShapeCapture.get().record(
        "attention",
        dispatch_mode,
        kernel.name,
        q.dtype,
        shape_params,
    )

    with kernel_scope(
        "attention",
        dispatch_mode,
        q.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        if projected_value:
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
                value_weight=value_weight,
                gate=gate,
                out=out,
                logit_cap=logit_cap,
            )
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
    q_scales: torch.Tensor | None = None,
    max_logits_bytes: int | None = None,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
    override: str | None = None,
    solution: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute DSA prefill top-k over packed workspace rows.

    Args:
        q: BF16 or FP8 E4M3 indexer query with shape
            [tokens, index_heads, head_dim]. FP8 queries require q_scales.
        weights: Per-token/head weights with shape [tokens, index_heads],
            FP32 or raw BF16 (implementations upcast on the fly).
        kv_workspace_slots: Global KV slot for each workspace row, shape
            [workspace_rows].
        row_starts: Inclusive workspace-row start per query token, shape [tokens].
        row_ends: Exclusive workspace-row end per query token, shape [tokens].
        topk: Number of workspace candidates to select.
        softmax_scale: Score scale. Each candidate score is exactly
            ``softmax_scale * sum_h(weights[h] * relu(dot(dequant(q[h]), dequant(k))))``.
            BF16 queries are already in their compute representation.
        index_k_cache: Packed paged FP8 index-K cache with scales (uint8). Used
            with kv_workspace_slots to resolve workspace rows inside the
            selected implementation.
        page_size: KV cache page size for index_k_cache.
        index_k_fp8: FP8 index-K rows already in workspace-row order. Must be
            provided together with index_k_scale.
        index_k_scale: FP8 index-K scales already in workspace-row order. Must
            be provided together with index_k_fp8.
        q_scales: Optional positive FP32 scale per token/head for FP8 queries,
            defining ``dequant(q[token, head]) = q[token, head].float() *
            q_scales[token, head]``.
        max_logits_bytes: Optional temporary logits memory cap.
        out: Optional contiguous int32 output buffer on q's device with shape
            [tokens, topk].
        lens_out: Optional contiguous int32 output buffer on q's device with
            shape [tokens].
        override: Optional exact kernel override name.
        solution: Optional kernel solution to force through normal selection.

    Implementations may accept a strided outer weight dimension, including the
    fused model projection view. q, index_k_cache, kv_workspace_slots,
    row_starts, and row_ends must be contiguous on q's device.
    kv_workspace_slots must be int64; row_starts and row_ends must be int32.

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
        "index_heads": q.shape[1],
        "head_dim": q.shape[-1],
        "topk": int(topk),
        "page_size": None if page_size is None else int(page_size),
    }
    has_workspace_rows = index_k_fp8 is not None and index_k_scale is not None
    if (index_k_fp8 is None) != (index_k_scale is None):
        raise ValueError(
            "index_k_fp8 and index_k_scale must be provided together for "
            "workspace-row input"
        )
    has_fp8 = index_k_cache is not None or has_workspace_rows
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
        kernel_kwargs = {
            "q": q,
            "weights": weights,
            "kv_workspace_slots": kv_workspace_slots,
            "row_starts": row_starts,
            "row_ends": row_ends,
            "topk": topk,
            "softmax_scale": softmax_scale,
            "index_k_cache": index_k_cache,
            "page_size": page_size,
            "index_k_fp8": index_k_fp8,
            "index_k_scale": index_k_scale,
            "max_logits_bytes": max_logits_bytes,
            "out": out,
            "lens_out": lens_out,
        }
        if q_scales is not None:
            kernel_kwargs["q_scales"] = q_scales
        return kernel(**kernel_kwargs)


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
    q_scales: torch.Tensor | None = None,
    seq_lens_2d: torch.Tensor | None = None,
    plan: object | None = None,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
    override: str | None = None,
    solution: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute DSA decode top-k over a paged KV cache.

    Args:
        q: BF16 or FP8 E4M3 indexer query with shape
            [tokens, index_heads, head_dim]. FP8 queries require q_scales.
        weights: Per-token/head weights with shape [tokens, index_heads],
            FP32 or raw BF16 (implementations upcast on the fly).
        seq_lens: Per-request full KV length, shape [num_reqs] (= tokens /
            q_len_per_req). Each query token's causal bound
            seq_lens[req] - (q_len_per_req - 1) + j is derived in-kernel.
        block_table: Paged KV block table with one row per request,
            shape [num_reqs, max_pages].
        page_size: Number of tokens per KV page.
        topk: Number of KV candidates to select.
        softmax_scale: Score scale. Each candidate score is exactly
            ``softmax_scale * sum_h(weights[h] * relu(dot(dequant(q[h]), dequant(k))))``.
            BF16 queries are already in their compute representation.
        q_len_per_req: Query rows per request (spec-verify next_n). Plain
            decode uses 1, where per-request is equivalent to per-token.
        index_k_cache: Packed FP8 index-K cache with scales (uint8). Used by
            both Triton and DeepGEMM.
        q_scales: Optional positive FP32 scale per token/head for FP8 queries,
            defining ``dequant(q[token, head]) = q[token, head].float() *
            q_scales[token, head]``.
        plan: Optional opaque backend-specific plan.
        out: Optional contiguous int32 output buffer on q's device with shape
            [tokens, topk].
        lens_out: Optional contiguous int32 output buffer on q's device with
            shape [tokens].
        override: Optional exact kernel override name.
        solution: Optional kernel solution to force through normal selection.

    Implementations may accept a strided outer weight dimension, including the
    fused model projection view. q, index_k_cache, seq_lens, and block_table
    must be contiguous on q's device. seq_lens and block_table must be int32.

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
        "index_heads": q.shape[1],
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
        kernel_kwargs = {
            "q": q,
            "weights": weights,
            "seq_lens": seq_lens,
            "block_table": block_table,
            "page_size": page_size,
            "topk": topk,
            "softmax_scale": softmax_scale,
            "q_len_per_req": q_len_per_req,
            "index_k_cache": index_k_cache,
            "seq_lens_2d": seq_lens_2d,
            "plan": plan,
            "out": out,
            "lens_out": lens_out,
        }
        if q_scales is not None:
            kernel_kwargs["q_scales"] = q_scales
        return kernel(**kernel_kwargs)


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


def _prefill_plan(
    operator: str,
    dtype: torch.dtype,
    traits: dict,
    solution: str | None,
) -> dict:
    """Shared extend-mode planning over a prefill operator's registry entries.

    FP8 currently prefers "prewrite" because the cache write and downcast
    path is easier to fuse. Other dtypes use "postwrite" only when a
    matching prefill kernel with at least performant priority exists;
    otherwise they use "prewrite".
    """
    if dtype == torch.float8_e4m3fn:
        return {"extend_mode": "prewrite"}

    signature = format_signature(
        q=dense_tensor_format(dtype),
        k=dense_tensor_format(dtype),
        v=dense_tensor_format(dtype),
    )
    candidates = KernelRegistry.get().get_for_operator(
        "attention",
        operator,
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
    """
    traits = {
        "head_dim": head_dim,
        "sliding_window": window_left >= 0,
        "support_logit_cap": logit_cap != 0.0,
        "support_sinks": sinks is not None,
        "return_lse": return_lse,
    }
    return _prefill_plan("mha_prefill", dtype, traits, solution)


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
    traits = {
        "head_dim": head_dim,
        "sliding_window": window_left >= 0,
        "return_lse": return_lse,
    }
    return _prefill_plan("rel_mha_prefill", dtype, traits, solution)

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
from tokenspeed_kernel.platform import pdl_enabled
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
    kv_seq_lens: torch.Tensor | None = None,
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
        kv_seq_lens: Optional physical KV length for every query row. Sparse
            backends that gather direct global slots use this separately from
            ``topk_lens``, which describes the selected sparse width.
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
            kv_seq_lens=kv_seq_lens,
            logit_cap=logit_cap,
            k_scale=k_scale,
            return_lse=return_lse,
            out=out,
            enable_pdl=pdl_enabled(),
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
    kv_seq_lens: torch.Tensor | None = None,
) -> AttentionResult:
    """Sparse DSA prefill over selected global KV slots.

    Args:
        q: Absorbed MLA query with shape [tokens, heads, R + D_rope] or
            [batch, q_len, heads, R + D_rope].
        kv_cache: Regular compressed MLA KV cache, flat [slots, dim] or paged.
        sparse_kv_cache: Packed sparse DSA KV cache, flat [slots, row_bytes] or
            paged.
        topk_slots: Global KV slot ids with shape [tokens, topk]. Invalid
            entries are -1.
        topk_lens: Valid selected-slot count per token.
        max_seqlen_k: Maximum dense visible context length for this batch.
        qk_nope_head_dim: Original non-RoPE q/k dimension.
        kv_lora_rank: MLA latent rank and output head dimension.
        qk_rope_head_dim: RoPE q/k dimension.
        softmax_scale: Scale applied to attention logits.
        page_size: KV cache page size.
        kv_seq_lens: Optional physical KV length for every query row. Sparse
            backends that gather direct global slots use this separately from
            ``topk_lens``, which describes the selected sparse width.
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
            kv_seq_lens=kv_seq_lens,
            logit_cap=logit_cap,
            k_scale=k_scale,
            return_lse=return_lse,
            out=out,
            enable_pdl=pdl_enabled(),
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
    candidate_lens_cpu: torch.Tensor | None = None,
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
        index_k_cache: Packed or page-planar FP8 index-K cache with scales
            (uint8). Page-planar caches may have a padded outer page stride.
            Used with kv_workspace_slots to resolve workspace rows inside the
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
        candidate_lens_cpu: Optional CPU mirror of ``row_ends - row_starts``.
            DeepGEMM uses it to select chunk launch bounds without synchronizing
            the CUDA stream; other implementations ignore it.
        out: Optional contiguous int32 output buffer on q's device with shape
            [tokens, topk].
        lens_out: Optional contiguous int32 output buffer on q's device with
            shape [tokens].
        override: Optional exact kernel override name.
        solution: Optional kernel solution to force through normal selection.

    Implementations may accept a strided outer weight dimension, including the
    fused model projection view. q, kv_workspace_slots, row_starts, and row_ends
    must be contiguous on q's device. index_k_cache may be a contiguous packed
    slot matrix or a page-planar matrix with contiguous bytes within each page.
    kv_workspace_slots must be int64; row_starts and row_ends must be int32.

    Returns:
        Tuple of workspace row ids and valid counts. Returned indices are
        absolute row ids into kv_workspace_slots; invalid entries are -1.
    """
    if candidate_lens_cpu is not None and (
        candidate_lens_cpu.device.type != "cpu"
        or candidate_lens_cpu.shape != (q.shape[0],)
    ):
        raise ValueError(
            "candidate_lens_cpu must be a CPU tensor with shape "
            f"{(q.shape[0],)}, got device={candidate_lens_cpu.device}, "
            f"shape={tuple(candidate_lens_cpu.shape)}"
        )
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
    if index_k_cache is not None:
        row_bytes = q.shape[-1] + q.shape[-1] // 128 * 4
        traits["index_k_layout"] = (
            "packed"
            if index_k_cache.ndim == 2 and index_k_cache.shape[1] == row_bytes
            else "page_planar"
        )
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
        if candidate_lens_cpu is not None and kernel.name.startswith("deep_gemm_"):
            kernel_kwargs["candidate_lens_cpu"] = candidate_lens_cpu
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
    topk_layout: str = "global_slots",
    block_table_base_offsets: torch.Tensor | None = None,
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
        topk_layout: Return physical cache slots when ``global_slots`` or
            absolute logical row offsets when ``logical_offsets``.
        block_table_base_offsets: Optional compact-table base page per request.
            Used only with ``topk_layout="logical_offsets"``.
        index_k_cache: Packed or page-planar FP8 index-K cache with scales
            (uint8). Page-planar caches may have a padded outer page stride.
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
    fused model projection view. q, seq_lens, and block_table must be contiguous
    on q's device. index_k_cache may be a contiguous packed slot matrix or a
    page-planar matrix with contiguous bytes within each page. seq_lens and
    block_table must be int32.

    Returns:
        Tuple of selected indices and valid counts. Indices are global KV slots
        or absolute logical offsets according to ``topk_layout``; invalid
        entries are -1.
    """
    if out is not None and out.shape != (q.shape[0], int(topk)):
        raise ValueError(
            f"out must have shape {(q.shape[0], int(topk))}, got {tuple(out.shape)}"
        )
    if topk_layout not in ("global_slots", "logical_offsets"):
        raise ValueError(
            "topk_layout must be 'global_slots' or 'logical_offsets', got "
            f"{topk_layout!r}"
        )
    if q_len_per_req < 1 or q.shape[0] % int(q_len_per_req) != 0:
        raise ValueError(
            f"q_len_per_req={q_len_per_req} must divide tokens={q.shape[0]}"
        )
    if block_table_base_offsets is not None and topk_layout != "logical_offsets":
        raise ValueError(
            "block_table_base_offsets requires topk_layout='logical_offsets'"
        )
    kernel_seq_lens = seq_lens
    if block_table_base_offsets is not None:
        num_reqs = q.shape[0] // int(q_len_per_req)
        if (
            block_table_base_offsets.ndim != 1
            or block_table_base_offsets.numel() < num_reqs
            or block_table_base_offsets.device != seq_lens.device
        ):
            raise ValueError(
                "block_table_base_offsets must have one entry per request on "
                "the same device as seq_lens"
            )
        kernel_seq_lens = (
            (
                seq_lens.to(torch.int64)
                - block_table_base_offsets[:num_reqs].to(torch.int64) * int(page_size)
            )
            .clamp(0, int(block_table.shape[1]) * int(page_size))
            .to(torch.int32)
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
        row_bytes = q.shape[-1] + q.shape[-1] // 128 * 4
        traits["index_k_layout"] = (
            "packed"
            if index_k_cache.ndim == 2 and index_k_cache.shape[1] == row_bytes
            else "page_planar"
        )
    signature = _attention_format_signature(q=q, weights=weights)
    kernel = select_kernel(
        "attention",
        "dsa_decode_topk",
        signature,
        traits=traits,
        features=(
            frozenset({"logical_offsets"}) if topk_layout == "logical_offsets" else None
        ),
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
            "seq_lens": kernel_seq_lens,
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
        if topk_layout == "logical_offsets":
            kernel_kwargs["topk_layout"] = topk_layout
            kernel_kwargs["block_table_base_offsets"] = block_table_base_offsets
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
    traits = {"page_size": int(page_size)}
    try:
        kernel = select_kernel(
            "attention",
            "dsa_plan",
            format_signature(),
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


# Backend registration (side-effect imports)
# isort: off
import tokenspeed_kernel.ops.attention.dsa.cuda  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.dsa.cute_dsl  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.dsa.deep_gemm  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.dsa.flashinfer  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.dsa.triton  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.dsa.gluon  # noqa: E402,F401

# isort: on

__all__ = [
    "dsa_decode",
    "dsa_prefill",
    "dsa_prefill_topk",
    "dsa_decode_topk",
    "dsa_plan",
]

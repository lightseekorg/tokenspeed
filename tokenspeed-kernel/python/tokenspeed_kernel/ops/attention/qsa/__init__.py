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

import torch
from tokenspeed_kernel.selection import select_kernel
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


def select_qsa_sparse_attention_kernel(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    selected_slots: torch.Tensor,
    *,
    max_seqlen_q: int | None,
    override: str | None,
    solution: str | None,
):
    """Select the ``qsa_sparse_attention`` kernel for validated inputs.

    Arguments follow ``qsa_sparse_attention``; ``q`` must have at least one row.

    Returns:
        The selected kernel.
    """

    query_width = 1 if max_seqlen_q is None else max_seqlen_q
    traits = {
        "batch_size": q.shape[0] // query_width,
        "q_len": query_width,
        "num_q_heads": q.shape[1],
        "num_kv_heads": k_cache.shape[1],
        "head_dim": q.shape[-1],
        "value_head_dim": v_cache.shape[-1],
        "selected_width": selected_slots.shape[1],
        "is_decode": max_seqlen_q is not None,
    }
    signature = _attention_format_signature(q=q, k_cache=k_cache, v_cache=v_cache)
    return select_kernel(
        "attention",
        "qsa_sparse_attention",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )


def qsa_sparse_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    selected_slots: torch.Tensor,
    *,
    scale: float,
    max_seqlen_q: int | None,
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
            ``[cache_slots, kv_heads, head_dim]``. Slot 1 must contain finite
            values for masked reads; slot 0 may contain NaN/Inf.
        v_cache: Flattened value cache shaped
            ``[cache_slots, kv_heads, value_head_dim]`` with the same slot 1
            requirement. Both caches must have at least two slots.
        selected_slots: Physical cache slots shaped ``[tokens, budget]``;
            non-positive values are ignored.
        scale: Softmax scale applied to query-key scores.
        max_seqlen_q: Uniform query tokens per request for decode: 1 for
            normal decode and ``spec_num_tokens`` for compact speculative
            decode. Pass None for prefill or mixed/ragged queries, including
            a prefill containing only one token.
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

    The SM100/SM103 CuTe DSL implementation is preferred when its specialization
    matches a uniform decode. Prefill and mixed/ragged queries use FlashInfer
    FA2 on supported NVIDIA architectures; other decode geometries retain
    the registered fallback. GFX950 uses the Gluon kernel and other AMD GPUs
    the portable Triton kernel. Kernel
    selection preserves this distinction before adapting ragged inputs to
    independent one-token query rows.
    """

    if q.ndim != 3 or k_cache.ndim != 3 or v_cache.ndim != 3:
        raise ValueError("QSA sparse attention expects rank-three Q/K/V tensors")
    if selected_slots.ndim != 2 or selected_slots.shape[0] != q.shape[0]:
        raise ValueError("QSA selected slots must have one row per query token")
    query_width = 1 if max_seqlen_q is None else max_seqlen_q
    if query_width < 1:
        raise ValueError("QSA max_seqlen_q must be positive")
    if q.shape[0] % query_width:
        raise ValueError("QSA query rows must be divisible by max_seqlen_q")
    if q.shape[0] == 0:
        return q.new_empty((0, q.shape[1], v_cache.shape[-1]))
    if k_cache.shape[0] < 2 or v_cache.shape[0] < 2:
        raise ValueError("QSA caches must include slot 1 for masked reads")
    kernel = select_qsa_sparse_attention_kernel(
        q,
        k_cache,
        v_cache,
        selected_slots,
        max_seqlen_q=max_seqlen_q,
        override=override,
        solution=solution,
    )
    return kernel(
        q,
        k_cache,
        v_cache,
        selected_slots,
        scale=scale,
        max_seqlen_q=query_width,
        metadata_capacity_rows=metadata_capacity_rows,
        k_scale=k_scale,
        v_scale=v_scale,
    )


# ===-----------------------------------------------------------------------===#
# QSA Indexer Block Selection
# ===-----------------------------------------------------------------------===#


def qsa_block_scores(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    page_table: torch.Tensor,
    request_indices: torch.Tensor,
    complete_blocks: torch.Tensor,
    *,
    page_size: int,
    queries_per_request: int | None,
    enable_pdl: bool,
    override: str | None,
    solution: str | None,
) -> torch.Tensor:
    """Score every compressed QSA block for each query row.

    Args:
        query: Index queries shaped ``[rows, heads, head_dim]`` with a
            contiguous last dimension.
        key_cache: Compressed keys shaped ``[slots, 1, head_dim]``.
        page_table: Compressed-cache page ids shaped ``[requests, pages]``.
        request_indices: Owning request per query row.
        complete_blocks: Fully compressed block count per row.
        page_size: Compressed rows per page.
        queries_per_request: Uniform consecutive rows per request, or None
            for ragged layouts; implementations may share key tiles within
            these runs.
        enable_pdl: Use programmatic dependent launch where supported.
        override: Optional registered kernel name.
        solution: Optional kernel solution.

    Returns:
        FP32 scores shaped ``[rows, pages * page_size]``: the summed ReLU of
        the head dot products below each row's ``complete_blocks`` and
        ``-inf`` elsewhere.
    """

    kernel = select_kernel(
        "attention",
        "qsa_block_scores",
        format_signature(
            query=dense_tensor_format(query.dtype),
            key_cache=dense_tensor_format(key_cache.dtype),
        ),
        traits={"num_heads": query.shape[1], "head_dim": query.shape[2]},
        solution=solution,
        override=override,
    )
    return kernel(
        query,
        key_cache,
        page_table,
        request_indices,
        complete_blocks,
        page_size=page_size,
        queries_per_request=queries_per_request,
        enable_pdl=enable_pdl,
    )


def qsa_block_topk_select(
    logits: torch.Tensor,
    topk: int,
    *,
    lengths: torch.Tensor,
    persistent_workspace: torch.Tensor | None,
    enable_pdl: bool,
    override: str | None,
    solution: str | None,
) -> torch.Tensor:
    """Select each row's ``topk`` highest finite block scores.

    Args:
        logits: FP32 scores shaped ``[rows, cols]``; ``-inf`` is never
            selected.
        topk: Blocks selected per row.
        lengths: Per-row count of scored columns; later columns are ``-inf``.
        persistent_workspace: Optional caller-owned CUDA uint8 workspace of
            at least 1 MiB for implementations that keep persistent state.
        enable_pdl: Use programmatic dependent launch where supported.
        override: Optional registered kernel name.
        solution: Optional kernel solution.

    Returns:
        Int32 column ids shaped ``[rows, topk]`` in no particular order,
        padded with ``-1`` when a row has fewer finite scores. Among equal
        scores at the threshold the smaller id is kept.
    """

    kernel = select_kernel(
        "attention",
        "qsa_block_topk_select",
        format_signature(logits=dense_tensor_format(logits.dtype)),
        traits={"topk": topk},
        solution=solution,
        override=override,
    )
    return kernel(
        logits,
        topk,
        lengths=lengths,
        persistent_workspace=persistent_workspace,
        enable_pdl=enable_pdl,
    )


def qsa_block_topk_stream(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    page_table: torch.Tensor,
    request_indices: torch.Tensor,
    complete_blocks: torch.Tensor,
    *,
    page_size: int,
    block_topk: int,
    max_partial_bytes: int,
    enable_pdl: bool,
    override: str | None,
    solution: str | None,
) -> torch.Tensor:
    """Score and select each row's top-k QSA blocks without materializing scores.

    Args:
        query: Index queries shaped ``[rows, heads, head_dim]``.
        key_cache: Compressed keys shaped ``[slots, 1, head_dim]``.
        page_table: Compressed-cache page ids shaped ``[requests, pages]``.
        request_indices: Owning request per query row.
        complete_blocks: Fully compressed block count per row.
        page_size: Compressed rows per page.
        block_topk: Blocks selected per row.
        max_partial_bytes: Memory budget for partial top-k buffers.
        enable_pdl: Use programmatic dependent launch where supported.
        override: Optional registered kernel name.
        solution: Optional kernel solution.

    Returns:
        Int32 block ids shaped ``[rows, block_topk]`` in no particular order;
        invalid entries are ``-1``.
    """

    kernel = select_kernel(
        "attention",
        "qsa_block_topk_stream",
        format_signature(
            query=dense_tensor_format(query.dtype),
            key_cache=dense_tensor_format(key_cache.dtype),
        ),
        traits={"num_heads": query.shape[1], "head_dim": query.shape[2]},
        solution=solution,
        override=override,
    )
    return kernel(
        query,
        key_cache,
        page_table,
        request_indices,
        complete_blocks,
        page_size=page_size,
        block_topk=block_topk,
        max_partial_bytes=max_partial_bytes,
        enable_pdl=enable_pdl,
    )


def qwen4_exp_qsa_block_topk(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    page_table: torch.Tensor,
    request_indices: torch.Tensor,
    complete_blocks: torch.Tensor,
    *,
    page_size: int,
    block_topk: int,
    queries_per_request: int | None,
    max_partial_bytes: int,
    algorithm: str,
    persistent_topk_workspace: torch.Tensor | None,
    enable_pdl: bool,
) -> torch.Tensor:
    """Select the highest-scoring compressed blocks for each query row.

    Args:
        query: Query tensor shaped ``[rows, heads, head_dim]``.
        key_cache: Flattened compressed keys shaped ``[slots, 1, head_dim]``.
        page_table: Raw compressed-cache block ids shaped ``[requests, max_pages]``,
            with holes and padding normalized to zero; one entry per logical page.
        request_indices: Owning request id per query row.
        complete_blocks: Fully compressed block counts shaped ``[rows]``.
        page_size: Compressed-cache rows covered by one logical page.
        block_topk: Blocks selected per row; must be a power of two and at
            least 64.
        queries_per_request: Uniform number of consecutive query rows per
            request, or None for ragged/mixed layouts. A positive width must
            divide ``rows``; the caller guarantees that each width-sized run
            has one owning request. Logits scoring shares K tiles within these
            runs while preserving each row's complete-block frontier.
        max_partial_bytes: Memory budget for the partial top-k buffers
            (``stream`` algorithm only).
        algorithm: ``"stream"`` fuses scoring and selection without
            materializing scores; ``"logits"`` materializes the
            ``[rows, num_blocks]`` FP32 scores with ``qsa_block_scores`` and
            selects them with ``qsa_block_topk_select``.
        persistent_topk_workspace: Optional caller-owned CUDA uint8 workspace
            of at least 1 MiB. The ``"logits"`` algorithm uses it for the
            length-aware persistent radix top-k when that kernel is available;
            otherwise selection falls back to portable Triton.
        enable_pdl: Wait for query/cache/metadata producers and release
            successor setup early between scoring and selection on NVIDIA GPUs.

    Returns:
        Int32 block ids shaped ``[rows, block_topk]``, in no particular order;
        invalid entries are ``-1``.
    """

    if query.ndim != 3 or key_cache.ndim != 3:
        raise ValueError("Qwen4-Exp QSA block top-k expects rank-three tensors")
    if block_topk < 64 or (block_topk & (block_topk - 1)):
        raise ValueError("Qwen4-Exp QSA block top-k needs a power-of-two topk >= 64")
    if algorithm not in ("stream", "logits"):
        raise ValueError(
            "Qwen4-Exp QSA block top-k algorithm must be 'stream' or 'logits', "
            f"got {algorithm!r}"
        )
    rows = query.shape[0]
    if queries_per_request is not None and (
        queries_per_request < 1 or rows % queries_per_request
    ):
        raise ValueError("QSA queries_per_request must be positive and divide rows")
    num_blocks = page_table.shape[1] * page_size
    if rows == 0 or num_blocks == 0:
        return torch.full(
            (rows, block_topk), -1, dtype=torch.int32, device=query.device
        )
    if algorithm == "logits":
        scores = qsa_block_scores(
            query,
            key_cache,
            page_table,
            request_indices,
            complete_blocks,
            page_size=page_size,
            queries_per_request=queries_per_request,
            enable_pdl=enable_pdl,
            override=None,
            solution=None,
        )
        return qsa_block_topk_select(
            scores,
            block_topk,
            lengths=complete_blocks,
            persistent_workspace=persistent_topk_workspace,
            enable_pdl=enable_pdl,
            override=None,
            solution=None,
        )
    return qsa_block_topk_stream(
        query,
        key_cache,
        page_table,
        request_indices,
        complete_blocks,
        page_size=page_size,
        block_topk=block_topk,
        max_partial_bytes=max_partial_bytes,
        enable_pdl=enable_pdl,
        override=None,
        solution=None,
    )


# Backend registration (side-effect imports)
# isort: off
import tokenspeed_kernel.ops.attention.qsa.triton  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.qsa.cute_dsl  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.qsa.flashinfer  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.qsa.gluon  # noqa: E402,F401

# isort: on


__all__ = [
    "qsa_block_scores",
    "qsa_block_topk_select",
    "qsa_block_topk_stream",
    "qsa_sparse_attention",
    "qwen4_exp_qsa_block_topk",
]

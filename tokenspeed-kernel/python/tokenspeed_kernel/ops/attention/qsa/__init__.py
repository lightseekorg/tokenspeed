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
from tokenspeed_kernel.ops.attention.qsa.metadata import QSASparseMetadata, QSAWorkspace
from tokenspeed_kernel.ops.tuning import get_autotune_max_num_tokens, is_warming_up
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import resolve_kernel_override, select_kernel
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


def _warmup_uses_cute_dsl(
    mode: str,
    kernel_name: str,
    *,
    override: str | None,
    solution: str | None,
) -> bool:
    """Apply dispatch precedence: a resolved override takes priority over solution."""
    effective_override = resolve_kernel_override("attention", mode, override)
    if effective_override is not None:
        return effective_override in ("cute_dsl", kernel_name)
    return solution in (None, "cute_dsl")


def select_qsa_sparse_attention_kernel(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    selected_slots: torch.Tensor,
    *,
    max_seqlen_q: int | None,
    k_scale: float | torch.Tensor | None,
    v_scale: float | torch.Tensor | None,
    sparse_metadata: QSASparseMetadata | None,
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
        "equal_head_dims": q.shape[-1] == k_cache.shape[-1] == v_cache.shape[-1],
        "selected_width": selected_slots.shape[1],
        "is_decode": max_seqlen_q is not None,
        "has_sparse_metadata": sparse_metadata is not None,
        "block_size": None if sparse_metadata is None else sparse_metadata.block_size,
        "block_topk": (
            None
            if sparse_metadata is None
            else sparse_metadata.selected_blocks.shape[1]
        ),
        "q_heads_per_kv": (
            q.shape[1] // k_cache.shape[1]
            if q.shape[1] % k_cache.shape[1] == 0
            else None
        ),
        "page_size": None if sparse_metadata is None else sparse_metadata.page_size,
        "scalar_scales": not isinstance(k_scale, torch.Tensor)
        and not isinstance(v_scale, torch.Tensor),
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
    sparse_metadata: QSASparseMetadata | None,
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
        sparse_metadata: Logical representation of the same slots, enabling
            grouped attention kernels. Pass ``None`` explicitly for direct-slot
            kernels. Unsupported geometries use the physical slots.

    Returns:
        Attention output shaped
        ``[tokens, query_heads, value_head_dim]`` with the query dtype.

    The SM100/SM103 CuTe DSL implementation is preferred when its specialization
    matches a uniform decode. Other supported Blackwell BF16 inputs with logical
    metadata use FlashInfer PrimTS QSA. Other supported NVIDIA inputs retain FA2.
    GFX950 uses the Gluon kernel and other AMD GPUs the portable Triton kernel.
    Selection preserves prefill versus decode before adapting ragged inputs to
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
    if sparse_metadata is not None:
        sparse_metadata.validate(
            q.shape[0], selected_slots.shape[1], query_width=max_seqlen_q
        )
    if q.shape[0] == 0:
        return q.new_empty((0, q.shape[1], v_cache.shape[-1]))
    if k_cache.shape[0] < 2 or v_cache.shape[0] < 2:
        raise ValueError("QSA caches must include slot 1 for masked reads")
    if is_warming_up() and _warmup_uses_cute_dsl(
        "qsa_sparse_attention",
        "cute_dsl_blackwell_qsa_sparse_attention",
        override=override,
        solution=solution,
    ):
        from tokenspeed_kernel.ops.attention.qsa.cute_dsl import (
            _warmup_qsa_sparse_attention,
        )

        _warmup_qsa_sparse_attention(
            q,
            k_cache,
            v_cache,
            selected_slots,
            max_rows=get_autotune_max_num_tokens(),
        )
    kernel = select_qsa_sparse_attention_kernel(
        q,
        k_cache,
        v_cache,
        selected_slots,
        max_seqlen_q=max_seqlen_q,
        k_scale=k_scale,
        v_scale=v_scale,
        sparse_metadata=sparse_metadata,
        override=override,
        solution=solution,
    )
    metadata_kwargs = {}
    if "qsa_sparse_metadata" in KernelRegistry.get().get_by_name(kernel.name).features:
        metadata_kwargs["sparse_metadata"] = sparse_metadata
        # Grouped attention reads request boundaries from its logical metadata.
        query_width = max_seqlen_q
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
        **metadata_kwargs,
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
    """Score each query row's complete compressed QSA blocks.

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
        the head dot products below each row's ``complete_blocks``. Later
        columns are unspecified; selectors must receive ``complete_blocks``.
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
        features=frozenset({"logits"}),
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
    enable_pdl: bool,
    override: str | None,
    solution: str | None,
) -> torch.Tensor:
    """Select each row's ``topk`` highest QSA block scores.

    Args:
        logits: FP32 scores shaped ``[rows, cols]`` from ``qsa_block_scores``.
            Scored prefixes must be finite; later columns are ignored.
        topk: Blocks selected per row.
        lengths: Per-row count of scored columns in ``[0, cols]``.
            Later columns may be uninitialized.
        enable_pdl: Use programmatic dependent launch where supported.
        override: Optional registered kernel name.
        solution: Optional kernel solution.

    Returns:
        Int32 column ids shaped ``[rows, topk]`` in no particular order,
        padded with ``-1`` when a row has fewer scored columns. Equal scores
        at the threshold may select different column ids across solutions.
    """

    kernel = select_kernel(
        "attention",
        "qsa_block_topk_select",
        format_signature(logits=dense_tensor_format(logits.dtype)),
        traits={"topk": topk},
        solution=solution,
        override=override,
        features=frozenset({"logits"}),
    )
    return kernel(
        logits,
        topk,
        lengths=lengths,
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
    queries_per_request: int | None,
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
        queries_per_request: Uniform request-run width or None for ragged rows.
        max_partial_bytes: Memory budget for partial top-k buffers.
        enable_pdl: Use programmatic dependent launch where supported.
        override: Optional registered kernel name.
        solution: Optional kernel solution.

    Returns:
        Int32 block ids shaped ``[rows, block_topk]`` in no particular order;
        invalid entries are ``-1``.
    """

    from tokenspeed_kernel.ops.attention.qsa.cute_dsl import (
        can_use_cute_dsl_qwen4_exp_qsa_block_topk,
    )

    supported_layout = can_use_cute_dsl_qwen4_exp_qsa_block_topk(
        query,
        key_cache,
        page_table,
        request_indices,
        complete_blocks,
        page_size=page_size,
        block_topk=block_topk,
        queries_per_request=queries_per_request,
        max_partial_bytes=max_partial_bytes,
    )
    if is_warming_up() and _warmup_uses_cute_dsl(
        "qsa_block_topk_stream",
        "cute_dsl_qwen4_exp_qsa_block_topk",
        override=override,
        solution=solution,
    ):
        from tokenspeed_kernel.ops.attention.qsa.cute_dsl import _warmup_qsa_stream_topk

        _warmup_qsa_stream_topk(
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
    kernel = select_kernel(
        "attention",
        "qsa_block_topk_stream",
        format_signature(
            query=dense_tensor_format(query.dtype),
            key_cache=dense_tensor_format(key_cache.dtype),
        ),
        traits={
            "num_q_heads": query.shape[1],
            "head_dim": query.shape[2],
            "page_size": page_size,
            "block_topk": block_topk,
            "supported_layout": supported_layout,
        },
        solution=solution,
        override=override,
        features=frozenset({"stream"}),
    )
    return kernel(
        query,
        key_cache,
        page_table,
        request_indices,
        complete_blocks,
        page_size=page_size,
        block_topk=block_topk,
        queries_per_request=queries_per_request,
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
    max_logits_bytes: int,
    path: str,
    enable_pdl: bool,
    solution: str | None,
    override: str | None,
) -> torch.Tensor:
    """Select the highest-scoring compressed blocks through kernel selection.

    Args:
        query: Query tensor shaped ``[rows, heads, head_dim]``.
        key_cache: Flattened compressed keys shaped ``[slots, 1, head_dim]``.
        page_table: Physical compressed-cache pages shaped ``[requests, pages]``
            with contiguous columns; holes and padding are zero.
        request_indices: Contiguous owning request id per query row.
        complete_blocks: Contiguous fully compressed block counts shaped
            ``[rows]``, between zero and the page-table capacity.
        page_size: Compressed-cache rows covered by one logical page.
        block_topk: Power-of-two blocks selected per row, at least 64.
        queries_per_request: Uniform consecutive query rows per request, or
            ``None`` for ragged/mixed runs. A positive width must divide rows;
            callers guarantee one owner per run. Each row keeps its frontier.
        max_partial_bytes: Global scratch budget for streaming selection,
            excluding the output. Grouped CuTe streaming needs no scratch.
        max_logits_bytes: Score-matrix budget for ``auto``. Exceeding it
            selects streaming regardless of query size.
        path: ``"logits"`` materializes FP32 scores then radix-selects;
            ``"stream"`` fuses scoring and selection. ``"auto"`` uses the
            query-row count, table capacity and supported layout, independently
            of prefill/decode. Explicit paths bypass the logits budget.
        enable_pdl: Explicit programmatic dependent launch ordering.
        solution: Optional registered backend, such as ``"cute_dsl"`` or
            ``"triton"``, filtered within the selected path.
        override: Optional registered kernel name or backend override.

    Returns:
        Int32 logical block ids shaped ``[rows, block_topk]``, padded with
        ``-1``. Output order and boundary tie ids are unspecified.
    """
    from tokenspeed_kernel.ops.attention.qsa.cute_dsl import (
        can_use_cute_dsl_qwen4_exp_qsa_block_topk,
    )

    if query.ndim != 3 or key_cache.ndim != 3 or page_table.ndim != 2:
        raise ValueError("QSA block top-k expects rank-three Q/K and a rank-two table")
    if block_topk < 64 or block_topk & (block_topk - 1):
        raise ValueError("QSA block top-k needs a power-of-two topk >= 64")
    if path not in ("auto", "stream", "logits"):
        raise ValueError(
            f"QSA top-k path must be 'auto', 'stream' or 'logits', got {path!r}"
        )
    rows = query.shape[0]
    if key_cache.shape[1:] != (1, query.shape[2]):
        raise ValueError("QSA keys must have one head with the query head dimension")
    if request_indices.shape != (rows,) or complete_blocks.shape != (rows,):
        raise ValueError("QSA block top-k metadata must have one entry per query row")
    if (
        page_table.stride(1) != 1
        or not request_indices.is_contiguous()
        or not complete_blocks.is_contiguous()
    ):
        raise ValueError(
            "QSA block top-k requires contiguous table columns and row metadata"
        )
    if queries_per_request is not None and (
        queries_per_request < 1 or rows % queries_per_request
    ):
        raise ValueError("QSA queries_per_request must be positive and divide rows")
    num_blocks = page_table.shape[1] * page_size
    if rows == 0 or num_blocks == 0:
        return torch.full(
            (rows, block_topk), -1, dtype=torch.int32, device=query.device
        )

    supported_layout = can_use_cute_dsl_qwen4_exp_qsa_block_topk(
        query,
        key_cache,
        page_table,
        request_indices,
        complete_blocks,
        page_size=page_size,
        block_topk=block_topk,
        queries_per_request=queries_per_request,
        max_partial_bytes=max_partial_bytes,
    )
    if path == "auto":
        # Keep shape routing independent of live device frontiers.
        min_stream_rows = 512 if num_blocks >= 16384 else 8192
        prefer_stream = supported_layout and rows >= min_stream_rows
        fits_logits = rows * num_blocks * 4 <= max_logits_bytes
        path = "stream" if prefer_stream or not fits_logits else "logits"
        # A logits startup must still prepare later stream calls. Stream calls
        # share the warmup in qsa_block_topk_stream below.
        if (
            path == "logits"
            and is_warming_up()
            and _warmup_uses_cute_dsl(
                "qsa_block_topk_stream",
                "cute_dsl_qwen4_exp_qsa_block_topk",
                override=override,
                solution=solution,
            )
        ):
            from tokenspeed_kernel.ops.attention.qsa.cute_dsl import (
                _warmup_qsa_stream_topk,
            )

            _warmup_qsa_stream_topk(
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

    if path == "logits":
        scores = qsa_block_scores(
            query,
            key_cache,
            page_table,
            request_indices,
            complete_blocks,
            page_size=page_size,
            queries_per_request=queries_per_request,
            enable_pdl=enable_pdl,
            solution=solution,
            override=override,
        )
        return qsa_block_topk_select(
            scores,
            block_topk,
            lengths=complete_blocks,
            enable_pdl=enable_pdl,
            solution=solution,
            override=override,
        )
    return qsa_block_topk_stream(
        query,
        key_cache,
        page_table,
        request_indices,
        complete_blocks,
        page_size=page_size,
        block_topk=block_topk,
        queries_per_request=queries_per_request,
        max_partial_bytes=max_partial_bytes,
        enable_pdl=enable_pdl,
        solution=solution,
        override=override,
    )


# Backend registration (side-effect imports)
# isort: off
import tokenspeed_kernel.ops.attention.qsa.cuda  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.qsa.triton  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.qsa.cute_dsl  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.qsa.flashinfer  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.qsa.flashinfer_prims  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.qsa.gluon  # noqa: E402,F401

# isort: on


__all__ = [
    "qsa_block_scores",
    "qsa_block_topk_select",
    "qsa_block_topk_stream",
    "qsa_sparse_attention",
    "QSASparseMetadata",
    "QSAWorkspace",
    "qwen4_exp_qsa_block_topk",
]

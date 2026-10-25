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

"""CuTe DSL registration for Blackwell QSA selection and sparse attention."""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
    pdl_enabled,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

_PLATFORM = current_platform()
_IS_NVIDIA_BLACKWELL = _PLATFORM.is_nvidia and _PLATFORM.is_blackwell

if _IS_NVIDIA_BLACKWELL:
    from tokenspeed_kernel.thirdparty.cute_dsl.qsa_sparse import (
        kernel as _cute_dsl_qsa_sparse_attention,
    )
    from tokenspeed_kernel.thirdparty.cute_dsl.qsa_sparse import (
        warmup as _warmup_sparse,
    )
    from tokenspeed_kernel.thirdparty.cute_dsl.qsa_stream_topk import (
        _configuration,
    )
    from tokenspeed_kernel.thirdparty.cute_dsl.qsa_stream_topk import (
        kernel as _cute_dsl_qsa_stream_block_topk,
    )
    from tokenspeed_kernel.thirdparty.cute_dsl.qsa_stream_topk import (
        warmup as _warmup_stream,
    )

_HEAD_DIM = 256
_SELECTED_WIDTH = 2051


def _supported_stream_topk_tensors(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    page_table: torch.Tensor,
    request_indices: torch.Tensor,
    complete_blocks: torch.Tensor,
    *,
    page_size: int,
    block_topk: int,
) -> bool:
    """Validate tensor geometry independently of the scratch-dependent tactic."""
    if not _IS_NVIDIA_BLACKWELL:
        return False
    if query.ndim != 3 or key_cache.ndim != 3 or page_table.ndim != 2:
        return False
    rows = query.shape[0]
    if rows == 0:
        return False
    if (
        query.shape[1:] != (4, 128)
        or key_cache.shape[0] < 64
        or key_cache.shape[0] % 64
        or key_cache.shape[1:] != (1, 128)
    ):
        return False
    if page_size != 64 or block_topk != 512 or not 0 < page_table.shape[1] <= 2048:
        return False
    if query.dtype is not torch.bfloat16 or key_cache.dtype is not torch.bfloat16:
        return False
    if page_table.dtype is not torch.int32:
        return False
    if request_indices.dtype not in (torch.int32, torch.int64):
        return False
    if complete_blocks.dtype is not torch.int32:
        return False
    if request_indices.shape != (rows,) or complete_blocks.shape != (rows,):
        return False
    tensors = (query, key_cache, page_table, request_indices, complete_blocks)
    if not query.is_cuda or any(tensor.device != query.device for tensor in tensors):
        return False
    if query.stride(2) != 1 or query.stride(0) % 8 or query.stride(1) % 8:
        return False
    if query.data_ptr() % 16 or not key_cache.is_contiguous():
        return False
    if key_cache.data_ptr() % 16:
        return False
    if page_table.stride(1) != 1 or page_table.stride(0) < page_table.shape[1]:
        return False
    if page_table.data_ptr() % 16:
        return False
    return all(
        tensor.is_contiguous() and tensor.data_ptr() % 16 == 0
        for tensor in (request_indices, complete_blocks)
    )


def _warmup_qsa_stream_topk(
    query,
    keys,
    page_table,
    requests,
    complete,
    *,
    page_size,
    block_topk,
    max_partial_bytes,
    enable_pdl,
):
    """Compile every stream top-k tactic these tensors and the scratch budget allow."""
    # A single-row tactic needs scratch even when grouped prefill needs none.
    # Warmup checks the shared tensor contract, then enumerates valid tactics
    # with the caller's actual budget.
    if not _supported_stream_topk_tensors(
        query,
        keys,
        page_table,
        requests,
        complete,
        page_size=page_size,
        block_topk=block_topk,
    ):
        return
    # A shortened prefill view retains the allocation's full row stride.
    _warmup_stream(
        query,
        keys,
        requests,
        max_capacity=max(page_table.shape[1], page_table.stride(0)) * page_size,
        max_partial_bytes=max_partial_bytes,
        enable_pdl=enable_pdl,
    )


def _warmup_qsa_sparse_attention(q, k, v, selected, *, max_rows):
    """Warm the CuTe sparse-attention configurations these tensors legally select."""
    if (
        not _IS_NVIDIA_BLACKWELL
        or q.ndim != 3
        or k.ndim != 3
        or v.ndim != 3
        or selected.ndim != 2
        or not q.is_cuda
        or any(tensor.device != q.device for tensor in (k, v, selected))
        or q.shape[1] not in (6, 12, 24)
        or k.shape[1] not in (1, 2, 4)
        or k.shape[0] < 2
        or q.shape[1] <= k.shape[1]
        or q.shape[1] % k.shape[1]
        or q.shape[2] != _HEAD_DIM
        or k.shape[2] != _HEAD_DIM
        or v.shape != k.shape
        or selected.shape != (q.shape[0], _SELECTED_WIDTH)
        or q.dtype is not torch.bfloat16
        or k.dtype not in (torch.bfloat16, torch.float8_e4m3fn)
        or v.dtype is not k.dtype
        or selected.dtype is not torch.int32
    ):
        return
    _warmup_sparse(q, k, v, max_rows=max_rows, enable_pdl=pdl_enabled())


def can_use_cute_dsl_qwen4_exp_qsa_block_topk(
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
) -> bool:
    """Return whether a tuned Blackwell MMA specialization should be used.

    The selector excludes other layouts before ranking registered kernels.

    Args:
        query: Candidate BF16 query tensor shaped ``[rows, 4, 128]``.
        key_cache: Candidate contiguous BF16 key cache shaped
            ``[slots, 1, 128]``.
        page_table: Candidate int32 page table with contiguous columns.
        request_indices: Candidate contiguous request id per query row.
        complete_blocks: Candidate contiguous valid-block count per query row.
        page_size: Number of compressed blocks per cache page.
        block_topk: Number of logical blocks to select per row.
        queries_per_request: Uniform consecutive query-run width, or ``None``
            for a ragged layout.
        max_partial_bytes: Caller budget for global selection scratch,
            excluding the output. Grouped streaming needs no global scratch.

    Returns:
        Whether the tensors satisfy a tuned MMA specialization's contract.
    """

    if not _supported_stream_topk_tensors(
        query,
        key_cache,
        page_table,
        request_indices,
        complete_blocks,
        page_size=page_size,
        block_topk=block_topk,
    ):
        return False
    rows = query.shape[0]
    if queries_per_request is not None and (
        queries_per_request < 1 or rows % queries_per_request
    ):
        return False
    return (
        _configuration(
            rows,
            page_table.shape[1] * page_size,
            queries_per_request,
            max_partial_bytes,
        )
        is not None
    )


def cute_dsl_qwen4_exp_qsa_block_topk(
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
) -> torch.Tensor:
    """Run the registered Blackwell QSA streaming specialization.

    Args:
        query: Aligned BF16 queries shaped ``[rows, 4, 128]``.
        key_cache: Contiguous BF16 keys shaped ``[slots, 1, 128]``.
        page_table: Aligned int32 physical pages shaped ``[requests, max_pages]``
            with contiguous columns and a runtime row stride.
        request_indices: Int32 or int64 request id per query row.
        complete_blocks: Contiguous int32 valid-block count per query row.
        page_size: Number of compressed blocks per page; must be 64.
        block_topk: Number of selected blocks; must be 512.
        queries_per_request: Uniform consecutive query-run width, or ``None``
            for ragged rows. Groups mask tails and never cross requests.
        max_partial_bytes: Budget for global scratch, excluding the output.
        enable_pdl: Whether to use programmatic dependent launch ordering.

    Returns:
        Int32 logical block ids shaped ``[rows, block_topk]``, padded with ``-1``.
    """

    if not _IS_NVIDIA_BLACKWELL:
        raise RuntimeError("CuTe DSL QSA stream top-k requires NVIDIA Blackwell")
    return _cute_dsl_qsa_stream_block_topk(
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


if _IS_NVIDIA_BLACKWELL:
    register_kernel(
        "attention",
        "qsa_block_topk_stream",
        name="cute_dsl_qwen4_exp_qsa_block_topk",
        solution="cute_dsl",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 7),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    query=dense_tensor_format(torch.bfloat16),
                    key_cache=dense_tensor_format(torch.bfloat16),
                ),
            }
        ),
        traits={
            "num_q_heads": frozenset({4}),
            "head_dim": frozenset({128}),
            "page_size": frozenset({64}),
            "block_topk": frozenset({512}),
            "supported_layout": frozenset({True}),
        },
        priority=Priority.SPECIALIZED,
        features=frozenset({"stream"}),
    )(cute_dsl_qwen4_exp_qsa_block_topk)


def cute_dsl_blackwell_qsa_sparse_attention(
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
) -> torch.Tensor:
    """Run Blackwell QSA with exact query-prefix and KV operand reuse.

    Args:
        q: BF16 query tensor shaped [tokens, query_heads, 256], with 6,
            12 or 24 query heads, divisible by the KV-head count.
        k_cache: BF16 or FP8 E4M3 key cache shaped [slots, kv_heads, 256],
            with 1, 2 or 4 KV heads.
        v_cache: Value cache with the same shape and dtype as k_cache.
        selected_slots: Int32 physical slots shaped [tokens, 2051];
            non-positive entries are masked.
        scale: Softmax scale applied to query-key scores.
        max_seqlen_q: Uniform query-token count per request, including
            speculative decode. Query groups never cross request boundaries.
        metadata_capacity_rows: Capacity for stateful fallback plans, unused
            by this implementation's call-owned prefix flags.
        k_scale: Scalar key-cache descale, folded into scale.
        v_scale: Scalar value-cache descale, applied to the output.

    Returns:
        BF16 attention output with q's shape.

    Twelve BF16 query heads share one sixteen-head tile. Larger batches group
    consecutive queries only when their complete ordered 2048-entry prefixes
    match; each query keeps separate softmax state and three private tail slots.
    A distinct prefix retains one query per CTA. The measured head geometry
    selects query-group size, operand stages and warp/register allocation.

    Two-query groups and six-head geometries detect equality inside attention.
    Larger twelve-head groups refresh prefix flags in one preparation kernel
    before attention. Both launches are included in full-call benchmarks. KV
    data is gathered directly from the cache with 64-bit offsets; preparation
    neither copies KV nor caches decisions across calls or graph replays.
    FP8 retains its gather/conversion and split-cluster path.
    """

    del metadata_capacity_rows  # This backend does not use fallback plan storage.
    return _cute_dsl_qsa_sparse_attention(
        q,
        k_cache,
        v_cache,
        selected_slots,
        scale=scale,
        max_seqlen_q=max_seqlen_q,
        k_scale=k_scale,
        v_scale=v_scale,
        enable_pdl=pdl_enabled(),
    )


if _IS_NVIDIA_BLACKWELL:
    register_kernel(
        "attention",
        "qsa_sparse_attention",
        name="cute_dsl_blackwell_qsa_sparse_attention",
        solution="cute_dsl",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 7),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    q=dense_tensor_format(torch.bfloat16),
                    k_cache=dense_tensor_format(torch.float8_e4m3fn),
                    v_cache=dense_tensor_format(torch.float8_e4m3fn),
                ),
                format_signature(
                    q=dense_tensor_format(torch.bfloat16),
                    k_cache=dense_tensor_format(torch.bfloat16),
                    v_cache=dense_tensor_format(torch.bfloat16),
                ),
            }
        ),
        traits={
            "num_q_heads": frozenset({6, 12, 24}),
            "num_kv_heads": frozenset({1, 2, 4}),
            "head_dim": frozenset({_HEAD_DIM}),
            "value_head_dim": frozenset({_HEAD_DIM}),
            "selected_width": frozenset({_SELECTED_WIDTH}),
            "is_decode": frozenset({True}),
        },
        priority=Priority.SPECIALIZED + 2,
    )(cute_dsl_blackwell_qsa_sparse_attention)
    __all__ = [
        "can_use_cute_dsl_qwen4_exp_qsa_block_topk",
        "cute_dsl_qwen4_exp_qsa_block_topk",
        "cute_dsl_blackwell_qsa_sparse_attention",
    ]
else:
    __all__ = []

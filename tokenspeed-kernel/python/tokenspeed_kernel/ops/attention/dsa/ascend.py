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

"""Ascend solutions for the common DSA kernel interfaces.

Communication and context-parallel scheduling deliberately do not live here.
This module has the same role as :mod:`dsa.cuda`: it adapts device operators to
the vendor-neutral tokenspeed-kernel boundary.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = ["supports_fused_dsa_decode", "require_ascend_dsa_decode"]

platform = current_platform()
_kernels = None


def supports_fused_dsa_decode(layer) -> bool:
    """Admit the host's CP8 packed projection ABI from the loaded weights."""
    indexer = layer.indexer
    if (
        layer.hidden_size,
        layer.num_heads,
        layer.num_local_heads,
        layer.q_lora_rank,
        layer.kv_lora_rank,
        layer.qk_nope_head_dim,
        layer.qk_rope_head_dim,
        layer.v_head_dim,
        indexer.index_n_heads,
        indexer.index_head_dim,
    ) != (4096, 64, 8, 1536, 512, 128, 64, 128, 64, 128) or layer.use_output_gate:
        return False
    weights = (
        (layer.q_a_proj.weight, (1536, 4096), torch.bfloat16),
        (layer.kv_a_proj_with_mqa.weight, (576, 4096), torch.bfloat16),
        (layer.q_b_proj.weight, (1536, 1536), torch.bfloat16),
        (indexer.wq_b.weight, (8192, 1536), torch.bfloat16),
        (indexer.wk.weight, (128, 4096), torch.bfloat16),
        (indexer.weights_proj.weight, (64, 4096), torch.float32),
    )
    if any(
        value.shape != shape or value.dtype != dtype for value, shape, dtype in weights
    ):
        return False
    return all(
        rotary is None or (not rotary.is_neox_style and rotary.rotary_dim == 64)
        for rotary in (layer.rotary_emb, indexer.rotary_emb)
    )


def _dsa_adapter():
    """Return the process-local Ascend DSA operator adapter."""
    global _kernels
    if not platform.is_npu:
        raise RuntimeError("Ascend DSA kernels require an NPU platform")
    if _kernels is None:
        from tokenspeed_kernel_npu.ops.dsa import AscendDSAKernels

        _kernels = AscendDSAKernels()
    return _kernels


def require_ascend_dsa_decode():
    """Validate ordinary Decode dependencies before serving or graph capture."""
    _dsa_adapter().require_decode()


def validate_dsa_fused_decode(*, rank_size, num_heads):
    """Check installed native communication ABI before preparing its resources."""
    _dsa_adapter().require_fused_context_parallel(
        rank_size=rank_size, num_heads=num_heads
    )


def ascend_prepare_dsa_resources(*, group, rank):
    """Bind operator-owned views of the existing HCCL windows before capture."""
    from tokenspeed_kernel.ops.attention.dsa import DSAResources

    require_ascend_dsa_decode()
    window = torch.ops.custom.dsa_prepare_prolog_window(group, rank)
    try:
        prepared = {
            rows: torch.ops.custom.dsa_prepare_lse_buffers(group, rows, rank)
            for rows in (64, 128, 256, 512)
        }
    except BaseException:
        torch.ops.custom.dsa_release_prolog_window(window)
        raise
    return DSAResources(
        window,
        prepared,
        release=torch.ops.custom.dsa_release_prolog_window,
    )


if platform.is_npu:
    register_kernel(
        "attention",
        "dsa_prepare_resources",
        name="ascend_prepare_dsa_resources",
        solution="ascend",
        capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
        signatures=[format_signature()],
        priority=Priority.PERFORMANT,
    )(ascend_prepare_dsa_resources)

    @register_kernel(
        "attention",
        "dsa_interleave_rope",
        name="ascend_dsa_interleave_rope",
        solution="ascend",
        capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
        signatures=frozenset(
            {
                format_signature(tensor=dense_tensor_format(torch.bfloat16)),
                format_signature(tensor=dense_tensor_format(torch.float32)),
            }
        ),
        traits={"rope_dim": frozenset({64})},
        priority=Priority.PERFORMANT,
    )
    def ascend_dsa_interleave_rope(
        tensor: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        rope_dim: int,
    ) -> torch.Tensor:
        return _dsa_adapter().interleave_rope(
            tensor,
            positions,
            cos_sin_cache,
            rope_dim=rope_dim,
        )

    @register_kernel(
        "attention",
        "dsa_decode_topk",
        name="ascend_lightning_indexer_dsa_decode_topk",
        solution="ascend",
        capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
        signatures=frozenset(
            {
                format_signature(
                    q=dense_tensor_format(torch.bfloat16),
                    weights=dense_tensor_format(torch.bfloat16),
                ),
                format_signature(
                    q=dense_tensor_format(torch.bfloat16),
                    weights=dense_tensor_format(torch.float32),
                ),
            }
        ),
        traits={
            "index_heads": frozenset({16}),
            "head_dim": frozenset({128}),
            "topk": frozenset({2048}),
            "page_size": frozenset({128}),
            "q_len_per_req": frozenset({1}),
            "index_k_layout": frozenset({"page_planar"}),
        },
        features={"logical_offsets", "forced_initial_local"},
        priority=Priority.PERFORMANT,
    )
    def ascend_lightning_indexer_dsa_decode_topk(
        q: torch.Tensor,
        weights: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        page_size: int,
        topk: int,
        softmax_scale: float,
        q_len_per_req: int,
        index_k_cache: torch.Tensor | None,
        seq_lens_2d: torch.Tensor | None,
        plan: object | None,
        out: torch.Tensor | None,
        lens_out: torch.Tensor | None,
        topk_layout: str,
        block_table_base_offsets: torch.Tensor | None,
        initial_tokens: int,
        local_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del softmax_scale, seq_lens_2d, plan, block_table_base_offsets
        if index_k_cache is None:
            raise ValueError("Ascend LightningIndexer requires index_k_cache")
        if q_len_per_req != 1:
            raise ValueError(
                "Ascend LightningIndexer decode currently requires q_len=1"
            )
        if topk_layout != "logical_offsets":
            raise ValueError("Ascend LightningIndexer returns logical offsets")
        q_ends = torch.arange(1, q.shape[0] + 1, dtype=torch.int32, device=q.device)
        indices, _ = _dsa_adapter().index(
            q,
            index_k_cache,
            weights,
            q_ends,
            seq_lens,
            block_table,
            topk,
            initial_tokens,
            local_tokens,
        )
        if out is not None:
            out.copy_(indices)
            indices = out
        lengths = seq_lens.to(torch.int32).clamp(max=topk)
        if lens_out is not None:
            lens_out.copy_(lengths)
            lengths = lens_out
        return indices, lengths


def _native_operation(mode, dtypes):
    """Register a single callable with the format of its principal tensor."""

    def register(function):
        if not platform.is_npu:
            return function
        return register_kernel(
            "attention",
            mode,
            name=function.__name__,
            solution="ascend",
            capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
            signatures=[
                format_signature(input=dense_tensor_format(dtype)) for dtype in dtypes
            ],
            priority=Priority.PERFORMANT,
        )(function)

    return register


@_native_operation("store_index_keys", (torch.bfloat16,))
def ascend_store_index_keys(x, cache, locations):
    """Index-K rows at owner-local cache slots."""
    return _dsa_adapter().scatter(x, cache, locations)


@_native_operation("dsa_local_topk", (torch.bfloat16,))
def ascend_dsa_local_topk(
    query, key, weights, q_ends, kv_lengths, table, topk, initial, local
):
    """Decode TopK offsets and live chunk hints for each query."""
    return _dsa_adapter().index(
        query, key, weights, q_ends, kv_lengths, table, topk, initial, local
    )


@_native_operation("dsa_local_topk_candidates", (torch.bfloat16,))
def ascend_dsa_local_topk_candidates(
    query,
    key,
    weights,
    q_ends,
    kv_lengths,
    table,
    topk,
    initial_counts,
    local_counts,
    *,
    sparse_mode,
    visible_lengths=None,
):
    """Owner-local TopK offsets and scores with causal bounds."""
    return _dsa_adapter().index_partial(
        query,
        key,
        weights,
        q_ends,
        kv_lengths,
        table,
        topk,
        initial_counts,
        local_counts,
        sparse_mode=sparse_mode,
        visible_lengths=visible_lengths,
    )


@_native_operation("dsa_gathered_prefill_topk", (torch.bfloat16,))
def ascend_dsa_gathered_prefill_topk(
    query, key, weights, query_cu, key_cu, table, topk, initial_tokens, window_tokens
):
    """Prefill TopK offsets over gathered Index-K pages."""
    return _dsa_adapter().prefill_index(
        query,
        key,
        weights,
        query_cu,
        key_cu,
        table,
        topk,
        initial_tokens,
        window_tokens,
    )


@_native_operation("dsa_gathered_prefill_attention", (torch.bfloat16,))
def ascend_dsa_gathered_prefill_attention(
    query, pages, indices, scan_count, table, query_seq, scale
):
    """Prefill attention over gathered latent pages; return head-major output."""
    return _dsa_adapter().prefill_attention(
        query, pages, indices, scan_count, table, query_seq, scale
    )


@_native_operation("dsa_select_local_topk", (torch.int32,))
def ascend_dsa_select_local_topk(indices, positions, rank):
    """Map exchanged TopK positions to this owner; return local indices and counts."""
    return _dsa_adapter().select_local(indices, positions, rank)


@_native_operation("merge_dsa_topk", (torch.int32,))
def ascend_merge_dsa_topk(indices, values, *, group, rank_size):
    """Exchange owner candidates and merge TopK using prepared native communication."""
    return _dsa_adapter().merge_topk_fused(
        indices, values, group=group, rank_size=rank_size
    )


@_native_operation(
    "merge_dsa_lse",
    (
        torch.bfloat16,
        torch.float32,
    ),
)
def ascend_merge_dsa_lse(output, maximum, total, *, group, rank_size):
    """Merge attention partials using prepared native communication."""
    return _dsa_adapter().merge_lse_direct(
        output, maximum, total, group=group, rank_size=rank_size
    )


@_native_operation(
    "dsa_project_output_partial",
    (
        torch.bfloat16,
        torch.float32,
    ),
)
def ascend_dsa_project_output_partial(output, w_vc, w_o):
    """Project latent attention through value and output weights."""
    return _dsa_adapter().output_projection_partial(output, w_vc, w_o)


@_native_operation("dsa_fused_prologue", (torch.bfloat16,))
def ascend_dsa_fused_prologue(
    x,
    down,
    gamma_q,
    gamma_kv,
    gamma_index,
    score_weight,
    qb,
    uk,
    window,
    kv_cache,
    index_cache,
    slots,
    *,
    eps_q,
    eps_kv,
    eps_index,
    positions,
    cos_sin,
    rotary_mode,
    out,
):
    """Project queries, exchange heads and store owner KV and Index-K."""
    return _dsa_adapter().fused_prolog(
        x,
        down,
        gamma_q,
        gamma_kv,
        gamma_index,
        score_weight,
        qb,
        uk,
        window,
        kv_cache,
        index_cache,
        slots,
        eps_q=eps_q,
        eps_kv=eps_kv,
        eps_index=eps_index,
        positions=positions,
        cos_sin=cos_sin,
        rotary_mode=rotary_mode,
        out=out,
    )


@_native_operation("prepare_dsa_projection_weight", (torch.bfloat16,))
def ascend_prepare_dsa_projection_weight(value):
    """Prepare an independent NZ projection weight without changing coefficients."""
    return _dsa_adapter().prepare_projection_weight(value)


@_native_operation("dsa_sparse_decode", (torch.bfloat16,))
def ascend_dsa_sparse_decode(
    q, qr, key, indices, valid_chunks, q_ends, kv_lengths, table, scale
):
    """Attend selected local KV slots; return token-major output."""
    return _dsa_adapter().attention(
        q, qr, key, indices, valid_chunks, q_ends, kv_lengths, table, scale
    )


@_native_operation("dsa_sparse_decode_partial", (torch.bfloat16,))
def ascend_dsa_sparse_decode_partial(
    packed_query,
    key,
    indices,
    valid_chunks,
    q_ends,
    kv_lengths,
    table,
    scale,
    *,
    sparse_mode,
    out=None,
):
    """Attend owner KV pages; return output, softmax maximum and sum."""
    return _dsa_adapter().attention_partial(
        packed_query,
        key,
        indices,
        valid_chunks,
        q_ends,
        kv_lengths,
        table,
        scale,
        sparse_mode=sparse_mode,
        out=out,
    )


@_native_operation("dsa_cp_metadata", (torch.int32,))
def ascend_dsa_cp_metadata(
    table,
    lengths,
    *,
    page_size,
    block_granularity,
    virtual_block_count,
    degree,
    rank,
    initial_tokens,
    window_tokens,
    query_width,
    query_group_size,
    slots,
    out,
    local_lengths,
    initial_counts,
    window_counts,
    query_ends,
    index_table,
    index_lengths,
    index_initial_counts,
    index_query_ends,
    local_slots,
):
    """Refresh caller-owned causal page tables, lengths and owner write slots."""
    from tokenspeed_kernel_npu.ops.dsa import dsa_cp_metadata as dsa_cp_metadata_impl

    return dsa_cp_metadata_impl(
        table,
        lengths,
        page_size=page_size,
        block_granularity=block_granularity,
        virtual_block_count=virtual_block_count,
        degree=degree,
        rank=rank,
        initial_tokens=initial_tokens,
        window_tokens=window_tokens,
        query_width=query_width,
        query_group_size=query_group_size,
        slots=slots,
        out=out,
        local_lengths=local_lengths,
        initial_counts=initial_counts,
        window_counts=window_counts,
        query_ends=query_ends,
        index_table=index_table,
        index_lengths=index_lengths,
        index_initial_counts=index_initial_counts,
        index_query_ends=index_query_ends,
        local_slots=local_slots,
    )


@_native_operation(
    "dsa_prefill_query_metadata",
    (
        torch.int32,
        torch.int64,
    ),
)
def ascend_dsa_prefill_query_metadata(
    cumulative, prefixes, *, first, rows, topk, active_requests
):
    """Build DSA-only cumulative lengths, query request IDs and scan counts."""
    from tokenspeed_kernel_npu.ops.dsa import (
        dsa_prefill_query_metadata as dsa_prefill_query_metadata_impl,
    )

    return dsa_prefill_query_metadata_impl(
        cumulative,
        prefixes,
        first=first,
        rows=rows,
        topk=topk,
        active_requests=active_requests,
    )

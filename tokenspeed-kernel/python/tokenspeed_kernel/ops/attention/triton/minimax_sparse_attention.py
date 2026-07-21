# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Triton block-sparse GQA attention for MiniMax.

The prefill kernel uses one program per query position and local KV head.  The
decode path splits the selected blocks across programs and merges their online
softmax states, preserving exact attention over the selected tokens.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.triton.minimax_indexer import (
    SPARSE_BLOCK_SIZE,
    minimax_indexer,
)
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@triton.heuristics(
    {
        "BLOCK_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_H": lambda args: triton.next_power_of_2(args["gqa_group_size"]),
    }
)
@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _sparse_prefill_kernel(
    query,
    key_cache,
    value_cache,
    selected_blocks,
    output,
    block_table,
    cu_seqlens_q,
    seq_lens,
    prefix_lens,
    num_kv_heads,
    gqa_group_size,
    head_dim,
    topk: tl.constexpr,
    scale,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_k_page,
    stride_k_h,
    stride_k_pos,
    stride_k_d,
    stride_v_page,
    stride_v_h,
    stride_v_pos,
    stride_v_d,
    stride_t_n,
    stride_t_h,
    stride_t_k,
    stride_o_n,
    stride_o_h,
    stride_o_d,
    stride_bt_b,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    query_offset = tl.program_id(0)
    kv_head = tl.program_id(1)
    request = tl.program_id(2)

    query_start = tl.load(cu_seqlens_q + request)
    query_len = tl.load(cu_seqlens_q + request + 1) - query_start
    if query_offset >= query_len:
        return

    seq_len = tl.load(seq_lens + request)
    prefix_len = tl.load(prefix_lens + request)
    query_position = prefix_len + query_offset
    token = query_start + query_offset
    first_head = kv_head * gqa_group_size
    head_offsets = tl.arange(0, BLOCK_H)
    dims = tl.arange(0, BLOCK_D)
    head_mask = head_offsets < gqa_group_size
    dim_mask = dims < head_dim
    q = tl.load(
        query
        + token * stride_q_n
        + (first_head + head_offsets[:, None]) * stride_q_h
        + dims[None, :] * stride_q_d,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    max_score = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_H,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)
    key_offsets = tl.arange(0, BLOCK_K)
    block_table_row = block_table + request * stride_bt_b
    selected_ptr = selected_blocks + token * stride_t_n + kv_head * stride_t_h
    visible_blocks = (query_position + BLOCK_K) // BLOCK_K
    selected_count = tl.minimum(topk, visible_blocks)
    scale_log2e = scale * 1.4426950408889634

    for selected_offset in tl.range(0, selected_count):
        block = tl.load(selected_ptr + selected_offset * stride_t_k).to(tl.int32)
        page = tl.load(block_table_row + block).to(tl.int64)
        key_positions = block * BLOCK_K + key_offsets
        key_mask = key_positions <= query_position
        key = tl.load(
            key_cache
            + page * stride_k_page
            + kv_head * stride_k_h
            + key_offsets[None, :] * stride_k_pos
            + dims[:, None] * stride_k_d,
            mask=dim_mask[:, None] & key_mask[None, :],
            other=0.0,
        )
        logits = tl.dot(q, key, out_dtype=tl.float32) * scale_log2e
        logits = tl.where(
            head_mask[:, None] & key_mask[None, :],
            logits,
            -float("inf"),
        )
        block_max = tl.max(logits, axis=1)
        new_max = tl.maximum(max_score, block_max)
        probabilities = tl.exp2(logits - new_max[:, None])
        correction = tl.exp2(max_score - new_max)
        accumulator *= correction[:, None]
        normalizer = normalizer * correction + tl.sum(probabilities, axis=1)
        value = tl.load(
            value_cache
            + page * stride_v_page
            + kv_head * stride_v_h
            + key_offsets[:, None] * stride_v_pos
            + dims[None, :] * stride_v_d,
            mask=key_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        accumulator += tl.dot(probabilities.to(value.dtype), value)
        max_score = new_max

    accumulator /= normalizer[:, None]
    tl.store(
        output
        + token * stride_o_n
        + (first_head + head_offsets[:, None]) * stride_o_h
        + dims[None, :] * stride_o_d,
        accumulator,
        mask=head_mask[:, None] & dim_mask[None, :],
    )


@triton.heuristics(
    {
        "BLOCK_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_H": lambda args: max(16, triton.next_power_of_2(args["gqa_group_size"])),
    }
)
@triton.jit(do_not_specialize=["decode_query_len"])
def _sparse_decode_kernel(
    query,
    key_cache,
    value_cache,
    selected_blocks,
    partial_output,
    partial_lse,
    block_table,
    seq_lens,
    total_queries,
    gqa_group_size,
    head_dim,
    topk: tl.constexpr,
    scale,
    decode_query_len,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_k_page,
    stride_k_h,
    stride_k_pos,
    stride_k_d,
    stride_v_page,
    stride_v_h,
    stride_v_pos,
    stride_v_d,
    stride_t_n,
    stride_t_h,
    stride_t_k,
    stride_o_c,
    stride_o_n,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_n,
    stride_l_h,
    stride_bt_b,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    query_chunk = tl.program_id(0)
    kv_head = tl.program_id(1)
    token = query_chunk % total_queries
    chunk = query_chunk // total_queries
    request = token // decode_query_len
    query_offset = token - request * decode_query_len
    seq_len = tl.load(seq_lens + request)
    query_position = seq_len - decode_query_len + query_offset
    visible_blocks = (query_position + BLOCK_K) // BLOCK_K
    selected_count = tl.minimum(topk, visible_blocks)

    selected_per_chunk: tl.constexpr = (topk + NUM_CHUNKS - 1) // NUM_CHUNKS
    selected_start = selected_per_chunk * chunk
    selected_end = tl.minimum(selected_start + selected_per_chunk, selected_count)
    first_head = kv_head * gqa_group_size
    head_offsets = tl.arange(0, BLOCK_H)
    dims = tl.arange(0, BLOCK_D)
    head_mask = head_offsets < gqa_group_size
    dim_mask = dims < head_dim
    q = tl.load(
        query
        + token * stride_q_n
        + (first_head + head_offsets[:, None]) * stride_q_h
        + dims[None, :] * stride_q_d,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    max_score = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
    lse = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)
    key_offsets = tl.arange(0, BLOCK_K)
    block_table_row = block_table + request * stride_bt_b
    selected_ptr = selected_blocks + token * stride_t_n + kv_head * stride_t_h
    scale_log2e = scale * 1.4426950408889634

    for selected_offset in tl.range(selected_start, selected_end):
        block = tl.load(selected_ptr + selected_offset * stride_t_k).to(tl.int32)
        page = tl.load(block_table_row + block).to(tl.int64)
        key_positions = block * BLOCK_K + key_offsets
        key_mask = key_positions <= query_position
        key = tl.load(
            key_cache
            + page * stride_k_page
            + kv_head * stride_k_h
            + key_offsets[None, :] * stride_k_pos
            + dims[:, None] * stride_k_d,
            mask=dim_mask[:, None] & key_mask[None, :],
            other=0.0,
        )
        logits = tl.dot(q, key, out_dtype=tl.float32) * scale_log2e
        logits = tl.where(
            head_mask[:, None] & key_mask[None, :],
            logits,
            -float("inf"),
        )
        block_max = tl.max(logits, axis=1)
        new_max = tl.maximum(max_score, block_max)
        probabilities = tl.exp2(logits - new_max[:, None])
        block_sum = tl.sum(probabilities, axis=1)
        accumulator *= tl.exp2(max_score - new_max)[:, None]
        value = tl.load(
            value_cache
            + page * stride_v_page
            + kv_head * stride_v_h
            + key_offsets[:, None] * stride_v_pos
            + dims[None, :] * stride_v_d,
            mask=key_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        accumulator += tl.dot(probabilities.to(value.dtype), value)
        lse = new_max + tl.log2(tl.exp2(lse - new_max) + block_sum)
        max_score = new_max

    normalization = tl.where(
        lse > -float("inf"),
        tl.exp2(max_score - lse),
        0.0,
    )
    accumulator *= normalization[:, None]
    tl.store(
        partial_output
        + chunk * stride_o_c
        + token * stride_o_n
        + (first_head + head_offsets[:, None]) * stride_o_h
        + dims[None, :] * stride_o_d,
        accumulator,
        mask=head_mask[:, None] & dim_mask[None, :],
    )
    tl.store(
        partial_lse
        + chunk * stride_l_c
        + token * stride_l_n
        + (first_head + head_offsets) * stride_l_h,
        lse,
        mask=head_mask,
    )


@triton.heuristics({"BLOCK_D": lambda args: triton.next_power_of_2(args["head_dim"])})
@triton.jit
def _merge_decode_kernel(
    partial_output,
    partial_lse,
    output,
    head_dim,
    stride_o_c,
    stride_o_n,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_n,
    stride_l_h,
    stride_out_n,
    stride_out_h,
    stride_out_d,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    chunks = tl.arange(0, NUM_CHUNKS)
    dims = tl.arange(0, BLOCK_D)
    lse = tl.load(
        partial_lse + chunks * stride_l_c + token * stride_l_n + head * stride_l_h
    )
    max_lse = tl.max(lse, axis=0)
    weights = tl.exp2(lse - max_lse)
    weights /= tl.sum(weights, axis=0)
    partials = tl.load(
        partial_output
        + chunks[:, None] * stride_o_c
        + token * stride_o_n
        + head * stride_o_h
        + dims[None, :] * stride_o_d,
        mask=dims[None, :] < head_dim,
        other=0.0,
    )
    merged = tl.sum(partials * weights[:, None], axis=0)
    tl.store(
        output + token * stride_out_n + head * stride_out_h + dims * stride_out_d,
        merged,
        mask=dims < head_dim,
    )


def _validate_inputs(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    selected_blocks: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
) -> tuple[int, int, int, int]:
    if query.dim() != 3:
        raise ValueError(f"query must be [tokens, heads, dim], got {query.shape}")
    if key_cache.shape != value_cache.shape or key_cache.dim() != 4:
        raise ValueError("key/value caches must have matching [pages, heads, 128, dim]")
    if key_cache.shape[2] != SPARSE_BLOCK_SIZE:
        raise ValueError("MiniMax sparse attention requires 128-token pages")
    if selected_blocks.dim() != 3 or selected_blocks.shape[0] != query.shape[0]:
        raise ValueError("selected_blocks must be [tokens, local_kv_heads, topk]")
    if block_table.dim() != 2 or seq_lens.dim() != 1:
        raise ValueError("block_table must be 2-D and seq_lens must be 1-D")
    tokens, num_heads, head_dim = query.shape
    num_kv_heads = key_cache.shape[1]
    if head_dim != 128 or key_cache.shape[3] != head_dim:
        raise ValueError("MiniMax sparse attention requires head dim 128")
    if selected_blocks.shape[1] != num_kv_heads:
        raise ValueError("selected block heads must match local KV heads")
    if num_heads % num_kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    return tokens, num_heads, num_kv_heads, head_dim


@torch.no_grad()
def minimax_sparse_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    selected_blocks: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    scale: float,
    decode_query_len: int = 0,
    cu_seqlens_q: torch.Tensor | None = None,
    prefix_lens: torch.Tensor | None = None,
    max_query_len: int = 0,
) -> torch.Tensor:
    """Run exact GQA attention over MiniMax's selected KV blocks.

    Args:
        query: Main attention queries shaped ``[tokens, local_heads, 128]``.
        key_cache: Paged key cache shaped ``[pages, local_kv_heads, 128, 128]``.
        value_cache: Paged value cache with the same shape as ``key_cache``.
        selected_blocks: Logical block ids shaped
            ``[tokens, local_kv_heads, topk]``.
        block_table: Logical-to-physical page table.
        seq_lens: Total sequence lengths after this step.
        scale: Main attention softmax scale.
        decode_query_len: Uniform queries per request for decode; zero selects
            sparse prefill.
        cu_seqlens_q: Prefill cumulative query lengths.
        prefix_lens: Prefill prefix lengths.
        max_query_len: Maximum query length in the prefill batch.

    Returns:
        Attention output with the same shape and dtype as ``query``.
    """
    tokens, num_heads, num_kv_heads, head_dim = _validate_inputs(
        query,
        key_cache,
        value_cache,
        selected_blocks,
        block_table,
        seq_lens,
    )
    if query.dtype != torch.bfloat16:
        raise TypeError("MiniMax Triton sparse attention requires BF16 query")
    if key_cache.dtype != torch.bfloat16 or value_cache.dtype != torch.bfloat16:
        raise TypeError("MiniMax Triton sparse attention requires BF16 KV cache")
    if selected_blocks.dtype != torch.int32:
        raise TypeError("selected_blocks must be int32")
    if tokens == 0:
        return torch.empty_like(query)

    query = query.contiguous()
    selected_blocks = selected_blocks.contiguous()
    block_table = block_table.to(device=query.device, dtype=torch.int32).contiguous()
    seq_lens = seq_lens.to(device=query.device, dtype=torch.int32).contiguous()
    output = torch.empty_like(query)
    gqa_group_size = num_heads // num_kv_heads
    topk = selected_blocks.shape[2]

    if decode_query_len:
        decode_query_len = int(decode_query_len)
        if tokens != seq_lens.numel() * decode_query_len:
            raise ValueError("decode tokens must equal requests * decode_query_len")
        target_grid = 256
        target_chunks = max(
            1,
            min(topk, target_grid // max(1, tokens * num_kv_heads)),
        )
        num_chunks = 1 << (target_chunks.bit_length() - 1)
        partial_output = torch.empty(
            (num_chunks, tokens, num_heads, head_dim),
            dtype=query.dtype,
            device=query.device,
        )
        partial_lse = torch.empty(
            (num_chunks, tokens, num_heads),
            dtype=torch.float32,
            device=query.device,
        )
        _sparse_decode_kernel[(tokens * num_chunks, num_kv_heads)](
            query,
            key_cache,
            value_cache,
            selected_blocks,
            partial_output,
            partial_lse,
            block_table,
            seq_lens,
            tokens,
            gqa_group_size,
            head_dim,
            topk=topk,
            scale=float(scale),
            decode_query_len=decode_query_len,
            stride_q_n=query.stride(0),
            stride_q_h=query.stride(1),
            stride_q_d=query.stride(2),
            stride_k_page=key_cache.stride(0),
            stride_k_h=key_cache.stride(1),
            stride_k_pos=key_cache.stride(2),
            stride_k_d=key_cache.stride(3),
            stride_v_page=value_cache.stride(0),
            stride_v_h=value_cache.stride(1),
            stride_v_pos=value_cache.stride(2),
            stride_v_d=value_cache.stride(3),
            stride_t_n=selected_blocks.stride(0),
            stride_t_h=selected_blocks.stride(1),
            stride_t_k=selected_blocks.stride(2),
            stride_o_c=partial_output.stride(0),
            stride_o_n=partial_output.stride(1),
            stride_o_h=partial_output.stride(2),
            stride_o_d=partial_output.stride(3),
            stride_l_c=partial_lse.stride(0),
            stride_l_n=partial_lse.stride(1),
            stride_l_h=partial_lse.stride(2),
            stride_bt_b=block_table.stride(0),
            NUM_CHUNKS=num_chunks,
            BLOCK_K=SPARSE_BLOCK_SIZE,
            num_warps=4,
            num_stages=2,
        )
        _merge_decode_kernel[(tokens, num_heads)](
            partial_output,
            partial_lse,
            output,
            head_dim,
            partial_output.stride(0),
            partial_output.stride(1),
            partial_output.stride(2),
            partial_output.stride(3),
            partial_lse.stride(0),
            partial_lse.stride(1),
            partial_lse.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            NUM_CHUNKS=num_chunks,
            num_warps=4,
        )
        return output

    if cu_seqlens_q is None or prefix_lens is None or max_query_len <= 0:
        raise ValueError(
            "sparse prefill requires cu_seqlens_q, prefix_lens, and "
            "positive max_query_len"
        )
    cu_seqlens_q = cu_seqlens_q.to(device=query.device, dtype=torch.int32).contiguous()
    prefix_lens = prefix_lens.to(device=query.device, dtype=torch.int32).contiguous()
    batch = cu_seqlens_q.numel() - 1
    _sparse_prefill_kernel[(int(max_query_len), num_kv_heads, batch)](
        query,
        key_cache,
        value_cache,
        selected_blocks,
        output,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        num_kv_heads,
        gqa_group_size,
        head_dim,
        topk=topk,
        scale=float(scale),
        stride_q_n=query.stride(0),
        stride_q_h=query.stride(1),
        stride_q_d=query.stride(2),
        stride_k_page=key_cache.stride(0),
        stride_k_h=key_cache.stride(1),
        stride_k_pos=key_cache.stride(2),
        stride_k_d=key_cache.stride(3),
        stride_v_page=value_cache.stride(0),
        stride_v_h=value_cache.stride(1),
        stride_v_pos=value_cache.stride(2),
        stride_v_d=value_cache.stride(3),
        stride_t_n=selected_blocks.stride(0),
        stride_t_h=selected_blocks.stride(1),
        stride_t_k=selected_blocks.stride(2),
        stride_o_n=output.stride(0),
        stride_o_h=output.stride(1),
        stride_o_d=output.stride(2),
        stride_bt_b=block_table.stride(0),
        BLOCK_K=SPARSE_BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
    )
    return output


_MINIMAX_MSA_TRAITS = {
    "head_dim": frozenset({128}),
    "index_head_dim": frozenset({128}),
    "page_size": frozenset({128}),
    "topk": frozenset({16}),
}
_MINIMAX_MSA_SIGNATURES = format_signatures(
    ("q", "index_q", "index_k", "k_cache", "v_cache", "index_k_cache"),
    "dense",
    {torch.bfloat16},
)


@register_kernel(
    "attention",
    "msa_decode_with_kvcache",
    name="triton_minimax_msa_decode_with_kvcache",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=_MINIMAX_MSA_SIGNATURES,
    traits=_MINIMAX_MSA_TRAITS,
    priority=Priority.PORTABLE,
    tags={"portability"},
)
def triton_minimax_msa_decode_with_kvcache(
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
) -> torch.Tensor:
    """Run MiniMax sparse-attention decode over paged caches."""

    max_blocks = min(
        page_table.shape[1],
        (max_seqlen_k + page_size - 1) // page_size,
    )
    selected_blocks = minimax_indexer(
        index_q,
        index_k,
        index_k_cache,
        slot_mapping,
        page_table,
        cache_seqlens,
        topk=topk,
        scale=index_scale,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        decode_query_len=max_seqlen_q,
        max_blocks=max_blocks,
    )
    return minimax_sparse_attention(
        q,
        k_cache,
        v_cache,
        selected_blocks,
        page_table,
        cache_seqlens,
        scale=attention_scale,
        decode_query_len=max_seqlen_q,
    )


@register_kernel(
    "attention",
    "msa_extend_with_kvcache",
    name="triton_minimax_msa_extend_with_kvcache",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=_MINIMAX_MSA_SIGNATURES,
    traits=_MINIMAX_MSA_TRAITS,
    priority=Priority.PORTABLE,
    tags={"portability"},
)
def triton_minimax_msa_extend_with_kvcache(
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
) -> torch.Tensor:
    """Run MiniMax sparse-attention extend over paged caches."""

    max_blocks = min(
        page_table.shape[1],
        (max_seqlen_k + page_size - 1) // page_size,
    )
    selected_blocks = minimax_indexer(
        index_q,
        index_k,
        index_k_cache,
        slot_mapping,
        page_table,
        cache_seqlens,
        topk=topk,
        scale=index_scale,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        cu_seqlens_q=cu_seqlens_q,
        prefix_lens=prefix_lens,
        max_query_len=max_seqlen_q,
        max_blocks=max_blocks,
    )
    return minimax_sparse_attention(
        q,
        k_cache,
        v_cache,
        selected_blocks,
        page_table,
        cache_seqlens,
        scale=attention_scale,
        cu_seqlens_q=cu_seqlens_q,
        prefix_lens=prefix_lens,
        max_query_len=max_seqlen_q,
    )

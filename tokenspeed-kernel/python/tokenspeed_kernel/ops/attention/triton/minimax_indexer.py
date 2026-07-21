# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""MiniMax index-key cache, block scoring, and block Top-K kernels.

The implementation follows MiniMax Sparse Attention's inference algorithm:
one index-query head per GQA group scores a shared index-key head, token scores
are max-reduced inside 128-token blocks, and the current block is forced into
the selected set.  The code is adapted to TokenSpeed's paged-cache layout.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.triton.dsa_topk import _topk_with_padding

SPARSE_BLOCK_SIZE = 128


@triton.jit
def _store_index_k_kernel(
    index_k,
    index_k_cache,
    slot_mapping,
    stride_k_n,
    stride_k_d,
    stride_cache_n,
    stride_cache_d,
    head_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    slot = tl.load(slot_mapping + token).to(tl.int64)
    values = tl.load(
        index_k + token * stride_k_n + dims * stride_k_d,
        mask=dims < head_dim,
        other=0.0,
    )
    tl.store(
        index_k_cache + slot * stride_cache_n + dims * stride_cache_d,
        values,
        mask=dims < head_dim,
    )


@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _prefill_block_score_kernel(
    index_q,
    index_k_cache,
    scores,
    block_table,
    cu_seqlens_q,
    seq_lens,
    prefix_lens,
    num_index_heads,
    scale,
    init_blocks: tl.constexpr,
    local_blocks: tl.constexpr,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_k_page,
    stride_k_pos,
    stride_k_d,
    stride_s_n,
    stride_s_h,
    stride_s_b,
    stride_bt_b,
    head_dim: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    query_tile = tl.program_id(0)
    batch_head = tl.program_id(1)
    request = batch_head // num_index_heads
    head = batch_head % num_index_heads

    query_start = tl.load(cu_seqlens_q + request)
    query_len = tl.load(cu_seqlens_q + request + 1) - query_start
    if query_tile * BLOCK_Q >= query_len:
        return

    seq_len = tl.load(seq_lens + request)
    prefix_len = tl.load(prefix_lens + request)
    query_offsets = query_tile * BLOCK_Q + tl.arange(0, BLOCK_Q)
    query_positions = prefix_len + query_offsets
    query_mask = query_offsets < query_len
    dims = tl.arange(0, head_dim)
    key_offsets = tl.arange(0, BLOCK_K)

    query = tl.load(
        index_q
        + (query_start + query_offsets[:, None]) * stride_q_n
        + head * stride_q_h
        + dims[None, :] * stride_q_d,
        mask=query_mask[:, None],
        other=0.0,
    )
    block_table_row = block_table + request * stride_bt_b
    visible_end = tl.minimum(
        seq_len,
        prefix_len + (query_tile + 1) * BLOCK_Q,
    )

    for key_start in tl.range(0, visible_end, BLOCK_K):
        block = key_start // BLOCK_K
        page = tl.load(block_table_row + block).to(tl.int64)
        key_positions = key_start + key_offsets
        key = tl.load(
            index_k_cache
            + page * stride_k_page
            + key_offsets[None, :] * stride_k_pos
            + dims[:, None] * stride_k_d,
        )
        logits = tl.dot(query, key, out_dtype=tl.float32) * scale
        logits = tl.where(
            query_positions[:, None] >= key_positions[None, :],
            logits,
            -float("inf"),
        )
        block_scores = tl.max(logits, axis=1)
        query_blocks = query_positions // BLOCK_K
        forced_init = block < init_blocks
        forced_local = (block <= query_blocks) & (block > query_blocks - local_blocks)
        block_scores = tl.where(
            forced_init | forced_local,
            float("inf"),
            block_scores,
        )
        tl.store(
            scores
            + (query_start + query_offsets) * stride_s_n
            + head * stride_s_h
            + block * stride_s_b,
            block_scores,
            mask=query_mask,
        )


@triton.jit(do_not_specialize=["decode_query_len"])
def _decode_block_score_kernel(
    index_q,
    index_k_cache,
    scores,
    block_table,
    seq_lens,
    num_index_heads: tl.constexpr,
    scale,
    init_blocks: tl.constexpr,
    local_blocks: tl.constexpr,
    decode_query_len,
    max_blocks,
    num_chunks: tl.constexpr,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_k_page,
    stride_k_pos,
    stride_k_d,
    stride_s_n,
    stride_s_h,
    stride_s_b,
    stride_bt_b,
    head_dim: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    request = tl.program_id(0)
    chunk = tl.program_id(1)
    head_query_width: tl.constexpr = num_index_heads * BLOCK_Q
    packed_offsets = tl.arange(0, head_query_width)
    head_offsets = packed_offsets // BLOCK_Q
    query_offsets = packed_offsets % BLOCK_Q
    query_mask = query_offsets < decode_query_len
    query_ids = request * decode_query_len + query_offsets

    seq_len = tl.load(seq_lens + request)
    query_positions = seq_len - decode_query_len + query_offsets
    visible_lens = tl.maximum(query_positions + 1, 0)
    visible_blocks_per_query = (visible_lens + BLOCK_K - 1) // BLOCK_K
    visible_blocks = tl.max(tl.where(query_mask, visible_blocks_per_query, 0), axis=0)

    blocks_per_chunk = (max_blocks + num_chunks - 1) // num_chunks
    chunk_start = chunk * blocks_per_chunk
    chunk_end = tl.minimum(chunk_start + blocks_per_chunk, visible_blocks)
    if chunk_start >= chunk_end:
        return

    dims = tl.arange(0, head_dim)
    key_offsets = tl.arange(0, BLOCK_K)
    query = tl.load(
        index_q
        + query_ids[None, :] * stride_q_n
        + head_offsets[None, :] * stride_q_h
        + dims[:, None] * stride_q_d,
        mask=query_mask[None, :],
        other=0.0,
    )
    block_table_row = block_table + request * stride_bt_b

    for block in tl.range(chunk_start, chunk_end):
        page = tl.load(block_table_row + block).to(tl.int64)
        key_positions = block * BLOCK_K + key_offsets
        key = tl.load(
            index_k_cache
            + page * stride_k_page
            + key_offsets[:, None] * stride_k_pos
            + dims[None, :] * stride_k_d,
        )
        logits = tl.dot(key, query, out_dtype=tl.float32) * scale
        logits = tl.where(
            (key_positions[:, None] < visible_lens[None, :]) & query_mask[None, :],
            logits,
            -float("inf"),
        )
        block_scores = tl.max(logits, axis=0)
        forced_init = block < init_blocks
        local_start = tl.maximum(0, visible_blocks_per_query - local_blocks)
        forced_local = (block >= local_start) & (block < visible_blocks_per_query)
        block_scores = tl.where(
            forced_init | forced_local,
            float("inf"),
            block_scores,
        )
        tl.store(
            scores
            + query_ids * stride_s_n
            + head_offsets * stride_s_h
            + block * stride_s_b,
            block_scores,
            mask=query_mask,
        )


def _validate_inputs(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    index_k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    topk: int,
) -> tuple[int, int, int]:
    if index_q.dim() != 3:
        raise ValueError(f"index_q must be [tokens, heads, dim], got {index_q.shape}")
    if index_k.shape != (index_q.shape[0], index_q.shape[2]):
        raise ValueError(
            "index_k must be [tokens, dim] matching index_q, got "
            f"index_q={tuple(index_q.shape)}, index_k={tuple(index_k.shape)}"
        )
    if index_k_cache.dim() != 2 or index_k_cache.shape[1] != index_q.shape[2]:
        raise ValueError(
            "index_k_cache must be [slots, dim] matching index_q, got "
            f"{tuple(index_k_cache.shape)}"
        )
    if slot_mapping.numel() != index_q.shape[0]:
        raise ValueError("slot_mapping must contain one slot per query token")
    if block_table.dim() != 2 or seq_lens.dim() != 1:
        raise ValueError("block_table must be 2-D and seq_lens must be 1-D")
    if block_table.shape[0] < seq_lens.numel():
        raise ValueError("block_table must contain one row per request")
    if topk <= 0 or topk & (topk - 1):
        raise ValueError(f"topk must be a power of two, got {topk}")
    tokens, num_heads, head_dim = index_q.shape
    if head_dim != 128:
        raise ValueError(f"MiniMax index head dim must be 128, got {head_dim}")
    return tokens, num_heads, head_dim


@torch.no_grad()
def minimax_indexer(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    index_k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    topk: int,
    scale: float,
    init_blocks: int,
    local_blocks: int,
    decode_query_len: int = 0,
    cu_seqlens_q: torch.Tensor | None = None,
    prefix_lens: torch.Tensor | None = None,
    max_query_len: int = 0,
    max_blocks: int | None = None,
) -> torch.Tensor:
    """Write index keys, score visible 128-token blocks, and select Top-K.

    Args:
        index_q: Normalized and rotary-embedded index queries shaped
            ``[tokens, local_index_heads, 128]``.
        index_k: Normalized and rotary-embedded shared index keys shaped
            ``[tokens, 128]``.
        index_k_cache: Per-layer side cache shaped ``[slots, 128]``.
        slot_mapping: Cache slot for every input token.
        block_table: Logical-to-physical page table shaped ``[requests, pages]``.
        seq_lens: Total sequence length for each request after this step.
        topk: Number of selected blocks. MiniMax sparse attention uses 16.
        scale: Index score scale, normally ``128**-0.5``.
        init_blocks: Number of leading blocks that are always selected.
        local_blocks: Number of most-recent blocks that are always selected.
        decode_query_len: Uniform queries per request for decode; zero selects
            the prefill path.
        cu_seqlens_q: Prefill cumulative query lengths.
        prefix_lens: Prefill cached-prefix lengths.
        max_query_len: Maximum prefill query length.
        max_blocks: Number of logical block columns to score. Defaults to the
            page-table width; prefill callers should pass the current batch's
            upper bound to avoid allocating scores for the full 1M context.

    Returns:
        Selected logical block ids shaped ``[tokens, local_index_heads, topk]``.
    """
    tokens, num_heads, head_dim = _validate_inputs(
        index_q,
        index_k,
        index_k_cache,
        slot_mapping,
        block_table,
        seq_lens,
        int(topk),
    )
    if tokens == 0:
        return torch.empty(
            (0, num_heads, int(topk)),
            dtype=torch.int32,
            device=index_q.device,
        )
    if index_q.dtype != torch.bfloat16:
        raise TypeError("MiniMax Triton indexer requires BF16 index queries")
    if index_k.dtype != torch.bfloat16:
        raise TypeError("MiniMax Triton indexer requires BF16 index keys")
    if index_k_cache.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise TypeError(
            "MiniMax Triton indexer requires a BF16 or FP8 E4M3 index cache"
        )
    if index_k_cache.shape[0] % SPARSE_BLOCK_SIZE:
        raise ValueError("index_k_cache slot count must be divisible by 128")

    index_q = index_q.contiguous()
    index_k = index_k.contiguous()
    slot_mapping = slot_mapping.to(
        device=index_q.device, dtype=torch.int32
    ).contiguous()
    block_table = block_table.to(device=index_q.device, dtype=torch.int32).contiguous()
    seq_lens = seq_lens.to(device=index_q.device, dtype=torch.int32).contiguous()
    block_d = triton.next_power_of_2(head_dim)
    _store_index_k_kernel[(tokens,)](
        index_k,
        index_k_cache,
        slot_mapping,
        index_k.stride(0),
        index_k.stride(1),
        index_k_cache.stride(0),
        index_k_cache.stride(1),
        head_dim=head_dim,
        BLOCK_D=block_d,
        num_warps=4,
    )

    max_blocks = block_table.shape[1] if max_blocks is None else int(max_blocks)
    if not 0 < max_blocks <= block_table.shape[1]:
        raise ValueError(
            f"max_blocks must be in [1, {block_table.shape[1]}], got {max_blocks}"
        )
    scores = torch.full(
        (tokens, num_heads, max_blocks),
        -float("inf"),
        dtype=torch.float32,
        device=index_q.device,
    )
    cache_pages = index_k_cache.view(-1, SPARSE_BLOCK_SIZE, head_dim)
    if decode_query_len:
        decode_query_len = int(decode_query_len)
        if tokens != seq_lens.numel() * decode_query_len:
            raise ValueError(
                "decode tokens must equal requests * decode_query_len, got "
                f"tokens={tokens}, requests={seq_lens.numel()}, "
                f"decode_query_len={decode_query_len}"
            )
        num_chunks = 64
        block_q = triton.next_power_of_2(decode_query_len)
        _decode_block_score_kernel[(seq_lens.numel(), num_chunks)](
            index_q,
            cache_pages,
            scores,
            block_table,
            seq_lens,
            num_index_heads=num_heads,
            scale=float(scale),
            init_blocks=int(init_blocks),
            local_blocks=int(local_blocks),
            decode_query_len=decode_query_len,
            max_blocks=max_blocks,
            num_chunks=num_chunks,
            stride_q_n=index_q.stride(0),
            stride_q_h=index_q.stride(1),
            stride_q_d=index_q.stride(2),
            stride_k_page=cache_pages.stride(0),
            stride_k_pos=cache_pages.stride(1),
            stride_k_d=cache_pages.stride(2),
            stride_s_n=scores.stride(0),
            stride_s_h=scores.stride(1),
            stride_s_b=scores.stride(2),
            stride_bt_b=block_table.stride(0),
            head_dim=head_dim,
            BLOCK_Q=block_q,
            BLOCK_K=SPARSE_BLOCK_SIZE,
            num_warps=4,
            num_stages=2,
        )
    else:
        if cu_seqlens_q is None or prefix_lens is None or max_query_len <= 0:
            raise ValueError(
                "prefill indexer requires cu_seqlens_q, prefix_lens, and "
                "positive max_query_len"
            )
        cu_seqlens_q = cu_seqlens_q.to(
            device=index_q.device, dtype=torch.int32
        ).contiguous()
        prefix_lens = prefix_lens.to(
            device=index_q.device, dtype=torch.int32
        ).contiguous()
        batch = cu_seqlens_q.numel() - 1
        block_q = 64
        _prefill_block_score_kernel[
            (triton.cdiv(int(max_query_len), block_q), batch * num_heads)
        ](
            index_q,
            cache_pages,
            scores,
            block_table,
            cu_seqlens_q,
            seq_lens,
            prefix_lens,
            num_heads,
            float(scale),
            init_blocks=int(init_blocks),
            local_blocks=int(local_blocks),
            stride_q_n=index_q.stride(0),
            stride_q_h=index_q.stride(1),
            stride_q_d=index_q.stride(2),
            stride_k_page=cache_pages.stride(0),
            stride_k_pos=cache_pages.stride(1),
            stride_k_d=cache_pages.stride(2),
            stride_s_n=scores.stride(0),
            stride_s_h=scores.stride(1),
            stride_s_b=scores.stride(2),
            stride_bt_b=block_table.stride(0),
            head_dim=head_dim,
            BLOCK_Q=block_q,
            BLOCK_K=SPARSE_BLOCK_SIZE,
            num_warps=8,
            num_stages=2,
        )

    selected = _topk_with_padding(scores.view(tokens * num_heads, max_blocks), topk)
    return selected.view(tokens, num_heads, int(topk))


__all__ = ["SPARSE_BLOCK_SIZE", "minimax_indexer"]

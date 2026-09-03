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

"""Portable Triton attend kernel for Qwen4-Exp QSA-selected cache slots."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.cuda.dsa_topk import (
    has_ragged_decode_topk,
    ragged_decode_topk,
)
from tokenspeed_kernel.ops.attention.triton.dsa_topk import (
    triton_topk_from_logits,
)
from tokenspeed_kernel.platform import current_platform

_is_nvidia = current_platform().is_nvidia
_PERSISTENT_TOPK_WORKSPACE_BYTES = 1024 * 1024


@triton.heuristics(
    {
        "BLOCK_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_H": lambda args: triton.next_power_of_2(args["num_heads"]),
    }
)
@triton.jit
def _qwen4_exp_qsa_mqa_scores_kernel(
    query,
    key_cache,
    key_slots,
    valid_counts,
    output,
    num_heads,
    head_dim,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_k_n,
    stride_k_d,
    stride_s_n,
    stride_s_k,
    stride_vc_n,
    stride_o_n,
    stride_o_k,
    NUM_KEYS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    key_block = tl.program_id(1)
    key_offsets = key_block * BLOCK_N + tl.arange(0, BLOCK_N)
    head_offsets = tl.arange(0, BLOCK_H)
    dim_offsets = tl.arange(0, BLOCK_D)
    head_mask = head_offsets < num_heads
    dim_mask = dim_offsets < head_dim
    key_mask = key_offsets < tl.load(valid_counts + row * stride_vc_n)
    slots = tl.load(
        key_slots + row * stride_s_n + key_offsets * stride_s_k,
        mask=key_offsets < NUM_KEYS,
        other=0,
    ).to(tl.int64)
    query_values = tl.load(
        query
        + row * stride_q_n
        + head_offsets[:, None] * stride_q_h
        + dim_offsets[None, :] * stride_q_d,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    key_values = tl.load(
        key_cache + slots[None, :] * stride_k_n + dim_offsets[:, None] * stride_k_d,
        mask=dim_mask[:, None] & key_mask[None, :],
        other=0.0,
    )
    scores = tl.dot(query_values, key_values, out_dtype=tl.float32)
    scores = tl.maximum(scores, 0.0)
    scores = tl.sum(tl.where(head_mask[:, None], scores, 0.0), axis=0)
    # Every key column is stored, so the invalid tail carries -inf in-kernel
    # instead of relying on a separate fill of the output buffer.
    scores = tl.where(key_mask, scores, -float("inf"))
    tl.store(
        output + row * stride_o_n + key_offsets * stride_o_k,
        scores,
        mask=key_offsets < NUM_KEYS,
    )


def qwen4_exp_qsa_mqa_scores(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    key_slots: torch.Tensor,
    valid_counts: torch.Tensor,
) -> torch.Tensor:
    """Score compressed QSA keys with the weight-free MQA indexer."""

    if query.ndim != 3 or key_cache.ndim != 3:
        raise ValueError("Qwen4-Exp QSA scoring expects rank-three Q and K tensors")
    if key_cache.shape[1] != 1 or query.shape[2] != key_cache.shape[2]:
        raise ValueError("Qwen4-Exp QSA scoring requires one matching KV index head")
    if key_slots.ndim != 2 or key_slots.shape[0] != query.shape[0]:
        raise ValueError("Qwen4-Exp QSA key slots must have one row per query")
    rows, num_keys = key_slots.shape
    # The scoring kernel writes every column including the -inf invalid
    # tail, so the buffer starts uninitialized.
    output = torch.empty((rows, num_keys), dtype=torch.float32, device=query.device)
    if rows == 0 or num_keys == 0:
        return output
    # Strided reads and in-kernel casts keep the launch free of host-side
    # copy/cast kernels; ``.to`` only enforces the device.
    key_slots = key_slots.to(device=query.device)
    valid_counts = valid_counts.to(device=query.device)
    _qwen4_exp_qsa_mqa_scores_kernel[(rows, triton.cdiv(num_keys, 64))](
        query,
        key_cache,
        key_slots,
        valid_counts,
        output,
        query.shape[1],
        query.shape[2],
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key_cache.stride(0),
        key_cache.stride(2),
        key_slots.stride(0),
        key_slots.stride(1),
        valid_counts.stride(0),
        output.stride(0),
        output.stride(1),
        NUM_KEYS=num_keys,
        BLOCK_N=64,
        num_warps=4,
        num_stages=2,
    )
    return output


@triton.heuristics(
    {
        "BLOCK_D": lambda args: triton.next_power_of_2(
            max(args["head_dim"], args["value_head_dim"])
        ),
        "BLOCK_H": lambda args: max(16, triton.next_power_of_2(args["gqa_group_size"])),
    }
)
@triton.jit
def _qwen4_exp_qsa_sparse_attention_kernel(
    query,
    key_cache,
    value_cache,
    selected_slots,
    output,
    num_kv_heads,
    gqa_group_size,
    head_dim,
    value_head_dim,
    scale,
    k_descale,
    v_descale,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_k_n,
    stride_k_h,
    stride_k_d,
    stride_v_n,
    stride_v_h,
    stride_v_d,
    stride_s_n,
    stride_s_k,
    stride_o_n,
    stride_o_h,
    stride_o_d,
    TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    USE_FP8: tl.constexpr,
):
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    head_offsets = tl.arange(0, BLOCK_H)
    dim_offsets = tl.arange(0, BLOCK_D)
    head_mask = head_offsets < gqa_group_size
    q_dim_mask = dim_offsets < head_dim
    v_dim_mask = dim_offsets < value_head_dim
    first_head = kv_head * gqa_group_size
    q = tl.load(
        query
        + row * stride_q_n
        + (first_head + head_offsets[:, None]) * stride_q_h
        + dim_offsets[None, :] * stride_q_d,
        mask=head_mask[:, None] & q_dim_mask[None, :],
        other=0.0,
    )
    q = (q * scale * 1.4426950408889634).to(q.dtype)

    maximum = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_H,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)
    token_offsets = tl.arange(0, BLOCK_N)
    slot_row = selected_slots + row * stride_s_n

    for start in range(0, TOPK, BLOCK_N):
        selected_offsets = start + token_offsets
        slots = tl.load(
            slot_row + selected_offsets * stride_s_k,
            mask=selected_offsets < TOPK,
            other=-1,
        ).to(tl.int64)
        valid = (selected_offsets < TOPK) & (slots > 0)
        keys = tl.load(
            key_cache
            + slots[None, :] * stride_k_n
            + kv_head * stride_k_h
            + dim_offsets[:, None] * stride_k_d,
            mask=q_dim_mask[:, None] & valid[None, :],
            other=0.0,
        )
        values = tl.load(
            value_cache
            + slots[:, None] * stride_v_n
            + kv_head * stride_v_h
            + dim_offsets[None, :] * stride_v_d,
            mask=valid[:, None] & v_dim_mask[None, :],
            other=0.0,
        )
        if USE_FP8:
            keys = (keys.to(tl.float32) * k_descale).to(q.dtype)
            values = (values.to(tl.float32) * v_descale).to(q.dtype)
        scores = tl.dot(q, keys, out_dtype=tl.float32)
        scores = tl.where(head_mask[:, None] & valid[None, :], scores, -float("inf"))
        has_valid = head_mask & (tl.sum(valid.to(tl.int32), axis=0) > 0)
        block_maximum = tl.max(scores, axis=1)
        next_maximum = tl.where(has_valid, tl.maximum(maximum, block_maximum), maximum)
        correction = tl.where(has_valid, tl.exp2(maximum - next_maximum), 1.0)
        probabilities = tl.where(
            head_mask[:, None] & valid[None, :],
            tl.exp2(scores - next_maximum[:, None]),
            0.0,
        )
        accumulator *= correction[:, None]
        accumulator += tl.dot(probabilities.to(values.dtype), values)
        normalizer = normalizer * correction + tl.sum(probabilities, axis=1)
        maximum = next_maximum

    result = tl.where(
        normalizer[:, None] > 0,
        accumulator / normalizer[:, None],
        0.0,
    )
    tl.store(
        output
        + row * stride_o_n
        + (first_head + head_offsets[:, None]) * stride_o_h
        + dim_offsets[None, :] * stride_o_d,
        result,
        mask=head_mask[:, None] & v_dim_mask[None, :],
    )


def qwen4_exp_qsa_sparse_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    selected_slots: torch.Tensor,
    *,
    scale: float,
    k_scale: float | None = None,
    v_scale: float | None = None,
) -> torch.Tensor:
    """Attend exactly to selected physical slots in a flattened paged cache.

    Args:
        query: Query tensor shaped ``[tokens, query_heads, head_dim]``.
        key_cache: Flattened paged keys shaped ``[slots, kv_heads, head_dim]``.
        value_cache: Flattened paged values shaped
            ``[slots, kv_heads, value_head_dim]``.
        selected_slots: Physical cache slots shaped ``[tokens, budget]``;
            non-positive entries are ignored.
        scale: Softmax scale applied to query-key scores.
        k_scale: Optional scalar descale for FP8 keys.
        v_scale: Optional scalar descale for FP8 values.

    Returns:
        Attention output shaped ``[tokens, query_heads, value_head_dim]``.
    """

    if query.ndim != 3 or key_cache.ndim != 3 or value_cache.ndim != 3:
        raise ValueError("Qwen4-Exp QSA expects rank-three query and cache tensors")
    if selected_slots.ndim != 2 or selected_slots.shape[0] != query.shape[0]:
        raise ValueError("selected_slots must have one row per query token")
    if key_cache.shape[:2] != value_cache.shape[:2]:
        raise ValueError("Qwen4-Exp QSA key/value cache geometry must match")
    if query.shape[2] != key_cache.shape[2]:
        raise ValueError("Qwen4-Exp QSA query/key head dimensions must match")
    num_kv_heads = key_cache.shape[1]
    if query.shape[1] % num_kv_heads:
        raise ValueError("Qwen4-Exp QSA query heads must be divisible by KV heads")
    if query.shape[0] == 0:
        return query.new_empty((0, query.shape[1], value_cache.shape[2]))

    use_fp8 = key_cache.dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2fnuz,
    )
    if key_cache.dtype != value_cache.dtype:
        raise TypeError("Qwen4-Exp QSA key and value caches must share one dtype")
    if use_fp8 != (k_scale is not None and v_scale is not None):
        raise ValueError("FP8 Qwen4-Exp QSA cache requires both K and V scales")
    # Query and slot tensors are read through their strides with in-kernel
    # casts, so no host-side copy/cast kernel is launched.
    output = torch.empty(
        (query.shape[0], query.shape[1], value_cache.shape[2]),
        dtype=query.dtype,
        device=query.device,
    )
    group_size = query.shape[1] // num_kv_heads
    _qwen4_exp_qsa_sparse_attention_kernel[(query.shape[0], num_kv_heads)](
        query,
        key_cache,
        value_cache,
        selected_slots,
        output,
        num_kv_heads,
        group_size,
        query.shape[2],
        value_cache.shape[2],
        float(scale),
        1.0 if k_scale is None else float(k_scale),
        1.0 if v_scale is None else float(v_scale),
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        selected_slots.stride(0),
        selected_slots.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        TOPK=selected_slots.shape[1],
        BLOCK_N=32,
        USE_FP8=use_fp8,
        num_warps=4,
        num_stages=2,
    )
    return output


@triton.jit
def _qwen4_exp_qsa_logical_layout_kernel(
    seq_lens,
    query_lengths,
    positions,
    requests,
    uniform_len,
    stride_seq_b,
    stride_len_b,
    HAS_UNIFORM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(0)
    if HAS_UNIFORM:
        length = tl.full((), uniform_len, tl.int64)
        start = request.to(tl.int64) * uniform_len
    else:
        length = tl.load(query_lengths + request * stride_len_b).to(tl.int64)
        start = tl.zeros((), dtype=tl.int64)
        for other in range(0, request):
            start += tl.load(query_lengths + other * stride_len_b).to(tl.int64)
    base = tl.load(seq_lens + request * stride_seq_b).to(tl.int64) - length
    for offset in range(0, length, BLOCK):
        slots = offset + tl.arange(0, BLOCK)
        mask = slots < length
        rows = start + slots
        values = base + slots
        tl.store(positions + rows, values, mask=mask)
        tl.store(requests + rows, tl.full((BLOCK,), request, tl.int64), mask=mask)


def qwen4_exp_qsa_logical_layout(
    seq_lens: torch.Tensor,
    query_lengths: torch.Tensor | int,
    total_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand per-request sequence geometry into per-token layout vectors.

    One kernel replaces the repeat-interleave/cumsum chain used to derive
    logical positions and request ids for flattened batches.

    Args:
        seq_lens: Cached sequence length per request.
        query_lengths: Tokens contributed by each request in this step,
            either a per-request tensor (read in place, any int dtype) or a
            Python int when every request contributes the same length.
        total_tokens: Number of flattened token rows to materialize.

    Returns:
        ``(positions, requests)`` int64 tensors shaped ``[total_tokens]``.
    """

    device = seq_lens.device
    positions = torch.empty((total_tokens,), dtype=torch.int64, device=device)
    requests = torch.empty((total_tokens,), dtype=torch.int64, device=device)
    batch = seq_lens.shape[0]
    if total_tokens and batch:
        if isinstance(query_lengths, int):
            uniform_len = int(query_lengths)
            lengths_arg = seq_lens
            stride_len_b = 0
            has_uniform = True
        else:
            uniform_len = 0
            lengths_arg = query_lengths
            stride_len_b = query_lengths.stride(0)
            has_uniform = False
        _qwen4_exp_qsa_logical_layout_kernel[(batch,)](
            seq_lens,
            lengths_arg,
            positions,
            requests,
            uniform_len,
            seq_lens.stride(0),
            stride_len_b,
            HAS_UNIFORM=has_uniform,
            BLOCK=128,
        )
    return positions, requests


@triton.jit
def _qwen4_exp_qsa_group_cache_locs_kernel(
    logical_positions,
    request_indices,
    qsa_page_table,
    qsa_expansion,
    qsa_page_size,
    recent_page_table,
    recent_expansion,
    recent_page_size,
    qsa_locs,
    recent_locs,
    complete_blocks,
    compress_ratio,
    rows,
    stride_qsa_pt_b,
    stride_recent_pt_b,
    BLOCK: tl.constexpr,
):
    block = tl.program_id(0)
    row_ids = block * BLOCK + tl.arange(0, BLOCK)
    mask = row_ids < rows
    positions = tl.load(logical_positions + row_ids, mask=mask, other=0).to(tl.int64)
    requests = tl.load(request_indices + row_ids, mask=mask, other=0).to(tl.int64)
    valid = positions >= 0
    safe_positions = tl.maximum(positions, 0)
    # Complete compressed blocks before each row ride along for free: both
    # outputs are per-row functions of the same logical position vector.
    counts = (positions + 1) // compress_ratio
    tl.store(complete_blocks + row_ids, counts.to(tl.int32), mask=mask)
    # Compressed group: the backend page table is stored at consumer
    # granularity, so undo the expansion inline (entry ``col * expansion``
    # holds ``expansion`` consumer pages of one logical page).
    qsa_columns = safe_positions // qsa_page_size
    qsa_pages = tl.load(
        qsa_page_table + requests * stride_qsa_pt_b + qsa_columns * qsa_expansion,
        mask=mask & valid,
        other=0,
    ).to(tl.int64)
    qsa_pages = qsa_pages // qsa_expansion
    qsa_values = qsa_pages * qsa_page_size + safe_positions % qsa_page_size
    tl.store(
        qsa_locs + row_ids,
        tl.where(valid & (qsa_pages > 0), qsa_values, 0).to(tl.int32),
        mask=mask,
    )
    # Recent raw group, same expansion reversal with its own geometry.
    recent_columns = safe_positions // recent_page_size
    recent_pages = tl.load(
        recent_page_table
        + requests * stride_recent_pt_b
        + recent_columns * recent_expansion,
        mask=mask & valid,
        other=0,
    ).to(tl.int64)
    recent_pages = recent_pages // recent_expansion
    recent_values = recent_pages * recent_page_size + safe_positions % recent_page_size
    tl.store(
        recent_locs + row_ids,
        tl.where(valid & (recent_pages > 0), recent_values, 0).to(tl.int32),
        mask=mask,
    )


def qwen4_exp_qsa_group_cache_locs(
    logical_positions: torch.Tensor,
    request_indices: torch.Tensor,
    qsa_page_table: torch.Tensor,
    qsa_expansion: int,
    qsa_page_size: int,
    recent_page_table: torch.Tensor,
    recent_expansion: int,
    recent_page_size: int,
    compress_ratio: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map flattened token positions into both QSA cache groups at once.

    Reads the backend page tables at consumer granularity and reverses the
    logical-to-kernel expansion in-kernel, so no reconstructed page table is
    materialized. The per-row complete compressed-block counts share the
    same position vector, so they are fused into this single launch instead
    of running ``qwen4_exp_qsa_complete_blocks`` separately.

    Args:
        logical_positions: Absolute logical position per token row.
        request_indices: Owning request id per token row.
        qsa_page_table: Compressed-group page table shaped
            ``[requests, max_pages]`` at consumer granularity.
        qsa_expansion: Consumer pages covered by one compressed logical page.
        qsa_page_size: Logical tokens covered by one compressed page.
        recent_page_table: Recent-group page table shaped
            ``[requests, max_pages]`` at consumer granularity.
        recent_expansion: Consumer pages covered by one recent logical page.
        recent_page_size: Logical tokens covered by one recent page.
        compress_ratio: Tokens grouped into one compressed block; used for
            the fused complete-block counts.

    Returns:
        ``(qsa_locs, recent_locs, complete_blocks)`` int32 slot vectors plus
        int32 complete-block counts; rows with negative positions or
        unallocated pages map to zero.
    """

    rows = logical_positions.shape[0]
    device = logical_positions.device
    qsa_locs = torch.empty((rows,), dtype=torch.int32, device=device)
    recent_locs = torch.empty((rows,), dtype=torch.int32, device=device)
    complete_blocks = torch.empty((rows,), dtype=torch.int32, device=device)
    if rows:
        _qwen4_exp_qsa_group_cache_locs_kernel[(triton.cdiv(rows, 256),)](
            logical_positions,
            request_indices,
            qsa_page_table,
            qsa_expansion,
            qsa_page_size,
            recent_page_table,
            recent_expansion,
            recent_page_size,
            qsa_locs,
            recent_locs,
            complete_blocks,
            compress_ratio,
            rows,
            qsa_page_table.stride(0),
            recent_page_table.stride(0),
            BLOCK=256,
        )
    return qsa_locs, recent_locs, complete_blocks


@triton.jit
def _qwen4_exp_qsa_complete_blocks_kernel(
    logical_positions,
    complete_blocks,
    rows,
    compress_ratio,
    BLOCK: tl.constexpr,
):
    block = tl.program_id(0)
    row_ids = block * BLOCK + tl.arange(0, BLOCK)
    mask = row_ids < rows
    positions = tl.load(logical_positions + row_ids, mask=mask, other=0).to(tl.int64)
    counts = (positions + 1) // compress_ratio
    tl.store(complete_blocks + row_ids, counts.to(tl.int32), mask=mask)


def qwen4_exp_qsa_complete_blocks(
    logical_positions: torch.Tensor,
    compress_ratio: int,
) -> torch.Tensor:
    """Count fully compressed blocks available before each token row.

    Args:
        logical_positions: Absolute logical position per token row.
        compress_ratio: Tokens grouped into one compressed block.

    Returns:
        Int32 complete-block counts shaped ``[rows]``.
    """

    rows = logical_positions.shape[0]
    counts = torch.empty((rows,), dtype=torch.int32, device=logical_positions.device)
    if rows:
        _qwen4_exp_qsa_complete_blocks_kernel[(triton.cdiv(rows, 256),)](
            logical_positions,
            counts,
            rows,
            compress_ratio,
            BLOCK=256,
        )
    return counts


@triton.jit
def _qwen4_exp_qsa_compress_and_store_kernel(
    token_k,
    logical_positions,
    request_indices,
    recent_locs,
    raw_cache,
    position_values,
    position_cache,
    norm_weight,
    cos_sin_cache,
    qsa_locs,
    compressed_cache,
    write_mask,
    draft_raw_cache,
    draft_logical_positions,
    draft_position_cache,
    head_dim,
    rotary_dim,
    norm_epsilon,
    recent_page_size,
    compress_ratio,
    compressed_token_page_size,
    compressed_rows_per_page,
    section0,
    section1,
    section2,
    stride_k_n,
    stride_k_d,
    stride_raw_p,
    stride_raw_s,
    stride_raw_d,
    stride_pv_n,
    stride_pv_a,
    stride_pc_p,
    stride_cs_n,
    stride_cc_row,
    stride_cc_d,
    stride_drc_r,
    stride_drc_s,
    stride_drc_d,
    stride_dlp_r,
    stride_dlp_s,
    stride_dpc_r,
    COMPRESS_RATIO: tl.constexpr,
    HAS_WRITE_MASK: tl.constexpr,
    HAS_DRAFT: tl.constexpr,
    HAS_SECTIONS: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    row = tl.program_id(0)
    half = rotary_dim // 2
    half_offsets = tl.arange(0, BLOCK_HALF)
    half_mask = half_offsets < half
    second_offsets = half + half_offsets
    dim_offsets = tl.arange(0, BLOCK_D)
    pass_mask = (dim_offsets >= rotary_dim) & (dim_offsets < head_dim)
    position = tl.load(logical_positions + row).to(tl.int64)
    request = tl.load(request_indices + row).to(tl.int64)
    recent_loc = tl.load(recent_locs + row).to(tl.int64)
    page = tl.maximum(recent_loc, 0) // recent_page_size

    # Gather the compression group and average its raw keys. Members still
    # present in the current batch come from the input keys; older members
    # fall back to the raw cache page. The fused path must finish these
    # reads before any kernel writes new raw keys.
    acc_first = tl.zeros((BLOCK_HALF,), dtype=tl.float32)
    acc_second = tl.zeros((BLOCK_HALF,), dtype=tl.float32)
    acc_pass = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for member in tl.static_range(COMPRESS_RATIO):
        offset = COMPRESS_RATIO - 1 - member
        source = row - offset
        expected = position - offset
        from_current = source >= 0
        safe_source = tl.maximum(source, 0)
        if from_current:
            from_current = (
                tl.load(request_indices + source).to(tl.int64) == request
            ) & (tl.load(logical_positions + source).to(tl.int64) == expected)
        slot = (expected % COMPRESS_RATIO + COMPRESS_RATIO) % COMPRESS_RATIO
        from_draft = False
        if HAS_DRAFT:
            from_draft = (
                (expected >= 0)
                & (not from_current)
                & (
                    tl.load(
                        draft_logical_positions
                        + request * stride_dlp_r
                        + slot * stride_dlp_s
                    ).to(tl.int64)
                    == expected
                )
            )
        current_base = token_k + safe_source * stride_k_n
        draft_base = draft_raw_cache + request * stride_drc_r + slot * stride_drc_s
        cached_base = raw_cache + page * stride_raw_p + slot * stride_raw_s
        current_mask = half_mask & from_current
        draft_mask = half_mask & from_draft
        cached_mask = half_mask & (not from_current) & (not from_draft)
        acc_first += tl.load(
            current_base + half_offsets * stride_k_d, mask=current_mask, other=0.0
        ).to(tl.float32)
        acc_first += tl.load(
            draft_base + half_offsets * stride_drc_d, mask=draft_mask, other=0.0
        ).to(tl.float32)
        acc_first += tl.load(
            cached_base + half_offsets * stride_raw_d, mask=cached_mask, other=0.0
        ).to(tl.float32)
        acc_second += tl.load(
            current_base + second_offsets * stride_k_d, mask=current_mask, other=0.0
        ).to(tl.float32)
        acc_second += tl.load(
            draft_base + second_offsets * stride_drc_d, mask=draft_mask, other=0.0
        ).to(tl.float32)
        acc_second += tl.load(
            cached_base + second_offsets * stride_raw_d, mask=cached_mask, other=0.0
        ).to(tl.float32)
        acc_pass += tl.load(
            current_base + dim_offsets * stride_k_d,
            mask=pass_mask & from_current,
            other=0.0,
        ).to(tl.float32)
        acc_pass += tl.load(
            draft_base + dim_offsets * stride_drc_d,
            mask=pass_mask & from_draft,
            other=0.0,
        ).to(tl.float32)
        acc_pass += tl.load(
            cached_base + dim_offsets * stride_raw_d,
            mask=pass_mask & (not from_current) & (not from_draft),
            other=0.0,
        ).to(tl.float32)
    pooled_first = acc_first / COMPRESS_RATIO
    pooled_second = acc_second / COMPRESS_RATIO
    pooled_pass = acc_pass / COMPRESS_RATIO

    # Gemma RMSNorm on the pooled key (weight already holds 1 + gamma).
    squares = (
        tl.sum(pooled_first * pooled_first, axis=0)
        + tl.sum(pooled_second * pooled_second, axis=0)
        + tl.sum(pooled_pass * pooled_pass, axis=0)
    )
    scale = 1.0 / tl.sqrt(squares / head_dim + norm_epsilon)
    weight_first = tl.load(norm_weight + half_offsets, mask=half_mask, other=0.0).to(
        tl.float32
    )
    weight_second = tl.load(norm_weight + second_offsets, mask=half_mask, other=0.0).to(
        tl.float32
    )
    weight_pass = tl.load(norm_weight + dim_offsets, mask=pass_mask, other=0.0).to(
        tl.float32
    )
    norm_first = pooled_first * scale * weight_first
    norm_second = pooled_second * scale * weight_second
    norm_pass = pooled_pass * scale * weight_pass

    # Group-start RoPE positions: current batch row or cached page header.
    head_source = row - (COMPRESS_RATIO - 1)
    head_expected = position - (COMPRESS_RATIO - 1)
    from_current_head = head_source >= 0
    safe_head_source = tl.maximum(head_source, 0)
    if from_current_head:
        from_current_head = (
            tl.load(request_indices + head_source).to(tl.int64) == request
        ) & (tl.load(logical_positions + head_source).to(tl.int64) == head_expected)
    head_slot = (head_expected % COMPRESS_RATIO + COMPRESS_RATIO) % COMPRESS_RATIO
    from_draft_head = False
    if HAS_DRAFT:
        from_draft_head = (
            (head_expected >= 0)
            & (not from_current_head)
            & (
                tl.load(
                    draft_logical_positions
                    + request * stride_dlp_r
                    + head_slot * stride_dlp_s
                ).to(tl.int64)
                == head_expected
            )
        )
    axes = tl.arange(0, 4)
    axis_mask = axes < 3
    current_positions = tl.load(
        position_values + safe_head_source * stride_pv_n + axes * stride_pv_a,
        mask=axis_mask & from_current_head,
        other=0,
    )
    draft_positions = tl.load(
        draft_position_cache + request * stride_dpc_r + axes,
        mask=axis_mask & from_draft_head,
        other=0,
    )
    cached_positions = tl.load(
        position_cache + page * stride_pc_p + axes,
        mask=axis_mask & (not from_current_head) & (not from_draft_head),
        other=0,
    )
    first_positions = current_positions + draft_positions + cached_positions
    p0 = tl.sum(tl.where(axes == 0, first_positions, 0))
    p1 = tl.sum(tl.where(axes == 1, first_positions, 0))
    p2 = tl.sum(tl.where(axes == 2, first_positions, 0))

    # Neox-style rotation with optional multimodal section selection.
    if HAS_SECTIONS:
        if INTERLEAVED:
            axis = tl.where(
                (half_offsets % 3 == 1) & (half_offsets < 3 * section1),
                1,
                tl.where((half_offsets % 3 == 2) & (half_offsets < 3 * section2), 2, 0),
            )
        else:
            axis = tl.where(
                half_offsets < section0,
                0,
                tl.where(half_offsets < section0 + section1, 1, 2),
            )
        selected_positions = tl.where(axis == 0, p0, tl.where(axis == 1, p1, p2))
    else:
        selected_positions = tl.where(half_mask, p0, 0)
    cos = tl.load(
        cos_sin_cache + selected_positions * stride_cs_n + half_offsets,
        mask=half_mask,
        other=0.0,
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_cache + selected_positions * stride_cs_n + half + half_offsets,
        mask=half_mask,
        other=0.0,
    ).to(tl.float32)
    rotated_first = norm_first * cos - norm_second * sin
    rotated_second = norm_second * cos + norm_first * sin

    # Scatter at compression-group boundaries only.
    qsa_loc = tl.load(qsa_locs + row).to(tl.int64)
    boundary = ((position + 1) % compress_ratio == 0) & (qsa_loc > 0) & (recent_loc > 0)
    if HAS_WRITE_MASK:
        boundary &= tl.load(write_mask + row) != 0
    if boundary:
        compressed_page = qsa_loc // compressed_token_page_size
        within = qsa_loc % compressed_token_page_size
        target = (
            compressed_page * compressed_rows_per_page + within // compress_ratio
        ) * stride_cc_row
        out_dtype = compressed_cache.dtype.element_ty
        tl.store(
            compressed_cache + target + half_offsets * stride_cc_d,
            rotated_first.to(out_dtype),
            mask=half_mask,
        )
        tl.store(
            compressed_cache + target + second_offsets * stride_cc_d,
            rotated_second.to(out_dtype),
            mask=half_mask,
        )
        tl.store(
            compressed_cache + target + dim_offsets * stride_cc_d,
            norm_pass.to(out_dtype),
            mask=pass_mask,
        )
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def qwen4_exp_qsa_compress_and_store(
    token_k: torch.Tensor,
    logical_positions: torch.Tensor,
    request_indices: torch.Tensor,
    recent_locs: torch.Tensor,
    raw_cache: torch.Tensor,
    position_values: torch.Tensor,
    position_cache: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_epsilon: float,
    cos_sin_cache: torch.Tensor,
    qsa_locs: torch.Tensor,
    compressed_cache: torch.Tensor,
    recent_page_size: int,
    compress_ratio: int,
    compressed_token_page_size: int,
    *,
    sections: tuple[int, ...] | None = None,
    interleaved: bool = False,
    write_mask: torch.Tensor | None = None,
    draft_raw_cache: torch.Tensor | None = None,
    draft_logical_positions: torch.Tensor | None = None,
    draft_position_cache: torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> None:
    """Pool, normalize, rotate, and scatter compressed QSA keys in one kernel.

    Each row gathers its compression group from current-batch keys, optional
    speculative draft keys, or the committed raw-cache fallback. It averages
    the members, applies Gemma RMSNorm and neox RoPE at the group-start
    position, then scatters the result into the compressed cache when the row
    ends a compression group. The kernel only reads raw-key stores; raw-key
    writes must happen afterwards in a separate launch.

    Args:
        token_k: Raw indexer keys shaped ``[rows, 1, head_dim]``.
        logical_positions: Absolute logical position per row.
        request_indices: Owning request id per row.
        recent_locs: Recent-cache slots per row; non-positive means invalid.
        raw_cache: Raw key cache shaped ``[pages, compress_ratio, 1, head_dim]``.
        position_values: RoPE positions shaped ``[rows, 3]``.
        position_cache: Group-start RoPE positions shaped ``[pages, 3]``.
        norm_weight: Gemma RMSNorm scale shaped ``[head_dim]``, pre-offset
            by one.
        norm_epsilon: RMSNorm variance epsilon.
        cos_sin_cache: Fused cos/sin table shaped
            ``[max_positions, rotary_dim]``.
        qsa_locs: Compressed-cache slots per row; non-positive means invalid.
        compressed_cache: Compressed key cache shaped
            ``[pages, rows_per_page, 1, head_dim]``.
        recent_page_size: Rows covered by one raw-cache page.
        compress_ratio: Tokens grouped into one compressed block.
        compressed_token_page_size: Logical tokens covered by one
            compressed page.
        sections: Optional multimodal section sizes selecting the RoPE
            position axis per rotary half-dimension.
        interleaved: Whether multimodal sections are interleaved by axis.
        write_mask: Optional authoritative per-row compression write mask.
        draft_raw_cache: Optional request-local speculative raw keys shaped
            ``[requests, compress_ratio, 1, head_dim]``.
        draft_logical_positions: Exact logical position tags for
            ``draft_raw_cache``, shaped ``[requests, compress_ratio]``.
        draft_position_cache: Group-start RoPE positions for speculative
            draft groups, shaped ``[requests, 3]``.
        enable_pdl: Allow the follow-up raw-key write kernel to launch early
            on NVIDIA GPUs; the dependent kernel still waits for this grid.

    Returns:
        None.
    """

    rows = logical_positions.shape[0]
    if not rows:
        return
    rotary_dim = cos_sin_cache.shape[-1]
    if rotary_dim % 2:
        raise ValueError("Qwen4-Exp QSA compression needs an even rotary dimension")
    head_dim = token_k.shape[-1]
    if rotary_dim > head_dim:
        raise ValueError("Qwen4-Exp QSA index head is narrower than the rotary dim")
    draft_buffers = (
        draft_raw_cache,
        draft_logical_positions,
        draft_position_cache,
    )
    has_draft = any(value is not None for value in draft_buffers)
    if has_draft and not all(value is not None for value in draft_buffers):
        raise ValueError("Qwen4-Exp QSA draft compression needs all scratch buffers")
    if has_draft:
        if (
            draft_raw_cache.ndim != 4
            or draft_raw_cache.shape[1:] != (compress_ratio, 1, head_dim)
            or draft_logical_positions.shape != draft_raw_cache.shape[:2]
            or draft_position_cache.shape != (draft_raw_cache.shape[0], 3)
        ):
            raise ValueError("Qwen4-Exp QSA draft scratch buffers have invalid shapes")
    else:
        draft_raw_cache = raw_cache
        draft_logical_positions = logical_positions
        draft_position_cache = position_cache
    if write_mask is None:
        write_mask = recent_locs
    # ``token_k`` is usually a strided view of the QK projection output; the
    # kernel reads it through its strides directly, and index tensors are
    # cast in-kernel, so no host-side copy/cast kernel is launched.
    use_pdl = _is_nvidia and enable_pdl
    _qwen4_exp_qsa_compress_and_store_kernel[(rows,)](
        token_k,
        logical_positions,
        request_indices,
        recent_locs,
        raw_cache,
        position_values,
        position_cache,
        norm_weight,
        cos_sin_cache,
        qsa_locs,
        compressed_cache,
        write_mask,
        draft_raw_cache,
        draft_logical_positions,
        draft_position_cache,
        head_dim,
        rotary_dim,
        float(norm_epsilon),
        recent_page_size,
        compress_ratio,
        compressed_token_page_size,
        compressed_cache.shape[1],
        0 if sections is None else int(sections[0]),
        0 if sections is None else int(sections[1]),
        0 if sections is None else int(sections[2]),
        token_k.stride(0),
        token_k.stride(-1),
        raw_cache.stride(0),
        raw_cache.stride(1),
        raw_cache.stride(-1),
        position_values.stride(0),
        position_values.stride(1),
        position_cache.stride(0),
        cos_sin_cache.stride(0),
        compressed_cache.stride(1),
        compressed_cache.stride(-1),
        draft_raw_cache.stride(0),
        draft_raw_cache.stride(1),
        draft_raw_cache.stride(-1),
        draft_logical_positions.stride(0),
        draft_logical_positions.stride(1) if has_draft else 0,
        draft_position_cache.stride(0),
        COMPRESS_RATIO=compress_ratio,
        HAS_WRITE_MASK=write_mask is not recent_locs,
        HAS_DRAFT=has_draft,
        HAS_SECTIONS=sections is not None,
        INTERLEAVED=interleaved,
        BLOCK_HALF=triton.next_power_of_2(max(rotary_dim // 2, 1)),
        BLOCK_D=triton.next_power_of_2(head_dim),
        ENABLE_PDL=use_pdl,
        **({"launch_pdl": True} if use_pdl else {}),
    )


@triton.jit
def _qwen4_exp_qsa_recent_write_kernel(
    token_k,
    logical_positions,
    request_indices,
    recent_locs,
    position_values,
    raw_cache,
    position_cache,
    write_mask,
    rows,
    head_dim,
    recent_page_size,
    request_limit,
    stride_k_n,
    stride_k_d,
    stride_raw_p,
    stride_raw_s,
    stride_raw_d,
    stride_pv_n,
    stride_pv_a,
    stride_pc_p,
    COMPRESS_RATIO: tl.constexpr,
    HAS_EXTRA_MASK: tl.constexpr,
    HAS_LIMIT: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    row = tl.program_id(0)
    loc = tl.load(recent_locs + row).to(tl.int64)
    if HAS_EXTRA_MASK:
        write = tl.load(write_mask + row) != 0
    else:
        write = loc > 0
    if write:
        # Keep only the last writer of every ring slot. The shadow check is
        # valid whenever rows are grouped by request in logical order; in
        # other shapes no later row can shadow the current one.
        future = row + COMPRESS_RATIO
        if future < rows:
            if HAS_EXTRA_MASK:
                has_future = tl.load(write_mask + future) != 0
            else:
                has_future = tl.load(recent_locs + future).to(tl.int64) > 0
            if has_future:
                has_future &= tl.load(request_indices + future).to(tl.int64) == tl.load(
                    request_indices + row
                ).to(tl.int64)
            if has_future:
                has_future &= (
                    tl.load(logical_positions + future).to(tl.int64)
                    == tl.load(logical_positions + row).to(tl.int64) + COMPRESS_RATIO
                )
            write &= not has_future
    if write and HAS_LIMIT:
        write &= tl.load(request_indices + row).to(tl.int64) < request_limit
    if write:
        position = tl.load(logical_positions + row).to(tl.int64)
        slot = (position % COMPRESS_RATIO + COMPRESS_RATIO) % COMPRESS_RATIO
        page = loc // recent_page_size
        dim_offsets = tl.arange(0, BLOCK_D)
        dim_mask = dim_offsets < head_dim
        values = tl.load(
            token_k + row * stride_k_n + dim_offsets * stride_k_d,
            mask=dim_mask,
            other=0.0,
        )
        if ENABLE_PDL:
            # Only the raw-cache and position-header writes must observe the
            # compression kernel's reads of the old ring slots; everything
            # above already overlapped that kernel's tail.
            tl.extra.cuda.gdc_wait()
        tl.store(
            raw_cache
            + page * stride_raw_p
            + slot * stride_raw_s
            + dim_offsets * stride_raw_d,
            values.to(raw_cache.dtype.element_ty),
            mask=dim_mask,
        )
        if slot == 0:
            axes = tl.arange(0, 4)
            axis_mask = axes < 3
            rope_positions = tl.load(
                position_values + row * stride_pv_n + axes * stride_pv_a,
                mask=axis_mask,
                other=0,
            )
            tl.store(
                position_cache + page * stride_pc_p + axes,
                rope_positions,
                mask=axis_mask,
            )


def qwen4_exp_qsa_recent_write(
    token_k: torch.Tensor,
    logical_positions: torch.Tensor,
    request_indices: torch.Tensor,
    recent_locs: torch.Tensor,
    position_values: torch.Tensor,
    raw_cache: torch.Tensor,
    position_cache: torch.Tensor,
    recent_page_size: int,
    compress_ratio: int,
    *,
    write_mask: torch.Tensor | None = None,
    request_limit: int | None = None,
    enable_pdl: bool = False,
) -> None:
    """Write the latest raw keys and group-start RoPE positions.

    Without ``write_mask`` the kernel keeps only the last writer of every
    ring slot and can additionally skip rows whose request id is not smaller
    than ``request_limit``. With ``write_mask`` the boolean mask selects the
    candidate rows; later masked writers of the same ring slot still shadow
    earlier ones so every physical slot receives exactly one deterministic
    write.

    Args:
        token_k: Raw indexer keys shaped ``[rows, 1, head_dim]``.
        logical_positions: Absolute logical position per row.
        request_indices: Owning request id per row.
        recent_locs: Recent-cache slots per row; non-positive means invalid.
        position_values: RoPE positions shaped ``[rows, 3]``.
        raw_cache: Raw key cache shaped ``[pages, compress_ratio, 1, head_dim]``.
        position_cache: Group-start RoPE positions shaped ``[pages, 3]``.
        recent_page_size: Rows covered by one raw-cache page.
        compress_ratio: Tokens grouped into one compressed block.
        write_mask: Optional authoritative per-row write mask.
        request_limit: Optional exclusive request-id write bound.
        enable_pdl: Wait on the compression kernel via programmatic dependent
            launch on NVIDIA GPUs, hiding the launch gap between the two.

    Returns:
        None.
    """

    rows = logical_positions.shape[0]
    if not rows:
        return
    # Strided ``token_k`` reads and in-kernel index casts avoid host-side
    # copy/cast kernels before the launch.
    head_dim = token_k.shape[-1]
    if write_mask is None:
        write_mask = recent_locs
    use_pdl = _is_nvidia and enable_pdl
    _qwen4_exp_qsa_recent_write_kernel[(rows,)](
        token_k,
        logical_positions,
        request_indices,
        recent_locs,
        position_values,
        raw_cache,
        position_cache,
        write_mask,
        rows,
        head_dim,
        recent_page_size,
        -1 if request_limit is None else request_limit,
        token_k.stride(0),
        token_k.stride(-1),
        raw_cache.stride(0),
        raw_cache.stride(1),
        raw_cache.stride(-1),
        position_values.stride(0),
        position_values.stride(1),
        position_cache.stride(0),
        COMPRESS_RATIO=compress_ratio,
        HAS_EXTRA_MASK=write_mask is not recent_locs,
        HAS_LIMIT=request_limit is not None,
        BLOCK_D=triton.next_power_of_2(head_dim),
        ENABLE_PDL=use_pdl,
        **({"launch_pdl": True} if use_pdl else {}),
    )


@triton.jit
def _qwen4_exp_qsa_stage_draft_kernel(
    token_k,
    position_values,
    logical_positions,
    request_indices,
    recent_locs,
    staged_k,
    staged_positions,
    staged_logical,
    head_dim,
    stride_k_n,
    stride_k_d,
    stride_pv_n,
    stride_pv_a,
    stride_sk_r,
    stride_sk_s,
    stride_sk_d,
    stride_sp_r,
    stride_sl_r,
    stride_sl_s,
    COMPRESS_RATIO: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    row = tl.program_id(0)
    recent_loc = tl.load(recent_locs + row).to(tl.int64)
    if recent_loc > 0:
        position = tl.load(logical_positions + row).to(tl.int64)
        request = tl.load(request_indices + row).to(tl.int64)
        slot = (position % COMPRESS_RATIO + COMPRESS_RATIO) % COMPRESS_RATIO
        dim_offsets = tl.arange(0, BLOCK_D)
        dim_mask = dim_offsets < head_dim
        values = tl.load(
            token_k + row * stride_k_n + dim_offsets * stride_k_d,
            mask=dim_mask,
            other=0.0,
        )
        axes = tl.arange(0, 4)
        axis_mask = axes < 3
        rope_positions = tl.load(
            position_values + row * stride_pv_n + axes * stride_pv_a,
            mask=axis_mask,
            other=0,
        )
        if ENABLE_PDL:
            # Compression may still be reading older values from this ring.
            tl.extra.cuda.gdc_wait()
        tl.store(
            staged_k
            + request * stride_sk_r
            + slot * stride_sk_s
            + dim_offsets * stride_sk_d,
            values.to(staged_k.dtype.element_ty),
            mask=dim_mask,
        )
        tl.store(
            staged_logical + request * stride_sl_r + slot * stride_sl_s,
            position,
        )
        if slot == 0:
            tl.store(
                staged_positions + request * stride_sp_r + axes,
                rope_positions,
                mask=axis_mask,
            )


def qwen4_exp_qsa_stage_draft(
    token_k: torch.Tensor,
    position_values: torch.Tensor,
    logical_positions: torch.Tensor,
    request_indices: torch.Tensor,
    recent_locs: torch.Tensor,
    staged_k: torch.Tensor,
    staged_positions: torch.Tensor,
    staged_logical: torch.Tensor,
    compress_ratio: int,
    *,
    enable_pdl: bool = False,
) -> None:
    """Stage speculative draft raw keys without mutating committed QSA state.

    The caller must provide at most one row per request. Exact logical-position
    tags let the compression kernel distinguish live speculative ring entries
    from slots left by an earlier MTP round.

    Args:
        token_k: Raw indexer keys shaped ``[rows, 1, head_dim]``.
        position_values: RoPE positions shaped ``[rows, 3]``.
        logical_positions: Absolute logical position per row.
        request_indices: Owning request id per row.
        recent_locs: Recent-cache slots per row; non-positive means invalid.
        staged_k: Request-local raw-key ring shaped
            ``[requests, compress_ratio, 1, head_dim]``.
        staged_positions: Request-local group-start RoPE positions shaped
            ``[requests, 3]``.
        staged_logical: Exact position tags shaped
            ``[requests, compress_ratio]``.
        compress_ratio: Tokens grouped into one compressed block.
        enable_pdl: Wait on a preceding compression kernel through
            programmatic dependent launch on NVIDIA GPUs.

    Returns:
        None.
    """

    rows = logical_positions.shape[0]
    if not rows:
        return
    head_dim = token_k.shape[-1]
    if (
        token_k.shape != (rows, 1, head_dim)
        or position_values.shape != (rows, 3)
        or request_indices.shape != (rows,)
        or recent_locs.shape != (rows,)
        or staged_k.ndim != 4
        or staged_k.shape[1:] != (compress_ratio, 1, head_dim)
        or staged_positions.shape != (staged_k.shape[0], 3)
        or staged_logical.shape != staged_k.shape[:2]
    ):
        raise ValueError("Qwen4-Exp QSA draft staging buffers have invalid shapes")
    use_pdl = _is_nvidia and enable_pdl
    _qwen4_exp_qsa_stage_draft_kernel[(rows,)](
        token_k,
        position_values,
        logical_positions,
        request_indices,
        recent_locs,
        staged_k,
        staged_positions,
        staged_logical,
        head_dim,
        token_k.stride(0),
        token_k.stride(-1),
        position_values.stride(0),
        position_values.stride(1),
        staged_k.stride(0),
        staged_k.stride(1),
        staged_k.stride(-1),
        staged_positions.stride(0),
        staged_logical.stride(0),
        staged_logical.stride(1),
        COMPRESS_RATIO=compress_ratio,
        BLOCK_D=triton.next_power_of_2(head_dim),
        ENABLE_PDL=use_pdl,
        **({"launch_pdl": True} if use_pdl else {}),
    )


@triton.jit
def _qwen4_exp_qsa_stage_verify_kernel(
    token_k,
    position_values,
    logical_positions,
    recent_locs,
    staged_k,
    staged_positions,
    staged_logical,
    staged_recent,
    head_dim,
    stride_k_n,
    stride_k_d,
    stride_pv_n,
    stride_pv_a,
    stride_lp,
    stride_rl,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    dim_offsets = tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < head_dim
    values = tl.load(
        token_k + row * stride_k_n + dim_offsets * stride_k_d,
        mask=dim_mask,
        other=0.0,
    )
    tl.store(staged_k + row * head_dim + dim_offsets, values, mask=dim_mask)
    axes = tl.arange(0, 4)
    axis_mask = axes < 3
    rope_positions = tl.load(
        position_values + row * stride_pv_n + axes * stride_pv_a,
        mask=axis_mask,
        other=0,
    )
    tl.store(staged_positions + row * 3 + axes, rope_positions, mask=axis_mask)
    tl.store(staged_logical + row, tl.load(logical_positions + row * stride_lp))
    tl.store(staged_recent + row, tl.load(recent_locs + row * stride_rl))


def qwen4_exp_qsa_stage_verify(
    token_k: torch.Tensor,
    position_values: torch.Tensor,
    logical_positions: torch.Tensor,
    recent_locs: torch.Tensor,
    staged_k: torch.Tensor,
    staged_positions: torch.Tensor,
    staged_logical: torch.Tensor,
    staged_recent: torch.Tensor,
) -> None:
    """Snapshot target-verify staging buffers with one fused copy kernel.

    One launch replaces the four ``copy_`` kernels that stage raw indexer
    keys, RoPE positions, logical positions, and recent-cache slots while
    the acceptance widths are still unknown. Sources may be strided views
    (the transposed ``[rows, 3]`` mrope layout included); every destination
    must be contiguous and share its source's dtype.

    Args:
        token_k: Raw indexer keys shaped ``[rows, 1, head_dim]``.
        position_values: RoPE positions shaped ``[rows, 3]``, any strides.
        logical_positions: Absolute logical position per row.
        recent_locs: Recent-cache slot per row.
        staged_k: Contiguous destination for ``token_k``.
        staged_positions: Contiguous destination for ``position_values``.
        staged_logical: Contiguous destination for ``logical_positions``.
        staged_recent: Contiguous destination for ``recent_locs``.

    Returns:
        None.
    """

    rows = logical_positions.shape[0]
    if not rows:
        return
    if token_k.ndim != 3 or token_k.shape[1] != 1:
        raise ValueError(
            "Qwen4-Exp QSA staging expects token_k shaped [rows, 1, head_dim]"
        )
    if (
        token_k.shape[0] != rows
        or position_values.shape[0] != rows
        or position_values.shape[1] != 3
        or recent_locs.shape[0] != rows
    ):
        raise ValueError(
            "Qwen4-Exp QSA staging sources need one shared leading row "
            "count and three position axes"
        )
    for source, staged in (
        (token_k, staged_k),
        (position_values, staged_positions),
        (logical_positions, staged_logical),
        (recent_locs, staged_recent),
    ):
        if source.dtype != staged.dtype or source.numel() != staged.numel():
            raise ValueError(
                "Qwen4-Exp QSA staging needs matching dtype and size "
                "between every source and destination"
            )
        if not staged.is_contiguous():
            raise ValueError("Qwen4-Exp QSA staging destinations must be contiguous")
    head_dim = token_k.shape[-1]
    _qwen4_exp_qsa_stage_verify_kernel[(rows,)](
        token_k,
        position_values,
        logical_positions,
        recent_locs,
        staged_k,
        staged_positions,
        staged_logical,
        staged_recent,
        head_dim,
        token_k.stride(0),
        token_k.stride(-1),
        position_values.stride(0),
        position_values.stride(1),
        logical_positions.stride(0),
        recent_locs.stride(0),
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
    )


@triton.jit
def _qwen4_exp_qsa_norm_rope_kernel(
    inputs,
    positions,
    norm_weight,
    cos_sin_cache,
    outputs,
    num_heads,
    head_dim,
    rotary_dim,
    norm_epsilon,
    section0,
    section1,
    section2,
    stride_in_n,
    stride_in_d,
    stride_pos_axis,
    stride_pos_row,
    stride_cs_n,
    stride_out_n,
    stride_out_h,
    POS_3D: tl.constexpr,
    HAS_SECTIONS: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    half = rotary_dim // 2
    half_offsets = tl.arange(0, BLOCK_HALF)
    half_mask = half_offsets < half
    dim_offsets = tl.arange(0, BLOCK_D)
    pass_mask = (dim_offsets >= rotary_dim) & (dim_offsets < head_dim)
    p0 = tl.load(positions + row * stride_pos_row).to(tl.int64)
    if POS_3D:
        p1 = tl.load(positions + stride_pos_axis + row * stride_pos_row).to(tl.int64)
        p2 = tl.load(positions + 2 * stride_pos_axis + row * stride_pos_row).to(
            tl.int64
        )
    else:
        p1 = p0
        p2 = p0
    # RoPE axis selection per rotary half-dimension; the cos/sin vectors are
    # token-level and shared by every head of this row.
    if HAS_SECTIONS:
        if INTERLEAVED:
            axis = tl.where(
                (half_offsets % 3 == 1) & (half_offsets < 3 * section1),
                1,
                tl.where((half_offsets % 3 == 2) & (half_offsets < 3 * section2), 2, 0),
            )
        else:
            axis = tl.where(
                half_offsets < section0,
                0,
                tl.where(half_offsets < section0 + section1, 1, 2),
            )
        selected_positions = tl.where(axis == 0, p0, tl.where(axis == 1, p1, p2))
    else:
        selected_positions = tl.where(half_mask, p0, 0)
    cos = tl.load(
        cos_sin_cache + selected_positions * stride_cs_n + half_offsets,
        mask=half_mask,
        other=0.0,
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_cache + selected_positions * stride_cs_n + half + half_offsets,
        mask=half_mask,
        other=0.0,
    ).to(tl.float32)
    weight_first = tl.load(norm_weight + half_offsets, mask=half_mask, other=0.0).to(
        tl.float32
    )
    weight_second = tl.load(
        norm_weight + half + half_offsets, mask=half_mask, other=0.0
    ).to(tl.float32)
    weight_pass = tl.load(norm_weight + dim_offsets, mask=pass_mask, other=0.0).to(
        tl.float32
    )
    for head in range(0, num_heads):
        input_head = inputs + row * stride_in_n + head * head_dim * stride_in_d
        first = tl.load(
            input_head + half_offsets * stride_in_d, mask=half_mask, other=0.0
        )
        second = tl.load(
            input_head + (half + half_offsets) * stride_in_d, mask=half_mask, other=0.0
        )
        passthrough = tl.load(
            input_head + dim_offsets * stride_in_d, mask=pass_mask, other=0.0
        )
        # Gemma RMSNorm per head row (weight already holds 1 + gamma).
        squares = (
            tl.sum(first.to(tl.float32) * first.to(tl.float32), axis=0)
            + tl.sum(second.to(tl.float32) * second.to(tl.float32), axis=0)
            + tl.sum(passthrough.to(tl.float32) * passthrough.to(tl.float32), axis=0)
        )
        scale = 1.0 / tl.sqrt(squares / head_dim + norm_epsilon)
        norm_first = first.to(tl.float32) * scale * weight_first
        norm_second = second.to(tl.float32) * scale * weight_second
        norm_pass = passthrough.to(tl.float32) * scale * weight_pass
        rotated_first = norm_first * cos - norm_second * sin
        rotated_second = norm_second * cos + norm_first * sin
        output_head = outputs + row * stride_out_n + head * stride_out_h
        out_dtype = outputs.dtype.element_ty
        tl.store(
            output_head + half_offsets, rotated_first.to(out_dtype), mask=half_mask
        )
        tl.store(
            output_head + half + half_offsets,
            rotated_second.to(out_dtype),
            mask=half_mask,
        )
        tl.store(output_head + dim_offsets, norm_pass.to(out_dtype), mask=pass_mask)


def qwen4_exp_qsa_norm_rope(
    inputs: torch.Tensor,
    positions: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_epsilon: float,
    cos_sin_cache: torch.Tensor,
    *,
    num_heads: int,
    sections: tuple[int, ...] | None = None,
    interleaved: bool = False,
) -> torch.Tensor:
    """Apply per-head Gemma RMSNorm and neox RoPE in one fused kernel.

    Args:
        inputs: Projected queries shaped ``[tokens, num_heads * head_dim]``;
            strided views (for example a split of one QK projection output)
            are read in place.
        positions: Token positions shaped ``[tokens]`` or ``[3, tokens]`` for
            multimodal section selection; any integer dtype and layout is
            read in place.
        norm_weight: Gemma RMSNorm scale shaped ``[head_dim]``, pre-offset
            by one.
        norm_epsilon: RMSNorm variance epsilon.
        cos_sin_cache: Fused cos/sin table shaped
            ``[max_positions, rotary_dim]``.
        num_heads: Query heads packed into each input row.
        sections: Optional multimodal section sizes selecting the position
            axis per rotary half-dimension.
        interleaved: Whether multimodal sections are interleaved by axis.

    Returns:
        Normalized and rotated queries shaped
        ``[tokens, num_heads, head_dim]`` in the input dtype.
    """

    if inputs.ndim != 2:
        raise ValueError("Qwen4-Exp QSA norm-rope expects flattened head rows")
    rotary_dim = cos_sin_cache.shape[-1]
    if rotary_dim % 2:
        raise ValueError("Qwen4-Exp QSA norm-rope needs an even rotary dimension")
    head_dim = norm_weight.shape[0]
    if rotary_dim > head_dim:
        raise ValueError("Qwen4-Exp QSA index head is narrower than the rotary dim")
    tokens, width = inputs.shape
    if width != num_heads * head_dim:
        raise ValueError("Qwen4-Exp QSA norm-rope row width must match the heads")
    outputs = torch.empty(
        (tokens, num_heads, head_dim), dtype=inputs.dtype, device=inputs.device
    )
    if not tokens:
        return outputs
    # Positions are read through their strides with in-kernel casts; no
    # host-side cast/copy kernel is launched.
    if positions.ndim == 2:
        stride_pos_axis = positions.stride(0)
    else:
        stride_pos_axis = 0
    stride_pos_row = positions.stride(-1)
    half = rotary_dim // 2
    _qwen4_exp_qsa_norm_rope_kernel[(tokens,)](
        inputs,
        positions,
        norm_weight,
        cos_sin_cache,
        outputs,
        num_heads,
        head_dim,
        rotary_dim,
        float(norm_epsilon),
        0 if sections is None else int(sections[0]),
        0 if sections is None else int(sections[1]),
        0 if sections is None else int(sections[2]),
        inputs.stride(0),
        inputs.stride(-1),
        stride_pos_axis,
        stride_pos_row,
        cos_sin_cache.stride(0),
        outputs.stride(0),
        outputs.stride(1),
        POS_3D=positions.ndim == 2,
        HAS_SECTIONS=sections is not None,
        INTERLEAVED=interleaved,
        BLOCK_HALF=triton.next_power_of_2(max(half, 1)),
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
    )
    return outputs


@triton.jit
def _qwen4_exp_qsa_pad_keys_to_topk(
    keys, BLOCK_TOPK: tl.constexpr, BLOCK_N: tl.constexpr
):
    """Zero-pad a packed-key tile up to the top-k width via constexpr joins."""

    if BLOCK_TOPK >= 2 * BLOCK_N:
        keys = tl.reshape(
            tl.join(keys, tl.full((BLOCK_N,), 0, tl.int64)), (2 * BLOCK_N,)
        )
    if BLOCK_TOPK >= 4 * BLOCK_N:
        keys = tl.reshape(
            tl.join(keys, tl.full((2 * BLOCK_N,), 0, tl.int64)), (4 * BLOCK_N,)
        )
    if BLOCK_TOPK >= 8 * BLOCK_N:
        keys = tl.reshape(
            tl.join(keys, tl.full((4 * BLOCK_N,), 0, tl.int64)), (8 * BLOCK_N,)
        )
    if BLOCK_TOPK >= 16 * BLOCK_N:
        keys = tl.reshape(
            tl.join(keys, tl.full((8 * BLOCK_N,), 0, tl.int64)), (16 * BLOCK_N,)
        )
    if BLOCK_TOPK >= 32 * BLOCK_N:
        keys = tl.reshape(
            tl.join(keys, tl.full((16 * BLOCK_N,), 0, tl.int64)), (32 * BLOCK_N,)
        )
    return keys


@triton.jit
def _qwen4_exp_qsa_stream_block_topk_kernel(
    query,
    key_cache,
    page_table,
    request_indices,
    complete_blocks,
    partial_keys,
    num_heads,
    head_dim,
    num_blocks,
    page_size,
    page_expansion,
    blocks_per_split,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_k_n,
    stride_k_d,
    stride_pt_b,
    stride_pk_n,
    stride_pk_s,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    row = tl.program_id(0)
    split = tl.program_id(1)
    block_start = split * blocks_per_split
    complete = tl.load(complete_blocks + row).to(tl.int64)
    block_end = tl.minimum(block_start + blocks_per_split, num_blocks)
    block_end = tl.minimum(block_end, complete)
    if block_start >= block_end:
        if ENABLE_PDL:
            tl.extra.cuda.gdc_launch_dependents()
        return
    request = tl.load(request_indices + row).to(tl.int64)
    head_offsets = tl.arange(0, BLOCK_H)
    dim_offsets = tl.arange(0, BLOCK_D)
    head_mask = head_offsets < num_heads
    dim_mask = dim_offsets < head_dim
    q = tl.load(
        query
        + row * stride_q_n
        + head_offsets[:, None] * stride_q_h
        + dim_offsets[None, :] * stride_q_d,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    acc = tl.zeros((BLOCK_TOPK,), dtype=tl.int64)
    for tile_start in range(block_start, block_end, BLOCK_N):
        block_ids = tile_start + tl.arange(0, BLOCK_N)
        tile_mask = block_ids < block_end
        columns = block_ids // page_size
        offsets = block_ids % page_size
        # Page tables live at consumer granularity; entry ``col * expansion``
        # maps back to one logical compressed page.
        pages = tl.load(
            page_table + request * stride_pt_b + columns * page_expansion,
            mask=tile_mask,
            other=0,
        ).to(tl.int64)
        pages = pages // page_expansion
        slots = pages * page_size + offsets
        keys = tl.load(
            key_cache + slots[None, :] * stride_k_n + dim_offsets[:, None] * stride_k_d,
            mask=tile_mask[None, :] & dim_mask[:, None],
            other=0.0,
        )
        scores = tl.dot(q, keys, out_dtype=tl.float32)
        scores = tl.maximum(scores, 0.0)
        scores = tl.sum(tl.where(head_mask[:, None], scores, 0.0), axis=0)
        # Pack (score, block id) into one monotonic int64 key: shifted
        # score bits in the high word, block id in the low word, zero
        # reserved as the invalid sentinel.
        score_bits = scores.to(tl.int32, bitcast=True)
        packed = ((score_bits.to(tl.int64) + 1) << 32) | block_ids.to(tl.int64)
        packed = tl.where(tile_mask, packed, 0)
        # Streaming merge in the style of the DSA top-k kernel: rotate
        # the running set into a bitonic layout every tile, then fold in
        # the tile's sorted top-k with an elementwise maximum instead of
        # re-selecting over a joined 2k-wide buffer. The prune still
        # skips the tile sort when no packed key can enter the set.
        acc = tl.bitonic_merge(acc)
        if tl.max(packed, axis=0) > tl.min(acc, axis=0):
            padded = _qwen4_exp_qsa_pad_keys_to_topk(packed, BLOCK_TOPK, BLOCK_N)
            acc = tl.maximum(acc, tl.sort(padded, descending=True))
    topk_offsets = tl.arange(0, BLOCK_TOPK)
    tl.store(partial_keys + row * stride_pk_n + split * stride_pk_s + topk_offsets, acc)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _qwen4_exp_qsa_merge_pairs(cur, ROWS: tl.constexpr, BLOCK_TOPK: tl.constexpr):
    """Merge adjacent bitonic rows and retain each pair's largest K keys."""

    # Every partial row is bitonic: the streaming producer stores
    # maximum(ascending, descending), while a previous tree level stores the
    # same layout. Negating odd rows lets one batched ascending bitonic merge
    # arrange every pair as (ascending, descending); the pairwise maximum is
    # then exactly the largest K keys from their union and remains bitonic.
    odd = (tl.arange(0, ROWS) % 2 == 1)[:, None]
    ordered = tl.where(odd, -cur, cur)
    ordered = tl.bitonic_merge(ordered)
    ordered = tl.where(odd, -ordered, ordered)
    return tl.max(tl.reshape(ordered, (ROWS // 2, 2, BLOCK_TOPK)), axis=1)


@triton.jit
def _qwen4_exp_qsa_merge_tree(cur, ROWS: tl.constexpr, BLOCK_TOPK: tl.constexpr):
    """Reduce power-of-two bitonic rows to one descending top-k row."""

    if ROWS >= 256:
        cur = _qwen4_exp_qsa_merge_pairs(cur, 256, BLOCK_TOPK)
    if ROWS >= 128:
        cur = _qwen4_exp_qsa_merge_pairs(cur, 128, BLOCK_TOPK)
    if ROWS >= 64:
        cur = _qwen4_exp_qsa_merge_pairs(cur, 64, BLOCK_TOPK)
    if ROWS >= 32:
        cur = _qwen4_exp_qsa_merge_pairs(cur, 32, BLOCK_TOPK)
    if ROWS >= 16:
        cur = _qwen4_exp_qsa_merge_pairs(cur, 16, BLOCK_TOPK)
    if ROWS >= 8:
        cur = _qwen4_exp_qsa_merge_pairs(cur, 8, BLOCK_TOPK)
    if ROWS >= 4:
        cur = _qwen4_exp_qsa_merge_pairs(cur, 4, BLOCK_TOPK)
    if ROWS >= 2:
        cur = _qwen4_exp_qsa_merge_pairs(cur, 2, BLOCK_TOPK)
    return tl.reshape(tl.bitonic_merge(cur, descending=True), (BLOCK_TOPK,))


@triton.jit
def _qwen4_exp_qsa_merge_block_topk_kernel(
    partial_keys,
    selected_blocks,
    complete_blocks,
    blocks_per_split,
    stride_pk_n,
    stride_pk_s,
    stride_o_n,
    BLOCK_TOPK: tl.constexpr,
    SPLITS: tl.constexpr,
    POW2_SPLITS: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    row = tl.program_id(0)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    complete = tl.load(complete_blocks + row).to(tl.int64)
    split_ids = tl.arange(0, POW2_SPLITS)[:, None]
    entries = tl.arange(0, BLOCK_TOPK)[None, :]
    # Mask only on SPLITS so the partials load issues in parallel with the
    # complete scalar load; splits past the complete frontier were skipped by
    # the streaming kernel and carry garbage, so they are zeroed afterwards
    # instead of being masked off before the load.
    partials = tl.load(
        partial_keys + row * stride_pk_n + split_ids * stride_pk_s + entries,
        mask=split_ids < SPLITS,
        other=0,
    )
    valid_splits = (split_ids < SPLITS) & (split_ids * blocks_per_split < complete)
    partials = tl.where(valid_splits, partials, 0)
    acc = _qwen4_exp_qsa_merge_tree(partials, POW2_SPLITS, BLOCK_TOPK)
    block_ids = (acc & 0xFFFFFFFF).to(tl.int32)
    block_ids = tl.where(acc != 0, block_ids, -1)
    tl.store(selected_blocks + row * stride_o_n + tl.arange(0, BLOCK_TOPK), block_ids)


@triton.jit
def _qwen4_exp_qsa_merge_chunk_kernel(
    partial_keys,
    merged_keys,
    complete_blocks,
    blocks_per_split,
    stride_pk_n,
    stride_pk_s,
    stride_mk_n,
    BLOCK_TOPK: tl.constexpr,
    SPLITS: tl.constexpr,
    CHUNK: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    complete = tl.load(complete_blocks + row).to(tl.int64)
    split_ids = chunk * CHUNK + tl.arange(0, CHUNK)[:, None]
    entries = tl.arange(0, BLOCK_TOPK)[None, :]
    # Same late-zeroing contract as the single-level merge, and no early-out
    # for fully invalid chunks: keeping every chunk CTA on the same path
    # avoids an uneven grid, which measurably delays the dependent launch in
    # latency-bound single-request cases.
    partials = tl.load(
        partial_keys + row * stride_pk_n + split_ids * stride_pk_s + entries,
        mask=split_ids < SPLITS,
        other=0,
    )
    valid_splits = (split_ids < SPLITS) & (split_ids * blocks_per_split < complete)
    partials = tl.where(valid_splits, partials, 0)
    acc = _qwen4_exp_qsa_merge_tree(partials, CHUNK, BLOCK_TOPK)
    tl.store(
        merged_keys + row * stride_mk_n + chunk * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK),
        acc,
    )
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _qwen4_exp_qsa_score_blocks_kernel(
    query,
    key_cache,
    page_table,
    request_indices,
    complete_blocks,
    logits,
    num_heads,
    head_dim,
    num_blocks,
    page_size,
    page_expansion,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_k_n,
    stride_k_d,
    stride_pt_b,
    stride_l_n,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    request = tl.load(request_indices + row).to(tl.int64)
    complete = tl.load(complete_blocks + row).to(tl.int64)
    head_offsets = tl.arange(0, BLOCK_H)
    dim_offsets = tl.arange(0, BLOCK_D)
    head_mask = head_offsets < num_heads
    dim_mask = dim_offsets < head_dim
    q = tl.load(
        query
        + row * stride_q_n
        + head_offsets[:, None] * stride_q_h
        + dim_offsets[None, :] * stride_q_d,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    block_ids = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    tile_mask = block_ids < num_blocks
    columns = block_ids // page_size
    offsets = block_ids % page_size
    # Page tables live at consumer granularity; entry ``col * expansion``
    # maps back to one logical compressed page.
    pages = tl.load(
        page_table + request * stride_pt_b + columns * page_expansion,
        mask=tile_mask,
        other=0,
    ).to(tl.int64)
    pages = pages // page_expansion
    slots = pages * page_size + offsets
    valid = tile_mask & (block_ids < complete)
    keys = tl.load(
        key_cache + slots[None, :] * stride_k_n + dim_offsets[:, None] * stride_k_d,
        mask=valid[None, :] & dim_mask[:, None],
        other=0.0,
    )
    scores = tl.dot(q, keys, out_dtype=tl.float32)
    scores = tl.maximum(scores, 0.0)
    scores = tl.sum(tl.where(head_mask[:, None], scores, 0.0), axis=0)
    # Invalid blocks become -inf so the downstream selection drops them;
    # the PDL trigger lets selection launch while the tail tiles drain.
    scores = tl.where(valid, scores, -float("inf"))
    tl.store(logits + row * stride_l_n + block_ids, scores, mask=tile_mask)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def _qwen4_exp_qsa_block_topk_stream(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    page_table: torch.Tensor,
    request_indices: torch.Tensor,
    complete_blocks: torch.Tensor,
    *,
    page_size: int,
    block_topk: int,
    page_expansion: int,
    max_partial_bytes: int,
    enable_pdl: bool = True,
) -> torch.Tensor:
    """Fused streaming path: score tiles and fold them into a running top-k.

    Streams each row's page-table blocks tile by tile, scores them against
    the row's query, and maintains a packed-key running top-k, so neither
    the expanded slot matrix nor the full score matrix is materialized.
    Block ranges are split across programs to keep small-row batches busy;
    the partial top-k buffers are bounded by ``max_partial_bytes``.
    """

    rows = query.shape[0]
    num_blocks = triton.cdiv(page_table.shape[1], page_expansion) * page_size
    if rows == 0 or num_blocks == 0:
        return torch.full(
            (rows, block_topk), -1, dtype=torch.int32, device=query.device
        )
    # The merge kernel unconditionally rewrites every row, including the
    # ``-1`` sentinels for empty slots, so skip the fill and leave the
    # buffer uninitialized.
    selected = torch.empty((rows, block_topk), dtype=torch.int32, device=query.device)
    # Wide tiles amortize the running top-k merge for large block_topk;
    # small topk keeps the original 64-block tiles.
    block_n = min(512, block_topk)
    splits = max(1, max_partial_bytes // max(rows * block_topk * 8, 1))
    # Target roughly 256 streaming programs so single-row decode batches
    # still fill the GPU while large batches do not oversubscribe it; the
    # partial buffers stay within ``max_partial_bytes``.
    split_cap = min(256, max(8, 256 // rows))
    # The merge selects at most ``merge_slots`` x ``block_topk`` packed keys
    # per program; two-stage merging needs the same bound on both levels.
    merge_slots = max(1, 8192 // block_topk)
    split_cap = min(split_cap, merge_slots * merge_slots)
    splits = min(splits, split_cap, triton.cdiv(num_blocks, block_n))
    blocks_per_split = triton.cdiv(num_blocks, splits)
    while splits > 1 and blocks_per_split * (splits - 1) >= num_blocks:
        splits -= 1
        blocks_per_split = triton.cdiv(num_blocks, splits)
    partial = torch.empty(
        (rows, splits, block_topk), dtype=torch.int64, device=query.device
    )
    # PDL overlaps the merge launch with the tail of the streaming grid on
    # NVIDIA GPUs; the merge's gdc_wait still enforces the full dependency.
    use_pdl = _is_nvidia and enable_pdl
    pdl_kwargs = {"launch_pdl": True} if use_pdl else {}
    # Strided query reads keep the GEMM-view input copy-free.
    _qwen4_exp_qsa_stream_block_topk_kernel[(rows, splits)](
        query,
        key_cache,
        page_table,
        request_indices,
        complete_blocks,
        partial,
        query.shape[1],
        query.shape[2],
        num_blocks,
        page_size,
        page_expansion,
        blocks_per_split,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key_cache.stride(0),
        key_cache.stride(2),
        page_table.stride(0),
        partial.stride(0),
        partial.stride(1),
        BLOCK_TOPK=block_topk,
        BLOCK_N=block_n,
        BLOCK_H=triton.next_power_of_2(query.shape[1]),
        BLOCK_D=triton.next_power_of_2(query.shape[2]),
        ENABLE_PDL=use_pdl,
        num_warps=8,
        num_stages=2,
        **pdl_kwargs,
    )
    if splits * block_topk <= 8192:
        _qwen4_exp_qsa_merge_block_topk_kernel[(rows,)](
            partial,
            selected,
            complete_blocks,
            blocks_per_split,
            partial.stride(0),
            partial.stride(1),
            selected.stride(0),
            BLOCK_TOPK=block_topk,
            SPLITS=splits,
            POW2_SPLITS=triton.next_power_of_2(splits),
            ENABLE_PDL=use_pdl,
            num_warps=8,
            **pdl_kwargs,
        )
    else:
        # Two-stage merge: chunked programs reduce the split partials first,
        # one final program selects the global top-k of the chunk results.
        chunks = triton.cdiv(splits, merge_slots)
        scratch = torch.empty(
            (rows, chunks, block_topk), dtype=torch.int64, device=query.device
        )
        _qwen4_exp_qsa_merge_chunk_kernel[(rows, chunks)](
            partial,
            scratch,
            complete_blocks,
            blocks_per_split,
            partial.stride(0),
            partial.stride(1),
            scratch.stride(0),
            BLOCK_TOPK=block_topk,
            SPLITS=splits,
            CHUNK=merge_slots,
            ENABLE_PDL=use_pdl,
            num_warps=8,
            **pdl_kwargs,
        )
        _qwen4_exp_qsa_merge_block_topk_kernel[(rows,)](
            scratch,
            selected,
            complete_blocks,
            blocks_per_split * merge_slots,
            scratch.stride(0),
            scratch.stride(1),
            selected.stride(0),
            BLOCK_TOPK=block_topk,
            SPLITS=chunks,
            POW2_SPLITS=triton.next_power_of_2(chunks),
            ENABLE_PDL=use_pdl,
            num_warps=8,
            **pdl_kwargs,
        )
    return selected


def _qwen4_exp_qsa_block_topk_logits(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    page_table: torch.Tensor,
    request_indices: torch.Tensor,
    complete_blocks: torch.Tensor,
    *,
    page_size: int,
    block_topk: int,
    page_expansion: int,
    persistent_topk_workspace: torch.Tensor | None,
    enable_pdl: bool = True,
) -> torch.Tensor:
    """Materialized path: score every block to FP32, then radix-select.

    Writes the ``[rows, num_blocks]`` score matrix (blocks at or beyond
    ``complete_blocks`` are ``-inf``). On NVIDIA, a valid caller-owned
    workspace routes selection through the length-aware persistent CUDA radix
    kernel; ``complete_blocks`` prevents the selector from rescanning
    graph-padded columns. Other platforms and direct callers without a
    workspace retain the portable Triton top-k fallback. Callers bound the
    score matrix via the routing heuristic in ``qwen4_exp_qsa_block_topk``.
    """

    rows = query.shape[0]
    num_blocks = triton.cdiv(page_table.shape[1], page_expansion) * page_size
    logits = torch.empty((rows, num_blocks), dtype=torch.float32, device=query.device)
    use_pdl = _is_nvidia and enable_pdl
    pdl_kwargs = {"launch_pdl": True} if use_pdl else {}
    _qwen4_exp_qsa_score_blocks_kernel[(rows, triton.cdiv(num_blocks, 512))](
        query,
        key_cache,
        page_table,
        request_indices,
        complete_blocks,
        logits,
        query.shape[1],
        query.shape[2],
        num_blocks,
        page_size,
        page_expansion,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key_cache.stride(0),
        key_cache.stride(2),
        page_table.stride(0),
        logits.stride(0),
        BLOCK_N=512,
        BLOCK_H=triton.next_power_of_2(query.shape[1]),
        BLOCK_D=triton.next_power_of_2(query.shape[2]),
        ENABLE_PDL=use_pdl,
        num_warps=8,
        num_stages=2,
        **pdl_kwargs,
    )
    if (
        _is_nvidia
        and persistent_topk_workspace is not None
        and block_topk in (512, 1024, 2048)
        and persistent_topk_workspace.is_cuda
        and persistent_topk_workspace.device == logits.device
        and persistent_topk_workspace.dtype == torch.uint8
        and persistent_topk_workspace.numel() >= _PERSISTENT_TOPK_WORKSPACE_BYTES
        and has_ragged_decode_topk()
    ):
        selected = torch.empty(
            (rows, block_topk), dtype=torch.int32, device=query.device
        )
        ragged_decode_topk(
            logits,
            selected,
            block_topk,
            lengths=complete_blocks,
            workspace=persistent_topk_workspace,
            max_seq_len=num_blocks,
        )
        return selected
    return triton_topk_from_logits(logits, block_topk, enable_pdl=use_pdl)


def qwen4_exp_qsa_block_topk(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    page_table: torch.Tensor,
    request_indices: torch.Tensor,
    complete_blocks: torch.Tensor,
    *,
    page_size: int,
    block_topk: int,
    page_expansion: int = 1,
    max_partial_bytes: int = 32 * 1024 * 1024,
    solution: str = "stream",
    persistent_topk_workspace: torch.Tensor | None = None,
    enable_pdl: bool = True,
) -> torch.Tensor:
    """Select the highest-scoring compressed blocks for each query row.

    Args:
        query: Query tensor shaped ``[rows, heads, head_dim]``.
        key_cache: Flattened compressed keys shaped ``[slots, 1, head_dim]``.
        page_table: Page table shaped ``[requests, max_pages]`` stored at
            consumer granularity.
        request_indices: Owning request id per query row.
        complete_blocks: Fully compressed block counts shaped ``[rows]``.
        page_size: Compressed-cache rows covered by one logical page.
        block_topk: Blocks selected per row; must be a power of two and at
            least 64.
        page_expansion: Consumer page-table entries covered by one logical
            page.
        max_partial_bytes: Memory budget for the partial top-k buffers
            (``stream`` solution only).
        solution: ``"stream"`` fuses scoring and selection without
            materializing scores (default); ``"logits"`` materializes the
            ``[rows, num_blocks]`` FP32 scores and radix-selects them.
        persistent_topk_workspace: Optional caller-owned CUDA uint8 workspace
            of at least 1 MiB. The ``"logits"`` solution uses it for the
            length-aware persistent radix top-k when that kernel is available;
            otherwise selection falls back to portable Triton.
        enable_pdl: Allow programmatic dependent launch between the
            producer and selection kernels on NVIDIA GPUs.

    Returns:
        Int32 block ids shaped ``[rows, block_topk]``; invalid entries are
        ``-1``.
    """

    if query.ndim != 3 or key_cache.ndim != 3:
        raise ValueError("Qwen4-Exp QSA block top-k expects rank-three tensors")
    if block_topk < 64 or (block_topk & (block_topk - 1)):
        raise ValueError("Qwen4-Exp QSA block top-k needs a power-of-two topk >= 64")
    page_expansion = int(page_expansion)
    if page_expansion < 1:
        raise ValueError("Qwen4-Exp QSA block top-k needs a positive expansion")
    if solution not in ("stream", "logits"):
        raise ValueError(
            "Qwen4-Exp QSA block top-k solution must be 'stream' or 'logits', "
            f"got {solution!r}"
        )
    rows = query.shape[0]
    num_blocks = triton.cdiv(page_table.shape[1], page_expansion) * page_size
    if rows == 0 or num_blocks == 0:
        return torch.full(
            (rows, block_topk), -1, dtype=torch.int32, device=query.device
        )
    if solution == "logits":
        return _qwen4_exp_qsa_block_topk_logits(
            query,
            key_cache,
            page_table,
            request_indices,
            complete_blocks,
            page_size=page_size,
            block_topk=block_topk,
            page_expansion=page_expansion,
            persistent_topk_workspace=persistent_topk_workspace,
            enable_pdl=enable_pdl,
        )
    return _qwen4_exp_qsa_block_topk_stream(
        query,
        key_cache,
        page_table,
        request_indices,
        complete_blocks,
        page_size=page_size,
        block_topk=block_topk,
        page_expansion=page_expansion,
        max_partial_bytes=max_partial_bytes,
        enable_pdl=enable_pdl,
    )


@triton.jit
def _qwen4_exp_qsa_selected_slots_kernel(
    selected_blocks,
    complete_blocks,
    logical_positions,
    request_indices,
    page_table,
    selected_slots,
    stride_b_n,
    stride_pt_b,
    stride_o_n,
    TOKEN_TOPK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = columns < WIDTH
    complete = tl.load(complete_blocks + row).to(tl.int64)
    is_token = columns < TOKEN_TOPK
    block_ids = tl.load(
        selected_blocks + row * stride_b_n + columns // COMPRESS_RATIO,
        mask=mask & is_token,
        other=-1,
    ).to(tl.int64)
    token_ok = (block_ids >= 0) & (block_ids < complete)
    token_values = block_ids * COMPRESS_RATIO + columns % COMPRESS_RATIO
    position = tl.load(logical_positions + row).to(tl.int64)
    suffix_values = complete * COMPRESS_RATIO + (columns - TOKEN_TOPK)
    suffix_ok = suffix_values <= position
    logical_tokens = tl.where(
        is_token,
        tl.where(token_ok, token_values, -1),
        tl.where(suffix_ok, suffix_values, -1),
    )
    # Keep the final causal guard local to the physical mapping. Besides
    # protecting externally supplied block lists, this prevents a stale
    # complete-block count from exposing later rows during MTP compaction.
    valid = (logical_tokens >= 0) & (logical_tokens <= position)
    safe_tokens = tl.maximum(logical_tokens, 0)
    page_columns = safe_tokens // PAGE_SIZE
    request = tl.load(request_indices + row).to(tl.int64)
    pages = tl.load(
        page_table + request * stride_pt_b + page_columns,
        mask=mask & valid,
        other=0,
    ).to(tl.int64)
    slots = pages * PAGE_SIZE + safe_tokens % PAGE_SIZE
    tl.store(
        selected_slots + row * stride_o_n + columns,
        tl.where(valid & (pages > 0), slots, -1).to(tl.int32),
        mask=mask,
    )


def qwen4_exp_qsa_selected_slots(
    selected_blocks: torch.Tensor,
    complete_blocks: torch.Tensor,
    logical_positions: torch.Tensor,
    request_indices: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int,
    compress_ratio: int,
    token_topk: int,
) -> torch.Tensor:
    """Expand selected blocks directly into physical full-attention slots.

    Args:
        selected_blocks: Top-k block ids shaped ``[rows, block_topk]``;
            negative entries are invalid.
        complete_blocks: Fully compressed block counts shaped ``[rows]``.
        logical_positions: Absolute logical position per row.
        request_indices: Owning request id per row.
        page_table: Full-attention page table shaped
            ``[requests, max_pages]``.
        page_size: Tokens covered by one full-attention page-table entry.
        compress_ratio: Tokens grouped into one compressed block.
        token_topk: Number of block-derived tokens kept per row.

    Returns:
        Int32 physical cache slots shaped
        ``[rows, token_topk + compress_ratio - 1]``; invalid entries are
        ``-1`` and the trailing columns cover the in-progress compression
        group.
    """

    if selected_blocks.ndim != 2:
        raise ValueError("Qwen4-Exp QSA selected blocks must be rank two")
    rows = selected_blocks.shape[0]
    if complete_blocks.shape != (rows,):
        raise ValueError("complete_blocks must have one entry per selected row")
    if logical_positions.shape != (rows,):
        raise ValueError("logical_positions must have one entry per selected row")
    if request_indices.shape != (rows,):
        raise ValueError("request_indices must have one entry per selected row")
    if page_table.ndim != 2:
        raise ValueError("Qwen4-Exp QSA full page table must be rank two")
    if page_size <= 0:
        raise ValueError("Qwen4-Exp QSA full page size must be positive")
    width = token_topk + compress_ratio - 1
    # The kernel stores every column including the -1 invalid entries, so
    # the buffer starts uninitialized.
    output = torch.empty(
        (rows, width), dtype=torch.int32, device=selected_blocks.device
    )
    if rows:
        _qwen4_exp_qsa_selected_slots_kernel[(rows, triton.cdiv(width, 256))](
            selected_blocks,
            complete_blocks,
            logical_positions,
            request_indices,
            page_table,
            output,
            selected_blocks.stride(0),
            page_table.stride(0),
            output.stride(0),
            TOKEN_TOPK=token_topk,
            COMPRESS_RATIO=compress_ratio,
            PAGE_SIZE=page_size,
            WIDTH=width,
            BLOCK=256,
        )
    return output


@triton.jit
def _qwen4_exp_qsa_rope_kernel(
    inputs,
    positions,
    cos_sin_cache,
    outputs,
    rotary_dim,
    num_heads,
    head_dim,
    section0,
    section1,
    section2,
    stride_x_n,
    stride_x_h,
    stride_x_d,
    stride_p_axis,
    stride_p_row,
    stride_c_n,
    stride_o_n,
    stride_o_h,
    POS_3D: tl.constexpr,
    HAS_SECTIONS: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    half = rotary_dim // 2
    p0 = tl.load(positions + row * stride_p_row).to(tl.int64)
    if POS_3D:
        p1 = tl.load(positions + stride_p_axis + row * stride_p_row).to(tl.int64)
        p2 = tl.load(positions + 2 * stride_p_axis + row * stride_p_row).to(tl.int64)
    else:
        p1 = p0
        p2 = p0
    dim_offsets = tl.arange(0, BLOCK_D)
    half_mask = dim_offsets < half
    if HAS_SECTIONS:
        if INTERLEAVED:
            axis = tl.where(
                (dim_offsets % 3 == 1) & (dim_offsets < 3 * section1),
                1,
                tl.where((dim_offsets % 3 == 2) & (dim_offsets < 3 * section2), 2, 0),
            )
        else:
            axis = tl.where(
                dim_offsets < section0,
                0,
                tl.where(dim_offsets < section0 + section1, 1, 2),
            )
        selected_positions = tl.where(axis == 0, p0, tl.where(axis == 1, p1, p2))
    else:
        selected_positions = tl.where(half_mask, p0, 0)
    cos = tl.load(
        cos_sin_cache + selected_positions * stride_c_n + dim_offsets,
        mask=half_mask,
        other=0.0,
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_cache + selected_positions * stride_c_n + half + dim_offsets,
        mask=half_mask,
        other=0.0,
    ).to(tl.float32)
    pass_mask = (dim_offsets >= rotary_dim) & (dim_offsets < head_dim)
    for head in range(0, num_heads):
        input_row = inputs + row * stride_x_n + head * stride_x_h
        output_row = outputs + row * stride_o_n + head * stride_o_h
        first = tl.load(
            input_row + dim_offsets * stride_x_d, mask=half_mask, other=0.0
        ).to(tl.float32)
        second = tl.load(
            input_row + (half + dim_offsets) * stride_x_d,
            mask=half_mask,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            output_row + dim_offsets,
            (first * cos - second * sin).to(outputs.dtype.element_ty),
            mask=half_mask,
        )
        tl.store(
            output_row + half + dim_offsets,
            (second * cos + first * sin).to(outputs.dtype.element_ty),
            mask=half_mask,
        )
        passthrough = tl.load(
            input_row + dim_offsets * stride_x_d, mask=pass_mask, other=0.0
        )
        tl.store(output_row + dim_offsets, passthrough, mask=pass_mask)


def qwen4_exp_qsa_rope(
    inputs: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    sections: tuple[int, ...] | None = None,
    interleaved: bool = False,
) -> torch.Tensor:
    """Apply neox-style RoPE to indexer tensors in one fused kernel.

    Args:
        inputs: Tensor shaped ``[tokens, heads, head_dim]``; strided views are
            read in place.
        positions: Token positions shaped ``[tokens]`` or ``[3, tokens]`` for
            multimodal section selection; any integer dtype and layout is
            read in place.
        cos_sin_cache: Fused cos/sin table shaped
            ``[max_positions, rotary_dim]``.
        sections: Optional multimodal section sizes selecting the position
            axis per rotary half-dimension.
        interleaved: Whether multimodal sections are interleaved by axis.

    Returns:
        Rotated tensor with the same shape and dtype as ``inputs``.
    """

    if inputs.ndim != 3:
        raise ValueError("Qwen4-Exp QSA RoPE expects rank-three tensors")
    rotary_dim = cos_sin_cache.shape[-1]
    if rotary_dim % 2:
        raise ValueError("Qwen4-Exp QSA RoPE needs an even rotary dimension")
    tokens, num_heads, head_dim = inputs.shape
    # Strided reads with in-kernel casts keep the launch free of host-side
    # copy/cast kernels; the output is freshly allocated and contiguous.
    outputs = torch.empty(inputs.shape, dtype=inputs.dtype, device=inputs.device)
    if not tokens:
        return outputs
    if positions.ndim == 2:
        stride_p_axis = positions.stride(0)
    else:
        stride_p_axis = 0
    stride_p_row = positions.stride(-1)
    _qwen4_exp_qsa_rope_kernel[(tokens,)](
        inputs,
        positions,
        cos_sin_cache,
        outputs,
        rotary_dim,
        num_heads,
        head_dim,
        0 if sections is None else int(sections[0]),
        0 if sections is None else int(sections[1]),
        0 if sections is None else int(sections[2]),
        inputs.stride(0),
        inputs.stride(1),
        inputs.stride(2),
        stride_p_axis,
        stride_p_row,
        cos_sin_cache.stride(0),
        outputs.stride(0),
        outputs.stride(1),
        POS_3D=positions.ndim == 2,
        HAS_SECTIONS=sections is not None,
        INTERLEAVED=interleaved,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
    )
    return outputs


__all__ = [
    "qwen4_exp_qsa_block_topk",
    "qwen4_exp_qsa_complete_blocks",
    "qwen4_exp_qsa_compress_and_store",
    "qwen4_exp_qsa_group_cache_locs",
    "qwen4_exp_qsa_logical_layout",
    "qwen4_exp_qsa_mqa_scores",
    "qwen4_exp_qsa_norm_rope",
    "qwen4_exp_qsa_recent_write",
    "qwen4_exp_qsa_rope",
    "qwen4_exp_qsa_selected_slots",
    "qwen4_exp_qsa_sparse_attention",
    "qwen4_exp_qsa_stage_draft",
    "qwen4_exp_qsa_stage_verify",
]

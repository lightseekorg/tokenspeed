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

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.triton.dsa_sparse_layout import (
    local_topk_to_global_slots,
)


@triton.jit
def _dsa_decode_logits_kernel(
    q,
    index_k,
    weights,
    seq_lens,
    block_table,
    logits,
    block_table_stride: tl.constexpr,
    logits_stride: tl.constexpr,
    page_size: tl.constexpr,
    max_seq_len: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    softmax_scale: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0)
    block_id = tl.program_id(1)
    offsets = block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    seq_len = tl.load(seq_lens + token).to(tl.int32)
    valid = offsets < seq_len
    block_idx = offsets // page_size
    block_offset = offsets - block_idx * page_size
    valid = valid & (offsets < max_seq_len)
    page = tl.load(
        block_table + token * block_table_stride + block_idx,
        mask=valid,
        other=0,
    ).to(tl.int64)
    slots = page * page_size + block_offset
    scores = tl.zeros((BLOCK_N,), tl.float32)

    dim_offsets = tl.arange(0, BLOCK_D)
    for head in tl.static_range(0, num_heads):
        head_weight = tl.load(weights + token * num_heads + head).to(tl.float32)
        head_score = tl.zeros((BLOCK_N,), tl.float32)
        for dim_start in tl.static_range(0, head_dim, BLOCK_D):
            dims = dim_start + dim_offsets
            q_vals = tl.load(
                q + (token * num_heads + head) * head_dim + dims,
                mask=dims < head_dim,
                other=0.0,
            ).to(tl.float32)
            k_vals = tl.load(
                index_k + slots[:, None] * head_dim + dims[None, :],
                mask=valid[:, None] & (dims[None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)
            head_score += tl.sum(k_vals * q_vals[None, :], axis=1)
        scores += head_score * head_weight

    scores *= softmax_scale
    scores = tl.where(valid, scores, -float("inf"))
    tl.store(
        logits + token * logits_stride + offsets,
        scores,
        mask=offsets < max_seq_len,
    )


@triton.jit
def _dsa_prefill_logits_kernel(
    q,
    index_k,
    weights,
    kv_workspace_slots,
    row_starts,
    row_ends,
    logits,
    logits_stride: tl.constexpr,
    seq_len_sum: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    softmax_scale: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0)
    block_id = tl.program_id(1)
    offsets = block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    row_start = tl.load(row_starts + token).to(tl.int32)
    row_end = tl.load(row_ends + token).to(tl.int32)
    valid = (offsets >= row_start) & (offsets < row_end) & (offsets < seq_len_sum)
    slots = tl.load(
        kv_workspace_slots + offsets,
        mask=offsets < seq_len_sum,
        other=0,
    )
    slots = slots.to(tl.int64)
    scores = tl.zeros((BLOCK_N,), tl.float32)

    dim_offsets = tl.arange(0, BLOCK_D)
    for head in tl.static_range(0, num_heads):
        head_weight = tl.load(weights + token * num_heads + head).to(tl.float32)
        head_score = tl.zeros((BLOCK_N,), tl.float32)
        for dim_start in tl.static_range(0, head_dim, BLOCK_D):
            dims = dim_start + dim_offsets
            q_vals = tl.load(
                q + (token * num_heads + head) * head_dim + dims,
                mask=dims < head_dim,
                other=0.0,
            ).to(tl.float32)
            k_vals = tl.load(
                index_k + slots[:, None] * head_dim + dims[None, :],
                mask=valid[:, None] & (dims[None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)
            head_score += tl.sum(k_vals * q_vals[None, :], axis=1)
        scores += head_score * head_weight

    scores *= softmax_scale
    scores = tl.where(valid, scores, -float("inf"))
    tl.store(
        logits + token * logits_stride + offsets,
        scores,
        mask=offsets < seq_len_sum,
    )


def _check_common_inputs(
    q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
) -> None:
    if q.dtype != torch.bfloat16:
        raise TypeError(f"DSA Triton top-k expects BF16 q, got {q.dtype}")
    if index_k.dtype != torch.bfloat16:
        raise TypeError(f"DSA Triton top-k expects BF16 index_k, got {index_k.dtype}")
    if weights.dtype != torch.float32:
        raise TypeError(f"DSA Triton top-k expects FP32 weights, got {weights.dtype}")
    if q.dim() != 3:
        raise ValueError(f"q must be [tokens, heads, dim], got {tuple(q.shape)}")
    if index_k.dim() != 2 or index_k.shape[1] != q.shape[2]:
        raise ValueError(
            "index_k must be [slots, dim] matching q dim, got "
            f"index_k={tuple(index_k.shape)}, q={tuple(q.shape)}"
        )
    if weights.shape != q.shape[:2]:
        raise ValueError(
            "weights must be [tokens, heads] matching q, got "
            f"weights={tuple(weights.shape)}, q={tuple(q.shape)}"
        )
    if q.shape[2] % 64 != 0:
        raise ValueError(
            f"DSA Triton top-k requires dim multiple of 64, got {q.shape[2]}"
        )


@triton.jit
def _fp32_to_ordered_key(x):
    bits = x.to(tl.uint32, bitcast=True)
    sign = bits & 0x80000000
    return bits ^ tl.where(sign != 0, 0xFFFFFFFF, 0x80000000)


@triton.jit
def _ordered_key_to_fp32(x):
    sign = x & 0x80000000
    bits = x ^ tl.where(sign != 0, 0x80000000, 0xFFFFFFFF)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _dsa_logits_topk_kernel(
    logits,
    out,
    logits_stride: tl.constexpr,
    out_stride: tl.constexpr,
    n_cols: tl.constexpr,
    n_cols_padded: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = (n_cols_padded - BLOCK_N) + tl.arange(0, BLOCK_N)
    valid = offsets < n_cols
    values = tl.load(
        logits + row * logits_stride + offsets,
        mask=valid,
        other=-float("inf"),
    )
    value_keys = _fp32_to_ordered_key(values).to(tl.uint64)
    index_keys = (n_cols_padded - offsets).to(tl.uint64)
    packed = (value_keys << 32) | index_keys
    acc = tl.topk(packed[None, :], topk, dim=1)

    loop_iterations: tl.constexpr = n_cols_padded // BLOCK_N - 1
    for _ in tl.static_range(0, loop_iterations):
        acc = tl.bitonic_merge(acc)
        offsets -= BLOCK_N
        valid = offsets < n_cols
        values = tl.load(
            logits + row * logits_stride + offsets,
            mask=valid,
            other=-float("inf"),
        )
        value_keys = _fp32_to_ordered_key(values).to(tl.uint64)
        index_keys = (n_cols_padded - offsets).to(tl.uint64)
        packed = (value_keys << 32) | index_keys
        acc = tl.maximum(acc, tl.topk(packed[None, :], topk, dim=1))

    acc = tl.sort(acc, dim=1, descending=True)
    top_offsets = tl.arange(0, topk)
    packed_top = tl.reshape(acc, (topk,))
    indices = n_cols_padded - (packed_top & 0xFFFFFFFF).to(tl.int32)
    values = _ordered_key_to_fp32((packed_top >> 32).to(tl.uint32))
    valid_top = (top_offsets < n_cols) & (indices >= 0) & (indices < n_cols)
    valid_top = valid_top & (values != -float("inf"))
    tl.store(
        out + row * out_stride + top_offsets,
        tl.where(valid_top, indices, -1),
        mask=top_offsets < topk,
    )


def _is_power_of_2(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _next_power_of_2(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (int(value) - 1).bit_length()


def _topk_with_padding(logits: torch.Tensor, topk: int) -> torch.Tensor:
    topk = int(topk)
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    if not _is_power_of_2(topk):
        raise ValueError(f"DSA Triton top-k requires power-of-two topk, got {topk}")
    if logits.dim() != 2:
        raise ValueError(f"logits must be [rows, cols], got {tuple(logits.shape)}")
    if logits.dtype != torch.float32:
        raise TypeError(f"logits must be FP32, got {logits.dtype}")

    rows, cols = logits.shape
    out = torch.full((rows, topk), -1, dtype=torch.int32, device=logits.device)
    if rows == 0 or cols == 0:
        return out
    n_cols_padded = _next_power_of_2(max(cols, topk))
    block_n = min(n_cols_padded, 2048)
    block_n = max(block_n, topk)
    if n_cols_padded % block_n != 0:
        raise ValueError(
            "DSA Triton top-k requires padded cols divisible by block size, got "
            f"cols={cols}, padded={n_cols_padded}, block={block_n}"
        )
    _dsa_logits_topk_kernel[(rows,)](
        logits.contiguous(),
        out,
        logits.stride(0),
        out.stride(0),
        n_cols=cols,
        n_cols_padded=n_cols_padded,
        topk=topk,
        BLOCK_N=block_n,
        num_warps=8,
        num_stages=1,
    )
    return out


def dsa_decode_topk(
    q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    page_size: int,
    topk: int,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GLM DSA decode top-k global KV slots.

    Args:
        q: BF16 indexer query with shape [tokens, index_heads, head_dim].
        index_k: BF16 index-K cache with shape [slots, head_dim].
        weights: FP32 per-token/head indexer weights with shape
            [tokens, index_heads].
        seq_lens: Visible context lengths for each query token.
        block_table: Paged KV block table for each query token.
        page_size: Number of token slots in a KV page.
        topk: Number of sparse KV slots to select.
        softmax_scale: Multiplicative score scale, normally
            index_head_dim ** -0.5.

    Returns:
        (topk_slots, topk_lens). topk_slots contains global KV slots with
        shape [tokens, topk] and topk_lens contains valid counts.
    """
    _check_common_inputs(q, index_k, weights)
    if seq_lens.dim() != 1 or seq_lens.numel() != q.shape[0]:
        raise ValueError(
            "seq_lens must be [tokens], got "
            f"{tuple(seq_lens.shape)} for q={tuple(q.shape)}"
        )
    if block_table.dim() != 2 or block_table.shape[0] < q.shape[0]:
        raise ValueError(
            "block_table must have at least one row per token, got "
            f"block_table={tuple(block_table.shape)}, q={tuple(q.shape)}"
        )
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    if q.shape[0] == 0:
        return (
            torch.empty((0, int(topk)), dtype=torch.int32, device=q.device),
            torch.empty((0,), dtype=torch.int32, device=q.device),
        )
    if not q.is_cuda:
        raise RuntimeError("DSA Triton decode top-k requires CUDA tensors")

    q = q.contiguous()
    index_k = index_k.contiguous()
    weights = weights.contiguous()
    seq_lens = seq_lens.to(device=q.device, dtype=torch.int32).contiguous()
    block_table = block_table.to(device=q.device, dtype=torch.int32).contiguous()
    max_seq_len = int(block_table.shape[1]) * int(page_size)
    logits = torch.empty(
        (q.shape[0], max_seq_len), dtype=torch.float32, device=q.device
    )
    block_n = 64
    grid = (q.shape[0], triton.cdiv(max_seq_len, block_n))
    _dsa_decode_logits_kernel[grid](
        q,
        index_k,
        weights,
        seq_lens,
        block_table,
        logits,
        block_table.stride(0),
        logits.stride(0),
        page_size=int(page_size),
        max_seq_len=max_seq_len,
        num_heads=q.shape[1],
        head_dim=q.shape[2],
        softmax_scale=float(softmax_scale),
        BLOCK_N=block_n,
        BLOCK_D=64,
        num_warps=4,
        num_stages=1,
    )
    local_topk_offsets = _topk_with_padding(logits, int(topk))
    return local_topk_to_global_slots(
        local_topk_offsets=local_topk_offsets,
        block_table=block_table,
        block_size=int(page_size),
        seq_lens=seq_lens,
    )


def dsa_prefill_topk(
    q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    kv_workspace_slots: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    *,
    topk: int,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GLM DSA prefill top-k workspace-row indices.

    Args:
        q: BF16 indexer query with shape [tokens, index_heads, head_dim].
        index_k: BF16 index-K cache with shape [slots, head_dim].
        weights: FP32 per-token/head indexer weights with shape
            [tokens, index_heads].
        kv_workspace_slots: Global KV slot for each packed prefill workspace row.
        row_starts: Inclusive workspace-row start per query token.
        row_ends: Exclusive workspace-row end per query token.
        topk: Number of sparse workspace rows to select.
        softmax_scale: Multiplicative score scale, normally
            index_head_dim ** -0.5.

    Returns:
        (workspace_indices, topk_lens). workspace_indices contains absolute rows
        into kv_workspace_slots with shape [tokens, topk].
    """
    _check_common_inputs(q, index_k, weights)
    if kv_workspace_slots.dim() != 1:
        raise ValueError(
            f"kv_workspace_slots must be 1-D, got {tuple(kv_workspace_slots.shape)}"
        )
    if row_starts.shape != (q.shape[0],) or row_ends.shape != (q.shape[0],):
        raise ValueError(
            "row_starts/row_ends must be [tokens], got "
            f"row_starts={tuple(row_starts.shape)}, row_ends={tuple(row_ends.shape)}, "
            f"q={tuple(q.shape)}"
        )
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    if q.shape[0] == 0:
        return (
            torch.empty((0, int(topk)), dtype=torch.int32, device=q.device),
            torch.empty((0,), dtype=torch.int32, device=q.device),
        )
    if not q.is_cuda:
        raise RuntimeError("DSA Triton prefill top-k requires CUDA tensors")

    q = q.contiguous()
    index_k = index_k.contiguous()
    weights = weights.contiguous()
    kv_workspace_slots = kv_workspace_slots.to(
        device=q.device, dtype=torch.int64
    ).contiguous()
    row_starts = row_starts.to(device=q.device, dtype=torch.int32).contiguous()
    row_ends = row_ends.to(device=q.device, dtype=torch.int32).contiguous()
    seq_len_sum = int(kv_workspace_slots.numel())
    if seq_len_sum == 0:
        return (
            torch.full((q.shape[0], int(topk)), -1, dtype=torch.int32, device=q.device),
            torch.zeros((q.shape[0],), dtype=torch.int32, device=q.device),
        )

    logits = torch.empty(
        (q.shape[0], seq_len_sum), dtype=torch.float32, device=q.device
    )
    block_n = 64
    grid = (q.shape[0], triton.cdiv(seq_len_sum, block_n))
    _dsa_prefill_logits_kernel[grid](
        q,
        index_k,
        weights,
        kv_workspace_slots,
        row_starts,
        row_ends,
        logits,
        logits.stride(0),
        seq_len_sum=seq_len_sum,
        num_heads=q.shape[1],
        head_dim=q.shape[2],
        softmax_scale=float(softmax_scale),
        BLOCK_N=block_n,
        BLOCK_D=64,
        num_warps=4,
        num_stages=1,
    )
    workspace_indices = _topk_with_padding(logits, int(topk))
    valid = (workspace_indices >= row_starts[:, None]) & (
        workspace_indices < row_ends[:, None]
    )
    workspace_indices = torch.where(valid, workspace_indices, -1)
    topk_lens = torch.minimum(
        (row_ends - row_starts).clamp_min_(0),
        torch.full_like(row_ends, int(topk)),
    ).to(torch.int32)
    return workspace_indices, topk_lens


__all__ = ["dsa_decode_topk", "dsa_prefill_topk"]

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
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

_RADIX_TOPK_MIN_COLS = 65536
_RADIX_TOPK_BLOCK_N = 4096


@triton.jit
def _local_topk_to_global_slots_kernel(
    global_topk_slots_ptr,
    global_topk_slots_stride,
    topk_lens_ptr,
    local_topk_offsets_ptr,
    local_topk_offsets_stride,
    seq_lens_ptr,
    block_table_ptr,
    block_table_stride,
    block_table_cols: tl.constexpr,
    block_size: tl.constexpr,
    topk: tl.constexpr,
    has_seq_lens: tl.constexpr,
    q_len_per_req: tl.constexpr,
    block: tl.constexpr,
):
    token_idx = tl.program_id(0)
    req_idx = token_idx // q_len_per_req
    count = tl.zeros((), dtype=tl.int32)
    seq_len = tl.full((), block_table_cols * block_size, dtype=tl.int32)
    if has_seq_lens:
        base = tl.load(seq_lens_ptr + req_idx).to(tl.int32)
        seq_len = base - (q_len_per_req - 1) + (token_idx % q_len_per_req)
        seq_len = tl.maximum(seq_len, 0)

    for start in range(0, topk, block):
        offsets = start + tl.arange(0, block)
        mask = offsets < topk
        local_idx = tl.load(
            local_topk_offsets_ptr + token_idx * local_topk_offsets_stride + offsets,
            mask=mask,
            other=-1,
        )
        valid = (local_idx >= 0) & (local_idx < seq_len)
        block_idx = local_idx // block_size
        block_offset = local_idx % block_size
        valid = valid & (block_idx >= 0) & (block_idx < block_table_cols)
        page = tl.load(
            block_table_ptr + req_idx * block_table_stride + block_idx,
            mask=mask & valid,
            other=0,
        )
        slot = page * block_size + block_offset
        tl.store(
            global_topk_slots_ptr + token_idx * global_topk_slots_stride + offsets,
            tl.where(valid, slot, -1),
            mask=mask,
        )
        count += tl.sum(valid.to(tl.int32), axis=0)

    tl.store(topk_lens_ptr + token_idx, count)


def local_topk_to_global_slots(
    *,
    local_topk_offsets: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    seq_lens: torch.Tensor | None = None,
    q_len_per_req: int = 1,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if local_topk_offsets.dtype != torch.int32:
        raise TypeError(
            f"local_topk_offsets must be int32, got {local_topk_offsets.dtype}"
        )
    if local_topk_offsets.dim() != 2:
        raise ValueError(
            "local_topk_offsets must be [tokens, topk], got "
            f"{tuple(local_topk_offsets.shape)}"
        )
    if block_table.dim() != 2:
        raise ValueError(
            f"block_table must be [tokens, pages], got {block_table.shape}"
        )
    if block_table.shape[1] == 0:
        raise ValueError("block_table must have at least one page column")
    num_tokens, topk = local_topk_offsets.shape
    q_len_per_req = int(q_len_per_req)
    if q_len_per_req < 1 or num_tokens % q_len_per_req != 0:
        raise ValueError(
            f"q_len_per_req={q_len_per_req} must divide tokens={num_tokens}"
        )
    num_reqs = num_tokens // q_len_per_req
    if block_table.shape[0] < num_reqs:
        raise ValueError(
            "block_table must have at least one row per request: "
            f"rows={block_table.shape[0]}, reqs={num_reqs}"
        )
    if seq_lens is not None and seq_lens.dim() != 1:
        raise ValueError(f"seq_lens must be 1-D, got {tuple(seq_lens.shape)}")
    if seq_lens is not None and seq_lens.numel() < num_reqs:
        raise ValueError(
            "seq_lens must have at least one entry per request: "
            f"lens={seq_lens.numel()}, reqs={num_reqs}"
        )

    if out is None:
        global_slots = torch.empty_like(local_topk_offsets)
    else:
        if out.shape != local_topk_offsets.shape:
            raise ValueError(
                f"out must have shape {tuple(local_topk_offsets.shape)}, "
                f"got {tuple(out.shape)}"
            )
        if out.dtype != torch.int32 or out.device != local_topk_offsets.device:
            raise TypeError(
                "out must be int32 on the same device as local_topk_offsets, "
                f"got {out.dtype} on {out.device}"
            )
        global_slots = out
    if lens_out is None:
        lens = torch.empty(
            num_tokens,
            dtype=torch.int32,
            device=local_topk_offsets.device,
        )
    else:
        if lens_out.shape != (num_tokens,):
            raise ValueError(
                "lens_out must have shape "
                f"({num_tokens},), got {tuple(lens_out.shape)}"
            )
        if (
            lens_out.dtype != torch.int32
            or lens_out.device != local_topk_offsets.device
        ):
            raise TypeError(
                "lens_out must be int32 on the same device as local_topk_offsets, "
                f"got {lens_out.dtype} on {lens_out.device}"
            )
        lens = lens_out
    if num_tokens == 0:
        return global_slots, lens

    if not local_topk_offsets.is_cuda:
        raise RuntimeError("DSA local top-k slot conversion requires CUDA tensors.")

    block_table = block_table.to(device=local_topk_offsets.device, dtype=torch.int32)
    if seq_lens is not None:
        seq_lens = seq_lens.to(device=local_topk_offsets.device, dtype=torch.int32)
    seq_lens_arg = block_table[:, 0] if seq_lens is None else seq_lens
    _local_topk_to_global_slots_kernel[(num_tokens,)](
        global_slots,
        global_slots.stride(0),
        lens,
        local_topk_offsets,
        local_topk_offsets.stride(0),
        seq_lens_arg,
        block_table,
        block_table.stride(0),
        block_table.shape[1],
        block_size=int(block_size),
        topk=topk,
        has_seq_lens=seq_lens is not None,
        q_len_per_req=q_len_per_req,
        block=1024,
    )
    return global_slots, lens


@triton.jit
def _workspace_topk_to_global_slots_kernel(
    out_ptr,
    workspace_indices_ptr,
    workspace_indices_stride: tl.constexpr,
    kv_workspace_slots_ptr,
    total: tl.constexpr,
    topk: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    workspace_idx = tl.load(workspace_indices_ptr + offsets, mask=mask, other=-1)
    valid = mask & (workspace_idx >= 0)
    slots = tl.load(kv_workspace_slots_ptr + workspace_idx, mask=valid, other=-1)
    tl.store(out_ptr + offsets, tl.where(valid, slots, -1), mask=mask)


def workspace_topk_to_global_slots(
    *,
    workspace_indices: torch.Tensor,
    kv_workspace_slots: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if workspace_indices.dtype != torch.int32:
        raise TypeError(
            f"workspace_indices must be int32, got {workspace_indices.dtype}"
        )
    if workspace_indices.dim() != 2:
        raise ValueError(
            "workspace_indices must be [tokens, topk], got "
            f"{tuple(workspace_indices.shape)}"
        )
    if kv_workspace_slots.dim() != 1:
        raise ValueError(
            f"kv_workspace_slots must be 1-D, got {tuple(kv_workspace_slots.shape)}"
        )
    if out is None:
        out = torch.empty_like(workspace_indices)
    elif (
        out.shape != workspace_indices.shape
        or out.dtype != torch.int32
        or out.device != workspace_indices.device
    ):
        raise ValueError(
            "out must be int32 with shape "
            f"{tuple(workspace_indices.shape)} on {workspace_indices.device}, "
            f"got {tuple(out.shape)} {out.dtype} on {out.device}"
        )
    if workspace_indices.numel() == 0:
        return out
    if not workspace_indices.is_cuda:
        raise RuntimeError("DSA workspace top-k slot conversion requires CUDA tensors.")

    workspace_indices = workspace_indices.contiguous()
    kv_workspace_slots = kv_workspace_slots.to(
        device=workspace_indices.device,
        dtype=torch.int64,
    ).contiguous()
    out_view = out.contiguous() if not out.is_contiguous() else out
    total = int(workspace_indices.numel())
    block = 256
    _workspace_topk_to_global_slots_kernel[(triton.cdiv(total, block),)](
        out_view,
        workspace_indices,
        workspace_indices.stride(0),
        kv_workspace_slots,
        total=total,
        topk=workspace_indices.shape[1],
        BLOCK=block,
        num_warps=4,
        num_stages=1,
    )
    if out_view.data_ptr() != out.data_ptr():
        out.copy_(out_view)
    return out


@triton.jit
def _dsa_decode_logits_fp8_kernel(
    q,
    index_k_fp8,
    index_k_scale,
    weights,
    seq_lens,
    block_table,
    logits,
    block_table_stride: tl.constexpr,
    logits_stride: tl.constexpr,
    page_size: tl.constexpr,
    row_bytes: tl.constexpr,
    max_seq_len: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_groups: tl.constexpr,
    softmax_scale: tl.constexpr,
    q_len_per_req: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0)
    block_id = tl.program_id(1)
    req = token // q_len_per_req
    offsets = block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    base = tl.load(seq_lens + req).to(tl.int32)
    seq_len = base - (q_len_per_req - 1) + (token % q_len_per_req)
    seq_len = tl.maximum(seq_len, 0)
    valid = offsets < seq_len
    block_idx = offsets // page_size
    block_offset = offsets - block_idx * page_size
    valid = valid & (offsets < max_seq_len)
    page = tl.load(
        block_table + req * block_table_stride + block_idx,
        mask=valid,
        other=0,
    ).to(tl.int64)
    page_bytes = page_size * row_bytes
    fp8_base = page * page_bytes + block_offset * head_dim
    scale_base = (
        page * (page_bytes // 4)
        + (page_size * head_dim) // 4
        + block_offset * num_groups
    )
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
                index_k_fp8 + fp8_base[:, None] + dims[None, :],
                mask=valid[:, None] & (dims[None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)
            k_scale = tl.load(
                index_k_scale + scale_base + dim_start // 128,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            head_score += tl.sum(k_vals * k_scale[:, None] * q_vals[None, :], axis=1)
        scores += head_score * head_weight

    scores *= softmax_scale
    scores = tl.where(valid, scores, -float("inf"))
    tl.store(
        logits + token * logits_stride + offsets,
        scores,
        mask=offsets < max_seq_len,
    )


@triton.jit
def _dsa_prefill_logits_fp8_kernel(
    q,
    index_k_fp8,
    index_k_scale,
    weights,
    kv_workspace_slots,
    row_starts,
    row_ends,
    logits,
    logits_stride: tl.constexpr,
    seq_len_sum: tl.constexpr,
    page_size: tl.constexpr,
    row_bytes: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_groups: tl.constexpr,
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
    ).to(tl.int64)
    page = slots // page_size
    block_offset = slots - page * page_size
    page_bytes = page_size * row_bytes
    fp8_base = page * page_bytes + block_offset * head_dim
    scale_base = (
        page * (page_bytes // 4)
        + (page_size * head_dim) // 4
        + block_offset * num_groups
    )
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
                index_k_fp8 + fp8_base[:, None] + dims[None, :],
                mask=valid[:, None] & (dims[None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)
            k_scale = tl.load(
                index_k_scale + scale_base + dim_start // 128,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            head_score += tl.sum(k_vals * k_scale[:, None] * q_vals[None, :], axis=1)
        scores += head_score * head_weight

    scores *= softmax_scale
    scores = tl.where(valid, scores, -float("inf"))
    tl.store(
        logits + token * logits_stride + offsets,
        scores,
        mask=offsets < seq_len_sum,
    )


def _check_q_weights(q: torch.Tensor, weights: torch.Tensor) -> None:
    if q.dtype != torch.bfloat16:
        raise TypeError(f"DSA Triton top-k expects BF16 q, got {q.dtype}")
    if weights.dtype != torch.float32:
        raise TypeError(f"DSA Triton top-k expects FP32 weights, got {weights.dtype}")
    if q.dim() != 3:
        raise ValueError(f"q must be [tokens, heads, dim], got {tuple(q.shape)}")
    if weights.shape != q.shape[:2]:
        raise ValueError(
            "weights must be [tokens, heads] matching q, got "
            f"weights={tuple(weights.shape)}, q={tuple(q.shape)}"
        )
    if q.shape[2] % 64 != 0:
        raise ValueError(
            f"DSA Triton top-k requires dim multiple of 64, got {q.shape[2]}"
        )


def _check_packed_fp8_inputs(
    q: torch.Tensor,
    index_k_cache: torch.Tensor,
    weights: torch.Tensor,
    page_size: int,
) -> int:
    _check_q_weights(q, weights)
    if q.shape[2] % 128 != 0:
        raise ValueError(
            "DSA Triton FP8 top-k requires dim multiple of 128, got " f"{q.shape[2]}"
        )
    if index_k_cache.dtype != torch.uint8:
        raise TypeError(
            "DSA Triton FP8 top-k expects uint8 packed index_k_cache, got "
            f"{index_k_cache.dtype}"
        )
    row_bytes = q.shape[2] + q.shape[2] // 128 * 4
    if index_k_cache.dim() != 2 or index_k_cache.shape[1] != row_bytes:
        raise ValueError(
            "index_k_cache must be [slots, row_bytes] matching q dim, got "
            f"index_k_cache={tuple(index_k_cache.shape)}, "
            f"expected row_bytes={row_bytes}, q={tuple(q.shape)}"
        )
    if index_k_cache.shape[0] % int(page_size) != 0:
        raise ValueError(
            "index_k_cache slot count must be divisible by page_size, got "
            f"slots={index_k_cache.shape[0]}, page_size={page_size}"
        )
    return row_bytes


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


@triton.jit
def _dsa_radix_hist_kernel(
    logits,
    prefixes,
    hist,
    logits_stride: tl.constexpr,
    hist_tiles: tl.constexpr,
    n_cols: tl.constexpr,
    shift: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offsets = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < n_cols
    values = tl.load(
        logits + row * logits_stride + offsets,
        mask=mask,
        other=-float("inf"),
    )
    keys = _fp32_to_ordered_key(values)
    prefix = tl.load(prefixes + row).to(tl.uint32)
    if shift == 28:
        prefix_match = mask
    else:
        prefix_match = (keys >> (shift + 4)) == prefix
    bucket = (keys >> shift) & 0xF
    base = (row * hist_tiles + tile) * 16
    for b in tl.static_range(0, 16):
        count = tl.sum(tl.where(mask & prefix_match & (bucket == b), 1, 0))
        tl.store(hist + base + b, count)


@triton.jit
def _dsa_radix_update_kernel(
    prefixes,
    remaining,
    hist,
    hist_tiles: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    row = tl.program_id(0)
    tile_offsets = tl.arange(0, BLOCK_TILES)
    tile_mask = tile_offsets < hist_tiles
    row_hist = hist + row * hist_tiles * 16
    kth = tl.load(remaining + row).to(tl.int32)
    cumulative = tl.full((), 0, dtype=tl.int32)
    selected = tl.full((), 0, dtype=tl.uint32)
    selected_remaining = kth
    found = False
    for b_desc in tl.static_range(0, 16):
        b = 15 - b_desc
        counts = tl.load(row_hist + tile_offsets * 16 + b, mask=tile_mask, other=0)
        count = tl.sum(counts).to(tl.int32)
        take = (found == False) & (kth <= cumulative + count)
        selected = tl.where(take, b, selected)
        selected_remaining = tl.where(take, kth - cumulative, selected_remaining)
        cumulative += tl.where(found == False, count, 0)
        found = found | take
    prefix = tl.load(prefixes + row).to(tl.uint32)
    tl.store(prefixes + row, (prefix << 4) | selected)
    tl.store(remaining + row, selected_remaining)


@triton.jit
def _dsa_radix_scatter_kernel(
    logits,
    prefixes,
    remaining,
    out_values,
    out_indices,
    logits_stride: tl.constexpr,
    out_stride: tl.constexpr,
    n_cols: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    top_offsets = tl.arange(0, topk)
    tl.store(out_values + row * out_stride + top_offsets, -float("inf"))
    tl.store(out_indices + row * out_stride + top_offsets, -1)

    threshold = tl.load(prefixes + row).to(tl.uint32)
    keep_equal = tl.load(remaining + row).to(tl.int32)
    count_greater = topk - keep_equal
    num_greater = tl.full((), 0, dtype=tl.int32)
    num_equal = tl.full((), 0, dtype=tl.int32)
    for start in tl.range(0, n_cols, BLOCK_N):
        offsets = start + tl.arange(0, BLOCK_N)
        mask = offsets < n_cols
        values = tl.load(
            logits + row * logits_stride + offsets,
            mask=mask,
            other=-float("inf"),
        )
        finite = values != -float("inf")
        keys = _fp32_to_ordered_key(values)
        greater = mask & finite & (keys > threshold)
        equal = mask & finite & (keys == threshold)

        greater_pos = num_greater + tl.cumsum(greater.to(tl.int32), 0) - 1
        greater_mask = greater & (greater_pos < topk)
        tl.store(
            out_values + row * out_stride + greater_pos,
            values,
            mask=greater_mask,
        )
        tl.store(
            out_indices + row * out_stride + greater_pos,
            offsets,
            mask=greater_mask,
        )
        num_greater += tl.sum(greater.to(tl.int32))

        equal_pos = count_greater + num_equal + tl.cumsum(equal.to(tl.int32), 0) - 1
        equal_mask = (
            equal & (equal_pos < topk) & (equal_pos < count_greater + keep_equal)
        )
        tl.store(
            out_values + row * out_stride + equal_pos,
            values,
            mask=equal_mask,
        )
        tl.store(
            out_indices + row * out_stride + equal_pos,
            offsets,
            mask=equal_mask,
        )
        num_equal += tl.sum(equal.to(tl.int32))


@triton.jit
def _dsa_radix_sort_selected_kernel(
    values,
    indices,
    out,
    stride: tl.constexpr,
    n_cols_padded: tl.constexpr,
    topk: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, topk)
    vals = tl.load(values + row * stride + offsets)
    idx = tl.load(indices + row * stride + offsets).to(tl.int32)
    valid = idx >= 0
    value_keys = _fp32_to_ordered_key(vals).to(tl.uint64)
    index_keys = (n_cols_padded - idx).to(tl.uint64)
    packed = (value_keys << 32) | index_keys
    packed = tl.where(valid, packed, tl.zeros_like(packed))
    packed = tl.sort(packed[None, :], dim=1, descending=True)
    packed = tl.reshape(packed, (topk,))
    sorted_idx = n_cols_padded - (packed & 0xFFFFFFFF).to(tl.int32)
    sorted_vals = _ordered_key_to_fp32((packed >> 32).to(tl.uint32))
    valid = (
        (sorted_idx >= 0)
        & (sorted_idx < n_cols_padded)
        & (sorted_vals != -float("inf"))
    )
    tl.store(out + row * stride + offsets, tl.where(valid, sorted_idx, -1))


def _is_power_of_2(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _next_power_of_2(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (int(value) - 1).bit_length()


def _radix_topk(logits: torch.Tensor, topk: int) -> torch.Tensor:
    rows, cols = logits.shape
    block_n = _RADIX_TOPK_BLOCK_N
    tiles = triton.cdiv(cols, block_n)
    hist = torch.empty((rows, tiles, 16), dtype=torch.int32, device=logits.device)
    prefixes = torch.zeros((rows,), dtype=torch.int32, device=logits.device)
    remaining = torch.full(
        (rows,), min(int(topk), cols), dtype=torch.int32, device=logits.device
    )
    out_values = torch.empty((rows, topk), dtype=torch.float32, device=logits.device)
    out_indices = torch.empty((rows, topk), dtype=torch.int32, device=logits.device)
    out = torch.empty((rows, topk), dtype=torch.int32, device=logits.device)
    block_tiles = _next_power_of_2(tiles)

    for shift in range(28, -1, -4):
        _dsa_radix_hist_kernel[(rows, tiles)](
            logits,
            prefixes,
            hist,
            logits.stride(0),
            tiles,
            n_cols=cols,
            shift=shift,
            BLOCK_N=block_n,
            num_warps=8,
            num_stages=1,
        )
        _dsa_radix_update_kernel[(rows,)](
            prefixes,
            remaining,
            hist,
            hist_tiles=tiles,
            BLOCK_TILES=block_tiles,
            num_warps=8,
            num_stages=1,
        )

    _dsa_radix_scatter_kernel[(rows,)](
        logits,
        prefixes,
        remaining,
        out_values,
        out_indices,
        logits.stride(0),
        out_indices.stride(0),
        n_cols=cols,
        topk=int(topk),
        BLOCK_N=block_n,
        num_warps=8,
        num_stages=1,
    )
    _dsa_radix_sort_selected_kernel[(rows,)](
        out_values,
        out_indices,
        out,
        out.stride(0),
        n_cols_padded=_next_power_of_2(max(cols, int(topk))),
        topk=int(topk),
        num_warps=8,
        num_stages=1,
    )
    return out


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
    if rows == 0 or cols == 0:
        return torch.full((rows, topk), -1, dtype=torch.int32, device=logits.device)
    logits = logits.contiguous()
    if cols >= _RADIX_TOPK_MIN_COLS:
        return _radix_topk(logits, topk)
    out = torch.full((rows, topk), -1, dtype=torch.int32, device=logits.device)
    n_cols_padded = _next_power_of_2(max(cols, topk))
    block_n = min(n_cols_padded, 2048)
    block_n = max(block_n, topk)
    if n_cols_padded % block_n != 0:
        raise ValueError(
            "DSA Triton top-k requires padded cols divisible by block size, got "
            f"cols={cols}, padded={n_cols_padded}, block={block_n}"
        )
    _dsa_logits_topk_kernel[(rows,)](
        logits,
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


def dsa_decode_topk_fp8(
    q: torch.Tensor,
    index_k_cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    page_size: int,
    topk: int,
    softmax_scale: float,
    q_len_per_req: int = 1,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    row_bytes = _check_packed_fp8_inputs(q, index_k_cache, weights, page_size)
    q_len_per_req = int(q_len_per_req)
    if q_len_per_req < 1 or q.shape[0] % q_len_per_req != 0:
        raise ValueError(
            f"q_len_per_req={q_len_per_req} must divide tokens={q.shape[0]}"
        )
    num_reqs = q.shape[0] // q_len_per_req
    if seq_lens.dim() != 1 or seq_lens.numel() != num_reqs:
        raise ValueError(
            "seq_lens must be [num_reqs], got "
            f"{tuple(seq_lens.shape)} for q={tuple(q.shape)}, "
            f"q_len_per_req={q_len_per_req}"
        )
    if block_table.dim() != 2 or block_table.shape[0] < num_reqs:
        raise ValueError(
            "block_table must have at least one row per request, got "
            f"block_table={tuple(block_table.shape)}, num_reqs={num_reqs}"
        )
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    if q.shape[0] == 0:
        return (
            (
                torch.empty((0, int(topk)), dtype=torch.int32, device=q.device)
                if out is None
                else out
            ),
            (
                torch.empty((0,), dtype=torch.int32, device=q.device)
                if lens_out is None
                else lens_out
            ),
        )
    if not q.is_cuda:
        raise RuntimeError("DSA Triton FP8 decode top-k requires CUDA tensors")

    q = q.contiguous()
    index_k_cache = index_k_cache.contiguous()
    weights = weights.contiguous()
    seq_lens = seq_lens.to(device=q.device, dtype=torch.int32).contiguous()
    block_table = block_table.to(device=q.device, dtype=torch.int32).contiguous()
    max_seq_len = int(block_table.shape[1]) * int(page_size)
    logits = torch.empty(
        (q.shape[0], max_seq_len), dtype=torch.float32, device=q.device
    )
    block_n = 64
    grid = (q.shape[0], triton.cdiv(max_seq_len, block_n))
    _dsa_decode_logits_fp8_kernel[grid](
        q,
        index_k_cache.view(torch.float8_e4m3fn),
        index_k_cache.view(torch.float32),
        weights,
        seq_lens,
        block_table,
        logits,
        block_table.stride(0),
        logits.stride(0),
        page_size=int(page_size),
        row_bytes=row_bytes,
        max_seq_len=max_seq_len,
        num_heads=q.shape[1],
        head_dim=q.shape[2],
        num_groups=q.shape[2] // 128,
        softmax_scale=float(softmax_scale),
        q_len_per_req=q_len_per_req,
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
        q_len_per_req=q_len_per_req,
        out=out,
        lens_out=lens_out,
    )


def dsa_prefill_topk_fp8(
    q: torch.Tensor,
    index_k_cache: torch.Tensor,
    weights: torch.Tensor,
    kv_workspace_slots: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    *,
    page_size: int,
    topk: int,
    softmax_scale: float,
    max_logits_bytes: int | None = None,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    row_bytes = _check_packed_fp8_inputs(q, index_k_cache, weights, page_size)
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
    if out is None:
        out = torch.empty((q.shape[0], int(topk)), dtype=torch.int32, device=q.device)
    if lens_out is None:
        lens_out = torch.empty((q.shape[0],), dtype=torch.int32, device=q.device)
    out.fill_(-1)
    lens_out.zero_()
    if q.shape[0] == 0:
        return out, lens_out
    if not q.is_cuda:
        raise RuntimeError("DSA Triton FP8 prefill top-k requires CUDA tensors")

    q = q.contiguous()
    index_k_cache = index_k_cache.contiguous()
    weights = weights.contiguous()
    kv_workspace_slots = kv_workspace_slots.to(
        device=q.device, dtype=torch.int64
    ).contiguous()
    row_starts = row_starts.to(device=q.device, dtype=torch.int32).contiguous()
    row_ends = row_ends.to(device=q.device, dtype=torch.int32).contiguous()
    seq_len_sum = int(kv_workspace_slots.numel())
    candidate_lens = (row_ends - row_starts).clamp_min(0)
    lens_out.copy_(
        torch.minimum(candidate_lens, torch.full_like(candidate_lens, int(topk)))
    )
    if seq_len_sum == 0:
        return out, lens_out

    if max_logits_bytes is None:
        max_query_rows = q.shape[0]
    else:
        max_query_rows = max(1, int(max_logits_bytes) // (max(seq_len_sum, 1) * 4))
    block_n = 64
    for start in range(0, q.shape[0], max_query_rows):
        end = min(start + max_query_rows, q.shape[0])
        logits = torch.empty(
            (end - start, seq_len_sum), dtype=torch.float32, device=q.device
        )
        grid = (end - start, triton.cdiv(seq_len_sum, block_n))
        _dsa_prefill_logits_fp8_kernel[grid](
            q[start:end],
            index_k_cache.view(torch.float8_e4m3fn),
            index_k_cache.view(torch.float32),
            weights[start:end],
            kv_workspace_slots,
            row_starts[start:end],
            row_ends[start:end],
            logits,
            logits.stride(0),
            seq_len_sum=seq_len_sum,
            page_size=int(page_size),
            row_bytes=row_bytes,
            num_heads=q.shape[1],
            head_dim=q.shape[2],
            num_groups=q.shape[2] // 128,
            softmax_scale=float(softmax_scale),
            BLOCK_N=block_n,
            BLOCK_D=64,
            num_warps=4,
            num_stages=1,
        )
        workspace_indices = _topk_with_padding(logits, int(topk))
        valid = (workspace_indices >= row_starts[start:end, None]) & (
            workspace_indices < row_ends[start:end, None]
        )
        out[start:end].copy_(torch.where(valid, workspace_indices, -1))
    return out, lens_out


@register_kernel(
    "attention",
    "dsa_plan",
    name="triton_dsa_plan",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=frozenset({format_signature()}),
    traits={
        "page_size": frozenset({64}),
    },
    priority=Priority.PORTABLE,
    tags={"portability"},
)
def triton_dsa_plan(
    *,
    page_size: int,
    seq_lens_2d: torch.Tensor,
    out: object | None = None,
) -> torch.Tensor:
    # plan is unused
    return object() if out is None else out


@register_kernel(
    "attention",
    "dsa_decode_topk",
    name="triton_dsa_decode_topk_fp8",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=frozenset(
        {
            format_signature(
                q=dense_tensor_format(torch.bfloat16),
                weights=dense_tensor_format(torch.float32),
            )
        }
    ),
    traits={
        "head_dim": frozenset({128}),
        "topk": frozenset({512, 1024, 2048}),
        "page_size": frozenset({64}),
        "index_k_format": frozenset({"fp8_scaled"}),
    },
    priority=Priority.PORTABLE,
    tags={"portability"},
)
def triton_dsa_decode_topk_fp8(
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
    seq_lens_2d: torch.Tensor | None = None,
    plan: object | None = None,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if index_k_cache is None:
        raise RuntimeError("Triton DSA paged top-k requires packed FP8 index_k_cache")
    return dsa_decode_topk_fp8(
        q,
        index_k_cache,
        weights,
        seq_lens,
        block_table,
        page_size=page_size,
        topk=topk,
        softmax_scale=softmax_scale,
        q_len_per_req=q_len_per_req,
        out=out,
        lens_out=lens_out,
    )


@register_kernel(
    "attention",
    "dsa_prefill_topk",
    name="triton_dsa_prefill_topk_fp8",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=frozenset(
        {
            format_signature(
                q=dense_tensor_format(torch.bfloat16),
                weights=dense_tensor_format(torch.float32),
            )
        }
    ),
    traits={
        "head_dim": frozenset({128}),
        "topk": frozenset({512, 1024, 2048}),
        "index_k_format": frozenset({"fp8_scaled"}),
    },
    priority=Priority.PORTABLE,
    tags={"portability"},
)
def triton_dsa_prefill_topk_fp8(
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
    max_logits_bytes: int | None = None,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if index_k_cache is None or page_size is None:
        raise RuntimeError(
            "Triton DSA top-k requires packed FP8 index_k_cache and page_size"
        )
    return dsa_prefill_topk_fp8(
        q,
        index_k_cache,
        weights,
        kv_workspace_slots,
        row_starts,
        row_ends,
        page_size=page_size,
        topk=topk,
        softmax_scale=softmax_scale,
        max_logits_bytes=max_logits_bytes,
        out=out,
        lens_out=lens_out,
    )


__all__ = [
    "dsa_decode_topk_fp8",
    "dsa_prefill_topk_fp8",
    "local_topk_to_global_slots",
    "triton_dsa_plan",
    "triton_dsa_decode_topk_fp8",
    "triton_dsa_prefill_topk_fp8",
    "workspace_topk_to_global_slots",
]

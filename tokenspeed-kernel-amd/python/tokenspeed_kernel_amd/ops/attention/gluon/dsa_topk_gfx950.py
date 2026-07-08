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

"""Correctness-first DSA top-k Gluon kernels for AMD GFX950."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton

__all__ = [
    "gluon_dsa_decode_topk_fp8_gfx950",
    "gluon_dsa_prefill_topk_fp8_gfx950",
]


@gluon.constexpr_function
def _vector_layout(
    BLOCK: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    LOAD_ELEMS: gl.constexpr,
):
    return gl.BlockedLayout([LOAD_ELEMS], [64], [NUM_WARPS], [0])


@gluon.jit
def _dsa_decode_logits_fp8_kernel(
    q,
    index_k_fp8,
    index_k_scale,
    weights,
    seq_lens,
    block_table,
    logits,
    block_table_stride: gl.constexpr,
    logits_stride: gl.constexpr,
    page_size: gl.constexpr,
    row_bytes: gl.constexpr,
    max_seq_len: gl.constexpr,
    num_heads: gl.constexpr,
    head_dim: gl.constexpr,
    num_groups: gl.constexpr,
    softmax_scale: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    token = gl.program_id(0)
    block_id = gl.program_id(1)
    offsets = block_id * BLOCK_N + gl.arange(
        0, BLOCK_N, layout=_vector_layout(BLOCK_N, gl.num_warps(), 1)
    )
    seq_len = gl.load(seq_lens + token).to(gl.int32)
    valid = (offsets < seq_len) & (offsets < max_seq_len)
    block_idx = offsets // page_size
    block_offset = offsets - block_idx * page_size
    page = gl.load(
        block_table + token * block_table_stride + block_idx,
        mask=valid,
        other=0,
    ).to(gl.int64)
    page_bytes = page_size * row_bytes
    fp8_base = page * page_bytes + block_offset.to(gl.int64) * head_dim
    scale_base = (
        page * (page_bytes // 4)
        + (page_size * head_dim) // 4
        + block_offset.to(gl.int64) * num_groups
    )
    scores = gl.full(
        [BLOCK_N],
        value=0.0,
        dtype=gl.float32,
        layout=_vector_layout(BLOCK_N, gl.num_warps(), 1),
    )

    for head in gl.static_range(0, num_heads):
        head_weight = gl.load(weights + token * num_heads + head).to(gl.float32)
        head_score = gl.full(
            [BLOCK_N],
            value=0.0,
            dtype=gl.float32,
            layout=_vector_layout(BLOCK_N, gl.num_warps(), 1),
        )
        for dim in gl.static_range(0, head_dim):
            q_val = gl.load(q + (token * num_heads + head) * head_dim + dim).to(
                gl.float32
            )
            k_val = gl.load(
                index_k_fp8 + fp8_base + dim,
                mask=valid,
                other=0.0,
            ).to(gl.float32)
            k_scale = gl.load(
                index_k_scale + scale_base + dim // 128,
                mask=valid,
                other=0.0,
            ).to(gl.float32)
            head_score += k_val * k_scale * q_val
        scores += head_score * head_weight

    scores *= softmax_scale
    scores = gl.where(valid, scores, -float("inf"))
    gl.store(
        logits + token * logits_stride + offsets,
        scores,
        mask=offsets < max_seq_len,
    )


@gluon.jit
def _dsa_prefill_logits_fp8_kernel(
    q,
    index_k_fp8,
    index_k_scale,
    weights,
    kv_workspace_slots,
    row_starts,
    row_ends,
    logits,
    logits_stride: gl.constexpr,
    seq_len_sum: gl.constexpr,
    page_size: gl.constexpr,
    row_bytes: gl.constexpr,
    num_heads: gl.constexpr,
    head_dim: gl.constexpr,
    num_groups: gl.constexpr,
    softmax_scale: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    token = gl.program_id(0)
    block_id = gl.program_id(1)
    offsets = block_id * BLOCK_N + gl.arange(
        0, BLOCK_N, layout=_vector_layout(BLOCK_N, gl.num_warps(), 1)
    )
    row_start = gl.load(row_starts + token).to(gl.int32)
    row_end = gl.load(row_ends + token).to(gl.int32)
    valid = (offsets >= row_start) & (offsets < row_end) & (offsets < seq_len_sum)
    slots = gl.load(kv_workspace_slots + offsets, mask=offsets < seq_len_sum, other=0)
    page = slots // page_size
    block_offset = slots - page * page_size
    page_bytes = page_size * row_bytes
    fp8_base = page * page_bytes + block_offset * head_dim
    scale_base = (
        page * (page_bytes // 4)
        + (page_size * head_dim) // 4
        + block_offset * num_groups
    )
    scores = gl.full(
        [BLOCK_N],
        value=0.0,
        dtype=gl.float32,
        layout=_vector_layout(BLOCK_N, gl.num_warps(), 1),
    )

    for head in gl.static_range(0, num_heads):
        head_weight = gl.load(weights + token * num_heads + head).to(gl.float32)
        head_score = gl.full(
            [BLOCK_N],
            value=0.0,
            dtype=gl.float32,
            layout=_vector_layout(BLOCK_N, gl.num_warps(), 1),
        )
        for dim in gl.static_range(0, head_dim):
            q_val = gl.load(q + (token * num_heads + head) * head_dim + dim).to(
                gl.float32
            )
            k_val = gl.load(
                index_k_fp8 + fp8_base + dim,
                mask=valid,
                other=0.0,
            ).to(gl.float32)
            k_scale = gl.load(
                index_k_scale + scale_base + dim // 128,
                mask=valid,
                other=0.0,
            ).to(gl.float32)
            head_score += k_val * k_scale * q_val
        scores += head_score * head_weight

    scores *= softmax_scale
    scores = gl.where(valid, scores, -float("inf"))
    gl.store(
        logits + token * logits_stride + offsets, scores, mask=offsets < seq_len_sum
    )


def _check_packed_fp8_inputs(
    q: torch.Tensor,
    index_k_cache: torch.Tensor,
    weights: torch.Tensor,
    page_size: int,
) -> int:
    if q.dtype != torch.bfloat16:
        raise TypeError(f"DSA Gluon top-k expects BF16 q, got {q.dtype}")
    if weights.dtype != torch.float32:
        raise TypeError(f"DSA Gluon top-k expects FP32 weights, got {weights.dtype}")
    if q.dim() != 3:
        raise ValueError(f"q must be [tokens, heads, dim], got {tuple(q.shape)}")
    if weights.shape != q.shape[:2]:
        raise ValueError(
            f"weights must have shape {tuple(q.shape[:2])}, got {tuple(weights.shape)}"
        )
    if q.shape[2] != 128:
        raise ValueError(f"DSA Gluon top-k supports head_dim=128, got {q.shape[2]}")
    if page_size != 64:
        raise ValueError(f"DSA Gluon top-k supports page_size=64, got {page_size}")
    if index_k_cache.dtype != torch.uint8:
        raise TypeError(
            "DSA Gluon FP8 top-k expects uint8 packed index_k_cache, got "
            f"{index_k_cache.dtype}"
        )
    num_groups = q.shape[2] // 128
    row_bytes = q.shape[2] + num_groups * 4
    if index_k_cache.dim() != 2 or index_k_cache.shape[1] != row_bytes:
        raise ValueError(
            "packed index_k_cache must have shape [slots, row_bytes="
            f"{row_bytes}], got {tuple(index_k_cache.shape)}"
        )
    if index_k_cache.shape[0] % page_size != 0:
        raise ValueError("packed index_k_cache slot count must be page aligned")
    return row_bytes


def _topk_with_padding(logits: torch.Tensor, topk: int) -> torch.Tensor:
    topk = int(topk)
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    out = torch.full(
        (logits.shape[0], topk), -1, device=logits.device, dtype=torch.int32
    )
    count = min(topk, logits.shape[1])
    if count == 0:
        return out
    out[:, :count].copy_(torch.topk(logits, count, dim=1).indices.to(torch.int32))
    return out


def _local_topk_to_global_slots(
    local_topk_offsets: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    page_size: int,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens, topk = local_topk_offsets.shape
    if out is None:
        out = torch.empty_like(local_topk_offsets)
    if lens_out is None:
        lens_out = torch.empty(
            (tokens,), dtype=torch.int32, device=local_topk_offsets.device
        )
    seq_lens = seq_lens.to(device=local_topk_offsets.device, dtype=torch.int32)
    block_table = block_table.to(device=local_topk_offsets.device, dtype=torch.int32)
    lens_out.copy_(torch.minimum(seq_lens, torch.full_like(seq_lens, int(topk))))

    local = local_topk_offsets.to(torch.int64)
    valid = (local >= 0) & (local < seq_lens[:, None].to(torch.int64))
    safe_local = torch.where(valid, local, torch.zeros_like(local))
    block_idx = torch.div(safe_local, int(page_size), rounding_mode="floor")
    block_idx = block_idx.clamp(max=block_table.shape[1] - 1)
    pages = torch.gather(block_table.to(torch.int64), 1, block_idx)
    slots = pages * int(page_size) + safe_local.remainder(int(page_size))
    out.copy_(torch.where(valid, slots.to(torch.int32), torch.full_like(out, -1)))
    return out, lens_out


def _workspace_topk_filter(
    workspace_indices: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    valid = (workspace_indices >= row_starts[:, None]) & (
        workspace_indices < row_ends[:, None]
    )
    out.copy_(torch.where(valid, workspace_indices, torch.full_like(out, -1)))
    return out


def gluon_dsa_decode_topk_fp8_gfx950(
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
    plan: object | None = None,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    del plan, q_len_per_req
    if index_k_cache is None:
        raise RuntimeError("Gluon DSA paged top-k requires packed FP8 index_k_cache")
    row_bytes = _check_packed_fp8_inputs(q, index_k_cache, weights, int(page_size))
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
    if q.shape[0] == 0:
        empty_out = (
            torch.empty((0, int(topk)), dtype=torch.int32, device=q.device)
            if out is None
            else out
        )
        empty_lens = (
            torch.empty((0,), dtype=torch.int32, device=q.device)
            if lens_out is None
            else lens_out
        )
        return empty_out, empty_lens
    if not q.is_cuda:
        raise RuntimeError("DSA Gluon FP8 decode top-k requires CUDA tensors")

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
    _dsa_decode_logits_fp8_kernel[(q.shape[0], triton.cdiv(max_seq_len, block_n))](
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
        BLOCK_N=block_n,
        num_warps=4,
    )
    local_topk_offsets = _topk_with_padding(logits, int(topk))
    return _local_topk_to_global_slots(
        local_topk_offsets,
        block_table,
        seq_lens,
        page_size=int(page_size),
        out=out,
        lens_out=lens_out,
    )


def gluon_dsa_prefill_topk_fp8_gfx950(
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
    del index_k_fp8, index_k_scale
    if index_k_cache is None or page_size is None:
        raise RuntimeError(
            "Gluon DSA top-k requires packed FP8 index_k_cache and page_size"
        )
    row_bytes = _check_packed_fp8_inputs(q, index_k_cache, weights, int(page_size))
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
    if out is None:
        out = torch.empty((q.shape[0], int(topk)), dtype=torch.int32, device=q.device)
    if lens_out is None:
        lens_out = torch.empty((q.shape[0],), dtype=torch.int32, device=q.device)
    out.fill_(-1)
    lens_out.zero_()
    if q.shape[0] == 0:
        return out, lens_out
    if not q.is_cuda:
        raise RuntimeError("DSA Gluon FP8 prefill top-k requires CUDA tensors")

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
        _dsa_prefill_logits_fp8_kernel[
            (end - start, triton.cdiv(seq_len_sum, block_n))
        ](
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
            num_warps=4,
        )
        workspace_indices = _topk_with_padding(logits, int(topk))
        _workspace_topk_filter(
            workspace_indices,
            row_starts[start:end],
            row_ends[start:end],
            out=out[start:end],
        )
    return out, lens_out

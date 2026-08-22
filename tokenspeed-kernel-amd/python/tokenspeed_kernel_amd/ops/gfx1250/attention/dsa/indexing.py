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

"""DSA indexer-logits Gluon kernels for AMD GFX1250."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx1250.attention._common import maximum

__all__ = [
    "_check_packed_fp8_inputs",
    "_dsa_decode_logits_fp8_kernel",
    "_dsa_prefill_logits_fp8_kernel",
]


@gluon.constexpr_function
def _score_layout(
    BLOCK_N: gl.constexpr,
    BLOCK_D: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    return gl.BlockedLayout([1, 8], [4, 8], [NUM_WARPS, 1], [1, 0])


@gluon.jit
def _dsa_decode_logits_fp8_kernel(
    q,
    index_k_fp8,
    index_k_scale,
    weights,
    seq_lens,
    block_table,
    logits,
    q_stride_token,
    q_stride_head,
    q_stride_dim,
    weight_stride_token,
    weight_stride_head,
    block_table_stride: gl.constexpr,
    logits_stride: gl.constexpr,
    page_size: gl.constexpr,
    row_bytes: gl.constexpr,
    page_stride_bytes: gl.constexpr,
    max_seq_len: gl.constexpr,
    num_heads: gl.constexpr,
    head_dim: gl.constexpr,
    num_groups: gl.constexpr,
    softmax_scale: gl.constexpr,
    q_len_per_req: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_D: gl.constexpr,
):
    token = gl.program_id(0)
    block_id = gl.program_id(1)
    layout: gl.constexpr = _score_layout(BLOCK_N, BLOCK_D, gl.num_warps())
    row_layout: gl.constexpr = gl.SliceLayout(1, layout)
    dim_layout: gl.constexpr = gl.SliceLayout(0, layout)
    offsets = block_id * BLOCK_N + gl.arange(0, BLOCK_N, layout=row_layout)
    dims = gl.arange(0, BLOCK_D, layout=dim_layout)

    req = token // q_len_per_req
    q_offset = token - req * q_len_per_req
    seq_len = gl.load(seq_lens + req).to(gl.int32)
    if q_len_per_req != 1:
        seq_len = seq_len - (q_len_per_req - 1) + q_offset
    valid = (offsets < seq_len) & (offsets < max_seq_len)

    block_idx = offsets // page_size
    block_offset = offsets - block_idx * page_size
    page = gl.load(
        block_table + req * block_table_stride + block_idx,
        mask=valid,
        other=0,
    ).to(gl.int64)
    fp8_base = page * page_stride_bytes + block_offset.to(gl.int64) * head_dim
    scale_base = (
        page * (page_stride_bytes // 4)
        + (page_size * head_dim) // 4
        + block_offset.to(gl.int64) * num_groups
    )

    scores = gl.zeros([BLOCK_N], gl.float32, layout=row_layout)
    for head in gl.static_range(0, num_heads):
        head_weight = gl.load(
            weights + token * weight_stride_token + head * weight_stride_head
        ).to(gl.float32)
        q_vals = gl.load(
            q + token * q_stride_token + head * q_stride_head + dims * q_stride_dim,
            mask=dims < head_dim,
            other=0.0,
        ).to(gl.float32)
        k_vals = gl.load(
            index_k_fp8 + fp8_base[:, None] + dims[None, :],
            mask=valid[:, None] & (dims[None, :] < head_dim),
            other=0.0,
        ).to(gl.float32)
        # Per-row FP32 scales follow each page's FP8 payload.
        k_scale = gl.load(
            index_k_scale + scale_base,
            mask=valid,
            other=0.0,
        ).to(gl.float32)
        head_score = gl.sum(k_vals * k_scale[:, None] * q_vals[None, :], axis=1)
        scores += maximum(head_score, 0.0) * head_weight

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
    q_stride_token,
    q_stride_head,
    q_stride_dim,
    weight_stride_token,
    weight_stride_head,
    logits_stride: gl.constexpr,
    seq_len_sum: gl.constexpr,
    page_size: gl.constexpr,
    row_bytes: gl.constexpr,
    page_stride_bytes: gl.constexpr,
    num_heads: gl.constexpr,
    head_dim: gl.constexpr,
    num_groups: gl.constexpr,
    softmax_scale: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_D: gl.constexpr,
):
    token = gl.program_id(0)
    block_id = gl.program_id(1)
    layout: gl.constexpr = _score_layout(BLOCK_N, BLOCK_D, gl.num_warps())
    row_layout: gl.constexpr = gl.SliceLayout(1, layout)
    dim_layout: gl.constexpr = gl.SliceLayout(0, layout)
    offsets = block_id * BLOCK_N + gl.arange(0, BLOCK_N, layout=row_layout)
    dims = gl.arange(0, BLOCK_D, layout=dim_layout)

    row_start = gl.load(row_starts + token).to(gl.int32)
    row_end = gl.load(row_ends + token).to(gl.int32)
    valid = (offsets >= row_start) & (offsets < row_end) & (offsets < seq_len_sum)
    slots = gl.load(
        kv_workspace_slots + offsets,
        mask=offsets < seq_len_sum,
        other=0,
    ).to(gl.int64)
    page = slots // page_size
    block_offset = slots - page * page_size
    fp8_base = page * page_stride_bytes + block_offset * head_dim
    scale_base = (
        page * (page_stride_bytes // 4)
        + (page_size * head_dim) // 4
        + block_offset * num_groups
    )

    scores = gl.zeros([BLOCK_N], gl.float32, layout=row_layout)
    for head in gl.static_range(0, num_heads):
        head_weight = gl.load(
            weights + token * weight_stride_token + head * weight_stride_head
        ).to(gl.float32)
        q_vals = gl.load(
            q + token * q_stride_token + head * q_stride_head + dims * q_stride_dim,
            mask=dims < head_dim,
            other=0.0,
        ).to(gl.float32)
        k_vals = gl.load(
            index_k_fp8 + fp8_base[:, None] + dims[None, :],
            mask=valid[:, None] & (dims[None, :] < head_dim),
            other=0.0,
        ).to(gl.float32)
        k_scale = gl.load(
            index_k_scale + scale_base,
            mask=valid,
            other=0.0,
        ).to(gl.float32)
        head_score = gl.sum(k_vals * k_scale[:, None] * q_vals[None, :], axis=1)
        scores += maximum(head_score, 0.0) * head_weight

    scores *= softmax_scale
    scores = gl.where(valid, scores, -float("inf"))
    gl.store(
        logits + token * logits_stride + offsets,
        scores,
        mask=offsets < seq_len_sum,
    )


def _check_packed_fp8_inputs(
    q: torch.Tensor,
    index_k_cache: torch.Tensor,
    weights: torch.Tensor,
    page_size: int,
) -> tuple[int, int]:
    if q.dtype != torch.bfloat16:
        raise TypeError(f"DSA Gluon top-k expects BF16 q, got {q.dtype}")
    if weights.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(
            f"DSA Gluon top-k expects BF16 or FP32 weights, got {weights.dtype}"
        )
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
    if index_k_cache.dim() != 2:
        raise ValueError(
            "index_k_cache must be a packed slot matrix or page-planar matrix, "
            f"got shape {tuple(index_k_cache.shape)}"
        )
    page_bytes = page_size * row_bytes
    if index_k_cache.shape[1] == row_bytes:
        if not index_k_cache.is_contiguous():
            raise ValueError("packed index_k_cache must be contiguous")
        if index_k_cache.shape[0] % page_size != 0:
            raise ValueError("packed index_k_cache slot count must be page aligned")
        page_stride_bytes = page_bytes
    elif (
        index_k_cache.shape[1] >= page_bytes
        and index_k_cache.stride(1) == 1
        and index_k_cache.stride(0) >= page_bytes
    ):
        page_stride_bytes = index_k_cache.stride(0)
    else:
        raise ValueError(
            "index_k_cache must be contiguous [slots, row_bytes] or page-planar "
            f"[pages, at least {page_bytes} bytes], got "
            f"shape={tuple(index_k_cache.shape)}, stride={index_k_cache.stride()}"
        )
    if index_k_cache.storage_offset() % 4 or page_stride_bytes % 4:
        raise ValueError("index_k_cache page storage must be float32 aligned")
    return row_bytes, page_stride_bytes

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

"""DSA TopK scoring Gluon kernels for AMD GFX950."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon

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
    return gl.BlockedLayout([1, 8], [8, 8], [NUM_WARPS, 1], [1, 0])


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
    dim_offsets = gl.arange(0, BLOCK_D, layout=dim_layout)
    req = token // q_len_per_req
    q_offset = token - req * q_len_per_req
    seq_len = gl.load(seq_lens + req).to(gl.int32)
    if q_len_per_req != 1:
        seq_len = seq_len - (q_len_per_req - 1) + q_offset
    valid = (offsets < seq_len) & (offsets < max_seq_len)
    block_idx = offsets // page_size
    block_offset = offsets - block_idx * page_size
    page = gl.amd.cdna4.buffer_load(
        ptr=block_table,
        offsets=(req * block_table_stride + block_idx).to(gl.int32),
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
        layout=row_layout,
    )

    for head in gl.static_range(0, num_heads):
        head_weight = gl.load(weights + token * num_heads + head).to(gl.float32)
        head_score = gl.full(
            [BLOCK_N],
            value=0.0,
            dtype=gl.float32,
            layout=row_layout,
        )
        for dim_start in gl.static_range(0, head_dim, BLOCK_D):
            dims = dim_start + dim_offsets
            q_vals = gl.amd.cdna4.buffer_load(
                ptr=q,
                offsets=((token * num_heads + head) * head_dim + dims).to(gl.int32),
                mask=dims < head_dim,
                other=0.0,
            ).to(gl.float32)
            k_vals = gl.amd.cdna4.buffer_load(
                ptr=index_k_fp8,
                offsets=(fp8_base[:, None] + dims[None, :]).to(gl.int32),
                mask=valid[:, None] & (dims[None, :] < head_dim),
                other=0.0,
            ).to(gl.float32)
            k_scale = gl.amd.cdna4.buffer_load(
                ptr=index_k_scale,
                offsets=(scale_base + dim_start // 128).to(gl.int32),
                mask=valid,
                other=0.0,
            ).to(gl.float32)
            head_score += gl.sum(k_vals * k_scale[:, None] * q_vals[None, :], axis=1)
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
    BLOCK_D: gl.constexpr,
):
    token = gl.program_id(0)
    block_id = gl.program_id(1)
    layout: gl.constexpr = _score_layout(BLOCK_N, BLOCK_D, gl.num_warps())
    row_layout: gl.constexpr = gl.SliceLayout(1, layout)
    dim_layout: gl.constexpr = gl.SliceLayout(0, layout)
    offsets = block_id * BLOCK_N + gl.arange(0, BLOCK_N, layout=row_layout)
    dim_offsets = gl.arange(0, BLOCK_D, layout=dim_layout)
    row_start = gl.load(row_starts + token).to(gl.int32)
    row_end = gl.load(row_ends + token).to(gl.int32)
    valid = (offsets >= row_start) & (offsets < row_end) & (offsets < seq_len_sum)
    slots = gl.amd.cdna4.buffer_load(
        ptr=kv_workspace_slots,
        offsets=offsets.to(gl.int32),
        mask=offsets < seq_len_sum,
        other=0,
    )
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
        layout=row_layout,
    )

    for head in gl.static_range(0, num_heads):
        head_weight = gl.load(weights + token * num_heads + head).to(gl.float32)
        head_score = gl.full(
            [BLOCK_N],
            value=0.0,
            dtype=gl.float32,
            layout=row_layout,
        )
        for dim_start in gl.static_range(0, head_dim, BLOCK_D):
            dims = dim_start + dim_offsets
            q_vals = gl.amd.cdna4.buffer_load(
                ptr=q,
                offsets=((token * num_heads + head) * head_dim + dims).to(gl.int32),
                mask=dims < head_dim,
                other=0.0,
            ).to(gl.float32)
            k_vals = gl.amd.cdna4.buffer_load(
                ptr=index_k_fp8,
                offsets=(fp8_base[:, None] + dims[None, :]).to(gl.int32),
                mask=valid[:, None] & (dims[None, :] < head_dim),
                other=0.0,
            ).to(gl.float32)
            k_scale = gl.amd.cdna4.buffer_load(
                ptr=index_k_scale,
                offsets=(scale_base + dim_start // 128).to(gl.int32),
                mask=valid,
                other=0.0,
            ).to(gl.float32)
            head_score += gl.sum(k_vals * k_scale[:, None] * q_vals[None, :], axis=1)
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

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


@triton.jit
def _dsa_sparse_decode_kernel(
    q,
    kv_fp8,
    kv_scale,
    kv_rope,
    topk_indices,
    topk_lens,
    out,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    kv_lora_rank: tl.constexpr,
    qk_rope_head_dim: tl.constexpr,
    row_bytes: tl.constexpr,
    topk: tl.constexpr,
    softmax_scale: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    v_block = tl.program_id(2)

    topk_offsets = tl.arange(0, BLOCK_TOPK)
    k_offsets = tl.arange(0, BLOCK_K)
    rope_offsets = tl.arange(0, 64)
    v_offsets = v_block * BLOCK_V + tl.arange(0, BLOCK_V)

    q_base = (token * num_heads + head) * head_dim
    q_nope_base = q_base
    q_rope_base = q_base + kv_lora_rank

    q_rope = tl.load(
        q + q_rope_base + rope_offsets,
        mask=rope_offsets < qk_rope_head_dim,
        other=0.0,
    ).to(tl.float32)

    valid_len = tl.load(topk_lens + token).to(tl.int32)
    max_score = tl.full((), -float("inf"), tl.float32)

    for start in range(0, topk, BLOCK_TOPK):
        cols = start + topk_offsets
        valid = cols < valid_len
        slots = tl.load(
            topk_indices + token * topk + cols,
            mask=valid,
            other=0,
        ).to(tl.int64)
        valid = valid & (slots >= 0)
        score = tl.zeros((BLOCK_TOPK,), tl.float32)

        for k_start in range(0, kv_lora_rank, BLOCK_K):
            ks = k_start + k_offsets
            q_vals = tl.load(q + q_nope_base + ks).to(tl.float32)
            k_vals = tl.load(
                kv_fp8 + slots[:, None] * row_bytes + ks[None, :],
                mask=valid[:, None],
                other=0.0,
            ).to(tl.float32)
            k_scale = tl.load(
                kv_scale
                + (slots * row_bytes + kv_lora_rank + (k_start // 128) * 4) // 4,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            score += tl.sum(k_vals * k_scale[:, None] * q_vals[None, :], axis=1)

        k_rope = tl.load(
            kv_rope
            + (slots[:, None] * row_bytes + kv_lora_rank + (kv_lora_rank // 128) * 4)
            // 2
            + rope_offsets[None, :],
            mask=valid[:, None] & (rope_offsets[None, :] < qk_rope_head_dim),
            other=0.0,
        ).to(tl.float32)
        score += tl.sum(k_rope * q_rope[None, :], axis=1)
        score *= softmax_scale
        score = tl.where(valid, score, -float("inf"))
        max_score = tl.maximum(max_score, tl.max(score, axis=0))

    denom = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_V,), tl.float32)
    v_mask = v_offsets < kv_lora_rank
    for start in range(0, topk, BLOCK_TOPK):
        cols = start + topk_offsets
        valid = cols < valid_len
        slots = tl.load(
            topk_indices + token * topk + cols,
            mask=valid,
            other=0,
        ).to(tl.int64)
        valid = valid & (slots >= 0)
        score = tl.zeros((BLOCK_TOPK,), tl.float32)

        for k_start in range(0, kv_lora_rank, BLOCK_K):
            ks = k_start + k_offsets
            q_vals = tl.load(q + q_nope_base + ks).to(tl.float32)
            k_vals = tl.load(
                kv_fp8 + slots[:, None] * row_bytes + ks[None, :],
                mask=valid[:, None],
                other=0.0,
            ).to(tl.float32)
            k_scale = tl.load(
                kv_scale
                + (slots * row_bytes + kv_lora_rank + (k_start // 128) * 4) // 4,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            score += tl.sum(k_vals * k_scale[:, None] * q_vals[None, :], axis=1)

        k_rope = tl.load(
            kv_rope
            + (slots[:, None] * row_bytes + kv_lora_rank + (kv_lora_rank // 128) * 4)
            // 2
            + rope_offsets[None, :],
            mask=valid[:, None] & (rope_offsets[None, :] < qk_rope_head_dim),
            other=0.0,
        ).to(tl.float32)
        score += tl.sum(k_rope * q_rope[None, :], axis=1)
        score *= softmax_scale
        score = tl.where(valid, score, -float("inf"))
        probs = tl.exp(score - max_score)
        probs = tl.where(valid, probs, 0.0)
        denom += tl.sum(probs, axis=0)

        v_vals = tl.load(
            kv_fp8 + slots[:, None] * row_bytes + v_offsets[None, :],
            mask=valid[:, None] & v_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        v_scale = tl.load(
            kv_scale
            + (
                slots[:, None] * row_bytes
                + kv_lora_rank
                + (v_offsets[None, :] // 128) * 4
            )
            // 4,
            mask=valid[:, None] & v_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(probs[:, None] * v_vals * v_scale, axis=0)

    result = acc / denom
    result = tl.where(denom > 0.0, result, 0.0)
    out_base = (token * num_heads + head) * kv_lora_rank
    tl.store(out + out_base + v_offsets, result, mask=v_mask)


def dsa_sparse_decode(
    q: torch.Tensor,
    sparse_kv: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_lens: torch.Tensor,
    *,
    softmax_scale: float,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> torch.Tensor:
    if q.dtype != torch.bfloat16:
        raise TypeError(f"dsa_sparse_decode expects BF16 q, got {q.dtype}")
    if sparse_kv.dtype != torch.uint8:
        raise TypeError(
            f"dsa_sparse_decode expects uint8 sparse_kv, got {sparse_kv.dtype}"
        )
    if topk_indices.dtype != torch.int32:
        raise TypeError(
            f"dsa_sparse_decode expects int32 topk_indices, got {topk_indices.dtype}"
        )
    if topk_lens.dtype != torch.int32:
        raise TypeError(
            f"dsa_sparse_decode expects int32 topk_lens, got {topk_lens.dtype}"
        )
    if q.dim() != 3:
        raise ValueError(
            f"dsa_sparse_decode expects q [tokens, heads, dim], got {q.shape}"
        )
    if topk_indices.dim() != 2 or topk_indices.shape[0] != q.shape[0]:
        raise ValueError(
            "dsa_sparse_decode top-k shape mismatch: "
            f"q={tuple(q.shape)}, topk={tuple(topk_indices.shape)}"
        )
    if topk_lens.shape != (q.shape[0],):
        raise ValueError(
            "dsa_sparse_decode top-k lens shape mismatch: "
            f"expected {(q.shape[0],)}, got {tuple(topk_lens.shape)}"
        )
    row_bytes = int(sparse_kv.shape[1])
    expected_row_bytes = kv_lora_rank + kv_lora_rank // 128 * 4 + qk_rope_head_dim * 2
    if row_bytes != expected_row_bytes:
        raise ValueError(
            "dsa_sparse_decode sparse KV row size mismatch: "
            f"got {row_bytes}, expected {expected_row_bytes}"
        )
    if q.shape[-1] != kv_lora_rank + qk_rope_head_dim:
        raise ValueError(
            "dsa_sparse_decode q dim mismatch: "
            f"got {q.shape[-1]}, expected {kv_lora_rank + qk_rope_head_dim}"
        )

    q = q.contiguous()
    topk_indices = topk_indices.contiguous()
    topk_lens = topk_lens.contiguous()
    out = torch.empty(
        (q.shape[0], q.shape[1], kv_lora_rank),
        dtype=q.dtype,
        device=q.device,
    )
    grid = (q.shape[0], q.shape[1], triton.cdiv(kv_lora_rank, 64))
    _dsa_sparse_decode_kernel[grid](
        q,
        sparse_kv.view(torch.float8_e4m3fn),
        sparse_kv.view(torch.float32),
        sparse_kv.view(torch.bfloat16),
        topk_indices,
        topk_lens,
        out,
        q.shape[1],
        q.shape[2],
        kv_lora_rank,
        qk_rope_head_dim,
        row_bytes,
        topk_indices.shape[1],
        float(softmax_scale),
        BLOCK_TOPK=32,
        BLOCK_K=64,
        BLOCK_V=64,
        num_warps=4,
        num_stages=1,
    )
    return out


__all__ = ["dsa_sparse_decode"]

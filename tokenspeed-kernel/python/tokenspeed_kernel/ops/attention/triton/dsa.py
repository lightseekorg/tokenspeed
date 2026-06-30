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

_DSA_QUERY_DTYPES = frozenset({torch.bfloat16, torch.float8_e4m3fn})


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


@triton.jit
def _dsa_dense_kv_decode_kernel(
    q,
    kv,
    topk_indices,
    topk_lens,
    out,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    kv_lora_rank: tl.constexpr,
    qk_rope_head_dim: tl.constexpr,
    kv_dim: tl.constexpr,
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
                kv + slots[:, None] * kv_dim + ks[None, :],
                mask=valid[:, None],
                other=0.0,
            ).to(tl.float32)
            score += tl.sum(k_vals * q_vals[None, :], axis=1)

        k_rope = tl.load(
            kv + slots[:, None] * kv_dim + kv_lora_rank + rope_offsets[None, :],
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
                kv + slots[:, None] * kv_dim + ks[None, :],
                mask=valid[:, None],
                other=0.0,
            ).to(tl.float32)
            score += tl.sum(k_vals * q_vals[None, :], axis=1)

        k_rope = tl.load(
            kv + slots[:, None] * kv_dim + kv_lora_rank + rope_offsets[None, :],
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
            kv + slots[:, None] * kv_dim + v_offsets[None, :],
            mask=valid[:, None] & v_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(probs[:, None] * v_vals, axis=0)

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
    if q.dtype not in _DSA_QUERY_DTYPES:
        raise TypeError(f"dsa_sparse_decode expects BF16 or FP8 q, got {q.dtype}")
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
        dtype=torch.bfloat16 if q.dtype == torch.float8_e4m3fn else q.dtype,
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


def dsa_dense_kv_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_lens: torch.Tensor,
    *,
    softmax_scale: float,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> torch.Tensor:
    if q.dtype not in _DSA_QUERY_DTYPES:
        raise TypeError(f"dsa_dense_kv_decode expects BF16 or FP8 q, got {q.dtype}")
    if kv_cache.dtype not in _DSA_QUERY_DTYPES:
        raise TypeError(
            f"dsa_dense_kv_decode expects BF16 or FP8 kv_cache, got {kv_cache.dtype}"
        )
    if topk_indices.dtype != torch.int32:
        raise TypeError(
            f"dsa_dense_kv_decode expects int32 topk_indices, got {topk_indices.dtype}"
        )
    if topk_lens.dtype != torch.int32:
        raise TypeError(
            f"dsa_dense_kv_decode expects int32 topk_lens, got {topk_lens.dtype}"
        )
    if q.dim() != 3:
        raise ValueError(
            f"dsa_dense_kv_decode expects q [tokens, heads, dim], got {q.shape}"
        )
    if topk_indices.dim() != 2 or topk_indices.shape[0] != q.shape[0]:
        raise ValueError(
            "dsa_dense_kv_decode top-k shape mismatch: "
            f"q={tuple(q.shape)}, topk={tuple(topk_indices.shape)}"
        )
    if topk_lens.shape != (q.shape[0],):
        raise ValueError(
            "dsa_dense_kv_decode top-k lens shape mismatch: "
            f"expected {(q.shape[0],)}, got {tuple(topk_lens.shape)}"
        )
    kv_dim = int(kv_lora_rank) + int(qk_rope_head_dim)
    if kv_cache.dim() != 2 or kv_cache.shape[1] != kv_dim:
        raise ValueError(
            "dsa_dense_kv_decode kv_cache must be [slots, kv_dim], got "
            f"{tuple(kv_cache.shape)}, expected kv_dim={kv_dim}"
        )
    if q.shape[-1] != kv_dim:
        raise ValueError(
            "dsa_dense_kv_decode q dim mismatch: "
            f"got {q.shape[-1]}, expected {kv_dim}"
        )

    q = q.contiguous()
    kv_cache = kv_cache.contiguous()
    topk_indices = topk_indices.contiguous()
    topk_lens = topk_lens.contiguous()
    out = torch.empty(
        (q.shape[0], q.shape[1], kv_lora_rank),
        dtype=torch.bfloat16 if q.dtype == torch.float8_e4m3fn else q.dtype,
        device=q.device,
    )
    grid = (q.shape[0], q.shape[1], triton.cdiv(kv_lora_rank, 64))
    _dsa_dense_kv_decode_kernel[grid](
        q,
        kv_cache,
        topk_indices,
        topk_lens,
        out,
        q.shape[1],
        q.shape[2],
        kv_lora_rank,
        qk_rope_head_dim,
        kv_dim,
        topk_indices.shape[1],
        float(softmax_scale),
        BLOCK_TOPK=32,
        BLOCK_K=64,
        BLOCK_V=64,
        num_warps=4,
        num_stages=1,
    )
    return out


def _flatten_sparse_kv_cache(
    sparse_kv_cache: torch.Tensor, page_size: int
) -> torch.Tensor:
    if sparse_kv_cache.dim() == 2:
        return sparse_kv_cache
    if sparse_kv_cache.dim() == 4:
        if sparse_kv_cache.shape[1] != int(page_size) or sparse_kv_cache.shape[2] != 1:
            raise ValueError(
                "paged sparse_kv_cache must be [pages, page_size, 1, row_bytes], "
                f"got {tuple(sparse_kv_cache.shape)} with page_size={page_size}"
            )
        return sparse_kv_cache.reshape(-1, sparse_kv_cache.shape[-1])
    raise ValueError(
        "sparse_kv_cache must be [slots, row_bytes] or "
        f"[pages, page_size, 1, row_bytes], got {tuple(sparse_kv_cache.shape)}"
    )


def _flatten_dense_kv_cache(kv_cache: torch.Tensor, page_size: int) -> torch.Tensor:
    if kv_cache.dim() == 2:
        return kv_cache
    if kv_cache.dim() == 3 and kv_cache.shape[1] == 1:
        return kv_cache.squeeze(1)
    if kv_cache.dim() == 4:
        if kv_cache.shape[1] == int(page_size) and kv_cache.shape[2] == 1:
            return kv_cache.reshape(-1, kv_cache.shape[-1])
        if kv_cache.shape[1] == 1 and kv_cache.shape[2] == int(page_size):
            return kv_cache.permute(0, 2, 1, 3).reshape(-1, kv_cache.shape[-1])
    raise ValueError(
        "kv_cache must be [slots, dim], [slots, 1, dim], or paged MLA cache, "
        f"got {tuple(kv_cache.shape)}"
    )


def _flatten_sparse_query(q: torch.Tensor, q_len_per_req: int) -> torch.Tensor:
    if q.dim() == 3:
        return q
    if q.dim() == 4:
        return q.reshape(-1, q.shape[2], q.shape[3])
    raise ValueError(
        "sparse MLA q must be [tokens, heads, dim] or "
        f"[batch, q_len, heads, dim], got {tuple(q.shape)}"
    )


def _copy_or_return_sparse_out(
    result: torch.Tensor,
    out: torch.Tensor | None,
) -> torch.Tensor:
    if out is None:
        return result
    out_view = out.reshape_as(result)
    out_view.copy_(result)
    return out


@register_kernel(
    "attention",
    "dsa_decode",
    name="triton_dsa_decode",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=frozenset(
        {format_signature(q=dense_tensor_format(dtype)) for dtype in _DSA_QUERY_DTYPES}
    ),
    traits={
        "page_size": frozenset({64}),
        "q_len_per_req": frozenset({1, 2, 3, 4, 5, 6}),
        "qk_nope_head_dim": frozenset({128, 192}),
        "kv_lora_rank": frozenset({128, 512}),
        "qk_rope_head_dim": frozenset({64}),
        "topk": frozenset({512, 1024, 2048}),
        "kv_cache_available": frozenset({False, True}),
        "sparse_kv_cache_available": frozenset({False, True}),
        "topk_layout": frozenset({"global_slots"}),
        "support_logit_cap": frozenset({False}),
        "return_lse": frozenset({False}),
    },
    priority=Priority.PORTABLE,
    tags={"portability"},
)
def triton_dsa_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor | None,
    sparse_kv_cache: torch.Tensor | None,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor | None,
    max_seqlen_k: int,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
    page_size: int,
    q_len_per_req: int = 1,
    logit_cap: float = 0.0,
    k_scale: float = 1.0,
    return_lse: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if sparse_kv_cache is None and kv_cache is None:
        raise RuntimeError("Triton sparse MLA requires kv_cache or sparse_kv_cache")
    if topk_lens is None:
        raise RuntimeError("Triton sparse MLA requires topk_lens")
    if return_lse:
        raise RuntimeError("Triton sparse MLA does not support return_lse")
    if logit_cap != 0.0:
        raise RuntimeError("Triton sparse MLA does not support logit_cap")
    q_flat = _flatten_sparse_query(q, q_len_per_req).contiguous()
    if sparse_kv_cache is not None:
        sparse_kv = _flatten_sparse_kv_cache(sparse_kv_cache, page_size).contiguous()
        result = dsa_sparse_decode(
            q_flat,
            sparse_kv,
            topk_slots.contiguous(),
            topk_lens.contiguous(),
            softmax_scale=softmax_scale * float(k_scale),
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
        )
    else:
        dense_kv = _flatten_dense_kv_cache(kv_cache, page_size).contiguous()
        result = dsa_dense_kv_decode(
            q_flat,
            dense_kv,
            topk_slots.contiguous(),
            topk_lens.contiguous(),
            softmax_scale=softmax_scale * float(k_scale),
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
        )
    return _copy_or_return_sparse_out(result, out)


@register_kernel(
    "attention",
    "dsa_prefill",
    name="triton_dsa_prefill",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=frozenset(
        {format_signature(q=dense_tensor_format(dtype)) for dtype in _DSA_QUERY_DTYPES}
    ),
    traits={
        "page_size": frozenset({64}),
        "q_len_per_req": frozenset({1}),
        "qk_nope_head_dim": frozenset({128, 192}),
        "kv_lora_rank": frozenset({128, 512}),
        "qk_rope_head_dim": frozenset({64}),
        "topk": frozenset({512, 1024, 2048}),
        "kv_cache_available": frozenset({False, True}),
        "sparse_kv_cache_available": frozenset({False, True}),
        "topk_layout": frozenset({"global_slots"}),
        "support_logit_cap": frozenset({False}),
        "return_lse": frozenset({False}),
    },
    priority=Priority.PORTABLE,
    tags={"portability"},
)
def triton_dsa_prefill(
    q: torch.Tensor,
    kv_cache: torch.Tensor | None,
    sparse_kv_cache: torch.Tensor | None,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor | None,
    max_seqlen_k: int,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
    page_size: int,
    q_len_per_req: int = 1,
    logit_cap: float = 0.0,
    k_scale: float = 1.0,
    return_lse: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    return triton_dsa_decode(
        q=q,
        kv_cache=kv_cache,
        sparse_kv_cache=sparse_kv_cache,
        topk_slots=topk_slots,
        topk_lens=topk_lens,
        max_seqlen_k=max_seqlen_k,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        softmax_scale=softmax_scale,
        page_size=page_size,
        q_len_per_req=q_len_per_req,
        logit_cap=logit_cap,
        k_scale=k_scale,
        return_lse=return_lse,
        out=out,
    )

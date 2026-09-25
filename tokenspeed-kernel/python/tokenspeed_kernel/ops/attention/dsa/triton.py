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


@triton.jit
def _dsa_packed_kv_kernel(
    q,
    kv_fp8,
    kv_scale,
    kv_rope,
    topk_indices,
    topk_lens,
    out,
    lse,
    RETURN_LSE: tl.constexpr,
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
    if RETURN_LSE:
        if v_block == 0:
            tl.store(
                lse + token * num_heads + head,
                tl.where(denom > 0.0, max_score + tl.log(denom), -float("inf")),
            )


@triton.jit
def _dsa_dense_kv_kernel(
    q,
    kv,
    topk_indices,
    topk_lens,
    out,
    lse,
    RETURN_LSE: tl.constexpr,
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
    if RETURN_LSE:
        if v_block == 0:
            tl.store(
                lse + token * num_heads + head,
                tl.where(denom > 0.0, max_score + tl.log(denom), -float("inf")),
            )


def _run_packed_kv(
    q: torch.Tensor,
    packed_kv: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_lens: torch.Tensor,
    *,
    softmax_scale: float,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    return_lse: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    row_bytes = int(packed_kv.shape[1])
    out = torch.empty(
        (q.shape[0], q.shape[1], kv_lora_rank),
        dtype=torch.bfloat16 if q.dtype == torch.float8_e4m3fn else q.dtype,
        device=q.device,
    )
    lse = (
        torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
        if return_lse
        else None
    )
    grid = (q.shape[0], q.shape[1], triton.cdiv(kv_lora_rank, 64))
    _dsa_packed_kv_kernel[grid](
        q,
        packed_kv.view(torch.float8_e4m3fn),
        packed_kv.view(torch.float32),
        packed_kv.view(torch.bfloat16),
        topk_indices,
        topk_lens,
        out,
        lse,
        return_lse,
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
    return out, lse


def _run_dense_kv(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_lens: torch.Tensor,
    *,
    softmax_scale: float,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    return_lse: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    kv_dim = int(kv_lora_rank) + int(qk_rope_head_dim)
    out = torch.empty(
        (q.shape[0], q.shape[1], kv_lora_rank),
        dtype=torch.bfloat16 if q.dtype == torch.float8_e4m3fn else q.dtype,
        device=q.device,
    )
    lse = (
        torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
        if return_lse
        else None
    )
    grid = (q.shape[0], q.shape[1], triton.cdiv(kv_lora_rank, 64))
    _dsa_dense_kv_kernel[grid](
        q,
        kv_cache,
        topk_indices,
        topk_lens,
        out,
        lse,
        return_lse,
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
    return out, lse


def _flatten_packed_kv_cache(packed_kv_cache: torch.Tensor) -> torch.Tensor:
    if packed_kv_cache.dim() == 2:
        return packed_kv_cache
    return packed_kv_cache.reshape(-1, packed_kv_cache.shape[-1])


def _flatten_dense_kv_cache(kv_cache: torch.Tensor) -> torch.Tensor:
    if kv_cache.dim() == 2:
        return kv_cache
    if kv_cache.dim() == 3:
        return kv_cache.squeeze(1)
    if kv_cache.shape[1] == 1:
        kv_cache = kv_cache.permute(0, 2, 1, 3)
    return kv_cache.reshape(-1, kv_cache.shape[-1])


def _flatten_query(q: torch.Tensor) -> torch.Tensor:
    if q.dim() == 3:
        return q
    return q.reshape(-1, q.shape[-2], q.shape[-1])


def _run_dsa(
    *,
    q: torch.Tensor,
    kv_cache: torch.Tensor | None,
    packed_kv_cache: torch.Tensor | None,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
    k_scale: float,
    out: torch.Tensor | None,
    return_lse: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    q = _flatten_query(q).contiguous()
    topk_slots = topk_slots.contiguous()
    topk_lens = (
        torch.full(
            (q.shape[0],), topk_slots.shape[-1], dtype=torch.int32, device=q.device
        )
        if topk_lens is None
        else topk_lens.contiguous()
    )
    softmax_scale = float(softmax_scale) * float(k_scale)

    if packed_kv_cache is not None:
        result, lse = _run_packed_kv(
            q,
            _flatten_packed_kv_cache(packed_kv_cache).contiguous(),
            topk_slots,
            topk_lens,
            softmax_scale=softmax_scale,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            return_lse=return_lse,
        )
    else:
        result, lse = _run_dense_kv(
            q,
            _flatten_dense_kv_cache(kv_cache).contiguous(),
            topk_slots,
            topk_lens,
            softmax_scale=softmax_scale,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            return_lse=return_lse,
        )

    if out is not None:
        out.reshape_as(result).copy_(result)
        result = out
    return (result, lse) if return_lse else result


@register_kernel(
    "attention",
    "dsa_decode",
    name="triton_dsa_decode",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=frozenset(
        {
            format_signature(q=dense_tensor_format(torch.bfloat16)),
            format_signature(q=dense_tensor_format(torch.float8_e4m3fn)),
        }
    ),
    traits={
        "q_len": frozenset({1, 2, 3, 4, 5, 6}),
        "qk_nope_head_dim": frozenset({128, 192, 256}),
        "kv_lora_rank": frozenset({128, 512}),
        "qk_rope_head_dim": frozenset({0, 64}),
        "page_size": frozenset({64}),
        "topk": frozenset({512, 1024, 2048, 2049, 2050, 2051}),
        "has_kv_cache": frozenset({False, True}),
        "has_sparse_kv_cache": frozenset({False, True}),
        "topk_layout": frozenset({"global_slots"}),
        "logit_cap": frozenset({False}),
        "return_lse": frozenset({False, True}),
    },
    priority=Priority.PORTABLE,
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
    kv_seq_lens: torch.Tensor | None = None,
    logit_cap: float = 0.0,
    k_scale: float = 1.0,
    return_lse: bool = False,
    out: torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    del kv_seq_lens
    return _run_dsa(
        q=q,
        kv_cache=kv_cache,
        packed_kv_cache=sparse_kv_cache,
        topk_slots=topk_slots,
        topk_lens=topk_lens,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        softmax_scale=softmax_scale,
        k_scale=k_scale,
        out=out,
        return_lse=return_lse,
    )


@register_kernel(
    "attention",
    "dsa_prefill",
    name="triton_dsa_prefill",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=frozenset(
        {
            format_signature(q=dense_tensor_format(torch.bfloat16)),
            format_signature(q=dense_tensor_format(torch.float8_e4m3fn)),
        }
    ),
    traits={
        "q_len": frozenset({1}),
        "qk_nope_head_dim": frozenset({128, 192, 256}),
        "kv_lora_rank": frozenset({128, 512}),
        "qk_rope_head_dim": frozenset({0, 64}),
        "page_size": frozenset({64}),
        "topk": frozenset({512, 1024, 2048, 2049, 2050, 2051}),
        "has_kv_cache": frozenset({False, True}),
        "has_sparse_kv_cache": frozenset({False, True}),
        "topk_layout": frozenset({"global_slots"}),
        "logit_cap": frozenset({False}),
        "return_lse": frozenset({False, True}),
    },
    priority=Priority.PORTABLE,
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
    kv_seq_lens: torch.Tensor | None = None,
    logit_cap: float = 0.0,
    k_scale: float = 1.0,
    return_lse: bool = False,
    out: torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    del kv_seq_lens
    return _run_dsa(
        q=q,
        kv_cache=kv_cache,
        packed_kv_cache=sparse_kv_cache,
        topk_slots=topk_slots,
        topk_lens=topk_lens,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        softmax_scale=softmax_scale,
        k_scale=k_scale,
        out=out,
        return_lse=return_lse,
    )


from tokenspeed_kernel.ops.attention.dsa._triton.topk import *  # noqa: E402,F403
from tokenspeed_kernel.ops.attention.dsa._triton.topk import (  # noqa: E402
    _topk_with_padding,
    _triton_dsa_decode_topk_fp8_impl,
    _triton_dsa_plan_impl,
    _triton_dsa_prefill_topk_fp8_impl,
)


@register_kernel(
    "attention",
    "dsa_plan",
    name="triton_dsa_plan",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=frozenset({format_signature()}),
    traits={"page_size": frozenset({64})},
    priority=Priority.PORTABLE,
)
def triton_dsa_plan(
    *,
    page_size: int,
    seq_lens_2d: torch.Tensor,
    out: object | None = None,
) -> torch.Tensor:
    return _triton_dsa_plan_impl(
        page_size=page_size,
        seq_lens_2d=seq_lens_2d,
        out=out,
    )


_TOPK_SIGNATURES = frozenset(
    {
        format_signature(
            q=dense_tensor_format(torch.bfloat16),
            weights=dense_tensor_format(torch.float32),
        ),
        format_signature(
            q=dense_tensor_format(torch.bfloat16),
            weights=dense_tensor_format(torch.bfloat16),
        ),
    }
)


@register_kernel(
    "attention",
    "dsa_decode_topk",
    name="triton_dsa_decode_topk_fp8",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=_TOPK_SIGNATURES,
    traits={
        "head_dim": frozenset({128}),
        "page_size": frozenset({64}),
        "topk": frozenset({512, 1024, 2048}),
        "index_k_format": frozenset({"fp8_scaled"}),
        "index_k_layout": frozenset({"packed", "page_planar"}),
    },
    features={"logical_offsets"},
    priority=Priority.PORTABLE,
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
    topk_layout: str = "global_slots",
    block_table_base_offsets: torch.Tensor | None = None,
    index_k_cache: torch.Tensor | None = None,
    seq_lens_2d: torch.Tensor | None = None,
    plan: object | None = None,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _triton_dsa_decode_topk_fp8_impl(
        q=q,
        weights=weights,
        seq_lens=seq_lens,
        block_table=block_table,
        page_size=page_size,
        topk=topk,
        softmax_scale=softmax_scale,
        q_len_per_req=q_len_per_req,
        topk_layout=topk_layout,
        block_table_base_offsets=block_table_base_offsets,
        index_k_cache=index_k_cache,
        seq_lens_2d=seq_lens_2d,
        plan=plan,
        out=out,
        lens_out=lens_out,
    )


@register_kernel(
    "attention",
    "dsa_prefill_topk",
    name="triton_dsa_prefill_topk_fp8",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=_TOPK_SIGNATURES,
    traits={
        "head_dim": frozenset({128}),
        "topk": frozenset({512, 1024, 2048}),
        "index_k_format": frozenset({"fp8_scaled"}),
        "index_k_layout": frozenset({"packed", "page_planar"}),
    },
    priority=Priority.PORTABLE,
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
    return _triton_dsa_prefill_topk_fp8_impl(
        q=q,
        weights=weights,
        kv_workspace_slots=kv_workspace_slots,
        row_starts=row_starts,
        row_ends=row_ends,
        topk=topk,
        softmax_scale=softmax_scale,
        index_k_cache=index_k_cache,
        page_size=page_size,
        index_k_fp8=index_k_fp8,
        index_k_scale=index_k_scale,
        max_logits_bytes=max_logits_bytes,
        out=out,
        lens_out=lens_out,
    )


def triton_dsa_index_candidates(
    q: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    local_page_table: torch.Tensor,
    query_requests: torch.Tensor,
    causal_lens: torch.Tensor,
    *,
    page_size: int,
    topk: int,
    softmax_scale: float,
    initial_tokens: int,
    local_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score owned Index-K pages and return (logical offsets, FP32 scores).

    Page-table positions retain request order; foreign/null pages are -1.
    Query requests and causal lengths have one entry per query, so the same
    contract covers prefill and decode. Invalid candidates have offset -1 and
    score -inf. Forced initial/tail windows use global request positions.
    The caller bounds the query tile to cap logits scratch space.
    """
    from tokenspeed_kernel.ops.attention.dsa._triton.index_candidates import (
        candidate_topk_offsets,
    )
    from tokenspeed_kernel.ops.attention.dsa._triton.topk import (
        _check_packed_fp8_inputs,
        _dsa_decode_logits_fp8_kernel,
    )

    if initial_tokens < 0 or local_tokens < 0 or initial_tokens + local_tokens > topk:
        raise ValueError("Forced windows must fit in topk")
    if query_requests.shape != (q.shape[0],) or causal_lens.shape != (q.shape[0],):
        raise ValueError("Index candidate rows must match queries")
    if local_page_table.shape[0] == 0 or local_page_table.shape[1] == 0:
        raise ValueError("Index candidates require a nonempty page table")
    valid_requests = (query_requests >= 0) & (
        query_requests < local_page_table.shape[0]
    )
    query_requests = query_requests.clamp(0, local_page_table.shape[0] - 1)
    causal_lens = torch.where(valid_requests, causal_lens, 0)
    q, weights = q.contiguous(), weights.float().contiguous()
    row_bytes, page_stride = _check_packed_fp8_inputs(
        q, index_k_cache, weights, page_size
    )
    width = local_page_table.shape[1] * page_size
    logits = torch.empty((q.shape[0], width), device=q.device, dtype=torch.float32)
    _dsa_decode_logits_fp8_kernel[(q.shape[0], triton.cdiv(width, 64))](
        q,
        index_k_cache.view(torch.float8_e4m3fn),
        index_k_cache.view(torch.float32),
        weights,
        causal_lens,
        local_page_table,
        logits,
        local_page_table.stride(0),
        logits.stride(0),
        query_requests,
        EXPLICIT_ROWS=True,
        INITIAL_TOKENS=initial_tokens,
        LOCAL_TOKENS=local_tokens,
        page_size=page_size,
        row_bytes=row_bytes,
        page_stride_bytes=page_stride,
        max_seq_len=width,
        num_heads=q.shape[1],
        head_dim=q.shape[2],
        num_groups=q.shape[2] // 128,
        softmax_scale=softmax_scale,
        q_len_per_req=1,
        BLOCK_N=64,
        BLOCK_D=64,
        num_warps=4,
        num_stages=1,
    )
    offsets = candidate_topk_offsets(logits, topk).to(torch.int32)
    scores = logits.gather(1, offsets.clamp_min(0).long())
    valid = (offsets >= 0) & (scores > -float("inf"))
    return torch.where(valid, offsets, -1), torch.where(valid, scores, -float("inf"))


@register_kernel(
    "attention",
    "dsa_index_candidates",
    name="triton_dsa_sharded_index_candidates",
    solution="triton",
    signatures=_TOPK_SIGNATURES,
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    priority=Priority.PORTABLE,
)
def triton_dsa_sharded_index_candidates(
    q: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    local_page_table: torch.Tensor,
    query_requests: torch.Tensor,
    causal_lens: torch.Tensor,
    *,
    page_size: int,
    topk: int,
    softmax_scale: float,
    initial_tokens: int,
    local_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from tokenspeed_kernel.ops.quantization import quantize_fp8_with_scale
    from tokenspeed_kernel.platform import current_platform

    if current_platform().is_nvidia:
        quantized, scale = quantize_fp8_with_scale(
            q.reshape(-1, q.shape[-1]),
            granularity="token_group",
            group_size=128,
            scale_encoding="float32",
        )
        scale = scale[: q.shape[0] * q.shape[1]].contiguous()
        weights = combine_topk_weights(weights, scale, softmax_scale)
        q = quantized.view_as(q).to(torch.bfloat16)
        softmax_scale = 1.0
    return triton_dsa_index_candidates(
        q,
        weights,
        index_k_cache,
        local_page_table,
        query_requests,
        causal_lens,
        page_size=page_size,
        topk=topk,
        softmax_scale=softmax_scale,
        initial_tokens=initial_tokens,
        local_tokens=local_tokens,
    )

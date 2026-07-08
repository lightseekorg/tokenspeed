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

"""Correctness-first selected-slot DSA Gluon kernels for AMD GFX950."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton

__all__ = [
    "gluon_dsa_decode_gfx950",
    "gluon_dsa_prefill_gfx950",
]


@gluon.constexpr_function
def _value_layout(
    BLOCK_TOPK: gl.constexpr,
    BLOCK_V: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    return gl.BlockedLayout([1, 1], [1, 64], [NUM_WARPS, 1], [1, 0])


@gluon.jit
def _dense_score(
    q,
    kv,
    slots,
    valid,
    q_base,
    kv_dim: gl.constexpr,
    kv_lora_rank: gl.constexpr,
    qk_rope_head_dim: gl.constexpr,
    BLOCK_TOPK: gl.constexpr,
    layout: gl.constexpr,
):
    score = gl.full(
        [BLOCK_TOPK],
        value=0.0,
        dtype=gl.float32,
        layout=gl.SliceLayout(1, layout),
    )
    for dim in gl.static_range(0, kv_lora_rank):
        q_val = gl.load(q + q_base + dim).to(gl.float32)
        k_val = gl.load(
            kv + slots * kv_dim + dim,
            mask=valid,
            other=0.0,
        ).to(gl.float32)
        score += k_val * q_val
    for dim in gl.static_range(0, qk_rope_head_dim):
        q_val = gl.load(q + q_base + kv_lora_rank + dim).to(gl.float32)
        k_val = gl.load(
            kv + slots * kv_dim + kv_lora_rank + dim,
            mask=valid,
            other=0.0,
        ).to(gl.float32)
        score += k_val * q_val
    return score


@gluon.jit
def _packed_score(
    q,
    kv_fp8,
    kv_scale,
    kv_rope,
    slots,
    valid,
    q_base,
    row_bytes: gl.constexpr,
    kv_lora_rank: gl.constexpr,
    qk_rope_head_dim: gl.constexpr,
    BLOCK_TOPK: gl.constexpr,
    layout: gl.constexpr,
):
    score = gl.full(
        [BLOCK_TOPK],
        value=0.0,
        dtype=gl.float32,
        layout=gl.SliceLayout(1, layout),
    )
    for dim in gl.static_range(0, kv_lora_rank):
        q_val = gl.load(q + q_base + dim).to(gl.float32)
        k_val = gl.load(
            kv_fp8 + slots * row_bytes + dim,
            mask=valid,
            other=0.0,
        ).to(gl.float32)
        k_scale = gl.load(
            kv_scale + (slots * row_bytes + kv_lora_rank + (dim // 128) * 4) // 4,
            mask=valid,
            other=0.0,
        ).to(gl.float32)
        score += k_val * k_scale * q_val
    rope_base = (slots * row_bytes + kv_lora_rank + (kv_lora_rank // 128) * 4) // 2
    for dim in gl.static_range(0, qk_rope_head_dim):
        q_val = gl.load(q + q_base + kv_lora_rank + dim).to(gl.float32)
        k_val = gl.load(
            kv_rope + rope_base + dim,
            mask=valid,
            other=0.0,
        ).to(gl.float32)
        score += k_val * q_val
    return score


@gluon.jit
def _dsa_dense_kv_kernel(
    q,
    kv,
    topk_indices,
    topk_lens,
    out,
    num_heads: gl.constexpr,
    head_dim: gl.constexpr,
    kv_lora_rank: gl.constexpr,
    qk_rope_head_dim: gl.constexpr,
    kv_dim: gl.constexpr,
    topk: gl.constexpr,
    softmax_scale: gl.constexpr,
    BLOCK_TOPK: gl.constexpr,
    BLOCK_V: gl.constexpr,
):
    token = gl.program_id(0)
    head = gl.program_id(1)
    v_block = gl.program_id(2)
    layout: gl.constexpr = _value_layout(BLOCK_TOPK, BLOCK_V, gl.num_warps())
    topk_offsets = gl.arange(0, BLOCK_TOPK, layout=gl.SliceLayout(1, layout))
    v_offsets = v_block * BLOCK_V + gl.arange(
        0, BLOCK_V, layout=gl.SliceLayout(0, layout)
    )
    q_base = (token * num_heads + head) * head_dim
    valid_len = gl.load(topk_lens + token).to(gl.int32)
    max_score = gl.full((), -float("inf"), gl.float32)

    for start in range(0, topk, BLOCK_TOPK):
        cols = start + topk_offsets
        valid = cols < valid_len
        slots = gl.load(topk_indices + token * topk + cols, mask=valid, other=0).to(
            gl.int64
        )
        valid = valid & (slots >= 0)
        score = _dense_score(
            q,
            kv,
            slots,
            valid,
            q_base,
            kv_dim,
            kv_lora_rank,
            qk_rope_head_dim,
            BLOCK_TOPK,
            layout,
        )
        score = gl.where(valid, score * softmax_scale, -float("inf"))
        max_score = gl.maximum(max_score, gl.max(score, axis=0))

    denom = gl.full((), 0.0, gl.float32)
    acc = gl.full(
        [BLOCK_V],
        value=0.0,
        dtype=gl.float32,
        layout=gl.SliceLayout(0, layout),
    )
    v_mask = v_offsets < kv_lora_rank
    for start in range(0, topk, BLOCK_TOPK):
        cols = start + topk_offsets
        valid = cols < valid_len
        slots = gl.load(topk_indices + token * topk + cols, mask=valid, other=0).to(
            gl.int64
        )
        valid = valid & (slots >= 0)
        score = _dense_score(
            q,
            kv,
            slots,
            valid,
            q_base,
            kv_dim,
            kv_lora_rank,
            qk_rope_head_dim,
            BLOCK_TOPK,
            layout,
        )
        score = gl.where(valid, score * softmax_scale, -float("inf"))
        probs = gl.exp(score - max_score)
        probs = gl.where(valid, probs, 0.0)
        denom += gl.sum(probs, axis=0)
        v_vals = gl.load(
            kv + slots[:, None] * kv_dim + v_offsets[None, :],
            mask=valid[:, None] & v_mask[None, :],
            other=0.0,
        ).to(gl.float32)
        acc += gl.sum(probs[:, None] * v_vals, axis=0)

    result = acc / denom
    result = gl.where(denom > 0.0, result, 0.0)
    out_base = (token * num_heads + head) * kv_lora_rank
    gl.store(out + out_base + v_offsets, result, mask=v_mask)


@gluon.jit
def _dsa_packed_kv_kernel(
    q,
    kv_fp8,
    kv_scale,
    kv_rope,
    topk_indices,
    topk_lens,
    out,
    num_heads: gl.constexpr,
    head_dim: gl.constexpr,
    kv_lora_rank: gl.constexpr,
    qk_rope_head_dim: gl.constexpr,
    row_bytes: gl.constexpr,
    topk: gl.constexpr,
    softmax_scale: gl.constexpr,
    BLOCK_TOPK: gl.constexpr,
    BLOCK_V: gl.constexpr,
):
    token = gl.program_id(0)
    head = gl.program_id(1)
    v_block = gl.program_id(2)
    layout: gl.constexpr = _value_layout(BLOCK_TOPK, BLOCK_V, gl.num_warps())
    topk_offsets = gl.arange(0, BLOCK_TOPK, layout=gl.SliceLayout(1, layout))
    v_offsets = v_block * BLOCK_V + gl.arange(
        0, BLOCK_V, layout=gl.SliceLayout(0, layout)
    )
    q_base = (token * num_heads + head) * head_dim
    valid_len = gl.load(topk_lens + token).to(gl.int32)
    max_score = gl.full((), -float("inf"), gl.float32)

    for start in range(0, topk, BLOCK_TOPK):
        cols = start + topk_offsets
        valid = cols < valid_len
        slots = gl.load(topk_indices + token * topk + cols, mask=valid, other=0).to(
            gl.int64
        )
        valid = valid & (slots >= 0)
        score = _packed_score(
            q,
            kv_fp8,
            kv_scale,
            kv_rope,
            slots,
            valid,
            q_base,
            row_bytes,
            kv_lora_rank,
            qk_rope_head_dim,
            BLOCK_TOPK,
            layout,
        )
        score = gl.where(valid, score * softmax_scale, -float("inf"))
        max_score = gl.maximum(max_score, gl.max(score, axis=0))

    denom = gl.full((), 0.0, gl.float32)
    acc = gl.full(
        [BLOCK_V],
        value=0.0,
        dtype=gl.float32,
        layout=gl.SliceLayout(0, layout),
    )
    v_mask = v_offsets < kv_lora_rank
    for start in range(0, topk, BLOCK_TOPK):
        cols = start + topk_offsets
        valid = cols < valid_len
        slots = gl.load(topk_indices + token * topk + cols, mask=valid, other=0).to(
            gl.int64
        )
        valid = valid & (slots >= 0)
        score = _packed_score(
            q,
            kv_fp8,
            kv_scale,
            kv_rope,
            slots,
            valid,
            q_base,
            row_bytes,
            kv_lora_rank,
            qk_rope_head_dim,
            BLOCK_TOPK,
            layout,
        )
        score = gl.where(valid, score * softmax_scale, -float("inf"))
        probs = gl.exp(score - max_score)
        probs = gl.where(valid, probs, 0.0)
        denom += gl.sum(probs, axis=0)
        v_vals = gl.load(
            kv_fp8 + slots[:, None] * row_bytes + v_offsets[None, :],
            mask=valid[:, None] & v_mask[None, :],
            other=0.0,
        ).to(gl.float32)
        v_scale = gl.load(
            kv_scale
            + (
                slots[:, None] * row_bytes
                + kv_lora_rank
                + (v_offsets[None, :] // 128) * 4
            )
            // 4,
            mask=valid[:, None] & v_mask[None, :],
            other=0.0,
        ).to(gl.float32)
        acc += gl.sum(probs[:, None] * v_vals * v_scale, axis=0)

    result = acc / denom
    result = gl.where(denom > 0.0, result, 0.0)
    out_base = (token * num_heads + head) * kv_lora_rank
    gl.store(out + out_base + v_offsets, result, mask=v_mask)


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


def _check_inputs(
    q: torch.Tensor,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor | None,
    *,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    page_size: int,
) -> None:
    if q.dtype != torch.bfloat16:
        raise TypeError(f"Gluon DSA supports BF16 q for this milestone, got {q.dtype}")
    if page_size != 64:
        raise ValueError(f"Gluon DSA supports page_size=64, got {page_size}")
    if qk_nope_head_dim != 192:
        raise ValueError(
            f"Gluon DSA supports qk_nope_head_dim=192, got {qk_nope_head_dim}"
        )
    if kv_lora_rank != 512:
        raise ValueError(f"Gluon DSA supports kv_lora_rank=512, got {kv_lora_rank}")
    if qk_rope_head_dim != 64:
        raise ValueError(
            f"Gluon DSA supports qk_rope_head_dim=64, got {qk_rope_head_dim}"
        )
    if topk_slots.dtype != torch.int32 or topk_slots.dim() != 2:
        raise ValueError("topk_slots must be int32 with shape [tokens, topk]")
    if topk_lens is None:
        raise ValueError("Gluon DSA requires topk_lens for this milestone")
    if topk_lens.dtype != torch.int32 or topk_lens.shape != (topk_slots.shape[0],):
        raise ValueError("topk_lens must be int32 with shape [tokens]")


def _run_dense_kv(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor,
    *,
    softmax_scale: float,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> torch.Tensor:
    kv_dim = int(kv_lora_rank) + int(qk_rope_head_dim)
    out = torch.empty(
        (q.shape[0], q.shape[1], kv_lora_rank), dtype=q.dtype, device=q.device
    )
    _dsa_dense_kv_kernel[(q.shape[0], q.shape[1], triton.cdiv(kv_lora_rank, 64))](
        q,
        kv_cache,
        topk_slots,
        topk_lens,
        out,
        q.shape[1],
        q.shape[2],
        kv_lora_rank,
        qk_rope_head_dim,
        kv_dim,
        topk_slots.shape[1],
        float(softmax_scale),
        BLOCK_TOPK=32,
        BLOCK_V=64,
        num_warps=4,
    )
    return out


def _run_packed_kv(
    q: torch.Tensor,
    packed_kv: torch.Tensor,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor,
    *,
    softmax_scale: float,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> torch.Tensor:
    row_bytes = int(packed_kv.shape[1])
    out = torch.empty(
        (q.shape[0], q.shape[1], kv_lora_rank), dtype=q.dtype, device=q.device
    )
    _dsa_packed_kv_kernel[(q.shape[0], q.shape[1], triton.cdiv(kv_lora_rank, 64))](
        q,
        packed_kv.view(torch.float8_e4m3fn),
        packed_kv.view(torch.float32),
        packed_kv.view(torch.bfloat16),
        topk_slots,
        topk_lens,
        out,
        q.shape[1],
        q.shape[2],
        kv_lora_rank,
        qk_rope_head_dim,
        row_bytes,
        topk_slots.shape[1],
        float(softmax_scale),
        BLOCK_TOPK=32,
        BLOCK_V=64,
        num_warps=4,
    )
    return out


def _run_dsa(
    *,
    q: torch.Tensor,
    kv_cache: torch.Tensor | None,
    sparse_kv_cache: torch.Tensor | None,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
    page_size: int,
    k_scale: float,
    out: torch.Tensor | None,
) -> torch.Tensor:
    _check_inputs(
        q,
        topk_slots,
        topk_lens,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        page_size=page_size,
    )
    q = _flatten_query(q).contiguous()
    topk_slots = topk_slots.contiguous()
    topk_lens = topk_lens.contiguous()
    softmax_scale = float(softmax_scale) * float(k_scale)
    if sparse_kv_cache is not None:
        result = _run_packed_kv(
            q,
            _flatten_packed_kv_cache(sparse_kv_cache).contiguous(),
            topk_slots,
            topk_lens,
            softmax_scale=softmax_scale,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
        )
    elif kv_cache is not None:
        result = _run_dense_kv(
            q,
            _flatten_dense_kv_cache(kv_cache).contiguous(),
            topk_slots,
            topk_lens,
            softmax_scale=softmax_scale,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
        )
    else:
        raise ValueError("Gluon DSA requires kv_cache or sparse_kv_cache")
    if out is None:
        return result
    out_view = out.reshape_as(result)
    out_view.copy_(result)
    return out


def gluon_dsa_decode_gfx950(
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
    del max_seqlen_k, q_len_per_req
    if logit_cap != 0.0 or return_lse:
        raise ValueError("Gluon DSA does not support logit_cap or return_lse")
    return _run_dsa(
        q=q,
        kv_cache=kv_cache,
        sparse_kv_cache=sparse_kv_cache,
        topk_slots=topk_slots,
        topk_lens=topk_lens,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        softmax_scale=softmax_scale,
        page_size=page_size,
        k_scale=k_scale,
        out=out,
    )


def gluon_dsa_prefill_gfx950(
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
    del max_seqlen_k, q_len_per_req
    if logit_cap != 0.0 or return_lse:
        raise ValueError("Gluon DSA does not support logit_cap or return_lse")
    return _run_dsa(
        q=q,
        kv_cache=kv_cache,
        sparse_kv_cache=sparse_kv_cache,
        topk_slots=topk_slots,
        topk_lens=topk_lens,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        softmax_scale=softmax_scale,
        page_size=page_size,
        k_scale=k_scale,
        out=out,
    )

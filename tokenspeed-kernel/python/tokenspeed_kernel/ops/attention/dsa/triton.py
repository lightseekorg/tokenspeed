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
from tokenspeed_kernel.ops.attention.dsa._triton.split_partials import (
    merge_split_partials,
)
from tokenspeed_kernel.ops.quantization import quantize_fp8
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
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
    sink,
    RETURN_LSE: tl.constexpr,
    HAS_SINK: tl.constexpr,
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
    has_keys = tl.full((), False, tl.int1)

    for start in range(0, topk, BLOCK_TOPK):
        cols = start + topk_offsets
        valid = cols < valid_len
        slots = tl.load(
            topk_indices + token * topk + cols,
            mask=valid,
            other=0,
        ).to(tl.int64)
        valid = valid & (slots >= 0)
        has_keys = has_keys | (tl.sum(valid.to(tl.int32), axis=0) > 0)
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

    if HAS_SINK:
        # The learnable sink is one extra candidate with no value vector: it
        # only shifts the softmax denominator, so it joins the max fold instead
        # of producing an output term.
        max_score = tl.maximum(max_score, tl.load(sink + head).to(tl.float32))

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

    # A row with no selected key stays empty (zero output, -inf LSE) even
    # with a sink, matching the other DSA backends.
    # Key probabilities may underflow under a strong sink; emptiness is
    # determined by valid slots, not by the floating-point denominator.
    if HAS_SINK:
        sink_value = tl.load(sink + head).to(tl.float32)
        denom += tl.exp(sink_value - max_score)

    result = acc / denom
    result = tl.where(has_keys, result, 0.0)
    out_base = (token * num_heads + head) * kv_lora_rank
    tl.store(out + out_base + v_offsets, result, mask=v_mask)
    if RETURN_LSE:
        if v_block == 0:
            tl.store(
                lse + token * num_heads + head,
                tl.where(has_keys, max_score + tl.log(denom), -float("inf")),
            )


@triton.jit
def _dsa_dense_kv_kernel(
    q,
    kv,
    topk_indices,
    topk_lens,
    out,
    lse,
    sink,
    RETURN_LSE: tl.constexpr,
    HAS_SINK: tl.constexpr,
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
    has_keys = tl.full((), False, tl.int1)

    for start in range(0, topk, BLOCK_TOPK):
        cols = start + topk_offsets
        valid = cols < valid_len
        slots = tl.load(
            topk_indices + token * topk + cols,
            mask=valid,
            other=0,
        ).to(tl.int64)
        valid = valid & (slots >= 0)
        has_keys = has_keys | (tl.sum(valid.to(tl.int32), axis=0) > 0)
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

    if HAS_SINK:
        # The learnable sink is one extra candidate with no value vector: it
        # only shifts the softmax denominator, so it joins the max fold instead
        # of producing an output term.
        max_score = tl.maximum(max_score, tl.load(sink + head).to(tl.float32))

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

    # A row with no selected key stays empty (zero output, -inf LSE) even
    # with a sink, matching the other DSA backends.
    # Key probabilities may underflow under a strong sink; emptiness is
    # determined by valid slots, not by the floating-point denominator.
    if HAS_SINK:
        sink_value = tl.load(sink + head).to(tl.float32)
        denom += tl.exp(sink_value - max_score)

    result = acc / denom
    result = tl.where(has_keys, result, 0.0)
    out_base = (token * num_heads + head) * kv_lora_rank
    tl.store(out + out_base + v_offsets, result, mask=v_mask)
    if RETURN_LSE:
        if v_block == 0:
            tl.store(
                lse + token * num_heads + head,
                tl.where(has_keys, max_score + tl.log(denom), -float("inf")),
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
    sink: torch.Tensor | None,
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
        # ``HAS_SINK`` is False whenever the caller has no sink, but Triton still
        # needs a pointer of the right type, so pass a one-element stand-in.
        sink if sink is not None else q,
        return_lse,
        sink is not None,
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
    sink: torch.Tensor | None,
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
        sink if sink is not None else q,
        return_lse,
        sink is not None,
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
    sink: torch.Tensor | None,
    out: torch.Tensor | None,
    return_lse: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    q = _flatten_query(q).contiguous()
    output_dtype = torch.bfloat16 if q.dtype == torch.float8_e4m3fn else q.dtype
    output_elements = q.shape[0] * q.shape[1] * kv_lora_rank
    if out is not None and (
        out.dtype != output_dtype
        or out.device != q.device
        or out.numel() != output_elements
        or not out.is_contiguous()
    ):
        raise ValueError(
            f"DSA output buffer must be contiguous {output_dtype} with "
            f"{output_elements} elements on {q.device}; got {out.dtype} "
            f"{tuple(out.shape)} on {out.device} with strides {out.stride()}"
        )
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
            sink=sink,
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
            sink=sink,
        )

    if out is not None:
        out.view_as(result).copy_(result)
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
        "sinks": frozenset({False, True}),
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
    attn_sink: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    del kv_seq_lens, enable_pdl
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
        sink=attn_sink,
        out=out,
        return_lse=return_lse,
    )


# Triton dots take at least 16 rows, so the split decode carries the heads of
# a row through its MMAs padded to 16; the padding lives in registers only.
_SPLIT_DECODE_MIN_HEADS = 16
_SPLIT_DECODE_BLOCK_N = 32
_SPLIT_DECODE_NUM_WARPS = 4
# Split programs per SM the column split aims for: a program's gathered KV rows
# arrive serially, so more than one per SM keeps loads in flight.
_SPLIT_DECODE_PROGRAMS_PER_SM = 2
# Padded columns a split covers at least, so a program's fixed work (its query
# load and its partial's store and merge) stays small against its KV gathers.
_SPLIT_DECODE_MIN_SPLIT_COLUMNS = 64


@triton.jit
def _dsa_dense_kv_split_kernel(
    q,
    kv,
    topk_indices,
    topk_lens,
    partial_out,
    partial_lse,
    split_lens,
    HAS_LENS: tl.constexpr,
    num_heads: tl.constexpr,
    HEADS_PAD: tl.constexpr,
    kv_lora_rank: tl.constexpr,
    qk_rope_head_dim: tl.constexpr,
    topk: tl.constexpr,
    softmax_scale: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # One program attends one column split of one query row for all its heads,
    # so every gathered latent row is loaded once and serves as key and value.
    token = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    part = token * tl.num_programs(1) + split
    row_dim: tl.constexpr = kv_lora_rank + qk_rope_head_dim

    heads = tl.arange(0, HEADS_PAD)
    live_heads = heads < num_heads
    dims = tl.arange(0, kv_lora_rank)
    rope_dims = kv_lora_rank + tl.arange(0, qk_rope_head_dim)
    q_rows = q + (token * num_heads + heads[:, None]) * row_dim
    q_nope = tl.load(q_rows + dims[None, :], mask=live_heads[:, None], other=0.0)
    q_rope = tl.load(q_rows + rope_dims[None, :], mask=live_heads[:, None], other=0.0)

    if HAS_LENS:
        valid_len = tl.minimum(tl.load(topk_lens + token), topk)
    else:
        valid_len = topk
    # The splits share the row's live columns evenly, in whole blocks, so a row
    # shorter than its padded width still keeps every split busy.
    split_width = tl.cdiv(tl.cdiv(valid_len, tl.num_programs(1)), BLOCK_N) * BLOCK_N
    first = split * split_width
    live = tl.minimum(tl.maximum(valid_len - first, 0), split_width)
    tl.store(split_lens + part, live)

    max_score = tl.full((HEADS_PAD,), -float("inf"), tl.float32)
    denom = tl.zeros((HEADS_PAD,), tl.float32)
    acc = tl.zeros((HEADS_PAD, kv_lora_rank), tl.float32)
    columns = tl.arange(0, BLOCK_N)
    split_slots = topk_indices + token * topk + first
    for start in range(0, live, BLOCK_N):
        slots = tl.load(
            split_slots + start + columns, mask=start + columns < live, other=-1
        )
        valid = slots >= 0
        kv_rows = kv + tl.where(valid, slots, 0).to(tl.int64)[:, None] * row_dim
        k_nope = tl.load(kv_rows + dims[None, :], mask=valid[:, None], other=0.0)
        k_rope = tl.load(kv_rows + rope_dims[None, :], mask=valid[:, None], other=0.0)
        scores = tl.dot(q_nope, tl.trans(k_nope))
        scores = tl.dot(q_rope, tl.trans(k_rope), scores)
        scores = tl.where(valid[None, :], scores * softmax_scale, -float("inf"))
        new_max = tl.maximum(max_score, tl.max(scores, axis=1))
        # A head without a live column yet keeps a finite reference, so its
        # terms are exp(-inf) = 0 rather than NaN.
        reference = tl.where(new_max == -float("inf"), 0.0, new_max)
        probs = tl.exp(scores - reference[:, None])
        rescale = tl.exp(max_score - reference)
        denom = denom * rescale + tl.sum(probs, axis=1)
        acc = tl.dot(probs.to(k_nope.dtype), k_nope, acc * rescale[:, None])
        max_score = new_max

    # NaN mass is a corrupt live partial, not the +inf empty sentinel.
    # Keep it in the merge so invalid data cannot become a finite result.
    attended = denom != 0.0
    result = acc / tl.where(attended, denom, 1.0)[:, None]
    out_rows = partial_out + (part * num_heads + heads[:, None]) * kv_lora_rank
    tl.store(
        out_rows + dims[None, :],
        result.to(partial_out.dtype.element_ty),
        mask=live_heads[:, None],
    )
    # +inf marks a split that attended nothing; the merge gives it no weight.
    lse = tl.where(attended, max_score + tl.log(denom), float("inf"))
    tl.store(partial_lse + part * num_heads + heads, lse, mask=live_heads)


def _split_decode_splits(tokens: int, topk: int) -> int:
    """Column splits per query row for the dense-KV split decode.

    The largest power of two that keeps the launch within
    ``_SPLIT_DECODE_PROGRAMS_PER_SM`` programs per SM and leaves every split at
    least ``_SPLIT_DECODE_MIN_SPLIT_COLUMNS`` of the padded columns.
    """
    budget = current_platform().sm_count * _SPLIT_DECODE_PROGRAMS_PER_SM
    splits = 1
    while (
        tokens * splits * 2 <= budget
        and splits * 2 * _SPLIT_DECODE_MIN_SPLIT_COLUMNS <= topk
    ):
        splits *= 2
    return splits


def _split_decode_partials(
    q: torch.Tensor,
    kv: torch.Tensor,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor | None,
    *,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
    splits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Attend every query row's column splits for the dense-KV split decode.

    Args:
        q: Contiguous ``[tokens, heads, kv_lora_rank + qk_rope_head_dim]`` queries.
        kv: Contiguous ``[slots, kv_lora_rank + qk_rope_head_dim]`` latent rows.
        topk_slots: ``[tokens, topk]`` global KV slots, ``-1`` where absent.
        topk_lens: Live slots per row, or None to attend the whole padded width.
        kv_lora_rank: Width of the latent part, which is also the value.
        qk_rope_head_dim: Width of the RoPE part after the latent part.
        softmax_scale: Scale applied to the attention scores.
        splits: Column splits per row.

    Returns:
        ``(partial, partial_lse, split_lens)``: each split's BF16 normalized
        ``[tokens * splits, heads, kv_lora_rank]`` output, its FP32 natural-log
        LSE (``+inf`` when it attended nothing) and its live column count, row
        major over ``(token, split)``.
    """
    tokens, num_heads, _ = q.shape
    partial = q.new_empty((tokens * splits, num_heads, kv_lora_rank))
    partial_lse = torch.empty(
        (tokens * splits, num_heads), dtype=torch.float32, device=q.device
    )
    split_lens = torch.empty((tokens * splits,), dtype=torch.int32, device=q.device)
    if tokens:
        _dsa_dense_kv_split_kernel[(tokens, splits)](
            q,
            kv,
            topk_slots.contiguous(),
            # Unread when HAS_LENS is False; Triton still needs a pointer.
            split_lens if topk_lens is None else topk_lens.contiguous(),
            partial,
            partial_lse,
            split_lens,
            topk_lens is not None,
            num_heads,
            max(_SPLIT_DECODE_MIN_HEADS, triton.next_power_of_2(num_heads)),
            kv_lora_rank,
            qk_rope_head_dim,
            int(topk_slots.shape[-1]),
            float(softmax_scale),
            BLOCK_N=_SPLIT_DECODE_BLOCK_N,
            num_warps=_SPLIT_DECODE_NUM_WARPS,
        )
    return partial, partial_lse, split_lens


@register_kernel(
    "attention",
    "dsa_decode",
    name="triton_dsa_split_decode",
    solution="triton",
    # Measured against FlashMLA on Hopper only.
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(9, 0),
        max_arch_version=ArchVersion(9, 0),
        vendors=frozenset({"nvidia"}),
    ),
    signatures=frozenset({format_signature(q=dense_tensor_format(torch.bfloat16))}),
    traits={
        "q_len": frozenset({1, 2, 3, 4, 5, 6}),
        "num_heads": frozenset({1, 2, 4, 8, 16}),
        "qk_nope_head_dim": frozenset({128, 192}),
        "kv_lora_rank": frozenset({512}),
        "qk_rope_head_dim": frozenset({64}),
        "page_size": frozenset({64}),
        "topk": frozenset({512, 1024, 2048}),
        "has_kv_cache": frozenset({True}),
        "has_sparse_kv_cache": frozenset({False}),
        "topk_layout": frozenset({"global_slots"}),
        "logit_cap": frozenset({False}),
        "return_lse": frozenset({False, True}),
        # Sinkless dense-KV decode keeps its existing kernels.
        "sinks": frozenset({True}),
    },
    priority=Priority.SPECIALIZED,
)
def triton_dsa_split_decode(
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
    attn_sink: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    del max_seqlen_k, qk_nope_head_dim, page_size, q_len_per_req
    del kv_seq_lens, logit_cap, enable_pdl
    if kv_cache is None or sparse_kv_cache is not None:
        raise RuntimeError("Triton split DSA decode reads a regular BF16 KV cache")
    q = _flatten_query(q).contiguous()
    kv = _flatten_dense_kv_cache(kv_cache)
    row_dim = int(kv_lora_rank) + int(qk_rope_head_dim)
    if q.shape[-1] != row_dim or kv.shape[-1] != row_dim or not kv.is_contiguous():
        raise ValueError(
            "Triton split DSA decode needs contiguous "
            f"[slots, {row_dim}] KV rows and [tokens, heads, {row_dim}] queries, "
            f"got KV {tuple(kv.shape)} with strides {kv.stride()} and query "
            f"{tuple(q.shape)}"
        )
    tokens, num_heads, _ = q.shape
    splits = _split_decode_splits(tokens, int(topk_slots.shape[-1]))
    partial, partial_lse, split_lens = _split_decode_partials(
        q,
        kv,
        topk_slots,
        topk_lens,
        kv_lora_rank=int(kv_lora_rank),
        qk_rope_head_dim=int(qk_rope_head_dim),
        softmax_scale=float(softmax_scale) * float(k_scale),
        splits=splits,
    )
    return merge_split_partials(
        partial,
        partial_lse,
        split_lens,
        splits=splits,
        heads=num_heads,
        sink=None if attn_sink is None else attn_sink[:num_heads].contiguous(),
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
        "sinks": frozenset({False, True}),
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
    attn_sink: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    del kv_seq_lens, enable_pdl
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
        sink=attn_sink,
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
    max_logits_bytes: int | None = None,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Reads the plane only: the leaf declares no INDEX_K_WORKSPACE_ROWS_FEATURE,
    # so the facade never hands it rows in workspace-row order.
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
    if current_platform().is_nvidia:
        quantized, scale = quantize_fp8(
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

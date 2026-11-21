# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
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

# TokenSpeed-owned penalty, logit-bias, and count kernels.

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton


@triton.jit
def _apply_penalties_logit_bias_inplace_kernel(
    logits_ptr,
    req_pool_indices_ptr,
    counts_ptr,
    candidates_ptr,
    logit_bias_ptr,
    freq_pen_pool_ptr,
    pres_pen_pool_ptr,
    rep_pen_pool_ptr,
    candidates_row_stride,
    vocab_size: tl.constexpr,
    logits_row_stride: tl.constexpr,
    counts_row_stride: tl.constexpr,
    bias_row_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_TOKENS_PER_REQ: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    req_row = row // NUM_TOKENS_PER_REQ
    pool_idx = tl.load(req_pool_indices_ptr + req_row)
    cols = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < vocab_size

    logits_offsets = row.to(tl.int64) * logits_row_stride + cols
    state_offsets = pool_idx * counts_row_stride + cols
    bias_offsets = pool_idx * bias_row_stride + cols

    vals = tl.load(logits_ptr + logits_offsets, mask=mask, other=0.0).to(tl.float32)
    counts = tl.load(counts_ptr + state_offsets, mask=mask, other=0).to(tl.float32)
    if NUM_TOKENS_PER_REQ > 1:
        # Verify row j scores the token after draft tokens 1..j, so those
        # count as emitted, as they would have been in plain decoding. Column
        # zero, the anchor, is in the decode counts already.
        depth = row % NUM_TOKENS_PER_REQ
        chain_ptr = candidates_ptr + req_row.to(tl.int64) * candidates_row_stride
        for k in tl.static_range(1, NUM_TOKENS_PER_REQ):
            token = tl.load(chain_ptr + k)
            counts += ((cols == token) & (k <= depth)).to(tl.float32)
    active = counts > 0.0

    rep = tl.load(rep_pen_pool_ptr + pool_idx).to(tl.float32)
    freq = tl.load(freq_pen_pool_ptr + pool_idx).to(tl.float32)
    presence = tl.load(pres_pen_pool_ptr + pool_idx).to(tl.float32)

    rep_vals = tl.where(vals > 0.0, vals / rep, vals * rep)
    vals = tl.where(active, rep_vals, vals)
    vals = vals - freq * counts - presence * active.to(tl.float32)
    vals += tl.load(logit_bias_ptr + bias_offsets, mask=mask, other=0.0).to(tl.float32)

    tl.store(logits_ptr + logits_offsets, vals, mask=mask)


def apply_penalties_logit_bias_inplace(
    logits: torch.Tensor,
    req_pool_indices: torch.Tensor,
    counts: torch.Tensor,
    logit_bias: torch.Tensor,
    freq_pen_pool: torch.Tensor,
    pres_pen_pool: torch.Tensor,
    rep_pen_pool: torch.Tensor,
    *,
    candidates: torch.Tensor | None,
    num_tokens_per_req: int = 1,
) -> torch.Tensor:
    """Apply repetition/frequency/presence penalties and logit_bias in-place.

    Args:
        logits: ``[requests * num_tokens_per_req, V]`` rows, updated in place.
        req_pool_indices: Int32 ``[requests]`` pool row of each request.
        counts: Int32 ``[pool_rows, >= V]`` tokens each request has emitted.
        logit_bias: ``[pool_rows, >= V]`` additive bias of each request.
        freq_pen_pool, pres_pen_pool, rep_pen_pool: ``[pool_rows]`` penalties.
        candidates: Int32 ``[requests, num_tokens_per_req]`` draft chain of a
            chain verify, column zero its anchor (already in ``counts``). Row
            ``j`` of a request also counts its draft tokens ``1..j``, the
            tokens plain decoding would have emitted before that row. Required
            when ``num_tokens_per_req`` > 1; with one row per request no draft
            precedes it, and ``None`` is allowed.
        num_tokens_per_req: Rows per request.

    Returns:
        ``logits``.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D, got {logits.ndim}D")
    if counts.ndim != 2 or logit_bias.ndim != 2:
        raise ValueError("counts and logit_bias must be 2D")
    if logits.device.type != "cuda":
        raise ValueError("apply_penalties_logit_bias_inplace requires CUDA logits")
    if logits.stride(-1) != 1:
        raise ValueError(
            "apply_penalties_logit_bias_inplace requires stride-1 vocab dimension, "
            f"got stride={logits.stride()}"
        )
    if req_pool_indices.dtype != torch.int32:
        raise ValueError(
            f"req_pool_indices must be int32, got {req_pool_indices.dtype}"
        )
    if counts.dtype != torch.int32:
        raise ValueError(f"counts must be int32, got {counts.dtype}")
    if num_tokens_per_req <= 0:
        raise ValueError("num_tokens_per_req must be positive")

    rows, vocab_size = logits.shape
    if rows % num_tokens_per_req != 0:
        raise ValueError(
            "logits rows must be divisible by num_tokens_per_req, "
            f"got rows={rows}, num_tokens_per_req={num_tokens_per_req}"
        )
    request_rows = rows // num_tokens_per_req
    if req_pool_indices.shape[0] != request_rows:
        raise ValueError(
            "req_pool_indices length must match request rows, "
            f"got {req_pool_indices.shape[0]} and {request_rows}"
        )
    if candidates is None and num_tokens_per_req > 1:
        raise ValueError(
            "candidates are required with num_tokens_per_req > 1: row j of a "
            "chain counts the chain's draft tokens 1..j"
        )
    if candidates is not None:
        if candidates.dtype != torch.int32:
            raise ValueError(f"candidates must be int32, got {candidates.dtype}")
        if candidates.shape != (request_rows, num_tokens_per_req):
            raise ValueError(
                "candidates must be [requests, num_tokens_per_req], got "
                f"{tuple(candidates.shape)} for {request_rows} requests and "
                f"num_tokens_per_req={num_tokens_per_req}"
            )
        if candidates.stride(1) != 1:
            raise ValueError(
                f"candidates rows must be contiguous, got stride={candidates.stride()}"
            )
    if counts.shape[1] < vocab_size or logit_bias.shape[1] < vocab_size:
        raise ValueError(
            "counts/logit_bias vocab dimension must cover logits vocab, "
            f"got counts={counts.shape}, logit_bias={logit_bias.shape}, logits={logits.shape}"
        )
    if rows == 0:
        return logits

    num_blocks = triton.cdiv(vocab_size, 1024)
    _apply_penalties_logit_bias_inplace_kernel[(rows, num_blocks)](
        logits,
        req_pool_indices,
        counts,
        # Never read when num_tokens_per_req is 1.
        req_pool_indices if candidates is None else candidates,
        logit_bias,
        freq_pen_pool,
        pres_pen_pool,
        rep_pen_pool,
        candidates_row_stride=0 if candidates is None else candidates.stride(0),
        vocab_size=vocab_size,
        logits_row_stride=logits.stride(0),
        counts_row_stride=counts.stride(0),
        bias_row_stride=logit_bias.stride(0),
        BLOCK_SIZE=1024,
        NUM_TOKENS_PER_REQ=num_tokens_per_req,
        num_warps=4,
        num_stages=3,
    )
    return logits


@triton.jit
def _accumulate_counts_inplace_kernel(
    counts_ptr,
    pool_idx_ptr,
    tokens_ptr,
    weights_ptr,
    # Per-step token count; runtime so every batch shape shares one binary.
    total,
    counts_row_stride: tl.constexpr,
    counts_col_stride: tl.constexpr,
    pool_idx_stride: tl.constexpr,
    tokens_stride: tl.constexpr,
    weights_stride: tl.constexpr,
    vocab_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total
    weights = tl.load(weights_ptr + offs * weights_stride, mask=mask, other=0).to(
        tl.int32
    )
    pool_idx = tl.load(pool_idx_ptr + offs * pool_idx_stride, mask=mask, other=0).to(
        tl.int64
    )
    tokens = tl.load(tokens_ptr + offs * tokens_stride, mask=mask, other=0).to(tl.int64)
    valid = mask & (weights != 0) & (tokens >= 0) & (tokens < vocab_size)
    tl.atomic_add(
        counts_ptr + pool_idx * counts_row_stride + tokens * counts_col_stride,
        weights,
        sem="relaxed",
        mask=valid,
    )


def accumulate_counts_inplace(
    counts: torch.Tensor,
    pool_idx: torch.Tensor,
    tokens: torch.Tensor,
    weights: torch.Tensor,
) -> None:
    """Graph-safe ``counts[pool_idx, tokens] += weights``."""
    if counts.ndim != 2:
        raise ValueError(f"counts must be 2D, got {counts.ndim}D")
    if counts.device.type != "cuda":
        raise ValueError("accumulate_counts_inplace requires CUDA counts")
    if counts.dtype != torch.int32:
        raise ValueError(f"counts must be int32, got {counts.dtype}")
    if pool_idx.dtype != torch.int32:
        raise ValueError(f"pool_idx must be int32, got {pool_idx.dtype}")
    if weights.dtype != torch.int32:
        raise ValueError(f"weights must be int32, got {weights.dtype}")
    if tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"tokens must be int32 or int64, got {tokens.dtype}")
    total = int(tokens.numel())
    if pool_idx.numel() != total or weights.numel() != total:
        raise ValueError(
            "pool_idx, tokens, and weights must have the same number of elements"
        )
    if total == 0:
        return

    pool_idx = pool_idx.reshape(-1)
    tokens = tokens.reshape(-1)
    weights = weights.reshape(-1)
    _accumulate_counts_inplace_kernel[(triton.cdiv(total, 256),)](
        counts,
        pool_idx,
        tokens,
        weights,
        total=total,
        counts_row_stride=counts.stride(0),
        counts_col_stride=counts.stride(1),
        pool_idx_stride=pool_idx.stride(0),
        tokens_stride=tokens.stride(0),
        weights_stride=weights.stride(0),
        vocab_size=counts.shape[1],
        BLOCK_SIZE=256,
        num_warps=4,
    )

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

# Contains TokenSpeed-native full sampling kernels adapted from vLLM MRV2
# sampling primitives:
#   https://vllm.ai/blog/2026-03-24-mrv2
#   https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/min_p.py
#   https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/penalties.py
#   https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/logit_bias.py
# Existing probability-route helpers moved here from TokenSpeed's previous
# sampling Triton module remain TokenSpeed compatibility code.

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

from .common import _GUMBEL_BLOCK_SIZE, _MIN_P_GUMBEL_BLOCK_SIZE
from .gumbel import _check_gumbel_pool_inputs, _gumbel_sample_stage2_kernel


@triton.jit
def _gumbel_sample_min_p_pool_kernel(
    logits_ptr,
    req_pool_indices_ptr,
    temperature_pool_ptr,
    min_p_pool_ptr,
    seed_pool_ptr,
    offsets_pool_ptr,
    out_ptr,
    logits_row_stride: tl.constexpr,
    vocab_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_TOKENS_PER_REQ: tl.constexpr,
):
    row = tl.program_id(0)
    req_row = row // NUM_TOKENS_PER_REQ
    spec_pos = row - req_row * NUM_TOKENS_PER_REQ
    pool_idx = tl.load(req_pool_indices_ptr + req_row)
    offsets = tl.arange(0, BLOCK_SIZE)

    temperature = tl.maximum(
        tl.load(temperature_pool_ptr + pool_idx).to(tl.float32), 1.0e-20
    )
    min_p = tl.load(min_p_pool_ptr + pool_idx).to(tl.float32)
    min_p_log_threshold = tl.log(tl.maximum(min_p, 1.0e-20))

    row_max = tl.full((), float("-inf"), tl.float32)
    for start in tl.range(0, vocab_size, BLOCK_SIZE, num_stages=3):
        cols = start + offsets
        mask = cols < vocab_size
        vals = tl.load(
            logits_ptr + row * logits_row_stride + cols,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        vals = vals / temperature
        row_max = tl.maximum(
            row_max, tl.max(tl.where(mask, vals, float("-inf")), axis=0)
        )

    threshold = row_max + min_p_log_threshold
    seed = tl.load(seed_pool_ptr + pool_idx).to(tl.int64)
    step_offset = tl.load(offsets_pool_ptr + pool_idx).to(tl.int64) + spec_pos
    rng_seed = tl.randint(seed, step_offset)

    best_score = tl.full((), float("-inf"), tl.float32)
    best_id = tl.full((), 2147483647, tl.int32)
    for start in tl.range(0, vocab_size, BLOCK_SIZE, num_stages=3):
        cols = start + offsets
        mask = cols < vocab_size
        vals = tl.load(
            logits_ptr + row * logits_row_stride + cols,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        vals = vals / temperature
        keep = mask & (vals >= threshold)
        uniform = tl.maximum(tl.rand(rng_seed, cols), 1.0e-7)
        gumbel = -tl.log(-tl.log(uniform))
        scores = tl.where(keep, vals + gumbel, float("-inf"))
        block_score = tl.max(scores, axis=0)
        block_id = tl.min(tl.where(scores == block_score, cols, 2147483647), axis=0)
        better = (block_score > best_score) | (
            (block_score == best_score) & (block_id < best_id)
        )
        best_score = tl.where(better, block_score, best_score)
        best_id = tl.where(better, block_id, best_id)

    tl.store(out_ptr + row, best_id)


def gumbel_sample_min_p_from_pools(
    logits: torch.Tensor,
    req_pool_indices: torch.Tensor,
    temperature_pool: torch.Tensor,
    min_p_pool: torch.Tensor,
    seed_pool: torch.Tensor,
    offsets_pool: torch.Tensor,
    out: torch.Tensor,
    *,
    block_size: int = _MIN_P_GUMBEL_BLOCK_SIZE,
    num_tokens_per_req: int = 1,
) -> torch.Tensor:
    """Gumbel-Max sampler for no top-k/top-p rows with a min-p cutoff."""
    if logits.ndim != 2:
        raise ValueError(f"gumbel_sample_min_p_from_pools expects 2D logits")
    if logits.device.type != "cuda":
        raise ValueError("gumbel_sample_min_p_from_pools requires CUDA logits")
    if logits.stride(-1) != 1:
        raise ValueError(
            "gumbel_sample_min_p_from_pools requires stride-1 vocab dimension, "
            f"got stride={logits.stride()}"
        )
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
    if req_pool_indices.dtype != torch.int32:
        raise ValueError(
            f"req_pool_indices must be int32, got {req_pool_indices.dtype}"
        )
    if min_p_pool.ndim != 1:
        raise ValueError(f"min_p_pool must be 1D, got {min_p_pool.ndim}D")
    if seed_pool.dtype != torch.int64:
        raise ValueError(f"seed_pool must be int64, got {seed_pool.dtype}")
    if out.dtype != torch.int32:
        raise ValueError(f"out must be int32, got {out.dtype}")
    if out.shape[0] < rows:
        raise ValueError(f"out is too small: {out.shape[0]} < {rows}")
    if rows == 0:
        return out[:0]

    _gumbel_sample_min_p_pool_kernel[(rows,)](
        logits,
        req_pool_indices,
        temperature_pool,
        min_p_pool,
        seed_pool,
        offsets_pool,
        out,
        logits_row_stride=logits.stride(0),
        vocab_size=vocab_size,
        BLOCK_SIZE=block_size,
        NUM_TOKENS_PER_REQ=num_tokens_per_req,
        num_warps=4,
        num_stages=3,
    )
    return out[:rows]


@triton.jit
def _min_p_local_max_kernel(
    logits_ptr,
    req_pool_indices_ptr,
    temperature_pool_ptr,
    local_max_ptr,
    logits_row_stride: tl.constexpr,
    local_row_stride: tl.constexpr,
    vocab_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_TOKENS_PER_REQ: tl.constexpr,
):
    row = tl.program_id(0)
    block_idx = tl.program_id(1)
    req_row = row // NUM_TOKENS_PER_REQ
    pool_idx = tl.load(req_pool_indices_ptr + req_row)
    cols = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < vocab_size
    temperature = tl.maximum(
        tl.load(temperature_pool_ptr + pool_idx).to(tl.float32), 1.0e-20
    )
    vals = tl.load(
        logits_ptr + row * logits_row_stride + cols,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    vals = vals / temperature
    tl.store(local_max_ptr + row * local_row_stride + block_idx, tl.max(vals, axis=0))


@triton.jit
def _min_p_row_max_kernel(
    local_max_ptr,
    row_max_ptr,
    local_row_stride: tl.constexpr,
    num_blocks: tl.constexpr,
    NUM_BLOCKS_PAD: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, NUM_BLOCKS_PAD)
    mask = offsets < num_blocks
    vals = tl.load(
        local_max_ptr + row * local_row_stride + offsets,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    tl.store(row_max_ptr + row, tl.max(vals, axis=0))


@triton.jit
def _min_p_local_gumbel_kernel(
    logits_ptr,
    req_pool_indices_ptr,
    temperature_pool_ptr,
    min_p_pool_ptr,
    seed_pool_ptr,
    offsets_pool_ptr,
    row_max_ptr,
    local_ids_ptr,
    local_scores_ptr,
    logits_row_stride: tl.constexpr,
    local_row_stride: tl.constexpr,
    vocab_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_TOKENS_PER_REQ: tl.constexpr,
):
    row = tl.program_id(0)
    block_idx = tl.program_id(1)
    req_row = row // NUM_TOKENS_PER_REQ
    spec_pos = row - req_row * NUM_TOKENS_PER_REQ
    pool_idx = tl.load(req_pool_indices_ptr + req_row)
    cols = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < vocab_size

    temperature = tl.maximum(
        tl.load(temperature_pool_ptr + pool_idx).to(tl.float32), 1.0e-20
    )
    min_p = tl.load(min_p_pool_ptr + pool_idx).to(tl.float32)
    threshold = tl.load(row_max_ptr + row).to(tl.float32) + tl.log(
        tl.maximum(min_p, 1.0e-20)
    )
    seed = tl.load(seed_pool_ptr + pool_idx).to(tl.int64)
    step_offset = tl.load(offsets_pool_ptr + pool_idx).to(tl.int64) + spec_pos
    rng_seed = tl.randint(seed, step_offset)

    vals = tl.load(
        logits_ptr + row * logits_row_stride + cols,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    vals = vals / temperature
    keep = mask & (vals >= threshold)
    uniform = tl.maximum(tl.rand(rng_seed, cols), 1.0e-7)
    gumbel = -tl.log(-tl.log(uniform))
    scores = tl.where(keep, vals + gumbel, float("-inf"))
    block_score = tl.max(scores, axis=0)
    block_id = tl.min(tl.where(scores == block_score, cols, 2147483647), axis=0)
    tl.store(local_ids_ptr + row * local_row_stride + block_idx, block_id)
    tl.store(local_scores_ptr + row * local_row_stride + block_idx, block_score)


def gumbel_sample_min_p_from_pools_parallel(
    logits: torch.Tensor,
    req_pool_indices: torch.Tensor,
    temperature_pool: torch.Tensor,
    min_p_pool: torch.Tensor,
    seed_pool: torch.Tensor,
    offsets_pool: torch.Tensor,
    local_ids: torch.Tensor,
    local_scores: torch.Tensor,
    row_max: torch.Tensor,
    out: torch.Tensor,
    *,
    block_size: int = _GUMBEL_BLOCK_SIZE,
    num_tokens_per_req: int = 1,
) -> torch.Tensor:
    """Parallel large-vocab min-p Gumbel sampler."""
    rows, vocab_size, num_blocks = _check_gumbel_pool_inputs(
        logits,
        req_pool_indices,
        temperature_pool,
        seed_pool,
        offsets_pool,
        local_ids,
        local_scores,
        out,
        fn_name="gumbel_sample_min_p_from_pools_parallel",
        block_size=block_size,
        num_tokens_per_req=num_tokens_per_req,
    )
    if min_p_pool.device.type != "cuda":
        raise ValueError("min_p_pool must be CUDA")
    if min_p_pool.ndim != 1:
        raise ValueError(f"min_p_pool must be 1D, got {min_p_pool.ndim}D")
    if row_max.device.type != "cuda":
        raise ValueError("row_max must be CUDA")
    if row_max.dtype != torch.float32:
        raise ValueError(f"row_max must be float32, got {row_max.dtype}")
    if row_max.shape[0] < rows:
        raise ValueError(f"row_max is too small: {row_max.shape[0]} < {rows}")
    if rows == 0:
        return out[:0]

    _min_p_local_max_kernel[(rows, num_blocks)](
        logits,
        req_pool_indices,
        temperature_pool,
        local_scores,
        logits_row_stride=logits.stride(0),
        local_row_stride=local_scores.stride(0),
        vocab_size=vocab_size,
        BLOCK_SIZE=block_size,
        NUM_TOKENS_PER_REQ=num_tokens_per_req,
        num_warps=4,
    )
    _min_p_row_max_kernel[(rows,)](
        local_scores,
        row_max,
        local_row_stride=local_scores.stride(0),
        num_blocks=num_blocks,
        NUM_BLOCKS_PAD=triton.next_power_of_2(num_blocks),
        num_warps=1,
    )
    _min_p_local_gumbel_kernel[(rows, num_blocks)](
        logits,
        req_pool_indices,
        temperature_pool,
        min_p_pool,
        seed_pool,
        offsets_pool,
        row_max,
        local_ids,
        local_scores,
        logits_row_stride=logits.stride(0),
        local_row_stride=local_ids.stride(0),
        vocab_size=vocab_size,
        BLOCK_SIZE=block_size,
        NUM_TOKENS_PER_REQ=num_tokens_per_req,
        num_warps=4,
    )
    _gumbel_sample_stage2_kernel[(rows,)](
        local_ids,
        local_scores,
        out,
        local_row_stride=local_ids.stride(0),
        num_blocks=num_blocks,
        NUM_BLOCKS_PAD=triton.next_power_of_2(num_blocks),
        num_warps=1,
    )
    return out[:rows]


@triton.jit
def _apply_penalties_logit_bias_inplace_kernel(
    logits_ptr,
    req_pool_indices_ptr,
    counts_ptr,
    logit_bias_ptr,
    freq_pen_pool_ptr,
    pres_pen_pool_ptr,
    rep_pen_pool_ptr,
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

    logits_offsets = row * logits_row_stride + cols
    state_offsets = pool_idx * counts_row_stride + cols
    bias_offsets = pool_idx * bias_row_stride + cols

    vals = tl.load(logits_ptr + logits_offsets, mask=mask, other=0.0).to(tl.float32)
    counts = tl.load(counts_ptr + state_offsets, mask=mask, other=0).to(tl.float32)
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
    num_tokens_per_req: int = 1,
    block_size: int = 1024,
) -> torch.Tensor:
    """Apply repetition/frequency/presence penalties and logit_bias in-place."""
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
    if counts.shape[1] < vocab_size or logit_bias.shape[1] < vocab_size:
        raise ValueError(
            "counts/logit_bias vocab dimension must cover logits vocab, "
            f"got counts={counts.shape}, logit_bias={logit_bias.shape}, logits={logits.shape}"
        )
    if rows == 0:
        return logits

    num_blocks = triton.cdiv(vocab_size, block_size)
    _apply_penalties_logit_bias_inplace_kernel[(rows, num_blocks)](
        logits,
        req_pool_indices,
        counts,
        logit_bias,
        freq_pen_pool,
        pres_pen_pool,
        rep_pen_pool,
        vocab_size=vocab_size,
        logits_row_stride=logits.stride(0),
        counts_row_stride=counts.stride(0),
        bias_row_stride=logit_bias.stride(0),
        BLOCK_SIZE=block_size,
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
    total: tl.constexpr,
    counts_row_stride: tl.constexpr,
    vocab_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total
    weights = tl.load(weights_ptr + offs, mask=mask, other=0).to(tl.int32)
    pool_idx = tl.load(pool_idx_ptr + offs, mask=mask, other=0).to(tl.int64)
    tokens = tl.load(tokens_ptr + offs, mask=mask, other=0).to(tl.int64)
    valid = mask & (weights != 0) & (tokens >= 0) & (tokens < vocab_size)
    tl.atomic_add(
        counts_ptr + pool_idx * counts_row_stride + tokens,
        weights,
        sem="relaxed",
        mask=valid,
    )


def accumulate_counts_inplace(
    counts: torch.Tensor,
    pool_idx: torch.Tensor,
    tokens: torch.Tensor,
    weights: torch.Tensor,
    *,
    block_size: int = 256,
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

    _accumulate_counts_inplace_kernel[(triton.cdiv(total, block_size),)](
        counts,
        pool_idx.reshape(-1),
        tokens.reshape(-1),
        weights.reshape(-1),
        total=total,
        counts_row_stride=counts.stride(0),
        vocab_size=counts.shape[1],
        BLOCK_SIZE=block_size,
        num_warps=4,
    )


@triton.jit
def _min_p_renorm_prob_kernel(
    probs_ptr,
    min_p_ptr,
    out_ptr,
    vocab_size: tl.constexpr,
    probs_row_stride: tl.constexpr,
    out_row_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    probs_row = probs_ptr + row * probs_row_stride
    out_row = out_ptr + row * out_row_stride

    max_prob = tl.full((), 0.0, tl.float32)
    for start in tl.range(0, vocab_size, BLOCK_SIZE, num_stages=3):
        cols = start + offs
        mask = cols < vocab_size
        vals = tl.load(probs_row + cols, mask=mask, other=0.0).to(tl.float32)
        max_prob = tl.maximum(max_prob, tl.max(tl.where(mask, vals, 0.0), axis=0))

    threshold = max_prob * tl.load(min_p_ptr + row).to(tl.float32)
    denom = tl.full((), 0.0, tl.float32)
    for start in tl.range(0, vocab_size, BLOCK_SIZE, num_stages=3):
        cols = start + offs
        mask = cols < vocab_size
        vals = tl.load(probs_row + cols, mask=mask, other=0.0).to(tl.float32)
        keep = mask & (vals >= threshold)
        denom += tl.sum(tl.where(keep, vals, 0.0), axis=0)

    inv_denom = 1.0 / tl.maximum(denom, 1.0e-20)
    for start in tl.range(0, vocab_size, BLOCK_SIZE, num_stages=3):
        cols = start + offs
        mask = cols < vocab_size
        vals = tl.load(probs_row + cols, mask=mask, other=0.0).to(tl.float32)
        keep = mask & (vals >= threshold)
        out = tl.where(keep, vals * inv_denom, 0.0)
        tl.store(out_row + cols, out, mask=mask)

    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def min_p_renorm_prob(
    probs: torch.Tensor,
    min_p: torch.Tensor,
    *,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Renormalize probabilities after applying a per-row min-p cutoff.

    For each row, this computes ``threshold = min_p[row] * max(probs[row])``,
    zeros probabilities below the threshold, and renormalizes the surviving
    probabilities so the row sums to one.
    """
    if probs.ndim != 2:
        raise ValueError(f"min_p_renorm_prob expects 2D probs, got {probs.ndim}D")
    if min_p.ndim != 1:
        raise ValueError(f"min_p_renorm_prob expects 1D min_p, got {min_p.ndim}D")
    if min_p.shape[0] != probs.shape[0]:
        raise ValueError(
            "min_p length must match probs rows, "
            f"got {min_p.shape[0]} and {probs.shape[0]}"
        )
    if probs.device.type != "cuda" or min_p.device.type != "cuda":
        raise ValueError("min_p_renorm_prob requires CUDA tensors")
    if probs.stride(-1) != 1:
        raise ValueError(
            f"min_p_renorm_prob requires stride-1 vocab dimension, got stride={probs.stride()}"
        )
    if not min_p.is_contiguous():
        min_p = min_p.contiguous()

    out = torch.empty_like(probs)
    rows, vocab_size = probs.shape
    if rows == 0:
        return out

    block_size = min(4096, triton.next_power_of_2(vocab_size))
    num_warps = 4 if block_size <= 1024 else 8
    extra_kwargs = {"launch_pdl": True} if enable_pdl else {}
    _min_p_renorm_prob_kernel[(rows,)](
        probs,
        min_p,
        out,
        vocab_size=vocab_size,
        probs_row_stride=probs.stride(0),
        out_row_stride=out.stride(0),
        BLOCK_SIZE=block_size,
        ENABLE_PDL=enable_pdl,
        num_warps=num_warps,
        num_stages=3,
        **extra_kwargs,
    )
    return out

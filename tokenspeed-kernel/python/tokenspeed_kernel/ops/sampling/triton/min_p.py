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

# TokenSpeed-native min-p Gumbel kernels.

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

from .gumbel import _check_gumbel_pool_inputs, _gumbel_sample_stage2_kernel

_MIN_P_GUMBEL_BLOCK_SIZE = 1024
_MIN_P_PARALLEL_BLOCK_SIZE = 1024


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
        BLOCK_SIZE=_MIN_P_GUMBEL_BLOCK_SIZE,
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
        block_size=_MIN_P_PARALLEL_BLOCK_SIZE,
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
        BLOCK_SIZE=_MIN_P_PARALLEL_BLOCK_SIZE,
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
        BLOCK_SIZE=_MIN_P_PARALLEL_BLOCK_SIZE,
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

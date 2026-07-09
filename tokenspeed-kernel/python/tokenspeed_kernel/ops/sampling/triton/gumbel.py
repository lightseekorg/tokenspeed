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

# TokenSpeed-specific pool state, scratch ownership, and tie-breaking.

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

_GUMBEL_BLOCK_SIZE = 1024
_GUMBEL_COMPACT_BLOCK_SIZE = 2048


@triton.jit
def _gumbel_sample_pool_stage1_kernel(
    logits_ptr,
    req_pool_indices_ptr,
    temperature_pool_ptr,
    seed_pool_ptr,
    offsets_pool_ptr,
    local_ids_ptr,
    local_scores_ptr,
    logits_row_stride: tl.constexpr,
    local_row_stride: tl.constexpr,
    vocab_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_TOKENS_PER_REQ: tl.constexpr,
):
    row = tl.program_id(0)
    req_row = row // NUM_TOKENS_PER_REQ
    spec_pos = row - req_row * NUM_TOKENS_PER_REQ
    block_idx = tl.program_id(1)
    token_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = token_offsets < vocab_size
    pool_idx = tl.load(req_pool_indices_ptr + req_row)

    logits = tl.load(
        logits_ptr + row * logits_row_stride + token_offsets,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    temperature = tl.maximum(
        tl.load(temperature_pool_ptr + pool_idx).to(tl.float32), 1.0e-20
    )

    seed = tl.load(seed_pool_ptr + pool_idx).to(tl.int64)
    offset = tl.load(offsets_pool_ptr + pool_idx).to(tl.int64) + spec_pos
    gumbel_seed = tl.randint(seed, offset)
    uniform = tl.maximum(tl.rand(gumbel_seed, token_offsets), 1.0e-7)
    gumbel = -tl.log(-tl.log(uniform))
    scores = tl.where(mask, logits / temperature + gumbel, float("-inf"))

    max_score = tl.max(scores, axis=0)
    token_id = tl.min(
        tl.where(scores == max_score, token_offsets, vocab_size + BLOCK_SIZE),
        axis=0,
    )

    tl.store(local_ids_ptr + row * local_row_stride + block_idx, token_id)
    tl.store(local_scores_ptr + row * local_row_stride + block_idx, max_score)


@triton.jit
def _gumbel_sample_stage2_kernel(
    local_ids_ptr,
    local_scores_ptr,
    out_ptr,
    local_row_stride: tl.constexpr,
    num_blocks: tl.constexpr,
    NUM_BLOCKS_PAD: tl.constexpr,
):
    row = tl.program_id(0)
    block_offsets = tl.arange(0, NUM_BLOCKS_PAD)
    mask = block_offsets < num_blocks

    scores = tl.load(
        local_scores_ptr + row * local_row_stride + block_offsets,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    ids = tl.load(
        local_ids_ptr + row * local_row_stride + block_offsets,
        mask=mask,
        other=2147483647,
    )

    max_score = tl.max(scores, axis=0)
    token_id = tl.min(tl.where(scores == max_score, ids, 2147483647), axis=0)
    tl.store(out_ptr + row, token_id)


@triton.jit
def _gumbel_sample_compact_pool_kernel(
    logits_ptr,
    req_pool_indices_ptr,
    temperature_pool_ptr,
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
    token_offsets = tl.arange(0, BLOCK_SIZE)
    temperature = tl.maximum(
        tl.load(temperature_pool_ptr + pool_idx).to(tl.float32), 1.0e-20
    )
    seed = tl.load(seed_pool_ptr + pool_idx).to(tl.int64)
    offset = tl.load(offsets_pool_ptr + pool_idx).to(tl.int64) + spec_pos
    gumbel_seed = tl.randint(seed, offset)

    best_score = tl.full((), float("-inf"), tl.float32)
    best_id = tl.full((), 2147483647, tl.int32)
    for start in tl.range(0, vocab_size, BLOCK_SIZE, num_stages=3):
        cols = start + token_offsets
        mask = cols < vocab_size
        logits = tl.load(
            logits_ptr + row * logits_row_stride + cols,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        uniform = tl.maximum(tl.rand(gumbel_seed, cols), 1.0e-7)
        gumbel = -tl.log(-tl.log(uniform))
        scores = tl.where(mask, logits / temperature + gumbel, float("-inf"))

        block_score = tl.max(scores, axis=0)
        block_id = tl.min(tl.where(scores == block_score, cols, 2147483647), axis=0)
        better = (block_score > best_score) | (
            (block_score == best_score) & (block_id < best_id)
        )
        best_score = tl.where(better, block_score, best_score)
        best_id = tl.where(better, block_id, best_id)

    tl.store(out_ptr + row, best_id)


def _check_gumbel_pool_inputs(
    logits: torch.Tensor,
    req_pool_indices: torch.Tensor,
    temperature_pool: torch.Tensor,
    seed_pool: torch.Tensor,
    offsets_pool: torch.Tensor,
    local_ids: torch.Tensor,
    local_scores: torch.Tensor,
    out: torch.Tensor,
    *,
    fn_name: str,
    block_size: int,
    num_tokens_per_req: int,
) -> tuple[int, int, int]:
    if logits.ndim != 2:
        raise ValueError(f"{fn_name} expects 2D logits, got {logits.ndim}D")
    if logits.device.type != "cuda":
        raise ValueError(f"{fn_name} requires CUDA logits")
    if logits.stride(-1) != 1:
        raise ValueError(
            f"{fn_name} requires stride-1 vocab dimension, "
            f"got stride={logits.stride()}"
        )
    rows, vocab_size = logits.shape
    if vocab_size <= 0:
        raise ValueError(f"{fn_name} requires non-empty vocab dimension")

    for name, tensor, ndim in (
        ("req_pool_indices", req_pool_indices, 1),
        ("temperature_pool", temperature_pool, 1),
        ("seed_pool", seed_pool, 1),
        ("offsets_pool", offsets_pool, 1),
        ("out", out, 1),
    ):
        if tensor.device.type != "cuda":
            raise ValueError(f"{name} must be CUDA")
        if tensor.ndim != ndim:
            raise ValueError(f"{name} must be {ndim}D, got {tensor.ndim}D")
    if num_tokens_per_req <= 0:
        raise ValueError("num_tokens_per_req must be positive")
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
    if out.shape[0] < rows:
        raise ValueError(f"out is too small: {out.shape[0]} < {rows}")
    if req_pool_indices.dtype != torch.int32:
        raise ValueError(
            f"req_pool_indices must be int32, got {req_pool_indices.dtype}"
        )
    if seed_pool.dtype != torch.int64:
        raise ValueError(f"seed_pool must be int64, got {seed_pool.dtype}")

    num_blocks = triton.cdiv(vocab_size, block_size)
    if local_ids.device.type != "cuda" or local_scores.device.type != "cuda":
        raise ValueError("gumbel pool scratch tensors must be CUDA")
    if local_ids.ndim != 2 or local_scores.ndim != 2:
        raise ValueError("gumbel pool scratch tensors must be 2D")
    if local_ids.shape[0] < rows or local_scores.shape[0] < rows:
        raise ValueError("gumbel pool scratch tensors have too few rows")
    if local_ids.shape[1] < num_blocks or local_scores.shape[1] < num_blocks:
        raise ValueError(
            "gumbel pool scratch tensors have too few blocks: "
            f"need {num_blocks}, got {local_ids.shape[1]} / {local_scores.shape[1]}"
        )
    if local_ids.dtype != torch.int32:
        raise ValueError(f"local_ids must be int32, got {local_ids.dtype}")
    if local_scores.dtype != torch.float32:
        raise ValueError(f"local_scores must be float32, got {local_scores.dtype}")
    if local_ids.stride(-1) != 1 or local_scores.stride(-1) != 1:
        raise ValueError("gumbel pool scratch tensors require stride-1 block dimension")
    return rows, vocab_size, num_blocks


def gumbel_sample_from_pools(
    logits: torch.Tensor,
    req_pool_indices: torch.Tensor,
    temperature_pool: torch.Tensor,
    seed_pool: torch.Tensor,
    offsets_pool: torch.Tensor,
    local_ids: torch.Tensor,
    local_scores: torch.Tensor,
    out: torch.Tensor,
    *,
    num_tokens_per_req: int = 1,
) -> torch.Tensor:
    """Sample token ids from logits with Gumbel-Max and pool-indexed scalars."""
    rows, vocab_size, num_blocks = _check_gumbel_pool_inputs(
        logits,
        req_pool_indices,
        temperature_pool,
        seed_pool,
        offsets_pool,
        local_ids,
        local_scores,
        out,
        fn_name="gumbel_sample_from_pools",
        block_size=_GUMBEL_BLOCK_SIZE,
        num_tokens_per_req=num_tokens_per_req,
    )
    if rows == 0:
        return out[:0]

    _gumbel_sample_pool_stage1_kernel[(rows, num_blocks)](
        logits,
        req_pool_indices,
        temperature_pool,
        seed_pool,
        offsets_pool,
        local_ids,
        local_scores,
        logits_row_stride=logits.stride(0),
        local_row_stride=local_ids.stride(0),
        vocab_size=vocab_size,
        BLOCK_SIZE=_GUMBEL_BLOCK_SIZE,
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


def gumbel_sample_from_pools_compact(
    logits: torch.Tensor,
    req_pool_indices: torch.Tensor,
    temperature_pool: torch.Tensor,
    seed_pool: torch.Tensor,
    offsets_pool: torch.Tensor,
    out: torch.Tensor,
    *,
    block_size: int = _GUMBEL_COMPACT_BLOCK_SIZE,
    num_tokens_per_req: int = 1,
) -> torch.Tensor:
    """Single-kernel Gumbel-Max path for vocabularies that fit one scan loop."""
    if logits.ndim != 2:
        raise ValueError(
            f"gumbel_sample_from_pools_compact expects 2D logits, got {logits.ndim}D"
        )
    if logits.device.type != "cuda":
        raise ValueError("gumbel_sample_from_pools_compact requires CUDA logits")
    if logits.stride(-1) != 1:
        raise ValueError(
            "gumbel_sample_from_pools_compact requires stride-1 vocab dimension, "
            f"got stride={logits.stride()}"
        )
    rows, vocab_size = logits.shape
    if vocab_size <= 0:
        raise ValueError("gumbel_sample_from_pools_compact requires non-empty vocab")
    if num_tokens_per_req <= 0:
        raise ValueError("num_tokens_per_req must be positive")
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
    if out.shape[0] < rows:
        raise ValueError(f"out is too small: {out.shape[0]} < {rows}")
    if rows == 0:
        return out[:0]

    _gumbel_sample_compact_pool_kernel[(rows,)](
        logits,
        req_pool_indices,
        temperature_pool,
        seed_pool,
        offsets_pool,
        out,
        logits_row_stride=logits.stride(0),
        vocab_size=vocab_size,
        BLOCK_SIZE=block_size,
        NUM_TOKENS_PER_REQ=num_tokens_per_req,
        num_warps=8,
        num_stages=3,
    )
    return out[:rows]

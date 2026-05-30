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

# Adapted from the vLLM MRV2 sampler direction of logits-space sampling with
# stateless Gumbel RNG:
#   https://vllm.ai/blog/2026-03-24-mrv2
#   https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/gumbel.py
# The top-p-only rejection/repair layout here is TokenSpeed-specific.

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

from .common import (
    _TOP_P_PARALLEL_BLOCK_SIZE,
    _TOP_P_PARALLEL_NUM_ATTEMPTS,
    _TOP_P_REPAIR_NUM_ATTEMPTS,
)


@triton.jit
def _top_p_parallel_stage1_kernel(
    logits_ptr,
    req_pool_indices_ptr,
    temperature_pool_ptr,
    seed_pool_ptr,
    offsets_pool_ptr,
    local_max_ptr,
    local_sum_ptr,
    local_argmax_ptr,
    local_scores_ptr,
    local_logits_ptr,
    local_ids_ptr,
    logits_row_stride: tl.constexpr,
    vocab_size: tl.constexpr,
    num_blocks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_ATTEMPTS: tl.constexpr,
    NUM_TOKENS_PER_REQ: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    req_row = row // NUM_TOKENS_PER_REQ
    spec_pos = row - req_row * NUM_TOKENS_PER_REQ
    pool_idx = tl.load(req_pool_indices_ptr + req_row)
    token_offsets = tl.arange(0, BLOCK_SIZE)
    cols = block * BLOCK_SIZE + token_offsets
    mask = cols < vocab_size

    temperature = tl.maximum(
        tl.load(temperature_pool_ptr + pool_idx).to(tl.float32), 1.0e-20
    )
    seed = tl.load(seed_pool_ptr + pool_idx).to(tl.int64)
    offset = tl.load(offsets_pool_ptr + pool_idx).to(tl.int64) + spec_pos

    vals = tl.load(
        logits_ptr + row * logits_row_stride + cols,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    vals = vals / temperature

    block_max = tl.max(vals, axis=0)
    safe_block_max = tl.where(block_max > -float("inf"), block_max, 0.0)
    block_sum = tl.sum(
        tl.where(mask & (vals > -float("inf")), tl.exp(vals - safe_block_max), 0.0),
        axis=0,
    )
    block_argmax = tl.min(tl.where(vals == block_max, cols, 2147483647), axis=0)
    block_base = row * num_blocks + block
    tl.store(local_max_ptr + block_base, block_max)
    tl.store(local_sum_ptr + block_base, block_sum)
    tl.store(local_argmax_ptr + block_base, block_argmax)

    for attempt in tl.static_range(0, NUM_ATTEMPTS):
        attempt_seed = tl.randint(seed, offset + attempt)
        uniform = tl.maximum(tl.rand(attempt_seed, cols), 1.0e-7)
        gumbel = -tl.log(-tl.log(uniform))
        scores = tl.where(mask, vals + gumbel, float("-inf"))
        best_score = tl.max(scores, axis=0)
        best_id = tl.min(tl.where(scores == best_score, cols, 2147483647), axis=0)
        best_logit = tl.max(tl.where(cols == best_id, vals, float("-inf")), axis=0)
        out_offset = block_base * NUM_ATTEMPTS + attempt
        tl.store(local_scores_ptr + out_offset, best_score)
        tl.store(local_logits_ptr + out_offset, best_logit)
        tl.store(local_ids_ptr + out_offset, best_id)


@triton.jit
def _top_p_parallel_stage2_kernel(
    local_max_ptr,
    local_sum_ptr,
    local_argmax_ptr,
    local_scores_ptr,
    local_logits_ptr,
    local_ids_ptr,
    row_max_ptr,
    row_total_ptr,
    row_argmax_ptr,
    row_candidate_logits_ptr,
    row_candidate_ids_ptr,
    num_blocks: tl.constexpr,
    NUM_BLOCKS_PAD: tl.constexpr,
    NUM_ATTEMPTS: tl.constexpr,
):
    row = tl.program_id(0)
    block_offsets = tl.arange(0, NUM_BLOCKS_PAD)
    block_mask = block_offsets < num_blocks
    block_base = row * num_blocks + block_offsets

    local_max = tl.load(
        local_max_ptr + block_base, mask=block_mask, other=-float("inf")
    )
    local_sum = tl.load(local_sum_ptr + block_base, mask=block_mask, other=0.0)
    row_max = tl.max(local_max, axis=0)
    safe_row_max = tl.where(row_max > -float("inf"), row_max, 0.0)
    total = tl.sum(
        tl.where(block_mask, local_sum * tl.exp(local_max - safe_row_max), 0.0),
        axis=0,
    )
    local_argmax = tl.load(
        local_argmax_ptr + block_base, mask=block_mask, other=2147483647
    )
    row_argmax = tl.min(
        tl.where((local_max == row_max) & block_mask, local_argmax, 2147483647),
        axis=0,
    )

    tl.store(row_max_ptr + row, row_max)
    tl.store(row_total_ptr + row, total)
    tl.store(row_argmax_ptr + row, row_argmax)

    for attempt in tl.static_range(0, NUM_ATTEMPTS):
        candidate_base = (row * num_blocks + block_offsets) * NUM_ATTEMPTS + attempt
        scores = tl.load(
            local_scores_ptr + candidate_base, mask=block_mask, other=-float("inf")
        )
        ids = tl.load(local_ids_ptr + candidate_base, mask=block_mask, other=2147483647)
        logits = tl.load(
            local_logits_ptr + candidate_base, mask=block_mask, other=-float("inf")
        )
        best_score = tl.max(scores, axis=0)
        best_id = tl.min(
            tl.where((scores == best_score) & block_mask, ids, 2147483647),
            axis=0,
        )
        best_logit = tl.max(tl.where(ids == best_id, logits, -float("inf")), axis=0)
        row_candidate_offset = row * NUM_ATTEMPTS + attempt
        tl.store(row_candidate_ids_ptr + row_candidate_offset, best_id)
        tl.store(row_candidate_logits_ptr + row_candidate_offset, best_logit)


@triton.jit
def _top_p_parallel_stage3_kernel(
    logits_ptr,
    req_pool_indices_ptr,
    temperature_pool_ptr,
    row_max_ptr,
    row_candidate_logits_ptr,
    row_candidate_ids_ptr,
    partial_before_ptr,
    logits_row_stride: tl.constexpr,
    vocab_size: tl.constexpr,
    num_blocks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_ATTEMPTS: tl.constexpr,
    NUM_TOKENS_PER_REQ: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    req_row = row // NUM_TOKENS_PER_REQ
    pool_idx = tl.load(req_pool_indices_ptr + req_row)
    token_offsets = tl.arange(0, BLOCK_SIZE)
    cols = block * BLOCK_SIZE + token_offsets
    mask = cols < vocab_size
    row_max = tl.load(row_max_ptr + row)
    temperature = tl.maximum(
        tl.load(temperature_pool_ptr + pool_idx).to(tl.float32), 1.0e-20
    )
    vals = tl.load(
        logits_ptr + row * logits_row_stride + cols,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    vals = vals / temperature
    weights = tl.exp(vals - row_max)

    for attempt in tl.static_range(0, NUM_ATTEMPTS):
        row_candidate_offset = row * NUM_ATTEMPTS + attempt
        candidate_logit = tl.load(row_candidate_logits_ptr + row_candidate_offset)
        candidate_id = tl.load(row_candidate_ids_ptr + row_candidate_offset)
        before_mask = (vals > candidate_logit) | (
            (vals == candidate_logit) & (cols < candidate_id)
        )
        before = tl.sum(tl.where(mask & before_mask, weights, 0.0), axis=0)
        out_offset = (row * num_blocks + block) * NUM_ATTEMPTS + attempt
        tl.store(partial_before_ptr + out_offset, before)


@triton.jit
def _top_p_parallel_stage4_kernel(
    top_p_pool_ptr,
    req_pool_indices_ptr,
    row_total_ptr,
    row_argmax_ptr,
    row_candidate_ids_ptr,
    partial_before_ptr,
    accepted_ptr,
    out_ptr,
    num_blocks: tl.constexpr,
    NUM_BLOCKS_PAD: tl.constexpr,
    NUM_ATTEMPTS: tl.constexpr,
    NUM_TOKENS_PER_REQ: tl.constexpr,
):
    row = tl.program_id(0)
    req_row = row // NUM_TOKENS_PER_REQ
    pool_idx = tl.load(req_pool_indices_ptr + req_row)
    top_p = tl.load(top_p_pool_ptr + pool_idx).to(tl.float32)
    target_mass = top_p * tl.load(row_total_ptr + row)
    block_offsets = tl.arange(0, NUM_BLOCKS_PAD)
    block_mask = block_offsets < num_blocks
    token = tl.load(row_argmax_ptr + row)
    found = tl.full((), 0, tl.int32)

    for attempt in tl.static_range(0, NUM_ATTEMPTS):
        before_base = (row * num_blocks + block_offsets) * NUM_ATTEMPTS + attempt
        before = tl.sum(
            tl.load(partial_before_ptr + before_base, mask=block_mask, other=0.0),
            axis=0,
        )
        accepted = before < target_mass
        candidate_id = tl.load(row_candidate_ids_ptr + row * NUM_ATTEMPTS + attempt)
        take = (found == 0) & accepted
        token = tl.where(take, candidate_id, token)
        found = tl.where(take, 1, found)

    tl.store(accepted_ptr + row, found)
    tl.store(out_ptr + row, token)


@triton.jit
def _top_p_parallel_repair_kernel(
    logits_ptr,
    req_pool_indices_ptr,
    temperature_pool_ptr,
    top_p_pool_ptr,
    seed_pool_ptr,
    offsets_pool_ptr,
    row_max_ptr,
    row_total_ptr,
    row_argmax_ptr,
    accepted_ptr,
    out_ptr,
    logits_row_stride: tl.constexpr,
    vocab_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    START_ATTEMPT: tl.constexpr,
    NUM_ATTEMPTS_TOTAL: tl.constexpr,
    NUM_TOKENS_PER_REQ: tl.constexpr,
):
    row = tl.program_id(0)
    accepted_found = tl.load(accepted_ptr + row)
    accepted_token = tl.load(out_ptr + row)
    req_row = row // NUM_TOKENS_PER_REQ
    spec_pos = row - req_row * NUM_TOKENS_PER_REQ
    pool_idx = tl.load(req_pool_indices_ptr + req_row)
    token_offsets = tl.arange(0, BLOCK_SIZE)

    temperature = tl.maximum(
        tl.load(temperature_pool_ptr + pool_idx).to(tl.float32), 1.0e-20
    )
    top_p = tl.load(top_p_pool_ptr + pool_idx).to(tl.float32)
    seed = tl.load(seed_pool_ptr + pool_idx).to(tl.int64)
    offset = tl.load(offsets_pool_ptr + pool_idx).to(tl.int64) + spec_pos
    row_max = tl.load(row_max_ptr + row)
    total = tl.load(row_total_ptr + row)
    target_mass = top_p * total
    row_argmax = tl.load(row_argmax_ptr + row)

    attempt = tl.full((), START_ATTEMPT, tl.int32)
    while (attempt < NUM_ATTEMPTS_TOTAL) & (accepted_found == 0):
        attempt_seed = tl.randint(seed, offset + attempt)
        best_score = tl.full((), float("-inf"), tl.float32)
        best_id = tl.full((), 2147483647, tl.int32)
        best_logit = tl.full((), float("-inf"), tl.float32)

        for start in tl.range(0, vocab_size, BLOCK_SIZE, num_stages=3):
            cols = start + token_offsets
            mask = cols < vocab_size
            vals = tl.load(
                logits_ptr + row * logits_row_stride + cols,
                mask=mask,
                other=float("-inf"),
            ).to(tl.float32)
            vals = vals / temperature
            uniform = tl.maximum(tl.rand(attempt_seed, cols), 1.0e-7)
            gumbel = -tl.log(-tl.log(uniform))
            scores = tl.where(mask, vals + gumbel, float("-inf"))
            block_score = tl.max(scores, axis=0)
            block_id = tl.min(tl.where(scores == block_score, cols, 2147483647), axis=0)
            block_logit = tl.max(
                tl.where(cols == block_id, vals, float("-inf")), axis=0
            )
            better = (block_score > best_score) | (
                (block_score == best_score) & (block_id < best_id)
            )
            best_score = tl.where(better, block_score, best_score)
            best_id = tl.where(better, block_id, best_id)
            best_logit = tl.where(better, block_logit, best_logit)

        before = tl.full((), 0.0, tl.float32)
        for start in tl.range(0, vocab_size, BLOCK_SIZE, num_stages=3):
            cols = start + token_offsets
            mask = cols < vocab_size
            vals = tl.load(
                logits_ptr + row * logits_row_stride + cols,
                mask=mask,
                other=float("-inf"),
            ).to(tl.float32)
            vals = vals / temperature
            weights = tl.exp(vals - row_max)
            before_mask = (vals > best_logit) | (
                (vals == best_logit) & (cols < best_id)
            )
            before += tl.sum(tl.where(mask & before_mask, weights, 0.0), axis=0)

        accepted = before < target_mass
        accepted_token = tl.where(accepted, best_id, accepted_token)
        accepted_found = tl.where(accepted, 1, accepted_found)
        attempt += 1

    token = tl.where(accepted_found != 0, accepted_token, row_argmax)
    tl.store(out_ptr + row, token)
    tl.store(accepted_ptr + row, accepted_found)


def gumbel_sample_top_p_parallel_from_pools(
    logits: torch.Tensor,
    req_pool_indices: torch.Tensor,
    temperature_pool: torch.Tensor,
    top_p_pool: torch.Tensor,
    seed_pool: torch.Tensor,
    offsets_pool: torch.Tensor,
    local_max: torch.Tensor,
    local_sum: torch.Tensor,
    local_argmax: torch.Tensor,
    local_scores: torch.Tensor,
    local_logits: torch.Tensor,
    local_ids: torch.Tensor,
    row_max: torch.Tensor,
    row_total: torch.Tensor,
    row_argmax: torch.Tensor,
    row_candidate_logits: torch.Tensor,
    row_candidate_ids: torch.Tensor,
    accepted: torch.Tensor,
    out: torch.Tensor,
    *,
    block_size: int = _TOP_P_PARALLEL_BLOCK_SIZE,
    num_attempts: int = _TOP_P_PARALLEL_NUM_ATTEMPTS,
    num_tokens_per_req: int = 1,
) -> torch.Tensor:
    """Block-parallel top-p-only Gumbel sampler for normal single-token sample.

    This path trades extra graph launches for much higher vocab-scan
    parallelism. It is intended for low-row normal sampling where the serial
    top-p rejection kernel underutilizes the GPU.
    """
    if logits.ndim != 2:
        raise ValueError("gumbel_sample_top_p_parallel_from_pools expects 2D logits")
    if logits.device.type != "cuda":
        raise ValueError("gumbel_sample_top_p_parallel_from_pools requires CUDA logits")
    if logits.stride(-1) != 1:
        raise ValueError(
            "gumbel_sample_top_p_parallel_from_pools requires stride-1 vocab dimension, "
            f"got stride={logits.stride()}"
        )
    rows, vocab_size = logits.shape
    if vocab_size <= 0:
        raise ValueError(
            "gumbel_sample_top_p_parallel_from_pools requires non-empty vocab"
        )
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
            "req_pool_indices length must match request rows for parallel top-p sample, "
            f"got {req_pool_indices.shape[0]} and {request_rows}"
        )
    if num_attempts <= 0:
        raise ValueError("num_attempts must be positive")

    num_blocks = triton.cdiv(vocab_size, block_size)
    num_blocks_pad = triton.next_power_of_2(num_blocks)
    for name, tensor, dtype in (
        ("req_pool_indices", req_pool_indices, torch.int32),
        ("seed_pool", seed_pool, torch.int64),
        ("local_argmax", local_argmax, torch.int32),
        ("local_ids", local_ids, torch.int32),
        ("row_argmax", row_argmax, torch.int32),
        ("row_candidate_ids", row_candidate_ids, torch.int32),
        ("accepted", accepted, torch.int32),
        ("out", out, torch.int32),
    ):
        if tensor.device.type != "cuda":
            raise ValueError(f"{name} must be CUDA")
        if tensor.dtype != dtype:
            raise ValueError(f"{name} must be {dtype}, got {tensor.dtype}")
    for name, tensor in (
        ("temperature_pool", temperature_pool),
        ("top_p_pool", top_p_pool),
        ("offsets_pool", offsets_pool),
        ("local_max", local_max),
        ("local_sum", local_sum),
        ("local_scores", local_scores),
        ("local_logits", local_logits),
        ("row_max", row_max),
        ("row_total", row_total),
        ("row_candidate_logits", row_candidate_logits),
    ):
        if tensor.device.type != "cuda":
            raise ValueError(f"{name} must be CUDA")
    if out.shape[0] < rows:
        raise ValueError(f"out is too small: {out.shape[0]} < {rows}")
    if rows == 0:
        return out[:0]

    local_shape = (rows, num_blocks)
    candidate_shape = (rows, num_blocks, num_attempts)
    row_candidate_shape = (rows, num_attempts)
    if local_max.shape[0] < rows or local_max.shape[1] < num_blocks:
        raise ValueError(f"local_max must cover {local_shape}, got {local_max.shape}")
    if local_sum.shape[0] < rows or local_sum.shape[1] < num_blocks:
        raise ValueError(f"local_sum must cover {local_shape}, got {local_sum.shape}")
    if local_argmax.shape[0] < rows or local_argmax.shape[1] < num_blocks:
        raise ValueError(
            f"local_argmax must cover {local_shape}, got {local_argmax.shape}"
        )
    for name, tensor in (
        ("local_scores", local_scores),
        ("local_logits", local_logits),
        ("local_ids", local_ids),
    ):
        if (
            tensor.shape[0] < rows
            or tensor.shape[1] < num_blocks
            or tensor.shape[2] < num_attempts
        ):
            raise ValueError(f"{name} must cover {candidate_shape}, got {tensor.shape}")
    for name, tensor in (
        ("row_max", row_max),
        ("row_total", row_total),
        ("row_argmax", row_argmax),
        ("accepted", accepted),
    ):
        if tensor.shape[0] < rows:
            raise ValueError(f"{name} is too small: {tensor.shape[0]} < {rows}")
    for name, tensor in (
        ("row_candidate_logits", row_candidate_logits),
        ("row_candidate_ids", row_candidate_ids),
    ):
        if tensor.shape[0] < rows or tensor.shape[1] < num_attempts:
            raise ValueError(
                f"{name} must cover {row_candidate_shape}, got {tensor.shape}"
            )

    _top_p_parallel_stage1_kernel[(rows, num_blocks)](
        logits,
        req_pool_indices,
        temperature_pool,
        seed_pool,
        offsets_pool,
        local_max,
        local_sum,
        local_argmax,
        local_scores,
        local_logits,
        local_ids,
        logits_row_stride=logits.stride(0),
        vocab_size=vocab_size,
        num_blocks=num_blocks,
        BLOCK_SIZE=block_size,
        NUM_ATTEMPTS=num_attempts,
        NUM_TOKENS_PER_REQ=num_tokens_per_req,
        num_warps=4,
        num_stages=3,
    )
    _top_p_parallel_stage2_kernel[(rows,)](
        local_max,
        local_sum,
        local_argmax,
        local_scores,
        local_logits,
        local_ids,
        row_max,
        row_total,
        row_argmax,
        row_candidate_logits,
        row_candidate_ids,
        num_blocks=num_blocks,
        NUM_BLOCKS_PAD=num_blocks_pad,
        NUM_ATTEMPTS=num_attempts,
        num_warps=8,
        num_stages=3,
    )
    _top_p_parallel_stage3_kernel[(rows, num_blocks)](
        logits,
        req_pool_indices,
        temperature_pool,
        row_max,
        row_candidate_logits,
        row_candidate_ids,
        local_scores,
        logits_row_stride=logits.stride(0),
        vocab_size=vocab_size,
        num_blocks=num_blocks,
        BLOCK_SIZE=block_size,
        NUM_ATTEMPTS=num_attempts,
        NUM_TOKENS_PER_REQ=num_tokens_per_req,
        num_warps=4,
        num_stages=3,
    )
    _top_p_parallel_stage4_kernel[(rows,)](
        top_p_pool,
        req_pool_indices,
        row_total,
        row_argmax,
        row_candidate_ids,
        local_scores,
        accepted,
        out,
        num_blocks=num_blocks,
        NUM_BLOCKS_PAD=num_blocks_pad,
        NUM_ATTEMPTS=num_attempts,
        NUM_TOKENS_PER_REQ=num_tokens_per_req,
        num_warps=8,
        num_stages=3,
    )
    if num_attempts < _TOP_P_REPAIR_NUM_ATTEMPTS:
        _top_p_parallel_repair_kernel[(rows,)](
            logits,
            req_pool_indices,
            temperature_pool,
            top_p_pool,
            seed_pool,
            offsets_pool,
            row_max,
            row_total,
            row_argmax,
            accepted,
            out,
            logits_row_stride=logits.stride(0),
            vocab_size=vocab_size,
            BLOCK_SIZE=block_size,
            START_ATTEMPT=num_attempts,
            NUM_ATTEMPTS_TOTAL=_TOP_P_REPAIR_NUM_ATTEMPTS,
            NUM_TOKENS_PER_REQ=num_tokens_per_req,
            num_warps=4,
            num_stages=3,
        )
    return out[:rows]

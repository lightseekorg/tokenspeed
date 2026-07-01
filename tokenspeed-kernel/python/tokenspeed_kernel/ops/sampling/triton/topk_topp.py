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

# Adapted from vLLM's MRV2/Qrita-style top-k/top-p Triton sampler:
#   https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/ops/topk_topp_triton.py
#   https://arxiv.org/abs/2602.01518
# TokenSpeed adapts the candidate/pivot ideas for candidate-space Gumbel-Max
# instead of vLLM's in-place logits masking API.

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

_TOP_K_TOP_P_BLOCK_SIZE = 2048
_TOP_K_TOP_P_PAD = 128
_QRITA_BLOCK_SIZE = 8192
_QRITA_BLOCK_SIZE_TRUNC = 4096
_QRITA_NUM_WARPS = 16

_QRITA_PERCENTILE_TO_STD_TABLE = [
    2.576,
    2.319,
    2.178,
    2.064,
    1.968,
    1.892,
    1.819,
    1.757,
    1.708,
    1.659,
    1.616,
    1.568,
    1.526,
    1.492,
    1.456,
    1.420,
    1.382,
    1.342,
    1.309,
    1.280,
    1.249,
    1.221,
    1.193,
    1.169,
    1.145,
    1.121,
    1.095,
    1.073,
    1.050,
    1.030,
    1.008,
    0.987,
    0.966,
    0.945,
    0.926,
    0.910,
    0.891,
    0.871,
    0.854,
    0.837,
    0.819,
    0.803,
    0.784,
    0.767,
    0.753,
    0.734,
    0.719,
    0.702,
    0.690,
    0.675,
    0.658,
    0.640,
    0.625,
    0.609,
    0.595,
    0.578,
    0.564,
    0.550,
    0.537,
    0.521,
    0.509,
    0.495,
    0.481,
    0.466,
    0.453,
    0.439,
    0.424,
    0.410,
    0.397,
    0.383,
    0.370,
    0.356,
    0.343,
    0.330,
    0.316,
    0.302,
    0.289,
    0.274,
    0.261,
    0.247,
    0.235,
    0.223,
    0.209,
    0.196,
    0.184,
    0.172,
    0.159,
    0.149,
    0.137,
    0.124,
    0.112,
    0.100,
    0.086,
    0.074,
    0.062,
    0.050,
    0.035,
    0.023,
    0.009,
    -0.003,
    -0.015,
    -0.027,
    -0.039,
    -0.052,
    -0.063,
    -0.074,
    -0.085,
    -0.097,
    -0.109,
    -0.122,
    -0.134,
    -0.147,
    -0.158,
    -0.171,
    -0.184,
    -0.196,
    -0.210,
    -0.223,
    -0.235,
    -0.248,
    -0.261,
    -0.275,
    -0.289,
    -0.302,
    -0.317,
    -0.328,
    -0.341,
    -0.353,
    -0.368,
    -0.382,
    -0.396,
    -0.410,
    -0.426,
    -0.439,
    -0.452,
    -0.465,
    -0.480,
    -0.493,
    -0.507,
    -0.521,
    -0.537,
    -0.551,
    -0.568,
    -0.582,
    -0.597,
    -0.614,
    -0.628,
    -0.643,
    -0.658,
    -0.673,
    -0.691,
    -0.706,
    -0.721,
    -0.738,
    -0.754,
    -0.769,
    -0.789,
    -0.808,
    -0.824,
    -0.838,
    -0.857,
    -0.877,
    -0.893,
    -0.912,
    -0.929,
    -0.947,
    -0.965,
    -0.983,
    -1.003,
    -1.027,
    -1.050,
    -1.070,
    -1.092,
    -1.117,
    -1.139,
    -1.162,
    -1.189,
    -1.216,
    -1.241,
    -1.272,
    -1.300,
    -1.330,
    -1.367,
    -1.404,
    -1.441,
    -1.485,
    -1.523,
    -1.564,
    -1.607,
    -1.658,
    -1.710,
    -1.778,
    -1.832,
    -1.901,
    -1.978,
    -2.068,
    -2.174,
    -2.325,
    -2.577,
    -3.813,
]


@triton.jit
def _top_k_top_p_candidates_stage1_kernel(
    logits_ptr,
    candidate_ids_ptr,
    candidate_logits_ptr,
    logits_row_stride: tl.constexpr,
    candidate_row_stride: tl.constexpr,
    vocab_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TOPK_PAD: tl.constexpr,
):
    row = tl.program_id(0)
    block_idx = tl.program_id(1)
    token_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = token_offsets < vocab_size

    vals = tl.load(
        logits_ptr + row * logits_row_stride + token_offsets,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    bits = vals.to(tl.uint32, bitcast=True)
    sign = bits & 0x80000000
    ordered_bits = tl.where(sign != 0, ~bits, bits ^ 0x80000000)
    ordered_bits = tl.where(mask, ordered_bits, 0)
    stable_id_key = (2147483647 - token_offsets).to(tl.uint32)
    packed = (ordered_bits.to(tl.uint64) << 31) | stable_id_key.to(tl.uint64)
    top_packed = tl.topk(packed, TOPK_PAD)

    top_ids = (2147483647 - (top_packed & 2147483647).to(tl.uint32)).to(tl.int32)
    top_valid = top_ids < vocab_size
    top_ordered_bits = (top_packed >> 31).to(tl.uint32)
    top_sign = top_ordered_bits & 0x80000000
    top_bits = tl.where(
        top_sign != 0,
        top_ordered_bits ^ 0x80000000,
        ~top_ordered_bits,
    )
    top_vals = top_bits.to(tl.float32, bitcast=True)
    top_vals = tl.where(top_valid, top_vals, float("-inf"))
    top_ids = tl.where(top_valid, top_ids, 2147483647)

    rank_offsets = tl.arange(0, TOPK_PAD)
    out_base = row * candidate_row_stride + block_idx * TOPK_PAD
    tl.store(candidate_logits_ptr + out_base + rank_offsets, top_vals)
    tl.store(candidate_ids_ptr + out_base + rank_offsets, top_ids)


@triton.jit
def _top_k_top_p_gumbel_stage2_kernel(
    candidate_ids_ptr,
    candidate_logits_ptr,
    req_pool_indices_ptr,
    temperature_pool_ptr,
    top_k_pool_ptr,
    top_p_pool_ptr,
    min_p_pool_ptr,
    seed_pool_ptr,
    offsets_pool_ptr,
    out_ptr,
    candidate_row_stride: tl.constexpr,
    num_blocks: tl.constexpr,
    vocab_size: tl.constexpr,
    TOPK_PAD: tl.constexpr,
    NUM_BLOCKS_PAD: tl.constexpr,
    NUM_TOKENS_PER_REQ: tl.constexpr,
):
    row = tl.program_id(0)
    req_row = row // NUM_TOKENS_PER_REQ
    spec_pos = row - req_row * NUM_TOKENS_PER_REQ
    pool_idx = tl.load(req_pool_indices_ptr + req_row)

    temperature = tl.maximum(
        tl.load(temperature_pool_ptr + pool_idx).to(tl.float32), 1.0e-20
    )
    top_k = tl.load(top_k_pool_ptr + pool_idx)
    top_k = tl.minimum(tl.maximum(top_k, 1), vocab_size)
    top_p = tl.load(top_p_pool_ptr + pool_idx).to(tl.float32)
    min_p_log_threshold = tl.full((), float("-inf"), tl.float32)
    if min_p_pool_ptr is not None:
        min_p = tl.load(min_p_pool_ptr + pool_idx).to(tl.float32)
        min_p_log_threshold = tl.log(tl.maximum(min_p, 1.0e-20))

    seed = tl.load(seed_pool_ptr + pool_idx).to(tl.int64)
    offset = tl.load(offsets_pool_ptr + pool_idx).to(tl.int64) + spec_pos
    gumbel_seed = tl.randint(seed, offset)

    rank_offsets = tl.arange(0, TOPK_PAD)
    block_offsets = tl.arange(0, NUM_BLOCKS_PAD)
    block_mask = block_offsets < num_blocks
    block_cursors = tl.full((NUM_BLOCKS_PAD,), 0, tl.int32)
    top_logits = tl.full((TOPK_PAD,), float("-inf"), tl.float32)
    top_ids = tl.full((TOPK_PAD,), 2147483647, tl.int32)

    # Stage 1 writes a sorted top-K list per vocab block. Merge those per-block
    # lists directly instead of repeatedly scanning the full candidate scratch.
    for rank in tl.range(0, TOPK_PAD):
        pos = block_offsets * TOPK_PAD + block_cursors
        active_blocks = block_mask & (block_cursors < TOPK_PAD)
        head_logits = tl.load(
            candidate_logits_ptr + row * candidate_row_stride + pos,
            mask=active_blocks,
            other=float("-inf"),
        ).to(tl.float32)
        head_ids = tl.load(
            candidate_ids_ptr + row * candidate_row_stride + pos,
            mask=active_blocks,
            other=2147483647,
        )

        best_logit = tl.max(head_logits, axis=0)
        best_id = tl.min(
            tl.where(head_logits == best_logit, head_ids, 2147483647), axis=0
        )
        best_block = tl.min(
            tl.where(
                (head_logits == best_logit) & (head_ids == best_id),
                block_offsets,
                2147483647,
            ),
            axis=0,
        )

        active_rank = rank < top_k
        top_logits = tl.where(
            (rank_offsets == rank) & active_rank, best_logit, top_logits
        )
        top_ids = tl.where((rank_offsets == rank) & active_rank, best_id, top_ids)
        block_cursors += tl.where(
            active_rank & (block_offsets == best_block),
            1,
            0,
        )

    scaled_logits = top_logits / temperature
    max_logit = tl.max(scaled_logits, axis=0)
    weights = tl.exp(scaled_logits - max_logit)
    denom = tl.maximum(tl.sum(weights, axis=0), 1.0e-20)
    probs = weights / denom

    cumulative_before = tl.cumsum(probs) - probs
    keep = (
        (rank_offsets < top_k)
        & (cumulative_before < top_p)
        & (scaled_logits >= max_logit + min_p_log_threshold)
    )

    rand_offsets = tl.where(keep, top_ids, 0)
    uniform = tl.maximum(tl.rand(gumbel_seed, rand_offsets), 1.0e-7)
    gumbel = -tl.log(-tl.log(uniform))
    scores = tl.where(keep, scaled_logits + gumbel, float("-inf"))

    max_score = tl.max(scores, axis=0)
    token_id = tl.min(tl.where(scores == max_score, top_ids, 2147483647), axis=0)
    tl.store(out_ptr + row, token_id)


def _check_top_k_top_p_gumbel_inputs(
    logits: torch.Tensor,
    req_pool_indices: torch.Tensor,
    temperature_pool: torch.Tensor,
    top_k_pool: torch.Tensor,
    top_p_pool: torch.Tensor,
    min_p_pool: torch.Tensor | None,
    seed_pool: torch.Tensor,
    offsets_pool: torch.Tensor,
    candidate_ids: torch.Tensor,
    candidate_logits: torch.Tensor,
    out: torch.Tensor,
    *,
    block_size: int,
    top_k_pad: int,
    num_tokens_per_req: int,
) -> tuple[int, int, int]:
    if logits.ndim != 2:
        raise ValueError(
            f"gumbel_sample_top_k_top_p_from_pools expects 2D logits, got {logits.ndim}D"
        )
    if logits.device.type != "cuda":
        raise ValueError("gumbel_sample_top_k_top_p_from_pools requires CUDA logits")
    if logits.stride(-1) != 1:
        raise ValueError(
            "gumbel_sample_top_k_top_p_from_pools requires stride-1 vocab dimension, "
            f"got stride={logits.stride()}"
        )
    rows, vocab_size = logits.shape
    if vocab_size <= 0:
        raise ValueError(
            "gumbel_sample_top_k_top_p_from_pools requires non-empty vocab"
        )

    for name, tensor, ndim in (
        ("req_pool_indices", req_pool_indices, 1),
        ("temperature_pool", temperature_pool, 1),
        ("top_k_pool", top_k_pool, 1),
        ("top_p_pool", top_p_pool, 1),
        ("seed_pool", seed_pool, 1),
        ("offsets_pool", offsets_pool, 1),
        ("candidate_ids", candidate_ids, 2),
        ("candidate_logits", candidate_logits, 2),
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
    if req_pool_indices.dtype != torch.int32:
        raise ValueError(
            f"req_pool_indices must be int32, got {req_pool_indices.dtype}"
        )
    if top_k_pool.dtype != torch.int32:
        raise ValueError(f"top_k_pool must be int32, got {top_k_pool.dtype}")
    if min_p_pool is not None:
        if min_p_pool.device.type != "cuda":
            raise ValueError("min_p_pool must be CUDA")
        if min_p_pool.ndim != 1:
            raise ValueError(f"min_p_pool must be 1D, got {min_p_pool.ndim}D")
    if seed_pool.dtype != torch.int64:
        raise ValueError(f"seed_pool must be int64, got {seed_pool.dtype}")
    if candidate_ids.dtype != torch.int32:
        raise ValueError(f"candidate_ids must be int32, got {candidate_ids.dtype}")
    if candidate_logits.dtype != torch.float32:
        raise ValueError(
            f"candidate_logits must be float32, got {candidate_logits.dtype}"
        )
    if out.dtype != torch.int32:
        raise ValueError(f"out must be int32, got {out.dtype}")
    if out.shape[0] < rows:
        raise ValueError(f"out is too small: {out.shape[0]} < {rows}")
    if candidate_ids.shape[0] < rows or candidate_logits.shape[0] < rows:
        raise ValueError("candidate scratch tensors have too few rows")
    if candidate_ids.stride(-1) != 1 or candidate_logits.stride(-1) != 1:
        raise ValueError("candidate scratch tensors require stride-1 dimension")

    num_blocks = triton.cdiv(vocab_size, block_size)
    num_candidates = num_blocks * top_k_pad
    if candidate_ids.shape[1] < num_candidates:
        raise ValueError(
            f"candidate_ids is too small: need {num_candidates}, got {candidate_ids.shape[1]}"
        )
    if candidate_logits.shape[1] < num_candidates:
        raise ValueError(
            "candidate_logits is too small: "
            f"need {num_candidates}, got {candidate_logits.shape[1]}"
        )
    return rows, vocab_size, num_blocks


def gumbel_sample_top_k_top_p_from_pools(
    logits: torch.Tensor,
    req_pool_indices: torch.Tensor,
    temperature_pool: torch.Tensor,
    top_k_pool: torch.Tensor,
    top_p_pool: torch.Tensor,
    seed_pool: torch.Tensor,
    offsets_pool: torch.Tensor,
    candidate_ids: torch.Tensor,
    candidate_logits: torch.Tensor,
    out: torch.Tensor,
    *,
    min_p_pool: torch.Tensor | None = None,
    block_size: int = _TOP_K_TOP_P_BLOCK_SIZE,
    top_k_pad: int = _TOP_K_TOP_P_PAD,
    num_tokens_per_req: int = 1,
) -> torch.Tensor:
    """Sample finite top-k/top-p candidates with Gumbel-Max."""
    if top_k_pad & (top_k_pad - 1):
        raise ValueError("top_k_pad must be a power of two for tl.topk")
    rows, vocab_size, num_blocks = _check_top_k_top_p_gumbel_inputs(
        logits,
        req_pool_indices,
        temperature_pool,
        top_k_pool,
        top_p_pool,
        min_p_pool,
        seed_pool,
        offsets_pool,
        candidate_ids,
        candidate_logits,
        out,
        block_size=block_size,
        top_k_pad=top_k_pad,
        num_tokens_per_req=num_tokens_per_req,
    )
    if rows == 0:
        return out[:0]

    stage1_num_warps = 1 if block_size <= 1024 else 8
    stage2_num_warps = 1 if block_size <= 1024 else 4

    _top_k_top_p_candidates_stage1_kernel[(rows, num_blocks)](
        logits,
        candidate_ids,
        candidate_logits,
        logits_row_stride=logits.stride(0),
        candidate_row_stride=candidate_ids.stride(0),
        vocab_size=vocab_size,
        BLOCK_SIZE=block_size,
        TOPK_PAD=top_k_pad,
        num_warps=stage1_num_warps,
    )
    _top_k_top_p_gumbel_stage2_kernel[(rows,)](
        candidate_ids,
        candidate_logits,
        req_pool_indices,
        temperature_pool,
        top_k_pool,
        top_p_pool,
        min_p_pool,
        seed_pool,
        offsets_pool,
        out,
        candidate_row_stride=candidate_ids.stride(0),
        num_blocks=num_blocks,
        vocab_size=vocab_size,
        TOPK_PAD=top_k_pad,
        NUM_BLOCKS_PAD=triton.next_power_of_2(num_blocks),
        NUM_TOKENS_PER_REQ=num_tokens_per_req,
        num_warps=stage2_num_warps,
        num_stages=3,
    )
    return out[:rows]


@triton.jit
def _qrita_update_min_larger_stats(
    data,
    above_mask,
    min_larger,
    num_min_larger,
    sentinel,
):
    tile_min = tl.min(tl.where(above_mask, data, sentinel))
    tile_eq = above_mask & (tl.abs(data - tile_min) < 1.0e-9)
    tile_cnt = tl.sum(tile_eq)
    is_new = tile_min < min_larger
    is_same = tl.abs(tile_min - min_larger) < 1.0e-9
    num_min_larger = tl.where(is_new, tile_cnt, num_min_larger + tile_cnt * is_same)
    min_larger = tl.minimum(min_larger, tile_min)
    return min_larger, num_min_larger


@triton.jit
def _top_k_top_p_qrita_gumbel_kernel(
    logits_ptr,
    req_pool_indices_ptr,
    temperature_pool_ptr,
    top_k_pool_ptr,
    top_p_pool_ptr,
    seed_pool_ptr,
    offsets_pool_ptr,
    qrita_buffer_ptr,
    percentile_to_std_table_ptr,
    out_ptr,
    logits_row_stride: tl.constexpr,
    qrita_buffer_row_stride: tl.constexpr,
    batch_size: tl.constexpr,
    vocab_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_TRUNC: tl.constexpr,
    NUM_TOKENS_PER_REQ: tl.constexpr,
):
    num_tiles: tl.constexpr = (vocab_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    for row in tl.range(pid, batch_size, num_programs):
        req_row = row // NUM_TOKENS_PER_REQ
        spec_pos = row - req_row * NUM_TOKENS_PER_REQ
        pool_idx = tl.load(req_pool_indices_ptr + req_row)
        temperature = tl.maximum(
            tl.load(temperature_pool_ptr + pool_idx).to(tl.float32), 1.0e-20
        )
        k = tl.minimum(tl.maximum(tl.load(top_k_pool_ptr + pool_idx), 1), vocab_size)
        p = tl.load(top_p_pool_ptr + pool_idx).to(tl.float32)
        seed = tl.load(seed_pool_ptr + pool_idx).to(tl.int64)
        offset = tl.load(offsets_pool_ptr + pool_idx).to(tl.int64) + spec_pos
        gumbel_seed = tl.randint(seed, offset)

        logits_row = logits_ptr + row * logits_row_stride
        buffer_row = qrita_buffer_ptr + pid * qrita_buffer_row_stride

        final_pivot = -float("inf")
        duplicate_logit = float("inf")
        num_duplicate_logit = tl.zeros((), dtype=tl.uint32)
        num_keep = tl.zeros((), dtype=tl.uint32)
        num_kept = tl.zeros((), dtype=tl.uint32)
        max_logit = -float("inf")
        min_logit = float("inf")
        row_argmax = tl.full((), 2147483647, tl.int32)

        if k < vocab_size:
            offs = tl.arange(0, BLOCK_SIZE)
            mask_n = offs < vocab_size
            logits_blk0 = tl.load(
                logits_row + offs, mask=mask_n, other=-float("inf")
            ).to(tl.float32)
            logits_blk0 = logits_blk0 / temperature
            finite_mask = (logits_blk0 > -float("inf")) & mask_n
            num_finite = tl.sum(finite_mask)
            finite_logits = tl.where(finite_mask, logits_blk0, 0.0)
            avg_logit = tl.where(
                num_finite > 0, tl.sum(finite_logits) / num_finite, 0.0
            )
            sq_avg_logit = tl.where(
                num_finite > 0,
                tl.sum(finite_logits * finite_logits) / num_finite,
                0.0,
            )
            std_logit = tl.sqrt(tl.maximum(sq_avg_logit - avg_logit * avg_logit, 0.0))

            percentile = tl.cast(k / vocab_size * 200, tl.uint32)
            percentile = tl.minimum(percentile, 199)
            sigma = tl.load(percentile_to_std_table_ptr + percentile)
            sigma = sigma + tl.abs(sigma) * -0.15
            outlier_pivot = avg_logit + std_logit * sigma
            num_outliers = tl.zeros((), dtype=tl.uint32)
            num_finite_total = tl.zeros((), dtype=tl.uint32)

            for i in range(0, num_tiles):
                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask_n = offs_n < vocab_size
                logits_blk = tl.load(
                    logits_row + offs_n, mask=mask_n, other=-float("inf")
                ).to(tl.float32)
                logits_blk = logits_blk / temperature

                block_max = tl.max(logits_blk)
                block_argmax = tl.min(
                    tl.where(logits_blk == block_max, offs_n, 2147483647)
                )
                better_max = (block_max > max_logit) | (
                    (block_max == max_logit) & (block_argmax < row_argmax)
                )
                max_logit = tl.where(better_max, block_max, max_logit)
                row_argmax = tl.where(better_max, block_argmax, row_argmax)

                finite_blk_mask = logits_blk > -float("inf")
                finite_blk = tl.where(finite_blk_mask, logits_blk, float("inf"))
                min_logit = tl.minimum(min_logit, tl.min(finite_blk))
                num_finite_total += tl.sum(finite_blk_mask & mask_n)

                outlier_mask = (logits_blk > outlier_pivot) & mask_n
                cumulative_pos = tl.cast(
                    tl.cumsum(outlier_mask) - 1 + num_outliers, tl.int32
                )
                num_outliers += tl.sum(outlier_mask)
                write_pos = tl.where(outlier_mask, cumulative_pos, -1)
                tl.store(buffer_row + write_pos, logits_blk, mask=outlier_mask)

            min_logit = tl.minimum(min_logit, max_logit)

            num_iters = 0
            k_pivot = float("inf")
            k_pivots_num = tl.zeros((), dtype=tl.uint32)
            min_larger = float("inf")
            num_min_larger = tl.zeros((), dtype=tl.uint32)
            if num_outliers > k:
                max_range = max_logit
                min_range = outlier_pivot
                search_range = tl.cast(num_outliers, tl.int32)
                search_iters = tl.cast(
                    (num_outliers + BLOCK_SIZE_TRUNC - 1) // BLOCK_SIZE_TRUNC,
                    tl.int32,
                )
                found_pivot = 0
                while found_pivot == 0:
                    k_pivot_0 = (max_range - min_range) * 1.0 / 3.0 + min_range
                    k_pivots_num_0 = tl.zeros((), dtype=tl.uint32)
                    min_larger_0 = float("inf")
                    num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

                    k_pivot_1 = (max_range - min_range) * 2.0 / 3.0 + min_range
                    k_pivots_num_1 = tl.zeros((), dtype=tl.uint32)
                    min_larger_1 = float("inf")
                    num_min_larger_1 = tl.zeros((), dtype=tl.uint32)

                    for i in range(0, search_iters):
                        offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(0, BLOCK_SIZE_TRUNC)
                        mask_n_2 = offs_n < search_range
                        logits_blk2 = tl.load(
                            buffer_row + offs_n, mask=mask_n_2, other=-float("inf")
                        )
                        above_0 = logits_blk2 > k_pivot_0
                        above_1 = logits_blk2 > k_pivot_1
                        k_pivots_num_0 += tl.sum(above_0)
                        k_pivots_num_1 += tl.sum(above_1)
                        min_larger_0, num_min_larger_0 = _qrita_update_min_larger_stats(
                            logits_blk2,
                            above_0,
                            min_larger_0,
                            num_min_larger_0,
                            float("inf"),
                        )
                        min_larger_1, num_min_larger_1 = _qrita_update_min_larger_stats(
                            logits_blk2,
                            above_1,
                            min_larger_1,
                            num_min_larger_1,
                            float("inf"),
                        )

                    if k_pivots_num_0 >= k and k_pivots_num_0 - num_min_larger_0 < k:
                        k_pivot = k_pivot_0
                        k_pivots_num = k_pivots_num_0
                        min_larger = min_larger_0
                        num_min_larger = num_min_larger_0
                        found_pivot = 1
                    if k_pivots_num_1 >= k and k_pivots_num_1 - num_min_larger_1 < k:
                        k_pivot = k_pivot_1
                        k_pivots_num = k_pivots_num_1
                        min_larger = min_larger_1
                        num_min_larger = num_min_larger_1
                        found_pivot = 1

                    if k_pivots_num_1 > k:
                        min_range = k_pivot_1
                    elif k_pivots_num_0 > k:
                        min_range = k_pivot_0
                    if k_pivots_num_0 < k:
                        max_range = k_pivot_0
                    elif k_pivots_num_1 < k:
                        max_range = k_pivot_1

                    num_iters += 1
                    if num_iters >= 18 or tl.abs(min_range - max_range) < 1.0e-9:
                        k_pivot = (max_range + min_range) / 2.0
                        found_pivot = 1
            else:
                max_range = max_logit
                min_range = min_logit
                found_pivot = 0
                while found_pivot == 0:
                    k_pivot_0 = (max_range - min_range) * 1.0 / 4.0 + min_range
                    k_pivots_num_0 = tl.zeros((), dtype=tl.uint32)
                    min_larger_0 = float("inf")
                    num_min_larger_0 = tl.zeros((), dtype=tl.uint32)
                    k_pivot_1 = (max_range - min_range) * 2.0 / 4.0 + min_range
                    k_pivots_num_1 = tl.zeros((), dtype=tl.uint32)
                    min_larger_1 = float("inf")
                    num_min_larger_1 = tl.zeros((), dtype=tl.uint32)

                    for i in range(0, num_tiles):
                        offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                        mask_n = offs_n < vocab_size
                        logits_blk2 = tl.load(
                            logits_row + offs_n, mask=mask_n, other=-float("inf")
                        ).to(tl.float32)
                        logits_blk2 = logits_blk2 / temperature
                        above_0 = logits_blk2 > k_pivot_0
                        above_1 = logits_blk2 > k_pivot_1
                        k_pivots_num_0 += tl.sum(above_0)
                        k_pivots_num_1 += tl.sum(above_1)
                        min_larger_0, num_min_larger_0 = _qrita_update_min_larger_stats(
                            logits_blk2,
                            above_0,
                            min_larger_0,
                            num_min_larger_0,
                            float("inf"),
                        )
                        min_larger_1, num_min_larger_1 = _qrita_update_min_larger_stats(
                            logits_blk2,
                            above_1,
                            min_larger_1,
                            num_min_larger_1,
                            float("inf"),
                        )

                    if k_pivots_num_0 >= k and k_pivots_num_0 - num_min_larger_0 < k:
                        k_pivot = k_pivot_0
                        k_pivots_num = k_pivots_num_0
                        min_larger = min_larger_0
                        num_min_larger = num_min_larger_0
                        found_pivot = 1
                    if k_pivots_num_1 >= k and k_pivots_num_1 - num_min_larger_1 < k:
                        k_pivot = k_pivot_1
                        k_pivots_num = k_pivots_num_1
                        min_larger = min_larger_1
                        num_min_larger = num_min_larger_1
                        found_pivot = 1

                    if k_pivots_num_1 > k:
                        min_range = k_pivot_1
                    elif k_pivots_num_0 > k:
                        min_range = k_pivot_0
                    if k_pivots_num_0 < k:
                        max_range = k_pivot_0
                    elif k_pivots_num_1 < k:
                        max_range = k_pivot_1

                    num_iters += 1
                    if num_iters >= 18 or tl.abs(min_range - max_range) < 1.0e-9:
                        k_pivot = (max_range + min_range) / 2.0
                        found_pivot = 1

            duplicate_logit = min_larger
            num_duplicate_logit = num_min_larger
            num_keep = num_duplicate_logit - (k_pivots_num - k)
            num_kept = tl.zeros((), dtype=tl.uint32)
            final_pivot = k_pivot if num_finite_total > k else -float("inf")

            if p < 1.0 and num_finite_total > k:
                min_logit = k_pivot
                sum_exp_logits = 0.0
                num_outliers_2 = tl.zeros((), dtype=tl.uint32)
                search_range = tl.cast(num_outliers, tl.int32)
                search_iters = tl.cast(
                    (num_outliers + BLOCK_SIZE_TRUNC - 1) // BLOCK_SIZE_TRUNC,
                    tl.int32,
                )

                if num_outliers > k:
                    for i in range(0, search_iters):
                        offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(0, BLOCK_SIZE_TRUNC)
                        mask_n_2 = offs_n < search_range
                        probs_blk = tl.load(
                            buffer_row + offs_n, mask=mask_n_2, other=-float("inf")
                        )
                        outlier_mask = (probs_blk > min_logit) & mask_n_2
                        if num_keep < num_duplicate_logit:
                            duplicate_mask = (
                                tl.abs(probs_blk - duplicate_logit) < 1.0e-9
                            )
                            duplicate_count = tl.cumsum(duplicate_mask) + num_kept
                            duplicate_keep_mask = (
                                duplicate_count <= num_keep
                            ) & duplicate_mask
                            duplicate_remove_mask = (
                                duplicate_mask & ~duplicate_keep_mask
                            )
                            outlier_mask = outlier_mask & (~duplicate_remove_mask)
                            num_kept += tl.sum(duplicate_keep_mask)
                        probs_blk = tl.where(outlier_mask, probs_blk, -float("inf"))
                        probs_blk = tl.exp(probs_blk - max_logit)
                        sum_exp_logits += tl.sum(probs_blk)

                    for i in range(0, search_iters):
                        offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(0, BLOCK_SIZE_TRUNC)
                        mask_n_2 = offs_n < search_range
                        probs_blk = tl.load(
                            buffer_row + offs_n, mask=mask_n_2, other=-float("inf")
                        )
                        probs_blk = tl.exp(probs_blk - max_logit)
                        probs_blk = probs_blk / sum_exp_logits
                        tl.store(buffer_row + offs_n, probs_blk, mask=mask_n_2)
                else:
                    num_kept = tl.zeros((), dtype=tl.uint32)
                    for i in range(0, num_tiles):
                        offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                        mask_n = offs_n < vocab_size
                        probs_blk = tl.load(
                            logits_row + offs_n, mask=mask_n, other=-float("inf")
                        ).to(tl.float32)
                        probs_blk = probs_blk / temperature
                        outlier_mask = (probs_blk > min_logit) & mask_n
                        duplicate_mask = tl.abs(probs_blk - duplicate_logit) < 1.0e-9
                        duplicate_count = tl.cumsum(duplicate_mask) + num_kept
                        duplicate_keep_mask = (
                            duplicate_count <= num_keep
                        ) & duplicate_mask
                        duplicate_remove_mask = duplicate_mask & ~duplicate_keep_mask
                        outlier_mask = outlier_mask & (~duplicate_remove_mask)
                        num_kept += tl.sum(duplicate_keep_mask)

                        probs_blk = tl.where(outlier_mask, probs_blk, -float("inf"))
                        probs_blk = tl.exp(probs_blk - max_logit)
                        sum_exp_logits += tl.sum(probs_blk)
                        cumulative_pos = tl.cast(
                            tl.cumsum(outlier_mask) - 1 + num_outliers_2, tl.int32
                        )
                        num_outliers_2 += tl.sum(outlier_mask)
                        write_pos = tl.where(outlier_mask, cumulative_pos, -1)
                        tl.store(buffer_row + write_pos, probs_blk, mask=outlier_mask)

                    search_range = tl.cast(num_outliers_2, tl.int32)
                    search_iters = tl.cast(
                        (num_outliers_2 + BLOCK_SIZE_TRUNC - 1) // BLOCK_SIZE_TRUNC,
                        tl.int32,
                    )
                    for i in range(0, search_iters):
                        offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(0, BLOCK_SIZE_TRUNC)
                        mask_n_2 = offs_n < search_range
                        probs_blk = tl.load(
                            buffer_row + offs_n, mask=mask_n_2, other=0.0
                        )
                        probs_blk = probs_blk / sum_exp_logits
                        tl.store(buffer_row + offs_n, probs_blk, mask=mask_n_2)

                max_range = 1.0 / sum_exp_logits
                min_range = tl.exp(min_logit - max_logit) / sum_exp_logits
                p_pivot = 1.0
                num_iters = 0
                min_larger_prob = 1.0
                num_min_larger = tl.zeros((), dtype=tl.uint32)
                p_pivots_sum = 0.0
                found_pivot = 0
                while found_pivot == 0:
                    p_pivot_0 = (max_range - min_range) * 1.0 / 3.0 + min_range
                    p_pivots_sum_0 = 0.0
                    min_larger_0 = 1.0
                    num_min_larger_0 = tl.zeros((), dtype=tl.uint32)
                    p_pivot_1 = (max_range - min_range) * 2.0 / 3.0 + min_range
                    p_pivots_sum_1 = 0.0
                    min_larger_1 = 1.0
                    num_min_larger_1 = tl.zeros((), dtype=tl.uint32)

                    for i in range(0, search_iters):
                        offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(0, BLOCK_SIZE_TRUNC)
                        mask_n_2 = offs_n < search_range
                        probs_blk = tl.load(
                            buffer_row + offs_n, mask=mask_n_2, other=0.0
                        )
                        above_0 = probs_blk > p_pivot_0
                        above_1 = probs_blk > p_pivot_1
                        p_pivots_sum_0 += tl.sum(probs_blk * above_0)
                        p_pivots_sum_1 += tl.sum(probs_blk * above_1)
                        masked_larger_0 = tl.where(above_0, probs_blk, 1.0)
                        masked_larger_1 = tl.where(above_1, probs_blk, 1.0)
                        min_larger_0 = tl.minimum(min_larger_0, tl.min(masked_larger_0))
                        min_larger_1 = tl.minimum(min_larger_1, tl.min(masked_larger_1))

                    for i in range(0, search_iters):
                        offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(0, BLOCK_SIZE_TRUNC)
                        mask_n_2 = offs_n < search_range
                        probs_blk = tl.load(
                            buffer_row + offs_n, mask=mask_n_2, other=0.0
                        )
                        num_min_larger_0 += tl.sum(
                            tl.abs(probs_blk - min_larger_0) < 1.0e-9
                        )
                        num_min_larger_1 += tl.sum(
                            tl.abs(probs_blk - min_larger_1) < 1.0e-9
                        )

                    if p_pivots_sum_1 >= p and (
                        p_pivots_sum_1 - (min_larger_1 * num_min_larger_1) < p
                    ):
                        p_pivot = p_pivot_1
                        min_larger_prob = min_larger_1
                        num_min_larger = num_min_larger_1
                        p_pivots_sum = p_pivots_sum_1
                        found_pivot = 1
                    if p_pivots_sum_0 >= p and (
                        p_pivots_sum_0 - (min_larger_0 * num_min_larger_0) < p
                    ):
                        p_pivot = p_pivot_0
                        min_larger_prob = min_larger_0
                        num_min_larger = num_min_larger_0
                        p_pivots_sum = p_pivots_sum_0
                        found_pivot = 1

                    if p_pivots_sum_1 > p:
                        min_range = p_pivot_1
                    elif p_pivots_sum_0 > p:
                        min_range = p_pivot_0
                    if p_pivots_sum_0 < p:
                        max_range = p_pivot_0
                    elif p_pivots_sum_1 < p:
                        max_range = p_pivot_1

                    num_iters += 1
                    if (max_range - min_range) < 1.0e-9 or num_iters >= 18:
                        p_pivot = (max_range + min_range) / 2.0
                        found_pivot = 1

                duplicate_logit = tl.log(min_larger_prob * sum_exp_logits) + max_logit
                num_duplicate_logit = num_min_larger
                num_keep = num_duplicate_logit - tl.cast(
                    (p_pivots_sum - p) / min_larger_prob, tl.uint32
                )
                num_kept = tl.zeros((), dtype=tl.uint32)
                final_pivot = tl.log(p_pivot * sum_exp_logits) + max_logit

        if not (final_pivot < max_logit):
            final_pivot = -float("inf")

        best_score = tl.full((), float("-inf"), tl.float32)
        best_id = tl.full((), 2147483647, tl.int32)
        num_kept = tl.zeros((), dtype=tl.uint32)
        for i in range(0, num_tiles):
            offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask_n = offs_n < vocab_size
            logits_blk = tl.load(
                logits_row + offs_n, mask=mask_n, other=-float("inf")
            ).to(tl.float32)
            logits_blk = logits_blk / temperature
            keep_mask = (logits_blk > final_pivot) & mask_n

            if num_keep < num_duplicate_logit:
                duplicate_mask = (
                    tl.abs(logits_blk - duplicate_logit) < 1.0e-9
                ) & mask_n
                duplicate_count = tl.cumsum(duplicate_mask) + num_kept
                duplicate_keep_mask = (duplicate_count <= num_keep) & duplicate_mask
                duplicate_remove_mask = duplicate_mask & ~duplicate_keep_mask
                num_kept += tl.sum(duplicate_keep_mask)
                keep_mask = keep_mask & (~duplicate_remove_mask)

            uniform = tl.maximum(tl.rand(gumbel_seed, offs_n), 1.0e-7)
            gumbel = -tl.log(-tl.log(uniform))
            scores = tl.where(keep_mask, logits_blk + gumbel, float("-inf"))
            block_score = tl.max(scores)
            block_id = tl.min(tl.where(scores == block_score, offs_n, 2147483647))
            better = (block_score > best_score) | (
                (block_score == best_score) & (block_id < best_id)
            )
            best_score = tl.where(better, block_score, best_score)
            best_id = tl.where(better, block_id, best_id)

        best_id = tl.where(best_id == 2147483647, row_argmax, best_id)
        tl.store(out_ptr + row, best_id)


def _check_top_k_top_p_qrita_inputs(
    logits: torch.Tensor,
    req_pool_indices: torch.Tensor,
    temperature_pool: torch.Tensor,
    top_k_pool: torch.Tensor,
    top_p_pool: torch.Tensor,
    seed_pool: torch.Tensor,
    offsets_pool: torch.Tensor,
    qrita_buffer: torch.Tensor,
    percentile_to_std_table: torch.Tensor,
    out: torch.Tensor,
    *,
    num_tokens_per_req: int,
    num_programs: int,
) -> tuple[int, int]:
    if logits.ndim != 2:
        raise ValueError(
            f"gumbel_sample_top_k_top_p_qrita_from_pools expects 2D logits, got {logits.ndim}D"
        )
    if logits.device.type != "cuda":
        raise ValueError(
            "gumbel_sample_top_k_top_p_qrita_from_pools requires CUDA logits"
        )
    if logits.stride(-1) != 1:
        raise ValueError(
            "gumbel_sample_top_k_top_p_qrita_from_pools requires stride-1 vocab dimension, "
            f"got stride={logits.stride()}"
        )
    rows, vocab_size = logits.shape
    if vocab_size <= 0:
        raise ValueError(
            "gumbel_sample_top_k_top_p_qrita_from_pools requires non-empty vocab"
        )

    for name, tensor, ndim in (
        ("req_pool_indices", req_pool_indices, 1),
        ("temperature_pool", temperature_pool, 1),
        ("top_k_pool", top_k_pool, 1),
        ("top_p_pool", top_p_pool, 1),
        ("seed_pool", seed_pool, 1),
        ("offsets_pool", offsets_pool, 1),
        ("qrita_buffer", qrita_buffer, 2),
        ("percentile_to_std_table", percentile_to_std_table, 1),
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
    if req_pool_indices.dtype != torch.int32:
        raise ValueError(
            f"req_pool_indices must be int32, got {req_pool_indices.dtype}"
        )
    if top_k_pool.dtype != torch.int32:
        raise ValueError(f"top_k_pool must be int32, got {top_k_pool.dtype}")
    if seed_pool.dtype != torch.int64:
        raise ValueError(f"seed_pool must be int64, got {seed_pool.dtype}")
    if qrita_buffer.dtype != torch.float32:
        raise ValueError(f"qrita_buffer must be float32, got {qrita_buffer.dtype}")
    if percentile_to_std_table.dtype != torch.float32:
        raise ValueError(
            "percentile_to_std_table must be float32, "
            f"got {percentile_to_std_table.dtype}"
        )
    if percentile_to_std_table.shape[0] < len(_QRITA_PERCENTILE_TO_STD_TABLE):
        raise ValueError("percentile_to_std_table is too small")
    if out.dtype != torch.int32:
        raise ValueError(f"out must be int32, got {out.dtype}")
    if out.shape[0] < rows:
        raise ValueError(f"out is too small: {out.shape[0]} < {rows}")
    if qrita_buffer.shape[0] < num_programs or qrita_buffer.shape[1] < vocab_size:
        raise ValueError(
            "qrita_buffer must be at least "
            f"({num_programs}, {vocab_size}), got {tuple(qrita_buffer.shape)}"
        )
    if qrita_buffer.stride(-1) != 1:
        raise ValueError("qrita_buffer requires stride-1 vocab dimension")
    return rows, vocab_size


def gumbel_sample_top_k_top_p_qrita_from_pools(
    logits: torch.Tensor,
    req_pool_indices: torch.Tensor,
    temperature_pool: torch.Tensor,
    top_k_pool: torch.Tensor,
    top_p_pool: torch.Tensor,
    seed_pool: torch.Tensor,
    offsets_pool: torch.Tensor,
    qrita_buffer: torch.Tensor,
    percentile_to_std_table: torch.Tensor,
    out: torch.Tensor,
    *,
    num_tokens_per_req: int = 1,
    num_programs: int | None = None,
) -> torch.Tensor:
    """Sample finite top-k/top-p rows using Qrita-style pivots."""
    rows = logits.shape[0]
    if num_programs is None:
        num_sms = torch.cuda.get_device_properties(logits.device).multi_processor_count
        num_programs = min(num_sms, max(int(rows), 1))
    rows, vocab_size = _check_top_k_top_p_qrita_inputs(
        logits,
        req_pool_indices,
        temperature_pool,
        top_k_pool,
        top_p_pool,
        seed_pool,
        offsets_pool,
        qrita_buffer,
        percentile_to_std_table,
        out,
        num_tokens_per_req=num_tokens_per_req,
        num_programs=num_programs,
    )
    if rows == 0:
        return out[:0]

    _top_k_top_p_qrita_gumbel_kernel[(num_programs,)](
        logits,
        req_pool_indices,
        temperature_pool,
        top_k_pool,
        top_p_pool,
        seed_pool,
        offsets_pool,
        qrita_buffer,
        percentile_to_std_table,
        out,
        logits_row_stride=logits.stride(0),
        qrita_buffer_row_stride=qrita_buffer.stride(0),
        batch_size=rows,
        vocab_size=vocab_size,
        BLOCK_SIZE=_QRITA_BLOCK_SIZE,
        BLOCK_SIZE_TRUNC=_QRITA_BLOCK_SIZE_TRUNC,
        NUM_TOKENS_PER_REQ=num_tokens_per_req,
        num_warps=_QRITA_NUM_WARPS,
        num_stages=3,
    )
    return out[:rows]

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

"""Rejection-sampling verify of a sampled draft chain on the no-filter route.

Row ``r`` of a request's verify window carries draft token ``x = x_{r+1}``,
drawn from ``q`` over ``K`` recorded candidates; the target distribution is
``p = softmax(logits / temperature)``. The row accepts ``x`` when
``coin * q(x) < p(x)``. Otherwise it emits a draw from ``max(p - q, 0)``
(renormalized), made by Gumbel-max with the plain sampler's noise: a
non-candidate token scores ``logit / temperature + G`` exactly as the plain
sampler does, a candidate ``log(max(p - q, 0)) + LSE + G``. Rows without a
recorded ``q`` (and the last row) emit the plain draw and accept ``x`` when
the draw equals it.

Noise streams for a request with seed ``s`` and verify offset ``o``:

* plain draw at row ``r``: ``rand(randint(s, o + r), token_id)``;
* coin at row ``r``: ``rand(randint(s, COIN_OFFSET_SALT + o + r), 0)``;
* draft step ``l``: ``rand(randint(s, DRAFT_SAMPLE_OFFSET_SALT + base + l),
  token_id)``.

Plain offsets stay below ``2**40`` and draft offsets below ``2**41``, so the
three streams never share a draw.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.sampling.triton.common import check_draft_q_inputs

__all__ = [
    "COIN_OFFSET_SALT",
    "DRAFT_SAMPLE_OFFSET_SALT",
    "verify_rows_no_filter_from_pools",
    "verify_rows_no_filter_scratch_shape",
]

# Philox offset salts of the draft and coin streams (see the module docstring).
DRAFT_SAMPLE_OFFSET_SALT = 1 << 40
COIN_OFFSET_SALT = 1 << 41
# Top-p within this distance of 1 means no top-p cutoff.
TOP_P_ONE_EPS = 1.0e-6

_REJECTION_BLOCK_SIZE = 1024
# Per-block statistics stage 1 leaves for stage 2: the max of logits / T, the
# sum of exp(logits / T - max) and the best plain score (fp32), and the token
# holding that score (int32).
_NUM_FLOAT_STATS = 3


@triton.jit
def _plain_gumbel(seed, offset, token_ids):
    """Gumbel noise of the plain sampler's stream at ``token_ids``."""
    uniform = tl.maximum(tl.rand(tl.randint(seed, offset), token_ids), 1.0e-7)
    return -tl.log(-tl.log(uniform))


@triton.jit
def _coin(seed, offset, COIN_SALT: tl.constexpr):
    """The accept coin in ``[0, 1)`` of the row at verify offset ``offset``."""
    return tl.rand(tl.randint(seed, COIN_SALT + offset), tl.full((), 0, tl.int32))


@triton.jit
def _row_draft_q(
    req_row,
    spec_pos,
    pool_idx,
    slot_ptr,
    pos_ptr,
    q_ids_ptr,
    candidates_ptr,
    slot_pool_stride: tl.constexpr,
    q_pool_stride: tl.constexpr,
    q_slot_stride: tl.constexpr,
    candidates_row_stride: tl.constexpr,
    candidates_col_stride: tl.constexpr,
    vocab_size: tl.constexpr,
    NUM_TOKENS_PER_REQ: tl.constexpr,
    NUM_CANDIDATES: tl.constexpr,
):
    """Whether the row verifies against a recorded q, and that q row's offset.

    The row needs a slot, and its recorded lane must hold the draft token;
    otherwise the draft did not come from that q and the row falls back to
    the target-only rule.
    """
    draft_row = spec_pos < NUM_TOKENS_PER_REQ - 1
    node = pool_idx * slot_pool_stride + spec_pos + 1
    slot = tl.load(slot_ptr + node, mask=draft_row, other=-1)
    lane = tl.load(pos_ptr + node, mask=draft_row, other=-1)
    draft = tl.load(
        candidates_ptr
        + req_row * candidates_row_stride
        + (spec_pos + 1) * candidates_col_stride,
        mask=draft_row,
        other=-1,
    )
    q_base = pool_idx * q_pool_stride + tl.maximum(slot, 0).to(tl.int64) * q_slot_stride
    lane_ok = (slot >= 0) & (lane >= 0) & (lane < NUM_CANDIDATES)
    recorded = tl.load(q_ids_ptr + q_base + tl.maximum(lane, 0), mask=lane_ok, other=-1)
    in_vocab = (draft >= 0) & (draft < vocab_size)
    return lane_ok & in_vocab & (recorded == draft), q_base


@triton.jit
def _rejection_no_filter_stage1_kernel(
    logits_ptr,
    req_pool_indices_ptr,
    temperature_pool_ptr,
    seed_pool_ptr,
    offsets_pool_ptr,
    slot_ptr,
    pos_ptr,
    q_ids_ptr,
    candidates_ptr,
    block_stats_ptr,
    block_ids_ptr,
    logits_row_stride: tl.constexpr,
    slot_pool_stride: tl.constexpr,
    q_pool_stride: tl.constexpr,
    q_slot_stride: tl.constexpr,
    candidates_row_stride: tl.constexpr,
    candidates_col_stride: tl.constexpr,
    stats_row_stride: tl.constexpr,
    stats_field_stride: tl.constexpr,
    ids_row_stride: tl.constexpr,
    vocab_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_TOKENS_PER_REQ: tl.constexpr,
    NUM_CANDIDATES: tl.constexpr,
):
    row = tl.program_id(0)
    block_idx = tl.program_id(1)
    req_row = row // NUM_TOKENS_PER_REQ
    spec_pos = row - req_row * NUM_TOKENS_PER_REQ
    pool_idx = tl.load(req_pool_indices_ptr + req_row).to(tl.int64)
    token_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = token_offsets < vocab_size

    logits = tl.load(
        logits_ptr + row.to(tl.int64) * logits_row_stride + token_offsets,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    temperature = tl.maximum(
        tl.load(temperature_pool_ptr + pool_idx).to(tl.float32), 1.0e-20
    )
    seed = tl.load(seed_pool_ptr + pool_idx).to(tl.int64)
    offset = tl.load(offsets_pool_ptr + pool_idx).to(tl.int64) + spec_pos
    scaled = logits / temperature

    block_max = tl.max(scaled, axis=0)
    safe_max = tl.where(block_max == float("-inf"), 0.0, block_max)
    block_sum = tl.sum(tl.where(mask, tl.exp(scaled - safe_max), 0.0), axis=0)

    # The plain sampler's scores, same arithmetic as gumbel_sample_from_pools.
    gumbel = _plain_gumbel(seed, offset, token_offsets)
    scores = tl.where(mask, logits / temperature + gumbel, float("-inf"))
    # A row with a recorded q leaves its candidates to stage 2, which scores
    # them from the residual max(p - q, 0) instead.
    has_q, q_base = _row_draft_q(
        req_row,
        spec_pos,
        pool_idx,
        slot_ptr,
        pos_ptr,
        q_ids_ptr,
        candidates_ptr,
        slot_pool_stride,
        q_pool_stride,
        q_slot_stride,
        candidates_row_stride,
        candidates_col_stride,
        vocab_size,
        NUM_TOKENS_PER_REQ,
        NUM_CANDIDATES,
    )
    block_start = block_idx * BLOCK_SIZE
    for lane in tl.static_range(NUM_CANDIDATES):
        candidate = tl.load(q_ids_ptr + q_base + lane, mask=has_q, other=-1)
        # Few blocks hold a candidate; the others skip the masking pass.
        if (candidate >= block_start) & (candidate < block_start + BLOCK_SIZE):
            scores = tl.where(token_offsets == candidate, float("-inf"), scores)
    best_score = tl.max(scores, axis=0)
    best_id = tl.min(
        tl.where(scores == best_score, token_offsets, vocab_size + BLOCK_SIZE),
        axis=0,
    )

    stats = block_stats_ptr + row * stats_row_stride + block_idx
    tl.store(stats, block_max)
    tl.store(stats + stats_field_stride, block_sum)
    tl.store(stats + 2 * stats_field_stride, best_score)
    tl.store(block_ids_ptr + row * ids_row_stride + block_idx, best_id)


@triton.jit
def _rejection_no_filter_stage2_kernel(
    logits_ptr,
    req_pool_indices_ptr,
    temperature_pool_ptr,
    seed_pool_ptr,
    offsets_pool_ptr,
    slot_ptr,
    pos_ptr,
    q_ptr,
    q_ids_ptr,
    candidates_ptr,
    block_stats_ptr,
    block_ids_ptr,
    row_accept_ptr,
    row_token_ptr,
    logits_row_stride: tl.constexpr,
    slot_pool_stride: tl.constexpr,
    q_pool_stride: tl.constexpr,
    q_slot_stride: tl.constexpr,
    candidates_row_stride: tl.constexpr,
    candidates_col_stride: tl.constexpr,
    stats_row_stride: tl.constexpr,
    stats_field_stride: tl.constexpr,
    ids_row_stride: tl.constexpr,
    vocab_size: tl.constexpr,
    num_blocks: tl.constexpr,
    NUM_BLOCKS_PAD: tl.constexpr,
    NUM_TOKENS_PER_REQ: tl.constexpr,
    NUM_CANDIDATES: tl.constexpr,
    K_PAD: tl.constexpr,
    COIN_SALT: tl.constexpr,
):
    row = tl.program_id(0)
    req_row = row // NUM_TOKENS_PER_REQ
    spec_pos = row - req_row * NUM_TOKENS_PER_REQ
    pool_idx = tl.load(req_pool_indices_ptr + req_row).to(tl.int64)
    temperature = tl.maximum(
        tl.load(temperature_pool_ptr + pool_idx).to(tl.float32), 1.0e-20
    )
    seed = tl.load(seed_pool_ptr + pool_idx).to(tl.int64)
    offset = tl.load(offsets_pool_ptr + pool_idx).to(tl.int64) + spec_pos

    blocks = tl.arange(0, NUM_BLOCKS_PAD)
    block_mask = blocks < num_blocks
    stats = block_stats_ptr + row * stats_row_stride + blocks
    ids = block_ids_ptr + row * ids_row_stride + blocks
    block_max = tl.load(stats, mask=block_mask, other=float("-inf"))
    block_sum = tl.load(stats + stats_field_stride, mask=block_mask, other=0.0)
    block_best = tl.load(
        stats + 2 * stats_field_stride, mask=block_mask, other=float("-inf")
    )
    block_best_id = tl.load(ids, mask=block_mask, other=2147483647)

    # Log of the softmax denominator over the whole vocabulary.
    row_max = tl.max(block_max, axis=0)
    safe_max = tl.where(row_max == float("-inf"), 0.0, row_max)
    total = tl.sum(block_sum * tl.exp(block_max - safe_max), axis=0)
    lse = safe_max + tl.log(total)
    # Best plain score; for a row with q, over the non-candidates only.
    plain_score = tl.max(block_best, axis=0)
    plain_id = tl.min(
        tl.where(block_best == plain_score, block_best_id, 2147483647), axis=0
    )

    draft_row = spec_pos < NUM_TOKENS_PER_REQ - 1
    draft = tl.load(
        candidates_ptr
        + req_row * candidates_row_stride
        + (spec_pos + 1) * candidates_col_stride,
        mask=draft_row,
        other=-1,
    )
    draft_lane = tl.load(
        pos_ptr + pool_idx * slot_pool_stride + spec_pos + 1,
        mask=draft_row,
        other=-1,
    )
    # Same decision as stage 1, which already left the candidates out.
    has_q, q_base = _row_draft_q(
        req_row,
        spec_pos,
        pool_idx,
        slot_ptr,
        pos_ptr,
        q_ids_ptr,
        candidates_ptr,
        slot_pool_stride,
        q_pool_stride,
        q_slot_stride,
        candidates_row_stride,
        candidates_col_stride,
        vocab_size,
        NUM_TOKENS_PER_REQ,
        NUM_CANDIDATES,
    )

    lanes = tl.arange(0, K_PAD)
    lane_ok = (lanes < NUM_CANDIDATES) & has_q
    q = tl.load(q_ptr + q_base + lanes, mask=lane_ok, other=0.0)
    cand_ids = tl.load(q_ids_ptr + q_base + lanes, mask=lane_ok, other=-1)
    cand_ok = lane_ok & (cand_ids >= 0) & (cand_ids < vocab_size)
    cand_logits = tl.load(
        logits_ptr + row.to(tl.int64) * logits_row_stride + cand_ids,
        mask=cand_ok,
        other=float("-inf"),
    ).to(tl.float32)
    cand_p = tl.where(cand_ok, tl.exp(cand_logits / temperature - lse), 0.0)

    # has_q guarantees the draft's recorded lane holds it.
    is_draft = cand_ok & (lanes == draft_lane) & (cand_ids == draft)
    q_draft = tl.max(tl.where(is_draft, q, 0.0), axis=0)
    p_draft = tl.max(tl.where(is_draft, cand_p, 0.0), axis=0)
    coin = _coin(seed, offset, COIN_SALT)
    accept_q = (q_draft > 0.0) & (coin * q_draft < p_draft)

    residual = cand_p - q
    cand_gumbel = _plain_gumbel(seed, offset, cand_ids)
    cand_scores = tl.where(
        cand_ok & (residual > 0.0),
        tl.log(residual) + lse + cand_gumbel,
        float("-inf"),
    )
    cand_best = tl.max(cand_scores, axis=0)
    cand_id = tl.min(tl.where(cand_scores == cand_best, cand_ids, 2147483647), axis=0)
    take_cand = (cand_best > plain_score) | (
        (cand_best == plain_score) & (cand_id < plain_id)
    )
    resampled = tl.where(take_cand, cand_id, plain_id)
    # p <= q everywhere only through rounding; fall back to the target's mode.
    # Every non-candidate then has p = 0, so the mode is a candidate.
    cand_max = tl.max(tl.where(cand_ok, cand_logits, float("-inf")), axis=0)
    mode = tl.min(
        tl.where(cand_ok & (cand_logits == cand_max), cand_ids, 2147483647), axis=0
    )
    resampled = tl.where(
        tl.maximum(cand_best, plain_score) == float("-inf"), mode, resampled
    )

    token = tl.where(has_q, resampled, plain_id)
    accept = draft_row & tl.where(has_q, accept_q, plain_id == draft)
    tl.store(row_token_ptr + row, token)
    tl.store(row_accept_ptr + row, accept.to(tl.int32))


def verify_rows_no_filter_scratch_shape(
    max_rows: int, vocab_size: int
) -> tuple[tuple[int, int, int], tuple[int, int]]:
    """Shapes of the ``scratch_scores`` / ``scratch_ids`` tensors of
    ``verify_rows_no_filter_from_pools``. Host arithmetic only.

    Args:
        max_rows: Most verify rows (requests times window width) of a call.
        vocab_size: Width of the logits rows.

    Returns:
        ``(scores_shape, ids_shape)`` for an fp32 ``scratch_scores`` and an
        int32 ``scratch_ids`` tensor, allocated once and sliced by rows.
    """
    if max_rows < 0 or vocab_size <= 0:
        raise ValueError(
            "verify_rows_no_filter_scratch_shape needs max_rows >= 0 and "
            f"vocab_size > 0, got {max_rows=} {vocab_size=}"
        )
    num_blocks = triton.cdiv(vocab_size, _REJECTION_BLOCK_SIZE)
    # The verify kernels index the scratch with int32 row offsets, which
    # stay far below 2^31 for any real batch; refuse sizes past that.
    if max_rows * _NUM_FLOAT_STATS * num_blocks >= 2**31:
        raise ValueError(
            "verify_rows_no_filter_scratch_shape keeps scratch offsets in "
            f"int32 and needs max_rows * {_NUM_FLOAT_STATS} * {num_blocks} < "
            f"2**31, got {max_rows=}"
        )
    return (max_rows, _NUM_FLOAT_STATS, num_blocks), (max_rows, num_blocks)


def verify_rows_no_filter_from_pools(
    logits: torch.Tensor,
    req_pool_indices: torch.Tensor,
    temperature_pool: torch.Tensor,
    seed_pool: torch.Tensor,
    offsets_pool: torch.Tensor,
    candidates: torch.Tensor,
    q: torch.Tensor,
    q_ids: torch.Tensor,
    slot: torch.Tensor,
    pos: torch.Tensor,
    scratch_scores: torch.Tensor,
    scratch_ids: torch.Tensor,
    row_accept: torch.Tensor,
    row_token: torch.Tensor,
    *,
    num_tokens_per_req: int,
) -> None:
    """Per-row rejection-sampling verify for rows without top-k or top-p.

    Every request must be served without top-k or top-p (the no-filter
    route): the target distribution is ``softmax(logits / temperature)`` over
    the whole vocabulary. Row ``r`` of request ``b`` is verify row ``b * N +
    r`` (``N = num_tokens_per_req``); it checks draft ``candidates[b, r + 1]``
    against ``q[pool, slot[pool, r + 1]]``, recorded at lane ``pos[pool, r +
    1]``. A slot of ``-1`` means no recorded q; such rows, and each request's
    last row, take the plain sampler's draw (the token
    ``gumbel_sample_from_pools`` samples at that offset) and accept the draft
    when it equals the draw.

    Args:
        logits: ``[bs * N, V]`` target logits, stride-1 vocabulary.
        req_pool_indices: ``[bs]`` int32 pool slot of each request.
        temperature_pool: ``[pool]`` fp32 temperature.
        seed_pool: ``[pool]`` int64 seed.
        offsets_pool: ``[pool]`` int64 verify offset of each slot's row 0.
        candidates: ``[bs, N]`` int32 verify window; column 0 is the anchor.
        q: ``[pool, S, K]`` fp32 recorded draft probabilities.
        q_ids: ``[pool, S, K]`` int32 their token ids.
        slot: ``[pool, >= N]`` int32 per window node, the ``q`` row its token
            was drawn from; ``-1`` for none.
        pos: ``[pool, >= N]`` int32 per window node, its lane in that row.
        scratch_scores: fp32 ``[bs * N, ...]`` slice of a tensor shaped by
            ``verify_rows_no_filter_scratch_shape``.
        scratch_ids: int32 ``[bs * N, ...]`` slice of a tensor shaped by
            ``verify_rows_no_filter_scratch_shape``.
        row_accept: ``[bs * N]`` int32 output; 1 when row ``r < N - 1``
            accepts draft ``x_{r+1}``, 0 otherwise and for the last row.
        row_token: ``[bs * N]`` int32 output; the token row ``r`` emits when
            the chain stops there.
        num_tokens_per_req: Verify window width ``N``.

    Returns:
        None; ``row_accept`` and ``row_token`` are written in place.
    """
    fn_name = "verify_rows_no_filter_from_pools"
    if logits.ndim != 2 or logits.stride(-1) != 1:
        raise ValueError(f"{fn_name} expects 2D logits with a stride-1 vocabulary")
    if logits.device.type != "cuda":
        raise ValueError(f"{fn_name} requires CUDA logits")
    rows, vocab_size = logits.shape
    if vocab_size <= 0:
        raise ValueError(f"{fn_name} requires a non-empty vocabulary")
    num_candidates = check_draft_q_inputs(
        fn_name=fn_name,
        rows=rows,
        num_tokens_per_req=num_tokens_per_req,
        req_pool_indices=req_pool_indices,
        candidates=candidates,
        q=q,
        q_ids=q_ids,
        slot=slot,
        pos=pos,
        row_accept=row_accept,
        row_token=row_token,
    )
    if seed_pool.dtype != torch.int64:
        raise ValueError(f"seed_pool must be int64, got {seed_pool.dtype}")
    scores_shape, ids_shape = verify_rows_no_filter_scratch_shape(rows, vocab_size)
    num_blocks = scores_shape[2]
    for name, tensor, shape, dtype in (
        ("scratch_scores", scratch_scores, scores_shape, torch.float32),
        ("scratch_ids", scratch_ids, ids_shape, torch.int32),
    ):
        if (
            tensor.ndim != len(shape)
            or tensor.shape[0] < rows
            or tuple(tensor.shape[1:]) != shape[1:]
            or tensor.dtype != dtype
            or tensor.stride(-1) != 1
        ):
            raise ValueError(
                f"{name} must be {dtype} [>= {rows}, *{shape[1:]}] with stride-1 "
                f"blocks, got {tuple(tensor.shape)} {tensor.dtype}"
            )
    if rows == 0:
        return

    common = dict(
        logits_row_stride=logits.stride(0),
        slot_pool_stride=slot.stride(0),
        q_pool_stride=q.stride(0),
        q_slot_stride=q.stride(1),
        candidates_row_stride=candidates.stride(0),
        candidates_col_stride=candidates.stride(1),
        stats_row_stride=scratch_scores.stride(0),
        stats_field_stride=scratch_scores.stride(1),
        ids_row_stride=scratch_ids.stride(0),
        vocab_size=vocab_size,
        NUM_TOKENS_PER_REQ=num_tokens_per_req,
        NUM_CANDIDATES=num_candidates,
    )
    _rejection_no_filter_stage1_kernel[(rows, num_blocks)](
        logits,
        req_pool_indices,
        temperature_pool,
        seed_pool,
        offsets_pool,
        slot,
        pos,
        q_ids,
        candidates,
        scratch_scores,
        scratch_ids,
        BLOCK_SIZE=_REJECTION_BLOCK_SIZE,
        num_warps=4,
        **common,
    )
    _rejection_no_filter_stage2_kernel[(rows,)](
        logits,
        req_pool_indices,
        temperature_pool,
        seed_pool,
        offsets_pool,
        slot,
        pos,
        q,
        q_ids,
        candidates,
        scratch_scores,
        scratch_ids,
        row_accept,
        row_token,
        num_blocks=num_blocks,
        NUM_BLOCKS_PAD=triton.next_power_of_2(num_blocks),
        K_PAD=triton.next_power_of_2(num_candidates),
        COIN_SALT=COIN_OFFSET_SALT,
        num_warps=1,
        **common,
    )

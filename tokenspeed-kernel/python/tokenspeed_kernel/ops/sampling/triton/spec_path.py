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

"""Path selection over a block drafter's candidate lattice.

``scores[b, l, p, c]`` is the edge weight of taking candidate ``c`` at step
``l`` after candidate ``p`` at step ``l - 1``; step 0's predecessor is the
anchor, so all of its ``p`` rows are equal. One Triton program walks one
request.

A greedy request takes the best edge at every step (ties toward the lowest
lane). A sampled request draws each step's lane from ``q = softmax(scores /
temperature)`` by Gumbel-max, so the verifier can accept the draft by
rejection sampling against the recorded ``q``. The noise is keyed by the
candidate's token id, never its lane, so tensor-parallel ranks holding the
same scores draw the same token whatever order their top-k returned.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.sampling.triton.common import TOP_K_DISABLED
from tokenspeed_kernel.ops.sampling.triton.rejection import (
    DRAFT_SAMPLE_OFFSET_SALT,
    TOP_P_ONE_EPS,
)

__all__ = ["dflash2_sample_path"]


@triton.jit
def _dflash2_sample_path_kernel(
    candidate_ids_ptr,
    scores_ptr,
    anchor_ptr,
    req_pool_indices_ptr,
    temperature_pool_ptr,
    top_k_pool_ptr,
    top_p_pool_ptr,
    seed_pool_ptr,
    draft_offsets_ptr,
    tokens_ptr,
    q_ptr,
    q_ids_ptr,
    pos_ptr,
    slot_ptr,
    tokens_row_stride,
    NUM_STEPS: tl.constexpr,
    TOP_K: tl.constexpr,
    K_PAD: tl.constexpr,
    TOP_K_OFF: tl.constexpr,
    TOP_P_ONE: tl.constexpr,
    DRAFT_SALT: tl.constexpr,
):
    req = tl.program_id(0).to(tl.int64)
    pool_idx = tl.load(req_pool_indices_ptr + req)
    raw_temperature = tl.load(temperature_pool_ptr + pool_idx).to(tl.float32)
    temperature = tl.maximum(raw_temperature, 1.0e-20)
    top_k = tl.load(top_k_pool_ptr + pool_idx)
    top_p = tl.load(top_p_pool_ptr + pool_idx).to(tl.float32)
    seed = tl.load(seed_pool_ptr + pool_idx).to(tl.int64)
    draft_offset = tl.load(draft_offsets_ptr + req).to(tl.int64) + DRAFT_SALT

    # Greedy requests take the best lane. So do requests served by top-p
    # alone: the verifier has no exact target distribution for them and falls
    # back to exact match, which the best lane passes more often than a draw.
    sampled = (top_k != 1) & (raw_temperature > 0.0)
    sampled = sampled & ~((top_k == TOP_K_OFF) & (top_p < TOP_P_ONE))
    proposal = sampled

    tokens_base = req * tokens_row_stride
    tl.store(tokens_ptr + tokens_base, tl.load(anchor_ptr + req).to(tl.int32))

    lanes = tl.arange(0, K_PAD)
    lane_ok = lanes < TOP_K
    previous = 0
    for step in tl.static_range(NUM_STEPS):
        step_base = (req * NUM_STEPS + step) * TOP_K
        row = tl.load(
            scores_ptr + (step_base + previous) * TOP_K + lanes,
            mask=lane_ok,
            other=float("-inf"),
        ).to(tl.float32)
        ids = tl.load(candidate_ids_ptr + step_base + lanes, mask=lane_ok, other=0).to(
            tl.int32
        )

        row_max = tl.max(row, axis=0)
        has_nan = tl.sum(tl.where(lane_ok & (row != row), 1, 0), axis=0) > 0
        finite = (row_max > float("-inf")) & (row_max < float("inf")) & ~has_nan
        best_lane = tl.min(tl.where(row == row_max, lanes, TOP_K), axis=0)
        best_lane = tl.where(best_lane < TOP_K, best_lane, 0)

        # A row with no finite distribution proposes nothing for the request.
        take_sample = sampled & finite
        proposal = proposal & finite
        if take_sample:
            scaled = row / temperature
            scaled_max = tl.max(scaled, axis=0)
            weights = tl.where(lane_ok, tl.exp(scaled - scaled_max), 0.0)
            q_out = weights / tl.sum(weights, axis=0)
            uniform = tl.maximum(
                tl.rand(tl.randint(seed, draft_offset + step), ids), 1.0e-7
            )
            gumbel = -tl.log(-tl.log(uniform))
            noisy = tl.where(lane_ok, scaled + gumbel, float("-inf"))
            noisy_max = tl.max(noisy, axis=0)
            # Break exact ties by token id so the pick does not depend on lane
            # order.
            sample_id = tl.min(tl.where(noisy == noisy_max, ids, 2147483647), axis=0)
            lane = tl.min(
                tl.where((noisy == noisy_max) & (ids == sample_id), lanes, TOP_K),
                axis=0,
            )
        else:
            # Greedy and fallback rows skip the draw: the best lane, one-hot q.
            lane = best_lane
            q_out = tl.where(lanes == best_lane, 1.0, 0.0)

        tl.store(q_ptr + step_base + lanes, q_out, mask=lane_ok)
        tl.store(q_ids_ptr + step_base + lanes, ids, mask=lane_ok)
        tl.store(pos_ptr + req * (NUM_STEPS + 1) + step + 1, lane.to(tl.int32))
        token = tl.load(candidate_ids_ptr + step_base + lane).to(tl.int32)
        tl.store(tokens_ptr + tokens_base + step + 1, token)
        previous = lane

    # The anchor (node 0) has no q and no lane. Node l + 1 reads q row l when
    # the request proposes; otherwise every node is -1 (the target-only rule).
    node_base = req * (NUM_STEPS + 1)
    tl.store(pos_ptr + node_base, 0)
    tl.store(slot_ptr + node_base, -1)
    for step in tl.static_range(NUM_STEPS):
        tl.store(slot_ptr + node_base + step + 1, tl.where(proposal, step, -1))


def dflash2_sample_path(
    candidate_ids: torch.Tensor,
    scores: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    req_pool_indices: torch.Tensor,
    temperature_pool: torch.Tensor,
    top_k_pool: torch.Tensor,
    top_p_pool: torch.Tensor,
    seed_pool: torch.Tensor,
    draft_offsets: torch.Tensor,
    tokens: torch.Tensor,
    q: torch.Tensor,
    q_ids: torch.Tensor,
    pos: torch.Tensor,
    slot: torch.Tensor,
) -> torch.Tensor:
    """Walk the lattice, sampling each step's lane or taking the best one.

    Step ``l`` reads the scores out of the previous step's lane (lane 0 for
    step 0). A sampled request draws the lane from ``softmax(scores /
    temperature)`` with Gumbel noise from seed ``seed`` at offset
    ``DRAFT_SAMPLE_OFFSET_SALT + draft_offsets[b] + l``, keyed by the
    candidate's token id. Greedy requests (``top_k == 1`` or temperature 0),
    requests with top-p but no top-k, and requests that meet a non-finite
    score row take the lowest-index best lane, record a one-hot ``q`` for
    such steps, and propose nothing.

    ``q``, ``q_ids``, ``pos`` and ``slot`` are the request's chain in the
    verifier's draft-q layout: node 0 is the anchor (slot -1, pos 0) and node
    ``l + 1`` (draft token ``l + 1``) reads q row ``l`` at lane ``pos``. A
    request that proposes nothing has slot -1 on every node.

    Args:
        candidate_ids: ``[bs, L, K]`` int32/int64 candidate token ids.
        scores: ``[bs, L, K, K]`` predecessor-conditioned edge weights.
        anchor_token_ids: ``[bs]`` token each request's walk starts from.
        req_pool_indices: ``[bs]`` int32 pool slot of each request.
        temperature_pool: ``[pool]`` fp32 temperature.
        top_k_pool: ``[pool]`` int32 top-k (``TOP_K_DISABLED`` when off).
        top_p_pool: ``[pool]`` fp32 top-p.
        seed_pool: ``[pool]`` int64 seed.
        draft_offsets: ``[bs]`` int64 per-request offset base (the verify
            window's start position times its width); the kernel adds the
            draft salt and the step.
        tokens: ``[bs, L + 1]`` int32 output; column 0 receives the anchor
            and column ``l + 1`` the step-``l`` token.
        q: ``[bs, L, K]`` fp32 output; each step's lane probabilities.
        q_ids: ``[bs, L, K]`` int32 output; each step's candidate ids.
        pos: ``[bs, L + 1]`` int32 output; each node's lane (0 for node 0).
        slot: ``[bs, L + 1]`` int32 output; each node's q row, or -1.

    Returns:
        ``tokens``, filled in place.

    Raises:
        ValueError: An input is not the shape, dtype or layout the kernel
            indexes with.
    """
    if candidate_ids.ndim != 3:
        raise ValueError(f"candidate_ids must be 3D, got {candidate_ids.ndim}D")
    batch_size, num_steps, top_k = candidate_ids.shape
    if scores.shape != (batch_size, num_steps, top_k, top_k):
        raise ValueError(
            f"scores shape {tuple(scores.shape)} must be "
            f"{(batch_size, num_steps, top_k, top_k)}"
        )
    for name, tensor, shape in (
        ("anchor_token_ids", anchor_token_ids, (batch_size,)),
        ("req_pool_indices", req_pool_indices, (batch_size,)),
        ("draft_offsets", draft_offsets, (batch_size,)),
        ("tokens", tokens, (batch_size, num_steps + 1)),
        ("q", q, (batch_size, num_steps, top_k)),
        ("q_ids", q_ids, (batch_size, num_steps, top_k)),
        ("pos", pos, (batch_size, num_steps + 1)),
        ("slot", slot, (batch_size, num_steps + 1)),
    ):
        if tensor.shape != shape:
            raise ValueError(f"{name} shape {tuple(tensor.shape)} must be {shape}")
    for name, tensor, dtype in (
        ("tokens", tokens, torch.int32),
        ("q", q, torch.float32),
        ("q_ids", q_ids, torch.int32),
        ("pos", pos, torch.int32),
        ("slot", slot, torch.int32),
        ("req_pool_indices", req_pool_indices, torch.int32),
        ("top_k_pool", top_k_pool, torch.int32),
        ("seed_pool", seed_pool, torch.int64),
        ("draft_offsets", draft_offsets, torch.int64),
    ):
        if tensor.dtype != dtype:
            raise ValueError(f"{name} must be {dtype}, got {tensor.dtype}")
    if not all(
        tensor.is_contiguous()
        for tensor in (candidate_ids, scores, q, q_ids, pos, slot, draft_offsets)
    ):
        raise ValueError(
            "candidate_ids, scores, q, q_ids, pos, slot and draft_offsets must "
            "be contiguous"
        )
    if tokens.stride(1) != 1:
        raise ValueError("tokens rows must be contiguous")
    tensors = (
        candidate_ids,
        scores,
        anchor_token_ids,
        req_pool_indices,
        temperature_pool,
        top_k_pool,
        top_p_pool,
        seed_pool,
        draft_offsets,
        tokens,
        q,
        q_ids,
        pos,
        slot,
    )
    if not all(tensor.device.type == "cuda" for tensor in tensors):
        raise ValueError("dflash2_sample_path requires CUDA tensors")
    if batch_size == 0 or num_steps == 0:
        return tokens

    _dflash2_sample_path_kernel[(batch_size,)](
        candidate_ids,
        scores,
        anchor_token_ids,
        req_pool_indices,
        temperature_pool,
        top_k_pool,
        top_p_pool,
        seed_pool,
        draft_offsets,
        tokens,
        q,
        q_ids,
        pos,
        slot,
        tokens.stride(0),
        NUM_STEPS=num_steps,
        TOP_K=top_k,
        K_PAD=triton.next_power_of_2(top_k),
        TOP_K_OFF=TOP_K_DISABLED,
        TOP_P_ONE=1.0 - TOP_P_ONE_EPS,
        DRAFT_SALT=DRAFT_SAMPLE_OFFSET_SALT,
    )
    return tokens

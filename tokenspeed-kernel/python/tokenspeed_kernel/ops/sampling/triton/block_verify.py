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

"""Block verification with vocabulary-block residual reductions.

Steps and notation follow ``tokenspeed_kernel.ops.sampling.block_verify``. Four
launches, each request owned by one program per launch:

- ``_prefix_ratios`` (B): prefix ratios ``r_i`` in log space and the number of
  live proposals ``K`` (the first sentinel truncates it).
- ``_residual_mass`` (B, N, chunks): per 1024-token vocabulary chunk, the
  positive residual mass and the target mass of every position.
- ``_select_prefix`` (B): thresholds ``h_i`` and the accepted length; picks the
  chunk holding the final token by inverse CDF over chunk masses; writes the
  accepted prefix in chain layout.
- ``_sample_residual`` (B): inverse CDF inside the chosen chunk.

No atomics and fixed grids: equal inputs give equal outputs on every TP rank.
No per-request host synchronization. N/V are fixed server geometry; batch
size and input strides remain runtime values. Scratch is captured with the
same ordinary execution path used for eager calls.
"""

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

# Largest fp32 below 1: keeps goal = coin * total strictly below total, so the
# inverse CDF lands inside the mass even for a coin of exactly 1.
_MAX_COIN = 0.99999994
_NOT_FOUND = 2147483647


@triton.jit(do_not_specialize=["candidate_stride"])
def _prefix_ratios(
    TargetProbs,
    DraftProbs,
    Candidates,
    Ratios,
    NumLive,
    candidate_stride,
    threshold,
    N: tl.constexpr,
    V: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    log_r = 0.0
    live_so_far = True
    num_live = N - 1
    tl.store(Ratios + b * N, 1.0)
    for i in range(N - 1):
        token = tl.load(Candidates + b * candidate_stride + i + 1).to(tl.int64)
        in_range = (token >= 0) & (token < V)
        q = tl.load(DraftProbs + (b * N + i) * V + token, in_range, 0.0)
        p = tl.load(TargetProbs + (b * N + i) * V + token, in_range, 0.0)
        # q outside (0, threshold] is a sentinel: no proposal from here on.
        live = in_range & (q > 0) & (q <= threshold)
        num_live = tl.where(live_so_far & ~live, i, num_live)
        live_so_far = live_so_far & live
        log_r = tl.where(
            live_so_far, tl.minimum(0.0, log_r + tl.log(p) - tl.log(q)), log_r
        )
        tl.store(Ratios + b * N + i + 1, tl.exp(log_r))
    tl.store(NumLive + b, num_live)


@triton.jit
def _residual_mass(
    TargetProbs,
    DraftProbs,
    Ratios,
    NumLive,
    ResidualMass,
    TargetMass,
    N: tl.constexpr,
    V: tl.constexpr,
    C: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    i = tl.program_id(1)
    c = tl.program_id(2)
    x = c * BLOCK + tl.arange(0, BLOCK)
    k = tl.load(NumLive + b)
    p = tl.load(TargetProbs + (b * N + i) * V + x, x < V, 0.0)
    q = tl.load(DraftProbs + (b * N + i) * V + x, x < V, 0.0)
    r = tl.load(Ratios + b * N + i)
    # Positions past the live proposals have no q: their law is the target's.
    mass = tl.where(i < k, tl.maximum(r * p - q, 0.0), p)
    tl.store(ResidualMass + (b * N + i) * C + c, tl.sum(mass, 0))
    tl.store(TargetMass + (b * N + i) * C + c, tl.sum(p, 0))


@triton.jit(do_not_specialize=["coin_stride", "final_stride", "candidate_stride"])
def _select_prefix(
    Ratios,
    NumLive,
    ResidualMass,
    TargetMass,
    Coins,
    FinalCoins,
    Candidates,
    Out,
    Index,
    Length,
    Chunk,
    Goal,
    Fallback,
    coin_stride,
    final_stride,
    candidate_stride,
    N: tl.constexpr,
    C: tl.constexpr,
    PAD_C: tl.constexpr,
    PAD_N: tl.constexpr,
    MAX_COIN: tl.constexpr,
    NOT_FOUND: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    chunks = tl.arange(0, PAD_C)
    k = tl.load(NumLive + b)
    # Steps 2-4: keep the longest prefix whose coin is below its threshold h.
    accepted = 0
    for i in range(N - 1):
        if i < k:
            ratio = tl.load(Ratios + b * N + i + 1)
            z = tl.sum(
                tl.load(ResidualMass + (b * N + i + 1) * C + chunks, chunks < C, 0.0),
                0,
            )
            denom = z + 1.0 - ratio
            h = tl.where(i + 1 == k, ratio, tl.where(denom > 0, z / denom, 1.0))
            coin = tl.load(Coins + b * coin_stride + i)
            accepted = tl.where(coin < h, i + 1, accepted)
    # Step 5, first level: the chunk holding the final token.
    mass = tl.load(ResidualMass + (b * N + accepted) * C + chunks, chunks < C, 0.0)
    # fp32 rounding can leave a rejection with no positive residual although
    # its exact probability is zero; draw from the target law instead of
    # emitting an arbitrary token (the torch oracle does the same).
    empty = tl.sum(mass, 0) <= 0.0
    target_mass = tl.load(TargetMass + (b * N + accepted) * C + chunks, chunks < C, 0.0)
    mass = tl.where(empty, target_mass, mass)
    cumulative = tl.cumsum(mass, 0)
    total = tl.sum(mass, 0)
    goal = tl.minimum(tl.load(FinalCoins + b * final_stride), MAX_COIN) * total
    chosen = tl.min(tl.where((chunks < C) & (cumulative > goal), chunks, NOT_FOUND), 0)
    # A last-bit difference between sum and scan must never create OOB access.
    last = tl.max(tl.where((chunks < C) & (mass > 0), chunks, 0), 0)
    chosen = tl.where(chosen == NOT_FOUND, last, chosen)
    before = tl.sum(tl.where(chunks < chosen, mass, 0.0), 0)
    tl.store(Chunk + b, chosen)
    tl.store(Goal + b, tl.maximum(goal - before, 0.0))
    tl.store(Length + b, accepted)
    tl.store(Fallback + b, empty.to(tl.int32))
    # Chain layout: accepted proposals, then the final token at ``accepted``.
    pos = tl.arange(0, PAD_N)
    token = tl.load(Candidates + b * candidate_stride + pos + 1, pos < accepted, 0)
    tl.store(Out + b * N + pos, token, pos < accepted)
    tl.store(Index + b * N + pos, tl.where(pos <= accepted, b * N + pos, -1), pos < N)


@triton.jit
def _sample_residual(
    TargetProbs,
    DraftProbs,
    Ratios,
    NumLive,
    Length,
    Chunk,
    Goal,
    Fallback,
    Out,
    N: tl.constexpr,
    V: tl.constexpr,
    BLOCK: tl.constexpr,
    NOT_FOUND: tl.constexpr,
):
    # Step 5, second level: inverse CDF inside the chosen chunk, on the same
    # law _select_prefix used (residual, or target after a bonus/fallback).
    b = tl.program_id(0).to(tl.int64)
    i = tl.load(Length + b)
    k = tl.load(NumLive + b)
    c = tl.load(Chunk + b)
    x = c * BLOCK + tl.arange(0, BLOCK)
    p = tl.load(TargetProbs + (b * N + i) * V + x, x < V, 0.0)
    q = tl.load(DraftProbs + (b * N + i) * V + x, x < V, 0.0)
    ratio = tl.load(Ratios + b * N + i)
    residual = (i < k) & (tl.load(Fallback + b) == 0)
    mass = tl.where(residual, tl.maximum(ratio * p - q, 0.0), p)
    cumulative = tl.cumsum(mass, 0)
    goal = tl.load(Goal + b)
    token = tl.min(tl.where((x < V) & (cumulative > goal), x, NOT_FOUND), 0)
    last = tl.max(tl.where((x < V) & (mass > 0), x, 0), 0)
    token = tl.where(token == NOT_FOUND, last, token)
    tl.store(Out + b * N + i, token)


@register_kernel(
    "sampling",
    "block_verify",
    name="triton_block_verify",
    solution="triton",
    priority=Priority.PORTABLE,
    signatures=format_signatures("target_probs", "dense", {torch.float32}),
)
def triton_block_verify(
    *,
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_probs,
    reject_draft_prob_threshold,
):
    """GPU implementation of the block verifier probability/chain-layout contract."""
    bs, n, v = target_probs.shape
    block = 1024
    chunks = triton.cdiv(v, block)
    device = target_probs.device
    ratios = torch.empty((bs, n), device=device, dtype=torch.float32)
    num_live = torch.empty((bs,), device=device, dtype=torch.int32)
    residual_mass = torch.empty((bs, n, chunks), device=device, dtype=torch.float32)
    target_mass = torch.empty_like(residual_mass)
    chosen_chunk = torch.empty_like(num_live)
    goals = torch.empty((bs,), device=device, dtype=torch.float32)
    fallback = torch.empty_like(num_live)
    _prefix_ratios[(bs,)](
        target_probs,
        draft_probs,
        candidates,
        ratios,
        num_live,
        candidates.stride(0),
        reject_draft_prob_threshold,
        N=n,
        V=v,
    )
    _residual_mass[(bs, n, chunks)](
        target_probs,
        draft_probs,
        ratios,
        num_live,
        residual_mass,
        target_mass,
        N=n,
        V=v,
        C=chunks,
        BLOCK=block,
    )
    _select_prefix[(bs,)](
        ratios,
        num_live,
        residual_mass,
        target_mass,
        uniform_samples,
        uniform_samples_for_final_sampling,
        candidates,
        predicts,
        accept_index,
        accept_token_num,
        chosen_chunk,
        goals,
        fallback,
        uniform_samples.stride(0),
        uniform_samples_for_final_sampling.stride(0),
        candidates.stride(0),
        N=n,
        C=chunks,
        PAD_C=triton.next_power_of_2(chunks),
        PAD_N=triton.next_power_of_2(n),
        MAX_COIN=_MAX_COIN,
        NOT_FOUND=_NOT_FOUND,
    )
    _sample_residual[(bs,)](
        target_probs,
        draft_probs,
        ratios,
        num_live,
        accept_token_num,
        chosen_chunk,
        goals,
        fallback,
        predicts,
        N=n,
        V=v,
        BLOCK=block,
        NOT_FOUND=_NOT_FOUND,
    )

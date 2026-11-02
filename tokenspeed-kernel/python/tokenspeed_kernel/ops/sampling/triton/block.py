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

No per-request host synchronization. N/V are fixed server geometry; batch
size and input strides remain runtime values. Scratch is captured with the
same ordinary execution path used for eager calls.
"""

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@triton.jit(do_not_specialize=["candidate_stride"])
def _prefix_ratios(
    P, Q, C, R, K, candidate_stride, threshold, N: tl.constexpr, V: tl.constexpr
):
    b = tl.program_id(0).to(tl.int64)
    log_r = 0.0
    valid = True
    count = N - 1
    tl.store(R + b * N, 1.0)
    for i in range(N - 1):
        token = tl.load(C + b * candidate_stride + i + 1).to(tl.int64)
        in_range = (token >= 0) & (token < V)
        q = tl.load(Q + (b * N + i) * V + token, in_range, 0.0)
        p = tl.load(P + (b * N + i) * V + token, in_range, 0.0)
        live = in_range & (q > 0) & (q <= threshold)
        count = tl.where(valid & ~live, i, count)
        valid = valid & live
        log_r = tl.where(valid, tl.minimum(0.0, log_r + tl.log(p) - tl.log(q)), log_r)
        tl.store(R + b * N + i + 1, tl.exp(log_r))
    tl.store(K + b, count)


@triton.jit
def _residual_mass(
    P,
    Q,
    R,
    K,
    M,
    PM,
    N: tl.constexpr,
    V: tl.constexpr,
    C: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    i = tl.program_id(1)
    c = tl.program_id(2)
    x = c * BLOCK + tl.arange(0, BLOCK)
    k = tl.load(K + b)
    p = tl.load(P + (b * N + i) * V + x, x < V, 0.0)
    q = tl.load(Q + (b * N + i) * V + x, x < V, 0.0)
    r = tl.load(R + b * N + i)
    mass = tl.where(i < k, tl.maximum(r * p - q, 0.0), p)
    tl.store(M + (b * N + i) * C + c, tl.sum(mass, 0))
    tl.store(PM + (b * N + i) * C + c, tl.sum(p, 0))


@triton.jit(do_not_specialize=["coin_stride", "final_stride", "candidate_stride"])
def _select_prefix(
    R,
    K,
    M,
    PM,
    U,
    F,
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
):
    b = tl.program_id(0).to(tl.int64)
    chunks = tl.arange(0, PAD_C)
    k = tl.load(K + b)
    accepted = 0
    for i in range(N - 1):
        if i < k:
            ratio = tl.load(R + b * N + i + 1)
            mass = tl.sum(tl.load(M + (b * N + i + 1) * C + chunks, chunks < C, 0.0), 0)
            denom = mass + 1.0 - ratio
            h = tl.where(i + 1 == k, ratio, tl.where(denom > 0, mass / denom, 1.0))
            coin = tl.load(U + b * coin_stride + i)
            accepted = tl.where(coin < h, i + 1, accepted)
    mass = tl.load(M + (b * N + accepted) * C + chunks, chunks < C, 0.0)
    # fp32 rounding can leave a rejection with no positive residual although
    # its exact probability is zero; draw from the target law instead of
    # emitting an arbitrary token (the torch oracle does the same).
    empty = tl.sum(mass, 0) <= 0.0
    target_mass = tl.load(PM + (b * N + accepted) * C + chunks, chunks < C, 0.0)
    mass = tl.where(empty, target_mass, mass)
    cumulative = tl.cumsum(mass, 0)
    total = tl.sum(mass, 0)
    goal = tl.minimum(tl.load(F + b * final_stride), 0.99999994) * total
    chosen = tl.min(tl.where((chunks < C) & (cumulative > goal), chunks, 2147483647), 0)
    # A last-bit difference between sum and scan must never create OOB access.
    last = tl.max(tl.where((chunks < C) & (mass > 0), chunks, 0), 0)
    chosen = tl.where(chosen == 2147483647, last, chosen)
    before = tl.sum(tl.where(chunks < chosen, mass, 0.0), 0)
    tl.store(Chunk + b, chosen)
    tl.store(Goal + b, tl.maximum(goal - before, 0.0))
    tl.store(Length + b, accepted)
    tl.store(Fallback + b, empty.to(tl.int32))
    pos = tl.arange(0, PAD_N)
    token = tl.load(Candidates + b * candidate_stride + pos + 1, pos < accepted, 0)
    tl.store(Out + b * N + pos, token, pos < accepted)
    tl.store(Index + b * N + pos, tl.where(pos <= accepted, b * N + pos, -1), pos < N)


@triton.jit
def _sample_residual(
    P,
    Q,
    R,
    K,
    Length,
    Chunk,
    Goal,
    Fallback,
    Out,
    N: tl.constexpr,
    V: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    i = tl.load(Length + b)
    k = tl.load(K + b)
    c = tl.load(Chunk + b)
    x = c * BLOCK + tl.arange(0, BLOCK)
    p = tl.load(P + (b * N + i) * V + x, x < V, 0.0)
    q = tl.load(Q + (b * N + i) * V + x, x < V, 0.0)
    ratio = tl.load(R + b * N + i)
    residual = (i < k) & (tl.load(Fallback + b) == 0)
    mass = tl.where(residual, tl.maximum(ratio * p - q, 0.0), p)
    cumulative = tl.cumsum(mass, 0)
    goal = tl.load(Goal + b)
    token = tl.min(tl.where((x < V) & (cumulative > goal), x, 2147483647), 0)
    last = tl.max(tl.where((x < V) & (mass > 0), x, 0), 0)
    token = tl.where(token == 2147483647, last, token)
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
    ratios = torch.empty((bs, n), device=target_probs.device, dtype=torch.float32)
    valid = torch.empty((bs,), device=target_probs.device, dtype=torch.int32)
    mass = torch.empty((bs, n, chunks), device=target_probs.device, dtype=torch.float32)
    target_mass = torch.empty_like(mass)
    chosen = torch.empty_like(valid)
    goals = torch.empty_like(valid, dtype=torch.float32)
    fallback = torch.empty_like(valid)
    _prefix_ratios[(bs,)](
        target_probs,
        draft_probs,
        candidates,
        ratios,
        valid,
        candidates.stride(0),
        reject_draft_prob_threshold,
        N=n,
        V=v,
    )
    _residual_mass[(bs, n, chunks)](
        target_probs,
        draft_probs,
        ratios,
        valid,
        mass,
        target_mass,
        N=n,
        V=v,
        C=chunks,
        BLOCK=block,
    )
    _select_prefix[(bs,)](
        ratios,
        valid,
        mass,
        target_mass,
        uniform_samples,
        uniform_samples_for_final_sampling,
        candidates,
        predicts,
        accept_index,
        accept_token_num,
        chosen,
        goals,
        fallback,
        uniform_samples.stride(0),
        uniform_samples_for_final_sampling.stride(0),
        candidates.stride(0),
        N=n,
        C=chunks,
        PAD_C=triton.next_power_of_2(chunks),
        PAD_N=triton.next_power_of_2(n),
    )
    _sample_residual[(bs,)](
        target_probs,
        draft_probs,
        ratios,
        valid,
        accept_token_num,
        chosen,
        goals,
        fallback,
        predicts,
        N=n,
        V=v,
        BLOCK=block,
    )

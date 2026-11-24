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

"""Target-only chain verification over each row's top-k logits.

The dense chain verify normalizes every row over the whole vocabulary,
renormalizes it to the request's top-k / top-p set, and then accepts and
samples. When every request keeps at most ``K`` tokens, the row's ``K`` best
logits (``vocab_shard_topk``) carry the whole kept set, so the same steps run
on them alone:

1. ``w_j = exp(l_j / T - l_0 / T)`` over the ``k`` best (``l_0`` the row max).
2. Top-p cutoff on the inclusive prefix sums ``c_j`` of ``w`` in descending
   order: the first ``j`` with ``c_j >= p * c_{k-1}`` when ``p < 1``, else
   ``k - 1``; the kept probabilities are ``w_j / c_cutoff``.
3. Position ``i`` accepts draft ``x_i`` against row ``i - 1`` when
   ``q(x_i) >= threshold_single`` or ``coin_{i-1} <= q(x_i) / threshold_acc``;
   the first rejection stops the chain.
4. The token after the accepted prefix is drawn from the stopping row with
   the rejected draft removed (the bonus row unmodified when all accept): the
   first token in ascending id order whose cumulative mass exceeds
   ``final_coin * total``.

Outputs follow ``chain_speculative_sampling_target_only``. Tokens differ from
the dense route only where a coin falls within rounding of a decision boundary
(the dense route sums over the whole vocabulary before renormalizing).
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

__all__ = ["triton_topk_chain_verify"]

_INT32_MAX = tl.constexpr(2147483647)


# A mixed batch verifies a decode suffix: its chains and coins start at any row offset.
@triton.jit(
    do_not_specialize=[
        "stride_candidates",
        "stride_coins",
        "stride_topk",
        "threshold_single",
        "threshold_acc",
    ],
    do_not_specialize_on_alignment=["candidates_ptr", "coins_ptr", "final_coins_ptr"],
)
def _topk_chain_verify_kernel(
    predicts_ptr,
    accept_index_ptr,
    accept_token_num_ptr,
    candidates_ptr,
    coins_ptr,
    final_coins_ptr,
    topk_ptr,
    temperature_ptr,
    top_k_ptr,
    top_p_ptr,
    stride_candidates,
    stride_coins,
    stride_topk,
    threshold_single,
    threshold_acc,
    N: tl.constexpr,
    N_PAD: tl.constexpr,
    K: tl.constexpr,
    K_PAD: tl.constexpr,
):
    request = tl.program_id(0).to(tl.int64)
    base = request * N
    chain = candidates_ptr + request * stride_candidates
    position = tl.arange(0, N_PAD)
    lane = tl.arange(0, K_PAD)
    row_ok = position < N
    rows = base + position
    pair = topk_ptr + rows[:, None] * stride_topk + lane[None, :]
    tile_ok = row_ok[:, None] & (lane[None, :] < K)
    logit = tl.load(pair, mask=tile_ok, other=float("-inf"))
    token = tl.load(pair + K, mask=tile_ok, other=-1.0).to(tl.int32)

    # Step 1, with the dense softmax's scaling: logit * (1 / T) less the row max.
    temperature = tl.load(temperature_ptr + rows, mask=row_ok, other=1.0)
    inv_temp = tl.where(temperature == 0.0, 0.0, tl.div_rn(1.0, temperature))
    scaled = logit * inv_temp[:, None]
    row_max = tl.max(scaled, axis=1)
    keep = tl.maximum(1, tl.minimum(tl.load(top_k_ptr + rows, mask=row_ok, other=1), K))
    in_top_k = lane[None, :] < keep[:, None]
    weight = tl.where(in_top_k, tl.exp(scaled - row_max[:, None]), 0.0)

    # Step 2, the fused top-k / top-p renorm's cutoff rule.
    cumulative = tl.cumsum(weight, axis=1)
    top_k_mass = tl.sum(
        tl.where(lane[None, :] == keep[:, None] - 1, cumulative, 0.0), axis=1
    )
    top_p = tl.load(top_p_ptr + rows, mask=row_ok, other=1.0)
    reached = (cumulative >= (top_p * top_k_mass)[:, None]) & in_top_k
    first = tl.min(tl.where(reached, lane[None, :], K_PAD), axis=1)
    cutoff = tl.where(top_p < 1.0, tl.minimum(first, keep - 1), keep - 1)
    kept_mass = tl.sum(
        tl.where(lane[None, :] == cutoff[:, None], cumulative, 0.0), axis=1
    )
    inv_mass = tl.where(kept_mass > 1e-30, tl.div_rn(1.0, kept_mass), 0.0)
    prob = tl.where(lane[None, :] <= cutoff[:, None], weight * inv_mass[:, None], 0.0)

    # Step 3: row r judges draft r + 1 with coin r.
    judged = position < N - 1
    draft = tl.load(chain + position + 1, mask=judged, other=-2).to(tl.int32)
    draft_prob = tl.sum(tl.where(token == draft[:, None], prob, 0.0), axis=1)
    coin = tl.load(
        coins_ptr + request * stride_coins + position, mask=judged, other=0.0
    )
    accept = (draft_prob >= threshold_single) | (coin <= draft_prob / threshold_acc)
    accepted = tl.min(tl.where(judged & ~accept, position, N - 1), axis=0)
    tl.store(
        accept_index_ptr + rows,
        tl.where(position <= accepted, rows, -1).to(tl.int32),
        mask=row_ok,
    )
    tl.store(predicts_ptr + rows, draft, mask=position < accepted)
    tl.store(accept_token_num_ptr + request, accepted)

    # Step 4: inverse CDF in ascending token id over the stopping row.
    stop_row = position[:, None] == accepted
    final_prob = tl.sum(tl.where(stop_row, prob, 0.0), axis=0)
    final_token = tl.sum(tl.where(stop_row, token, 0), axis=0)
    rejected = tl.load(chain + accepted + 1, mask=accepted < N - 1, other=-2).to(
        tl.int32
    )
    mass = tl.where(final_token == rejected, 0.0, final_prob)
    at_or_before = final_token[None, :] <= final_token[:, None]
    cdf = tl.sum(tl.where(at_or_before, mass[None, :], 0.0), axis=1)
    # A masked sum never exceeds the full one, so the max is the total mass.
    threshold = tl.load(final_coins_ptr + request) * tl.max(cdf, axis=0)
    hit = (mass > 0.0) & (cdf > threshold)
    sampled = tl.min(tl.where(hit, final_token, _INT32_MAX), axis=0)
    # Rounding can leave the threshold at the total: take the last token with mass.
    last = tl.max(tl.where(mass > 0.0, final_token, -1), axis=0)
    best = tl.max(tl.where(lane == 0, final_token, -1), axis=0)
    fallback = tl.where(last >= 0, last, best)
    tl.store(
        predicts_ptr + base + accepted,
        tl.where(sampled == _INT32_MAX, fallback, sampled),
    )


@register_kernel(
    "sampling",
    "topk_chain_verify",
    name="triton_topk_chain_verify",
    solution="triton",
    signatures=format_signatures("topk", "dense", {torch.float32}),
    priority=Priority.PORTABLE,
)
def triton_topk_chain_verify(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    topk: torch.Tensor,
    k: int,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    *,
    threshold_single: float,
    threshold_acc: float,
) -> None:
    """Target-only chain verification from each row's ``k`` best logits.

    Args:
        predicts: Contiguous int32 ``[>= bs * N]`` output tokens, written for
            the accepted prefix and the slot after it.
        accept_index: Contiguous int32 ``[bs, N]`` accepted row ids, ``-1``
            past the accepted prefix.
        accept_token_num: Contiguous int32 ``[>= bs]`` accepted draft counts
            (bonus token excluded).
        candidates: int32 or int64 ``[bs, N]`` chains, unit column stride; column 0
            is the verified token, columns ``1..N-1`` the drafts.
        uniform_samples: fp32 ``[bs, N]`` accept coins, unit column stride.
        uniform_samples_for_final_sampling: Contiguous fp32 ``[>= bs]``
            residual coins.
        topk: fp32 ``[bs * N, >= 2 * k]``, unit column stride: each row's
            ``k`` best logits in descending order (lowest id first among
            equal ones), then their ids as exact fp32 integers, as
            ``vocab_shard_topk`` writes them.
        k: Candidates per row; every row's top-k must be at most ``k``.
        temperatures: Contiguous fp32 ``[bs * N]`` per-row temperatures.
        top_ks: Contiguous int32 ``[bs * N]`` per-row top-k, at most ``k``.
        top_ps: Contiguous fp32 ``[bs * N]`` per-row top-p.
        threshold_single: Accept outright at or above this probability, in
            ``[0, 1]``.
        threshold_acc: Accept with probability ``q / threshold_acc``, in
            ``[0, 1]`` (raised to 1e-9 when smaller).
    """
    bs, n = candidates.shape
    rows = bs * n
    for name, tensor, dtype, size in (
        ("predicts", predicts, torch.int32, rows),
        ("accept_index", accept_index, torch.int32, rows),
        ("accept_token_num", accept_token_num, torch.int32, bs),
        (
            "uniform_samples_for_final_sampling",
            uniform_samples_for_final_sampling,
            torch.float32,
            bs,
        ),
        ("temperatures", temperatures, torch.float32, rows),
        ("top_ks", top_ks, torch.int32, rows),
        ("top_ps", top_ps, torch.float32, rows),
    ):
        if tensor.dtype != dtype or not tensor.is_contiguous() or tensor.numel() < size:
            raise ValueError(
                f"{name} must be a contiguous {dtype} of >= {size} elements"
            )
    if accept_index.shape != candidates.shape:
        raise ValueError(
            f"accept_index must be [{bs}, {n}], got {tuple(accept_index.shape)}"
        )
    for name, tensor, dtypes in (
        ("candidates", candidates, (torch.int32, torch.int64)),
        ("uniform_samples", uniform_samples, (torch.float32,)),
    ):
        if (
            tensor.dtype not in dtypes
            or tensor.shape != candidates.shape
            or tensor.stride(1) != 1
        ):
            raise ValueError(
                f"{name} must be {dtypes} [{bs}, {n}] with unit column stride"
            )
    if (
        topk.dtype != torch.float32
        or topk.ndim != 2
        or topk.shape[0] < rows
        or topk.shape[1] < 2 * k
        or topk.stride(1) != 1
    ):
        raise ValueError(
            f"topk must be fp32 [>= {rows}, >= {2 * k}] with unit column stride"
        )
    if not (0.0 <= threshold_single <= 1.0 and 0.0 <= threshold_acc <= 1.0):
        raise ValueError(
            f"thresholds must lie in [0, 1], got single {threshold_single}, acc {threshold_acc}"
        )
    if bs == 0:
        return
    _topk_chain_verify_kernel[(bs,)](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        uniform_samples,
        uniform_samples_for_final_sampling,
        topk,
        temperatures,
        top_ks,
        top_ps,
        candidates.stride(0),
        uniform_samples.stride(0),
        topk.stride(0),
        float(threshold_single),
        max(float(threshold_acc), 1e-9),
        N=n,
        N_PAD=triton.next_power_of_2(n),
        K=k,
        K_PAD=triton.next_power_of_2(k),
        num_warps=1,
    )

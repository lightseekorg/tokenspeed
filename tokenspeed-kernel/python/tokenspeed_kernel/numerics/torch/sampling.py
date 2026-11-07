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

"""PyTorch references for ``tokenspeed_kernel.ops.sampling``."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.sampling.triton.common import TOP_K_DISABLED
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@register_kernel(
    "sampling",
    "gather_and_expand_scalars",
    name="torch_gather_and_expand_scalars",
    solution="torch",
    signatures=format_signatures("temperature", "dense", {torch.float32}),
    priority=Priority.REFERENCE,
)
def torch_gather_and_expand_scalars(
    index: torch.Tensor,
    *,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    min_p: torch.Tensor | None,
    seed: torch.Tensor | None,
    offsets: torch.Tensor | None,
    n: int,
) -> tuple[torch.Tensor | None, ...]:
    """``pool.index_select(0, index).repeat_interleave(n)`` of each pool.

    Returns:
        ``(temperatures, top_ks, top_ps, min_ps, seeds, offsets)``, ``None``
        for an omitted pool; offsets widen to int64.
    """

    def expand(pool: torch.Tensor | None) -> torch.Tensor | None:
        if pool is None:
            return None
        return pool.index_select(0, index.long()).repeat_interleave(n, dim=0)

    expanded_offsets = expand(offsets)
    return (
        expand(temperature),
        expand(top_k),
        expand(top_p),
        expand(min_p),
        expand(seed),
        None if expanded_offsets is None else expanded_offsets.to(torch.int64),
    )


@register_kernel(
    "sampling",
    "min_p_renorm_prob",
    name="torch_min_p_renorm_prob",
    solution="torch",
    signatures=format_signatures("probs", "dense", {torch.float32}),
    priority=Priority.REFERENCE,
)
def torch_min_p_renorm_prob(probs: torch.Tensor, min_p: torch.Tensor) -> torch.Tensor:
    """Zero each row's probabilities below ``min_p * max`` and renormalize."""
    max_probs = probs.max(dim=-1, keepdim=True).values
    out = torch.where(
        probs >= min_p.to(probs.dtype).view(-1, 1) * max_probs,
        probs,
        torch.zeros_like(probs),
    )
    return out / out.sum(dim=-1, keepdim=True)


@register_kernel(
    "sampling",
    "topk_topp_renorm",
    name="torch_topk_topp_renorm",
    solution="torch",
    signatures=format_signatures("probs", "dense", {torch.float32}),
    priority=Priority.REFERENCE,
)
def torch_topk_topp_renorm(
    probs: torch.Tensor, top_ks: torch.Tensor, top_ps: torch.Tensor
) -> torch.Tensor:
    """Top-k renormalization, then deterministic top-p renormalization.

    Mirrors FlashInfer's ``top_k_renorm_prob`` followed by
    ``top_p_renorm_prob(is_deterministic=True)``: ``k >= vocab`` means no
    top-k cutoff, ``p >= 1`` no top-p cutoff (renormalize only). Top-p keeps
    the smallest prefix of the sorted row with mass ``>= p``.
    """
    bs, vocab = probs.shape
    out = probs.clone()
    for i in range(bs):
        k = min(int(top_ks[i].item()), vocab)
        if k < vocab:
            kth = torch.topk(out[i], k, sorted=False).values.min()
            out[i] = torch.where(out[i] >= kth, out[i], torch.zeros_like(out[i]))
            s = out[i].sum()
            if s > 0:
                out[i] = out[i] / s
        p = float(top_ps[i].item())
        if p < 1.0:
            sorted_vals, _ = torch.sort(out[i], descending=True)
            cs = torch.cumsum(sorted_vals, 0)
            # Clamp to the vocabulary in case FP32 rounding leaves the row
            # total below p.
            keep = min((cs < p).sum().item() + 1, vocab)
            thresh = sorted_vals[keep - 1]
            out[i] = torch.where(out[i] >= thresh, out[i], torch.zeros_like(out[i]))
        s = out[i].sum()
        if s > 0:
            out[i] = out[i] / s
    return out


# The distance from 1 within which top-p means "no top-p cutoff".
_TOP_P_ONE_EPS = 1.0e-6


@register_kernel(
    "sampling",
    "dflash2_sample_path",
    name="torch_dflash2_sample_path",
    solution="torch",
    signatures=format_signatures(
        "scores", "dense", {torch.float32, torch.bfloat16, torch.float16}
    ),
    priority=Priority.REFERENCE,
)
def torch_dflash2_sample_path(
    candidate_ids: torch.Tensor,
    scores: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    gumbel: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Walk the lattice, sampling each step's lane or taking the best one.

    Step ``l`` reads the scores out of the previous step's lane (lane 0 for
    step 0). A sampled request picks ``argmax(scores / T + gumbel)`` (exact
    ties toward the lowest token id) and records ``q = softmax(scores / T)``.
    Greedy requests (``top_k == 1`` or ``T == 0``), requests with top-p but
    no top-k, and requests that meet a non-finite score row take the lowest
    best lane, record a one-hot ``q`` for such steps, and report no proposal.

    Args:
        candidate_ids: ``[bs, L, K]`` candidate token ids.
        scores: ``[bs, L, K, K]`` predecessor-conditioned edge weights.
        anchor_token_ids: ``[bs]`` token each walk starts from.
        temperature: ``[bs]`` fp32 temperature of each request.
        top_k: ``[bs]`` top-k of each request (``1 << 30`` when disabled).
        top_p: ``[bs]`` top-p of each request.
        gumbel: ``[bs, L, K]`` fp32 Gumbel noise of each step's lanes.

    Returns:
        ``(tokens [bs, L + 1] int32, q [bs, L, K] fp32, lanes [bs, L] int32,
        has_proposal [bs] int32)`` on ``candidate_ids``' device.
    """
    batch, num_steps, num_lanes = candidate_ids.shape
    device = candidate_ids.device
    candidate_ids = candidate_ids.cpu()
    scores = scores.cpu().float()
    gumbel = gumbel.cpu().float()
    temperature = temperature.cpu().float()
    tokens = torch.empty(batch, num_steps + 1, dtype=torch.int32)
    tokens[:, 0] = anchor_token_ids.cpu().to(torch.int32)
    q = torch.empty(batch, num_steps, num_lanes, dtype=torch.float32)
    lanes = torch.empty(batch, num_steps, dtype=torch.int32)
    has_proposal = torch.empty(batch, dtype=torch.int32)
    for request in range(batch):
        k = int(top_k[request])
        sampled = k != 1 and float(temperature[request]) > 0.0
        sampled &= not (
            k == TOP_K_DISABLED and float(top_p[request]) < 1.0 - _TOP_P_ONE_EPS
        )
        proposal = sampled
        t = temperature[request].clamp_min(1.0e-20)
        previous = 0
        for step in range(num_steps):
            row = scores[request, step, previous]
            ids = candidate_ids[request, step]
            row_max = row.max()
            finite = bool(torch.isfinite(row_max)) and not bool(row.isnan().any())
            best = (row == row_max).nonzero()
            best_lane = int(best[0]) if best.numel() else 0
            if sampled and finite:
                scaled = row / t
                noisy = scaled + gumbel[request, step]
                ties = (noisy == noisy.max()).nonzero().flatten()
                lane = int(ties[torch.argmin(ids[ties])])
                q[request, step] = torch.softmax(scaled, dim=0)
            else:
                lane = best_lane
                q[request, step] = 0.0
                q[request, step, lane] = 1.0
            proposal &= finite
            lanes[request, step] = lane
            tokens[request, step + 1] = int(ids[lane])
            previous = lane
        has_proposal[request] = int(proposal)
    return (
        tokens.to(device),
        q.to(device),
        lanes.to(device),
        has_proposal.to(device),
    )


def _kept_tokens(
    logits: torch.Tensor,
    scaled: torch.Tensor,
    top_k: int,
    top_p: float,
    top_k_pad: int | None,
) -> torch.Tensor:
    """Boolean mask of the tokens a top-k / top-p sampler of one row keeps."""
    vocab = logits.numel()
    if top_k == TOP_K_DISABLED:
        k = vocab
    else:
        k = min(max(top_k, 1), vocab, vocab if top_k_pad is None else top_k_pad)
    keep = torch.zeros(vocab, dtype=torch.bool)
    if k == vocab and top_p >= 1.0 - _TOP_P_ONE_EPS:
        keep[:] = True
        return keep
    # Best logits first, ties toward the lower id (the kernels' order).
    order = torch.sort(logits, descending=True, stable=True).indices[:k]
    top = scaled[order]
    weights = torch.exp(top - top.max())
    probs = weights / weights.sum().clamp_min(1.0e-20)
    keep[order[torch.cumsum(probs, 0) - probs < top_p]] = True
    return keep


def _argmax_lowest(values: torch.Tensor, ids: torch.Tensor) -> int:
    ties = (values == values.max()).nonzero().flatten()
    return int(ids[ties].min())


@register_kernel(
    "sampling",
    "verify_rows",
    name="torch_verify_rows",
    solution="torch",
    signatures=format_signatures(
        "logits", "dense", {torch.float32, torch.float64, torch.bfloat16}
    ),
    priority=Priority.REFERENCE,
)
def torch_verify_rows(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    q: torch.Tensor,
    q_ids: torch.Tensor,
    has_q: torch.Tensor,
    draft_lane: torch.Tensor,
    draft_tokens: torch.Tensor,
    coins: torch.Tensor,
    gumbels: torch.Tensor,
    *,
    top_k_pad: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-row rejection-sampling verify of a sampled draft, given its noise.

    The target distribution ``p`` of a row is the softmax of ``logits / T``
    over its kept tokens (the ``min(top_k, top_k_pad)`` best after the top-p
    cut; every token without top-k or top-p), normalized over the kept set. A
    row with a recorded ``q`` accepts draft ``x`` (at lane ``draft_lane``)
    when ``coin * q(x) < p(x)`` and otherwise emits ``argmax(residual + G)``,
    where a kept candidate scores ``log(max(p - q, 0)) + LSE`` and a kept
    non-candidate ``logit / T``; with no finite score it emits the mode of
    ``p``. Greedy rows (``top_k == 1``), rows without ``q`` and rows whose
    lane ``draft_lane`` does not hold ``x`` emit the plain
    draw ``argmax over kept of logit / T + G`` and accept ``x`` when it equals
    the draw. Exact ties go to the lowest token id.

    Args:
        logits: ``[rows, V]`` target logits; fp64 for exact enumeration.
        temperature: ``[rows]`` temperature.
        top_k: ``[rows]`` top-k (``1 << 30`` when disabled).
        top_p: ``[rows]`` top-p.
        q: ``[rows, K]`` recorded draft probabilities.
        q_ids: ``[rows, K]`` their token ids.
        has_q: ``[rows]`` bool, whether the row has a recorded ``q``.
        draft_lane: ``[rows]`` lane of the draft in ``q``.
        draft_tokens: ``[rows]`` the draft token; ``-1`` for none (last row).
        coins: ``[rows]`` uniform accept coins in ``[0, 1)``.
        gumbels: ``[rows, V]`` the plain stream's Gumbel noise per token.
        top_k_pad: Cap on the top-k support (the kernel's pad), or None.

    Returns:
        ``(row_accept [rows] int32, row_token [rows] int32, residual
        [rows, V])``: the accept decisions, the tokens emitted when the chain
        stops at each row, and each row's noiseless residual scores
        (``-inf`` everywhere for rows that emit the plain draw).
    """
    dtype = logits.dtype if logits.dtype == torch.float64 else torch.float32
    logits = logits.cpu().to(dtype)
    rows, vocab = logits.shape
    scaled = logits / temperature.cpu().to(dtype).clamp_min(1.0e-20)[:, None]
    q = q.cpu().to(dtype)
    gumbels = gumbels.cpu().to(dtype)
    token_ids = torch.arange(vocab)
    row_accept = torch.zeros(rows, dtype=torch.int32)
    row_token = torch.zeros(rows, dtype=torch.int32)
    residual = torch.full((rows, vocab), float("-inf"), dtype=dtype)
    for row in range(rows):
        k = int(top_k[row])
        keep = _kept_tokens(logits[row], scaled[row], k, float(top_p[row]), top_k_pad)
        kept_scaled = torch.where(keep, scaled[row], float("-inf"))
        draft = int(draft_tokens[row])
        plain = _argmax_lowest(kept_scaled + gumbels[row], token_ids)
        lane = int(draft_lane[row])
        # A draft outside the vocabulary or not at its recorded lane did not
        # come from this q.
        lane_holds_draft = (
            0 <= draft < vocab
            and 0 <= lane < q.shape[1]
            and int(q_ids[row, lane]) == draft
        )
        if k == 1 or not bool(has_q[row]) or not lane_holds_draft:
            row_token[row] = plain
            row_accept[row] = int(plain == draft)
            continue
        kept_max = kept_scaled.max()
        lse = kept_max + torch.log(torch.exp(kept_scaled - kept_max).sum())
        p = torch.where(keep, torch.exp(scaled[row] - lse), 0.0)
        zero = torch.zeros((), dtype=dtype)
        q_draft = q[row, lane]
        p_draft = p[draft] if 0 <= draft < vocab else zero
        # In the logits' precision, like the kernels.
        coin = coins[row].cpu().to(dtype)
        row_accept[row] = int(bool(q_draft > 0) and bool(coin * q_draft < p_draft))

        support_q = torch.zeros(vocab, dtype=dtype)
        is_candidate = torch.zeros(vocab, dtype=torch.bool)
        for c in range(q.shape[1]):
            cid = int(q_ids[row, c])
            if 0 <= cid < vocab:
                support_q[cid] = torch.maximum(support_q[cid], q[row, c])
                is_candidate[cid] = True
        diff = p - support_q
        candidate_scores = torch.where(
            diff > 0, torch.log(diff.clamp_min(0.0)) + lse, float("-inf")
        )
        scores = torch.where(is_candidate, candidate_scores, scaled[row])
        scores = torch.where(keep, scores, float("-inf"))
        residual[row] = scores
        noisy = scores + gumbels[row]
        if bool(torch.isinf(noisy).all()):
            row_token[row] = _argmax_lowest(kept_scaled, token_ids)
        else:
            row_token[row] = _argmax_lowest(noisy, token_ids)
    return row_accept, row_token, residual


@register_kernel(
    "sampling",
    "dspark_block_greedy",
    name="torch_dspark_block_greedy",
    solution="torch",
    signatures=format_signatures("base", "dense", {torch.float32}),
    priority=Priority.REFERENCE,
)
def torch_dspark_block_greedy(
    base: torch.Tensor,
    anchors: torch.Tensor,
    embedding: torch.Tensor,
    projection: torch.Tensor,
) -> torch.Tensor:
    """Greedy block decode: each step adds the previous token's low-rank bias.

    Step ``s`` picks ``argmax(base[:, s] + embedding[prev] @ projection.T)``
    over the whole vocabulary in FP32, starting from ``anchors``.

    Returns:
        ``[rows, block]`` int64 token ids.
    """
    previous = anchors.long()
    expected = torch.empty(base.shape[:2], dtype=torch.int64, device=base.device)
    for step in range(base.shape[1]):
        bias = embedding[previous].float() @ projection.float().T
        previous = torch.argmax(base[:, step] + bias, dim=-1)
        expected[:, step] = previous
    return expected


@register_kernel(
    "sampling",
    "verify_tree",
    name="torch_verify_tree",
    solution="torch",
    signatures=format_signatures("candidates", "dense", {torch.int32, torch.int64}),
    priority=Priority.REFERENCE,
)
def torch_verify_tree(
    candidates: torch.Tensor, parents: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Greedy verification of draft trees against the target's picks.

    From the root, each step follows the first child whose candidate token
    equals the target's pick at the current node; the walk stops when none
    does. ``predicts`` records each step's pick, then the last node's.

    Args:
        candidates: ``[bs, n]`` draft token of each node (node 0 is the root).
        parents: ``[bs, n]`` parent of each node.
        target: ``[bs * n]`` the target's pick at each node.

    Returns:
        On the host: ``(predicts [bs * n] int32, lengths [bs] int32, paths [bs,
        n] int32)``: picks per accepted step, accepted path lengths, and the
        accepted node ids (``-1`` padded).
    """
    bs, n = candidates.shape
    predicts = torch.zeros(bs * n, dtype=torch.int32)
    lengths, paths = [], torch.full((bs, n), -1, dtype=torch.int32)
    for b in range(bs):
        cur, path = 0, [0]
        while True:
            pick = int(target[b * n + cur])
            kids = [
                j
                for j in range(n)
                if int(parents[b, j]) == cur and int(candidates[b, j]) == pick
            ]
            if not kids:
                break
            predicts[b * n + len(path) - 1] = pick
            cur = kids[0]
            path.append(cur)
        predicts[b * n + len(path) - 1] = int(target[b * n + cur])
        lengths.append(len(path))
        paths[b, : len(path)] = torch.tensor(path, dtype=torch.int32)
    return predicts, torch.tensor(lengths, dtype=torch.int32), paths


@register_kernel(
    "sampling",
    "top_k_top_p_support",
    name="torch_top_k_top_p_support",
    solution="torch",
    signatures=format_signatures(
        "logits", "dense", {torch.float32, torch.bfloat16, torch.float16}
    ),
    priority=Priority.REFERENCE,
)
def torch_top_k_top_p_support(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    min_p: float | None,
) -> set[int]:
    """Token ids a top-k, top-p (and min-p) sampler of one row may draw.

    Among the ``top_k`` best logits at ``temperature``, a token is kept while
    the probability mass before it is below ``top_p``; a ``min_p`` also drops
    tokens whose probability is below ``min_p`` times the best one.
    """
    top_k = min(top_k, logits.numel())
    top_vals, top_ids = torch.topk(logits.float(), top_k, sorted=True)
    scaled = top_vals / temperature
    probs = torch.softmax(scaled, dim=-1)
    keep = torch.cumsum(probs, dim=-1) - probs < top_p
    if min_p is not None:
        keep &= scaled >= scaled.max() + torch.log(
            torch.tensor(min_p, device=logits.device)
        )
    return set(top_ids[keep].int().cpu().tolist())

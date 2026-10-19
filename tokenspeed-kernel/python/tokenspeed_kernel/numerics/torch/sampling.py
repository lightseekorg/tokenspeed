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


@register_kernel(
    "sampling",
    "dflash2_greedy_path",
    name="torch_dflash2_greedy_path",
    solution="torch",
    signatures=format_signatures(
        "scores", "dense", {torch.float32, torch.bfloat16, torch.float16}
    ),
    priority=Priority.REFERENCE,
)
def torch_dflash2_greedy_path(
    candidate_ids: torch.Tensor,
    scores: torch.Tensor,
    anchor_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Walk the lattice one locally best edge at a time.

    Step 0 takes the best edge out of lane 0; each later step the best edge
    out of the previous step's pick.

    Returns:
        ``[batch, num_steps + 1]`` int32: the anchor, then each step's token.
    """
    batch, num_steps, _ = candidate_ids.shape
    out = torch.empty(batch, num_steps + 1, dtype=torch.int32)
    out[:, 0] = anchor_token_ids.cpu().to(torch.int32)
    for request in range(batch):
        previous = 0
        for step in range(num_steps):
            previous = int(torch.argmax(scores[request, step, previous]))
            out[request, step + 1] = int(candidate_ids[request, step, previous])
    return out.to(candidate_ids.device)


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

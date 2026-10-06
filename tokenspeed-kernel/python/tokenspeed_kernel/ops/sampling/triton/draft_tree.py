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

"""Select and number the nodes of a dynamic draft tree (EAGLE-2 style).

Drafting records ``E`` scored candidates per request: token, cumulative
log-probability, parent candidate (``-1`` under the root) and depth. The tree
keeps the best ``N - 1`` of them under the root and numbers them depth first,
each node's children best first, so the most likely path is ``0, 1, 2, ..``.
Selection ranks candidates in parallel blocks; ordering runs on the kept nodes only.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

__all__ = ["draft_tree_expand", "draft_tree_finalize", "tree_ancestry"]


@triton.jit
def _finite_scores(scores_ptr, offs, ok):
    score = tl.load(scores_ptr + offs, mask=ok, other=float("-inf"))
    return tl.where(score == score, score, float("-inf"))


@triton.jit
def _draft_tree_select_kernel(
    scores_ptr,  # [bs, E] float32
    kept_ptr,  # [bs, N - 1] int32 out: kept entry of each global rank
    slot_ptr,  # [bs, E] int32 out: global rank of each kept entry
    E: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    CHUNK: tl.constexpr,
):
    """Program (req, block): global rank of BLOCK entries -- entries strictly
    better, ties to the lower entry id -- and the best N - 1 by rank."""
    req = tl.program_id(0)
    base = req * E
    ent = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    ent_ok = ent < E
    score = _finite_scores(scores_ptr, base + ent, ent_ok)
    rank = tl.zeros([BLOCK], dtype=tl.int32)
    for start in range(0, E, CHUNK):
        other = start + tl.arange(0, CHUNK)
        other_ok = other < E
        other_score = _finite_scores(scores_ptr, base + other, other_ok)
        better = (other_score[None, :] > score[:, None]) | (
            (other_score[None, :] == score[:, None]) & (other[None, :] < ent[:, None])
        )
        rank += tl.sum((better & other_ok[None, :]).to(tl.int32), axis=1)
    # A child never outranks its parent (score <= parent's, ties to the lower id).
    kept = ent_ok & (rank < N - 1)
    tl.store(kept_ptr + req * (N - 1) + rank, ent.to(tl.int32), mask=kept)
    tl.store(slot_ptr + base + ent, rank, mask=kept)


@triton.jit
def _draft_tree_order_kernel(
    parent_ptr,  # [bs, E] int64 parent candidate, -1 under the root
    depth_ptr,  # [bs, E] int64
    tokens_ptr,  # [bs, E] int64
    root_ptr,  # [bs] root token
    kept_ptr,  # [bs, N - 1] int32 kept entry by global rank
    slot_ptr,  # [bs, E] int32 global rank of each kept entry
    out_tokens_ptr,  # [bs, N] int32
    out_parent_ptr,  # [bs, N] int32
    sibling_ptr,  # [bs, K_PAD] int32 scratch
    pos_ptr,  # [bs, K_PAD] int32 scratch
    root_stride,
    E: tl.constexpr,
    N: tl.constexpr,
    MAX_DEPTH: tl.constexpr,
    RANK_BITS: tl.constexpr,
    K_PAD: tl.constexpr,
):
    """Number the kept entries depth first, children best first."""
    req = tl.program_id(0)
    base = req * E
    scratch = req * K_PAD
    slots = tl.arange(0, K_PAD)
    ok = slots < N - 1
    ent = tl.load(kept_ptr + req * (N - 1) + slots, mask=ok, other=0)
    parent = tl.load(parent_ptr + base + ent, mask=ok, other=-2)

    # Sibling rank: kept siblings of better global rank.
    ahead = (parent[None, :] == parent[:, None]) & (slots[None, :] < slots[:, None])
    sibling = tl.sum((ahead & ok[None, :]).to(tl.int32), axis=1)
    tl.store(sibling_ptr + scratch + slots, sibling)
    tl.debug_barrier()

    # Depth-first key: (sibling rank + 1) per level from the root down.
    key = tl.zeros([K_PAD], dtype=tl.int64)
    cursor = tl.where(ok, slots, -1)
    for _ in tl.static_range(MAX_DEPTH):
        live = cursor >= 0
        safe = tl.where(live, cursor, 0)
        cur_ent = tl.load(kept_ptr + req * (N - 1) + safe, mask=live, other=0)
        level = tl.load(depth_ptr + base + cur_ent, mask=live, other=0)
        digit = tl.load(sibling_ptr + scratch + safe, mask=live, other=0).to(tl.int64)
        key += tl.where(live, (digit + 1) << ((MAX_DEPTH - level) * RANK_BITS), 0)
        up = tl.load(parent_ptr + base + cur_ent, mask=live, other=-1)
        cursor = tl.where(
            up >= 0,
            tl.load(slot_ptr + base + tl.where(up >= 0, up, 0), mask=up >= 0, other=-1),
            -1,
        )

    pos = tl.sum(((key[None, :] < key[:, None]) & ok[None, :]).to(tl.int32), axis=1)
    tl.store(pos_ptr + scratch + slots, pos)
    tl.debug_barrier()

    has_parent = ok & (parent >= 0)
    parent_slot = tl.load(
        slot_ptr + base + tl.where(has_parent, parent, 0), mask=has_parent, other=0
    )
    parent_pos = tl.load(pos_ptr + scratch + parent_slot, mask=has_parent, other=-1)
    node = pos + 1
    tokens = tl.load(tokens_ptr + base + ent, mask=ok, other=0)
    tl.store(out_tokens_ptr + req * N + node, tokens.to(tl.int32), mask=ok)
    tl.store(out_parent_ptr + req * N + node, parent_pos + 1, mask=ok)
    tl.store(
        out_tokens_ptr + req * N, tl.load(root_ptr + req * root_stride).to(tl.int32)
    )
    tl.store(out_parent_ptr + req * N, -1)


@triton.jit
def _tree_ancestry_kernel(
    parent_ptr,  # [bs, N] int32
    depth_ptr,  # [bs, N] int32
    mask_ptr,  # [bs, N] int64
    N: tl.constexpr,
    N_PAD: tl.constexpr,
):
    req = tl.program_id(0)
    nodes = tl.arange(0, N_PAD)
    ok = nodes < N
    mask = tl.full([N_PAD], 1, tl.int64) << nodes.to(tl.int64)
    depth = tl.zeros([N_PAD], dtype=tl.int32)
    cursor = tl.load(parent_ptr + req * N + nodes, mask=ok, other=-1)
    live = cursor >= 0
    while tl.max(live.to(tl.int32), axis=0) > 0:
        safe = tl.where(live, cursor, 0)
        mask |= tl.where(live, tl.full([N_PAD], 1, tl.int64) << safe.to(tl.int64), 0)
        depth += live.to(tl.int32)
        cursor = tl.where(
            live, tl.load(parent_ptr + req * N + safe, mask=live, other=-1), -1
        )
        live = cursor >= 0
    tl.store(depth_ptr + req * N + nodes, depth, mask=ok)
    tl.store(mask_ptr + req * N + nodes, mask, mask=ok)


def tree_ancestry(
    parent: torch.Tensor, depth: torch.Tensor, mask: torch.Tensor
) -> None:
    """Depth and ancestor-or-self mask of every tree node.

    Args:
        parent: ``[bs, N]`` int32 parent node, ``-1`` for the root, every
            parent index below its child's; ``N <= 64``.
        depth: ``[bs, N]`` int32 output.
        mask: ``[bs, N]`` int64 output; bit ``j`` marks node ``j`` as an
            ancestor of, or equal to, the row's node.
    """
    bs, num_nodes = parent.shape
    if num_nodes > 64:
        raise ValueError(f"a 64-bit mask holds at most 64 nodes, got {num_nodes}")
    if bs == 0:
        return
    _tree_ancestry_kernel[(bs,)](
        parent,
        depth,
        mask,
        N=num_nodes,
        N_PAD=max(16, triton.next_power_of_2(num_nodes)),
    )


@triton.jit
def _draft_tree_expand_kernel(
    child_scores_ptr,  # [bs, K * K] float32, lane-major
    child_tokens_ptr,  # [bs, K * K] int64
    lane_scores_ptr,  # [bs, K] float32, updated in place
    lane_entry_ptr,  # [bs, K] int64, updated in place
    entry_scores_ptr,  # [bs, E] float32
    entry_parent_ptr,  # [bs, E] int64
    entry_depth_ptr,  # [bs, E] int64
    entry_tokens_ptr,  # [bs, E] int64
    lane_tokens_ptr,  # [bs, K] int64 out
    lane_mask_ptr,  # [bs, K] int64 lane ancestor masks, updated in place
    hidden_src_ptr,  # [bs * K, HIDDEN] this step's lane hidden rows
    hidden_dst_ptr,  # [bs * K, HIDDEN] next step's lane hidden rows
    stride_hidden_src,
    stride_hidden_dst,
    start,
    depth,
    num_entries,
    mask_bit_base,
    K: tl.constexpr,
    KK_PAD: tl.constexpr,
    PREPARE_NEXT: tl.constexpr,
    HIDDEN: tl.constexpr,
    HBLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    idx = tl.arange(0, KK_PAD)
    ok = idx < K * K
    lane = idx // K
    child = tl.load(child_scores_ptr + req * K * K + idx, mask=ok, other=float("-inf"))
    # A child never scores above its parent, which keeps finalize's kept set a tree.
    child = tl.where(child > 0.0, 0.0, child)
    token = tl.load(child_tokens_ptr + req * K * K + idx, mask=ok, other=0)
    score = tl.load(lane_scores_ptr + req * K + lane, mask=ok, other=0.0) + child
    # NaN (e.g. a padded request's garbage logits) ranks last, so ranks stay a permutation.
    score = tl.where(score == score, score, float("-inf"))
    parent = tl.load(lane_entry_ptr + req * K + lane, mask=ok, other=-1)

    entry = req * num_entries + start + idx
    tl.store(entry_scores_ptr + entry, score, mask=ok)
    tl.store(entry_parent_ptr + entry, parent, mask=ok)
    tl.store(entry_depth_ptr + entry, tl.full([KK_PAD], depth, tl.int64), mask=ok)
    tl.store(entry_tokens_ptr + entry, token, mask=ok)

    # Rank among the K * K children, ties to the lower index; the best K become the lanes.
    rank = tl.zeros([KK_PAD], dtype=tl.int32)
    for j in tl.static_range(K * K):
        other_child = tl.load(child_scores_ptr + req * K * K + j)
        other = tl.where(other_child > 0.0, 0.0, other_child) + tl.load(
            lane_scores_ptr + req * K + j // K
        )
        other = tl.where(other == other, other, float("-inf"))
        rank += ((other > score) | ((other == score) & (j < idx))).to(tl.int32)
    if PREPARE_NEXT:
        parent_mask = tl.load(lane_mask_ptr + req * K + lane, mask=ok, other=0)
    tl.debug_barrier()
    best = ok & (rank < K)
    tl.store(lane_scores_ptr + req * K + rank, score, mask=best)
    tl.store(lane_entry_ptr + req * K + rank, (start + idx).to(tl.int64), mask=best)
    tl.store(lane_tokens_ptr + req * K + rank, token, mask=best)
    if PREPARE_NEXT:
        # The next step's lane r: its parent lane's ancestors plus its own lane window slot.
        own_bit = tl.full([KK_PAD], 1, tl.int64) << (mask_bit_base + rank).to(tl.int64)
        tl.store(lane_mask_ptr + req * K + rank, parent_mask | own_bit, mask=best)
        cols = tl.arange(0, HBLOCK)
        for r in tl.static_range(K):
            src_lane = tl.sum(tl.where(best & (rank == r), lane, 0), axis=0)
            src = hidden_src_ptr + (req * K + src_lane).to(tl.int64) * stride_hidden_src
            dst = hidden_dst_ptr + (req * K + r).to(tl.int64) * stride_hidden_dst
            for c0 in range(0, HIDDEN, HBLOCK):
                c = c0 + cols
                tl.store(dst + c, tl.load(src + c, mask=c < HIDDEN), mask=c < HIDDEN)


def draft_tree_expand(
    child_scores: torch.Tensor,
    child_tokens: torch.Tensor,
    lane_scores: torch.Tensor,
    lane_entry: torch.Tensor,
    entry_scores: torch.Tensor,
    entry_parent: torch.Tensor,
    entry_depth: torch.Tensor,
    entry_tokens: torch.Tensor,
    *,
    start: int,
    depth: int,
    next_lanes: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
) -> torch.Tensor:
    """Record every lane's ``K`` children and keep the best ``K`` as new lanes.

    Args:
        child_scores: ``[bs, K * K]`` float32 child log-probabilities, lane-major.
        child_tokens: ``[bs, K * K]`` int64 child tokens.
        lane_scores: ``[bs, K]`` float32 cumulative lane scores, updated in place.
        lane_entry: ``[bs, K]`` int64 candidate id of each lane, updated in place.
        entry_scores: ``[bs, E]`` float32 candidate record; children land at
            ``[start, start + K * K)``, as do ``entry_parent`` (int64),
            ``entry_depth`` (int64) and ``entry_tokens`` (int64).
        start: first candidate id of this step's children.
        depth: depth of this step's children.
        next_lanes: ``(lane_mask, hidden_src, hidden_dst)`` to prepare the next
            drafting step, or ``None`` after the last one. ``lane_mask`` is the
            ``[bs, K]`` int64 lane ancestor masks, updated in place: new lane
            ``r`` gets its parent lane's mask plus bit ``depth * K - K + r``
            (its slot in the lane window, counting from the first
            expanded step); ``hidden_dst[b * K + r] = hidden_src[b * K + parent]``.

    Returns:
        ``[bs, K]`` int64 token of each new lane, best first.
    """
    bs, topk = lane_scores.shape
    lane_tokens = torch.empty((bs, topk), dtype=torch.int64, device=lane_scores.device)
    if bs == 0:
        return lane_tokens
    if next_lanes is None:
        lane_mask, hidden_src, hidden_dst = lane_tokens, lane_tokens, lane_tokens
        hidden = 1
    else:
        lane_mask, hidden_src, hidden_dst = next_lanes
        hidden = hidden_src.shape[1]
    _draft_tree_expand_kernel[(bs,)](
        child_scores,
        child_tokens,
        lane_scores,
        lane_entry,
        entry_scores,
        entry_parent,
        entry_depth,
        entry_tokens,
        lane_tokens,
        lane_mask,
        hidden_src,
        hidden_dst,
        hidden_src.stride(0),
        hidden_dst.stride(0),
        start,
        depth,
        entry_scores.shape[1],
        (depth - 1) * topk,
        K=topk,
        KK_PAD=max(16, triton.next_power_of_2(topk * topk)),
        PREPARE_NEXT=next_lanes is not None,
        HIDDEN=hidden,
        HBLOCK=1024,
    )
    return lane_tokens


def draft_tree_finalize(
    scores: torch.Tensor,
    parent: torch.Tensor,
    depth: torch.Tensor,
    tokens: torch.Tensor,
    root_tokens: torch.Tensor,
    *,
    num_nodes: int,
    max_depth: int,
    rank_bits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the best ``num_nodes - 1`` candidates under the root, depth first.

    Args:
        scores: ``[bs, E]`` float32 cumulative log-probability per candidate;
            a child never scores above its parent (``draft_tree_expand``
            guarantees it).
        parent: ``[bs, E]`` int64 parent candidate, ``-1`` under the root; a
            parent precedes its children.
        depth: ``[bs, E]`` int64 candidate depth, 1 under the root.
        tokens: ``[bs, E]`` int64 candidate token.
        root_tokens: ``[bs]`` token of node 0, any stride.
        num_nodes: ``N``, nodes including the root; ``N - 1 <= E``.
        max_depth: deepest candidate depth.
        rank_bits: bits per level of the depth-first key; ``max_depth *
            rank_bits <= 62`` and every sibling rank + 1 fits.

    Returns:
        ``(tokens, parent)``: ``[bs, N]`` int32 node tokens and parent nodes
        (``-1`` for the root).
    """
    bs, num_entries = scores.shape
    if not 1 <= num_nodes - 1 <= num_entries:
        raise ValueError(f"cannot keep {num_nodes - 1} of {num_entries} candidates")
    if max_depth * rank_bits > 62:
        raise ValueError(f"depth {max_depth} x {rank_bits} bits overflows the key")
    device = scores.device
    out_tokens = torch.empty((bs, num_nodes), dtype=torch.int32, device=device)
    out_parent = torch.empty((bs, num_nodes), dtype=torch.int32, device=device)
    if bs == 0:
        return out_tokens, out_parent
    kept = torch.empty((bs, num_nodes - 1), dtype=torch.int32, device=device)
    slot = torch.empty((bs, num_entries), dtype=torch.int32, device=device)
    block = 64
    _draft_tree_select_kernel[(bs, triton.cdiv(num_entries, block))](
        scores,
        kept,
        slot,
        E=num_entries,
        N=num_nodes,
        BLOCK=block,
        CHUNK=64,
    )
    k_pad = max(16, triton.next_power_of_2(num_nodes - 1))
    sibling = torch.empty((bs, k_pad), dtype=torch.int32, device=device)
    pos = torch.empty_like(sibling)
    _draft_tree_order_kernel[(bs,)](
        parent,
        depth,
        tokens,
        root_tokens,
        kept,
        slot,
        out_tokens,
        out_parent,
        sibling,
        pos,
        root_tokens.stride(0),
        E=num_entries,
        N=num_nodes,
        MAX_DEPTH=max_depth,
        RANK_BITS=rank_bits,
        K_PAD=k_pad,
    )
    return out_tokens, out_parent

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

"""EAGLE-Harness draft-tree bookkeeping kernels (EAGLE-4); see
``csrc/eagle_harness_tree.cu`` for the node pool layout."""

import functools
from pathlib import Path

import torch

__all__ = [
    "eagle_harness_draft_depth_update",
    "eagle_harness_publish_draft_q",
    "eagle_harness_residual_sample",
    "eagle_harness_sample_children",
    "eagle_harness_tree_reject_sample",
    "eagle_harness_stable_topk_desc",
    "eagle_harness_update_node_data",
    "eagle_harness_work_deltas",
    "eagle_harness_write_verify_tree",
]


@functools.cache
def _module():
    import tvm_ffi

    so_path = (
        Path(__file__).parent / "objs" / "eagle_harness_tree" / "eagle_harness_tree.so"
    )
    if not so_path.exists():
        raise RuntimeError(
            f"tokenspeed_kernel eagle_harness_tree library not found at {so_path}. "
            "Run: pip install -e tokenspeed_kernel/python/"
        )
    return tvm_ffi.load_module(str(so_path))


def eagle_harness_update_node_data(
    pool_scores: torch.Tensor,
    pool_tokens: torch.Tensor,
    work_scores: torch.Tensor,
    sort_scores: torch.Tensor,
    child_scores: torch.Tensor,
    child_tokens: torch.Tensor,
) -> None:
    """Write the root's ``[bs, K]`` children into columns ``1..K`` of the pool,
    work and sort lists."""
    _module().eagle_harness_update_node_data(
        pool_scores, pool_tokens, work_scores, sort_scores, child_scores, child_tokens
    )


def eagle_harness_work_deltas(
    work_scores: torch.Tensor,
    work_deltas: torch.Tensor,
    live_rows: torch.Tensor,
    node_count: torch.Tensor,
    verify_num: torch.Tensor,
) -> None:
    """``work_deltas[j]`` (contiguous fp32 ``[K + 1]``): the mean over the
    first ``live_rows`` rows of the best ``min(node_count + j, verify_num)``
    scores of each row's two sorted segments (``node_count`` best nodes, then
    the next depth's ``j`` best frontier scores). The counts are device int32
    ``[1]`` (a captured launch serves every round); a zero ``node_count``
    writes nothing. ``work_scores`` holds at most 512 rows; up to 256 live rows
    the batch sum's order is that of a 256-row sum."""
    _module().eagle_harness_work_deltas(
        work_scores, work_deltas, live_rows, node_count, verify_num
    )


def eagle_harness_draft_depth_update(
    child_probs: torch.Tensor,
    child_tokens: torch.Tensor,
    expanded: torch.Tensor,
    child_hidden: torch.Tensor,
    frontier_tokens: torch.Tensor,
    frontier_scores: torch.Tensor,
    frontier_hidden: torch.Tensor,
    frontier_index: torch.Tensor,
    pool_tokens: torch.Tensor,
    pool_scores: torch.Tensor,
    pool_prev: torch.Tensor,
    work_scores: torch.Tensor,
    sort_scores: torch.Tensor,
    positions: torch.Tensor,
    slot_starts: torch.Tensor,
    lane_mask: torch.Tensor,
    width: torch.Tensor,
    depth: int,
    cumu_count: torch.Tensor,
    verify_num: torch.Tensor,
    num_slots: int,
) -> None:
    """One draft depth's bookkeeping after its lane forward, in one launch
    (one CTA per request row; per-request tensors are contiguous ``[rows,
    ...]``). ``width``, ``cumu_count`` and ``verify_num`` are device int32
    ``[1]`` counts, so a captured launch serves every round.

    The forward expanded the first ``width`` of each request's ``K`` lanes:
    ``child_probs`` / ``child_tokens`` (``[rows, K, K]`` fp32 / int64, lanes
    past ``width`` ignored) are their children, ``expanded`` (``[rows, K]``
    int64) their pool indices and ``child_hidden`` (``[rows, K, H]``, not
    overlapping ``frontier_hidden``) their output hidden rows. A child's path
    score is its probability times its lane's ``frontier_scores`` entry. The
    children, sorted descending by the bitonic network the frontier's tie
    order comes from, give the next frontier, the best ``K`` (``[rows, K]``):
    ``frontier_tokens``, ``frontier_scores``, ``frontier_index`` (pool
    indices) and ``frontier_hidden`` (``[rows, K, H]``, their lanes'
    ``child_hidden`` rows). All children are appended to the pool
    (``pool_*``, ``[rows, nodes]``) at node ``cumu_count`` on: their tokens,
    ``path score * 0.9999^(depth + 1)`` and their parents' pool indices.
    ``sort_scores`` keeps the best ``verify_num`` path scores of its first
    ``min(verify_num, cumu_count)`` and the children (stable, ties to the
    earlier); ``work_scores`` gets that head and the frontier's scores for
    the next work deltas.

    The lanes move one slot on: ``positions`` (``[rows, K]``) + 1,
    ``slot_starts`` (``[rows]``) + K. When depth ``depth + 1`` fits in the
    ``num_slots``-slot lane window, ``lane_mask`` (``[rows, K, words]``
    int64, window slot ``j`` at bit ``j % 64`` of word ``j // 64``) becomes
    the next depth's: lane ``r``'s parent lane's row plus slot
    ``depth * K + r``.
    """
    _module().eagle_harness_draft_depth_update(
        child_probs,
        child_tokens,
        expanded,
        child_hidden,
        frontier_tokens,
        frontier_scores,
        frontier_hidden,
        frontier_index,
        pool_tokens,
        pool_scores,
        pool_prev,
        work_scores,
        sort_scores,
        positions,
        slot_starts,
        lane_mask,
        width,
        depth,
        cumu_count,
        verify_num,
        num_slots,
    )


def eagle_harness_stable_topk_desc(
    scores: torch.Tensor,
    live_rows: torch.Tensor,
    n: torch.Tensor,
    k: torch.Tensor,
    out_values: torch.Tensor,
    out_indices: torch.Tensor,
) -> None:
    """The first ``k`` entries of a stable descending sort (NaN first, ties to
    the lower index) of the first ``n`` columns of each of the first
    ``live_rows`` rows of ``[rows, <= 1024]`` float32 ``scores``, into the
    first ``k`` columns of ``out_values`` / ``out_indices`` (``[rows, max_k]``
    float32 / int64). ``live_rows``, ``n`` and ``k`` (``1 <= k <= n``) are
    device int32 ``[1]``."""
    _module().eagle_harness_stable_topk_desc(
        scores, live_rows, n, k, out_values, out_indices
    )


def eagle_harness_write_verify_tree(
    ranked: torch.Tensor,
    pool_tokens: torch.Tensor,
    pool_prev: torch.Tensor,
    rows: torch.Tensor,
    input_map: torch.Tensor,
    parent_map: torch.Tensor,
    live_rows: torch.Tensor,
    verify_num: torch.Tensor,
) -> None:
    """Write each of the first ``live_rows`` requests' verify tree into its
    ``rows`` of the next round's ``input_map`` / ``parent_map`` (``[*, N]``
    int32): slot ``i < verify_num`` is pool node ``ranked[b, i]``
    (ancestor-first) with its parent's slot, the rest repeat the root (pool
    node 0) with parent -1. ``live_rows`` and ``verify_num`` are device int32
    ``[1]``."""
    _module().eagle_harness_write_verify_tree(
        ranked,
        pool_tokens,
        pool_prev,
        rows,
        input_map,
        parent_map,
        live_rows,
        verify_num,
    )


def eagle_harness_sample_children(
    logits: torch.Tensor,
    candidates: torch.Tensor,
    tokens: torch.Tensor,
    scores: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    q_store: torch.Tensor,
    qid_store: torch.Tensor,
    node_slot: torch.Tensor,
    node_pos: torch.Tensor,
    width: torch.Tensor | None,
    slot_base: int,
    cumu_start: torch.Tensor,
    vocab_limit: int,
    seed: int,
    round_counter: torch.Tensor,
    launch: int,
) -> None:
    """Draw each expanded row's children from its draft distribution.

    The rows are ``R`` per request, as ``[B, R, *]`` views read through their
    strides, of which each request expands its first ``width`` (a device
    int32 ``[1]``; None: all ``R``). ``cumu_start`` is a device int32 ``[1]``. For rows of requests with ``temperatures > 0``,
    overwrite ``tokens`` (``[B, R, K]`` int64) with ``K`` candidates drawn
    without replacement from the softmax of ``logits / temperature`` over
    the row's candidates (``[B, R, NC]`` fp32 logits, int64 ids), cut to the
    request's top-k / top-p; ``scores`` (``[B, R, K]``, q's sorted top-K)
    are kept, a child past the cut's support set to 0. Row ``r``'s q and ids
    go to slot ``slot_base + r`` of ``q_store`` / ``qid_store``
    (``[B, S, NC]``) and each child's slot and candidate position to
    ``node_slot`` / ``node_pos`` (``[B, nodes]`` int32) at pool nodes
    ``cumu_start + r * K`` on; greedy rows keep their children with slot -1.
    The draws come from Philox stream ``round_counter[0] * 64 + launch``
    keyed by expanded row ``b * width + r`` and candidate, so a captured
    graph draws afresh as the device counter moves.
    """
    _module().eagle_harness_sample_children(
        logits,
        candidates,
        tokens,
        scores,
        temperatures,
        top_ks,
        top_ps,
        q_store,
        qid_store,
        node_slot,
        node_pos,
        width,
        slot_base,
        cumu_start,
        vocab_limit,
        seed,
        round_counter,
        launch,
    )


def eagle_harness_tree_reject_sample(
    probas: torch.Tensor,
    tokens: torch.Tensor,
    parents: torch.Tensor,
    q_rows: torch.Tensor | None,
    q_slot: torch.Tensor,
    q_pos: torch.Tensor,
    q_store: torch.Tensor,
    qid_store: torch.Tensor,
    coins: torch.Tensor,
    top_ks: torch.Tensor,
    argmax: torch.Tensor,
    path: torch.Tensor,
    predicts: torch.Tensor,
    accept_length: torch.Tensor,
    last_row: torch.Tensor,
) -> None:
    """Accept each request's draft tree against the target by rejection sampling.

    ``probas`` (``[bs * N, V]`` fp32) holds the target's distribution at every
    node row; the rows of tried nodes become their residuals. A node's children
    (nodes whose ``parents`` entry it is, in index order) are tried in turn:
    sampled ones (``q_slot`` >= 0 at the request's ``q_rows`` row of the
    ``[R, N]`` tables, with q in ``q_store`` / ``qid_store`` ``[R, S, NC]``) by
    recursive rejection without replacement, the others by their target mass
    over the mass left; ``coins`` (``[bs, N]``) hold one uniform per child.
    ``q_rows`` None: every child is deterministic. A sampled row whose residual
    vanishes takes the target's argmax (``torch.argmax``'s) as its bonus.

    Greedy rows (``top_ks``, ``[bs * N]`` int32, equal to 1) have a one-hot
    target at ``argmax`` (``[bs * N]`` int64, read for them only): the first
    child holding it is accepted, the others are rejected, and their
    ``probas`` rows are never read (they may be unwritten).

    Writes ``path`` (``[bs, N]`` int32, accepted nodes root first, -1 past the
    path), ``predicts`` (``[bs * N]`` int32; the accepted children's tokens, the
    bonus slot left for ``eagle_harness_residual_sample``), ``accept_length``
    (``[bs]``, accepted drafts plus the bonus) and ``last_row`` (``[bs]`` int64,
    the last accepted node's row of ``probas``).
    """
    _module().eagle_harness_tree_reject_sample(
        probas,
        tokens,
        parents,
        q_rows,
        q_slot,
        q_pos,
        q_store,
        qid_store,
        coins,
        top_ks,
        argmax,
        path,
        predicts,
        accept_length,
        last_row,
    )


def eagle_harness_residual_sample(
    probas: torch.Tensor,
    last_row: torch.Tensor,
    coins: torch.Tensor,
    top_ks: torch.Tensor,
    argmax: torch.Tensor,
    accept_length: torch.Tensor,
    predicts: torch.Tensor,
    num_nodes: int,
) -> None:
    """Draw each request's bonus token from row ``last_row[b]`` of ``probas``
    (normalized or not) by inverse CDF with ``coins[b]``, into the end of its
    accepted path in ``predicts`` (``[bs * num_nodes]``). A greedy last row
    (``top_ks`` 1, per ``probas`` row) gives its ``argmax`` (int64, per
    ``probas`` row) instead, its ``probas`` row unread."""
    _module().eagle_harness_residual_sample(
        probas, last_row, coins, top_ks, argmax, accept_length, predicts, num_nodes
    )


def eagle_harness_publish_draft_q(
    ranked: torch.Tensor,
    node_slot: torch.Tensor,
    node_pos: torch.Tensor,
    q_round: torch.Tensor,
    qid_round: torch.Tensor,
    rows: torch.Tensor,
    q_tab: torch.Tensor,
    qid_tab: torch.Tensor,
    slot_tab: torch.Tensor,
    pos_tab: torch.Tensor,
    live_rows: torch.Tensor,
    verify_num: torch.Tensor,
) -> None:
    """Copy each of the first ``live_rows`` requests' round q (``[rows, S,
    NC]``) into row ``rows[b]`` of the ``[R, S, NC]`` tables, and write per
    verify node ``i < verify_num`` (pool node ``ranked[b, i]``) its parent's
    slot and candidate position into the ``[R, N]`` tables; -1 for the root
    and slots past ``verify_num``. ``live_rows`` and ``verify_num`` are device
    int32 ``[1]``."""
    _module().eagle_harness_publish_draft_q(
        ranked,
        node_slot,
        node_pos,
        q_round,
        qid_round,
        rows,
        q_tab,
        qid_tab,
        slot_tab,
        pos_tab,
        live_rows,
        verify_num,
    )

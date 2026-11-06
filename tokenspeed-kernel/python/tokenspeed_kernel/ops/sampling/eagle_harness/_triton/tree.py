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

"""Triton kernels of the EAGLE-Harness draft tree.

Each request owns one row of a node pool (``pool_tokens`` / ``pool_scores``
/ ``pool_prev``, ``[rows, nodes]``): node 0 is the root, then the children
in expansion order, each with its token, its decayed path score and its
parent's pool index. Alongside it, ``work_scores`` (``[rows, >= verify_num +
K]``) holds the best path scores so far followed by the next frontier's,
and ``sort_scores`` (``[rows, verify_num]``) the best path scores so far in
descending order. ``K`` is the lane count (children kept per depth).

Every kernel fixes its arithmetic and reduction order and is launched
without FMA contraction, so its results are deterministic bit for bit.
Kernels that rewrite a buffer they also read (the frontier scores, the sort
list, the lane masks) load everything first and store after a CTA barrier.
"""

from __future__ import annotations

from tokenspeed_kernel._triton import libdevice, tl, triton

# The depth update's sort network size (width * K children at most) and sort list head.
MAX_CHILDREN = 64
MAX_VERIFY = 256
MAX_MASK_WORDS = 4


@triton.jit(do_not_specialize=["rows", "node_stride", "work_stride", "sort_stride"])
def _update_node_data_kernel(
    pool_scores_ptr,
    pool_tokens_ptr,
    work_scores_ptr,
    sort_scores_ptr,
    child_scores_ptr,  # [rows, K] contiguous
    child_tokens_ptr,
    rows,
    node_stride,
    work_stride,
    sort_stride,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Scatters the root's children into pool nodes 1..K.

    Element ``i`` of the flattened contiguous ``[rows, K]`` children goes to
    row ``i // K``, column ``i % K + 1`` of the pool scores and tokens, the
    work list and the sort list. Grid: ``cdiv(rows * K, BLOCK)`` programs of
    ``BLOCK`` elements; ``K`` is constexpr, the row count and row strides are
    runtime values.
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = idx < rows * K
    b = (idx // K).to(tl.int64)
    node = idx % K + 1  # the root's children are nodes 1..K
    score = tl.load(child_scores_ptr + idx, mask=ok)
    token = tl.load(child_tokens_ptr + idx, mask=ok)
    tl.store(pool_scores_ptr + b * node_stride + node, score, mask=ok)
    tl.store(pool_tokens_ptr + b * node_stride + node, token, mask=ok)
    tl.store(work_scores_ptr + b * work_stride + node, score, mask=ok)
    tl.store(sort_scores_ptr + b * sort_stride + node, score, mask=ok)


@triton.jit(do_not_specialize=["work_stride"])
def _work_deltas_kernel(
    work_scores_ptr,
    work_deltas_ptr,
    live_rows_ptr,
    node_count_ptr,
    verify_num_ptr,
    work_stride,
    THREADS: tl.constexpr,
    LOG_THREADS: tl.constexpr,
):
    """``work_deltas[j]``: the mean over the first ``live_rows`` rows of the
    sum of each row's best ``min(node_count + j, verify_num)`` scores.

    A row's scores are two descending segments of its work list: the
    ``node_count`` best nodes so far, then the next frontier's scores. The
    row takes its best entries by a two-way merge (ties to the first
    segment). Grid: one program per ``j``. Each program holds one row per
    lane of a ``THREADS``-lane vector (``THREADS`` a power of two covering
    the rows, ``LOG_THREADS = log2(THREADS)``; padded lanes are zero) and
    sums them by a halving tree (lane ``t`` adds lane ``t + THREADS / 2``,
    then ``THREADS / 4``, ...), then divides once, correctly rounded, by the
    live row count. The counts are device int32 ``[1]`` scalars; a zero
    ``node_count`` writes nothing.
    """
    node_count = tl.load(node_count_ptr)
    if node_count != 0:
        j = tl.program_id(0)
        batch_size = tl.load(live_rows_ptr)
        verify_num = tl.load(verify_num_ptr)
        row = tl.arange(0, THREADS)
        live = row < batch_size
        head = work_scores_ptr + row.to(tl.int64) * work_stride
        tail = head + node_count
        k = tl.minimum(node_count + j, verify_num)
        pa = tl.zeros([THREADS], dtype=tl.int32)
        pb = tl.zeros([THREADS], dtype=tl.int32)
        total = tl.zeros([THREADS], dtype=tl.float32)
        for _ in range(0, k):
            va = tl.load(head + pa, mask=live & (pa < node_count), other=0.0)
            vb = tl.load(tail + pb, mask=live & (pb < j), other=0.0)
            take_a = va >= vb
            total = tl.where(live, total + tl.where(take_a, va, vb), total)
            pa += take_a.to(tl.int32)
            pb += (~take_a).to(tl.int32)
        # The halving tree: lane t adds lane t + half, half = THREADS / 2 .. 1.
        for _ in tl.static_range(LOG_THREADS):
            total = tl.sum(tl.reshape(total, (2, total.shape[0] // 2)), axis=0)
        mean = tl.math.div_rn(tl.reshape(total, ()), batch_size.to(tl.float32))
        tl.store(work_deltas_ptr + j, mean)


@triton.jit(
    do_not_specialize=[
        "node_stride",
        "work_stride",
        "sort_stride",
        "depth",
        "num_slots",
        "hidden_size",
    ]
)
def _draft_depth_update_kernel(
    child_probs_ptr,  # [rows, K, K]
    child_tokens_ptr,  # [rows, K, K]
    expanded_ptr,  # [rows, K]
    child_hidden_ptr,  # [rows, K, H]
    frontier_tokens_ptr,  # [rows, K]
    frontier_scores_ptr,  # [rows, K]
    frontier_hidden_ptr,  # [rows, K, H]
    frontier_index_ptr,  # [rows, K]
    pool_tokens_ptr,
    pool_scores_ptr,
    pool_prev_ptr,
    work_scores_ptr,
    sort_scores_ptr,
    positions_ptr,  # [rows, K]
    slot_starts_ptr,  # [rows]
    lane_mask_ptr,  # [rows, K, MASK_WORDS]
    width_ptr,
    cumu_count_ptr,
    verify_num_ptr,
    node_stride,
    work_stride,
    sort_stride,
    depth,
    num_slots,
    hidden_size,
    K: tl.constexpr,
    K_PAD: tl.constexpr,
    MASK_WORDS: tl.constexpr,
    MASK_PAD: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """One draft depth's bookkeeping for one request (grid: one program per
    request row).

    The first ``width`` of the row's ``K`` frontier lanes were expanded,
    each into ``K`` children (``child_probs`` / ``child_tokens``, ``[rows,
    K, K]``). A child's path score is its probability times its lane's
    frontier score. The program:

    * sorts the ``n = width * K`` children by path score with a descending
      64-entry bitonic network (entries past ``n`` never swap; equal scores
      never swap, so the order is deterministic);
    * makes the best ``K`` the next frontier (``frontier_tokens``,
      ``frontier_scores``, ``frontier_index`` = their pool indices, and
      ``frontier_hidden`` = the ``child_hidden`` rows of their parent
      lanes);
    * appends all ``n`` children to the pool at node ``cumu_count``:
      tokens, ``path score * DEPTH_DECAY ** (depth + 1)`` and their parents'
      pool indices (``expanded``);
    * merges the children into the sort list: each entry of
      ``[sort list head, sorted children]`` lands at its stable descending
      rank (ties to the earlier entry), keeping the best ``verify_num``;
      the work list gets the head followed by the new frontier's scores;
    * moves each lane one slot on (``positions + 1``, ``slot_starts + K``)
      and, when depth ``depth + 1`` fits in the ``num_slots``-slot window,
      sets lane ``r``'s ``lane_mask`` (``MASK_WORDS`` int64 words, slot
      ``j`` at bit ``j % 64`` of word ``j // 64``) to its parent lane's mask
      plus slot ``depth * K + r``.

    ``K`` (with ``K * K <= 64``), ``K_PAD`` / ``MASK_PAD`` (powers of two
    covering ``K`` / ``MASK_WORDS``) and ``BLOCK_H`` (hidden columns copied
    per step) are constexpr; ``width``, ``cumu_count`` and ``verify_num``
    are device int32 ``[1]`` scalars, so one compiled launch serves every
    round.
    """
    # Draft depth d >= 1's children enter the pool at path score * DEPTH_DECAY^(d + 1).
    DEPTH_DECAY: tl.constexpr = 0.9999

    b = tl.program_id(0).to(tl.int64)
    width = tl.load(width_ptr)
    cumu_count = tl.load(cumu_count_ptr)
    verify_num = tl.load(verify_num_ptr)
    n = width * K
    head = tl.minimum(
        verify_num, cumu_count
    )  # sort list entries merged with the children
    kept = tl.minimum(verify_num, cumu_count + n)  # sort list entries after the merge
    lanes = b * K
    node_row = b * node_stride
    sort_row = sort_scores_ptr + b * sort_stride
    work_row = work_scores_ptr + b * work_stride

    # --- Loads (every buffer rewritten below is read before the barrier).
    p = tl.arange(0, 64)
    in_lanes = p < K * K
    probs = tl.load(child_probs_ptr + lanes * K + p, mask=in_lanes, other=0.0)
    parent_score = tl.load(
        frontier_scores_ptr + lanes + p // K, mask=in_lanes, other=0.0
    )
    score = probs * parent_score
    tokens = tl.load(child_tokens_ptr + lanes * K + p, mask=in_lanes, other=0)
    prev = tl.load(expanded_ptr + lanes + p // K, mask=in_lanes, other=0)

    # Descending bitonic sort of 64 positions; pairs reaching past n skip, ties never swap.
    v = tl.where(p < n, score, -1e10)
    x = p
    for size_log in tl.static_range(1, 7):
        size = 1 << size_log
        for stride_log in tl.static_range(size_log - 1, -1, -1):
            stride = 1 << stride_log
            partner = p ^ stride
            o = tl.gather(v, partner, 0)
            ox = tl.gather(x, partner, 0)
            lower = (p & stride) == 0
            descending = (p & size) == 0
            swap = ((p | stride) < n) & tl.where(lower == descending, v < o, v > o)
            v = tl.where(swap, o, v)
            x = tl.where(swap, ox, x)

    # The frontier: the best K children, their parent lanes.
    lane = tl.arange(0, K_PAD)
    lane_ok = lane < K
    lane_safe = tl.where(lane_ok, lane, 0)
    best_child = tl.gather(x, lane_safe, 0)
    best_score = tl.gather(v, lane_safe, 0)
    best_parent = best_child // K
    best_token = tl.load(
        child_tokens_ptr + lanes * K + best_child, mask=lane_ok, other=0
    )

    # The stable rank of each entry of [sort list head, sorted children]: the best
    # `kept` land at their rank.
    total = head + n
    e = tl.arange(0, 512)
    e_head = e < head
    e_ok = e < total
    key = tl.where(
        e_head,
        tl.load(sort_row + e, mask=e_head, other=0.0),
        tl.gather(v, tl.where(e_ok & ~e_head, e - head, 0), 0),
    )
    rank = tl.zeros([512], dtype=tl.int32)
    for start in range(0, total, 32):
        jj = start + tl.arange(0, 32)
        j_head = jj < head
        j_ok = jj < total
        other = tl.where(
            j_head,
            tl.load(sort_row + jj, mask=j_head, other=0.0),
            tl.gather(v, tl.where(j_ok & ~j_head, jj - head, 0), 0),
        )
        before = (other[None, :] > key[:, None]) | (
            (other[None, :] == key[:, None]) & (jj[None, :] < e[:, None])
        )
        rank += tl.sum((before & j_ok[None, :]).to(tl.int32), axis=1)

    # The next depth's lane masks: the parent lane's, plus the lane's own slot.
    word = tl.arange(0, MASK_PAD)
    mask_ok = lane_ok[:, None] & (word[None, :] < MASK_WORDS)
    write_masks = (depth + 1) * K <= num_slots
    old_mask = tl.load(
        lane_mask_ptr + (lanes + best_parent[:, None]) * MASK_WORDS + word[None, :],
        mask=mask_ok & write_masks,
        other=0,
    )
    slot = depth * K + lane
    own = tl.where(
        (slot // 64)[:, None] == word[None, :],
        tl.full([K_PAD, MASK_PAD], 1, tl.int64) << (slot % 64).to(tl.int64)[:, None],
        0,
    )
    new_mask = old_mask | own
    positions = tl.load(positions_ptr + lanes + lane, mask=lane_ok, other=0)
    slot_start = tl.load(slot_starts_ptr + b)

    tl.debug_barrier()

    # --- Stores.
    decay = libdevice.pow(
        tl.full([], DEPTH_DECAY, tl.float32), (depth + 1).to(tl.float32)
    )
    child_ok = p < n
    tl.store(pool_tokens_ptr + node_row + cumu_count + p, tokens, mask=child_ok)
    tl.store(pool_scores_ptr + node_row + cumu_count + p, score * decay, mask=child_ok)
    tl.store(pool_prev_ptr + node_row + cumu_count + p, prev, mask=child_ok)

    tl.store(frontier_index_ptr + lanes + lane, cumu_count + best_child, mask=lane_ok)
    tl.store(frontier_scores_ptr + lanes + lane, best_score, mask=lane_ok)
    tl.store(frontier_tokens_ptr + lanes + lane, best_token, mask=lane_ok)
    tl.store(work_row + head + lane, best_score, mask=lane_ok)
    tl.store(work_row + e, key, mask=e_head)
    tl.store(sort_row + rank, key, mask=e_ok & (rank < kept))
    tl.store(
        lane_mask_ptr + (lanes + lane[:, None]) * MASK_WORDS + word[None, :],
        new_mask,
        mask=mask_ok & write_masks,
    )
    tl.store(positions_ptr + lanes + lane, positions + 1, mask=lane_ok)
    tl.store(slot_starts_ptr + b, slot_start + K)

    # The frontier's input hidden rows: their parent lanes' output rows.
    src = child_hidden_ptr + (lanes + best_parent).to(tl.int64)[:, None] * hidden_size
    dst = frontier_hidden_ptr + (lanes + lane).to(tl.int64)[:, None] * hidden_size
    for h0 in range(0, hidden_size, BLOCK_H):
        h = h0 + tl.arange(0, BLOCK_H)
        ok = lane_ok[:, None] & (h < hidden_size)[None, :]
        tl.store(dst + h[None, :], tl.load(src + h[None, :], mask=ok), mask=ok)


@triton.jit
def _argmax_before(v, i, w, j):
    """Whether ``(v, i)`` precedes ``(w, j)``: NaN first, then the larger
    value, then the lower index."""
    v_nan = v != v
    w_nan = w != w
    return tl.where(v_nan != w_nan, v_nan, tl.where(~v_nan & (v != w), v > w, i < j))


@triton.jit(do_not_specialize=["row_stride", "out_stride"])
def _stable_topk_desc_kernel(
    scores_ptr,
    live_rows_ptr,
    n_ptr,
    k_ptr,
    out_values_ptr,
    out_indices_ptr,
    row_stride,
    out_stride,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
):
    """The first ``k`` entries of a stable descending sort of each live
    row's first ``n`` scores (grid: one program per row; rows at or past
    ``live_rows`` exit).

    Order: NaN first, then the larger value, then the lower index. Each
    score's output position is the number of scores preceding it in that
    strict total order, counted ``CHUNK`` columns at a time, so the result
    is unique and does not depend on the count's order. ``BLOCK_N`` (a
    power of two covering the columns) and ``CHUNK`` are constexpr;
    ``live_rows``, ``n`` and ``k`` are device int32 ``[1]`` scalars.
    """
    row = tl.program_id(0)
    if row < tl.load(live_rows_ptr):
        n = tl.load(n_ptr)
        k = tl.load(k_ptr)
        base = scores_ptr + row.to(tl.int64) * row_stride
        i = tl.arange(0, BLOCK_N)
        i_ok = i < n
        v = tl.load(base + i, mask=i_ok, other=0.0)
        rank = tl.zeros([BLOCK_N], dtype=tl.int32)
        for start in range(0, n, CHUNK):
            j = start + tl.arange(0, CHUNK)
            j_ok = j < n
            w = tl.load(base + j, mask=j_ok, other=0.0)
            before = _argmax_before(w[None, :], j[None, :], v[:, None], i[:, None])
            rank += tl.sum((before & j_ok[None, :]).to(tl.int32), axis=1)
        out = row.to(tl.int64) * out_stride + rank
        keep = i_ok & (rank < k)
        tl.store(out_values_ptr + out, v, mask=keep)
        tl.store(out_indices_ptr + out, i.to(tl.int64), mask=keep)


@triton.jit(
    do_not_specialize=["ranked_stride", "node_stride", "input_stride", "parent_stride"]
)
def _write_verify_tree_kernel(
    ranked_ptr,
    pool_tokens_ptr,
    pool_prev_ptr,
    rows_ptr,
    input_map_ptr,
    parent_map_ptr,
    live_rows_ptr,
    verify_num_ptr,
    ranked_stride,
    node_stride,
    input_stride,
    parent_stride,
    NUM_SLOTS: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Writes one request's verify tree into row ``rows[b]`` of the next
    round's ``input_map`` / ``parent_map`` (grid: one program per request;
    requests at or past ``live_rows`` exit).

    Slot ``i < verify_num`` holds pool node ``ranked[b, i]`` (ranked nodes
    come ancestor first): its token, and as parent the latest earlier slot
    holding its pool parent. The remaining slots repeat the root's token
    with parent -1. ``NUM_SLOTS`` (the map width, at most
    ``MAX_TREE_NODES``) and ``BLOCK_S`` (a power of two covering it) are
    constexpr; ``live_rows`` and ``verify_num`` are device int32 ``[1]``.
    """
    # Verify tree nodes per request.
    MAX_TREE_NODES: tl.constexpr = 64

    b = tl.program_id(0)
    if b < tl.load(live_rows_ptr):
        verify_num = tl.load(verify_num_ptr)
        ranked_row = ranked_ptr + b.to(tl.int64) * ranked_stride
        node_row = b.to(tl.int64) * node_stride
        j = tl.arange(0, MAX_TREE_NODES)
        nodes = tl.load(ranked_row + j, mask=j < verify_num, other=-1)
        i = tl.arange(0, BLOCK_S)
        ranked = i < verify_num
        node = tl.load(ranked_row + i, mask=ranked, other=0)
        prev = tl.load(pool_prev_ptr + node_row + node, mask=ranked, other=0)
        match = (nodes[None, :] == prev[:, None]) & (j[None, :] < i[:, None])
        parent = tl.max(tl.where(match, j[None, :], -1), axis=1)
        parent = tl.where(ranked, parent, -1)
        token = tl.load(pool_tokens_ptr + node_row + node)
        row = tl.load(rows_ptr + b)
        slot_ok = i < NUM_SLOTS
        tl.store(
            input_map_ptr + row * input_stride + i, token.to(tl.int32), mask=slot_ok
        )
        tl.store(
            parent_map_ptr + row * parent_stride + i, parent.to(tl.int32), mask=slot_ok
        )

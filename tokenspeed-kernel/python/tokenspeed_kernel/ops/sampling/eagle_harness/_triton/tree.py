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

"""Triton kernels of the EAGLE-Harness draft tree (the per-request node pool:
the root, node 0, then the children in expansion order). Each keeps the
floating-point operation order of the CUDA build, so both give the same bits;
the launchers live in ``ops/sampling/eagle_harness/triton.py``.

Kernels that rewrite a buffer they also read (the frontier scores, the sort
list, the lane masks) load everything first and store after a CTA barrier.
"""

from __future__ import annotations

from tokenspeed_kernel._triton import libdevice, tl, triton

# Draft depth d >= 1's children enter the pool at path score * DEPTH_DECAY^(d + 1).
DEPTH_DECAY = tl.constexpr(0.9999)
# The depth update's sort network size (width * K children at most) and sort list head.
MAX_CHILDREN = 64
MAX_VERIFY = 256
MAX_MASK_WORDS = 4
# Verify tree nodes per request.
MAX_TREE_NODES = tl.constexpr(64)


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
    """work_deltas[j]: the batch mean of each row's best min(node_count + j,
    verify_num) merged scores; one row per lane, then a halving tree over
    THREADS lanes (the padded ones zero)."""
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
    """One request's depth: sort its children, the best K become the
    frontier, all join the pool; the sort list merges the children by stable
    rank."""
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
    """Whether (v, i) precedes (w, j) in torch.argmax's order: NaN first, then
    the larger value, then the lower index."""
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
    """Each live row's best k of n scores in torch.argmax's order: every
    score's rank is the count of scores before it, a strict total order, so
    the result is the CUDA build's sort network's."""
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
    """Verify tree into the next round's inputs: slot i < verify_num is ranked
    node i (its parent the latest earlier slot holding its pool parent), later
    slots the root."""
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

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

"""Triton kernels of the EAGLE-Harness sampled trees.

A sampled request draws each expanded row's ``K`` children from the draft
distribution ``q`` over the row's ``NC`` candidate ids (Gumbel-top-K, after
the request's top-k / top-p cut) and records ``q`` and the ids in a slot of
its q store, each child's (slot, position) in the node tables. The verify
accepts a node's children by recursive rejection against the target
probabilities, rewriting a rejected node's row into its residual, and the
bonus token is drawn from the last accepted node's (unnormalized) row.

The children's Gumbel noise is keyed like the CUDA build's (seed, round
counter, launch, expanded row, candidate position) but drawn from Triton's
Philox stream, so the two builds draw different (equally distributed)
children.
"""

from __future__ import annotations

from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.sampling.eagle_harness._triton.policy import _take_f32
from tokenspeed_kernel.ops.sampling.eagle_harness._triton.tree import _argmax_before

# Verify tree nodes per request, and the row and candidate counts the noise key packs into 10 bits each.
MAX_TREE_NODES = 64
MAX_KEYED = 1024

_FLT_MAX = tl.constexpr(3.4028234663852886e38)


@triton.jit
def _argmax_fold(v, i, mask, best_v, best_i):
    """``(best_v, best_i)`` folded with every masked ``(v, i)``: NaN first,
    then the larger value, then the lower index."""
    nan = mask & (v != v)
    any_nan = tl.max(nan.to(tl.int32)) > 0
    big = 2147483647
    nan_i = tl.min(tl.where(nan, i, big))
    num = mask & (v == v)
    top = tl.max(tl.where(num, v, -float("inf")))
    top_i = tl.min(tl.where(num & (v == top), i, big))
    cand_v = tl.where(any_nan, float("nan"), top)
    cand_i = tl.where(any_nan, nan_i, top_i)
    take = (cand_i != big) & _argmax_before(cand_v, cand_i, best_v, best_i)
    return tl.where(take, cand_v, best_v), tl.where(take, cand_i, best_i)


@triton.jit(
    do_not_specialize=[
        "logits_s0",
        "logits_s1",
        "cand_s0",
        "cand_s1",
        "tokens_s0",
        "tokens_s1",
        "scores_s0",
        "scores_s1",
        "rows_per_req",
        "slot_base",
        "num_slots",
        "max_nodes",
        "nc",
        "vocab_limit",
        "seed",
        "launch",
    ]
)
def _sample_children_kernel(
    logits_ptr,  # fp32 [req, rows, nc], unit last stride
    logits_s0,
    logits_s1,
    cand_ptr,  # int64 [req, rows, nc]
    cand_s0,
    cand_s1,
    tokens_ptr,  # int64 [req, rows, K]
    tokens_s0,
    tokens_s1,
    scores_ptr,  # fp32 [req, rows, K]
    scores_s0,
    scores_s1,
    temps_ptr,
    top_ks_ptr,
    top_ps_ptr,
    q_store_ptr,  # fp32 [req, num_slots, nc] contiguous
    qid_store_ptr,
    node_slot_ptr,  # int32 [req, max_nodes]
    node_pos_ptr,
    rows_per_req,
    width_ptr,
    slot_base,
    cumu_start_ptr,
    num_slots,
    max_nodes,
    nc,
    vocab_limit,
    seed,
    round_counter_ptr,
    launch,
    K: tl.constexpr,
    K_PAD: tl.constexpr,
    HAS_WIDTH: tl.constexpr,
    BLOCK_NC: tl.constexpr,
    CHUNK: tl.constexpr,
):
    """Draws the ``K`` children of request ``b``'s row ``w`` (one program
    each; rows at or past the request's width exit).

    A greedy request (temperature <= 0) only marks its children as drawn
    without q (node slot -1). Otherwise ``q = softmax(logits / T)`` over the
    candidates below ``vocab_limit``, cut to the entries with fewer than
    ``top_k`` candidates above them and less than ``top_p`` mass above them,
    renormalized, is stored in slot ``slot_base + w`` with the candidate ids;
    the children are the ``K`` best Gumbel keys ``log q + G`` (ties to the
    lower position), ``G`` keyed by (round counter, launch, expanded row,
    position). A child past the cut's support gets score 0.
    """
    pid = tl.program_id(0)
    b = pid // rows_per_req
    w = pid - b * rows_per_req
    if HAS_WIDTH:
        width = tl.load(width_ptr)
    else:
        width = rows_per_req
    if w < width:
        b64 = b.to(tl.int64)
        w64 = w.to(tl.int64)
        m = b64 * width + w64
        node_row = b64 * max_nodes + tl.load(cumu_start_ptr) + w64 * K
        kk = tl.arange(0, K_PAD)
        temp = tl.load(temps_ptr + b)
        if temp <= 0.0:
            tl.store(
                node_slot_ptr + node_row + kk,
                tl.full([K_PAD], -1, tl.int32),
                mask=kk < K,
            )
        else:
            j = tl.arange(0, BLOCK_NC)
            in_row = j < nc
            cand = tl.load(
                cand_ptr + b64 * cand_s0 + w64 * cand_s1 + j, mask=in_row, other=0
            ).to(tl.int64)
            valid = in_row & (cand < vocab_limit)
            x = tl.load(
                logits_ptr + b64 * logits_s0 + w64 * logits_s1 + j,
                mask=valid,
                other=0.0,
            )
            x = tl.where(valid, x * (1.0 / temp), -_FLT_MAX)
            x_max = tl.max(x)
            e = tl.where(in_row & (x > -_FLT_MAX), tl.exp(x - x_max), 0.0)
            total = tl.sum(e)
            top_k = tl.load(top_ks_ptr + b)
            top_p = tl.load(top_ps_ptr + b)
            store_row = (b64 * num_slots + slot_base + w) * nc
            if (top_k < nc) | (top_p < 1.0):
                # Rank each entry against the whole row, CHUNK entries at a time read back from the store.
                q = e / total
                tl.store(q_store_ptr + store_row + j, q, mask=in_row)
                tl.debug_barrier()
                above = tl.zeros([BLOCK_NC], dtype=tl.int32)
                mass = tl.zeros([BLOCK_NC], dtype=tl.float32)
                for start in range(0, nc, CHUNK):
                    i = start + tl.arange(0, CHUNK)
                    q_i = tl.load(q_store_ptr + store_row + i, mask=i < nc, other=0.0)
                    higher = q_i[None, :] > q[:, None]
                    above += tl.sum(higher.to(tl.int32), axis=1)
                    mass += tl.sum(tl.where(higher, q_i[None, :], 0.0), axis=1)
                e = tl.where((above < top_k) & (mass < top_p), e, 0.0)
                total = tl.sum(e)
                tl.debug_barrier()
            q = e / total
            tl.store(q_store_ptr + store_row + j, q, mask=in_row)
            tl.store(qid_store_ptr + store_row + j, cand.to(tl.int32), mask=in_row)
            stream = tl.load(round_counter_ptr).to(tl.int64) * 64 + launch
            u = tl.rand(seed, (stream << 20) | (m << 10) | j.to(tl.int64))
            u = tl.minimum(tl.maximum(u, 1.0e-7), 1.0 - 2.0**-24)
            key = tl.where(
                q > 0.0,
                tl.log(q) - tl.log(-tl.log(u)),
                tl.where(cand < vocab_limit, -1.0e30, -2.0e30),
            )
            key = tl.where(in_row, key, -float("inf"))
            out_tok = tl.zeros([K_PAD], dtype=tl.int64)
            out_pos = tl.zeros([K_PAD], dtype=tl.int32)
            out_dead = tl.zeros([K_PAD], dtype=tl.int1)
            for i in tl.static_range(K):
                best_v = tl.max(key)
                best_j = tl.min(tl.where(in_row & (key == best_v), j, BLOCK_NC))
                key = tl.where(j == best_j, -float("inf"), key)
                token = tl.sum(tl.where(j == best_j, cand, 0))
                out_tok = tl.where(kk == i, token, out_tok)
                out_pos = tl.where(kk == i, best_j, out_pos)
                out_dead = tl.where(kk == i, best_v <= -1.0e29, out_dead)
            ok = kk < K
            tl.store(
                tokens_ptr + b64 * tokens_s0 + w64 * tokens_s1 + kk, out_tok, mask=ok
            )
            tl.store(
                scores_ptr + b64 * scores_s0 + w64 * scores_s1 + kk,
                tl.zeros([K_PAD], dtype=tl.float32),
                mask=ok & out_dead,
            )
            slot = (slot_base + w).to(tl.int32)
            tl.store(
                node_slot_ptr + node_row + kk,
                tl.zeros([K_PAD], tl.int32) + slot,
                mask=ok,
            )
            tl.store(node_pos_ptr + node_row + kk, out_pos, mask=ok)


@triton.jit
def _next_child(parents, idx, n, node, after):
    """The first child of ``node`` past index ``after``, or -1."""
    c = tl.min(tl.where((idx < n) & (idx > after) & (parents == node), idx, n))
    return tl.where(c < n, c, -1)


@triton.jit(
    do_not_specialize=[
        "probas_stride",
        "coins_stride",
        "num_nodes",
        "num_slots",
        "nc",
        "vocab",
    ]
)
def _tree_reject_nodes_kernel(
    probas_ptr,  # fp32 [rows >= bs * n, vocab], unit column stride
    probas_stride,
    tokens_ptr,  # int32 [bs, n]
    parents_ptr,  # int32 [bs, n]
    q_rows_ptr,  # int64 [bs] (HAS_Q)
    q_slot_ptr,  # int32 [pool, n]
    q_pos_ptr,
    q_store_ptr,  # fp32 [pool, num_slots, nc] contiguous
    qid_store_ptr,
    coins_ptr,  # fp32 [bs, >= n], unit column stride
    coins_stride,
    top_ks_ptr,  # int32 [rows]
    argmax_ptr,  # int64 [rows]
    accepted_ptr,  # int32 [bs, n]: each node's accepted child or -1
    num_nodes,
    num_slots,
    nc,
    vocab,
    HAS_Q: tl.constexpr,
    N_PAD: tl.constexpr,
    BLOCK_NC: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Picks node ``node`` of request ``b``'s accepted child (one program
    each), rewriting the node's target row where it rejects.

    * sampled (the first child has a q slot): recursive rejection. Child
      ``x`` at q position ``pos`` is accepted when ``coin * q_i * z <
      r(x)``, ``q_i = q(x) / q_left`` its share of the draft mass left and
      ``z`` the residual mass; a rejection subtracts ``q * z / q_left`` from
      the row at every candidate not yet rejected and drops ``x`` from
      ``q_left``. A residual that vanishes accepts ``x`` when ``r(x) > 0``,
      else sets the row's first argmax (taken before the updates) to 1;
    * greedy (top-k 1): the first child holding the row's argmax;
    * otherwise: child ``x`` is accepted with probability ``p'(x)`` over the
      mass its rejected siblings left, which are zeroed.

    The candidates' target values stay in registers through the rejections
    and are written back once.
    """
    b = tl.program_id(0)
    node = tl.program_id(1)
    if node < num_nodes:
        b64 = b.to(tl.int64)
        idx = tl.arange(0, N_PAD)
        tree = b64 * num_nodes
        parents = tl.load(parents_ptr + tree + idx, mask=idx < num_nodes, other=-2)
        row = tree + node
        p_row = probas_ptr + row * probas_stride
        child = _next_child(parents, idx, num_nodes, node, node)
        accepted = tl.full((), -1, tl.int32)
        greedy = tl.load(top_ks_ptr + row) == 1
        if HAS_Q:
            q_row = tl.load(q_rows_ptr + b)
        else:
            q_row = tl.full((), -1, tl.int64)
        slot = tl.full((), -1, tl.int32)
        if (~greedy) & (q_row >= 0) & (child >= 0):
            slot = tl.load(q_slot_ptr + q_row * num_nodes + child)
        if slot >= 0:
            j = tl.arange(0, BLOCK_NC)
            store_row = (q_row * num_slots + slot) * nc
            q = tl.load(q_store_ptr + store_row + j, mask=j < nc, other=0.0)
            ids = tl.load(qid_store_ptr + store_row + j, mask=j < nc, other=0)
            live = (j < nc) & (q > 0.0)
            p = tl.load(p_row + ids, mask=live, other=0.0)
            q_left = tl.sum(tl.where(live, q, 0.0))
            z_out = tl.maximum(1.0 - tl.sum(p), 0.0)
            in_vocab = (ids >= 0) & (ids < vocab)
            best_v, best_i = _argmax_fold(
                p, ids, live & in_vocab, float("-inf"), 2147483647
            )
            z = tl.full((), 1.0, tl.float32)
            rejected = j < 0
            vanished = child < 0
            while child >= 0:
                pos = tl.load(q_pos_ptr + q_row * num_nodes + child)
                token = tl.load(tokens_ptr + tree + child)
                at = j == pos
                q_child = _take_f32(q, j, pos)
                nxt = _next_child(parents, idx, num_nodes, node, child)
                if (q_child > 0.0) & (q_left > 0.0) & (z > 0.0):
                    q_i = q_child / q_left
                    # The child's token is the candidate at its q position (sample_children's draw).
                    r_child = _take_f32(p, j, pos)
                    if (token >= 0) & (token < vocab):
                        if _argmax_before(r_child, token, best_v, best_i):
                            best_v = r_child
                            best_i = token
                    coin = tl.load(coins_ptr + b64 * coins_stride + child)
                    if coin * q_i * z < r_child:
                        accepted = child
                        nxt = -1
                    else:
                        scale = z / q_left
                        others = live & ~at
                        keep = others & ~rejected
                        p = tl.where(keep, tl.maximum(p - scale * q, 0.0), p)
                        r_sum = tl.sum(tl.where(others, p, 0.0))
                        q_sum = tl.sum(tl.where(keep, q, 0.0))
                        r_y = tl.maximum(r_child - scale * q_child, 0.0)
                        p = tl.where(at, r_y, p)
                        z_new = z_out + r_sum + r_y
                        if not (z_new > 1.0e-6 * z):
                            if r_child > 0.0:
                                accepted = child
                            else:
                                vanished = True
                            nxt = -1
                        else:
                            z = z_new
                            q_left = q_sum
                            rejected = rejected | at
                child = nxt
            tl.store(p_row + ids, p, mask=live)
            if vanished:
                # Bonus = the target's argmax: untouched entries keep target values (rare, so scanned).
                tl.debug_barrier()
                for start in range(0, vocab, BLOCK_V):
                    v_idx = start + tl.arange(0, BLOCK_V)
                    v_ok = v_idx < vocab
                    v = tl.load(p_row + v_idx, mask=v_ok, other=0.0)
                    best_v, best_i = _argmax_fold(v, v_idx, v_ok, best_v, best_i)
                tl.debug_barrier()
                tl.store(p_row + best_i, 1.0)
        elif greedy:
            best = tl.load(argmax_ptr + row)
            tokens = tl.load(tokens_ptr + tree + idx, mask=idx < num_nodes, other=0)
            hit = (idx < num_nodes) & (idx > node) & (parents == node)
            c = tl.min(tl.where(hit & (tokens.to(tl.int64) == best), idx, num_nodes))
            accepted = tl.where(c < num_nodes, c, -1)
        else:
            total = tl.full((), 1.0, tl.float32)
            while child >= 0:
                nxt = _next_child(parents, idx, num_nodes, node, child)
                token = tl.load(tokens_ptr + tree + child)
                if (token >= 0) & (token < vocab):
                    p_child = tl.load(p_row + token)
                    coin = tl.load(coins_ptr + b64 * coins_stride + child)
                    if (p_child > 0.0) & (coin * total < p_child):
                        accepted = child
                        nxt = -1
                    else:
                        total = tl.maximum(total - p_child, 0.0)
                        tl.store(p_row + token, 0.0)
                        # A later sibling may hold the same token.
                        tl.debug_barrier()
                child = nxt
        tl.store(accepted_ptr + tree + node, accepted)


@triton.jit(do_not_specialize=["num_nodes"])
def _tree_reject_path_kernel(
    tokens_ptr,  # int32 [bs, n]
    path_ptr,  # int32 [bs, n]: the accepted children in, the path out
    predicts_ptr,  # int32 [bs * n]
    accept_length_ptr,
    last_row_ptr,
    num_nodes,
    N_PAD: tl.constexpr,
):
    """Walks request ``b``'s accepted children from the root: the path
    (node indices, -1 past it), the accepted tokens as predicts, the accept
    length and the last accepted node's row."""
    b = tl.program_id(0)
    tree = b.to(tl.int64) * num_nodes
    idx = tl.arange(0, N_PAD)
    accepted = tl.load(path_ptr + tree + idx, mask=idx < num_nodes, other=-1)
    path = tl.where(idx == 0, 0, tl.full([N_PAD], -1, tl.int32))
    length = tl.full((), 1, tl.int32)
    node = tl.full((), 0, tl.int32)
    c = tl.sum(tl.where(idx == 0, accepted, 0))
    while c >= 0:
        tl.store(predicts_ptr + tree + length - 1, tl.load(tokens_ptr + tree + c))
        path = tl.where(idx == length, c, path)
        length += 1
        node = c
        c = tl.sum(tl.where(idx == node, accepted, 0))
    tl.debug_barrier()
    tl.store(path_ptr + tree + idx, path, mask=idx < num_nodes)
    tl.store(accept_length_ptr + b, length)
    tl.store(last_row_ptr + b, tree + node)


@triton.jit(do_not_specialize=["probas_stride", "num_nodes", "vocab"])
def _residual_sample_kernel(
    probas_ptr,
    probas_stride,
    last_row_ptr,
    coins_ptr,  # fp32 [bs]
    top_ks_ptr,
    argmax_ptr,
    accept_length_ptr,
    predicts_ptr,
    num_nodes,
    vocab,
    BLOCK_V: tl.constexpr,
):
    """Request ``b``'s bonus token, at its last predicts entry: the argmax of
    a greedy row, else the inverse-CDF draw of ``coin * sum(row)`` from the
    last accepted row (unnormalized): the first positive entry whose
    cumulative mass passes it, or the last positive entry when rounding
    leaves the target past every entry."""
    b = tl.program_id(0)
    row = tl.load(last_row_ptr + b)
    out = predicts_ptr + b.to(tl.int64) * num_nodes + tl.load(accept_length_ptr + b) - 1
    if tl.load(top_ks_ptr + row) == 1:
        tl.store(out, tl.load(argmax_ptr + row).to(tl.int32))
    else:
        p_row = probas_ptr + row * probas_stride
        total = tl.zeros((), tl.float32)
        last = tl.full((), -1, tl.int32)
        for start in range(0, vocab, BLOCK_V):
            v_idx = start + tl.arange(0, BLOCK_V)
            v = tl.load(p_row + v_idx, mask=v_idx < vocab, other=0.0)
            total += tl.sum(v)
            last = tl.maximum(last, tl.max(tl.where(v > 0.0, v_idx, -1)))
        target = tl.load(coins_ptr + b) * total
        token = tl.full((), -1, tl.int32)
        acc = tl.zeros((), tl.float32)
        start = tl.full((), 0, tl.int32)
        while (token < 0) & (start < vocab):
            v_idx = start + tl.arange(0, BLOCK_V)
            v = tl.load(p_row + v_idx, mask=v_idx < vocab, other=0.0)
            cum = acc + tl.cumsum(v, 0)
            hit = (cum > target) & (v > 0.0)
            first = tl.min(tl.where(hit, v_idx, vocab))
            token = tl.where(first < vocab, first, -1)
            acc += tl.sum(v)
            start += BLOCK_V
        token = tl.where(token >= 0, token, tl.maximum(last, 0))
        tl.store(out, token)


@triton.jit(
    do_not_specialize=["ranked_stride", "node_stride", "num_nodes", "store_size"]
)
def _publish_draft_q_kernel(
    ranked_ptr,  # int64 [bs, *]
    ranked_stride,
    node_slot_ptr,  # int32 [bs, max_nodes]
    node_pos_ptr,
    node_stride,
    q_round_ptr,  # fp32 [bs, store_size] contiguous
    qid_round_ptr,
    rows_ptr,  # int64 [bs]
    q_tab_ptr,  # fp32 [pool, store_size]
    qid_tab_ptr,
    slot_tab_ptr,  # int32 [pool, num_nodes]
    pos_tab_ptr,
    live_rows_ptr,
    verify_num_ptr,
    num_nodes,
    store_size,
    N_PAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Request ``b``'s round q, candidate ids and its ranked nodes' (slot,
    position) into the next round's row ``rows[b]``; the root and nodes past
    the verify width get slot -1."""
    b = tl.program_id(0)
    if b < tl.load(live_rows_ptr):
        b64 = b.to(tl.int64)
        verify_num = tl.load(verify_num_ptr)
        row = tl.load(rows_ptr + b)
        i = tl.arange(0, N_PAD)
        ranked = (i > 0) & (i < verify_num) & (i < num_nodes)
        node = tl.load(ranked_ptr + b64 * ranked_stride + i, mask=ranked, other=0)
        slot = tl.load(node_slot_ptr + b64 * node_stride + node, mask=ranked, other=-1)
        pos = tl.load(node_pos_ptr + b64 * node_stride + node, mask=ranked, other=0)
        tl.store(slot_tab_ptr + row * num_nodes + i, slot, mask=i < num_nodes)
        tl.store(pos_tab_ptr + row * num_nodes + i, pos, mask=i < num_nodes)
        for start in range(0, store_size, BLOCK):
            k = start + tl.arange(0, BLOCK)
            ok = k < store_size
            src = b64 * store_size + k
            dst = row * store_size + k
            tl.store(q_tab_ptr + dst, tl.load(q_round_ptr + src, mask=ok), mask=ok)
            tl.store(qid_tab_ptr + dst, tl.load(qid_round_ptr + src, mask=ok), mask=ok)

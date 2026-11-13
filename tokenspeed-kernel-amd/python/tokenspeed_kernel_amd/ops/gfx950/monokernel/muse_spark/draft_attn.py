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

"""The attention half of a DFlash2 draft layer of Muse Spark, one TP8 rank.

The half starts from the previous layer's MLP partials (the layer's input
norm sums them, as the MLP half sums the attention partials) and ends in this
rank's attention partial, the MLP half's input. Its first and last stages are
the MLP half's (``draft_mlp``) with this half's weights: iscat, isum1, inorm
(the input RMSNorm), acoef, aconv0 (the attention convolution), qkv, then the
stages below, oproj, and afin (the convolution after the output projection).

A rank holds 8 query heads and one KV head of 128 (64 / 8 heads). The block
of 8 rows a request is non-causal and every row shares one window, as in the
regular draft (``attention/backends/paged/mha.py``, block decode): the keys
``[seq_len - 2048, seq_len)``, the block's 8 and the 2040 before them (the
layers' ``window_left`` of 2047 plus the row's own key).

* head, 10 tasks per row (8 q heads, then k, then v): the qkv projection's K
  parts summed and rounded once to BF16 (the projection's output), then for q
  and k the per-head RMSNorm and NeoX RoPE in FP32, rounded once, as the
  attention prologue does (``docs/design/attention-prologue.md``). q goes to
  ``qh``; k and v go into the draft's KV cache at the row's slot and into
  ``kb`` / ``vb``: attention tasks of this launch read the block's own keys
  from the mailbox, since a CTA never reads another CTA's store of the same
  launch.
* attn, ``SPLITS`` tasks a request (a launch serves the blocks of one or
  two requests, each with its own length and page table): one task per
  split of the request's cached keys
  ``[max(seq_len - 2048, 0), seq_len - 8)``, ``ceil(n / SPLITS)`` each, one
  chunk of at most 128 (16 splits hold the 2040) with a base-2 softmax;
  every row's 8 heads are one MFMA row each (64 rows). The last split also
  takes the block's 8 keys from ``kb`` / ``vb``, as one more chunk, for
  every row (non-causal). A task publishes FP32 ``acc / l`` and the LSE
  (``-inf`` for an empty split) as plain words, then one flag
  (``Mailbox.store_words`` / ``raise_flag``): at 8 rows a task writes
  64 x 128 values, and tagged pairs would double the stores.
* merge, 8 tasks per row, one per q head: after the splits' flags, the LSE
  merge of the splits, BF16, published in ``ao`` for the output projection.

The softmax step and the cache loads follow the target layer's
``attention.py`` at the draft's head of 128 and single KV head.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.monokernel import mx, schedule, trace
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import draft_mlp
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.attention import (
    softmax_step,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.dims import (
    DRAFT_BLOCK,
    DRAFT_HEAD_DIM,
    DRAFT_Q_HEADS,
    HIDDEN,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import Stage, placement

HEAD = DRAFT_HEAD_DIM
HALF = gl.constexpr(HEAD // 2)  # rotary pairs of a head (NeoX: i and i + 64)
Q_HEADS = DRAFT_Q_HEADS
Q_ROWS = gl.constexpr(Q_HEADS * HEAD)  # q rows a rank: the attention width
QKV = gl.constexpr(Q_ROWS + 2 * HEAD)  # q, k (128), v (128) rows a rank
HEAD_TASKS = gl.constexpr(Q_HEADS + 2)  # tasks per row: 8 q heads, k, v
EPS = draft_mlp.EPS
PAGE = gl.constexpr(64)  # tokens a KV page
BN = gl.constexpr(128)  # keys a chunk
QM = gl.constexpr(64)  # MFMA rows of attention: 8 block rows x 8 heads
KT = gl.constexpr(16)  # MFMA rows of the block's keys (8, the rest zero)
WINDOW_KEYS = gl.constexpr(2048)  # every row's keys: the window and its own
BLOCK_ROWS = DRAFT_BLOCK
# exp2 softmax of (q . k) / sqrt(128).
QK_SCALE = gl.constexpr(0.08838834764831845 * 1.4426950408889634)


# The projections' tasks: QKV_ROWS (OPROJ_ROWS) output rows x a K part of
# QKV_KS (OPROJ_KS) values.
QKV_ROWS = gl.constexpr(128)
QKV_KS = gl.constexpr(512)
QKV_KPARTS = gl.constexpr(HIDDEN // QKV_KS)  # 16
QKV_TASKS = gl.constexpr(QKV // QKV_ROWS * QKV_KPARTS)  # 10 groups x 16 K parts
OPROJ_ROWS = gl.constexpr(128)
OPROJ_KS = gl.constexpr(512)
OPROJ_KPARTS = gl.constexpr(Q_ROWS // OPROJ_KS)  # o_proj's K (1024): 2 parts
OPROJ_TASKS = gl.constexpr(HIDDEN // OPROJ_ROWS * OPROJ_KPARTS)  # 64 x 2
AFIN_TASKS = gl.constexpr(256)
AFIN_COLS = gl.constexpr(HIDDEN // AFIN_TASKS)  # 32


def regions(splits: int) -> tuple[tuple[str, int], ...]:
    """The attention half's regions. Those of the stages it shares with the
    MLP half (draft_mlp) carry an ``a`` prefix, so one plan can hold both."""
    head, q = int(HEAD), int(Q_HEADS)
    world, hidden = int(draft_mlp.WORLD), int(draft_mlp.HIDDEN)
    owned, sums = int(draft_mlp.OWNED), int(draft_mlp.SUM_TASKS)
    return (
        ("ared", world * owned // 2),  # BF16 pairs [source rank][owned column]
        ("ass", world * sums),  # FP32 sums of squares [rank][task]
        ("ahin", hidden // 2),  # BF16 pairs of the whole attention input
        ("acpart", int(draft_mlp.COEF_KPARTS) * int(draft_mlp.COEFS)),  # FP32
        ("ahc", hidden // 2),  # BF16 pairs of the convolved attention input
        ("qpart", int(QKV_KPARTS) * int(QKV)),  # FP32 [K part][q, k, v row]
        ("qh", q * head // 2),  # BF16 pairs of the rotated query heads
        ("kb", head // 2),  # BF16 pairs of the block row's rotated key
        ("vb", head // 2),  # BF16 pairs of the block row's value
        ("apart", splits * q * head),  # FP32 words [split][head][dim] acc / l
        ("alse", splits * q),  # FP32 words [split][head] LSE
        ("aflag", splits),  # a flag per attn task, in row 0
        ("ao", q * head // 2),  # BF16 pairs of the attention output
        ("opart", int(OPROJ_KPARTS) * hidden),  # FP32 [K part][column]
    )


PEERS = ("ared", "ass", "ahin")


def stages(bases: dict[str, int], splits: int, rows: int) -> tuple[Stage, ...]:
    """The attention half of ``rows`` token rows (one or two requests'
    blocks), placed by ``bases``."""
    names = (
        "iscat",
        "isum",
        "acoef",
        "aconv0",
        "qkv",
        "head",
        "attn",
        "merge",
        "oproj",
        "afin",
    )
    iscat, isum, acoef, aconv0, qkv, head, attn, merge, oproj, afin = placement(
        bases, names
    )
    tasks = int(draft_mlp.SUM_TASKS) * rows
    coefs = (
        int(draft_mlp.COEF_KPARTS) * int(draft_mlp.COEFS) // int(draft_mlp.COEF_ROWS)
    )
    return (
        Stage("iscat", int(draft_mlp.SCAT_TASKS), iscat, writes=("ared",)),
        Stage("isum1", tasks, isum, reads=("ared",), writes=("ass",)),
        Stage("inorm", tasks, isum, reads=("ass",), writes=("ahin",)),
        Stage("acoef", coefs, acoef, reads=("ahin",), writes=("acpart",)),
        Stage(
            "aconv0",
            int(draft_mlp.CONV_TASKS),
            aconv0,
            reads=("ahin", "acpart"),
            writes=("ahc",),
        ),
        Stage("qkv", int(QKV_TASKS), qkv, reads=("ahc",), writes=("qpart",)),
        Stage(
            "head",
            int(HEAD_TASKS) * rows,
            head,
            reads=("qpart",),
            writes=("qh", "kb", "vb"),
        ),
        Stage(
            "attn",
            splits * max(rows // int(BLOCK_ROWS), 1),
            attn,
            reads=("qh", "kb", "vb"),
            writes=("apart", "alse", "aflag"),
        ),
        Stage(
            "merge",
            int(Q_HEADS) * rows,
            merge,
            reads=("apart", "alse", "aflag"),
            writes=("ao",),
        ),
        Stage("oproj", int(OPROJ_TASKS), oproj, reads=("ao",), writes=("opart",)),
        Stage("afin", int(AFIN_TASKS), afin, reads=("opart", "acpart")),
    )


@gluon.jit
def head_stage(
    mb,
    task,
    q_norm,
    k_norm,
    cos_sin,
    positions,
    slots,
    k_cache,
    v_cache,
    STAGE: gl.constexpr,
):
    """Head task ``task``: head ``task % 10`` (q 0..7, k, v) of row
    ``task // 10``.

    Args:
        q_norm, k_norm: BF16 ``[128]`` per-head RMSNorm weights.
        cos_sin: FP32 ``[positions, 128]`` RoPE table, cos then sin.
        positions: int32 ``[rows]`` the rows' positions.
        slots: int32 or int64 ``[rows]`` the rows' cache slots.
        k_cache, v_cache: BF16 ``[slots, 1, 128]`` this layer's cache.
    """
    mb.stamp(STAGE, trace.START)
    row, head, _ = schedule.row_task(mb.plan, task, HEAD_TASKS)
    mb = mb.row(row)
    nw: gl.constexpr = gl.num_warps()
    # [K part, pair]: the K parts across lanes, a pair (i, i + 64) a lane.
    kl: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [nw, 1], [1, 0])
    k = gl.arange(0, QKV_KPARTS, gl.SliceLayout(1, kl))[:, None]
    i = gl.arange(0, HALF, gl.SliceLayout(0, kl))
    is_q = head < Q_HEADS
    is_k = head == Q_HEADS
    rotated = head <= Q_HEADS
    # The parameters load before the poll: a load after a poll would start
    # only when the poll returns.
    w1 = gl.where(is_q, gl.load(q_norm + i), gl.load(k_norm + i)).to(gl.float32)
    w2 = gl.where(is_q, gl.load(q_norm + i + HALF), gl.load(k_norm + i + HALF))
    w2 = w2.to(gl.float32)
    position = gl.load(positions + row)
    cos = gl.load(cos_sin + position.to(gl.int64) * HEAD + i)
    sin = gl.load(cos_sin + position.to(gl.int64) * HEAD + HALF + i)
    slot = gl.load(slots + row).to(gl.int64)
    at = k * QKV + head * HEAD + i[None, :]
    lo_w, hi_w = mb.poll_many(("qpart", "qpart"), (at, at + HALF))
    mb.stamp(STAGE, trace.READY)
    vals = head_values(lo_w, hi_w, w1, w2, cos, sin, rotated)
    pl: gl.constexpr = vals.type.layout
    pair = gl.reshape(vals, [HALF, 2])
    word = mx.pack_bf16_pairs(pair)
    p = gl.arange(0, HALF, word.type.layout)
    if is_q:
        mb.put("qh", head * (HEAD // 2) + p, word)
    elif is_k:
        mb.put("kb", p, word)
        cols = gl.arange(0, HEAD, pl)
        gl.store(k_cache + slot * HEAD + cols, vals)
    else:
        mb.put("vb", p, word)
        cols = gl.arange(0, HEAD, pl)
        gl.store(v_cache + slot * HEAD + cols, vals)
    mb.stamp(STAGE, trace.END)


@gluon.jit
def head_values(lo_w, hi_w, w1, w2, cos, sin, rotated):
    """A head's K-part words ``[K part, 64]`` (lo_w: dims i, hi_w: i + 64) as
    its BF16 values ``[128]`` in head order, two a lane: the K parts summed
    and rounded once to BF16 (the projection's output), then if ``rotated``
    the per-head RMSNorm (gains ``w1``, ``w2``) and NeoX RoPE in FP32,
    rounded once, as the attention prologue does. The draft context write
    (``draft_context``) forms its K and V the same way."""
    x1 = gl.sum(lo_w.to(gl.float32, bitcast=True), axis=0)
    x2 = gl.sum(hi_w.to(gl.float32, bitcast=True), axis=0)
    x1 = x1.to(gl.bfloat16).to(gl.float32)
    x2 = x2.to(gl.bfloat16).to(gl.float32)
    if rotated:
        inv = gl.rsqrt(gl.sum(x1 * x1 + x2 * x2, axis=0) / HEAD + EPS)
        n1 = x1 * inv * w1
        n2 = x2 * inv * w2
        # The association of the attention prologue's kernels.
        o1 = gl.fma(n1, cos, -n2 * sin)
        o2 = gl.fma(n2, cos, n1 * sin)
    else:
        o1 = x1
        o2 = x2
    b1 = o1.to(gl.bfloat16)
    b2 = o2.to(gl.bfloat16)
    # Pairs (j, j + 1) of the head: lanes hold i and i + 64, so the even and
    # odd halves of a pair sit on two lanes; regroup through a layout change.
    full = gl.join(b1, b2)  # [64, 2]: (i, i + 64)
    vals = gl.reshape(gl.permute(full, [1, 0]), [HEAD])  # head order
    pl: gl.constexpr = gl.BlockedLayout([2], [64], [gl.num_warps()], [0])
    return gl.convert_layout(vals, pl)


@gluon.constexpr_function
def _layouts(nw):
    """(qk MFMA, pv MFMA, K operand, V load) layouts, as the target layer's."""
    qk_mma = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, nw]
    )
    pv_mma = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 16], transposed=True, warps_per_cta=[1, nw]
    )
    k_layout = gl.DotOperandLayout(1, qk_mma, 8)
    v_load = gl.BlockedLayout([1, 8], [8, 8], [nw, 1], [1, 0])
    return qk_mma, pv_mma, k_layout, v_load


@gluon.jit
def _load_kv(
    table,
    k_cache,
    v_cache,
    c0,
    hi,
    dk,
    dv,
    k_layout: gl.constexpr,
    v_load: gl.constexpr,
):
    """Keys ``[128, BN]`` and values ``[BN, 128]`` of chunk ``c0 ..`` from the
    cache (zero from ``hi``)."""
    jk = c0 + gl.arange(0, BN, gl.SliceLayout(0, k_layout))
    jv = c0 + gl.arange(0, BN, gl.SliceLayout(1, v_load))
    old_k = jk < hi
    old_v = jv < hi
    page_k = gl.load(table + jk // PAGE, mask=old_k, other=0)
    page_v = gl.load(table + jv // PAGE, mask=old_v, other=0)
    slot_k = page_k.to(gl.int64) * PAGE + jk % PAGE
    slot_v = page_v.to(gl.int64) * PAGE + jv % PAGE
    keys = gl.load(
        k_cache + slot_k[None, :] * HEAD + dk[:, None], mask=old_k[None, :], other=0.0
    )
    values = gl.load(
        v_cache + slot_v[:, None] * HEAD + dv[None, :], mask=old_v[:, None], other=0.0
    )
    return keys, values


@gluon.jit
def _bf16_rows(got):
    """``[N, 128]`` FP32 of polled BF16 pairs ``[N, 64]``."""
    lo = mx.bf16_half(got, 0)
    hi = mx.bf16_half(got, 1)
    return gl.reshape(gl.join(lo, hi), [got.shape[0], HEAD])


@gluon.jit
def attn_stage(
    mb,
    task,
    table,
    table_stride,
    seq_len,
    k_cache,
    v_cache,
    SPLITS: gl.constexpr,
    STAGE: gl.constexpr,
):
    """Attn task ``task``: split ``task % SPLITS`` of the cached keys of
    request ``task // SPLITS`` (block rows ``8 q ..``), for its rows.

    Args:
        table: int32 page tables, request ``q``'s at ``table + q
            table_stride``.
        seq_len: int32 ``[requests]``: each request's length after its
            block.
        k_cache, v_cache: BF16 ``[slots, 1, 128]`` this layer's cache.
    """
    R: gl.constexpr = min(schedule.rows(mb.plan), BLOCK_ROWS)
    gl.static_assert(R * Q_HEADS <= QM)
    # Two requests: request task // SPLITS's split task % SPLITS. A launch of
    # one request computes no request index.
    if schedule.rows(mb.plan) > BLOCK_ROWS:
        req = task // SPLITS
        task = task % SPLITS
        mb = mb.row(req * BLOCK_ROWS)
        table = table + req * table_stride
        seq_len = seq_len + req
    # Every split's cached keys fit one chunk: ceil(2040 / SPLITS) <= BN.
    gl.static_assert((WINDOW_KEYS - BLOCK_ROWS + SPLITS - 1) // SPLITS <= BN)
    mb.stamp(STAGE, trace.START)
    s = task
    nw: gl.constexpr = gl.num_warps()
    qk_mma: gl.constexpr = _layouts(nw)[0]
    pv_mma: gl.constexpr = _layouts(nw)[1]
    k_layout: gl.constexpr = _layouts(nw)[2]
    v_load: gl.constexpr = _layouts(nw)[3]
    q_layout: gl.constexpr = gl.DotOperandLayout(0, qk_mma, 8)
    p_layout: gl.constexpr = gl.DotOperandLayout(0, pv_mma, 4)
    v_layout: gl.constexpr = gl.DotOperandLayout(1, pv_mma, 4)

    end = gl.load(seq_len)
    prefix = end - R
    first = gl.maximum(end - WINDOW_KEYS, 0)
    per = (prefix - first + SPLITS - 1) // SPLITS
    lo = first + s * per
    hi = gl.minimum(lo + per, prefix)

    # The first chunk's cache loads go before the polls: they do not depend
    # on this launch.
    dk = gl.arange(0, HEAD, gl.SliceLayout(1, k_layout))
    dv = gl.arange(0, HEAD, gl.SliceLayout(0, v_load))
    keys, values = _load_kv(table, k_cache, v_cache, lo, hi, dk, dv, k_layout, v_load)
    # The values go to LDS before the poll too: the poll's loads return only
    # after the cache loads (loads complete in order), so the store waits for
    # nothing the poll would not, and leaves the path after the poll.
    plain: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    v_smem = gl.allocate_shared_memory(gl.bfloat16, [BN, HEAD], plain)
    v_smem.store(values)
    js = lo + gl.arange(0, BN, gl.SliceLayout(0, qk_mma))
    valid = (js[None, :] < hi) & (gl.arange(0, QM, gl.SliceLayout(1, qk_mma)) >= 0)[
        :, None
    ]

    # q of the 8 heads of every row, and the block's keys and values.
    pl: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [nw, 1], [1, 0])
    qr = gl.arange(0, QM, gl.SliceLayout(1, pl))[:, None]
    kr = gl.arange(0, KT, gl.SliceLayout(1, pl))[:, None]
    pp = gl.arange(0, HEAD // 2, gl.SliceLayout(0, pl))[None, :]
    q_row = gl.minimum(qr // Q_HEADS, R - 1)
    k_row = gl.minimum(kr, R - 1)
    got_q, got_k, got_v = mb.poll_many(
        ("qh", "kb", "vb"),
        (
            mb.row_offset(q_row) + (qr % Q_HEADS) * (HEAD // 2) + pp,
            mb.row_offset(k_row) + pp,
            mb.row_offset(k_row) + pp,
        ),
    )
    mb.stamp(STAGE, trace.READY)
    q = _bf16_rows(got_q)
    q = gl.where(
        gl.arange(0, QM, gl.SliceLayout(1, q.type.layout))[:, None] < R * Q_HEADS,
        q,
        0.0,
    )
    q = gl.convert_layout(q.to(gl.bfloat16), q_layout)
    cur_k = gl.allocate_shared_memory(gl.bfloat16, [KT, HEAD], plain)
    cur_v = gl.allocate_shared_memory(gl.bfloat16, [KT, HEAD], plain)
    cur_k.store(_bf16_rows(got_k).to(gl.bfloat16))
    cur_v.store(_bf16_rows(got_v).to(gl.bfloat16))

    m = gl.full([QM], float("-inf"), gl.float32, gl.SliceLayout(1, qk_mma))
    l = gl.zeros([QM], gl.float32, gl.SliceLayout(1, qk_mma))
    acc = gl.zeros([QM, HEAD], gl.float32, pv_mma)
    # A split's cached keys fit one chunk (see the static assert): no loop,
    # and no second chunk's registers.
    # Every wave's LDS stores above (v_smem, cur_k, cur_v) land before any
    # wave loads them.
    gl.barrier()
    m, l, acc = softmax_step(
        q, keys, v_smem.load(v_layout), valid, m, l, acc, p_layout, QK_SCALE
    )
    # The block's keys, for every row (the block is non-causal). The other
    # splits mask every key, so for them the step leaves m, l and acc as
    # they were.
    # Every wave is done with the first step's LDS reads (v_smem and its
    # layout changes) before the block's step writes LDS again.
    gl.barrier()
    qi = gl.arange(0, QM, gl.SliceLayout(1, qk_mma))
    u = gl.arange(0, KT, gl.SliceLayout(0, qk_mma))
    own = (u[None, :] < R) & (qi >= 0)[:, None] & (s == SPLITS - 1)
    m, l, acc = softmax_step(
        q,
        cur_k.permute([1, 0]).load(gl.DotOperandLayout(1, qk_mma, 8)),
        cur_v.load(v_layout),
        own,
        m,
        l,
        acc,
        p_layout,
        QK_SCALE,
    )
    mb.stamp(STAGE, trace.COMPUTED)
    l_acc = gl.convert_layout(l, gl.SliceLayout(1, pv_mma))
    out = acc / gl.where(l_acc > 0, l_acc, 1.0)[:, None]
    lse = gl.where(l > 0, m + gl.log2(gl.where(l > 0, l, 1.0)), float("-inf"))
    r = gl.arange(0, QM, gl.SliceLayout(1, pv_mma))
    c = gl.arange(0, HEAD, gl.SliceLayout(0, pv_mma))
    part = (s * Q_HEADS + r[:, None] % Q_HEADS) * HEAD + c[None, :]
    # Plain words (row r's copy starts at word 2 row_offset(r)), then one
    # flag: tagged pairs would double the 64 x 128 values' stores.
    mb.store_words(
        "apart",
        2 * mb.row_offset(r[:, None] // Q_HEADS) + part,
        out,
        mask=(r < R * Q_HEADS)[:, None] & (c >= 0)[None, :],
    )
    mb.store_words(
        "alse",
        2 * mb.row_offset(qi // Q_HEADS) + s * Q_HEADS + qi % Q_HEADS,
        lse,
        mask=qi < R * Q_HEADS,
    )
    mb.raise_flag("aflag", s)
    mb.stamp(STAGE, trace.END)


@gluon.jit
def merge_stage(mb, task, SPLITS: gl.constexpr, STAGE: gl.constexpr):
    """Merge task ``task`` = q head ``h = task % 8`` of row ``task // 8``: the
    LSE merge of the splits, rounded to BF16, into ``ao``."""
    mb.stamp(STAGE, trace.START)
    row, h, _ = schedule.row_task(mb.plan, task, Q_HEADS)
    # The splits' flags sit in the request's row 0.
    if schedule.rows(mb.plan) > BLOCK_ROWS:
        flag0 = -(row % BLOCK_ROWS) * schedule.row_pairs(mb.plan)
    else:
        flag0 = -row * schedule.row_pairs(mb.plan)
    mb = mb.row(row)
    nw: gl.constexpr = gl.num_warps()
    ml: gl.constexpr = gl.BlockedLayout([1, 1], [2, 32], [nw, 1], [1, 0])
    s = gl.arange(0, SPLITS, gl.SliceLayout(1, ml))
    c = gl.arange(0, HEAD, gl.SliceLayout(0, ml))
    mb.poll_flags("aflag", flag0 + s)
    # Each wave polls some of the splits' flags: the words load after every
    # wave's poll returned, so every split's flag is in.
    gl.barrier()
    got_p = mb.load_words("apart", (s[:, None] * Q_HEADS + h) * HEAD + c[None, :])
    got_l = mb.load_words("alse", s * Q_HEADS + h)
    mb.stamp(STAGE, trace.READY)
    lse = got_l.to(gl.float32, bitcast=True)
    top = gl.max(lse, axis=0)
    w = gl.where(lse == float("-inf"), 0.0, gl.exp2(lse - top))
    p = got_p.to(gl.float32, bitcast=True)
    out = gl.sum(p * w[:, None], axis=0) / gl.sum(w, axis=0)
    mb.stamp(STAGE, trace.COMPUTED)
    mx.put_bf16(mb, "ao", h * (HEAD // 2), out)
    mb.stamp(STAGE, trace.END)

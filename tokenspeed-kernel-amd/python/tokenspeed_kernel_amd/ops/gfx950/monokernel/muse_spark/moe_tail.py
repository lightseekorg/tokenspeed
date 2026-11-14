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

"""Combine, post-expert projection and layer boundary of one Muse Spark rank.

Rank ``r`` owns pooled columns ``512 r ..`` of the expert outputs and hidden
columns ``1024 r ..`` of the residual.

* comb1, comb2, 8 tasks of 64 owned pooled columns per row (same CTAs): the
  rank-order
  sum of the 8 ranks' BF16 route partials, rounded to BF16, times the
  post-expert gain; then each route's RMS norm over all 4096 columns (one
  exchange of partial sums of squares per route), and the weighted sum of
  the 8 routes (a tree within a wave, not the regular path's pick order),
  rounded to BF16, put into every rank's ``cmb``.
* unpool, 64 tasks of 16 owned hidden columns (128 at 8 or 16 rows: each
  column block's rows in two groups, so a task reads half the combined rows
  for the same 128 KiB of weights): the post-expert projection (BF16
  ``[1024, 4096]``) of every whole combined row,
  rounded to BF16 (``p``), and ``y = p`` times the post-FFN gain. For each
  row, a task puts ``p`` (two BF16 values a word: the out tasks apply the
  gain, so ``y`` is the same FP32 product) and ``alpha h'``, and four partial
  sums, of ``y^2``, ``(beta y)^2``, ``(alpha h')(beta y)`` and ``(alpha
  h')^2``, into every rank's ``ffn``, in one store per rank. The puts of all
  64 tasks to 8 ranks are bound by the links between the GPUs, so ``p``
  goes in 3 pairs per 2 columns instead of 4.
* out, ``PER_ROW`` tasks a row (16, or 8 when the launch ends with qnext),
  each of ``8192 / PER_ROW`` hidden columns: the row's 512 x 4 partial
  sums from every rank's ``ffn``, summed into the inverse RMS of the post-FFN
  norm (over 8192 columns) and of the next layer's input norm, whose sum of
  squares is the expansion of ``(beta y inv + alpha h')^2`` in the four sums;
  then the residual gate ``h'' = beta y inv + alpha h'`` and the next layer's
  input norm. Each task sums its own row's sums (16 KB) itself, rather than
  reading them from a stage that sums each row once: such a stage would add
  one more hand-off at the end of the layer. The result is rounded to BF16
  and stored as the next attention input, for the next launch. Every rank
  computes the whole row from the ``ffn`` of all ranks: putting the owned
  slice of the result to the other ranks instead would add a second
  exchange to the end of the layer. The tasks of a row that cover the
  rank's owned columns (``PER_ROW / 8``) also store ``h''`` (FP32, the
  launch's residual output), not unpool's CTAs: a poll waits for every
  earlier store of its wave, so after their ``ffn`` puts unpool's wait for
  the scales would also wait for those stores. With a draft's tap, every
  task also stores its columns of ``bf16(h'')`` to its own rank's tap: each
  rank computes the whole row the same way, so no rank sends its owned
  columns to the others.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.monokernel import (
    bf16,
    mailbox,
    mx,
    schedule,
    trace,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.attn_tail import (
    OWNED,
    norm_scales,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.dims import (
    HIDDEN,
    POOLED,
    POST_EPS,
    WORLD,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.moe_experts import (
    OWNED_OUT,
    TOP_K,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import Stage, placement

COMB_TASKS = gl.constexpr(8)
COMB_COLS = gl.constexpr(64)
UNPOOL_TASKS = gl.constexpr(64)
UNPOOL_COLS = gl.constexpr(16)
SUMS = gl.constexpr(4)
OUT_TASKS = gl.constexpr(16)
AH_AT = gl.constexpr(int(HIDDEN) // 2)  # where ``alpha h'`` starts in ``ffn``
SUMS_AT = gl.constexpr(3 * int(HIDDEN) // 2)  # where the sums start in ``ffn``

PEERS = ("ss3", "cmb", "ffn")
# Past 8 rows (two requests) unpool publishes ``ffn`` as plain words and one
# flag per task in ``ffnf``: the link is the limit there (1.8 MB a launch
# out of each GPU as tagged pairs). Up to 8 rows it keeps tagged pairs: a
# flag waits for the remote stores' acknowledgments, on the tail's
# critical path.
WORDS_PEERS = ("ffnf",)


@gluon.constexpr_function
def ffn_words(rows: int) -> bool:
    """Whether unpool hands ``ffn`` to out as plain words and flags."""
    return int(rows) > 8


def rank_blocks(region: str, pairs: int) -> tuple[tuple[int, int], ...]:
    """``(start, pairs)`` of each block of peer region ``region`` (``pairs``
    pairs a row) that the ranks split in rank order: rank ``r`` writes the
    ``r``-th ``WORLD``-th of every block. ``ffn`` holds three such blocks
    (``p``, ``alpha h'``, the sums); the other regions are one block."""
    if region != "ffn":
        return ((0, pairs),)
    world, hidden = int(WORLD), int(HIDDEN)
    return (
        (0, hidden // 2),
        (int(AH_AT), hidden),
        (int(SUMS_AT), world * int(UNPOOL_TASKS) * int(SUMS)),
    )


def peers(rows: int) -> tuple[str, ...]:
    """The peer regions of ``rows`` token rows."""
    return PEERS + flags(rows)


def flags(rows: int) -> tuple[str, ...]:
    """The peer regions of ``rows`` token rows that hold flags (Plan.flags)."""
    return WORDS_PEERS if ffn_words(rows) else ()


def regions(rows: int) -> tuple[tuple[str, int], ...]:
    """The regions of ``rows`` token rows (past 8 rows also ``ffnf``)."""
    world, k = int(WORLD), int(TOP_K)
    flags = (("ffnf", world * int(UNPOOL_TASKS)),) if ffn_words(rows) else ()
    return flags + (
        ("ss3", world * int(COMB_TASKS) * k),  # FP32 [rank][task][route]
        ("cmb", int(POOLED) // 2),  # BF16 pairs of the combined row
        # BF16 pairs of p, FP32 alpha h' [column], then the FP32 sums:
        # [rank][task][sum].
        ("ffn", int(SUMS_AT) + world * int(UNPOOL_TASKS) * int(SUMS)),
        # BF16 words of the next attention input, and a flag per out task:
        # for the next layer's qkvg tasks at the end of the launch (qnext).
        ("anx", int(HIDDEN) // 4),
        ("anxf", int(OUT_TASKS)),
    )


def stages(
    bases: dict[str, int], rows: int, out_per_row: int, publish: bool
) -> tuple[Stage, ...]:
    """Comb1, comb2, unpool and out of ``rows`` token rows, placed by
    ``bases``, with ``out_per_row`` out tasks a row (a divisor of 16 and a
    multiple of 8: a rank's owned columns are whole tasks). With ``publish``
    out also publishes the next attention input for qnext (``anx``)."""
    if out_per_row not in (8, 16):
        raise ValueError(f"out tasks a row: 8 or 16, got {out_per_row}")
    comb, unpool, out = placement(bases, ("comb", "unpool", "out"))
    c, u = int(COMB_TASKS) * rows, int(UNPOOL_TASKS) * unpool_halves(rows)
    ffn = ("ffn", "ffnf") if ffn_words(rows) else ("ffn",)
    return (
        Stage("comb1", c, comb, reads=("rr", "route"), writes=("ss3",)),
        Stage("comb2", c, comb, reads=("ss3",), writes=("cmb",)),
        Stage("unpool", u, unpool, reads=("cmb", "hp"), writes=ffn),
        Stage(
            "out",
            out_per_row * rows,
            out,
            reads=ffn,
            writes=("anx", "anxf") if publish else (),
        ),
    )


@gluon.jit
def _put_all(mb, region: gl.constexpr, index, values):
    """Put ``values`` (FP32) at ``index`` of every rank's ``region``."""
    word = values.to(gl.int32, bitcast=True)
    for p in gl.static_range(WORLD):
        mb.put_peer(region, p, index, word)


@gluon.jit
def _bf16_word(lo, hi):
    """FP32 ``lo`` and ``hi`` rounded to BF16, as one int32 pair word
    (``lo`` in the low half), in ``lo``'s layout: ``mx.pack_bf16_pairs``
    builds the same word from a ``[..., 2]`` pair tensor, whose split
    would leave the word in another layout."""
    return (lo.to(gl.bfloat16).to(gl.int16, bitcast=True).to(gl.int32) & 0xFFFF) | (
        hi.to(gl.bfloat16).to(gl.int16, bitcast=True).to(gl.int32) << 16
    )


@gluon.jit
def comb_stages(mb, task, pen_gain, RANK: gl.constexpr):
    """Comb1 and comb2 task ``task``: owned pooled columns ``64 (task % 8)
    ..`` of row ``task // 8``.

    Args:
        pen_gain: FP32 ``bf16(w + 1)`` of the post-expert norm, owned columns.
    """
    mb.stamp("comb1", trace.START)
    row, task, _ = schedule.row_task(mb.plan, task, COMB_TASKS)
    mb = mb.row(row)
    nw: gl.constexpr = gl.num_warps()
    gl.static_assert(nw == 8, "comb lays out 8 waves")
    # [source, route, pair]: the 8 source ranks in a thread's registers, so
    # the rank-order sum stays in a thread; the routes over 2 lanes and 4
    # waves; the pairs over 32 lanes (2 waves repeat them).
    L3: gl.constexpr = gl.BlockedLayout([WORLD, 1, 1], [1, 2, 32], [1, 4, 2], [2, 1, 0])
    src = gl.arange(0, WORLD, gl.SliceLayout(1, gl.SliceLayout(2, L3)))[:, None, None]
    k = gl.arange(0, TOP_K, gl.SliceLayout(0, gl.SliceLayout(2, L3)))[None, :, None]
    j = gl.arange(0, COMB_COLS // 2, gl.SliceLayout(0, gl.SliceLayout(1, L3)))[
        None, None, :
    ]
    L2: gl.constexpr = gl.SliceLayout(0, L3)
    # The gains load before the poll: a load after a poll would start only
    # when the poll returns.
    jj = gl.arange(0, COMB_COLS // 2, gl.SliceLayout(0, L2))
    c = task * COMB_COLS + 2 * jj
    gain_lo = gl.load(pen_gain + c)
    gain_hi = gl.load(pen_gain + c + 1)
    (got,) = mb.poll_many(
        ("rr",), (((src * TOP_K + k) * OWNED_OUT + task * COMB_COLS) // 2 + j,)
    )
    mb.stamp("comb1", trace.READY)
    lo_all = mx.bf16_half(got, 0)
    hi_all = mx.bf16_half(got, 1)
    # The rank order of the regular path's reduction: 0, then 1, ...
    lo = gl.zeros([TOP_K, COMB_COLS // 2], gl.float32, L2)
    hi = gl.zeros([TOP_K, COMB_COLS // 2], gl.float32, L2)
    for p in gl.static_range(WORLD):
        lo += gl.sum(gl.where(src == p, lo_all, 0.0), axis=0)
        hi += gl.sum(gl.where(src == p, hi_all, 0.0), axis=0)
    lo = lo.to(gl.bfloat16).to(gl.float32) * gain_lo[None, :]
    hi = hi.to(gl.bfloat16).to(gl.float32) * gain_hi[None, :]
    ssq = gl.sum(lo * lo + hi * hi, axis=1)  # [8 routes]
    kk = gl.arange(0, TOP_K, gl.SliceLayout(1, L2))
    _put_all(mb, "ss3", (RANK * COMB_TASKS + task) * TOP_K + kk, ssq)
    mb.stamp("comb1", trace.END)

    mb.stamp("comb2", trace.START)
    tl_: gl.constexpr = gl.BlockedLayout([1, 1], [8, 8], [nw, 1], [1, 0])
    rt = gl.arange(0, WORLD * COMB_TASKS, gl.SliceLayout(1, tl_))[:, None]
    kt = gl.arange(0, TOP_K, gl.SliceLayout(0, tl_))[None, :]
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [nw], [0])
    got_s, got_w = mb.poll_many(
        ("ss3", "route"), (rt * TOP_K + kt, TOP_K + gl.arange(0, TOP_K, lanes))
    )
    mb.stamp("comb2", trace.READY)
    total = gl.sum(got_s.to(gl.float32, bitcast=True), axis=0)  # [8 routes]
    inv = gl.rsqrt(total / POOLED + POST_EPS)
    scale = gl.convert_layout(inv, gl.SliceLayout(1, L2)) * gl.convert_layout(
        got_w.to(gl.float32, bitcast=True), gl.SliceLayout(1, L2)
    )
    out_lo = gl.sum(lo * scale[:, None], axis=0)
    out_hi = gl.sum(hi * scale[:, None], axis=0)
    word = _bf16_word(out_lo, out_hi)
    for p in gl.static_range(WORLD):
        mb.put_peer("cmb", p, (RANK * OWNED_OUT + task * COMB_COLS) // 2 + jj, word)
    mb.stamp("comb2", trace.END)


@gluon.jit
def _ffn_norms(mb):
    """The inverse RMS of the post-FFN norm and of the next layer's input
    norm of ``mb``'s row, from every rank's sums in ``ffn``."""
    _pre_poll_sums(mb)
    (got,) = mb.poll_many(("ffn",), (_ffn_sums_index(),))
    return norm_scales(got)


@gluon.jit
def _ffn_word_flags(mb, first):
    """Past 8 rows: poll every rank's and unpool task's flag of the row
    group whose first row is ``first`` rows from ``mb``'s (``ffnf`` sits in
    that row), then meet at a barrier: the words load after."""
    lanes = gl.arange(0, 64 * gl.num_warps(), mailbox.lane_layout(gl.num_warps()))
    N: gl.constexpr = WORLD * UNPOOL_TASKS
    at = first * schedule.row_pairs(mb.plan)
    mb.pre_poll("ffnf", at + gl.minimum(lanes * N // 64, N - 1))
    f = gl.arange(0, N, lanes.type.layout)
    mb.poll_flags("ffnf", at + f)
    gl.barrier()


@gluon.jit
def _pre_poll_sums(mb):
    """:meth:`Mailbox.pre_poll` on 64 of ``ffn``'s sums, spread evenly (each
    unpool task puts its sums last in its one put)."""
    N: gl.constexpr = WORLD * UNPOOL_TASKS * SUMS
    lanes = gl.arange(0, 64 * gl.num_warps(), mailbox.lane_layout(gl.num_warps()))
    mb.pre_poll("ffn", SUMS_AT + gl.minimum(lanes * N // 64, N - 1))


@gluon.jit
def _ffn_sums_index():
    """The ``ffn`` pair index of every rank's and task's 4 sums."""
    nw: gl.constexpr = gl.num_warps()
    sl: gl.constexpr = gl.BlockedLayout([1, SUMS], [64, 1], [nw, 1], [1, 0])
    rt = gl.arange(0, WORLD * UNPOOL_TASKS, gl.SliceLayout(1, sl))[:, None]
    qt = gl.arange(0, SUMS, gl.SliceLayout(0, sl))[None, :]
    return SUMS_AT + rt * SUMS + qt


@gluon.constexpr_function
def unpool_halves(rows):
    """Row groups of unpool: at 8 or 16 rows, two tasks of each column
    block take half the rows each, so a task reads half the combined rows
    (the read is bytes-bound) with the same weights; the GEMV rows are
    independent, so the values do not change."""
    rows = int(rows)
    return 2 if rows in (8, 16) else 1


@gluon.jit
def unpool_stages(
    mb,
    task,
    w,
    post_gain,
    alpha,
    beta,
    RANK: gl.constexpr,
):
    """Unpool task ``task``: owned hidden columns ``16 (task % 64) ..`` of
    row group ``task // 64`` (:func:`unpool_halves`).

    Args:
        w: This rank's post-expert projection rows, BF16 ``[1024, 4096]``
            (``weights.bf16_tiles``).
        post_gain: FP32 ``bf16(w + 1)`` of the post-FFN norm, all 8192
            columns.
        alpha: FP32 residual-gate coefficient of the FFN, owned columns.
        beta: FP32 residual-gate coefficient of the FFN, all 8192 columns.
    """
    H: gl.constexpr = unpool_halves(schedule.rows(mb.plan))
    R: gl.constexpr = schedule.rows(mb.plan) // H
    nw: gl.constexpr = gl.num_warps()
    mb.stamp("unpool", trace.START)
    # The task's row group: the mailbox view of its first row.
    mb = mb.row((task // UNPOOL_TASKS) * R)
    task = task % UNPOOL_TASKS
    tiles = bf16.load_weight_tiles(
        w,
        nb0=task,
        K=POOLED,
        kit0=0,
        KITS=POOLED // 32,
        SPLIT=gl.num_warps(),
        NW=1,
        GROUP=1,
        GROUP_STRIDE=0,
    )
    # The parameters load before the polls: a load after a poll would start
    # only when the poll returns.
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [nw], [0])
    j = gl.arange(0, UNPOOL_COLS // 2, lanes)
    c = task * UNPOOL_COLS + 2 * j
    post_lo = gl.load(post_gain + RANK * OWNED + c)
    post_hi = gl.load(post_gain + RANK * OWNED + c + 1)
    b_lo = gl.load(beta + RANK * OWNED + c)
    b_hi = gl.load(beta + RANK * OWNED + c + 1)
    a_lo = gl.load(alpha + c)
    a_hi = gl.load(alpha + c + 1)
    if R > 1:
        # Past one row every wave of many CTAs polls several whole rows:
        # one wave spins on hints first (unthrottled, the flood of polls
        # starves the producers' stores across ranks and the grid livelocks).
        mb.row(R - 1).pre_poll_spread("cmb", POOLED // 2)
    # Every row's h' of the task's pairs, in the epilogue's [row, pair] layout.
    RPP: gl.constexpr = triton.next_power_of_2(R)
    hl: gl.constexpr = gl.SliceLayout(
        1, gl.BlockedLayout([1, 1, 1], [1, 8, 8], [nw, 1, 1], [2, 1, 0])
    )
    hr = gl.arange(0, RPP, gl.SliceLayout(1, hl))[:, None]
    hc = task * UNPOOL_COLS + 2 * gl.arange(0, UNPOOL_COLS // 2, gl.SliceLayout(0, hl))
    hat = mb.row_offset(hr) + hc[None, :]
    # hp is polled with cmb's first row group, so on the launch's critical
    # path it costs no round trip of its own.
    x_smem, hp = bf16.load_rows(mb, "cmb", 0, POOLED, R, ("hp", "hp"), (hat, hat + 1))
    mb.stamp("unpool", trace.READY)
    out = bf16.gemv(x_smem, tiles, FIRST=0, ROWS=R)  # [RP, 16]
    mb.stamp("unpool", trace.COMPUTED)
    RP: gl.constexpr = out.shape[0]
    # split needs a pair's two values in one thread.
    pairs: gl.constexpr = gl.BlockedLayout([1, 2], [1, 64], [nw, 1], [1, 0])
    lo, hi = gl.split(
        gl.reshape(gl.convert_layout(out, pairs), [RP, UNPOOL_COLS // 2, 2])
    )
    # Every row at once: [row, kind, pair], a wave per row. Kinds 0 .. 2 are
    # p (both columns), (alpha h')_lo, (alpha h')_hi of the 8 pairs; kind 3
    # holds the row's 4 sums. One put per peer serves every row.
    L3: gl.constexpr = gl.BlockedLayout([1, 1, 1], [1, 8, 8], [nw, 1, 1], [2, 1, 0])
    P2: gl.constexpr = gl.SliceLayout(1, L3)
    r2 = gl.arange(0, RP, gl.SliceLayout(1, P2))[:, None]
    j2 = gl.arange(0, UNPOOL_COLS // 2, gl.SliceLayout(0, P2))[None, :]
    hp_lo = hp[0].to(gl.float32, bitcast=True)
    hp_hi = hp[1].to(gl.float32, bitcast=True)
    p_lo = gl.convert_layout(lo, P2).to(gl.bfloat16).to(gl.float32)
    p_hi = gl.convert_layout(hi, P2).to(gl.bfloat16).to(gl.float32)
    p_word = _bf16_word(p_lo, p_hi).to(gl.float32, bitcast=True)
    y_lo = p_lo * gl.convert_layout(post_lo, gl.SliceLayout(0, P2))[None, :]
    y_hi = p_hi * gl.convert_layout(post_hi, gl.SliceLayout(0, P2))[None, :]
    a2_lo = gl.convert_layout(a_lo, gl.SliceLayout(0, P2))[None, :]
    a2_hi = gl.convert_layout(a_hi, gl.SliceLayout(0, P2))[None, :]
    b2_lo = gl.convert_layout(b_lo, gl.SliceLayout(0, P2))[None, :]
    b2_hi = gl.convert_layout(b_hi, gl.SliceLayout(0, P2))[None, :]
    ah_lo = a2_lo * hp_lo
    ah_hi = a2_hi * hp_hi
    by_lo = b2_lo * y_lo
    by_hi = b2_hi * y_hi
    s_y = gl.sum(y_lo * y_lo + y_hi * y_hi, axis=1)[:, None]
    s_by = gl.sum(by_lo * by_lo + by_hi * by_hi, axis=1)[:, None]
    s_x = gl.sum(ah_lo * by_lo + ah_hi * by_hi, axis=1)[:, None]
    s_ah = gl.sum(ah_lo * ah_lo + ah_hi * ah_hi, axis=1)[:, None]
    sr = gl.where(j2 == 0, s_y, gl.where(j2 == 1, s_by, gl.where(j2 == 2, s_x, s_ah)))
    k = gl.arange(0, 8, gl.SliceLayout(0, gl.SliceLayout(2, L3)))[None, :, None]
    j3 = j2[:, None, :]
    r3 = r2[:, None, :]
    vals = gl.where(
        k == 0,
        p_word[:, None, :],
        gl.where(
            k == 1,
            ah_lo[:, None, :],
            gl.where(k == 2, ah_hi[:, None, :], sr[:, None, :]),
        ),
    )
    col = RANK * OWNED + task * UNPOOL_COLS + 2 * j3
    idx = mb.row_offset(r3) + gl.where(
        k == 0,
        col // 2,
        gl.where(
            k < 3,
            AH_AT + col + k - 1,
            SUMS_AT + (RANK * UNPOOL_TASKS + task) * SUMS + j3,
        ),
    )
    word = vals.to(gl.int32, bitcast=True)
    put = ((k < 3) | ((k == 3) & (j3 < SUMS))) & (r3 < R)
    if ffn_words(schedule.rows(mb.plan)):
        # Plain words (row r's copy at word 2 row_offset(r)), then one flag:
        # the stores go out before any later load (vmcnt is in order).
        widx = 2 * mb.row_offset(r3) + (idx - mb.row_offset(r3))
        for p in gl.static_range(WORLD):
            mb.store_words_peer("ffn", p, widx, word, mask=put)
        mb.raise_flag_peers("ffnf", RANK * UNPOOL_TASKS + task)
    else:
        for p in gl.static_range(WORLD):
            mb.put_peer("ffn", p, idx, word, mask=put)
    mb.stamp("unpool", trace.END)


@gluon.jit
def out_stage(
    mb,
    task,
    post_gain,
    beta,
    in_gain,
    a_out,
    h_next,
    tap,
    tap_col,
    tap_row,
    RANK: gl.constexpr,
    TAP: gl.constexpr,
    WHOLE_ROWS: gl.constexpr,
    PER_ROW: gl.constexpr,
    PUBLISH: gl.constexpr,
    STAGE: gl.constexpr,
):
    """Out task ``task``: next attention input values ``(8192 / PER_ROW)
    (task % PER_ROW) ..`` of row ``task // PER_ROW`` to ``a_out`` (BF16
    ``[rows, 8192]``), from every rank's ``ffn``. The residual ``h''`` goes
    to ``h_next``: with ``WHOLE_ROWS`` every task stores its columns (each
    rank then holds whole rows, for a next layer on the regular path), else
    the tasks of the rank's owned columns store those. With ``PUBLISH`` the
    task first publishes its values as plain words in ``anx`` and raises
    its flag in ``anxf`` (for qnext: the launch's own CTAs cannot read
    ``a_out``'s plain stores).

    Args:
        post_gain: FP32 ``bf16(w + 1)`` of the post-FFN norm, all 8192
            columns.
        beta: FP32 residual-gate coefficient of the FFN, all 8192 columns.
        in_gain: FP32 ``bf16(w + 1)`` of the next layer's input norm, all
            8192 columns.
        h_next: Where the FP32 residual ``h''`` goes: ``[rows, 8192]`` with
            ``WHOLE_ROWS``, else the owned columns, ``[rows, 1024]``.
        tap, tap_col, tap_row: With ``TAP``, this rank's BF16 buffer that
            gets ``bf16(h'')`` of the task's columns: row ``r``, column ``c``
            (of 8192) at ``tap + r tap_row + tap_col + c``. A draft reads
            this residual (the one entering the next layer) as a tap. Unused
            without ``TAP``.
    """
    mb.stamp(STAGE, trace.START)
    row, task, _ = schedule.row_task(mb.plan, task, PER_ROW)
    mb = mb.row(row)
    a_out = a_out + row * HIDDEN
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    WORDS: gl.constexpr = HIDDEN // 2 // PER_ROW
    i = task * WORDS + gl.arange(0, WORDS, lanes)
    c = 2 * i
    post_lo = gl.load(post_gain + c)
    post_hi = gl.load(post_gain + c + 1)
    b_lo = gl.load(beta + c)
    b_hi = gl.load(beta + c + 1)
    in_lo = gl.load(in_gain + c)
    in_hi = gl.load(in_gain + c + 1)
    if ffn_words(schedule.rows(mb.plan)):
        # Past 8 rows: the row group's flags, then the words.
        G: gl.constexpr = schedule.rows(mb.plan) // unpool_halves(
            schedule.rows(mb.plan)
        )
        _ffn_word_flags(mb, (row // G) * G - row)
        inv, inv2 = norm_scales(mb.load_words("ffn", _ffn_sums_index()))
        pw = mb.load_words("ffn", i)
        ah_lo = mb.load_words("ffn", AH_AT + c)
        ah_hi = mb.load_words("ffn", AH_AT + c + 1)
    else:
        # Each unpool task puts its columns and its sums in one store, so the
        # columns' poll after the sums' barely waits.
        inv, inv2 = _ffn_norms(mb)
        pw, ah_lo, ah_hi = mb.poll_many(
            ("ffn", "ffn", "ffn"), (i, AH_AT + c, AH_AT + c + 1)
        )
    mb.stamp(STAGE, trace.READY)
    # y as unpool computed it: the BF16 projection times the gain.
    y_lo = mx.bf16_half(pw, 0) * post_lo
    y_hi = mx.bf16_half(pw, 1) * post_hi
    h_lo = gl.fma(b_lo, y_lo * inv, ah_lo.to(gl.float32, bitcast=True))
    h_hi = gl.fma(b_hi, y_hi * inv, ah_hi.to(gl.float32, bitcast=True))
    word = _bf16_word(h_lo * inv2 * in_lo, h_hi * inv2 * in_hi)
    if PUBLISH:
        # First: raise_flag waits for every earlier store of the wave.
        mb.store_words("anx", i, word)
        mb.raise_flag("anxf", task)
    gl.store(a_out.to(gl.pointer_type(gl.int32), bitcast=True) + i, word)
    if WHOLE_ROWS:
        # The same arithmetic on every rank as on the owner of the columns:
        # the whole rows match a gather of every rank's owned columns.
        at = h_next + row * HIDDEN + c
        gl.store(at, h_lo)
        gl.store(at + 1, h_hi)
    if not WHOLE_ROWS and task // (PER_ROW // WORLD) == RANK:
        # The owned columns of the residual output.
        at = h_next + row * OWNED + c - RANK * OWNED
        gl.store(at, h_lo)
        gl.store(at + 1, h_hi)
    if TAP:
        # Rounded once to BF16 as the regular capture rounds the FP32 stream.
        # Written through: the draft reads the tap past the caches.
        t_at = tap + row.to(gl.int64) * tap_row + tap_col
        gl.store(
            t_at.to(gl.pointer_type(gl.int32), bitcast=True) + i,
            _bf16_word(h_lo, h_hi),
            cache_modifier=".wt",
        )
    mb.stamp(STAGE, trace.END)

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

"""Output projection and attention tail of one Muse Spark rank.

Rank ``r`` owns hidden columns ``1024 r .. 1024 r + 1023``: it keeps only
that slice of the FP32 residual, and it reduces only those columns of the
output projection (a reduce-scatter).

* oproj, 128 tasks of 64 output columns: the rank's ``o_proj`` partial
  (BF16 ``[8192, 1024]``, 128 KiB a task) of the gated head-norm output
  ``z``, rounded to BF16 like the regular path's partial. A task puts its
  columns of every row into the owner rank's ``red`` slot of this rank.
* tail1, tail2, 8 tasks of 128 owned columns per row on the same CTAs (each
  task keeps its values in registers from one stage to the next):

  - tail1 sums the 8 ranks' partials in rank order and rounds to BF16
    (``b``). It puts four partial sums of its columns into every rank's
    ``ss1``: of ``b^2``, ``(beta b)^2``, ``(alpha h)(beta b)`` and ``(alpha
    h)^2``.
  - tail2, after one exchange: ``inv = rsqrt(mean(b^2) + 1e-8)`` (the
    weightless post-attention norm over all 8192 columns), the residual
    gate ``h' = beta b inv + alpha h`` (published in ``hp``), and ``u = h'
    rsqrt(mean(h'^2) + 1e-5) bf16(w + 1)`` (the pre-FFN norm), whose sum of
    squares is the expansion of ``(beta b inv + alpha h)^2`` in the four
    sums (:func:`norm_scales`). ``u`` goes as BF16 into every rank's
    ``xin`` (the MoE input, which the pooler reads whole) as plain words,
    then one flag per rank in ``xflag``: the pooler reads half the bytes of
    tagged pairs. The router does not wait for it: it sums ``b`` itself from
    ``red`` and takes the scales from ``ss1`` (see moe_route).
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.monokernel import bf16, mx, schedule, trace
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.dims import (
    HIDDEN,
    OWNED,
    WORLD,
    Q,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import Stage, placement

OPROJ_TASKS = gl.constexpr(128)
OPROJ_COLS = gl.constexpr(64)
TAIL_TASKS = gl.constexpr(8)
TAIL_COLS = gl.constexpr(128)
SUMS = gl.constexpr(4)  # partial sums a tail task puts (see tail_stages)
POST_EPS = gl.constexpr(1e-8)
PRE_EPS = gl.constexpr(1e-5)

PEERS = ("red", "ss1", "xin", "xflag")


def regions() -> tuple[tuple[str, int], ...]:
    world, owned, tasks = int(WORLD), int(OWNED), int(TAIL_TASKS)
    return (
        ("red", world * owned // 2),  # BF16 pairs [source rank][owned column]
        ("ss1", world * tasks * int(SUMS)),  # FP32 partial sums [rank][task][sum]
        ("hp", owned),  # FP32 owned slice of the residual after attention (h')
        # Words of BF16 pairs of the whole MoE input, rank-major (2 words a
        # pair of the region), and a flag per tail task: [rank][task].
        ("xin", world * owned // 4),
        ("xflag", world * tasks),
    )


def stages(bases: dict[str, int], rows: int) -> tuple[Stage, ...]:
    """Oproj, tail1 and tail2 of ``rows`` token rows, placed by ``bases``."""
    oproj, tail = placement(bases, ("oproj", "tail"))
    tasks = int(TAIL_TASKS) * rows
    return (
        Stage("oproj", int(OPROJ_TASKS), oproj, reads=("z",), writes=("red",)),
        Stage("tail1", tasks, tail, reads=("red",), writes=("ss1",)),
        Stage("tail2", tasks, tail, reads=("ss1",), writes=("hp", "xin", "xflag")),
    )


@gluon.jit
def oproj_stage(mb, task, w, RANK: gl.constexpr, STAGE: gl.constexpr):
    """Oproj task ``task``: output columns ``64 task ..`` of every row.

    Args:
        w: The rank's ``o_proj``, BF16 ``[8192, 1024]`` (``weights.bf16_tiles``).
    """
    R: gl.constexpr = schedule.rows(mb.plan)
    mb.stamp(STAGE, trace.START)
    NW: gl.constexpr = OPROJ_COLS // 16
    tiles = bf16.load_weight_tiles(
        w,
        nb0=task * NW,
        K=Q,
        kit0=0,
        KITS=Q // 32,
        SPLIT=gl.num_warps() // NW,
        NW=NW,
        GROUP=NW,
        GROUP_STRIDE=0,
    )
    x_smem, _ = bf16.load_rows(mb, "z", 0, Q, R)
    mb.stamp(STAGE, trace.READY)
    out = bf16.gemv(x_smem, tiles, FIRST=0, ROWS=R)  # FP32 [RP, 64]
    mb.stamp(STAGE, trace.COMPUTED)
    RP: gl.constexpr = out.shape[0]
    pairs: gl.constexpr = gl.BlockedLayout([1, 2], [1, 64], [gl.num_warps(), 1], [1, 0])
    values = gl.convert_layout(out.to(gl.bfloat16), pairs)
    word = gl.reshape(
        mx.pack_bf16_pairs(gl.reshape(values, [RP, OPROJ_COLS // 2, 2])),
        [RP, OPROJ_COLS // 2],
    )
    wl: gl.constexpr = word.type.layout
    r = gl.arange(0, RP, gl.SliceLayout(1, wl))[:, None]
    p = gl.arange(0, OPROJ_COLS // 2, gl.SliceLayout(0, wl))[None, :]
    cols_per_rank: gl.constexpr = OWNED // OPROJ_COLS
    owner = task // cols_per_rank
    at = RANK * (OWNED // 2) + (task % cols_per_rank) * (OPROJ_COLS // 2) + p
    # One put serves every row; the padding rows past R are masked.
    mb.put_peer("red", owner, mb.row_offset(r) + at, word, mask=(r < R) & (p >= 0))
    mb.stamp(STAGE, trace.END)


@gluon.constexpr_function
def sums_layout(num_warps):
    """Layout of the ``[WORLD * TAIL_TASKS, SUMS]`` partial sums in ``ss1``."""
    return gl.BlockedLayout([1, SUMS], [64, 1], [num_warps, 1], [1, 0])


@gluon.jit
def sums_index(num_warps: gl.constexpr):
    """The ``ss1`` pair index of every partial sum, in :func:`sums_layout`."""
    sl: gl.constexpr = sums_layout(num_warps)
    rt = gl.arange(0, WORLD * TAIL_TASKS, gl.SliceLayout(1, sl))[:, None]
    q = gl.arange(0, SUMS, gl.SliceLayout(0, sl))[None, :]
    return rt * SUMS + q


@gluon.jit
def norm_scales(got):
    """The post-attention and pre-FFN norm scales ``(inv, inv2)`` from the
    polled ``ss1`` pairs (:func:`sums_index`)."""
    total = gl.sum(got.to(gl.float32, bitcast=True), axis=0)  # [SUMS]
    q = gl.arange(0, SUMS, total.type.layout)
    ss_b = gl.sum(gl.where(q == 0, total, 0.0), axis=0)
    ss_bb = gl.sum(gl.where(q == 1, total, 0.0), axis=0)
    ss_x = gl.sum(gl.where(q == 2, total, 0.0), axis=0)
    ss_ah = gl.sum(gl.where(q == 3, total, 0.0), axis=0)
    inv = gl.rsqrt(ss_b / HIDDEN + POST_EPS)
    # The sum of h'^2 over all columns: inv^2 ss_bb + 2 inv ss_x + ss_ah.
    ss_h = gl.maximum(inv * inv * ss_bb + 2.0 * inv * ss_x + ss_ah, 0.0)
    return inv, gl.rsqrt(ss_h / HIDDEN + PRE_EPS)


@gluon.jit
def tail_stages(mb, task, h, alpha, beta, pre_gain, RANK: gl.constexpr):
    """Tail tasks ``task`` of tail1 and tail2: owned columns ``128 (task %
    8) ..`` (of the rank's 1024) of row ``task // 8``.

    Args:
        h: The rank's FP32 residual slice ``[rows, 1024]`` (the previous
            launch's).
        alpha, beta: FP32 residual-gate coefficients of the owned columns.
        pre_gain: FP32 ``bf16(w + 1)`` of the pre-FFN norm, owned columns.
    """
    mb.stamp("tail1", trace.START)
    row, task, _ = schedule.row_task(mb.plan, task, TAIL_TASKS)
    mb = mb.row(row)
    h = h + row * OWNED
    nw: gl.constexpr = gl.num_warps()
    rl: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [1, nw], [1, 0])
    src = gl.arange(0, WORLD, gl.SliceLayout(1, rl))
    j = gl.arange(0, TAIL_COLS // 2, gl.SliceLayout(0, rl))
    # The parameters load before the first poll: a load after a poll would
    # start only when the poll returns.
    c = task * TAIL_COLS + 2 * j
    beta_lo = gl.load(beta + c)
    beta_hi = gl.load(beta + c + 1)
    ah_lo = gl.load(alpha + c) * gl.load(h + c)
    ah_hi = gl.load(alpha + c + 1) * gl.load(h + c + 1)
    gain_lo = gl.load(pre_gain + c)
    gain_hi = gl.load(pre_gain + c + 1)
    (got,) = mb.poll_many(
        ("red",), (src[:, None] * (OWNED // 2) + task * (TAIL_COLS // 2) + j[None, :],)
    )
    mb.stamp("tail1", trace.READY)
    lo_all = (got << 16).to(gl.float32, bitcast=True)
    hi_all = (got & -65536).to(gl.float32, bitcast=True)
    # The rank order of the regular path's reduction: 0, then 1, ...
    lo = gl.zeros([TAIL_COLS // 2], gl.float32, gl.SliceLayout(0, rl))
    hi = gl.zeros([TAIL_COLS // 2], gl.float32, gl.SliceLayout(0, rl))
    for p in gl.static_range(WORLD):
        lo += gl.sum(gl.where(src[:, None] == p, lo_all, 0.0), axis=0)
        hi += gl.sum(gl.where(src[:, None] == p, hi_all, 0.0), axis=0)
    lo = lo.to(gl.bfloat16).to(gl.float32)
    hi = hi.to(gl.bfloat16).to(gl.float32)
    bb_lo = beta_lo * lo
    bb_hi = beta_hi * hi
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [nw], [0])
    q = gl.arange(0, SUMS, lanes)
    sums = gl.where(
        q == 0,
        gl.sum(lo * lo + hi * hi, axis=0),
        gl.where(
            q == 1,
            gl.sum(bb_lo * bb_lo + bb_hi * bb_hi, axis=0),
            gl.where(
                q == 2,
                gl.sum(ah_lo * bb_lo + ah_hi * bb_hi, axis=0),
                gl.sum(ah_lo * ah_lo + ah_hi * ah_hi, axis=0),
            ),
        ),
    )
    word = gl.convert_layout(sums, lanes).to(gl.int32, bitcast=True)
    for p in gl.static_range(WORLD):
        mb.put_peer("ss1", p, (RANK * TAIL_TASKS + task) * SUMS + q, word)
    mb.stamp("tail1", trace.END)

    mb.stamp("tail2", trace.START)
    (got,) = mb.poll_many(("ss1",), (sums_index(nw),))
    mb.stamp("tail2", trace.READY)
    inv, inv2 = norm_scales(got)
    hp_lo = gl.fma(beta_lo, lo * inv, ah_lo)
    hp_hi = gl.fma(beta_hi, hi * inv, ah_hi)
    # The MoE tail reads h' on other CTAs: through the mailbox, not memory.
    mb.put("hp", c, hp_lo.to(gl.int32, bitcast=True))
    mb.put("hp", c + 1, hp_hi.to(gl.int32, bitcast=True))
    u_lo = hp_lo * inv2 * gain_lo
    u_hi = hp_hi * inv2 * gain_hi
    word = (u_lo.to(gl.bfloat16).to(gl.int16, bitcast=True).to(gl.int32) & 0xFFFF) | (
        u_hi.to(gl.bfloat16).to(gl.int16, bitcast=True).to(gl.int32) << 16
    )
    for p in gl.static_range(WORLD):
        mb.store_words_peer("xin", p, RANK * (OWNED // 2) + c // 2, word)
    mb.raise_flag_peers("xflag", RANK * TAIL_TASKS + task)
    mb.stamp("tail2", trace.END)

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

"""The MLP half of a DFlash2 draft layer of Muse Spark, one TP8 rank.

The draft (RadixArk/Muse-Spark-Open-DFlash2, ``runtime/models/dflash2.py``)
hands each layer's attention sublayer to the MLP as a per-rank partial, after
the attention convolution. This half sums the ranks' partials, adds the
residual, applies the post-attention RMSNorm, and runs the MLP with its
convolution, ending in this rank's partial again (the next layer's input norm
sums the ranks').

Rank ``r`` owns hidden columns ``1024 r .. 1024 r + 1023`` for the sum and the
norm, as in ``attn_tail``:

* scat, 128 tasks of 64 columns: a task reads its columns of this rank's BF16
  partial ``z`` (every row) and puts them into the owner rank's ``red`` slot
  of this rank.
* sum1, norm, 8 tasks of 128 owned columns per row on the same CTAs (each
  task keeps its values in registers from one stage to the next):

  - sum1 sums the 8 ranks' partials in rank order and rounds the sum to BF16,
    as the regular path's all-reduce does, adds the residual in FP32 (the
    stream), stores the BF16 residual, and puts the stream's sum of squares
    over its columns into every rank's ``ss``.
  - norm, after that exchange: ``h = bf16(stream rsqrt(mean(stream^2) +
    1e-5) w)`` over all 8192 columns, put as BF16 into every rank's ``hin``
    (the MLP input, which the projections read whole).

* coef, 256 tasks: the MLP convolution's coefficients ``h W_kp^T`` (2048 =
  side x tap x 512 groups; the projection is replicated, 32 MiB): 128 groups
  of 16 output rows x 2 K parts of 4096, 128 KiB of weight a task, FP32
  partials into ``cpart``. The convolutions add a coefficient's two partials
  and round to BF16 themselves (:func:`coefficient`): no reduce stage.
* conv0, 64 tasks of 128 channels: the convolution before the MLP,
  ``(base0 + delta0) h[t] + (base1 + delta1) h[t - 1]`` within each block of
  8 rows, rounded once to BF16, into ``hc``.
* gu, 896 tasks in rounds: gate/up (3584 + 3584 rows a rank, 112 MiB) in 56
  groups of 128 rows x 16 K parts of 512, FP32 partials into ``gpart``.
* act, 112 tasks of 32 dimensions: the K parts' sums rounded to BF16, then
  ``silu(g) u`` rounded to BF16, into ``act``.
* down, 448 tasks in rounds: down (8192 x 3584, 56 MiB) in 64 groups of 128
  rows x 7 K parts of 512, FP32 partials into ``dpart``.
* fin, 256 tasks of 32 columns: the K parts' sum rounded to BF16, then the
  convolution after the MLP with side 1's coefficients, rounded to BF16:
  this rank's partial, stored in ``out``.

The projections share :func:`proj_issue`, :func:`proj_poll` and
:func:`proj_compute`; their
weights are in ``weights.bf16_tiles`` order. A task issues its weight loads
before it polls its input, so the weights stream while the earlier stages
finish.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.monokernel import bf16, mx, schedule, trace
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.dims import (
    DRAFT_BLOCK,
    HIDDEN,
    OWNED,
    WORLD,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import Stage, placement

SCAT_TASKS = gl.constexpr(128)
SCAT_COLS = gl.constexpr(64)
SUM_TASKS = gl.constexpr(8)  # tasks per row of sum1 and norm
SUM_COLS = gl.constexpr(128)
EPS = gl.constexpr(1e-5)
COEFS = gl.constexpr(2048)  # side x tap x group of the MLP convolution
# Gate/up and down tasks: 128 output rows x a 512-wide K part (128 KiB of
# weight). Longer K parts make fewer partials to sum, but each task then polls
# a longer stretch of its input rows: 64 x 1024 and 32 x 2048 for gate/up, and
# 32 x 1792 for down, measured equal or slower.
GU_ROWS = gl.constexpr(128)
GU_KS = gl.constexpr(512)
GU_KPARTS = gl.constexpr(16)  # HIDDEN / GU_KS
DOWN_ROWS = gl.constexpr(128)
DOWN_KS = gl.constexpr(512)
DOWN_KPARTS = gl.constexpr(7)  # INTER / DOWN_KS
COEF_KS = gl.constexpr(4096)  # K values of a coef task
COEF_KPARTS = gl.constexpr(2)  # HIDDEN / COEF_KS
COEF_ROWS = gl.constexpr(16)  # output rows of a coef task
GROUPS = gl.constexpr(512)  # convolution groups of 16 channels
GROUP_SIZE = gl.constexpr(16)
BLOCK = DRAFT_BLOCK  # the convolution's span
CONV_TASKS = gl.constexpr(64)
CONV_COLS = gl.constexpr(128)
INTER = gl.constexpr(3584)  # gate (and up) rows a rank
GU = gl.constexpr(7168)  # gate then up rows
ACT_TASKS = gl.constexpr(112)
ACT_COLS = gl.constexpr(32)
FIN_TASKS = gl.constexpr(256)
FIN_COLS = gl.constexpr(32)

PEERS = ("red", "ss", "hin")


def regions() -> tuple[tuple[str, int], ...]:
    world, owned, tasks = int(WORLD), int(OWNED), int(SUM_TASKS)
    return (
        ("red", world * owned // 2),  # BF16 pairs [source rank][owned column]
        ("ss", world * tasks),  # FP32 sums of squares [rank][task]
        ("hin", int(HIDDEN) // 2),  # BF16 pairs of the whole MLP input
        ("cpart", int(COEF_KPARTS) * int(COEFS)),  # FP32 [K part][coefficient]
        ("hc", int(HIDDEN) // 2),  # BF16 pairs of the convolved MLP input
        ("gpart", int(GU_KPARTS) * int(GU)),  # FP32 [K part][gate, up row]
        ("act", int(INTER) // 2),  # BF16 pairs of silu(g) u
        ("dpart", int(DOWN_KPARTS) * int(HIDDEN)),  # FP32 [K part][column]
    )


def stages(bases: dict[str, int], rows: int) -> tuple[Stage, ...]:
    """The stages of ``rows`` token rows, placed by ``bases``."""
    names = ("scat", "sum", "coef", "conv0", "gu", "act", "down", "fin")
    scat, tail, coef, conv0, gu, act, down, fin = placement(bases, names)
    tasks = int(SUM_TASKS) * rows
    return (
        Stage("scat", int(SCAT_TASKS), scat, writes=("red",)),
        Stage("sum1", tasks, tail, reads=("red",), writes=("ss",)),
        Stage("norm", tasks, tail, reads=("ss",), writes=("hin",)),
        Stage(
            "coef",
            int(COEF_KPARTS) * int(COEFS) // int(COEF_ROWS),
            coef,
            reads=("hin",),
            writes=("cpart",),
        ),
        Stage("conv0", int(CONV_TASKS), conv0, reads=("hin", "cpart"), writes=("hc",)),
        Stage(
            "gu",
            int(GU_KPARTS) * int(GU) // int(GU_ROWS),
            gu,
            reads=("hc",),
            writes=("gpart",),
        ),
        Stage("act", int(ACT_TASKS), act, reads=("gpart",), writes=("act",)),
        Stage(
            "down",
            int(DOWN_KPARTS) * int(HIDDEN) // int(DOWN_ROWS),
            down,
            reads=("act",),
            writes=("dpart",),
        ),
        Stage("fin", int(FIN_TASKS), fin, reads=("dpart", "cpart")),
    )


@gluon.jit
def scat_stage(mb, task, z, RANK: gl.constexpr, RED: gl.constexpr, STAGE: gl.constexpr):
    """Scat task ``task``: columns ``64 task ..`` of every row of ``z``, into
    the owner ranks' ``RED`` (the peer region of the sum).

    Args:
        z: This rank's BF16 partial ``[rows, 8192]``, written by an earlier
            launch.
    """
    R: gl.constexpr = schedule.rows(mb.plan)
    RP: gl.constexpr = triton.next_power_of_2(R)
    mb.stamp(STAGE, trace.START)
    pairs: gl.constexpr = gl.BlockedLayout([1, 2], [1, 64], [gl.num_warps(), 1], [1, 0])
    rr = gl.arange(0, RP, gl.SliceLayout(1, pairs))[:, None]
    cc = gl.arange(0, SCAT_COLS, gl.SliceLayout(0, pairs))[None, :]
    values = gl.load(
        z + rr * HIDDEN + task * SCAT_COLS + cc, mask=(rr < R) & (cc >= 0), other=0.0
    )
    word = gl.reshape(
        mx.pack_bf16_pairs(gl.reshape(values, [RP, SCAT_COLS // 2, 2])),
        [RP, SCAT_COLS // 2],
    )
    wl: gl.constexpr = word.type.layout
    r = gl.arange(0, RP, gl.SliceLayout(1, wl))[:, None]
    p = gl.arange(0, SCAT_COLS // 2, gl.SliceLayout(0, wl))[None, :]
    cols_per_rank: gl.constexpr = OWNED // SCAT_COLS
    owner = task // cols_per_rank
    at = RANK * (OWNED // 2) + (task % cols_per_rank) * (SCAT_COLS // 2) + p
    # One put serves every row; the padding rows past R are masked.
    mb.put_peer(RED, owner, mb.row_offset(r) + at, word, mask=(r < R) & (p >= 0))
    mb.stamp(STAGE, trace.END)


@gluon.jit
def residual_zeros():
    """Zeros in the layout of :func:`sum_norm_regs`' residual (even or odd
    owned columns of one sum task), for a CTA that has no such task."""
    nw: gl.constexpr = gl.num_warps()
    rl: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [1, nw], [1, 0])
    return gl.zeros([SUM_COLS // 2], gl.float32, gl.SliceLayout(0, rl))


@gluon.jit
def sum_norm_stages(
    mb,
    task,
    residual,
    residual_out,
    gain,
    RANK: gl.constexpr,
    RED: gl.constexpr,
    SS: gl.constexpr,
    HIN: gl.constexpr,
    SUM: gl.constexpr,
    NORM: gl.constexpr,
    STORE: gl.constexpr,
):
    """Tasks ``task`` of the sum and the norm (stages ``SUM``, ``NORM``):
    owned columns ``128 (task % 8) ..`` (of the rank's 1024) of row
    ``task // 8``, with the residual from memory (:func:`sum_norm_regs`).

    Args:
        residual: BF16 ``[rows, 8192]`` residual before the sublayer (the
            same on every rank), written by an earlier launch.
        residual_out: BF16 ``[rows, 8192]``; with ``STORE`` this rank stores
            its owned columns of the residual after the sublayer.
        gain: FP32 ``[1024]`` norm gain, owned columns.

    Returns:
        The residual after the sublayer, as :func:`sum_norm_regs`.
    """
    mb.stamp(SUM, trace.START)
    row, t, _ = schedule.row_task(mb.plan, task, SUM_TASKS)
    nw: gl.constexpr = gl.num_warps()
    rl: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [1, nw], [1, 0])
    j = gl.arange(0, SUM_COLS // 2, gl.SliceLayout(0, rl))
    at = row * HIDDEN + RANK * OWNED + t * SUM_COLS + 2 * j
    res_lo = gl.load(residual + at).to(gl.float32)
    res_hi = gl.load(residual + at + 1).to(gl.float32)
    return _sum_norm(
        mb,
        task,
        res_lo,
        res_hi,
        residual_out,
        gain,
        RANK,
        RED,
        SS,
        HIN,
        SUM,
        NORM,
        STORE,
    )


@gluon.jit
def sum_norm_regs(
    mb,
    task,
    res_lo,
    res_hi,
    residual_out,
    gain,
    RANK: gl.constexpr,
    RED: gl.constexpr,
    SS: gl.constexpr,
    HIN: gl.constexpr,
    SUM: gl.constexpr,
    NORM: gl.constexpr,
    STORE: gl.constexpr,
):
    """:func:`sum_norm_stages` with the residual from registers: the even and
    odd owned columns that an earlier sum task of this CTA returned (a CTA
    never reads another CTA's store of the same launch, so the stages that
    pass the residual on run on the same CTAs).

    Returns:
        ``(res_lo, res_hi)``: the residual after the sublayer, BF16-rounded
        (FP32), as the regular path stores it.
    """
    mb.stamp(SUM, trace.START)
    return _sum_norm(
        mb,
        task,
        res_lo,
        res_hi,
        residual_out,
        gain,
        RANK,
        RED,
        SS,
        HIN,
        SUM,
        NORM,
        STORE,
    )


@gluon.jit
def _sum_norm(
    mb,
    task,
    res_lo,
    res_hi,
    residual_out,
    gain,
    RANK: gl.constexpr,
    RED: gl.constexpr,
    SS: gl.constexpr,
    HIN: gl.constexpr,
    SUM: gl.constexpr,
    NORM: gl.constexpr,
    STORE: gl.constexpr,
):
    row, task, _ = schedule.row_task(mb.plan, task, SUM_TASKS)
    mb = mb.row(row)
    nw: gl.constexpr = gl.num_warps()
    rl: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [1, nw], [1, 0])
    src = gl.arange(0, WORLD, gl.SliceLayout(1, rl))
    j = gl.arange(0, SUM_COLS // 2, gl.SliceLayout(0, rl))
    # The parameters load before the first poll: a load after a poll would
    # start only when the poll returns.
    c = task * SUM_COLS + 2 * j  # owned column of the low half
    at = row * HIDDEN + RANK * OWNED + c
    gain_lo = gl.load(gain + c)
    gain_hi = gl.load(gain + c + 1)
    (got,) = mb.poll_many(
        (RED,), (src[:, None] * (OWNED // 2) + task * (SUM_COLS // 2) + j[None, :],)
    )
    mb.stamp(SUM, trace.READY)
    lo_all = (got << 16).to(gl.float32, bitcast=True)
    hi_all = (got & -65536).to(gl.float32, bitcast=True)
    # The rank order of the regular path's reduction: 0, then 1, ...
    lo = gl.zeros([SUM_COLS // 2], gl.float32, gl.SliceLayout(0, rl))
    hi = gl.zeros([SUM_COLS // 2], gl.float32, gl.SliceLayout(0, rl))
    for p in gl.static_range(WORLD):
        lo += gl.sum(gl.where(src[:, None] == p, lo_all, 0.0), axis=0)
        hi += gl.sum(gl.where(src[:, None] == p, hi_all, 0.0), axis=0)
    # The all-reduce returns BF16; the norm then adds the residual in FP32.
    lo = lo.to(gl.bfloat16).to(gl.float32) + res_lo
    hi = hi.to(gl.bfloat16).to(gl.float32) + res_hi
    if STORE:
        gl.store(residual_out + at, lo.to(gl.bfloat16))
        gl.store(residual_out + at + 1, hi.to(gl.bfloat16))
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [nw], [0])
    one = gl.arange(0, 1, lanes)
    total = gl.sum(lo * lo + hi * hi, axis=0)
    word = (gl.zeros([1], gl.float32, lanes) + total).to(gl.int32, bitcast=True)
    for p in gl.static_range(WORLD):
        mb.put_peer(SS, p, RANK * SUM_TASKS + task + one, word)
    mb.stamp(SUM, trace.END)

    mb.stamp(NORM, trace.START)
    (got,) = mb.poll_many((SS,), (gl.arange(0, WORLD * SUM_TASKS, lanes),))
    mb.stamp(NORM, trace.READY)
    inv = gl.rsqrt(gl.sum(got.to(gl.float32, bitcast=True), axis=0) / HIDDEN + EPS)
    h_lo = (lo * inv * gain_lo).to(gl.bfloat16)
    h_hi = (hi * inv * gain_hi).to(gl.bfloat16)
    word = (h_lo.to(gl.int16, bitcast=True).to(gl.int32) & 0xFFFF) | (
        h_hi.to(gl.int16, bitcast=True).to(gl.int32) << 16
    )
    for p in gl.static_range(WORLD):
        mb.put_peer(HIN, p, RANK * (OWNED // 2) + c // 2, word)
    mb.stamp(NORM, trace.END)
    return lo.to(gl.bfloat16).to(gl.float32), hi.to(gl.bfloat16).to(gl.float32)


@gluon.jit
def proj_issue(
    task, w, N: gl.constexpr, K: gl.constexpr, NR: gl.constexpr, KS: gl.constexpr
):
    """Issue the weight loads of task ``task`` of a projection ``x W^T``
    (``W [N, K]``): output rows ``NR (task // KP) ..``, K part ``task % KP``
    (``KP = K / KS``). The task's waves split the rows into 16-row blocks and
    the K part between them. Nothing waits here.

    Args:
        w: BF16 ``[N, K]`` in ``weights.bf16_tiles`` order.

    Returns:
        The task's weight tiles, for :func:`proj_compute`.
    """
    KP: gl.constexpr = K // KS
    NW: gl.constexpr = NR // 16
    return bf16.load_weight_tiles(
        w,
        nb0=(task // KP) * NW,
        K=K,
        kit0=(task % KP) * (KS // 32),
        KITS=KS // 32,
        SPLIT=gl.num_warps() // NW,
        NW=NW,
        GROUP=NW,
        GROUP_STRIDE=0,
    )


@gluon.jit
def proj_poll(
    mb, task, SRC: gl.constexpr, K: gl.constexpr, KS: gl.constexpr, STAGE: gl.constexpr
):
    """Poll the input rows of task ``task`` of a projection into LDS: K part
    ``task % (K / KS)`` of every row of ``SRC`` (BF16 pairs, ``K`` wide)."""
    R: gl.constexpr = schedule.rows(mb.plan)
    mb.stamp(STAGE, trace.START)
    x_smem, _ = bf16.load_rows(mb, SRC, (task % (K // KS)) * KS, KS, R)
    mb.stamp(STAGE, trace.READY)
    return x_smem


@gluon.jit
def proj_compute(
    mb,
    task,
    tiles,
    x_smem,
    DST: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    NR: gl.constexpr,
    KS: gl.constexpr,
    STAGE: gl.constexpr,
):
    """Task ``task`` of a projection from its weight tiles (:func:`proj_issue`)
    and input rows (:func:`proj_poll`): every row's FP32 partial into
    ``DST [K part][N]``."""
    R: gl.constexpr = schedule.rows(mb.plan)
    KP: gl.constexpr = K // KS
    kpart = task % KP
    group = task // KP
    out = bf16.gemv(x_smem, tiles, FIRST=0, ROWS=R)  # FP32 [RP, NR]
    mb.stamp(STAGE, trace.COMPUTED)
    RP: gl.constexpr = out.shape[0]
    bl: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [gl.num_warps(), 1], [1, 0])
    out = gl.convert_layout(out, bl)
    r = gl.arange(0, RP, gl.SliceLayout(1, bl))[:, None]
    n = gl.arange(0, NR, gl.SliceLayout(0, bl))[None, :]
    at = mb.row_offset(r) + kpart * N + group * NR + n
    mb.put(DST, at, out.to(gl.int32, bitcast=True), mask=(r < R) & (n >= 0))
    mb.stamp(STAGE, trace.END)


@gluon.jit
def proj_task(
    mb,
    task,
    w,
    SRC: gl.constexpr,
    DST: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    NR: gl.constexpr,
    KS: gl.constexpr,
    STAGE: gl.constexpr,
):
    """A projection task: :func:`proj_issue`, :func:`proj_poll`, then
    :func:`proj_compute` (``NR`` output rows, a ``KS``-wide K part)."""
    tiles = proj_issue(task, w, N, K, NR, KS)
    x_smem = proj_poll(mb, task, SRC, K, KS, STAGE)
    proj_compute(mb, task, tiles, x_smem, DST, N, K, NR, KS, STAGE)


@gluon.constexpr_function
def kq_layout(KP, Q, num_warps):
    """A ``[rows, KP, Q]`` layout: lanes across the ``Q`` values first, then
    across the K parts, a warp per row (or more rows a warp)."""
    q_t = min(Q, 64)
    k_t = min(KP, 64 // q_t)
    r_t = 64 // (q_t * k_t)
    return gl.BlockedLayout(
        [1, KP // k_t, Q // q_t], [r_t, k_t, q_t], [num_warps, 1, 1], [2, 1, 0]
    )


@gluon.jit
def coefficient(mb, row, index, CPART: gl.constexpr):
    """BF16-rounded convolution coefficient ``index`` (side x tap x group) of
    row ``row``: the sum of coef's two K parts, rounded as the regular path's
    GEMV output is. Polls ``CPART``."""
    lo, hi = mb.poll_many(
        (CPART, CPART),
        (mb.row_offset(row) + index, mb.row_offset(row) + COEFS + index),
    )
    total = lo.to(gl.float32, bitcast=True) + hi.to(gl.float32, bitcast=True)
    return total.to(gl.bfloat16).to(gl.float32)


@gluon.jit
def conv0_stage(
    mb,
    task,
    base,
    HIN: gl.constexpr,
    CPART: gl.constexpr,
    HC: gl.constexpr,
    STAGE: gl.constexpr,
):
    """Conv0 task ``task``: channels ``128 task ..`` of every row.

    Args:
        base: BF16 ``[2, 2, 8192]`` ``base_kernel`` (side, tap, channel).
    """
    R: gl.constexpr = schedule.rows(mb.plan)
    RP: gl.constexpr = triton.next_power_of_2(R)
    mb.stamp(STAGE, trace.START)
    rl: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [gl.num_warps(), 1], [1, 0])
    r = gl.arange(0, RP, gl.SliceLayout(1, rl))[:, None]
    j = gl.arange(0, CONV_COLS // 2, gl.SliceLayout(0, rl))[None, :]
    c = task * CONV_COLS + 2 * j  # channel of the low half
    b0_lo = gl.load(base + c).to(gl.float32)
    b0_hi = gl.load(base + c + 1).to(gl.float32)
    b1_lo = gl.load(base + HIDDEN + c).to(gl.float32)
    b1_hi = gl.load(base + HIDDEN + c + 1).to(gl.float32)
    rc = gl.minimum(r, R - 1)
    rp = gl.maximum(rc - 1, 0)
    g = c // GROUP_SIZE  # both halves share a group
    h_w, hp_w = mb.poll_many(
        (HIN, HIN), (mb.row_offset(rc) + c // 2, mb.row_offset(rp) + c // 2)
    )
    # Side 0's coefficients: tap 0 at g, tap 1 at GROUPS + g.
    d0 = coefficient(mb, rc, g, CPART)
    d1 = coefficient(mb, rc, GROUPS + g, CPART)
    mb.stamp(STAGE, trace.READY)
    live = (rc % BLOCK) >= 1  # tap 1 stays within the row's block
    h_lo = (h_w << 16).to(gl.float32, bitcast=True)
    h_hi = (h_w & -65536).to(gl.float32, bitcast=True)
    p_lo = (hp_w << 16).to(gl.float32, bitcast=True)
    p_hi = (hp_w & -65536).to(gl.float32, bitcast=True)
    y_lo = (b0_lo + d0) * h_lo + gl.where(live, (b1_lo + d1) * p_lo, 0.0)
    y_hi = (b0_hi + d0) * h_hi + gl.where(live, (b1_hi + d1) * p_hi, 0.0)
    word = (y_lo.to(gl.bfloat16).to(gl.int16, bitcast=True).to(gl.int32) & 0xFFFF) | (
        y_hi.to(gl.bfloat16).to(gl.int16, bitcast=True).to(gl.int32) << 16
    )
    mb.put(HC, mb.row_offset(r) + c // 2, word, mask=(r < R) & (j >= 0))
    mb.stamp(STAGE, trace.END)


@gluon.jit
def act_stage(mb, task, STAGE: gl.constexpr):
    """Act task ``task``: dimensions ``ACT_COLS task ..`` of every row."""
    R: gl.constexpr = schedule.rows(mb.plan)
    RP: gl.constexpr = triton.next_power_of_2(R)
    mb.stamp(STAGE, trace.START)
    nw: gl.constexpr = gl.num_warps()
    ll: gl.constexpr = kq_layout(GU_KPARTS, ACT_COLS // 2, nw)
    r = gl.arange(0, RP, gl.SliceLayout(1, gl.SliceLayout(2, ll)))[:, None, None]
    k = gl.arange(0, GU_KPARTS, gl.SliceLayout(0, gl.SliceLayout(2, ll)))[None, :, None]
    q = gl.arange(0, ACT_COLS // 2, gl.SliceLayout(0, gl.SliceLayout(1, ll)))[
        None, None, :
    ]
    at = mb.row_offset(gl.minimum(r, R - 1)) + k * GU + task * ACT_COLS + 2 * q
    g_lo, g_hi, u_lo, u_hi = mb.poll_many(
        ("gpart", "gpart", "gpart", "gpart"), (at, at + 1, at + INTER, at + INTER + 1)
    )
    mb.stamp(STAGE, trace.READY)
    # The regular path's GEMV rounds gate and up to BF16 before silu * mul.
    lo_g = (
        gl.sum(g_lo.to(gl.float32, bitcast=True), axis=1).to(gl.bfloat16).to(gl.float32)
    )
    hi_g = (
        gl.sum(g_hi.to(gl.float32, bitcast=True), axis=1).to(gl.bfloat16).to(gl.float32)
    )
    lo_u = (
        gl.sum(u_lo.to(gl.float32, bitcast=True), axis=1).to(gl.bfloat16).to(gl.float32)
    )
    hi_u = (
        gl.sum(u_hi.to(gl.float32, bitcast=True), axis=1).to(gl.bfloat16).to(gl.float32)
    )
    lo = (lo_g / (1.0 + gl.exp(-lo_g)) * lo_u).to(gl.bfloat16)
    hi = (hi_g / (1.0 + gl.exp(-hi_g)) * hi_u).to(gl.bfloat16)
    word = (lo.to(gl.int16, bitcast=True).to(gl.int32) & 0xFFFF) | (
        hi.to(gl.int16, bitcast=True).to(gl.int32) << 16
    )
    sl: gl.constexpr = gl.SliceLayout(1, ll)
    r2 = gl.arange(0, RP, gl.SliceLayout(1, sl))[:, None]
    q2 = gl.arange(0, ACT_COLS // 2, gl.SliceLayout(0, sl))[None, :]
    mb.put(
        "act",
        mb.row_offset(r2) + task * (ACT_COLS // 2) + q2,
        word,
        mask=(r2 < R) & (q2 >= 0),
    )
    mb.stamp(STAGE, trace.END)


@gluon.jit
def _fin_values(
    mb,
    task,
    base,
    SRC: gl.constexpr,
    KP: gl.constexpr,
    CPART: gl.constexpr,
    COLS: gl.constexpr,
    STAGE: gl.constexpr,
):
    """Fin task ``task``: columns ``COLS task ..`` of every row.

    Args:
        base: BF16 ``[2, 2, 8192]`` ``base_kernel`` (side, tap, channel).

    Returns:
        ``(r2, c2, y)``: the row and column indices of the ``[rows, COLS]``
        tile, and its FP32 values, this rank's partial of the MLP half.
    """
    R: gl.constexpr = schedule.rows(mb.plan)
    RP: gl.constexpr = triton.next_power_of_2(R)
    mb.stamp(STAGE, trace.START)
    nw: gl.constexpr = gl.num_warps()
    KQ: gl.constexpr = triton.next_power_of_2(KP)
    ll: gl.constexpr = kq_layout(KQ, COLS, nw)
    r = gl.arange(0, RP, gl.SliceLayout(1, gl.SliceLayout(2, ll)))[:, None, None]
    k = gl.arange(0, KQ, gl.SliceLayout(0, gl.SliceLayout(2, ll)))[None, :, None]
    n = gl.arange(0, COLS, gl.SliceLayout(0, gl.SliceLayout(1, ll)))[None, None, :]
    sl: gl.constexpr = gl.SliceLayout(1, ll)
    r2 = gl.arange(0, RP, gl.SliceLayout(1, sl))[:, None]
    c2 = task * COLS + gl.arange(0, COLS, gl.SliceLayout(0, sl))[None, :]
    b0 = gl.load(base + 2 * HIDDEN + c2).to(gl.float32)  # side 1, tap 0
    b1 = gl.load(base + 3 * HIDDEN + c2).to(gl.float32)  # side 1, tap 1
    rc = gl.minimum(r, R - 1)
    rp = gl.maximum(rc - 1, 0)
    col = gl.minimum(k, KP - 1) * HIDDEN + task * COLS + n
    rc2 = gl.minimum(r2, R - 1)
    g2 = c2 // GROUP_SIZE
    # Side 1's coefficients: tap 0 at 2 GROUPS + g, tap 1 at 3 GROUPS + g.
    t0 = coefficient(mb, rc2, 2 * GROUPS + g2, CPART)
    t1 = coefficient(mb, rc2, 3 * GROUPS + g2, CPART)
    cur_w, prev_w = mb.poll_many(
        (SRC, SRC), (mb.row_offset(rc) + col, mb.row_offset(rp) + col)
    )
    mb.stamp(STAGE, trace.READY)
    live_k = k < KP
    # The regular path's GEMV rounds the down projection to BF16.
    d = gl.sum(gl.where(live_k, cur_w.to(gl.float32, bitcast=True), 0.0), axis=1)
    dp = gl.sum(gl.where(live_k, prev_w.to(gl.float32, bitcast=True), 0.0), axis=1)
    d = d.to(gl.bfloat16).to(gl.float32)
    dp = dp.to(gl.bfloat16).to(gl.float32)
    live = (rc2 % BLOCK) >= 1
    y = (b0 + t0) * d + gl.where(live, (b1 + t1) * dp, 0.0)
    return r2, c2, y


@gluon.jit
def fin_stage(
    mb,
    task,
    base,
    out,
    SRC: gl.constexpr,
    KP: gl.constexpr,
    CPART: gl.constexpr,
    COLS: gl.constexpr,
    STAGE: gl.constexpr,
):
    """Fin task ``task`` (:func:`_fin_values`), stored in ``out``, BF16
    ``[rows, 8192]``: this rank's partial of the half, for a later launch."""
    R: gl.constexpr = schedule.rows(mb.plan)
    r2, c2, y = _fin_values(mb, task, base, SRC, KP, CPART, COLS, STAGE)
    gl.store(out + r2 * HIDDEN + c2, y.to(gl.bfloat16), mask=(r2 < R) & (c2 >= 0))
    mb.stamp(STAGE, trace.END)


@gluon.jit
def fin_to_owners(
    mb,
    task,
    base,
    SRC: gl.constexpr,
    KP: gl.constexpr,
    CPART: gl.constexpr,
    COLS: gl.constexpr,
    RANK: gl.constexpr,
    RED: gl.constexpr,
    STAGE: gl.constexpr,
):
    """Fin task ``task`` (:func:`_fin_values`) put into the owner rank's
    ``RED`` slot of this rank, as scat would: the next sum in the same
    launch reads it from there."""
    R: gl.constexpr = schedule.rows(mb.plan)
    RP: gl.constexpr = triton.next_power_of_2(R)
    r2, c2, y = _fin_values(mb, task, base, SRC, KP, CPART, COLS, STAGE)
    pairs: gl.constexpr = gl.BlockedLayout(
        [1, 2], [64 // (COLS // 2), COLS // 2], [gl.num_warps(), 1], [1, 0]
    )
    values = gl.convert_layout(y.to(gl.bfloat16), pairs)
    word = gl.reshape(
        mx.pack_bf16_pairs(gl.reshape(values, [RP, COLS // 2, 2])),
        [RP, COLS // 2],
    )
    wl: gl.constexpr = word.type.layout
    r = gl.arange(0, RP, gl.SliceLayout(1, wl))[:, None]
    p = gl.arange(0, COLS // 2, gl.SliceLayout(0, wl))[None, :]
    col0 = task * COLS
    at = RANK * (OWNED // 2) + (col0 % OWNED) // 2 + p
    mb.put_peer(
        RED, col0 // OWNED, mb.row_offset(r) + at, word, mask=(r < R) & (p >= 0)
    )
    mb.stamp(STAGE, trace.END)

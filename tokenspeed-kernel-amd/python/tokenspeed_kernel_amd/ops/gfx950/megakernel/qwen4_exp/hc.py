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

"""Hyper-connection mixer stages (TokenSpeed ``GatedResidualSimple.mix``).

The residual is 4 streams of ``HIDDEN`` values. A mixer turns them into one
sublayer input and 4 inject logits:

* ``y_b = bf16(h_b * rsqrt(mean(h_b**2) + eps) * (1 + w_b))`` (Gemma norm per
  stream).
* ``down = silu(y @ W_down.T)`` (``LOWRANK`` values), ``inject = y @ W_inj.T``.
* ``mixed = mean_b(sigmoid(down @ W_up_b.T) * y_b)``.

TokenSpeed folds the ``1 / 4`` projection scale into ``W_down`` and ``W_inj``.

Two stages run it:

* reduce (A1 / M1), ``20 + 8 rows`` tasks. Task ``2 p + h`` (``p < 10``) runs
  low-rank outputs ``32 p .. 32 p + 31``. These are two 16-row blocks of
  ``W_down``. The task runs them over the K half of streams ``2 h, 2 h + 1``
  and publishes the FP32 partial sums. One CU's memory path issues the 80 KiB
  of weights of a task in about 0.6 us. Task ``20 + 8 q + 4 h + r`` publishes
  row ``q``'s partial inject logit ``r`` of the same stream half. Consumers
  add the halves.
* mix (A2 / M2), 80 tasks. Task ``t`` runs the up projection of columns
  ``32 t .. 32 t + 31`` of every stream (8 blocks, one per wave). Its input is
  the low-rank activation, quantized to MXFP8. The task publishes that slice
  of ``mixed`` as MXFP8 (for the next GEMVs) and as BF16.

Every reduce task normalizes the streams that it reads by itself (the norm is
per stream). Tasks 0 and 1 also publish the inverse RMS of their half for the
mix stage. The mix stage needs only its own columns of the streams. A stream
is a ``[128, 32]`` tile with groups 80-127 masked. Thus a normalized stream
sits in LDS as a segment of 32 k-iterations, with 20 of them used. The GEMV
reads this segment in place.

The streams come from global memory or from a mailbox region of BF16 pairs.
The attention mixer reads global memory: the previous launch wrote the
streams. The MLP mixer reads a mailbox: the output projection's combine
publishes the streams. ``SRC`` names the region, or is ``""`` for global
memory.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import aggregate, gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.megakernel import mx, schedule, trace
from tokenspeed_kernel_amd.ops.gfx950.megakernel.plan import Stage
from tokenspeed_kernel_amd.ops.gfx950.megakernel.qwen4_exp.dims import (
    EPS,
    HIDDEN,
    WIDTH,
)

LOWRANK = gl.constexpr(320)
GROUPS = gl.constexpr(80)  # 32-value groups per stream
DOWN_TASKS = gl.constexpr(20)
INJ_TASKS = gl.constexpr(8)  # per row: 4 inject rows x 2 stream halves
MIX_TASKS = gl.constexpr(80)
UP_K = gl.constexpr(384)  # LOWRANK padded to whole k-iterations

# Mailbox regions of one mixer (prefix "a_" attention, "m_" MLP).
REGIONS = (
    ("lr", 640),  # y @ W_down.T over each stream half (f32), [2, 320]
    ("inj", 8),  # inject logits over each stream half (f32), [2, 4]
    ("inv", 4),  # inverse RMS of each stream (f32)
    ("x8", 640),  # MXFP8 words of the mixed row
    ("xe", 80),  # its exponents
    ("xbf", 1280),  # BF16 pairs of the mixed row
)


def regions(prefix: str) -> tuple[tuple[str, int], ...]:
    """The mailbox regions of the mixer whose region names have the prefix
    ``prefix``."""
    return tuple((prefix + name, pairs) for name, pairs in REGIONS)


def stages(
    prefix: str, bases: dict[str, int], rows: int, reads: tuple[str, ...]
) -> tuple[Stage, ...]:
    """The reduce and mix stages of the mixer whose region names have the
    prefix ``prefix``. ``bases`` names and places the stages (reduce first).
    ``reads`` is the mailbox region of the mixer's input streams, if any."""
    if len(bases) != 2:
        raise ValueError(f"a mixer has a reduce and a mix stage, got {tuple(bases)}")
    (reduce_name, reduce_base), (mix_name, mix_base) = bases.items()
    return (
        Stage(
            reduce_name,
            int(DOWN_TASKS) + int(INJ_TASKS) * rows,
            reduce_base,
            reads=reads,
            writes=(prefix + "lr", prefix + "inj", prefix + "inv"),
        ),
        Stage(
            mix_name,
            int(MIX_TASKS),
            mix_base,
            reads=(prefix + "lr", prefix + "inv") + reads,
            writes=(prefix + "x8", prefix + "xe", prefix + "xbf"),
        ),
    )


@aggregate
class Streams:
    """A launch's input streams: BF16 ``[rows, ...]`` at ``ptr`` with the
    streams ``stride`` elements apart (see :meth:`element`), or mailbox
    region ``src`` (``""``: global memory)."""

    ptr: gl.tensor | gl.constexpr
    stride: gl.tensor | gl.constexpr
    src: gl.constexpr

    @gluon.jit
    def element(self, r, e):
        """Address of element ``e`` (of ``4 * HIDDEN``) of row ``r``. Rows
        are ``3 * stride + HIDDEN`` elements apart: ``4 * HIDDEN`` for four
        streams, ``HIDDEN`` for one row that stands for all four (stride 0:
        the first layer's input, an embedding)."""
        return self.row_ptr(r) + self.offset(e)

    @gluon.jit
    def row_ptr(self, r):
        """Address of row ``r``'s element 0."""
        return self.ptr + r * (3 * self.stride + HIDDEN)

    @gluon.jit
    def offset(self, e):
        """Offset of element ``e`` from its row's element 0."""
        return (e // HIDDEN) * self.stride + e % HIDDEN

    @gluon.jit
    def row(self, r):
        """These streams as seen by row ``r`` (its row 0)."""
        return Streams(self.element(r, 0), self.stride, self.src)


@aggregate
class Mixer:
    """One mixer's weights (``weights.MixerWeights``)."""

    norm_w: gl.tensor  # raw Gemma norm weight [4 * HIDDEN] (BF16)
    w_down: gl.tensor  # W_down / 4, N16 tiles
    s_down: gl.tensor  # and scale dwords
    w_inj: gl.tensor  # W_inj / 4, BF16 [4, 4 * HIDDEN]
    w_up: gl.tensor  # W_up [4 * HIDDEN, UP_K] (K zero-padded), N16 tiles
    s_up: gl.tensor  # and scale dwords


@gluon.constexpr_function
def _stream_layout(num_warps):
    """``[streams, 128, 32]``: 8 consecutive values per lane (the scaled
    downcast needs 8), 4 lanes per 32-value group, groups over lanes and waves.
    """
    return gl.BlockedLayout([1, 1, 8], [1, 16, 4], [1, num_warps, 1], [2, 1, 0])


@gluon.jit
def _stream_index(first, NS: gl.constexpr):
    """Element indices ``[NS, 128, 32]`` of ``NS`` streams from stream element
    ``first`` (groups past 80 clamped to a valid group), and the group mask."""
    layout: gl.constexpr = _stream_layout(gl.num_warps())
    s = gl.arange(0, NS, gl.SliceLayout(1, gl.SliceLayout(2, layout)))
    g = gl.arange(0, 128, gl.SliceLayout(0, gl.SliceLayout(2, layout)))
    j = gl.arange(0, 32, gl.SliceLayout(0, gl.SliceLayout(1, layout)))
    g_valid = gl.minimum(g, GROUPS - 1)
    k = (
        first
        + s[:, None, None] * HIDDEN
        + g_valid[None, :, None] * 32
        + j[None, None, :]
    )
    return k, (g < GROUPS)[None, :, None]


@gluon.jit
def _load_streams(ptr, first, NS: gl.constexpr):
    """BF16 ``[NS, 128, 32]`` streams of ``[4 * HIDDEN]`` values at ``ptr``
    from element ``first``."""
    k, valid = _stream_index(first, NS)
    return gl.load(ptr + k, mask=valid, other=0.0)


@gluon.jit
def _load_row_streams(streams, r, first, NS: gl.constexpr):
    """:func:`_load_streams` of row ``r`` of ``streams``."""
    k, valid = _stream_index(first, NS)
    # The code adds the terms to the pointer one by one. This order compiles
    # to a faster QSA layer (by about 0.4 us) than the order of Streams.element.
    return gl.load(
        streams.row_ptr(r) + (k // HIDDEN) * streams.stride + k % HIDDEN,
        mask=valid,
        other=0.0,
    )


@gluon.jit
def _reduce_down(mb, task, streams, mixer, P: gl.constexpr, STAGE: gl.constexpr):
    """A low-rank task of :func:`reduce_stage`, for every row."""
    R: gl.constexpr = schedule.rows(mb.plan)
    # Global streams load first and the weights load right after. The norm
    # then waits for the streams only (vmcnt counts loads in issue order).
    half = task % 2
    first = half * (2 * HIDDEN)
    h_raw = ()
    if streams.src == "":
        for r in gl.static_range(R):
            h_raw += (_load_row_streams(streams, r, first, 2),)
    w_raw = _load_streams(mixer.norm_w, first, 2)
    tiles, scales = mx.load_weight_tiles(
        mixer.w_down,
        mixer.s_down,
        nb0=(task // 2) * 2,
        K=WIDTH,
        kit0=half * 40,
        KITS=40,
        SPLIT=4,
        NW=2,
        GROUP=2,
        GROUP_STRIDE=0,
        SCALES="k4",
    )
    gl.amd.hint.sched_barrier()
    RP: gl.constexpr = triton.next_power_of_2(R)
    # Rows of 32 FP8 bytes (one MXFP8 block) and one exponent per block.
    # The GEMV reads them as [RP, 64 k-iterations, 128] and [RP, 64, 4].
    flat: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    x_smem = gl.allocate_shared_memory(gl.uint8, [RP * 256, 32], flat)
    xs_smem = gl.allocate_shared_memory(
        gl.uint8, [RP * 256], gl.SwizzledSharedLayout(1, 1, 1, [0])
    )
    xs = _stream_rows(mb, h_raw, streams.src, first, R)
    mb.stamp(STAGE, trace.READY)
    normed = _normalize_rows(xs, w_raw)
    for r in gl.static_range(R):
        y, inv = normed[r]
        if task < 2:
            two = gl.arange(0, 2, inv.type.layout)
            mb.row(r).put(P + "inv", half * 2 + two, inv)
        fp8, exponent = mx.quantize_mxfp8(gl.reshape(y, [256, 32]))
        x_smem.slice(r * 256, 256).store(fp8.to(gl.uint8, bitcast=True))
        xs_smem.slice(r * 256, 256).store(exponent)
    gl.barrier()
    rows = mx.gemv(
        x_smem.reshape([RP, 64, 128]),
        xs_smem.reshape([RP, 64, 4]),
        tiles,
        scales,
        SPLIT=4,
        NW=2,
        SEG=GROUPS // 4,
        SEG_STRIDE=32,
        FIRST=0,
        ROWS=R,
    )
    mb.stamp(STAGE, trace.COMPUTED)
    r = gl.arange(0, RP, gl.SliceLayout(1, rows.type.layout))
    c = gl.arange(0, 32, gl.SliceLayout(0, rows.type.layout))
    index = mb.row_offset(r)[:, None] + (half * LOWRANK + (task // 2) * 32 + c)[None, :]
    mb.put(P + "lr", index, rows, mask=(r < R)[:, None])


@gluon.jit
def _reduce_inject(mb, task, streams, mixer, P: gl.constexpr, STAGE: gl.constexpr):
    """An inject task of :func:`reduce_stage`: one row."""
    # One row per inject task: the rows' norms run on their own
    # CTAs, not one after another.
    req, local = (task - DOWN_TASKS) // INJ_TASKS, (task - DOWN_TASKS) % INJ_TASKS
    half = local // 4
    row = local % 4
    first = half * (2 * HIDDEN)
    h_raw = ()
    if streams.src == "":
        h_raw = (_load_row_streams(streams, req, first, 2),)
    w_raw = _load_streams(mixer.norm_w, first, 2)
    wi_raw = _load_streams(mixer.w_inj + row * WIDTH, first, 2)
    gl.amd.hint.sched_barrier()
    xs = _stream_rows(mb.row(req), h_raw, streams.src, first, 1)
    mb.stamp(STAGE, trace.READY)
    y, _ = _normalize(xs[0], w_raw)
    logit = gl.sum(gl.sum(gl.sum(y * wi_raw.to(gl.float32), axis=2), axis=1), axis=0)
    _put_scalar(mb.row(req), P + "inj", half * 4 + row, logit)
    mb.stamp(STAGE, trace.COMPUTED)


@gluon.jit
def _stream_rows(mb, h_raw, SRC: gl.constexpr, first, R: gl.constexpr):
    """FP32 ``[2, 128, 32]`` stream halves of every row: the global
    streams already loaded in ``h_raw``, or polled from mailbox ``SRC`` (one
    poll per BF16 pair, the rows in shared round trips)."""
    xs = ()
    if SRC == "":
        for r in gl.static_range(R):
            xs += (h_raw[r].to(gl.float32),)
    else:
        pl: gl.constexpr = gl.BlockedLayout(
            [1, 1, 4], [1, 16, 4], [1, gl.num_warps(), 1], [2, 1, 0]
        )
        s = gl.arange(0, 2, gl.SliceLayout(1, gl.SliceLayout(2, pl)))
        g = gl.arange(0, 128, gl.SliceLayout(0, gl.SliceLayout(2, pl)))
        j = gl.arange(0, 16, gl.SliceLayout(0, gl.SliceLayout(1, pl)))
        pair = (
            first // 2
            + s[:, None, None] * (HIDDEN // 2)
            + gl.minimum(g, GROUPS - 1)[None, :, None] * 16
            + j[None, None, :]
        )
        words = mb.poll_rows((SRC,), (pair,), R)
        _, valid = _stream_index(first, 2)
        for r in gl.static_range(R):
            lo = (words[r][0] << 16).to(gl.float32, bitcast=True)
            hi = (words[r][0] & -65536).to(gl.float32, bitcast=True)
            x = gl.reshape(gl.join(lo, hi), [2, 128, 32])
            x = gl.convert_layout(x, _stream_layout(gl.num_warps()))
            xs += (gl.where(valid, x, 0.0),)
    return xs


@gluon.jit
def _normalize(x, w_raw):
    """``(y, inv)``: BF16-rounded normalized streams and per-stream inverse RMS."""
    inv = gl.rsqrt(gl.sum(gl.sum(x * x, axis=2), axis=1) * (1.0 / HIDDEN) + EPS)
    y = x * inv[:, None, None] * (1.0 + w_raw.to(gl.float32))
    return y.to(gl.bfloat16).to(gl.float32), inv


@gluon.jit
def _normalize_rows(xs, w_raw):
    """:func:`_normalize` of every row of the tuple ``xs``. The rows' sums
    meet in one reduction across waves: one LDS round trip for all rows."""
    partial = ()
    for r in gl.static_range(len(xs)):
        partial += (gl.sum(xs[r] * xs[r], axis=2),)
    if len(xs) == 1:
        sums = (gl.sum(partial[0], axis=1),)
    elif len(xs) == 2:
        sums = gl.reduce(partial, 1, _add2)
    elif len(xs) == 3:
        sums = gl.reduce(partial, 1, _add3)
    else:
        sums = gl.reduce(partial, 1, _add4)
    out = ()
    for r in gl.static_range(len(xs)):
        inv = gl.rsqrt(sums[r] * (1.0 / HIDDEN) + EPS)
        y = xs[r] * inv[:, None, None] * (1.0 + w_raw.to(gl.float32))
        out += ((y.to(gl.bfloat16).to(gl.float32), inv),)
    return out


@gluon.jit
def _add2(a0, a1, b0, b1):
    return a0 + b0, a1 + b1


@gluon.jit
def _add3(a0, a1, a2, b0, b1, b2):
    return a0 + b0, a1 + b1, a2 + b2


@gluon.jit
def _add4(a0, a1, a2, a3, b0, b1, b2, b3):
    return a0 + b0, a1 + b1, a2 + b2, a3 + b3


@gluon.jit
def _put_scalar(mb, region: gl.constexpr, index, value):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    one = gl.arange(0, 1, layout)
    mb.put(region, index + one, gl.full([1], 0.0, gl.float32, layout) + value)


@gluon.jit
def reduce_stage(mb, task, streams, mixer, P: gl.constexpr, STAGE: gl.constexpr):
    """A1 / M1 task ``task`` of the mixer whose region names have the prefix ``P``,
    for every row of the launch.

    Args:
        mb: The launch's :class:`Mailbox`.
        streams: The mixer's input :class:`Streams`.
        mixer: The mixer's :class:`Mixer` weights.
        P: Region name prefix of the mixer.
        STAGE: Plan name of this stage (for tracing).
    """
    mb.stamp(STAGE, trace.START)
    if task < DOWN_TASKS:
        _reduce_down(mb, task, streams, mixer, P, STAGE)
    else:
        _reduce_inject(mb, task, streams, mixer, P, STAGE)
    mb.stamp(STAGE, trace.END)


@gluon.jit
def mix_stage(mb, task, streams, mixer, P: gl.constexpr, out, STAGE: gl.constexpr):
    """A2 / M2 task ``task``: columns ``32 task ..`` of ``mixed``, for every
    row of the launch.

    Args:
        streams, mixer, P: As :func:`reduce_stage`.
        out: ``None``, or a BF16 ``[rows, HIDDEN]`` tensor that also receives
            ``mixed`` (the model's final mixer feeds the LM head).
    """
    SRC: gl.constexpr = streams.src
    R: gl.constexpr = schedule.rows(mb.plan)
    RP: gl.constexpr = triton.next_power_of_2(R)
    mb.stamp(STAGE, trace.START)
    # Wave v = 2 b + half reads block b * 160 + 2 task + half. Thus result
    # column n = 32 b + c is column 32 task + c of stream b. The epilogue
    # runs in the GEMV result layout, 4 columns per lane.
    tiles, scales = mx.load_weight_tiles(
        mixer.w_up,
        mixer.s_up,
        nb0=task * 2,
        K=UP_K,
        kit0=0,
        KITS=3,
        SPLIT=1,
        NW=8,
        GROUP=2,
        GROUP_STRIDE=160,
        SCALES="k4",
    )
    el: gl.constexpr = mx.gemv_template(1, 8, RP).type.layout
    r = gl.arange(0, RP, gl.SliceLayout(1, el))
    n = gl.arange(0, 128, gl.SliceLayout(0, el))
    col = (n // 32) * HIDDEN + task * 32 + n % 32
    w_col = gl.load(mixer.norm_w + col).to(gl.float32)[None, :]
    row_ok = (r < R)[:, None]
    if SRC == "":
        h_col = gl.load(
            streams.element(r[:, None], col[None, :]),
            mask=row_ok,
            other=0.0,
        ).to(gl.float32)
    gl.amd.hint.sched_barrier()

    # One poll for every row reads both stream halves of the 320 low-rank
    # sums as [rows, 2, 16 groups, 32]. It also reads the streams' inverse
    # RMS (per column). For streams in a mailbox, it also reads the task's
    # stream columns.
    ql: gl.constexpr = _lowrank_layout(gl.num_warps())
    rq = gl.arange(0, RP, gl.SliceLayout(1, gl.SliceLayout(2, gl.SliceLayout(3, ql))))
    hf = gl.arange(0, 2, gl.SliceLayout(0, gl.SliceLayout(2, gl.SliceLayout(3, ql))))
    g = gl.arange(0, 16, gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(3, ql))))
    j = gl.arange(0, 32, gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(2, ql))))
    lowrank = (
        mb.row_offset(rq)[:, None, None, None]
        + (hf * LOWRANK)[None, :, None, None]
        + (gl.minimum(g, 9) * 32)[None, None, :, None]
        + j[None, None, None, :]
    )
    columns = mb.row_offset(r)[:, None]
    indices = (lowrank, columns + (n // 32)[None, :])
    if SRC != "":
        indices = indices + (columns + (col // 2)[None, :],)
    polled = mb.poll_many(_mix_regions(P, SRC), indices)
    if SRC != "":
        h_col = mx.bf16_half(polled[2], col[None, :])
    inv = polled[1].to(gl.float32, bitcast=True)
    y = (h_col * inv * (1.0 + w_col)).to(gl.bfloat16).to(gl.float32)

    # The low-rank activation silu(sum of the halves) as MXFP8, one block per
    # group, with TokenSpeed's scale rule. With this rule, |x / 2**e| <= 448,
    # so a plain E4M3 conversion matches the scaled downcast.
    lr = mx.silu(gl.sum(polled[0].to(gl.float32, bitcast=True), axis=1))
    gg = gl.arange(0, 16, gl.SliceLayout(0, gl.SliceLayout(2, lr.type.layout)))
    used = (gg < 10)[None, :, None]
    lr = gl.where(used, lr, 0.0)
    exponent = mx.mxfp8_exponent(gl.max(gl.abs(lr), axis=2))
    exponent = gl.where((gg < 10)[None, :], exponent, 127)
    q = lr * gl.exp2(127.0 - exponent.to(gl.float32))[:, :, None]
    rows: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    x_smem = gl.allocate_shared_memory(gl.uint8, [RP * 16, 32], rows)
    xs_smem = gl.allocate_shared_memory(
        gl.uint8, [RP * 16], gl.SwizzledSharedLayout(1, 1, 1, [0])
    )
    x_smem.store(
        gl.reshape(q.to(gl.float8e4nv).to(gl.uint8, bitcast=True), [RP * 16, 32])
    )
    xs_smem.store(gl.reshape(exponent.to(gl.uint8), [RP * 16]))
    gl.barrier()
    mb.stamp(STAGE, trace.READY)
    gates = mx.gemv(
        x_smem.reshape([RP, 4, 128]),
        xs_smem.reshape([RP, 4, 4]),
        tiles,
        scales,
        SPLIT=1,
        NW=8,
        SEG=0,
        SEG_STRIDE=0,
        FIRST=0,
        ROWS=R,
    )
    mb.stamp(STAGE, trace.COMPUTED)

    # One LDS round trip regroups the 4 streams of a column into one lane.
    # The stream mean is then in-thread and lands in the MXFP8 block layout.
    weighted = gl.allocate_shared_memory(gl.float32, [RP * 4, 32], rows)
    weighted.store(gl.reshape(y * mx.sigmoid(gates), [RP * 4, 32]))
    gl.barrier()
    cl: gl.constexpr = gl.BlockedLayout(
        [1, 4, 8], [16, 1, 4], [gl.num_warps(), 1, 1], [2, 1, 0]
    )
    mixed = gl.sum(weighted.reshape([RP, 4, 32]).load(cl), axis=1) * 0.25
    mixed = mixed.to(gl.bfloat16).to(gl.float32)
    mx.put_mxfp8_rows(mb, P + "x8", P + "xe", task, mixed)
    mx.put_bf16_rows(mb, P + "xbf", task * 16, mixed)
    if out is not None:
        ol: gl.constexpr = gl.SliceLayout(1, cl)
        ro = gl.arange(0, RP, gl.SliceLayout(1, ol))
        c = gl.arange(0, 32, gl.SliceLayout(0, ol))
        gl.store(
            out + (ro * HIDDEN + task * 32)[:, None] + c[None, :],
            mixed.to(gl.bfloat16),
            mask=(ro < R)[:, None],
        )
    mb.stamp(STAGE, trace.END)


@gluon.constexpr_function
def _lowrank_layout(num_warps):
    """``[rows, 2 halves, 16 groups, 32]`` low-rank sums, both halves of a
    value in one thread."""
    return gl.BlockedLayout(
        [1, 2, 1, 1], [1, 1, 2, 32], [1, 1, num_warps, 1], [3, 2, 1, 0]
    )


@gluon.constexpr_function
def _mix_regions(P, SRC):
    """Regions the mix stage polls: the low-rank sums and inverse RMS, and
    the streams when they come from a mailbox."""
    return (P + "lr", P + "inv") if SRC == "" else (P + "lr", P + "inv", SRC)

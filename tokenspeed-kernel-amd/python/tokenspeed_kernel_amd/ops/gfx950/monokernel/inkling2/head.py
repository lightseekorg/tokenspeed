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

"""The attention head prologue of a local layer (TokenSpeed
``_rel_mha_head_norms_kernel``), one row a task group.

With ``Hq`` query and ``Hkv`` KV heads a rank (the plan's ``Dims``; a rank
of Inkling 2 Small at TP4 has 12 and 2), the projection row ``qkvr`` holds
``q [Hq, 128] | k [Hkv, 128] | v [Hkv, 128] | rel [Hq, 16] | gate [Hq, 128]
| sink [Hq] | pad`` (``Dims.k_off``, ``v_off``, ``rel_off``, ...). A task
takes a group of a row's heads, a wave a head (the regular kernel runs one
warp a head; ``plan.HEAD_SHAPES`` gives the groups a row count):

* the Q heads ``h`` of a group:
  ``q_h = bf16(clip(q_h * scale_h))`` and the head's relative logits
  ``bf16(bf16(clip_8(rel_h * rel_scale_h)) @ P)``, ``P`` BF16
  ``[16, 1024]`` (index: query-to-key distance).
* K heads ``g`` of a group: the short conv of the raw k (as ``conv.py``,
  ring channels ``128 g ..``), ``k = bf16(clip(bf16(conv)))``, the KV cache
  row and the ring row of the raw k; V heads ``g``: the short conv of the
  raw v (ring channels ``128 Hkv + 128 g ..``),
  ``v = bf16(c rsqrt(sum(c^2) / 128 + eps))`` of ``c = bf16(conv)``, the KV
  cache row and the ring row.

``clip(y) = y min(1, thr / (|y| + eps))`` with ``|y| = ss rsqrt(ss)``
(0 when ``ss = 0``). Every ``rsqrt`` is correctly rounded, as the regular
kernel's on ROCm (``torch.rsqrt``'s rounding there), not the approximate
instruction. The conv taps of earlier rows of the request in this
launch are those rows' raw k / v from their projection rows.

Bit for bit ``_rel_mha_head_norms_kernel`` as the regular prologue launches
it (``relative_head_prologue``: one warp a (row, head) program, FP
contraction off; the head counts only map its programs to heads, so the
head math is every geometry's). A head row of 128 values sits 2 adjacent
values a lane of its wave, as there, so its sums reduce in the same order,
and the relative logits come from the same 16x16x16 BF16 MFMA instruction
(one row of it a head).
"""

from __future__ import annotations

from tokenspeed_kernel_amd._extern import arithmetic_fence
from tokenspeed_kernel_amd._triton import aggregate, gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.monokernel import mx, schedule, trace
from tokenspeed_kernel_amd.ops.gfx950.monokernel.inkling2 import conv, mix
from tokenspeed_kernel_amd.ops.gfx950.monokernel.inkling2 import plan as ip

# Every Inkling2 geometry's; the head counts and the projection row's
# offsets are the plan's (schedule.dims).
DIM = gl.constexpr(128)
REL = gl.constexpr(16)
REL_EXTENT = gl.constexpr(1024)
QK_CLIP = gl.constexpr(22.627416997969522)
REL_CLIP = gl.constexpr(8.0)
EPS = gl.constexpr(1e-6)


@aggregate
class Weights:
    """A local layer's prologue weights."""

    q_scale: gl.tensor  # BF16 [Q heads, DIM]
    rel_scale: gl.tensor  # BF16 [Q heads, REL]
    rel_proj: gl.tensor  # BF16 [REL, REL_EXTENT]
    k_taps: gl.tensor  # BF16 [KV heads * DIM, 4]
    v_taps: gl.tensor  # BF16 [KV heads * DIM, 4]


@aggregate
class Cache:
    """The layer's KV cache rows and the launch's write slots."""

    k_cache: gl.tensor  # BF16 [slots, KV heads, DIM]
    v_cache: gl.tensor
    locations: gl.tensor  # int32 [rows] flat cache slot of each row's token


@gluon.jit
def _rsqrt(x):
    """FP32 ``rsqrt`` correctly rounded, as the regular kernel's
    ``_torch_rsqrt`` computes it on ROCm: through FP64."""
    return gl.extra.libdevice.rsqrt(x.to(gl.float64)).to(gl.float32)


@gluon.jit
def _clip(y, thr: gl.constexpr):
    """``clip(y)`` of each row of ``y [H, n]`` (a wave a row)."""
    # The regular kernel runs without FP contraction: the fence keeps each
    # square rounded before the sum, and the norm is a product rounded
    # before the epsilon is added (a plain product would contract with it).
    ss = gl.sum(arithmetic_fence(y * y), axis=1)
    n = gl.where(ss > 0, gl.fma(ss, _rsqrt(ss), 0.0), 0.0)
    return y * gl.minimum(1.0, gl.div_rn(thr, n + EPS))[:, None]


@gluon.constexpr_function
def _rows_layout(num_warps):
    """``[8, n]``: a wave a row (a head), the row's values over the lanes."""
    return gl.BlockedLayout([1, 1], [1, 64], [num_warps, 1], [1, 0])


@gluon.constexpr_function
def head_shape(plan):
    """``(Q heads, KV heads)`` a task of ``plan`` (``plan.HEAD_SHAPES``)."""
    return ip.HEAD_SHAPES[plan.plan.dims.hidden][plan.plan.rows]


@gluon.jit
def _q_group_task(
    mb,
    g,
    weights,
    q: gl.constexpr,
    rel: gl.constexpr,
    qkvr: gl.constexpr,
    tau,
    GROUP: gl.constexpr,
):
    """The Q heads ``GROUP g ..`` and their relative logits, a wave a head
    (the regular kernel's one-warp program for each). A full layer scales
    both, once rounded, by the row's FP32 ``tau`` (a pointer to it; None for
    a local layer) and rounds again, as the regular kernel's output scale.

    The relative logits are summed as the regular kernel sums them: a
    16x16x16 BF16 MFMA over 16-column tiles of P, the FP32 accumulator from
    zero. Row ``j`` of its A operand holds head ``GROUP g + j`` (the regular
    kernel's row 0; the other rows zero), and an MFMA output depends only
    on its own row, so the bits are the regular kernel's."""
    D: gl.constexpr = schedule.dims(mb.plan)
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 16],
        transposed=True,
        warps_per_cta=[1, gl.num_warps()],
    )
    a_layout: gl.constexpr = gl.DotOperandLayout(0, mma, 4)
    b_layout: gl.constexpr = gl.DotOperandLayout(1, mma, 4)
    layout: gl.constexpr = _rows_layout(gl.num_warps())
    # The constant operands first, the projection rows in one round trip.
    kr = gl.arange(0, REL, gl.SliceLayout(1, b_layout))
    e = gl.arange(0, REL_EXTENT, gl.SliceLayout(0, b_layout))
    proj = gl.load(weights.rel_proj + kr[:, None] * REL_EXTENT + e[None, :])
    w = gl.arange(0, 8, gl.SliceLayout(1, layout))
    # Rows past the group's heads repeat its last head and publish nothing.
    h = (g * GROUP + gl.minimum(w, GROUP - 1))[:, None]
    p = gl.arange(0, DIM // 2, gl.SliceLayout(0, layout))[None, :]
    s_lo, s_hi = mix.load_pairs(weights.q_scale + h * DIM, p)
    rl: gl.constexpr = gl.BlockedLayout([1, 1], [4, 16], [gl.num_warps(), 1], [1, 0])
    rr = gl.arange(0, 16, gl.SliceLayout(1, rl))
    rh = (g * GROUP + gl.minimum(rr, GROUP - 1))[:, None]
    kk = gl.arange(0, REL, gl.SliceLayout(0, rl))[None, :]
    rel_scale = gl.load(weights.rel_scale + rh * REL + kk).to(gl.float32)
    word, rel_word = mb.poll_many(
        (qkvr, qkvr), (h * (DIM // 2) + p, (D.rel_off + rh * REL) // 2 + kk // 2)
    )
    lo, hi = mix.unpack(word)
    y = gl.reshape(gl.join(lo * s_lo, hi * s_hi), [8, DIM])
    y = _clip(y, QK_CLIP)
    if tau is not None:
        y = y.to(gl.bfloat16).to(gl.float32) * gl.load(tau)
    y = gl.reshape(y, [8, DIM // 2, 2])
    lo, hi = gl.split(y)
    word = gl.convert_layout(mix.pack(lo, hi), layout)
    live = w[:, None] < GROUP
    mb.put(q, h * (DIM // 2) + p, word, mask=live & (p >= 0))
    r_lo, r_hi = mix.unpack(rel_word)
    raw = gl.where(kk % 2 == 0, r_lo, r_hi)
    rc = _clip(raw * rel_scale, REL_CLIP)
    if tau is not None:
        rc = rc.to(gl.bfloat16).to(gl.float32) * gl.load(tau)
    rc = gl.where((rr < GROUP)[:, None], rc, 0.0).to(gl.bfloat16)
    a = gl.convert_layout(rc, a_layout)
    acc = gl.amd.cdna4.mfma(a, proj, gl.zeros([16, REL_EXTENT], gl.float32, mma))
    # A thread holds 4 adjacent columns of a row, so its pairs form in
    # registers.
    lo, hi = gl.split(gl.reshape(acc, [16, REL_EXTENT // 2, 2]))
    rows = gl.arange(0, 16, gl.SliceLayout(1, lo.type.layout))[:, None]
    pe = gl.arange(0, REL_EXTENT // 2, gl.SliceLayout(0, lo.type.layout))[None, :]
    mb.put(
        rel,
        (g * GROUP + gl.minimum(rows, GROUP - 1)) * (REL_EXTENT // 2) + pe,
        mix.pack(lo, hi),
        mask=(rows < GROUP) & (pe >= 0),
    )


@gluon.jit
def _kv_store(
    mb,
    row,
    ring,
    cache,
    kv: gl.constexpr,
    channel,
    j,
    g,
    is_k2,
    live,
    p,
    acc_lo,
    acc_hi,
    x_lo,
    x_hi,
    SCALE: gl.constexpr,
):
    """The K or V heads of a KV task after the conv's taps: the conv output
    ``bf16((acc + x) scale)`` of the tap sums ``acc`` and the raw values
    ``x``, the ring rows of ``x``, ``k = bf16(clip(conv))`` or the normed
    ``v``, the KV cache rows and the row's ``kv`` region."""
    D: gl.constexpr = schedule.dims(mb.plan)
    layout: gl.constexpr = _rows_layout(gl.num_warps())
    w = gl.arange(0, 8, gl.SliceLayout(1, layout))
    lo = ((acc_lo + x_lo) * SCALE).to(gl.bfloat16).to(gl.float32)
    hi = ((acc_hi + x_hi) * SCALE).to(gl.bfloat16).to(gl.float32)
    slot, position = conv.ring_slot(ring, row, 0)
    at = conv.ring_offset(ring, slot, position) + channel
    gl.store(ring.ring + at, x_lo.to(gl.bfloat16), live & (slot >= 0))
    gl.store(ring.ring + at + 1, x_hi.to(gl.bfloat16), live & (slot >= 0))
    c = gl.reshape(gl.join(lo, hi), [8, DIM])
    k = gl.reshape(_clip(c, QK_CLIP), [8, DIM // 2, 2])
    k_lo, k_hi = gl.split(k)
    k_lo = gl.convert_layout(k_lo, layout)
    k_hi = gl.convert_layout(k_hi, layout)
    # Each square rounded before the sum, in the clip's layout.
    ss = gl.sum(arithmetic_fence(c * c), axis=1)
    inv = gl.convert_layout(_rsqrt(gl.div_rn(ss, DIM * 1.0) + EPS), w.type.layout)
    out_lo = gl.where(is_k2, k_lo, lo * inv[:, None])
    out_hi = gl.where(is_k2, k_hi, hi * inv[:, None])
    word = mix.pack(out_lo, out_hi)
    # int32 slots: the offsets that grow with the slot are int64.
    location = gl.load(cache.locations + row).to(gl.int64)
    at = location * (D.kv_heads * DIM // 2) + g[:, None] * (DIM // 2) + p
    k_words = cache.k_cache.to(gl.pointer_type(gl.int32), bitcast=True)
    v_words = cache.v_cache.to(gl.pointer_type(gl.int32), bitcast=True)
    gl.store(k_words + at, word, live & is_k2)
    gl.store(v_words + at, word, live & ~is_k2)
    mb.row(row).put(kv, j[:, None] * (DIM // 2) + p, word, mask=live & (p >= 0))


@gluon.jit
def _kv_task(
    mb,
    row,
    t,
    weights,
    cache,
    ring,
    kv: gl.constexpr,
    k_src: gl.constexpr,
    v_src: gl.constexpr,
    GROUP: gl.constexpr,
):
    """Heads ``GROUP t ..`` of the K heads, then the V heads (the row's
    ``kv`` region in that order) of ``row``, a wave a head (waves past the
    group's repeat its last head and store nothing): the short conv of the
    raw values (as ``conv.py``), which come from the projection row
    ``k_src`` (= ``v_src``: the one source's ``qkvr`` with k and v at their
    offsets) or from the K and V sources' own projections (``kproj`` and
    ``vproj``: k or v at 0),
    ``k = bf16(clip(bf16(conv)))`` or ``v = bf16(c rsqrt(sum(c^2) / 128 +
    eps))`` of ``c = bf16(conv)``, the KV cache rows and the ring rows of
    the raw values. The conv taps of earlier rows of the request in this
    launch are those rows' raw values from their projection rows."""
    D: gl.constexpr = schedule.dims(mb.plan)
    layout: gl.constexpr = _rows_layout(gl.num_warps())
    CHAIN: gl.constexpr = schedule.chain(mb.plan)
    tok = row % CHAIN
    w = gl.arange(0, 8, gl.SliceLayout(1, layout))
    j = t * GROUP + gl.minimum(w, GROUP - 1)
    is_k = j < D.kv_heads
    g = j % D.kv_heads
    live = (w < GROUP)[:, None]
    p = gl.arange(0, DIM // 2, gl.SliceLayout(0, layout))[None, :]
    ch = 2 * p
    # K head g: raw values at k0 + 128 g, ring channels 128 g; V head g:
    # v0 + 128 g, ring channels kv_dim + 128 g. A V source of its own sits
    # at its region's offset from the K source's (one poll for both).
    ONE: gl.constexpr = k_src == v_src
    K0: gl.constexpr = D.k_off if ONE else 0
    V0: gl.constexpr = (
        D.v_off
        if ONE
        else 2 * (schedule.region(mb.plan, v_src) - schedule.region(mb.plan, k_src))
    )
    off = (gl.where(is_k, K0, V0) + g * DIM)[:, None] // 2 + p
    channel = (gl.where(is_k, 0, D.kv_dim) + DIM * g)[:, None] + ch
    taps = (g * DIM)[:, None] * conv.TAPS + ch * conv.TAPS
    is_k2 = is_k[:, None] & (p >= 0)
    # Every tap's raw rows in one round trip: this row and the request's
    # earlier rows in this launch (else the ring).
    gl.static_assert(conv.TAPS == 4)
    offs = ()
    for i in gl.static_range(conv.TAPS):
        offs += (mb.row_offset(gl.maximum(row - (conv.TAPS - 1 - i), 0)) + off,)
    got = mb.poll_many((k_src, k_src, k_src, k_src), offs)
    acc_lo = gl.zeros([8, DIM // 2], gl.float32, layout)
    acc_hi = gl.zeros([8, DIM // 2], gl.float32, layout)
    x_lo = acc_lo
    x_hi = acc_hi
    for i in gl.static_range(conv.TAPS):
        back = conv.TAPS - 1 - i
        k_lo = gl.load(weights.k_taps + taps + i, is_k2, other=0.0)
        k_hi = gl.load(weights.k_taps + taps + conv.TAPS + i, is_k2, other=0.0)
        v_lo = gl.load(weights.v_taps + taps + i, ~is_k2, other=0.0)
        v_hi = gl.load(weights.v_taps + taps + conv.TAPS + i, ~is_k2, other=0.0)
        w_lo = gl.where(is_k2, k_lo, v_lo).to(gl.float32)
        w_hi = gl.where(is_k2, k_hi, v_hi).to(gl.float32)
        n_lo, n_hi = mix.unpack(got[i])
        if back == 0:
            x_lo = n_lo
            x_hi = n_hi
        else:
            slot, position = conv.ring_slot(ring, row, back)
            at = conv.ring_offset(ring, slot, position) + channel
            valid = (slot >= 0) & (position >= 0)
            o_lo = gl.load(ring.ring + at, valid & live, other=0.0).to(gl.float32)
            o_hi = gl.load(ring.ring + at + 1, valid & live, other=0.0).to(gl.float32)
            x_lo = gl.where(tok >= back, n_lo, o_lo)
            x_hi = gl.where(tok >= back, n_hi, o_hi)
        acc_lo = gl.fma(x_lo, w_lo, acc_lo)
        acc_hi = gl.fma(x_hi, w_hi, acc_hi)
    _kv_store(
        mb,
        row,
        ring,
        cache,
        kv,
        channel,
        j,
        g,
        is_k2,
        live,
        p,
        acc_lo,
        acc_hi,
        x_lo,
        x_hi,
        conv.OUTPUT_SCALE,
    )


@gluon.jit
def head_stage(
    mb,
    task,
    weights,
    cache,
    ring,
    qkvr: gl.constexpr,
    q: gl.constexpr,
    kv: gl.constexpr,
    rel: gl.constexpr,
    STAGE: gl.constexpr,
):
    """Prologue task ``task`` of the launch: a group of the row's Q heads,
    or of its K and V heads (``head_shape``: the heads a task)."""
    _head_task(mb, task, weights, cache, ring, qkvr, qkvr, qkvr, q, kv, rel, STAGE)


@gluon.jit
def three_source_head_stage(
    mb,
    task,
    weights,
    cache,
    ring,
    qkvr: gl.constexpr,
    kproj: gl.constexpr,
    vproj: gl.constexpr,
    q: gl.constexpr,
    kv: gl.constexpr,
    rel: gl.constexpr,
    STAGE: gl.constexpr,
):
    """:func:`head_stage` of a local layer of three sources (a looped
    core's later passes): its Q heads, relative logits, gate and sink from
    the Q source's projection ``qkvr``, its raw K and V from the K and V
    sources' projections ``kproj`` and ``vproj`` (static taps)."""
    _head_task(mb, task, weights, cache, ring, qkvr, kproj, vproj, q, kv, rel, STAGE)


@gluon.jit
def _head_task(
    mb,
    task,
    weights,
    cache,
    ring,
    qkvr: gl.constexpr,
    k_src: gl.constexpr,
    v_src: gl.constexpr,
    q: gl.constexpr,
    kv: gl.constexpr,
    rel: gl.constexpr,
    STAGE: gl.constexpr,
):
    mb.stamp(STAGE, trace.START)
    D: gl.constexpr = schedule.dims(mb.plan)
    QG: gl.constexpr = head_shape(mb.plan)[0]
    KVG: gl.constexpr = head_shape(mb.plan)[1]
    Q_TASKS: gl.constexpr = D.q_heads // QG
    TASKS: gl.constexpr = Q_TASKS + 2 * D.kv_heads // KVG
    row = task // TASKS
    t = task % TASKS
    if t < Q_TASKS:
        _q_group_task(mb.row(row), t, weights, q, rel, qkvr, None, QG)
    else:
        _kv_task(mb, row, t - Q_TASKS, weights, cache, ring, kv, k_src, v_src, KVG)
    mb.stamp(STAGE, trace.END)


# A full layer's K and V conv: 4 static taps plus the row's projected taps a
# head, (4 * 2 + 1) ** -0.5 output scale.
FULL_CONV_SCALE = gl.constexpr(0.3333333333333333)


@gluon.jit
def _full_kv_task(
    mb,
    row,
    t,
    weights,
    cache,
    ring,
    kv: gl.constexpr,
    kproj: gl.constexpr,
    vproj: gl.constexpr,
    GROUP: gl.constexpr,
):
    """:func:`_kv_task` of a full layer: the raw K head values and their
    projected taps come from the K source's projection row (``kproj``: k
    [KV heads, 128] | taps [KV heads][4]), V's from ``vproj``. The conv runs the regular
    kernel's dynamic order: lag 0 first, each coefficient ``(static + 0.0)
    + bf16(projected)`` in FP32, ``acc = x c`` then ``fma(x, c, acc)``."""
    D: gl.constexpr = schedule.dims(mb.plan)
    layout: gl.constexpr = _rows_layout(gl.num_warps())
    CHAIN: gl.constexpr = schedule.chain(mb.plan)
    tok = row % CHAIN
    w = gl.arange(0, 8, gl.SliceLayout(1, layout))
    j = t * GROUP + gl.minimum(w, GROUP - 1)
    is_k = j < D.kv_heads
    g = j % D.kv_heads
    live = (w < GROUP)[:, None]
    p = gl.arange(0, DIM // 2, gl.SliceLayout(0, layout))[None, :]
    ch = 2 * p
    off = (g * DIM)[:, None] // 2 + p
    channel = (gl.where(is_k, 0, D.kv_dim) + DIM * g)[:, None] + ch
    taps = (g * DIM)[:, None] * conv.TAPS + ch * conv.TAPS
    is_k2 = is_k[:, None] & (p >= 0)
    gl.static_assert(conv.TAPS == 4)
    # Lag i's raw rows (this row and the request's earlier rows in this
    # launch, else the ring), and this row's projected taps: K's and V's
    # regions both, each head its own.
    offs_k = ()
    offs_v = ()
    for i in gl.static_range(conv.TAPS):
        o = mb.row_offset(gl.maximum(row - i, 0)) + off
        offs_k += (o,)
        offs_v += (o,)
    got_k = mb.poll_many((kproj, kproj, kproj, kproj), offs_k)
    got_v = mb.poll_many((vproj, vproj, vproj, vproj), offs_v)
    # Taps [head][4] after the kv_dim values: pairs kv_dim / 2 + 2 g and
    # kv_dim / 2 + 2 g + 1.
    tw: gl.constexpr = gl.BlockedLayout([1, 4], [1, 64], [gl.num_warps(), 1], [1, 0])
    four = gl.arange(0, 4, gl.SliceLayout(0, tw))[None, :]
    gt = gl.convert_layout(g, gl.SliceLayout(1, tw))[:, None]
    tap_pair = mb.row_offset(row) + (D.kv_dim + gt * conv.TAPS + four) // 2
    dk, dv = mb.poll_many((kproj, vproj), (tap_pair, tap_pair))
    is_kt = gl.convert_layout(is_k, gl.SliceLayout(1, tw))[:, None]
    tap_word = gl.where(is_kt, dk, dv)
    dyn = mx.bf16_half(tap_word, gt * conv.TAPS + four)
    acc_lo = gl.zeros([8, DIM // 2], gl.float32, layout)
    acc_hi = gl.zeros([8, DIM // 2], gl.float32, layout)
    x_lo = acc_lo
    x_hi = acc_hi
    cur_lo = acc_lo
    cur_hi = acc_hi
    for i in gl.static_range(conv.TAPS):
        k_lo = gl.load(weights.k_taps + taps + i, is_k2, other=0.0)
        k_hi = gl.load(weights.k_taps + taps + conv.TAPS + i, is_k2, other=0.0)
        v_lo = gl.load(weights.v_taps + taps + i, ~is_k2, other=0.0)
        v_hi = gl.load(weights.v_taps + taps + conv.TAPS + i, ~is_k2, other=0.0)
        # The tap's projected coefficient of the wave's head.
        d_i = gl.sum(gl.where(four == i, dyn, 0.0), axis=1)
        d_i = gl.convert_layout(d_i, gl.SliceLayout(1, layout))[:, None]
        c_lo = (gl.where(is_k2, k_lo, v_lo).to(gl.float32) + 0.0) + d_i
        c_hi = (gl.where(is_k2, k_hi, v_hi).to(gl.float32) + 0.0) + d_i
        n_lo, n_hi = mix.unpack(gl.where(is_k2, got_k[i], got_v[i]))
        if i == 0:
            x_lo = n_lo
            x_hi = n_hi
            cur_lo = n_lo
            cur_hi = n_hi
        else:
            slot, position = conv.ring_slot(ring, row, i)
            at = conv.ring_offset(ring, slot, position) + channel
            valid = (slot >= 0) & (position >= 0)
            o_lo = gl.load(ring.ring + at, valid & live, other=0.0).to(gl.float32)
            o_hi = gl.load(ring.ring + at + 1, valid & live, other=0.0).to(gl.float32)
            x_lo = gl.where(tok >= i, n_lo, o_lo)
            x_hi = gl.where(tok >= i, n_hi, o_hi)
        if i == 0:
            acc_lo = x_lo * c_lo
            acc_hi = x_hi * c_hi
        else:
            acc_lo = gl.fma(x_lo, c_lo, acc_lo)
            acc_hi = gl.fma(x_hi, c_hi, acc_hi)
    _kv_store(
        mb,
        row,
        ring,
        cache,
        kv,
        channel,
        j,
        g,
        is_k2,
        live,
        p,
        acc_lo,
        acc_hi,
        cur_lo,
        cur_hi,
        FULL_CONV_SCALE,
    )


@gluon.jit
def full_head_stage(
    mb,
    task,
    weights,
    cache,
    ring,
    tau,
    qkvr: gl.constexpr,
    kproj: gl.constexpr,
    vproj: gl.constexpr,
    q: gl.constexpr,
    kv: gl.constexpr,
    rel: gl.constexpr,
    STAGE: gl.constexpr,
):
    """:func:`head_stage` of a full layer: the Q heads and relative logits
    scaled by the row's ``tau`` (FP32 ``[rows]``), the K and V heads from
    their sources' projections with projected conv taps
    (:func:`_full_kv_task`)."""
    mb.stamp(STAGE, trace.START)
    D: gl.constexpr = schedule.dims(mb.plan)
    QG: gl.constexpr = head_shape(mb.plan)[0]
    KVG: gl.constexpr = head_shape(mb.plan)[1]
    Q_TASKS: gl.constexpr = D.q_heads // QG
    TASKS: gl.constexpr = Q_TASKS + 2 * D.kv_heads // KVG
    row = task // TASKS
    t = task % TASKS
    if t < Q_TASKS:
        _q_group_task(mb.row(row), t, weights, q, rel, qkvr, tau + row, QG)
    else:
        _full_kv_task(mb, row, t - Q_TASKS, weights, cache, ring, kv, kproj, vproj, KVG)
    mb.stamp(STAGE, trace.END)

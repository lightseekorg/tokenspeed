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

"""Depth-mix stages of a mid local layer (TokenSpeed ``AttentionResidualState``
``attention_input`` / ``mlp_input`` with the set partials of the layer's
block, ``_attnres_combine_kernel``).

A layer input is a softmax mixture of the RMS-normalized completed blocks and
the current block's partial ``cand``, scored by ``q . rmsnorm(src)``. The
completed blocks enter as the set's FP32 partials ``(m, s, acc)``, which an
earlier launch computed once for every query of the set. Per row:

* attention mix: ``cand = bf16(((0 + T0) + T1) [+ T2])`` of the pending BF16
  terms that the previous layer left; ``cand`` becomes this layer's first
  pending term ``P``.
* MLP mix: ``cand = P + A`` in FP32, not rounded (``A``: the attention
  output conv).
* ``r = rsqrt(sum(cand^2) / H + eps)``, ``l = (cand . w) r``,
  ``m* = max(m, l)``, ``c = exp(m - m*)``, ``e = exp(l - m*)``,
  ``mix = (acc c + e cand r) / (s c + e)``.
* ``out = bf16(mix rsqrt(sum(mix^2) / H + eps) gain)``.

Layer 0 has no completed block: its attention input is the embedding's
RMSNorm (:func:`attention_norm_stage`), not a mix.

``w`` is the query's score weight ``q * H ** -0.5`` in FP32.

Bit for bit ``_attnres_combine_kernel`` as the regular path launches it
(``attnres_combine_queries``: one program a row, the row as two tiles of 4096
columns, the second masked past ``H``, 4 warps, FP contraction on). A row
sum is ``sum(tile 0) + sum(tile 1)``, each reduced in the regular kernel's
layout, so the in-thread, lane and warp reduction orders are its own:
tile 0 is ``[1, 4096]`` with 8 adjacent columns a thread, 64 lanes and 4
warps along the row, the other warps holding copies. The regular kernel
keeps the last 512 columns in the first warp of its second tile (+0.0
elsewhere, which adds nothing but the sign of a zero): tile 1 is ``[1, 512]``
with one warp along the row. The mailbox regions hold BF16 pairs (pair
``p``: values ``2 p`` and ``2 p + 1``), read and written in the tiles'
layouts.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import aggregate, gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.monokernel import schedule, trace

HIDDEN = gl.constexpr(4608)
EPS = gl.constexpr(1e-6)
# (first column, columns) of each tile; the regular kernel's tiles are
# next_power_of_2(H) / 2 columns wide.
PIECES = gl.constexpr((0, 4096, 4096, 512))


@aggregate
class Query:
    """One depth-mix query's operands, seen from one row (:meth:`row`)."""

    w: gl.tensor  # FP32 [HIDDEN] score weight q * HIDDEN ** -0.5
    gain: gl.tensor  # BF16 [HIDDEN] output norm gain
    acc: gl.tensor  # FP32 [rows, HIDDEN] set partial: sum of e * rmsnorm(src)
    m: gl.tensor  # FP32 [rows] set partial: running max score
    s: gl.tensor  # FP32 [rows] set partial: sum of e

    @gluon.jit
    def row(self, r):
        return Query(self.w, self.gain, self.acc + r * HIDDEN, self.m + r, self.s + r)


@gluon.constexpr_function
def _tile_layout(columns, per, num_warps):
    """``[1, columns]`` with ``per`` consecutive values a thread, 64 lanes and
    up to 4 warps along the row (the regular kernel's ``[8], [64], [4]``
    tile); the other warps hold copies."""
    along = min(columns // (per * 64), 4)
    return gl.BlockedLayout([1, per], [1, 64], [num_warps // along, along], [1, 0])


@gluon.jit
def _cols(piece: gl.constexpr):
    """Columns ``[1, n]`` of tile ``piece`` of a row."""
    first: gl.constexpr = PIECES[2 * piece]
    n: gl.constexpr = PIECES[2 * piece + 1]
    layout: gl.constexpr = _tile_layout(n, 8, gl.num_warps())
    return first + gl.arange(0, n, gl.SliceLayout(0, layout))[None, :]


@gluon.jit
def _pairs(piece: gl.constexpr):
    """Pair indices ``[1, n / 2]`` of tile ``piece`` of a row."""
    first: gl.constexpr = PIECES[2 * piece] // 2
    n: gl.constexpr = PIECES[2 * piece + 1] // 2
    layout: gl.constexpr = _tile_layout(n, 4, gl.num_warps())
    return first + gl.arange(0, n, gl.SliceLayout(0, layout))[None, :]


@gluon.jit
def unpack(word):
    """FP32 ``(lo, hi)`` of BF16 pair words (element 0 in the low half)."""
    return (word << 16).to(gl.float32, bitcast=True), (word & -65536).to(
        gl.float32, bitcast=True
    )


@gluon.jit
def pack(lo, hi):
    """BF16 pair words of FP32 ``lo`` and ``hi``, each rounded to BF16."""
    lo16 = lo.to(gl.bfloat16).to(gl.int16, bitcast=True).to(gl.int32) & 0xFFFF
    hi16 = hi.to(gl.bfloat16).to(gl.int16, bitcast=True).to(gl.int32)
    return lo16 | (hi16 << 16)


@gluon.jit
def load_pairs(x, p):
    """FP32 ``(lo, hi)`` of global row ``x`` (BF16) at pairs ``p``."""
    word = gl.load(x.to(gl.pointer_type(gl.int32), bitcast=True) + p)
    return unpack(word)


@gluon.jit
def _values(words, piece: gl.constexpr):
    """FP32 values ``[1, n]`` of tile ``piece`` from its BF16 pair words."""
    lo, hi = unpack(words)
    n: gl.constexpr = PIECES[2 * piece + 1]
    x = gl.reshape(gl.join(lo, hi), [1, n])
    return gl.convert_layout(x, _cols(piece).type.layout)


@gluon.jit
def _words(x, piece: gl.constexpr):
    """BF16 pair words ``[1, n / 2]`` of tile ``piece``'s FP32 values."""
    n: gl.constexpr = PIECES[2 * piece + 1]
    lo, hi = gl.split(gl.reshape(x, [1, n // 2, 2]))
    return gl.convert_layout(pack(lo, hi), _pairs(piece).type.layout)


@gluon.jit
def _tile_sum(x):
    return gl.sum(gl.sum(x, axis=1), axis=0)


@gluon.jit
def _row_sum(a, b):
    """``sum(a[0] * b[0]) + sum(a[1] * b[1])`` of two tiles, as the regular
    kernel sums a row."""
    return _tile_sum(a[0] * b[0]) + _tile_sum(a[1] * b[1])


@gluon.jit
def operands(q):
    """Query row ``q``'s operands of :func:`mix`, loaded before the row's
    inputs are polled: the score weight, set partial and gain tiles, and the
    partial's ``(m, s)``."""
    ws = ()
    accs = ()
    gains = ()
    for i in gl.static_range(len(PIECES) // 2):
        cols = _cols(i)
        ws += (gl.load(q.w + cols),)
        accs += (gl.load(q.acc + cols),)
        gains += (gl.load(q.gain + cols).to(gl.float32),)
    return ws, accs, gains, gl.load(q.m), gl.load(q.s)


@gluon.jit
def mix(cand, ops):
    """The mixed, normalized row ``bf16(...)`` as FP32 tiles from ``cand``
    (FP32 tiles) and a query row's :func:`operands`, in the regular kernel's
    operation order."""
    ws, accs, gains, m, s = ops
    r = gl.rsqrt(_row_sum(cand, cand) / HIDDEN + EPS)
    logit = _row_sum(cand, ws) * r
    m_new = gl.maximum(m, logit)
    c = gl.exp(m - m_new)
    e = gl.exp(logit - m_new)
    # The FMAs that the regular kernel's FP contraction forms, made explicit:
    # LLVM may contract either product of a * c + e * v, and here it picks
    # the other one for some packed elements.
    inv = 1.0 / gl.fma(s, c, e)
    mixed = ()
    for i in gl.static_range(len(cand)):
        mixed += (gl.fma(accs[i], c, e * (cand[i] * r)) * inv,)
    rs = gl.rsqrt(gl.div_rn(_row_sum(mixed, mixed), HIDDEN * 1.0) + EPS)
    out = ()
    for i in gl.static_range(len(cand)):
        out += ((mixed[i] * rs) * gains[i],)
    return out


@gluon.jit
def boundary_operands(q, mb, set_acc: gl.constexpr, set_ms: gl.constexpr):
    """:func:`operands` of a boundary layer's MLP query, whose set partial
    this launch computes (:func:`sweep_stage`): the score weight and gain
    tiles from ``q``, the partial's accumulator and ``(m, s)`` polled from
    regions ``set_acc`` (one FP32 a pair) and ``set_ms``."""
    ws = ()
    accs = ()
    gains = ()
    for i in gl.static_range(len(PIECES) // 2):
        cols = _cols(i)
        ws += (gl.load(q.w + cols),)
        gains += (gl.load(q.gain + cols).to(gl.float32),)
        accs += (mb.poll(set_acc, cols).to(gl.float32, bitcast=True),)
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    two = gl.arange(0, 2, lanes)
    ms = mb.poll(set_ms, two).to(gl.float32, bitcast=True)
    m = gl.sum(gl.where(two == 0, ms, 0.0), axis=0)
    s = gl.sum(gl.where(two == 1, ms, 0.0), axis=0)
    return ws, accs, gains, m, s


@gluon.jit
def _publish(mb, region: gl.constexpr, tiles):
    for i in gl.static_range(len(tiles)):
        mb.put(region, _pairs(i), _words(tiles[i], i))


@gluon.jit
def _attention_mix(
    mb, terms, cand_out, q, x: gl.constexpr, cand: gl.constexpr, PUBLISH: gl.constexpr
):
    """The attention mix of one query of a row: ``cand`` from ``terms``,
    published (``cand_out`` and region ``cand``) when ``PUBLISH``, then the
    mixed row into region ``x``."""
    ops = operands(q)
    pieces = ()
    for i in gl.static_range(2):
        cols = _cols(i)
        if len(terms) == 1:
            # One pending term is the candidate itself, -0.0 included (the
            # regular kernel adds no zero without a candidate to store).
            total = gl.load(terms[0] + cols).to(gl.float32)
        else:
            # Summed from +0.0: a -0.0 first term becomes +0.0, as the
            # regular path's sum does.
            total = 0.0 + gl.load(terms[0] + cols).to(gl.float32)
            for t in gl.static_range(1, len(terms)):
                total += gl.load(terms[t] + cols).to(gl.float32)
        rounded = total.to(gl.bfloat16)
        if PUBLISH:
            gl.store(cand_out + cols, rounded)
            mb.put(cand, _pairs(i), _words(total, i))
        pieces += (rounded.to(gl.float32),)
    return mix(pieces, ops)


@gluon.jit
def attention_mix_stage(
    mb,
    terms,
    cand_out,
    q,
    x: gl.constexpr,
    cand: gl.constexpr,
    STAGE: gl.constexpr,
):
    """The attention mix of the row that ``mb``, ``terms``, ``cand_out`` and
    ``q`` view.

    Args:
        terms: The previous layer's pending BF16 rows (1 to 3 global
            pointers), summed in order.
        cand_out: Global BF16 row that receives ``cand``, the next pending
            term ``P`` of the regular path's state.
        q: The attention query (:class:`Query`) row.
        x: Region of the BF16 layer input pairs.
        cand: Region of the BF16 ``cand`` pairs, for the MLP mix.
    """
    mb.stamp(STAGE, trace.START)
    out = _attention_mix(mb, terms, cand_out, q, x, cand, True)
    mb.stamp(STAGE, trace.READY)
    _publish(mb, x, out)
    mb.stamp(STAGE, trace.END)


@gluon.jit
def full_attention_mix_stage(
    mb,
    terms,
    cand_out,
    q,
    j,
    cand: gl.constexpr,
    STAGE: gl.constexpr,
):
    """Query ``j`` of a full layer's attention mix of the row that ``mb``,
    ``terms`` and ``cand_out`` view: the Q (0), K (1) or V (2) source, each
    its own query row ``q`` (:class:`Query`) and gain, into region
    ``x_attn``, ``x_k`` or ``x_v`` (as ``attnres_combine_queries`` mixes
    the three queries of one layer). Query 0 publishes ``cand``."""
    mb.stamp(STAGE, trace.START)
    if j == 0:
        out = _attention_mix(mb, terms, cand_out, q, "x_attn", cand, True)
        mb.stamp(STAGE, trace.READY)
        _publish(mb, "x_attn", out)
    elif j == 1:
        out = _attention_mix(mb, terms, cand_out, q, "x_k", cand, False)
        mb.stamp(STAGE, trace.READY)
        _publish(mb, "x_k", out)
    else:
        out = _attention_mix(mb, terms, cand_out, q, "x_v", cand, False)
        mb.stamp(STAGE, trace.READY)
        _publish(mb, "x_v", out)
    mb.stamp(STAGE, trace.END)


# Layer 0's attention input (``AttentionResidualState.attention_input`` with
# no block stored: ``attn_norm(partial)``, ``rmsnorm_precision`` as the
# regular path launches it: ``_rmsnorm_precision_kernel``, one program a row
# of 8 warps, the row as one 8192-wide tile with 8 adjacent columns a thread,
# FP contraction off).
NORM_TILE = gl.constexpr(8192)


@gluon.constexpr_function
def _norm_layout(num_warps):
    return gl.BlockedLayout([8], [64], [num_warps], [0])


@gluon.jit
def _squares(v):
    """``v * v``, each product rounded: the regular kernel sums rounded
    squares (no FP contraction), and an opaque multiply keeps the compiler
    from fusing it into the sum."""
    return gl.inline_asm_elementwise(
        "v_mul_f32 $0, $1, $1", "=v,v", [v], dtype=gl.float32, is_pure=True, pack=1
    )


@gluon.jit
def attention_norm_stage(
    mb,
    term,
    cand_out,
    gain,
    x: gl.constexpr,
    cand: gl.constexpr,
    STAGE: gl.constexpr,
):
    """Layer 0's attention input of the row that ``mb``, ``term`` and
    ``cand_out`` view: the block starts from the embedding row ``term``
    (BF16), stored as is in ``cand_out`` (its block) and region ``cand``
    (the sweep's candidate); region ``x`` gets ``bf16(rmsnorm(term)
    gain)``, ``gain`` the BF16 attention norm weight, with a correctly
    rounded ``rsqrt`` as the regular kernel's.
    """
    mb.stamp(STAGE, trace.START)
    nl: gl.constexpr = _norm_layout(gl.num_warps())
    c = gl.arange(0, NORM_TILE, nl)
    v = gl.load(term + c, mask=c < HIDDEN, other=0.0).to(gl.float32)
    total = gl.sum(_squares(v), axis=0)
    # _rmsnorm_precision_kernel's rsqrt is correctly rounded on ROCm
    # (torch.rsqrt's rounding there), computed through FP64.
    r = gl.extra.libdevice.rsqrt(
        (gl.div_rn(total, HIDDEN * 1.0) + EPS).to(gl.float64)
    ).to(gl.float32)
    mb.stamp(STAGE, trace.READY)
    out = ()
    for i in gl.static_range(2):
        cols = _cols(i)
        value = gl.load(term + cols)
        gl.store(cand_out + cols, value)
        value = value.to(gl.float32)
        mb.put(cand, _pairs(i), _words(value, i))
        out += ((value * r) * gl.load(gain + cols).to(gl.float32),)
    _publish(mb, x, out)
    mb.stamp(STAGE, trace.END)


@gluon.jit
def mlp_mix_stage(
    mb,
    q,
    cand: gl.constexpr,
    attn: gl.constexpr,
    x: gl.constexpr,
    set_acc: gl.constexpr,
    set_ms: gl.constexpr,
    BOUNDARY: gl.constexpr,
    STAGE: gl.constexpr,
):
    """The MLP mix of the row that ``mb`` and ``q`` view: ``cand = P + A`` in
    FP32 from regions ``cand`` (``P``) and ``attn`` (``A``). A boundary
    layer (``BOUNDARY``) starts a block: ``cand = A``, and the query's set
    partial comes from this launch's sweep (:func:`boundary_operands`)."""
    mb.stamp(STAGE, trace.START)
    p0 = _pairs(0)
    p1 = _pairs(1)
    if BOUNDARY:
        ops = boundary_operands(q, mb, set_acc, set_ms)
        words = mb.poll_many((attn, attn), (p0, p1))
        pieces = (_values(words[0], 0), _values(words[1], 1))
    else:
        ops = operands(q)
        # One round trip for both tiles of both regions.
        words = mb.poll_many((cand, attn, cand, attn), (p0, p0, p1, p1))
        pieces = (
            _values(words[0], 0) + _values(words[1], 0),
            _values(words[2], 1) + _values(words[3], 1),
        )
    mb.stamp(STAGE, trace.READY)
    _publish(mb, x, mix(pieces, ops))
    mb.stamp(STAGE, trace.END)


# The set partials sweep of a boundary layer (TokenSpeed
# ``attention_residual_partials``, ``_attnres_partial_kernel``): one program a
# (token, query), 8 warps, each block row as two 4096-wide tiles (the second
# masked past HIDDEN) with 8 adjacent columns a thread.
SWEEP_TILE = gl.constexpr(4096)


@gluon.constexpr_function
def _sweep_layout(num_warps):
    return gl.BlockedLayout([SWEEP_TILE // (64 * num_warps)], [64], [num_warps], [0])


@aggregate
class Sweep:
    """A boundary layer's set partials sweep operands."""

    blocks: gl.tensor  # BF16 [blocks, rows, HIDDEN] completed blocks
    count: gl.tensor  # int32 [1]: the blocks stored before this layer's
    weights: gl.tensor  # FP32 [queries, HIDDEN] score weights of the new set
    m: gl.tensor  # FP32 [queries, rows] out
    s: gl.tensor  # FP32 [queries, rows] out
    acc: gl.tensor  # FP32 [queries, rows, HIDDEN] out


@gluon.jit
def _sweep_step(v0, v1, wp0, wp1, acc0, acc1, m_run, s_run):
    """One block of the regular kernel's recurrence, in its operation order
    (FP contraction as that kernel compiles it)."""
    sq = gl.sum(v0 * v0, axis=0) + gl.sum(v1 * v1, axis=0)
    dot = gl.sum(v0 * wp0, axis=0) + gl.sum(v1 * wp1, axis=0)
    rsig = gl.rsqrt(sq / HIDDEN + EPS)
    logit = dot * rsig
    m_new = gl.maximum(m_run, logit)
    corr = gl.exp(m_run - m_new)
    wgt = gl.exp(logit - m_new)
    v0 = v0 * rsig
    v1 = v1 * rsig
    acc0 = acc0 * corr + wgt * v0
    acc1 = acc1 * corr + wgt * v1
    s_run = s_run * corr + wgt
    return acc0, acc1, m_new, s_run


@gluon.jit
def sweep_stage(
    mb,
    task,
    sweep,
    cand: gl.constexpr,
    set_acc: gl.constexpr,
    set_ms: gl.constexpr,
    QUERIES: gl.constexpr,
    FIRST: gl.constexpr,
    STAGE: gl.constexpr,
):
    """Sweep task ``task`` = (row, query ``FIRST + task % QUERIES``): the
    online-softmax partial of the query over the stored blocks and this
    launch's candidate (region ``cand``, the block this layer stores), in
    the regular kernel's order. Stores ``(m, s, acc)`` to ``sweep``'s
    outputs; query 0 (the layer's own MLP query) also publishes them in
    ``set_acc`` and ``set_ms`` for the MLP mix."""
    mb.stamp(STAGE, trace.START)
    R: gl.constexpr = schedule.rows(mb.plan)
    t = task // QUERIES
    qi = FIRST + task % QUERIES
    layout: gl.constexpr = _sweep_layout(gl.num_warps())
    col0 = gl.arange(0, SWEEP_TILE, layout)
    col1 = SWEEP_TILE + col0
    mask1 = col1 < HIDDEN
    wp0 = gl.load(sweep.weights + qi * HIDDEN + col0)
    wp1 = gl.load(sweep.weights + qi * HIDDEN + col1, mask=mask1, other=0.0)
    acc0 = gl.zeros([SWEEP_TILE], gl.float32, layout)
    acc1 = gl.zeros([SWEEP_TILE], gl.float32, layout)
    m_run = -float("inf")
    s_run = 0.0
    row_base = sweep.blocks + t.to(gl.int64) * HIDDEN
    for b in range(gl.load(sweep.count)):
        base = row_base + b.to(gl.int64) * (R * HIDDEN)
        v0 = gl.load(base + col0).to(gl.float32)
        v1 = gl.load(base + col1, mask=mask1, other=0.0).to(gl.float32)
        acc0, acc1, m_run, s_run = _sweep_step(
            v0, v1, wp0, wp1, acc0, acc1, m_run, s_run
        )
    # The last block is this layer's candidate, from this launch's mix.
    view = mb.row(t)
    pl: gl.constexpr = gl.BlockedLayout(
        [SWEEP_TILE // (128 * gl.num_warps())], [64], [gl.num_warps()], [0]
    )
    p = gl.arange(0, SWEEP_TILE // 2, pl)
    w0, w1 = view.poll_many(
        (cand, cand), (p, gl.minimum(SWEEP_TILE // 2 + p, HIDDEN // 2 - 1))
    )
    mb.stamp(STAGE, trace.READY)
    lo, hi = unpack(w0)
    v0 = gl.convert_layout(gl.reshape(gl.join(lo, hi), [SWEEP_TILE]), layout)
    lo, hi = unpack(w1)
    v1 = gl.convert_layout(gl.reshape(gl.join(lo, hi), [SWEEP_TILE]), layout)
    v1 = gl.where(mask1, v1, 0.0)
    acc0, acc1, m_run, s_run = _sweep_step(v0, v1, wp0, wp1, acc0, acc1, m_run, s_run)
    out = sweep.acc + (qi * R + t).to(gl.int64) * HIDDEN
    gl.store(out + col0, acc0)
    gl.store(out + col1, acc1, mask=mask1)
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    one = gl.arange(0, 1, lanes)
    gl.store(sweep.m + qi * R + t + one, m_run + one * 0.0)
    gl.store(sweep.s + qi * R + t + one, s_run + one * 0.0)
    if qi == 0:
        view.put(set_acc, col0, acc0.to(gl.int32, bitcast=True))
        view.put(set_acc, col1, acc1.to(gl.int32, bitcast=True), mask=mask1)
        two = gl.arange(0, 2, lanes)
        ms = gl.where(two == 0, m_run, s_run)
        view.put(set_ms, two, ms.to(gl.int32, bitcast=True))
    mb.stamp(STAGE, trace.END)

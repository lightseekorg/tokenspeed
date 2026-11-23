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
earlier launch computed once for every query of the set. Per row of width
``H`` (the plan's ``dims.hidden``):

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
(``attnres_combine_queries``: one program a row, the row as two tiles of
``B = next_power_of_2(H) / 2`` columns, the second masked past ``H``, 4
warps, FP contraction on). The regular kernel's tiles hold 8 adjacent
columns a thread, 64 lanes and 4 warps along the row (2048 columns, then
the next 2048 in each thread's second 8). A row sum is ``sum(tile 0) +
sum(tile 1)``, each reduced in that layout, so the in-thread, lane and warp
reduction orders are its own: tile 0 is ``[1, B]`` with the regular layout
(the CTA's other warps holding copies). Its second tile holds the last
``H - B`` columns in its first warps (+0.0 elsewhere, which adds nothing
but the sign of a zero): tile 1 is ``[1, n]``, ``n`` the power of two of at
least 512 columns that holds them, with ``n / 512`` warps along the row;
columns past ``H`` in it (lanes the regular tile masks too) are masked to
+0.0 and add as they do there. Small's 4608 is ``4096 + 512`` (no mask), a
2304-wide row ``2048 + 256`` of a 512-wide tile. The mailbox regions hold
BF16 pairs (pair ``p``: values ``2 p`` and ``2 p + 1``), read and written in
the tiles' layouts.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import aggregate, gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.monokernel import schedule, trace

# Inkling 2 Small's width, for callers that still index rows by it; the
# stages take the width from the plan (``schedule.dims(plan).hidden``).
HIDDEN = gl.constexpr(4608)
EPS = gl.constexpr(1e-6)


@gluon.constexpr_function
def pieces(hidden):
    """``(first column, columns, live columns)`` of each tile of a row of
    ``hidden`` columns, flat: the regular kernel's tiles are
    ``next_power_of_2(hidden) / 2`` columns wide, the second one's live
    columns held in its first warps (see the module docstring)."""
    hidden = int(hidden)
    half = triton.next_power_of_2(hidden) // 2
    rest = hidden - half
    if not 2048 < hidden <= 8192:
        # Below, the regular launch's 4-warp tile would hold copies.
        raise ValueError(f"no depth-mix tiles for a {hidden}-wide row")
    return (0, half, half, half, max(triton.next_power_of_2(rest), 512), rest)


@gluon.constexpr_function
def _masked(hidden, piece):
    """Whether tile ``piece`` of a ``hidden``-wide row holds columns past
    the row (masked to +0.0)."""
    p = pieces(hidden)
    return p[3 * int(piece) + 2] < p[3 * int(piece) + 1]


@aggregate
class Query:
    """One depth-mix query's operands, seen from one row (:meth:`row_of`)."""

    w: gl.tensor  # FP32 [H] score weight q * H ** -0.5
    gain: gl.tensor  # BF16 [H] output norm gain
    acc: gl.tensor  # FP32 [rows, H] set partial: sum of e * rmsnorm(src)
    m: gl.tensor  # FP32 [rows] set partial: running max score
    s: gl.tensor  # FP32 [rows] set partial: sum of e

    @gluon.jit
    def row_of(self, r, hidden: gl.constexpr):
        """The query seen from row ``r`` of ``hidden``-wide rows."""
        return Query(self.w, self.gain, self.acc + r * hidden, self.m + r, self.s + r)

    @gluon.jit
    def row(self, r):
        """:meth:`row_of` at Small's width (``HIDDEN``)."""
        return self.row_of(r, HIDDEN)


@gluon.constexpr_function
def _tile_layout(columns, per, num_warps):
    """``[1, columns]`` with ``per`` consecutive values a thread, 64 lanes and
    up to 4 warps along the row (the regular kernel's ``[8], [64], [4]``
    tile); the other warps hold copies."""
    along = min(columns // (per * 64), 4)
    return gl.BlockedLayout([1, per], [1, 64], [num_warps // along, along], [1, 0])


@gluon.jit
def _cols_of(piece: gl.constexpr, hidden: gl.constexpr):
    """Columns ``[1, n]`` of tile ``piece`` of a ``hidden``-wide row."""
    P: gl.constexpr = pieces(hidden)
    first: gl.constexpr = P[3 * piece]
    n: gl.constexpr = P[3 * piece + 1]
    layout: gl.constexpr = _tile_layout(n, 8, gl.num_warps())
    return first + gl.arange(0, n, gl.SliceLayout(0, layout))[None, :]


@gluon.jit
def _pairs_of(piece: gl.constexpr, hidden: gl.constexpr):
    """Pair indices ``[1, n / 2]`` of tile ``piece`` of a ``hidden``-wide
    row (past the row's pairs where the tile is masked)."""
    P: gl.constexpr = pieces(hidden)
    first: gl.constexpr = P[3 * piece] // 2
    n: gl.constexpr = P[3 * piece + 1] // 2
    layout: gl.constexpr = _tile_layout(n, 4, gl.num_warps())
    return first + gl.arange(0, n, gl.SliceLayout(0, layout))[None, :]


@gluon.jit
def _poll_pairs(piece: gl.constexpr, hidden: gl.constexpr):
    """:func:`_pairs_of` to poll: a masked tile's pairs past the row poll
    the row's last pair instead (their values are masked after)."""
    p = _pairs_of(piece, hidden)
    if _masked(hidden, piece):
        p = gl.minimum(p, hidden // 2 - 1)
    return p


@gluon.jit
def _load(ptr, cols, piece: gl.constexpr, hidden: gl.constexpr):
    """Tile ``piece`` (columns ``cols``) of global row ``ptr``, +0.0 past
    the row."""
    if _masked(hidden, piece):
        value = gl.load(ptr + cols, mask=cols < hidden, other=0.0)
    else:
        value = gl.load(ptr + cols)
    return value


@gluon.jit
def _store(ptr, cols, piece: gl.constexpr, hidden: gl.constexpr, value):
    """Store tile ``piece`` (columns ``cols``) of global row ``ptr``, its
    columns in the row."""
    if _masked(hidden, piece):
        gl.store(ptr + cols, value, mask=cols < hidden)
    else:
        gl.store(ptr + cols, value)


@gluon.jit
def _put(mb, region: gl.constexpr, piece: gl.constexpr, hidden: gl.constexpr, x):
    """Publish tile ``piece``'s FP32 values ``x`` as BF16 pairs in
    ``region`` (its pairs in the row)."""
    p = _pairs_of(piece, hidden)
    words = _words(x, piece)
    if _masked(hidden, piece):
        mb.put(region, p, words, mask=p < hidden // 2)
    else:
        mb.put(region, p, words)


@gluon.jit
def _live(x, piece: gl.constexpr, hidden: gl.constexpr):
    """Tile ``piece``'s values ``x`` with +0.0 past the row."""
    if _masked(hidden, piece):
        x = gl.where(_cols_of(piece, hidden) < hidden, x, 0.0)
    return x


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
    """FP32 values ``[1, n]`` of a tile from its ``n / 2`` BF16 pair words
    (``piece`` names the tile; the layout follows its width)."""
    lo, hi = unpack(words)
    n: gl.constexpr = 2 * words.shape[1]
    x = gl.reshape(gl.join(lo, hi), [1, n])
    layout: gl.constexpr = _tile_layout(n, 8, gl.num_warps())
    return gl.convert_layout(x, layout)


@gluon.jit
def _words(x, piece: gl.constexpr):
    """BF16 pair words ``[1, n / 2]`` of a tile's ``n`` FP32 values
    (``piece`` names the tile; the layout follows its width)."""
    n: gl.constexpr = x.shape[1]
    lo, hi = gl.split(gl.reshape(x, [1, n // 2, 2]))
    layout: gl.constexpr = _tile_layout(n // 2, 4, gl.num_warps())
    return gl.convert_layout(pack(lo, hi), layout)


@gluon.jit
def _tile_sum(x):
    return gl.sum(gl.sum(x, axis=1), axis=0)


@gluon.jit
def _row_sum(a, b):
    """``sum(a[0] * b[0]) + sum(a[1] * b[1])`` of two tiles, as the regular
    kernel sums a row."""
    return _tile_sum(a[0] * b[0]) + _tile_sum(a[1] * b[1])


@gluon.jit
def operands(q, H: gl.constexpr):
    """Query row ``q``'s operands of :func:`mix` (``H``-wide rows), loaded
    before the row's inputs are polled: the score weight, set partial and
    gain tiles (+0.0 past the row), and the partial's ``(m, s)``."""
    ws = ()
    accs = ()
    gains = ()
    for i in gl.static_range(2):
        cols = _cols_of(i, H)
        ws += (_load(q.w, cols, i, H),)
        accs += (_load(q.acc, cols, i, H),)
        gains += (_load(q.gain, cols, i, H).to(gl.float32),)
    return ws, accs, gains, gl.load(q.m), gl.load(q.s)


@gluon.jit
def mix(cand, ops, H: gl.constexpr):
    """The mixed, normalized row ``bf16(...)`` as FP32 tiles from ``cand``
    (FP32 tiles, +0.0 past the row) and a query row's :func:`operands`
    (``H``-wide rows), in the regular kernel's operation order. Columns past
    the row come out +0.0 (or NaN where the regular kernel's are)."""
    ws, accs, gains, m, s = ops
    r = gl.rsqrt(_row_sum(cand, cand) / H + EPS)
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
    rs = gl.rsqrt(gl.div_rn(_row_sum(mixed, mixed), H * 1.0) + EPS)
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
    H: gl.constexpr = schedule.dims(mb.plan).hidden
    ws = ()
    accs = ()
    gains = ()
    for i in gl.static_range(2):
        cols = _cols_of(i, H)
        ws += (_load(q.w, cols, i, H),)
        gains += (_load(q.gain, cols, i, H).to(gl.float32),)
        if _masked(H, i):
            got = mb.poll(set_acc, gl.minimum(cols, H - 1))
            accs += (_live(got.to(gl.float32, bitcast=True), i, H),)
        else:
            accs += (mb.poll(set_acc, cols).to(gl.float32, bitcast=True),)
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    two = gl.arange(0, 2, lanes)
    ms = mb.poll(set_ms, two).to(gl.float32, bitcast=True)
    m = gl.sum(gl.where(two == 0, ms, 0.0), axis=0)
    s = gl.sum(gl.where(two == 1, ms, 0.0), axis=0)
    return ws, accs, gains, m, s


@gluon.jit
def _publish(mb, region: gl.constexpr, tiles):
    H: gl.constexpr = schedule.dims(mb.plan).hidden
    for i in gl.static_range(len(tiles)):
        _put(mb, region, i, H, tiles[i])


@gluon.jit
def _attention_mix(
    mb, terms, cand_out, q, x: gl.constexpr, cand: gl.constexpr, PUBLISH: gl.constexpr
):
    """The attention mix of one query of a row: ``cand`` from ``terms``,
    published (``cand_out`` and region ``cand``) when ``PUBLISH``, then the
    mixed row into region ``x``."""
    H: gl.constexpr = schedule.dims(mb.plan).hidden
    ops = operands(q, H)
    tiles = ()
    for i in gl.static_range(2):
        cols = _cols_of(i, H)
        if len(terms) == 1:
            # One pending term is the candidate itself, -0.0 included (the
            # regular kernel adds no zero without a candidate to store).
            total = _load(terms[0], cols, i, H).to(gl.float32)
        else:
            # Summed from +0.0: a -0.0 first term becomes +0.0, as the
            # regular path's sum does.
            total = 0.0 + _load(terms[0], cols, i, H).to(gl.float32)
            for t in gl.static_range(1, len(terms)):
                total += _load(terms[t], cols, i, H).to(gl.float32)
        rounded = total.to(gl.bfloat16)
        if PUBLISH:
            _store(cand_out, cols, i, H, rounded)
            _put(mb, cand, i, H, total)
        tiles += (rounded.to(gl.float32),)
    return mix(tiles, ops, H)


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
# regular path launches it: ``_rmsnorm_precision_kernel``, one program a row,
# the row as one ``next_power_of_2(H)``-wide tile masked past ``H`` with 8
# adjacent columns a thread, 8 warps from 4096 columns and 4 below (Small's
# 4608: 8192 columns on 8 warps; 2304: 4096 on 4), FP contraction off).


@gluon.constexpr_function
def _norm_shape(hidden):
    """``(tile, warps)`` of ``triton_rmsnorm_precision``'s launch over a
    ``hidden``-wide row."""
    hidden = int(hidden)
    return triton.next_power_of_2(hidden), 4 if hidden < 4096 else 8


@gluon.constexpr_function
def _norm_layout(num_warps):
    return gl.BlockedLayout([8], [64], [num_warps], [0])


@gluon.constexpr_function
def _norm_layout_2d(warps, num_warps):
    """``[1, tile]`` with the regular kernel's ``[8], [64], [warps]`` layout
    along the row; the CTA's other warps hold copies."""
    return gl.BlockedLayout([1, 8], [1, 64], [num_warps // warps, warps], [1, 0])


@gluon.jit
def _squares(v):
    """``v * v``, each product rounded: the regular kernel sums rounded
    squares (no FP contraction), and an opaque multiply keeps the compiler
    from fusing it into the sum."""
    return gl.inline_asm_elementwise(
        "v_mul_f32 $0, $1, $1", "=v,v", [v], dtype=gl.float32, is_pure=True, pack=1
    )


@gluon.jit
def _norm_squares(term, H: gl.constexpr):
    """The sum of the rounded squares of the ``H``-wide BF16 row ``term`` in
    ``_rmsnorm_precision_kernel``'s reduction layout."""
    shape: gl.constexpr = _norm_shape(H)
    TILE: gl.constexpr = shape[0]
    WARPS: gl.constexpr = shape[1]
    if WARPS == gl.num_warps():
        nl: gl.constexpr = _norm_layout(gl.num_warps())
        c = gl.arange(0, TILE, nl)
        v = gl.load(term + c, mask=c < H, other=0.0).to(gl.float32)
        total = gl.sum(_squares(v), axis=0)
    else:
        gl.static_assert(WARPS < gl.num_warps())
        nl2: gl.constexpr = _norm_layout_2d(WARPS, gl.num_warps())
        c2 = gl.arange(0, TILE, gl.SliceLayout(0, nl2))[None, :]
        v2 = gl.load(term + c2, mask=c2 < H, other=0.0).to(gl.float32)
        total = gl.sum(gl.sum(_squares(v2), axis=1), axis=0)
    return total


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
    H: gl.constexpr = schedule.dims(mb.plan).hidden
    total = _norm_squares(term, H)
    # _rmsnorm_precision_kernel's rsqrt is correctly rounded on ROCm
    # (torch.rsqrt's rounding there), computed through FP64.
    r = gl.extra.libdevice.rsqrt((gl.div_rn(total, H * 1.0) + EPS).to(gl.float64)).to(
        gl.float32
    )
    mb.stamp(STAGE, trace.READY)
    out = ()
    for i in gl.static_range(2):
        cols = _cols_of(i, H)
        value = _load(term, cols, i, H)
        _store(cand_out, cols, i, H, value)
        value = value.to(gl.float32)
        _put(mb, cand, i, H, value)
        out += ((value * r) * _load(gain, cols, i, H).to(gl.float32),)
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
    H: gl.constexpr = schedule.dims(mb.plan).hidden
    p0 = _poll_pairs(0, H)
    p1 = _poll_pairs(1, H)
    if BOUNDARY:
        ops = boundary_operands(q, mb, set_acc, set_ms)
        words = mb.poll_many((attn, attn), (p0, p1))
        tiles = (_values(words[0], 0), _live(_values(words[1], 1), 1, H))
    else:
        ops = operands(q, H)
        # One round trip for both tiles of both regions.
        words = mb.poll_many((cand, attn, cand, attn), (p0, p0, p1, p1))
        tiles = (
            _values(words[0], 0) + _values(words[1], 0),
            _live(_values(words[2], 1) + _values(words[3], 1), 1, H),
        )
    mb.stamp(STAGE, trace.READY)
    _publish(mb, x, mix(tiles, ops, H))
    mb.stamp(STAGE, trace.END)


# The set partials sweep of a boundary layer (TokenSpeed
# ``attention_residual_partials``, ``_attnres_partial_kernel``): one program a
# (token, query), 8 warps past 2048 columns (4 below), each block row as two
# ``next_power_of_2(H) / 2``-wide tiles (the second masked past H) spread
# evenly over the threads: Small's 4096-wide tiles 8 adjacent columns a
# thread, 2304's 2048-wide ones 4.


@gluon.constexpr_function
def _sweep_shape(hidden):
    """``(tile, warps)`` of ``attnres_query_partials``'s launch over
    ``hidden``-wide rows."""
    hidden = int(hidden)
    return triton.next_power_of_2(hidden) // 2, 8 if hidden > 2048 else 4


@gluon.constexpr_function
def _sweep_layout(tile, num_warps):
    return gl.BlockedLayout([tile // (64 * num_warps)], [64], [num_warps], [0])


@aggregate
class Sweep:
    """A boundary layer's set partials sweep operands."""

    blocks: gl.tensor  # BF16 [blocks, rows, H] completed blocks
    count: gl.tensor  # int32 [1]: the blocks stored before this layer's
    weights: gl.tensor  # FP32 [queries, H] score weights of the new set
    m: gl.tensor  # FP32 [queries, rows] out
    s: gl.tensor  # FP32 [queries, rows] out
    acc: gl.tensor  # FP32 [queries, rows, H] out


@gluon.jit
def _sweep_step(v0, v1, wp0, wp1, acc0, acc1, m_run, s_run, H: gl.constexpr):
    """One block of the regular kernel's recurrence over ``H``-wide rows, in
    its operation order (FP contraction as that kernel compiles it)."""
    sq = gl.sum(v0 * v0, axis=0) + gl.sum(v1 * v1, axis=0)
    dot = gl.sum(v0 * wp0, axis=0) + gl.sum(v1 * wp1, axis=0)
    rsig = gl.rsqrt(sq / H + EPS)
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
    H: gl.constexpr = schedule.dims(mb.plan).hidden
    shape: gl.constexpr = _sweep_shape(H)
    TILE: gl.constexpr = shape[0]
    # The regular launch's warps are the CTA's: its reductions' layout.
    gl.static_assert(shape[1] == gl.num_warps())
    t = task // QUERIES
    qi = FIRST + task % QUERIES
    layout: gl.constexpr = _sweep_layout(TILE, gl.num_warps())
    col0 = gl.arange(0, TILE, layout)
    col1 = TILE + col0
    mask1 = col1 < H
    wp0 = gl.load(sweep.weights + qi * H + col0)
    wp1 = gl.load(sweep.weights + qi * H + col1, mask=mask1, other=0.0)
    acc0 = gl.zeros([TILE], gl.float32, layout)
    acc1 = gl.zeros([TILE], gl.float32, layout)
    m_run = -float("inf")
    s_run = 0.0
    row_base = sweep.blocks + t.to(gl.int64) * H
    for b in range(gl.load(sweep.count)):
        base = row_base + b.to(gl.int64) * (R * H)
        v0 = gl.load(base + col0).to(gl.float32)
        v1 = gl.load(base + col1, mask=mask1, other=0.0).to(gl.float32)
        acc0, acc1, m_run, s_run = _sweep_step(
            v0, v1, wp0, wp1, acc0, acc1, m_run, s_run, H
        )
    # The last block is this layer's candidate, from this launch's mix.
    view = mb.row(t)
    pl: gl.constexpr = gl.BlockedLayout(
        [TILE // (128 * gl.num_warps())], [64], [gl.num_warps()], [0]
    )
    p = gl.arange(0, TILE // 2, pl)
    w0, w1 = view.poll_many((cand, cand), (p, gl.minimum(TILE // 2 + p, H // 2 - 1)))
    mb.stamp(STAGE, trace.READY)
    lo, hi = unpack(w0)
    v0 = gl.convert_layout(gl.reshape(gl.join(lo, hi), [TILE]), layout)
    lo, hi = unpack(w1)
    v1 = gl.convert_layout(gl.reshape(gl.join(lo, hi), [TILE]), layout)
    v1 = gl.where(mask1, v1, 0.0)
    acc0, acc1, m_run, s_run = _sweep_step(
        v0, v1, wp0, wp1, acc0, acc1, m_run, s_run, H
    )
    out = sweep.acc + (qi * R + t).to(gl.int64) * H
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

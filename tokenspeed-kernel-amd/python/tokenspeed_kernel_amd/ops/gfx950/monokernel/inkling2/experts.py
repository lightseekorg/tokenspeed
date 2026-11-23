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

"""The NVFP4 or MXFP8 routed experts of a rank (TokenSpeed's block-scaled
MoE routes on gfx950: ``gluon_block_scaled_moe_routes_mm_gfx950`` with BF16
activations, the capped SwiGLU and ``_fold_routes_kernel``). The format is
a static parameter of every stage (``NVFP4``).

The plan's dimensions (``Dims``) give the widths: ``hidden`` H, the expert
width ``inter`` I and the rank's ``local_experts`` (Inkling 2 Small at TP4:
4608, 2304 and 96). Every rank routes every row and runs its own experts on
the routes that select them. The rows that route to one of the rank's experts, in row order,
fill that expert's tiles 16 at a time (the 16 MFMA rows of a tile): its
items, one for up to 16 rows. The GEMM tasks walk ``item x output tile`` in
rounds over the grid, the items in ascending expert id. Each row's result
depends only on its own row, so it does not depend on the rows that share
its tile (the regular expert GEMM groups a rank's routes by expert as well).

* ``ex1``: ``gate_up = bf16(acc (xa wa) / (1536 WD))``, ``acc`` the FP32 dot
  of the row's quantized BF16 input and the expert's ``w13`` (N ``2 I``, K
  H, gate and up interleaved), ``xa`` the row's amax, ``wa`` / ``WD`` the
  checkpoint's outer factors. MXFP8 has no outer factor:
  ``gate_up = bf16(acc)``.
* ``swiglu``, one task a (row, route slot) of a local expert:
  ``h = bf16(((4 tanh(g / 4) sigmoid(g)) 25 tanh(u / 25)) gamma)``
  (``gamma`` the route weight), then the quantization of ``h``
  (``quant.py``).
* ``ex2``: as ``ex1`` with ``w2`` (N H, K I) on each route's quantized ``h``.
* ``fold``: per row, the FP32 sum of its local routes' outputs in route slot
  order from +0.0, rounded once to BF16, stored to every rank's peer region
  for the expert exchange (``conv.reduce_conv_stage``).

The quantized inputs (``xq``, ``hq``) are plain words that their producers
publish with a flag. The GEMM tiles are those of the regular expert GEMM's
LDS ring (``ring_dot`` in ``moe/block_scaled.py``): 128 output columns
a CTA (16 a wave), each tile's K blocks in order through direct-to-LDS
copies, the same MFMA K order and the same outer factor, so ``ex1`` and
``ex2`` equal the regular expert GEMM bit for bit. A K that is not a whole
number of the ring's 256-wide K blocks (``w2`` at I 1152) takes the regular
GEMM's masked loop there (``RING`` off): its last block multiplies the
values past K as +0.0 activations and weights. The tiles keep the ring and
give that last block the same operands (:func:`_tail_block`), so the MFMA
chain is the masked loop's. ``swiglu`` equals the
regular SwiGLU and quantizer (``_swiglu_quantize_routes``) and ``fold`` the
regular route fold (``_fold_routes_kernel``) bit for bit as well.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import aggregate
from tokenspeed_kernel_amd._triton import cdna4_async_copy as async_copy
from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.moe import block_scaled
from tokenspeed_kernel_amd.ops.gfx950.monokernel import schedule, trace
from tokenspeed_kernel_amd.ops.gfx950.monokernel.inkling2 import mix, projection, quant

TOP_K = gl.constexpr(8)
# Bins of the local experts' route counts (local_routes): at least a rank's
# experts.
EXPERT_BINS = gl.constexpr(128)
BN = gl.constexpr(128)  # output columns of a tile: 16 a wave of 8
BK = gl.constexpr(256)  # K of a ring stage
NVFP4_DEPTH = gl.constexpr(4)  # NVFP4 ring stages: 26 KB each
MXFP8_DEPTH = gl.constexpr(2)  # MXFP8 ring stages: 42 KB each
MAX_ROWS = gl.constexpr(16)  # rows of the MFMA tile
GATE_CAP = gl.constexpr(4.0)
UP_CAP = gl.constexpr(25.0)


@aggregate
class Bank:
    """One NVFP4 expert projection of this rank's experts."""

    w: gl.tensor  # uint8 [local_experts, N, K / 2] packed E2M1, low nibble even column
    ws: gl.tensor  # float8e4nv [local_experts, N, K / 16] group scales
    wa: gl.tensor  # FP32 [local_experts] outer factor
    wd: gl.tensor  # FP32 [1] outer denominator


@aggregate
class Mxfp8Bank:
    """One MXFP8 expert projection of this rank's experts."""

    w: gl.tensor  # float8e4nv [local_experts, N, K] E4M3 values
    ws: gl.tensor  # uint8 [local_experts, N, K / 32] UE8M0 group exponents


@gluon.jit
def bank(fields, NVFP4: gl.constexpr):
    """The ``Bank`` (NVFP4) or ``Mxfp8Bank`` of a projection's ``fields``."""
    if NVFP4:
        b = Bank(*fields)
    else:
        b = Mxfp8Bank(*fields)
    return b


@gluon.constexpr_function
def _depth(NVFP4):
    """Ring stages of the format's GEMM tiles."""
    return NVFP4_DEPTH.value if NVFP4 else MXFP8_DEPTH.value


@gluon.jit
def _wide_tiles(tiles, G: gl.constexpr):
    """Of a stage's ``tiles`` (``BN`` columns each) over ``G`` programs, the
    ones that NVFP4 wide tiles (``2 BN`` columns) cover: whole waves of
    ``2 G``, and the last partial wave too when it holds more than ``G``.
    A program multiplies a wide tile in less time than two tiles but more
    than one, so a last wave that one round of tiles covers stays narrow."""
    full = tiles // (2 * G) * (2 * G)
    return gl.where(tiles - full > G, tiles, full)


@gluon.jit
def _first_narrow(tile, wide):
    """A program's first narrow tile after its wide tiles, ``tile`` the
    first wide tile index past them: the programs keep their order, so the
    narrow tiles from ``wide`` on go out like the wide ones."""
    return wide + tile - wide // 2


@gluon.jit
def _wide_rings(rings):
    """The NVFP4 ring of ``ring_stages`` (``NVFP4_DEPTH`` stages of ``BN``
    rows) as ``NVFP4_DEPTH // 2`` stages of ``2 BN`` rows: the same LDS,
    wide stage ``s`` over stages ``2 s`` and ``2 s + 1``."""
    w_ring, s_ring, x_ring = rings
    D: gl.constexpr = NVFP4_DEPTH // 2
    w = w_ring.reinterpret(gl.int32, [D, 2 * BN, BK // 8], w_ring.layout)
    sc = s_ring.reinterpret(s_ring.dtype, [D, 2 * BN, BK // 16], s_ring.layout)
    return w, sc, x_ring.slice(0, D)


@gluon.constexpr_function
def route_rows(rows):
    """Rows of :func:`local_routes`: a power of two, at least a tile's."""
    rp = 1
    while rp < max(rows, MAX_ROWS):
        rp *= 2
    return rp


@gluon.jit
def local_routes(mb, route: gl.constexpr, RANK: gl.constexpr):
    """This rank's routes of every row: ``(local, first, items)``. A launch
    computes them once, for both ``up_stage`` and ``down_stage``.

    ``local [RP, TOP_K]`` (``RP = route_rows(rows)``): the local expert (0 ..
    ``local_experts - 1``) of each route, or -1. ``first [EXPERT_BINS]``: each
    local expert's
    first item (the items of the experts below it). ``items``: the items,
    ``ceil(n / MAX_ROWS)`` an expert of ``n`` routed rows."""
    R: gl.constexpr = schedule.rows(mb.plan)
    RP: gl.constexpr = route_rows(R)
    LOCAL: gl.constexpr = schedule.dims(mb.plan).local_experts
    gl.static_assert(LOCAL <= EXPERT_BINS)
    nw: gl.constexpr = gl.num_warps()
    rl: gl.constexpr = gl.BlockedLayout([1, 1], [8, 8], [nw, 1], [1, 0])
    r = gl.arange(0, RP, gl.SliceLayout(1, rl))
    k = gl.arange(0, TOP_K, gl.SliceLayout(0, rl))
    ids = mb.poll(route, mb.row_offset(gl.minimum(r, R - 1))[:, None] + k[None, :])
    local = ids - RANK * LOCAL
    local = gl.where((local >= 0) & (local < LOCAL) & (r < R)[:, None], local, -1)
    flat = gl.reshape(local, [RP * TOP_K])
    # A row routes to an expert at most once: its routes count its rows.
    count = gl.histogram(
        flat, EXPERT_BINS, mask=flat >= 0, layout=gl.BlockedLayout([1], [64], [nw], [0])
    )
    chunks = (count + MAX_ROWS - 1) // MAX_ROWS
    return local, gl.cumsum(chunks, 0) - chunks, gl.sum(chunks, 0)


@gluon.jit
def _item_expert(first, item):
    """``(expert, chunk)`` of item ``item``: the expert whose items hold it,
    and its index among them."""
    e = gl.arange(0, EXPERT_BINS, first.type.layout)
    # The expert is the last one whose first item is at most ``item``: the
    # experts without rows share their first item with the next expert.
    below = gl.sum((first <= item).to(gl.int32), 0)
    expert = below - 1
    return expert, item - gl.sum(gl.where(e == expert, first, 0), 0)


@gluon.jit
def _tile_rows(local, expert, chunk, values):
    """The tile rows of chunk ``chunk`` of ``expert``: ``(row, k_of, ok,
    value)``, each ``[MAX_ROWS]``. Tile row ``m`` holds routed row ``MAX_ROWS
    chunk + m`` of the expert in row order (``row``, its route slot ``k_of``,
    and ``values [RP]`` of the row), where ``ok``."""
    RP: gl.constexpr = local.shape[0]
    rl: gl.constexpr = local.type.layout
    hit_k = local == expert
    k = gl.arange(0, TOP_K, gl.SliceLayout(0, rl))
    k_r = gl.sum(gl.where(hit_k, k[None, :], 0), 1)
    hit = gl.max(hit_k.to(gl.int32), 1)
    nw: gl.constexpr = gl.num_warps()
    sq: gl.constexpr = gl.BlockedLayout([1, 1], [8, 8], [1, nw], [0, 1])
    # Each row's rank among the expert's rows.
    below = gl.convert_layout(hit, gl.SliceLayout(1, sq))
    lower = gl.arange(0, RP, gl.SliceLayout(1, sq))[:, None]
    upper = gl.arange(0, RP, gl.SliceLayout(0, sq))[None, :]
    rank = gl.sum(gl.where(lower < upper, below[:, None], 0), axis=0)
    tl_: gl.constexpr = gl.BlockedLayout([1, 1], [16, 4], [1, nw], [1, 0])
    m = gl.arange(0, MAX_ROWS, gl.SliceLayout(1, tl_))[:, None]
    r = gl.arange(0, RP, gl.SliceLayout(0, tl_))[None, :]
    rank = gl.convert_layout(rank, gl.SliceLayout(0, tl_))[None, :]
    hit = gl.convert_layout(hit, gl.SliceLayout(0, tl_))[None, :]
    k_r = gl.convert_layout(k_r, gl.SliceLayout(0, tl_))[None, :]
    v = gl.convert_layout(values, gl.SliceLayout(0, tl_))[None, :]
    sel = (hit > 0) & (rank == chunk * MAX_ROWS + m)
    row = gl.sum(gl.where(sel, r, 0), 1)
    k_of = gl.sum(gl.where(sel, k_r, 0), 1)
    value = gl.sum(gl.where(sel, v, 0), 1)
    ok = gl.max(sel.to(gl.int32), 1) > 0
    return row, k_of, ok, value


@gluon.jit
def _rows_view(values, layout: gl.constexpr):
    return gl.convert_layout(values, gl.SliceLayout(1, layout))


@gluon.jit
def _finish_tile(
    mb,
    out: gl.constexpr,
    acc,
    outer_rows,
    bank,
    expert,
    pair0,
    ok,
    NVFP4: gl.constexpr,
):
    """Scale a tile's FP32 rows ``acc [RP, BN]`` by ``(xa wa) / (1536 WD)``
    (NVFP4; ``outer_rows``: each row's amax), round to BF16 and publish the
    rows where ``ok`` at pairs ``pair0[r] ..`` of region ``out``."""
    RP: gl.constexpr = acc.shape[0]
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 2], [4, 16], [4, gl.num_warps() // 4], [1, 0]
    )
    v = gl.convert_layout(acc, layout)
    if NVFP4:
        xa = _rows_view(outer_rows, layout)
        wa = gl.load(bank.wa + expert)
        factor = (xa * wa) * gl.div_rn(1.0, 1536.0 * gl.load(bank.wd))
        v = v * factor[:, None]
    lo, hi = gl.split(gl.reshape(v, [RP, BN // 2, 2]))
    word = mix.pack(lo, hi)
    pl: gl.constexpr = word.type.layout
    p = gl.arange(0, BN // 2, gl.SliceLayout(0, pl))
    at = _rows_view(pair0, pl)
    m_ok = _rows_view(ok, pl)
    mb.put(out, at[:, None] + p[None, :], word, mask=m_ok[:, None])


@gluon.jit
def _operands(
    mb,
    bank,
    local,
    first,
    values,
    x: gl.constexpr,
    xa: gl.constexpr,
    x_flag: gl.constexpr,
    item,
    column,
    UP: gl.constexpr,
    COLUMNS: gl.constexpr,
    NVFP4: gl.constexpr,
):
    """The ring operands of tile ``column`` (``COLUMNS`` columns) of item
    ``item`` of ``ex1`` (``UP``) or ``ex2``: ``(w_at, s_at, x_at, x_mask,
    expert, pair0, ok, amax)``, the last four for ``_finish_tile``. ``ex1``
    takes the rows' amaxes as ``values``; ``ex2`` polls its routes' flags
    (``x_flag``) and amaxes (``xa``)."""
    expert, chunk = _item_expert(first, item)
    row, k_of, ok, amax = _tile_rows(local, expert, chunk, values)
    D: gl.constexpr = schedule.dims(mb.plan)
    HIDDEN: gl.constexpr = D.hidden
    INTER: gl.constexpr = D.inter
    if UP:
        N: gl.constexpr = 2 * INTER
        K: gl.constexpr = HIDDEN
        # A row's xq values start at word 2 row_offset, value 4 row_offset.
        starts = 4 * mb.row_offset(row).to(gl.int64)
        pair0 = mb.row_offset(row) + k_of * INTER + column * (BN // 2)
    else:
        N: gl.constexpr = HIDDEN
        K: gl.constexpr = INTER
        own = mb.row_offset(row) + k_of
        # The tile's routes' flags; the other lanes poll the first
        # route's (tile row 0 always holds one), which is raised too.
        m = gl.arange(0, MAX_ROWS, own.type.layout)
        at = gl.where(ok, own, gl.sum(gl.where(m == 0, own, 0), 0))
        mb.poll_flags(x_flag, at)
        gl.barrier()
        amax = gl.zeros([MAX_ROWS], gl.float32, own.type.layout)
        if NVFP4:
            (bits,) = mb.poll_many((xa,), (at,))
            amax = gl.where(ok, bits.to(gl.float32, bitcast=True), 0.0)
        # Route k's h values start at word 2 row_offset + k INTER / 2.
        starts = 4 * mb.row_offset(row).to(gl.int64) + k_of * INTER
        pair0 = mb.row_offset(row) + k_of * (HIDDEN // 2) + column * (BN // 2)
    x_at, x_mask = block_scaled.ring_activations(
        _region_values(mb, x), starts, ok, MAX_ROWS, BK
    )
    w_at, s_at = block_scaled.ring_weights(
        bank.w, bank.ws, expert.to(gl.int64) * N + column * BN, K, COLUMNS, BK, NVFP4
    )
    return w_at, s_at, x_at, x_mask, expert, pair0, ok, amax


@gluon.jit
def _issue(ring, slot, block, ops, K: gl.constexpr, NVFP4: gl.constexpr):
    """Copy K block ``block`` of a tile (``ops``: its :func:`_operands`) of
    reduction width ``K`` into ring slot ``slot``. Of a partial last block
    (``K`` not a whole number of ``BK`` blocks) only the values below ``K``
    copy: no copy reads past a row (or the bank's end), and the rest of the
    slot keeps stale bytes, which :func:`_tail_block` replaces with +0.0
    after its LDS reads."""
    if K % BK == 0:
        block_scaled.ring_issue_weights(ring, slot, block, ops[0], ops[1], BK, NVFP4)
        block_scaled.ring_issue_activations(
            ring, slot, block, ops[2], ops[3], BK, ".cv"
        )
    else:
        w_ring, s_ring, x_ring = ring
        NW: gl.constexpr = gl.num_warps()
        layouts: gl.constexpr = block_scaled._ring_layouts(BN, BK, NW, NVFP4)
        if NVFP4:
            W_STEP: gl.constexpr = BK // 8  # int32 words of E2M1 pairs
            S_STEP: gl.constexpr = BK // 16
        else:
            W_STEP: gl.constexpr = BK
            S_STEP: gl.constexpr = BK // 32
        wc = block * W_STEP + gl.arange(0, W_STEP, gl.SliceLayout(0, layouts[0]))
        sc = block * S_STEP + gl.arange(0, S_STEP, gl.SliceLayout(0, layouts[1]))
        x_layout: gl.constexpr = block_scaled._ring_layouts(16 * NW, BK, NW, True)[2]
        xc = block * BK + gl.arange(0, BK, gl.SliceLayout(0, x_layout))
        # The bounds fall on whole 16-byte (4-byte for scales) lane copies.
        async_copy.global_load_to_shared(
            w_ring.index(slot),
            ops[0] + block * W_STEP,
            mask=(wc < K * W_STEP // BK)[None, :],
            cache_modifier=".cg",
        )
        async_copy.global_load_to_shared(
            s_ring.index(slot),
            ops[1] + block * S_STEP,
            mask=(sc < K * S_STEP // BK)[None, :],
            cache_modifier=".cg",
        )
        async_copy.global_load_to_shared(
            x_ring.index(slot),
            ops[2] + block * BK,
            mask=ops[3] & (xc < K)[None, :],
            cache_modifier=".cv",
        )


@gluon.jit
def _live_k(x, LIVE: gl.constexpr, AXIS: gl.constexpr):
    """``x`` with its values from ``LIVE`` on along K (axis ``AXIS``) +0.0."""
    k = gl.arange(0, x.shape[AXIS], gl.SliceLayout(1 - AXIS, x.type.layout))
    if AXIS == 0:
        keep = (k < LIVE)[:, None]
    else:
        keep = (k < LIVE)[None, :]
    return gl.where(keep, x, gl.zeros_like(x))


@gluon.jit
def _tail_operand_b(value, dot_b: gl.constexpr, LIVE: gl.constexpr):
    """The MFMA B operand ``[BK, BN]`` of upcast weights ``value [BN, BK]``,
    +0.0 from K value ``LIVE`` on."""
    return _live_k(gl.convert_layout(gl.permute(value, (1, 0)), dot_b), LIVE, 0)


@gluon.jit
def _tail_block(acc, rings, slot, LIVE: gl.constexpr, NVFP4: gl.constexpr):
    """``block_scaled.ring_block`` of a tile's partial last K block: the
    activations and weights from K value ``LIVE`` of the block on are +0.0,
    the operands the regular GEMM's masked loop multiplies there (its masked
    loads give +0.0 values, codes and scales)."""
    mfma_layout: gl.constexpr = acc.type.layout
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mfma_layout, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mfma_layout, 8)
    _, _, x_ring = rings
    words, scale = block_scaled._ring_load_weights(rings, slot, BN, BK, NVFP4)
    a = _live_k(x_ring.index(slot).load(dot_a), LIVE, 1)
    value = block_scaled._upcast_ring_weights(words, scale, BN, BK, NVFP4)
    return gl.amd.cdna4.mfma(a, _tail_operand_b(value, dot_b, LIVE), acc)


@gluon.jit
def _tail_block_pair(acc0, acc1, rings, halves, slot, LIVE: gl.constexpr):
    """``block_scaled.ring_block_pair`` of a tile's partial last K block, as
    :func:`_tail_block`."""
    mfma_layout: gl.constexpr = acc0.type.layout
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mfma_layout, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mfma_layout, 8)
    _, _, x_ring = rings
    a = _live_k(x_ring.index(slot).load(dot_a), LIVE, 1)
    words, scale = block_scaled._ring_load_weights(halves, 2 * slot, BN, BK, True)
    value = block_scaled._upcast_nvfp4_words(words, scale, BN, BK)
    acc0 = gl.amd.cdna4.mfma(a, _tail_operand_b(value, dot_b, LIVE), acc0)
    words, scale = block_scaled._ring_load_weights(halves, 2 * slot + 1, BN, BK, True)
    value = block_scaled._upcast_nvfp4_words(words, scale, BN, BK)
    acc1 = gl.amd.cdna4.mfma(a, _tail_operand_b(value, dot_b, LIVE), acc1)
    return acc0, acc1


@gluon.jit
def _run_tiles(
    mb,
    rings,
    bank,
    local,
    first,
    values,
    tile,
    end,
    x: gl.constexpr,
    xa: gl.constexpr,
    x_flag: gl.constexpr,
    out: gl.constexpr,
    task,
    STAGE: gl.constexpr,
    UP: gl.constexpr,
    WIDE: gl.constexpr,
    NVFP4: gl.constexpr,
):
    """Tiles ``tile, tile + grid, ..`` below ``end`` of ``ex1`` (``UP``) or
    ``ex2``, ``BN`` columns each, or ``2 BN`` (``WIDE``, NVFP4: through the
    ring as ``_wide_rings``, a pair of results). Returns the first tile index
    past them.

    The ring runs on across tiles: the copies of a tile's first K blocks
    issue during the previous tile's last ones, so a program's tiles stream
    back to back instead of each waiting for its first blocks. Each tile's
    K blocks still multiply in order: the same bits."""
    G: gl.constexpr = schedule.grid(mb.plan)
    # ``halves``: the ring of ``BN`` rows a stage that a wide stage pairs.
    halves = rings
    if WIDE:
        ring = _wide_rings(rings)
        COLUMNS: gl.constexpr = 2 * BN
        DEPTH: gl.constexpr = NVFP4_DEPTH // 2
    else:
        ring = rings
        COLUMNS: gl.constexpr = BN
        DEPTH: gl.constexpr = _depth(NVFP4)
    D: gl.constexpr = schedule.dims(mb.plan)
    PER_ITEM: gl.constexpr = (2 * D.inter if UP else D.hidden) // COLUMNS
    SPAN: gl.constexpr = COLUMNS // BN
    K: gl.constexpr = D.hidden if UP else D.inter
    blocks: gl.constexpr = (K + BK - 1) // BK
    # Values of the last K block below K (BK: a whole block).
    LIVE: gl.constexpr = K - (blocks - 1) * BK
    gl.static_assert(blocks >= DEPTH - 1, "a tile of at least the ring's lead")
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, gl.num_warps()],
    )
    if tile < end:
        cur = _operands(
            mb,
            bank,
            local,
            first,
            values,
            x,
            xa,
            x_flag,
            tile // PER_ITEM,
            SPAN * (tile % PER_ITEM),
            UP,
            COLUMNS,
            NVFP4,
        )
        if not UP:
            if tile == task:
                mb.stamp(STAGE, trace.READY)
        # One commit group a K block, as the steps below commit them.
        # Mailbox words of another XCD's producer: read past both caches.
        for s in gl.static_range(DEPTH - 1):
            _issue(ring, s, min(s, blocks - 1), cur, K, NVFP4)
            async_copy.commit_group()
        step = 0
        while tile < end:
            later = tile + G
            more = later < end
            # The next tile's operands (this tile's again past the last):
            # ex2 polls its routes now, before this tile's K blocks. Every
            # program runs swiglu before ex2, so the wait is as safe as it
            # was at the tile itself.
            if more:
                nxt = _operands(
                    mb,
                    bank,
                    local,
                    first,
                    values,
                    x,
                    xa,
                    x_flag,
                    later // PER_ITEM,
                    SPAN * (later % PER_ITEM),
                    UP,
                    COLUMNS,
                    NVFP4,
                )
            else:
                nxt = cur
            acc0 = gl.zeros([MAX_ROWS, BN], gl.float32, mfma_layout)
            acc1 = gl.zeros([MAX_ROWS, BN], gl.float32, mfma_layout)
            # This tile's blocks past the prologue, then the next tile's
            # first DEPTH - 1 (this tile's last again past the last tile).
            for i in range(blocks - DEPTH + 1):
                async_copy.wait_group(DEPTH - 2)
                gl.barrier()
                slot = (step + DEPTH - 1) % DEPTH
                _issue(ring, slot, i + DEPTH - 1, cur, K, NVFP4)
                async_copy.commit_group()
                if WIDE:
                    acc0, acc1 = block_scaled.ring_block_pair(
                        acc0, acc1, ring, halves, step % DEPTH, BN, BK
                    )
                else:
                    acc0 = block_scaled.ring_block(
                        acc0, ring, step % DEPTH, BN, BK, NVFP4
                    )
                step += 1
            for i in gl.static_range(DEPTH - 1):
                async_copy.wait_group(DEPTH - 2)
                gl.barrier()
                slot = (step + DEPTH - 1) % DEPTH
                block = gl.where(more, i, blocks - 1)
                _issue(ring, slot, block, nxt, K, NVFP4)
                async_copy.commit_group()
                # The last step multiplies the tile's last K block.
                if (LIVE < BK) & (i == DEPTH - 2):
                    if WIDE:
                        acc0, acc1 = _tail_block_pair(
                            acc0, acc1, ring, halves, step % DEPTH, LIVE
                        )
                    else:
                        acc0 = _tail_block(acc0, ring, step % DEPTH, LIVE, NVFP4)
                elif WIDE:
                    acc0, acc1 = block_scaled.ring_block_pair(
                        acc0, acc1, ring, halves, step % DEPTH, BN, BK
                    )
                else:
                    acc0 = block_scaled.ring_block(
                        acc0, ring, step % DEPTH, BN, BK, NVFP4
                    )
                step += 1
            _finish_tile(mb, out, acc0, cur[7], bank, cur[4], cur[5], cur[6], NVFP4)
            if WIDE:
                _finish_tile(
                    mb,
                    out,
                    acc1,
                    cur[7],
                    bank,
                    cur[4],
                    cur[5] + BN // 2,
                    cur[6],
                    NVFP4,
                )
            cur = nxt
            tile = later
        # The ring is free again for the stage's next tiles.
        async_copy.wait_group(0)
        gl.barrier()
    return tile


@gluon.jit
def _region_values(mb, region: gl.constexpr):
    """BF16 base pointer of this rank's copy of ``region`` (2 values a word)."""
    return mb._words(region).to(gl.pointer_type(gl.bfloat16), bitcast=True)


@gluon.jit
def up_stage(
    mb,
    task,
    bank,
    routes,
    xq: gl.constexpr,
    xa: gl.constexpr,
    xq_flag: gl.constexpr,
    gate_up: gl.constexpr,
    STAGE: gl.constexpr,
    NVFP4: gl.constexpr,
):
    """``ex1`` task ``task``: tiles ``task, task + grid, ..`` of ``item x (2
    inter / BN)``. A tile multiplies its item's rows. ``routes``: the
    launch's ``local_routes``. ``xa``: the rows' amax region (NVFP4), else
    None."""
    mb.stamp(STAGE, trace.START)
    G: gl.constexpr = schedule.grid(mb.plan)
    N: gl.constexpr = 2 * schedule.dims(mb.plan).inter
    TILES: gl.constexpr = N // BN
    local, first, items = routes
    if task < items * TILES:
        rings = block_scaled.ring_stages(
            bank.ws, MAX_ROWS, BN, BK, _depth(NVFP4), NVFP4
        )
        RP: gl.constexpr = local.shape[0]
        rows = gl.arange(0, RP, gl.BlockedLayout([1], [64], [gl.num_warps()], [0]))
        mb.poll_flags(xq_flag, mb.row_offset(rows))
        gl.barrier()
        if NVFP4:
            # Every row's amax once (the quant stage puts it before its flag).
            (amaxes,) = mb.poll_many((xa,), (mb.row_offset(rows),))
            amaxes = amaxes.to(gl.float32, bitcast=True)
        else:
            amaxes = gl.zeros([RP], gl.float32, rows.type.layout)
        mb.stamp(STAGE, trace.READY)
        tile = task
        if NVFP4:
            wide = _wide_tiles(items * TILES, G)
            tile = _run_tiles(
                mb,
                rings,
                bank,
                local,
                first,
                amaxes,
                tile,
                wide // 2,
                xq,
                None,
                None,
                gate_up,
                task,
                STAGE,
                True,
                True,
                NVFP4,
            )
            tile = _first_narrow(tile, wide)
        _run_tiles(
            mb,
            rings,
            bank,
            local,
            first,
            amaxes,
            tile,
            items * TILES,
            xq,
            None,
            None,
            gate_up,
            task,
            STAGE,
            True,
            False,
            NVFP4,
        )
    mb.stamp(STAGE, trace.END)


@gluon.jit
def swiglu_stage(
    mb,
    task,
    route: gl.constexpr,
    gate_up: gl.constexpr,
    hq: gl.constexpr,
    ha: gl.constexpr,
    hq_flag: gl.constexpr,
    RANK: gl.constexpr,
    STAGE: gl.constexpr,
    NVFP4: gl.constexpr,
):
    """``swiglu`` task ``task`` = (row, route slot ``k``), when the route's
    expert is local: the capped SwiGLU times the route weight, then the NVFP4
    or MXFP8 quantization of ``h``, published as words with a flag (and the
    NVFP4 amax in ``ha``; None for MXFP8)."""
    mb.stamp(STAGE, trace.START)
    D: gl.constexpr = schedule.dims(mb.plan)
    LOCAL: gl.constexpr = D.local_experts
    INTER: gl.constexpr = D.inter
    row = task // TOP_K
    k = task % TOP_K
    view = mb.row(row)
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    one = gl.arange(0, 1, lanes)
    got = view.poll(route, k + one * 0)
    expert = gl.sum(got, 0) - RANK * LOCAL
    if (expert >= 0) & (expert < LOCAL):
        gamma = gl.sum(
            view.poll(route, TOP_K + k + one * 0).to(gl.float32, bitcast=True), 0
        )
        layout: gl.constexpr = gl.BlockedLayout(
            [1, 8], [64, 1], [gl.num_warps(), 1], [1, 0]
        )
        # The route's groups of 16 h values (144 at inter 2304), padded to a
        # power of two with +0.0 values that are never stored (the regular
        # quantizer's masked columns past inter).
        GROUPS: gl.constexpr = INTER // quant.GROUP
        GP: gl.constexpr = triton.next_power_of_2(GROUPS)
        g = gl.arange(0, GP, gl.SliceLayout(1, layout))
        c = gl.arange(0, 8, gl.SliceLayout(0, layout))
        live = (g < GROUPS)[:, None]
        # Pair j holds gate_j and up_j (the interleaved w13 columns).
        j = gl.minimum(g, GROUPS - 1)[:, None] * 16 + c[None, :] * 2
        words = view.poll_many((gate_up, gate_up), (k * INTER + j, k * INTER + j + 1))
        mb.stamp(STAGE, trace.READY)
        h = ()
        for w in gl.static_range(2):
            gate, up = mix.unpack(words[w])
            sg = 1.0 / (1.0 + gl.exp(-gate))
            gate = GATE_CAP * gl.extra.libdevice.tanh(gate * (1.0 / GATE_CAP))
            up = UP_CAP * gl.extra.libdevice.tanh(gl.div_rn(up, UP_CAP))
            h += (((gate * sg) * up * gamma).to(gl.bfloat16).to(gl.float32),)
        h_lo = gl.where(live, h[0], 0.0)
        h_hi = gl.where(live, h[1], 0.0)
        values = gl.reshape(gl.join(h_lo, h_hi), [GP, 16])
        if NVFP4:
            amax = gl.max(gl.max(gl.abs(values), axis=1), axis=0)
            enc = gl.minimum(gl.div_rn(1536.0, amax), quant.FLT_MAX)
            dec = gl.div_rn(1.0, enc)
            outer = amax * (1.0 / 1536.0)
            q = quant.quantize_groups(values, enc, dec, outer)
        else:
            # Two 16-value rows are one MXFP8 group.
            q = quant.mxfp8_groups(gl.reshape(values, [GP // 2, quant.MX_GROUP]))
        q_lo, q_hi = gl.split(gl.reshape(q, [GP, 8, 2]))
        word = gl.convert_layout(mix.pack(q_lo, q_hi), layout)
        pair = gl.minimum(g, GROUPS - 1)[:, None] * 8 + c[None, :]
        # Words of route k of the row: INTER / 2 words a route.
        view.store_words(hq, k * (INTER // 2) + pair, word, mask=live)
        if NVFP4:
            view.put(ha, k + one * 0, amax.to(gl.int32, bitcast=True) + one * 0)
        view.raise_flag(hq_flag, k)
    mb.stamp(STAGE, trace.END)


@gluon.jit
def down_stage(
    mb,
    task,
    bank,
    routes,
    hq: gl.constexpr,
    ha: gl.constexpr,
    hq_flag: gl.constexpr,
    out: gl.constexpr,
    STAGE: gl.constexpr,
    NVFP4: gl.constexpr,
):
    """``ex2`` task ``task``: tiles ``task, task + grid, ..`` of ``item x
    (hidden / BN)`` of ``w2`` on the routes' quantized ``h``; publishes each
    route's BF16 output. ``routes``: the launch's ``local_routes``. ``ha``:
    the routes' amax region (NVFP4), else None."""
    mb.stamp(STAGE, trace.START)
    G: gl.constexpr = schedule.grid(mb.plan)
    N: gl.constexpr = schedule.dims(mb.plan).hidden
    TILES: gl.constexpr = N // BN
    local, first, items = routes
    if task < items * TILES:
        rings = block_scaled.ring_stages(
            bank.ws, MAX_ROWS, BN, BK, _depth(NVFP4), NVFP4
        )
        nothing = gl.zeros(
            [local.shape[0]], gl.int32, gl.SliceLayout(1, local.type.layout)
        )
        tile = task
        if NVFP4:
            wide = _wide_tiles(items * TILES, G)
            tile = _run_tiles(
                mb,
                rings,
                bank,
                local,
                first,
                nothing,
                tile,
                wide // 2,
                hq,
                ha,
                hq_flag,
                out,
                task,
                STAGE,
                False,
                True,
                NVFP4,
            )
            tile = _first_narrow(tile, wide)
        _run_tiles(
            mb,
            rings,
            bank,
            local,
            first,
            nothing,
            tile,
            items * TILES,
            hq,
            ha,
            hq_flag,
            out,
            task,
            STAGE,
            False,
            False,
            NVFP4,
        )
    mb.stamp(STAGE, trace.END)


@gluon.jit
def fold_stage(
    mb,
    task,
    route: gl.constexpr,
    out: gl.constexpr,
    part: gl.constexpr,
    flag: gl.constexpr,
    COLUMNS: gl.constexpr,
    RANK: gl.constexpr,
    STAGE: gl.constexpr,
):
    """``fold`` task ``task``: columns ``COLUMNS * task ..`` of every row, the
    FP32 slot-order sum of the row's local routes from +0.0, rounded once,
    stored as words to every rank's peer region ``part`` (``[source
    rank][hidden / 2 words]`` a row), then flag ``RANK * tasks + task``."""
    mb.stamp(STAGE, trace.START)
    D: gl.constexpr = schedule.dims(mb.plan)
    LOCAL: gl.constexpr = D.local_experts
    HIDDEN: gl.constexpr = D.hidden
    R: gl.constexpr = schedule.rows(mb.plan)
    RP: gl.constexpr = triton.next_power_of_2(R)
    HALF: gl.constexpr = COLUMNS // 2
    layout: gl.constexpr = projection.word_layout(gl.num_warps())
    r = gl.arange(0, RP, gl.SliceLayout(1, layout))
    row = gl.minimum(r, R - 1)
    c = gl.arange(0, HALF, gl.SliceLayout(0, layout))
    lo = gl.zeros([RP, HALF], gl.float32, layout)
    hi = gl.zeros([RP, HALF], gl.float32, layout)
    for k in gl.static_range(TOP_K):
        ids = mb.poll(route, mb.row_offset(row) + k)
        is_local = (ids >= RANK * LOCAL) & (ids < (RANK + 1) * LOCAL)
        # A remote route has no output: its lanes poll the row's route ids,
        # which are published.
        at = gl.where(
            is_local[:, None],
            mb.pairs(
                out,
                (mb.row_offset(row) + k * (HIDDEN // 2) + task * HALF)[:, None]
                + c[None, :],
            ),
            mb.pairs(route, mb.row_offset(row)[:, None] + c[None, :] * 0),
        )
        (got,) = mb.poll_pairs((at,))
        g_lo, g_hi = mix.unpack(got)
        lo += gl.where(is_local[:, None], g_lo, 0.0)
        hi += gl.where(is_local[:, None], g_hi, 0.0)
    mb.stamp(STAGE, trace.COMPUTED)
    word = mix.pack(lo, hi)
    index = (2 * mb.row_offset(row) + RANK * (HIDDEN // 2) + task * HALF)[:, None] + c[
        None, :
    ]
    W: gl.constexpr = schedule.world(mb.plan)
    for p in gl.static_range(W):
        mb.store_words_peer(part, p, index, word, mask=(r < R)[:, None])
    mb.raise_flag_peers(flag, RANK * (HIDDEN // COLUMNS) + task)
    mb.stamp(STAGE, trace.END)

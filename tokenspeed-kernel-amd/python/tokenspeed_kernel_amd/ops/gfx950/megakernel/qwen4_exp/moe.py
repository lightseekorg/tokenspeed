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

"""Sparse MoE stages (TokenSpeed ``Qwen3_5MoeSparseMoeBlock``).

``out = sum_k w_k E_k(x) + sigmoid(g . x) S(x)`` over the top 10 of 512
routed experts (softmax weights renormalized over the 10) and the shared
expert ``S``. ``E(x) = W_down (silu(W_gate x) * W_up x)``, with the
intermediate rounded to BF16 and quantized to MXFP8 like TokenSpeed's MXFP8
expert pipeline. The block output joins the residual streams with the MLP
mixer's inject logits: ``h_b + bf16(out) * 2 sigmoid(inject_b)``.

Every task serves every row of the launch:

* router (M3), 65 tasks. Each task runs 8 router rows (BF16), but the last
  task runs the shared expert gate. M3 publishes the 512 logits as BF16.
* expert up (M4), 220 tasks: rank ``r`` (0-9 routed, 10 shared) x 32
  intermediate columns. A task polls its input rows first (M2 published them
  before M3 ran). Then it routes (:func:`routing`, the CTA's waves together).
  Then it streams each row's expert weights, two rows in flight. Each row's
  chain lands in its own MFMA rows of one accumulator. Thus one epilogue
  publishes ``silu(gate) * up`` of every row as MXFP8 blocks.
* expert down (M5), 160 tasks of 16 hidden columns. A task routes again and
  polls every row's 11 intermediates. It runs each row's 11 down projections
  as one rank-3 MFMA whose batch dimension is the expert (into its own MFMA
  rows of one accumulator). One epilogue combines them and writes the
  layer's output streams.

The stages read routed experts in place from TokenSpeed's N16 MoE banks. The
MXFP8 pipeline pads the intermediate to ``INTER_PAD``, and its scales use the
``"n32"`` cells of ``mx._scale_bytes``. The shared expert's weights are the
dense projections' preshuffled tiles, with scale copies in the same cells.
Its gate / up blocks alternate like the banks' (``weights.alternate_gate_up``).
Thus every M4 rank reads its weights the same way.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._scheduling import wave_uniform_i32
from tokenspeed_kernel_amd._triton import aggregate, gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.megakernel import mx, schedule, trace
from tokenspeed_kernel_amd.ops.gfx950.megakernel.plan import Stage, placement
from tokenspeed_kernel_amd.ops.gfx950.megakernel.qwen4_exp.dims import HIDDEN

EXPERTS = gl.constexpr(512)
TOP_K = gl.constexpr(10)
RANKS = gl.constexpr(11)  # the routed top 10, then the shared expert
INTER = gl.constexpr(640)
INTER_PAD = gl.constexpr(768)
ROUTER_TASKS = gl.constexpr(65)
UP_TASKS = gl.constexpr(220)  # RANKS * INTER / 32
DOWN_TASKS = gl.constexpr(160)  # HIDDEN / 16
W13_CELLS = gl.constexpr(10)  # 256-wide K cells of HIDDEN
W2_CELLS = gl.constexpr(3)  # of INTER_PAD
# Rows of M5 weights a task keeps in flight. Each row takes about 45 VGPRs.
# Two spill at four rows. The task issues the first while M4 ends, and the
# next as a row's chain frees its registers.
DOWN_INFLIGHT = gl.constexpr(1)
# Rows whose M5 intermediates one poll round trip covers.
DOWN_POLL_ROWS = gl.constexpr(2)

REGIONS = (
    ("lg", 256),  # router logits, BF16 pairs
    ("sg", 1),  # shared expert gate logit (f32)
    ("dn8", 1760),  # MXFP8 words of each rank's intermediate, [RANKS, 160]
    ("dne", 220),  # its exponents, [RANKS, 20]
)


def stages(
    bases: dict[str, int], X: str, W: str, OUT: tuple[str, ...]
) -> tuple[Stage, ...]:
    """M3-M5, placed by ``bases``. ``X`` is the MLP mixer's region prefix and
    ``W`` is the region of the residual streams. M5 also publishes the output
    streams in the regions ``OUT`` (none or one)."""
    m3, m4, m5 = placement(bases, ("m3", "m4", "m5"))
    return (
        Stage("m3", int(ROUTER_TASKS), m3, reads=(X + "xbf",), writes=("lg", "sg")),
        Stage(
            "m4",
            int(UP_TASKS),
            m4,
            reads=("lg", X + "x8", X + "xe"),
            writes=("dn8", "dne"),
        ),
        Stage(
            "m5",
            int(DOWN_TASKS),
            m5,
            reads=("lg", "dn8", "dne", W, X + "inj", "sg"),
            writes=OUT,
        ),
    )


@aggregate
class Weights:
    """The MoE block's weights (``weights.MoeWeights``)."""

    router_w: gl.tensor  # router, BF16 [EXPERTS, HIDDEN]
    gate_w: gl.tensor  # shared expert gate, BF16 [HIDDEN]
    # Routed experts' N16 gate / up bank [EXPERTS, 2 INTER_PAD, HIDDEN]
    # (16-row gate and up blocks alternate) and its "n32" scales.
    w13: gl.tensor
    s13: gl.tensor
    # The shared expert's gate / up [2 INTER, HIDDEN] preshuffled tiles and
    # "n32" scales, in the banks' block order (weights.alternate_gate_up).
    shared_gu: gl.tensor
    shared_gu_s: gl.tensor
    # Routed experts' N16 down bank [EXPERTS, HIDDEN, INTER_PAD] and its
    # "n32" scales.
    w2: gl.tensor
    s2: gl.tensor
    # The shared expert's down [HIDDEN, INTER] preshuffled tiles and "n32"
    # scale copy (W2_CELLS cells).
    shared_dn: gl.tensor
    shared_dn_s: gl.tensor


@gluon.constexpr_function
def _rank_layout(num_warps):
    """``[16]`` per-rank values, rank ``r`` on lane ``r`` of every wave."""
    return gl.BlockedLayout([1], [64], [num_warps], [0])


@gluon.jit
def _pick(values, k):
    """Rank ``k`` (a runtime scalar) of a :func:`_route_row` vector."""
    ranks = gl.arange(0, 16, values.type.layout)
    return gl.sum(gl.where(ranks == k, values, 0), axis=0)


@gluon.jit
def _by_rank(values, ranks):
    """A :func:`_route_row` vector in the layout of ``ranks`` (``arange(16)``)."""
    return gl.convert_layout(values, ranks.type.layout)


@gluon.jit
def router_stage(mb, task, weights, X: gl.constexpr, STAGE: gl.constexpr):
    """M3 task ``task``: logits of experts ``8 task .. 8 task + 7`` (BF16), or
    for the last task the shared expert gate logit (FP32).

    Args:
        weights: The block's :class:`Weights`.
        X: Region prefix of the MLP mixer (its BF16 row ``xbf``).
    """
    mb.stamp(STAGE, trace.START)
    is_gate = task == ROUTER_TASKS - 1
    rl: gl.constexpr = gl.BlockedLayout([1, 8], [1, 64], [gl.num_warps(), 1], [1, 0])
    r = gl.arange(0, 8, gl.SliceLayout(1, rl))
    k = gl.arange(0, 4096, gl.SliceLayout(0, rl))
    rows = gl.where(
        is_gate,
        weights.gate_w + 0 * (r[:, None] * HIDDEN),
        weights.router_w + (task * 8 + r)[:, None] * HIDDEN,
    )
    w = gl.load(rows + k[None, :], mask=(k < HIDDEN)[None, :], other=0.0)
    gl.amd.hint.sched_barrier()
    # The BF16 rows: poll compactly, then share them through LDS.
    R: gl.constexpr = schedule.rows(mb.plan)
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    i = gl.arange(0, 2048, lanes)
    flat: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    x_words = gl.allocate_shared_memory(
        gl.int32, [triton.next_power_of_2(R) * 2048], flat
    )
    words = mb.poll_rows((X + "xbf",), (gl.minimum(i, HIDDEN // 2 - 1),), R)
    for row in gl.static_range(R):
        x_words.slice(row * 2048, 2048).store(
            gl.where(i < HIDDEN // 2, words[row][0], 0)
        )
    gl.barrier()
    mb.stamp(STAGE, trace.READY)
    for row in gl.static_range(R):
        x = x_words.slice(row * 2048, 2048).reinterpret(gl.bfloat16, [4096], flat)
        x = x.load(gl.SliceLayout(0, rl))
        logit = gl.sum(w.to(gl.float32) * x.to(gl.float32)[None, :], axis=1)
        if is_gate:
            one = gl.arange(0, 1, lanes)
            mb.row(row).put(
                "sg", one, gl.full([1], 0.0, gl.float32, lanes) + gl.max(logit, axis=0)
            )
        else:
            mx.put_bf16(mb.row(row), "lg", task * 4, logit)
    mb.stamp(STAGE, trace.COMPUTED)
    mb.stamp(STAGE, trace.END)


@gluon.jit
def _up_tiles(task, experts, weights):
    """Issue the weight loads of M4 task ``task`` (``experts`` from
    :func:`_route_row`): blocks ``4 q .. 4 q + 3`` (gate, up, gate, up rows
    of columns ``32 q ..``) of its rank's expert.

    The shared expert's blocks alternate like the banks'. Thus every rank is
    one load sequence with a scalar base. A branch per rank would join two
    sequences' registers, and the copies there wait for the loads.
    """
    rank = task // 20
    q = task % 20
    routed = rank < TOP_K
    # Wave-uniform, so the loads take a scalar base (else every load becomes
    # a waterfall loop over the lanes' bases).
    expert = wave_uniform_i32(_pick(experts, rank)).to(gl.int64)
    if routed:
        tiles, scales = mx.load_weight_tiles(
            weights.w13 + expert * (2 * INTER_PAD * HIDDEN // 2),
            weights.s13 + expert * (INTER_PAD // 16 * W13_CELLS * 256),
            nb0=q * 4,
            K=HIDDEN,
            kit0=0,
            KITS=20,
            SPLIT=2,
            NW=4,
            GROUP=4,
            GROUP_STRIDE=0,
            SCALES="n32",
        )
    else:
        tiles, scales = mx.load_weight_tiles(
            weights.shared_gu,
            weights.shared_gu_s,
            nb0=q * 4,
            K=HIDDEN,
            kit0=0,
            KITS=20,
            SPLIT=2,
            NW=4,
            GROUP=4,
            GROUP_STRIDE=0,
            SCALES="n32",
        )
    return tiles, scales


@gluon.jit
def _up_publish(mb, task, out):
    """SwiGLU of M4 task ``task``'s GEMV result ``out [RP, 64]`` (a row per
    row), published as one MXFP8 block of ``dn8`` / ``dne`` per row."""
    RP: gl.constexpr = out.shape[0]
    # Result column n is block n // 16: gate, up, gate, up. Round to BF16
    # like TokenSpeed's epilogues.
    out = gl.reshape(out.to(gl.bfloat16).to(gl.float32), [RP, 2, 2, 16])
    lay: gl.constexpr = out.type.layout
    i1 = gl.arange(0, 2, gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(3, lay))))
    gate = gl.sum(gl.where((i1 == 0)[None, None, :, None], out, 0.0), axis=2)
    up = gl.sum(gl.where((i1 == 1)[None, None, :, None], out, 0.0), axis=2)
    h = (mx.silu(gate) * up).to(gl.bfloat16).to(gl.float32)
    mx.put_mxfp8_rows(mb, "dn8", "dne", task, gl.reshape(h, [RP, 32]))


@gluon.constexpr_function
def _batch_layouts(num_warps):
    """Rank-3 MFMA over the 16 (11 used) ranks: 2 ranks per wave."""
    mma = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 128],
        transposed=True,
        warps_per_cta=[num_warps, 1, 1],
    )
    return mma, gl.DotOperandLayout(0, mma, 16), gl.DotOperandLayout(1, mma, 16)


@gluon.constexpr_function
def _batch_scale_layout(num_warps):
    """E8M0 scales ``[16 ranks, 16 rows, 4 K groups]`` of :func:`_batch_layouts`.

    Waves own ranks ``0 .. num_warps - 1`` and a register repeat adds
    ``num_warps`` (as the MFMA layout places its batch dimension).
    """
    warp_bits = num_warps.bit_length() - 1
    return gl.DistributedLinearLayout(
        reg_bases=[[1 << i, 0, 0] for i in range(warp_bits, 4)],
        lane_bases=[[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 0, 1], [0, 0, 2]],
        warp_bases=[[1 << i, 0, 0] for i in range(warp_bits)],
        block_bases=[],
        shape=[16, 16, 4],
    )


@gluon.jit
def _down_tiles(task, experts, weights):
    """Issue the weight loads of M5 task ``task`` (``experts`` from
    :func:`_route_row`)."""
    bl: gl.constexpr = _batch_layouts(gl.num_warps())[2]
    # B tiles of every rank: rank 10 is the shared expert, 11-15 unused.
    rb = gl.arange(0, 16, gl.SliceLayout(1, gl.SliceLayout(2, bl)))
    kb = gl.arange(0, 64, gl.SliceLayout(0, gl.SliceLayout(2, bl)))
    nb = gl.arange(0, 16, gl.SliceLayout(0, gl.SliceLayout(1, bl)))
    eb = _by_rank(experts, rb)
    routed = rb < TOP_K
    tile_base = gl.where(
        routed,
        (eb.to(gl.int64) * HIDDEN + task * 16) * (INTER_PAD // 2),
        task * 16 * (INTER // 2),
    )
    tile = (
        tile_base[:, None, None]
        + ((kb // 16) * 256 + kb % 16)[None, :, None]
        + (nb * 16)[None, None, :]
    )
    base_ptr = gl.where(
        routed[:, None, None], weights.w2 + tile, weights.shared_dn + tile
    )
    sl: gl.constexpr = _batch_scale_layout(gl.num_warps())
    rs = gl.arange(0, 16, gl.SliceLayout(1, gl.SliceLayout(2, sl)))
    ns = gl.arange(0, 16, gl.SliceLayout(0, gl.SliceLayout(2, sl)))
    gs = gl.arange(0, 4, gl.SliceLayout(0, gl.SliceLayout(1, sl)))
    es = _by_rank(experts, rs)
    cell = (
        (task // 2) * (W2_CELLS * 256)
        + gs[None, None, :] * 64
        + ns[None, :, None] * 4
        + task % 2
    )
    s_base = gl.where(
        (rs < TOP_K)[:, None, None],
        weights.s2
        + (es.to(gl.int64) * (HIDDEN // 32 * W2_CELLS * 256))[:, None, None]
        + cell,
        weights.shared_dn_s + cell,
    )
    used = (rb < RANKS)[:, None, None]
    tiles = ()
    scales = ()
    for i in gl.static_range(INTER // 128):
        tiles += (gl.load(base_ptr + i * 1024, mask=used, other=0),)
        scales += (
            gl.load(
                s_base + (i // 2) * 256 + (i % 2) * 2,
                mask=(rs < RANKS)[:, None, None],
                other=127,
            ),
        )
    return tiles, scales


@gluon.constexpr_function
def _stream_layout(num_warps):
    """``[rows, 4, 16]`` (row, stream, column) with a BF16 pair per lane."""
    return gl.BlockedLayout([1, 1, 2], [1, 8, 8], [1, num_warps, 1], [2, 1, 0])


@gluon.jit
def _down_inputs(mb, task, M: gl.constexpr, W: gl.constexpr):
    """Poll M5 task ``task``'s inputs for every row: each rank's
    intermediate (into LDS), the residual columns of region ``W``, and the
    MLP mixer's (prefix ``M``) inject gates and the shared expert gate.

    The intermediates poll ``DOWN_POLL_ROWS`` rows per round trip, which
    bounds a thread's polled pairs (and their registers). The first trip
    also polls the other inputs of every row.

    Returns:
        ``(x_words, xs_smem, residual, mix, shared_gate)``: LDS ``[RP * 128,
        32]`` int32 and ``[RP * 128, 4]`` uint8 (row ``r``'s k-iteration
        ``i``, rank ``k`` at LDS row ``(8 r + i) * 16 + k``), then FP32
        ``[RP, 4, 16]`` (:func:`_stream_layout`), ``[RP, 4]`` and ``[RP]``.
    """
    R: gl.constexpr = schedule.rows(mb.plan)
    RP: gl.constexpr = triton.next_power_of_2(R)
    G: gl.constexpr = _rows_per_poll(RP)
    NW: gl.constexpr = gl.num_warps()
    # Residual columns (a BF16 pair per lane), and the inject half-logits
    # ([row, stream, 0..1]) with the shared gate ([row, stream, 2..3]).
    sl: gl.constexpr = _stream_layout(NW)
    r = gl.arange(0, RP, gl.SliceLayout(1, gl.SliceLayout(2, sl)))[:, None, None]
    b = gl.arange(0, 4, gl.SliceLayout(0, gl.SliceLayout(2, sl)))[None, :, None]
    c = gl.arange(0, 16, gl.SliceLayout(0, gl.SliceLayout(1, sl)))[None, None, :]
    column = b * HIDDEN + task * 16 + c
    gate_l: gl.constexpr = gl.BlockedLayout(
        [1, 1, 1], [1, 4, 16], [1, 1, NW], [2, 1, 0]
    )
    gr = gl.arange(0, RP, gl.SliceLayout(1, gl.SliceLayout(2, gate_l)))[:, None, None]
    gb = gl.arange(0, 4, gl.SliceLayout(0, gl.SliceLayout(2, gate_l)))[None, :, None]
    gh = gl.arange(0, 4, gl.SliceLayout(0, gl.SliceLayout(1, gate_l)))[None, None, :]
    gates = gl.where(
        gh < 2,
        mb.pairs(M + "inj", mb.row_offset(gr) + gh * 4 + gb),
        mb.pairs("sg", mb.row_offset(gr) + gb * 0),
    )
    flat: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    x_words = gl.allocate_shared_memory(gl.int32, [RP * 128, 32], flat)
    xs_smem = gl.allocate_shared_memory(gl.uint8, [RP * 128, 4], flat)
    # Throttle: wave 0 waits for M4's exponents (each M4 task writes its
    # block's exponent after its words) before every wave polls the inputs.
    mb.row(R - 1).pre_poll_spread("dne", UP_TASKS)
    word_ok, word_i, exp_ok, exp_i = _intermediate_index(mb, 0, G)
    got_w, got_e, got_h, got_g = mb.poll_pairs(
        (
            mb.pairs("dn8", word_i),
            mb.pairs("dne", exp_i),
            mb.pairs(W, mb.row_offset(r) + column // 2),
            gates,
        )
    )
    _store_intermediates(x_words, xs_smem, 0, word_ok, got_w, exp_ok, got_e)
    for first in gl.static_range(G, RP, G):
        word_ok, word_i, exp_ok, exp_i = _intermediate_index(mb, first, G)
        got_w, got_e = mb.poll_pairs((mb.pairs("dn8", word_i), mb.pairs("dne", exp_i)))
        _store_intermediates(x_words, xs_smem, first, word_ok, got_w, exp_ok, got_e)
    residual = mx.bf16_half(got_h, column)
    halves = gl.where(gh < 2, got_g.to(gl.float32, bitcast=True), 0.0)
    logit = gl.sum(halves, axis=2).to(gl.bfloat16).to(gl.float32)
    mix = gl.convert_layout(2.0 * mx.sigmoid(logit), gl.SliceLayout(2, sl))
    gate = gl.max(gl.where(gh == 2, got_g.to(gl.float32, bitcast=True), -1e30), axis=2)
    shared_gate = gl.convert_layout(
        mx.sigmoid(gl.max(gate, axis=1)), gl.SliceLayout(1, gl.SliceLayout(1, sl))
    )
    gl.barrier()
    return x_words, xs_smem, residual, mix, shared_gate


@gluon.constexpr_function
def _rows_per_poll(rows):
    return min(rows, DOWN_POLL_ROWS)


@gluon.jit
def _intermediate_index(mb, FIRST: gl.constexpr, G: gl.constexpr):
    """``(valid, pair offsets)`` of the words ``[G, 8, 16, 32]`` and the
    exponents ``[G, 8, 16, 4]`` (row ``FIRST + g``, k-iteration, rank, ..)
    of the ranks' MXFP8 intermediates.

    A wave polls one k-iteration, and an instruction polls two ranks' words
    (or 16 ranks' exponents). Padding (k-iterations 5 to 7, ranks 11 to 15)
    polls the row's pair 0, so it costs one cache line per instruction.
    """
    NW: gl.constexpr = gl.num_warps()
    wl: gl.constexpr = gl.BlockedLayout(
        [1, 1, 1, 1], [1, 1, 2, 32], [1, NW, 1, 1], [3, 2, 1, 0]
    )
    el: gl.constexpr = gl.BlockedLayout(
        [1, 1, 1, 1], [1, 1, 16, 4], [1, NW, 1, 1], [3, 2, 1, 0]
    )
    word_ok, word_i = _rank_index(mb, wl, FIRST, G, 160, 32)
    exp_ok, exp_i = _rank_index(mb, el, FIRST, G, 20, 4)
    return word_ok, word_i, exp_ok, exp_i


@gluon.jit
def _rank_index(
    mb,
    layout: gl.constexpr,
    FIRST: gl.constexpr,
    G: gl.constexpr,
    RANK_PAIRS: gl.constexpr,
    N: gl.constexpr,
):
    s3: gl.constexpr = gl.SliceLayout(3, layout)
    r = (
        FIRST
        + gl.arange(0, G, gl.SliceLayout(1, gl.SliceLayout(2, s3)))[:, None, None, None]
    )
    k = gl.arange(0, 8, gl.SliceLayout(0, gl.SliceLayout(2, s3)))[None, :, None, None]
    rank = gl.arange(0, 16, gl.SliceLayout(0, gl.SliceLayout(1, s3)))[
        None, None, :, None
    ]
    w = gl.arange(
        0, N, gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(2, layout)))
    )[None, None, None, :]
    valid = (k < INTER // 128) & (rank < RANKS)
    return valid, mb.row_offset(r) + gl.where(valid, rank * RANK_PAIRS + k * N + w, 0)


@gluon.jit
def _store_intermediates(
    x_words, xs_smem, FIRST: gl.constexpr, word_ok, words, exp_ok, exps
):
    """Rows ``FIRST ..`` of the polled intermediates into :func:`_down_inputs`'
    LDS (padding as zero words and unit scales)."""
    G: gl.constexpr = words.shape[0]
    x_words.slice(FIRST * 128, G * 128, 0).store(
        gl.reshape(gl.where(word_ok, words, 0), [G * 128, 32])
    )
    xs_smem.slice(FIRST * 128, G * 128, 0).store(
        gl.reshape(gl.where(exp_ok, exps, 127).to(gl.uint8), [G * 128, 4])
    )


@gluon.jit
def _down_chain(x_view, xs_smem, tiles, scales, row: gl.constexpr):
    """M5's rank-batched MFMA chain of request ``row`` (its A rows from
    :func:`_down_inputs`' LDS), broadcast to every MFMA row."""
    mma: gl.constexpr = _batch_layouts(gl.num_warps())[0]
    al: gl.constexpr = _batch_layouts(gl.num_warps())[1]
    sl: gl.constexpr = _batch_scale_layout(gl.num_warps())
    a_full = gl.zeros([16, 16, 128], gl.uint8, al)
    as_full = gl.zeros([16, 16, 4], gl.uint8, sl)
    acc = gl.zeros([16, 16, 16], gl.float32, mma)
    for i in gl.static_range(INTER // 128):
        x = x_view.slice((row * 8 + i) * 16, 16, 0).load(gl.SliceLayout(1, al))
        a, _ = gl.broadcast(gl.expand_dims(x, 1), a_full)
        xs = xs_smem.slice((row * 8 + i) * 16, 16, 0).load(gl.SliceLayout(1, sl))
        a_s, _ = gl.broadcast(gl.expand_dims(xs, 1), as_full)
        acc = gl.amd.cdna4.mfma_scaled(
            a.to(gl.float8e4nv, bitcast=True),
            a_s,
            "e4m3",
            tiles[i],
            scales[i],
            "e2m1",
            acc,
        )
    return acc


@gluon.jit
def _down_epilogue(mb, task, acc, route_weights, inputs, out, OUT: gl.constexpr):
    """Every row of M5 task ``task`` from its gathered accumulator
    ``acc`` (MFMA row ``m`` holds row ``m % RP``). The epilogue:

    1. Combines the ranks with ``route_weights`` (:func:`routing`).
    2. Adds the result to the residual.
    3. Stores the layer's output streams ``out`` (BF16 ``[rows, 4 * HIDDEN]``).

    ``OUT`` names a region that also receives the output streams as BF16
    pairs (for stages after the layer in the same launch), or is ``""``.
    """
    R: gl.constexpr = schedule.rows(mb.plan)
    RP: gl.constexpr = triton.next_power_of_2(R)
    _, _, residual, mix, shared_gate = inputs
    # [rank, row, column]. The MFMA rows of a row agree.
    y = gl.max(gl.reshape(acc, [16, 16 // RP, RP, 16]), axis=1)
    y = y.to(gl.bfloat16).to(gl.float32)
    yl: gl.constexpr = y.type.layout
    rk = gl.arange(0, 16, gl.SliceLayout(1, gl.SliceLayout(2, yl)))[:, None, None]
    w = gl.convert_layout(gl.permute(route_weights, [1, 0]), gl.SliceLayout(2, yl))
    routed = gl.sum(gl.where(rk < TOP_K, y * w[:, :, None], 0.0), axis=0)
    shared = gl.sum(gl.where(rk == TOP_K, y, 0.0), axis=0)
    sl: gl.constexpr = _stream_layout(gl.num_warps())
    routed = gl.convert_layout(
        routed.to(gl.bfloat16).to(gl.float32), gl.SliceLayout(1, sl)
    )
    shared = gl.convert_layout(shared, gl.SliceLayout(1, sl))
    block = (routed + shared_gate[:, None] * shared).to(gl.bfloat16).to(gl.float32)
    streams = (residual + block[:, None, :] * mix[:, :, None]).to(gl.bfloat16)
    r = gl.arange(0, RP, gl.SliceLayout(1, gl.SliceLayout(2, sl)))[:, None, None]
    b = gl.arange(0, 4, gl.SliceLayout(0, gl.SliceLayout(2, sl)))[None, :, None]
    c = gl.arange(0, 16, gl.SliceLayout(0, gl.SliceLayout(1, sl)))[None, None, :]
    gl.store(out + r * (4 * HIDDEN) + b * HIDDEN + task * 16 + c, streams, mask=r < R)
    if OUT != "":
        streams = streams.to(gl.float32)
        for i in gl.static_range(4):
            stream = gl.sum(gl.where(b == i, streams, 0.0), axis=1)
            mx.put_bf16_rows(mb, OUT, (i * HIDDEN + task * 16) // 2, stream)


@gluon.jit
def routing(mb):
    """Every row's top 10 of the router logits (ties: the lower
    expert) and their renormalized softmax weights.

    The key ``(monotone(bf16) << 9) | (511 - expert)`` orders by logit then
    by lower expert. Rounds of a max pick experts in rank order. The CTA's
    waves share the work. Wave ``w`` takes row ``w % RP`` and one slice of
    the experts and keeps its slice's top 10. The slices' picks meet in LDS.
    Then wave ``w`` takes row ``w % RP``'s top 10 of those picks and their
    softmax weights. The top 10 of the slices' top 10s is the top 10.

    Returns:
        ``(experts, weights)``: ``[rows, 16]`` by rank. Ranks past the top 10
        hold expert 0 and weight 0. :func:`_route_row` reads one row.
    """
    R: gl.constexpr = schedule.rows(mb.plan)
    RP: gl.constexpr = triton.next_power_of_2(R)
    NW: gl.constexpr = gl.num_warps()
    SLICES: gl.constexpr = NW // RP
    SLICE: gl.constexpr = EXPERTS // SLICES
    # One wave per [row, slice] pair: wave w = slice * RP + row.
    wl: gl.constexpr = gl.BlockedLayout([1, SLICE // 64], [1, 64], [NW, 1], [1, 0])
    w = gl.arange(0, NW, gl.SliceLayout(1, wl))[:, None]
    e = (w // RP) * SLICE + gl.arange(0, SLICE, gl.SliceLayout(0, wl))[None, :]
    word = mb.poll("lg", mb.row_offset(w % RP) + e // 2)
    bits = (word >> (16 * (e % 2))) & 0xFFFF
    # BF16 bits to an order-preserving unsigned key.
    monotone = gl.where(bits >= 0x8000, bits ^ 0xFFFF, bits | 0x8000)
    key = (monotone << 9) | (EXPERTS - 1 - e)
    picks = _top_keys(key)
    # Each row's slice picks side by side, [RP, SLICES * 16], in LDS.
    flat: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    shared = gl.allocate_shared_memory(gl.int32, [RP, SLICES * 16], flat)
    shared.store(
        gl.reshape(
            gl.permute(gl.reshape(picks, [SLICES, RP, 16]), [1, 0, 2]),
            [RP, SLICES * 16],
        )
    )
    gl.barrier()
    # Wave w merges row w % RP: rows over waves (repeated past RP).
    ml: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [NW, 1], [1, 0])
    best = _top_keys(shared.load(ml))
    rank = gl.arange(0, 16, gl.SliceLayout(0, best.type.layout))[None, :]
    m = best >> 9
    logits = (gl.where(m >= 0x8000, m & 0x7FFF, m ^ 0xFFFF) << 16).to(
        gl.float32, bitcast=True
    )
    first = gl.max(gl.where(rank == 0, logits, float("-inf")), axis=1)
    weights = gl.where(rank < TOP_K, gl.exp(logits - first[:, None]), 0.0)
    normed = weights / gl.sum(weights, axis=1)[:, None]
    experts = gl.where(rank < TOP_K, EXPERTS - 1 - (best & (EXPERTS - 1)), 0)
    # Every wave needs every row.
    rl: gl.constexpr = gl.BlockedLayout([RP, 1], [1, 64], [NW, 1], [1, 0])
    experts = gl.convert_layout(experts, rl)
    normed = gl.convert_layout(normed, rl)
    return experts, normed


@gluon.jit
def _top_keys(key):
    """The 16 largest of each row of ``key [rows, N]`` (N >= 16, distinct
    keys) in descending order, ``[rows, 16]`` in ``key``'s layout. Callers
    use only the first ``TOP_K`` (the rest are padding)."""
    rank = gl.arange(0, 16, gl.SliceLayout(0, key.type.layout))[None, :]
    picks = gl.zeros([key.shape[0], 16], gl.int32, key.type.layout)
    for k in gl.static_range(TOP_K):
        best = gl.max(key, axis=1)
        key = gl.where(key == best[:, None], 0, key)
        picks = gl.where(rank == k, best[:, None], picks)
    return picks


@gluon.jit
def _route_row(routes, r: gl.constexpr):
    """Row ``r`` of :func:`routing` as ``[16]`` per-rank vectors
    (:func:`_rank_layout`)."""
    experts, weights = routes
    rows = gl.arange(0, experts.shape[0], gl.SliceLayout(1, experts.type.layout))[
        :, None
    ]
    ranks: gl.constexpr = _rank_layout(gl.num_warps())
    return (
        gl.convert_layout(gl.sum(gl.where(rows == r, experts, 0), axis=0), ranks),
        gl.convert_layout(gl.sum(gl.where(rows == r, weights, 0.0), axis=0), ranks),
    )


@gluon.jit
def up_stage(mb, task, weights, X: gl.constexpr):
    """M4 task ``task``: rank ``task // 20``, intermediate columns
    ``32 (task % 20) ..`` of every row, published as MXFP8 blocks
    of ``dn8`` / ``dne``.

    Args:
        weights: The block's :class:`Weights`.
        X: Region prefix of the MLP mixer (its MXFP8 rows ``x8`` / ``xe``).
    """
    mb.stamp("m4", trace.START)
    x = _up_inputs(mb, X)
    routes = routing(mb)
    _up_finish(mb, task, routes, x, weights, 0, False)


@gluon.jit
def down_stage(
    mb, task, weights, out, M: gl.constexpr, W: gl.constexpr, OUT: gl.constexpr
):
    """M5 task ``task``: hidden columns ``16 task .. 16 task + 15`` of every
    row's block output, combined into the layer's output streams.

    Args:
        weights: The block's :class:`Weights`.
        out: The layer's output streams, BF16 ``[rows, 4 * HIDDEN]`` (global).
        M: Region prefix of the MLP mixer (its ``inj`` logits).
        W: Region of the residual streams (the output projection's ``w``).
        OUT: A region that also receives the output streams as BF16 pairs
            (for stages after the layer in the same launch), or ``""``.
    """
    mb.stamp("m5", trace.START)
    routes = routing(mb)
    head = _down_issue(task, routes, weights, schedule.rows(mb.plan))
    _down_finish(mb, task, routes, head, weights, out, M, W, OUT)


@gluon.jit
def up_down_stage(
    mb,
    up_task,
    down_task,
    weights,
    out,
    X: gl.constexpr,
    M: gl.constexpr,
    W: gl.constexpr,
    OUT: gl.constexpr,
):
    """M4 task ``up_task`` then M5 task ``down_task`` on one CTA (arguments
    as :func:`up_stage` and :func:`down_stage`).

    Both tasks' expert weights depend only on the routing. Thus the CTA issues
    the weight loads of the first M5 rows before M4's last row computes. These
    loads stream while M4 finishes, not after.
    """
    mb.stamp("m4", trace.START)
    x = _up_inputs(mb, X)
    routes = routing(mb)
    head = _up_finish(mb, up_task, routes, x, weights, down_task, True)
    mb.stamp("m5", trace.START)
    _down_finish(mb, down_task, routes, head, weights, out, M, W, OUT)


@gluon.jit
def _up_inputs(mb, X: gl.constexpr):
    """Poll M4's input, the mixed rows (MLP mixer prefix ``X``), into LDS.

    M4 polls them before the routing and its weight loads. M2 published
    them before M3 ran, so the poll returns at once. Each row's GEMV then
    waits only for its own weights (a poll waits for every earlier load).
    """
    R: gl.constexpr = schedule.rows(mb.plan)
    x_smem, xs_smem, _ = mx.load_mxfp8_rows(mb, X + "x8", X + "xe", 20, R)
    return x_smem, xs_smem


@gluon.jit
def _up_finish(
    mb,
    task,
    routes,
    x,
    weights,
    down_task,
    DOWN: gl.constexpr,
):
    """M4 task ``task`` for every row (``routes`` from :func:`routing`, ``x``
    from :func:`_up_inputs`). With ``DOWN``, it also issues the first rows'
    loads of M5 task ``down_task`` before the last GEMV (see
    :func:`_down_issue`) and returns them. Else it returns ``()``.

    Every row runs the rank's own weights: the routed rows' experts, or the
    shared expert for each row of a shared-rank task (a re-read from L2).
    """
    R: gl.constexpr = schedule.rows(mb.plan)
    x_smem, xs_smem = x
    mb.stamp("m4", trace.READY)
    # Two rows' weights are in flight. Rows 0 and 1 load at once. Row r + 2
    # loads as soon as row r computes (its registers are free then).
    loaded = ()
    for r in gl.static_range(_up_ahead(R)):
        loaded += (_up_tiles(task, _route_row(routes, r)[0], weights),)
    head = ()
    if (R == 1) & DOWN:
        head = _down_issue(down_task, routes, weights, R)
    # Every row's chain adds into one accumulator (its own MFMA rows), so
    # one epilogue serves all rows.
    acc = mx.gemv_zero(2, 4)
    for r in gl.static_range(R):
        tiles, scales = loaded[r]
        acc = mx.gemv_row(acc, x_smem, xs_smem, tiles, scales, r)
        # Keep each row's loads after the chain that frees their registers.
        gl.amd.hint.sched_barrier()
        if r + 2 < R:
            loaded += (_up_tiles(task, _route_row(routes, r + 2)[0], weights),)
        elif (r + 2 == R) & DOWN:
            # M5's first rows stream while M4's last row computes.
            head = _down_issue(down_task, routes, weights, R)
    _up_publish(mb, task, mx.gemv_result(acc, xs_smem.shape[0]))
    mb.stamp("m4", trace.COMPUTED)
    mb.stamp("m4", trace.END)
    return head


@gluon.constexpr_function
def _up_ahead(rows):
    return min(rows, 2)


@gluon.constexpr_function
def _inflight(rows):
    return min(rows, DOWN_INFLIGHT)


@gluon.jit
def _down_issue(task, routes, weights, R: gl.constexpr):
    """Issue M5 task ``task``'s weight loads of its first ``DOWN_INFLIGHT`` rows."""
    head = ()
    for r in gl.static_range(_inflight(R)):
        head += (_down_tiles(task, _route_row(routes, r)[0], weights),)
    return head


@gluon.jit
def _down_finish(
    mb,
    task,
    routes,
    head,
    weights,
    out,
    M: gl.constexpr,
    W: gl.constexpr,
    OUT: gl.constexpr,
):
    """M5 task ``task`` for every row, with the first rows' loads (``head``)
    already issued. The task issues loads up to ``DOWN_INFLIGHT`` rows and
    polls every row's inputs. A poll waits for the loads, which land while
    the slowest M4 task finishes. Then the task runs each row's chain into
    one accumulator and one epilogue. Row ``r + DOWN_INFLIGHT``'s loads issue
    once row ``r``'s chain frees its registers.
    """
    R: gl.constexpr = schedule.rows(mb.plan)
    RP: gl.constexpr = triton.next_power_of_2(R)
    loaded = head
    for r in gl.static_range(len(head), _inflight(R)):
        loaded += (_down_tiles(task, _route_row(routes, r)[0], weights),)
    inputs = _down_inputs(mb, task, M, W)
    mb.stamp("m5", trace.READY)
    flat: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    x_view = inputs[0].reinterpret(gl.uint8, [RP * 128, 128], flat)
    mma: gl.constexpr = _batch_layouts(gl.num_warps())[0]
    acc = gl.zeros([16, 16, 16], gl.float32, mma)
    m = gl.arange(0, 16, gl.SliceLayout(0, gl.SliceLayout(2, mma)))[None, :, None]
    for r in gl.static_range(R):
        tiles, scales = loaded[r]
        own = _down_chain(x_view, inputs[1], tiles, scales, r)
        acc = gl.where(m % RP == r, own, acc)
        # Keep the next row's loads after this chain, which frees their
        # registers.
        gl.amd.hint.sched_barrier()
        if r + DOWN_INFLIGHT < R:
            loaded += (
                _down_tiles(task, _route_row(routes, r + DOWN_INFLIGHT)[0], weights),
            )
    mb.stamp("m5", trace.COMPUTED)
    _down_epilogue(mb, task, acc, routes[1], inputs, out, OUT)
    mb.stamp("m5", trace.END)

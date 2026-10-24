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

"""Sparse MoE stages with BF16 experts (``moe.py`` for MXFP4 experts).

The block computes what ``moe.py`` computes, with BF16 weights and
activations and the roundings of TokenSpeed's BF16 MoE path at one row
(``gluon_bf16_moe`` with FP32 atomic combine, and BF16 dense projections for
the shared expert):

* a routed expert's gate and up projections stay FP32, and its intermediate
  is ``bf16(silu(gate) * up)``;
* the shared expert's gate and up projections are rounded to BF16 first;
* each routed expert's down projection stays FP32: the routed sum
  ``sum_k w_k down_k`` is rounded to BF16 once. The shared expert's down
  projection is rounded to BF16.

Routing (M3 and :func:`moe.routing`) and the block output are as in
``moe.py``. The stages read the weights in place: the routed experts'
``w13`` (gate rows, then up rows) and ``w2``, and the shared expert's fused
``gate_up_proj`` and ``down_proj``, all row-major.

* expert up (M4), :data:`UP_BLOCKS` tasks: rank ``r`` (0-10, 10 the shared
  expert) x 16 intermediate columns. A routed task streams one unit per row
  (the row's expert), two in flight; a shared task reads its weights once
  for every row. M4 publishes the intermediates as BF16 pairs (``dnb``).
* expert down (M5), 160 tasks of 16 hidden columns. A task's chains are the
  (row, rank) down projections, 16 per rank-3 BF16 MFMA, as in ``moe.py``.
  The epilogue is ``moe.down_epilogue``.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._scheduling import wave_uniform_i32
from tokenspeed_kernel_amd._triton import aggregate, gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.megakernel import bf16, mx, schedule, trace
from tokenspeed_kernel_amd.ops.gfx950.megakernel.plan import Stage, placement
from tokenspeed_kernel_amd.ops.gfx950.megakernel.qwen4_exp import moe
from tokenspeed_kernel_amd.ops.gfx950.megakernel.qwen4_exp.dims import HIDDEN

INTER = moe.INTER
TOP_K = moe.TOP_K
RANKS = moe.RANKS
COLUMNS = gl.constexpr(16)  # intermediate columns of an M4 task
BLOCKS = gl.constexpr(40)  # INTER / COLUMNS
UP_BLOCKS = gl.constexpr(440)  # RANKS * BLOCKS
ROW_PAIRS = gl.constexpr(3520)  # RANKS * INTER / 2: a row's intermediates

REGIONS = (
    ("lg", 256),  # router logits, BF16 pairs
    ("sg", 1),  # shared expert gate logit (f32)
    ("dnb", int(ROW_PAIRS)),  # each rank's intermediate, BF16 pairs
)


def stages(
    bases: dict[str, int], rows: int, X: str, W: str, OUT: tuple[str, ...]
) -> tuple[Stage, ...]:
    """M3-M5 for ``rows`` rows (arguments as ``moe.stages``)."""
    m3, m4, m5 = placement(bases, ("m3", "m4", "m5"))
    return (
        Stage(
            "m3",
            int(moe.ROUTER_TASKS) * -(-rows // moe.router_rows(rows)),
            m3,
            reads=(X + "xbf",),
            writes=("lg", "sg"),
        ),
        Stage("m4", int(UP_BLOCKS), m4, reads=("lg", X + "xbf"), writes=("dnb",)),
        Stage(
            "m5",
            int(moe.DOWN_TASKS),
            m5,
            reads=("lg", "dnb", W, X + "inj", "sg"),
            writes=OUT,
        ),
    )


@aggregate
class Weights:
    """The MoE block's BF16 weights (``weights.MoeBf16Weights``), read in
    place."""

    router_w: gl.tensor  # router, BF16 [EXPERTS, HIDDEN]
    gate_w: gl.tensor  # shared expert gate, BF16 [HIDDEN]
    w13: gl.tensor  # routed gate / up, BF16 [EXPERTS, 2 INTER, HIDDEN]
    w2: gl.tensor  # routed down, BF16 [EXPERTS, HIDDEN, INTER]
    shared_gu: gl.tensor  # shared gate / up, BF16 [2 INTER, HIDDEN]
    shared_dn: gl.tensor  # shared down, BF16 [HIDDEN, INTER]


@gluon.jit
def _up_tiles(w):
    """Issue the loads of one M4 unit: the gate rows of the task's 16
    columns (16-row block ``q`` of ``w``) and their up rows (block
    ``BLOCKS + q``). ``w`` points at block ``q`` of an expert's gate / up."""
    return bf16.load_row_major_tiles(
        w,
        nb0=0,
        K=HIDDEN,
        kit0=0,
        KITS=HIDDEN // 32,
        SPLIT=4,
        NW=2,
        GROUP=1,
        GROUP_STRIDE=BLOCKS,
    )


@gluon.jit
def _swiglu(out, ROUND: gl.constexpr):
    """``bf16(silu(gate) * up)`` of a GEMV result ``out [RP, 32]`` (gate in
    columns 0-15, up in 16-31), as FP32 ``[RP, 16]``. With ``ROUND``, gate
    and up are rounded to BF16 first."""
    RP: gl.constexpr = out.shape[0]
    if ROUND:
        out = out.to(gl.bfloat16).to(gl.float32)
    out = gl.reshape(out, [RP, 2, 16])
    lay: gl.constexpr = out.type.layout
    half = gl.arange(0, 2, gl.SliceLayout(0, gl.SliceLayout(2, lay)))[None, :, None]
    gate = gl.sum(gl.where(half == 0, out, 0.0), axis=1)
    up = gl.sum(gl.where(half == 1, out, 0.0), axis=1)
    sig = 1.0 / (1.0 + gl.exp(-gate))
    return ((gate * sig) * up).to(gl.bfloat16).to(gl.float32)


@gluon.jit
def up_stage(mb, task, weights, X: gl.constexpr):
    """M4 task ``task``: rank ``task // 40``, intermediate columns
    ``16 (task % 40) ..`` of every row, published as BF16 pairs of ``dnb``.

    Args:
        weights: The block's :class:`Weights`.
        X: Region prefix of the MLP mixer (its BF16 rows ``xbf``).
    """
    mb.stamp("m4", trace.START)
    R: gl.constexpr = schedule.rows(mb.plan)
    rank = task // BLOCKS
    q = task % BLOCKS
    # M2 published the rows before M3 ran: the poll returns at once, and each
    # GEMV then waits only for its own weights.
    x_smem, _ = bf16.load_rows(mb, X + "xbf", 0, HIDDEN, R)
    routes = moe.routing(mb)
    mb.stamp("m4", trace.READY)
    pair = (rank * INTER + q * COLUMNS) // 2
    if rank < TOP_K:
        # One unit per row: the row's expert. Two in flight; unit u + 2
        # issues once unit u's registers are free.
        loaded = ()
        for u in gl.static_range(min(R, 2)):
            loaded += (_up_tiles(_expert_rows(weights, routes, rank, u, q)),)
        for u in gl.static_range(R):
            out = bf16.gemv(x_smem, loaded[u], FIRST=u, ROWS=1)
            gl.amd.hint.sched_barrier()
            if u + 2 < R:
                loaded += (_up_tiles(_expert_rows(weights, routes, rank, u + 2, q)),)
            mx.put_bf16(mb.row(u), "dnb", pair, gl.reshape(_swiglu(out, False), [16]))
    else:
        tiles = _up_tiles(weights.shared_gu + q * (COLUMNS * HIDDEN))
        rows = bf16.gemv(x_smem, tiles, FIRST=0, ROWS=R)
        mx.put_bf16_rows(mb, "dnb", pair, _swiglu(rows, True))
    mb.stamp("m4", trace.COMPUTED)
    mb.stamp("m4", trace.END)


@gluon.jit
def _expert_rows(weights, routes, rank, r: gl.constexpr, q):
    """The gate rows of block ``q`` of row ``r``'s rank-``rank`` expert."""
    # Wave-uniform, so the loads take a scalar base.
    expert = wave_uniform_i32(moe._pick(moe._route_row(routes, r)[0], rank))
    return weights.w13 + (expert.to(gl.int64) * (2 * INTER) + q * COLUMNS) * HIDDEN


@gluon.constexpr_function
def _batch_layouts(num_warps):
    """Rank-3 BF16 MFMA over a group of 16 M5 chains, and its operands."""
    return (
        bf16.mfma_layout(num_warps, 1),
        bf16.a_layout(num_warps, 1),
        bf16.b_layout(num_warps, 1),
    )


@gluon.jit
def _down_tiles(task, group: gl.constexpr, experts, weights, R: gl.constexpr):
    """Issue the weight loads of chains ``16 group ..`` of M5 task ``task``:
    each chain's ``[16 hidden columns, INTER]`` rows, one ``[16, 32, 16]``
    B tile per k-iteration. Chains past the last row load nothing."""
    bl: gl.constexpr = _batch_layouts(gl.num_warps())[2]
    cb = group * 16 + gl.arange(0, 16, gl.SliceLayout(1, gl.SliceLayout(2, bl)))
    kb = gl.arange(0, 32, gl.SliceLayout(0, gl.SliceLayout(2, bl)))
    nb = gl.arange(0, 16, gl.SliceLayout(0, gl.SliceLayout(1, bl)))
    eb = moe._chain_values(experts, cb, R)
    used, _, rank = moe._chain_rows(cb, R)
    routed = used & (rank < TOP_K)
    rows = (task * 16 + nb)[None, None, :] * INTER + kb[None, :, None]
    base = gl.where(
        routed[:, None, None],
        weights.w2 + eb.to(gl.int64)[:, None, None] * (HIDDEN * INTER) + rows,
        weights.shared_dn + rows,
    )
    tiles = ()
    for i in gl.static_range(INTER // 32):
        tiles += (gl.load(base + i * 32, mask=used[:, None, None], other=0.0),)
    return tiles


@gluon.jit
def _down_inputs(mb, task, M: gl.constexpr, W: gl.constexpr):
    """Poll M5 task ``task``'s inputs for every row: ``moe.block_inputs``,
    then each chain's intermediate into LDS.

    Returns:
        ``(x_lo, x_hi, block)``: BF16 ``[S, 512]`` and ``[S, 128]`` in LDS,
        values ``0 .. 511`` and ``512 .. 639`` of each chain (chain ``c`` in
        row ``c``, ``S = moe._chain_slots(rows)``; rows past the chains hold
        0; LDS shapes are powers of 2), and ``moe.block_inputs``.
    """
    R: gl.constexpr = schedule.rows(mb.plan)
    S: gl.constexpr = moe._chain_slots(R)
    NW: gl.constexpr = gl.num_warps()
    block = moe.block_inputs(mb, task, M, W)
    flat: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    lo = gl.allocate_shared_memory(gl.int32, [S, 256], flat)
    hi = gl.allocate_shared_memory(gl.int32, [S, 64], flat)
    # A chain's 320 pairs in 5 parts of 64, a part's chains across the
    # waves. Throttle: wave 0 first waits for M4's tasks.
    wl: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [NW, 1], [1, 0])
    ok, row, rank = moe._chain_rows(gl.arange(0, S, gl.SliceLayout(1, wl)), R)
    p = gl.arange(0, 64, gl.SliceLayout(0, wl))[None, :]
    first = mb.row_offset(row)[:, None] + gl.where(ok[:, None], rank[:, None] * 320, 0)
    mb.row(R - 1).pre_poll_spread("dnb", ROW_PAIRS)
    for part in gl.static_range(0, 5, 4):
        offsets = ()
        for j in gl.static_range(part, min(part + 4, 5)):
            offsets += (mb.pairs("dnb", first + j * 64 + p),)
        got = mb.poll_pairs(offsets)
        for j in gl.static_range(len(got)):
            value = gl.where(ok[:, None], got[j], 0)
            if part + j < 4:
                lo.slice((part + j) * 64, 64, 1).store(value)
            else:
                hi.store(value)
    gl.barrier()
    return (
        lo.reinterpret(gl.bfloat16, [S, 512], flat),
        hi.reinterpret(gl.bfloat16, [S, 128], flat),
        block,
    )


@gluon.jit
def _down_chain_group(x_lo, x_hi, tiles, group: gl.constexpr):
    """The rank-batched MFMA chain of chains ``16 group ..``: each chain's
    intermediate (LDS rows of :func:`_down_inputs`, broadcast to every MFMA
    row) times its tiles."""
    mma: gl.constexpr = _batch_layouts(gl.num_warps())[0]
    al: gl.constexpr = _batch_layouts(gl.num_warps())[1]
    a_full = gl.zeros([16, 16, 32], gl.bfloat16, al)
    acc = gl.zeros([16, 16, 16], gl.float32, mma)
    lo = x_lo.slice(group * 16, 16, 0)
    hi = x_hi.slice(group * 16, 16, 0)
    for i in gl.static_range(INTER // 32):
        if i < 16:
            x = lo.slice(i * 32, 32, 1).load(gl.SliceLayout(1, al))
        else:
            x = hi.slice((i - 16) * 32, 32, 1).load(gl.SliceLayout(1, al))
        a, _ = gl.broadcast(gl.expand_dims(x, 1), a_full)
        acc = gl.amd.cdna4.mfma(a, tiles[i], acc)
    return acc


@gluon.jit
def down_stage(
    mb, task, weights, out, M: gl.constexpr, W: gl.constexpr, OUT: gl.constexpr
):
    """M5 task ``task``: hidden columns ``16 task .. 16 task + 15`` of every
    row's block output, combined into the layer's output streams (arguments
    as ``moe.down_stage``).

    The first chain group's weights issue before the input poll; each later
    group's issue once the group before it computed (a group's tiles take
    160 registers).
    """
    mb.stamp("m5", trace.START)
    R: gl.constexpr = schedule.rows(mb.plan)
    G: gl.constexpr = moe._chain_groups(R)
    routes = moe.routing(mb)
    tiles = _down_tiles(task, 0, routes[0], weights, R)
    x_lo, x_hi, block = _down_inputs(mb, task, M, W)
    mb.stamp("m5", trace.READY)
    accs = ()
    for g in gl.static_range(G):
        accs += (_down_chain_group(x_lo, x_hi, tiles, g),)
        gl.amd.hint.sched_barrier()
        if g + 1 < G:
            tiles = _down_tiles(task, g + 1, routes[0], weights, R)
    mb.stamp("m5", trace.COMPUTED)
    moe.down_epilogue(mb, task, accs, routes, block, out, OUT, False)
    mb.stamp("m5", trace.END)


__all__ = [
    "REGIONS",
    "UP_BLOCKS",
    "Weights",
    "down_stage",
    "stages",
    "up_stage",
]

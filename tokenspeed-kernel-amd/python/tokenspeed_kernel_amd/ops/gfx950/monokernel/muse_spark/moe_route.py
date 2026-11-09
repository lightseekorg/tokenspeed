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

"""Expert routing of one Muse Spark rank.

The router multiplies the FP32 pre-FFN norm output ``u = h' s g`` (``s =
rsqrt(mean(h'^2) + 1e-5)``, ``g = bf16(w + 1)``) by the FP32 router weight.
Rank ``r`` holds ``h'`` only for its 1024 owned columns, so it takes the
matching 1024 columns of the router weight (1 MiB, not the replicated 8 MiB)
and every rank adds the 8 partial logits. ``u`` is not rounded on this path,
and ``h' = beta b inv + alpha h`` (see attn_tail) is linear in the
post-attention norm's ``inv``. So the router multiplies ``beta b g`` and
``alpha h g``, and top-8 applies the scalars ``inv`` and ``s``: the router
waits for neither norm exchange, only for ``b``, which it sums itself from
the ranks' o_proj partials (``red``) in rank order, as tail1 does: the same
bits, without a hop through tail1.

* router, 16 tasks of 16 experts (past one row, 16 per pair of rows): this
  rank's partial logits of ``alpha h g`` (from the launch's inputs, at once)
  and of ``beta b g`` of its rows, put into every rank's ``lg2`` and ``lg``
  slots of this rank.
* topk, 1 task per row: the rank-order sum of ``inv lg + lg2`` times ``s`` (both
  from ``ss1``) times ``output_multiplier``, the sigmoid
  scores, and the 8 experts of the largest ``score + bias`` (8 argmax passes,
  a tie to the lowest id). The weights are the bias-free scores of the 8,
  divided by their sum. Every rank gets the same bits, published in ``route``
  as 8 ids, then 8 FP32 weights.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.monokernel import schedule, trace
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.attn_tail import (
    OWNED,
    norm_scales,
    sums_index,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.dims import EXPERTS, WORLD
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import Stage, placement

TOP_K = gl.constexpr(8)
ROUTER_TASKS = gl.constexpr(16)
ROUTER_ROWS = gl.constexpr(16)  # experts a router task
ROW_PAIR = gl.constexpr(2)  # token rows a router task serves past one row
ROUTER_SCALE = gl.constexpr(0.17677669529663687)  # output_multiplier

PEERS = ("lg", "lg2")


def regions() -> tuple[tuple[str, int], ...]:
    return (
        # FP32 partial logits [source rank][expert] of beta b and of alpha h
        ("lg", int(WORLD) * int(EXPERTS)),
        ("lg2", int(WORLD) * int(EXPERTS)),
        ("route", 2 * int(TOP_K)),  # int32 ids, then FP32 weights
    )


def stages(bases: dict[str, int], rows: int) -> tuple[Stage, ...]:
    """Router and topk of ``rows`` token rows, placed by ``bases``."""
    router, topk = placement(bases, ("router", "topk"))
    return (
        Stage(
            "router",
            int(ROUTER_TASKS) * (1 if rows == 1 else -(-rows // int(ROW_PAIR))),
            router,
            reads=("red",),
            writes=("lg", "lg2"),
        ),
        Stage("topk", rows, topk, reads=("lg", "lg2", "ss1"), writes=("route",)),
    )


@gluon.jit
def _owned_b(mb, row, src, j):
    """``b`` of the rank's owned columns ``2 j`` and ``2 j + 1`` (FP32 of the
    BF16 rounding) of rows ``row``: the 8 ranks' o_proj partials in ``red``
    (``src`` along the axis before ``j``) summed in rank order, as
    :func:`attn_tail.tail_stages` sums them. ``row``, ``src`` and ``j``
    broadcast to the poll's shape; returns ``(lo, hi)`` without the ``src``
    axis."""
    (got,) = mb.poll_many(("red",), (mb.row_offset(row) + src * (OWNED // 2) + j,))
    lo_all = (got << 16).to(gl.float32, bitcast=True)
    hi_all = (got & -65536).to(gl.float32, bitcast=True)
    AX: gl.constexpr = len(got.shape) - 2
    # tail1 starts from zeros; 0 + x is x (and +0 for -0), as is rank 0's
    # sum with zeros, so rank 0's term starts the sum here.
    lo = gl.sum(gl.where(src == 0, lo_all, 0.0), axis=AX)
    hi = gl.sum(gl.where(src == 0, hi_all, 0.0), axis=AX)
    for p in gl.static_range(1, WORLD):
        lo += gl.sum(gl.where(src == p, lo_all, 0.0), axis=AX)
        hi += gl.sum(gl.where(src == p, hi_all, 0.0), axis=AX)
    return lo.to(gl.bfloat16).to(gl.float32), hi.to(gl.bfloat16).to(gl.float32)


@gluon.jit
def router_stage(
    mb, task, w, pre_gain, h, alpha, beta, RANK: gl.constexpr, STAGE: gl.constexpr
):
    """Router task ``task``: experts ``16 (task % 16) ..`` of one row, or of
    rows ``2 (task // 16)`` and ``+ 1`` of several.

    ``h' = beta b inv + alpha h`` (see attn_tail), so the logits of ``h' g``
    are ``inv`` times those of ``beta b g`` plus those of ``alpha h g``. The
    second part needs only the launch's inputs, so it goes out first; the
    first needs only ``b``, not the norm exchange. Top-8 applies ``inv``.

    Args:
        w: This rank's router columns, FP32 ``[256, 1024]`` (row ``e`` is
            expert ``e``'s weights for the owned columns).
        pre_gain: FP32 ``bf16(w + 1)`` of the pre-FFN norm, owned columns.
        h, alpha, beta: The owned FP32 residual ``[rows, 1024]`` and
            residual-gate coefficients (as tail_stages takes them).
    """
    mb.stamp(STAGE, trace.START)
    if schedule.rows(mb.plan) == 1:
        _router_row(mb, task, w, pre_gain, h, alpha, beta, RANK, STAGE)
    else:
        _router_rows(mb, task, w, pre_gain, h, alpha, beta, RANK, STAGE)
    mb.stamp(STAGE, trace.COMPUTED)
    mb.stamp(STAGE, trace.END)


@gluon.jit
def _router_row(
    mb, task, w, pre_gain, h, alpha, beta, RANK: gl.constexpr, STAGE: gl.constexpr
):
    """:func:`router_stage` of one row. The waves split the columns: each
    wave polls only its 128 values of b (fewer polls of the same lines),
    and one sum crosses the waves."""
    nw: gl.constexpr = gl.num_warps()
    wl: gl.constexpr = gl.BlockedLayout([1, 4], [2, 32], [1, nw], [1, 0])
    e = task * ROUTER_ROWS + gl.arange(0, ROUTER_ROWS, gl.SliceLayout(1, wl))
    k = gl.arange(0, OWNED, gl.SliceLayout(0, wl))
    weights = gl.load(w + e[:, None] * OWNED + k[None, :])
    gain = gl.load(pre_gain + k)
    ahg = gl.load(alpha + k) * gl.load(h + k) * gain
    bg = gl.load(beta + k) * gain
    word2 = gl.sum(weights * ahg[None, :], axis=1).to(gl.int32, bitcast=True)
    for p in gl.static_range(WORLD):
        mb.put_peer("lg2", p, RANK * EXPERTS + e, word2)
    # b of each column in the columns' own layout (wl's): every column polls
    # its pair of red and takes its half, so b needs no layout change.
    src = gl.arange(0, WORLD, gl.SliceLayout(1, wl))[:, None]
    lo, hi = _owned_b(mb, 0, src, k[None, :] // 2)
    mb.stamp(STAGE, trace.READY)
    v = gl.where(k % 2 == 0, lo, hi) * bg
    word = gl.sum(weights * v[None, :], axis=1).to(gl.int32, bitcast=True)
    for p in gl.static_range(WORLD):
        mb.put_peer("lg", p, RANK * EXPERTS + e, word)


@gluon.jit
def _router_rows(
    mb, task, w, pre_gain, h, alpha, beta, RANK: gl.constexpr, STAGE: gl.constexpr
):
    """:func:`router_stage` of a pair of rows. The waves split the experts (2
    each) and the lanes the columns, so a row's sum stays within a wave: no
    LDS round trip per row. The rows' b goes through LDS once."""
    R: gl.constexpr = schedule.rows(mb.plan)
    RP: gl.constexpr = ROW_PAIR
    nw: gl.constexpr = gl.num_warps()
    first = (task // ROUTER_TASKS) * ROW_PAIR
    task = task % ROUTER_TASKS
    wl: gl.constexpr = gl.BlockedLayout([2, 16], [1, 64], [nw, 1], [1, 0])
    e = task * ROUTER_ROWS + gl.arange(0, ROUTER_ROWS, gl.SliceLayout(1, wl))
    k = gl.arange(0, OWNED, gl.SliceLayout(0, wl))
    col = gl.arange(0, RP, gl.SliceLayout(0, wl))[None, :]
    weights = gl.load(w + e[:, None] * OWNED + k[None, :])
    gain = gl.load(pre_gain + k)
    ag = gl.load(alpha + k) * gain
    bg = gl.load(beta + k) * gain
    # The logits of every row, [expert, row], then one put per peer.
    at = mb.row_offset(first + col) + RANK * EXPERTS + e[:, None]
    put = (first + col < R) & (e[:, None] >= 0)
    lg2 = gl.zeros([ROUTER_ROWS, RP], gl.float32, wl)
    for r in gl.static_range(RP):
        ahg = ag * gl.load(h + gl.minimum(first + r, R - 1) * OWNED + k)
        lg2 = gl.where(col == r, gl.sum(weights * ahg[None, :], axis=1)[:, None], lg2)
    # The puts go out from a compact layout (a value a lane): from the
    # column layout every lane holds 32 copies, one store each per peer.
    cl: gl.constexpr = gl.BlockedLayout([1, 1], [32, 2], [nw, 1], [1, 0])
    at_c = gl.convert_layout(at, cl)
    put_c = gl.convert_layout(put, cl)
    word2 = gl.convert_layout(lg2.to(gl.int32, bitcast=True), cl)
    for p in gl.static_range(WORLD):
        mb.put_peer("lg2", p, at_c, word2, mask=put_c)
    # Every row's b, summed here from the ranks' partials (one poll), through
    # LDS into the column layout.
    bl: gl.constexpr = gl.BlockedLayout([1, 1, 1], [1, 1, 64], [1, 1, nw], [2, 1, 0])
    br = gl.arange(0, RP, gl.SliceLayout(1, gl.SliceLayout(2, bl)))[:, None, None]
    bs = gl.arange(0, WORLD, gl.SliceLayout(0, gl.SliceLayout(2, bl)))[None, :, None]
    bj = gl.arange(0, OWNED // 2, gl.SliceLayout(0, gl.SliceLayout(1, bl)))[
        None, None, :
    ]
    lo, hi = _owned_b(mb, first + br, bs, bj)
    mb.stamp(STAGE, trace.READY)
    flat: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    b_smem = gl.allocate_shared_memory(gl.float32, [RP * OWNED], flat)
    b_smem.store(gl.reshape(gl.join(lo, hi), [RP * OWNED]))
    gl.barrier()
    lg = gl.zeros([ROUTER_ROWS, RP], gl.float32, wl)
    for r in gl.static_range(RP):
        v = b_smem.slice(r * OWNED, OWNED).load(gl.SliceLayout(0, wl)) * bg
        lg = gl.where(col == r, gl.sum(weights * v[None, :], axis=1)[:, None], lg)
    word = gl.convert_layout(lg.to(gl.int32, bitcast=True), cl)
    for p in gl.static_range(WORLD):
        mb.put_peer("lg", p, at_c, word, mask=put_c)


# The top-8 asm operands: the ids and the picked scores (lane k holds pass
# k's), the choices' working copy, a VGPR temporary that returns the scores'
# total, 15 SGPR temporaries, then the 2 inputs, and the clobbered SCC (the
# SALU compares and arithmetic write it). One string, element by element: a
# tuple repeats an entry for every element of its operand, so a clobber on the
# 4-element score input would come before its other elements.
_TOP8_CONSTRAINTS = gl.constexpr(
    ",".join(["=&v"] * (1 + 1 + 4 + 1) + ["=&s"] * 15 + ["v"] * (4 + 4) + ["~{scc}"])
)


@gluon.constexpr_function
def _top8_asm(outputs, inputs):
    """The top 8 of a wave's 256 choices (4 a lane, expert 4 lane + j in
    register j): a loop of 8 passes of the wave max (DPP), the lowest expert
    at it (lane masks and s_ff1) and its removal. Pass k writes its id and
    score to lane k. A loop, not 8 copies: the code runs once a launch, so
    its size (instruction fetch), not its arithmetic, sets its time. Then the
    scores' total, in the pairwise order of a lane butterfly (gl.sum's)."""
    (ids,), (picks,), cw, (t,) = outputs[0:4]
    masks = [o[0] for o in outputs[4:8]]
    any_ = outputs[8][0]
    lane, j, k, best, sid, spick = (o[0] for o in outputs[9:15])
    r = [o[0] for o in outputs[15:19]]
    c_in, score = inputs
    nop = "s_nop 1"
    a = [f"v_mov_b32 {d}, {x}" for d, x in zip(cw, c_in)]
    a += [
        f"v_mov_b32 {ids}, 0",
        f"v_mov_b32 {picks}, 0",
        f"s_mov_b32 {k}, 0",
        "top8_${:uid}:",
        f"v_max3_f32 {t}, {cw[0]}, {cw[1]}, {cw[2]}",
        f"v_max_f32 {t}, {t}, {cw[3]}",
    ]
    for ctrl in (
        "quad_perm:[1,0,3,2]",
        "quad_perm:[2,3,0,1]",
        "row_half_mirror",
        "row_mirror",
        "row_bcast:15 row_mask:0xa",
        "row_bcast:31 row_mask:0xc",
    ):
        a += [nop, f"v_max_f32_dpp {t}, {t}, {t} {ctrl}"]
    # A VALU reads an SGPR that a VALU (v_readlane) wrote only 2 wait states
    # later, so the compares read a SALU copy (sid, set again below).
    a += [nop, f"v_readlane_b32 {best}, {t}, 63", f"s_mov_b32 {sid}, {best}"]
    for jj in range(4):
        a += [f"v_cmp_eq_f32_e64 {masks[jj]}, {cw[jj]}, {sid}"]
    a += [
        f"s_or_b64 {any_}, {masks[0]}, {masks[1]}",
        f"s_or_b64 {any_}, {any_}, {masks[2]}",
        f"s_or_b64 {any_}, {any_}, {masks[3]}",
        f"s_ff1_i32_b64 {lane}, {any_}",
        f"s_mov_b32 {j}, 3",
    ]
    for jj in (2, 1, 0):
        a += [f"s_bitcmp1_b64 {masks[jj]}, {lane}", f"s_cselect_b32 {j}, {jj}, {j}"]
    a += [f"s_lshl_b32 {sid}, {lane}, 2", f"s_add_u32 {sid}, {sid}, {j}", "s_nop 3"]
    for jj in range(4):
        a += [f"v_readlane_b32 {r[jj]}, {score[jj]}, {lane}"]
    a += [f"s_mov_b32 {spick}, {r[3]}"]
    for jj in (2, 1, 0):
        a += [f"s_cmp_eq_u32 {j}, {jj}", f"s_cselect_b32 {spick}, {r[jj]}, {spick}"]
    # Lane k takes the id and score (a one-lane mask: v_writelane cannot
    # read two SGPRs on gfx9).
    a += [
        f"s_lshl_b64 {any_}, 1, {k}",
        f"v_mov_b32 {t}, {sid}",
        f"v_cndmask_b32_e64 {ids}, {ids}, {t}, {any_}",
        f"v_mov_b32 {t}, {spick}",
        f"v_cndmask_b32_e64 {picks}, {picks}, {t}, {any_}",
    ]
    # Remove the pick: -inf in register j of lane `lane` (a one-lane mask;
    # a VALU op reads one SGPR on gfx9, so -inf goes in a VGPR).
    a += [f"v_mov_b32 {t}, 0xff800000", f"s_lshl_b64 {any_}, 1, {lane}"]
    for jj in range(4):
        a += [
            f"s_cmp_eq_u32 {j}, {jj}",
            f"s_cselect_b64 {masks[jj]}, {any_}, 0",
            f"v_cndmask_b32_e64 {cw[jj]}, {cw[jj]}, {t}, {masks[jj]}",
        ]
    a += [
        f"s_add_u32 {k}, {k}, 1",
        f"s_cmp_lt_u32 {k}, 8",
        "s_cbranch_scc1 top8_${:uid}",
    ]
    # The total: ((p0 + p1) + (p2 + p3)) + ((p4 + p5) + (p6 + p7)).
    p = r + [best, sid, spick, j]
    a += [nop] + [f"v_readlane_b32 {p[i]}, {picks}, {i}" for i in range(8)]
    a += [
        f"v_mov_b32 {t}, {p[0]}",
        f"v_add_f32 {t}, {p[1]}, {t}",
        f"v_mov_b32 {cw[0]}, {p[2]}",
        f"v_add_f32 {cw[0]}, {p[3]}, {cw[0]}",
        f"v_add_f32 {t}, {t}, {cw[0]}",
        f"v_mov_b32 {cw[0]}, {p[4]}",
        f"v_add_f32 {cw[0]}, {p[5]}, {cw[0]}",
        f"v_mov_b32 {cw[1]}, {p[6]}",
        f"v_add_f32 {cw[1]}, {p[7]}, {cw[1]}",
        f"v_add_f32 {cw[0]}, {cw[0]}, {cw[1]}",
        f"v_add_f32 {t}, {t}, {cw[0]}",
    ]
    return "\n".join(a)


@gluon.jit
def topk_stage(mb, task, bias, STAGE: gl.constexpr):
    """The top-8 of row ``task``'s summed logits (``bias``: FP32 ``[256]``)."""
    mb.stamp(STAGE, trace.START)
    mb = mb.row(task)
    nw: gl.constexpr = gl.num_warps()
    ll: gl.constexpr = gl.BlockedLayout([WORLD, 1], [1, 64], [1, nw], [0, 1])
    src = gl.arange(0, WORLD, gl.SliceLayout(1, ll))
    e = gl.arange(0, EXPERTS, gl.SliceLayout(0, ll))
    # Every wave holds all 256 experts (4 a lane), so each argmax below
    # reduces within a wave: no LDS round trip and barrier per pass.
    wl: gl.constexpr = gl.BlockedLayout([EXPERTS // 64], [64], [nw], [0])
    ew = gl.arange(0, EXPERTS, wl)
    bias_w = gl.load(bias + ew)  # before the poll, not after it
    at = src[:, None] * EXPERTS + e[None, :]
    # The alpha h part goes out at the router's start: poll it first, alone.
    (got2,) = mb.poll_many(("lg2",), (at,))
    # Its per-source rows, while the rest is in flight: this code runs
    # once a launch, so its fetch is off the critical path here.
    parts2 = got2.to(gl.float32, bitcast=True)
    h_parts = ()
    for p in gl.static_range(WORLD):
        h_parts += (gl.sum(gl.where(src[:, None] == p, parts2, 0.0), axis=0),)
    got, ss = mb.poll_many(("lg", "ss1"), (at, sums_index(nw)))
    mb.stamp(STAGE, trace.READY)
    # The norm scales, from the attention tail's partial sums.
    inv, s = norm_scales(ss)
    parts = got.to(gl.float32, bitcast=True)
    logit = gl.zeros([EXPERTS], gl.float32, gl.SliceLayout(0, ll))
    for p in gl.static_range(WORLD):
        mine = src[:, None] == p
        logit += gl.fma(gl.sum(gl.where(mine, parts, 0.0), axis=0), inv, h_parts[p])
    logit = gl.convert_layout(logit * s, wl)
    # An exact division: the scores decide which experts a near tie picks.
    score = 1.0 / (1.0 + gl.exp(-(logit * ROUTER_SCALE)))
    choice = score + bias_w
    # The 8 passes (max, then the lowest expert at it) by hand: the wave
    # reductions Triton emits cost hundreds of cycles each, 3 a pass.
    one: gl.constexpr = gl.BlockedLayout([1], [64], [nw], [0])
    slot = gl.arange(0, 64 * nw, one)
    vt: gl.constexpr = slot.type.with_element_ty(gl.float32)
    # asm: the passes need lane masks in SGPRs (v_cmp, s_ff1, v_readlane by
    # an SGPR lane); Gluon has no ballot or lane read.
    res = gl.inline_asm(
        _top8_asm,
        _TOP8_CONSTRAINTS,
        [choice, score],
        [
            slot.type,
            vt,
            choice.type,
            vt,
            gl.int64,
            gl.int64,
            gl.int64,
            gl.int64,
            gl.int64,
            gl.int32,
            gl.int32,
            gl.int32,
            gl.int32,
            gl.int32,
            gl.int32,
            gl.int32,
            gl.int32,
            gl.int32,
            gl.int32,
        ],
        is_pure=True,
    )
    ids, picked, total = res[0], res[1], res[3]
    first = slot < TOP_K
    # The experts stages read only the ids: publish them first.
    mb.put("route", slot, ids, mask=first)
    weight = picked / gl.where(total == 0.0, 1.0, total)
    mb.stamp(STAGE, trace.COMPUTED)
    mb.put("route", TOP_K + slot, weight.to(gl.int32, bitcast=True), mask=first)
    mb.stamp(STAGE, trace.END)

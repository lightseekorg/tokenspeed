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


"""MXFP4 MoE decode as a persistent per-wave stream (gfx950).

The MoE phase of a persistent decode layer: 256 workgroups x 8 waves, one per
CU, each rank holding a slice of every expert's intermediate (EP1). Every
token is an A row of every expert (MFMA M = 16), so each touched expert's
weights are read once.

Work units, all per wave:

- A gate/up *strip* is (expert, 16 gate/up columns, all K): one 1 KB
  ``buffer_load_dwordx4`` per lane per 128-K step, which is exactly the MFMA
  B operand in the gdot128 layout, plus one E8M0 dword per lane per two
  steps. A *group* is 4 strips = 64 gate/up columns = one 32-value MXFP4
  group of mids; the 4 strips of a group run on one quad of 4 waves.
- A down *unit* is (expert, 64 output columns, all 512 K): 16 chunks of 1 KB
  plus scales, with the expert's mids as the A operand.

Each wave keeps ``RING`` weight chunks in flight in registers and never
drains them: the chunk it just consumed is refilled ``RING`` steps ahead,
into the next strip or unit past the current one's end. Nothing a stream
wave waits for is a vector load issued behind its weights (vector memory
completes in issue order): item metadata comes from LDS, readiness polls are
scalar loads, and the quad exchange runs in LDS without a barrier
(``persistent/wave.py``).

The plan is closed-form from the routed ids, in every workgroup: whole
experts go to XCDs round-robin and the ``E % 8`` leftover experts are split by
group over all XCDs, so every XCD carries the same bytes; within an XCD the
16 tail-role workgroups (slots 0 and 16-30) get whole groups worth half a
full workgroup's bytes and no down units, so they finish early, and the 16
others take the remaining groups, then down units, earliest-ready expert
first. :func:`native_moe_plan` is the same plan on the host.

Numerics are today's direct decode (``decode_common``, ``decode_stage1``,
``quantize_gluon``): the same per-column MFMA chain over K, SwiGLU, 32-value
MXFP4 quantization of the mids and down chain, so the output is bitwise equal.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch
from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.decode_common import (
    _cdna4_swizzled_mxfp4_scale_offset,
    _direct_mxfp4_mfma,
    _direct_mxfp4_mfma_layouts,
    _gluon_dot_preshuffled_w_offset,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.decode_stage1 import (
    _swiglu_interleaved,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.quantize_gluon import (
    _mxfp4_quantize_tile_in_layout,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.scale_layout import (
    pack_cdna4_mxfp4_scale_lanes,
    unswizzle_cdna4_mxfp4_scale,
)
from tokenspeed_kernel_amd.ops.gfx950.persistent import (
    ENTRY_WORDS,
    ERROR_WORDS,
    FIRST_KERNEL_SITE,
    GRID,
    enter_check,
    enter_issue,
    fail_stop,
    persistent_compile_options,
    poll_check,
    poll_tags16,
    release_drain,
    vote,
)
from tokenspeed_kernel_amd.ops.gfx950.persistent.arena import UncachedBuffer
from tokenspeed_kernel_amd.ops.gfx950.persistent.wave import (
    LDS_SPIN_TRIES,
    after,
    lds_spin_bounded,
    lds_store_count,
    opaque_zero,
    wave_scalar,
)

__all__ = [
    "GROUPS_PER_EXPERT",
    "TAG_WORDS",
    "TAG_WORDS_PER_EXPERT",
    "UNITS_PER_EXPERT",
    "MOE_SINK_LAYOUT",
    "MoeBuffers",
    "NativeMoeState",
    "NativeMoeWeights",
    "gluon_mxfp4_moe_native_stream_gfx950",
    "launch_gluon_mxfp4_moe_native_stream_gfx950",
    "native_moe_plan",
    "native_moe_weights",
    "prefetch_moe",
    "run_moe",
    "store_routes_sink",
]

XCDS = 8
SLOTS = 32
# Tail-role slots per XCD: 0 and TAIL_FIRST..TAIL_LAST (16 of 32).
TAIL_FIRST = 16
TAIL_LAST = 30
HIDDEN = 4096
INTER = 512
GATE_UP = 2 * INTER
GROUP_COLUMNS = 64
GROUPS_PER_EXPERT = GATE_UP // GROUP_COLUMNS  # 16
UNIT_COLUMNS = 64
UNITS_PER_EXPERT = HIDDEN // UNIT_COLUMNS  # 64
# Group tags: one word per (group, strip) — each wave publishes its own rows.
TAG_WORDS_PER_EXPERT = GROUPS_PER_EXPERT * 4
# Distinct experts of one launch: at most 16 rows x top 8.
MAX_RANKS = 128
# Group tags live in two copies, indexed by expert rank. A launch polls and
# publishes copy (epoch & 1) and zeroes the other one for the next launch.
# Launch tags repeat after 2**32 - 1 launches, and a rank no launch touches
# for that long would otherwise still hold a matching tag; zeroed slots never
# match (tags are never 0).
TAG_COPY_WORDS = MAX_RANKS * TAG_WORDS_PER_EXPERT
TAG_WORDS = 2 * TAG_COPY_WORDS
# Bytes of one group (64 gate/up columns x K, plus scales) and one unit.
GROUP_BYTES = GROUP_COLUMNS * HIDDEN // 2 + GROUP_COLUMNS * HIDDEN // 32
UNIT_BYTES = UNIT_COLUMNS * INTER // 2 + UNIT_COLUMNS * INTER // 32

# Failure site of a down unit's readiness wait (detail: the expert's rank).
SITE_MOE_READY = FIRST_KERNEL_SITE + 8
_SITE_MOE_READY = gl.constexpr(SITE_MOE_READY)
# Failure site of a quad's mids exchange wait (detail: ``(rank * 16 + group)
# * 2 + quad`` of the group being exchanged).
SITE_MOE_QUAD = FIRST_KERNEL_SITE + 9
_SITE_MOE_QUAD = gl.constexpr(SITE_MOE_QUAD)
_LDS_SPIN_TRIES = gl.constexpr(LDS_SPIN_TRIES)
_XCDS = gl.constexpr(XCDS)
_HIDDEN = gl.constexpr(HIDDEN)
_TAIL_FIRST = gl.constexpr(TAIL_FIRST)
_TAIL_LAST = gl.constexpr(TAIL_LAST)
_GROUPS = gl.constexpr(GROUPS_PER_EXPERT)
_UNITS = gl.constexpr(UNITS_PER_EXPERT)
_TAGS = gl.constexpr(TAG_WORDS_PER_EXPERT)
_TAG_COPY_WORDS = gl.constexpr(TAG_COPY_WORDS)
_MAX_RANKS = gl.constexpr(MAX_RANKS)
_K_STEPS = gl.constexpr(HIDDEN // 128)  # 32 gate/up steps
_DN_STEPS = gl.constexpr(INTER // 128)  # 4 down steps
_DN_CHUNKS = gl.constexpr(4 * (INTER // 128))  # 16
_MID_ROW = gl.constexpr(INTER // 2)  # packed MXFP4 bytes per mid row
_MID_SCALES = gl.constexpr(INTER // 32)  # E8M0 bytes per mid row


# ----------------------------------------------------------------------------
# The plan, on the host (the device derives the same numbers per workgroup)
# ----------------------------------------------------------------------------


def _tail_index(slot):
    """Tail-role index 0-15 of a slot, or -1 for a full slot."""
    if slot == 0:
        return 0
    if TAIL_FIRST <= slot <= TAIL_LAST:
        return slot - TAIL_FIRST + 1
    return -1


def _full_index(slot):
    """Full-role index 0-15 of a slot, or -1 for a tail slot."""
    if _tail_index(slot) >= 0:
        return -1
    return slot - 1 if slot < TAIL_FIRST else slot - 1 - (TAIL_LAST - TAIL_FIRST + 1)


def _ready_order(p, t, n_whole):
    """Whole-expert index of the p-th down expert, earliest ready first.

    Tail workgroups run whole experts 0..t-1 one per round, full workgroups
    experts t.. one per round, so round r makes experts r and t + r ready.
    """
    m = min(t, n_whole - t)
    if p < 2 * m:
        return p // 2 + (p % 2) * t
    rest = p - 2 * m
    return m + rest if t > n_whole - t else t + m + rest


def _plan_counts(experts, tail_num, tail_den):
    whole = experts - experts % XCDS
    leftover = experts % XCDS
    n_whole = whole // XCDS
    groups = GROUPS_PER_EXPERT * n_whole + 2 * leftover
    units = UNITS_PER_EXPERT * n_whole + 8 * leftover
    # In unit-sized pieces (a group is 8 units of bytes): 16 tail workgroups
    # at w = tail_num / tail_den of a full one, rounded to whole groups.
    pieces = 8 * groups + units
    denom = 8 * 16 * (tail_den + tail_num)
    t = (2 * pieces * tail_num + denom) // (2 * denom)
    t = min(t, groups // 16)
    return whole, n_whole, groups, units, t


def _full_units(groups, units, t, b):
    """Down-unit range [first, end) of full index ``b`` within its XCD."""
    rest = groups - 16 * t
    total = units + 8 * rest
    first = 0
    need_b = 0
    for bb in range(16):
        groups_bb = rest // 16 + (1 if bb < rest % 16 else 0)
        target = total // 16 + (1 if bb < total % 16 else 0)
        need = max(target - 8 * groups_bb, 0)
        if bb < b:
            first += need
        if bb == b:
            need_b = need
    first = min(first, units)
    end = units if b == 15 else min(first + need_b, units)
    return first, end


def native_moe_plan(topk_ids, tail_num=1, tail_den=2):
    """The closed-form plan on the host.

    Args:
        topk_ids: ``(rows, top_k)`` integer expert ids.
        tail_num, tail_den: A tail-role workgroup's bytes as a fraction of a
            full one's (``prefetch_moe``'s ``TAIL_NUM`` / ``TAIL_DEN``).

    Returns:
        A list over the 256 logical workgroups (``cta = slot * 8 + xcd``) of
        lists of work items in stream order per workgroup: ``("ug", expert,
        group)`` then ``("dn", expert, unit)``, with global expert ids.
    """
    ids = np.asarray(topk_ids).reshape(-1)
    distinct = np.unique(ids)
    experts = len(distinct)
    whole, n_whole, groups, units, t = _plan_counts(experts, tail_num, tail_den)
    plan = [[] for _ in range(XCDS * SLOTS)]

    def group_item(x, q):
        if q < GROUPS_PER_EXPERT * n_whole:
            rank = XCDS * (q // GROUPS_PER_EXPERT) + x
            return int(distinct[rank]), q % GROUPS_PER_EXPERT
        r = q - GROUPS_PER_EXPERT * n_whole
        return int(distinct[whole + r // 2]), 2 * x + r % 2

    def unit_item(x, v):
        if v < UNITS_PER_EXPERT * n_whole:
            j = _ready_order(v // UNITS_PER_EXPERT, t, n_whole)
            return int(distinct[XCDS * j + x]), v % UNITS_PER_EXPERT
        r = v - UNITS_PER_EXPERT * n_whole
        return int(distinct[whole + r // 8]), 8 * x + r % 8

    for slot in range(SLOTS):
        a, b = _tail_index(slot), _full_index(slot)
        for x in range(XCDS):
            items = plan[slot * XCDS + x]
            if a >= 0:
                qs = [a + 16 * k for k in range(t)]
            else:
                qs = list(range(16 * t + b, groups, 16))
            items += [("ug",) + group_item(x, q) for q in qs]
            if b >= 0:
                first, end = _full_units(groups, units, t, b)
                items += [("dn",) + unit_item(x, v) for v in range(first, end)]
    return plan


# ----------------------------------------------------------------------------
# Device: plan, per workgroup and per wave
# ----------------------------------------------------------------------------


@gluon.jit
def _stage_ids(ids_ptr, rank_smem, ids_t_smem, ROWS: gl.constexpr, TOPK: gl.constexpr):
    """Sorted distinct experts into ``rank_smem`` (rank -> expert id) and the
    ids transposed into ``ids_t_smem`` (``[TOPK, 16]``, -1 past ``ROWS``).
    Returns the number of distinct experts. Runs once at entry (workgroup-wide
    reductions, before the barrier that ends the entry).
    """
    PICKS: gl.constexpr = ROWS * TOPK
    L2: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [gl.num_warps(), 1], [1, 0])
    ra = gl.arange(0, PICKS, layout=gl.SliceLayout(1, L2))
    rb = gl.arange(0, PICKS, layout=gl.SliceLayout(0, L2))
    ida = gl.load(ids_ptr + ra)
    idb = gl.load(ids_ptr + rb)
    same = ida[:, None] == idb[None, :]
    # The first pick of each expert stands for it.
    first_b = gl.max((same & (ra[:, None] < rb[None, :])).to(gl.int32), axis=0) == 0
    first_a = gl.max((same & (rb[None, :] < ra[:, None])).to(gl.int32), axis=1) == 0
    below = (first_a[:, None] & (ida[:, None] < idb[None, :])).to(gl.int32)
    rank_b = gl.sum(below, axis=0)
    # rank -> expert: the first pick whose rank is i.
    sel = (first_b[None, :] & (rank_b[None, :] == ra[:, None])).to(gl.int32)
    rank_smem.store(gl.sum(sel * idb[None, :], axis=1))
    # The count comes out of a cross-wave reduction in a VGPR; move it to an
    # SGPR so the plan's trip counts and branches stay scalar (a divergent
    # loop would merge its waits conservatively).
    LANE: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    experts = wave_scalar(
        gl.full([64 * gl.num_warps()], 0, gl.int32, LANE)
        + gl.sum(first_b.to(gl.int32), axis=0)
    )
    TL: gl.constexpr = gl.BlockedLayout(
        [1, 1], [TOPK, 64 // TOPK], [1, gl.num_warps()], [1, 0]
    )
    tj = gl.arange(0, TOPK, layout=gl.SliceLayout(1, TL))[:, None]
    tr = gl.arange(0, 16, layout=gl.SliceLayout(0, TL))[None, :]
    ids_t_smem.store(gl.load(ids_ptr + tr * TOPK + tj, mask=tr < ROWS, other=-1))
    return experts


@gluon.jit
def _routes(ids_smem, expert, token, TOPK: gl.constexpr):
    """Route ``token * TOPK + slot`` of each row of ``token`` (a tensor of
    token rows, laid out like the result) for ``expert``, or -1 where the
    token does not pick it. ``ids_smem[j]`` holds slot ``j``'s id per row, in
    the same row order. Reads LDS only."""
    route = gl.full(token.shape, -1, gl.int32, token.type.layout)
    by_slot = ids_smem.reinterpret(
        gl.int32, ids_smem.shape, gl.SwizzledSharedLayout(1, 1, 1, order=[0])
    )
    for j in gl.static_range(TOPK):
        hit = by_slot.index(j).load(token.type.layout) == expert
        route = gl.where(hit, token * TOPK + j, route)
    return route


@gluon.jit
def _counts(experts, TAIL_NUM: gl.constexpr, TAIL_DEN: gl.constexpr):
    # Device form of ``_plan_counts``.
    whole = experts - experts % _XCDS
    leftover = experts % _XCDS
    n_whole = whole // _XCDS
    groups = _GROUPS * n_whole + 2 * leftover
    units = _UNITS * n_whole + 8 * leftover
    pieces = 8 * groups + units
    denom: gl.constexpr = 8 * 16 * (TAIL_DEN + TAIL_NUM)
    t = gl.minimum((2 * pieces * TAIL_NUM + denom) // (2 * denom), groups // 16)
    return whole, n_whole, groups, units, t


@gluon.jit
def _full_units_dev(groups, units, t, b):
    # Device form of ``_full_units`` (scalar loop over the 16 full slots).
    rest = groups - 16 * t
    total = units + 8 * rest
    first = units * 0
    need_b = units * 0
    for bb in gl.static_range(16):
        groups_bb = rest // 16 + (bb < rest % 16).to(gl.int32)
        target = total // 16 + (bb < total % 16).to(gl.int32)
        need = gl.maximum(target - 8 * groups_bb, 0)
        first += gl.where(bb < b, need, 0)
        need_b += gl.where(bb == b, need, 0)
    first = gl.minimum(first, units)
    end = gl.where(b == 15, units, gl.minimum(first + need_b, units))
    return first, end


@gluon.jit
def _group_item(q, xcd, whole, n_whole):
    # (rank, group) of the XCD's q-th group; device form of group_item.
    in_whole = q < _GROUPS * n_whole
    r = q - _GROUPS * n_whole
    rank = gl.where(in_whole, _XCDS * (q // _GROUPS) + xcd, whole + r // 2)
    group = gl.where(in_whole, q % _GROUPS, 2 * xcd + r % 2)
    return rank, group


@gluon.jit
def _unit_item(v, xcd, whole, n_whole, t):
    # (rank, unit) of the XCD's v-th down unit; device form of unit_item.
    in_whole = v < _UNITS * n_whole
    p = v // _UNITS
    m = gl.minimum(t, n_whole - t)
    rest = p - 2 * m
    j = gl.where(
        p < 2 * m,
        p // 2 + (p % 2) * t,
        gl.where(t > n_whole - t, m + rest, t + m + rest),
    )
    r = v - _UNITS * n_whole
    rank = gl.where(in_whole, _XCDS * j + xcd, whole + r // 8)
    unit = gl.where(in_whole, v % _UNITS, 8 * xcd + r % 8)
    return rank, unit


@gluon.aggregate
class MoeWave:
    """One wave's share of the plan (per-wave scalars, see ``wave_scalar``)."""

    wave: gl.tensor
    quad: gl.tensor
    strip: gl.tensor
    xcd: gl.tensor
    whole: gl.tensor
    n_whole: gl.tensor
    t: gl.tensor
    q0: gl.tensor  # first group of this wave's quad; the next ones are 32 apart
    n_ug: gl.tensor
    v0: gl.tensor  # first down unit of this wave
    n_dn: gl.tensor

    @gluon.constexpr_function
    def __init__(self, wave, quad, strip, xcd, whole, n_whole, t, q0, n_ug, v0, n_dn):
        self.wave = wave
        self.quad = quad
        self.strip = strip
        self.xcd = xcd
        self.whole = whole
        self.n_whole = n_whole
        self.t = t
        self.q0 = q0
        self.n_ug = n_ug
        self.v0 = v0
        self.n_dn = n_dn


@gluon.jit
def _plan_wave(
    xcd,
    slot,
    experts,
    TAIL_NUM: gl.constexpr,
    TAIL_DEN: gl.constexpr,
    QUAD_MAP: gl.constexpr,
):
    whole, n_whole, groups, units, t = _counts(experts, TAIL_NUM, TAIL_DEN)
    is_tail = (slot == 0) | ((slot >= _TAIL_FIRST) & (slot <= _TAIL_LAST))
    a = gl.where(slot == 0, 0, slot - _TAIL_FIRST + 1)
    b = gl.where(
        slot < _TAIL_FIRST, slot - 1, slot - 1 - (_TAIL_LAST - _TAIL_FIRST + 1)
    )
    # This workgroup's groups: q = start + 16 k, k < count.
    start = gl.where(is_tail, a, 16 * t + b)
    count = gl.where(is_tail, t, gl.maximum(groups - 16 * t - b + 15, 0) // 16)
    first, end = _full_units_dev(groups, units, t, b)
    first = gl.where(is_tail, 0, first)
    end = gl.where(is_tail, 0, end)
    # Per wave: quad and strip; the quad takes every other group of the list.
    WL: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    wave = wave_scalar(gl.arange(0, 64 * gl.num_warps(), WL) // 64)
    if QUAD_MAP == 0:
        quad = wave // 4
        strip = wave % 4
    else:
        quad = wave % 2
        strip = wave // 2
    n_ug = gl.maximum(count - quad + 1, 0) // 2
    v0 = first + wave
    n_dn = gl.maximum(end - v0 + 7, 0) // 8
    return MoeWave(
        wave, quad, strip, xcd, whole, n_whole, t, start + 16 * quad, n_ug, v0, n_dn
    )


@gluon.jit
def _token_of_quant_row(rho, QUAD_MAP: gl.constexpr):
    """Token row held by quant-exchange row ``rho`` (inverse of ``_lds_row``)."""
    wave = rho // 4
    if QUAD_MAP == 0:
        strip = wave % 4
    else:
        strip = wave // 2
    return 4 * strip + rho % 4


@gluon.jit
def _lds_row(quad, token_row, QUAD_MAP: gl.constexpr):
    """Row of the quant exchange holding (quad, token row): rows 4w..4w+3
    belong to wave w, so a wave quantizes only its own quad's group."""
    if QUAD_MAP == 0:
        wave = 4 * quad + token_row // 4
    else:
        wave = 2 * (token_row // 4) + quad
    return 4 * wave + token_row % 4


# ----------------------------------------------------------------------------
# Device: the stream
# ----------------------------------------------------------------------------


@gluon.aggregate
class MoeBuffers:
    """Weights, activations and the mids hand-off of one MoE call."""

    xq: gl.tensor
    xs: gl.tensor
    w13: gl.tensor
    s13: gl.tensor
    s13_lanes: gl.tensor  # lane-major gate/up scales (SCALE_LANES), int32 words
    w2: gl.tensor
    s2: gl.tensor
    mids: gl.tensor
    mids_s: gl.tensor
    tags: gl.tensor
    stride_xslin: gl.tensor
    stride_xsnb: gl.tensor
    stride_w13: gl.tensor
    stride_s13: gl.tensor
    stride_s13lin: gl.tensor
    stride_s13nb: gl.tensor
    stride_w2: gl.tensor
    stride_s2: gl.tensor
    stride_s2lin: gl.tensor
    stride_s2nb: gl.tensor

    @gluon.constexpr_function
    def __init__(
        self,
        xq,
        xs,
        w13,
        s13,
        s13_lanes,
        w2,
        s2,
        mids,
        mids_s,
        tags,
        stride_xslin,
        stride_xsnb,
        stride_w13,
        stride_s13,
        stride_s13lin,
        stride_s13nb,
        stride_w2,
        stride_s2,
        stride_s2lin,
        stride_s2nb,
    ):
        self.xq = xq
        self.xs = xs
        self.w13 = w13
        self.s13 = s13
        self.s13_lanes = s13_lanes
        self.w2 = w2
        self.s2 = s2
        self.mids = mids
        self.mids_s = mids_s
        self.tags = tags
        self.stride_xslin = stride_xslin
        self.stride_xsnb = stride_xsnb
        self.stride_w13 = stride_w13
        self.stride_s13 = stride_s13
        self.stride_s13lin = stride_s13lin
        self.stride_s13nb = stride_s13nb
        self.stride_w2 = stride_w2
        self.stride_s2 = stride_s2
        self.stride_s2lin = stride_s2lin
        self.stride_s2nb = stride_s2nb


# The layout of a down unit's result as handed to the sink: [16 rows, 128
# columns], wave w owns columns 16w..16w+15 (its own unit), and lane l of a
# wave holds row l % 16 and 4 consecutive columns from 4 (l // 16), i.e. two
# whole bf16 pairs. It is the transposed MFMA result layout.
MOE_SINK_LAYOUT = gl.constexpr(gl.BlockedLayout([1, 4], [16, 4], [1, 8], [0, 1]))


@gluon.constexpr_function
def _layouts():
    # MFMA 16x16x128 (transposed result: a lane holds one row, 4 consecutive
    # columns), warps along N: wave w owns columns 16w..16w+15.
    mfma, dot_a, dot_b, a_scale, b_scale = _direct_mxfp4_mfma_layouts(16, 128, 4, 8)
    rows = MOE_SINK_LAYOUT  # == mfma, as a blocked layout
    lane = gl.BlockedLayout([1], [64], [8], [0])  # one element per lane
    quant = gl.BlockedLayout(
        [1, 8], [4, 16], [8, 1], [1, 0]
    )  # [32, 32]: wave w rows 4w..4w+3
    return mfma, dot_a, dot_b, a_scale, b_scale, rows, lane, quant


@gluon.jit
def _expert_at(rank_smem, rank):
    """Expert id of ``rank`` (a per-wave scalar) from the shared rank table."""
    PICKS: gl.constexpr = rank_smem.shape[0]
    one: gl.constexpr = gl.BlockedLayout([1, 1], [64, 1], [8, 1], [1, 0])
    rows = rank_smem.reinterpret(
        gl.int32, [PICKS, 1, 1], gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    )
    return wave_scalar(rows.index(rank).load(one))


@gluon.jit
def _line_column(pos, st: gl.constexpr):
    """Column (within a 64-column down unit) that MFMA output position
    ``pos`` (0-15) of strip ``st`` computes when a unit's strips are mapped
    to whole lines. Lane ``k = pos // 4`` holds positions ``4 k + j``; strips
    0 and 2 give it the even and odd columns of ``base(k) + 0..7``, strips 1
    and 3 those of ``base(k) + 16..23``, with ``base(k) = 32 (k // 2) + 8 (k
    % 2)``. Strips ``st`` and ``st ^ 1`` stay 16 columns apart, so they share
    their E8M0 dwords as the plain mapping does."""
    k = pos // 4
    return 32 * (k // 2) + 8 * (k % 2) + 16 * (st % 2) + st // 2 + 2 * (pos % 4)


@gluon.jit
def _strip_offsets(
    expert,
    column,
    w_stride,
    s_stride,
    s_lin,
    s_nb,
    N_PHYS: gl.constexpr,
    LINE_STRIP: gl.constexpr,
):
    """Lane offsets of a 16-column strip's first K step and the per-step
    advances, all from ``decode_common``'s layout functions.

    ``LINE_STRIP`` -1 computes columns ``column + 0..15``; 0-3 computes strip
    ``LINE_STRIP`` of the 64-column unit at ``column`` under the whole-line
    mapping (:func:`_line_column`).

    Returns ``(w, w_pair, w_odd, s, s_pair)``: step ``kt`` reads weight bytes
    at ``w + (kt // 2) * w_pair + (kt % 2) * w_odd`` and its E8M0 byte at
    ``s + (kt // 2) * s_pair + (kt % 2) * 2``; the swizzle keeps a step
    pair's two bytes in one dword. Advances are per-wave scalars.
    """
    lb: gl.constexpr = _layouts()[2]
    lsb: gl.constexpr = _layouts()[4]
    bk = gl.arange(0, 64, layout=gl.SliceLayout(1, lb))[:, None]
    bn = gl.arange(0, 128, layout=gl.SliceLayout(0, lb))[None, :]
    # Strips start on 16-column boundaries; telling the compiler keeps every
    # lane's 16 bytes one dwordx4.
    column = gl.multiple_of(column, 16)
    w_expert = expert.to(gl.int64) * w_stride
    if LINE_STRIP < 0:
        n_col = column + bn % 16
    else:
        n_col = column + _line_column(bn % 16, LINE_STRIP)
    w0 = _gluon_dot_preshuffled_w_offset(w_expert, bk, n_col, N_PHYS)
    w1 = _gluon_dot_preshuffled_w_offset(w_expert, 64 + bk, n_col, N_PHYS)
    w2 = _gluon_dot_preshuffled_w_offset(w_expert, 128 + bk, n_col, N_PHYS)
    bsn = gl.arange(0, 128, layout=gl.SliceLayout(1, lsb))[:, None]
    bsk = gl.arange(0, 4, layout=gl.SliceLayout(0, lsb))[None, :]
    s_expert = expert.to(gl.int64) * s_stride
    if LINE_STRIP < 0:
        s_col = column + bsn % 16
    else:
        s_col = column + _line_column(bsn % 16, LINE_STRIP)
    s0 = _cdna4_swizzled_mxfp4_scale_offset(s_expert, s_col, bsk, s_lin, s_nb)
    s2 = _cdna4_swizzled_mxfp4_scale_offset(s_expert, s_col, 8 + bsk, s_lin, s_nb)
    # The advances are whole 1 KB chunks and whole scale dwords (asserted by
    # the codegen test's dwordx4 / dword loads).
    return (
        w0.to(gl.int32),
        gl.multiple_of(wave_scalar((w2 - w0).to(gl.int32)), 1024),
        gl.multiple_of(wave_scalar((w1 - w0).to(gl.int32)), 1024),
        s0.to(gl.int32),
        gl.multiple_of(wave_scalar((s2 - s0).to(gl.int32)), 4),
    )


@gluon.jit
def _load_chunk(
    desc, base, pair, odd, kt: gl.constexpr, live, ADDR_SOFFSET: gl.constexpr
):
    """One 1 KB weight chunk (the MFMA B operand of step ``kt``), nontemporal.

    ``ADDR_SOFFSET`` puts the step's advance in the scalar offset register
    instead of adding it to every lane's offset.
    """
    policy: gl.constexpr = gl.amd.cdna3.CachePolicy(nontemporal=True)
    step = (kt // 2) * pair + (kt % 2) * odd
    mask = None if live is None else (base == base) & live
    if ADDR_SOFFSET:
        return gl.amd.cdna3.buffer.load(
            desc, base, mask=mask, soffset=step, cache_policy=policy
        )
    return gl.amd.cdna3.buffer.load(desc, base + step, mask=mask, cache_policy=policy)


@gluon.jit
def _load_scale_dword(desc, base, pair, p: gl.constexpr, live):
    """The E8M0 dword of step pair ``p`` (bytes ``base + p * pair``).
    Cached normally: the two strips of a 32-column block read the same
    lines."""
    mask = None if live is None else (base == base) & live
    return gl.amd.cdna3.buffer.load(desc, (base + p * pair) // 4, mask=mask)


@gluon.jit
def _scale_byte(dword, base, kt: gl.constexpr):
    """Step ``kt``'s E8M0 byte out of its pair's dword."""
    shift = ((base + (kt % 2) * 2) % 4) * 8
    return ((dword >> shift) & 255).to(gl.uint8)


@gluon.constexpr_function
def _lanes_layout():
    """The B-scale layout of the MFMA (a lane holds one (column, K quad)) with
    4 consecutive dwords per lane added as the last dimension."""
    return gl.DistributedLinearLayout(
        reg_bases=[[0, 0, 1], [0, 0, 2]],
        lane_bases=[[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [0, 1, 0], [0, 2, 0]],
        warp_bases=[[16, 0, 0], [32, 0, 0], [64, 0, 0]],
        block_bases=[],
        shape=[128, 4, 4],
    )


@gluon.jit
def _lanes_base(expert, column, N_PHYS: gl.constexpr, K_STEPS: gl.constexpr):
    """Dword offsets of a 16-column strip's lane-major E8M0 scales (see
    ``scale_layout.pack_cdna4_mxfp4_scale_lanes``): lane (k quad, column)
    holds its K steps' bytes contiguously, 16 steps per dwordx4."""
    L: gl.constexpr = _lanes_layout()
    n = gl.arange(0, 128, layout=gl.SliceLayout(1, gl.SliceLayout(2, L)))[:, None, None]
    kq = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(2, L)))[None, :, None]
    j = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(1, L)))[None, None, :]
    strip = expert * (N_PHYS // 16) + gl.multiple_of(column, 16) // 16
    return (strip * 64 + kq * 16 + n % 16) * (K_STEPS // 4) + j


@gluon.jit
def _load_lanes(desc, base, half: gl.constexpr, live):
    """Steps 16 half .. 16 half + 15 of a strip's lane-major scales: one
    dwordx4 per lane, nontemporal like the weights (each byte is read once)."""
    policy: gl.constexpr = gl.amd.cdna3.CachePolicy(nontemporal=True)
    mask = None if live is None else (base == base) & live
    return gl.amd.cdna3.buffer.load(
        desc, base + half * 4, mask=mask, cache_policy=policy
    )


@gluon.jit
def _lanes_byte(quad, kt: gl.constexpr):
    """Step ``kt``'s E8M0 byte (B-scale layout) out of its half's dwordx4."""
    pairs = quad.reshape([128, 4, 2, 2])
    lo, hi = gl.split(pairs)
    pick = hi if ((kt % 16) // 4) % 2 else lo
    d0, d1 = gl.split(pick)
    dword = d1 if ((kt % 16) // 4) // 2 else d0
    return ((dword >> ((kt % 4) * 8)) & 255).to(gl.uint8)


@gluon.jit
def _group_of(plan, k, rank_smem):
    """This wave's k-th group: (expert id, rank, group index, leftover?)."""
    # The quad takes every other group of the workgroup's list (16 apart).
    q = plan.q0 + 32 * k
    rank, group = _group_item(q, plan.xcd, plan.whole, plan.n_whole)
    expert = _expert_at(rank_smem, rank)
    return expert, rank, group, rank >= plan.whole


@gluon.jit
def _unit_of(plan, u, rank_smem):
    """This wave's u-th down unit: (expert id, rank, unit index)."""
    rank, unit = _unit_item(plan.v0 + 8 * u, plan.xcd, plan.whole, plan.n_whole, plan.t)
    expert = _expert_at(rank_smem, rank)
    return expert, rank, unit


@gluon.jit
def _publish_tags(tags_d, word, tag, live, cross_xcd, dep):
    """Store ``tag`` at word ``word`` of ``tags_d`` from one lane of this wave,
    computed from ``dep``: the store issues only after the wave waited for
    the load ``dep`` came from, which also covers this wave's earlier stores
    (one wave's vector memory operations complete in issue order)."""
    mfma: gl.constexpr = _layouts()[0]
    r = gl.arange(0, 16, layout=gl.SliceLayout(1, mfma))[:, None]
    c = gl.arange(0, 128, layout=gl.SliceLayout(0, mfma))[None, :]
    value = after(
        gl.full([16, 128], 0, gl.int32, mfma) + tag, dep.to(gl.int32, bitcast=True)
    )
    _store(
        tags_d, value, word + r * 0 + c * 0, live & (r == 0) & (c % 16 == 0), cross_xcd
    )


@gluon.jit
def prefetch_moe(
    ctx,
    ids_ptr,
    bufs,
    ROWS: gl.constexpr,
    TOPK: gl.constexpr,
    N_PHYS1: gl.constexpr,
    RING: gl.constexpr,
    QUAD_MAP: gl.constexpr,
    TAIL_NUM: gl.constexpr,
    TAIL_DEN: gl.constexpr,
    ADDR_SOFFSET: gl.constexpr,
    SCALE_LANES: gl.constexpr,
):
    """Plan this workgroup's MoE work and issue its first weights.

    ``ctx`` is the launch (``xcd``, ``slot``); ``ids_ptr`` holds the
    ``(ROWS, TOPK)`` routed expert ids, published before the call. Weights
    depend only on the ids, so they go out before the activations exist.
    Ends with a workgroup barrier (the ids tables are shared).

    Returns the handles :func:`run_moe` continues from.
    """
    PICKS: gl.constexpr = ROWS * TOPK
    rank_smem = gl.allocate_shared_memory(
        gl.int32, [PICKS], gl.SwizzledSharedLayout(1, 1, 1, order=[0])
    )
    ids_t = gl.allocate_shared_memory(
        gl.int32, [TOPK, 16], gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    )
    experts = _stage_ids(ids_ptr, rank_smem, ids_t, ROWS, TOPK)
    # The ids again in quant-exchange row order: ids_q[j, rho] = ids[token(rho), j].
    ids_q = gl.allocate_shared_memory(
        gl.int32, [TOPK, 32], gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    )
    QJ: gl.constexpr = gl.BlockedLayout([1, 1], [2, 32], [gl.num_warps(), 1], [1, 0])
    qj = gl.arange(0, TOPK, layout=gl.SliceLayout(1, QJ))[:, None]
    rho = gl.arange(0, 32, layout=gl.SliceLayout(0, QJ))[None, :]
    token = _token_of_quant_row(rho, QUAD_MAP)
    ids_q.store(gl.load(ids_ptr + token * TOPK + qj, mask=token < ROWS, other=-1))
    gl.barrier()
    plan = _plan_wave(ctx.xcd, ctx.slot, experts, TAIL_NUM, TAIL_DEN, QUAD_MAP)
    live = plan.n_ug > 0
    expert, rank, group, left = _group_of(plan, 0, rank_smem)
    w, w_pair, w_odd, s, s_pair = _strip_offsets(
        expert,
        group * 64 + plan.strip * 16,
        bufs.stride_w13,
        bufs.stride_s13,
        bufs.stride_s13lin,
        bufs.stride_s13nb,
        N_PHYS1,
        -1,
    )
    wdesc = gl.amd.cdna3.buffer.make_buffer_descriptor(bufs.w13, 0x7FFFFFF0)
    sdesc = gl.amd.cdna3.buffer.make_buffer_descriptor(bufs.s13, 0x7FFFFFF0)
    ring = ()
    sring = ()
    if SCALE_LANES:
        # Steps 0-15's scales; steps 16-31 go out mid-strip (slot 1 starts as
        # a copy, not a load).
        ldesc = gl.amd.cdna3.buffer.make_buffer_descriptor(bufs.s13_lanes, 0x7FFFFFF0)
        half0 = _load_lanes(
            ldesc,
            _lanes_base(expert, group * 64 + plan.strip * 16, N_PHYS1, _K_STEPS),
            0,
            live,
        )
        sring = (half0, half0)
    for j in gl.static_range(RING):
        # Ring order (slot 0 first): the loop's first wait is the minimum
        # over this entry and its back edge.
        if not SCALE_LANES and j % 2 == 0:
            sring = sring + (_load_scale_dword(sdesc, s, s_pair, j // 2, live),)
        ring = ring + (_load_chunk(wdesc, w, w_pair, w_odd, j, live, ADDR_SOFFSET),)
        gl.amd.hint.sched_barrier()
    return plan, ring, sring, rank_smem, ids_t, ids_q


@gluon.jit
def _bf16_pairs(even, odd):
    """bf16 column pairs as 32-bit words: ``even`` in the low half."""
    lo = even.to(gl.bfloat16).to(gl.uint16, bitcast=True).to(gl.uint32)
    hi = odd.to(gl.bfloat16).to(gl.uint16, bitcast=True).to(gl.uint32)
    return lo | (hi << 16)


@gluon.jit
def store_routes_sink(args, ys, route, cols, col0):
    """Default down-unit sink (see :func:`run_moe`): ``out[route, cols] =
    bf16(y)`` with ``out = args[0]`` of shape ``(rows * top_k, hidden)``;
    rows with ``route < 0`` are skipped. With 4 strips (whole lines) each
    lane stores its two lines as 32-bit column pairs, one 16-byte store per
    line. The layer passes its H4 push."""
    out = args[0]
    if len(ys) == 1:
        gl.amd.cdna4.buffer_store(
            ys[0].to(gl.bfloat16),
            out,
            (route * _HIDDEN + cols[0]).to(gl.int32),
            mask=(route >= 0) & (cols[0] >= 0),
        )
    else:
        words = out.to(gl.pointer_type(gl.uint32), bitcast=True)
        for line in gl.static_range(2):
            # A lane's 4 words are one aligned run (the compiler cannot see
            # it through the halving).
            word = gl.max_contiguous(gl.multiple_of(cols[line] // 2, [1, 4]), [1, 4])
            gl.amd.cdna4.buffer_store(
                _bf16_pairs(ys[line], ys[line + 2]),
                words,
                (route * (_HIDDEN // 2) + word).to(gl.int32),
                mask=(route >= 0) & (word >= 0),
            )


@gluon.jit
def _store(desc, value, offsets, mask, cross_xcd):
    """Plain store, or a device-scope one when the reader may sit on another
    XCD (the split leftover experts)."""
    if cross_xcd:
        gl.amd.cdna3.buffer.store(value, desc, offsets, mask=mask, scope="gpu")
    else:
        gl.amd.cdna3.buffer.store(value, desc, offsets, mask=mask)


@gluon.constexpr_function
def _dn_kt(c, order):
    """K step of down chunk ``c`` (``order`` 0: K-step-major, 1: strip-major);
    every strip's chain stays in K order either way."""
    return c // 4 if order == 0 else c % 4


@gluon.constexpr_function
def _dn_st(c, order):
    """Strip of down chunk ``c`` (see :func:`_dn_kt`)."""
    return c % 4 if order == 0 else c // 4


@gluon.constexpr_function
def _line_strip(st, lines):
    """``_strip_offsets``' LINE_STRIP for strip ``st`` of a down unit."""
    return st if lines else -1


@gluon.jit
def _unit_offsets(expert, unit, bufs, N_PHYS2: gl.constexpr, DN_LINES: gl.constexpr):
    """Per-strip lane offsets of a down unit (4 strips of 16 columns; with
    ``DN_LINES`` the whole-line column mapping)."""
    n0 = unit * 64
    w = ()
    s = ()
    for st in gl.static_range(4):
        ws, w_pair, w_odd, ss, s_pair = _strip_offsets(
            expert,
            n0 + (0 if DN_LINES else 16 * st),
            bufs.stride_w2,
            bufs.stride_s2,
            bufs.stride_s2lin,
            bufs.stride_s2nb,
            N_PHYS2,
            _line_strip(st, DN_LINES),
        )
        w = w + (ws,)
        s = s + (ss,)
    return w, w_pair, w_odd, s, s_pair


@gluon.constexpr_function
def _dn_sdw(c, order):
    """Scale dword (32-column block, step pair) of down chunk ``c``."""
    return (_dn_st(c, order) // 2) * 2 + _dn_kt(c, order) // 2


@gluon.constexpr_function
def _dn_chunk_of(kt, st, order):
    """Chunk index of (K step, strip) in a down unit (inverse of _dn_kt/_dn_st)."""
    return 4 * kt + st if order == 0 else 4 * st + kt


@gluon.constexpr_function
def _a_release(kt, order, poll_at):
    """Chunk after which this unit's A operand of K step ``kt`` is dead and
    the next unit's may take its registers (never before the readiness poll)."""
    return max(poll_at, max(_dn_chunk_of(kt, st, order) for st in range(4)))


@gluon.constexpr_function
def _s_release(blk, p, order, poll_at):
    """Chunk after which the scale dword of (32-column block, step pair) is dead."""
    return max(
        poll_at,
        max(
            _dn_chunk_of(kt, st, order)
            for kt in (2 * p, 2 * p + 1)
            for st in (2 * blk, 2 * blk + 1)
        ),
    )


@gluon.jit
def _unit_routes(ids_t, expert, TOPK: gl.constexpr):
    """A-operand row routes of a down unit, in the A and A-scale layouts."""
    la: gl.constexpr = _layouts()[1]
    lsa: gl.constexpr = _layouts()[3]
    route_a = _routes(
        ids_t, expert, gl.arange(0, 16, layout=gl.SliceLayout(1, la)), TOPK
    )
    route_s = _routes(
        ids_t, expert, gl.arange(0, 16, layout=gl.SliceLayout(1, lsa)), TOPK
    )
    return route_a[:, None], route_s[:, None]


@gluon.jit
def _load_a(mids_d, route_a, kt: gl.constexpr, live):
    """K step ``kt`` of the expert's mids as the A operand: one dwordx4 per
    lane; rows are tokens, masked where the token does not pick the expert."""
    la: gl.constexpr = _layouts()[1]
    ak = gl.arange(0, 64, layout=gl.SliceLayout(0, la))[None, :]
    return gl.amd.cdna3.buffer.load(
        mids_d,
        route_a * _MID_ROW + kt * 64 + ak,
        mask=live & (route_a >= 0),
        scope="gpu",
    )


@gluon.jit
def _load_a_scale(mids_sd, route_s, kt: gl.constexpr, live):
    """The A operand's E8M0 bytes of K step ``kt``: ``mids_s[route][g]`` is
    group g = 4 kt + k quad, so one dword holds a step's 4 quads."""
    lsa: gl.constexpr = _layouts()[3]
    ask = gl.arange(0, 4, layout=gl.SliceLayout(0, lsa))[None, :]
    return gl.amd.cdna3.buffer.load(
        mids_sd,
        route_s * (_MID_SCALES // 4) + kt + ask * 0,
        mask=live & (route_s >= 0),
        scope="gpu",
    )


@gluon.jit
def _wait_quad(ctx, ex_count, counter, target, detail):
    """Wait until the quad's exchange counter (byte ``counter`` into the
    shared ``ex_count``) reaches ``target``: all 4 waves stored the group's
    SwiGLU values. LDS reads only; the deadline is checked between bounded
    spins and fails the launch with ``SITE_MOE_QUAD``."""
    lane: gl.constexpr = _layouts()[6]
    offsets = counter + gl.zeros([64 * gl.num_warps()], gl.int32, lane)
    goal = target + gl.zeros([64 * gl.num_warps()], gl.int32, lane)
    seen = wave_scalar(lds_spin_bounded(ex_count, offsets, goal))
    start = gl.to_tensor(0).to(gl.int64)
    retries = 0
    stop = retries < 0
    while (seen < target) & (not stop):
        retries += _LDS_SPIN_TRIES
        stop, start = poll_check(ctx, start, retries, True)
        seen = wave_scalar(lds_spin_bounded(ex_count, offsets, goal))
    if stop:
        fail_stop(ctx, _SITE_MOE_QUAD, detail)


@gluon.jit
def _wait_ready(ctx, tags, rank, TAG_POLL: gl.constexpr):
    """Wait until every group tag of the expert at ``rank`` carries this
    launch's tag (64 words: 16 groups x 4 strips, each published by its
    wave). Per wave, without a workgroup barrier: the wave's own mids loads
    follow its poll in program order. Fails the launch on expiry.

    ``TAG_POLL`` 0 and 1 poll with scalar loads (1 past the scalar cache),
    which do not wait behind the wave's weight loads; 2 polls with one
    uncached vector load per lane (a lane per tag word) and a wave vote,
    which waits for every load the wave issued before it.
    """
    if TAG_POLL == 2:
        lane: gl.constexpr = _layouts()[6]
        words = tags + rank * _TAGS + gl.arange(0, 64 * gl.num_warps(), lane) % 64
        missing = gl.load(words, cache_modifier=".cv") != ctx.tag
        start = gl.to_tensor(0).to(gl.int64)
        retries = 0
        stop = retries < 0
        while vote(missing.to(gl.int32)) & (not stop):
            retries += 1
            missing = gl.load(words, cache_modifier=".cv") != ctx.tag
            if retries % 64 == 0:
                stop, start = poll_check(ctx, start, retries, True)
        if stop:
            fail_stop(ctx, _SITE_MOE_READY, rank)
    else:
        for j in gl.static_range(_TAGS // 16):
            seen = poll_tags16(
                tags + rank * _TAGS + 16 * j, ctx.tag, ctx, TAG_POLL == 1, False
            )
            if not seen:
                fail_stop(ctx, _SITE_MOE_READY, rank)


@gluon.jit
def run_moe(
    ctx,
    handles,
    bufs,
    sink: gl.constexpr,
    sink_args,
    ROWS: gl.constexpr,
    TOPK: gl.constexpr,
    N_PHYS1: gl.constexpr,
    N_PHYS2: gl.constexpr,
    RING: gl.constexpr,
    QUAD_MAP: gl.constexpr,
    DN_ORDER: gl.constexpr,
    DN_LINES: gl.constexpr,
    ADDR_SOFFSET: gl.constexpr,
    TAG_POLL: gl.constexpr,
    SCALE_LANES: gl.constexpr,
    SWIGLU_ALPHA: gl.constexpr,
    SWIGLU_LIMIT: gl.constexpr,
    SWIGLU_BETA: gl.constexpr,
):
    """Stream this workgroup's MoE work: gate/up strips, then down units.

    ``ctx`` is the launch (``persistent.Launch``); its ``tag`` marks this
    launch's mids. ``bufs.xq`` and ``bufs.xs`` must be published before the
    call. ``bufs.tags`` holds ``TAG_WORDS`` int32 words, zeroed once; every
    launch that advances the epoch must run the stream, since it zeroes the
    tag copy the next launch polls.

    Every down unit's output goes to ``sink(sink_args, ys, route, cols,
    col0)``: ``ys`` a tuple of fp32 ``[16, 128]`` strip results in
    ``MOE_SINK_LAYOUT`` (wave w's own 16 columns; each lane holds 4 values
    of one row), ``cols`` the matching tuple of ``[1, 128]`` output columns,
    ``route`` the ``[16, 1]`` route per token row (-1: the token does not
    pick the expert), all in its slice layouts, and ``col0`` the wave's
    first column as a per-wave scalar. Without ``DN_LINES`` the sink is
    called once per strip with 1-tuples: lane l holds columns ``col0 + 4 (l
    // 16) + 0..3`` and ``col0`` is a multiple of 16. With ``DN_LINES`` it is
    called once per unit with the 4 strips and ``col0`` the unit's first
    column (a multiple of 64): with ``base = 32 (k // 2) + 8 (k % 2)`` for
    ``k = (l // 16) % 4``, value j of strip 0 is column ``col0 + base + 2
    j`` and of strip 2 the next one; strips 1 and 3 likewise from ``col0 +
    base + 16``. Each lane's strips 0 and 2 (1 and 3) thus form one whole
    16-byte bf16 line, 4 column pairs. Either way a
    call's columns lie within one 512-column slice. Like other per-wave
    scalars, use ``col0`` for addresses and branches only. The waves call
    the sink independently, with different trip counts, so the sink must
    stay within the wave: no layout conversion across waves, no barrier, no
    shared memory a wave does not own. Elementwise math, in-lane
    ``reshape`` / ``split`` / ``join`` and stores are fine.

    Plan fields: ``RING`` weight chunks in flight per wave; ``QUAD_MAP`` 0
    runs a group's 4 strips on waves 4q..4q+3, 1 on waves q, q+2, q+4, q+6;
    ``DN_ORDER`` 0 runs a down unit K-step-major, 1 strip-major;
    ``DN_LINES`` maps a down unit's columns to strips so each lane's results
    form whole lines (above; same sums, bit for bit); ``ADDR_SOFFSET`` advances weight addresses in the scalar offset;
    ``TAG_POLL`` picks the readiness poll (:func:`_wait_ready`); ``SCALE_LANES`` reads the
    gate/up scales lane-major (``bufs.s13_lanes``, one dwordx4 per lane per
    16 steps) instead of one swizzled dword per 2 steps.
    """
    mfma: gl.constexpr = _layouts()[0]
    la: gl.constexpr = _layouts()[1]
    lsa: gl.constexpr = _layouts()[3]
    rowl: gl.constexpr = _layouts()[5]
    lane: gl.constexpr = _layouts()[6]
    quant: gl.constexpr = _layouts()[7]
    gl.static_assert(_K_STEPS % RING == 0)
    gl.static_assert(RING % 2 == 0)
    gl.static_assert(RING <= 16)
    plan, ring, sring, rank_smem, ids_t, ids_q = handles
    tag = ctx.tag
    K_PACKED: gl.constexpr = 4096 // 2

    # ---- entry: the activations (every token an A row) and the zeroed
    # exchange counters, then the phase's only workgroup barrier.
    XL: gl.constexpr = gl.BlockedLayout([1, 16], [1, 64], [8, 1], [1, 0])
    x_all = gl.allocate_shared_memory(
        gl.uint8,
        [1, 16, K_PACKED],
        gl.PaddedSharedLayout.with_identity_for(
            [[K_PACKED, 16]], [16, K_PACKED], [1, 0]
        ),
    )
    xr = gl.arange(0, 16, layout=gl.SliceLayout(1, XL))[:, None]
    xc = gl.arange(0, K_PACKED, layout=gl.SliceLayout(0, XL))[None, :]
    x_all.index(0).store(gl.load(bufs.xq + xr * K_PACKED + xc, mask=xr < ROWS, other=0))
    SL: gl.constexpr = gl.BlockedLayout([1, 4], [16, 4], [1, 8], [1, 0])
    xs_all = gl.allocate_shared_memory(
        gl.uint8, [1, 16, 128], gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    )
    sr = gl.arange(0, 16, layout=gl.SliceLayout(1, SL))[:, None]
    sk = gl.arange(0, 128, layout=gl.SliceLayout(0, SL))[None, :]
    xs_all.index(0).store(
        gl.load(
            bufs.xs
            + _cdna4_swizzled_mxfp4_scale_offset(
                0, sr, sk, bufs.stride_xslin, bufs.stride_xsnb
            ),
            mask=sr < ROWS,
            other=127,
        )
    )
    # Quant exchange: [parity][32 rows][32 mids] fp32; rows 4w..4w+3 belong
    # to wave w. One counter word per (parity, quad), monotonic per launch.
    ex = gl.allocate_shared_memory(
        gl.float32, [2, 32, 32], gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    )
    ex_count = gl.allocate_shared_memory(
        gl.int32, [64], gl.SwizzledSharedLayout(1, 1, 1, order=[0])
    )
    ex_count.store(gl.zeros([64], gl.int32, gl.BlockedLayout([1], [64], [8], [0])))
    gl.barrier()

    w13d = gl.amd.cdna3.buffer.make_buffer_descriptor(bufs.w13, 0x7FFFFFF0)
    s13d = gl.amd.cdna3.buffer.make_buffer_descriptor(bufs.s13, 0x7FFFFFF0)
    l13d = gl.amd.cdna3.buffer.make_buffer_descriptor(bufs.s13_lanes, 0x7FFFFFF0)
    w2d = gl.amd.cdna3.buffer.make_buffer_descriptor(bufs.w2, 0x7FFFFFF0)
    s2d = gl.amd.cdna3.buffer.make_buffer_descriptor(bufs.s2, 0x7FFFFFF0)
    mids_d = gl.amd.cdna3.buffer.make_buffer_descriptor(bufs.mids, 0x7FFFFFF0)
    mids_sd = gl.amd.cdna3.buffer.make_buffer_descriptor(bufs.mids_s, 0x7FFFFFF0)
    tags_d = gl.amd.cdna3.buffer.make_buffer_descriptor(bufs.tags, 0x7FFFFFF0)
    lanes = gl.arange(0, 64 * 8, layout=lane)
    # This launch's tag copy; zero this workgroup's share of the other one
    # (32 words, one line) for the next launch.
    gl.static_assert(ROWS * TOPK <= _MAX_RANKS)
    gl.static_assert(_TAG_COPY_WORDS == 32 * 256)
    tag_base = ctx.copy * _TAG_COPY_WORDS
    gl.amd.cdna3.buffer.store(
        gl.zeros_like(lanes),
        tags_d,
        (1 - ctx.copy) * _TAG_COPY_WORDS + ctx.cta * 32 + lanes,
        mask=lanes < 32,
        scope="gpu",
    )

    # ---- gate/up strips
    expert, rank, group, left = _group_of(plan, 0, rank_smem)
    w, w_pair, w_odd, s, s_pair = _strip_offsets(
        expert,
        group * 64 + plan.strip * 16,
        bufs.stride_w13,
        bufs.stride_s13,
        bufs.stride_s13lin,
        bufs.stride_s13nb,
        N_PHYS1,
        -1,
    )
    lscale = _lanes_base(expert, group * 64 + plan.strip * 16, N_PHYS1, _K_STEPS)
    pend_word = rank * 0 - 1
    pend_left = left
    for k in range(plan.n_ug):
        # The next strip's plan, from LDS only.
        nexpert, nrank, ngroup, nleft = _group_of(plan, k + 1, rank_smem)
        nlive = k + 1 < plan.n_ug
        nw, nw_pair, nw_odd, ns, ns_pair = _strip_offsets(
            nexpert,
            ngroup * 64 + plan.strip * 16,
            bufs.stride_w13,
            bufs.stride_s13,
            bufs.stride_s13lin,
            bufs.stride_s13nb,
            N_PHYS1,
            -1,
        )
        nlscale = _lanes_base(nexpert, ngroup * 64 + plan.strip * 16, N_PHYS1, _K_STEPS)
        # An opaque index keeps the 32 steps' A reads in the loop (hoisted,
        # they would hold 160 VGPRs).
        z = opaque_zero()
        xa = x_all.index(z)
        xsa = xs_all.index(z)
        acc = gl.zeros((16, 128), gl.float32, mfma)
        for kt in gl.static_range(_K_STEPS):
            if SCALE_LANES:
                b_scale = _lanes_byte(sring[kt // 16], kt)
            else:
                b_scale = _scale_byte(sring[(kt // 2) % (RING // 2)], s, kt)
            a = xa.slice(kt * 64, 64, dim=1).load(la)
            a_scale = xsa.slice(kt * 4, 4, dim=1).load(lsa)
            acc = _direct_mxfp4_mfma(acc, a, ring[kt % RING], a_scale, b_scale)
            # Refill the consumed slot RING steps ahead: this strip, or the next.
            if kt + RING < _K_STEPS:
                nb = _load_chunk(w13d, w, w_pair, w_odd, kt + RING, None, ADDR_SOFFSET)
            else:
                nb = _load_chunk(
                    w13d, nw, nw_pair, nw_odd, kt + RING - _K_STEPS, nlive, ADDR_SOFFSET
                )
            ring = ring[: kt % RING] + (nb,) + ring[kt % RING + 1 :]
            if SCALE_LANES:
                # Each half is refilled once its last step was used: steps
                # 16-31 of this strip, then steps 0-15 of the next one.
                if kt == 16 - RING:
                    sring = (sring[0], _load_lanes(l13d, lscale, 1, None))
                if kt == _K_STEPS - RING:
                    sring = (_load_lanes(l13d, nlscale, 0, nlive), sring[1])
            elif kt % 2 == 1:
                if (kt - 1 + RING) // 2 < _K_STEPS // 2:
                    nsd = _load_scale_dword(s13d, s, s_pair, (kt - 1 + RING) // 2, None)
                else:
                    nsd = _load_scale_dword(
                        s13d, ns, ns_pair, (kt - 1 + RING) // 2 - _K_STEPS // 2, nlive
                    )
                sring = (
                    sring[: (kt // 2) % (RING // 2)]
                    + (nsd,)
                    + sring[(kt // 2) % (RING // 2) + 1 :]
                )
            if kt == RING:
                # The previous strip's tag: this step's chunk was issued after
                # that strip's mids stores, and the wave's memory operations
                # complete in issue order, so waiting for it covers them.
                _publish_tags(
                    tags_d, tag_base + pend_word, tag, pend_word >= 0, pend_left, acc
                )
            gl.amd.hint.sched_barrier()
        pend_word = rank * 0 - 1

        # ---- epilogue: SwiGLU in the lane, quant exchange, mids, tag pending
        mid = _swiglu_interleaved(
            gl.convert_layout(acc, rowl, assert_trivial=True),
            SWIGLU_ALPHA,
            SWIGLU_LIMIT,
            SWIGLU_BETA,
            64,
        )
        ML: gl.constexpr = mid.type.layout
        tr = gl.arange(0, 16, layout=gl.SliceLayout(1, ML))[:, None]
        jc = gl.arange(0, 64, layout=gl.SliceLayout(0, ML))[None, :]
        par = k % 2
        offs = (
            (par * 32 + _lds_row(plan.quad, tr, QUAD_MAP)) * 128
            + plan.strip * 32
            + (jc % 8) * 4
        )
        counter = (par * 2 + plan.quad) * 4
        lds_store_count(ex, offs, mid, ex_count, counter + offs * 0)
        _wait_quad(
            ctx,
            ex_count,
            counter,
            256 * (k // 2 + 1),
            (rank * 16 + group) * 2 + plan.quad,
        )
        packed, sbyte = _mxfp4_quantize_tile_in_layout(ex.index(par).load(quant))
        PL: gl.constexpr = packed.type.layout
        rho = gl.arange(0, 32, layout=gl.SliceLayout(1, PL))
        route = _routes(ids_q, expert, _token_of_quant_row(rho, QUAD_MAP), TOPK)[
            :, None
        ]
        pb = gl.arange(0, 16, layout=gl.SliceLayout(0, PL))[None, :]
        _store(mids_d, packed, route * _MID_ROW + group * 16 + pb, route >= 0, left)
        SLY: gl.constexpr = sbyte.type.layout
        srho = gl.arange(0, 32, layout=gl.SliceLayout(1, SLY))
        sroute = _routes(ids_q, expert, _token_of_quant_row(srho, QUAD_MAP), TOPK)[
            :, None
        ]
        sg = gl.arange(0, 1, layout=gl.SliceLayout(0, SLY))[None, :]
        # mids_s[route][g]: group g is K step g // 4, quad g % 4.
        _store(mids_sd, sbyte, sroute * _MID_SCALES + group + sg, sroute >= 0, left)
        pend_word = rank * _TAGS + group * 4 + plan.strip
        pend_left = left
        expert, rank, group, left = nexpert, nrank, ngroup, nleft
        w, w_pair, w_odd, s, s_pair = nw, nw_pair, nw_odd, ns, ns_pair
        lscale = nlscale

    # ---- gate/up -> down: publish the last strip's tag after a drain (once
    # per wave; its next loads would wait on this wave's own tag otherwise)
    if pend_word >= 0:
        release_drain()
        _publish_tags(
            tags_d,
            tag_base + pend_word,
            tag,
            pend_word >= 0,
            pend_left,
            gl.zeros((16, 128), gl.float32, mfma),
        )

    # ---- down units. Every operand that changes per unit is replaced in
    # place after its last use (no loop-carried copy, which would wait for a
    # load just issued): the weight ring by chunk, the A operand by K step,
    # the scale dwords by step pair, all after the next unit's readiness poll.
    POLL_AT: gl.constexpr = max(_DN_CHUNKS - RING - 1, 0)
    dlive = plan.n_dn > 0
    dexpert, drank, dunit = _unit_of(plan, 0, rank_smem)
    wb, dw_pair, dw_odd, sb, ds_pair = _unit_offsets(
        dexpert, dunit, bufs, N_PHYS2, DN_LINES
    )
    # First unit, in the steady state's issue order (side loads, then the
    # ring): the loop's first waits are the minimum over this entry and its
    # back edge. Its readiness wait therefore precedes its weights, once per
    # wave, right after the gate/up drain.
    if dlive:
        # Mids ready: the expert's group tags, polled with scalar loads.
        _wait_ready(ctx, bufs.tags + tag_base, drank, TAG_POLL)
    route_a, route_s = _unit_routes(ids_t, dexpert, TOPK)
    sdw = ()
    for i in gl.static_range(4):
        sdw = sdw + (_load_scale_dword(s2d, sb[2 * (i // 2)], ds_pair, i % 2, dlive),)
    a_t = ()
    a_sc = ()
    for kt in gl.static_range(_DN_STEPS):
        a_t = a_t + (_load_a(mids_d, route_a, kt, dlive),)
        a_sc = a_sc + (_load_a_scale(mids_sd, route_s, kt, dlive),)
    ring2 = ()
    for j in gl.static_range(RING):
        ring2 = ring2 + (
            _load_chunk(
                w2d,
                wb[_dn_st(j, DN_ORDER)],
                dw_pair,
                dw_odd,
                _dn_kt(j, DN_ORDER),
                dlive,
                ADDR_SOFFSET,
            ),
        )
        gl.amd.hint.sched_barrier()
    ask = gl.arange(0, 4, layout=gl.SliceLayout(0, lsa))[None, :]
    for u in range(plan.n_dn):
        nexpert, nrank, nunit = _unit_of(plan, u + 1, rank_smem)
        nlive = u + 1 < plan.n_dn
        nwb, nw_pair, nw_odd, nsb, ns_pair = _unit_offsets(
            nexpert, nunit, bufs, N_PHYS2, DN_LINES
        )
        nroute_a = route_a
        nroute_s = route_s
        zero = gl.zeros((16, 128), gl.float32, mfma)
        accs = (zero, zero, zero, zero)
        for c in gl.static_range(_DN_CHUNKS):
            b_scale = _scale_byte(
                sdw[_dn_sdw(c, DN_ORDER)], sb[_dn_st(c, DN_ORDER)], _dn_kt(c, DN_ORDER)
            )
            a_scale = ((a_sc[_dn_kt(c, DN_ORDER)] >> (ask * 8)) & 255).to(gl.uint8)
            acc = _direct_mxfp4_mfma(
                accs[_dn_st(c, DN_ORDER)],
                a_t[_dn_kt(c, DN_ORDER)],
                ring2[c % RING],
                a_scale,
                b_scale,
            )
            accs = (
                accs[: _dn_st(c, DN_ORDER)] + (acc,) + accs[_dn_st(c, DN_ORDER) + 1 :]
            )
            if c == POLL_AT:
                # The next unit's readiness: scalar loads only.
                if nlive:
                    _wait_ready(ctx, bufs.tags + tag_base, nrank, TAG_POLL)
                nroute_a, nroute_s = _unit_routes(ids_t, nexpert, TOPK)
            for kt in gl.static_range(_DN_STEPS):
                if c == _a_release(kt, DN_ORDER, POLL_AT):
                    a_t = (
                        a_t[:kt]
                        + (_load_a(mids_d, nroute_a, kt, nlive),)
                        + a_t[kt + 1 :]
                    )
                    a_sc = (
                        a_sc[:kt]
                        + (_load_a_scale(mids_sd, nroute_s, kt, nlive),)
                        + a_sc[kt + 1 :]
                    )
            for i in gl.static_range(4):
                if c == _s_release(i // 2, i % 2, DN_ORDER, POLL_AT):
                    sdw = (
                        sdw[:i]
                        + (
                            _load_scale_dword(
                                s2d, nsb[2 * (i // 2)], ns_pair, i % 2, nlive
                            ),
                        )
                        + sdw[i + 1 :]
                    )
            if c + RING < _DN_CHUNKS:
                nb = _load_chunk(
                    w2d,
                    wb[_dn_st(c + RING, DN_ORDER)],
                    dw_pair,
                    dw_odd,
                    _dn_kt(c + RING, DN_ORDER),
                    None,
                    ADDR_SOFFSET,
                )
            else:
                nb = _load_chunk(
                    w2d,
                    nwb[_dn_st(c + RING - _DN_CHUNKS, DN_ORDER)],
                    nw_pair,
                    nw_odd,
                    _dn_kt(c + RING - _DN_CHUNKS, DN_ORDER),
                    nlive,
                    ADDR_SOFFSET,
                )
            ring2 = ring2[: c % RING] + (nb,) + ring2[c % RING + 1 :]
            gl.amd.hint.sched_barrier()
        # Down unit -> sink (the layer's H4 push). Handed over in
        # MOE_SINK_LAYOUT, which is the MFMA result's own layout: no data
        # moves, and each wave owns its 16 columns.
        rows = gl.arange(0, 16, layout=gl.SliceLayout(1, rowl))
        route = _routes(ids_t, dexpert, rows, TOPK)[:, None]
        pos = gl.arange(0, 128, layout=gl.SliceLayout(0, rowl))[None, :] % 16
        if DN_LINES:
            ys = ()
            cols = ()
            for st in gl.static_range(4):
                ys = ys + (gl.convert_layout(accs[st], rowl, assert_trivial=True),)
                cols = cols + (dunit * 64 + _line_column(pos, st),)
            sink(sink_args, ys, route, cols, dunit * 64)
        else:
            for st in gl.static_range(4):
                y = gl.convert_layout(accs[st], rowl, assert_trivial=True)
                col0 = dunit * 64 + st * 16
                sink(sink_args, (y,), route, (col0 + pos,), col0)
        dexpert, drank, dunit = nexpert, nrank, nunit
        wb, dw_pair, dw_odd, sb, ds_pair = nwb, nw_pair, nw_odd, nsb, ns_pair
        route_a, route_s = nroute_a, nroute_s


# ----------------------------------------------------------------------------
# Standalone launch (tests and benchmarks; the decoder layer composes
# prefetch_moe / run_moe itself)
# ----------------------------------------------------------------------------


@gluon.jit(do_not_specialize=["deadline_ticks"])
def gluon_mxfp4_moe_native_stream_gfx950(
    epochs,
    claims,
    errors,
    arena,
    ids,
    xq,
    xs,
    w13,
    s13,
    s13_lanes,
    w2,
    s2,
    mids,
    mids_s,
    tags,
    out,
    stride_xslin,
    stride_xsnb,
    stride_w13,
    stride_s13,
    stride_s13lin,
    stride_s13nb,
    stride_w2,
    stride_s2,
    stride_s2lin,
    stride_s2nb,
    deadline_ticks,
    ROWS: gl.constexpr,
    TOPK: gl.constexpr,
    N_PHYS1: gl.constexpr,
    N_PHYS2: gl.constexpr,
    RING: gl.constexpr,
    QUAD_MAP: gl.constexpr,
    DN_ORDER: gl.constexpr,
    DN_LINES: gl.constexpr,
    ADDR_SOFFSET: gl.constexpr,
    TAG_POLL: gl.constexpr,
    SCALE_LANES: gl.constexpr,
    TAIL_NUM: gl.constexpr,
    TAIL_DEN: gl.constexpr,
    SWIGLU_ALPHA: gl.constexpr,
    SWIGLU_LIMIT: gl.constexpr,
    SWIGLU_BETA: gl.constexpr,
    SINK: gl.constexpr,
):
    """The MoE stream as its own persistent launch of 256 workgroups x 8
    waves; ``SINK((out,), ys, route, cols, col0)`` takes the down units' output
    (``store_routes_sink``: ``out[route, :]`` in bf16)."""
    # Entry counters first, then the plan and the first weights (they depend
    # only on the ids), then the entry check.
    entry = enter_issue(epochs, claims)
    # The scales' row strides are 1, which a launch passes as constants.
    bufs = MoeBuffers(
        xq,
        xs,
        w13,
        s13,
        s13_lanes,
        w2,
        s2,
        mids,
        mids_s,
        tags,
        gl.to_tensor(stride_xslin),
        stride_xsnb,
        stride_w13,
        stride_s13,
        gl.to_tensor(stride_s13lin),
        stride_s13nb,
        stride_w2,
        stride_s2,
        gl.to_tensor(stride_s2lin),
        stride_s2nb,
    )
    handles = prefetch_moe(
        entry,
        ids,
        bufs,
        ROWS,
        TOPK,
        N_PHYS1,
        RING,
        QUAD_MAP,
        TAIL_NUM,
        TAIL_DEN,
        ADDR_SOFFSET,
        SCALE_LANES,
    )
    # The serving entry form (vector counters).
    ctx = enter_check(entry, errors, arena, deadline_ticks, 0, fail_stop)
    run_moe(
        ctx,
        handles,
        bufs,
        SINK,
        (out,),
        ROWS,
        TOPK,
        N_PHYS1,
        N_PHYS2,
        RING,
        QUAD_MAP,
        DN_ORDER,
        DN_LINES,
        ADDR_SOFFSET,
        TAG_POLL,
        SCALE_LANES,
        SWIGLU_ALPHA,
        SWIGLU_LIMIT,
        SWIGLU_BETA,
    )


class NativeMoeState:
    """Launch state of the standalone stream on one device: epochs, claims and
    error words (``persistent``), the mids hand-off for up to ``rows`` tokens
    and the readiness tags. Reused across launches; the launch tag keeps them
    apart. Free the tags with ``state.tags.free()``."""

    def __init__(self, device, rows: int, top_k: int):
        self.epochs = torch.zeros(ENTRY_WORDS, dtype=torch.int64, device=device)
        self.claims = torch.zeros(ENTRY_WORDS, dtype=torch.int64, device=device)
        self.errors = torch.zeros(ERROR_WORDS, dtype=torch.int32, device=device)
        # The launch's flag arena; the standalone stream polls none of it.
        self.arena = torch.zeros(ENTRY_WORDS, dtype=torch.int64, device=device)
        routes = rows * top_k
        self.mids = torch.zeros(routes * INTER // 2, dtype=torch.uint8, device=device)
        self.mids_scale = torch.zeros(
            routes * INTER // 32 // 4, dtype=torch.int32, device=device
        )
        # Written on one XCD and polled with scalar loads on others (the split
        # leftover experts), so uncached: scalar polls of cached device memory
        # only see writes made on the same XCD.
        self.tags = UncachedBuffer(TAG_WORDS, torch.int32, device)


def _int32_words(scale):
    """The uint8 scale tensor's bytes from its first element on, as int32
    words: the stream reads E8M0 scales one dword at a time."""
    storage = scale.untyped_storage()
    if scale.dtype != torch.uint8 or scale.storage_offset() % 4:
        raise ValueError("scales must be uint8 starting on a 4-byte boundary")
    words = (storage.nbytes() - scale.storage_offset()) // 4
    alias = torch.empty(0, dtype=torch.int32, device=scale.device)
    return alias.set_(storage, scale.storage_offset() // 4, (words,), (1,))


@dataclasses.dataclass(frozen=True)
class NativeMoeWeights:
    """The experts' prepared tensors as the stream reads them, in place.

    ``s13`` / ``s2`` are the CDNA4-swizzled E8M0 scales as int32 words (the
    stream reads a dword at a time); strides are in bytes of the uint8
    tensors. ``s13_lanes`` is the optional lane-major copy of the gate/up
    scales (``SCALE_LANES``), as int32 words, or None.
    """

    w13: torch.Tensor
    s13: torch.Tensor
    s13_lanes: torch.Tensor | None
    w2: torch.Tensor
    s2: torch.Tensor
    stride_w13: int
    stride_s13: int
    stride_s13lin: int
    stride_s13nb: int
    stride_w2: int
    stride_s2: int
    stride_s2lin: int
    stride_s2nb: int
    n_phys1: int
    n_phys2: int


def native_moe_weights(experts, *, scale_lanes: bool) -> NativeMoeWeights:
    """The stream's view of a processed gfx950 MXFP4 MoE layer.

    Reads the tensors today's EP1 decode reads (``w13_weight_triton_tensor``
    and ``w2_weight_triton_tensor``, gdot128-shuffled, with their
    precision configs' CDNA4-swizzled scales), without copying them.

    Args:
        experts: The MoE layer after ``preprocess_gluon_mxfp4_gfx950_moe_weights``
            (the model's ``mlp.experts``).
        scale_lanes: Also build the lane-major gate/up scales
            (``SCALE_LANES``). That is a copy: 32 MiB per layer of 256
            local experts (2 GB over 60 layers).

    Raises:
        ValueError: The layout is not the EP1 slice the stream tiles
            (``(E, 2048, 1024)`` gate/up, ``(E, 256, 4096)`` down), or a
            stride the kernels specialize on is not a multiple of 16.
    """
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused import (
        _extract_gluon_raw_s,
        _extract_gluon_raw_w,
    )

    w13 = _extract_gluon_raw_w(experts.w13_weight_triton_tensor)
    s13 = _extract_gluon_raw_s(experts.w13_precision_config.b_mx_scale)
    w2 = _extract_gluon_raw_w(experts.w2_weight_triton_tensor)
    s2 = _extract_gluon_raw_s(experts.w2_precision_config.b_mx_scale)
    experts_n = w13.shape[0]
    if w13.dtype != torch.uint8 or w13.shape[1:] != (HIDDEN // 2, GATE_UP):
        raise ValueError(
            f"native MoE stream: gate/up weights must be (E, {HIDDEN // 2}, "
            f"{GATE_UP}) uint8 gdot128, got {tuple(w13.shape)} {w13.dtype}"
        )
    if w2.dtype != torch.uint8 or w2.shape != (experts_n, INTER // 2, HIDDEN):
        raise ValueError(
            f"native MoE stream: down weights must be (E, {INTER // 2}, "
            f"{HIDDEN}) uint8 gdot128, got {tuple(w2.shape)} {w2.dtype}"
        )
    for name, scale in (("gate/up", s13), ("down", s2)):
        if scale.dtype != torch.uint8 or scale.stride(1) != 1:
            raise ValueError(
                f"native MoE stream: {name} scales must be CDNA4-swizzled uint8 "
                f"(stride(1) == 1), got {scale.dtype} strides {scale.stride()}"
            )
    strides = dict(
        stride_w13=w13.stride(0),
        stride_s13=s13.stride(0),
        stride_s13lin=s13.stride(1),
        stride_s13nb=s13.stride(2),
        stride_w2=w2.stride(0),
        stride_s2=s2.stride(0),
        stride_s2lin=s2.stride(1),
        stride_s2nb=s2.stride(2),
    )
    uneven = {k: v for k, v in strides.items() if not k.endswith("lin") and v % 16}
    if uneven:
        raise ValueError(
            f"native MoE stream: strides must be multiples of 16, got {uneven}"
        )
    lanes = None
    if scale_lanes:
        canonical = unswizzle_cdna4_mxfp4_scale(s13, GATE_UP, HIDDEN // 32)
        lanes = _int32_words(pack_cdna4_mxfp4_scale_lanes(canonical))
    return NativeMoeWeights(
        w13=w13,
        s13=_int32_words(s13),
        s13_lanes=lanes,
        w2=w2,
        s2=_int32_words(s2),
        n_phys1=w13.shape[2],
        n_phys2=w2.shape[2],
        **strides,
    )


def launch_gluon_mxfp4_moe_native_stream_gfx950(
    xq,
    xs,
    weights: NativeMoeWeights,
    topk_ids,
    out,
    state,
    *,
    ring: int,
    quad_map: int,
    dn_order: int,
    dn_lines: int,
    addr_soffset: int,
    tag_poll: int,
    tail_num: int,
    tail_den: int,
    swiglu_alpha: float,
    swiglu_limit: float,
    swiglu_beta: float,
    deadline_ticks: int,
):
    """Run the MoE stream as its own launch.

    Args:
        xq, xs: MXFP4 activation rows ``(rows, 2048)`` and their CDNA4-swizzled
            E8M0 scales (``_quantize_mxfp4_activation``).
        weights: From :func:`native_moe_weights`; ``SCALE_LANES`` follows
            whether it holds lane-major scales.
        topk_ids: ``(rows, top_k)`` int32 global expert ids (EP1: all local).
        out: ``(rows * top_k, 4096)`` bf16, written per route.
        state: :class:`NativeMoeState`.
        ring, quad_map, dn_order, dn_lines, addr_soffset, tag_poll, tail_num,
            tail_den:
            Plan fields, see :func:`run_moe` and :func:`prefetch_moe`.
        swiglu_alpha, swiglu_limit, swiglu_beta: The model's SwiGLU.
        deadline_ticks: How long a readiness wait may last (10 ns ticks).
    """
    rows, top_k = topk_ids.shape
    if xq.shape[1] != HIDDEN // 2:
        raise ValueError(
            f"native MoE stream: activations must be (rows, {HIDDEN // 2}), "
            f"got {tuple(xq.shape)}"
        )
    if rows > 16 or out.shape != (rows * top_k, HIDDEN):
        raise ValueError(
            "native MoE stream: at most 16 rows, out is (rows * top_k, hidden)"
        )
    w = weights
    gluon_mxfp4_moe_native_stream_gfx950[(GRID,)](
        state.epochs, state.claims, state.errors, state.arena, topk_ids, xq, xs,
        w.w13, w.s13, w.s13 if w.s13_lanes is None else w.s13_lanes, w.w2, w.s2,
        state.mids, state.mids_scale, state.tags, out,
        xs.stride(0), xs.stride(1), w.stride_w13, w.stride_s13, w.stride_s13lin,
        w.stride_s13nb, w.stride_w2, w.stride_s2, w.stride_s2lin, w.stride_s2nb,
        deadline_ticks,
        ROWS=rows, TOPK=top_k, N_PHYS1=w.n_phys1, N_PHYS2=w.n_phys2, RING=ring,
        QUAD_MAP=quad_map, DN_ORDER=dn_order, DN_LINES=dn_lines,
        ADDR_SOFFSET=addr_soffset,
        TAG_POLL=tag_poll, SCALE_LANES=int(w.s13_lanes is not None),
        TAIL_NUM=tail_num, TAIL_DEN=tail_den, SWIGLU_ALPHA=swiglu_alpha,
        SWIGLU_LIMIT=swiglu_limit, SWIGLU_BETA=swiglu_beta, SINK=store_routes_sink,
        num_warps=8, **persistent_compile_options()
    )  # fmt: skip

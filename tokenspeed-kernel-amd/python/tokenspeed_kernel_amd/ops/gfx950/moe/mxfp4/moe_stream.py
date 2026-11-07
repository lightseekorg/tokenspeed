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


"""MXFP4 MoE decode as one launch of per-wave weight streams (gfx950).

Replaces the direct decode's gate/up (stage 1) and down (stage 2) launches
at a few decode row counts, for a rank that holds a slice of every expert's
intermediate (expert tensor parallelism, all experts local): hidden 4096,
intermediate slice 512, top 8. The output is the direct decode's unweighted
per-route rows, ``out[token * top_k + slot, :]``, bit for bit.

256 workgroups x 8 waves, one per CU. Every token is an A row of every
expert (MFMA M = 16, at most 16 rows), so each touched expert's weights are
read once. Work items, all per wave:

- A gate/up *strip* is (expert, 16 gate/up columns, all K): one 1 KB
  ``buffer_load_dwordx4`` per lane per 128-K step (the MFMA B operand in the
  gdot128 layout) plus one E8M0 dword per lane per two steps.
- A *group* is 4 strips: 64 gate/up columns, one 32-value MXFP4 block of
  mids. A group's 4 strips run on a *quad* of 4 waves (waves 0-3 or 4-7)
  that exchange their SwiGLU values through LDS before quantizing.
- A down *unit* is (expert, 64 output columns, all 512 K): 16 chunks of 1 KB
  plus scales, with the expert's mids as the A operand.

Each wave keeps ``RING`` weight chunks in flight in registers and never
drains them: the chunk it just consumed is refilled ``RING`` steps ahead,
into the next strip or unit when the current one ends. A wave's vector
memory operations complete in issue order, so the stream never waits on a
vector load issued behind its weights: item metadata comes from LDS,
readiness polls are scalar loads, and the quad exchange spins on LDS
without ``s_barrier``.

The plan is closed-form from the routed ids, computed in every workgroup.
Whole experts go to XCDs round-robin, and the ``E % 8`` leftover experts are
split by group over all XCDs, so every XCD carries the same bytes. Within an
XCD, 16 *tail* workgroups (slots 0 and 16-30) get whole groups worth
``TAIL_NUM / TAIL_DEN`` of a *full* workgroup's bytes and no down units, so
they finish first; the 16 full workgroups take the remaining groups, then
the down units, earliest-ready expert first. :func:`moe_stream_plan` is the
same plan on the host.

Progress: a workgroup's gate/up groups never wait on another workgroup (only
on its own waves, through LDS), and its down units wait only on gate/up
groups. Every workgroup finishes its groups before its first down wait, so
the waits end once all 256 workgroups have started. One workgroup fits on
every CU, so a device with 256 CUs runs them all as soon as the work queued
before the launch has left the CUs; the launch must not share the device
with a kernel that waits on its results. The waits have no deadline, like
the other cross-workgroup waits in TokenSpeed's decode kernels.

Numerics follow the direct decode (``decode_common``, ``decode_stage1``,
``quantize_gluon``): the same per-column MFMA chain over K, SwiGLU, 32-value
MXFP4 quantization of the mids and down chain.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch
from tokenspeed_kernel_amd._scheduling import release_drain
from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4._stream_sync import (
    UncachedBuffer,
    after,
    lds_spin_until,
    lds_store_count,
    opaque_zero,
    scalar_load_glc_i64,
    stream_sync_compile_options,
    trap,
    wait_tags16,
    wave_scalar,
    xcc_id,
)
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
    is_swizzled_cdna4_mxfp4_scale,
)

__all__ = [
    "GRID",
    "GROUP_BYTES",
    "GROUPS_PER_EXPERT",
    "MAX_ROWS",
    "TAG_COPY_WORDS",
    "TAIL_SHARE",
    "UNIT_BYTES",
    "UNITS_PER_EXPERT",
    "MoeStreamState",
    "MoeStreamWeights",
    "gluon_mxfp4_moe_stream_gfx950",
    "launch_gluon_mxfp4_moe_stream_gfx950",
    "moe_stream_plan",
    "moe_stream_weights",
]

# Workgroups per launch (one per CU), XCDs per GPU and workgroups per XCD.
GRID = 256
XCDS = 8
SLOTS = GRID // XCDS
# Tail slots per XCD: 0 and TAIL_FIRST..TAIL_LAST (16 of 32).
TAIL_FIRST = 16
TAIL_LAST = 30
# The tail workgroups' bytes as a fraction of a full workgroup's
# (numerator, denominator), as measured best for a standalone launch.
TAIL_SHARE = (1, 2)
HIDDEN = 4096
INTER = 512
GATE_UP = 2 * INTER
TOPK = 8
# Rows of a launch: every token is an A row of the 16-row MFMA.
MAX_ROWS = 16
GROUP_COLUMNS = 64
GROUPS_PER_EXPERT = GATE_UP // GROUP_COLUMNS  # 16
UNIT_COLUMNS = 64
UNITS_PER_EXPERT = HIDDEN // UNIT_COLUMNS  # 64
# Weight chunks each wave keeps in flight.
RING = 8
# Group tags: one word per (group, strip); each wave publishes its own rows.
TAG_WORDS_PER_EXPERT = GROUPS_PER_EXPERT * 4
# Distinct experts of one launch: at most 16 rows x top 8.
MAX_RANKS = MAX_ROWS * TOPK
# Group tags live in two copies, indexed by expert rank. A launch polls and
# publishes copy (launch count & 1) and zeroes the other one for the next
# launch: launch tags repeat after 2**32 - 1 launches, and a zeroed word
# never matches (tags are never 0).
TAG_COPY_WORDS = MAX_RANKS * TAG_WORDS_PER_EXPERT
TAG_WORDS = 2 * TAG_COPY_WORDS
# int32 words per 128 B line; each launch counter and role claim owns a line.
LINE_WORDS = 32
COUNTER_WORDS = GRID * LINE_WORDS // 2
# Bytes of one group (64 gate/up columns x K, plus scales) and one unit.
GROUP_BYTES = GROUP_COLUMNS * HIDDEN // 2 + GROUP_COLUMNS * HIDDEN // 32
UNIT_BYTES = UNIT_COLUMNS * INTER // 2 + UNIT_COLUMNS * INTER // 32

_XCDS = gl.constexpr(XCDS)
_SLOTS = gl.constexpr(SLOTS)
_HIDDEN = gl.constexpr(HIDDEN)
_TAIL_FIRST = gl.constexpr(TAIL_FIRST)
_TAIL_LAST = gl.constexpr(TAIL_LAST)
_GROUPS = gl.constexpr(GROUPS_PER_EXPERT)
_UNITS = gl.constexpr(UNITS_PER_EXPERT)
_RING = gl.constexpr(RING)
_TAGS = gl.constexpr(TAG_WORDS_PER_EXPERT)
_TAG_COPY_WORDS = gl.constexpr(TAG_COPY_WORDS)
_MAX_RANKS = gl.constexpr(MAX_RANKS)
_COUNTER_STRIDE = gl.constexpr(LINE_WORDS // 2)
_K_STEPS = gl.constexpr(HIDDEN // 128)  # 32 gate/up steps
_DN_STEPS = gl.constexpr(INTER // 128)  # 4 down steps
_DN_CHUNKS = gl.constexpr(4 * (INTER // 128))  # 16
_MID_ROW = gl.constexpr(INTER // 2)  # packed MXFP4 bytes per mid row
_MID_SCALES = gl.constexpr(INTER // 32)  # E8M0 bytes per mid row


# ----------------------------------------------------------------------------
# The plan, on the host (the device derives the same numbers per workgroup)
# ----------------------------------------------------------------------------


def _tail_index(slot):
    """Tail index 0-15 of a slot, or -1 for a full slot."""
    if slot == 0:
        return 0
    if TAIL_FIRST <= slot <= TAIL_LAST:
        return slot - TAIL_FIRST + 1
    return -1


def _full_index(slot):
    """Full index 0-15 of a slot, or -1 for a tail slot."""
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
    # at tail_num / tail_den of a full one's bytes, rounded to whole groups.
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


def moe_stream_plan(topk_ids, *, tail_num, tail_den):
    """The closed-form plan on the host.

    Args:
        topk_ids: ``(rows, top_k)`` integer expert ids.
        tail_num, tail_den: A tail workgroup's bytes as a fraction of a full
            one's (the kernel's ``TAIL_NUM`` / ``TAIL_DEN``).

    Returns:
        A list over the 256 logical workgroups (``cta = slot * 8 + xcd``) of
        lists of work items in stream order per workgroup: ``("ug", expert,
        group)`` then ``("dn", expert, unit)``, with the routed expert ids.
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
    Returns the number of distinct experts. Runs once at entry, in every
    wave (the reductions span the workgroup).
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
    # SGPR so the plan's trip counts and branches stay uniform (the compiler
    # merges the waits of a divergent loop conservatively).
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
    def __init__(self, quad, strip, xcd, whole, n_whole, t, q0, n_ug, v0, n_dn):
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
def _plan_wave(xcd, slot, experts, TAIL_NUM: gl.constexpr, TAIL_DEN: gl.constexpr):
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
    # Per wave: quad (waves 0-3 or 4-7) and strip; each of the 2 quads takes
    # every other group of the list.
    WL: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    wave = wave_scalar(gl.arange(0, 64 * gl.num_warps(), WL) // 64)
    quad = wave // 4
    strip = wave % 4
    n_ug = gl.maximum(count - quad + 1, 0) // 2
    v0 = first + wave
    n_dn = gl.maximum(end - v0 + 7, 0) // 8
    return MoeWave(
        quad, strip, xcd, whole, n_whole, t, start + 16 * quad, n_ug, v0, n_dn
    )


@gluon.jit
def _token_of_quant_row(rho):
    """Token row held by quant-exchange row ``rho`` (inverse of ``_lds_row``)."""
    return rho % 16


@gluon.jit
def _lds_row(quad, token_row):
    """Row of the quant exchange holding (quad, token row): rows 4w..4w+3
    belong to wave w, so a wave quantizes only its own quad's group."""
    return 16 * quad + token_row


# ----------------------------------------------------------------------------
# Device: the stream
# ----------------------------------------------------------------------------


@gluon.aggregate
class MoeBuffers:
    """Weights, activations and the mids hand-off of one launch."""

    xq: gl.tensor
    xs: gl.tensor
    w13: gl.tensor
    s13: gl.tensor
    w2: gl.tensor
    s2: gl.tensor
    mids: gl.tensor
    mids_s: gl.tensor
    tags: gl.tensor
    stride_xq: gl.tensor
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
    def __init__(self, xq, xs, w13, s13, w2, s2, mids, mids_s, tags, stride_xq, stride_xslin, stride_xsnb, stride_w13, stride_s13, stride_s13lin, stride_s13nb, stride_w2, stride_s2, stride_s2lin, stride_s2nb):  # fmt: skip
        self.xq = xq
        self.xs = xs
        self.w13 = w13
        self.s13 = s13
        self.w2 = w2
        self.s2 = s2
        self.mids = mids
        self.mids_s = mids_s
        self.tags = tags
        self.stride_xq = stride_xq
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


@gluon.constexpr_function
def _layouts():
    # MFMA 16x16x128 (transposed result: a lane holds one row, 4 consecutive
    # columns), warps along N: wave w owns columns 16w..16w+15.
    mfma, dot_a, dot_b, a_scale, b_scale = _direct_mxfp4_mfma_layouts(16, 128, 4, 8)
    # The MFMA result as a blocked layout: lane l of wave w holds row l % 16
    # and columns 16 w + 4 (l // 16) + 0..3.
    rows = gl.BlockedLayout([1, 4], [16, 4], [1, 8], [0, 1])
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
def _strip_offsets(
    expert, column, w_stride, s_stride, s_lin, s_nb, N_PHYS: gl.constexpr
):
    """Lane offsets of the 16-column strip at ``column``: its first K step
    and the per-step advances, all from ``decode_common``'s layout functions.

    Returns ``(w, w_pair, w_odd, s, s_pair)``: step ``kt`` reads weight bytes
    at ``w + (kt // 2) * w_pair + (kt % 2) * w_odd`` and its E8M0 byte at
    ``s + (kt // 2) * s_pair + (kt % 2) * 2``; the swizzle keeps a step
    pair's two bytes in one dword. Advances are per-wave scalars.
    """
    lb: gl.constexpr = _layouts()[2]
    lsb: gl.constexpr = _layouts()[4]
    bk = gl.arange(0, 64, layout=gl.SliceLayout(1, lb))[:, None]
    bn = gl.arange(0, 128, layout=gl.SliceLayout(0, lb))[None, :]
    # Strips start on 16-column boundaries; saying so keeps every lane's 16
    # bytes one dwordx4.
    column = gl.multiple_of(column, 16)
    w_expert = expert.to(gl.int64) * w_stride
    n_col = column + bn % 16
    w0 = _gluon_dot_preshuffled_w_offset(w_expert, bk, n_col, N_PHYS)
    w1 = _gluon_dot_preshuffled_w_offset(w_expert, 64 + bk, n_col, N_PHYS)
    w2 = _gluon_dot_preshuffled_w_offset(w_expert, 128 + bk, n_col, N_PHYS)
    bsn = gl.arange(0, 128, layout=gl.SliceLayout(1, lsb))[:, None]
    bsk = gl.arange(0, 4, layout=gl.SliceLayout(0, lsb))[None, :]
    s_expert = expert.to(gl.int64) * s_stride
    s_col = column + bsn % 16
    s0 = _cdna4_swizzled_mxfp4_scale_offset(s_expert, s_col, bsk, s_lin, s_nb)
    s2 = _cdna4_swizzled_mxfp4_scale_offset(s_expert, s_col, 8 + bsk, s_lin, s_nb)
    # The advances are whole 1 KB chunks and whole scale dwords (the codegen
    # test checks the dwordx4 / dword loads).
    return (
        w0.to(gl.int32),
        gl.multiple_of(wave_scalar((w2 - w0).to(gl.int32)), 1024),
        gl.multiple_of(wave_scalar((w1 - w0).to(gl.int32)), 1024),
        s0.to(gl.int32),
        gl.multiple_of(wave_scalar((s2 - s0).to(gl.int32)), 4),
    )


@gluon.jit
def _load_chunk(desc, base, pair, odd, kt: gl.constexpr, live):
    """One 1 KB weight chunk (the MFMA B operand of step ``kt``), nontemporal:
    each weight byte is read once per launch."""
    policy: gl.constexpr = gl.amd.cdna4.CachePolicy(nontemporal=True)
    step = (kt // 2) * pair + (kt % 2) * odd
    mask = None if live is None else (base == base) & live
    return gl.amd.cdna4.buffer.load(desc, base + step, mask=mask, cache_policy=policy)


@gluon.jit
def _load_scale_dword(desc, base, pair, p: gl.constexpr, live):
    """The E8M0 dword of step pair ``p`` (bytes ``base + p * pair``).
    Cached normally: the two strips of a 32-column block read the same
    lines."""
    mask = None if live is None else (base == base) & live
    return gl.amd.cdna4.buffer.load(desc, (base + p * pair) // 4, mask=mask)


@gluon.jit
def _scale_byte(dword, base, kt: gl.constexpr):
    """Step ``kt``'s E8M0 byte out of its pair's dword."""
    shift = ((base + (kt % 2) * 2) % 4) * 8
    return ((dword >> shift) & 255).to(gl.uint8)


@gluon.jit
def _group_of(plan, k, rank_smem):
    """This wave's k-th group: (expert id, rank, group index, leftover?)."""
    # A quad takes every other group of the workgroup's list.
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
def _store_mids(desc, value, offsets, mask):
    """Store intermediates written through to memory (SC0 SC1).

    Each XCD has its own L2, and the L2s do not snoop one another. Readers of
    a split leftover expert sit on other XCDs, and every launch reuses the
    same buffer, so a reader's L2 may still hold the previous launch's line.
    Writing through here and reading with SC0 SC1 (:func:`_load_a`) bypasses
    every L2 coherently (CDNA4 ISA 9.1.10.2), whichever XCD reads.
    """
    gl.amd.cdna4.buffer.store(value, desc, offsets, mask=mask, scope="sys")


@gluon.jit
def _publish_tags(tags_d, word, tag, live, dep):
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
    # The tags are uncached memory, so the store reaches every XCD's readers.
    gl.amd.cdna4.buffer.store(
        value,
        tags_d,
        word + r * 0 + c * 0,
        mask=live & (r == 0) & (c % 16 == 0),
        scope="sys",
    )


@gluon.jit
def _unit_offsets(expert, unit, bufs, N_PHYS2: gl.constexpr):
    """Per-strip lane offsets of a down unit (4 strips of 16 columns)."""
    n0 = unit * 64
    w = ()
    s = ()
    for st in gl.static_range(4):
        ws, w_pair, w_odd, ss, s_pair = _strip_offsets(
            expert,
            n0 + 16 * st,
            bufs.stride_w2,
            bufs.stride_s2,
            bufs.stride_s2lin,
            bufs.stride_s2nb,
            N_PHYS2,
        )
        w = w + (ws,)
        s = s + (ss,)
    return w, w_pair, w_odd, s, s_pair


# A down unit runs K-step-major: chunk c is K step c // 4 of strip c % 4.
@gluon.constexpr_function
def _dn_kt(c):
    return c // 4


@gluon.constexpr_function
def _dn_st(c):
    return c % 4


@gluon.constexpr_function
def _dn_sdw(c):
    """Scale dword (32-column block, step pair) of down chunk ``c``."""
    return (_dn_st(c) // 2) * 2 + _dn_kt(c) // 2


@gluon.constexpr_function
def _a_release(kt, poll_at):
    """Chunk after which this unit's A operand of K step ``kt`` is dead and
    the next unit's may take its registers (never before the readiness poll)."""
    return max(poll_at, 4 * kt + 3)


@gluon.constexpr_function
def _s_release(blk, p, poll_at):
    """Chunk after which the scale dword of (32-column block, step pair) is dead."""
    return max(poll_at, 4 * (2 * p + 1) + 2 * blk + 1)


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
    return gl.amd.cdna4.buffer.load(
        mids_d,
        route_a * _MID_ROW + kt * 64 + ak,
        mask=live & (route_a >= 0),
        scope="sys",  # see _store_mids
    )


@gluon.jit
def _load_a_scale(mids_sd, route_s, kt: gl.constexpr, live):
    """The A operand's E8M0 bytes of K step ``kt``: ``mids_s[route][g]`` is
    group g = 4 kt + K slice, so one dword holds a step's 4 slices."""
    lsa: gl.constexpr = _layouts()[3]
    ask = gl.arange(0, 4, layout=gl.SliceLayout(0, lsa))[None, :]
    return gl.amd.cdna4.buffer.load(
        mids_sd,
        route_s * (_MID_SCALES // 4) + kt + ask * 0,
        mask=live & (route_s >= 0),
        scope="sys",  # see _store_mids
    )


@gluon.jit
def _wait_quad(ex_count, counter, target):
    """Wait until the quad's exchange counter (byte ``counter`` into the
    shared ``ex_count``) reaches ``target``: all 4 waves stored the group's
    SwiGLU values. Reads LDS only."""
    lane: gl.constexpr = _layouts()[6]
    offsets = counter + gl.zeros([64 * gl.num_warps()], gl.int32, lane)
    goal = target + gl.zeros([64 * gl.num_warps()], gl.int32, lane)
    lds_spin_until(ex_count, offsets, goal)


@gluon.jit
def _wait_ready(tags, rank, tag):
    """Wait until every group tag of the expert at ``rank`` carries this
    launch's tag (64 words: 16 groups x 4 strips, each published by its
    wave). Per wave, without a workgroup barrier: the wave's own mids loads
    follow its poll in program order. Scalar loads, which do not wait behind
    the wave's weight loads."""
    for j in gl.static_range(_TAGS // 16):
        wait_tags16(tags + rank * _TAGS + 16 * j, tag)


@gluon.jit
def _store_routes(out_d, y, route, cols):
    """``out[route, cols] = bf16(y)`` for one strip's ``[16, 128]`` result
    (wave w's 16 columns, 4 consecutive per lane); rows with ``route < 0``
    are skipped. The rows are read after the launch ends, so the store needs
    no scope."""
    gl.amd.cdna4.buffer.store(
        y.to(gl.bfloat16),
        out_d,
        (route * _HIDDEN + cols).to(gl.int32),
        mask=(route >= 0) & (cols >= 0),
    )


def _moe_stream_launch_metadata(grid, kernel, args):
    return {"name": f"{kernel.name} (rows={args['ROWS']})"}


@gluon.jit(launch_metadata=_moe_stream_launch_metadata)
def gluon_mxfp4_moe_stream_gfx950(
    epochs,
    claims,
    ids,
    xq,
    xs,
    w13,
    s13,
    w2,
    s2,
    mids,
    mids_s,
    tags,
    out,
    stride_xq,
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
    ROWS: gl.constexpr,
    TOPK: gl.constexpr,
    N_PHYS1: gl.constexpr,
    N_PHYS2: gl.constexpr,
    TAIL_NUM: gl.constexpr,
    TAIL_DEN: gl.constexpr,
    SWIGLU_ALPHA: gl.constexpr,
    SWIGLU_LIMIT: gl.constexpr,
    SWIGLU_BETA: gl.constexpr,
):
    """The gate/up and down projections of ``ROWS`` decode rows in one launch
    of 256 workgroups x 8 waves; see the module docstring and
    :func:`launch_gluon_mxfp4_moe_stream_gfx950`."""
    mfma: gl.constexpr = _layouts()[0]
    la: gl.constexpr = _layouts()[1]
    lsa: gl.constexpr = _layouts()[3]
    rowl: gl.constexpr = _layouts()[5]
    lane: gl.constexpr = _layouts()[6]
    quant: gl.constexpr = _layouts()[7]
    gl.static_assert(_K_STEPS % _RING == 0)
    gl.static_assert(_RING % 2 == 0)
    gl.static_assert(ROWS * TOPK <= _MAX_RANKS)
    gl.static_assert(_TAG_COPY_WORDS == 32 * 256)
    K_PACKED: gl.constexpr = _HIDDEN // 2
    PICKS: gl.constexpr = ROWS * TOPK
    # The scales' row strides are 1, which a launch passes as constants.
    bufs = MoeBuffers(
        xq, xs, w13, s13, w2, s2, mids, mids_s, tags, stride_xq,
        gl.to_tensor(stride_xslin), stride_xsnb, stride_w13, stride_s13,
        gl.to_tensor(stride_s13lin), stride_s13nb, stride_w2, stride_s2,
        gl.to_tensor(stride_s2lin), stride_s2nb,
    )  # fmt: skip

    # ---- role: the plan places whole experts by the XCD that runs the
    # workgroup, so the role comes from
    # the hardware XCD id and the program id; the entry check below proves
    # every role is taken once.
    pid = gl.program_id(0)
    xcd = xcc_id()
    slot = pid // _XCDS
    cta = slot * _XCDS + xcd

    # ---- plan and first weights: they depend only on the routed ids, so
    # they go out before the entry check.
    rank_smem = gl.allocate_shared_memory(
        gl.int32, [PICKS], gl.SwizzledSharedLayout(1, 1, 1, order=[0])
    )
    ids_t = gl.allocate_shared_memory(
        gl.int32, [TOPK, 16], gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    )
    experts = _stage_ids(ids, rank_smem, ids_t, ROWS, TOPK)
    # The ids again in quant-exchange row order: ids_q[j, rho] = ids[token(rho), j].
    ids_q = gl.allocate_shared_memory(
        gl.int32, [TOPK, 32], gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    )
    QJ: gl.constexpr = gl.BlockedLayout([1, 1], [2, 32], [gl.num_warps(), 1], [1, 0])
    qj = gl.arange(0, TOPK, layout=gl.SliceLayout(1, QJ))[:, None]
    rho = gl.arange(0, 32, layout=gl.SliceLayout(0, QJ))[None, :]
    token = _token_of_quant_row(rho)
    ids_q.store(gl.load(ids + token * TOPK + qj, mask=token < ROWS, other=-1))
    plan = _plan_wave(xcd, slot, experts, TAIL_NUM, TAIL_DEN)
    expert, rank, group, _ = _group_of(plan, 0, rank_smem)
    w, w_pair, w_odd, s, s_pair = _strip_offsets(
        expert,
        group * 64 + plan.strip * 16,
        bufs.stride_w13,
        bufs.stride_s13,
        bufs.stride_s13lin,
        bufs.stride_s13nb,
        N_PHYS1,
    )
    w13d = gl.amd.cdna4.buffer.make_buffer_descriptor(bufs.w13, 0x7FFFFFF0)
    # The E8M0 scales are read a dword at a time.
    s13d = gl.amd.cdna4.buffer.make_buffer_descriptor(
        bufs.s13.to(gl.pointer_type(gl.int32), bitcast=True), 0x7FFFFFF0
    )
    live = plan.n_ug > 0
    ring = ()
    sring = ()
    for j in gl.static_range(_RING):
        # Ring order (slot 0 first): the loop's first wait is the minimum
        # over this entry and its back edge.
        if j % 2 == 0:
            sring = sring + (_load_scale_dword(s13d, s, s_pair, j // 2, live),)
        ring = ring + (_load_chunk(w13d, w, w_pair, w_odd, j, live),)
        gl.amd.hint.sched_barrier()

    # ---- entry check. Each workgroup counts its launches in its program
    # id's line of ``epochs``: every workgroup runs once per launch, so all
    # hold the same count, which gives the launch's tag and its tag copy.
    # Every wave reads the count with a scalar load (it does not wait behind
    # the weights above), the waves join so that no wave reads the word after
    # the store, and the count goes back with a relaxed atomic store. The workgroup then exchanges (tag, pid) into its
    # role's line of ``claims``; finding this launch's tag there means two
    # workgroups took one role, so some role is missing and its work would
    # be skipped: stop the launch instead.
    epoch_line = epochs + pid * _COUNTER_STRIDE
    epoch = scalar_load_glc_i64(epoch_line) + 1
    gl.barrier()
    gl.atomic_store(epoch_line, epoch, sem="relaxed", scope="gpu")
    # Nonzero and different in consecutive launches; repeats after
    # 2**32 - 1 launches (the zeroed tag copy covers that, see TAG_WORDS).
    tag = (epoch % (2**32 - 1) + 1).to(gl.int32)
    copy = (epoch & 1).to(gl.int32)
    claim_line = claims + (xcd * _SLOTS + slot) * _COUNTER_STRIDE
    mark = (tag.to(gl.int64) << 32) | pid.to(gl.int64)
    old = gl.atomic_xchg(claim_line, mark, sem="relaxed", scope="gpu")
    if (old >> 32).to(gl.int32) == tag:
        trap()

    # ---- activations (every token an A row) and the zeroed exchange
    # counters.
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
    x_all.index(0).store(
        gl.load(bufs.xq + xr * bufs.stride_xq + xc, mask=xr < ROWS, other=0)
    )
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
    # The counters are then updated and read by inline asm, which the
    # compiler's LDS hazard tracking does not see: every wave's zeroing must
    # land before any wave's first increment.
    gl.barrier()

    w2d = gl.amd.cdna4.buffer.make_buffer_descriptor(bufs.w2, 0x7FFFFFF0)
    s2d = gl.amd.cdna4.buffer.make_buffer_descriptor(
        bufs.s2.to(gl.pointer_type(gl.int32), bitcast=True), 0x7FFFFFF0
    )
    mids_d = gl.amd.cdna4.buffer.make_buffer_descriptor(bufs.mids, 0x7FFFFFF0)
    # The mids' E8M0 bytes: stored a byte per (route, group), read back as
    # int32 words (one per K step), so two views of the same buffer.
    mids_sd = gl.amd.cdna4.buffer.make_buffer_descriptor(bufs.mids_s, 0x7FFFFFF0)
    mids_sb = gl.amd.cdna4.buffer.make_buffer_descriptor(
        bufs.mids_s.to(gl.pointer_type(gl.uint8), bitcast=True), 0x7FFFFFF0
    )
    tags_d = gl.amd.cdna4.buffer.make_buffer_descriptor(bufs.tags, 0x7FFFFFF0)
    out_d = gl.amd.cdna4.buffer.make_buffer_descriptor(out, 0x7FFFFFF0)
    lanes = gl.arange(0, 64 * 8, layout=lane)
    # This launch's tag copy; zero this workgroup's share of the other one
    # (32 words, one line) for the next launch. Waves on other XCDs read the
    # tags, so the clear uses system scope like every other tag access.
    tag_base = copy * _TAG_COPY_WORDS
    gl.amd.cdna4.buffer.store(
        gl.zeros_like(lanes),
        tags_d,
        (1 - copy) * _TAG_COPY_WORDS + cta * 32 + lanes,
        mask=lanes < 32,
        scope="sys",
    )

    # ---- gate/up strips
    pend_word = rank * 0 - 1
    for k in range(plan.n_ug):
        # The next strip's plan, from LDS only.
        nexpert, nrank, ngroup, _ = _group_of(plan, k + 1, rank_smem)
        nlive = k + 1 < plan.n_ug
        nw, nw_pair, nw_odd, ns, ns_pair = _strip_offsets(
            nexpert,
            ngroup * 64 + plan.strip * 16,
            bufs.stride_w13,
            bufs.stride_s13,
            bufs.stride_s13lin,
            bufs.stride_s13nb,
            N_PHYS1,
        )
        # An opaque index keeps the 32 steps' A reads in the loop (hoisted,
        # they would hold 160 VGPRs).
        z = opaque_zero()
        xa = x_all.index(z)
        xsa = xs_all.index(z)
        acc = gl.zeros((16, 128), gl.float32, mfma)
        for kt in gl.static_range(_K_STEPS):
            b_scale = _scale_byte(sring[(kt // 2) % (_RING // 2)], s, kt)
            a = xa.slice(kt * 64, 64, dim=1).load(la)
            a_scale = xsa.slice(kt * 4, 4, dim=1).load(lsa)
            acc = _direct_mxfp4_mfma(acc, a, ring[kt % _RING], a_scale, b_scale)
            # Refill the consumed slot RING steps ahead: this strip, or the next.
            if kt + _RING < _K_STEPS:
                nb = _load_chunk(w13d, w, w_pair, w_odd, kt + _RING, None)
            else:
                nb = _load_chunk(
                    w13d, nw, nw_pair, nw_odd, kt + _RING - _K_STEPS, nlive
                )
            ring = ring[: kt % _RING] + (nb,) + ring[kt % _RING + 1 :]
            if kt % 2 == 1:
                if (kt - 1 + _RING) // 2 < _K_STEPS // 2:
                    nsd = _load_scale_dword(
                        s13d, s, s_pair, (kt - 1 + _RING) // 2, None
                    )
                else:
                    nsd = _load_scale_dword(
                        s13d, ns, ns_pair, (kt - 1 + _RING) // 2 - _K_STEPS // 2, nlive
                    )
                sring = (
                    sring[: (kt // 2) % (_RING // 2)]
                    + (nsd,)
                    + sring[(kt // 2) % (_RING // 2) + 1 :]
                )
            if kt == _RING:
                # The previous strip's tag: this step's chunk was issued after
                # that strip's mids stores, and the wave's memory operations
                # complete in issue order, so waiting for it covers them.
                _publish_tags(tags_d, tag_base + pend_word, tag, pend_word >= 0, acc)
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
            (par * 32 + _lds_row(plan.quad, tr)) * 128 + plan.strip * 32 + (jc % 8) * 4
        )
        counter = (par * 2 + plan.quad) * 4
        lds_store_count(ex, offs, mid, ex_count, counter + offs * 0)
        _wait_quad(ex_count, counter, 256 * (k // 2 + 1))
        packed, sbyte = _mxfp4_quantize_tile_in_layout(ex.index(par).load(quant))
        PL: gl.constexpr = packed.type.layout
        prho = gl.arange(0, 32, layout=gl.SliceLayout(1, PL))
        route = _routes(ids_q, expert, _token_of_quant_row(prho), TOPK)[:, None]
        pb = gl.arange(0, 16, layout=gl.SliceLayout(0, PL))[None, :]
        _store_mids(mids_d, packed, route * _MID_ROW + group * 16 + pb, route >= 0)
        SLY: gl.constexpr = sbyte.type.layout
        srho = gl.arange(0, 32, layout=gl.SliceLayout(1, SLY))
        sroute = _routes(ids_q, expert, _token_of_quant_row(srho), TOPK)[:, None]
        sg = gl.arange(0, 1, layout=gl.SliceLayout(0, SLY))[None, :]
        # mids_s[route][g]: group g is K step g // 4, K slice g % 4.
        _store_mids(mids_sb, sbyte, sroute * _MID_SCALES + group + sg, sroute >= 0)
        pend_word = rank * _TAGS + group * 4 + plan.strip
        expert, rank, group = nexpert, nrank, ngroup
        w, w_pair, w_odd, s, s_pair = nw, nw_pair, nw_odd, ns, ns_pair

    # ---- gate/up -> down: publish the last strip's tag after a drain, once
    # per wave (otherwise the wave's next loads would wait on its own tag)
    if pend_word >= 0:
        release_drain()
        _publish_tags(
            tags_d,
            tag_base + pend_word,
            tag,
            pend_word >= 0,
            gl.zeros((16, 128), gl.float32, mfma),
        )

    # ---- down units. Every operand that changes per unit is replaced in
    # place after its last use (no loop-carried copy, which would wait for a
    # load just issued): the weight ring by chunk, the A operand by K step,
    # the scale dwords by step pair, all after the next unit's readiness poll.
    POLL_AT: gl.constexpr = max(_DN_CHUNKS - _RING - 1, 0)
    dlive = plan.n_dn > 0
    dexpert, drank, dunit = _unit_of(plan, 0, rank_smem)
    wb, dw_pair, dw_odd, sb, ds_pair = _unit_offsets(dexpert, dunit, bufs, N_PHYS2)
    # First unit, in the steady state's issue order (side loads, then the
    # ring): the loop's first waits are the minimum over this entry and its
    # back edge. Its readiness wait therefore precedes its weights, once per
    # wave, right after the gate/up drain.
    if dlive:
        # Mids ready: the expert's group tags, polled with scalar loads.
        _wait_ready(bufs.tags + tag_base, drank, tag)
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
    for j in gl.static_range(_RING):
        ring2 = ring2 + (
            _load_chunk(w2d, wb[_dn_st(j)], dw_pair, dw_odd, _dn_kt(j), dlive),
        )
        gl.amd.hint.sched_barrier()
    ask = gl.arange(0, 4, layout=gl.SliceLayout(0, lsa))[None, :]
    for u in range(plan.n_dn):
        nexpert, nrank, nunit = _unit_of(plan, u + 1, rank_smem)
        nlive = u + 1 < plan.n_dn
        nwb, nw_pair, nw_odd, nsb, ns_pair = _unit_offsets(
            nexpert, nunit, bufs, N_PHYS2
        )
        nroute_a = route_a
        nroute_s = route_s
        zero = gl.zeros((16, 128), gl.float32, mfma)
        accs = (zero, zero, zero, zero)
        for c in gl.static_range(_DN_CHUNKS):
            b_scale = _scale_byte(sdw[_dn_sdw(c)], sb[_dn_st(c)], _dn_kt(c))
            a_scale = ((a_sc[_dn_kt(c)] >> (ask * 8)) & 255).to(gl.uint8)
            acc = _direct_mxfp4_mfma(
                accs[_dn_st(c)], a_t[_dn_kt(c)], ring2[c % _RING], a_scale, b_scale
            )
            accs = accs[: _dn_st(c)] + (acc,) + accs[_dn_st(c) + 1 :]
            if c == POLL_AT:
                # The next unit's readiness: scalar loads only.
                if nlive:
                    _wait_ready(bufs.tags + tag_base, nrank, tag)
                nroute_a, nroute_s = _unit_routes(ids_t, nexpert, TOPK)
            for kt in gl.static_range(_DN_STEPS):
                if c == _a_release(kt, POLL_AT):
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
                if c == _s_release(i // 2, i % 2, POLL_AT):
                    sdw = (
                        sdw[:i]
                        + (
                            _load_scale_dword(
                                s2d, nsb[2 * (i // 2)], ns_pair, i % 2, nlive
                            ),
                        )
                        + sdw[i + 1 :]
                    )
            if c + _RING < _DN_CHUNKS:
                nb = _load_chunk(
                    w2d, wb[_dn_st(c + _RING)], dw_pair, dw_odd, _dn_kt(c + _RING), None
                )
            else:
                nb = _load_chunk(
                    w2d,
                    nwb[_dn_st(c + _RING - _DN_CHUNKS)],
                    nw_pair,
                    nw_odd,
                    _dn_kt(c + _RING - _DN_CHUNKS),
                    nlive,
                )
            ring2 = ring2[: c % _RING] + (nb,) + ring2[c % _RING + 1 :]
            gl.amd.hint.sched_barrier()
        # Down unit -> out, in the MFMA result's own layout: no data moves,
        # and each wave stores its own 16 columns.
        rows = gl.arange(0, 16, layout=gl.SliceLayout(1, rowl))
        route = _routes(ids_t, dexpert, rows, TOPK)[:, None]
        pos = gl.arange(0, 128, layout=gl.SliceLayout(0, rowl))[None, :] % 16
        for st in gl.static_range(4):
            y = gl.convert_layout(accs[st], rowl, assert_trivial=True)
            _store_routes(out_d, y, route, dunit * 64 + st * 16 + pos)
        dexpert, drank, dunit = nexpert, nrank, nunit
        wb, dw_pair, dw_odd, sb, ds_pair = nwb, nw_pair, nw_odd, nsb, ns_pair
        route_a, route_s = nroute_a, nroute_s


# ----------------------------------------------------------------------------
# Host
# ----------------------------------------------------------------------------


class MoeStreamState:
    """Launch state of the stream on one device, reused by every launch and
    every layer: per-workgroup launch counters and role claims, the mids
    hand-off for up to ``MAX_ROWS`` rows, and the readiness tags.

    Launches that share a state must run one after another (one stream, or
    graph replays of it): each launch zeroes the tag copy the next one polls
    and overwrites the mids. The counters and tags carry over graph replays,
    so a captured launch stays correct when replayed. Create it outside graph
    capture; :meth:`free` releases the uncached tags.

    Args:
        device: The GPU.
    """

    def __init__(self, device: torch.device) -> None:
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            # Launches compare this with their tensors' devices, which carry
            # an index.
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        self.epochs = torch.zeros(COUNTER_WORDS, dtype=torch.int64, device=device)
        self.claims = torch.zeros(COUNTER_WORDS, dtype=torch.int64, device=device)
        routes = MAX_ROWS * TOPK
        self.mids = torch.zeros(routes * INTER // 2, dtype=torch.uint8, device=device)
        self.mids_scale = torch.zeros(
            routes * INTER // 32 // 4, dtype=torch.int32, device=device
        )
        # Written on one XCD and polled with scalar loads on others (the split
        # leftover experts), so uncached (see UncachedBuffer).
        self.tags = UncachedBuffer(TAG_WORDS, torch.int32, self.device)

    def free(self) -> None:
        self.tags.free()


@dataclasses.dataclass(frozen=True)
class MoeStreamWeights:
    """A processed MoE layer's tensors as the stream reads them, in place.

    ``s13`` / ``s2`` are the CDNA4-swizzled E8M0 scales (uint8, read a dword
    at a time); strides are in bytes.
    """

    w13: torch.Tensor
    s13: torch.Tensor
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


# Bytes a buffer descriptor covers (num_records of the weight, scale, and
# workspace descriptors).
_BUFFER_BYTES = 0x7FFFFFF0


def moe_stream_weights(experts) -> MoeStreamWeights:
    """The stream's view of a processed gfx950 MXFP4 MoE layer.

    Reads the tensors the direct decode reads (``w13_weight_triton_tensor``
    and ``w2_weight_triton_tensor``, gdot128-shuffled, with their precision
    configs' CDNA4-swizzled scales), without copying them.

    Args:
        experts: The MoE layer after ``preprocess_gluon_mxfp4_gfx950_moe_weights``.

    Raises:
        ValueError: The layout is not the slice the stream tiles (``(E, 2048,
            1024)`` gate/up, ``(E, 256, 4096)`` down, gdot128-shuffled), a
            scale buffer does not start on a dword, a stride the kernel
            specializes on is not a multiple of 16, or a weight or scale
            tensor spans more bytes than the kernel's int32 buffer offsets
            reach.
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
            f"moe_stream_weights: gate/up weights must be (E, {HIDDEN // 2}, "
            f"{GATE_UP}) uint8 gdot128, got {tuple(w13.shape)} {w13.dtype}"
        )
    if w2.dtype != torch.uint8 or w2.shape != (experts_n, INTER // 2, HIDDEN):
        raise ValueError(
            f"moe_stream_weights: down weights must be (E, {INTER // 2}, "
            f"{HIDDEN}) uint8 gdot128, got {tuple(w2.shape)} {w2.dtype}"
        )
    for name, weight in (("gate/up", w13), ("down", w2)):
        if (
            not getattr(weight, "is_shuffled_for_gluon_dot", False)
            or getattr(weight, "gluon_dot_block_k_pk", 0) != 128
            or getattr(weight, "gluon_dot_block_n", 0) != 128
        ):
            raise ValueError(
                f"moe_stream_weights: {name} weights must be gdot128-shuffled"
            )
    for name, scale in (("gate/up", s13), ("down", s2)):
        if (
            scale.dtype != torch.uint8
            or scale.ndim != 3
            or scale.stride(1) != 1
            or scale.data_ptr() % 4
        ):
            raise ValueError(
                f"moe_stream_weights: {name} scales must be CDNA4-swizzled uint8 "
                "(stride(1) == 1) starting on a dword, got "
                f"{scale.dtype} strides {scale.stride()}"
            )
    # _strip_offsets keeps each lane's byte offset into a weight or scale
    # tensor (expert * stride plus the in-expert offset) in int32, and the
    # buffer descriptors cover 0x7FFFFFF0 bytes, past which loads return
    # zeros. Refuse tensors that reach past that window.
    for name, tensor in (
        ("gate/up weights", w13),
        ("gate/up scales", s13),
        ("down weights", w2),
        ("down scales", s2),
    ):
        extent = 1 + sum((n - 1) * st for n, st in zip(tensor.shape, tensor.stride()))
        if extent > _BUFFER_BYTES:
            raise ValueError(
                f"moe_stream_weights: {name} must span at most {_BUFFER_BYTES} "
                f"bytes, the reach of the kernel's int32 buffer offsets, got {extent}"
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
            f"moe_stream_weights: strides must be multiples of 16, got {uneven}"
        )
    return MoeStreamWeights(
        w13=w13,
        s13=s13,
        w2=w2,
        s2=s2,
        n_phys1=w13.shape[2],
        n_phys2=w2.shape[2],
        **strides,
    )


def launch_gluon_mxfp4_moe_stream_gfx950(
    xq: torch.Tensor,
    xs: torch.Tensor,
    weights: MoeStreamWeights,
    topk_ids: torch.Tensor,
    out: torch.Tensor,
    state: MoeStreamState,
    *,
    tail_num: int,
    tail_den: int,
    swiglu_alpha: float,
    swiglu_limit: float,
    swiglu_beta: float,
) -> torch.Tensor:
    """Run the gate/up and down projections of a few decode rows in one launch.

    Args:
        xq, xs: MXFP4 activation rows ``(rows, 2048)`` (unit column stride)
            and their CDNA4-swizzled E8M0 scales.
        weights: From :func:`moe_stream_weights`.
        topk_ids: Contiguous ``(rows, 8)`` int32 expert ids, indices into
            the layer's experts; ``rows`` is a compile-time size, so call it
            only for a fixed small set of row counts (at most 16).
        out: Contiguous ``(rows * 8, 4096)`` bf16, written per route.
        state: :class:`MoeStreamState` on the same device.
        tail_num, tail_den: The tail workgroups' bytes as a fraction of a
            full one's; :data:`TAIL_SHARE` is the measured choice.
        swiglu_alpha, swiglu_limit, swiglu_beta: The layer's SwiGLU.

    Returns:
        ``out``.
    """
    rows, top_k = topk_ids.shape
    if not 0 < rows <= MAX_ROWS or top_k != TOPK:
        raise ValueError(
            "launch_gluon_mxfp4_moe_stream_gfx950: topk_ids must be (rows, "
            f"{TOPK}) with 1 <= rows <= {MAX_ROWS}, got {tuple(topk_ids.shape)}"
        )
    if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
        raise ValueError(
            "launch_gluon_mxfp4_moe_stream_gfx950: topk_ids must be contiguous int32"
        )
    if xq.dtype != torch.uint8 or xq.shape != (rows, HIDDEN // 2) or xq.stride(1) != 1:
        raise ValueError(
            "launch_gluon_mxfp4_moe_stream_gfx950: activations must be "
            f"(rows, {HIDDEN // 2}) uint8 rows, got {tuple(xq.shape)} {xq.dtype}"
        )
    if (
        out.dtype != torch.bfloat16
        or out.shape != (rows * TOPK, HIDDEN)
        or not out.is_contiguous()
    ):
        raise ValueError(
            "launch_gluon_mxfp4_moe_stream_gfx950: out must be contiguous "
            f"({rows * TOPK}, {HIDDEN}) bf16, got {tuple(out.shape)} {out.dtype}"
        )
    if xs.dtype != torch.uint8 or not is_swizzled_cdna4_mxfp4_scale(xs):
        raise ValueError(
            "launch_gluon_mxfp4_moe_stream_gfx950: activation scales must be "
            f"CDNA4-swizzled E8M0, got {xs.dtype} shape {tuple(xs.shape)} "
            f"strides {xs.stride()}"
        )
    if state.device != out.device:
        raise ValueError(
            "launch_gluon_mxfp4_moe_stream_gfx950: state is on "
            f"{state.device}, out on {out.device}"
        )
    w = weights
    gluon_mxfp4_moe_stream_gfx950[(GRID,)](
        state.epochs, state.claims, topk_ids, xq, xs, w.w13, w.s13, w.w2, w.s2,
        state.mids, state.mids_scale, state.tags, out,
        xq.stride(0), xs.stride(0), xs.stride(1), w.stride_w13, w.stride_s13,
        w.stride_s13lin, w.stride_s13nb, w.stride_w2, w.stride_s2,
        w.stride_s2lin, w.stride_s2nb,
        ROWS=rows, TOPK=TOPK, N_PHYS1=w.n_phys1, N_PHYS2=w.n_phys2,
        TAIL_NUM=tail_num, TAIL_DEN=tail_den, SWIGLU_ALPHA=swiglu_alpha,
        SWIGLU_LIMIT=swiglu_limit, SWIGLU_BETA=swiglu_beta,
        num_warps=8, **stream_sync_compile_options(),
    )  # fmt: skip
    return out

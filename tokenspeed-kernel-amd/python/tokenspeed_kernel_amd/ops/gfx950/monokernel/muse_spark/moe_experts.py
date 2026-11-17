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

"""MXFP4 experts of one Muse Spark rank (8 picks a row).

At TP8 / EP1 every rank holds all 256 experts, each sharded over its
intermediate dimension: the rank's 512 of 4096. The stages read the regular
path's preprocessed weights in place (gdot128-shuffled E2M1 bytes and
CDNA4-swizzled E8M0 scales, as ``decode_stage1`` / ``decode_stage2`` read
them): a second copy of every expert would not fit.

The launch's rows pick ``U`` distinct experts (8 for one row; consecutive
tokens share many). A pick owns its expert when no earlier row picked it;
slot ``i`` is the ``i``-th owning pick in (row, pick) order. Row 0 owns all
its picks, so slot ``i < 8`` is row 0's pick ``i`` at any row count. A task
of lane ``l`` takes slots ``l, l + 8, ...``: each expert's weights stream
once for every row that picked it. Row ``r``'s result for slot ``i`` goes to
row ``r``'s entry of the pick that chose that expert, so the combine reads
each row's 8 picks as for one row.

* xq, a task a K quarter of a group of up to 2 rows (:func:`xq_rows`):
  polls the group's whole pooled rows, applies the pre-expert RMS norm, and
  quantizes the quarter to MXFP4 (the regular path's "even" rule), then publishes the
  E2M1 bytes and E8M0 scales as plain words under one flag.
* ex1, 256 tasks: lane ``l``, 128 of the 1024 interleaved gate / up columns,
  a quarter of K (64 KiB of weights a slot). A task issues its first slot's
  weights once routing names the experts, then loads its K quarter of every
  row from xq into LDS, one MFMA row each (136 words a row, not every whole
  pooled row on each of the 256 CTAs). It publishes the FP32 partial sums of
  the rows that picked the slot's expert.
* exr, 64 tasks at one row (lane ``l``, one ex1 column block), 256 past one
  row (each lane's slots split over EXR_PARTS tasks, one a CTA). For each of
  its slots, a task polls the 4 K quarters of every row that picked the
  slot's expert, sums them in K order, applies SwiGLU (``silu(gate) up``)
  and the regular path's MXFP4 requantization (groups of 32: the block's 64
  values are two whole groups), and publishes them as words (``h1``). Done
  once here, an ex2 task reads 72 words a row instead of 4096 partials and
  quantizes nothing: polled by all 256 CTAs at once, the partials cost a
  round trip a row bound by the memory-side bandwidth, so a slot that many
  rows picked would hold its lane up, and each of a lane's 32 ex2 tasks
  would requantize the same rows.
* ex2, 256 tasks: lane ``l``, 128 of the 4096 output columns over the rank's
  512 intermediate values (32 KiB a slot). A task polls the MXFP4
  intermediates of its lane's first 8 slots at once (every row of each)
  into LDS, its first 4 slots' weights issued before that poll, then runs
  those slots with each slot's weights 4 slots ahead and no poll between
  them. Past one row the gather also keeps each slot's row picks in LDS, so
  a slot reads them in one load rather than a reduction a row. The BF16
  partials go to the rank that owns those columns (``rr``), which sums the
  8 ranks' partials.

The slot table (:func:`slots`: each pick's slot, whether it owns its
expert, every row's pick of it) is built once, by one task after the
routing (:func:`slot_table_stage`), and published in ``slt``; built on all
256 CTAs it would lengthen each ex1 task's prelude. Past 8 rows each CTA
also keeps its lane's slots' experts and row picks in LDS
(:func:`cta_slots`), which ex1, exr and ex2
read with one load a slot instead of two reductions over the table.

Expert weights cannot load before routing picks the experts. So a CTA runs
the first slot of its ex1 and ex2 tasks in parts: :func:`ex1_poll_routes`
(the first slots' experts and the slot table), :func:`ex1_issue`, then
:func:`union_of` and :func:`ex1_xq` (the A operand from xq, while the
weights stream), :func:`ex1_finish`, then :func:`ex2_issue` and
:func:`ex2_finish` at ex2's place in the plan. ex2's first weights issue
after ex1's products: issued with ex1's they sit on ex1's critical path,
and they still arrive before ex2's poll returns. The loads come before the polls: a poll
waits for every earlier load of its wave (``vmcnt`` counts in order), so a
load after a poll would start only when the poll returns. The finish steps
then run the lane's other slots, one after another.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._extern import (
    wave_ballot_i64,
    wave_mbcnt,
    wave_popcount_i64,
)
from tokenspeed_kernel_amd._triton import cdna4_async_copy, gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.decode_common import (
    _cdna4_swizzled_mxfp4_scale_offset,
    _direct_mxfp4_mfma,
    _direct_mxfp4_mfma_layouts,
    _gluon_dot_preshuffled_w_offset,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.quantize_gluon import (
    _mxfp4_quantize_tile,
    _mxfp4_quantize_tile_in_layout,
    scaled_downcast_layout,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel import mx, schedule, trace
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.dims import (
    POOLED,
    PRE_EPS,
    WORLD,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import Stage, placement

TOP_K = gl.constexpr(8)
LANES = gl.constexpr(8)  # slot lanes: the tasks of a column block
INTER = gl.constexpr(512)  # the rank's intermediate values of an expert
W13_N = gl.constexpr(1024)  # interleaved gate / up columns
BLOCK_N = gl.constexpr(128)
M_DUP = gl.constexpr(16)  # MFMA rows; the activation row repeats in each
KSPLIT = gl.constexpr(4)
K_SLICE = gl.constexpr(POOLED // KSPLIT)
NB1 = gl.constexpr(W13_N // BLOCK_N)
NB2 = gl.constexpr(POOLED // BLOCK_N)
EX1_TASKS = gl.constexpr(LANES * NB1 * KSPLIT)
# Past one row each lane's slots split over EXR_PARTS exr tasks (a slot of
# every EXR_PARTS-th round each), one task a CTA: an exr task then holds its
# CTA's ex2 task back by about one slot, whatever the slot count. On CTAs
# 192-255 alone, the lanes there would run a round of exr, and when they
# also had the most slots (slots % 8 = 7 or 0) every ex2 lane would wait
# for it. One row has one slot a lane: one part. A launch
# that ends with qnext splits exr in 2 parts instead (layer.NEXT_BASES).
EXR_PARTS = gl.constexpr(4)
EXR_LANE_TASKS = gl.constexpr(LANES * NB1)  # one part
# An intermediate row in MXFP4 words: 64 of E2M1 values, then a word per
# ex1 block holding its 2 scale bytes.
H1_WORDS = gl.constexpr(INTER // 8 + NB1)
# h1's room a pick, past the H1_WORDS it holds. The scratch layout (each
# region's offset within its 4 KiB page, and the row stride) affects the
# launch time, and this sizing sets the offsets of the regions after h1 and
# the row stride: change it only with a layer perf A/B.
H1_REGION_WORDS = 256
EX2_TASKS = gl.constexpr(LANES * NB2)  # a 128-column block each
# An ex2 task's first slots (all of them up to 64 distinct experts) are
# gathered in one poll and run unrolled (:func:`ex2_finish`).
EX2_GATHER = gl.constexpr(8)
# Slots whose weights an ex2 task issues before that gather (then one more
# a slot). The gather waits for them anyway (vmcnt counts in order), and it
# waits on exr too, so up to 4 slots a lane (32 distinct experts) stream
# while exr finishes instead of after the gather.
EX2_AHEAD = gl.constexpr(4)
OWNED_OUT = gl.constexpr(POOLED // WORLD)  # output columns a rank reduces

# Group ``g`` of 32 expert CTAs serves lane ``g``: slot i runs on lane
# i % 8, so when U % 8 != 0 lanes 0 .. U % 8 - 1 carry one slot more.


PEERS = ("rr",)
# The slot table (:func:`slot_table_stage`), plain words in row 0 under one
# flag (sltf): a word a pick (its id, slot, whether it owns its expert, the
# slot count), then each pick's row picks ``[pick, row]``. Sized for 16
# rows. Plain words: as tagged pairs, all 256 ex1 CTAs would poll one CTA's
# 2176 pairs.
SLT_PICKS = gl.constexpr(128)
SLT_WORDS = gl.constexpr(128 + 128 * 16)
# The expert input (:func:`xq_stage`): each row's K quarters normed and
# quantized to MXFP4 once, as plain words in the row's copy of xq: a
# quarter's 512 E2M1 bytes (128 words), then its 32 E8M0 scales (8 words).
# An xq task does one K quarter of a group of rows (one poll round trip)
# and raises flag ``quarter * XQ_GROUPS + group`` in row 0 of xqf. ex1 then
# loads 136 words a row instead of polling every whole pooled row (16 KiB of
# pairs) on each of its 256 CTAs. A launch's rows split into up to
# XQ_GROUPS groups (:func:`xq_rows`): one row a group up to 8 rows, two at
# 16. An xq task polls its group's whole pooled rows (8 KiB of tagged pairs
# a row) for the norm, so fewer rows a group shorten xq's poll; at 4 rows a
# group the xq CTAs' ex1 falls behind the others at 1 x 8. With one row a
# group at 16 rows, the 64 tasks would wrap onto the heaviest ex1 lanes'
# CTAs.
XQ_QUARTER_WORDS = gl.constexpr(K_SLICE // 8 + K_SLICE // 128)
XQ_GROUPS = gl.constexpr(8)


@gluon.constexpr_function
def xq_rows(rows):
    """Rows of an xq row group in a launch of ``rows`` rows."""
    rp = 1 << (int(rows) - 1).bit_length()
    return max(1, rp // int(XQ_GROUPS))


def regions() -> tuple[tuple[str, int], ...]:
    k, nb1, ks, bn = int(TOP_K), int(NB1), int(KSPLIT), int(BLOCK_N)
    return (
        ("slt", int(SLT_WORDS) // 2),  # words (half the region)
        ("sltf", 1),
        ("xq", int(KSPLIT) * int(XQ_QUARTER_WORDS) // 2),  # words
        ("xqf", int(KSPLIT) * int(XQ_GROUPS)),
        ("p1", k * nb1 * ks * bn),  # FP32 [pick][block][K quarter][column] a row
        # The MXFP4 intermediate [pick][word] a row (H1_WORDS a pick) in a
        # region of H1_REGION_WORDS a pick (see there).
        ("h1", k * int(H1_REGION_WORDS)),
        (
            "rr",
            int(WORLD) * k * int(OWNED_OUT) // 2,
        ),  # BF16 pairs [source][pick][column] a row
    )


def xq_tasks(rows: int) -> int:
    """The xq tasks of a launch of ``rows``: a K quarter of a row group each."""
    rp = 1 << (rows - 1).bit_length()
    return int(KSPLIT) * (rp // xq_rows(rows))


def stages(bases: dict[str, int], rows: int, exr_parts: int) -> tuple[Stage, ...]:
    """slt, xq, ex1, exr and ex2 of ``rows`` token rows, placed by ``bases``;
    each lane's slots split over ``exr_parts`` exr tasks (1, 2, or EXR_PARTS
    to spread exr over every CTA)."""
    if exr_parts not in (1, 2, int(EXR_PARTS)):
        raise ValueError(f"exr parts: 1, 2 or {int(EXR_PARTS)}, got {exr_parts}")
    slt, xq, ex1, exr, ex2 = placement(bases, ("slt", "xq", "ex1", "exr", "ex2"))
    return (
        Stage("slt", 1, slt, reads=("route",), writes=("slt", "sltf")),
        Stage("xq", xq_tasks(rows), xq, reads=("cg",), writes=("xq", "xqf")),
        Stage(
            "ex1",
            int(EX1_TASKS),
            ex1,
            reads=("slt", "sltf", "xq", "xqf", "route"),
            writes=("p1",),
        ),
        Stage(
            "exr",
            int(EXR_LANE_TASKS) * exr_parts,
            exr,
            reads=("route", "p1"),
            writes=("h1",),
        ),
        Stage("ex2", int(EX2_TASKS), ex2, reads=("route", "h1"), writes=("rr",)),
    )


@gluon.constexpr_function
def _pick_layout(NP, num_warps):
    """``[pick, pick]`` of the launch's ``NP`` (row, pick) ids: a lane per
    pick along axis 0 (the waves repeat it), every pick in each thread along
    axis 1."""
    return gl.BlockedLayout([1, NP], [64, 1], [num_warps, 1], [1, 0])


@gluon.jit
def slots(ids):
    """The slots of the launch's picks ``ids [NP]`` (from :func:`ex1_poll_routes`, a
    lane each, rows major). Returns ``(ids, owner, slot, count, table)``:
    the ids, whether each pick owns its expert, and its slot (a lane each),
    the number of slots, and each pick's row picks ``[NP, rows]`` (for row
    ``r``, the pick of ``r`` that chose the same expert, or -1). Only a
    wave's lanes talk.

    It issues no global load: it runs while the first slots' weights
    stream."""
    NP: gl.constexpr = ids.shape[0]
    L2: gl.constexpr = _pick_layout(NP, gl.num_warps())
    ids = gl.convert_layout(ids, gl.SliceLayout(0, L2))
    q = gl.arange(0, NP, gl.SliceLayout(0, L2))[None, :]
    p = gl.arange(0, NP, gl.SliceLayout(1, L2))[:, None]
    mine = gl.convert_layout(ids, gl.SliceLayout(1, L2))[:, None]
    # A pick owns its expert unless an earlier row picked it (a row's picks
    # are distinct).
    seen = (ids[None, :] == mine) & (q // TOP_K < p // TOP_K)
    # A pick a lane; the scan lowers on a plain layout, not on a slice.
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    owner = gl.convert_layout(1 - gl.max(seen.to(gl.int32), axis=1), lanes)
    slot = gl.cumsum(owner, axis=0) - owner
    # Each pick's row picks: for row r, the pick of r that chose the same
    # expert (or -1). A slot then reads its rows' picks in one reduction
    # (:func:`_row_picks`), not one a row.
    RP: gl.constexpr = NP // TOP_K
    same = ids[None, :] == mine
    k = gl.where(same, q % TOP_K, -1)
    table = gl.max(gl.reshape(k, [NP, RP, TOP_K]), axis=2)
    # The ids stay a lane each, not a copy in every thread: the expert steps
    # run at the register limit past one row, and the copies spill.
    return (
        gl.convert_layout(ids, lanes),
        owner,
        slot,
        gl.sum(owner, axis=0),
        table,
    )


@gluon.jit
def slot_table_stage(mb, STAGE: gl.constexpr):
    """The launch's slots (:func:`slots`), once: every row's picks, then
    each pick's word (id, slot, owner, slot count) and row picks as plain
    words in ``slt`` (row 0) under one flag (``sltf``). ex1 polls the flag
    with the pooled rows, then loads the words, instead of building the
    table on every CTA (:func:`union_of`)."""
    mb.stamp(STAGE, trace.START)
    R: gl.constexpr = schedule.rows(mb.plan)
    RP: gl.constexpr = triton.next_power_of_2(R)
    NP: gl.constexpr = RP * TOP_K
    nw: gl.constexpr = gl.num_warps()
    if NP <= 64:
        # A pick a lane of every wave: the slots by wave ballots.
        pk = gl.arange(0, NP, gl.BlockedLayout([1], [64], [nw], [0]))
        (ids,) = mb.poll_many(
            ("route",), (mb.row_offset(gl.minimum(pk // TOP_K, R - 1)) + pk % TOP_K,)
        )
        mb.stamp(STAGE, trace.READY)
        ids, owner, slot, count, table = _ballot_slots(ids, RP)
    elif NP == 128:
        # Two picks a lane (2 l and 2 l + 1, one row) of every wave.
        pk = gl.arange(0, NP, gl.BlockedLayout([2], [64], [nw], [0]))
        (ids,) = mb.poll_many(
            ("route",), (mb.row_offset(gl.minimum(pk // TOP_K, R - 1)) + pk % TOP_K,)
        )
        mb.stamp(STAGE, trace.READY)
        ids, owner, slot, count, table = _ballot_slots2(ids, RP)
    else:
        pk = gl.arange(0, NP, gl.SliceLayout(1, _pick_layout(NP, nw)))
        (ids,) = mb.poll_many(
            ("route",), (mb.row_offset(gl.minimum(pk // TOP_K, R - 1)) + pk % TOP_K,)
        )
        mb.stamp(STAGE, trace.READY)
        ids, owner, slot, count, table = slots(ids)
    lanes: gl.constexpr = ids.type.layout
    p = gl.arange(0, NP, lanes)
    word = ids | (slot << 8) | (owner << 15) | (count << 16)
    mb.stamp(STAGE, trace.COMPUTED)
    mb.store_words("slt", p, word)
    tl: gl.constexpr = table.type.layout
    tp = gl.arange(0, NP, gl.SliceLayout(1, tl))[:, None]
    tr = gl.arange(0, RP, gl.SliceLayout(0, tl))[None, :]
    mb.store_words("slt", SLT_PICKS + tp * RP + tr, table)
    mb.raise_flag("sltf", 0)
    mb.stamp(STAGE, trace.END)


@gluon.constexpr_function
def _table_layout(NP, RP, num_warps):
    """``[pick, row]`` of the polled table: a pick's row a thread."""
    return gl.BlockedLayout([1, RP], [64, 1], [num_warps, 1], [1, 0])


@gluon.jit
def union_of(polled):
    """:func:`slots`'s result from the polled slot table (``ex1_poll_routes``)."""
    words, table = polled
    ids = words & 0xFF
    slot = (words >> 8) & 0x7F
    owner = (words >> 15) & 1
    count = gl.max(words >> 16, axis=0)
    return ids, owner, slot, count, table


@gluon.constexpr_function
def _slots_asm(outputs, inputs):
    """:func:`slots` of up to 64 picks (rows of 8) by wave ballots: lane
    ``p`` holds pick ``p``'s id (``NP = 8 len(table)`` picks; lanes past
    them repeat them). For each pick ``q`` (v_readlane), every lane compares
    its id: a hit gives the lane's row-``q // 8`` pick (``q % 8``) and, for a
    lane of a later row, marks its expert as seen in an earlier row. A pick
    owns its expert unless seen; its slot is the owner picks below it
    (v_mbcnt of the owners' ballot), the count all owner picks."""
    (owner,), (slot,), (count,), table = outputs[0:4]
    lane, row, seen, hit, t = (o[0] for o in outputs[4:9])
    (sq,) = outputs[9]
    ((ident,),) = inputs
    RP = len(table)
    NP = RP * 8
    a = [
        # Five independent VALUs precede the first input lane read.
        f"v_mbcnt_lo_u32_b32 {lane}, -1, 0",
        f"v_mbcnt_hi_u32_b32 {lane}, -1, {lane}",
        # Past NP lanes repeat the picks: lane l holds pick l % NP.
        f"v_and_b32 {lane}, {NP - 1}, {lane}",
        f"v_lshrrev_b32 {row}, 3, {lane}",
        f"v_mov_b32 {seen}, 0",
    ]
    # A VALU reads an SGPR that a VALU (v_readlane) wrote only 2 wait states
    # later: pick q + 1's id is read right after pick q's compare, and its
    # two v_cndmask (or more) pass before the next compare.
    a += [f"v_readlane_b32 {sq}, {ident}, 0"]
    a += [f"v_mov_b32 {x}, -1" for x in table]
    for r in range(RP):
        a += [f"v_mov_b32 {hit}, 0"]
        for k in range(8):
            q = r * 8 + k
            a += [f"v_cmp_eq_u32_e32 vcc, {sq}, {ident}"]
            if q + 1 < NP:
                a += [f"v_readlane_b32 {sq}, {ident}, {q + 1}"]
            a += [
                f"v_cndmask_b32_e64 {table[r]}, {table[r]}, {k}, vcc",
                f"v_cndmask_b32_e64 {hit}, {hit}, 1, vcc",
            ]
        # A lane of a later row saw row r's experts.
        a += [
            f"v_cmp_lt_u32_e32 vcc, {r}, {row}",
            f"v_or_b32 {t}, {seen}, {hit}",
            f"v_cndmask_b32_e32 {seen}, {seen}, {t}, vcc",
        ]
    a += [
        f"v_sub_u32 {owner}, 1, {seen}",
        # The owner picks' ballot, of the first NP lanes (the rest repeat).
        f"v_cmp_ne_u32_e32 vcc, 0, {owner}",
    ]
    if NP == 64:
        a += [
            # CDNA4 ISA Table 11: VALU VCC -> mbcnt constant; s_bcnt fills state 2.
            "s_nop 0",
            f"s_bcnt1_i32_b64 {count}, vcc",
            f"v_mbcnt_lo_u32_b32 {slot}, vcc_lo, 0",
            f"v_mbcnt_hi_u32_b32 {slot}, vcc_hi, {slot}",
        ]
    else:
        # A slot counts the owners below its pick: the ballot's bits below
        # l % NP (a mask of the first NP <= 32 lanes).
        a += [
            # VALU -> SALU is interlocked; later VALUs read SALU-written VCC.
            f"s_and_b32 vcc_lo, vcc_lo, {(1 << NP) - 1:#x}",
            f"s_bcnt1_i32_b32 {count}, vcc_lo",
            f"v_bfm_b32 {t}, {lane}, 0",
            f"v_and_b32 {t}, vcc_lo, {t}",
            f"v_bcnt_u32_b32 {slot}, {t}, 0",
        ]
    return "\n".join(a)


@gluon.constexpr_function
def _slots2_rows_asm(outputs, inputs):
    """The table entries of one wave's two rows for 128 picks (16 rows of 8),
    two picks a lane: lane ``l`` holds picks ``2 l`` and ``2 l + 1`` (both of
    row ``l // 4``). Element ``(e, j)`` of a lane is pick ``2 l + e``'s entry
    for row ``2 w + j`` (``w`` the wave): that row's pick (``q % 8``) of the
    same expert, or -1. Each of the rows' 16 picks ``q`` (v_readlane of its
    lane's pick) is compared with both of every lane's picks, in a loop of 4
    passes (row picks ``2 kk`` and ``2 kk + 1`` in pass ``kk``). A row's
    picks are distinct, so at most one pick hits an entry."""
    tab, tmp = outputs[0], outputs[1]
    (sb,), (sl,), (sc,) = outputs[2:5]
    sq = [o[0] for o in outputs[5:9]]
    ident, rows = inputs
    picks = (ident[0], ident[2])  # elements (0, 0) and (1, 0)
    kv = (tmp[0], tmp[1])
    a = [
        # rows[0] holds 2 w in every lane: the wave's first row pick 2 w * 8
        # sits in lane 8 w. CDNA4 ISA Table 11: a VALU write of a VGPR needs one
        # wait state before a lane read of it, and the compiler's write of
        # rows[0] may be the instruction just before this block.
        "s_nop 0",
        f"v_readfirstlane_b32 {sb}, {rows[0]}",
        # VALU -> SALU is interlocked.
        f"s_lshl_b32 {sb}, {sb}, 2",
    ]
    a += [f"v_mov_b32 {x}, -1" for x in tab]
    a += [
        f"v_mov_b32 {kv[0]}, 0",
        f"v_mov_b32 {kv[1]}, 1",
        f"s_mov_b32 {sc}, 0",
        "slots2_${:uid}:",
    ]
    # The pass's 4 row picks (rows 2 w + j, picks 2 kk and 2 kk + 1: lane
    # 8 w + 4 j + kk) are read first, each into its own SGPR: a v_readlane's
    # SGPR may feed a VALU only 2 wait states later, and here every compare
    # comes at least 3 VALU after the read of its pick. The lane selects are
    # SALU-written, which needs no wait.
    a += [f"s_add_u32 {sl}, {sb}, {sc}"]
    for j in range(2):
        if j:
            a += [f"s_add_u32 {sl}, {sl}, 4"]
        for par in range(2):
            a += [f"v_readlane_b32 {sq[2 * j + par]}, {picks[par]}, {sl}"]
    for j in range(2):
        for par in range(2):
            for e in range(2):
                a += [
                    f"v_cmp_eq_u32_e32 vcc, {sq[2 * j + par]}, {picks[e]}",
                    f"v_cndmask_b32_e32 {tab[2 * e + j]}, {tab[2 * e + j]}, {kv[par]}",
                ]
    a += [
        f"v_add_u32 {kv[0]}, 2, {kv[0]}",
        f"v_add_u32 {kv[1]}, 2, {kv[1]}",
        f"s_add_u32 {sc}, {sc}, 1",
        f"s_cmp_lt_u32 {sc}, 4",
        "s_cbranch_scc1 slots2_${:uid}",
    ]
    return "\n".join(a)


@gluon.constexpr_function
def _slots2_rows_constraints():
    """:func:`_slots2_rows_asm`'s constraints, one entry an element: the
    entries and a scratch of 4 a thread, 7 SGPRs, then the picks and rows."""
    return ",".join(["=&v"] * 8 + ["=&s"] * 7 + ["v"] * 8 + ["~{vcc}", "~{scc}"])


@gluon.jit
def _ballot_slots2(ids, RP: gl.constexpr):
    """:func:`slots` of ``ids`` two a lane (128 picks, 16 rows), the rows
    split over the waves: each wave builds its two rows' table entries
    (:func:`_slots2_rows_asm`), a pick's ``seen`` (an earlier row picked its
    expert) is a reduction of the entries across the waves, and the whole
    table goes to every wave in LDS. Every wave then holds the same owner,
    slot and count from wave ballots and the table. Built in every
    wave, the 16 rows would take ~900 VALU instructions a wave.
    Returns the same ``(ids, owner, slot, count, table)``."""
    NP: gl.constexpr = ids.shape[0]
    nw: gl.constexpr = gl.num_warps()
    gl.static_assert(RP == 2 * nw, "two rows a wave")
    # asm: by hand, the compares and ballots take fewer instructions than
    # with the _wave.ll intrinsics.
    # [pick, row]: two picks a lane, two rows a wave.
    la: gl.constexpr = gl.BlockedLayout([2, 2], [64, 1], [1, nw], [1, 0])
    mine = gl.convert_layout(ids, gl.SliceLayout(1, la))[:, None]
    r = gl.arange(0, RP, gl.SliceLayout(0, la))[None, :]
    mine, r = gl.broadcast(mine, r)
    out = gl.inline_asm(
        _slots2_rows_asm,
        _slots2_rows_constraints(),
        [mine, r],
        [mine.type, mine.type] + [gl.int32] * 7,
        is_pure=True,
    )
    entries = out[0]
    p = gl.arange(0, NP, gl.SliceLayout(1, la))[:, None]
    seen = gl.max(gl.where((entries >= 0) & (r < p // TOP_K), 1, 0), axis=1)
    owner = gl.convert_layout(1 - seen, ids.type.layout)
    # Separate the two register elements so each ballot counts one pick per lane.
    pair: gl.constexpr = gl.BlockedLayout([1, 2], [64, 1], [nw, 1], [0, 1])
    owners = gl.reshape(owner, [64, 2])
    owners = gl.convert_layout(owners, pair)
    even, odd = gl.split(owners)
    even_mask = wave_ballot_i64(even)
    odd_mask = wave_ballot_i64(odd)
    below = wave_mbcnt(even_mask) + wave_mbcnt(odd_mask)
    count = wave_popcount_i64(even_mask) + wave_popcount_i64(odd_mask)
    slot = gl.reshape(gl.join(below, below + even), [NP])
    slot = gl.convert_layout(slot, ids.type.layout)
    # Every lane has the same count; select one instead of reducing the wave.
    count = gl.sum(gl.gather(count, gl.full([1], 0, gl.int32, count.type.layout), 0), 0)
    tt: gl.constexpr = gl.BlockedLayout([2, RP], [64, 1], [nw, 1], [1, 0])
    table = gl.convert_layout(entries, tt)
    return ids, owner, slot, count, table


@gluon.constexpr_function
def _table_lanes_layout(RP, num_warps):
    """``[pick, row]`` with a pick's row picks in its lane's registers."""
    return gl.BlockedLayout([1, RP], [64, 1], [num_warps, 1], [1, 0])


@gluon.jit
def _ballot_slots(ids, RP: gl.constexpr):
    """:func:`slots` by :func:`_slots_asm`: ``ids`` a lane each (up to 64
    picks). Returns the same ``(ids, owner, slot, count, table)``."""
    NP: gl.constexpr = ids.shape[0]
    nw: gl.constexpr = gl.num_warps()
    tt: gl.constexpr = gl.distributed_type(
        gl.int32, [NP, RP], _table_lanes_layout(RP, nw)
    )
    # asm: by hand, the compares and ballots take fewer instructions than
    # with the _wave.ll intrinsics.
    out = gl.inline_asm(
        _slots_asm,
        (
            "=&v",
            "=&v",
            "=&s",
            "=&v",
            "=&v",
            "=&v",
            "=&v",
            "=&v",
            "=&v",
            "=&s",
            "v,~{vcc},~{scc}",
        ),
        [ids],
        [
            ids.type,
            ids.type,
            gl.int32,
            tt,
            ids.type,
            ids.type,
            ids.type,
            ids.type,
            ids.type,
            gl.int32,
        ],
        is_pure=True,
    )
    return ids, out[0], out[1], out[2], out[3]


@gluon.jit
def _slot_expert(union, i):
    """The expert of slot ``i``, or -1 past the last slot."""
    ids, owner, slot, _, _ = union
    mine = gl.convert_layout(ids, owner.type.layout)
    return gl.max(gl.where((owner > 0) & (slot == i), mine, -1), axis=0)


@gluon.jit
def _row_picks(union, expert, R: gl.constexpr):
    """The pick of each row that chose ``expert``, or -1: a tuple of ``R``
    scalars. One reduction across the waves gives every thread the whole
    row of the table (:func:`slots`) of the expert's owner pick; the scalars
    then come from the thread's own registers."""
    ids, owner, _, _, table = union
    tl: gl.constexpr = table.type.layout
    mine = gl.convert_layout(ids, gl.SliceLayout(1, tl))
    own = gl.convert_layout(owner, gl.SliceLayout(1, tl))
    hit = ((own > 0) & (mine == expert))[:, None]
    row = gl.max(gl.where(hit, table, -1), axis=0)  # [RP], in every thread
    r = gl.arange(0, table.shape[1], row.type.layout)
    picks = ()
    for i in gl.static_range(R):
        picks += (gl.max(gl.where(r == i, row, -1), axis=0),)
    return picks


@gluon.jit
def _load_b(
    w,
    ws,
    expert,
    stride_we,
    stride_se,
    stride_slin,
    stride_snb,
    n0,
    kt0,
    KT: gl.constexpr,
    N_PHYS: gl.constexpr,
    active=True,
):
    """Issue the B tiles of K tiles ``kt0 ..`` (128 values each) of columns
    ``n0 .. n0 + 127``: ``KT`` (weights, scales) pairs. Nothing loads when
    ``active`` is false or ``expert`` is -1."""
    layouts: gl.constexpr = _direct_mxfp4_mfma_layouts(
        M_DUP, BLOCK_N, 4, gl.num_warps()
    )
    dot_b: gl.constexpr = layouts[2]
    b_scale: gl.constexpr = layouts[4]
    bk = gl.arange(0, 64, gl.SliceLayout(1, dot_b))[:, None]
    bn = n0 + gl.arange(0, BLOCK_N, gl.SliceLayout(0, dot_b))[None, :]
    sn = n0 + gl.arange(0, BLOCK_N, gl.SliceLayout(1, b_scale))[:, None]
    sk = gl.arange(0, 4, gl.SliceLayout(0, b_scale))[None, :]
    on = active & (expert >= 0)
    expert = gl.maximum(expert, 0)
    w_off = expert.to(gl.int64) * stride_we
    s_off = expert.to(gl.int64) * stride_se
    tiles = ()
    for i in gl.static_range(KT):
        kt = kt0 + i
        b = gl.amd.cdna4.buffer_load(
            w,
            _gluon_dot_preshuffled_w_offset(w_off, kt * 64 + bk, bn, N_PHYS).to(
                gl.int32
            ),
            # (bk >= 0) is always true: it broadcasts the scalar ``on`` to
            # the tile's shape (the masks below use the same idiom).
            mask=(bk >= 0) & on,
            # sc0 nt on gfx950: a slot's weights stream once a launch, so
            # they must not evict what the launch reuses from L2 and the
            # memory-side cache (its code, the KV, the warmed next-layer
            # QKVG). The scales stay default: small, and read again.
            cache=".cg",
        )
        s = gl.amd.cdna4.buffer_load(
            ws,
            _cdna4_swizzled_mxfp4_scale_offset(
                s_off, sn, kt * 4 + sk, stride_slin, stride_snb
            ).to(gl.int32),
            mask=(sk >= 0) & on,
        )
        tiles += ((b, s),)
    return tiles


@gluon.jit
def _mma_rows(packed_smem, scale_smem, tiles):
    """The MXFP4 product of the LDS A rows and the B ``tiles``: FP32 ``[16,
    128]``, one row per A row."""
    layouts: gl.constexpr = _direct_mxfp4_mfma_layouts(
        M_DUP, BLOCK_N, 4, gl.num_warps()
    )
    mfma: gl.constexpr = layouts[0]
    acc = gl.zeros([M_DUP, BLOCK_N], gl.float32, mfma)
    for i in gl.static_range(len(tiles)):
        a = packed_smem.slice(i * 64, 64, 1).load(layouts[1])
        a_s = scale_smem.slice(i * 4, 4, 1).load(layouts[3])
        acc = _direct_mxfp4_mfma(acc, a, tiles[i][0], a_s, tiles[i][1])
    return acc


@gluon.constexpr_function
def _cg_layout(num_warps):
    """``[rows, K quarter, pair]`` of polled pooled rows."""
    return gl.BlockedLayout([1, 1, 1], [1, 1, 64], [1, 1, num_warps], [2, 1, 0])


@gluon.jit
def _xq_quantize(got, ks, gain_lo, gain_hi):
    """K quarter ``ks`` of the polled pooled rows ``got`` (``[G, KSPLIT,
    K_SLICE / 2]`` BF16 pairs, :func:`_cg_layout`) with each row's
    pre-expert RMS norm, quantized to MXFP4 as the regular path does (BF16
    input, groups of 32 along K). Returns the words: E2M1 ``[G, K_SLICE /
    8]`` and E8M0 scales ``[G, K_SLICE / 128]``.

    Only the G rows quantize: groups are per row, so the bytes are the ones
    of a 16-row MFMA tile would quantize mostly padding rows."""
    G: gl.constexpr = got.shape[0]
    nw: gl.constexpr = gl.num_warps()
    cl: gl.constexpr = _cg_layout(nw)
    q = gl.arange(0, KSPLIT, gl.SliceLayout(0, gl.SliceLayout(2, cl)))[None, :, None]
    lo = mx.bf16_half(got, 0)
    hi = mx.bf16_half(got, 1)
    inv = gl.rsqrt(
        gl.sum(gl.sum(lo * lo + hi * hi, axis=1), axis=1) / POOLED + PRE_EPS
    )[:, None]
    mine = q == ks
    lo = gl.sum(gl.where(mine, lo, 0.0), axis=1) * inv * gain_lo[None, :]
    hi = gl.sum(gl.where(mine, hi, 0.0), axis=1) * inv * gain_hi[None, :]
    x = gl.reshape(gl.join(lo, hi), [G, K_SLICE]).to(gl.bfloat16)
    dl: gl.constexpr = scaled_downcast_layout(G, K_SLICE, nw)
    packed, scales = _mxfp4_quantize_tile_in_layout(gl.convert_layout(x, dl))
    # A word takes 4 consecutive bytes in a thread; the scales come one a
    # thread.
    sl: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [nw, 1], [1, 0])
    return _words(packed), _words(gl.convert_layout(scales, sl))


@gluon.constexpr_function
def _words_layout(cols, num_warps):
    """``[row, word]`` of an xq quarter's words (a word a thread)."""
    lanes = min(64, cols)
    return gl.BlockedLayout([1, 1], [64 // lanes, lanes], [num_warps, 1], [1, 0])


@gluon.jit
def xq_stage(mb, task, pre_gain, STAGE: gl.constexpr):
    """xq task ``task``: K quarter ``task % KSPLIT`` of the rows of group
    ``task // KSPLIT`` (``G = xq_rows(R)`` rows), normed and quantized
    by :func:`_xq_quantize`, published as plain words in each row's copy of
    xq under one flag (xqf)."""
    mb.stamp(STAGE, trace.START)
    R: gl.constexpr = schedule.rows(mb.plan)
    G: gl.constexpr = xq_rows(R)
    ks = task % KSPLIT
    grp = task // KSPLIT
    nw: gl.constexpr = gl.num_warps()
    cl: gl.constexpr = _cg_layout(nw)
    gr = gl.arange(0, G, gl.SliceLayout(1, gl.SliceLayout(2, cl)))[:, None, None]
    q = gl.arange(0, KSPLIT, gl.SliceLayout(0, gl.SliceLayout(2, cl)))[None, :, None]
    j = gl.arange(0, K_SLICE // 2, gl.SliceLayout(0, gl.SliceLayout(1, cl)))
    c = ks * K_SLICE + 2 * j
    gain_lo = gl.load(pre_gain + c)
    gain_hi = gl.load(pre_gain + c + 1)
    pair = q * (K_SLICE // 2) + j[None, None, :]
    (got,) = mb.poll_many(("cg",), (mb.row_offset(grp * G + gr) + pair,))
    mb.stamp(STAGE, trace.READY)
    words, scales = _xq_quantize(got, ks, gain_lo, gain_hi)
    mb.stamp(STAGE, trace.COMPUTED)
    wl: gl.constexpr = words.type.layout
    pr = grp * G + gl.arange(0, G, gl.SliceLayout(1, wl))[:, None]
    pw = gl.arange(0, K_SLICE // 8, gl.SliceLayout(0, wl))[None, :]
    mb.store_words(
        "xq",
        2 * mb.row_offset(pr) + ks * XQ_QUARTER_WORDS + pw,
        words,
        mask=(pr < R) & (pw >= 0),
    )
    sl: gl.constexpr = scales.type.layout
    sr = grp * G + gl.arange(0, G, gl.SliceLayout(1, sl))[:, None]
    sw = gl.arange(0, K_SLICE // 128, gl.SliceLayout(0, sl))[None, :]
    mb.store_words(
        "xq",
        2 * mb.row_offset(sr) + ks * XQ_QUARTER_WORDS + K_SLICE // 8 + sw,
        scales,
        mask=(sr < R) & (sw >= 0),
    )
    mb.raise_flag("xqf", ks * XQ_GROUPS + grp)
    mb.stamp(STAGE, trace.END)


@gluon.jit
def ex1_poll_routes(mb, task, task2, active, STAGE: gl.constexpr):
    """The routes of ex1 task ``task`` (the pooled rows' quantized words come
    from xq, :func:`ex1_xq`): the slot table's flag with the experts of the
    first slots of ex1 task ``task`` and ex2 task ``task2``, then the table's
    words. Returns ``(ids, e1, e2)``; the weights can stream from here."""
    if active:
        mb.stamp(STAGE, trace.START)
    R: gl.constexpr = schedule.rows(mb.plan)
    RP: gl.constexpr = triton.next_power_of_2(R)
    nw: gl.constexpr = gl.num_warps()
    NP: gl.constexpr = RP * TOP_K
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [nw], [0])
    pk = gl.arange(0, NP, lanes)
    tl: gl.constexpr = _table_layout(NP, RP, nw)
    tp = gl.arange(0, NP, gl.SliceLayout(1, tl))[:, None]
    tr = gl.arange(0, RP, gl.SliceLayout(0, tl))[None, :]
    i = gl.arange(0, 2, lanes)
    first = gl.where(
        i == 0,
        task // (NB1 * KSPLIT),
        task2 // NB2,
    )
    # One tagged poll for the routes and the table's flag: two polls in a
    # row would add a round trip on ex1's critical path.
    firsts, _ = mb.poll_many(("route", "sltf"), (first, i * 0))
    # The table's flag is in: its words load after the poll returned.
    gl.barrier()
    ids = (mb.load_words("slt", pk), mb.load_words("slt", SLT_PICKS + tp * RP + tr))
    e1 = gl.max(gl.where(i == 0, firsts, -1), axis=0)
    e2 = gl.max(gl.where(i == 1, firsts, -1), axis=0)
    return ids, e1, e2


@gluon.jit
def ex1_xq(mb, task, active, STAGE: gl.constexpr):
    """ex1 task ``task``'s A operand from xq: poll its K quarter's flags (one
    a row group), then load the quarter's words of every row into LDS, one
    MFMA row each (rows past the launch's rows repeat its last row, rows
    past ``RP`` are zero). Returns the LDS (values, scales)."""
    R: gl.constexpr = schedule.rows(mb.plan)
    RP: gl.constexpr = triton.next_power_of_2(R)
    NG: gl.constexpr = RP // xq_rows(R)
    ks = task % KSPLIT
    nw: gl.constexpr = gl.num_warps()
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [nw], [0])
    f = gl.arange(0, XQ_GROUPS, lanes)
    mb.poll_flags("xqf", ks * XQ_GROUPS + gl.minimum(f, NG - 1))
    gl.barrier()
    if active:
        mb.stamp(STAGE, trace.READY)
    plain: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    PW: gl.constexpr = K_SLICE // 8
    SW: gl.constexpr = K_SLICE // 128
    pl: gl.constexpr = _words_layout(PW, nw)
    sl: gl.constexpr = _words_layout(SW, nw)
    pr = gl.arange(0, M_DUP, gl.SliceLayout(1, pl))[:, None]
    pw = gl.arange(0, PW, gl.SliceLayout(0, pl))[None, :]
    sr = gl.arange(0, M_DUP, gl.SliceLayout(1, sl))[:, None]
    sw = gl.arange(0, SW, gl.SliceLayout(0, sl))[None, :]
    base = ks * XQ_QUARTER_WORDS
    words = mb.load_words("xq", 2 * mb.row_offset(pr) + base + pw)
    scales = mb.load_words("xq", 2 * mb.row_offset(sr) + base + PW + sw)
    p_smem = gl.allocate_shared_memory(gl.uint8, [M_DUP, K_SLICE // 2], plain)
    s_smem = gl.allocate_shared_memory(gl.uint8, [M_DUP, K_SLICE // 32], plain)
    p_smem.reinterpret(gl.int32, [M_DUP, PW], plain).store(gl.where(pr < RP, words, 0))
    s_smem.reinterpret(gl.int32, [M_DUP, SW], plain).store(gl.where(sr < RP, scales, 0))
    gl.barrier()
    return p_smem, s_smem


# An ex1 B tile in LDS (ex1_prefetch): the K-tile's 8 chunks of 1 KiB (16
# columns each), raw-copied, read back in the MFMA's dot_b layout through
# this byte permutation: offset = (n // 16) 1024 + (k // 16) 256 + (n % 16) 16
# + k % 16 for packed K row k and column n. Bit-identical to _load_b's
# register read of the same tile.
B_TILE_LDS = gl.constexpr(
    gl.SharedLinearLayout(
        offset_bases=[
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [16, 0],
            [32, 0],
            [0, 16],
            [0, 32],
            [0, 64],
        ]
    )
)
B_TILE_BYTES = gl.constexpr(8192)  # 64 packed K x 128 columns


@gluon.jit
def ex1_prefetch(PLAN: gl.constexpr, task, active, table, union, w13, stride_we):
    """Past one row: copy the w13 B tiles (not the scales) of the lane's second
    slot for ex1 task ``task`` into LDS: 8 buffers, ``uint8 [8192]`` a K-tile,
    while the CTA waits on the routing chain for its A operand. Only the first
    slot's tiles are in flight then (in registers: a second there spills past 8
    rows). The slot's expert comes from ``table`` (:func:`cta_slots`) past 8
    rows, else from ``union``. With one row a lane has one slot at most (8
    picks over 8 lanes): nothing loads (a placeholder).

    The copies count in ``vmcnt`` in order, so :func:`ex1_xq`'s poll waits for
    them: by its return they have landed."""
    R: gl.constexpr = schedule.rows(PLAN)
    if R > 1:
        nw: gl.constexpr = gl.num_warps()
        KT: gl.constexpr = K_SLICE // 128
        blk = (task // KSPLIT) % NB1
        ks = task % KSPLIT
        if R > 8:
            expert = lane_slot_expert(table[0], 1)
        else:
            lane = task // (NB1 * KSPLIT)
            expert = _slot_expert(union, lane + LANES)
        on = active & (expert >= 0)
        w_off = gl.maximum(expert, 0).to(gl.int64) * stride_we
        raw: gl.constexpr = gl.BlockedLayout([16], [64], [nw], [0])
        i = gl.arange(0, B_TILE_BYTES, raw)
        # The tile's chunks (16 columns each) sit 2 KiB apart.
        chunk = (i // 1024) * 2048 + i % 1024
        plain: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
        out = ()
        for t in gl.static_range(KT):
            smem = gl.allocate_shared_memory(gl.uint8, [B_TILE_BYTES], plain)
            base = _gluon_dot_preshuffled_w_offset(
                w_off, (ks * KT + t) * 64, blk * BLOCK_N, W13_N
            )
            # Each lane's 16 bytes are contiguous and 16-byte aligned: the
            # tile starts at a K row multiple of 64 and a column multiple of
            # 128 (the chunk offsets are multiples of 1 KiB).
            offs = gl.max_contiguous(
                gl.multiple_of((base + chunk).to(gl.int32), 16), 16
            )
            cdna4_async_copy.buffer_load_to_shared(
                smem,
                w13,
                offs,
                mask=on & (i >= 0),
                cache_modifier=".cg",
            )
            out += (smem,)
        cdna4_async_copy.commit_group()
    else:
        out = 0
    return out


@gluon.jit
def _lds_b_tiles(pre, scales):
    """The B tiles of a slot from LDS (``pre``, :func:`ex1_prefetch`) with its
    scales from registers (``scales``, :func:`_load_scales`), as
    :func:`_load_b` returns them."""
    layouts: gl.constexpr = _direct_mxfp4_mfma_layouts(
        M_DUP, BLOCK_N, 4, gl.num_warps()
    )
    dot_b: gl.constexpr = layouts[2]
    tiles = ()
    for t in gl.static_range(len(scales)):
        b = pre[t].reinterpret(gl.uint8, [64, BLOCK_N], B_TILE_LDS).load(dot_b)
        tiles += ((b, scales[t]),)
    return tiles


@gluon.jit
def _load_scales(
    ws, expert, stride_se, stride_slin, stride_snb, n0, kt0, KT: gl.constexpr
):
    """:func:`_load_b`'s scale loads alone: ``KT`` tiles' scales of columns
    ``n0 ..`` from K tile ``kt0``; nothing loads when ``expert`` is -1."""
    layouts: gl.constexpr = _direct_mxfp4_mfma_layouts(
        M_DUP, BLOCK_N, 4, gl.num_warps()
    )
    b_scale: gl.constexpr = layouts[4]
    sn = n0 + gl.arange(0, BLOCK_N, gl.SliceLayout(1, b_scale))[:, None]
    sk = gl.arange(0, 4, gl.SliceLayout(0, b_scale))[None, :]
    on = expert >= 0
    s_off = gl.maximum(expert, 0).to(gl.int64) * stride_se
    scales = ()
    for i in gl.static_range(KT):
        scales += (
            gl.amd.cdna4.buffer_load(
                ws,
                _cdna4_swizzled_mxfp4_scale_offset(
                    s_off, sn, (kt0 + i) * 4 + sk, stride_slin, stride_snb
                ).to(gl.int32),
                mask=(sk >= 0) & on,
            ),
        )
    return scales


@gluon.jit
def ex1_issue(
    task, active, expert, w13, w13s, stride_we, stride_se, stride_slin, stride_snb
):
    """Issue ex1 task ``task``'s w13 loads (column block, K quarter) of
    expert ``expert``; nothing loads when ``active`` is false or ``expert``
    is -1.

    Args:
        w13, w13s: The rank's preprocessed w13 bytes ``[256, 2048, 1024]`` and
            scales (as ``decode_stage1`` takes them), with their strides.
    """
    blk = (task // KSPLIT) % NB1
    ks = task % KSPLIT
    return _load_b(
        w13,
        w13s,
        expert,
        stride_we,
        stride_se,
        stride_slin,
        stride_snb,
        blk * BLOCK_N,
        ks * (K_SLICE // 128),
        K_SLICE // 128,
        W13_N,
        active,
    )


@gluon.constexpr_function
def _p1_layout(num_warps):
    """``[M_DUP, BLOCK_N]`` with a warp's 16 columns of the MFMA tile, a
    row's 16 columns in 16 lanes (4 rows a store)."""
    return gl.BlockedLayout([1, 1], [4, 16], [1, num_warps], [1, 0])


@gluon.jit
def _put_p1(mb, acc, k, blk, ks):
    """Put an ex1 slot's FP32 partials ``acc [M_DUP, BLOCK_N]`` (MFMA layout)
    into ``p1``: MFMA row ``m`` goes to row ``m``'s copy under pick ``k[m]``
    (``k`` in ``SliceLayout(1, _p1_layout)``; -1 skips the row).

    In the MFMA layout a lane holds 4 adjacent columns of a row, so each
    8-byte store writes 4 lanes 32 bytes apart and fills a quarter of every
    segment it touches. :func:`_p1_layout` puts a row's 16 columns of a
    warp in 16 lanes: one store writes 128 contiguous bytes a row. Each
    warp keeps its own columns, but the conversion moves both of a lane's
    column bits from registers to lanes, which Triton lowers through LDS
    with a workgroup barrier rather than lane shuffles."""
    cl: gl.constexpr = _p1_layout(gl.num_warps())
    v = gl.convert_layout(acc, cl)
    m = gl.arange(0, M_DUP, gl.SliceLayout(1, cl))[:, None]
    n = gl.arange(0, BLOCK_N, gl.SliceLayout(0, cl))[None, :]
    at = ((k[:, None] * NB1 + blk) * KSPLIT + ks) * BLOCK_N + n
    mb.put(
        "p1",
        mb.row_offset(m) + at,
        v.to(gl.int32, bitcast=True),
        mask=(k[:, None] >= 0) & (n >= 0),
    )


@gluon.jit
def _ex1_slot(mb, task, a, tiles, union, expert):
    """Multiply the A rows by one slot's w13 ``tiles`` and publish the FP32
    partial sums of the rows that picked ``expert``."""
    R: gl.constexpr = schedule.rows(mb.plan)
    blk = (task // KSPLIT) % NB1
    ks = task % KSPLIT
    p_smem, s_smem = a
    acc = _mma_rows(p_smem, s_smem, tiles)
    picks = _row_picks(union, expert, R)
    m = gl.arange(0, M_DUP, gl.SliceLayout(1, _p1_layout(gl.num_warps())))
    k = gl.full([M_DUP], -1, gl.int32, m.type.layout)
    for r in gl.static_range(R):
        k = gl.where(m == r, picks[r], k)
    _put_p1(mb, acc, k, blk, ks)


@gluon.jit
def lane_slots(union, lane):
    """Past 8 rows: the expert and the row picks of each of lane ``lane``'s
    slots, from the slot table (``union``), in LDS: ``(experts,
    picks)``, ``int32 [16, 16]`` each, row ``k`` for slot ``lane + 8 k`` (at
    most 8 picks a row make at most 16 slots a lane). A picks row holds each
    row's pick of the slot's expert, or -1 (as :func:`_ex2_gather_lane`); an
    experts row holds the slot's expert in every column, or -1 past the
    lane's last slot.

    Past 8 rows ex1, ex2 and (placed from CTA 0 in EXR_PARTS parts) exr task
    ``t`` all run on CTA ``t`` with lane ``t // 32``, so the CTA builds one
    table, after :func:`union_of`, for all three. Per slot they then read it
    with one LDS load: the lookups it replaces (:func:`_slot_expert`,
    :func:`_row_picks`) are reductions over the 128-pick table a slot."""
    nw: gl.constexpr = gl.num_warps()
    ids, owner, slot, count, table = union
    NP: gl.constexpr = table.shape[0]
    RP: gl.constexpr = table.shape[1]
    gl.static_assert(RP == M_DUP, "past 8 rows the table has 16 row columns")
    # The [slot, pick] layout below: two slots a wave, the picks over the 64
    # lanes.
    gl.static_assert(M_DUP == 2 * nw, "two slots a wave")
    gl.static_assert(NP % 64 == 0, "the picks fill whole lanes")
    # Each slot has exactly one owner pick, however many rows share its
    # expert: a row picks an expert at most once, and the owner is its pick
    # in the first row that chose it (later rows' picks of it are "seen",
    # see _slots_asm); slots number the owners. So find the owner with one
    # reduction over the picks a slot ([slot, pick]: a slot a wave, two
    # picks a lane) and gather its table row (every row's pick of the
    # expert) from LDS: the same values as a max over the slot's owner
    # picks. Reducing the whole [slot, pick, row] table instead took a DPP
    # max step per row column on every CTA.
    l2: gl.constexpr = gl.BlockedLayout([2, NP // 64], [1, 64], [nw, 1], [1, 0])
    j = gl.arange(0, M_DUP, gl.SliceLayout(1, l2))[:, None]
    p = gl.arange(0, NP, gl.SliceLayout(0, l2))[None, :]
    pl: gl.constexpr = gl.SliceLayout(0, l2)
    hit = (gl.convert_layout(owner, pl)[None, :] > 0) & (
        gl.convert_layout(slot, pl)[None, :] == lane + j * LANES
    )
    owner_pick = gl.max(gl.where(hit, p, -1), axis=1)  # [16]
    expert = gl.max(gl.where(hit, gl.convert_layout(ids, pl)[None, :], -1), axis=1)
    plain: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    table_smem = gl.allocate_shared_memory(gl.int32, [NP, RP], plain)
    table_smem.store(table)
    gl.barrier()
    rl: gl.constexpr = gl.BlockedLayout([1, 1], [4, 16], [nw, 1], [1, 0])
    # A slot without an owner (past the lane's last slot) gathers pick 0's
    # row; ``has`` masks it to -1 below.
    at = gl.convert_layout(gl.maximum(owner_pick, 0), gl.SliceLayout(1, rl))[:, None]
    at = at + gl.zeros([M_DUP, RP], gl.int32, rl)
    got = table_smem.gather(at, 0)  # [16, RP]: row k is slot lane + 8 k's
    has = gl.convert_layout(owner_pick, gl.SliceLayout(1, rl))[:, None] >= 0
    picks = gl.where(has, got, -1)
    experts = gl.convert_layout(expert, gl.SliceLayout(1, rl))[:, None] + gl.zeros(
        [M_DUP, RP], gl.int32, rl
    )
    picks_smem = gl.allocate_shared_memory(gl.int32, [M_DUP, M_DUP], plain)
    experts_smem = gl.allocate_shared_memory(gl.int32, [M_DUP, M_DUP], plain)
    picks_smem.store(picks)
    experts_smem.store(experts)
    gl.barrier()
    # The same row-major words, as rows: index(k) is slot k's row.
    rows: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    return (
        experts_smem.reinterpret(gl.int32, [M_DUP, M_DUP], rows),
        picks_smem.reinterpret(gl.int32, [M_DUP, M_DUP], rows),
    )


@gluon.jit
def cta_slots(PLAN: gl.constexpr, union):
    """Past 8 rows, this CTA's :func:`lane_slots`: ex1 and ex2 task ``t``
    (placed from CTA 0) run on CTA ``t``, lane ``t // 32``, and so does exr
    task ``t`` when it is placed from CTA 0 in EXR_PARTS parts. Up to 8 rows
    the stages look their slots up in ``union`` themselves (a placeholder).
    It ends in a barrier, so every CTA calls it, outside any task branch."""
    if schedule.rows(PLAN) > 8:
        gl.static_assert(
            schedule.stage_base(PLAN, "ex1") == 0
            and schedule.stage_base(PLAN, "ex2") == 0,
            "the slot table serves the lane of CTA t: t // 32",
        )
        # The table is the lane of ex1's task on this CTA; ex2 reads it for
        # its own task's lane, task // NB2: the two must agree.
        gl.static_assert(
            NB2 == NB1 * KSPLIT,
            "ex1 and ex2 tasks of a CTA must be of one lane",
        )
        table = lane_slots(union, gl.program_id(0) // (NB1 * KSPLIT))
    else:
        table = (0, 0)
    return table


@gluon.jit
def lane_slot_expert(experts_smem, k):
    """The expert of the lane's ``k``-th slot (:func:`lane_slots`), or -1
    (past its last slot, ``k`` 16 or more too): every thread loads the whole
    row, so the max is its own."""
    whole: gl.constexpr = gl.BlockedLayout([M_DUP], [64], [gl.num_warps()], [0])
    k = gl.to_tensor(k)
    row = experts_smem.index(gl.minimum(k, M_DUP - 1)).load(whole)
    return gl.where(k < M_DUP, gl.max(row, axis=0), -1)


@gluon.jit
def _slot_rows_lds(picks_row, RP: gl.constexpr, layout: gl.constexpr):
    """:func:`_slot_rows` of a slot's row picks from LDS (``picks_row``
    ``[16]``, :func:`lane_slots`): the same ``(row, pick, routed)``."""
    whole: gl.constexpr = gl.BlockedLayout([M_DUP], [64], [gl.num_warps()], [0])
    w = picks_row.load(whole)
    iw = gl.arange(0, M_DUP, whole)
    first = gl.min(gl.where(w >= 0, iw, M_DUP), axis=0)
    first_pick = gl.max(gl.where(iw == first, w, -1), axis=0)
    pick = picks_row.slice(0, RP).load(layout)
    i = gl.arange(0, RP, layout)
    routed = pick >= 0
    return gl.where(routed, i, first), gl.where(routed, pick, first_pick), routed


@gluon.jit
def _ex1_slot_picked(mb, task, a, tiles, picks_row):
    """:func:`_ex1_slot` past 8 rows, with the slot's row picks from LDS
    (``picks_row`` ``[16]``, kept by :func:`_ex1_lane_slots`): MFMA row ``r``
    is row ``r``, as there."""
    blk = (task // KSPLIT) % NB1
    ks = task % KSPLIT
    p_smem, s_smem = a
    acc = _mma_rows(p_smem, s_smem, tiles)
    k = picks_row.load(gl.SliceLayout(1, _p1_layout(gl.num_warps())))
    _put_p1(mb, acc, k, blk, ks)


@gluon.jit
def ex1_finish(
    mb,
    task,
    a,
    tiles,
    expert,
    union,
    table,
    pre,
    w13,
    w13s,
    stride_we,
    stride_se,
    stride_slin,
    stride_snb,
    STAGE: gl.constexpr,
):
    """Ex1 task ``task`` after :func:`ex1_xq` (``a``) and :func:`ex1_issue`
    (``tiles``, of the lane's first slot, ``expert``) and :func:`slots`
    (``union``): multiply and publish the FP32 partial sums, then the same
    for the lane's other slots. ``table``: past 8 rows, the CTA's
    :func:`lane_slots`; ``pre``: past one row, the second slot's B tiles in
    LDS (:func:`ex1_prefetch`)."""
    lane = task // (NB1 * KSPLIT)
    count = union[3]
    R: gl.constexpr = schedule.rows(mb.plan)
    if R > 8:
        # Past 8 rows each slot's expert and row picks come from LDS, and the
        # second slot's B tiles too: they landed during the A poll (vmcnt
        # counts in order), so the wait is free and covers no later load.
        experts_smem, picks_smem = table
        cdna4_async_copy.wait_group(0)
        gl.barrier()
        blk = (task // KSPLIT) % NB1
        ks = task % KSPLIT
        second = _load_scales(
            w13s,
            lane_slot_expert(experts_smem, 1),
            stride_se,
            stride_slin,
            stride_snb,
            blk * BLOCK_N,
            ks * (K_SLICE // 128),
            K_SLICE // 128,
        )
        following = lane_slot_expert(experts_smem, 2)
        more = ex1_issue(
            task,
            True,
            following,
            w13,
            w13s,
            stride_we,
            stride_se,
            stride_slin,
            stride_snb,
        )
        _ex1_slot_picked(mb, task, a, tiles, picks_smem.index(0))
        if lane + LANES < count:
            _ex1_slot_picked(
                mb, task, a, _lds_b_tiles(pre, second), picks_smem.index(1)
            )
        for i in range(lane + 2 * LANES, count, LANES):
            k = i // LANES
            tiles = more
            following = lane_slot_expert(experts_smem, k + 1)
            more = ex1_issue(
                task,
                True,
                following,
                w13,
                w13s,
                stride_we,
                stride_se,
                stride_slin,
                stride_snb,
            )
            _ex1_slot_picked(mb, task, a, tiles, picks_smem.index(k))
    elif R > 1:
        # Slot 1's B tiles landed in LDS during the A poll (ex1_prefetch), so
        # the wait is free; its scales load here, and slot 2's tiles go out
        # into registers before slot 0's product. Every CTA's waves see the
        # copies after the barrier.
        cdna4_async_copy.wait_group(0)
        gl.barrier()
        blk = (task // KSPLIT) % NB1
        ks = task % KSPLIT
        following = _slot_expert(union, lane + 2 * LANES)
        more = ex1_issue(
            task,
            True,
            following,
            w13,
            w13s,
            stride_we,
            stride_se,
            stride_slin,
            stride_snb,
        )
        _ex1_slot(mb, task, a, tiles, union, expert)
        if lane + LANES < count:
            # Slot 1's scales load after slot 0's product: not live across it.
            expert1 = _slot_expert(union, lane + LANES)
            second = _load_scales(
                w13s,
                expert1,
                stride_se,
                stride_slin,
                stride_snb,
                blk * BLOCK_N,
                ks * (K_SLICE // 128),
                K_SLICE // 128,
            )
            _ex1_slot(mb, task, a, _lds_b_tiles(pre, second), union, expert1)
        for i in range(lane + 2 * LANES, count, LANES):
            expert = following
            tiles = more
            following = _slot_expert(union, i + LANES)
            more = ex1_issue(
                task,
                True,
                following,
                w13,
                w13s,
                stride_we,
                stride_se,
                stride_slin,
                stride_snb,
            )
            _ex1_slot(mb, task, a, tiles, union, expert)
    else:
        # Each slot's loads go out before the previous slot's product. With
        # one row the loop never runs; it stays there too, since the backend
        # schedules the slot's code around it better than without it.
        following = _slot_expert(union, lane + LANES)
        more = ex1_issue(
            task,
            True,
            following,
            w13,
            w13s,
            stride_we,
            stride_se,
            stride_slin,
            stride_snb,
        )
        _ex1_slot(mb, task, a, tiles, union, expert)
        for i in range(lane + LANES, count, LANES):
            expert = following
            tiles = more
            following = _slot_expert(union, i + LANES)
            more = ex1_issue(
                task,
                True,
                following,
                w13,
                w13s,
                stride_we,
                stride_se,
                stride_slin,
                stride_snb,
            )
            _ex1_slot(mb, task, a, tiles, union, expert)
    mb.stamp(STAGE, trace.COMPUTED)
    mb.stamp(STAGE, trace.END)


@gluon.jit
def ex2_issue(
    mb,
    task,
    active,
    expert,
    w2,
    w2s,
    stride_we,
    stride_se,
    stride_slin,
    stride_snb,
    STAGE: gl.constexpr,
):
    """Issue ex2 task ``task``'s w2 loads (column block) of expert
    ``expert``; nothing loads (and nothing stamps) when ``active`` is false;
    nothing loads when ``expert`` is -1. Every CTA calls it (Gluon cannot
    carry the tiles out of a branch); :func:`ex2_finish` uses them.

    Args:
        w2, w2s: The rank's preprocessed w2 bytes ``[256, 256, 4096]`` and
            scales, with their strides.
    """
    if active:
        mb.stamp(STAGE, trace.START)
    return _ex2_loads(
        task, active, expert, w2, w2s, stride_we, stride_se, stride_slin, stride_snb
    )


@gluon.jit
def _ex2_loads(
    task, active, expert, w2, w2s, stride_we, stride_se, stride_slin, stride_snb
):
    """:func:`ex2_issue`'s loads, without the trace point."""
    return _load_b(
        w2,
        w2s,
        expert,
        stride_we,
        stride_se,
        stride_slin,
        stride_snb,
        (task % NB2) * BLOCK_N,
        0,
        INTER // 128,
        POOLED,
        active,
    )


@gluon.jit
def _slot_rows(mb, picks, RP: gl.constexpr, layout: gl.constexpr):
    """``(row, pick, routed)`` ``[RP]`` in ``layout``: each row that picked
    the slot's expert with its pick; the other rows (and the rows past the
    launch's) stand in for the first routed row, so a poll of them reads
    pairs that are published."""
    R: gl.constexpr = schedule.rows(mb.plan)
    i = gl.arange(0, RP, layout)
    pick = gl.full([RP], -1, gl.int32, layout)
    first = gl.to_tensor(-1)
    first_pick = gl.to_tensor(-1)
    for r in gl.static_range(R - 1, -1, -1):
        pick = gl.where(i == r, picks[r], pick)
        first = gl.where(picks[r] >= 0, r, first)
        first_pick = gl.where(picks[r] >= 0, picks[r], first_pick)
    routed = pick >= 0
    return gl.where(routed, i, first), gl.where(routed, pick, first_pick), routed


@gluon.jit
def _words(x):
    """Bytes ``[M, 4 W]`` as little-endian int32 words ``[M, W]``."""
    M: gl.constexpr = x.shape[0]
    W: gl.constexpr = x.shape[1] // 4
    lo, hi = gl.split(gl.reshape(x.to(gl.int32), [M, W, 2, 2]))
    half = lo | (hi << 8)
    lo, hi = gl.split(half)
    return lo | (hi << 16)


@gluon.jit
def exr_stage(mb, task, union, table, STAGE: gl.constexpr):
    """Exr task ``task`` (slot lane, ex1 column block): for each slot of the
    lane, sum the 4 K quarters of the ex1 partials of every row that picked
    the slot's expert, apply SwiGLU (``silu(gate) up``) and publish the
    block's 64 intermediate values a row. The 32 ex2 tasks of the lane then
    read 512 values a row, not 4 partials of 1024 columns each, requantized
    to MXFP4.
    ``table``: past 8 rows, the CTA's :func:`lane_slots`, which serves the
    task's slots when exr runs from CTA 0 in EXR_PARTS parts (its lane is
    then the CTA's)."""
    mb.stamp(STAGE, trace.START)
    R: gl.constexpr = schedule.rows(mb.plan)
    RP: gl.constexpr = triton.next_power_of_2(R)
    nw: gl.constexpr = gl.num_warps()
    PARTS: gl.constexpr = schedule.stage_tasks(mb.plan, STAGE) // EXR_LANE_TASKS
    LDS: gl.constexpr = (
        R > 8 and schedule.stage_base(mb.plan, STAGE) == 0 and PARTS == EXR_PARTS
    )
    if LDS:
        # Then exr task t, lane t // (NB1 * EXR_PARTS), is the lane of the
        # CTA's table (cta_slots: CTA // (NB1 * KSPLIT)).
        gl.static_assert(
            EXR_PARTS == KSPLIT, "exr's lane must be the CTA's slot-table lane"
        )
    lane = task // (NB1 * PARTS)
    blk = task // PARTS % NB1
    first = lane + LANES * (task % PARTS)
    count = union[3]
    # [row, K quarter, column]: the K quarters in a lane's registers.
    pl: gl.constexpr = gl.BlockedLayout(
        [1, KSPLIT, 2], [1, 1, 64], [nw, 1, 1], [2, 1, 0]
    )
    rl: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, pl))
    ks = gl.arange(0, KSPLIT, gl.SliceLayout(0, gl.SliceLayout(2, pl)))[None, :, None]
    n = gl.arange(0, BLOCK_N, gl.SliceLayout(0, gl.SliceLayout(1, pl)))[None, None, :]
    for i in range(first, count, LANES * PARTS):
        if LDS:
            picks_row = table[1].index(i // LANES)
            row, pick, _ = _slot_rows_lds(picks_row, RP, rl)
        else:
            expert = _slot_expert(union, i)
            picks = _row_picks(union, expert, R)
            row, pick, _ = _slot_rows(mb, picks, RP, rl)
        at = ((pick[:, None, None] * NB1 + blk) * KSPLIT + ks) * BLOCK_N + n
        (got,) = mb.poll_many(("p1",), (mb.row_offset(row[:, None, None]) + at,))
        if i == first:
            mb.stamp(STAGE, trace.READY)
        parts = got.to(gl.float32, bitcast=True)
        # The regular path's order: K quarter 0, then 1, ...
        acc = gl.zeros([RP, BLOCK_N], gl.float32, gl.SliceLayout(1, pl))
        for k in gl.static_range(KSPLIT):
            acc += gl.sum(gl.where(ks == k, parts, 0.0), axis=1)
        gate, up = gl.split(gl.reshape(acc, [RP, BLOCK_N // 2, 2]))
        inter = gate / (1.0 + gl.exp(-gate)) * up
        # The regular path's requantization: groups of 32 values, and the
        # block's 64 values are two whole groups.
        packed, scales = _mxfp4_quantize_tile(inter)
        words = _words(packed)
        wl: gl.constexpr = words.type.layout
        if LDS:
            row, pick, routed = _slot_rows_lds(picks_row, RP, gl.SliceLayout(1, wl))
        else:
            row, pick, routed = _slot_rows(mb, picks, RP, gl.SliceLayout(1, wl))
        w = gl.arange(0, BLOCK_N // 16, gl.SliceLayout(0, wl))[None, :]
        mb.put(
            "h1",
            mb.row_offset(row[:, None])
            + pick[:, None] * H1_WORDS
            + blk * (BLOCK_N // 16)
            + w,
            words,
            mask=routed[:, None] & (w >= 0),
        )
        # A row's 2 scale bytes in one thread, for the split.
        pl2: gl.constexpr = gl.BlockedLayout([1, 2], [64, 1], [nw, 1], [1, 0])
        lo, hi = gl.split(gl.convert_layout(scales.to(gl.int32), pl2))
        sl: gl.constexpr = lo.type.layout
        if LDS:
            row, pick, routed = _slot_rows_lds(picks_row, RP, sl)
        else:
            row, pick, routed = _slot_rows(mb, picks, RP, sl)
        mb.put(
            "h1",
            mb.row_offset(row) + pick * H1_WORDS + INTER // 8 + blk,
            lo | (hi << 8),
            mask=routed,
        )
    mb.stamp(STAGE, trace.COMPUTED)
    mb.stamp(STAGE, trace.END)


@gluon.jit
def _ex2_gather_lane(
    mb,
    union,
    slot_picks,
    lane,
    S: gl.constexpr,
    KEEP_PICKS: gl.constexpr,
    K0: gl.constexpr,
):
    """The polls of ex2 lane ``lane``'s first ``S`` slots in one round trip:
    each slot's MXFP4 intermediates (:func:`exr_stage`) of every row that
    picked its expert, MFMA row ``r`` row ``r`` (the other rows zero), into
    LDS. Returns the LDS (values ``[S 16, 256]`` and scales ``[S 16, 16]``
    bytes) for :func:`_ex2_compute`, a slot's 16 rows a slice.

    The picks of all ``S`` slots come from one reduction over the slot table
    (``union``), or past 8 rows from the CTA's table (``slot_picks``: the
    picks of :func:`lane_slots`, the same picks; up to 8 rows a placeholder).
    A slot past the lane's last, and a row that did not pick the slot's
    expert, poll words that are published (the lane's first slot, the slot's
    first routed row). With ``KEEP_PICKS`` (past one row) it also returns
    those picks in LDS, ``int32 [S, 16]``: row ``k`` is slot ``lane + 8 k``,
    column ``r`` row ``r``'s pick of its expert, or -1 (a row that did not
    pick it, a row past the launch's, or a slot past the lane's last).
    ``K0``: the first slot is the lane's ``K0``-th (past 8 rows,
    :func:`_ex2_tail_batch` gathers slots 8 .. 15)."""
    R: gl.constexpr = schedule.rows(mb.plan)
    RP: gl.constexpr = triton.next_power_of_2(R)
    nw: gl.constexpr = gl.num_warps()
    ids, owner, slot, count, table = union
    # [slot, pick, row]: a slot a wave, a pick a lane, the rows in registers.
    l3: gl.constexpr = gl.BlockedLayout([1, 1, RP], [1, 64, 1], [nw, 1, 1], [2, 1, 0])
    tl: gl.constexpr = gl.SliceLayout(1, l3)  # [slot, row]
    if R > 8:
        # The table's row k is slot lane + 8 k; past the lane's last slot it
        # holds -1, where the reduction below takes the lane's first slot
        # (whose words are published), so those rows take row 0's picks.
        gl.static_assert(RP == M_DUP, "past 8 rows the table has 16 row columns")
        flat: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
        tab = slot_picks.reinterpret(gl.int32, [M_DUP, M_DUP], flat)
        js = gl.arange(0, S, gl.SliceLayout(1, tl))[:, None]
        picks = gl.where(
            lane + (js + K0) * LANES < count,
            tab.slice(K0, S, dim=0).load(tl),
            tab.slice(0, 1, dim=0).load(tl),
        )
    else:
        pl: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(2, l3))
        j = gl.arange(0, S, gl.SliceLayout(1, gl.SliceLayout(2, l3)))[:, None, None]
        if K0 > 0:
            want = lane + (j + K0) * LANES
        else:
            want = lane + j * LANES
        want = gl.where(want < count, want, lane)
        own = gl.convert_layout(owner, pl)[None, :, None]
        mine = gl.convert_layout(slot, pl)[None, :, None]
        t3 = gl.convert_layout(table, gl.SliceLayout(0, l3))[None, :, :]
        picks = gl.max(gl.where((own > 0) & (mine == want), t3, -1), axis=1)  # [S, RP]
    if KEEP_PICKS:
        ks = gl.arange(0, S, gl.SliceLayout(1, tl))[:, None]
        plain_p: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
        picks_smem = gl.allocate_shared_memory(gl.int32, [S, M_DUP], plain_p)
        if K0 > 0:
            valid = lane + (ks + K0) * LANES < count
        else:
            valid = lane + ks * LANES < count
        picks_smem.slice(0, RP, dim=1).store(gl.where(valid, picks, -1))
        # MFMA rows past the launch's rows: no pick. A block's width must be a
        # power of two, so the columns go in blocks [RP, 2 RP), [2 RP, 4 RP),
        # ... up to M_DUP (one block at 8 rows, none at 16).
        fill: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [nw, 1], [1, 0])
        for k in gl.static_range(4):  # at most log2(M_DUP) blocks
            # RP << k inline: a constexpr name cannot be reassigned per pass.
            if (RP << k) < M_DUP:
                picks_smem.slice(RP << k, RP << k, dim=1).store(
                    gl.full([S, RP << k], -1, gl.int32, fill)
                )
    r = gl.arange(0, RP, gl.SliceLayout(0, tl))[None, :]
    routed = picks >= 0
    first = gl.min(gl.where(routed, r, RP), axis=1)[:, None]
    first_pick = gl.max(gl.where(r == first, picks, -1), axis=1)[:, None]
    row = gl.where(routed, r, first)
    pick = gl.where(routed, picks, first_pick)
    wl: gl.constexpr = gl.BlockedLayout([1, 1, 1], [1, 1, 64], [nw, 1, 1], [2, 1, 0])
    sl: gl.constexpr = gl.BlockedLayout([1, 1, 2], [1, 16, 4], [nw, 1, 1], [2, 1, 0])
    v = gl.arange(0, INTER // 8, gl.SliceLayout(0, gl.SliceLayout(1, wl)))
    b = gl.arange(0, NB1, gl.SliceLayout(0, gl.SliceLayout(1, sl)))
    row_w = gl.convert_layout(row, gl.SliceLayout(2, wl))[:, :, None]
    pick_w = gl.convert_layout(pick, gl.SliceLayout(2, wl))[:, :, None]
    row_s = gl.convert_layout(row, gl.SliceLayout(2, sl))[:, :, None]
    pick_s = gl.convert_layout(pick, gl.SliceLayout(2, sl))[:, :, None]
    got, sc = mb.poll_many(
        ("h1", "h1"),
        (
            mb.row_offset(row_w) + pick_w * H1_WORDS + v[None, None, :],
            mb.row_offset(row_s) + pick_s * H1_WORDS + INTER // 8 + b[None, None, :],
        ),
    )
    got = gl.where(gl.convert_layout(routed, gl.SliceLayout(2, wl))[:, :, None], got, 0)
    sc = gl.where(gl.convert_layout(routed, gl.SliceLayout(2, sl))[:, :, None], sc, 0)
    lo, hi = gl.split(gl.reshape(sc, [S, RP, NB1 // 2, 2]))
    # Row-major (unswizzled): the [slot, row, word] buffers are read back as
    # [slot * 16 + row, byte], a slot's 16 MFMA rows a slice.
    plain3: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [2, 1, 0])
    plain: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    p_smem = gl.allocate_shared_memory(gl.uint8, [S, M_DUP, INTER // 2], plain3)
    s_smem = gl.allocate_shared_memory(gl.uint8, [S, M_DUP, INTER // 32], plain3)
    p_words = p_smem.reinterpret(gl.int32, [S, M_DUP, INTER // 8], plain3)
    s_words = s_smem.reinterpret(gl.int32, [S, M_DUP, NB1 // 2], plain3)
    p_words.slice(0, RP, dim=1).store(got)
    s_words.slice(0, RP, dim=1).store(lo | (hi << 16))
    for k in gl.static_range(4):  # M_DUP = 16 rows
        if (RP << k) < M_DUP:
            p_words.slice(RP << k, RP << k, dim=1).store(
                gl.zeros([S, RP << k, INTER // 8], gl.int32, wl)
            )
            s_words.slice(RP << k, RP << k, dim=1).store(
                gl.zeros([S, RP << k, NB1 // 2], gl.int32, sl)
            )
    gl.barrier()
    out = (
        p_smem.reinterpret(gl.uint8, [S * M_DUP, INTER // 2], plain),
        s_smem.reinterpret(gl.uint8, [S * M_DUP, INTER // 32], plain),
    )
    if KEEP_PICKS:
        out += (picks_smem,)
    return out


@gluon.jit
def _ex2_compute_picked(mb, task, loads, a, picks_smem, RANK: gl.constexpr):
    """:func:`_ex2_compute` past one row, with the slot's row picks from LDS
    (``picks_smem`` ``[1, 16]``, kept by :func:`_ex2_gather_lane`): MFMA row ``r`` is
    row ``r``, as there."""
    blk = task % NB2
    p_smem, s_smem = a
    owner = (blk * BLOCK_N) // OWNED_OUT
    acc = _mma_rows(p_smem, s_smem, loads)
    out = gl.reshape(acc.to(gl.bfloat16), [M_DUP, BLOCK_N // 2, 2])
    word = mx.pack_bf16_pairs(out)
    rl: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [gl.num_warps(), 1], [1, 0])
    col = (blk * BLOCK_N) % OWNED_OUT
    R: gl.constexpr = schedule.rows(mb.plan)
    if R >= 8:
        # In the accumulator's layout adjacent lanes hold pairs 16 bytes
        # apart, so each 8-byte remote store fills half of every segment it
        # touches. Here lane i holds pair i of a row: one store writes a
        # row's 64 pairs as 512 contiguous bytes. The same values go to the
        # same addresses. Below 8 rows the put keeps the accumulator's layout.
        cl: gl.constexpr = gl.BlockedLayout(
            [1, 1], [1, 64], [gl.num_warps(), 1], [1, 0]
        )
        word = gl.convert_layout(word, cl)
        mm = gl.arange(0, M_DUP, gl.SliceLayout(1, cl))[:, None]
        p = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, cl))[None, :]
        pick_of = gl.convert_layout(
            gl.reshape(picks_smem.load(rl), [M_DUP]), gl.SliceLayout(1, cl)
        )[:, None]
    else:
        wl: gl.constexpr = word.type.layout
        mm = gl.arange(0, M_DUP, gl.SliceLayout(1, wl))[:, None]
        p = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, wl))[None, :]
        pick_of = gl.convert_layout(
            gl.reshape(picks_smem.load(rl), [M_DUP]), gl.SliceLayout(1, wl)
        )[:, None]
    at = ((RANK * TOP_K + pick_of) * OWNED_OUT + col) // 2 + p
    mb.put_peer(
        "rr",
        owner,
        mb.row_offset(mm) + at,
        word,
        mask=(pick_of >= 0) & (p >= 0),
    )


@gluon.jit
def _ex2_compute(mb, task, loads, gathered, RANK: gl.constexpr):
    """One ex2 slot after :func:`_ex2_gather_lane` (one row): the MXFP4 intermediate times
    the slot's w2 ``loads``, the BF16 partials put to the owner rank."""
    R: gl.constexpr = schedule.rows(mb.plan)
    blk = task % NB2
    a, picks, order = gathered
    p_smem, s_smem = a
    owner = (blk * BLOCK_N) // OWNED_OUT
    acc = _mma_rows(p_smem, s_smem, loads)
    out = gl.reshape(acc.to(gl.bfloat16), [M_DUP, BLOCK_N // 2, 2])
    word = mx.pack_bf16_pairs(out)
    wl: gl.constexpr = word.type.layout
    mm = gl.arange(0, M_DUP, gl.SliceLayout(1, wl))[:, None]
    p = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, wl))[None, :]
    # The row and pick of each MFMA row.
    row_of = gl.full([M_DUP, 1], 0, gl.int32, wl)
    pick_of = gl.full([M_DUP, 1], -1, gl.int32, wl)
    for r in gl.static_range(R):
        mine = (mm == order[r]) & (picks[r] >= 0)
        row_of = gl.where(mine, r, row_of)
        pick_of = gl.where(mine, picks[r], pick_of)
    col = (blk * BLOCK_N) % OWNED_OUT
    at = ((RANK * TOP_K + pick_of) * OWNED_OUT + col) // 2 + p
    mb.put_peer(
        "rr",
        owner,
        mb.row_offset(row_of) + at,
        word,
        mask=(pick_of >= 0) & (p >= 0),
    )


@gluon.jit
def _ex2_tail_batch(
    mb,
    task,
    union,
    table,
    lane,
    count,
    w2,
    w2s,
    stride_we,
    stride_se,
    stride_slin,
    stride_snb,
    S: gl.constexpr,
    A: gl.constexpr,
    RANK: gl.constexpr,
):
    """:func:`ex2_finish`'s second batch past 8 rows: the lane's slots ``S`` ..
    ``2 S - 1`` as its first ``S``, from one gather (:func:`_ex2_gather_lane`
    with ``K0 = S``) and each slot's weights ``A`` slots ahead of its product."""
    tiles = ()
    for k in gl.static_range(A):
        tiles += (
            _ex2_loads(
                task,
                True,
                lane_slot_expert(table[0], S + k),
                w2,
                w2s,
                stride_we,
                stride_se,
                stride_slin,
                stride_snb,
            ),
        )
    p_smem, s_smem, picks_smem = _ex2_gather_lane(mb, union, table[1], lane, S, True, S)
    for k in gl.static_range(S):
        i = lane + (S + k) * LANES
        if k + A < S:
            tiles += (
                _ex2_loads(
                    task,
                    True,
                    lane_slot_expert(table[0], S + k + A),
                    w2,
                    w2s,
                    stride_we,
                    stride_se,
                    stride_slin,
                    stride_snb,
                ),
            )
        if i < count:
            a = (
                p_smem.slice(k * M_DUP, M_DUP, dim=0),
                s_smem.slice(k * M_DUP, M_DUP, dim=0),
            )
            _ex2_compute_picked(
                mb, task, tiles[k], a, picks_smem.slice(k, 1, dim=0), RANK
            )


@gluon.jit
def ex2_finish(
    mb,
    task,
    loads,
    expert,
    union,
    table,
    w2,
    w2s,
    stride_we,
    stride_se,
    stride_slin,
    stride_snb,
    RANK: gl.constexpr,
    STAGE: gl.constexpr,
):
    """Ex2 task ``task`` after :func:`ex2_issue` (``loads``, of the lane's
    first slot, ``expert``) and :func:`slots` (``union``): the lane's first
    ``EX2_GATHER`` slots from one gather of their intermediates, the first
    ``EX2_AHEAD`` slots' weights issued before the gather and each later
    slot's that many slots ahead of its product; then (past 64 distinct
    experts, so past 8 rows) the lane's other slots, up to ``EX2_GATHER``
    more, in a second batch of the same form (:func:`_ex2_tail_batch`).

    A poll waits for every earlier load of its wave (``vmcnt`` counts in
    order): polled a slot at a time between the slots' weights, each slot
    would pay its weights' latency and a poll round trip, well past their
    stream at 8 rows. The unrolled slots carry no weights
    across a loop, whose copies would wait for the loads in flight.

    ``table``: past 8 rows, the CTA's :func:`lane_slots`, which gives the
    slots' experts with one LDS load each."""
    R: gl.constexpr = schedule.rows(mb.plan)
    RP: gl.constexpr = triton.next_power_of_2(R)
    S: gl.constexpr = min(RP, EX2_GATHER)
    A: gl.constexpr = max(2, min(S, EX2_AHEAD))
    lane = task // NB2
    count = union[3]
    TABLE: gl.constexpr = R > 8
    tiles = (loads,)
    for k in gl.static_range(1, A):
        tiles += (
            _ex2_loads(
                task,
                True,
                (
                    lane_slot_expert(table[0], k)
                    if TABLE
                    else _slot_expert(union, lane + k * LANES)
                ),
                w2,
                w2s,
                stride_we,
                stride_se,
                stride_slin,
                stride_snb,
            ),
        )
    if R > 1:
        # Past one row, each slot's row picks come from LDS (kept by the
        # gather from its one reduction): per slot, the R row scalars of
        # _row_picks were R cross-lane reductions.
        p_smem, s_smem, picks_smem = _ex2_gather_lane(
            mb, union, table[1], lane, S, True, 0
        )
    else:
        p_smem, s_smem = _ex2_gather_lane(mb, union, table[1], lane, S, False, 0)
    mb.stamp(STAGE, trace.READY)
    # MFMA row r is row r (rows that did not pick the slot's expert are zero).
    rows = ()
    for r in gl.static_range(R):
        rows += (r,)
    for k in gl.static_range(S):
        i = lane + k * LANES
        if k + A < S:
            tiles += (
                _ex2_loads(
                    task,
                    True,
                    (
                        lane_slot_expert(table[0], k + A)
                        if TABLE
                        else _slot_expert(union, i + A * LANES)
                    ),
                    w2,
                    w2s,
                    stride_we,
                    stride_se,
                    stride_slin,
                    stride_snb,
                ),
            )
        if i < count:
            a = (
                p_smem.slice(k * M_DUP, M_DUP, dim=0),
                s_smem.slice(k * M_DUP, M_DUP, dim=0),
            )
            if R > 1:
                _ex2_compute_picked(
                    mb, task, tiles[k], a, picks_smem.slice(k, 1, dim=0), RANK
                )
            else:
                picks = _row_picks(union, _slot_expert(union, i), R)
                _ex2_compute(mb, task, tiles[k], (a, picks, rows), RANK)
    if S < RP:
        # Past 64 distinct experts (more than S slots a lane, so past 8 rows):
        # the lane's slots S .. 2 S - 1 in a second batch, one gather of their
        # intermediates and the same A-ahead weights, instead of a poll round
        # trip and the weights' latency a slot. A lane has at most 2 S slots.
        gl.static_assert(TABLE and RP == 2 * S, "the tail batch serves slots S .. 2S-1")
        if count > lane + S * LANES:
            _ex2_tail_batch(
                mb,
                task,
                union,
                table,
                lane,
                count,
                w2,
                w2s,
                stride_we,
                stride_se,
                stride_slin,
                stride_snb,
                S,
                A,
                RANK,
            )
    mb.stamp(STAGE, trace.COMPUTED)
    mb.stamp(STAGE, trace.END)

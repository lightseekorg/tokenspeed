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

"""The fused Muse Spark decoder layer on gfx950: one persistent launch.

One plain launch of ``GRID`` workgroups x 8 waves, one per CU, runs a whole
decoder layer and the next layer's query projection. A workgroup's roles
come from its physical place, ``(xcd, slot)`` (the hardware XCD and the
program id // 8, see ``persistent.enter_issue``), and the plan key's
constexprs; ``plan.py`` models the same roles and checks their waits. Roles
hand work to each other through memory; each hand-off below states its
producer's ordering (drain, barrier, flag or single-store tagged words) and
its consumer's loads.

Phases, in the order a workgroup runs its roles:

- front: attention splits and their record mergers, the o_proj senders
  (SND) and column owners (OWN1), the router and pooler K parts (RTR,
  POOL), the top-k (TOPK) and the pooled rows' norm and MXFP4 quantization
  (XN);
- MoE: every workgroup streams its share of the experts;
- tail: route owners (OWN4) and finishers (FIN) on slots 0 and 24-30,
  post-expert senders (UNP) and sum reducers (RED) on 16-23, one QKV task
  of the next layer on every workgroup, then the prologue heads (HEAD) on
  the FIN workgroups; when the next layer
  is regular, BND2 on the FIN workgroups writes the regular boundary
  instead of QKV and HEAD.

Every wait carries the launch deadline and abort check and stops the queue
through ``fail_stop`` instead of hanging.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tokenspeed_kernel._triton import gl, gluon, tl
from tokenspeed_kernel.ops.communication._iris.all_gather import (
    iris_rmsnorm_mxfp4_row,
)
from tokenspeed_kernel.ops.communication._iris.all_reduce import (
    iris_wait_lamport_slots,
)
from tokenspeed_kernel.ops.communication._iris.muse_spark.attntail import (
    muse_attntail_gated_rows,
    muse_attntail_publish_attention,
    muse_attntail_publish_rows,
    muse_attntail_read_box,
    muse_attntail_read_box_rows,
)
from tokenspeed_kernel.ops.communication._iris.muse_spark.fused_layer import (
    BOX_WORDS,
    H1_SHARD,
    H1_WORDS,
    H6_SLOT,
    H6_WORDS,
    MAX_ROWS,
    H4SinkArgs,
    MuseNativeView,
    clear_next_copies,
    fin,
    h1_rs_sink,
    h3_sink,
    h4_sink,
    h6_sink,
    iris_fail_stop,
    own1,
    own4_complete,
    red_sums,
    unp_sums_publish,
    xn_wait,
)
from tokenspeed_kernel.ops.decoder_layer import PlanKey
from tokenspeed_kernel.ops.decoder_layer._gluon.muse_spark import plan_variants
from tokenspeed_kernel_amd.ops.gfx950.attention._common import InputStrides
from tokenspeed_kernel_amd.ops.gfx950.attention.mha.decode import (
    AttentionConfig,
    AttentionProgram,
    merge_balanced_rows_gfx950,
    mha_decode_balanced_item_gfx950,
    mha_decode_balanced_split_gfx950,
    mha_decode_sliding_split_gfx950,
    publish_balanced_record_gfx950,
    store_balanced_partial_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.attention.prologue.gqa import (
    gqa_prologue_key_gfx950,
    gqa_prologue_query_gfx950,
    gqa_prologue_rope_gfx950,
    gqa_prologue_rows_layout,
    gqa_prologue_value_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.gemv_splitk import (
    gemv_a16w16_splitk_b_step_gfx950,
    gemv_a16w16_splitk_finish_steps_gfx950,
    gemv_a16w16_splitk_hi_lo_sums_steps_gfx950,
    gemv_a16w16_splitk_k_vector_step_gfx950,
    gemv_a16w16_splitk_k_vector_to_shared_gfx950,
    gemv_a16w16_splitk_publish_gfx950,
    gemv_a16w16_splitk_sum_gfx950,
    gemv_a16w16_splitk_wave_sums_gfx950,
    gemv_a16w16_splitk_wave_sums_steps_gfx950,
    gemv_a16w16_splitk_weight_step_tiles_gfx950,
    gemv_a16w16_splitk_weight_steps_gfx950,
    gemv_a16w16_splitk_weight_steps_windowed_gfx950,
    gemv_a16w16_splitk_weights_gfx950,
    gemv_a16w16_splitk_weights_paced_gfx950,
    gemv_a16w16_splitk_x_steps_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm import (
    mm_a16w16_column_relay_gfx950,
    mm_a16w16_column_relay_weights_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.layernorm.deferred import (
    deferred_norm_rows_gfx950,
    deferred_norm_scales_gfx950,
    deferred_norm_terms_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.native_stream import (
    MoeBuffers,
    prefetch_moe,
    run_moe,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.routing import (
    sigmoid_bias_topk_keys_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.persistent import (
    FIRST_KERNEL_SITE,
    GRID,
    enter_check,
    enter_issue,
    fail_stop,
    persistent_compile_options,
    poll_tags16,
    poll_until_all,
    release_drain,
    trace_stamp,
)

__all__ = [
    "NUM_WARPS",
    "PLAN_VARIANTS",
    "gluon_decoder_layer_muse_spark_gfx950",
    "DIVISIBLE_ARGS",
    "offline_attrs",
    "offline_compile_args",
    "offline_signature",
]

NUM_WARPS = 8
WORLD_SIZE = 8
HIDDEN = 8192
# Per rank (TP 8): query heads, KV heads, head size, attention width.
Q_HEADS = 16
KV_HEADS = 2
HEAD_DIM = 64
Q_WIDTH = Q_HEADS * HEAD_DIM
EXPERTS = 256
TOPK = 8
# The pooled (pre-expert) width and this rank's columns of it.
POOLED = 4096
POOLED_LOCAL = POOLED // WORLD_SIZE
ROWS_PER_REQUEST = 8
# K parts of the router and pooler (one per XCD, an owner rank's columns).
KPARTS = 8
ROUTER_GROUPS = EXPERTS // 16
POOL_TILE = 64
POOL_TILES = POOLED_LOCAL // POOL_TILE
# Sliding attention: 8 splits per (request, KV head) block, on the block's XCD.
SLIDING_SPLITS = 8
# Full attention: the balanced decode's items (its grid: at least 512
# program indices), two per workgroup at most; one merge task per 8 rows of
# a (request, KV head) block (one row per wave).
FULL_PROGRAMS = 512
FULL_MERGE_TASKS_PER_BLOCK = 64 // 8
ATT_FIRST_SLOT = 24
# One 16-row MFMA tile per wave tiles the attention rows.
ATT_BLOCK_M = 16 * NUM_WARPS

# The next layer's fused QKV + output-gate projection: 16 q heads, 2 K and 2
# V heads, 16 gate heads of 64; one 144-row task per (group, K part).
QKV_HEADS = Q_HEADS + 2 * KV_HEADS + Q_HEADS
QKV_WIDTH = QKV_HEADS * HEAD_DIM
QKV_GROUPS = 16
QKV_PARTS = 16
QKV_TASK_ROWS = QKV_WIDTH // QKV_GROUPS
# HEAD tasks: the q heads, the (K, V) head pairs, then the gate heads.
HEAD_TASKS = Q_HEADS + KV_HEADS + Q_HEADS
# Post-expert senders (UNP) per rank: 16 hidden columns each.
UNP_COLUMNS = 16

# Tagged K-part partials (router, pooler, QKV) keep 16 token slots per part
# whatever the launch's rows: every slot is rewritten with the launch's tag
# each launch (rows past the launch's carry unused values), so no slot keeps
# a tag old enough to repeat (tags repeat after 2**32 - 1 launches).
PART_ROWS = 16
# Attention records (int32, one per wave): two copies by launch parity; each
# launch zeroes the other copy, so no record keeps an old tag either.
RECORD_WORDS = max(2 * KV_HEADS * SLIDING_SPLITS, FULL_PROGRAMS) * NUM_WARPS
assert RECORD_WORDS % GRID == 0

# Stamps (``STAMPS``): per workgroup, the clock at these points of its
# flow (int64, ``STAMP_STRIDE`` per workgroup, logical id order).
STAMP_ENTRY = 0
STAMP_CHECKED = 1
STAMP_FRONT = 2
STAMP_IDS = 3
STAMP_XQ = 4
STAMP_MOE = 5
STAMP_TAIL = 6
STAMP_QKV = 7
STAMP_END = 8
# The tail's hand-offs: OWN4, FIN and S published (last row of the
# workgroup), UNP's S rows seen and H6 pushed, RED done, QKV x'' ready.
STAMP_OWN4 = 9
STAMP_FIN = 10
STAMP_S = 11
STAMP_UNP_S = 12
STAMP_H6 = 13
STAMP_RED = 14
STAMP_QKV_X = 15
STAMP_STRIDE = 16

# The uncached exchange (int32 words): routed ids, route weights, the ids
# and xq row tags (one 64 B line per 16 rows), the h' part flags (one line
# per 1024-column part) and the S part tags ([part][row]). The xq rows and
# scales are uncached buffers of their own.
XQ_ROW_BYTES = POOLED // 2
XS_ROW_BYTES = POOLED // 32
EX_IDS = 0
EX_WEIGHTS = EX_IDS + MAX_ROWS * TOPK
EX_IDS_TAGS = EX_WEIGHTS + MAX_ROWS * TOPK
EX_XQ_TAGS = EX_IDS_TAGS + MAX_ROWS
EX_HP_TAGS = EX_XQ_TAGS + MAX_ROWS
EX_S_TAGS = EX_HP_TAGS + KPARTS * 16
# The xq scales are CDNA4-swizzled (``empty_swizzled_cdna4_mxfp4_scale``):
# one 32-row block, stride 1 within it.
XS_MBLOCK_STRIDE = MAX_ROWS * XS_ROW_BYTES
EXCHANGE_WORDS = EX_S_TAGS + KPARTS * MAX_ROWS
# Every polled tag line starts on a 64 B line (16 words).
assert all(
    offset % 16 == 0 for offset in (EX_IDS_TAGS, EX_XQ_TAGS, EX_HP_TAGS, EX_S_TAGS)
)
assert MAX_ROWS % 16 == 0

# Failure sites of the layer's own waits (fused_layer's hop ends use
# FIRST_KERNEL_SITE + 16 .. + 22, the MoE stream + 8).
SITE_RECORDS = FIRST_KERNEL_SITE + 32
SITE_BOX = SITE_RECORDS + 1
SITE_SUMS = SITE_RECORDS + 2
SITE_AG = SITE_RECORDS + 3
SITE_LOGITS = SITE_RECORDS + 4
SITE_POOL = SITE_RECORDS + 5
SITE_IDS = SITE_RECORDS + 6
SITE_XQ = SITE_RECORDS + 7
SITE_HPRIME = SITE_RECORDS + 8
SITE_S = SITE_RECORDS + 9
SITE_H6 = SITE_RECORDS + 10
SITE_H6_SUMS = SITE_RECORDS + 11
SITE_QKV = SITE_RECORDS + 12

_NUM_WARPS = gl.constexpr(NUM_WARPS)
_HIDDEN = gl.constexpr(HIDDEN)
_Q_WIDTH = gl.constexpr(Q_WIDTH)
_EXPERTS = gl.constexpr(EXPERTS)
_TOPK = gl.constexpr(TOPK)
_POOLED = gl.constexpr(POOLED)
_POOLED_LOCAL = gl.constexpr(POOLED_LOCAL)
_KPARTS = gl.constexpr(KPARTS)
_ROUTER_GROUPS = gl.constexpr(ROUTER_GROUPS)
_POOL_TILE = gl.constexpr(POOL_TILE)
_POOL_TILES = gl.constexpr(POOL_TILES)
_ATT_FIRST_SLOT = gl.constexpr(ATT_FIRST_SLOT)
_ATT_BLOCK_M = gl.constexpr(ATT_BLOCK_M)
_FULL_PROGRAMS = gl.constexpr(FULL_PROGRAMS)
_FULL_MERGE_TASKS = gl.constexpr(FULL_MERGE_TASKS_PER_BLOCK)
_GRID = gl.constexpr(GRID)
_MAX_ROWS = gl.constexpr(MAX_ROWS)
_BOX_WORDS = gl.constexpr(BOX_WORDS)
_H1_WORDS = gl.constexpr(H1_WORDS)
_H1_SHARD = gl.constexpr(H1_SHARD)
_OWNERS = gl.constexpr(WORLD_SIZE * (HIDDEN // WORLD_SIZE // 128))
_EX_IDS = gl.constexpr(EX_IDS)
_EX_WEIGHTS = gl.constexpr(EX_WEIGHTS)
_EX_IDS_TAGS = gl.constexpr(EX_IDS_TAGS)
_EX_XQ_TAGS = gl.constexpr(EX_XQ_TAGS)
_XS_MBLOCK_STRIDE = gl.constexpr(XS_MBLOCK_STRIDE)
_EX_HP_TAGS = gl.constexpr(EX_HP_TAGS)
_EX_S_TAGS = gl.constexpr(EX_S_TAGS)
_H6_SLOT = gl.constexpr(H6_SLOT)
_H6_WORDS = gl.constexpr(H6_WORDS)
_Q_HEADS = gl.constexpr(Q_HEADS)
_KV_HEADS = gl.constexpr(KV_HEADS)
_HEAD_DIM = gl.constexpr(HEAD_DIM)
_QKV_WIDTH = gl.constexpr(QKV_WIDTH)
_QKV_GROUPS = gl.constexpr(QKV_GROUPS)
_QKV_PARTS = gl.constexpr(QKV_PARTS)
_QKV_TASK_ROWS = gl.constexpr(QKV_TASK_ROWS)
_HEAD_TASKS = gl.constexpr(HEAD_TASKS)
_UNP_COLUMNS = gl.constexpr(UNP_COLUMNS)
_STAMP_STRIDE = gl.constexpr(STAMP_STRIDE)
_STAMP_OWN4 = gl.constexpr(STAMP_OWN4)
_STAMP_FIN = gl.constexpr(STAMP_FIN)
_STAMP_S = gl.constexpr(STAMP_S)
_STAMP_UNP_S = gl.constexpr(STAMP_UNP_S)
_STAMP_H6 = gl.constexpr(STAMP_H6)
_STAMP_RED = gl.constexpr(STAMP_RED)
_STAMP_QKV_X = gl.constexpr(STAMP_QKV_X)
_PART_ROWS = gl.constexpr(PART_ROWS)
_RECORD_WORDS = gl.constexpr(RECORD_WORDS)
_STAMP_ENTRY = gl.constexpr(STAMP_ENTRY)
_STAMP_CHECKED = gl.constexpr(STAMP_CHECKED)
_STAMP_FRONT = gl.constexpr(STAMP_FRONT)
_STAMP_IDS = gl.constexpr(STAMP_IDS)
_STAMP_XQ = gl.constexpr(STAMP_XQ)
_STAMP_MOE = gl.constexpr(STAMP_MOE)
_STAMP_TAIL = gl.constexpr(STAMP_TAIL)
_STAMP_QKV = gl.constexpr(STAMP_QKV)
_STAMP_END = gl.constexpr(STAMP_END)
_SITE_HPRIME = gl.constexpr(SITE_HPRIME)
_SITE_S = gl.constexpr(SITE_S)
_SITE_H6 = gl.constexpr(SITE_H6)
_SITE_H6_SUMS = gl.constexpr(SITE_H6_SUMS)
_SITE_QKV = gl.constexpr(SITE_QKV)
_SITE_RECORDS = gl.constexpr(SITE_RECORDS)
_SITE_BOX = gl.constexpr(SITE_BOX)
_SITE_AG = gl.constexpr(SITE_AG)
_SITE_SUMS = gl.constexpr(SITE_SUMS)
_SITE_LOGITS = gl.constexpr(SITE_LOGITS)
_SITE_POOL = gl.constexpr(SITE_POOL)
_SITE_IDS = gl.constexpr(SITE_IDS)
_SITE_XQ = gl.constexpr(SITE_XQ)


# ---------------------------------------------------------------------------
# Layouts
# ---------------------------------------------------------------------------


@gluon.constexpr_function
def _sums_layout(columns: int, vector: int):
    """``gemv_a16w16_splitk_wave_sums_gfx950``'s ``OUT_LAYOUT`` for 16
    tokens x ``columns``: ``vector`` contiguous columns per thread."""
    lanes = columns // vector
    return gl.BlockedLayout(
        [NUM_WARPS, 1, vector], [1, 64 // lanes, lanes], [1, NUM_WARPS, 1], [2, 1, 0]
    )


@gluon.constexpr_function
def _norm_sums_layout():
    """[16 tokens, owners, 4 sums]: a token's owners across one wave's lanes,
    its four sums in one thread (``deferred_norm_scales_gfx950``)."""
    return gl.BlockedLayout(
        [16 // NUM_WARPS, 1, 4], [1, 64, 1], [NUM_WARPS, 1, 1], [2, 1, 0]
    )


@gluon.jit
def _stamp(stamps_ptr, cta, point: gl.constexpr, STAMPS: gl.constexpr):
    """With ``STAMPS``, store the clock for this workgroup's ``point`` (one
    plain store; compiled out otherwise)."""
    if STAMPS:
        trace_stamp(stamps_ptr + cta * _STAMP_STRIDE, point)


@gluon.jit
def _launch_records(att_records, launch):
    """This launch's copy of the attention records; zero this workgroup's
    share of the other copy, which the next launch uses (nothing reads it in
    this one)."""
    lanes_l: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    lane = gl.arange(0, 64 * gl.num_warps(), lanes_l)
    SHARE: gl.constexpr = _RECORD_WORDS // _GRID
    gl.amd.cdna4.buffer_store(
        gl.zeros([64 * gl.num_warps()], gl.int32, lanes_l),
        att_records + (1 - launch.copy) * _RECORD_WORDS,
        launch.cta * SHARE + lane,
        mask=lane < SHARE,
    )
    return att_records + launch.copy * _RECORD_WORDS


@gluon.jit
def _reserve_lds():
    """Hold 96 KB of LDS live once, more than half of the CU's 160 KB, so
    exactly one workgroup sits on each CU; the roles reuse the space."""
    layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    zeros = gl.zeros([64 * gl.num_warps()], gl.int32, lanes)
    # Allocation sizes are powers of two: 64 KB + 32 KB, live together.
    big = gl.allocate_shared_memory(gl.int32, [16384], layout)
    small = gl.allocate_shared_memory(gl.int32, [8192], layout)
    big.slice(0, 64 * gl.num_warps()).store(zeros)
    small.slice(0, 64 * gl.num_warps()).store(zeros)


# ---------------------------------------------------------------------------
# Front: sliding attention
# ---------------------------------------------------------------------------


@gluon.jit
def _wait_records(record_ptr, first, count: gl.constexpr, tag, launch):
    """The record merger's wait (same XCD, scalar ``glc``): every split's per-wave records
    carry ``tag``; 16 records per 64 B line."""
    gl.static_assert(count % 16 == 0)
    # Every wave polls and then reads the partials itself (no barrier).
    for line in gl.static_range(count // 16):
        if not poll_tags16(record_ptr + first + 16 * line, tag, launch, True, False):
            iris_fail_stop(launch, _SITE_RECORDS, line)


@gluon.jit
def _attention_split(
    launch,
    block,
    split,
    view,
    q_ptr,
    gate_ptr,
    gate_stride,
    k_cache_ptr,
    v_cache_ptr,
    page_table_ptr,
    cache_seqlens_ptr,
    mid_o_ptr,
    mid_lse_ptr,
    record_ptr,
    cfg: gl.constexpr,
    GATE_EPS: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    ATT_MERGE: gl.constexpr,
):
    """One sliding split of block ``block`` = (request, KV head).

    ``ATT_MERGE`` picks who merges: ``"designated"`` (split 0 merges the
    block's rows) or ``"spread"`` (split ``s`` merges block rows ``[s R, (s +
    1) R)``, ``R = ROWS / NUM_KV_SPLITS``, one row per wave; every split
    then reads 1/NUM_KV_SPLITS of the partials). Either way the rows are
    bitwise the same.

    The split streams its pages and stores its partial (plain, read on
    this XCD); each wave drains its stores, then stores its record
    (``publish_split_record``); the merger polls the block's records,
    merges in split order (bitwise to the last-arriver merge) and publishes
    the gated, normalized rows to the attention box (written through).
    """
    program = AttentionProgram.create(
        cfg,
        q_ptr,
        k_cache_ptr,
        v_cache_ptr,
        page_table_ptr,
        cache_seqlens_ptr,
        mid_o_ptr,
        mid_lse_ptr,
        block // cfg.NUM_KV_HEADS,
        (block % cfg.NUM_KV_HEADS) * cfg.NUM_ROW_BLOCKS,
        split,
    )
    mha_decode_sliding_split_gfx950(
        program, q_ptr, k_cache_ptr, v_cache_ptr, NUM_STAGES
    )
    program.publish_split_record(record_ptr, launch.tag)
    box = view.box + launch.copy * _BOX_WORDS
    if ATT_MERGE == "spread":
        SHARE: gl.constexpr = cfg.ROWS // cfg.NUM_KV_SPLITS
        gl.static_assert(SHARE * cfg.NUM_KV_SPLITS == cfg.ROWS)
        gl.static_assert(cfg.NUM_ROW_BLOCKS == 1)
        rows_layout: gl.constexpr = gl.BlockedLayout(
            [1, 1], [1, 64], [cfg.NUM_WARPS, 1], [1, 0]
        )
        m = split * SHARE + gl.arange(0, SHARE, layout=gl.SliceLayout(1, rows_layout))
        d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, rows_layout))
        gate = gl.amd.cdna4.buffer_load(
            gate_ptr,
            (program.row_tokens(m) * gate_stride + program.row_heads(m) * cfg.HEAD_DIM)[
                :, None
            ]
            + d[None, :],
            mask=program.row_valid(m)[:, None] & (d >= 0)[None, :],
        )
        merged = program.merge_split_records_rows(
            record_ptr, launch.tag, m, d, _wait_records, launch
        )
        muse_attntail_publish_rows(program, m, merged, gate, box, GATE_EPS)
    elif split == 0:
        gl.static_assert(ATT_MERGE == "designated")
        m = gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.store_layout))
        d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.store_layout))
        gate = gl.amd.cdna4.buffer_load(
            gate_ptr,
            (program.row_tokens(m) * gate_stride + program.row_heads(m) * cfg.HEAD_DIM)[
                :, None
            ]
            + d[None, :],
            mask=program.row_valid(m)[:, None] & (d >= 0)[None, :],
        )
        merged = program.merge_split_records(
            record_ptr, launch.tag, q_ptr, False, _wait_records, launch
        )
        muse_attntail_publish_attention(program, merged, gate, box, GATE_EPS)


# ---------------------------------------------------------------------------
# Front: full attention
# ---------------------------------------------------------------------------


@gluon.jit
def _wait_records_gpu(record_ptr, first, count, tag, launch):
    """The full-layer merge's wait: ``count`` int32 records from ``first``,
    published on any XCD with device-scope stores; one per lane, relaxed
    device-scope polls (``poll_until_all`` ``"sc1"``)."""
    lanes = gl.arange(
        0,
        64 * gl.num_warps(),
        layout=gl.BlockedLayout([1], [64], [gl.num_warps()], [0]),
    )
    for chunk in range(0, count, 64 * gl.num_warps()):
        index = chunk + lanes
        if not poll_until_all(
            record_ptr + first + index, index < count, tag, launch, "sc1"
        ):
            iris_fail_stop(launch, _SITE_RECORDS, chunk)


@gluon.jit
def _no_weights(args):
    """No resident weights (a value, since the issue point sits in a branch)."""
    return gl.to_tensor(0)


# Front weight pacing. Each stream that overlaps a latency-bound chain keeps
# few enough bytes in flight per CU that the GPU's total stays below ~4.5
# TB/s (polls stay flat up to ~4.9 TB/s of other traffic and double above
# ~6), at ~1.27 us per round trip:
# - o_proj at a sliding layer's entry, one 32 KB tile per CU in flight: 128
#   CUs ~3.3 TB/s, landed ~5 us after entry, well before the box;
# - the pooler, two 8 KB steps per CU: 64 CUs ~0.8 TB/s, landed ~10 us
#   after entry (sliding) or ~5 us after the items (full: four steps), before
#   the all-gather;
# - the router after the sender's push (slot 0: after OWN1), four steps per
#   CU: 128 CUs ~3.3 TB/s, landed ~2.5 us later, before the all-gather.
# A full layer's o_proj stays unpaced at the issue hook: the sender waits
# for it, so it takes the whole fabric.
_O_PROJ_WINDOW = gl.constexpr(1)
_POOLER_LOADS_SLIDING = gl.constexpr(2)
_POOLER_LOADS_FULL = gl.constexpr(4)
_ROUTER_LOADS = gl.constexpr(4)


@gluon.jit
def _sender_weights(args):
    """SND's resident weights at a full layer's issue hook (unpaced): the
    o_proj rows of ``sender``."""
    o_proj_ptr, sender = args
    return gemv_a16w16_splitk_weights_gfx950(o_proj_ptr, sender, _Q_WIDTH, 64, 1, 128)


@gluon.jit
def _sender_weights_paced(o_proj_ptr, sender):
    """SND's resident weights at a sliding layer's entry, one tile in
    flight per CU."""
    return gemv_a16w16_splitk_weights_paced_gfx950(
        o_proj_ptr, sender, _Q_WIDTH, 64, 1, 128, _O_PROJ_WINDOW
    )


@gluon.jit
def _router_weights(router_hi_ptr, router_lo_ptr, task):
    """RTR's resident weights: the router's hi / lo tiles of its K-part
    task, paced as one stream of K steps."""
    STEPS: gl.constexpr = _HIDDEN // _KPARTS // _NUM_WARPS // 32
    hi = gemv_a16w16_splitk_weight_steps_windowed_gfx950(
        router_hi_ptr, task, _HIDDEN, 16, _KPARTS, _ROUTER_GROUPS, _ROUTER_LOADS, 0
    )
    lo = gemv_a16w16_splitk_weight_steps_windowed_gfx950(
        router_lo_ptr, task, _HIDDEN, 16, _KPARTS, _ROUTER_GROUPS, _ROUTER_LOADS,
        STEPS,
    )  # fmt: skip
    return hi, lo


@gluon.jit
def _pooler_weights_sliding(pooler_ptr, task):
    """POOL's resident weights at a sliding layer's entry (paced)."""
    return gemv_a16w16_splitk_weight_steps_windowed_gfx950(
        pooler_ptr, task, _HIDDEN, _POOL_TILE, _KPARTS, _POOL_TILES,
        _POOLER_LOADS_SLIDING, 0,
    )  # fmt: skip


@gluon.jit
def _pooler_weights(args):
    """POOL's resident weights at a full layer's issue hook (paced)."""
    pooler_ptr, task = args
    return gemv_a16w16_splitk_weight_steps_windowed_gfx950(
        pooler_ptr, task, _HIDDEN, _POOL_TILE, _KPARTS, _POOL_TILES,
        _POOLER_LOADS_FULL, 0,
    )  # fmt: skip


@gluon.jit
def _full_items(
    launch,
    cfg: gl.constexpr,
    q_ptr,
    cache_seqlens_ptr,
    schedule_ptr,
    page_table_ptr,
    k_cache_ptr,
    v_cache_ptr,
    mid_o_ptr,
    mid_lse_ptr,
    record_ptr,
    REQUESTS: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    ISSUE: gl.constexpr,
    issue_args,
):
    """A full layer's splits: this workgroup's balanced decode items
    (program indices ``cta`` and ``cta + GRID``). Each partial is written
    through; its record (device scope) waits only for the partial's stores.
    The role's weights (``ISSUE(issue_args)``) issue after the last item's
    partial store, so they hold up neither the pages, the stores nor the
    records.

    Returns:
        ``ISSUE``'s weights.
    """
    gl.static_assert(_FULL_PROGRAMS <= 2 * _GRID)
    program, live, block_item, splits = mha_decode_balanced_item_gfx950(
        cfg, q_ptr, cache_seqlens_ptr, REQUESTS, schedule_ptr, page_table_ptr,
        k_cache_ptr, v_cache_ptr, mid_o_ptr, mid_lse_ptr, launch.cta,
    )  # fmt: skip
    lanes_l: gl.constexpr = gl.BlockedLayout([1], [64], [cfg.NUM_WARPS], [0])
    token = gl.zeros([64 * cfg.NUM_WARPS], gl.int32, lanes_l)
    if live:
        m_i, l_i, acc = mha_decode_balanced_split_gfx950(
            program, k_cache_ptr, v_cache_ptr, NUM_STAGES
        )
        token = store_balanced_partial_gfx950(
            program, acc, l_i, m_i, block_item, splits, record_ptr
        )
    second, live2, block_item2, splits2 = mha_decode_balanced_item_gfx950(
        cfg, q_ptr, cache_seqlens_ptr, REQUESTS, schedule_ptr, page_table_ptr,
        k_cache_ptr, v_cache_ptr, mid_o_ptr, mid_lse_ptr, launch.cta + _GRID,
    )  # fmt: skip
    if live2:
        # Two items: the first's record goes out now, the weights after the
        # second's partial, so neither item's pages queue behind them.
        if live:
            publish_balanced_record_gfx950(
                program, record_ptr, block_item, launch.tag, token
            )
        m_i, l_i, acc = mha_decode_balanced_split_gfx950(
            second, k_cache_ptr, v_cache_ptr, NUM_STAGES
        )
        token2 = store_balanced_partial_gfx950(
            second, acc, l_i, m_i, block_item2, splits2, record_ptr
        )
        issued = ISSUE(issue_args)
        publish_balanced_record_gfx950(
            second, record_ptr, block_item2, launch.tag, token2
        )
    else:
        issued = ISSUE(issue_args)
        if live:
            publish_balanced_record_gfx950(
                program, record_ptr, block_item, launch.tag, token
            )
    return issued


@gluon.jit
def _full_merge(
    launch,
    task,
    view,
    cfg: gl.constexpr,
    schedule_ptr,
    mid_o_ptr,
    mid_lse_ptr,
    record_ptr,
    q_ptr,
    gate_ptr,
    gate_stride,
    GATE_EPS: gl.constexpr,
):
    """A full layer's merge: task ``task`` (8 rows of one (request, KV
    head) block, one per wave) polls the block's records, merges bitwise
    the balanced reduce's way, then gates, normalizes and publishes its rows
    to the attention box (written through), as the sliding merger does."""
    block = task // _FULL_MERGE_TASKS
    out, q_index, q_head = merge_balanced_rows_gfx950(
        cfg, schedule_ptr, mid_o_ptr, mid_lse_ptr, record_ptr, launch.tag, launch,
        q_ptr, block // cfg.NUM_KV_HEADS, block % cfg.NUM_KV_HEADS,
        (task % _FULL_MERGE_TASKS) * cfg.NUM_WARPS, False, _wait_records_gpu,
    )  # fmt: skip
    out_l: gl.constexpr = out.type.layout
    d = gl.arange(0, cfg.HEAD_DIM, gl.SliceLayout(0, out_l))
    q_index = gl.convert_layout(q_index, gl.SliceLayout(1, out_l))
    q_head = gl.convert_layout(q_head, gl.SliceLayout(1, out_l))
    gate = gl.load(
        gate_ptr + (q_index * gate_stride + q_head * cfg.HEAD_DIM)[:, None] + d[None, :]
    )
    rows = muse_attntail_gated_rows(out, gate, GATE_EPS)
    box = (view.box + launch.copy * _BOX_WORDS).to(
        gl.pointer_type(gl.bfloat16), bitcast=True
    )
    gl.amd.cdna4.buffer_store(
        rows,
        box,
        (q_index * _Q_WIDTH + q_head * cfg.HEAD_DIM)[:, None] + d[None, :],
        cache=".wt",
    )


# ---------------------------------------------------------------------------
# Front: o_proj senders and column owners
# ---------------------------------------------------------------------------


@gluon.jit
def _send(
    launch, view, sender, tiles, rows, RANK: gl.constexpr, PEER_CACHE: gl.constexpr
):
    """SND ``sender`` (0-127): re-arm its share of the next copies, wait for
    the attention box, multiply it by its resident o_proj rows and push
    the partial to the columns' owner rank."""
    clear_next_copies(view, launch, sender)
    x = muse_attntail_read_box(
        view.box + launch.copy * _BOX_WORDS, rows, _Q_WIDTH, launch, iris_fail_stop,
        _SITE_BOX,
    )  # fmt: skip
    out = gemv_a16w16_splitk_wave_sums_gfx950(tiles, x, _sums_layout(64, 2))
    # The entry clears land before any peer can see this launch's push.
    release_drain()
    h1_rs_sink(view, launch, out, sender, rows, RANK, PEER_CACHE)


# ---------------------------------------------------------------------------
# Front: router and pooler K parts, top-k, XN
# ---------------------------------------------------------------------------


@gluon.jit
def _norm_sums(launch, sums_ptr, rows, SITE: gl.constexpr):
    """A residual boundary's norm sums of the 16 token slots from its 64
    producers (H1: the column owners; H6: the sum reducers): tagged words
    ``[producer][MAX_ROWS][4]`` (tag high), polled until they carry the
    launch tag. Returns FP32 ``[16, 64, 4]`` in ``_norm_sums_layout``."""
    sums_l: gl.constexpr = _norm_sums_layout()
    t = gl.arange(0, 16, gl.SliceLayout(1, gl.SliceLayout(2, sums_l)))
    o = gl.arange(0, _OWNERS, gl.SliceLayout(0, gl.SliceLayout(2, sums_l)))
    w = gl.arange(0, 4, gl.SliceLayout(0, gl.SliceLayout(1, sums_l)))
    index = (o[None, :, None] * _MAX_ROWS + t[:, None, None]) * 4 + w[None, None, :]
    live = (t < rows)[:, None, None] & (o >= 0)[None, :, None] & (w >= 0)[None, None, :]
    sums = gemv_a16w16_splitk_sum_gfx950(
        sums_ptr,
        launch.tag,
        gl.reshape(index, [16 * _OWNERS * 4]),
        gl.reshape(live, [16 * _OWNERS * 4]),
        0,
        0,
        1,
        launch,
        iris_fail_stop,
        SITE,
    )
    return gl.convert_layout(gl.reshape(sums, [16, _OWNERS, 4]), sums_l)


@gluon.jit
def _vectors_to_shared(
    task,
    alpha_ptr,
    beta_ptr,
    post_ptr,
    pre_ptr,
    K_PARTS: gl.constexpr,
    GROUPS: gl.constexpr,
):
    """Copy task ``task``'s K slice of the four deferred-norm vectors into LDS
    (LDS-direct, one committed group): they do not depend on the polls, so
    they arrive while the task waits."""
    WAVES: gl.constexpr = gl.num_warps()
    KW: gl.constexpr = _HIDDEN // K_PARTS // WAVES
    flat: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    vectors = ()
    for ptr in gl.static_range(4):
        v_ptr = (alpha_ptr, beta_ptr, post_ptr, pre_ptr)[ptr]
        smem = gl.allocate_shared_memory(v_ptr.dtype.element_ty, [WAVES, KW], flat)
        gemv_a16w16_splitk_k_vector_to_shared_gfx950(
            smem, v_ptr, task, _HIDDEN, K_PARTS, GROUPS
        )
        vectors += (smem,)
    gl.amd.cdna4.async_copy.commit_group()
    return vectors


@gluon.jit
def _x_state(b_rows, h, vectors, inv1, inv2, column0, rows, h_ptr, store_h):
    """The state :func:`_x_step` builds x from, one K step at a time: ``b``
    (``[waves, KW, 16]``) stored to LDS, ``h`` per step, the vectors in LDS,
    the scales; with ``store_h`` each step's h' also goes to ``h_ptr`` rows
    from column ``column0``."""
    WAVES: gl.constexpr = gl.num_warps()
    b_s = gl.allocate_shared_memory(
        b_rows.dtype, list(b_rows.shape), gl.SwizzledSharedLayout(1, 1, 1, [1, 2, 0])
    )
    b_s.store(b_rows)
    token_l: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(1, h[0].type.layout))
    inv1 = gl.convert_layout(inv1, token_l)[None, None, :]
    inv2 = gl.convert_layout(inv2, token_l)[None, None, :]
    gl.amd.cdna4.async_copy.wait_group(0)
    gl.barrier()
    return (b_s, h, vectors, inv1, inv2, column0, rows, h_ptr, store_h)


@gluon.jit
def _x_step_stored(state, step, H_CACHE: gl.constexpr):
    """K step ``step`` of the deferred-norm x (FP32 ``[waves, 32, 16]``);
    h' stores use ``H_CACHE``."""
    b_s, h, vectors, inv1, inv2, column0, rows, h_ptr, store_h = state
    hp, x = deferred_norm_rows_gfx950(
        gemv_a16w16_splitk_b_step_gfx950(b_s, step),
        h[step],
        gemv_a16w16_splitk_k_vector_step_gfx950(vectors[0], step),
        gemv_a16w16_splitk_k_vector_step_gfx950(vectors[1], step),
        gemv_a16w16_splitk_k_vector_step_gfx950(vectors[2], step),
        gemv_a16w16_splitk_k_vector_step_gfx950(vectors[3], step),
        inv1,
        inv2,
    )
    if store_h:
        _store_step_rows(hp, h_ptr, column0, b_s.shape[1], step, rows, H_CACHE)
    return x


@gluon.jit
def _x_step(state, step):
    """The router's x step (FP32); h' is written through for readers on
    any XCD."""
    return _x_step_stored(state, step, ".wt")


@gluon.jit
def _x_step_bf16(state, step):
    """The pooler's and QKV's x step (BF16); h'' is stored plain (the next
    launch reads it)."""
    return _x_step_stored(state, step, "").to(gl.bfloat16)


@gluon.jit
def _kpart_state(
    launch,
    view,
    part,
    task,
    rows,
    residual_ptr,
    alpha_ptr,
    beta_ptr,
    post_ptr,
    pre_ptr,
    hprime_ptr,
    publish_h,
    GROUPS: gl.constexpr,
    POST_EPS: gl.constexpr,
    PRE_EPS: gl.constexpr,
):
    """K part ``part``'s x state (:func:`_x_step`): the deferred-norm build
    from the H1 all-gather words of owner rank ``part`` (Lamport
    payload-as-flag) and every owner's four norm sums (tagged words); with
    ``publish_h`` the steps write h' through to ``hprime``."""
    vectors = _vectors_to_shared(
        task, alpha_ptr, beta_ptr, post_ptr, pre_ptr, _KPARTS, GROUPS
    )
    # h is the previous launch's: its loads go ahead of the polls.
    h = gemv_a16w16_splitk_x_steps_gfx950(
        residual_ptr, _HIDDEN, task, rows, _HIDDEN, _KPARTS, GROUPS, ""
    )
    inv1, inv2 = deferred_norm_scales_gfx950(
        _norm_sums(launch, view.h1_sums, rows, _SITE_SUMS), _HIDDEN, POST_EPS, PRE_EPS
    )
    # Owner rank ``part``'s all-gather slot: [rows, 1024] BF16.
    ag = view.h1 + launch.copy * (2 * _H1_WORDS) + (_H1_WORDS + part * _H1_SHARD)
    b = muse_attntail_read_box_rows(
        ag, rows, _HIDDEN // _KPARTS, launch, iris_fail_stop, _SITE_AG
    )
    return _x_state(
        b, h, vectors, inv1, inv2, part * (_HIDDEN // _KPARTS), rows, hprime_ptr,
        publish_h,
    )  # fmt: skip


@gluon.jit
def _topk(
    launch,
    group,
    rows,
    logits_ptr,
    multiplier,
    bias_ptr,
    exchange_ptr,
    ROUTED_SCALING_FACTOR: gl.constexpr,
):
    """TOPK of rows ``8 group ..``, one wave per row: sum the router's K
    parts in part order (tagged words), pick the top 8,
    publish ids and weights to the uncached exchange, then each row's tag
    (rows past ``rows`` tag too, so the MoE polls whole 16-row lines)."""
    WAVES: gl.constexpr = gl.num_warps()
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 64], [WAVES, 1], [1, 0])
    row = group * WAVES + gl.arange(0, WAVES, gl.SliceLayout(1, layout))
    expert = gl.arange(0, _EXPERTS, gl.SliceLayout(0, layout))
    offsets = row[:, None] * _EXPERTS + expert[None, :]
    live = (row < rows)[:, None] & (expert >= 0)[None, :]
    logits = gemv_a16w16_splitk_sum_gfx950(
        logits_ptr,
        launch.tag,
        gl.reshape(offsets, [WAVES * _EXPERTS]),
        gl.reshape(live, [WAVES * _EXPERTS]),
        _PART_ROWS,
        _EXPERTS,
        _KPARTS,
        launch,
        iris_fail_stop,
        _SITE_LOGITS,
    )
    logits = (
        gl.convert_layout(gl.reshape(logits, [WAVES, _EXPERTS]), layout) * multiplier
    )
    out_l: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [WAVES, 1], [1, 0])
    ids, weights = sigmoid_bias_topk_keys_gfx950(
        logits, bias_ptr, _EXPERTS, _TOPK, _TOPK, True, ROUTED_SCALING_FACTOR,
        gl.float32, out_l,
    )  # fmt: skip
    out_row = group * WAVES + gl.arange(0, WAVES, gl.SliceLayout(1, out_l))
    pick = gl.arange(0, _TOPK, gl.SliceLayout(0, out_l))
    slots = out_row[:, None] * _TOPK + pick[None, :]
    mask = (out_row < rows)[:, None] & (pick >= 0)[None, :]
    gl.amd.cdna4.buffer_store(ids, exchange_ptr, _EX_IDS + slots, mask=mask)
    gl.amd.cdna4.buffer_store(
        weights.to(gl.int32, bitcast=True), exchange_ptr, _EX_WEIGHTS + slots, mask=mask
    )
    # Each wave's tag after its own ids (uncached: no cache bits).
    release_drain()
    _publish_row_tags(launch, exchange_ptr + _EX_IDS_TAGS, group * WAVES, rows)


@gluon.jit
def _publish_row_tags(launch, tags_ptr, first_row, rows):
    """Lane 0 of wave w stores the launch tag for row ``first_row + w`` and,
    when the launch has at most 8 rows, for row ``first_row + w + 8`` (a
    16-row line is polled whole)."""
    WAVES: gl.constexpr = gl.num_warps()
    lanes_l: gl.constexpr = gl.BlockedLayout([1], [64], [WAVES], [0])
    lane = gl.arange(0, 64 * WAVES, lanes_l)
    row = first_row + lane // 64
    first = lane % 64 == 0
    tag = gl.full([64 * WAVES], 0, gl.int32, lanes_l) + launch.tag
    gl.amd.cdna4.buffer_store(tag, tags_ptr, row, mask=first)
    gl.amd.cdna4.buffer_store(tag, tags_ptr, row + WAVES, mask=first & (rows <= WAVES))


@gluon.jit
def _xn(
    launch,
    view,
    row,
    rows,
    norm_ptr,
    exchange_ptr,
    xq_ptr,
    xs_ptr,
    RANK: gl.constexpr,
    EPS: gl.constexpr,
    WEIGHT_OFFSET: gl.constexpr,
):
    """XN of ``row``: wait for every rank's pooled columns (H3), RMSNorm, MXFP4 to the uncached exchange, then the row's tag."""
    quant_layout: gl.constexpr = gl.BlockedLayout(
        [1, 16], [1, 64], [gl.num_warps(), 1], [1, 0]
    )
    quant_columns = gl.arange(0, 1024, layout=gl.SliceLayout(0, quant_layout))
    norm_weights = ()
    for rep in gl.static_range(_POOLED // 1024):
        norm_weights += (gl.load(norm_ptr + rep * 1024 + quant_columns).to(gl.float32),)
    words = xn_wait(view, launch, row, iris_fail_stop, RANK)
    iris_rmsnorm_mxfp4_row(
        words, row, norm_weights, xq_ptr, xs_ptr, _POOLED, EPS, WEIGHT_OFFSET, 1,
        _XS_MBLOCK_STRIDE,
    )  # fmt: skip
    # Every wave's xq stores land before the tag: drain, then a barrier.
    release_drain()
    gl.barrier()
    lanes_l: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    lane = gl.arange(0, 64 * gl.num_warps(), lanes_l)
    tag = gl.full([64 * gl.num_warps()], 0, gl.int32, lanes_l) + launch.tag
    tags = exchange_ptr + _EX_XQ_TAGS
    gl.amd.cdna4.buffer_store(tag, tags, row + lane * 0, mask=lane == 0)
    gl.amd.cdna4.buffer_store(
        tag, tags, row + 8 + lane * 0, mask=(lane == 0) & (rows <= 8)
    )


# ---------------------------------------------------------------------------
# Hand-offs through the uncached exchange
# ---------------------------------------------------------------------------


@gluon.jit
def _wait_line(launch, line_ptr, SITE: gl.constexpr):
    """Wait, in this wave, for a 64 B line of 16 tags in the uncached
    exchange (scalar loads; no barrier: the wave reads what it guards)."""
    if not poll_tags16(line_ptr, launch.tag, launch, True, False):
        iris_fail_stop(launch, SITE, 0)


@gluon.jit
def _publish_line(launch, line_ptr):
    """After a workgroup's payload stores (every wave drained, then a
    barrier): store the launch tag in all 16 words of a 64 B line."""
    lanes_l: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    lane = gl.arange(0, 64 * gl.num_warps(), lanes_l)
    tag = gl.full([64 * gl.num_warps()], 0, gl.int32, lanes_l) + launch.tag
    gl.amd.cdna4.buffer_store(tag, line_ptr, lane, mask=lane < 16)


@gluon.jit
def _store_step_rows(
    values,
    ptr,
    column0,
    KW: gl.constexpr,
    step: gl.constexpr,
    rows,
    CACHE: gl.constexpr,
):
    """Store K step ``step`` (FP32 ``[waves, 32, 16]``) of a B operand whose
    waves hold ``KW`` columns each from ``column0``, as rows of a ``[*,
    HIDDEN]`` FP32 tensor (``CACHE``)."""
    layout: gl.constexpr = values.type.layout
    WAVES: gl.constexpr = gl.num_warps()
    wave = gl.arange(0, WAVES, gl.SliceLayout(1, gl.SliceLayout(2, layout)))
    k = gl.arange(0, values.shape[1], gl.SliceLayout(0, gl.SliceLayout(2, layout)))
    t = gl.arange(0, 16, gl.SliceLayout(0, gl.SliceLayout(1, layout)))
    offsets = (
        t[None, None, :] * _HIDDEN
        + column0
        + (wave * KW + step * values.shape[1])[:, None, None]
        + k[None, :, None]
    )
    mask = (
        (wave >= 0)[:, None, None] & (k >= 0)[None, :, None] & (t < rows)[None, None, :]
    )
    gl.amd.cdna4.buffer_store(values, ptr, offsets, mask=mask, cache=CACHE)


@gluon.jit
def _publish_hprime(launch, part, exchange_ptr):
    """After the steps wrote h' of K part ``part`` through to ``hprime
    [MAX_ROWS, HIDDEN]`` (:func:`_x_step`), the part's flag line. Its
    readers (UNP, QKV, BND2) sit on any XCD and read it ``.cv``."""
    release_drain()
    gl.barrier()
    _publish_line(launch, exchange_ptr + _EX_HP_TAGS + part * 16)


@gluon.jit
def _store_s(s, token, part, s_rows_ptr):
    """FIN's routed sum of (token, part), FP32 [512], as BF16 (the
    post-expert projection's input) to ``s_rows [MAX_ROWS, 4096]``, written
    through. :func:`_publish_s` then publishes it."""
    columns = gl.arange(0, 512, layout=s.type.layout)
    gl.amd.cdna4.buffer_store(
        s.to(gl.bfloat16), s_rows_ptr, token * _POOLED + part * 512 + columns,
        cache=".wt",
    )  # fmt: skip


@gluon.jit
def _publish_s(launch, first_token, ROWS: gl.constexpr, part, rows, exchange_ptr):
    """After this workgroup's :func:`_store_s` of tokens ``first_token + 8 r``
    (r < ROWS): one drain for them all, then their tags in the S tags
    ``[part][row]`` (also row ``token + 8`` when the launch has at most 8
    rows: UNP polls 16-row lines)."""
    release_drain()
    gl.barrier()
    lanes_l: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    lane = gl.arange(0, 64 * gl.num_warps(), lanes_l)
    tag = gl.full([64 * gl.num_warps()], 0, gl.int32, lanes_l) + launch.tag
    for r in gl.static_range(ROWS):
        tags = exchange_ptr + _EX_S_TAGS + part * _MAX_ROWS + first_token + 8 * r
        gl.amd.cdna4.buffer_store(tag, tags, lane * 0, mask=lane == 0)
        gl.amd.cdna4.buffer_store(
            tag, tags, lane * 0 + 8, mask=(lane == 0) & (rows <= 8)
        )


@gluon.jit
def _wait_b_rows(
    ptr,
    rows,
    K: gl.constexpr,
    ROW_STRIDE: gl.constexpr,
    launch,
    SITE: gl.constexpr,
):
    """Wait for ``rows`` (<= 16) rows of ``K`` BF16 Lamport words at ``ptr``
    (row stride ``ROW_STRIDE``, one source, written by another rank) and
    return them as ``[waves, K / waves, 16 tokens]``, each thread a contiguous
    K run of one token (for :func:`_x_state`)."""
    WAVES: gl.constexpr = gl.num_warps()
    KW: gl.constexpr = K // WAVES
    PACK: gl.constexpr = 16 * K // (64 * WAVES)
    packs: gl.constexpr = gl.BlockedLayout([1, PACK], [64, 1], [WAVES, 1], [1, 0])
    pack = gl.arange(0, 64 * WAVES, layout=gl.SliceLayout(1, packs))
    lane = gl.arange(0, PACK, layout=gl.SliceLayout(0, packs))
    row = pack * PACK // K
    offsets = (row * ROW_STRIDE + pack * PACK % K)[:, None] + lane[None, :]
    rows_in = iris_wait_lamport_slots(
        ptr, offsets, row < rows, launch, iris_fail_stop, SITE, 1, 2, 8, packs
    )[0]
    return gl.permute(gl.reshape(rows_in, [16, WAVES, KW]), [1, 2, 0])


# ---------------------------------------------------------------------------
# Tail: post-expert senders, sum reducers, QKV, HEAD, BND2 (the last fused
# layer's boundary)
# ---------------------------------------------------------------------------


@gluon.jit
def _unp(
    launch,
    view,
    sender,
    rows,
    w,
    s_rows_ptr,
    hprime_ptr,
    exchange_ptr,
    alpha2_ptr,
    beta2_ptr,
    post2_ptr,
    stamps_ptr,
    RANK: gl.constexpr,
    PEER_CACHE: gl.constexpr,
    STAMPS: gl.constexpr,
):
    """UNP ``sender`` (0-63 of this rank, 8 per XCD): its 16 hidden columns
    of the post-expert projection, bitwise the column GEMM's, relayed across
    the waves (``mm_a16w16_column_relay_gfx950``) from resident weights
    ``w``. It waits for every FIN part's S rows (written through, flag lines; read ``.cv``), pushes
    the BF16 columns to every rank (H6) and their four norm-sum partials to
    this XCD's reducer."""
    WAVES: gl.constexpr = gl.num_warps()
    for p in gl.static_range(_KPARTS):
        _wait_line(launch, exchange_ptr + _EX_S_TAGS + p * _MAX_ROWS, _SITE_S)
    _stamp(stamps_ptr, launch.cta, _STAMP_UNP_S, STAMPS)
    mfma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True,
        warps_per_cta=[WAVES, 1, 1],
    )  # fmt: skip
    a_l: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma, k_width=8)
    wave = gl.arange(0, WAVES, gl.SliceLayout(1, gl.SliceLayout(2, a_l)))
    token = gl.arange(0, 16, gl.SliceLayout(0, gl.SliceLayout(2, a_l)))
    k = gl.arange(0, _POOLED // WAVES, gl.SliceLayout(0, gl.SliceLayout(1, a_l)))
    s = gl.amd.cdna4.buffer_load(
        s_rows_ptr,
        token[None, :, None] * _POOLED
        + (wave * (_POOLED // WAVES))[:, None, None]
        + k[None, None, :],
        mask=(token < rows)[None, :, None] & (k >= 0)[None, None, :],
        other=0.0,
        cache=".cv",
    )
    relay = gl.allocate_shared_memory(
        gl.float32, [WAVES, 16, 16], gl.SwizzledSharedLayout(1, 1, 1, [2, 1, 0])
    )
    relay_count = gl.allocate_shared_memory(
        gl.int32, [1], gl.SwizzledSharedLayout(1, 1, 1, [0])
    )
    relay_count.store(
        gl.zeros([1], gl.int32, gl.BlockedLayout([1], [64], [WAVES], [0]))
    )
    gl.barrier()
    out_l: gl.constexpr = gl.BlockedLayout([1, 4], [16, 4], [WAVES, 1], [1, 0])
    b = mm_a16w16_column_relay_gfx950(s, w, relay, relay_count, out_l).to(gl.bfloat16)
    h6_sink(
        view, launch, b, sender * _UNP_COLUMNS, rows, 16, _UNP_COLUMNS, RANK, PEER_CACHE
    )
    _stamp(stamps_ptr, launch.cta, _STAMP_H6, STAMPS)
    # The four norm-sum partials of these columns (h' from part RANK).
    _wait_line(launch, exchange_ptr + _EX_HP_TAGS + RANK * 16, _SITE_HPRIME)
    t = gl.minimum(gl.arange(0, 16, gl.SliceLayout(1, out_l)), rows - 1)
    c = (
        RANK * (_HIDDEN // _KPARTS)
        + sender * _UNP_COLUMNS
        + gl.arange(0, _UNP_COLUMNS, gl.SliceLayout(0, out_l))
    )
    hp = gl.amd.cdna4.buffer_load(
        hprime_ptr, t[:, None] * _HIDDEN + c[None, :], cache=".cv"
    )
    alpha = gl.load(alpha2_ptr + c)[None, :]
    beta = gl.load(beta2_ptr + c)[None, :]
    post = gl.load(post2_ptr + c)[None, :]
    terms = deferred_norm_terms_gfx950(b, hp, alpha, beta, post)
    unp_sums_publish(view, launch, gl.sum(terms, 1), sender % 8, rows)


@gluon.jit
def _ffn_state(
    launch,
    view,
    part,
    task,
    rows,
    hprime_ptr,
    exchange_ptr,
    alpha2_ptr,
    beta2_ptr,
    post2_ptr,
    pre2_ptr,
    residual_ptr,
    store_h,
    POST_EPS: gl.constexpr,
    PRE_EPS: gl.constexpr,
):
    """QKV K part ``part``'s (512 columns) x'' state (:func:`_x_step`):
    the deferred FFN-boundary build from H6 (columns of rank ``part // 2``),
    h' and the 64 reducers' sums; with ``store_h`` the steps store h'' to
    ``residual`` for the next layer (plain: the next launch reads it)."""
    vectors = _vectors_to_shared(
        task, alpha2_ptr, beta2_ptr, post2_ptr, pre2_ptr, _QKV_PARTS, _QKV_GROUPS
    )
    inv1, inv2 = deferred_norm_scales_gfx950(
        _norm_sums(launch, view.h6_sums, rows, _SITE_H6_SUMS),
        _HIDDEN,
        POST_EPS,
        PRE_EPS,
    )
    PART: gl.constexpr = _HIDDEN // _QKV_PARTS
    b = _wait_b_rows(
        view.h6 + launch.copy * (2 * _H6_WORDS) + (part // 2) * _H6_SLOT
        + (part % 2) * PART,
        rows, PART, _HIDDEN // _KPARTS, launch, _SITE_H6,
    )  # fmt: skip
    _wait_line(launch, exchange_ptr + _EX_HP_TAGS + (part // 2) * 16, _SITE_HPRIME)
    h = gemv_a16w16_splitk_x_steps_gfx950(
        hprime_ptr, _HIDDEN, task, rows, _HIDDEN, _QKV_PARTS, _QKV_GROUPS, ".cv"
    )
    return _x_state(b, h, vectors, inv1, inv2, part * PART, rows, residual_ptr, store_h)


@gluon.jit
def _head_rows(launch, qkv_parts, column0, rows):
    """One head's 16 x 64 rows from the 16 QKV K parts' tagged
    partials, summed in part order and rounded once to BF16 (as FP32), in
    the prologue's rows layout."""
    rows_l: gl.constexpr = gqa_prologue_rows_layout(gl.num_warps())
    flat_l: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    index = gl.arange(0, 16 * _HEAD_DIM, flat_l)
    token = index // _HEAD_DIM
    sums = gemv_a16w16_splitk_sum_gfx950(
        qkv_parts, launch.tag, token * _QKV_WIDTH + column0 + index % _HEAD_DIM,
        token < rows, _PART_ROWS, _QKV_WIDTH, _QKV_PARTS, launch, iris_fail_stop,
        _SITE_QKV,
    )  # fmt: skip
    return gl.convert_layout(
        gl.reshape(sums.to(gl.bfloat16).to(gl.float32), [16, _HEAD_DIM]), rows_l
    )


@gluon.jit
def _head(
    launch,
    head,
    rows,
    qkv_parts,
    q_out_ptr,
    gate_out_ptr,
    q_norm_ptr,
    k_norm_ptr,
    cos_sin_ptr,
    cos_sin_stride,
    positions_ptr,
    next_k_cache_ptr,
    next_v_cache_ptr,
    slots_ptr,
    num_slots,
    k_inv_scale,
    v_inv_scale,
    ROTARY: gl.constexpr,
    QK_EPS: gl.constexpr,
    QK_OFFSET: gl.constexpr,
    KV_SCALED: gl.constexpr,
):
    """HEAD task ``head``: q heads (prologue to the FP8 query), (K, V) head
    pairs (prologue into the next layer's cache), then the output-gate
    heads (BF16). The operands that do not wait load first. Rows past
    ``num_slots`` (graph padding) write no cache row."""
    rows_l: gl.constexpr = gqa_prologue_rows_layout(gl.num_warps())
    tokens = gl.arange(0, 16, gl.SliceLayout(1, rows_l)).to(gl.int64)
    row_mask = tokens < rows
    kv_mask = row_mask & (tokens < num_slots)
    cos, sin = gqa_prologue_rope_gfx950(
        cos_sin_ptr, cos_sin_stride, positions_ptr, tokens, row_mask, _HEAD_DIM // 2
    )
    slot = gl.load(slots_ptr + tokens, mask=kv_mask, other=0).to(gl.int64)
    if head < _Q_HEADS:
        q = _head_rows(launch, qkv_parts, head * _HEAD_DIM, rows)
        gqa_prologue_query_gfx950(
            q, q_out_ptr + tokens * _Q_WIDTH + head * _HEAD_DIM, q_norm_ptr, cos, sin,
            row_mask, ROTARY, QK_EPS, QK_OFFSET,
        )  # fmt: skip
    elif head < _Q_HEADS + _KV_HEADS:
        kv = head - _Q_HEADS
        k_rows = _head_rows(launch, qkv_parts, (_Q_HEADS + kv) * _HEAD_DIM, rows)
        gqa_prologue_key_gfx950(
            k_rows, next_k_cache_ptr + (slot * _KV_HEADS + kv) * _HEAD_DIM, k_norm_ptr,
            cos, sin, kv_mask, k_inv_scale, ROTARY, QK_EPS, QK_OFFSET, KV_SCALED,
        )  # fmt: skip
        v_rows = _head_rows(
            launch, qkv_parts, (_Q_HEADS + _KV_HEADS + kv) * _HEAD_DIM, rows
        )
        gqa_prologue_value_gfx950(
            v_rows, next_v_cache_ptr + (slot * _KV_HEADS + kv) * _HEAD_DIM, kv_mask,
            v_inv_scale, KV_SCALED,
        )  # fmt: skip
    else:
        g = head - _Q_HEADS - _KV_HEADS
        gate = _head_rows(
            launch, qkv_parts, (_Q_HEADS + 2 * _KV_HEADS + g) * _HEAD_DIM, rows
        )
        d = gl.arange(0, _HEAD_DIM, gl.SliceLayout(0, rows_l))
        gl.store(
            gate_out_ptr + tokens[:, None] * _Q_WIDTH + g * _HEAD_DIM + d[None, :],
            gate.to(gl.bfloat16),
            mask=row_mask[:, None] & (d >= 0)[None, :],
        )


@gluon.jit
def _bnd2(
    launch,
    view,
    token,
    part,
    rows,
    hprime_ptr,
    exchange_ptr,
    residual_ptr,
    normed_ptr,
    alpha2_ptr,
    beta2_ptr,
    post2_ptr,
    pre2_ptr,
    POST_EPS: gl.constexpr,
    PRE_EPS: gl.constexpr,
):
    """BND2 of (token, part): the last fused layer's FFN boundary in the
    regular layout. h'' and the next layer's normed input (BF16) of the
    token's 1024 columns of rank ``part``, from H6, h' and the reducers'
    sums; h'' goes to the residual in place (plain: the next kernel reads)."""
    WAVES: gl.constexpr = gl.num_warps()
    COLUMNS: gl.constexpr = _HIDDEN // _KPARTS
    sums = _norm_sums(launch, view.h6_sums, rows, _SITE_H6_SUMS)
    inv1, inv2 = deferred_norm_scales_gfx950(sums, _HIDDEN, POST_EPS, PRE_EPS)
    # The row's 1024 columns as 512 two-column packs, one per lane.
    packs: gl.constexpr = gl.BlockedLayout([1, 2], [64, 1], [WAVES, 1], [1, 0])
    pack = gl.arange(0, COLUMNS // 2, gl.SliceLayout(1, packs))
    pair = gl.arange(0, 2, gl.SliceLayout(0, packs))
    local = pack[:, None] * 2 + pair[None, :]
    words = iris_wait_lamport_slots(
        view.h6 + launch.copy * (2 * _H6_WORDS) + part * _H6_SLOT,
        token * COLUMNS + local, pack >= 0, launch, iris_fail_stop, _SITE_H6, 1, 2,
        8, packs,
    )[0]  # fmt: skip
    _wait_line(launch, exchange_ptr + _EX_HP_TAGS + part * 16, _SITE_HPRIME)
    columns = part * COLUMNS + local
    hp = gl.amd.cdna4.buffer_load(hprime_ptr, token * _HIDDEN + columns, cache=".cv")
    alpha = gl.load(alpha2_ptr + columns)
    beta = gl.load(beta2_ptr + columns)
    post = gl.load(post2_ptr + columns)
    pre = gl.load(pre2_ptr + columns)
    pick = gl.arange(0, 16, inv1.type.layout) == token
    i1 = gl.sum(gl.where(pick, inv1, 0.0), 0)
    i2 = gl.sum(gl.where(pick, inv2, 0.0), 0)
    h2, x2 = deferred_norm_rows_gfx950(words, hp, alpha, beta, post, pre, i1, i2)
    gl.amd.cdna4.buffer_store(h2, residual_ptr, token * _HIDDEN + columns)
    gl.amd.cdna4.buffer_store(x2.to(gl.bfloat16), normed_ptr, token * _HIDDEN + columns)


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------


@gluon.jit(
    do_not_specialize=("deadline_ticks", "label", "rank", "gate_stride", "num_slots")
)
def gluon_decoder_layer_muse_spark_gfx950(
    # Launch state.
    epochs,
    claims,
    errors,
    arena,
    exchange,
    xq_ptr,
    xs_ptr,
    deadline_ticks,
    label,
    # Iris areas (fused_layer.MuseNativeIrisState.view_args()).
    heaps,
    rank,
    box,
    h1,
    h1_sums,
    h3,
    h4_own,
    routes,
    h6,
    red_in,
    h6_sums,
    # Attention.
    q_ptr,
    gate_ptr,
    gate_stride,
    k_cache_ptr,
    v_cache_ptr,
    page_table_ptr: tl.const,
    cache_seqlens_ptr,
    schedule_ptr: tl.const,
    mid_o_ptr,
    mid_lse_ptr,
    att_records,
    # o_proj, residual and the post-attention boundary.
    o_proj_ptr,
    residual_ptr,
    alpha_ptr,
    beta_ptr,
    post_attn_norm_ptr,
    post_attn_bf16_ptr,
    pre_ffn_bf16_ptr,
    # Router, pooler.
    router_hi_ptr,
    router_lo_ptr,
    router_multiplier,
    router_bias_ptr,
    logits_parts,
    pooler_ptr,
    pooled_parts,
    pooled_norm_ptr,
    # Experts.
    w13,
    s13,
    s13_lanes,
    w2,
    s2,
    mids,
    mids_s,
    moe_tags,
    stride_w13,
    stride_s13,
    stride_s13lin,
    stride_s13nb,
    stride_w2,
    stride_s2,
    stride_s2lin,
    stride_s2nb,
    # Routed sum and post-expert projection.
    route_norm_ptr,
    s_rows,
    hprime,
    post_expert_ptr,
    # FFN boundary (gate coefficients, the post-norm and the next input norm
    # weights with their offsets, BF16).
    alpha2_ptr,
    beta2_ptr,
    post2_bf16_ptr,
    pre2_bf16_ptr,
    # The next layer's QKV projection and prologue.
    qkv_w_ptr,
    qkv_parts,
    q_out_ptr,
    gate_out_ptr,
    q_norm_ptr,
    k_norm_ptr,
    cos_sin_ptr,
    cos_sin_stride,
    positions_ptr,
    next_k_cache_ptr,
    next_v_cache_ptr,
    slots_ptr,
    num_slots,
    k_inv_scale,
    v_inv_scale,
    # The last fused layer's boundary output (next layer regular).
    normed_out_ptr,
    # Clock stamps per workgroup (``STAMPS``).
    stamps_ptr,
    # Plan key.
    REQUESTS: gl.constexpr,
    SLIDING: gl.constexpr,
    NEXT_KIND: gl.constexpr,
    RANK: gl.constexpr,
    PEER_CACHE: gl.constexpr,
    # Attention constants.
    SM_SCALE: gl.constexpr,
    PAGE_TABLE_STRIDE: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    WINDOW_LEFT: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    GATE_EPS: gl.constexpr,
    # Norms.
    POST_ATTN_EPS: gl.constexpr,
    POST_ATTN_OFFSET: gl.constexpr,
    POST_ATTN_HAS_WEIGHT: gl.constexpr,
    PRE_FFN_EPS: gl.constexpr,
    POOLED_EPS: gl.constexpr,
    POOLED_OFFSET: gl.constexpr,
    ROUTE_EPS: gl.constexpr,
    ROUTE_OFFSET: gl.constexpr,
    ROUTED_SCALING_FACTOR: gl.constexpr,
    POST2_EPS: gl.constexpr,
    PRE2_EPS: gl.constexpr,
    QK_EPS: gl.constexpr,
    QK_OFFSET: gl.constexpr,
    KV_SCALED: gl.constexpr,
    NEXT_ROTARY: gl.constexpr,
    QKV_WINDOW: gl.constexpr,
    ATT_MERGE: gl.constexpr,
    STAMPS: gl.constexpr,
    # MoE plan fields (native_stream.run_moe).
    N_PHYS1: gl.constexpr,
    N_PHYS2: gl.constexpr,
    RING: gl.constexpr,
    QUAD_MAP: gl.constexpr,
    DN_ORDER: gl.constexpr,
    ADDR_SOFFSET: gl.constexpr,
    SCALE_LANES: gl.constexpr,
    TAIL_NUM: gl.constexpr,
    TAIL_DEN: gl.constexpr,
    SWIGLU_ALPHA: gl.constexpr,
    SWIGLU_LIMIT: gl.constexpr,
    SWIGLU_BETA: gl.constexpr,
):
    """One fused Muse Spark decoder layer (see the module docstring)."""
    ROWS: gl.constexpr = REQUESTS * 8
    rows = gl.to_tensor(ROWS)
    view = MuseNativeView(
        heaps, rank, box, h1, h1_sums, h3, h4_own, routes, h6, red_in, h6_sums
    )
    # Sliding layers: 8 splits per block; full layers: the balanced decode's
    # items (its schedule picks the splits).
    cfg: gl.constexpr = AttentionConfig(
        SM_SCALE, PAGE_TABLE_STRIDE, PAGE_SIZE, 8 if SLIDING else 1, 8, 16, 2, 64,
        _ATT_BLOCK_M, PAGE_SIZE, _NUM_WARPS, SLIDING, WINDOW_LEFT if SLIDING else -1,
        True, k_cache_ptr.dtype.element_ty, InputStrides(_Q_WIDTH, 64, 1),
    )  # fmt: skip

    _reserve_lds()
    # Entry: the two counters, then the role's first weights, then the check.
    entry = enter_issue(epochs, claims)
    xcd = entry.xcd
    slot = entry.slot
    _stamp(stamps_ptr, entry.cta, _STAMP_ENTRY, STAMPS)
    if slot < 16:
        # SND -> OWN1 on slot 0 -> RTR.
        sender = xcd * 16 + slot
        task = xcd * _ROUTER_GROUPS + slot
        if SLIDING:
            # o_proj goes out before the entry check (no attention here),
            # paced, and lands long before the box.
            o_tiles = _sender_weights_paced(o_proj_ptr, sender)
            launch = enter_check(entry, errors, arena, deadline_ticks, label, fail_stop)
            _stamp(stamps_ptr, launch.cta, _STAMP_CHECKED, STAMPS)
            records = _launch_records(att_records, launch)
            _send(launch, view, sender, o_tiles, rows, RANK, PEER_CACHE)
            if slot == 0:
                own1(
                    view, launch, xcd, rows, residual_ptr, post_attn_norm_ptr,
                    alpha_ptr, beta_ptr, iris_fail_stop, RANK, POST_ATTN_OFFSET,
                    POST_ATTN_HAS_WEIGHT, PEER_CACHE,
                )  # fmt: skip
        else:
            # Full layers: o_proj after this workgroup's attention items; the
            # box arrives soon after, and its read queues behind whatever
            # this workgroup has in flight.
            launch = enter_check(entry, errors, arena, deadline_ticks, label, fail_stop)
            _stamp(stamps_ptr, launch.cta, _STAMP_CHECKED, STAMPS)
            records = _launch_records(att_records, launch)
            o_tiles = _full_items(
                launch, cfg, q_ptr, cache_seqlens_ptr, schedule_ptr, page_table_ptr,
                k_cache_ptr, v_cache_ptr, mid_o_ptr, mid_lse_ptr, records,
                REQUESTS, NUM_STAGES, _sender_weights, (o_proj_ptr, sender),
            )  # fmt: skip
            _send(launch, view, sender, o_tiles, rows, RANK, PEER_CACHE)
            if slot == 0:
                own1(
                    view, launch, xcd, rows, residual_ptr, post_attn_norm_ptr,
                    alpha_ptr, beta_ptr, iris_fail_stop, RANK, POST_ATTN_OFFSET,
                    POST_ATTN_HAS_WEIGHT, PEER_CACHE,
                )  # fmt: skip
        # The router's tiles after the push and OWN1's reduce-scatter wait,
        # which they would hold up; needed only after the all-gather.
        hi, lo = _router_weights(router_hi_ptr, router_lo_ptr, task)
        router_x = _kpart_state(
            launch, view, xcd, task, rows, residual_ptr, alpha_ptr, beta_ptr,
            post_attn_bf16_ptr, pre_ffn_bf16_ptr, hprime, slot == 1, _ROUTER_GROUPS,
            POST_ATTN_EPS, PRE_FFN_EPS,
        )  # fmt: skip
        logits = gemv_a16w16_splitk_hi_lo_sums_steps_gfx950(
            hi, lo, _x_step, router_x, _sums_layout(16, 2)
        )
        gemv_a16w16_splitk_publish_gfx950(
            logits,
            logits_parts,
            launch.tag,
            slot * 16,
            xcd,
            _PART_ROWS,
            _EXPERTS,
            ".wt",
        )
        if slot == 1:
            # h' is read much later (UNP, QKV): its flag follows the logits.
            _publish_hprime(launch, xcd, exchange)
    elif slot < 24:
        # POOL; slot 16 + T on XCD T reduces tile T and pushes H3.
        tile = slot - 16
        task = xcd * _POOL_TILES + tile
        if SLIDING:
            p_tiles = _pooler_weights_sliding(pooler_ptr, task)
            launch = enter_check(entry, errors, arena, deadline_ticks, label, fail_stop)
            _stamp(stamps_ptr, launch.cta, _STAMP_CHECKED, STAMPS)
            records = _launch_records(att_records, launch)
        else:
            launch = enter_check(entry, errors, arena, deadline_ticks, label, fail_stop)
            _stamp(stamps_ptr, launch.cta, _STAMP_CHECKED, STAMPS)
            records = _launch_records(att_records, launch)
            p_tiles = _full_items(
                launch, cfg, q_ptr, cache_seqlens_ptr, schedule_ptr, page_table_ptr,
                k_cache_ptr, v_cache_ptr, mid_o_ptr, mid_lse_ptr, records,
                REQUESTS, NUM_STAGES, _pooler_weights, (pooler_ptr, task),
            )  # fmt: skip
        pool_x = _kpart_state(
            launch, view, xcd, task, rows, residual_ptr, alpha_ptr, beta_ptr,
            post_attn_bf16_ptr, pre_ffn_bf16_ptr, hprime, False, _POOL_TILES,
            POST_ATTN_EPS, PRE_FFN_EPS,
        )  # fmt: skip
        pooled = gemv_a16w16_splitk_wave_sums_steps_gfx950(
            p_tiles, _x_step_bf16, pool_x, _sums_layout(_POOL_TILE, 2)
        )
        gemv_a16w16_splitk_publish_gfx950(
            pooled, pooled_parts, launch.tag, tile * _POOL_TILE, xcd, _PART_ROWS,
            _POOLED_LOCAL, ".wt",
        )  # fmt: skip
        if xcd == tile:
            red_l: gl.constexpr = gl.BlockedLayout(
                [1, 2], [16, 4], [_NUM_WARPS, 1], [1, 0]
            )
            t = gl.arange(0, 16, gl.SliceLayout(1, red_l))
            c = gl.arange(0, _POOL_TILE, gl.SliceLayout(0, red_l))
            offsets = t[:, None] * _POOLED_LOCAL + tile * _POOL_TILE + c[None, :]
            live = (t < rows)[:, None] & (c >= 0)[None, :]
            reduced = gemv_a16w16_splitk_sum_gfx950(
                pooled_parts, launch.tag, gl.reshape(offsets, [16 * _POOL_TILE]),
                gl.reshape(live, [16 * _POOL_TILE]), _PART_ROWS, _POOLED_LOCAL, _KPARTS,
                launch, iris_fail_stop, _SITE_POOL,
            )  # fmt: skip
            h3_sink(
                view, launch, gl.reshape(reduced, [16, _POOL_TILE]), tile, rows, 16,
                RANK, PEER_CACHE,
            )  # fmt: skip
    else:
        launch = enter_check(entry, errors, arena, deadline_ticks, label, fail_stop)
        _stamp(stamps_ptr, launch.cta, _STAMP_CHECKED, STAMPS)
        records = _launch_records(att_records, launch)
        if SLIDING:
            if xcd < 2 * REQUESTS:
                # Split (slot - 24) of block xcd.
                _attention_split(
                    launch, xcd, slot - _ATT_FIRST_SLOT, view, q_ptr, gate_ptr,
                    gate_stride, k_cache_ptr, v_cache_ptr, page_table_ptr,
                    cache_seqlens_ptr, mid_o_ptr, mid_lse_ptr, records, cfg,
                    GATE_EPS, NUM_STAGES, ATT_MERGE,
                )  # fmt: skip
        else:
            _full_items(
                launch, cfg, q_ptr, cache_seqlens_ptr, schedule_ptr, page_table_ptr,
                k_cache_ptr, v_cache_ptr, mid_o_ptr, mid_lse_ptr, records,
                REQUESTS, NUM_STAGES, _no_weights, (),
            )  # fmt: skip
            # Merge tasks on slots 24.. (no weights in flight), 8 per XCD row.
            merge = (slot - _ATT_FIRST_SLOT) * 8 + xcd
            if merge < REQUESTS * 2 * _FULL_MERGE_TASKS:
                _full_merge(
                    launch, merge, view, cfg, schedule_ptr, mid_o_ptr, mid_lse_ptr,
                    records, q_ptr, gate_ptr, gate_stride, GATE_EPS,
                )  # fmt: skip
        if (slot == _ATT_FIRST_SLOT) & (xcd >= 8 - REQUESTS):
            _topk(
                launch, 7 - xcd, rows, logits_parts, router_multiplier, router_bias_ptr,
                exchange, ROUTED_SCALING_FACTOR,
            )  # fmt: skip
        if (slot >= 25) & (slot < 25 + REQUESTS):
            _xn(
                launch, view, xcd + 8 * (slot - 25), rows, pooled_norm_ptr, exchange,
                xq_ptr, xs_ptr, RANK, POOLED_EPS, POOLED_OFFSET,
            )  # fmt: skip

    _stamp(stamps_ptr, launch.cta, _STAMP_FRONT, STAMPS)
    # MoE: every workgroup, once the ids and then the xq rows land.
    if not poll_tags16(exchange + _EX_IDS_TAGS, launch.tag, launch, True, True):
        iris_fail_stop(launch, _SITE_IDS, 0)
    _stamp(stamps_ptr, launch.cta, _STAMP_IDS, STAMPS)
    bufs = MoeBuffers(
        xq_ptr,
        xs_ptr,
        w13, s13, s13_lanes, w2, s2, mids, mids_s, moe_tags, gl.to_tensor(1),
        gl.to_tensor(_XS_MBLOCK_STRIDE), stride_w13,
        stride_s13, stride_s13lin, stride_s13nb, stride_w2, stride_s2,
        stride_s2lin, stride_s2nb,
    )  # fmt: skip
    handles = prefetch_moe(
        launch, exchange + _EX_IDS, bufs, ROWS, _TOPK, N_PHYS1, RING, QUAD_MAP,
        TAIL_NUM, TAIL_DEN, ADDR_SOFFSET, SCALE_LANES,
    )  # fmt: skip
    if not poll_tags16(exchange + _EX_XQ_TAGS, launch.tag, launch, True, True):
        iris_fail_stop(launch, _SITE_XQ, 0)
    _stamp(stamps_ptr, launch.cta, _STAMP_XQ, STAMPS)
    run_moe(
        launch, handles, bufs, h4_sink, H4SinkArgs(view, launch, RANK), ROWS, _TOPK, N_PHYS1,
        N_PHYS2, RING, QUAD_MAP, DN_ORDER, ADDR_SOFFSET, True, SCALE_LANES,
        SWIGLU_ALPHA, SWIGLU_LIMIT, SWIGLU_BETA,
    )  # fmt: skip

    _stamp(stamps_ptr, launch.cta, _STAMP_MOE, STAMPS)
    # Tail. Weights first, at this workgroup's MoE end: the post-expert
    # relay's (UNP) and the next layer's QKV task (paced so the stream stays
    # under the fabric's latency knee while the tail's exchanges run).
    is_tail = (slot == 0) | ((slot >= 24) & (slot < 31))
    is_unp = (slot >= 16) & (slot < 24)
    index = gl.where(slot == 0, 0, slot - 23)
    if NEXT_KIND != 2:
        part = 2 * xcd + slot % 2
        group = slot // 2
        task = part * _QKV_GROUPS + group
    if is_unp:
        # UNP, then this XCD's sum reducer on slot 16. The relay
        # weights go out first: the relay needs them before the QKV task.
        sender = xcd * 8 + slot - 16
        w = mm_a16w16_column_relay_weights_gfx950(
            post_expert_ptr, RANK * (_HIDDEN // _KPARTS) + sender * _UNP_COLUMNS,
            _POOLED,
        )  # fmt: skip
        if NEXT_KIND != 2:
            # The QKV task's first window of tiles only: the rest would hold
            # registers through the relay (the same tiles in the same order,
            # issued in two parts).
            QKV_FIRST: gl.constexpr = min(QKV_WINDOW, _QKV_TASK_ROWS // 16)
            qkv_first = gemv_a16w16_splitk_weight_step_tiles_gfx950(
                qkv_w_ptr, task, _HIDDEN, _QKV_TASK_ROWS, _QKV_PARTS, _QKV_GROUPS,
                QKV_WINDOW, 0, QKV_FIRST,
            )  # fmt: skip
        _unp(
            launch, view, sender, rows, w, s_rows, hprime, exchange, alpha2_ptr,
            beta2_ptr, post2_bf16_ptr, stamps_ptr, RANK, PEER_CACHE, STAMPS,
        )  # fmt: skip
        if NEXT_KIND != 2:
            # The rest once the relay's weights are dead and UNP's pushes and
            # polls are issued: loads complete in issue order, so earlier
            # they would hold up those hand-offs. The reducer below polls
            # with scalar loads, which do not wait for them.
            qkv_tiles = qkv_first + gemv_a16w16_splitk_weight_step_tiles_gfx950(
                qkv_w_ptr, task, _HIDDEN, _QKV_TASK_ROWS, _QKV_PARTS, _QKV_GROUPS,
                QKV_WINDOW, QKV_FIRST, _QKV_TASK_ROWS // 16 - QKV_FIRST,
            )  # fmt: skip
        if slot == 16:
            red_sums(view, launch, rows, iris_fail_stop, RANK, PEER_CACHE)
            _stamp(stamps_ptr, launch.cta, _STAMP_RED, STAMPS)
    elif is_tail:
        if NEXT_KIND != 2:
            # Tail-role workgroups finish their MoE share early, inside the
            # MoE's full-rate window: their task goes out unpaced.
            qkv_tiles = gemv_a16w16_splitk_weight_steps_gfx950(
                qkv_w_ptr, task, _HIDDEN, _QKV_TASK_ROWS, _QKV_PARTS, _QKV_GROUPS,
                _QKV_TASK_ROWS // 16,
            )  # fmt: skip
        # OWN4 of every row's pick xcd first, so the second row's H5 push
        # does not wait for the first row's FIN; then FIN of part xcd per
        # row, and one drain and tag publish for all the rows' S.
        for r in gl.static_range(REQUESTS):
            own4_complete(
                view, launch, (index + 8 * r) * _TOPK + xcd, route_norm_ptr,
                iris_fail_stop, RANK, ROUTE_OFFSET, PEER_CACHE,
            )  # fmt: skip
            _stamp(stamps_ptr, launch.cta, _STAMP_OWN4, STAMPS)
        for r in gl.static_range(REQUESTS):
            s = fin(
                view, launch, index + 8 * r, xcd,
                (exchange + _EX_WEIGHTS).to(gl.pointer_type(gl.float32), bitcast=True),
                route_norm_ptr, iris_fail_stop, ROUTE_EPS, ROUTE_OFFSET,
            )  # fmt: skip
            _stamp(stamps_ptr, launch.cta, _STAMP_FIN, STAMPS)
            _store_s(s, index + 8 * r, xcd, s_rows)
        _publish_s(launch, index, REQUESTS, xcd, rows, exchange)
        _stamp(stamps_ptr, launch.cta, _STAMP_S, STAMPS)
    else:
        if NEXT_KIND != 2:
            # The other workgroups stream their task while the tail's
            # latency-bound exchanges run: paced below the fabric's knee.
            qkv_tiles = gemv_a16w16_splitk_weight_steps_gfx950(
                qkv_w_ptr, task, _HIDDEN, _QKV_TASK_ROWS, _QKV_PARTS, _QKV_GROUPS,
                QKV_WINDOW,
            )  # fmt: skip
    _stamp(stamps_ptr, launch.cta, _STAMP_TAIL, STAMPS)
    if NEXT_KIND != 2:
        # QKV: x'' of K part ``part``, one K step at a time; group 0
        # stores h'' for the next layer.
        qkv_x = _ffn_state(
            launch, view, part, task, rows, hprime, exchange, alpha2_ptr, beta2_ptr,
            post2_bf16_ptr, pre2_bf16_ptr, residual_ptr, group == 0, POST2_EPS,
            PRE2_EPS,
        )  # fmt: skip
        _stamp(stamps_ptr, launch.cta, _STAMP_QKV_X, STAMPS)
        gemv_a16w16_splitk_finish_steps_gfx950(
            qkv_tiles, _x_step_bf16, qkv_x, qkv_parts, launch.tag, task, _PART_ROWS,
            _QKV_WIDTH, _QKV_TASK_ROWS, _QKV_GROUPS, ".wt",
        )  # fmt: skip
        _stamp(stamps_ptr, launch.cta, _STAMP_QKV, STAMPS)
        head = index * 8 + xcd
        if is_tail & (head < _HEAD_TASKS):
            _head(
                launch, head, rows, qkv_parts, q_out_ptr, gate_out_ptr, q_norm_ptr,
                k_norm_ptr, cos_sin_ptr, cos_sin_stride, positions_ptr,
                next_k_cache_ptr, next_v_cache_ptr, slots_ptr, num_slots, k_inv_scale,
                v_inv_scale, NEXT_ROTARY, QK_EPS, QK_OFFSET, KV_SCALED,
            )  # fmt: skip
    elif is_tail:
        # BND2: the regular boundary of (token, part xcd).
        for r in gl.static_range(REQUESTS):
            _bnd2(
                launch, view, index + 8 * r, xcd, rows, hprime, exchange,
                residual_ptr, normed_out_ptr, alpha2_ptr, beta2_ptr, post2_bf16_ptr,
                pre2_bf16_ptr, POST2_EPS, PRE2_EPS,
            )  # fmt: skip
    _stamp(stamps_ptr, launch.cta, _STAMP_END, STAMPS)


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------

# Every plan variant (``PlanKey``) the leaf compiles.
PLAN_VARIANTS = plan_variants()

_NEXT_KINDS = {"sliding": 0, "full": 1, "regular": 2}

# Argument types of the kernel's tensors (the rest are int32 scalars).
_POINTER_TYPES = dict(
    epochs="*i64",
    claims="*i64",
    errors="*i32",
    arena="*i64",
    exchange="*i32",
    xq_ptr="*u8",
    xs_ptr="*u8",
    box="*u32",
    h1="*bf16",
    h1_sums="*i64",
    h3="*bf16",
    h4_own="*u32",
    routes="*u32",
    h6="*bf16",
    red_in="*i64",
    h6_sums="*i64",
    q_ptr="*fp8e4nv",
    gate_ptr="*bf16",
    k_cache_ptr="*fp8e4nv",
    v_cache_ptr="*fp8e4nv",
    page_table_ptr="*ki32",
    cache_seqlens_ptr="*i32",
    schedule_ptr="*i32",
    mid_o_ptr="*fp32",
    mid_lse_ptr="*fp32",
    att_records="*i32",
    o_proj_ptr="*bf16",
    residual_ptr="*fp32",
    alpha_ptr="*fp32",
    beta_ptr="*fp32",
    post_attn_norm_ptr="*bf16",
    post_attn_bf16_ptr="*bf16",
    pre_ffn_bf16_ptr="*bf16",
    router_hi_ptr="*bf16",
    router_multiplier="fp32",
    router_lo_ptr="*bf16",
    router_bias_ptr="*fp32",
    logits_parts="*i64",
    pooler_ptr="*bf16",
    pooled_parts="*i64",
    pooled_norm_ptr="*bf16",
    w13="*u8",
    s13="*i32",
    s13_lanes="*i32",
    w2="*u8",
    s2="*i32",
    mids="*u8",
    mids_s="*i32",
    moe_tags="*i32",
    route_norm_ptr="*bf16",
    s_rows="*bf16",
    hprime="*fp32",
    post_expert_ptr="*bf16",
    alpha2_ptr="*fp32",
    beta2_ptr="*fp32",
    post2_bf16_ptr="*bf16",
    pre2_bf16_ptr="*bf16",
    qkv_w_ptr="*bf16",
    qkv_parts="*i64",
    q_out_ptr="*fp8e4nv",
    gate_out_ptr="*bf16",
    q_norm_ptr="*bf16",
    k_norm_ptr="*bf16",
    cos_sin_ptr="*fp32",
    positions_ptr="*i64",
    next_k_cache_ptr="*fp8e4nv",
    next_v_cache_ptr="*fp8e4nv",
    slots_ptr="*i64",
    k_inv_scale="fp32",
    v_inv_scale="fp32",
    normed_out_ptr="*bf16",
    stamps_ptr="*i64",
    deadline_ticks="i64",
)


@dataclass(frozen=True)
class ModelConstants:
    """The model and deployment values the kernel specializes on (fixed once
    the server starts)."""

    sm_scale: float
    page_size: int
    page_table_stride: int
    window_left: int
    gate_eps: float
    post_attn_eps: float
    post_attn_offset: float
    post_attn_has_weight: bool
    pre_ffn_eps: float
    pooled_eps: float
    pooled_offset: float
    route_eps: float
    route_offset: float
    routed_scaling_factor: float
    post2_eps: float
    pre2_eps: float
    qk_eps: float
    qk_offset: float
    swiglu_alpha: float
    swiglu_limit: float
    swiglu_beta: float


def kernel_constexprs(
    key: PlanKey,
    model: ModelConstants,
    *,
    rank: int,
    peer_cache: str,
    ring: int,
    quad_map: int,
    dn_order: int,
    addr_soffset: int,
    scale_lanes: int,
    tail_num: int,
    tail_den: int,
    n_phys1: int,
    n_phys2: int,
    num_stages: int,
    kv_scaled: bool,
    next_rotary: bool,
    qkv_window: int,
    att_merge: str,
    stamps: bool,
) -> dict:
    """The kernel's constexprs for one plan variant on one rank."""
    return dict(
        REQUESTS=key.requests,
        SLIDING=key.layer_kind == "sliding",
        NEXT_KIND=_NEXT_KINDS[key.next_kind],
        RANK=rank,
        PEER_CACHE=peer_cache,
        SM_SCALE=model.sm_scale,
        PAGE_TABLE_STRIDE=model.page_table_stride,
        PAGE_SIZE=model.page_size,
        WINDOW_LEFT=model.window_left,
        NUM_STAGES=num_stages,
        GATE_EPS=model.gate_eps,
        POST_ATTN_EPS=model.post_attn_eps,
        POST_ATTN_OFFSET=model.post_attn_offset,
        POST_ATTN_HAS_WEIGHT=model.post_attn_has_weight,
        PRE_FFN_EPS=model.pre_ffn_eps,
        POOLED_EPS=model.pooled_eps,
        POOLED_OFFSET=model.pooled_offset,
        ROUTE_EPS=model.route_eps,
        ROUTE_OFFSET=model.route_offset,
        ROUTED_SCALING_FACTOR=model.routed_scaling_factor,
        POST2_EPS=model.post2_eps,
        PRE2_EPS=model.pre2_eps,
        QK_EPS=model.qk_eps,
        QK_OFFSET=model.qk_offset,
        KV_SCALED=kv_scaled,
        NEXT_ROTARY=next_rotary,
        QKV_WINDOW=qkv_window,
        ATT_MERGE=att_merge,
        STAMPS=stamps,
        N_PHYS1=n_phys1,
        N_PHYS2=n_phys2,
        RING=ring,
        QUAD_MAP=quad_map,
        DN_ORDER=dn_order,
        ADDR_SOFFSET=addr_soffset,
        SCALE_LANES=scale_lanes,
        TAIL_NUM=tail_num,
        TAIL_DEN=tail_den,
        SWIGLU_ALPHA=model.swiglu_alpha,
        SWIGLU_LIMIT=model.swiglu_limit,
        SWIGLU_BETA=model.swiglu_beta,
    )


# Muse Spark's values, for offline compiles (codegen tests, the ISA gate).
MUSE_SPARK_CONSTANTS = ModelConstants(
    sm_scale=0.125,
    page_size=64,
    page_table_stride=4096,
    window_left=2047,
    gate_eps=1e-5,
    post_attn_eps=1e-8,
    post_attn_offset=1.0,
    post_attn_has_weight=False,
    pre_ffn_eps=1e-5,
    pooled_eps=1e-5,
    pooled_offset=0.0,
    route_eps=1e-5,
    route_offset=0.0,
    routed_scaling_factor=1.0,
    post2_eps=1e-8,
    pre2_eps=1e-5,
    qk_eps=1e-5,
    qk_offset=0.0,
    swiglu_alpha=1.0,
    swiglu_limit=0.0,
    swiglu_beta=0.0,
)


def offline_signature(constexprs: dict) -> dict:
    """The kernel's non-constexpr argument types for an offline compile:
    ``heaps`` a tuple of 8 int64, tensors as ``_POINTER_TYPES``, the rest
    int32 scalars."""
    signature = {}
    for name in gluon_decoder_layer_muse_spark_gfx950.arg_names:
        if name in constexprs:
            continue
        if name == "heaps":
            signature[name] = ("i64",) * WORLD_SIZE
        else:
            signature[name] = _POINTER_TYPES.get(name, "i32")
    return signature


# Integer arguments a launch passes as multiples of 16, which Triton then
# specializes as such (weight and scale strides, the rotary table's stride).
DIVISIBLE_ARGS = frozenset(
    {
        "cos_sin_stride",
        "stride_w13",
        "stride_s13",
        "stride_s13lin",
        "stride_s13nb",
        "stride_w2",
        "stride_s2",
        "stride_s2lin",
        "stride_s2nb",
    }
)


def offline_attrs(signature: dict) -> dict:
    """The argument attributes a launch's specialization gives the kernel:
    16-byte divisible pointers within a 32-bit range (buffer operations),
    and the 16-divisible integers of :data:`DIVISIBLE_ARGS`."""
    attrs = {}
    for index, name in enumerate(gluon_decoder_layer_muse_spark_gfx950.arg_names):
        kind = signature.get(name)
        if isinstance(kind, tuple):
            attrs.update(
                {(index, p): [["tt.divisibility", 16]] for p in range(len(kind))}
            )
        elif isinstance(kind, str) and kind.startswith("*"):
            attrs[(index,)] = [["tt.divisibility", 16], ["tt.pointer_range", 32]]
        elif name in DIVISIBLE_ARGS:
            attrs[(index,)] = [["tt.divisibility", 16]]
    return attrs


def offline_compile_args(key: PlanKey, rank: int):
    """Everything to compile plan variant ``key`` of rank ``rank`` (0-7) for
    gfx950 without a GPU, with Muse Spark's constants and the default plan
    fields.

    Returns:
        ``(fn, signature, constexprs, options)``: the kernel, its argument
        types (``heaps`` a tuple of 8 int64), the constexprs, and the compile
        options (``num_warps`` and the persistent kernel options). Pass
        :func:`offline_attrs` of the signature as the source's attributes, as
        a launch specializes them.
    """
    fn = gluon_decoder_layer_muse_spark_gfx950
    constexprs = kernel_constexprs(
        key, MUSE_SPARK_CONSTANTS, rank=rank, peer_cache=".wt", ring=16, quad_map=0,
        dn_order=0, addr_soffset=0, scale_lanes=0, tail_num=1, tail_den=2,
        n_phys1=1024, n_phys2=4096, num_stages=2, kv_scaled=False,
        next_rotary=key.next_kind == "sliding", qkv_window=2, att_merge="designated",
        stamps=False,
    )  # fmt: skip
    signature = offline_signature(constexprs)
    options = {"num_warps": NUM_WARPS, **persistent_compile_options()}
    return fn, signature, constexprs, options

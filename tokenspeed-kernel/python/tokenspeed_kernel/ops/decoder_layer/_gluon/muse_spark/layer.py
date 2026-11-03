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
constexprs; the role table is ``plan.py`` (NATIVE-DESIGN §4.1). Roles hand
work to each other through memory (NATIVE-DESIGN §7): each hand-off is
named by its §7 id at its producer and consumer below.

Phases, in the order a workgroup runs its roles:

- front: attention splits and their record merger (S1-S3), the o_proj
  senders (SND, S4-S5) and owners (OWN1, S6-S7), the router and pooler K
  parts (RTR, POOL, S8-S9), the top-k (TOPK, S10) and the pooled rows'
  norm and MXFP4 quantization (XN, S11);
- MoE: every workgroup streams its share of the experts (S12-S15);
- tail: route owners (OWN4, S16-S17), finishers (FIN, S18).

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
from tokenspeed_kernel.ops.communication._iris.muse_spark.attntail import (
    muse_attntail_publish_attention,
    muse_attntail_read_box,
)
from tokenspeed_kernel.ops.communication._iris.muse_spark.fused_layer import (
    BOX_WORDS,
    H1_SHARD,
    H1_WORDS,
    MAX_ROWS,
    MuseNativeView,
    clear_next_copies,
    fin,
    h1_rs_sink,
    h3_sink,
    h4_sink,
    iris_fail_stop,
    own1,
    own4_complete,
    xn_wait,
)
from tokenspeed_kernel.ops.decoder_layer import PlanKey
from tokenspeed_kernel_amd.ops.gfx950.attention._common import InputStrides
from tokenspeed_kernel_amd.ops.gfx950.attention.mha.decode import (
    AttentionConfig,
    AttentionProgram,
    mha_decode_sliding_split_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.gemv_splitk import (
    gemv_a16w16_splitk_hi_lo_sums_gfx950,
    gemv_a16w16_splitk_publish_gfx950,
    gemv_a16w16_splitk_sum_gfx950,
    gemv_a16w16_splitk_wave_sums_gfx950,
    gemv_a16w16_splitk_weights_gfx950,
    gemv_a16w16_splitk_x_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.layernorm.deferred import (
    deferred_norm_rows_gfx950,
    deferred_norm_scales_gfx950,
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
    enter_check,
    enter_issue,
    fail_stop,
    persistent_compile_options,
    poll_tags16,
    release_drain,
    store_pairs,
)

__all__ = [
    "NUM_WARPS",
    "PLAN_VARIANTS",
    "gluon_decoder_layer_muse_spark_gfx950",
    "DIVISIBLE_ARGS",
    "offline_compile_args",
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
ATT_FIRST_SLOT = 24
# One 16-row MFMA tile per wave tiles the attention rows.
ATT_BLOCK_M = 16 * NUM_WARPS

# The uncached exchange (int32 words): routed ids, route weights, the ids
# and xq row tags (one 64 B line per 16 rows), the xq rows and scales.
XQ_ROW_BYTES = POOLED // 2
XS_ROW_BYTES = POOLED // 32
EX_IDS = 0
EX_WEIGHTS = EX_IDS + MAX_ROWS * TOPK
EX_IDS_TAGS = EX_WEIGHTS + MAX_ROWS * TOPK
EX_XQ_TAGS = EX_IDS_TAGS + MAX_ROWS
EX_XQ = EX_XQ_TAGS + MAX_ROWS
EX_XS = EX_XQ + MAX_ROWS * XQ_ROW_BYTES // 4
# The xq scales are CDNA4-swizzled (``empty_swizzled_cdna4_mxfp4_scale``):
# one 32-row block, stride 1 within it.
XS_MBLOCK_STRIDE = MAX_ROWS * XS_ROW_BYTES
EXCHANGE_WORDS = EX_XS + MAX_ROWS * XS_ROW_BYTES // 4

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
_MAX_ROWS = gl.constexpr(MAX_ROWS)
_BOX_WORDS = gl.constexpr(BOX_WORDS)
_H1_WORDS = gl.constexpr(H1_WORDS)
_H1_SHARD = gl.constexpr(H1_SHARD)
_OWNERS = gl.constexpr(WORLD_SIZE * (HIDDEN // WORLD_SIZE // 128))
_EX_IDS = gl.constexpr(EX_IDS)
_EX_WEIGHTS = gl.constexpr(EX_WEIGHTS)
_EX_IDS_TAGS = gl.constexpr(EX_IDS_TAGS)
_EX_XQ_TAGS = gl.constexpr(EX_XQ_TAGS)
_EX_XQ = gl.constexpr(EX_XQ)
_EX_XS = gl.constexpr(EX_XS)
_XS_MBLOCK_STRIDE = gl.constexpr(XS_MBLOCK_STRIDE)
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
# Front: attention (S1-S3)
# ---------------------------------------------------------------------------


@gluon.jit
def _wait_records(record_ptr, first, count: gl.constexpr, tag, launch):
    """S2 consumer (same XCD, scalar ``glc``): every split's per-wave records
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
):
    """One sliding split of block ``block`` = (request, KV head); split 0 is
    the block's merger.

    S1: the split streams its pages and stores its partial (plain, read on
    this XCD). S2: each wave drains its stores, then stores its record
    (``publish_split_record``). S3: the merger polls the block's records,
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
    if split == 0:
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
        muse_attntail_publish_attention(
            program, merged, gate, view.box + launch.copy * _BOX_WORDS, GATE_EPS
        )


# ---------------------------------------------------------------------------
# Front: o_proj senders and owners (S4-S7)
# ---------------------------------------------------------------------------


@gluon.jit
def _send(
    launch, view, sender, tiles, rows, RANK: gl.constexpr, PEER_CACHE: gl.constexpr
):
    """SND ``sender`` (0-127): re-arm its share of the next copies, wait for
    the attention box (S4), multiply it by its resident o_proj rows and push
    the partial to the columns' owner (S5)."""
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
# Front: router and pooler K parts (S8-S9), top-k (S10), XN (S11)
# ---------------------------------------------------------------------------


@gluon.jit
def _kpart_x(
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
    GROUPS: gl.constexpr,
    POST_EPS: gl.constexpr,
    PRE_EPS: gl.constexpr,
):
    """K part ``part``'s x (FP32 B operand ``[waves, 128, 16]``): the
    deferred-norm build from the H1 all-gather words of owner rank ``part``
    (S8, Lamport payload-as-flag) and every owner's four norm sums (tagged
    words). Returns ``(h', x)``."""
    # Every owner's sums of the 16 token slots (tag high, value low).
    sums_l: gl.constexpr = _norm_sums_layout()
    t = gl.arange(0, 16, gl.SliceLayout(1, gl.SliceLayout(2, sums_l)))
    o = gl.arange(0, _OWNERS, gl.SliceLayout(0, gl.SliceLayout(2, sums_l)))
    w = gl.arange(0, 4, gl.SliceLayout(0, gl.SliceLayout(1, sums_l)))
    index = (o[None, :, None] * _MAX_ROWS + t[:, None, None]) * 4 + w[None, None, :]
    live = (t < rows)[:, None, None] & (o >= 0)[None, :, None] & (w >= 0)[None, None, :]
    flat = gl.reshape(index, [16 * _OWNERS * 4])
    sums = gemv_a16w16_splitk_sum_gfx950(
        view.h1_sums,
        launch.tag,
        flat,
        gl.reshape(live, [16 * _OWNERS * 4]),
        0,
        0,
        1,
        launch,
        iris_fail_stop,
        _SITE_SUMS,
    )
    sums = gl.convert_layout(gl.reshape(sums, [16, _OWNERS, 4]), sums_l)
    inv1, inv2 = deferred_norm_scales_gfx950(sums, _HIDDEN, POST_EPS, PRE_EPS)
    # Owner rank ``part``'s all-gather slot: [rows, 1024] BF16.
    ag = view.h1 + launch.copy * (2 * _H1_WORDS) + (_H1_WORDS + part * _H1_SHARD)
    b = muse_attntail_read_box(
        ag, rows, _HIDDEN // _KPARTS, launch, iris_fail_stop, _SITE_AG
    )
    h = gemv_a16w16_splitk_x_gfx950(
        residual_ptr, _HIDDEN, task, rows, _HIDDEN, _KPARTS, GROUPS, ""
    )
    alpha = gemv_a16w16_splitk_x_gfx950(
        alpha_ptr, 0, task, rows, _HIDDEN, _KPARTS, GROUPS, ""
    )
    beta = gemv_a16w16_splitk_x_gfx950(
        beta_ptr, 0, task, rows, _HIDDEN, _KPARTS, GROUPS, ""
    )
    post = gemv_a16w16_splitk_x_gfx950(
        post_ptr, 0, task, rows, _HIDDEN, _KPARTS, GROUPS, ""
    )
    pre = gemv_a16w16_splitk_x_gfx950(
        pre_ptr, 0, task, rows, _HIDDEN, _KPARTS, GROUPS, ""
    )
    token_l: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(1, h.type.layout))
    b = gl.convert_layout(b, h.type.layout)
    inv1 = gl.convert_layout(inv1, token_l)[None, None, :]
    inv2 = gl.convert_layout(inv2, token_l)[None, None, :]
    return deferred_norm_rows_gfx950(b, h, alpha, beta, post, pre, inv1, inv2)


@gluon.jit
def _topk(
    launch,
    group,
    rows,
    logits_ptr,
    bias_ptr,
    exchange_ptr,
    ROUTED_SCALING_FACTOR: gl.constexpr,
):
    """TOPK of rows ``8 group ..``, one wave per row: sum the router's K
    parts in part order (S9 consumer, tagged words), pick the top 8 (S10),
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
        rows,
        _EXPERTS,
        _KPARTS,
        launch,
        iris_fail_stop,
        _SITE_LOGITS,
    )
    logits = gl.convert_layout(gl.reshape(logits, [WAVES, _EXPERTS]), layout)
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
    # S10: each wave's tag after its own ids (uncached: no cache bits).
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
    """XN of ``row``: wait for every rank's pooled columns (S11 consumer of
    H3), RMSNorm, MXFP4 to the uncached exchange, then the row's tag."""
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
    release_drain()
    lanes_l: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    lane = gl.arange(0, 64 * gl.num_warps(), lanes_l)
    tag = gl.full([64 * gl.num_warps()], 0, gl.int32, lanes_l) + launch.tag
    tags = exchange_ptr + _EX_XQ_TAGS
    gl.amd.cdna4.buffer_store(tag, tags, row + lane * 0, mask=lane == 0)
    gl.amd.cdna4.buffer_store(
        tag, tags, row + 8 + lane * 0, mask=(lane == 0) & (rows <= 8)
    )


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------


@gluon.jit(do_not_specialize=("deadline_ticks", "label", "gate_stride"))
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
    # Routed sum.
    route_norm_ptr,
    s_pairs,
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
    PRE_FFN_EPS: gl.constexpr,
    POOLED_EPS: gl.constexpr,
    POOLED_OFFSET: gl.constexpr,
    ROUTE_EPS: gl.constexpr,
    ROUTE_OFFSET: gl.constexpr,
    ROUTED_SCALING_FACTOR: gl.constexpr,
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
    gl.static_assert(SLIDING, "full-attention layers are not composed yet")
    rows = gl.to_tensor(ROWS)
    view = MuseNativeView(
        heaps, rank, box, h1, h1_sums, h3, h4_own, routes, h6, red_in, h6_sums
    )
    cfg: gl.constexpr = AttentionConfig(
        SM_SCALE, PAGE_TABLE_STRIDE, PAGE_SIZE, 8, 8, 16, 2, 64, _ATT_BLOCK_M,
        PAGE_SIZE, _NUM_WARPS, True, WINDOW_LEFT, True,
        k_cache_ptr.dtype.element_ty, InputStrides(_Q_WIDTH, 64, 1),
    )  # fmt: skip

    _reserve_lds()
    # Entry: the two counters, then the role's first weights, then the check.
    entry = enter_issue(epochs, claims)
    xcd = entry.xcd
    slot = entry.slot
    if slot < 16:
        # SND (S4-S5) -> OWN1 on slot 0 (S6-S7) -> RTR (S8-S9).
        sender = xcd * 16 + slot
        o_tiles = gemv_a16w16_splitk_weights_gfx950(
            o_proj_ptr, sender, _Q_WIDTH, 64, 1, 128
        )
        task = xcd * _ROUTER_GROUPS + slot
        hi = gemv_a16w16_splitk_weights_gfx950(
            router_hi_ptr, task, _HIDDEN, 16, _KPARTS, _ROUTER_GROUPS
        )
        lo = gemv_a16w16_splitk_weights_gfx950(
            router_lo_ptr, task, _HIDDEN, 16, _KPARTS, _ROUTER_GROUPS
        )
        launch = enter_check(entry, errors, arena, deadline_ticks, label, fail_stop)
        _send(launch, view, sender, o_tiles, rows, RANK, PEER_CACHE)
        if slot == 0:
            own1(
                view, launch, xcd, rows, residual_ptr, post_attn_norm_ptr, alpha_ptr,
                beta_ptr, iris_fail_stop, RANK, POST_ATTN_OFFSET, True, PEER_CACHE,
            )  # fmt: skip
        _, x = _kpart_x(
            launch, view, xcd, task, rows, residual_ptr, alpha_ptr, beta_ptr,
            post_attn_bf16_ptr, pre_ffn_bf16_ptr, _ROUTER_GROUPS, POST_ATTN_EPS,
            PRE_FFN_EPS,
        )  # fmt: skip
        logits = gemv_a16w16_splitk_hi_lo_sums_gfx950(hi, lo, x, _sums_layout(16, 2))
        gemv_a16w16_splitk_publish_gfx950(
            logits, logits_parts, launch.tag, slot * 16, xcd, rows, _EXPERTS, ".wt"
        )
    elif slot < 24:
        # POOL (S8-S9); slot 16 + T on XCD T reduces tile T and pushes H3.
        tile = slot - 16
        task = xcd * _POOL_TILES + tile
        p_tiles = gemv_a16w16_splitk_weights_gfx950(
            pooler_ptr, task, _HIDDEN, _POOL_TILE, _KPARTS, _POOL_TILES
        )
        launch = enter_check(entry, errors, arena, deadline_ticks, label, fail_stop)
        _, x = _kpart_x(
            launch, view, xcd, task, rows, residual_ptr, alpha_ptr, beta_ptr,
            post_attn_bf16_ptr, pre_ffn_bf16_ptr, _POOL_TILES, POST_ATTN_EPS,
            PRE_FFN_EPS,
        )  # fmt: skip
        pooled = gemv_a16w16_splitk_wave_sums_gfx950(
            p_tiles, x.to(gl.bfloat16), _sums_layout(_POOL_TILE, 2)
        )
        gemv_a16w16_splitk_publish_gfx950(
            pooled, pooled_parts, launch.tag, tile * _POOL_TILE, xcd, rows,
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
                gl.reshape(live, [16 * _POOL_TILE]), rows, _POOLED_LOCAL, _KPARTS,
                launch, iris_fail_stop, _SITE_POOL,
            )  # fmt: skip
            h3_sink(
                view, launch, gl.reshape(reduced, [16, _POOL_TILE]), tile, rows, 16,
                RANK, PEER_CACHE,
            )  # fmt: skip
    else:
        launch = enter_check(entry, errors, arena, deadline_ticks, label, fail_stop)
        if xcd < 2 * REQUESTS:
            # S1-S3: split (slot - 24) of block xcd.
            _attention_split(
                launch, xcd, slot - _ATT_FIRST_SLOT, view, q_ptr, gate_ptr, gate_stride,
                k_cache_ptr, v_cache_ptr, page_table_ptr, cache_seqlens_ptr, mid_o_ptr,
                mid_lse_ptr, att_records, cfg, GATE_EPS, NUM_STAGES,
            )  # fmt: skip
        if (slot == _ATT_FIRST_SLOT) & (xcd >= 8 - REQUESTS):
            _topk(
                launch, 7 - xcd, rows, logits_parts, router_bias_ptr, exchange,
                ROUTED_SCALING_FACTOR,
            )  # fmt: skip
        if (slot >= 25) & (slot < 25 + REQUESTS):
            _xn(
                launch, view, xcd + 8 * (slot - 25), rows, pooled_norm_ptr, exchange,
                xq_ptr, xs_ptr, RANK, POOLED_EPS, POOLED_OFFSET,
            )  # fmt: skip

    # MoE (S12-S15): every workgroup, once the ids and then the xq rows land.
    if not poll_tags16(exchange + _EX_IDS_TAGS, launch.tag, launch, True, True):
        iris_fail_stop(launch, _SITE_IDS, 0)
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
    run_moe(
        launch, handles, bufs, h4_sink, (view, launch), ROWS, _TOPK, N_PHYS1,
        N_PHYS2, RING, QUAD_MAP, DN_ORDER, ADDR_SOFFSET, True, SCALE_LANES,
        SWIGLU_ALPHA, SWIGLU_LIMIT, SWIGLU_BETA,
    )  # fmt: skip

    # Tail: OWN4 (S16-S17) then FIN (S18) on slots 0 and 24-30.
    if (slot == 0) | ((slot >= 24) & (slot < 31)):
        index = gl.where(slot == 0, 0, slot - 23)
        for r in gl.static_range(REQUESTS):
            token = index + 8 * r
            own4_complete(
                view, launch, token * _TOPK + xcd, route_norm_ptr, iris_fail_stop,
                RANK, ROUTE_OFFSET, PEER_CACHE,
            )  # fmt: skip
            s = fin(
                view, launch, token, xcd,
                (exchange + _EX_WEIGHTS).to(gl.pointer_type(gl.float32), bitcast=True),
                route_norm_ptr, iris_fail_stop, ROUTE_EPS, ROUTE_OFFSET,
            )  # fmt: skip
            columns = gl.arange(0, 512, layout=s.type.layout)
            store_pairs(
                s_pairs + token * _POOLED + xcd * 512 + columns,
                s.to(gl.int32, bitcast=True),
                launch,
                columns >= 0,
                "gpu",
            )


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------

# Every plan variant (``PlanKey``) the leaf compiles.
PLAN_VARIANTS = tuple(
    PlanKey(requests, layer_kind, next_kind)
    for requests in (1, 2)
    for layer_kind in ("sliding", "full")
    for next_kind in ("sliding", "full", "regular")
)

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
    page_table_ptr="*i32",
    cache_seqlens_ptr="*i32",
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
    s_pairs="*i64",
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
    pre_ffn_eps: float
    pooled_eps: float
    pooled_offset: float
    route_eps: float
    route_offset: float
    routed_scaling_factor: float
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
        PRE_FFN_EPS=model.pre_ffn_eps,
        POOLED_EPS=model.pooled_eps,
        POOLED_OFFSET=model.pooled_offset,
        ROUTE_EPS=model.route_eps,
        ROUTE_OFFSET=model.route_offset,
        ROUTED_SCALING_FACTOR=model.routed_scaling_factor,
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
    post_attn_offset=0.0,
    pre_ffn_eps=1e-5,
    pooled_eps=1e-5,
    pooled_offset=0.0,
    route_eps=1e-5,
    route_offset=0.0,
    routed_scaling_factor=1.0,
    swiglu_alpha=1.0,
    swiglu_limit=0.0,
    swiglu_beta=0.0,
)


# Integer arguments a launch passes as multiples of 16, which Triton then
# specializes as such (weight and scale strides).
DIVISIBLE_ARGS = frozenset(
    {
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


def offline_compile_args(key: PlanKey, rank: int):
    """Everything to compile plan variant ``key`` of rank ``rank`` (0-7) for
    gfx950 without a GPU, with Muse Spark's constants and the default plan
    fields.

    Returns:
        ``(fn, signature, constexprs, options)``: the kernel, its argument
        types (``heaps`` a tuple of 8 int64), the constexprs, and the compile
        options (``num_warps`` and the persistent kernel options). Declare
        every pointer and every :data:`DIVISIBLE_ARGS` argument 16-divisible,
        as a launch specializes them.
    """
    fn = gluon_decoder_layer_muse_spark_gfx950
    constexprs = kernel_constexprs(
        key, MUSE_SPARK_CONSTANTS, rank=rank, peer_cache=".wt", ring=8, quad_map=0,
        dn_order=0, addr_soffset=0, scale_lanes=1, tail_num=1, tail_den=2,
        n_phys1=1024, n_phys2=4096, num_stages=2,
    )  # fmt: skip
    signature = {}
    for name in fn.arg_names:
        if name in constexprs:
            continue
        if name == "heaps":
            signature[name] = ("i64",) * WORLD_SIZE
        else:
            signature[name] = _POINTER_TYPES.get(name, "i32")
    options = {"num_warps": NUM_WARPS, **persistent_compile_options()}
    return fn, signature, constexprs, options

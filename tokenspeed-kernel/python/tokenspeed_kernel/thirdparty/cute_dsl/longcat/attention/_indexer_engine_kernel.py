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

"""Attention-0 indexer on TP rank 0; see LongcatIndexerEngineKernel."""

from __future__ import annotations

from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass import (
    BFloat16,
    Float8E4M3FN,
    Float32,
    Int32,
    Int64,
    Uint8,
    Uint32,
)
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.utils.distributed import atomicAdd
from cutlass.utils.smem_allocator import SmemAllocator
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention import (
    selection_exchange as _sx,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention._indexer_qk_tcgen05 import (
    LongcatQKTile,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention._primitives import (
    _quantize_bits,
    cp_async_16_g2s,
    cp_async_mbarrier_arrive_noinc,
    emit_topk_index,
    fence_acq_rel_sys,
    make_l2_policy_evict_first,
    red_add_relaxed_gpu_s32,
    store_relaxed_sys_u32,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention._stream_gemv import (
    BAR_GEMV,
    MIN_THREADS,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention._stream_gemv import (
    mbar_offset as gemv_mbar_offset,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention._stream_gemv import (
    smem_bytes as gemv_smem_bytes,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention._stream_gemv import (
    stream_gemv_tile,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._collective_primitives import (
    fence_proxy_async_global,
    load_global_u32x4,
    store_global_u32x4,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._primitives import (
    atomic_add_acq_rel_gpu_u32,
    load_acquire_gpu_u32,
    prefetch_bulk_l2,
    prefetch_global_l2,
    red_add_relaxed_gpu_u32,
    red_add_release_gpu_u32,
    store_relaxed_gpu_u32,
)
from tokenspeed_kernel.thirdparty.cute_dsl.topk.top_k.filtered_top_k_varlen_util import (
    float_as_uint32,
)
from tokenspeed_kernel.thirdparty.cute_dsl.topk.top_k.single_pass_multi_cta_radix_topk import (
    fence_acq_rel_gpu,
)
from tokenspeed_kernel.thirdparty.cute_dsl.topk.top_k.single_pass_multi_cta_radix_topk_cluster import (
    _query_max_cluster_size,
)

# The optional collective import is required only by P5.
try:
    from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention._o_proj_collective import (
        LongcatOProjCollective,
    )
except ImportError:  # pragma: no cover - optional collective
    LongcatOProjCollective = None

OPROJ_COLLECTIVE_AVAILABLE = LongcatOProjCollective is not None

__all__ = [
    "LongcatIndexerEngineKernel",
    "OPROJ_COLLECTIVE_AVAILABLE",
]

# Geometry and model constants (LongCat 2.0).
# One CTA per SM across B300's 148 SMs, in two-CTA clusters.
CLUSTERS = 74
CTAS_PER_CLUSTER = 2
NUM_CTAS = CLUSTERS * CTAS_PER_CLUSTER  # 148
NUM_THREADS = 384
NUM_WARPS = 12

NUM_TOKENS = 8
NUM_HEADS = 32
HEAD_DIM = 128
HIDDEN = 8192
QKV_OUT = 2112
QKV_LATENT_OFFSET = 1536  # kv_a latent starts here in the QKV-A output
QKV_ROPE_OFFSET = 2048  # k_pe starts here
KV_LORA_DIM = 512
ROPE_DIM = 64
WQ_B_OUT = 4096
WK_OUT = 160
WK_WEIGHT_OFFSET = 128  # scoring weights live at raw_k[:, 128:160]
RMS_EPS = 1e-6
FP8_MAX = 448.0
QUANT_EPS = 1e-10

TOPK = 2048
# P3 stages this many block-table entries before waiting for logits;
# wider block tables use global memory for slot mapping.
P3_BT_WINDOW = 4096
PAGE_SIZE = 64
PAGE_SHIFT = 6  # log2(PAGE_SIZE), the shift the fused slot mapping uses
HALF_PAGE = PAGE_SIZE // 2  # 32: the P2 scoring task granularity
INDEX_K_ROW_BYTES = 132  # 128 FP8 + FP32 per-token scale
INDEX_K_PAGE_BYTES = PAGE_SIZE * INDEX_K_ROW_BYTES  # 8448
INDEX_K_ROW_WORDS = INDEX_K_ROW_BYTES // 4  # 33
CHUNK_SIZE_DEFAULT = 4096
INITIAL_TOKENS = 16
LOCAL_TOKENS = 1024
# Selection prefix [0, min(length, FORCED_TOKENS)): initial then local
# positions in row order, directly addressable by consumers.
FORCED_TOKENS = INITIAL_TOKENS + LOCAL_TOKENS

# P2 builds a 65536-bin histogram of ordered BF16 keys with relaxed reds.
# P3 finds threshold t and tie budget k*, then counts and emits keys > t
# plus the first k* atomic tie claims. Generation parity lets P0 zero the
# next histogram while P2 uses the current one.
HIST_BINS = 1 << 16
KEY_INF = 0xFF80  # ordered key of BF16 +inf: the forced initial/local slots
KEY_NEG_INF = 0x007F  # ordered key of BF16 -inf (also NaN's sink)
# P2 accumulates coarse key>>8 counts in CTA-private SMEM (8 KB), then
# flushes up to 2048 reds. P3 uses them to locate the 256-bin fine block.
COARSE_BINS = 256
COARSE_SHIFT = 8  # key >> 8 == the 256-bin block the key falls in
# row_states int32 words: [0] definite counter, [1] tie counter, [2:4] pad;
# fine bins at 4 + parity*HIST_BINS + key; coarse bins at
# ROW_STATE_COARSE_OFF + parity*256 + (key>>8); histogram bases align to 16 B.
ROW_STATE_HIST_OFF = 4
ROW_STATE_COARSE_OFF = ROW_STATE_HIST_OFF + 2 * HIST_BINS  # 131076
ROW_STATE_WORDS = ROW_STATE_COARSE_OFF + 2 * COARSE_BINS  # 131588 (16 B mult)

# P0 has no cluster barrier; each CTA owns a complete tile.
# Pass 0 (K=8192): CTAs 0-95 project q_lora, 96-105 project wk.
# Rank 0 has no MLA head; QKV-A's KV latent / k_pe columns are unused.
# Pass 1 (K=1536): CTAs 0-127 project wq_b after lora_done.
QKV_A_TILE_N = 16
WK_TILE_N = 16
WQ_B_TILE_N = 32
QKV_A_TILES = QKV_LATENT_OFFSET // QKV_A_TILE_N  # 96 exact (q_lora only)
WK_TILES = -(-WK_OUT // WK_TILE_N)  # 10 exact
WQ_B_TILES = -(-WQ_B_OUT // WQ_B_TILE_N)  # 128 exact
P0_TILES = QKV_A_TILES + WK_TILES  # 106 pass-0 CTAs
P0_PIPE_STAGES = 5
# Stream P0 reads the ordinary hidden rows through TMA.
STREAM_P0_THREADS = MIN_THREADS  # warps 0-8; 9-11 keep the L2 prefetches
STREAM_QB_STAGES = 3  # K = 1536 is three 512-element slices
HIST_CHUNK_WORDS = 4096  # 16 KB L2 warm-up chunks of the score histograms
HIST_CHUNKS_PER_ROW = HIST_BINS // HIST_CHUNK_WORDS
# Both passes add one proj_done per published tile.
PROJ_PER_STEP = P0_TILES + WQ_B_TILES  # 234
# wq_b gates on all projected q_lora tiles.
LORA_PER_STEP = QKV_A_TILES  # 96 -> lora_done adds

P1_TOKEN_CTAS = NUM_TOKENS  # one index-K task per token row
P1_Q_TASKS = NUM_TOKENS * NUM_HEADS  # 256 dynamically assigned (token, head) rows

SMEM_BUDGET_BYTES = 200 * 1024  # one CTA per SM

# P5 uses all warps: cluster_size * (comm_threads / 32) <= 32 (2*12=24).
P5_COMM_THREADS = NUM_THREADS
P5_SCRATCH_BYTES = 16 * 4  # the per-warp partials, then the inverse RMS
BAR_P5_NORM = 13

# Uint32 tickets: completion counts target generation * count_per_step;
# P2 pickup/logits counters alternate parity slots reset for the next step.
TICK_PROJ = 0
TICK_LORA = 1
TICK_K_READY = 2
TICK_Q_READY = 3
TICK_LOGITS = 4
TICK_ROWS = 5
TICK_P1_TASK = 6  # monotonic P1 task counter (dynamic pickup)
TICK_P2_TASK = 7  # P2 half-page task counter (dynamic pickup), slot gen & 1
TICK_PUB = 8  # per-(row, peer) selection publications (7 per row per step)
TICK_P2_TASK_ODD = 9  # second P2 counter slot; the idle slot is re-zeroed
TICK_LOGITS_ODD = 10  # second scored-half-page slot (TICK_LOGITS is even)
TICK_ROW_READY = 11  # 8 unused words
NUM_TICKETS = TICK_ROW_READY + 8

# P0 per-tile publication slots (``scratch``): the three GEMM families get
# disjoint ranges so P1 can gate each task on exactly the tiles it reads.
SLOT_QKV_A = 0
SLOT_WK = SLOT_QKV_A + QKV_A_TILES  # 96
SLOT_WQ_B = SLOT_WK + WK_TILES  # 106
NUM_TILE_SLOTS = SLOT_WQ_B + WQ_B_TILES  # 234
SCRATCH_WORDS = 256  # >= NUM_TILE_SLOTS, 16 B aligned
# P1 task pool: 8 token rows first (they gate P2's current page), then the
# 256 (token, head) Q rows ordered head-major.
P1_TASKS = NUM_TOKENS + NUM_TOKENS * NUM_HEADS  # 264
# P1 runs three HEAD_DIM-thread tasks concurrently: warps 0-3 / 4-7 / 8-11,
# each with an independent named barrier and four sWarp words.
P1_GROUPS = 3
P1_GROUP_THREADS = NUM_THREADS // P1_GROUPS  # 128
P1_GROUP_WARPS = NUM_WARPS // P1_GROUPS  # 4
assert P1_GROUP_THREADS == HEAD_DIM, "a P1 group must be exactly HEAD_DIM wide"
assert P1_TASKS % P1_GROUPS == 0, "the P1 pool must split into whole triples"
# One grab hands out a whole triple ``(3t, 3t+1, 3t+2)``.
P1_TRIPLES = P1_TASKS // P1_GROUPS  # 88
BAR_P1_GROUP = 4  # ids 4/5/6, one per 128-thread group
NUM_PEERS = 7
PUBS_PER_STEP = NUM_TOKENS * NUM_PEERS  # row-granular publish adds per step
# P4 publishers: the row's own P3 cluster, each CTA to up to four peers.
P4_PEERS_PER_CTA = -(-NUM_PEERS // CTAS_PER_CLUSTER)  # 4
P4_VECS = -(-TOPK // (NUM_THREADS * 4))  # 16 B vectors per thread per row: 2

# SMEM byte offsets: P0/P1/P2/P3/P5 reuse offset 0 in separate segments.

# P1: 12 warp-reduction words plus the task broadcast region.
_P1_TASK_OFF = NUM_WARPS * 4  # 48
_P1_BYTES = _P1_TASK_OFF + 16  # 64

# P2 Q storage budget: 144 B per row; q_offset selects the active layout.
_SQ_PITCH = 144
_SQ_ROWS = NUM_TOKENS * NUM_HEADS  # 256 = 8 tokens x 32 heads
_SQ_BYTES = _SQ_ROWS * _SQ_PITCH  # 36864
# P2 raw half-page: 32 FP8 rows plus 32 FP32 scales per stage.
# The constructor aligns tcgen05 K stages and derives the active offsets.
P2_K_STAGES = 2
# Warps 8-11 issue 16 B cp.async; warps 0-7 own one verify token each.
# Each issuer copies two value blocks; the first eight also copy scales.
P2_ISSUE_WARP0 = 8
P2_ISSUE_THREADS = (NUM_WARPS - P2_ISSUE_WARP0) * 32  # 128
_SK_PITCH = 144
_SK_VALUES_BYTES = HALF_PAGE * _SK_PITCH  # 4608
_SK_STAGE_BYTES = _SK_VALUES_BYTES + HALF_PAGE * 4  # 4736
_SK_BYTES = P2_K_STAGES * _SK_STAGE_BYTES  # 9472
_P2_MBAR_OFF = _SQ_BYTES + _SK_BYTES  # 46336
# Two task-id slots; reuse follows the CTA barriers separating stage reads.
_P2_TASK_OFF = _P2_MBAR_OFF + 8 * P2_K_STAGES  # 46352 at S=2
# Coarse histogram staging: one 256-bin int32 table per verify row, private
# to the CTA and aggregated with SMEM atomics by the scoring epilogue.
_P2_COARSE_OFF = -(-(_P2_TASK_OFF + 16) // 128) * 128  # 46464 at S=2
_P2_BYTES = _P2_COARSE_OFF + NUM_TOKENS * COARSE_BINS * 4  # 54656 at S=2

# P3 histogram, threshold and filter scratch.
_P3_HIST_OFF = 0
_P3_PREFIX_OFF = _P3_HIST_OFF + 256 * 4  # 1024
_P3_SCAL_OFF = _P3_PREFIX_OFF + 256 * 4  # 2048
_P3_WARP_OFF = 2176  # 128-aligned after scalar scratch
_P3_ORDERED_OFF = 2304  # 128-aligned
_RADIX_FIXED_BYTES = _P3_ORDERED_OFF  # 2304


def p3_bt_offset(chunk_size: int) -> int:
    """128-aligned block-table offset after fixed P3 scratch.

    ``chunk_size`` is retained for signature compatibility and ignored.
    """
    return _RADIX_FIXED_BYTES


def p3_arena_bytes(chunk_size: int) -> int:
    """P3 SMEM: 2304 B scratch plus a 16 KB block-table window."""
    return p3_bt_offset(chunk_size) + P3_BT_WINDOW * 4


@cute.jit
def _bf16_bits_rne(v: Float32) -> Uint32:
    """BF16 bit pattern of ``v`` under round-to-nearest-even (finite or inf).

    NaN is not handled here: callers scrub it to -inf first so the stored
    score and its histogram key agree.
    """
    u = Uint32(float_as_uint32(v))
    lsb = (u >> Uint32(16)) & Uint32(1)
    return ((u + Uint32(0x7FFF) + lsb) >> Uint32(16)) & Uint32(0xFFFF)


@cute.jit
def _ordered_key_bf16(bits: Uint32) -> Uint32:
    """Monotone 16-bit key: larger BF16 value -> larger unsigned key."""
    key = bits | Uint32(0x8000)
    if (bits & Uint32(0x8000)) != Uint32(0):
        key = (bits ^ Uint32(0xFFFF)) & Uint32(0xFFFF)
    return key


@cute.jit
def _selection_key(bits: Uint32, pos: Int32, length: Int32) -> Uint32:
    """Shared P2/P3 key: +inf for forced positions, ordered BF16 otherwise.

    Histogram counts and filter membership must use identical keys.
    """
    key = _ordered_key_bf16(bits)
    if pos < Int32(INITIAL_TOKENS):
        key = Uint32(KEY_INF)
    if pos >= length - Int32(LOCAL_TOKENS):
        key = Uint32(KEY_INF)
    return key


@cute.jit
def _forced_index(pos: Int32, length: Int32) -> Int32:
    """Canonical slot of forced position ``pos``, or -1 when it is not forced.

    Initial tokens precede the local window; short rows use slot=position.
    The identity path and both filter passes share this layout."""
    idx = Int32(-1)
    if length <= Int32(FORCED_TOKENS):
        idx = pos
    else:
        if pos < Int32(INITIAL_TOKENS):
            idx = pos
        if pos >= length - Int32(LOCAL_TOKENS):
            idx = Int32(INITIAL_TOKENS) + pos - (length - Int32(LOCAL_TOKENS))
    return idx


@cute.jit
def _warp_exclusive_sum(lane_sum: Int32, lane: Int32) -> Int32:
    """Count entries in the higher-ranked lanes of P3's descending scan."""
    excl = lane_sum
    for step in cutlass.range_constexpr(5):
        up = cute.arch.shuffle_sync_up(
            excl, 1 << step, mask=0xFFFFFFFF, mask_and_clamp=0
        )
        if lane >= (1 << step):
            excl = excl + up
    return excl - lane_sum


@cute.jit
def _load_score_vector(g_logits_row_u32: cute.Tensor, pos_base: Int32):
    """Load eight BF16 scores as four aligned u32 words for either filter pass."""
    ptr = cute.make_ptr(
        Uint32,
        (g_logits_row_u32.iterator + pos_base // Int32(2)).toint(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    return load_global_u32x4(ptr)


@cute.jit
def _spin_until_ticket(ptr, target: Uint32) -> None:
    """Acquire-spin until the Uint32 ticket reaches target (gpu scope).

    Outstanding ticket distance stays below 2**31 across Uint32 wrap.
    All CTA threads call: thread 0 polls, then a barrier shares the acquire.
    """
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        value = load_acquire_gpu_u32(ptr)
        while (value - target).to(Int32) < Int32(0):
            value = load_acquire_gpu_u32(ptr)
    cute.arch.barrier()


@dsl_user_op
def _p1_group_sync(group: Int32, *, loc=None, ip=None) -> None:
    """Synchronize a P1 group whose task may differ from the other CTA groups."""
    barrier_id = Int32(BAR_P1_GROUP) + group
    llvm.inline_asm(
        None,
        [barrier_id.ir_value(loc=loc, ip=ip)],
        f"barrier.sync $0, {P1_GROUP_THREADS};",
        "r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _group_wait_slot_ranges(
    slots: cute.Tensor,
    first_a: Int32,
    count_a: cutlass.Constexpr,
    first_b: Int32,
    count_b: cutlass.Constexpr,
    gen: Int32,
    group: Int32,
) -> None:
    """Acquire two tile-slot ranges in parallel, one slot per group thread.

    Threads [0, count_a) poll first_a; the next count_b poll first_b.
    The group barrier shares their acquires with all 128 threads.
    """
    tidx, _, _ = cute.arch.thread_idx()
    gtid = tidx - group * Int32(P1_GROUP_THREADS)
    if gtid < Int32(count_a + count_b):
        idx = first_a + gtid
        if gtid >= Int32(count_a):
            idx = first_b + gtid - Int32(count_a)
        value = load_acquire_gpu_u32(slots.iterator + idx)
        while (value - Uint32(gen)).to(Int32) < Int32(0):
            value = load_acquire_gpu_u32(slots.iterator + idx)
    _p1_group_sync(group)


@cute.jit
def _group_reduce(
    value: cutlass.Float32,
    warp_values: cute.Tensor,
    group: Int32,
    is_max: cutlass.Constexpr,
) -> cutlass.Float32:
    """Reduce one P1 group with the same two-level sum or max tree."""
    lane = cute.arch.lane_idx()
    base = group * Int32(P1_GROUP_WARPS)
    gwarp = cute.arch.warp_idx() - base
    for step in cutlass.range_constexpr(5):
        if cutlass.const_expr(is_max):
            other = cute.arch.shuffle_sync_bfly(
                value, offset=1 << step, mask=-1, mask_and_clamp=31
            )
            if other > value:
                value = other
        else:
            value = value + cute.arch.shuffle_sync_bfly(
                value, offset=1 << step, mask=-1, mask_and_clamp=31
            )
    if lane == 0:
        warp_values[base + gwarp] = value
    _p1_group_sync(group)
    if cutlass.const_expr(is_max):
        group_value = cutlass.Float32(-3.402823466e38)
    else:
        group_value = cutlass.Float32(0.0)
    if gwarp == 0:
        if lane < P1_GROUP_WARPS:
            group_value = warp_values[base + lane]
        for step in cutlass.range_constexpr(2):
            if cutlass.const_expr(is_max):
                other = cute.arch.shuffle_sync_bfly(
                    group_value, offset=1 << step, mask=-1, mask_and_clamp=31
                )
                if other > group_value:
                    group_value = other
            else:
                group_value = group_value + cute.arch.shuffle_sync_bfly(
                    group_value, offset=1 << step, mask=-1, mask_and_clamp=31
                )
        if lane == 0:
            warp_values[base] = group_value
    _p1_group_sync(group)
    return warp_values[base]


@cute.jit
def _store_published_word(address: Int64, value: Uint32) -> None:
    """Store one word of a peer's selection payload."""
    dst = cute.make_tensor(
        cute.make_ptr(
            Uint32,
            address,
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        cute.make_layout((1,)),
    )
    dst[0] = value


@cute.jit
def _absolute(value: cutlass.Float32) -> cutlass.Float32:
    result = value
    if value < cutlass.Float32(0.0):
        result = -value
    return result


class LongcatIndexerEngineKernel:
    """Rank-0 indexer with BF16 streaming projections and FP8 tcgen05 QK.

    P1 quantizes query/index-K after acquiring projection slots. P2 scores
    32 positions per tcgen05 task; P3 reserves the first 16 and last 1024
    tokens and selects the rest by BF16 histogram, bypassing scores for
    short contexts. P4 publishes selection and index-K to workers; P5
    consumes their TP8 collective. PDL waits at entry and triggers after P4.
    """

    def __init__(
        self,
        *,
        kv_dtype: cutlass.Numeric,
        softmax_scale: float,
        chunk_size: int = CHUNK_SIZE_DEFAULT,
        norm_eps: float = 1e-5,
        quant_outputs: int = 0,
    ) -> None:
        if kv_dtype not in (Float8E4M3FN, BFloat16):
            raise ValueError(f"unsupported kv_dtype: {kv_dtype!r}")
        if chunk_size <= 0 or chunk_size % 8:
            raise ValueError("chunk_size must be a positive multiple of 8")
        if LongcatOProjCollective is None:
            raise RuntimeError(
                "Segment P5 requires "
                "(thirdparty/cute_dsl/longcat/attention/_o_proj_collective.py,"
                " class LongcatOProjCollective)."
            )
        if int(quant_outputs) not in (0, 1, 2):
            raise ValueError("quant_outputs must be 0, 1 or 2")
        self.quant_outputs = int(quant_outputs)
        self.w_bytes = 2
        self.p0_stages = P0_PIPE_STAGES
        while (
            self.p0_stages > 1
            and gemv_smem_bytes(QKV_A_TILE_N, self.p0_stages, self.w_bytes, 512)
            > SMEM_BUDGET_BYTES
        ):
            self.p0_stages -= 1
        self.p0_stages = min(self.p0_stages, HIDDEN // 512)

        self.p2_tasks_per_page = PAGE_SIZE // 32
        self.p2_sk_values_bytes = 32 * _SK_PITCH
        self._p2_qk = LongcatQKTile(32)
        self.arena_alignment = 1024
        self.p2_sk_stage_bytes = _SK_STAGE_BYTES
        self.p2_sk_stage_bytes = -(-(self.p2_sk_values_bytes + 128) // 1024) * 1024
        self.p2_sk_bytes = P2_K_STAGES * self.p2_sk_stage_bytes
        self.p2_mbar_off = _SQ_BYTES + self.p2_sk_bytes
        self.p2_task_off = self.p2_mbar_off + 8 * P2_K_STAGES
        self.p2_coarse_off = -(-(self.p2_task_off + 20) // 128) * 128
        self.p2_bytes = self.p2_coarse_off + NUM_TOKENS * COARSE_BINS * 4
        self.p2_tcgen_mbar_off = self.p2_task_off + 8
        self.p2_tcgen_tmem_off = self.p2_task_off + 16
        assert self.p2_tcgen_tmem_off + 4 <= self.p2_coarse_off

        self.kv_dtype = kv_dtype
        self.softmax_scale = float(softmax_scale)
        self.chunk_size = int(chunk_size)
        self.is_fp8_kv = kv_dtype == Float8E4M3FN

        parts = []
        parts.append(gemv_smem_bytes(QKV_A_TILE_N, self.p0_stages, self.w_bytes, 512))
        parts.append(gemv_smem_bytes(WQ_B_TILE_N, STREAM_QB_STAGES, self.w_bytes))
        parts.append(_P1_BYTES)
        parts.append(self.p2_bytes)
        parts.append(p3_arena_bytes(self.chunk_size))
        parts.append(P5_SCRATCH_BYTES)
        self.arena_bytes = max(parts)
        if self.arena_bytes > SMEM_BUDGET_BYTES:
            raise ValueError(
                f"SMEM arena {self.arena_bytes} B exceeds the {SMEM_BUDGET_BYTES} B"
                " budget"
            )

        max_cluster_size = _query_max_cluster_size()
        if max_cluster_size < CTAS_PER_CLUSTER:
            raise RuntimeError(
                f"indexer requires {CTAS_PER_CLUSTER}-CTA clusters; "
                f"device supports at most {max_cluster_size}"
            )
        self.collective = LongcatOProjCollective(
            indexer_rank=0,
            cluster_size=CTAS_PER_CLUSTER,
            norm_eps=float(norm_eps),
        )

    # P1: postprocess (all CTAs)

    @cute.jit
    def _p1_device(
        self,
        gRawQ: cute.Tensor,
        gRawK: cute.Tensor,
        gRmsW: cute.Tensor,
        gCosSin: cute.Tensor,
        gPositions: cute.Tensor,
        gOutCacheLoc: cute.Tensor,
        gIndexKCache: cute.Tensor,
        gQFp8: cute.Tensor,
        gScaledW: cute.Tensor,
        gTickets: cute.Tensor,
        gScratch: cute.Tensor,
        sWarp: cute.Tensor,
        sTask: cute.Tensor,
        tidx: Int32,
        cta: Int32,
        gen: Int32,
    ) -> None:
        """P1: dynamically picked token rows and (token, head) Q rows.

        The index-K row uses BF16 RMSNorm/RoPE rounding before FP8 conversion.
        Intermediate BF16 rounding points match production: after RMSNorm,
        after RoPE.  Dynamic FP8 scale per quantization group is
        ``max(absmax, 1e-10) / 448``; the 132 B row scatter places the 128
        FP8 bytes at ``page*8448 + slot*128`` and the FP32 scale at word
        ``page*2112 + 2048 + slot``.

        Rank 0 holds no MLA head. The token row neither normalizes the KV
        latent nor writes the KV cache; P0 projects only the q_lora prefix.
        """
        # Each CTA grabs three tasks for its independent 128-thread groups.
        # Token tasks wait for all wk tiles; Q tasks wait for the two wk
        # weight tiles and their head's four wq_b tiles. Token tasks come
        # first because P2's fresh-page copies depend on their index-K rows.
        # One terminating grab per CTA advances the monotonic counter by
        # P1_TRIPLES + NUM_CTAS per step.
        task_base = (Uint32(gen) - Uint32(1)) * Uint32(P1_TRIPLES + NUM_CTAS)
        group = tidx // Int32(P1_GROUP_THREADS)
        gtid = tidx % Int32(P1_GROUP_THREADS)
        first_wait = Int32(1)
        running = Int32(1)
        # Release the previous round's Q rows before the next grab, including
        # the terminating grab. The round-end barrier orders all group stores.
        round_q = Int32(0)
        while running != Int32(0):
            if tidx == 0:
                if round_q > Int32(0):
                    # Fence publishes all Q stores before the relaxed count.
                    fence_acq_rel_gpu()
                    red_add_relaxed_gpu_u32(
                        gTickets.iterator + Int32(TICK_Q_READY), Uint32(round_q)
                    )
                sTask[0] = (
                    Uint32(
                        atomicAdd(gTickets.iterator + Int32(TICK_P1_TASK), Uint32(1))
                    )
                    - task_base
                ).to(Int32)
            cute.arch.barrier()
            triple = sTask[0]
            round_q = Int32(0)
            for g in cutlass.range_constexpr(P1_GROUPS):
                if triple * Int32(P1_GROUPS) + Int32(g) >= Int32(NUM_TOKENS):
                    round_q = round_q + Int32(1)
            if triple >= Int32(P1_TRIPLES):
                running = Int32(0)
            else:
                task = triple * Int32(P1_GROUPS) + group
                if task < Int32(NUM_TOKENS):
                    _group_wait_slot_ranges(
                        gScratch, Int32(SLOT_WK), WK_TILES, Int32(0), 0, gen, group
                    )
                else:
                    head_q = (task - Int32(NUM_TOKENS)) // Int32(NUM_TOKENS)
                    _group_wait_slot_ranges(
                        gScratch,
                        Int32(SLOT_WK + WK_TILES - 2),
                        2,
                        Int32(SLOT_WQ_B) + head_q * Int32(HEAD_DIM // WQ_B_TILE_N),
                        HEAD_DIM // WQ_B_TILE_N,
                        gen,
                        group,
                    )
                if first_wait != Int32(0):
                    first_wait = Int32(0)
                if task < Int32(NUM_TOKENS):
                    # Token row: index-K scatter.
                    token = task
                    position = gPositions[token]

                    value = gRawK[token, gtid].to(cutlass.Float32)
                    sum_sq = _group_reduce(value * value, sWarp, group, False)
                    rstd = cute.math.rsqrt(
                        sum_sq / cutlass.Float32(HEAD_DIM) + cutlass.Float32(RMS_EPS),
                        fastmath=True,
                    )
                    value = value * rstd * gRmsW[gtid].to(cutlass.Float32)
                    value = value.to(cutlass.BFloat16).to(cutlass.Float32)
                    if gtid < ROPE_DIM:
                        paired_raw = gRawK[token, gtid ^ 1].to(cutlass.Float32)
                        paired = paired_raw * rstd * gRmsW[gtid ^ 1].to(cutlass.Float32)
                        paired = paired.to(cutlass.BFloat16).to(cutlass.Float32)
                        cos = gCosSin[position, gtid // 2].to(cutlass.Float32)
                        sin = gCosSin[position, 32 + gtid // 2].to(cutlass.Float32)
                        if gtid % 2 == 0:
                            value = value * cos - paired * sin
                        else:
                            value = value * cos + paired * sin
                        value = value.to(cutlass.BFloat16).to(cutlass.Float32)
                    absmax = _group_reduce(_absolute(value), sWarp, group, True)
                    if absmax < cutlass.Float32(QUANT_EPS):
                        absmax = cutlass.Float32(QUANT_EPS)
                    k_scale = absmax / cutlass.Float32(FP8_MAX)

                    slot = gOutCacheLoc[token]
                    page = slot // Int64(PAGE_SIZE)
                    page_offset = slot - page * Int64(PAGE_SIZE)
                    page_bytes = Int64(INDEX_K_PAGE_BYTES)
                    fp8_offset = (
                        page * page_bytes + page_offset * Int64(HEAD_DIM) + Int64(gtid)
                    )
                    gIndexKCache[fp8_offset] = _quantize_bits(value, k_scale)
                    if gtid == 0:
                        scale_offset = (
                            page * Int64(PAGE_SIZE * INDEX_K_ROW_WORDS)
                            + Int64(PAGE_SIZE * HEAD_DIM // 4)
                            + page_offset
                        )
                        cache_words = cute.recast_tensor(gIndexKCache, cutlass.Float32)
                        cache_words[scale_offset] = k_scale

                    # Order every writer's stores before the group barrier
                    # and release-add publishing index-K to P2.
                    fence_proxy_async_global()
                    _p1_group_sync(group)
                    if gtid == 0:
                        red_add_release_gpu_u32(
                            gTickets.iterator + Int32(TICK_K_READY), Uint32(1)
                        )
                else:
                    q_task = task - Int32(NUM_TOKENS)
                    head = q_task // Int32(NUM_TOKENS)
                    token = q_task % Int32(NUM_TOKENS)
                    position = gPositions[token]

                    q_value = gRawQ[token, head * HEAD_DIM + gtid].to(cutlass.Float32)
                    if gtid < ROPE_DIM:
                        paired = gRawQ[token, head * HEAD_DIM + (gtid ^ 1)].to(
                            cutlass.Float32
                        )
                        cos = gCosSin[position, gtid // 2].to(cutlass.Float32)
                        sin = gCosSin[position, 32 + gtid // 2].to(cutlass.Float32)
                        if gtid % 2 == 0:
                            q_value = q_value * cos - paired * sin
                        else:
                            q_value = q_value * cos + paired * sin
                        q_value = q_value.to(cutlass.BFloat16).to(cutlass.Float32)
                    head_absmax = _group_reduce(_absolute(q_value), sWarp, group, True)
                    if head_absmax < cutlass.Float32(QUANT_EPS):
                        head_absmax = cutlass.Float32(QUANT_EPS)
                    q_scale = head_absmax / cutlass.Float32(FP8_MAX)
                    gQFp8[token, head, gtid] = _quantize_bits(q_value, q_scale)
                    if gtid == 0:
                        raw_weight = gRawK[token, WK_WEIGHT_OFFSET + head].to(
                            cutlass.Float32
                        )
                        # Fold q_scale and softmax_scale (128**-0.5 * 32**-0.5)
                        # as in dsa_topk.py's combine_topk_weights.
                        gScaledW[token, head] = (
                            raw_weight * q_scale * cutlass.Float32(self.softmax_scale)
                        )
            # Order Q stores before the next release and sTask overwrite.
            cute.arch.barrier()

    # P2: FP8 paged scoring (all CTAs, dynamic half-page pickup)

    @cute.jit
    def _p2_device(
        self,
        gQFp8: cute.Tensor,
        gScaledW: cute.Tensor,
        gIndexKCache: cute.Tensor,
        gSeqLens: cute.Tensor,
        gBlockTable: cute.Tensor,
        gLogits: cute.Tensor,
        gRowStates: cute.Tensor,
        gTickets: cute.Tensor,
        arena_i32,
        mbar: cute.Pointer,
        tidx: Int32,
        cta: Int32,
        gen: Int32,
    ) -> None:
        """P2: native FP8 tcgen05 scoring and BF16 histogram construction.

        logits[t,pos] = k_scale[pos] * sum_h scaled_w[t,h] * relu(dot(q,k)).
        P1 folds w*q_scale*softmax_scale into scaled_w, matching dsa_topk.py.
        Moving the positive k_scale after ReLU and the head sum preserves
        the formula. Causally masked positions and NaNs become -inf before
        BF16 rounding, so stored scores and histogram keys agree.

        Warps 8-11 copy raw 32x128 FP8 K bytes and 32 FP32 scales through
        two cp.async stages; warps 0-7 reduce one token each. Thread 0 grabs
        task=2*page+half and broadcasts it in SMEM. The deep pipeline holds
        cur/nxt and refills each freed stage; a single-wave pool uses one
        task per CTA. Each CTA makes one terminating grab, leaving the
        parity counter at num_tasks+NUM_CTAS.

        Fresh verify rows occupy seq_len-8+t. Before issuing the first task
        >= 2*((seq_len-8)//64), each CTA acquires k_ready once; earlier
        history copies can overlap P1. All scoring waits for q_ready.
        """
        qk_args = self._p2_qk.prepare()
        qk_mbar = cute.make_ptr(
            Int64,
            arena_i32 + Int32(self.p2_tcgen_mbar_off),
            cute.AddressSpace.smem,
            assumed_align=8,
        )
        qk_word = cute.make_ptr(
            Int32,
            arena_i32 + Int32(self.p2_tcgen_tmem_off),
            cute.AddressSpace.smem,
            assumed_align=4,
        )
        qk_tmem = self._p2_qk.begin(qk_mbar, qk_word, tidx)
        lane = tidx % 32
        warp = tidx // 32
        # Score live pages only; block-table capacity may exceed seq_len.
        num_pages = (gSeqLens[0] + Int32(PAGE_SIZE - 1)) // Int32(PAGE_SIZE)
        if num_pages > gBlockTable.shape[1]:
            num_pages = Int32(gBlockTable.shape[1])
        num_tasks = Int32(self.p2_tasks_per_page) * num_pages
        # Prefetch the block table before P3 stages it into SMEM.
        if tidx * 32 < gBlockTable.shape[1]:
            # one 128 B line per thread; the table is only 4 B aligned
            prefetch_global_l2(gBlockTable.iterator + tidx * Int32(32))
        gate_task = Int32(self.p2_tasks_per_page) * (
            (gSeqLens[0] - Int32(NUM_TOKENS)) // Int32(PAGE_SIZE)
        )
        # Request-sized pickup/logits counters start at zero in gen&1;
        # CTA 0 resets only the idle parity for the next generation.
        task_base = Int32(0)
        p2_counter = gTickets.iterator + Int32(TICK_P2_TASK)
        logits_ticket = gTickets.iterator + Int32(TICK_LOGITS)
        if (gen & Int32(1)) != Int32(0):
            p2_counter = gTickets.iterator + Int32(TICK_P2_TASK_ODD)
            logits_ticket = gTickets.iterator + Int32(TICK_LOGITS_ODD)
        pool_open = Int32(1)
        k_waited = Int32(0)
        ndone = Int32(0)

        sq_u8 = cute.make_tensor(
            cute.make_ptr(Uint8, arena_i32, cute.AddressSpace.smem, assumed_align=128),
            cute.make_layout((_SQ_ROWS, _SQ_PITCH), stride=(_SQ_PITCH, 1)),
        )
        sk_u8 = cute.make_tensor(
            cute.make_ptr(
                Uint8,
                arena_i32 + Int32(_SQ_BYTES),
                cute.AddressSpace.smem,
                assumed_align=128,
            ),
            cute.make_layout((self.p2_sk_bytes,), stride=(1,)),
        )
        s_task = cute.make_tensor(
            cute.make_ptr(
                Int32,
                arena_i32 + Int32(self.p2_task_off),
                cute.AddressSpace.smem,
                assumed_align=16,
            ),
            cute.make_layout((P2_K_STAGES,)),
        )
        s_coarse = cute.make_tensor(
            cute.make_ptr(
                Int32,
                arena_i32 + Int32(self.p2_coarse_off),
                cute.AddressSpace.smem,
                assumed_align=128,
            ),
            cute.make_layout((NUM_TOKENS * COARSE_BINS,)),
        )

        for i in range(tidx, NUM_TOKENS * COARSE_BINS, NUM_THREADS):
            s_coarse[i] = Int32(0)
        if tidx == 0:
            for s in cutlass.range_constexpr(P2_K_STAGES):
                cute.arch.mbarrier_init(mbar + s, P2_ISSUE_THREADS)
            cute.arch.mbarrier_init_fence()
            s_task[0] = Int32(atomicAdd(p2_counter, Uint32(1))) - task_base
            if cta == Int32(0):
                if (gen & Int32(1)) != Int32(0):
                    gTickets[TICK_P2_TASK] = Uint32(0)
                    gTickets[TICK_LOGITS] = Uint32(0)
                else:
                    gTickets[TICK_P2_TASK_ODD] = Uint32(0)
                    gTickets[TICK_LOGITS_ODD] = Uint32(0)
        cute.arch.barrier()  # mbar init + first task id + zeroed s_coarse
        cur = s_task[0]
        first_task = cur
        if cur >= num_tasks:
            pool_open = Int32(0)
        # Single-wave pools defer their terminating grab until after scoring.
        deep = Int32(1)
        if num_tasks <= Int32(NUM_CTAS):
            deep = Int32(0)

        # Prime K copies before staging the 32 KB Q tile.
        if tidx == 0:
            if pool_open != Int32(0):
                if deep != Int32(0):
                    s_task[1] = Int32(atomicAdd(p2_counter, Uint32(1))) - task_base
        k_waited = self._p2_k_gate(
            gTickets, cur, num_tasks, gate_task, k_waited, gen, tidx, cta
        )
        if warp >= P2_ISSUE_WARP0:
            if cur < num_tasks:
                self._p2_issue_task(
                    sk_u8,
                    mbar,
                    gIndexKCache,
                    gBlockTable,
                    cur,
                    0,
                    tidx,
                )
        cute.arch.barrier()  # the second task id is visible
        nxt = num_tasks
        if pool_open != Int32(0):
            if deep != Int32(0):
                nxt = s_task[1]
                if nxt >= num_tasks:
                    pool_open = Int32(0)
            else:
                pool_open = Int32(0)
        k_waited = self._p2_k_gate(
            gTickets, nxt, num_tasks, gate_task, k_waited, gen, tidx, cta
        )
        if warp >= P2_ISSUE_WARP0:
            if nxt < num_tasks:
                self._p2_issue_task(
                    sk_u8,
                    mbar,
                    gIndexKCache,
                    gBlockTable,
                    nxt,
                    1,
                    tidx,
                )

        # K copies are in flight; acquire P1's Q rows before staging Q.
        _spin_until_ticket(
            gTickets.iterator + Int32(TICK_Q_READY), Uint32(gen) * Uint32(P1_Q_TASKS)
        )
        fence_proxy_async_global()

        # Only task owners stage the 256x128 FP8 Q tile, with q_offset
        # mapping each 16 B block into tcgen05's SMEM layout.
        q_blocks = Int32(0)
        if cur < num_tasks:
            q_blocks = Int32(_SQ_ROWS * (HEAD_DIM // 16))
        for blk in range(tidx, q_blocks, NUM_THREADS):
            row = blk // (HEAD_DIM // 16)
            col = blk % (HEAD_DIM // 16)
            q_offset = row * Int32(_SQ_PITCH) + col * Int32(16)
            q_offset = self._p2_qk.q_offset(qk_args, row, col * Int32(16))
            cp_async_16_g2s(
                gQFp8.iterator + Int64(row) * Int64(HEAD_DIM) + Int64(col) * Int64(16),
                sq_u8.iterator + q_offset,
            )
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()  # Q tile visible

        # Warps 8-11 do not reduce scores; modulo keeps their loads in bounds.
        sw_row = warp % NUM_TOKENS
        sw_a = gScaledW[sw_row, lane // 4].to(cutlass.Float32)
        sw_b = gScaledW[sw_row, lane // 4 + 8].to(cutlass.Float32)
        sw_c = gScaledW[sw_row, lane // 4 + 16].to(cutlass.Float32)
        sw_d = gScaledW[sw_row, lane // 4 + 24].to(cutlass.Float32)
        causal_len = gSeqLens[0] - Int32(NUM_TOKENS) + Int32(warp) + Int32(1)
        negative_infinity = cutlass.Float32(float("-inf"))
        # BF16 scores are written as packed pairs; the two histograms per
        # row alternate with the generation parity (P0 zeroes the other).
        g_logits_u32 = cute.recast_tensor(gLogits, Uint32)
        hist_parity = gen & Int32(1)

        rS = cute.make_rmem_tensor((4, 2, 4), cutlass.Float32)
        # Reduced column pairs before/after the cross-group butterfly.
        rSum = cute.make_rmem_tensor((2, 4), cutlass.Float32)

        iter_i = Int32(0)
        while cur < num_tasks:
            stage = iter_i % Int32(P2_K_STAGES)
            cute.arch.mbarrier_wait(mbar + stage, (iter_i // Int32(P2_K_STAGES)) % 2)
            if tidx == 0:
                # Overlap the next grab with compute; the previous CTA
                # barrier ended all reads of this task-id slot.
                if pool_open != Int32(0):
                    s_task[stage] = Int32(atomicAdd(p2_counter, Uint32(1))) - task_base
            page_pos0 = cur * Int32(32)

            rS = self._p2_qk.compute(
                qk_args,
                sq_u8.iterator,
                sk_u8.iterator + stage * Int32(self.p2_sk_stage_bytes),
                qk_mbar,
                qk_tmem,
                tidx,
                iter_i & Int32(1),
            )

            if warp < 8:
                for half in cutlass.range_constexpr(32 // HALF_PAGE):
                    if cutlass.const_expr(half > 0):
                        rS = self._p2_qk.read(qk_args, qk_tmem, tidx, half=half)
                    for m in cutlass.range_constexpr(2):
                        wa = sw_a
                        wb = sw_b
                        if m == 1:
                            wa = sw_c
                            wb = sw_d
                        for n in cutlass.range_constexpr(4):
                            j0 = rS[0, m, n]
                            j1 = rS[1, m, n]
                            j2 = rS[2, m, n]
                            j3 = rS[3, m, n]
                            if j0 < cutlass.Float32(0.0):
                                j0 = cutlass.Float32(0.0)
                            if j1 < cutlass.Float32(0.0):
                                j1 = cutlass.Float32(0.0)
                            if j2 < cutlass.Float32(0.0):
                                j2 = cutlass.Float32(0.0)
                            if j3 < cutlass.Float32(0.0):
                                j3 = cutlass.Float32(0.0)
                            if m == 0:
                                rSum[0, n] = wa * j0 + wb * j2
                                rSum[1, n] = wa * j1 + wb * j3
                            else:
                                rSum[0, n] = rSum[0, n] + (wa * j0 + wb * j2)
                                rSum[1, n] = rSum[1, n] + (wa * j1 + wb * j3)
                    for off in cutlass.range_constexpr(3):
                        stride = 4 << off
                        for i in cutlass.range_constexpr(2):
                            for n in cutlass.range_constexpr(4):
                                rSum[i, n] = rSum[i, n] + cute.arch.shuffle_sync_bfly(
                                    rSum[i, n],
                                    offset=stride,
                                    mask=-1,
                                    mask_and_clamp=31,
                                )

                    # Each butterfly group holds the same reduced pairs.
                    # Lanes 0-15 store 32 BF16 scores as consecutive u32 words;
                    # lanes 16-31 hash the same rounded bits into histograms.
                    ks = cute.make_tensor(
                        cute.make_ptr(
                            cutlass.Float32,
                            (
                                sk_u8.iterator
                                + Int32(
                                    stage * self.p2_sk_stage_bytes
                                    + self.p2_sk_values_bytes
                                )
                            ).toint(),
                            cute.AddressSpace.smem,
                            assumed_align=4,
                        ),
                        cute.make_layout((32,)),
                    )
                    nsel = (lane // 4) % 4
                    sa = rSum[0, 0]
                    sb = rSum[1, 0]
                    if nsel == Int32(1):
                        sa = rSum[0, 1]
                        sb = rSum[1, 1]
                    if nsel == Int32(2):
                        sa = rSum[0, 2]
                        sb = rSum[1, 2]
                    if nsel == Int32(3):
                        sa = rSum[0, 3]
                        sb = rSum[1, 3]
                    col0 = Int32(half * HALF_PAGE) + nsel * Int32(8) + (lane % 4) * 2
                    pos0 = page_pos0 + col0
                    v0 = sa * ks[col0]
                    v1 = sb * ks[col0 + 1]
                    if pos0 >= causal_len:
                        v0 = negative_infinity
                    if pos0 + 1 >= causal_len:
                        v1 = negative_infinity
                    # Scrub NaNs to -inf before deriving both score and key.
                    if v0 != v0:
                        v0 = negative_infinity
                    if v1 != v1:
                        v1 = negative_infinity
                    b0 = _bf16_bits_rne(v0)
                    b1 = _bf16_bits_rne(v1)
                    if lane < 16:
                        g_logits_u32[warp, pos0 // 2] = b0 | (b1 << Uint32(16))
                    else:
                        hist_base = (
                            gRowStates.iterator
                            + Int32(warp * ROW_STATE_WORDS + ROW_STATE_HIST_OFF)
                            + hist_parity * Int32(HIST_BINS)
                        )
                        coarse_base = s_coarse.iterator + Int32(warp * COARSE_BINS)
                        if pos0 < causal_len:
                            key0 = Int32(_selection_key(b0, pos0, causal_len))
                            red_add_relaxed_gpu_s32(hist_base + key0, Int32(1))
                            atomicAdd(
                                coarse_base + (key0 >> Int32(COARSE_SHIFT)),
                                Int32(1),
                            )
                        if pos0 + 1 < causal_len:
                            key1 = Int32(_selection_key(b1, pos0 + 1, causal_len))
                            red_add_relaxed_gpu_s32(hist_base + key1, Int32(1))
                            atomicAdd(
                                coarse_base + (key1 >> Int32(COARSE_SHIFT)),
                                Int32(1),
                            )
            cute.arch.barrier()  # stage consumed; safe to re-issue
            new_task = num_tasks
            if pool_open != Int32(0):
                new_task = s_task[stage]
                if new_task >= num_tasks:
                    pool_open = Int32(0)
            k_waited = self._p2_k_gate(
                gTickets, new_task, num_tasks, gate_task, k_waited, gen, tidx, cta
            )
            if warp >= P2_ISSUE_WARP0:
                if new_task < num_tasks:
                    self._p2_issue_task(
                        sk_u8,
                        mbar,
                        gIndexKCache,
                        gBlockTable,
                        new_task,
                        stage,
                        tidx,
                    )
            cur = nxt
            nxt = new_task
            ndone = ndone + Int32(1)
            iter_i = iter_i + Int32(1)
        self._p2_qk.end(qk_tmem, tidx)
        # Single-wave task owners still owe one terminating grab.
        if tidx == 0:
            if deep == Int32(0):
                if first_task < num_tasks:
                    atomicAdd(p2_counter, Uint32(1))
        # Flush nonempty coarse bins under the same completion publication
        # as fine bins and scores; ndone is CTA-uniform.
        if ndone > Int32(0):
            coarse_row_base = Int32(ROW_STATE_COARSE_OFF) + (
                hist_parity * Int32(COARSE_BINS)
            )
            for i in range(tidx, NUM_TOKENS * COARSE_BINS, NUM_THREADS):
                count = s_coarse[i]
                if count != Int32(0):
                    red_add_relaxed_gpu_s32(
                        gRowStates.iterator
                        + (i // Int32(COARSE_BINS)) * Int32(ROW_STATE_WORDS)
                        + coarse_row_base
                        + i % Int32(COARSE_BINS),
                        count,
                    )
        # Join all score/histogram writes, then fence and add this CTA's
        # completed task count to publish them to P3.
        cute.arch.barrier()
        if tidx == 0:
            if ndone > Int32(0):
                fence_acq_rel_gpu()
                red_add_relaxed_gpu_u32(logits_ticket, Uint32(ndone))

    @cute.jit
    def _p2_k_gate(
        self,
        gTickets: cute.Tensor,
        task: Int32,
        num_tasks: Int32,
        gate_task: Int32,
        k_waited: Int32,
        gen: Int32,
        tidx: Int32,
        cta: Int32,
    ) -> Int32:
        """Acquire k_ready once before copying a task >= gate_task.

        Broadcast ids and all arguments are CTA-uniform: all 384 threads
        reach the ticket barrier together. Return the updated k_waited.
        """
        if task < num_tasks:
            if task >= gate_task:
                if k_waited == Int32(0):
                    k_waited = Int32(1)
                    _spin_until_ticket(
                        gTickets.iterator + Int32(TICK_K_READY),
                        Uint32(gen) * Uint32(P1_TOKEN_CTAS),
                    )
                    fence_proxy_async_global()
        return k_waited

    @cute.jit
    def _p2_issue_task(
        self,
        sk_u8: cute.Tensor,
        mbar: cute.Tensor,
        gIndexKCache: cute.Tensor,
        gBlockTable: cute.Tensor,
        task_id: Int32,
        slot: Int32,
        tidx: Int32,
    ) -> None:
        """Fetch one raw index-K half page with 16 B ``cp.async``.

        Warps 8-11 copy task_id=2*page+half into stage slot: 256 value
        blocks via k_offset and eight scale blocks, all 16 B aligned.
        Every issuer ends with cp.async.mbarrier.arrive.noinc; the stage
        mbarrier therefore requires P2_ISSUE_THREADS arrivals.
        """
        qk_args = self._p2_qk.prepare()
        t = tidx - Int32(P2_ISSUE_WARP0 * 32)
        stage = slot % P2_K_STAGES
        page = task_id // self.p2_tasks_per_page
        half = task_id % self.p2_tasks_per_page
        phys = gBlockTable[0, page]
        src_base = gIndexKCache.iterator + Int64(phys) * Int64(INDEX_K_PAGE_BYTES)
        dst_base = sk_u8.iterator + stage * Int32(self.p2_sk_stage_bytes)
        for rep in cutlass.range_constexpr(32 * HEAD_DIM // 16 // P2_ISSUE_THREADS):
            blk = t + Int32(rep * P2_ISSUE_THREADS)
            row = blk // (HEAD_DIM // 16)
            col = blk % (HEAD_DIM // 16)
            src = (
                src_base
                + Int64(half) * Int64(32 * HEAD_DIM)
                + Int64(row) * Int64(HEAD_DIM)
                + Int64(col) * Int64(16)
            )
            k_offset = row * Int32(_SK_PITCH) + col * Int32(16)
            k_offset = self._p2_qk.k_offset(qk_args, row, col * Int32(16))
            dst = dst_base + k_offset
            cp_async_16_g2s(src, dst)
        if t < Int32(8):
            src = (
                src_base
                + Int64(PAGE_SIZE * HEAD_DIM)
                + Int64(half) * Int64(128)
                + Int64(t) * Int64(16)
            )
            dst = dst_base + Int32(self.p2_sk_values_bytes) + t * Int32(16)
            cp_async_16_g2s(src, dst)
        cp_async_mbarrier_arrive_noinc(mbar + stage)

    # P3 histogram Top-K + P4 publication (clusters 0-7, one row each).

    @cute.jit
    def _p3_device(
        self,
        gLogits: cute.Tensor,
        gRowStates: cute.Tensor,
        gSeqLens: cute.Tensor,
        gBlockTable: cute.Tensor,
        gTopk: cute.Tensor,
        gTopkLens: cute.Tensor,
        gTickets: cute.Tensor,
        arena_i32,
        has_peers: cutlass.Constexpr,
        gOutCacheLoc: cute.Tensor,
        gIndexKCache: cute.Tensor,
        gPeerBases: cute.Tensor,
        gGeneration: cute.Tensor,
        tidx: Int32,
        bidy: Int32,
        bidx: Int32,
        gen: Int32,
    ) -> None:
        """P3: histogram Top-K for one causal verify row per two-CTA cluster.

        Length is seq_len-8+row+1. Short rows map positions directly; longer
        rows find a BF16 threshold, count definite slots, then emit them and
        atomically claim ties. Forced positions retain their reserved prefix.
        Outputs are physical cache slots, padded with -1 to TOPK, with
        lens=min(length, TOPK). The final cluster barrier orders both CTAs'
        stores before P4 publishes the row.

        Stage the block table before acquiring logits when its compile-time
        capacity fits P3_BT_WINDOW; otherwise map through the global table.
        """
        cta_in_group = bidx
        group_id = bidy
        chunk_size = cutlass.const_expr(self.chunk_size)
        top_k = cutlass.const_expr(TOPK)
        next_n = cutlass.const_expr(NUM_TOKENS)
        num_pages = cutlass.const_expr(gBlockTable.shape[1])
        bt_staged = cutlass.const_expr(num_pages <= P3_BT_WINDOW)

        s_block_table = cute.make_tensor(
            cute.make_ptr(
                Int32,
                arena_i32 + Int32(p3_bt_offset(chunk_size)),
                cute.AddressSpace.smem,
                assumed_align=128,
            ),
            cute.make_ordered_layout((min(num_pages, P3_BT_WINDOW),), order=(0,)),
        )
        if cutlass.const_expr(bt_staged):
            for i in range(tidx, num_pages, NUM_THREADS):
                s_block_table[i] = gBlockTable[0, i]
            cute.arch.barrier()

        # P2 contributes one ticket per tile of each live page; the slot
        # is per generation parity.
        live_pages = (gSeqLens[0] + Int32(PAGE_SIZE - 1)) // Int32(PAGE_SIZE)
        if live_pages > Int32(num_pages):
            live_pages = Int32(num_pages)
        logits_ticket = gTickets.iterator + Int32(TICK_LOGITS)
        if (gen & Int32(1)) != Int32(0):
            logits_ticket = gTickets.iterator + Int32(TICK_LOGITS_ODD)
        need_logits = Int32(1)
        if gSeqLens[0] <= Int32(TOPK):
            need_logits = Int32(0)
        if need_logits != Int32(0):
            _spin_until_ticket(
                logits_ticket, Uint32(self.p2_tasks_per_page) * Uint32(live_pages)
            )

        if cutlass.const_expr(bt_staged):
            slot_table = s_block_table
        else:
            slot_table = gBlockTable[0, None]

        # Scratch: 256 coarse counts, threshold scalars and 12 warp counts.
        s_block_sums = cute.make_tensor(
            cute.make_ptr(
                Int32,
                arena_i32 + Int32(_P3_HIST_OFF),
                cute.AddressSpace.smem,
                assumed_align=128,
            ),
            cute.make_ordered_layout((256,), order=(0,)),
        )
        # [0] threshold key t, [1] count of keys > t (gt_count),
        # [2] ties still needed k*, [3] this CTA's definite-slot base,
        # [4] this CTA's local definite counter.
        s_scalars = cute.make_tensor(
            cute.make_ptr(
                Int32,
                arena_i32 + Int32(_P3_SCAL_OFF),
                cute.AddressSpace.smem,
                assumed_align=16,
            ),
            cute.make_ordered_layout((8,), order=(0,)),
        )
        s_warp_sums = cute.make_tensor(
            cute.make_ptr(
                Int32,
                arena_i32 + Int32(_P3_WARP_OFF),
                cute.AddressSpace.smem,
                assumed_align=128,
            ),
            cute.make_ordered_layout((NUM_WARPS,), order=(0,)),
        )

        row = group_id
        row_state = gRowStates[row, None]
        seq_len = gSeqLens[0]
        length = seq_len - next_n + (row % next_n) + 1
        if length < 0:
            length = cutlass.Int32(0)
        lane = tidx % 32
        warp = tidx // 32

        # Each CTA filters a contiguous half, rounded to 8-score vectors.
        chunk = (
            (length + Int32(CTAS_PER_CLUSTER * 8 - 1))
            // Int32(CTAS_PER_CLUSTER * 8)
            * Int32(8)
        )
        chunk_start = cta_in_group * chunk
        chunk_end = chunk_start + chunk
        if chunk_end > length:
            chunk_end = length

        valid_count = length
        if valid_count > Int32(TOPK):
            valid_count = Int32(TOPK)

        g_logits_row_u32 = cute.recast_tensor(gLogits, Uint32)[row, None]
        output_indices_row = gTopk[row, None]
        if top_k >= length:
            # Identity path: forced prefix first, then middle positions in
            # row order; the mapping remains a bijection onto [0, length).
            for index in range(chunk_start + tidx, chunk_end, NUM_THREADS):
                slot = _forced_index(index, length)
                if slot < Int32(0):
                    slot = Int32(FORCED_TOKENS) + index - Int32(INITIAL_TOKENS)
                if slot < top_k:
                    emit_topk_index(
                        output_indices_row,
                        slot,
                        index,
                        slot_table,
                        PAGE_SHIFT,
                    )
            if cta_in_group == 0:
                for index in range(tidx + length, top_k, NUM_THREADS):
                    output_indices_row[index] = cutlass.Int32(-1)
        else:
            # Each CTA independently finds the threshold, avoiding a
            # cluster barrier before filtering.
            hist_off = Int32(ROW_STATE_HIST_OFF) + (gen & Int32(1)) * Int32(HIST_BINS)
            # Read P2's 1 KB coarse histogram, then only the crossing block.
            coarse_off = Int32(ROW_STATE_COARSE_OFF) + (
                (gen & Int32(1)) * Int32(COARSE_BINS)
            )
            if tidx < Int32(COARSE_BINS // 4):
                ptr = cute.make_ptr(
                    Uint32,
                    (row_state.iterator + coarse_off + tidx * Int32(4)).toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                vec = load_global_u32x4(ptr)
                for q in cutlass.range_constexpr(4):
                    s_block_sums[tidx * Int32(4) + Int32(q)] = Int32(vec[q])
            cute.arch.barrier()
            # Warp 0: descending suffix scan over the 256 block sums.  Lane
            # ``l`` owns blocks [255 - 8l - 7, 255 - 8l] (top to bottom), so
            # a shuffle prefix over lanes gives the count above each block.
            if warp == 0:
                lane_sum = Int32(0)
                for j in cutlass.range_constexpr(8):
                    lane_sum = lane_sum + s_block_sums[255 - lane * 8 - j]
                excl = _warp_exclusive_sum(lane_sum, lane)
                running = excl
                found_block = Int32(-1)
                found_before = Int32(0)
                for j in cutlass.range_constexpr(8):
                    b = 255 - lane * 8 - j
                    cnt = s_block_sums[b]
                    # Both tests are needed to identify the unique crossing.
                    if running < Int32(top_k):
                        if running + cnt >= Int32(top_k):
                            found_block = b
                            found_before = running
                    running = running + cnt
                if found_block >= 0:
                    s_scalars[5] = found_block
                    s_scalars[6] = found_before
            cute.arch.barrier()
            # Warp 0 again: the 256 bins of the found block, descending.
            if warp == 0:
                blk = s_scalars[5]
                before = s_scalars[6]
                bin_base = hist_off + blk * Int32(256)
                bins = cute.make_rmem_tensor((8,), Int32)
                for j in cutlass.range_constexpr(8):
                    bins[j] = row_state[
                        bin_base + Int32(255) - lane * Int32(8) - Int32(j)
                    ]
                lane_sum = Int32(0)
                for j in cutlass.range_constexpr(8):
                    lane_sum = lane_sum + bins[j]
                excl = _warp_exclusive_sum(lane_sum, lane)
                running = before + excl
                for j in cutlass.range_constexpr(8):
                    key = blk * Int32(256) + Int32(255) - lane * Int32(8) - Int32(j)
                    cnt = bins[j]
                    if running < Int32(top_k):
                        if running + cnt >= Int32(top_k):
                            s_scalars[0] = key
                            s_scalars[1] = running
                            s_scalars[2] = Int32(top_k) - running
                    running = running + cnt
                if lane == 0:
                    s_scalars[4] = Int32(0)
            cute.arch.barrier()
            t_key = Uint32(s_scalars[0])
            gt_count = s_scalars[1]
            # Forced +inf keys still own the prefix when the threshold is +inf.
            tie_base = gt_count
            if t_key == Uint32(KEY_INF):
                tie_base = Int32(FORCED_TOKENS)
            k_star = Int32(top_k) - tie_base

            # Pass 1 counts non-forced definite slots in 8-score vectors.
            # length > TOPK > FORCED_TOKENS reserves the full forced prefix.
            n_vec = (chunk_end - chunk_start + Int32(7)) // Int32(8)
            my_count = Int32(0)
            for v in range(tidx, n_vec, NUM_THREADS):
                pos_base = chunk_start + v * Int32(8)
                vec = _load_score_vector(g_logits_row_u32, pos_base)
                for j in cutlass.range_constexpr(8):
                    pos = pos_base + Int32(j)
                    bits = (vec[j // 2] >> Uint32(16 * (j % 2))) & Uint32(0xFFFF)
                    if pos < chunk_end:
                        if _forced_index(pos, length) < Int32(0):
                            if _selection_key(bits, pos, length) > t_key:
                                my_count = my_count + Int32(1)
            for step in cutlass.range_constexpr(5):
                my_count = my_count + cute.arch.shuffle_sync_bfly(
                    my_count, offset=1 << step, mask=-1, mask_and_clamp=31
                )
            if lane == 0:
                s_warp_sums[warp] = my_count
            cute.arch.barrier()
            if tidx == 0:
                total = Int32(0)
                for w in cutlass.range_constexpr(NUM_WARPS):
                    total = total + s_warp_sums[w]
                base = Int32(0)
                if total > 0:
                    base = atomicAdd(row_state.iterator, total)
                s_scalars[3] = base
            cute.arch.barrier()
            def_base = s_scalars[3]

            # Pass 2 emits definites and the first k* atomic tie claims.
            for v in range(tidx, n_vec, NUM_THREADS):
                pos_base = chunk_start + v * Int32(8)
                vec = _load_score_vector(g_logits_row_u32, pos_base)
                for j in cutlass.range_constexpr(8):
                    pos = pos_base + Int32(j)
                    bits = (vec[j // 2] >> Uint32(16 * (j % 2))) & Uint32(0xFFFF)
                    if pos < chunk_end:
                        slot = _forced_index(pos, length)
                        key = _selection_key(bits, pos, length)
                        if slot >= Int32(0):
                            emit_topk_index(
                                output_indices_row,
                                slot,
                                pos,
                                slot_table,
                                PAGE_SHIFT,
                            )
                        elif key > t_key:
                            local = atomicAdd(s_scalars.iterator + 4, Int32(1))
                            # gt_count includes forced keys; non-forced
                            # definites fill [FORCED_TOKENS, gt_count).
                            emit_topk_index(
                                output_indices_row,
                                Int32(FORCED_TOKENS) + def_base + local,
                                pos,
                                slot_table,
                                PAGE_SHIFT,
                            )
                        elif key == t_key:
                            tie = atomicAdd(row_state.iterator + 1, Int32(1))
                            if tie < k_star:
                                emit_topk_index(
                                    output_indices_row,
                                    tie_base + tie,
                                    pos,
                                    slot_table,
                                    PAGE_SHIFT,
                                )

        if cta_in_group == 0 and tidx == 0:
            gTopkLens[row] = valid_count
        # Release cluster arrival orders both CTAs' slot stores before
        # CTA 0's row completion and each CTA's P4 publication.
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        if cta_in_group == 0 and tidx == 0:
            # Fence the final row before relaxed completion publication.
            fence_acq_rel_gpu()
            # Cluster barrier retires all filter atomics before resetting
            # counters for the next launch; step 1 starts zeroed.
            row_state[0] = Int32(0)
            row_state[1] = Int32(0)
        if cta_in_group == 0 and tidx == 0:
            # Eight row completions per generation; standalone's last row
            # advances gGeneration.
            prev_row = Uint32(0)
            if cutlass.const_expr(not has_peers):
                prev_row = atomic_add_acq_rel_gpu_u32(
                    gTickets.iterator + Int32(TICK_ROWS), Uint32(1)
                )
            else:
                red_add_relaxed_gpu_u32(gTickets.iterator + Int32(TICK_ROWS), Uint32(1))
            if cutlass.const_expr(not has_peers):
                if prev_row % Uint32(NUM_TOKENS) == Uint32(NUM_TOKENS - 1):
                    gGeneration[0] = (Uint32(gen) + Uint32(1)).to(Int32)
        if cutlass.const_expr(has_peers):
            self._p4_device(
                gTopk,
                gOutCacheLoc,
                gIndexKCache,
                gPeerBases,
                gTickets,
                gGeneration,
                arena_i32,
                row,
                valid_count,
                cta_in_group,
                tidx,
                gen,
            )

    @cute.jit
    def _p4_device(
        self,
        gTopk: cute.Tensor,
        gOutCacheLoc: cute.Tensor,
        gIndexKCache: cute.Tensor,
        gPeerBases: cute.Tensor,
        gTickets: cute.Tensor,
        gGeneration: cute.Tensor,
        arena_i32,
        row: Int32,
        length: Int32,
        cta_in_group: Int32,
        tidx: Int32,
        gen: Int32,
    ) -> None:
        """P4: the row's selection publication, by the row's own cluster.

        After P3's cluster barrier, CTA c copies to peers starting at
        1+P4_PEERS_PER_CTA*c. TICK_PUB counts 56 row/peer publications per
        step; the closing CTA publishes all seven flags and advances gen.
        """
        first = Int32(1) + cta_in_group * Int32(P4_PEERS_PER_CTA)
        count = Int32(NUM_PEERS) - cta_in_group * Int32(P4_PEERS_PER_CTA)
        if count > Int32(P4_PEERS_PER_CTA):
            count = Int32(P4_PEERS_PER_CTA)
        s_closing = cute.make_tensor(
            cute.make_ptr(
                Int32,
                arena_i32 + Int32(_P3_SCAL_OFF),
                cute.AddressSpace.smem,
                assumed_align=16,
            ),
            cute.make_layout((1,)),
        )
        self._p4_publish_row(
            gTopk,
            gOutCacheLoc,
            gIndexKCache,
            gPeerBases,
            row,
            length,
            first,
            count,
            tidx,
            gen,
        )
        cute.arch.barrier()
        # Barrier + acq_rel add joins every CTA's P2P stores. The closing
        # CTA's flag lanes system-fence this causal chain before flag stores.
        if tidx == 0:
            prev = atomic_add_acq_rel_gpu_u32(
                gTickets.iterator + Int32(TICK_PUB), Uint32(count)
            )
            # Only this generation's final publication reaches its wrapped target.
            if prev + Uint32(count) == Uint32(gen) * Uint32(PUBS_PER_STEP):
                s_closing[0] = Int32(1)
            else:
                s_closing[0] = Int32(0)
        cute.arch.barrier()
        if s_closing[0] != Int32(0):
            cta = row * Int32(CTAS_PER_CLUSTER) + cta_in_group
            # Each of seven lanes system-fences before its peer flag.
            if tidx < NUM_PEERS:
                fence_acq_rel_sys()
                words = _sx.payload_words(INDEX_K_ROW_WORDS)
                flag_off = Int64(_sx.DOUBLE_SLOTS * words * 4)
                flag = cute.make_ptr(
                    Uint32,
                    gPeerBases[tidx + 1] + flag_off,
                    cute.AddressSpace.gmem,
                    assumed_align=4,
                )
                store_relaxed_sys_u32(flag, Uint32(gen))
            if tidx == 0:
                gGeneration[0] = (Uint32(gen) + Uint32(1)).to(Int32)

    @cute.jit
    def _p4_publish_row(
        self,
        gTopk: cute.Tensor,
        gOutCacheLoc: cute.Tensor,
        gIndexKCache: cute.Tensor,
        gPeerBases: cute.Tensor,
        row: Int32,
        length: Int32,
        first: Int32,
        count: Int32,
        tidx: Int32,
        gen: Int32,
    ) -> None:
        """Copy a verify row to peers [first, first+count), at most four.

        selection_exchange has two gen&1 slots, each containing
        [8x2048 indices | 8 lengths | 8x33 index-K words]. Indices/lengths
        carry generation tags; index-K (128 FP8 bytes + FP32 scale) does not.
        Load the whole row before peer stores to avoid alias serialization.
        """
        slot = gen & 1
        words = _sx.payload_words(INDEX_K_ROW_WORDS)
        # Top byte: generation tag; low 24 bits: value. Workers can acquire
        # individual tagged words before the final system-scope flag.
        tag_bits = (Uint32(gen) & Uint32(_sx.TAG_MASK)) << Uint32(_sx.TAG_SHIFT)
        g_topk_u32 = cute.recast_tensor(gTopk, cutlass.Uint32)
        row_words = g_topk_u32[row, None]
        vecs = cute.make_rmem_tensor((P4_VECS, 4), Uint32)
        for v in cutlass.range_constexpr(P4_VECS):
            i = tidx * Int32(4) + Int32(v * NUM_THREADS * 4)
            if i < Int32(TOPK):
                src = cute.make_ptr(
                    Uint32,
                    (row_words.iterator + i).toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                vec = load_global_u32x4(src)
                for j in cutlass.range_constexpr(4):
                    vecs[v, j] = (vec[j] & Uint32(_sx.VALUE_MASK)) | tag_bits
        index_k_word = Uint32(0)
        if tidx < INDEX_K_ROW_WORDS:
            cache_words = cute.recast_tensor(gIndexKCache, cutlass.Uint32)
            slot_id = Int64(gOutCacheLoc[row])
            page = slot_id // Int64(PAGE_SIZE)
            page_off = slot_id - page * Int64(PAGE_SIZE)
            # Paged 132 B rows: fp8 words at page*2112 + off*32, the fp32
            # scale word at page*2112 + 2048 + off.
            src_word = (
                page * Int64(PAGE_SIZE * INDEX_K_ROW_WORDS)
                + Int64(PAGE_SIZE * HEAD_DIM // 4)
                + page_off
            )
            if tidx < HEAD_DIM // 4:
                src_word = (
                    page * Int64(PAGE_SIZE * INDEX_K_ROW_WORDS)
                    + page_off * Int64(HEAD_DIM // 4)
                    + Int64(tidx)
                )
            index_k_word = cache_words[src_word]
        for p in cutlass.range_constexpr(P4_PEERS_PER_CTA):
            if Int32(p) < count:
                base = gPeerBases[first + Int32(p)] + Int64(slot * words * 4)
                for v in cutlass.range_constexpr(P4_VECS):
                    i = tidx * Int32(4) + Int32(v * NUM_THREADS * 4)
                    if i < Int32(TOPK):
                        tagged = cute.make_rmem_tensor((4,), Uint32)
                        for j in cutlass.range_constexpr(4):
                            tagged[j] = vecs[v, j]
                        store_global_u32x4(
                            base + Int64((row * Int32(TOPK) + i) * 4), tagged.load()
                        )
                if tidx == 0:
                    _store_published_word(
                        base + Int64((_sx.INDICES_WORDS + row) * 4),
                        Uint32(length) | tag_bits,
                    )
                if tidx < INDEX_K_ROW_WORDS:
                    _store_published_word(
                        base
                        + Int64(
                            (
                                _sx.INDICES_WORDS
                                + _sx.LENS_WORDS
                                + row * INDEX_K_ROW_WORDS
                                + tidx
                            )
                            * 4
                        ),
                        index_k_word,
                    )

    # Kernel entry

    @cute.kernel
    def kernel(
        self,
        gRmsW: cute.Tensor,
        gCosSin: cute.Tensor,
        gPositions: cute.Tensor,
        gOutCacheLoc: cute.Tensor,
        gIndexKCache: cute.Tensor,
        gQkvOut: cute.Tensor,
        gRawQ: cute.Tensor,
        gRawK: cute.Tensor,
        gQFp8: cute.Tensor,
        gScaledW: cute.Tensor,
        gSeqLens: cute.Tensor,
        gBlockTable: cute.Tensor,
        gTopk: cute.Tensor,
        gTopkLens: cute.Tensor,
        gLogits: cute.Tensor,
        gTickets: cute.Tensor,
        gRowStates: cute.Tensor,
        gScratch: cute.Tensor,
        gPeerBases: cute.Tensor,
        gGeneration: cute.Tensor,
        gResidual: cute.Tensor,
        gGamma: cute.Tensor,
        gResidualOut: cute.Tensor,
        gNormOut: cute.Tensor,
        gQuantOut: Optional[cute.Tensor],
        gQuantScale: Optional[cute.Tensor],
        gQuantGlobal: Optional[cute.Tensor],
        gQuant2Out: Optional[cute.Tensor],
        gQuant2Scale: Optional[cute.Tensor],
        gQuant2Global: Optional[cute.Tensor],
        gTransportBases: cute.Tensor,
        gRotationFlags: cute.Tensor,
        qkv_operands: tuple,
        wk_operands: tuple,
        qb_operands: tuple,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        cta = bidy * CTAS_PER_CLUSTER + bidx
        cute.arch.griddepcontrol_wait()
        gen = cutlass.Int32(gGeneration[0])
        # Zero only the next parity while P2 uses gen&1. The previous
        # launch prepared this parity; the workspace initially starts zeroed.
        next_parity = ((Uint32(gen) + Uint32(1)) & Uint32(1)).to(Int32)
        zero_off = Int32(ROW_STATE_HIST_OFF) + next_parity * Int32(HIST_BINS)
        for i in range(
            cta * NUM_THREADS + tidx, NUM_TOKENS * HIST_BINS, NUM_CTAS * NUM_THREADS
        ):
            gRowStates[i // Int32(HIST_BINS), zero_off + i % Int32(HIST_BINS)] = Int32(
                0
            )
        # Same for the coarse bins (2048 words: at most one per thread).
        coarse_zero_off = Int32(ROW_STATE_COARSE_OFF) + next_parity * Int32(COARSE_BINS)
        for i in range(
            cta * NUM_THREADS + tidx,
            NUM_TOKENS * COARSE_BINS,
            NUM_CTAS * NUM_THREADS,
        ):
            gRowStates[
                i // Int32(COARSE_BINS), coarse_zero_off + i % Int32(COARSE_BINS)
            ] = Int32(0)

        smem = SmemAllocator()
        arena_raw = smem.allocate(self.arena_bytes, byte_alignment=self.arena_alignment)
        arena_i32 = arena_raw.toint()
        # Hosted routines need this base alignment for vectorized SMEM.
        arena = cute.make_ptr(
            cutlass.Int8, arena_i32, cute.AddressSpace.smem, assumed_align=128
        )
        # P2 pipeline mbarriers follow its Q/K storage.
        mbar = cute.make_ptr(
            cutlass.Int64,
            arena_i32 + Int32(self.p2_mbar_off),
            cute.AddressSpace.smem,
            assumed_align=8,
        )
        has_peers = True

        # Inline P0 to keep operand layouts static for vectorized epilogues.
        lora_ticket = cute.make_tensor(
            gTickets.iterator + Int32(TICK_LORA), cute.make_layout((1,))
        )
        proj_ticket = cute.make_tensor(
            gTickets.iterator + Int32(TICK_PROJ), cute.make_layout((1,))
        )
        # P1 acquires disjoint tile slots; QKV-A starts at gScratch[0].
        scratch_wk = cute.make_tensor(
            gScratch.iterator + Int32(SLOT_WK), cute.make_layout((WK_TILES,))
        )
        scratch_wqb = cute.make_tensor(
            gScratch.iterator + Int32(SLOT_WQ_B), cute.make_layout((WQ_B_TILES,))
        )
        # wq_b reads the q_lora prefix with the full QKV-A row stride.
        mA_lora = cute.make_tensor(
            gQkvOut.iterator,
            cute.make_layout((NUM_TOKENS, QKV_LATENT_OFFSET), stride=(QKV_OUT, 1)),
        )

        p0_threads = STREAM_P0_THREADS
        # Pass 0: q_lora/wk projections; remaining warps prefetch.
        if tidx < Int32(p0_threads):
            w_policy = make_l2_policy_evict_first()
            mbar_p0 = cute.make_ptr(
                Int64,
                arena_i32
                + Int32(
                    gemv_mbar_offset(QKV_A_TILE_N, self.p0_stages, self.w_bytes, 512)
                ),
                cute.AddressSpace.smem,
                assumed_align=8,
            )
            slot = Int32(0)
            if cta < Int32(QKV_A_TILES):
                stream_gemv_tile(
                    *qkv_operands,
                    gQkvOut,
                    cta * Int32(QKV_A_TILE_N),
                    arena,
                    mbar_p0,
                    tidx,
                    HIDDEN,
                    QKV_A_TILE_N,
                    stages=self.p0_stages,
                    threads=STREAM_P0_THREADS,
                    k_chunk=512,
                    weight_policy=w_policy,
                )
                slot = Int32(SLOT_QKV_A) + cta
            elif cta < Int32(P0_TILES):
                wk_tile = cta - Int32(QKV_A_TILES)
                stream_gemv_tile(
                    *wk_operands,
                    gRawK,
                    wk_tile * Int32(WK_TILE_N),
                    arena,
                    mbar_p0,
                    tidx,
                    HIDDEN,
                    WK_TILE_N,
                    stages=self.p0_stages,
                    threads=STREAM_P0_THREADS,
                    k_chunk=512,
                    weight_policy=w_policy,
                )
                slot = Int32(SLOT_WK) + wk_tile
            if cta < Int32(P0_TILES):
                # Join stores, then fence before publishing the generation
                # tile slot and relaxed completion counts.
                cute.arch.barrier(
                    barrier_id=BAR_GEMV, number_of_threads=STREAM_P0_THREADS
                )
                if tidx == 0:
                    fence_acq_rel_gpu()
                    store_relaxed_gpu_u32(gScratch.iterator + slot, Uint32(gen))
                    red_add_relaxed_gpu_u32(proj_ticket.iterator, Uint32(1))
                    if cta < Int32(QKV_A_TILES):
                        red_add_relaxed_gpu_u32(lora_ticket.iterator, Uint32(1))
        else:
            # Prefetch 16 B-aligned 8448 B live index-K pages, striped by
            # CTA. L2 remains coherent with P1's later fresh-row writes.
            if tidx == Int32(p0_threads):
                live_pages = (gSeqLens[0] + Int32(PAGE_SIZE - 1)) // Int32(PAGE_SIZE)
                if live_pages > gBlockTable.shape[1]:
                    live_pages = Int32(gBlockTable.shape[1])
                page = cta
                while page < live_pages:
                    prefetch_bulk_l2(
                        cute.make_ptr(
                            Uint8,
                            Int64(gIndexKCache.iterator.toint())
                            + Int64(gBlockTable[0, page]) * Int64(INDEX_K_PAGE_BYTES),
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        Uint32(INDEX_K_PAGE_BYTES),
                    )
                    page += Int32(NUM_CTAS)
            # Warm this parity's histogram in 16 KB chunks striped by CTA.
            if tidx >= Int32(p0_threads):
                if tidx < Int32(p0_threads + 4):
                    chunk = cta + (tidx - Int32(p0_threads)) * Int32(NUM_CTAS)
                    if chunk < Int32(NUM_TOKENS * HIST_CHUNKS_PER_ROW):
                        row = chunk // Int32(HIST_CHUNKS_PER_ROW)
                        word0 = (
                            row * Int32(ROW_STATE_WORDS)
                            + Int32(ROW_STATE_HIST_OFF)
                            + (gen & Int32(1)) * Int32(HIST_BINS)
                            + (chunk % Int32(HIST_CHUNKS_PER_ROW))
                            * Int32(HIST_CHUNK_WORDS)
                        )
                        prefetch_bulk_l2(
                            cute.make_ptr(
                                Uint8,
                                Int64(gRowStates.iterator.toint())
                                + Int64(word0) * Int64(4),
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            ),
                            Uint32(HIST_CHUNK_WORDS * 4),
                        )
        # Pass 1: each wq_b tile waits for all 96 q_lora tiles.
        if cta < Int32(WQ_B_TILES):
            if tidx < Int32(p0_threads):
                stream_gemv_tile(
                    *qb_operands,
                    gRawQ,
                    cta * Int32(WQ_B_TILE_N),
                    arena,
                    cute.make_ptr(
                        Int64,
                        arena_i32
                        + Int32(
                            gemv_mbar_offset(
                                WQ_B_TILE_N, STREAM_QB_STAGES, self.w_bytes
                            )
                        ),
                        cute.AddressSpace.smem,
                        assumed_align=8,
                    ),
                    tidx,
                    QKV_LATENT_OFFSET,
                    WQ_B_TILE_N,
                    stages=STREAM_QB_STAGES,
                    threads=STREAM_P0_THREADS,
                    weight_policy=make_l2_policy_evict_first(),
                    x_ready=lora_ticket.iterator,
                    x_target=Uint32(gen) * Uint32(LORA_PER_STEP),
                )
                cute.arch.barrier(
                    barrier_id=BAR_GEMV, number_of_threads=STREAM_P0_THREADS
                )
                if tidx == 0:
                    fence_acq_rel_gpu()
                    store_relaxed_gpu_u32(
                        gScratch.iterator + Int32(SLOT_WQ_B) + cta, Uint32(gen)
                    )
                    red_add_relaxed_gpu_u32(proj_ticket.iterator, Uint32(1))

        s_warp = cute.make_tensor(
            cute.make_ptr(
                cutlass.Float32, arena_i32, cute.AddressSpace.smem, assumed_align=16
            ),
            cute.make_layout((NUM_WARPS,)),
        )
        s_task = cute.make_tensor(
            cute.make_ptr(
                Int32,
                arena_i32 + Int32(_P1_TASK_OFF),
                cute.AddressSpace.smem,
                assumed_align=16,
            ),
            cute.make_layout((1 + P1_GROUPS,)),
        )
        self._p1_device(
            gRawQ,
            gRawK,
            gRmsW,
            gCosSin,
            gPositions,
            gOutCacheLoc,
            gIndexKCache,
            gQFp8,
            gScaledW,
            gTickets,
            gScratch,
            s_warp,
            s_task,
            tidx,
            cta,
            gen,
        )
        score_logits = Int32(1)
        if gSeqLens[0] <= Int32(TOPK):
            score_logits = Int32(0)
        if score_logits == Int32(0):
            # Keep the next generation's counters empty across short/long
            # transitions, just as P2 does when it scores the current row.
            if tidx == 0 and cta == Int32(0):
                if (gen & Int32(1)) != Int32(0):
                    gTickets[TICK_P2_TASK] = Uint32(0)
                    gTickets[TICK_LOGITS] = Uint32(0)
                else:
                    gTickets[TICK_P2_TASK_ODD] = Uint32(0)
                    gTickets[TICK_LOGITS_ODD] = Uint32(0)
            # P4 copies fresh index-K rows. P2 normally provides this
            # acquire transitively through its logits completion ticket.
            _spin_until_ticket(
                gTickets.iterator + Int32(TICK_Q_READY),
                Uint32(gen) * Uint32(P1_Q_TASKS),
            )
            _spin_until_ticket(
                gTickets.iterator + Int32(TICK_K_READY),
                Uint32(gen) * Uint32(P1_TOKEN_CTAS),
            )
        else:
            self._p2_device(
                gQFp8,
                gScaledW,
                gIndexKCache,
                gSeqLens,
                gBlockTable,
                gLogits,
                gRowStates,
                gTickets,
                arena_i32,
                mbar,
                tidx,
                cta,
                gen,
            )
        if bidy < 8:
            self._p3_device(
                gLogits,
                gRowStates,
                gSeqLens,
                gBlockTable,
                gTopk,
                gTopkLens,
                gTickets,
                arena_i32,
                has_peers,
                gOutCacheLoc,
                gIndexKCache,
                gPeerBases,
                gGeneration,
                tidx,
                bidy,
                bidx,
                gen,
            )
        cute.arch.griddepcontrol_launch_dependents()
        cute.arch.barrier()
        if bidy < 8:
            self._p5_device(
                gResidual,
                gGamma,
                gResidualOut,
                gNormOut,
                gQuantOut,
                gQuantScale,
                gQuantGlobal,
                gQuant2Out,
                gQuant2Scale,
                gQuant2Global,
                gTransportBases,
                gRotationFlags,
                arena_i32,
                tidx,
                bidy,
                bidx,
            )

    # P5: AR consumer + residual + RMSNorm (clusters 0-7, one token each)

    @cute.jit
    def _p5_device(
        self,
        gResidual: cute.Tensor,
        gGamma: cute.Tensor,
        gResidualOut: cute.Tensor,
        gNormOut: cute.Tensor,
        gQuantOut: Optional[cute.Tensor],
        gQuantScale: Optional[cute.Tensor],
        gQuantGlobal: Optional[cute.Tensor],
        gQuant2Out: Optional[cute.Tensor],
        gQuant2Scale: Optional[cute.Tensor],
        gQuant2Global: Optional[cute.Tensor],
        gTransportBases: cute.Tensor,
        gRotationFlags: cute.Tensor,
        arena_i32,
        tidx: Int32,
        bidy: Int32,
        bidx: Int32,
    ) -> None:
        """Consume seven workers' o_proj payloads for token bidy, CTA bidx.

        All warps participate in reduction/residual/RMSNorm; rank 0 has no
        o_proj payload to publish.
        """
        token = bidy
        slot = load_acquire_gpu_u32(gRotationFlags.iterator)
        # transport_bases[0] is the multicast VA (producers only); [1] is
        # this rank's unicast alias where the replicas landed.
        local_base = gTransportBases[1]
        # P5 reuses offset 0 after kernel's barrier retires P2/P3 storage.
        norm_scratch = cute.make_ptr(
            Float32, arena_i32, cute.AddressSpace.smem, assumed_align=16
        )
        comm_tid = Int32(tidx)
        if tidx >= P5_COMM_THREADS:
            comm_tid = Int32(-1)
        self.collective.consume_reduce_norm(
            comm_tid,
            Int32(tidx),
            slot,
            local_base,
            Int32(token * CTAS_PER_CLUSTER + bidx),
            Int32(NUM_TOKENS * CTAS_PER_CLUSTER),
            token,
            bidx,
            gResidual,
            gGamma,
            gResidualOut,
            gNormOut,
            norm_scratch,
            gRotationFlags,
            P5_COMM_THREADS,
            Int32(NUM_THREADS),
            Int32(BAR_P5_NORM),
            Int32(NUM_TOKENS * CTAS_PER_CLUSTER),
            gQuantOut if self.quant_outputs >= 1 else None,
            gQuantScale if self.quant_outputs >= 1 else None,
            gQuantGlobal if self.quant_outputs >= 1 else None,
            gQuant2Out if self.quant_outputs >= 2 else None,
            gQuant2Scale if self.quant_outputs >= 2 else None,
            gQuant2Global if self.quant_outputs >= 2 else None,
            rotate_relaxed=True,
        )

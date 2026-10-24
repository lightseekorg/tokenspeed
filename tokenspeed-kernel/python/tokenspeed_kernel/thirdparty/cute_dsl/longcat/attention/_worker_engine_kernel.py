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

"""Attention-0 workers for verify-8; see LongcatWorkerEngineKernel."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint8, Uint32, utils
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention import (
    _o_proj_routine as _oproj,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention import (
    selection_exchange as _sx,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention._indexer_engine_kernel import (
    PAGE_SHIFT,
    PAGE_SIZE,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention._o_proj_collective import (
    SLOT_BYTES,
    LongcatOProjCollective,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention._primitives import (
    cpasync_bulk_g2s,
    make_l2_policy_evict_first,
    make_l2_policy_evict_normal,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention._sparse_mla_hetero_kernel import (
    ELEMS_16BIT_PITCH,
    FORCED_TOKENS,
    INITIAL_ROW,
    INITIAL_TOKENS,
    NUM_TOKENS,
    PARTIAL_FLOATS,
    PHASE_A_ROWS,
    PHASE_A_TILES,
    RAW_PITCH_FP8_ELEMS,
    SLOTS_PER_CTA,
    TILE_KV,
    TILES_PER_CTA,
    TOPK,
    LongcatSparseMlaHeteroKernel,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention._stream_gemv import (
    mbar_offset as gemv_mbar_offset,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention._stream_gemv import (
    stream_gemv_tile,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._attention1_finalize import (
    LongcatAttention1Finalize,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._collective_primitives import (
    fence_proxy_async_global,
    load_acquire_sys_u32,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._primitives import (
    atomic_add_acq_rel_gpu_u32,
    atomic_add_gpu_u32,
    load_acquire_gpu_u32,
    prefetch_bulk_l2,
    red_add_relaxed_gpu_u32,
    red_add_release_gpu_u32,
    store_relaxed_gpu_u32,
)
from tokenspeed_kernel.thirdparty.cute_dsl.topk.top_k.single_pass_multi_cta_radix_topk import (
    fence_acq_rel_gpu,
)

# Geometry.
CLUSTER_CTAS = 4
NUM_CLUSTERS = 32
NUM_CTAS = 128
THREADS_PER_CTA = 384
ROUTINE_THREADS = 192  # warps 0-5 host every corridor routine
ENGINE_THREADS = 192  # warps 6-11
# P7: warps 4-11 consume after o_proj publication; cluster*warps <= 32
# caps the DSMEM exchange at eight warps for a four-CTA cluster.
P7_COMM_THREADS = 256

# P0: one independent 24-column tile per CTA; 88 project QKV-A,
# of which the first 64 produce q_a for P2.
P0_FAST_TILE_N = 24
OPROJ_MBAR_W = TILES_PER_CTA  # zone C mbarrier slots past the gather tiles

Q_LORA = 1536
KV_LORA = 512
QKV_WIDTH = 2112
HEAD_DIM_V = 128
Q_B_ROPE = 64
KV_ROW = 576  # 512 latent + 64 RoPE channels
HIDDEN = 8192

QKV_NORM_EPS = 1e-5  # q_a / kv_a RMSNorm epsilon
ROW_SUMSQ_WORDS = 32  # fp32 [2 parity][2 slot][8 rows]
ROW_SUMSQ_PARITY_WORDS = 16


# Named barriers and participants.
BAR_SEGMENT = 11
BAR_ROUTINE = 15  # warps 0-5 only (the hosted routines' own id)
BAR_GATHER = 12
BAR_NORM = 13  # P7 collective CTA-wide barrier

# Engine warp roles.
W_ENGINE_FIRST = 6
W_GATHER_END = 8  # warps 6-8 issue the early gather
W_PREFETCH_FIRST = 9
PREFETCH_CHUNK_BYTES = 4096  # one bulk L2 prefetch per thread per chunk
GATHER_THREADS = 96  # warps 6-8
SEL_WORDS_PER_CTA = NUM_TOKENS * TOPK // NUM_CTAS  # untagged selection copy

# Workspace Uint32 tickets: completions/readiness use monotonic adds.
# The last CTA exit advances GEN.
TICK_GEN = 0
TICK_QKV_COMPLETED = 1  # P0 tile completions
TICK_QKV_COMP_Q = 2  # q_a tile completions gating P2
TICK_QB_COMPLETED = 6  # folded Q-B tile completions
TICK_QB_READY = 7  # per-head Q-B readiness
TICK_ABS_COMPLETED = 47
TICK_ABS_READY = 48  # per-head RoPE arrivals
TICK_PREPARE_COMPLETED = 128  # P4 task completions
TICK_EXIT = 129  # +128 per generation: CTA exits; the last one bumps TICK_GEN
TICK_HEAD_READY = (
    130  # [local_heads] +8 per generation (one per P5 head-CTA of the head)
)
TICK_P0_READY_SCRATCH = 259
TICKET_WORDS = 352
# Zone C layout (bytes from the zone C base).
ZONE_C_SLOTS = 0  # uint32[SLOTS_PER_CTA] staged forced-row slots (rows < 64)
ZONE_C_MASK = 512  # uint32[SLOTS_PER_CTA]: 0 issued, 1 invalid, 2 deferred, 3 routine's
ZONE_C_MBAR = 1024  # int64[8] mbarriers: TILES_PER_CTA gather tiles + the two P6 tiles
ZONE_C_NORM = 1088  # float32[16]: P7 collective scratch (partials, inverse RMS)
ZONE_C_BYTES = 1152

SMEM_LIMIT = 232448  # SM103 dynamic SMEM per CTA


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


class LongcatWorkerEngineKernel:
    """Attention-0 worker pipeline for TP ranks 1 through 7.

    BF16 QKV-A accumulates row statistics; folded Q-B writes the absorbed
    query. P4 applies interleaved RoPE and writes KV. P5 uses 16 contiguous
    splits per token and polls tagged live selection; P6 publishes o_proj,
    then P7 reduces, adds residual, normalizes and optionally packs NVFP4.
    PDL waits at entry and triggers dependents before P6 publication.

    Args:
        local_heads: this rank's head count, 9 or 10.
        kv_dtype: FP8 E4M3 or BF16 query and KV cache element type.
        sm_scale: sparse MLA softmax scale.
        k_scale: KV dequantization scale.
        rank: TP producer rank, 1 through 7.
        norm_eps: post-attention RMSNorm epsilon.
        selection_slot_words: stride between the two live receive slots.
    """

    def __init__(
        self,
        *,
        local_heads: int,
        kv_dtype,
        sm_scale: float,
        k_scale: float = 1.0,
        rank: int = 1,
        norm_eps: float = 1e-5,
        selection_slot_words: int = 0,
    ) -> None:
        if not isinstance(selection_slot_words, int) or selection_slot_words < 0:
            raise ValueError("selection_slot_words must be a non-negative int")
        self.p0_fast_tiles = QKV_WIDTH // P0_FAST_TILE_N
        self.p0_fast_q_tiles = Q_LORA // P0_FAST_TILE_N
        self.selection_slot_words = selection_slot_words
        if not isinstance(local_heads, int) or local_heads < 1:
            raise ValueError("local_heads must be a positive integer")
        if kv_dtype is not cutlass.Float8E4M3FN and kv_dtype is not cutlass.BFloat16:
            raise ValueError(
                "kv_dtype must be cutlass.Float8E4M3FN or cutlass.BFloat16"
            )
        if not isinstance(rank, int) or not 1 <= rank <= 7:
            raise ValueError("rank must be in [1, 7]")
        self.local_heads = local_heads
        if local_heads not in (9, 10):
            raise ValueError("worker requires 9 or 10 heads")
        self.p_b_width = local_heads * 576
        self.p_b_tile_n = 64
        self.p_b_stages = 2
        self.p_b_tiles = self.p_b_width // self.p_b_tile_n
        self.kv_dtype = kv_dtype
        self.is_fp8 = kv_dtype is cutlass.Float8E4M3FN
        self.sm_scale = float(sm_scale)
        self.k_scale = float(k_scale)
        self.rank = rank

        # P6 reuses the dead ring at arena offset 0.
        self.oproj_k_cols = local_heads * HEAD_DIM_V
        self.oproj_pitch = _oproj.row_pitch(self.oproj_k_cols)
        self.oproj_smem_bytes = _oproj.smem_bytes(self.oproj_k_cols, NUM_TOKENS)
        self.finalize = LongcatAttention1Finalize(
            local_heads=local_heads,
            kv_dtype=kv_dtype,
            kv_norm_in_epilogue=True,
            eps=QKV_NORM_EPS,
            qkv_output_tiles=self.p0_fast_tiles,
            rope_interleaved=True,
            wait_absorbed=False,
            rope_ready_per_head=True,
        )
        self.p5 = LongcatSparseMlaHeteroKernel(
            local_heads=local_heads,
            kv_dtype=kv_dtype,
            sm_scale=self.sm_scale,
            k_scale=self.k_scale,
            row_map="contiguous",
            splits=16,
        )
        self.quarters = self.p5.quarters
        self.num_clusters = self.p5.clusters
        self.num_ctas = NUM_TOKENS * 16
        self.collective = LongcatOProjCollective(
            indexer_rank=0, cluster_size=CLUSTER_CTAS, norm_eps=float(norm_eps)
        )
        self.p4_clusters = _ceil_div(self.finalize.active_task_count, CLUSTER_CTAS)
        # Place P-B after the P3/P4 CTAs when space permits, avoiding
        # serialization of a CTA's projection and finalization tasks.
        self.p_b_base = 0
        base = self.num_ctas - self.p_b_tiles
        if base >= CLUSTER_CTAS * self.p4_clusters:
            self.p_b_base = base
        self.ring_pitch = RAW_PITCH_FP8_ELEMS if self.is_fp8 else ELEMS_16BIT_PITCH * 2
        self.ring_bytes = self.p5.ring_tiles * TILE_KV * self.ring_pitch
        self.p5_arena_bytes = self.p5.arena_bytes
        self.row_bytes = KV_ROW if self.is_fp8 else KV_ROW * 2

    def arena_layout(self, zone_a1_bytes: int, zone_a2_bytes: int) -> tuple:
        """Compute constexpr arena offsets from routine storage sizes.

        Args:
            zone_a1_bytes: Storage at offset 0 while the ring is dead.
            zone_a2_bytes: Storage past the ring while it fills.

        Returns:
            ``(zone_a2_offset, zone_c_offset, arena_bytes)``, 128 B aligned.

        Raises:
            ValueError: Arena exceeds the SM103 dynamic SMEM limit.
        """
        align = 128
        zone_a2_offset = _ceil_div(self.ring_bytes, align) * align
        body = max(
            self.p5_arena_bytes,
            zone_a2_offset + zone_a2_bytes,
            zone_a1_bytes,
            self.oproj_smem_bytes,
        )
        zone_c_offset = _ceil_div(body, align) * align
        arena_bytes = zone_c_offset + ZONE_C_BYTES
        if arena_bytes > SMEM_LIMIT:
            raise ValueError(
                f"worker-engine SMEM arena {arena_bytes} B exceeds {SMEM_LIMIT} B "
                f"(ring {self.ring_bytes}, zone A1 {zone_a1_bytes}, "
                f"zone A2 {zone_a2_bytes} at {zone_a2_offset}, "
                f"P5 arena {self.p5_arena_bytes})"
            )
        return zone_a2_offset, zone_c_offset, arena_bytes

    # Engine-warp weight L2 prefetch.

    @cute.jit
    def _prefetch_addr_range(
        self,
        base: Int64,
        total_bytes: Int64,
        engine_tid: Int32,
        threads: Int32,
        policy: Int64,
    ) -> None:
        """Prefetch a global range into L2 under policy, chunked by thread.

        base and total_bytes must be multiples of 16.
        """
        offset = Int64(engine_tid) * Int64(PREFETCH_CHUNK_BYTES)
        stride = Int64(threads) * Int64(PREFETCH_CHUNK_BYTES)
        while offset < total_bytes:
            size = Int64(PREFETCH_CHUNK_BYTES)
            if total_bytes - offset < size:
                size = total_bytes - offset
            prefetch_bulk_l2(
                cute.make_ptr(
                    Uint8, base + offset, cute.AddressSpace.gmem, assumed_align=16
                ),
                Uint32(size),
                policy,
            )
            offset += stride

    @cute.jit
    def _prefetch_range(
        self,
        g_weight: cute.Tensor,
        byte0: Int64,
        total_bytes: Int64,
        engine_tid: Int32,
        threads: Int32,
        policy: Int64,
    ) -> None:
        """:meth:`_prefetch_addr_range` from byte ``byte0`` of ``g_weight``."""
        self._prefetch_addr_range(
            Int64(g_weight.iterator.toint()) + byte0,
            total_bytes,
            engine_tid,
            threads,
            policy,
        )

    # Early KV gather into the P5 ring (warps 6-8).

    @cute.jit
    def _forced_gather(
        self,
        g_block_table: cute.Tensor,
        g_positions: cute.Tensor,
        g_out_cache_loc: cute.Tensor,
        g_kv_cache: cute.Tensor,
        s_slots: cute.Tensor,
        s_mask: cute.Tensor,
        mbar_gather: cute.Pointer,
        arena_addr,
        token: Int32,
        quarter: Int32,
        bidx: Int32,
        gid: Int32,
        warp: Int32,
        lane: Int32,
    ) -> None:
        """Stage this CTA's 65 forced rows and issue the ones already in cache.

        Warps 6-8 run independently of selection. Contiguous ring rows
        r<64 map to canonical idx=16+64*split+r, position
        idx+max(length-1040, 0); row 127 maps to initial position split.
        Strided rows r<64 use idx=r*16+split, position idx for idx<16,
        otherwise idx+max(length-1040, 0); rows [64,128) belong to phase B.

        Mask 0: cached history via page table; 1: invalid, copy row 0;
        2: current verify row via out_cache_loc, deferred until P4 writes
        KV; 3: phase B owned by the routine. Contiguous row 127 completes
        on tile 3's mbarrier before the routine arms it; a negative tx-count
        is allowed while that barrier's arrival is pending.
        """
        split = quarter * 4 + bidx
        length = g_positions[token] + Int32(1)
        f_len = length
        if f_len > Int32(FORCED_TOKENS):
            f_len = Int32(FORCED_TOKENS)
        shift = length - Int32(FORCED_TOKENS)
        if shift < Int32(0):
            shift = Int32(0)
        first_cur = g_positions[0]
        for half in cutlass.range_constexpr(2):
            r = gid + half * GATHER_THREADS
            if r < SLOTS_PER_CTA:
                slot = Int32(0)
                mask = Uint32(3)
                phase_a = Int32(0)
                if r < Int32(PHASE_A_ROWS):
                    phase_a = Int32(1)
                if cutlass.const_expr(self.p5.row_map == "strided"):
                    idx = r * Int32(16) + split
                    pos = idx
                    if idx >= Int32(INITIAL_TOKENS):
                        pos = idx + shift
                else:
                    idx = Int32(INITIAL_TOKENS) + split * Int32(PHASE_A_ROWS) + r
                    pos = idx + shift
                    if r == Int32(INITIAL_ROW):
                        idx = split
                        pos = split
                        phase_a = Int32(1)
                if phase_a != Int32(0):
                    mask = Uint32(1)
                    if idx < f_len:
                        if pos >= first_cur:
                            # P4 writes current rows to out_cache_loc;
                            # the page table may not yet include these pages.
                            slot = g_out_cache_loc[pos - first_cur]
                            mask = Uint32(2)
                        else:
                            slot = (
                                g_block_table[0, pos >> Int32(PAGE_SHIFT)]
                                << Int32(PAGE_SHIFT)
                            ) | (pos & Int32(PAGE_SIZE - 1))
                            mask = Uint32(0)
                s_slots[r] = Uint32(slot)
                s_mask[r] = mask
        cute.arch.barrier(barrier_id=BAR_GATHER, number_of_threads=GATHER_THREADS)
        if warp == W_ENGINE_FIRST and lane == 0:
            # Initialize one mbarrier per ring tile. Arm only phase A,
            # including deferred rows; the routine arms phase B.
            for t in cutlass.range_constexpr(self.p5.ring_tiles):
                cute.arch.mbarrier_init(mbar_gather + t, 1)
            cute.arch.mbarrier_init_fence()
            for t in cutlass.range_constexpr(PHASE_A_TILES):
                cute.arch.mbarrier_arrive_and_expect_tx(
                    mbar_gather + t, Int32(TILE_KV * self.row_bytes)
                )
        cute.arch.barrier(barrier_id=BAR_GATHER, number_of_threads=GATHER_THREADS)
        self._gather_rows(g_kv_cache, s_slots, s_mask, mbar_gather, arena_addr, gid, 0)
        self._gather_rows(g_kv_cache, s_slots, s_mask, mbar_gather, arena_addr, gid, 1)
        cute.arch.sync_warp()

    @cute.jit
    def _gather_rows(
        self,
        g_kv_cache: cute.Tensor,
        s_slots: cute.Tensor,
        s_mask: cute.Tensor,
        mbar_gather: cute.Pointer,
        ring_addr: Int64,
        gid: Int32,
        want_mask: cutlass.Constexpr,
    ) -> None:
        """Issue the ring rows whose mask equals ``want_mask``.

        Gather thread gid copies rows gid and gid+96 (<128), one bulk
        copy each, completing on mbar_gather[row//TILE_KV]. Masks 0/1
        gather history/invalid rows early; mask 2 gathers current rows
        at P5 entry after P4 publication.
        """
        for half in cutlass.range_constexpr(2):
            r = gid + half * GATHER_THREADS
            if r < SLOTS_PER_CTA:
                if s_mask[r] == Uint32(want_mask):
                    src = g_kv_cache.iterator + Int64(s_slots[r]) * KV_ROW
                    dst = cute.make_ptr(
                        Uint8,
                        ring_addr + Int64(r) * Int64(self.ring_pitch),
                        cute.AddressSpace.smem,
                        assumed_align=16,
                    )
                    cpasync_bulk_g2s(
                        src, dst, mbar_gather + (r // TILE_KV), Int32(self.row_bytes)
                    )

    @cute.jit
    def _oproj_tile(
        self,
        g_attn_proj: cute.Tensor,
        g_o_proj: cute.Tensor,
        g_o_local: cute.Tensor,
        t_head_ready: cute.Tensor,
        gen: Uint32,
        tidx: Int32,
        tile: Int32,
        arena_addr: Int64,
        mbar_gather: cute.Pointer,
        multicast_base: Int64,
        slot: Uint32,
    ) -> None:
        """P6 o_proj tile ``tile`` (64 output columns)."""
        _oproj.oproj_tile(
            g_attn_proj,
            g_o_proj,
            g_o_local,
            t_head_ready,
            gen,
            tile,
            tidx,
            arena_addr,
            mbar_gather + OPROJ_MBAR_W,
            k_cols=self.oproj_k_cols,
            hidden=HIDDEN,
            tokens=NUM_TOKENS,
            heads=self.local_heads,
            bar_id=BAR_SEGMENT,
            threads=THREADS_PER_CTA,
            materialize_output=True,
            publish_collective=self.collective,
            publish_base=multicast_base,
            publish_slot_base=Int64(slot) * Int64(SLOT_BYTES),
            publish_rank=self.rank,
        )

    # The kernel.

    @cute.kernel
    def kernel(
        self,
        g_residual: cute.Tensor,
        g_kv_gamma: cute.Tensor,
        g_wkc: cute.Tensor,
        g_wvc_t: cute.Tensor,
        g_o_proj: cute.Tensor,
        g_post_gamma: cute.Tensor,
        g_cos_sin: cute.Tensor,
        g_positions: cute.Tensor,
        g_out_cache_loc: cute.Tensor,
        g_block_table: cute.Tensor,
        g_kv_cache: cute.Tensor,
        g_topk_slots: cute.Tensor,
        g_topk_lens: cute.Tensor,
        g_selection_flag: cute.Tensor,
        g_topk_out: cute.Tensor,
        g_lens_out: cute.Tensor,
        g_partial: cute.Tensor,
        g_counter: cute.Tensor,
        g_p5_gen: cute.Tensor,
        g_row_sumsq: cute.Tensor,
        g_qkv_out: cute.Tensor,
        g_q_b_out: cute.Tensor,
        g_q_b_heads: cute.Tensor,
        g_query: cute.Tensor,
        g_attn_proj: cute.Tensor,
        g_o_local: cute.Tensor,
        g_residual_out: cute.Tensor,
        g_norm_out: cute.Tensor,
        g_quant_out: cute.Tensor,
        g_quant_scale: cute.Tensor,
        g_quant_global: cute.Tensor,
        g_quant2_out: cute.Tensor,
        g_quant2_scale: cute.Tensor,
        g_quant2_global: cute.Tensor,
        g_transport_bases: cute.Tensor,
        g_rotation: cute.Tensor,
        t_gen: cute.Tensor,
        t_qkv_completed: cute.Tensor,
        t_comp_q: cute.Tensor,
        t_qb_completed: cute.Tensor,
        t_qb_ready: cute.Tensor,
        t_abs_ready: cute.Tensor,
        t_prepare: cute.Tensor,
        t_exit: cute.Tensor,
        t_head_ready: cute.Tensor,
        qkv_operands: tuple,
        qb_operands: tuple,
        zone_a2_offset: cutlass.Constexpr,
        zone_c_offset: cutlass.Constexpr,
        arena_bytes: cutlass.Constexpr,
    ):
        """Execute one worker step with the hosted routines' barrier contracts."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx % 32
        cta = bidy * CLUSTER_CTAS + bidx
        token = bidy // self.quarters
        quarter = bidy % self.quarters
        pair_id = bidy * 2 + bidx // 2
        pair_rank = bidx % 2
        engine_tid = tidx - ROUTINE_THREADS
        gid = tidx - ROUTINE_THREADS  # gather thread id of warps 6-8 (0..95)
        cute.arch.griddepcontrol_wait()
        gen = load_acquire_gpu_u32(t_gen.iterator)
        # Read-once streams use evict_first. Operands for later segments
        # use normal L2 priority.
        pol_first = make_l2_policy_evict_first()
        pol_last = make_l2_policy_evict_normal()
        # P0 uses this parity and zeroes the other for the next launch.
        parity = Int32(gen) & Int32(1)
        s_row_sumsq = cute.make_tensor(
            g_row_sumsq.iterator + parity * Int32(ROW_SUMSQ_PARITY_WORDS),
            cute.make_layout((ROW_SUMSQ_PARITY_WORDS,)),
        )
        if cutlass.const_expr(self.selection_slot_words > 0):
            # Match the indexer's double-slot publication at gen&1.
            sel_off = parity * Int32(self.selection_slot_words)
            g_topk_slots = cute.make_tensor(
                g_topk_slots.iterator + sel_off, g_topk_slots.layout
            )
            g_topk_lens = cute.make_tensor(
                g_topk_lens.iterator + sel_off, g_topk_lens.layout
            )

        # Shared memory arena.
        smem = utils.SmemAllocator()
        arena_ptr = smem.allocate(arena_bytes, byte_alignment=128)
        arena_addr = Int64(arena_ptr.toint())

        zone_a1_ptr = cute.make_ptr(
            cutlass.Int8, arena_addr, cute.AddressSpace.smem, assumed_align=128
        )
        zone_a2_ptr = cute.make_ptr(
            cutlass.Int8,
            arena_addr + Int64(zone_a2_offset),
            cute.AddressSpace.smem,
            assumed_align=128,
        )
        zone_c_addr = arena_addr + Int64(zone_c_offset)
        s_slots = cute.make_tensor(
            cute.make_ptr(
                Uint32,
                zone_c_addr + Int64(ZONE_C_SLOTS),
                cute.AddressSpace.smem,
                assumed_align=16,
            ),
            cute.make_layout((SLOTS_PER_CTA,)),
        )
        s_mask = cute.make_tensor(
            cute.make_ptr(
                Uint32,
                zone_c_addr + Int64(ZONE_C_MASK),
                cute.AddressSpace.smem,
                assumed_align=16,
            ),
            cute.make_layout((SLOTS_PER_CTA,)),
        )
        mbar_gather = cute.make_ptr(
            cutlass.Int64,
            zone_c_addr + Int64(ZONE_C_MBAR),
            cute.AddressSpace.smem,
            assumed_align=8,
        )
        norm_scratch = cute.make_ptr(
            Float32,
            zone_c_addr + Int64(ZONE_C_NORM),
            cute.AddressSpace.smem,
            assumed_align=16,
        )

        # P0: QKV-A projection
        if cta < self.p0_fast_tiles:
            stream_gemv_tile(
                *qkv_operands,
                g_qkv_out,
                cta * Int32(P0_FAST_TILE_N),
                zone_a1_ptr,
                cute.make_ptr(
                    cutlass.Int64,
                    arena_addr + Int64(gemv_mbar_offset(P0_FAST_TILE_N)),
                    cute.AddressSpace.smem,
                    assumed_align=8,
                ),
                tidx,
                HIDDEN,
                P0_FAST_TILE_N,
                row_sumsq=s_row_sumsq,
                sumsq_ranges=((0, Q_LORA, 0), (Q_LORA, Q_LORA + KV_LORA, 1)),
                weight_policy=pol_first,
            )
            # Join every lane's row-statistic stores before publication.
            cute.arch.barrier(barrier_id=BAR_SEGMENT, number_of_threads=THREADS_PER_CTA)
            if tidx == 0:
                # Fence both counts; publish P-B's comp_q gate first.
                fence_acq_rel_gpu()
                if cta < self.p0_fast_q_tiles:
                    red_add_relaxed_gpu_u32(t_comp_q.iterator, Uint32(1))
                red_add_relaxed_gpu_u32(t_qkv_completed.iterator, Uint32(1))
        else:
            if cta == self.p0_fast_tiles and tidx < ROW_SUMSQ_PARITY_WORDS:
                g_row_sumsq[
                    (Int32(1) - parity) * Int32(ROW_SUMSQ_PARITY_WORDS) + tidx
                ] = Float32(0.0)
        cute.arch.barrier(barrier_id=BAR_SEGMENT, number_of_threads=THREADS_PER_CTA)

        # P2: Q-B projection + early gather
        pb_tile = cta - Int32(self.p_b_base)
        if pb_tile >= Int32(0) and pb_tile < Int32(self.p_b_tiles):
            # Weight stages run ahead of the q_a activation copies, which wait
            # for the q tiles of P0 (``t_comp_q``).
            stream_gemv_tile(
                *qb_operands,
                g_q_b_out,
                pb_tile * Int32(self.p_b_tile_n),
                zone_a1_ptr,
                cute.make_ptr(
                    cutlass.Int64,
                    arena_addr
                    + Int64(gemv_mbar_offset(self.p_b_tile_n, self.p_b_stages)),
                    cute.AddressSpace.smem,
                    assumed_align=8,
                ),
                tidx,
                Q_LORA,
                self.p_b_tile_n,
                self.p_b_stages,
                row_sumsq=s_row_sumsq,
                rms_scale=(Q_LORA, QKV_NORM_EPS, 0),
                x_ready=t_comp_q.iterator,
                x_target=Uint32(gen) * Uint32(self.p0_fast_q_tiles),
                gFoldQuery=g_query,
            )
            cute.arch.barrier(barrier_id=BAR_SEGMENT, number_of_threads=THREADS_PER_CTA)
            if tidx == 0:
                # Fence the tile before publishing both completion words.
                fence_acq_rel_gpu()
                head = pb_tile // Int32(576 // self.p_b_tile_n)
                column = (pb_tile % Int32(576 // self.p_b_tile_n)) * Int32(
                    self.p_b_tile_n
                )
                if column + Int32(self.p_b_tile_n) > Int32(KV_LORA):
                    # absorbed-ready counts RoPE arrivals; the last writer
                    # releases Q-B readiness for the complete head.
                    parts = Uint32(_ceil_div(Q_B_ROPE, self.p_b_tile_n))
                    previous = atomic_add_acq_rel_gpu_u32(
                        t_abs_ready.iterator + head, Uint32(1)
                    )
                    if previous == Uint32(gen) * parts - Uint32(1):
                        red_add_release_gpu_u32(t_qb_ready.iterator + head, Uint32(1))
                red_add_release_gpu_u32(t_qb_completed.iterator, Uint32(1))
        # The ring is dead until the gather below issues into it.
        cute.arch.barrier(barrier_id=BAR_SEGMENT, number_of_threads=THREADS_PER_CTA)
        if not warp < W_ENGINE_FIRST:
            if warp <= W_GATHER_END:
                self._forced_gather(
                    g_block_table,
                    g_positions,
                    g_out_cache_loc,
                    g_kv_cache,
                    s_slots,
                    s_mask,
                    mbar_gather,
                    arena_addr,
                    token,
                    quarter,
                    bidx,
                    gid,
                    warp,
                    lane,
                )
            else:
                absorb_bytes = self.local_heads * KV_LORA * HEAD_DIM_V * 2
                slice_bytes = -(-absorb_bytes // (self.num_ctas * 16)) * 16
                slice0 = Int64(cta) * Int64(slice_bytes)
                if slice0 < Int64(absorb_bytes):
                    slice_len = Int64(slice_bytes)
                    if Int64(absorb_bytes) - slice0 < slice_len:
                        slice_len = Int64(absorb_bytes) - slice0
                    self._prefetch_range(
                        g_wkc,
                        slice0,
                        slice_len,
                        engine_tid - GATHER_THREADS,
                        GATHER_THREADS,
                        pol_last,
                    )
                    self._prefetch_range(
                        g_wvc_t,
                        slice0,
                        slice_len,
                        engine_tid - GATHER_THREADS,
                        GATHER_THREADS,
                        pol_last,
                    )
        cute.arch.barrier(barrier_id=BAR_SEGMENT, number_of_threads=THREADS_PER_CTA)

        # P3 boundary: absorbed-query projection is folded into P2.
        cute.arch.barrier(barrier_id=BAR_SEGMENT, number_of_threads=THREADS_PER_CTA)

        # P4: RoPE + query/KV assembly
        if bidy < self.p4_clusters:
            if warp < W_ENGINE_FIRST:
                self.finalize.finalize_device_routine(
                    g_q_b_heads,
                    g_qkv_out,
                    g_positions,
                    g_cos_sin,
                    g_out_cache_loc,
                    g_query,
                    g_kv_cache,
                    t_gen,
                    t_qkv_completed,
                    t_qkv_completed,  # kv_norm_completed: unused in epilogue mode
                    t_qb_ready,
                    t_abs_ready,
                    t_prepare,
                    s_row_sumsq,
                    g_kv_gamma,
                    logical_worker_id=pair_id,
                    routine_cta_rank=pair_rank,
                    barrier_threads=ROUTINE_THREADS,
                )
            else:
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
        cute.arch.barrier(barrier_id=BAR_SEGMENT, number_of_threads=THREADS_PER_CTA)

        # P5: sparse MLA decode
        if tidx == 0:
            qb_target = gen * Uint32(self.p_b_tiles)
            qb_completed = load_acquire_gpu_u32(t_qb_completed.iterator)
            while (qb_completed - qb_target).to(Int32) < Int32(0):
                qb_completed = load_acquire_gpu_u32(t_qb_completed.iterator)
            target = gen * Uint32(self.finalize.active_task_count)
            completed = load_acquire_gpu_u32(t_prepare.iterator)
            while (completed - target).to(Int32) < Int32(0):
                completed = load_acquire_gpu_u32(t_prepare.iterator)
        cute.arch.barrier(barrier_id=BAR_SEGMENT, number_of_threads=THREADS_PER_CTA)
        if warp >= W_ENGINE_FIRST and warp <= W_GATHER_END:
            # P4's KV stores were acquired above; fence the async proxy
            # before deferred bulk copies read them.
            fence_proxy_async_global()
            self._gather_rows(
                g_kv_cache, s_slots, s_mask, mbar_gather, arena_addr, gid, 2
            )
            cute.arch.sync_warp()
        # All 12 warps enter P5, which owns offset 0 with the ring as sK.
        # It waits on phase-A mbarriers and acquires selected words by tag.
        f_len = g_positions[token] + Int32(1)
        if f_len > Int32(FORCED_TOKENS):
            f_len = Int32(FORCED_TOKENS)
        self.p5.sparse_mla_device_routine(
            g_query,
            g_kv_cache,
            g_topk_slots,
            g_topk_lens,
            g_partial,
            g_counter,
            g_p5_gen,
            zone_a1_ptr,
            cta,
            token,
            quarter,
            bidx,
            tidx,
            gather_mbar=mbar_gather,
            generation=(gen - Uint32(1)).to(Int32),
            gWvc=g_wvc_t,
            gProj=g_attn_proj,
            forced_len=f_len,
            gBlockTable=g_block_table if self.selection_slot_words > 0 else None,
            gPositions=g_positions if self.selection_slot_words > 0 else None,
            gOutCacheLoc=g_out_cache_loc if self.selection_slot_words > 0 else None,
            page_shift=PAGE_SHIFT,
            gSelFlag=None,
            sel_target=gen,
            sel_tag=Uint32(gen) & Uint32(_sx.TAG_MASK),
        )
        if tidx == 0:
            # P5's final CTA barrier joins all attn_proj stores before
            # thread 0 releases this head-CTA's arrival to P6.
            if quarter * 4 + bidx < self.local_heads:
                red_add_release_gpu_u32(
                    t_head_ready.iterator + (quarter * 4 + bidx), Uint32(1)
                )
        cute.arch.barrier(barrier_id=BAR_SEGMENT, number_of_threads=THREADS_PER_CTA)

        # P6: fused o_proj
        slot = Uint32(0)
        multicast_base = Int64(0)
        # P6 publishes and P7 consumes this acquired slot; P7 rotates it.
        slot = load_acquire_gpu_u32(g_rotation.iterator)
        multicast_base = g_transport_bases[0]
        cute.arch.griddepcontrol_launch_dependents()
        p6_args = (g_attn_proj, g_o_proj, g_o_local, t_head_ready, gen, tidx)
        self._oproj_tile(
            *p6_args,
            cta,
            arena_addr,
            mbar_gather,
            multicast_base,
            slot,
        )
        if warp >= W_PREFETCH_FIRST:
            # Branch 1 needs the complete selection. Acquire the final
            # system flag because P5 only polls words it gathers and can
            # reconstruct forced rows locally.
            if cutlass.const_expr(self.selection_slot_words > 0):
                flag = load_acquire_sys_u32(g_selection_flag.iterator)
                while (flag - gen).to(Int32) < Int32(0):
                    flag = load_acquire_sys_u32(g_selection_flag.iterator)
                pub_tid = tidx - W_PREFETCH_FIRST * 32
                for i in range(pub_tid, SEL_WORDS_PER_CTA, GATHER_THREADS):
                    idx = cta * SEL_WORDS_PER_CTA + i
                    g_topk_out[idx] = self.p5._untag_word(
                        load_acquire_gpu_u32(g_topk_slots.iterator + idx)
                    )
                if cta == 0 and pub_tid < NUM_TOKENS:
                    g_lens_out[pub_tid] = self.p5._untag_word(
                        load_acquire_gpu_u32(g_topk_lens.iterator + pub_tid)
                    )
        cute.arch.barrier(barrier_id=BAR_SEGMENT, number_of_threads=THREADS_PER_CTA)

        # P7: all-reduce + residual + norm
        local_base = g_transport_bases[1]
        # All threads of the 32 quarter-0 CTAs enter: each physical cluster
        # is wholly quarter 0, and CTA bidx reduces [bidx*2048, (bidx+1)*2048).
        if quarter == 0:
            self.collective.consume_reduce_norm(
                Int32(tidx - (THREADS_PER_CTA - P7_COMM_THREADS)),
                Int32(tidx),
                slot,
                local_base,
                Int32(token * CLUSTER_CTAS + bidx),
                Int32(NUM_TOKENS * CLUSTER_CTAS),
                token,
                bidx,
                g_residual,
                g_post_gamma,
                g_residual_out,
                g_norm_out,
                norm_scratch,
                g_rotation,
                P7_COMM_THREADS,
                Int32(THREADS_PER_CTA),
                Int32(BAR_NORM),
                Int32(NUM_TOKENS * CLUSTER_CTAS),
                g_quant_out,
                g_quant_scale,
                g_quant_global,
                g_quant2_out,
                g_quant2_scale,
                g_quant2_global,
                rotate_relaxed=True,
            )
        cute.arch.barrier(barrier_id=BAR_SEGMENT, number_of_threads=THREADS_PER_CTA)
        if tidx == 0:
            # The last exiting CTA advances gen with relaxed operations.
            # All current readers captured it at entry; the next reader is
            # the next launch, ordered after this grid completes.
            exits = atomic_add_gpu_u32(t_exit.iterator, Uint32(1), acq_rel=False)
            if exits == gen * Uint32(self.num_ctas) - Uint32(1):
                store_relaxed_gpu_u32(t_gen.iterator, gen + Uint32(1))


__all__ = [
    "LongcatWorkerEngineKernel",
    "PARTIAL_FLOATS",
    "NUM_TOKENS",
    "TOPK",
    "SLOTS_PER_CTA",
    "TILE_KV",
    "TILES_PER_CTA",
    "TICKET_WORDS",
    "ZONE_C_BYTES",
    "SMEM_LIMIT",
]

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

"""TP8 Lamport one-shot all-reduce and clustered RMSNorm device routines.

Attention 0 excludes indexer rank 0; attention 1 uses all eight producers.
Each cluster owns one token row, split into 8192 / cluster_size columns per CTA.
The caller supplies the engine CTA; these routines do not launch kernels.

The DownReduce workspace has three 1 MiB slots. Each rank contributes
[8,8192] BF16 values (128 KiB), at byte offset
``slot*1048576 + ((token*8 + rank)*8192 + hidden)*2``.
Producers multimem-store through multicast ``transport_bases[0]``; consumers
poll their local unicast replica, ``transport_bases[1]``. The dirty sentinel
is 0x80000000; producer words are sanitized so clean data cannot equal it.

Per-rank uint32[9] ``rotation_flags`` holds current slot [0], retired-slot
state [1:6], and arrival counter [8]. After all consumer/clearer CTAs arrive,
the last marks the consumed slot for re-arming and release-publishes the next.
Callers acquire [0] before using that state. Per-rank consume-before-publish
order keeps producers at most one slot ahead of the slowest consumer, within
the three-slot Lamport tolerance. See ``arrive_rotation`` for grid-tail ordering.

Sum producers in ascending rank order, rounding to BF16 after each addition;
start with the first fragment unchanged. Then add the residual and round once
more. Seven producers require six rank-add rounds; eight require seven.
"""

from __future__ import annotations

from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import math
from flashinfer.fused_moe.cute_dsl.blackwell.utils import fmin
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention._primitives import (
    ld_shared_cluster_f32,
    mapa_shared_cluster_u32,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._collective_primitives import (
    bf16x8_to_packed_u32x4,
    load_global_u32x4,
    packed_u32x4_to_bf16x8,
    store_global_u32x4,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._downreduce_oneshot import (
    LongcatDownReduceOneShotWorkspace,
    create_longcat_downreduce_oneshot_workspace,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._downreduce_oneshot_primitives import (
    load_oneshot_fragment,
    oneshot_word_is_dirty,
    sanitize_oneshot_word,
    store_oneshot_fragment,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._primitives import (
    atomic_add_acq_rel_gpu_u32,
    atomic_add_gpu_u32,
    store_relaxed_gpu_u32,
    store_release_gpu_u32,
)

FP32_MAX = 3.40282346638528859812e38
SF_VEC_SIZE = 16
SLOT_BYTES = 1048576
TOKEN_COUNT = 8
RANK_COUNT = 8
HIDDEN_DIM = 8192
ELEMENT_BYTES = 2
DIRTY_SENTINEL = Uint32(0x80000000)


def _fragment_slots(fragments_per_cta: int, comm_threads: int) -> int:
    """Ceiling of eight-column fragments per consumer thread.

    Thread tid owns tid + i*comm_threads. This trace-time integer sizes register
    fragments and their constexpr loops for the configured cluster width.
    """
    return -(-fragments_per_cta // comm_threads)


class LongcatOProjCollective:
    """Device all-reduce and norm routines for the shared one-shot transport.

    Args:
        world_size: TP size, fixed to 8 by the payload layout.
        indexer_rank: Non-publishing rank to skip, or None for eight producers.
        cluster_size: Physical CTAs sharing a token's 8192-column norm.
        norm_eps: RMSNorm epsilon.

    Each routine specifies its thread-participation and ordering requirements.
    """

    def __init__(
        self,
        *,
        world_size: int = 8,
        indexer_rank: int | None = 0,
        cluster_size: int = 4,
        norm_eps: float = 1e-5,
    ) -> None:
        if world_size != RANK_COUNT:
            raise ValueError("the fixed payload layout requires world_size == 8")
        if indexer_rank is not None and (
            type(indexer_rank) is not int or not 0 <= indexer_rank < RANK_COUNT
        ):
            raise ValueError("indexer_rank must be None or a Python int in [0, 8)")
        if type(cluster_size) is not int or not 1 <= cluster_size <= RANK_COUNT:
            raise ValueError("cluster_size must be a Python int in [1, 8]")
        if HIDDEN_DIM % cluster_size or (HIDDEN_DIM // cluster_size) % 8:
            raise ValueError("each CTA's column range must be BF16x8 aligned")
        if type(norm_eps) is not float or not norm_eps > 0.0 or norm_eps != norm_eps:
            raise ValueError("norm_eps must be a finite positive float")
        self.world_size = world_size
        self.indexer_rank = indexer_rank
        self.cluster_size = cluster_size
        self.norm_eps = norm_eps
        self.cols_per_cta = HIDDEN_DIM // cluster_size
        self.fragments_per_cta = self.cols_per_cta // 8
        # ``indexer_rank is None``: every rank publishes (attention branch 1,
        # where no rank hosts an indexer) -- eight producers, seven roundings.
        self.producer_ranks = tuple(
            rank for rank in range(world_size) if rank != indexer_rank
        )
        self.producer_count = len(self.producer_ranks)

    # Producer side (o_proj GEMM epilogue, worker ranks only)

    @cute.jit
    def publish_tile_fragments(
        self,
        tile: cute.Tensor,
        multicast_base: Int64,
        slot_base: Int64,
        rank: cutlass.Int32,
        col0: cutlass.Int32,
        tidx: cutlass.Int32,
        threads: cutlass.Int32,
        token_rows: cutlass.Constexpr = 8,
        cols: cutlass.Constexpr = 128,
    ) -> None:
        """Sanitize and multicast one BF16 o_proj epilogue tile.

        Args:
            tile: [token_rows,cols], with token rows 0..7, cols divisible by 8,
                and a 16-byte-aligned element base.
            multicast_base: Workspace multicast VA, transport_bases[0].
            slot_base: Current slot's byte offset, slot*1048576.
            rank: Producer rank; the configured indexer skips publication.
            col0: First hidden column, BF16x8 aligned in [0,8192).
            tidx: Calling thread id.
            threads: Participants striding over fragments; all must invoke it.
            token_rows, cols: Compile-time tile dimensions.

        Returns:
            None. Stores use the module's payload offset; no barrier is taken here.
        """
        if cutlass.const_expr(self.indexer_rank is None):
            is_producer = cutlass.Boolean(True)
        else:
            is_producer = rank != self.indexer_rank
        if is_producer:
            fragments_per_row = cols // 8
            total_fragments = token_rows * fragments_per_row
            for fragment in cutlass.range(tidx, total_fragments, threads):
                row = fragment // fragments_per_row
                col_in_tile = (fragment % fragments_per_row) * 8
                values = cute.make_rmem_tensor((8,), BFloat16)
                for element in cutlass.range_constexpr(8):
                    values[element] = tile[row, col_in_tile + element]
                hidden = col0 + col_in_tile
                payload_offset = (
                    slot_base
                    + ((Int64(row) * 8 + Int64(rank)) * 8192 + Int64(hidden)) * 2
                )
                packed = bf16x8_to_packed_u32x4(values.load())
                sanitized = cute.make_rmem_tensor((4,), Uint32)
                for word in cutlass.range_constexpr(4):
                    sanitized[word] = sanitize_oneshot_word(packed[word])
                store_oneshot_fragment(
                    multicast_base + payload_offset, sanitized.load()
                )

    # Consumer side (all eight ranks)

    @cute.jit
    def clear_retired_slot(
        self,
        comm_tid: cutlass.Int32,
        local_base: Int64,
        rotation_flags: cute.Tensor,
        cta_linear: cutlass.Int32,
        cta_count: cutlass.Int32,
        comm_threads: cutlass.Int32,
    ) -> None:
        """Re-arm retired-slot words across cta_count CTAs, 16 bytes per store.

        Call after acquiring the current slot and before this CTA's rotation arrival.
        Consumers or separate clearers may do the work; all count as rotation
        participants. External clearers use consume_reduce_norm(clear=False).
        """
        sentinel = cute.make_rmem_tensor((4,), Uint32)
        for word in cutlass.range_constexpr(4):
            sentinel[word] = DIRTY_SENTINEL
        dirty_flag = rotation_flags[3]
        if dirty_flag != Uint32(0):
            dirty_slot = rotation_flags[1]
            dirty_words = rotation_flags[4] // Uint32(16)
            clear_index = cta_linear * comm_threads + comm_tid
            clear_stride = cta_count * comm_threads
            while clear_index < dirty_words:
                store_global_u32x4(
                    local_base + Int64(dirty_slot) * 1048576 + Int64(clear_index) * 16,
                    sentinel.load(),
                )
                clear_index += clear_stride

    @cute.jit
    def _reduce_norm_columns(
        self,
        comm_tid: cutlass.Int32,
        slot: Uint32,
        local_base: Int64,
        token: cutlass.Int32,
        cta_in_cluster: cutlass.Int32,
        residual: cute.Tensor,
        gamma: cute.Tensor,
        residual_out: cute.Tensor,
        r_result: cute.Tensor,
        r_gamma: cute.Tensor,
        scratch: cute.Pointer,
        comm_threads: cutlass.Constexpr,
    ) -> None:
        """Poll producers, sum in rank order and add residual on consumer warps.

        Thread comm_tid owns fragments comm_tid + i*comm_threads of this CTA's
        column range. Preload residual/gamma before polling; keep packed results
        and gamma in registers through the cluster exchange.
        """
        scratch_sumsq = cute.make_tensor(scratch, cute.make_layout((17,)))
        thread_sumsq = Float32(0.0)
        slots = _fragment_slots(self.fragments_per_cta, comm_threads)
        r_residual = cute.make_rmem_tensor((slots, 4), Uint32)
        for i in cutlass.range_constexpr(slots):
            fragment = comm_tid + Int32(i * comm_threads)
            if fragment < Int32(self.fragments_per_cta):
                hidden = cta_in_cluster * self.cols_per_cta + fragment * 8
                offset = Int64(token) * 8192 + Int64(hidden)
                res = load_global_u32x4(
                    cute.make_ptr(
                        BFloat16,
                        (residual.iterator + offset).toint(),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                )
                gam = load_global_u32x4(
                    cute.make_ptr(
                        BFloat16,
                        (gamma.iterator + Int64(hidden)).toint(),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                )
                for word in cutlass.range_constexpr(4):
                    r_residual[i, word] = res[word]
                    r_gamma[i, word] = gam[word]
        # Each sweep polls every producer of every owned fragment. Repeat
        # the complete sweep until all fragments are clean.
        producers = self.producer_count
        peer_samples = cute.make_rmem_tensor((slots * 4 * producers,), Uint32)
        dirty = cutlass.Boolean(True)
        while dirty:
            dirty = cutlass.Boolean(False)
            for i in cutlass.range_constexpr(slots):
                fragment = comm_tid + Int32(i * comm_threads)
                if fragment < Int32(self.fragments_per_cta):
                    hidden = cta_in_cluster * self.cols_per_cta + fragment * 8
                    for peer_index in cutlass.range_constexpr(producers):
                        peer = self.producer_ranks[peer_index]
                        pointer = cute.make_ptr(
                            BFloat16,
                            local_base
                            + Int64(slot) * 1048576
                            + ((Int64(token) * 8 + Int64(peer)) * 8192 + Int64(hidden))
                            * 2,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        )
                        sample = load_oneshot_fragment(pointer)
                        for word in cutlass.range_constexpr(4):
                            peer_samples[(i * producers + peer_index) * 4 + word] = (
                                sample[word]
                            )
                            dirty = dirty | oneshot_word_is_dirty(sample[word])

        for i in cutlass.range_constexpr(slots):
            fragment = comm_tid + Int32(i * comm_threads)
            if fragment < Int32(self.fragments_per_cta):
                hidden = cta_in_cluster * self.cols_per_cta + fragment * 8
                # Rank-order sum over the producers: FP32 accumulate, BF16 round
                # after every producer addition (production rank rounding
                # minus the skipped indexer rank).
                packed = cute.make_rmem_tensor((4,), Uint32)
                for word in cutlass.range_constexpr(4):
                    packed[word] = peer_samples[i * producers * 4 + word]
                reduced = packed_u32x4_to_bf16x8(packed.load())
                for peer_index in cutlass.range_constexpr(1, producers):
                    for word in cutlass.range_constexpr(4):
                        packed[word] = peer_samples[
                            (i * producers + peer_index) * 4 + word
                        ]
                    values = packed_u32x4_to_bf16x8(packed.load())
                    reduced = (reduced.to(Float32) + values.to(Float32)).to(BFloat16)

                # The residual add applies the final BF16 rounding.
                for word in cutlass.range_constexpr(4):
                    packed[word] = r_residual[i, word]
                residual_values = packed_u32x4_to_bf16x8(packed.load()).to(Float32)
                result = (reduced.to(Float32) + residual_values).to(BFloat16)
                result_packed = bf16x8_to_packed_u32x4(result)
                offset = Int64(token) * 8192 + Int64(hidden)
                store_global_u32x4(
                    (residual_out.iterator + offset).toint(), result_packed
                )
                for word in cutlass.range_constexpr(4):
                    r_result[i, word] = result_packed[word]
                result_f32 = result.to(Float32)
                for element in cutlass.range_constexpr(8):
                    value = result_f32[element]
                    thread_sumsq = thread_sumsq + value * value

        warp_sum = cute.arch.warp_reduction_sum(thread_sumsq)
        # Partials at scratch[0 .. warps-1]; the inverse RMS lands right after
        # them, in scratch[warps], after the cluster exchange.
        if comm_tid % 32 == 0:
            scratch_sumsq[comm_tid // 32] = warp_sum

    @cute.jit
    def _quantize_fragment(
        self,
        r_norm: cute.Tensor,
        block_amax: Float32,
        fragment: cutlass.Int32,
        hidden: cutlass.Int32,
        token: cutlass.Int32,
        quant_out: cute.Tensor,
        quant_scale: cute.Tensor,
        norm_const: Float32,
    ) -> None:
        """Pack eight BF16-rounded norm values with a shared 16-value block amax.

        The even fragment writes its E4M3 scale at
        ``(c//4)*512 + (r%32)*16 + (r//32)*4 + c%4``; token r is below 32.
        ``norm_const`` is the consumer's reciprocal global input scale.
        """
        pvscale = block_amax * Float32(1.0 / 6.0) * norm_const
        sfc = pvscale.to(cutlass.Float8E4M3FN)
        if fragment % 2 == 0:
            # Global block index, not the CTA-local fragment pair: every CTA of
            # the cluster owns a different column range and they would
            # otherwise collide on the same 128 slots.
            block = hidden // SF_VEC_SIZE
            quant_scale[(block // 4) * 512 + token * 16 + block % 4] = sfc
        acc_scale = cute.arch.rcp_approx(sfc.to(Float32)) * norm_const
        acc_scale = fmin(acc_scale, Float32(FP32_MAX), nan=True)
        scaled = cute.make_rmem_tensor((8,), Float32)
        for e in cutlass.range_constexpr(8):
            scaled[e] = r_norm[e] * acc_scale
        packed_fp4 = cute.make_rmem_tensor((8,), cutlass.Float4E2M1FN)
        packed_fp4.store(scaled.load().to(cutlass.Float4E2M1FN))
        # Eight e2m1 values are four bytes at byte offset ``hidden / 2``.
        gOut = cute.make_tensor(
            cute.make_ptr(
                cutlass.Uint8,
                quant_out.iterator.toint() + Int64(token) * 4096 + Int64(hidden) // 2,
                cute.AddressSpace.gmem,
                assumed_align=4,
            ),
            cute.make_layout((4,)),
        )
        cute.autovec_copy(cute.recast_tensor(packed_fp4, cutlass.Uint8), gOut)

    @cute.jit
    def _normalize_columns(
        self,
        comm_tid: cutlass.Int32,
        token: cutlass.Int32,
        cta_in_cluster: cutlass.Int32,
        norm_out: cute.Tensor,
        r_result: cute.Tensor,
        r_gamma: cute.Tensor,
        inverse_rms: Float32,
        comm_threads: cutlass.Constexpr,
        quant_out: Optional[cute.Tensor] = None,
        quant_scale: Optional[cute.Tensor] = None,
        quant_const: Optional[Float32] = None,
        quant2_out: Optional[cute.Tensor] = None,
        quant2_scale: Optional[cute.Tensor] = None,
        quant2_const: Optional[Float32] = None,
    ) -> None:
        """Apply cluster inverse RMS and gamma, then round/store norm_out as BF16.

        Optional NVFP4 outputs encode those rounded values. Two consumers may share
        the block amax while using distinct reciprocal global input scales.
        """
        packed = cute.make_rmem_tensor((4,), Uint32)
        r_norm = cute.make_rmem_tensor((8,), Float32)
        r_amax = cute.make_rmem_tensor((1,), Float32)
        for i in cutlass.range_constexpr(
            _fragment_slots(self.fragments_per_cta, comm_threads)
        ):
            fragment = comm_tid + Int32(i * comm_threads)
            if cutlass.const_expr(quant_out is not None):
                r_amax[0] = Float32(0.0)
            if fragment < Int32(self.fragments_per_cta):
                hidden = cta_in_cluster * self.cols_per_cta + fragment * 8
                offset = Int64(token) * 8192 + Int64(hidden)
                for word in cutlass.range_constexpr(4):
                    packed[word] = r_result[i, word]
                pre_norm = packed_u32x4_to_bf16x8(packed.load()).to(Float32)
                for word in cutlass.range_constexpr(4):
                    packed[word] = r_gamma[i, word]
                gamma_values = packed_u32x4_to_bf16x8(packed.load()).to(Float32)
                norm_bf16 = (pre_norm * inverse_rms * gamma_values).to(BFloat16)
                store_global_u32x4(
                    (norm_out.iterator + offset).toint(),
                    bf16x8_to_packed_u32x4(norm_bf16),
                )
                if cutlass.const_expr(quant_out is not None):
                    # Quantize BF16-rounded norm_out, matching downstream fp4_quantize
                    # values and block maxima at rounding boundaries.
                    rounded = norm_bf16.to(Float32)
                    r_norm.store(rounded)
                    abs_ir = math.absf(rounded.ir_value())
                    abs_frg = type(rounded)(abs_ir, rounded.shape, rounded.dtype)
                    r_amax[0] = abs_frg.reduce(cute.ReductionOp.MAX, Float32(0.0), 0)
            if cutlass.const_expr(quant_out is not None):
                # Neighbor lanes share a 16-value quantization block. Keep the
                # collective exchange outside the fragment guard; inactive
                # lanes contribute zero.
                block_amax = cute.arch.fmax(
                    r_amax[0], cute.arch.shuffle_sync_bfly(r_amax[0], offset=1)
                )
                if fragment < Int32(self.fragments_per_cta):
                    hidden = cta_in_cluster * self.cols_per_cta + fragment * 8
                    self._quantize_fragment(
                        r_norm,
                        block_amax,
                        fragment,
                        hidden,
                        token,
                        quant_out,
                        quant_scale,
                        quant_const,
                    )
                    if cutlass.const_expr(quant2_out is not None):
                        self._quantize_fragment(
                            r_norm,
                            block_amax,
                            fragment,
                            hidden,
                            token,
                            quant2_out,
                            quant2_scale,
                            quant2_const,
                        )

    @cute.jit
    def consume_reduce_norm(
        self,
        comm_tid: cutlass.Int32,
        tidx: cutlass.Int32,
        slot: Uint32,
        local_base: Int64,
        cta_linear: cutlass.Int32,
        cta_count: cutlass.Int32,
        token: cutlass.Int32,
        cta_in_cluster: cutlass.Int32,
        residual: cute.Tensor,
        gamma: cute.Tensor,
        residual_out: cute.Tensor,
        norm_out: cute.Tensor,
        scratch: cute.Pointer,
        rotation_flags: cute.Tensor,
        comm_threads: cutlass.Constexpr,
        threads_per_cta: cutlass.Int32,
        cta_bar_id: cutlass.Int32,
        rotation_participants: cutlass.Int32,
        quant_out: Optional[cute.Tensor] = None,
        quant_scale: Optional[cute.Tensor] = None,
        quant_global: Optional[cute.Tensor] = None,
        quant2_out: Optional[cute.Tensor] = None,
        quant2_scale: Optional[cute.Tensor] = None,
        quant2_global: Optional[cute.Tensor] = None,
        clear: cutlass.Constexpr = True,
        rotate: cutlass.Constexpr = True,
        rotate_relaxed: cutlass.Constexpr = False,
    ) -> None:
        """Clear retired data, reduce producers, add residual and RMS-normalize.

        Args:
            comm_tid: Consumer id in [0,comm_threads); negative ids only join syncs.
            tidx: Full CTA thread id in [0,threads_per_cta).
            slot: Current slot, acquired from rotation_flags[0] by the caller.
            local_base: Local unicast replica, transport_bases[1].
            cta_linear, cta_count: This CTA and the count distributing sentinel clears.
            token: Verify token owned by the physical cluster.
            cta_in_cluster: CTA's share of the token's 8192 columns.
            residual, residual_out, norm_out: Eight token-major BF16[8192] rows.
            gamma: BF16 [8192] RMSNorm weights.
            scratch: SMEM for comm_threads//32 FP32 partials, live through both
                cluster syncs. Every thread of every cluster CTA must call.
            rotation_flags: Per-rank uint32[9] rotation/arrival state.
            comm_threads: Consumer threads; cluster_size*(comm_threads//32) <= 32.
            threads_per_cta: Full CTA participation count, normally 384.
            cta_bar_id: Full-CTA named barrier used before rotation.
            rotation_participants: Consumer and clearer CTAs arriving per rank.
            quant_out: Optional uint8[8,4096] NVFP4 output from BF16-rounded norm_out.
            quant_scale: E4M3[128*512] swizzled block scales.
            quant_global: FP32[1] reciprocal global input scale.
            quant2_out, quant2_scale, quant2_global: Second packing with its own scale.
            clear: Clear this CTA's retired-slot share; False with external clearers.
            rotate: Finish with a full-CTA barrier and arrive_rotation. If False,
                the caller must perform both before any producer reads the new slot.
            rotate_relaxed: Grid-tail ordering described by arrive_rotation.

        Returns:
            None. Each consumer stores its residual/norm columns and optional packing.
            Two unconditional cluster barriers protect partial visibility and lifetime;
            with rotate, one last arriver per rank publishes the next slot.
        """
        slots = _fragment_slots(self.fragments_per_cta, comm_threads)
        r_result = cute.make_rmem_tensor((slots, 4), Uint32)
        r_gamma = cute.make_rmem_tensor((slots, 4), Uint32)
        # Load scale constants before polling and possibly aliasing norm stores.
        quant_const = quant2_const = None
        if cutlass.const_expr(quant_out is not None):
            quant_const = Float32(quant_global[0])
        if cutlass.const_expr(quant2_out is not None):
            quant2_const = Float32(quant2_global[0])
        if comm_tid >= 0:
            if cutlass.const_expr(clear):
                self.clear_retired_slot(
                    comm_tid,
                    local_base,
                    rotation_flags,
                    cta_linear,
                    cta_count,
                    comm_threads,
                )
            self._reduce_norm_columns(
                comm_tid,
                slot,
                local_base,
                token,
                cta_in_cluster,
                residual,
                gamma,
                residual_out,
                r_result,
                r_gamma,
                scratch,
                comm_threads,
            )

        # Every CTA thread joins both cluster syncs and, with rotate, the
        # threads_per_cta named barrier. Only computation is consumer-guarded.
        # Cluster release/acquire publishes scratch partials from every warp,
        # including local ones; no preceding CTA barrier is needed.
        partials = comm_threads // 32
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        # Each consumer warp applies the same reduction tree to the same
        # lane-ordered partials, producing identical inverse-RMS bits.
        inverse_rms = Float32(0.0)
        if comm_tid >= 0:
            lane = comm_tid % 32
            partial = Float32(0.0)
            if lane < self.cluster_size * partials:
                peer_sumsq = mapa_shared_cluster_u32(scratch, Int32(lane // partials))
                partial = ld_shared_cluster_f32(peer_sumsq + (lane % partials) * 4)
            full_sumsq = cute.arch.warp_reduction_sum(partial)
            inverse_rms = cute.math.rsqrt(
                full_sumsq / Float32(HIDDEN_DIM) + Float32(self.norm_eps),
                fastmath=True,
            )
        # Retain scratch and every CTA until all peers finish DSMEM reads;
        # reusing it or exiting early can fault with Xid 13 (CTA Not Present).
        # Arrive after local reads complete; join the final wait before return.
        cute.arch.cluster_arrive_relaxed()

        if comm_tid >= 0:
            self._normalize_columns(
                comm_tid,
                token,
                cta_in_cluster,
                norm_out,
                r_result,
                r_gamma,
                inverse_rms,
                comm_threads,
                quant_out,
                quant_scale,
                quant_const,
                quant2_out,
                quant2_scale,
                quant2_const,
            )

        if cutlass.const_expr(rotate):
            # The full-CTA barrier also proves this CTA's epilogue publishes,
            # sentinel clears, and norm stores are complete before the arrival.
            cute.arch.barrier(barrier_id=cta_bar_id, number_of_threads=threads_per_cta)
            self.arrive_rotation(
                tidx, slot, rotation_flags, rotation_participants, rotate_relaxed
            )
        cute.arch.cluster_wait()

    @cute.jit
    def arrive_rotation(
        self,
        tidx: cutlass.Int32,
        slot: Uint32,
        rotation_flags: cute.Tensor,
        rotation_participants: cutlass.Int32,
        relaxed: cutlass.Constexpr = False,
    ) -> cutlass.Boolean:
        """Arrive once per CTA; the last participant rotates this rank's slot.

        Every thread must call after a full-CTA barrier covering slot reads, retired
        slot stores and outputs. Thread 0 arrives; the last marks the consumed slot
        for re-arming and release-stores the next slot.

        With relaxed=True, this must end the grid: no later in-grid rotation reader,
        next reader acquires only after grid completion, and re-arming occurs in a
        later grid. Arrival and slot publication then use relaxed operations.

        Returns:
            True on the rotating thread, False elsewhere.
        """
        rotated = cutlass.Boolean(False)
        if tidx == 0:
            if cutlass.const_expr(relaxed):
                previous = atomic_add_gpu_u32(
                    rotation_flags.iterator + 8, Uint32(1), acq_rel=False
                )
            else:
                previous = atomic_add_acq_rel_gpu_u32(
                    rotation_flags.iterator + 8, Uint32(1)
                )
            if previous + Uint32(1) == Uint32(rotation_participants):
                rotation_flags[1] = slot
                rotation_flags[2] = Uint32(1048576)
                rotation_flags[3] = Uint32(1)
                rotation_flags[4] = Uint32(1048576)
                rotation_flags[5] = Uint32(0)
                rotation_flags[8] = Uint32(0)
                if cutlass.const_expr(relaxed):
                    store_relaxed_gpu_u32(
                        rotation_flags.iterator,
                        (slot + Uint32(1)) % Uint32(3),
                    )
                else:
                    store_release_gpu_u32(
                        rotation_flags.iterator,
                        (slot + Uint32(1)) % Uint32(3),
                    )
                rotated = cutlass.Boolean(True)
        return rotated


def make_o_proj_collective_workspace(
    group, device
) -> LongcatDownReduceOneShotWorkspace:
    """Collectively allocate the shared TP8 DownReduce one-shot workspace.

    Args:
        group: Process group spanning exactly eight ranks.
        device: This rank's CUDA device (SM103).

    Returns:
        LongcatDownReduceOneShotWorkspace with three sentinel-armed 1 MiB slots,
        multicast/unicast transport_bases and uint32[9] rotation_flags. Keep it
        alive for the engine's lifetime; slot geometry is documented above.
    """
    return create_longcat_downreduce_oneshot_workspace(group, device)


__all__ = [
    "LongcatOProjCollective",
    "make_o_proj_collective_workspace",
]

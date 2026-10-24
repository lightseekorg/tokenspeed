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

"""Caller-hosted absorbed-query BF16 tensor-core routine.

Compute [8,128] @ [512,128]^T independently per head using the ll_bf16
split-K pipeline. Each CTA owns 64 output columns; a two-CTA cluster holds
two independent tiles without a split-K exchange or separate launch.
"""

from __future__ import annotations

import math

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Uint32
from cutlass.pipeline import sm90 as pipeline
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._collective_primitives import (
    fence_proxy_async_global,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._primitives import (
    load_acquire_gpu_u32,
    red_add_relaxed_gpu_u32,
    store_relaxed_gpu_u32,
)
from tokenspeed_kernel.thirdparty.cute_dsl.topk.top_k.single_pass_multi_cta_radix_topk import (
    fence_acq_rel_gpu,
)

_TOKENS = 8
_HEADS = 8
_Q_NOPE = 128
_KV_LORA = 512
_OUTPUT_TILE_N = 64
_TILE_M = 16
_TILE_K = 128
_NUM_STAGES = 2
_DMA_WARPS = 2
_MMA_WARPS = 4
_THREADS = (_DMA_WARPS + _MMA_WARPS) * 32
_PHYSICAL_CLUSTERS = 32
_N_TILES = _KV_LORA // _OUTPUT_TILE_N
_ACTIVE_CTAS = _HEADS * _N_TILES
_Q_B_NOPE_TILES_PER_HEAD = 3
_Q_B_TILES_PER_HEAD = 4
_MMA_SYNC_BARRIER_ID = 14
_ROUTINE_SYNC_BARRIER_ID = 15

assert _ACTIVE_CTAS == _PHYSICAL_CLUSTERS * 2


class LongcatAttention1AbsorbedQuery:
    """Compute exact-shape absorbed queries in a caller-owned cluster grid."""

    physical_cluster_count = _PHYSICAL_CLUSTERS
    threads_per_cta = _THREADS

    def __init__(
        self,
        *,
        local_heads: int = 8,
        prefetch_weight_before_activation: bool = False,
        defer_pipeline_sync: bool = False,
        kv_dtype=cutlass.Float8E4M3FN,
    ) -> None:
        """Configure absorbed-query tiles and pipeline participation.

        Args:
            local_heads: Positive head count; eight 64-column tiles per head.
            kv_dtype: Query output type, Float8E4M3FN or BFloat16.
            prefetch_weight_before_activation: Issue w_kc before acquiring Q-B tiles.
            defer_pipeline_sync: Order deferred mbarrier init with named barrier 15.
                Required for a warp subset of a larger CTA; full sync_threads would
                wait on nonparticipants.

        Raises:
            ValueError: Invalid head count or unsupported dtype.
        """
        if not isinstance(local_heads, int) or local_heads < 1:
            raise ValueError("local_heads must be a positive integer")
        self.local_heads = local_heads
        self.active_task_count = local_heads * _N_TILES
        self.prefetch_weight_before_activation = prefetch_weight_before_activation
        self.defer_pipeline_sync = bool(defer_pipeline_sync)
        if kv_dtype is not cutlass.Float8E4M3FN and kv_dtype is not cutlass.BFloat16:
            raise ValueError(
                "kv_dtype must be cutlass.Float8E4M3FN or cutlass.BFloat16"
            )
        self.query_is_bf16 = kv_dtype is cutlass.BFloat16
        self.tile_m = _TILE_M
        self.tile_n = _OUTPUT_TILE_N
        self.tile_k = _TILE_K
        self.num_stages = _NUM_STAGES
        self.copy_bits = 128
        self.num_dma_threads = _DMA_WARPS * 32
        self.num_mma_threads = _MMA_WARPS * 32
        self.num_epilogue_elems = _TILE_M * _OUTPUT_TILE_N
        self.epilogue_elems_per_thread = self.num_epilogue_elems // self.num_mma_threads

    def _make_smem_layout(self, dtype, smem_tiler):
        major_size = min(smem_tiler[1], 64)
        swizzle_bits = int(math.log2(major_size * dtype.width // self.copy_bits))
        swizzle_bits = min(swizzle_bits, 3)
        atom_outer = cute.make_layout((8, major_size), stride=(major_size, 1))
        atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3), 0, atom_outer
        )
        return cute.tile_to_shape(atom, smem_tiler, (0, 1, 2))

    def _make_gmem_tiled_copy(self, atom_copy, dtype):
        copy_elems = self.copy_bits // dtype.width
        k_threads = self.tile_k // copy_elems
        thread_layout = cute.make_layout(
            (self.num_dma_threads // k_threads, k_threads),
            stride=(k_threads, 1),
        )
        return cute.make_tiled_copy_tv(
            atom_copy, thread_layout, cute.make_layout((1, copy_elems))
        )

    @cute.jit
    def _fill_pred(self, pred_flat, coord_tensor, dim_limit):
        coord_ktile = coord_tensor[None, None, 0, 0]
        for vector in cutlass.range_constexpr(pred_flat.shape[0]):
            for row in cutlass.range_constexpr(pred_flat.shape[1]):
                pred_flat[vector, row] = cute.elem_less(
                    coord_ktile[(0, vector), row], (dim_limit, _Q_NOPE)
                )

    @cute.jit
    def _wait_for_q_b_tiles(
        self,
        q_b_tile_ready: cute.Tensor,
        generation: cute.Tensor,
        head: cutlass.Int32,
    ) -> Uint32:
        """Acquire the three Q-B tiles containing one head's q_nope; returns
        the generation waited for."""
        target_generation = load_acquire_gpu_u32(generation.iterator)
        for q_b_nope_tile in cutlass.range_constexpr(_Q_B_NOPE_TILES_PER_HEAD):
            q_b_tile = head * _Q_B_TILES_PER_HEAD + q_b_nope_tile
            completed = load_acquire_gpu_u32(q_b_tile_ready.iterator + q_b_tile)
            while (completed - target_generation).to(Int32) < Int32(0):
                completed = load_acquire_gpu_u32(q_b_tile_ready.iterator + q_b_tile)
        return target_generation

    def _make_pred(self, coord_tensor):
        num_vectors = coord_tensor.shape[0][1]
        num_rows = cute.size(coord_tensor, mode=[1])
        pred_flat = cute.make_rmem_tensor(
            cute.make_layout((num_vectors, num_rows), stride=(num_rows, 1)),
            cutlass.Boolean,
        )
        pred = cute.make_tensor(
            pred_flat.iterator,
            cute.make_layout(
                (num_vectors, num_rows, cute.size(coord_tensor, mode=[2])),
                stride=(num_rows, 1, 0),
            ),
        )
        return pred_flat, pred

    @cute.jit
    def prepare_device_routine(self):
        """Build fixed MMA/copy layouts and the overlayable shared ABI."""
        a_layout = self._make_smem_layout(BFloat16, (_TILE_M, _TILE_K, _NUM_STAGES))
        b_layout = self._make_smem_layout(
            BFloat16, (_OUTPUT_TILE_N, _TILE_K, _NUM_STAGES)
        )

        @cute.struct
        class SharedStorage:
            a: cute.struct.Align[
                cute.struct.MemRange[BFloat16, cute.cosize(a_layout)], 16
            ]
            b: cute.struct.Align[
                cute.struct.MemRange[BFloat16, cute.cosize(b_layout)], 16
            ]
            mbar: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int64, _NUM_STAGES * 2], 8
            ]
            warp_partial: cute.struct.Align[
                cute.struct.MemRange[Float32, _MMA_WARPS * _TILE_M * _OUTPUT_TILE_N],
                16,
            ]

        copy_atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            BFloat16,
            num_bits_per_copy=self.copy_bits,
        )
        tiled_copy_a = self._make_gmem_tiled_copy(copy_atom, BFloat16)
        tiled_copy_b = self._make_gmem_tiled_copy(copy_atom, BFloat16)
        mma_op = cute.nvgpu.warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16))
        tiled_mma = cute.make_tiled_mma(
            mma_op,
            cute.make_layout((1, 1, 1)),
            permutation_mnk=(16, _OUTPUT_TILE_N, 16),
        )
        self.shared_storage = SharedStorage
        return a_layout, b_layout, tiled_copy_a, tiled_copy_b, tiled_mma

    @cute.jit
    def absorbed_query_device_routine(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        generation: cute.Tensor,
        q_b_tile_ready: cute.Tensor,
        absorbed_query_tile_ready: cute.Tensor,
        absorbed_query_completed: cute.Tensor,
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
        tiled_copy_a: cute.TiledCopy,
        tiled_copy_b: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        shared_storage_ptr: cute.Pointer,
        *,
        logical_worker_id=None,
        routine_cta_rank=None,
        barrier_threads: cutlass.Constexpr = 192,
        wait_ready: cutlass.Constexpr = True,
        tail_cluster_sync: cutlass.Constexpr = True,
    ) -> None:
        """Consume Q-B and publish one head/output tile per active CTA.

        Args:
            wait_ready: Acquire head Q-B tickets; False requires prior producer
                ordering, such as a cluster barrier after sibling stores.
            tail_cluster_sync: Use the routine's final cluster barrier; False if the
                caller supplies one, e.g. after invoking two tiles on the same CTA.
            logical_worker_id: Optional task-group override; task=id*2+routine_cta_rank.
            routine_cta_rank: Optional 0/1 override of block_idx_in_cluster().
            barrier_threads: Participants in routine named barrier 15, e.g. 192 for
                hosted W0-5.
        """
        tidx, _, _ = cute.arch.thread_idx()
        _, bidy, bidz = cute.arch.block_idx()
        _, grid_dim_y, _ = cute.arch.grid_dim()
        warp = cute.arch.warp_idx()
        lane = cute.arch.lane_idx()
        if cutlass.const_expr(logical_worker_id is None):
            physical_cluster_id = cutlass.Int32(bidy + grid_dim_y * bidz)
        else:
            physical_cluster_id = logical_worker_id
        if cutlass.const_expr(routine_cta_rank is None):
            cta_rank = cute.arch.block_idx_in_cluster()
        else:
            cta_rank = routine_cta_rank
        output_task = physical_cluster_id * 2 + cta_rank

        if output_task < self.active_task_count:
            head = output_task // _N_TILES
            # The generation tidx 0 publishes at the tail, read at the head
            # so its round trip is not on the tail: the default path's wait
            # returns it, the others load it here.
            generation_value = Uint32(0)
            if cutlass.const_expr(
                self.prefetch_weight_before_activation or not wait_ready
            ):
                if tidx == 0:
                    generation_value = load_acquire_gpu_u32(generation.iterator)
            # With prefetch_weight_before_activation, issue the immutable
            # head weights before acquiring the activation ticket.
            if cutlass.const_expr(not self.prefetch_weight_before_activation):
                if cutlass.const_expr(wait_ready):
                    if tidx == 0:
                        generation_value = self._wait_for_q_b_tiles(
                            q_b_tile_ready, generation, head
                        )
                cute.arch.barrier(
                    barrier_id=_ROUTINE_SYNC_BARRIER_ID,
                    number_of_threads=barrier_threads,
                )

            output_tile = output_task % _N_TILES
            head_a = mA[head, None, None]
            head_b = mB[head, None, None]
            head_c = mC[head, None, None]
            identity_a = cute.make_identity_tensor(head_a.layout.shape)
            identity_b = cute.make_identity_tensor(head_b.layout.shape)
            identity_c = cute.make_identity_tensor(head_c.layout.shape)
            cta_tiler = (_TILE_M, _OUTPUT_TILE_N, _TILE_K)
            coord = (0, output_tile, None)
            gA = cute.local_tile(
                head_a, tiler=cta_tiler, coord=coord, proj=(1, None, 1)
            )
            gB = cute.local_tile(
                head_b, tiler=cta_tiler, coord=coord, proj=(None, 1, 1)
            )
            gC = cute.local_tile(
                head_c, tiler=cta_tiler, coord=coord, proj=(1, 1, None)
            )
            cA = cute.local_tile(
                identity_a,
                tiler=cta_tiler,
                coord=coord,
                proj=(1, None, 1),
            )
            cB = cute.local_tile(
                identity_b,
                tiler=cta_tiler,
                coord=coord,
                proj=(None, 1, 1),
            )
            cC = cute.local_tile(
                identity_c,
                tiler=cta_tiler,
                coord=coord,
                proj=(1, 1, None),
            )
            gA = cute.make_tensor(gA.iterator.align(16), gA.layout)
            gB = cute.make_tensor(gB.iterator.align(16), gB.layout)

            storage = self.shared_storage(shared_storage_ptr)
            sA = storage.a.get_tensor(a_layout)
            sB = storage.b.get_tensor(b_layout)
            warp_partials = storage.warp_partial.get_tensor(
                cute.make_layout(
                    (_MMA_WARPS, self.num_epilogue_elems),
                    stride=(self.num_epilogue_elems, 1),
                )
            )
            producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_dma_threads
            )
            consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_mma_threads
            )
            mainloop_pipeline = pipeline.PipelineCpAsync.create(
                barrier_storage=storage.mbar.data_ptr(),
                num_stages=_NUM_STAGES,
                producer_group=producer_group,
                consumer_group=consumer_group,
                defer_sync=self.defer_pipeline_sync,
            )
            if cutlass.const_expr(self.defer_pipeline_sync):
                cute.arch.mbarrier_init_fence()
                cute.arch.barrier(
                    barrier_id=_ROUTINE_SYNC_BARRIER_ID,
                    number_of_threads=barrier_threads,
                )
            producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, _NUM_STAGES
            )
            consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, _NUM_STAGES
            )
            is_dma_warp = warp < _DMA_WARPS
            dma_tidx = tidx
            mma_tidx = tidx - self.num_dma_threads
            mma_warp = warp - _DMA_WARPS
            if is_dma_warp:
                thr_a = tiled_copy_a.get_slice(dma_tidx)
                thr_b = tiled_copy_b.get_slice(dma_tidx)
                tAgA = thr_a.partition_S(gA)
                tAsA = thr_a.partition_D(sA)
                tBgB = thr_b.partition_S(gB)
                tBsB = thr_b.partition_D(sB)
                tAcA = thr_a.partition_S(cA)
                tBcB = thr_b.partition_S(cB)
                pred_a_flat, pred_a = self._make_pred(tAcA)
                pred_b_flat, pred_b = self._make_pred(tBcB)
                self._fill_pred(pred_a_flat, tAcA, _TOKENS)
                self._fill_pred(pred_b_flat, tBcB, _KV_LORA)
                mainloop_pipeline.producer_acquire(producer_state)
                cute.copy(
                    tiled_copy_b,
                    tBgB[None, None, None, 0],
                    tBsB[None, None, None, producer_state.index],
                    pred=pred_b,
                )
                if cutlass.const_expr(self.prefetch_weight_before_activation):
                    # Commit only the independent B copies so they progress
                    # while this head's Q-B tiles are pending.  The pipeline
                    # full barrier remains blocked until A is also copied.
                    cute.arch.cp_async_commit_group()
                    if cutlass.const_expr(wait_ready):
                        if lane == 0:
                            self._wait_for_q_b_tiles(q_b_tile_ready, generation, head)
                    cute.arch.sync_warp()
                    # The early commit removed B from the later pipeline
                    # arrive-on group, so join that group before copying A.
                    cute.arch.cp_async_wait_group(0)
                cute.copy(
                    tiled_copy_a,
                    tAgA[None, None, None, 0],
                    tAsA[None, None, None, producer_state.index],
                    pred=pred_a,
                )
                mainloop_pipeline.producer_commit(producer_state)
                producer_state.advance()
                mainloop_pipeline.producer_tail(producer_state)
            else:
                thr_mma = tiled_mma.get_slice(lane)
                tCsA = thr_mma.partition_A(sA)
                tCsB = thr_mma.partition_B(sB)
                tCgC = thr_mma.partition_C(gC)
                tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
                tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
                tCrC = tiled_mma.make_fragment_C(tCgC)
                tCrC.fill(0.0)
                atom_a = cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), BFloat16
                )
                atom_b = cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), BFloat16
                )
                tiled_s2r_a = cute.make_tiled_copy_A(atom_a, tiled_mma)
                tiled_s2r_b = cute.make_tiled_copy_B(atom_b, tiled_mma)
                thr_s2r_a = tiled_s2r_a.get_slice(lane)
                thr_s2r_b = tiled_s2r_b.get_slice(lane)
                tCsA_warp = cute.logical_divide(
                    thr_s2r_a.partition_S(sA),
                    (None, None, _MMA_WARPS, None),
                )
                tCsB_warp = cute.logical_divide(
                    thr_s2r_b.partition_S(sB),
                    (None, None, _MMA_WARPS, None),
                )
                tCrA_view = thr_s2r_a.retile(tCrA)
                tCrB_view = thr_s2r_b.retile(tCrB)
                k_blocks_per_warp = cute.size(tCrA, mode=[2]) // _MMA_WARPS
                mainloop_pipeline.consumer_wait(consumer_state)
                for ki in cutlass.range_constexpr(k_blocks_per_warp):
                    cute.copy(
                        tiled_s2r_a,
                        tCsA_warp[None, None, (mma_warp, ki), consumer_state.index],
                        tCrA_view[None, None, 0],
                    )
                    cute.copy(
                        tiled_s2r_b,
                        tCsB_warp[None, None, (mma_warp, ki), consumer_state.index],
                        tCrB_view[None, None, 0],
                    )
                    cute.gemm(
                        tiled_mma,
                        tCrC,
                        tCrA[None, None, 0],
                        tCrB[None, None, 0],
                        tCrC,
                    )
                mainloop_pipeline.consumer_release(consumer_state)
                consumer_state.advance()

                epilogue_layout = cute.make_layout(
                    (
                        self.epilogue_elems_per_thread,
                        self.num_mma_threads,
                    ),
                    stride=(self.num_mma_threads, 1),
                )
                epilogue_slots = cute.make_tensor(0, epilogue_layout)
                epilogue_coords = cute.make_identity_tensor((_OUTPUT_TILE_N, _TILE_M))
                warp_storage = cute.make_tensor(
                    cute.domain_offset((mma_warp, 0), warp_partials).iterator,
                    cute.make_layout(
                        (_TILE_M, _OUTPUT_TILE_N),
                        stride=(_OUTPUT_TILE_N, 1),
                    ),
                )
                tCsC = thr_mma.partition_C(warp_storage)
                cute.autovec_copy(tCrC, tCsC)
                cute.arch.barrier(
                    barrier_id=_MMA_SYNC_BARRIER_ID,
                    number_of_threads=self.num_mma_threads,
                )
                for elem in cutlass.range_constexpr(self.epilogue_elems_per_thread):
                    elem_idx = epilogue_slots[elem, mma_tidx]
                    local_coord = cute.select(epilogue_coords[elem_idx], mode=[1, 0])
                    global_coord = cC[local_coord]
                    if cute.elem_less(global_coord, head_c.shape):
                        value = (
                            warp_partials[None, elem_idx]
                            .load()
                            .reduce(
                                cute.ReductionOp.ADD,
                                init_val=Float32(0.0),
                                reduction_profile=0,
                            )
                        )
                        # Match the current BF16 BMM boundary followed by the
                        # fused query store conversion, but avoid materializing
                        # that intermediate tensor a second time.
                        if cutlass.const_expr(self.query_is_bf16):
                            gC[local_coord] = BFloat16(value)
                        else:
                            gC[local_coord] = cutlass.Float8E4M3FN(BFloat16(value))

            # Writer-side proxy fence: the engine's P5 bulk-loads these query
            # rows (async proxy); order the generic stores before the publish.
            fence_proxy_async_global()
            cute.arch.barrier(
                barrier_id=_ROUTINE_SYNC_BARRIER_ID,
                number_of_threads=barrier_threads,
            )
            if tidx == 0:
                # One fence for both words; each release would also wait out
                # the previous word's round trip.
                fence_acq_rel_gpu()
                store_relaxed_gpu_u32(
                    absorbed_query_tile_ready.iterator + output_task,
                    generation_value,
                )
                red_add_relaxed_gpu_u32(absorbed_query_completed.iterator, Uint32(1))

        # Keep both CTAs converged before the parent reuses this cluster.
        if cutlass.const_expr(tail_cluster_sync):
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()


__all__ = ["LongcatAttention1AbsorbedQuery"]

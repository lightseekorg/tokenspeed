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

"""Weight-streaming tcgen05 GEMV: acc[t,n] += sum_k x[t,k]*w[n,k].

BF16 weights are the MMA M operand (128 rows), eight tokens its padded N
operand, with FP32 TMEM accumulation. Warp 5 issues direct weight TMA,
warp 4 issues MMA, and W0-3 handle the epilogue.

K splits own contiguous stage ranges and red.add FP32 partials into caller-
zeroed acc. Unsplit output can store BF16 directly; each activation stage
may wait for its own readiness ticket (one head per o_proj K stage).
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention._primitives import (
    make_l2_policy_evict_first,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._collective_primitives import (
    fence_proxy_async_global,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._primitives import (
    load_acquire_gpu_u32,
    red_add_gpu_f32,
)

TOKENS = 8
TILE_M = 128


class LongcatWeightStreamGemv:
    """BF16[N,K] weights and [8,K] activations on tcgen05, optionally K-split.

    Args:
        n_dim, k_dim: Output width and K. TMA zero-fills tail rows of 128-row tiles;
            the epilogue omits them.
        k_splits: CTAs per output tile, each owning contiguous K stages.
        stage_k: K elements per stage (128*stage_k*2 weight bytes).
        ab_stages: In-flight stage count.
        n_pad: Padded token dimension, 8 or 16.
    """

    def __init__(
        self,
        *,
        n_dim: int,
        k_dim: int,
        k_splits: int,
        stage_k: int = 128,
        ab_stages: int = 6,
        n_pad: int = 16,
    ) -> None:
        if k_dim % stage_k or (k_dim // stage_k) % k_splits:
            raise ValueError("k_splits must divide the K stages")
        self.n_dim = n_dim
        self.k_dim = k_dim
        self.k_splits = k_splits
        self.stage_k = stage_k
        self.k_stages = k_dim // stage_k
        self.split_stages = self.k_stages // k_splits
        self.m_tiles = -(-n_dim // TILE_M)
        self.tasks = self.m_tiles * k_splits
        self.num_ab_stage = ab_stages
        self.n_pad = n_pad
        self.mma_tiler = (TILE_M, n_pad, stage_k)
        self.acc_dtype = cutlass.Float32
        self.cta_group = tcgen05.CtaGroup.ONE
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 192
        self.num_tmem_alloc_cols = 32
        self.buffer_align_bytes = 1024
        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.threads_per_cta
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=32 * len((self.mma_warp_id, *self.epilog_warp_id)),
        )
        # The epilogue warps alone (``publish``).
        self.epilog_barrier = pipeline.NamedBarrier(
            barrier_id=2, num_threads=32 * len(self.epilog_warp_id)
        )

    @cute.jit
    def prepare(self, weight: cute.Tensor, act: cute.Tensor):
        """MMA, SMEM layouts and TMA atoms (``weight`` ``(N, K, 1)``,
        ``act`` ``(8, K, 1)``, both K-major bf16)."""
        self.a_dtype = weight.element_type
        self.b_dtype = act.element_type
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            utils.LayoutEnum.from_tensor(weight).mma_major_mode(),
            utils.LayoutEnum.from_tensor(act).mma_major_mode(),
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage
        )
        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,)
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        self.a_stage_bytes = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        self.b_stage_bytes = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        a_op = sm100_utils.cluster_shape_to_tma_atom_A((1, 1), tiled_mma.thr_id)
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            weight,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B((1, 1), tiled_mma.thr_id)
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            act,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        return (
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
        )

    # ------------------------------------------------------------------
    @cute.jit
    def device_routine(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        acc_out: cute.Tensor,
        shared_storage_ptr: cute.Pointer,
        task: cutlass.Int32,
        gate: cute.Tensor | None = None,
        gate_target: cutlass.Uint32 | None = None,
        out_bf16: cute.Tensor | None = None,
        publish=None,
    ):
        """Run task m_tile*k_splits+split on CTA warps 0..5.

        Args:
            acc_out: Token-major FP32[8*n_dim] atomic-add destination; caller zeroes it.
                Unused with out_bf16.
            gate, gate_target: Acquire gate[global_stage] with a signed-difference
                target check, then fence the async proxy before activation TMA.
                The stage's weights issue before this acquire.
            out_bf16: Direct token-major BF16[8*n_dim] output; requires k_splits==1.
            publish: With out_bf16, (collective,multicast_base,rotation,slot_bytes,
                rank,gates). After MMA completion, stage BF16 outputs in the retired
                first A stage and multicast. Epilogue warps wait on every gates
                (ticket,target) before reading rotation; MMA can overlap those waits.
                Publication uses named barrier 2.
        """
        if cutlass.const_expr(publish is not None):
            if cutlass.const_expr(out_bf16 is None or self.n_dim % TILE_M):
                raise ValueError("publish needs out_bf16 and whole tiles")
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        storage = self.shared_storage(shared_storage_ptr)
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sA_base = storage.sA.data_ptr().toint()
        a_stage_bytes = cute.size_in_bytes(
            self.a_dtype, cute.slice_(a_smem_layout_staged, (None, None, None, 0))
        )
        b_stage_bytes = cute.size_in_bytes(
            self.b_dtype, cute.slice_(b_smem_layout_staged, (None, None, None, 0))
        )
        if tidx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=a_stage_bytes + b_stage_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len(self.epilog_warp_id)
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        cute.arch.mbarrier_init_fence()
        self.cta_sync_barrier.arrive_and_wait()
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        thr_mma = tiled_mma.get_slice(0)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            cutlass.Int32(0),
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            cutlass.Int32(0),
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, 1))
        evict_first = make_l2_policy_evict_first()
        m_tile = task // self.k_splits
        split = task % self.k_splits
        k_first = split * self.split_stages
        self.cta_sync_barrier.arrive_and_wait()

        if warp_idx == self.tma_warp_id:
            prod = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            # Prefill weight stages, then issue each refill before its activation
            # gate. act follows the same slots for activation copies.
            pre = min(self.num_ab_stage, self.split_stages)
            for k in cutlass.range_constexpr(pre):
                ab_pipeline.producer_acquire(prod)
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, m_tile, k_first + k, 0)],
                    tAsA[(None, prod.index)],
                    tma_bar_ptr=ab_pipeline.producer_get_barrier(prod),
                    cache_policy=evict_first,
                )
                prod.advance()
            act = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            for k in cutlass.range(0, self.split_stages, 1, unroll=1):
                if k >= pre:
                    ab_pipeline.producer_acquire(prod)
                    cute.copy(
                        tma_atom_a,
                        tAgA[(None, m_tile, k_first + k, 0)],
                        tAsA[(None, prod.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(prod),
                        cache_policy=evict_first,
                    )
                    prod.advance()
                if cutlass.const_expr(gate is not None):
                    ready = load_acquire_gpu_u32(gate.iterator + (k_first + k))
                    while (ready - gate_target).to(cutlass.Int32) < cutlass.Int32(0):
                        ready = load_acquire_gpu_u32(gate.iterator + (k_first + k))
                    fence_proxy_async_global()
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, 0, k_first + k, 0)],
                    tBsB[(None, act.index)],
                    tma_bar_ptr=ab_pipeline.producer_get_barrier(act),
                )
                act.advance()
            ab_pipeline.producer_tail(prod)

        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)[
                (None, None, None, 0)
            ]
            cons = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_prod = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 1
            )
            acc_pipeline.producer_acquire(acc_prod)
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            k_blocks = cute.size(tCrA, mode=[2])
            for k in cutlass.range(0, self.split_stages, 1, unroll=1):
                ab_pipeline.consumer_wait(cons)
                for kb in cutlass.range_constexpr(k_blocks):
                    cute.gemm(
                        tiled_mma,
                        tCtAcc,
                        tCrA[(None, None, kb, cons.index)],
                        tCrB[(None, None, kb, cons.index)],
                        tCtAcc,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                ab_pipeline.consumer_release(cons)
                cons.advance()
            acc_pipeline.producer_commit(acc_prod)
            acc_prod.advance()
            acc_pipeline.producer_tail(acc_prod)

        if warp_idx < self.mma_warp_id:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
            epi_tile = (TILE_M, self.n_pad)
            copy_atom_t2r = sm100_utils.get_tmem_load_op(
                self.mma_tiler,
                utils.LayoutEnum.ROW_MAJOR,
                self.acc_dtype,
                self.acc_dtype,
                epi_tile,
                False,
            )
            tAcc_epi = cute.flat_divide(
                tCtAcc_base[((None, None), 0, 0, None)], epi_tile
            )
            tiled_copy_t2r = tcgen05.make_tmem_copy(
                copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
            )
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
            cC_epi = cute.flat_divide(cute.make_identity_tensor(epi_tile), epi_tile)
            tTR_cC = thr_copy_t2r.partition_D(cC_epi)
            tTR_rAcc = cute.make_rmem_tensor(
                tTR_cC[(None, None, None, 0, 0)].shape, self.acc_dtype
            )
            acc_cons = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 1
            )
            if cutlass.const_expr(publish is not None):
                collective, multicast_base, rotation, slot_bytes, rank, gates = publish
                if tidx == 0:
                    for word, target in gates:
                        seen = load_acquire_gpu_u32(word)
                        while (seen - target).to(cutlass.Int32) < cutlass.Int32(0):
                            seen = load_acquire_gpu_u32(word)
                self.epilog_barrier.arrive_and_wait()
                slot_base = cutlass.Int64(load_acquire_gpu_u32(rotation)) * slot_bytes
            acc_pipeline.consumer_wait(acc_cons)
            cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, 0, 0, 0)], tTR_rAcc)
            cute.arch.fence_view_async_tmem_load()
            with cute.arch.elect_one():
                acc_pipeline.consumer_release(acc_cons)
            # Row ``tidx`` of the tile (one weight row / output column), the
            # first eight N columns are the tokens.
            row = m_tile * TILE_M + tidx
            if cutlass.const_expr(publish is not None):
                s_tile = cute.make_tensor(
                    cute.make_ptr(
                        cutlass.BFloat16,
                        sA_base,
                        cute.AddressSpace.smem,
                        assumed_align=16,
                    ),
                    cute.make_layout((TOKENS, TILE_M), stride=(TILE_M, 1)),
                )
                for t in cutlass.range_constexpr(TOKENS):
                    s_tile[t, tidx] = tTR_rAcc[t].to(cutlass.BFloat16)
                self.epilog_barrier.arrive_and_wait()
                collective.publish_tile_fragments(
                    s_tile,
                    multicast_base,
                    slot_base,
                    rank,
                    m_tile * TILE_M,
                    tidx,
                    cutlass.Int32(32 * len(self.epilog_warp_id)),
                    cols=TILE_M,
                )
            if row < self.n_dim:
                for t in cutlass.range_constexpr(TOKENS):
                    if cutlass.const_expr(out_bf16 is not None):
                        out_bf16[t * self.n_dim + row] = tTR_rAcc[t].to(
                            cutlass.BFloat16
                        )
                    else:
                        red_add_gpu_f32(
                            acc_out.iterator + (t * self.n_dim + row), tTR_rAcc[t]
                        )
            tmem.relinquish_alloc_permit()
            tmem.free(acc_tmem_ptr)
        self.cta_sync_barrier.arrive_and_wait()

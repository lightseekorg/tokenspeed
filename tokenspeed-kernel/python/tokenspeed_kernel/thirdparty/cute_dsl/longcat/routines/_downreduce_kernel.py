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

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This file is ported from TensorRT-LLM's dense_blockscaled_gemm_persistent.py
# with modifications for FlashInfer integration.
# Original: https://github.com/NVIDIA/TensorRT-LLM

"""LongCat DownReduce with independent compute/communication warps.

Full-K TMA/MMA/TMEM loops ported from FlashInfer commit
60b49158ab4fb81718aef486c2d3c89aec4c1901:
flashinfer/gemm/kernels/dense_blockscaled_gemm_sm100.py.
Math/layout preparation uses that version's unchanged common base.
"""

from __future__ import annotations

from typing import Optional, Union

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm100 import (
    Sm100BlockScaledPersistentDenseGemmKernel,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._collective_primitives import (
    bf16x8_to_packed_u32x4,
    load_global_u32x4,
    packed_u32x4_to_bf16x8,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._downreduce_oneshot_primitives import (
    sanitize_oneshot_word,
    store_oneshot_fragment,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._primitives import (
    load_acquire_gpu_u32,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._stage import (
    BlockScaledStages,
)
from tokenspeed_kernel.thirdparty.cute_dsl.topk.top_k.single_pass_multi_cta_radix_topk import (
    fence_acq_rel_gpu,
)


class LongcatDownReduceKernel(
    BlockScaledStages, Sm100BlockScaledPersistentDenseGemmKernel
):
    """Fixed-shape full-K Down and TP8 reduction in one finite kernel entry."""

    def __init__(self, *, rank=0):
        if type(rank) is not int or not 0 <= rank < 8:
            raise ValueError("rank must be a Python int in [0,8)")
        super().__init__(
            sf_vec_size=16,
            mma_tiler_mn=(128, 8),
            cluster_shape_mn=(1, 1),
            use_prefetch=False,
            enable_pdl=False,
        )
        self.worker_count = 64
        self.threads_per_cta = 384
        self.tasks_per_worker = 1
        self.rank = rank

    @cute.jit
    def prepare_device_routine(
        self,
        weight: cute.Tensor,
        activation: cute.Tensor,
        weight_scale: cute.Tensor,
        activation_scale: cute.Tensor,
        alpha: cute.Tensor,
        output: cute.Tensor,
        peer_payload_bases: cute.Tensor,
        publish_arrivals: cute.Tensor,
    ):
        """Prepare native SwapAB inputs and caller-owned output views.

        Weight/activation are logical FP4 [8192,1536,1]/[8,1536,1].
        Scale tensors point to the established block16 FP8 storage. Output
        is BF16 [8192,8,1] with caller-defined strides. Peer tables contain eight
        immutable int64 device addresses. Transport controls are uint32[9].
        Workspace/pointer tables must survive every captured replay.
        """
        a_tensor, b_tensor = weight, activation
        sfa_tensor, sfb_tensor = weight_scale, activation_scale
        c_tensor = output
        (
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_c,
            tma_tensor_c,
        ) = self._prepare_tma(
            a_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            c_tensor,
        )

        if cutlass.const_expr(cute.size(self.epi_tile[1]) != (8)):
            raise ValueError("DownReduce epilogue width does not match its MMA mode")

        self.buffer_align_bytes = 1024

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.c_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        return (
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_c,
            tma_tensor_c,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.c_smem_layout_staged,
            alpha,
            peer_payload_bases,
            publish_arrivals,
        )

    def _copy_tma_stage(self, atom, gmem, smem, ab_pipeline, state, mask):
        """Emit one TMA stage copy at trace time without advancing its producer."""
        cute.copy(
            atom,
            gmem[(None, state.count)],
            smem[(None, state.index)],
            tma_bar_ptr=ab_pipeline.producer_get_barrier(state),
            mcast_mask=mask,
        )

    @cute.jit
    def downreduce_device_routine(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        alpha: cute.Tensor,
        peer_payload_bases: cute.Tensor,
        publish_arrivals: cute.Tensor,
        shared_storage_ptr: Optional[cute.Pointer] = None,
        logical_worker: Optional[cutlass.Int32] = None,
        gate_arrivals: Optional[cute.Tensor] = None,
        gate_epochs: Optional[cute.Tensor] = None,
        local_add: Optional[cute.Tensor] = None,
        local_add_ready: Optional[cute.Tensor] = None,
        local_add_gen: Optional[cute.Tensor] = None,
    ):
        """Publish Down fragments for the layer collective.

        ``local_add`` is rank-local, unreduced BF16 ``[8, 8192]`` MoE output.
        Add it before publication so one-shot all-reduce sums ``down + local_add``
        across ranks at LongCat's MLP1 boundary. Its producer on another stream
        release-adds ``local_add_ready`` once per invocation; epilogue warps wait
        for ``local_add_ready >= local_add_gen + 1`` before publishing. The layer
        collective bumps ``local_add_gen`` after all Down workers have read it.
        Both counters are device-owned and monotonic across captured replays.
        """
        # Layouts used by the epilogue must be built in the device JIT region.
        epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk, self.use_2cta_instrs, self.c_layout, self.c_dtype
        )
        tidx, _, _ = cute.arch.thread_idx()
        if cutlass.const_expr(local_add is not None):
            if cutlass.const_expr(local_add_ready is None or local_add_gen is None):
                raise ValueError("local_add needs its two tickets")
            # Publish-only: the consumer that bumps ``local_add_gen`` runs
            # after every worker has read it here (its host orders that).
            local_add_target = load_acquire_gpu_u32(
                local_add_gen.iterator
            ) + cutlass.Uint32(1)
        worker, _, _ = cute.arch.block_idx()
        if cutlass.const_expr(logical_worker is not None):
            worker = logical_worker
        gate_target = cutlass.Uint32(0)
        if cutlass.const_expr(gate_arrivals is not None):
            gate_target = (gate_epochs[worker] + cutlass.Uint32(1)) * cutlass.Uint32(4)
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        slot = load_acquire_gpu_u32(publish_arrivals.iterator)

        if cutlass.const_expr(shared_storage_ptr is None):
            smem = utils.SmemAllocator()
            shared_storage_ptr = smem.allocate(
                self.shared_storage.size_in_bytes(),
                byte_alignment=self.buffer_align_bytes,
            )

        if warp_idx < 6:
            alpha_value = alpha[0].to(cutlass.Float32)

            if warp_idx == self.tma_warp_id:
                cpasync.prefetch_descriptor(tma_atom_a)
                cpasync.prefetch_descriptor(tma_atom_b)
                cpasync.prefetch_descriptor(tma_atom_sfa)
                cpasync.prefetch_descriptor(tma_atom_sfb)
                cpasync.prefetch_descriptor(tma_atom_c)

            bidx, bidy, bidz = cute.arch.block_idx()
            if cutlass.const_expr(logical_worker is not None):
                bidx = logical_worker
            mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
            is_leader_cta = mma_tile_coord_v == 0
            # The layer hosts independent one-CTA MMA workers.
            cta_rank_in_cluster = cutlass.Int32(0)
            block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
                cta_rank_in_cluster
            )
            block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
                cta_rank_in_cluster
            )
            tidx, _, _ = cute.arch.thread_idx()

            storage = self.shared_storage(shared_storage_ptr)

            tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr.ptr
            tmem_holding_buf = storage.tmem_holding_buf.ptr

            ab_pipeline, acc_pipeline = self.stage_pipelines(
                storage,
                cluster_layout_vmnk,
                tx_bytes=self.num_tma_load_bytes,
                acc_stages=self.num_acc_stage,
                ab_consumers=self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1,
            )

            cute.arch.mbarrier_init_fence()

            if cute.size(self.cluster_shape_mn) > 1:
                cute.arch.cluster_arrive_relaxed(aligned=False)

            sC = storage.sC.get_tensor(
                c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
            )
            sA, sB, sSFA, sSFB = self.smem_views(
                storage,
                a_smem_layout_staged,
                b_smem_layout_staged,
                sfa_smem_layout_staged,
                sfb_smem_layout_staged,
            )

            a_full_mcast_mask = None
            b_full_mcast_mask = None
            sfa_full_mcast_mask = None
            sfb_full_mcast_mask = None
            if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast):
                a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
                )
                b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
                )
                sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
                )
                sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_sfb_vmnk,
                    block_in_cluster_coord_sfb_vmnk,
                    mcast_mode=1,
                )

            thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
            thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
            gA_mkl, tAsA, tAgA = self.partition_operand(
                mA_mkl,
                sA,
                tma_atom_a,
                cluster_layout_vmnk,
                thr_mma,
                block_in_cluster_coord_vmnk[2],
                operand="A",
                scale=False,
            )
            _, tBsB, tBgB = self.partition_operand(
                mB_nkl,
                sB,
                tma_atom_b,
                cluster_layout_vmnk,
                thr_mma,
                block_in_cluster_coord_vmnk[1],
                operand="B",
                scale=False,
            )
            _, tAsSFA, tAgSFA = self.partition_operand(
                mSFA_mkl,
                sSFA,
                tma_atom_sfa,
                cluster_layout_vmnk,
                thr_mma,
                block_in_cluster_coord_vmnk[2],
                operand="A",
                scale=True,
            )
            _, tBsSFB, tBgSFB = self.partition_operand(
                mSFB_nkl,
                sSFB,
                tma_atom_sfb,
                cluster_layout_sfb_vmnk,
                thr_mma_sfb,
                block_in_cluster_coord_sfb_vmnk[1],
                operand="B",
                scale=True,
            )
            gC_mnl = cute.local_tile(
                mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
            )
            k_block_cnt = cutlass.Int32(cute.size(gA_mkl, mode=[3]))
            tCgC = thr_mma.partition_C(gC_mnl)

            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)
            acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
            if cutlass.const_expr(self.overlapping_accum):
                num_acc_stage_overlapped = 2
                tCtAcc_fake = tiled_mma.make_fragment_C(
                    cute.append(acc_shape, num_acc_stage_overlapped)
                )
                tCtAcc_fake = cute.make_tensor(
                    tCtAcc_fake.iterator,
                    cute.make_layout(
                        tCtAcc_fake.shape,
                        stride=(
                            tCtAcc_fake.stride[0],
                            tCtAcc_fake.stride[1],
                            tCtAcc_fake.stride[2],
                            (256 - self.num_sf_tmem_cols) * tCtAcc_fake.stride[0][1],
                        ),
                    ),
                )
            else:
                tCtAcc_fake = tiled_mma.make_fragment_C(
                    cute.append(acc_shape, self.num_acc_stage)
                )

            if cute.size(self.cluster_shape_mn) > 1:
                cute.arch.cluster_wait()
            else:
                cute.arch.barrier(
                    barrier_id=self.cta_sync_bar_id, number_of_threads=192
                )

            if warp_idx <= self.mma_warp_id:
                cute.arch.barrier(barrier_id=2, number_of_threads=160)

            if warp_idx == self.tma_warp_id:
                ab_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_ab_stage
                )

                for task_index in cutlass.range(self.tasks_per_worker):
                    tile_id = bidx + task_index * self.worker_count
                    mma_tile_coord_mnl = (
                        tile_id,
                        cutlass.Int32(0),
                        cutlass.Int32(0),
                    )

                    tAgA_slice = tAgA[
                        (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                    ]
                    tBgB_slice = tBgB[
                        (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                    ]

                    tAgSFA_slice = tAgSFA[
                        (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                    ]
                    slice_n = mma_tile_coord_mnl[1]
                    if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                        slice_n = mma_tile_coord_mnl[1] // 2
                    tBgSFB_slice = tBgSFB[(None, slice_n, None, mma_tile_coord_mnl[2])]

                    if cutlass.const_expr(gate_arrivals is not None):
                        # The weight cursor reserves empty AB slots and may run
                        # ahead while activation is unavailable. The activation
                        # cursor only fills reserved slots. The full barrier
                        # expects both inputs; MMA releases the slot after use.
                        activation_state = pipeline.make_pipeline_state(
                            pipeline.PipelineUserType.Producer, self.num_ab_stage
                        )
                        # Preload weights before polling Gate, within ring
                        # capacity so activation can progress when full K won't fit.
                        preload_count = min(k_block_cnt, self.num_ab_stage)
                        for _ in cutlass.range(preload_count, unroll=1):
                            ab_pipeline.producer_acquire(ab_producer_state)
                            self._copy_tma_stage(
                                tma_atom_a,
                                tAgA_slice,
                                tAsA,
                                ab_pipeline,
                                ab_producer_state,
                                a_full_mcast_mask,
                            )
                            self._copy_tma_stage(
                                tma_atom_sfa,
                                tAgSFA_slice,
                                tAsSFA,
                                ab_pipeline,
                                ab_producer_state,
                                sfa_full_mcast_mask,
                            )
                            ab_producer_state.advance()
                        while activation_state.count < k_block_cnt:
                            if ab_producer_state.count < k_block_cnt and (True):
                                empty = (
                                    cute.arch.make_warp_uniform(
                                        ab_pipeline.producer_try_acquire(
                                            ab_producer_state
                                        ).to(cutlass.Int32)
                                    )
                                    != 0
                                )
                                if empty:
                                    ab_pipeline.producer_acquire(
                                        ab_producer_state, empty
                                    )
                                    self._copy_tma_stage(
                                        tma_atom_a,
                                        tAgA_slice,
                                        tAsA,
                                        ab_pipeline,
                                        ab_producer_state,
                                        a_full_mcast_mask,
                                    )
                                    self._copy_tma_stage(
                                        tma_atom_sfa,
                                        tAgSFA_slice,
                                        tAsSFA,
                                        ab_pipeline,
                                        ab_producer_state,
                                        sfa_full_mcast_mask,
                                    )
                                    ab_producer_state.advance()
                            if activation_state.count < ab_producer_state.count:
                                ready = (
                                    load_acquire_gpu_u32(
                                        gate_arrivals.iterator
                                        + (activation_state.count * self.mma_tiler[2])
                                        // 256
                                    )
                                    == gate_target
                                )
                                ready = (
                                    cute.arch.make_warp_uniform(ready.to(cutlass.Int32))
                                    != 0
                                )
                                if ready:
                                    self._copy_tma_stage(
                                        tma_atom_b,
                                        tBgB_slice,
                                        tBsB,
                                        ab_pipeline,
                                        activation_state,
                                        b_full_mcast_mask,
                                    )
                                    self._copy_tma_stage(
                                        tma_atom_sfb,
                                        tBgSFB_slice,
                                        tBsSFB,
                                        ab_pipeline,
                                        activation_state,
                                        sfb_full_mcast_mask,
                                    )
                                    activation_state.advance()
                    else:
                        ab_producer_state.reset_count()
                        peek_ab_empty_status = cutlass.Boolean(1)
                        if ab_producer_state.count < k_block_cnt:
                            peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                                ab_producer_state
                            )
                        for k_block in cutlass.range(0, k_block_cnt, 1, unroll=1):
                            ab_pipeline.producer_acquire(
                                ab_producer_state, peek_ab_empty_status
                            )

                            self._copy_tma_stage(
                                tma_atom_a,
                                tAgA_slice,
                                tAsA,
                                ab_pipeline,
                                ab_producer_state,
                                a_full_mcast_mask,
                            )
                            self._copy_tma_stage(
                                tma_atom_sfa,
                                tAgSFA_slice,
                                tAsSFA,
                                ab_pipeline,
                                ab_producer_state,
                                sfa_full_mcast_mask,
                            )
                            if cutlass.const_expr(gate_arrivals is not None):
                                while (
                                    load_acquire_gpu_u32(
                                        gate_arrivals.iterator
                                        + (k_block * self.mma_tiler[2]) // 256
                                    )
                                    != gate_target
                                ):
                                    pass

                            self._copy_tma_stage(
                                tma_atom_b,
                                tBgB_slice,
                                tBsB,
                                ab_pipeline,
                                ab_producer_state,
                                b_full_mcast_mask,
                            )
                            self._copy_tma_stage(
                                tma_atom_sfb,
                                tBgSFB_slice,
                                tBsSFB,
                                ab_pipeline,
                                ab_producer_state,
                                sfb_full_mcast_mask,
                            )

                            ab_producer_state.advance()
                            peek_ab_empty_status = cutlass.Boolean(1)
                            if ab_producer_state.count < k_block_cnt:
                                peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                                    ab_producer_state
                                )

                ab_pipeline.producer_tail(ab_producer_state)

            if warp_idx == self.mma_warp_id:
                tmem_ptr_read_threads = 32 * len(
                    (self.mma_warp_id, *self.epilog_warp_id)
                )
                cute.arch.barrier(
                    barrier_id=self.tmem_ptr_sync_bar_id,
                    number_of_threads=tmem_ptr_read_threads,
                )

                acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                    self.acc_dtype,
                    alignment=16,
                    ptr_to_buffer_holding_addr=tmem_holding_buf,
                )
                tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

                tCtSFA = self.sf_tmem_tensor(
                    acc_tmem_ptr + self.num_accumulator_tmem_cols,
                    tiled_mma,
                    sfa_smem_layout_staged,
                    operand="A",
                )
                tCtSFB = self.sf_tmem_tensor(
                    acc_tmem_ptr
                    + self.num_accumulator_tmem_cols
                    + self.num_sfa_tmem_cols,
                    tiled_mma,
                    sfb_smem_layout_staged,
                    operand="B",
                )
                tCtSFB_layout = tCtSFB.layout
                tiled_copy_s2t_sfa, tCsSFA_compact_s2t, tCtSFA_compact_s2t = (
                    self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
                )
                tiled_copy_s2t_sfb, tCsSFB_compact_s2t, tCtSFB_compact_s2t = (
                    self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)
                )

                ab_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_ab_stage
                )
                acc_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_acc_stage
                )

                for task_index in cutlass.range(self.tasks_per_worker):
                    tile_id = bidx + task_index * self.worker_count
                    mma_tile_coord_mnl = (
                        tile_id,
                        cutlass.Int32(0),
                        cutlass.Int32(0),
                    )

                    if cutlass.const_expr(self.overlapping_accum):
                        acc_stage_index = acc_producer_state.phase ^ 1
                    else:
                        acc_stage_index = acc_producer_state.index

                    tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]

                    ab_consumer_state.reset_count()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_block_cnt and is_leader_cta:
                        peek_ab_full_status = ab_pipeline.consumer_try_wait(
                            ab_consumer_state
                        )

                    if is_leader_cta:
                        acc_pipeline.producer_acquire(acc_producer_state)

                    tCtSFB_mma = tCtSFB
                    if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
                        offset = (
                            cutlass.Int32(2)
                            if mma_tile_coord_mnl[1] % 2 == 1
                            else cutlass.Int32(0)
                        )
                        shifted_ptr = cute.recast_ptr(
                            acc_tmem_ptr
                            + self.num_accumulator_tmem_cols
                            + self.num_sfa_tmem_cols
                            + offset,
                            dtype=self.sf_dtype,
                        )
                        tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)
                    elif cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                        offset = cutlass.Int32((mma_tile_coord_mnl[1] % 2) * 2)
                        shifted_ptr = cute.recast_ptr(
                            acc_tmem_ptr
                            + self.num_accumulator_tmem_cols
                            + self.num_sfa_tmem_cols
                            + offset,
                            dtype=self.sf_dtype,
                        )
                        tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)

                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                    for _k_block in range(k_block_cnt):
                        if is_leader_cta:
                            ab_pipeline.consumer_wait(
                                ab_consumer_state, peek_ab_full_status
                            )

                            s2t_stage_coord = (
                                None,
                                None,
                                None,
                                None,
                                ab_consumer_state.index,
                            )
                            tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[
                                s2t_stage_coord
                            ]
                            tCsSFB_compact_s2t_staged = tCsSFB_compact_s2t[
                                s2t_stage_coord
                            ]
                            cute.copy(
                                tiled_copy_s2t_sfa,
                                tCsSFA_compact_s2t_staged,
                                tCtSFA_compact_s2t,
                            )
                            cute.copy(
                                tiled_copy_s2t_sfb,
                                tCsSFB_compact_s2t_staged,
                                tCtSFB_compact_s2t,
                            )

                            num_kphases = cute.size(tCrA, mode=[2])
                            for kphase_idx in cutlass.range(
                                num_kphases, unroll_full=True
                            ):
                                kphase_coord = (
                                    None,
                                    None,
                                    kphase_idx,
                                    ab_consumer_state.index,
                                )

                                sf_kphase_coord = (None, None, kphase_idx)
                                tiled_mma.set(
                                    tcgen05.Field.SFA,
                                    tCtSFA[sf_kphase_coord].iterator,
                                )
                                tiled_mma.set(
                                    tcgen05.Field.SFB,
                                    tCtSFB_mma[sf_kphase_coord].iterator,
                                )

                                cute.gemm(
                                    tiled_mma,
                                    tCtAcc,
                                    tCrA[kphase_coord],
                                    tCrB[kphase_coord],
                                    tCtAcc,
                                )

                                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                            ab_pipeline.consumer_release(ab_consumer_state)

                        ab_consumer_state.advance()
                        peek_ab_full_status = cutlass.Boolean(1)
                        if ab_consumer_state.count < k_block_cnt:
                            if is_leader_cta:
                                peek_ab_full_status = ab_pipeline.consumer_try_wait(
                                    ab_consumer_state
                                )

                    if is_leader_cta:
                        acc_pipeline.producer_commit(acc_producer_state)
                    acc_producer_state.advance()

                acc_pipeline.producer_tail(acc_producer_state)

            if warp_idx < self.mma_warp_id:
                if warp_idx == self.epilog_warp_id[0]:
                    cute.arch.alloc_tmem(
                        self.num_tmem_alloc_cols,
                        tmem_holding_buf,
                        is_two_cta=False,
                    )

                tmem_ptr_read_threads = 32 * len(
                    (self.mma_warp_id, *self.epilog_warp_id)
                )
                cute.arch.barrier(
                    barrier_id=self.tmem_ptr_sync_bar_id,
                    number_of_threads=tmem_ptr_read_threads,
                )

                acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                    self.acc_dtype,
                    alignment=16,
                    ptr_to_buffer_holding_addr=tmem_holding_buf,
                )
                tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

                epi_tidx = tidx
                tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = (
                    self.epilog_tmem_copy_and_partition(
                        epi_tidx, tCtAcc_base, tCgC, epi_tile, False
                    )
                )

                tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
                tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                    tiled_copy_t2r, tTR_rC, epi_tidx, sC
                )
                acc_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_acc_stage
                )

                if cutlass.const_expr(local_add is not None):
                    # The MoE output lands on another stream: one epilogue
                    # thread spins, the barrier releases the rest, the
                    # gpu-scope fence makes the producer's stores visible
                    # to every publishing thread.
                    if epi_tidx == 0:
                        ready = load_acquire_gpu_u32(local_add_ready.iterator)
                        while (ready - local_add_target).to(
                            cutlass.Int32
                        ) < cutlass.Int32(0):
                            ready = load_acquire_gpu_u32(local_add_ready.iterator)
                    cute.arch.barrier(
                        barrier_id=self.epilog_sync_bar_id,
                        number_of_threads=32 * len(self.epilog_warp_id),
                    )
                    fence_acq_rel_gpu()
                for task_index in cutlass.range(self.tasks_per_worker):
                    tile_id = bidx + task_index * self.worker_count
                    mma_tile_coord_mnl = (
                        tile_id,
                        cutlass.Int32(0),
                        cutlass.Int32(0),
                    )

                    if cutlass.const_expr(self.overlapping_accum):
                        acc_stage_index = acc_consumer_state.phase
                        reverse_subtile = (
                            cutlass.Boolean(True)
                            if acc_stage_index == 0
                            else cutlass.Boolean(False)
                        )
                    else:
                        acc_stage_index = acc_consumer_state.index

                    tTR_tAcc = tTR_tAcc_base[
                        (None, None, None, None, None, acc_stage_index)
                    ]

                    acc_pipeline.consumer_wait(acc_consumer_state)

                    tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

                    subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                    num_prev_subtiles = task_index * subtile_cnt

                    for subtile_idx in cutlass.range(subtile_cnt):
                        real_subtile_idx = subtile_idx
                        if cutlass.const_expr(self.overlapping_accum):
                            if reverse_subtile:
                                real_subtile_idx = (
                                    self.cta_tile_shape_mnk[1] // self.epi_tile_n
                                    - 1
                                    - subtile_idx
                                )
                        tTR_tAcc_mn = tTR_tAcc[(None, None, None, real_subtile_idx)]
                        cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                        if cutlass.const_expr(self.overlapping_accum):
                            if subtile_idx == self.iter_acc_early_release_in_epilogue:
                                cute.arch.fence_view_async_tmem_load()
                                with cute.arch.elect_one():
                                    acc_pipeline.consumer_release(acc_consumer_state)
                                acc_consumer_state.advance()
                        acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                        acc_vec = (alpha_value * acc_vec).to(self.c_dtype)
                        tRS_rC.store(acc_vec)

                        c_buffer = (num_prev_subtiles + subtile_idx) % self.num_c_stage
                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rC,
                            tRS_sC[(None, None, None, c_buffer)],
                        )
                        # The payload gather below uses generic shared loads;
                        # the CTA barrier orders the generic shared stores.
                        epilog_threads = 32 * len(self.epilog_warp_id)
                        cute.arch.barrier(
                            barrier_id=self.epilog_sync_bar_id,
                            number_of_threads=epilog_threads,
                        )

                        epi_m = cute.size(epi_tile[0])
                        epi_n = cute.size(epi_tile[1])
                        if epi_tidx * 8 < epi_m * epi_n:
                            values = cute.make_rmem_tensor((8,), cutlass.BFloat16)
                            token = epi_tidx // 16
                            hidden_base = (epi_tidx % 16) * 8
                            for element in cutlass.range_constexpr(8):
                                values[element] = sC[
                                    hidden_base + element,
                                    token,
                                    c_buffer,
                                ]
                            payload_hidden = cutlass.Int64(tile_id) * 128 + hidden_base
                            if cutlass.const_expr(local_add is not None):
                                extra = packed_u32x4_to_bf16x8(
                                    load_global_u32x4(
                                        cute.make_ptr(
                                            cutlass.BFloat16,
                                            (
                                                local_add.iterator
                                                + cutlass.Int64(token) * 8192
                                                + payload_hidden
                                            ).toint(),
                                            cute.AddressSpace.gmem,
                                            assumed_align=16,
                                        )
                                    )
                                )
                                for element in cutlass.range_constexpr(8):
                                    values[element] = (
                                        values[element].to(cutlass.Float32)
                                        + extra[element].to(cutlass.Float32)
                                    ).to(cutlass.BFloat16)
                            payload_offset = (
                                cutlass.Int64(slot) * 1048576
                                + (
                                    (
                                        cutlass.Int64(token) * 8
                                        + cutlass.Int64(self.rank)
                                    )
                                    * 8192
                                    + payload_hidden
                                )
                                * 2
                            )
                            packed = bf16x8_to_packed_u32x4(values.load())
                            sanitized = cute.make_rmem_tensor((4,), cutlass.Uint32)
                            for word in cutlass.range_constexpr(4):
                                sanitized[word] = sanitize_oneshot_word(packed[word])
                            store_oneshot_fragment(
                                peer_payload_bases[0] + payload_offset,
                                sanitized.load(),
                            )

                    if cutlass.const_expr(not self.overlapping_accum):
                        with cute.arch.elect_one():
                            acc_pipeline.consumer_release(acc_consumer_state)
                        acc_consumer_state.advance()

                if warp_idx == self.epilog_warp_id[0]:
                    cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)
                epilog_threads = 32 * len(self.epilog_warp_id)
                cute.arch.barrier(
                    barrier_id=self.epilog_sync_bar_id, number_of_threads=epilog_threads
                )
                if warp_idx == self.epilog_warp_id[0]:
                    cute.arch.dealloc_tmem(
                        acc_tmem_ptr,
                        self.num_tmem_alloc_cols,
                        is_two_cta=False,
                    )

        if cutlass.const_expr(gate_epochs is not None):
            if tidx == 0:
                gate_epochs[worker] = gate_epochs[worker] + cutlass.Uint32(1)

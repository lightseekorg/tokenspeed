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

"""Expert GateUp v2: MMA M=128 weight rows, N=8 tokens; TMA stages weights.

A default MoE task is ``(group g, n_tile in 0..3)``: expert ``ids[g]``'s
``[128, 8192]`` gate/up weight slice times the shared ``[8, 8192]`` activation.
Rows ``0..63`` of the tile are Up, rows ``64..127`` Gate (the checkpoint
interleaves them with granularity 64), so the tile produces the 64 output
columns ``[64 * n_tile, 64 * n_tile + 64)`` of the group's ``[8, 256]`` NVFP4
GateUp output. Epilogue math, rounding order and output layout match
``gate_up_device_routine`` for bit-comparable output consumed by Down.
"""

from __future__ import annotations

from typing import Optional, Type

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass._mlir.dialects import math
from cutlass.cute.nvgpu import cpasync, tcgen05
from flashinfer.fused_moe.cute_dsl.blackwell.utils import fmin
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention._primitives import (
    make_l2_policy_evict_first,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._collective_primitives import (
    fence_proxy_async_global,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._primitives import (
    load_acquire_gpu_u32,
    red_add_relaxed_gpu_u32,
    red_add_release_gpu_u32,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._stage import (
    BlockScaledStages,
)
from tokenspeed_kernel.thirdparty.cute_dsl.nvfp4_gemm_swiglu_nvfp4_quant import (
    Sm100BlockScaledPersistentDenseGemmKernel as _BlockScaledGemm,
)

TOKENS = 8
GROUPS = 96
HIDDEN = 8192
INTERMEDIATE = 256
GATE_WEIGHT_N = 2 * INTERMEDIATE
M_TILES = GATE_WEIGHT_N // 128
SF_VEC_SIZE = 16
LOG2_E = 1.4426950408889634
FP32_MAX = 3.40282346638528859812e38


class LongcatExpertGateV2Kernel(BlockScaledStages):
    """Weight-as-M expert GateUp with a SwiGLU + NVFP4 quantization epilogue.

    Args:
        num_workers: CTAs in the launch; tasks are statically strided over them.
        k_blocks_per_stage: MMA K=64 blocks per pipeline stage (K per stage is
            ``64 * k_blocks_per_stage``; 8 gives 32 KiB weight TMA copies).
        ab_stages: weight/scale pipeline stages in flight.
        intermediate, groups: the GateUp output width per group and the group
            count.  The MoE's defaults are one expert's ``[8, 256]`` per
            group; a dense MLP is one group of its whole intermediate width
            (``1536`` per TP rank), with the output and scale layouts the
            tokens-as-M GateUp writes for ``L = 1``.
        stage_tiles: ``0`` counts ``gate_stage_arrivals[group]`` once per
            finished tile (the MoE Down waits per group); ``> 0`` counts
            ``gate_stage_arrivals[n_tile // stage_tiles]`` instead -- one word
            per Down K stage of ``64 * stage_tiles`` columns, the dense
            DownReduce's hand-off.
    """

    def __init__(
        self,
        *,
        num_workers: int,
        k_blocks_per_stage: int = 4,
        ab_stages: int = 10,
        intermediate: int = INTERMEDIATE,
        groups: int = GROUPS,
        stage_tiles: int = 0,
    ) -> None:
        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = SF_VEC_SIZE
        self.cta_group = tcgen05.CtaGroup.ONE
        self.k_blocks_per_stage = k_blocks_per_stage
        self.stage_k = 64 * k_blocks_per_stage
        if HIDDEN % self.stage_k:
            raise ValueError("k_blocks_per_stage must divide the hidden size")
        self.k_stages = HIDDEN // self.stage_k
        self.num_ab_stage = ab_stages
        self.num_acc_stage = 2
        self.num_workers = num_workers
        if intermediate % 64:
            raise ValueError("intermediate must be a multiple of 64")
        self.intermediate = intermediate
        self.gate_weight_n = 2 * intermediate
        self.m_tiles = self.gate_weight_n // 128
        self.groups = groups
        self.tasks = groups * self.m_tiles
        if stage_tiles and self.m_tiles % stage_tiles:
            raise ValueError("stage_tiles must divide the tiles per group")
        self.stage_tiles = stage_tiles
        self.mma_tiler = (128, TOKENS, self.stage_k)
        self.mma_tiler_sfb = (128, 128, self.stage_k)
        self.epi_tile = (128, TOKENS)
        # Physical warps 0-5 are required; 4-9 fails TMA launch even with a
        # warpgroup-aligned epilogue and rebased TMEM. Host other families on
        # separate CTAs.
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 192
        self.num_tmem_alloc_cols = 512
        self.buffer_align_bytes = 1024
        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.threads_per_cta
        )
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2, num_threads=32 * len(self.epilog_warp_id)
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=32 * len((self.mma_warp_id, *self.epilog_warp_id)),
        )

    @cute.jit
    def _prepare_layouts(self):
        """Set up the fixed NVFP4/E4M3, K-major MMA and shared storage."""
        self.a_dtype: Type[cutlass.Numeric] = cutlass.Float4E2M1FN
        self.b_dtype: Type[cutlass.Numeric] = cutlass.Float4E2M1FN
        self.sf_dtype: Type[cutlass.Numeric] = cutlass.Float8E4M3FN
        self.a_major_mode = tcgen05.OperandMajorMode.K
        self.b_major_mode = tcgen05.OperandMajorMode.K
        # Selects only the TMEM->register load op; all [128, 8] accumulator
        # elements are handled explicitly.
        self.c_layout = utils.LayoutEnum.ROW_MAJOR

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_tiler[:2],
        )
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage
        )
        # Share the weight ring: full-K activation residency costs 96 KiB
        # (scales pad 8 rows to 128), halving weight bytes in flight.
        # Re-reading 96 KiB per tile costs L2 bandwidth.
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage
        )

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,)
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        sfa_smem_layout = cute.slice_(
            self.sfa_smem_layout_staged, (None, None, None, 0)
        )
        sfb_smem_layout = cute.slice_(
            self.sfb_smem_layout_staged, (None, None, None, 0)
        )
        self.a_stage_bytes = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        self.b_stage_bytes = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.sfa_stage_bytes = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        self.sfb_stage_bytes = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # [128 accumulator rows, 8 tokens] fp32 epilogue exchange
            sExch: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, 128 * TOKENS],
                self.buffer_align_bytes,
            ]
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
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        self.check_smem()

        return tiled_mma, cluster_layout_vmnk

    @cute.jit
    def prepare(
        self,
        act: cute.Tensor,
        act_sf: cute.Tensor,
        weight: cute.Tensor,
        weight_sf: cute.Tensor,
    ):
        """Build activation and weight TMA descriptors and the device arguments.

        Args:
            act: ``(8, 8192, 1)`` NVFP4 activation (K major).
            act_sf: the matching swizzled e4m3 scale buffer (128-row padded).
            weight: K-major NVFP4 gate/up weights, with experts in mode 2.
            weight_sf: the matching swizzled e4m3 block16 scales.

        Returns:
            The ordered arguments consumed by
            :meth:`gate_v2_device_routine`, through its staged SMEM layouts.
        """
        tiled_mma, cluster_layout_vmnk = self._prepare_layouts()
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_tiler_sfb[:2],
        )

        cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (tiled_mma_sfb.thr_id.shape,)
        )

        sfb_tensor = cute.make_tensor(
            act_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(act.shape, self.sf_vec_size),
        )

        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        sfb_smem_layout = cute.slice_(
            self.sfb_smem_layout_staged, (None, None, None, 0)
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
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB((1, 1), tiled_mma.thr_id)
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb_tensor,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        tma_atom_a, tma_tensor_a, tma_atom_sfa, tma_tensor_sfa = self.weight_tma_atoms(
            tiled_mma, cluster_layout_vmnk, (1, 1), weight, weight_sf
        )

        return (
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfb,
            tma_tensor_sfb,
            cluster_layout_vmnk,
            cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_sfa,
            tma_tensor_sfa,
        )

    def check_smem(self) -> None:
        """Validate the SMEM budget (plain method: no ``raise`` inside a trace)."""
        capacity = utils.get_smem_capacity_in_bytes("sm_100")
        used = self.shared_storage.size_in_bytes()
        if used > capacity:
            raise ValueError(
                f"gate v2 SMEM {used} B exceeds the {capacity} B capacity "
                f"(k_blocks_per_stage={self.k_blocks_per_stage}, "
                f"ab_stages={self.num_ab_stage})"
            )

    @cute.jit
    def gate_v2_device_routine(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        group_expert_ids: cute.Tensor,
        gate_alpha: cute.Tensor,
        up_alpha: cute.Tensor,
        output_global_scale: cute.Tensor,
        gate_output: cute.Tensor,
        gate_output_scale: cute.Tensor,
        gate_stage_arrivals: cute.Tensor,
        shared_storage_ptr: cute.Pointer,
        worker_id: cutlass.Int32,
        gate: Optional[tuple] = None,
    ):
        """Run the weight-as-M GateUp on warps 0..5 of the calling CTA.

        Named barriers use explicit participant counts (192 / 128 / 160).
        The routine must occupy physical warps 0-5; see the constructor's
        warp-placement restriction.

        Args:
            tiled_mma, ..., mSFA_mkl: the :meth:`prepare` results in their
                direct return order.
            group_expert_ids: ``int32[G]`` expert id per group (-1 = empty).
            gate_alpha, up_alpha: ``float32[E]`` dequant scales.
            output_global_scale: ``float32[1]`` NVFP4 re-quant scale.
            gate_output, gate_output_scale: flat NVFP4 output and its swizzled
                e4m3 scales, both indexed by group.
            gate_stage_arrivals: ``uint32[G]`` per-group counters, or one uint32
                per Down K stage when ``stage_tiles > 0``; incremented once per
                finished ``(group, n_tile)``.
            shared_storage_ptr: caller-owned SMEM block of at least
                ``self.shared_storage.size_in_bytes()`` bytes.
            worker_id: this owner's ordinal; it runs tasks ``worker_id``,
                ``worker_id + num_workers``, ...
            gate: ``(ticket, target)`` when the caller has not
                waited for the activation yet (an engine phase issued before
                its gate): the TMA warp puts the first task's first
                ``num_ab_stage`` weight stages in flight, waits for the PDL
                predecessor grid (``griddepcontrol.wait``, a no-op once it
                has finished; in a kernel's first phase that wait is the real
                gate and ``target`` is already met), acquire-spins until
                ``*ticket >= target``, fences the async proxy (the
                activation was written with generic stores), then issues
                the activation copies. Static stride only.
        """
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        storage = self.shared_storage(shared_storage_ptr)

        sA, sB, sSFA, sSFB = self.smem_views(
            storage,
            a_smem_layout_staged,
            b_smem_layout_staged,
            sfa_smem_layout_staged,
            sfb_smem_layout_staged,
        )
        sExch = storage.sExch.get_tensor(cute.make_layout((128, TOKENS)))

        a_stage_bytes = self.stage_byte_size(self.a_dtype, a_smem_layout_staged)
        sfa_stage_bytes = self.stage_byte_size(self.sf_dtype, sfa_smem_layout_staged)
        b_stage_bytes = self.stage_byte_size(self.b_dtype, b_smem_layout_staged)
        sfb_stage_bytes = self.stage_byte_size(self.sf_dtype, sfb_smem_layout_staged)

        # Descriptor prefetch requires host-built kernel parameters (including
        # engine ``host_prepare``); device-built atoms fail to lower here.
        if tidx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfb)

        # The caller owns the init fence and participant-counted rendezvous.
        ab_pipeline, acc_pipeline = self.stage_pipelines(
            storage,
            cluster_layout_vmnk,
            tx_bytes=a_stage_bytes + sfa_stage_bytes + b_stage_bytes + sfb_stage_bytes,
            acc_stages=self.num_acc_stage,
            ab_consumers=1,
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

        thr_mma = tiled_mma.get_slice(0)
        _, tAsA, tAgA = self.partition_operand(
            mA_mkl,
            sA,
            tma_atom_a,
            cluster_layout_vmnk,
            thr_mma,
            cutlass.Int32(0),
            operand="A",
            scale=False,
        )
        _, tAsSFA, tAgSFA = self.partition_operand(
            mSFA_mkl,
            sSFA,
            tma_atom_sfa,
            cluster_layout_vmnk,
            thr_mma,
            cutlass.Int32(0),
            operand="A",
            scale=True,
        )
        _, tBsB, tBgB = self.partition_operand(
            mB_nkl,
            sB,
            tma_atom_b,
            cluster_layout_vmnk,
            thr_mma,
            cutlass.Int32(0),
            operand="B",
            scale=False,
        )
        _, tBsSFB, tBgSFB = self.partition_operand(
            mSFB_nkl,
            sSFB,
            tma_atom_sfb,
            cluster_layout_sfb_vmnk,
            tiled_mma_sfb.get_slice(0),
            cutlass.Int32(0),
            operand="B",
            scale=True,
        )

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        # Stream once-read expert weights with evict_first to protect the
        # main chain's staged L2 data.
        evict_first = make_l2_policy_evict_first()

        self.cta_sync_barrier.arrive_and_wait()

        # TMA producer warp
        if warp_idx == self.tma_warp_id:
            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            n = cutlass.Int32(0)
            task = self._static_task(n, worker_id)
            k_first = cutlass.Int32(0)
            if cutlass.const_expr(gate is not None):
                # Weights first: the first task's first stages do not depend
                # on the activation, so they go out before the gate.
                pre = min(self.num_ab_stage, self.k_stages)
                if task >= cutlass.Int32(0):
                    pre_expert = group_expert_ids[task // self.m_tiles]
                    if pre_expert >= 0:
                        pre_tile = task % self.m_tiles
                        for k_stage in cutlass.range_constexpr(pre):
                            ab_pipeline.producer_acquire(ab_producer_state)
                            cute.copy(
                                tma_atom_a,
                                tAgA[(None, pre_tile, k_stage, pre_expert)],
                                tAsA[(None, ab_producer_state.index)],
                                tma_bar_ptr=ab_pipeline.producer_get_barrier(
                                    ab_producer_state
                                ),
                                cache_policy=evict_first,
                            )
                            cute.copy(
                                tma_atom_sfa,
                                tAgSFA[(None, pre_tile, k_stage, pre_expert)],
                                tAsSFA[(None, ab_producer_state.index)],
                                tma_bar_ptr=ab_pipeline.producer_get_barrier(
                                    ab_producer_state
                                ),
                                cache_policy=evict_first,
                            )
                            ab_producer_state.advance()
                        k_first = cutlass.Int32(pre)
                gate_word, gate_target = gate
                # In a kernel's first phase the gate is also the PDL
                # predecessor (a no-op once it has finished).
                cute.arch.griddepcontrol_wait()
                with cute.arch.elect_one():
                    seen = load_acquire_gpu_u32(gate_word)
                    while (seen - gate_target).to(cutlass.Int32) < cutlass.Int32(0):
                        seen = load_acquire_gpu_u32(gate_word)
                    fence_proxy_async_global()
                cute.arch.sync_warp()
                if k_first > cutlass.Int32(0):
                    # The activation of the stages already armed, on the
                    # same full barriers (first lap: slot k is stage k).
                    act_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Producer, self.num_ab_stage
                    )
                    for k_stage in cutlass.range_constexpr(pre):
                        cute.copy(
                            tma_atom_b,
                            tBgB[(None, 0, k_stage, 0)],
                            tBsB[(None, act_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(act_state),
                        )
                        cute.copy(
                            tma_atom_sfb,
                            tBgSFB[(None, 0, k_stage, 0)],
                            tBsSFB[(None, act_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(act_state),
                        )
                        act_state.advance()
            while task >= cutlass.Int32(0):
                group = task // self.m_tiles
                n_tile = task % self.m_tiles
                expert = group_expert_ids[group]
                if expert >= 0:
                    for k_stage in cutlass.range(k_first, self.k_stages, 1, unroll=1):
                        ab_pipeline.producer_acquire(ab_producer_state)
                        cute.copy(
                            tma_atom_a,
                            tAgA[(None, n_tile, k_stage, expert)],
                            tAsA[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(
                                ab_producer_state
                            ),
                            cache_policy=evict_first,
                        )
                        cute.copy(
                            tma_atom_sfa,
                            tAgSFA[(None, n_tile, k_stage, expert)],
                            tAsSFA[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(
                                ab_producer_state
                            ),
                            cache_policy=evict_first,
                        )
                        # Warp-collective: issued by the whole TMA warp.
                        cute.copy(
                            tma_atom_b,
                            tBgB[(None, 0, k_stage, 0)],
                            tBsB[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(
                                ab_producer_state
                            ),
                        )
                        cute.copy(
                            tma_atom_sfb,
                            tBgSFB[(None, 0, k_stage, 0)],
                            tBsSFB[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(
                                ab_producer_state
                            ),
                        )
                        ab_producer_state.advance()
                n = n + cutlass.Int32(1)
                task = self._static_task(n, worker_id)
                k_first = cutlass.Int32(0)
            ab_pipeline.producer_tail(ab_producer_state)

        # MMA warp
        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
            tCtSFA = self.sf_tmem_tensor(
                acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base),
                tiled_mma,
                sfa_smem_layout_staged,
                operand="A",
            )
            tCtSFB = self.sf_tmem_tensor(
                acc_tmem_ptr
                + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base)
                + tcgen05.find_tmem_tensor_col_offset(tCtSFA),
                tiled_mma,
                sfb_smem_layout_staged,
                operand="B",
            )
            (
                tiled_copy_s2t_sfa,
                tCsSFA_compact_s2t,
                tCtSFA_compact_s2t,
            ) = self.s2t_copy_and_partition(sSFA, tCtSFA)
            (
                tiled_copy_s2t_sfb,
                tCsSFB_compact_s2t,
                tCtSFB_compact_s2t,
            ) = self.s2t_copy_and_partition(sSFB, tCtSFB)

            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            n = cutlass.Int32(0)
            task = self._static_task(n, worker_id)
            while task >= cutlass.Int32(0):
                group = task // self.m_tiles
                expert = group_expert_ids[group]
                if expert >= 0:
                    tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]
                    acc_pipeline.producer_acquire(acc_producer_state)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for k_stage in cutlass.range(0, self.k_stages, 1, unroll=1):
                        ab_pipeline.consumer_wait(ab_consumer_state)
                        cute.copy(
                            tiled_copy_s2t_sfa,
                            tCsSFA_compact_s2t[
                                (None, None, None, None, ab_consumer_state.index)
                            ],
                            tCtSFA_compact_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb,
                            tCsSFB_compact_s2t[
                                (None, None, None, None, ab_consumer_state.index)
                            ],
                            tCtSFB_compact_s2t,
                        )
                        for kblock_idx in cutlass.range_constexpr(
                            self.k_blocks_per_stage
                        ):
                            sf_kblock_coord = (None, None, kblock_idx)
                            tiled_mma.set(
                                tcgen05.Field.SFA, tCtSFA[sf_kblock_coord].iterator
                            )
                            tiled_mma.set(
                                tcgen05.Field.SFB, tCtSFB[sf_kblock_coord].iterator
                            )
                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[
                                    (
                                        None,
                                        None,
                                        kblock_idx,
                                        ab_consumer_state.index,
                                    )
                                ],
                                tCrB[
                                    (
                                        None,
                                        None,
                                        kblock_idx,
                                        ab_consumer_state.index,
                                    )
                                ],
                                tCtAcc,
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        ab_pipeline.consumer_release(ab_consumer_state)
                        ab_consumer_state.advance()
                    acc_pipeline.producer_commit(acc_producer_state)
                    acc_producer_state.advance()
                n = n + cutlass.Int32(1)
                task = self._static_task(n, worker_id)
            acc_pipeline.producer_tail(acc_producer_state)

        # Epilogue warps
        if warp_idx < self.mma_warp_id:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            copy_atom_t2r = sm100_utils.get_tmem_load_op(
                self.mma_tiler,
                self.c_layout,
                self.acc_dtype,
                self.acc_dtype,
                self.epi_tile,
                False,
            )
            tAcc_epi = cute.flat_divide(
                tCtAcc_base[((None, None), 0, 0, None)], self.epi_tile
            )
            tiled_copy_t2r = tcgen05.make_tmem_copy(
                copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
            )
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
            cC_epi = cute.flat_divide(
                cute.make_identity_tensor(self.epi_tile), self.epi_tile
            )
            tTR_cC = thr_copy_t2r.partition_D(cC_epi)
            tTR_rAcc = cute.make_rmem_tensor(
                tTR_cC[(None, None, None, 0, 0)].shape, self.acc_dtype
            )
            num_acc_vals = cute.size(tTR_rAcc)

            norm_const = output_global_scale[0]
            # [16 output rows, one token] per thread: the same 16-element
            # quantization block the inherited epilogue holds in one thread.
            rVals = cute.make_rmem_tensor((SF_VEC_SIZE,), self.acc_dtype)
            rOut = cute.make_rmem_tensor((SF_VEC_SIZE,), cutlass.Float4E2M1FN)
            token = tidx // 4
            blk = tidx % 4

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            n = cutlass.Int32(0)
            task = self._static_task(n, worker_id)
            while task >= cutlass.Int32(0):
                group = task // self.m_tiles
                n_tile = task % self.m_tiles
                arrival = group
                if cutlass.const_expr(self.stage_tiles > 0):
                    arrival = n_tile // self.stage_tiles
                expert = group_expert_ids[group]
                if expert >= 0:
                    alpha_gate = gate_alpha[expert]
                    alpha_up = up_alpha[expert]
                    acc_pipeline.consumer_wait(acc_consumer_state)
                    cute.copy(
                        tiled_copy_t2r,
                        tTR_tAcc[(None, None, None, 0, 0, acc_consumer_state.index)],
                        tTR_rAcc,
                    )
                    cute.arch.fence_view_async_tmem_load()
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

                    for i in cutlass.range_constexpr(num_acc_vals):
                        sExch[tidx, i] = tTR_rAcc[i]
                    self.epilog_sync_barrier.arrive_and_wait()

                    if tidx < 32:
                        # SwiGLU on (Up row j, Gate row j + 64), matching the
                        # inherited vectorized epilogue operation for operation.
                        for e in cutlass.range_constexpr(SF_VEC_SIZE):
                            row = blk * SF_VEC_SIZE + e
                            up_a = sExch[row, token] * alpha_up
                            gate_a = sExch[row + 64, token] * alpha_gate
                            sig = cute.math.exp2(
                                gate_a * cutlass.Float32(-LOG2_E), fastmath=True
                            )
                            sig = sig + cutlass.Float32(1.0)
                            sig = cute.arch.rcp_approx(sig)
                            sig = sig * gate_a
                            rVals[e] = sig * up_a

                        acc_frg = rVals.load()
                        abs_frg_ir = math.absf(acc_frg.ir_value())
                        abs_frg = type(acc_frg)(
                            abs_frg_ir, acc_frg.shape, acc_frg.dtype
                        )
                        pvscale = abs_frg.reduce(
                            cute.ReductionOp.MAX, cutlass.Float32(0.0), 0
                        )
                        pvscale = pvscale * cutlass.Float32(1.0 / 6.0)
                        pvscale = pvscale * norm_const
                        sfc = pvscale.to(cutlass.Float8E4M3FN)
                        # The 128x4 swizzled scale layout: 512 bytes per four
                        # scale columns (one n_tile), 128 rows per group.
                        gate_output_scale[
                            group * (8 * self.intermediate)
                            + token * 16
                            + blk
                            + n_tile * 512
                        ] = sfc
                        acc_scale = cute.arch.rcp_approx(sfc.to(cutlass.Float32))
                        acc_scale = acc_scale * norm_const
                        acc_scale = fmin(acc_scale, cutlass.Float32(FP32_MAX), nan=True)
                        for e in cutlass.range_constexpr(SF_VEC_SIZE):
                            rVals[e] = rVals[e] * acc_scale
                        rOut.store(rVals.load().to(cutlass.Float4E2M1FN))
                        rOut_bytes = cute.recast_tensor(rOut, cutlass.Uint8)
                        out_base = (
                            group * (TOKENS * self.intermediate // 2)
                            + token * (self.intermediate // 2)
                            + n_tile * 32
                            + blk * 8
                        )
                        gOut = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.Uint8,
                                gate_output.iterator.toint() + out_base,
                                cute.AddressSpace.gmem,
                                assumed_align=8,
                            ),
                            cute.make_layout((SF_VEC_SIZE // 2,)),
                        )
                        cute.autovec_copy(rOut_bytes, gOut)
                        # Down reads the output with TMA (the async proxy);
                        # these are generic-proxy stores.
                        fence_proxy_async_global()

                    self.epilog_sync_barrier.arrive_and_wait()
                    if tidx == 0:
                        # Release-publish barrier-ordered tile stores to Down's
                        # acquire poll; non-returning avoids a per-tile round trip.
                        red_add_release_gpu_u32(
                            gate_stage_arrivals.iterator + arrival,
                            cutlass.Uint32(1),
                        )
                else:
                    # Keep skipped-group counters aligned for later launches:
                    # Down expects (generation + 1) * tiles per counter. Down
                    # also skips the group, so relaxed non-returning is enough.
                    if tidx == 0:
                        red_add_relaxed_gpu_u32(
                            gate_stage_arrivals.iterator + arrival,
                            cutlass.Uint32(1),
                        )
                n = n + cutlass.Int32(1)
                task = self._static_task(n, worker_id)

            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(acc_tmem_ptr)

        self.cta_sync_barrier.arrive_and_wait()

    @cute.jit
    def _static_task(self, n, worker_id):
        """The n-th task of this CTA's fixed stride, -1 past the end."""
        task = cutlass.Int32(0) + worker_id + cutlass.Int32(self.num_workers) * n
        if task >= cutlass.Int32(self.task_bound()):
            task = cutlass.Int32(-1)
        return task

    def task_bound(self) -> int:
        """End of the statically strided task range."""
        return self.tasks

    s2t_copy_and_partition = _BlockScaledGemm.mainloop_s2t_copy_and_partition


def _make_operands(
    weight_ptr: cute.Pointer,
    weight_sf_ptr: cute.Pointer,
    expert_count: cutlass.Constexpr,
    gate_weight_n: cutlass.Constexpr = GATE_WEIGHT_N,
):
    """The weight operands in the layouts the expert entry already uses."""
    weight = cute.make_tensor(
        weight_ptr,
        cute.make_ordered_layout(
            (gate_weight_n, HIDDEN, expert_count), order=(1, 0, 2)
        ),
    )
    weight_sf = cute.make_tensor(
        weight_sf_ptr,
        cute.make_ordered_layout(
            (
                32,
                4,
                gate_weight_n // 128,
                4,
                (HIDDEN // 16) // 4,
                expert_count,
            ),
            order=(2, 1, 4, 0, 3, 5),
        ),
    )
    return weight, weight_sf


__all__ = ["LongcatExpertGateV2Kernel"]

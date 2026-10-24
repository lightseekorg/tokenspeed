# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""LongCat-owned verify-8 Down device routine.

Ported from FlashInfer's ``dense_blockscaled_gemm_sm100.py``. The K-stage
routine owns LongCat's static scheduling, readiness and publication protocol.
"""

from __future__ import annotations

from typing import Optional, Type

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.arch import griddepcontrol_launch_dependents
from cutlass.cute.nvgpu import cpasync, tcgen05
from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm100 import (
    Sm100BlockScaledPersistentDenseGemmKernel,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention._primitives import (
    make_l2_policy_evict_first,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._primitives import (
    load_acquire_gpu_u32,
    red_add_gpu_f32,
    red_add_release_gpu_u32,
    store_release_gpu_u32,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._stage import (
    BlockScaledStages,
)


class LongcatPersistentDownKStageKernel(
    BlockScaledStages, Sm100BlockScaledPersistentDenseGemmKernel
):
    """Consume expert GateUp generations with hidden-slab Down owners."""

    # Group-resident B/SFB ring depth for the hidden-slab routine.
    num_b_group_stage = 4
    # Two TMEM stages overlap the next pass's MMA with the epilogue.
    num_group_acc_stage = 2
    # Hidden M-tiles per TMEM->register copy; four tiles amortize the
    # ~450-cycle latency of an eight-fp32-column ``tcgen05.ld``.
    num_group_tiles_per_t2r = 4
    # Expert groups per hidden-slab TMEM pass. The trace-time group count
    # is capped by the TMEM budget and must divide the group count.
    num_slab_groups_per_pass = 1

    def __init__(
        self,
        *args,
        gate_stage_producers_per_generation: int = 4,
        weight_prefetch_depth: Optional[int] = None,
        **kwargs,
    ) -> None:
        use_prefetch = bool(kwargs.get("use_prefetch", False))
        if weight_prefetch_depth is None:
            weight_prefetch_depth = 1 if use_prefetch else 0
        if weight_prefetch_depth not in (0, 1, 2, 4, 6):
            raise ValueError("weight_prefetch_depth must be one of {0, 1, 2, 4, 6}")
        kwargs["use_prefetch"] = weight_prefetch_depth > 0
        super().__init__(*args, **kwargs)
        self.gate_stage_producers_per_generation = gate_stage_producers_per_generation
        self.weight_prefetch_depth = weight_prefetch_depth

    @cute.jit
    def _prepare_layouts(self):
        """Set up the fixed NVFP4/E4M3, K-major MMA and shared storage."""
        self.a_dtype: Type[cutlass.Numeric] = cutlass.Float4E2M1FN
        self.b_dtype: Type[cutlass.Numeric] = cutlass.Float4E2M1FN
        self.sf_dtype: Type[cutlass.Numeric] = cutlass.Float8E4M3FN
        self.c_dtype: Type[cutlass.Numeric] = cutlass.BFloat16
        self.a_major_mode = tcgen05.OperandMajorMode.K
        self.b_major_mode = tcgen05.OperandMajorMode.K
        self.c_layout = utils.LayoutEnum.COL_MAJOR

        self._setup_attributes()
        if cutlass.const_expr(self.weight_prefetch_depth > self.num_ab_stage):
            raise ValueError(
                "weight_prefetch_depth exceeds the compiled Down SMEM ring: "
                f"{self.weight_prefetch_depth} > {self.num_ab_stage}"
            )

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mnk[:2],
        )

        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mnk_sfb[:2],
        )

        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        sfa_smem_layout = cute.slice_(
            self.sfa_smem_layout_staged, (None, None, None, 0)
        )
        sfb_smem_layout = cute.slice_(
            self.sfb_smem_layout_staged, (None, None, None, 0)
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        # Expose compile-time A/B/SFA/SFB ring sizes for storage accounting.
        self.a_stage_bytes = a_copy_size
        self.b_stage_bytes = b_copy_size
        self.sfa_stage_bytes = sfa_copy_size
        self.sfb_stage_bytes = sfb_copy_size
        self.num_tma_load_bytes = (
            a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
        ) * atom_thr_size

        self.buffer_align_bytes = 1024

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            # Retain the base accumulator depth alongside the hidden-slab
            # pipeline's ``num_group_acc_stage``.
            acc_full_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, max(self.num_acc_stage, self.num_group_acc_stage)
            ]
            acc_empty_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, max(self.num_acc_stage, self.num_group_acc_stage)
            ]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            # B/SFB barriers cover both group-resident slots and weight-ring
            # stages.
            b_group_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, max(self.num_b_group_stage, self.num_ab_stage)
            ]
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

        return tiled_mma, tiled_mma_sfb

    @cute.jit
    def prepare_k_stage_device_routine(
        self,
        act: cute.Tensor,
        act_sf: cute.Tensor,
        weight: cute.Tensor,
        weight_sf: cute.Tensor,
    ):
        """Build A/SFA and B/SFB descriptors for NVFP4 weights and activations.

        Args:
            act: K-major NVFP4 activation tensor.
            act_sf: E4M3 block16 scales for ``act``.
            weight: K-major NVFP4 expert weight tensor.
            weight_sf: E4M3 block16 scales for ``weight``.

        Returns the two MMAs, B/SFB atoms and tensors, cluster layouts,
        staged A/B/SFA/SFB layouts and A/SFA atoms and tensors consumed by
        the hidden-slab device routine.
        """
        tiled_mma, tiled_mma_sfb = self._prepare_layouts()

        # ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            act.shape, self.sf_vec_size
        )
        act_sf = cute.make_tensor(act_sf.iterator, sfb_layout)

        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            act,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfb_smem_layout = cute.slice_(
            self.sfb_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            act_sf,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
            x = tma_tensor_sfb.stride[0][1]
            y = cute.ceil_div(tma_tensor_sfb.shape[0][1], 4)

            new_shape = (
                (tma_tensor_sfb.shape[0][0], ((2, 2), y)),
                tma_tensor_sfb.shape[1],
                tma_tensor_sfb.shape[2],
            )
            # Use right multiplication for ScaledBasis (3 * x instead of x * 3)
            x_times_3 = 3 * x
            new_stride = (
                (tma_tensor_sfb.stride[0][0], ((x, x), x_times_3)),
                tma_tensor_sfb.stride[1],
                tma_tensor_sfb.stride[2],
            )
            tma_tensor_sfb_new_layout = cute.make_layout(new_shape, stride=new_stride)
            tma_tensor_sfb = cute.make_tensor(
                tma_tensor_sfb.iterator, tma_tensor_sfb_new_layout
            )

        tma_atom_a, tma_tensor_a, tma_atom_sfa, tma_tensor_sfa = self.weight_tma_atoms(
            tiled_mma,
            self.cluster_layout_vmnk,
            self.cluster_shape_mn,
            weight,
            weight_sf,
        )

        return (
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfb,
            tma_tensor_sfb,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_sfa,
            tma_tensor_sfa,
        )

    @cute.jit
    def _live_passes(
        self,
        route_expert_ids: cute.Tensor,
        groups_per_split: cutlass.Constexpr,
        group_splits: cutlass.Constexpr,
        split_idx: cutlass.Int32,
        gpp: cutlass.Constexpr,
    ) -> cutlass.Int32:
        """Count ``gpp``-slot windows through this split's last routed slot.

        Live groups precede -1 slots, so interleaved split slots
        ``step * group_splits + split_idx`` also have a live prefix.
        Warp ballots count 32 steps at a time; every calling warp agrees.
        """
        lane = cute.arch.lane_idx()
        live = cutlass.Int32(0)
        for base in cutlass.range_constexpr(0, groups_per_split, 32):
            is_live = cutlass.Boolean(False)
            if lane + base < groups_per_split:
                step = lane + base
                is_live = route_expert_ids[step * group_splits + split_idx] >= 0
            live = live + cutlass.Int32(
                cute.arch.popc(cute.arch.vote_ballot_sync(is_live))
            )
        return (live + gpp - 1) // gpp

    @cute.jit
    def down_hidden_slab_device_routine(
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
        alpha: cute.Tensor,
        gate_stage_arrivals: cute.Tensor,
        route_output_arrivals: cute.Tensor,
        route_expert_ids: cute.Tensor,
        group_table: cute.Tensor,
        group_acc: cute.Tensor,
        group_done: cute.Tensor,
        shared_storage_ptr: cute.Pointer,
        worker_id: cutlass.Int32,
        worker_count: cutlass.Constexpr,
        group_count: cutlass.Constexpr,
        group_splits: cutlass.Constexpr = 1,
        release_pdl_dependents: cutlass.Constexpr = False,
    ):
        """Own a hidden slab over an interleaved partition of expert groups.

        With ``S = worker_count // group_splits``, owner ``w`` takes hidden
        tiles ``[(w % S) * slab, (w % S + 1) * slab)`` and groups
        ``step * group_splits + w // S``. Its ``slab * 128`` rows x 8 tokens
        accumulate in FP32 registers and are written once at exit: plain
        stores for one split, FP32 atomic adds to zeroed output otherwise.
        Each owner reads only its slab of each assigned expert shard.

        Pass windows have a trace-time constant width (dead slots included),
        so MMA and epilogue agree without sharing a routed-group count.

        Args:
            tiled_mma, ..., mSFA_mkl: MMA/TMA tensors and layouts from
                :meth:`prepare_k_stage_device_routine`.
            alpha: ``float32[E]`` Down dequant scale, indexed by ``ids[g]``.
            gate_stage_arrivals: ``uint32[G]`` GateUp producer counter.
            route_output_arrivals: entry ``worker_id`` is this owner's private
                launch generation (the per-group entries used by the other
                routines are untouched); every owner reads and bumps only its
                own, so no owner can observe another's advance.
            route_expert_ids: ``int32[G]``; ``ids[g] < 0`` skips the group.
            group_table: ``float32[G * 8]`` route weight per (group, token).
            group_acc: ``float32[8 * 8192]`` token-major output; must start
                at zero when ``group_splits > 1``.
            group_done: ``uint32[1]``; bumped once per owner at exit.
            shared_storage_ptr: 128 B aligned caller-owned SMEM block.
            worker_id, worker_count: this owner's ordinal and the owner count.
            group_count: total group slots.
            group_splits: interleaved group partitions; must divide both counts,
                and ``worker_count // group_splits`` must divide hidden tiles.
            release_pdl_dependents: emit ``griddepcontrol.launch_dependents``.
        """
        if cutlass.const_expr(self.cluster_shape_mn != (1, 1)):
            raise ValueError("the hidden-slab Down routine is one-CTA only")
        if cutlass.const_expr(self.cta_tile_shape_mnk[1] != 8):
            raise ValueError("the hidden-slab Down routine needs an N = 8 tile")

        epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk, self.use_2cta_instrs, self.c_layout, self.c_dtype
        )
        num_tokens = 8
        hidden = 8192
        tiles_per_group = hidden // self.cta_tile_shape_mnk[0]
        # Owners form a slab_owners x group_splits grid. Owners sharing a
        # slab reduce their accumulated rows once each in the final write.
        if cutlass.const_expr(worker_count % group_splits):
            raise ValueError("group_splits must divide the owner count")
        slab_owners = worker_count // group_splits
        if cutlass.const_expr(tiles_per_group % slab_owners):
            raise ValueError(
                f"slab owner count {slab_owners} must divide {tiles_per_group} "
                "hidden tiles"
            )
        if cutlass.const_expr(group_count % group_splits):
            raise ValueError("group_splits must divide the group count")
        groups_per_split = group_count // group_splits
        slab_tiles = tiles_per_group // slab_owners
        t2r_tiles = min(self.num_group_tiles_per_t2r, slab_tiles)
        if cutlass.const_expr(slab_tiles % t2r_tiles):
            raise ValueError("the slab must be a whole number of t2r copies")
        t2r_slots = slab_tiles // t2r_tiles
        # Bound pass width by the class default and TMEM budget; require
        # whole windows below. ``num_sfa_tmem_cols`` is staged, so reserve
        # 32 columns per scale region: double-buffered SFA plus SFB.
        tmem_budget = 512 - 96
        # Literal N=8 and two accumulator stages keep ``gpp`` trace-time;
        # reading them from ``self`` yields staged Int32 loop bounds.
        gpp = min(
            self.num_slab_groups_per_pass,
            max(1, tmem_budget // (slab_tiles * 8 * 2)),
        )
        # A ``while`` would stage ``gpp``, invalidating ``range_constexpr``.
        if cutlass.const_expr(groups_per_split % gpp):
            raise ValueError(
                f"num_slab_groups_per_pass {gpp} must divide {groups_per_split}"
            )
        num_passes = groups_per_split // gpp
        # Trace-time validation prevents TMEM overrun; a staged ``if``
        # cannot raise early.
        if cutlass.const_expr(gpp * slab_tiles * 8 * 2 > tmem_budget):
            raise ValueError(
                f"a {slab_tiles}-tile slab with {gpp} groups per pass needs "
                f"{gpp * slab_tiles * 8 * 2} TMEM columns, over the "
                f"{tmem_budget} available"
            )
        b_stages = self.num_b_group_stage
        if cutlass.const_expr(self.num_ab_stage > (b_stages - 1) * slab_tiles):
            raise ValueError("Down SMEM ring exceeds the safe B-ring reuse window")

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfb)

        storage = self.shared_storage(shared_storage_ptr)
        tmem_holding_buf = storage.tmem_holding_buf.ptr
        b_mbar_base = storage.b_group_mbar_ptr.data_ptr()

        sA, sB, sSFA, sSFB = self.smem_views(
            storage,
            a_smem_layout_staged,
            b_smem_layout_staged,
            sfa_smem_layout_staged,
            sfb_smem_layout_staged,
        )
        a_stage_bytes = self.stage_byte_size(self.a_dtype, a_smem_layout_staged)
        sfa_stage_bytes = self.stage_byte_size(self.sf_dtype, sfa_smem_layout_staged)
        b_stage_bytes = self.stage_byte_size(self.b_dtype, b_smem_layout_staged)
        sfb_stage_bytes = self.stage_byte_size(self.sf_dtype, sfb_smem_layout_staged)

        # The caller owns the init fence and participant-counted rendezvous.
        ab_pipeline, acc_pipeline = self.stage_pipelines(
            storage,
            cluster_layout_vmnk,
            tx_bytes=a_stage_bytes + sfa_stage_bytes,
            acc_stages=self.num_group_acc_stage,
            ab_consumers=1,
        )
        if warp_idx == self.tma_warp_id:
            with cute.arch.elect_one():
                for b_stage in cutlass.range_constexpr(b_stages):
                    cute.arch.mbarrier_init(b_mbar_base + b_stage, 1)
        cute.arch.mbarrier_init_fence()

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

        # One 8-column accumulator slot per slab tile, two slabs resident so
        # the MMA of group g + 1 overlaps the epilogue of group g.
        acc_slots = gpp * slab_tiles * self.num_group_acc_stage
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, acc_slots))
        acc_col_stride = tCtAcc_fake.stride[0][1]
        tCtAcc_fake = cute.make_tensor(
            tCtAcc_fake.iterator,
            cute.make_layout(
                tCtAcc_fake.shape,
                stride=(
                    tCtAcc_fake.stride[0],
                    tCtAcc_fake.stride[1],
                    tCtAcc_fake.stride[2],
                    self.cta_tile_shape_mnk[1] * acc_col_stride,
                ),
            ),
        )
        num_group_acc_cols = acc_slots * self.cta_tile_shape_mnk[1]
        t2r_cols = t2r_tiles * self.cta_tile_shape_mnk[1]
        tCtAcc_wide_layout = cute.make_layout(
            (
                (self.cta_tile_shape_mnk[0], t2r_cols),
                1,
                1,
                gpp * t2r_slots * self.num_group_acc_stage,
            ),
            stride=(
                (tCtAcc_fake.stride[0][0], acc_col_stride),
                tCtAcc_fake.stride[1],
                tCtAcc_fake.stride[2],
                t2r_cols * acc_col_stride,
            ),
        )
        epi_tile_wide = (epi_tile[0], cute.make_layout(t2r_cols))

        slab_base_tile = (worker_id % cutlass.Int32(slab_owners)) * cutlass.Int32(
            slab_tiles
        )
        # Interleave groups across owners to distribute GateUp arrival waits.
        split_idx = worker_id // cutlass.Int32(slab_owners)

        # Stream once-read expert weights with evict_first to protect the
        # main chain's staged L2 data.
        evict_first = make_l2_policy_evict_first()

        cute.arch.barrier(
            barrier_id=self.cta_sync_bar_id, number_of_threads=self.threads_per_cta
        )

        # TMA producer warp
        if warp_idx == self.tma_warp_id:
            weight_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            # Read the launch generation once into every lane before
            # elect_one, so each elected lane has the warp-uniform value.
            generation = load_acquire_gpu_u32(
                route_output_arrivals.iterator + worker_id
            )
            expected_count = (generation + cutlass.Uint32(1)) * cutlass.Uint32(
                self.gate_stage_producers_per_generation
            )
            b_count = cutlass.Int32(0)
            # Weights are independent of GateUp; B/SFB loads require its
            # ticket. The B ring has no empty barrier, so its producer must
            # never overtake unread slots. Weight-ring producer_acquire can
            # already advance ceil(num_ab_stage / slab_tiles) groups; only
            # the remaining B-ring capacity could support extra lookahead.
            # Extra lookahead is disabled.
            B_LOOKAHEAD = cutlass.const_expr(0)
            for local_step in cutlass.range(
                0, groups_per_split + B_LOOKAHEAD, 1, unroll=1
            ):
                step = local_step * cutlass.Int32(group_splits) + split_idx
                if local_step < groups_per_split:
                    issue_expert = route_expert_ids[step]
                    if issue_expert >= 0:
                        b_slot = b_count % b_stages
                        with cute.arch.elect_one():
                            producer_count = load_acquire_gpu_u32(
                                gate_stage_arrivals.iterator + step
                            )
                            while (producer_count - expected_count).to(
                                cutlass.Int32
                            ) < cutlass.Int32(0):
                                producer_count = load_acquire_gpu_u32(
                                    gate_stage_arrivals.iterator + step
                                )
                        cute.arch.sync_warp()
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                b_mbar_base + b_slot,
                                cutlass.Int32(b_stage_bytes + sfb_stage_bytes),
                            )
                        cute.copy(
                            tma_atom_b,
                            tBgB[(None, 0, 0, step)],
                            tBsB[(None, b_slot)],
                            tma_bar_ptr=b_mbar_base + b_slot,
                        )
                        cute.copy(
                            tma_atom_sfb,
                            tBgSFB[(None, 0, 0, step)],
                            tBsSFB[(None, b_slot)],
                            tma_bar_ptr=b_mbar_base + b_slot,
                        )
                        b_count = b_count + 1
                group = (local_step - B_LOOKAHEAD) * cutlass.Int32(
                    group_splits
                ) + split_idx
                if local_step >= B_LOOKAHEAD:
                    expert_id = route_expert_ids[group]
                    if expert_id >= 0:
                        for slab_idx in cutlass.range(0, slab_tiles, 1, unroll=1):
                            m_tile = slab_base_tile + slab_idx
                            ab_pipeline.producer_acquire(weight_state)
                            cute.copy(
                                tma_atom_a,
                                tAgA[(None, m_tile, 0, expert_id)],
                                tAsA[(None, weight_state.index)],
                                tma_bar_ptr=ab_pipeline.producer_get_barrier(
                                    weight_state
                                ),
                                cache_policy=evict_first,
                            )
                            cute.copy(
                                tma_atom_sfa,
                                tAgSFA[(None, m_tile, 0, expert_id)],
                                tAsSFA[(None, weight_state.index)],
                                tma_bar_ptr=ab_pipeline.producer_get_barrier(
                                    weight_state
                                ),
                                cache_policy=evict_first,
                            )
                            weight_state.advance()
            ab_pipeline.producer_tail(weight_state)
            with cute.arch.elect_one():
                store_release_gpu_u32(
                    route_output_arrivals.iterator + worker_id,
                    generation + cutlass.Uint32(1),
                )

        # MMA warp
        if warp_idx == self.mma_warp_id:
            tmem_ptr_read_threads = 32 * len((self.mma_warp_id, *self.epilog_warp_id))
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
            # Double-buffer SFA so tile i + 1's SMEM->TMEM scale copy does
            # not overwrite scales still read by tile i's MMAs.
            tCtSFA_buf = [
                self.sf_tmem_tensor(
                    acc_tmem_ptr + num_group_acc_cols + buf * self.num_sfa_tmem_cols,
                    tiled_mma,
                    sfa_smem_layout_staged,
                    operand="A",
                )
                for buf in range(2)
            ]
            tCtSFB = self.sf_tmem_tensor(
                acc_tmem_ptr + num_group_acc_cols + 2 * self.num_sfa_tmem_cols,
                tiled_mma,
                sfb_smem_layout_staged,
                operand="B",
            )
            s2t_sfa = [
                self.mainloop_s2t_copy_and_partition(sSFA, t) for t in tCtSFA_buf
            ]
            tiled_copy_s2t_sfb, tCsSFB_compact_s2t, tCtSFB_compact_s2t = (
                self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)
            )

            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_group_acc_stage
            )
            num_kphases = cute.size(tCrA, mode=[2])
            b_count = cutlass.Int32(0)
            # Acquire/commit once per fixed-width window, including its dead
            # slots; stop at the last live window to avoid empty round trips.
            live_passes = self._live_passes(
                route_expert_ids, groups_per_split, group_splits, split_idx, gpp
            )
            for pass_idx in cutlass.range(0, live_passes, 1, unroll=1):
                acc_pipeline.producer_acquire(acc_producer_state)
                acc_pass_base = acc_producer_state.index * (gpp * slab_tiles)
                for j in cutlass.range_constexpr(gpp):
                    group = (pass_idx * gpp + j) * cutlass.Int32(
                        group_splits
                    ) + split_idx
                    expert_id = route_expert_ids[group]
                    if expert_id >= 0:
                        b_slot = b_count % b_stages
                        b_phase = (b_count // b_stages) % 2
                        b_count = b_count + 1
                        cute.arch.mbarrier_wait(b_mbar_base + b_slot, b_phase)
                        cute.copy(
                            tiled_copy_s2t_sfb,
                            tCsSFB_compact_s2t[(None, None, None, None, b_slot)],
                            tCtSFB_compact_s2t,
                        )
                        acc_slot_base = acc_pass_base + j * slab_tiles
                        for slab_idx in cutlass.range_constexpr(slab_tiles):
                            buf = cutlass.const_expr(slab_idx % 2)
                            (
                                tiled_copy_s2t_sfa,
                                tCsSFA_compact_s2t,
                                tCtSFA_compact_s2t,
                            ) = s2t_sfa[buf]
                            ab_pipeline.consumer_wait(ab_consumer_state)
                            cute.copy(
                                tiled_copy_s2t_sfa,
                                tCsSFA_compact_s2t[
                                    (None, None, None, None, ab_consumer_state.index)
                                ],
                                tCtSFA_compact_s2t,
                            )
                            tCtAcc = tCtAcc_base[
                                (None, None, None, acc_slot_base + slab_idx)
                            ]
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                            for kphase_idx in cutlass.range(
                                num_kphases, unroll_full=True
                            ):
                                sf_kphase_coord = (None, None, kphase_idx)
                                tiled_mma.set(
                                    tcgen05.Field.SFA,
                                    tCtSFA_buf[buf][sf_kphase_coord].iterator,
                                )
                                tiled_mma.set(
                                    tcgen05.Field.SFB, tCtSFB[sf_kphase_coord].iterator
                                )
                                cute.gemm(
                                    tiled_mma,
                                    tCtAcc,
                                    tCrA[
                                        (
                                            None,
                                            None,
                                            kphase_idx,
                                            ab_consumer_state.index,
                                        )
                                    ],
                                    tCrB[(None, None, kphase_idx, b_slot)],
                                    tCtAcc,
                                )
                                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            ab_pipeline.consumer_release(ab_consumer_state)
                            ab_consumer_state.advance()
                acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()
            acc_pipeline.producer_tail(acc_producer_state)

        # Epilogue warps
        if warp_idx < self.mma_warp_id:
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.alloc_tmem(
                    self.num_tmem_alloc_cols, tmem_holding_buf, is_two_cta=False
                )
            tmem_ptr_read_threads = 32 * len((self.mma_warp_id, *self.epilog_warp_id))
            cute.arch.barrier(
                barrier_id=self.tmem_ptr_sync_bar_id,
                number_of_threads=tmem_ptr_read_threads,
            )
            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            tCtAcc_wide = cute.make_tensor(acc_tmem_ptr, tCtAcc_wide_layout)
            copy_atom_t2r = sm100_utils.get_tmem_load_op(
                self.cta_tile_shape_mnk,
                self.c_layout,
                self.c_dtype,
                self.acc_dtype,
                epi_tile_wide,
                False,
            )
            tAcc_epi = cute.flat_divide(
                tCtAcc_wide[((None, None), 0, 0, None)], epi_tile_wide
            )
            tiled_copy_t2r = tcgen05.make_tmem_copy(
                copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
            )
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_tAcc_base = thr_copy_t2r.partition_S(tAcc_epi)
            cC_epi = cute.flat_divide(
                cute.make_identity_tensor((self.cta_tile_shape_mnk[0], t2r_cols)),
                epi_tile_wide,
            )
            tTR_cC = thr_copy_t2r.partition_D(cC_epi)[(None, None, None, 0, 0)]
            tTR_rAcc = cute.make_rmem_tensor(tTR_cC.shape, self.acc_dtype)
            elem_cnt = cute.size(tTR_rAcc)
            elem_row_offset = []
            elem_token = []
            for element in cutlass.range_constexpr(elem_cnt):
                coord = tTR_cC[element]
                column = cutlass.Int32(coord[1])
                token = column % num_tokens
                elem_token.append(token)
                elem_row_offset.append(
                    token * hidden
                    + cutlass.Int32(coord[0])
                    + (column // num_tokens) * self.cta_tile_shape_mnk[0]
                )
            # Eight FP32 values per slab tile per thread; flat trace-time
            # indices keep the running sum in registers.
            rSlab = cute.make_rmem_tensor((t2r_slots * elem_cnt,), self.acc_dtype)
            for element in cutlass.range_constexpr(t2r_slots * elem_cnt):
                rSlab[element] = cutlass.Float32(0.0)

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_group_acc_stage
            )
            epilog_threads = 32 * len(self.epilog_warp_id)
            # Mirror of the MMA warp's pass windows: one wait/release per
            # window, the routed slots inside it drained in order.
            live_passes = self._live_passes(
                route_expert_ids, groups_per_split, group_splits, split_idx, gpp
            )
            for pass_idx in cutlass.range(0, live_passes, 1, unroll=1):
                acc_pipeline.consumer_wait(acc_consumer_state)
                t2r_pass_base = acc_consumer_state.index * (gpp * t2r_slots)
                for j in cutlass.range_constexpr(gpp):
                    group = (pass_idx * gpp + j) * cutlass.Int32(
                        group_splits
                    ) + split_idx
                    expert_id = route_expert_ids[group]
                    if expert_id >= 0:
                        group_alpha = alpha[expert_id].to(cutlass.Float32)
                        weights = []
                        for slot_idx in cutlass.range_constexpr(len(elem_token)):
                            weights.append(
                                group_alpha
                                * group_table[group * num_tokens + elem_token[slot_idx]]
                            )
                        t2r_slot_base = t2r_pass_base + j * t2r_slots
                        for slot in cutlass.range_constexpr(t2r_slots):
                            cute.copy(
                                tiled_copy_t2r,
                                tTR_tAcc_base[
                                    (None, None, None, 0, 0, t2r_slot_base + slot)
                                ],
                                tTR_rAcc,
                            )
                            for element in cutlass.range_constexpr(elem_cnt):
                                scale = weights[element]
                                if scale != cutlass.Float32(0.0):
                                    acc_idx = cutlass.const_expr(
                                        slot * elem_cnt + element
                                    )
                                    rSlab[acc_idx] = (
                                        rSlab[acc_idx] + tTR_rAcc[element] * scale
                                    )
                cute.arch.fence_view_async_tmem_load()
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

            # One final write per owner: single-split slabs need no clearing;
            # split owners atomically reduce into zeroed ``group_acc``.
            slab_row_base = slab_base_tile * self.cta_tile_shape_mnk[0]
            for slot in cutlass.range_constexpr(t2r_slots):
                row_base = slab_row_base + slot * t2r_tiles * self.cta_tile_shape_mnk[0]
                for element in cutlass.range_constexpr(elem_cnt):
                    acc_idx = cutlass.const_expr(slot * elem_cnt + element)
                    offset = elem_row_offset[element] + row_base
                    if cutlass.const_expr(group_splits == 1):
                        group_acc[offset] = rSlab[acc_idx]
                    else:
                        red_add_gpu_f32(group_acc.iterator + offset, rSlab[acc_idx])

            cute.arch.barrier(
                barrier_id=self.epilog_sync_bar_id, number_of_threads=epilog_threads
            )
            if warp_idx == self.epilog_warp_id[0]:
                with cute.arch.elect_one():
                    red_add_release_gpu_u32(group_done.iterator, cutlass.Uint32(1))
                cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)
            cute.arch.barrier(
                barrier_id=self.epilog_sync_bar_id, number_of_threads=epilog_threads
            )
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.dealloc_tmem(
                    acc_tmem_ptr, self.num_tmem_alloc_cols, is_two_cta=False
                )

        if cutlass.const_expr(release_pdl_dependents):
            griddepcontrol_launch_dependents()


__all__ = [
    "LongcatPersistentDownKStageKernel",
]

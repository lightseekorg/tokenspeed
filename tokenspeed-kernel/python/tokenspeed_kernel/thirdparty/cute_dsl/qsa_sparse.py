# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modifications Copyright (c) 2026 LightSeek Foundation
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

"""B200 direct-slot QSA attention derived from CUTLASS mixed-input FMHA."""

import math
from functools import partial
from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass._mlir import ir as _ir
from cutlass._mlir.dialects import llvm as _llvm
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import *
from cutlass.cutlass_dsl import T, dsl_user_op

# Kernel invariants
mma_modes = (0, 1, 2)
mma_dice = (None, None, None)  # (MMA, #MMA_M, #MMA_K)
cpy_dice = (None,) + mma_dice  # (CPY, #CPY_MMA, #CPY_M, #CPY_K)
warp_threads = 32
warpgroup_warps = 4
warpgroup_threads = 128

# Math helpers
log2_e = math.log2(math.e)  # change exponential base
fadd2 = cute.arch.add_packed_f32x2
fmul2 = cute.arch.mul_packed_f32x2
ffma2 = cute.arch.fma_packed_f32x2
exp2 = partial(cute.math.exp2, fastmath=True)
warp_fmax = partial(cute.arch.warp_redux_sync, kind="fmax", nan=True)
smem_fmax = partial(cute.arch.atomic_fmax, sem="relaxed", scope="cta")


@dsl_user_op
def set_block_rank(smem_ptr, peer_rank, *, loc=None, ip=None):
    dsmem_ptr = cute.arch.map_dsmem_ptr(smem_ptr, peer_rank, loc=loc, ip=ip)
    return cutlass.Int32(dsmem_ptr.toint(loc=loc, ip=ip))


@dsl_user_op
def st_shared_remote_f32(remote_addr, value, *, loc=None, ip=None):
    i32 = _ir.IntegerType.get_signless(32)
    _llvm.inline_asm(
        i32,
        [remote_addr.ir_value(loc=loc, ip=ip), value.ir_value(loc=loc, ip=ip)],
        "st.shared::cluster.f32 [$0], $1; mov.u32 $2, 0;",
        "r,f,=r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_shared_remote_f32(remote_addr, *, loc=None, ip=None):
    return cutlass.Float32(
        _llvm.inline_asm(
            T.f32(),
            [remote_addr.ir_value(loc=loc, ip=ip)],
            "ld.shared::cluster.f32 $0, [$1];",
            "=f,r",
            has_side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def copy_gmem_to_smem_u128(smem_ptr, gmem_ptr, *, loc=None, ip=None):
    """Issue one aligned 16-byte FP8 asynchronous global-to-shared copy."""
    _llvm.inline_asm(
        None,
        [
            smem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            gmem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
        ],
        "{\n"
        ".reg .u32 smem_addr;\n"
        "cvt.u32.u64 smem_addr, $0;\n"
        "cp.async.cg.shared.global [smem_addr], [$1], 16;\n"
        "}\n",
        "l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class MixedInputFusedMultiHeadAttentionDecode:
    def __init__(
        self,
        headdim,
        block_scaledim,  # headdim per scale factor; scale factor shape is (batches, heads_k, seqlen, headdim / block_scaledim)
        grouped_head_tile,  # GQA packing tile size, can be less than group size
        convert_warpgroups=1,  # Multiple warpgroups striding on convert stages
        kv_splits=4,
    ):
        self.headdim = headdim
        self.grouped_head_tile = grouped_head_tile
        self.block_scaledim = block_scaledim
        self.scaledim = headdim // block_scaledim
        self.convert_warpgroups = convert_warpgroups
        self.kv_splits = kv_splits

        assert headdim % block_scaledim == 0
        assert grouped_head_tile % 8 == 0 and 0 < grouped_head_tile <= 32

        warpgroup_id = 0

        self.softmax_warpgroup_id = warpgroup_id
        warpgroup_id += 1

        self.cvt_warpgroup_ids = tuple(
            range(warpgroup_id, warpgroup_id + convert_warpgroups)
        )
        warpgroup_id += convert_warpgroups

        # Why 2 MMA+TMA warps when not MMA bound?
        # Less register pressure per warp promotes concise SASS
        # hides MMA 'switching' latency that gets exposed with less concise SASS
        # We would have 2 leftover warps if we do warpgroup reg realloc
        # and less register pressure gives more realloc flexibility
        self.mma_kq_warp_id = warpgroup_id * warpgroup_warps + 0
        self.mma_vp_warp_id = warpgroup_id * warpgroup_warps + 1
        self.tma_kv_warp_id = warpgroup_id * warpgroup_warps + 2
        self.tma_qo_warp_id = warpgroup_id * warpgroup_warps + 3
        self.mma_tma_warpgroup_id = warpgroup_id
        warpgroup_id += 1

        self.threads_per_cta = warpgroup_id * warpgroup_threads

        self.use_reg_reconfig = grouped_head_tile > 16
        max_regs_per_wg_thread = 64 * 1024 // warpgroup_threads  # 64K regs per SM
        self.mma_tma_regs = 72
        self.cvt_regs = 112
        self.softmax_regs = (
            max_regs_per_wg_thread
            - self.mma_tma_regs
            - self.cvt_regs * convert_warpgroups
        )
        self.softmax_regs = max(128, min(256, self.softmax_regs))
        assert (
            self.mma_tma_regs + self.softmax_regs + self.cvt_regs * convert_warpgroups
        ) <= max_regs_per_wg_thread or not self.use_reg_reconfig

        self.bs_stages = 2
        self.sp_stages = 2
        self.o_stages = 1
        self.kv_ring_stages = 2

    @cute.jit
    def _issue_sparse_tile(
        self,
        cache_iter: cute.Pointer,
        selected_slots: cute.Tensor,
        batch_idx: cutlass.Int32,
        selected_tile: cutlass.Int32,
        dim_offset: cutlass.Int32,
        smem_base: cutlass.Int64,
        smem_stage: cutlass.Int32,
        warpgroup_tidx: cutlass.Int32,
        lane_idx: cutlass.Int32,
        kv_smem_dtype: Type[cutlass.Numeric],
    ):
        for iteration in cutlass.range_constexpr(8):
            vector = warpgroup_tidx + iteration * warpgroup_threads
            row = vector // 8
            col = (vector - row * 8) * 16
            selected_idx = selected_tile * 128 + row
            slot = cutlass.Int32(0)
            if lane_idx % 8 == 0 and selected_idx < selected_slots.shape[1]:
                candidate = selected_slots[batch_idx, selected_idx]
                if candidate > 0:
                    slot = candidate
            slot = cute.arch.shuffle_sync(slot, lane_idx - lane_idx % 8)
            source = cache_iter + slot * 256 + dim_offset + col
            logical_address = smem_base + row * 128 + col + smem_stage * 16384
            swizzled_address = logical_address ^ ((logical_address >> 3) & 0x70)
            destination = cute.make_ptr(
                kv_smem_dtype,
                swizzled_address,
                cute.AddressSpace.smem,
                assumed_align=16,
            )
            copy_gmem_to_smem_u128(destination, source)
        cute.arch.cp_async_commit_group()

    @cute.jit
    def __call__(
        self,
        q_tensor: cute.Tensor,
        k_tensor: cute.Tensor,
        v_tensor: cute.Tensor,
        selected_slots: cute.Tensor,
        output_tensor: cute.Tensor,
        scale_qs: cutlass.Float32,
        scale_o: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        problem_shape = self.problem_shape
        kv_splits = self.kv_splits
        q_iter = q_tensor.iterator
        k_iter = k_tensor.iterator
        v_iter = v_tensor.iterator
        o_iter = output_tensor.iterator
        ##############################
        # TiledMma creation
        ##############################
        mma_dtype = q_iter.dtype
        acc_dtype = cutlass.Float32

        # Block tile sets the granularity at which threadblocks consume work
        blk_tile_s = 128
        blk_tile_h = self.grouped_head_tile
        blk_tile_d = self.headdim
        blk_tile_shd = (blk_tile_s, blk_tile_h, blk_tile_d)

        # MMA tile sets the granularity at which TMAs + MMAs are issued
        mma_tile_m = 128
        mma_tile_n = self.grouped_head_tile
        mma_tile_k = 128 if self.headdim % 128 == 0 else 64
        mma_tile_mnk = (mma_tile_m, mma_tile_n, mma_tile_k)
        assert self.headdim % mma_tile_k == 0

        # GEMM1: (S_K, H_R, D, (H_K, B))
        tiled_mma_kq = sm100_utils.make_trivial_tiled_mma(
            mma_dtype,
            mma_dtype,
            OperandMajorMode.K,  # K
            OperandMajorMode.K,  # Q
            acc_dtype,
            tcgen05.CtaGroup.ONE,
            mma_tile_mnk[:2],
            tcgen05.OperandSource.TMEM,  # converted K in tmem
        )

        # GEMM2: (D, H_R, S_K, (H_K, B))
        tiled_mma_vp = sm100_utils.make_trivial_tiled_mma(  #
            mma_dtype,
            mma_dtype,
            OperandMajorMode.K,  # V
            OperandMajorMode.MN,  # P
            acc_dtype,
            tcgen05.CtaGroup.ONE,
            mma_tile_mnk[:2],
            tcgen05.OperandSource.TMEM,  # converted V in tmem
        )

        # Calculate Q stages
        self.q_stages = blk_tile_d // mma_tile_k

        # Perf heuristics
        cap_kv_stages = k_iter.dtype.width >= 8
        max_cvt_stages = 4 if self.grouped_head_tile == 32 and mma_tile_k == 128 else 8
        max_kv_stages = 8 if mma_tile_k == 128 else 14

        # Calculate KV tmem stages
        tmem_alloc_cols = mma_tile_n * self.sp_stages
        tmem_alloc_cols += mma_tile_n * self.o_stages * (blk_tile_d // mma_tile_m)
        tmem_capacity = 512
        cvt_stage_cols = mma_tile_k * mma_dtype.width // 32
        self.cvt_stages = (tmem_capacity - tmem_alloc_cols) // cvt_stage_cols
        self.cvt_stages = (
            min(self.cvt_stages, max_cvt_stages) if cap_kv_stages else self.cvt_stages
        )

        tmem_alloc_cols += self.cvt_stages * cvt_stage_cols
        self.tmem_alloc_cols = 2 ** math.ceil(
            math.log2(tmem_alloc_cols)
        )  # Tmem alloc must be PO2

        # Calculate KV smem stages
        self.mbarrier_reserved_bytes = 768
        smem_alloc_bits = self.mbarrier_reserved_bytes * 8
        smem_alloc_bits += mma_tile_n * acc_dtype.width  # colmax
        smem_alloc_bits += (
            self.scaledim * blk_tile_s * self.bs_stages * mma_dtype.width
        )  # block scale
        smem_alloc_bits += mma_tile_n * warpgroup_warps * acc_dtype.width  # colsum
        smem_alloc_bits += (
            mma_tile_n * mma_tile_k * self.q_stages * mma_dtype.width
        )  # Q
        smem_alloc_bits += (
            mma_tile_m * mma_tile_n * self.sp_stages * mma_dtype.width
        )  # P

        smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        kv_smem_dtype = cutlass.Int8 if k_iter.dtype.width < 8 else k_iter.dtype
        self.kv_stages = (smem_capacity * 8 - smem_alloc_bits) // (
            mma_tile_m * mma_tile_k * kv_smem_dtype.width
        )
        self.kv_stages = (
            min(self.kv_stages, max_kv_stages) if cap_kv_stages else self.kv_stages
        )
        assert self.kv_stages >= self.kv_ring_stages * self.convert_warpgroups

        ##############################
        # TMA creation
        ##############################
        b, h_q, h_k, s_k, d = problem_shape
        h_r = h_q // h_k

        q = cute.make_tensor(
            q_iter,
            cute.make_ordered_layout(shape=(h_r, d, (h_k, b)), order=(1, 0, (2, 3))),
        )
        assert k_iter.dtype is not q_iter.dtype
        assert v_iter.dtype is k_iter.dtype

        # (MMA, MMA_M/N, MMA_K, Stages)
        smem_layout_q = sm100_utils.make_smem_layout_b(
            tiled_mma_kq, mma_tile_mnk, q_iter.dtype, self.q_stages
        )
        smem_layout_k = sm100_utils.make_smem_layout_a(
            tiled_mma_kq, mma_tile_mnk, kv_smem_dtype, self.kv_stages
        )
        smem_layout_v = sm100_utils.make_smem_layout_a(
            tiled_mma_vp, mma_tile_mnk, kv_smem_dtype, self.kv_stages, is_k_major=False
        )  # V is always headdim-major (GEMM2 M-major) in gmem+smem

        smem_layout_atom_o = tcgen05.make_smem_layout_atom(
            tcgen05.mma.SmemLayoutAtomKind.MN_SW128, acc_dtype
        )
        smem_layout_o = cute.tile_to_shape(
            smem_layout_atom_o, (blk_tile_d, blk_tile_h), order=(1, 0)
        )
        smem_layout_o = cute.flat_divide(smem_layout_o, (mma_tile_m, mma_tile_n))

        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()

        tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            q,
            cute.select(smem_layout_q, mma_modes),
            mma_tile_mnk,
            tiled_mma_kq,
        )

        ##############################
        # Decode Kernel launch
        ##############################
        scale_qs_log2_e = scale_qs * log2_e

        n_tiles = cute.ceil_div(h_r, blk_tile_h)
        l_tiles = b * h_k
        grid = (kv_splits, n_tiles, l_tiles)
        o = cute.make_tensor(o_iter, cute.make_layout((d, h_q, b)))

        self.decode(
            blk_tile_shd,
            mma_tile_mnk,
            tiled_mma_kq,
            tiled_mma_vp,
            q_iter.dtype,
            smem_layout_q,
            tma_atom_q,
            tma_tensor_q,
            k_iter.dtype,
            smem_layout_k,
            k_iter,
            v_iter.dtype,
            smem_layout_v,
            v_iter,
            selected_slots,
            acc_dtype,
            smem_layout_o,
            o,
            scale_qs,
            scale_qs_log2_e,
            scale_o,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[kv_splits, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

        ##############################
        # Reduction Kernel launch
        ##############################
        # The selected-token splits reduce through DSM in the decode launch.

    @cute.kernel
    def decode(
        self,
        # MMA
        blk_tile_shd: cute.Tile,
        mma_tile_mnk: cute.Tile,
        tiled_mma_kq: cute.TiledMma,
        tiled_mma_vp: cute.TiledMma,
        # Q
        q_dtype: Type[cutlass.Numeric],
        smem_layout_q: cute.ComposedLayout,
        tma_atom_q: cute.CopyAtom,
        mQ: cute.Tensor,
        # K
        k_dtype: Type[cutlass.Numeric],
        smem_layout_k: cute.ComposedLayout,
        k_iter: cute.Pointer,
        # V
        v_dtype: Type[cutlass.Numeric],
        smem_layout_v: cute.ComposedLayout,
        v_iter: cute.Pointer,
        selected_slots: cute.Tensor,
        # O
        o_dtype: Type[cutlass.Numeric],
        smem_layout_o: cute.ComposedLayout,
        mOut: cute.Tensor,
        scale_qs: cutlass.Float32,
        scale_qs_log2_e: cutlass.Float32,
        scale_o: cutlass.Float32,
    ):
        # Read special registers
        kv_splits, tiles_hr, tiles_hb = cute.arch.grid_dim()
        kv_split_idx, coord_hr, coord_hb = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(tidx // warp_threads)
        warpgroup_idx = cute.arch.make_warp_uniform(tidx // warpgroup_threads)
        warpgroup_tidx = tidx % warpgroup_threads
        warpgroup_widx = warp_idx % warpgroup_warps
        init_warp = 0

        # No multicast
        mcast_coord = 0
        mcast_layout = cute.make_layout((1, 1, 1, 1))  # vmnk

        # Alias types
        mma_dtype = q_dtype
        acc_dtype = o_dtype
        kv_smem_dtype = cutlass.Int8 if k_dtype.width < 8 else k_dtype

        # Shapes for MMA tile indexing (Read TMA partition for example)
        blk_tile_s, blk_tile_h, blk_tile_d = blk_tile_shd
        mma_tile_m, mma_tile_n, mma_tile_k = mma_tile_mnk
        tiles_dm, tiles_sk = cute.ceil_div(
            (blk_tile_d, blk_tile_s), (mma_tile_m, mma_tile_k)
        )
        tiles_dk, tiles_sm = cute.ceil_div(
            (blk_tile_d, blk_tile_s), (mma_tile_k, mma_tile_m)
        )
        tiles_s = cute.ceil_div(selected_slots.shape[1], blk_tile_s)
        iters_s = cute.ceil_div(tiles_s - kv_split_idx, kv_splits)
        prefetch_iters = self.sp_stages - 1
        if iters_s < prefetch_iters:
            prefetch_iters = iters_s
        assert tiles_sm == 1

        # Runtime checks
        exit_early = kv_split_idx >= tiles_s
        lane_store_max = mma_tile_n == warp_threads or lane_idx < mma_tile_n

        # Smem alloc helper
        svector_align = 16
        stensor_align = 128
        smem = utils.SmemAllocator()

        ##############################
        # Prefetch TMA descriptor
        ##############################
        if warp_idx == init_warp and not exit_early:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
        init_warp += 1

        ##############################
        # Tmem Allocation
        ##############################
        tmem_ptr_smem_ptr = smem.allocate_array(cutlass.Int32)
        if warp_idx == init_warp and not exit_early:
            cute.arch.alloc_tmem(self.tmem_alloc_cols, tmem_ptr_smem_ptr)
        init_warp += 1

        ##############################
        # Pipeline Allocation + Init
        ##############################
        # Allocate Mbarriers
        q_pipeline_ptr = smem.allocate_array(cutlass.Int64, self.q_stages * 2)
        cvt_pipeline_ptr = smem.allocate_array(cutlass.Int64, self.cvt_stages * 2)
        s_pipeline_ptr = smem.allocate_array(cutlass.Int64, self.sp_stages * 2)
        p_pipeline_ptr = smem.allocate_array(cutlass.Int64, self.sp_stages * 2)
        o_pipeline_ptr = smem.allocate_array(cutlass.Int64, self.o_stages * 2)

        # Declare named barriers
        softmax_nbar = pipeline.NamedBarrier(
            barrier_id=1, num_threads=warpgroup_threads
        )
        mma_kq_nbar = pipeline.NamedBarrier(barrier_id=2, num_threads=64)
        mma_vp_nbar = pipeline.NamedBarrier(barrier_id=3, num_threads=64)

        # Alias thread cooperatives
        elect_one_cooperative = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        warpgroup_cooperative = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, warpgroup_threads
        )
        mma_group = elect_one_cooperative
        tma_group = elect_one_cooperative
        cvt_group = warpgroup_cooperative
        cvt_groups = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, warpgroup_threads * self.convert_warpgroups
        )
        softmax_group = warpgroup_cooperative

        # Initialize pipelines
        q_producer, q_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.q_stages,
            producer_group=tma_group,
            consumer_group=softmax_group,  # Reuse Q consumer mbarriers to sync O store
            tx_count=cute.size_in_bytes(q_dtype, cute.select(smem_layout_q, mma_modes)),
            barrier_storage=q_pipeline_ptr,
            tidx=mcast_coord,
            cta_layout_vmnk=mcast_layout,
            defer_sync=True,
        ).make_participants()
        cvt_producer, cvt_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.cvt_stages,
            producer_group=cvt_group,
            consumer_group=mma_group,
            barrier_storage=cvt_pipeline_ptr,
            defer_sync=True,
        ).make_participants()
        s_producer, s_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.sp_stages,
            producer_group=mma_group,
            consumer_group=softmax_group,
            barrier_storage=s_pipeline_ptr,
            defer_sync=True,
        ).make_participants()
        p_producer, p_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.sp_stages,
            producer_group=softmax_group,
            consumer_group=mma_group,
            barrier_storage=p_pipeline_ptr,
            defer_sync=True,
        ).make_participants()
        o_producer, o_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.o_stages,
            producer_group=mma_group,
            consumer_group=softmax_group,
            barrier_storage=o_pipeline_ptr,
            defer_sync=True,
        ).make_participants()

        # Ensure visibility of local mbarrier inits and tmem alloc
        cute.arch.sync_threads()

        ##############################
        # MMA Partition + Allocate
        ##############################
        # Threadblock slice
        thrblk_mma_kq = tiled_mma_kq.get_slice(0)
        thrblk_mma_vp = tiled_mma_vp.get_slice(0)

        # M - colmax
        sM_layout = cute.make_layout(shape=(mma_tile_m, mma_tile_n), stride=(0, 1))
        sM = smem.allocate_tensor(acc_dtype, sM_layout, svector_align)
        tCsM = thrblk_mma_kq.partition_C(sM)

        # L - colsum
        sL_layout = cute.make_layout(
            shape=(mma_tile_m, mma_tile_n, warpgroup_warps), stride=(0, 1, mma_tile_n)
        )
        sL = smem.allocate_tensor(acc_dtype, sL_layout, svector_align)
        tCsL = thrblk_mma_kq.partition_C(sL)

        # Q
        tBsQ = smem.allocate_tensor(
            q_dtype, smem_layout_q.outer, stensor_align, smem_layout_q.inner
        )  # (MMA, #MMA_N, #MMA_K, q_stages)

        # K
        tAsK = smem.allocate_tensor(
            kv_smem_dtype, smem_layout_k.outer, stensor_align, smem_layout_k.inner
        )  # (MMA, #MMA_M, #MMA_K, kv_stages)
        tAtK_cvt_shape = tiled_mma_kq.partition_shape_A(
            (mma_tile_m, mma_tile_k, self.cvt_stages)
        )  # (MMA, #MMA_M, #MMA_K, cvt_stages)
        tAtK_cvt = thrblk_mma_kq.make_fragment_A(tAtK_cvt_shape)

        # V
        tAsV_iterator = cute.recast_ptr(
            tAsK.iterator, smem_layout_v.inner, dtype=kv_smem_dtype
        )  # KV share input buffers
        tAsV = cute.make_tensor(
            tAsV_iterator, smem_layout_v.outer
        )  # (MMA, #MMA_M, #MMA_K, kv_stages)
        sKV_vector_ptr = cute.recast_ptr(tAsK.iterator, dtype=kv_smem_dtype)
        sKV_vector_address = sKV_vector_ptr.toint()
        tAtV_cvt_shape = tiled_mma_vp.partition_shape_A(
            (mma_tile_m, mma_tile_k, self.cvt_stages)
        )  # (MMA, #MMA_M, #MMA_K, cvt_stages)
        tAtV_cvt = thrblk_mma_vp.make_fragment_A(tAtV_cvt_shape)

        # S
        tCtS_shape = tiled_mma_kq.partition_shape_C(
            (mma_tile_m, mma_tile_n, self.sp_stages)
        )
        tCtS = thrblk_mma_kq.make_fragment_C(
            tCtS_shape
        )  # (MMA_MN, #MMA_M=1, #MMA_N=1, sp_stages)

        # P - Treat MN C tile of BMM0 as NM B tile of BMM1
        # (MMA_NK, #MMA_N, #MMA_K=MMA_TILE_M/MMA_K, sp_stages)
        mma_tile_nm = (None, mma_tile_n, mma_tile_m)
        tBsP_nm_layout = sm100_utils.make_smem_layout_b(
            tiled_mma_vp, mma_tile_nm, mma_dtype, self.sp_stages
        )
        tBsP_nm = smem.allocate_tensor(
            mma_dtype, tBsP_nm_layout.outer, stensor_align, tBsP_nm_layout.inner
        )

        # Tile for NK B tile iteration
        # (MMA_NK, #MMA_N, #MMA_K=MMA_TILE_K/MMA_K, #TILES_SK=MMA_TILE_M/MMA_TILE_K, sp_stages)
        tBsP_nk_tile = thrblk_mma_vp.partition_shape_B((mma_tile_n, mma_tile_k))
        tBsP_nk = cute.local_tile(tBsP_nm, tBsP_nk_tile, (0, 0, None, None))

        # Reshape NM B tile of BMM1 to become MN C tile of BMM0
        # (MMA_NK, #MMA_N, #MMA_K=MMA_TILE_M/MMA_K, sp_stages) ->
        # (MMA_MN, #MMA_M, #MMA_N, sp_stages)
        tCsP_tile = cute.make_ordered_layout(tCtS_shape, order=((2, 0), 3, 1, 4))
        tCsP = cute.composition(tBsP_nm, tCsP_tile)

        # O
        sO_iterator = cute.recast_ptr(
            tBsQ.iterator, smem_layout_o.inner, dtype=o_dtype
        )  # Reuse QKV smem for O TMA store
        sO_mma = cute.make_tensor(
            sO_iterator, smem_layout_o.outer
        )  # (MMA_TILE_M, MMA_TILE_N, #TILE_DM, #TILE_HN)
        tCsO = thrblk_mma_vp.partition_C(
            sO_mma
        )  # (MMA, #MMA_M, #MMA_N, #TILE_DM, #TILE_HN)
        tCtO = thrblk_mma_vp.make_fragment_C(tCsO.shape)

        # Tmem tensor allocation
        tmem_ptr = cute.arch.retrieve_tmem_ptr(cutlass.Int32, 16, tmem_ptr_smem_ptr)
        tmem_offset = 0

        tAtK_cvt = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + tmem_offset, dtype=mma_dtype), tAtK_cvt.layout
        )
        tAtV_cvt = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + tmem_offset, dtype=mma_dtype), tAtV_cvt.layout
        )
        tmem_offset += tcgen05.find_tmem_tensor_col_offset(tAtK_cvt)

        tCtS = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + tmem_offset, dtype=acc_dtype), tCtS.layout
        )
        tmem_offset += tcgen05.find_tmem_tensor_col_offset(tCtS)

        tCtO = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + tmem_offset, dtype=acc_dtype), tCtO.layout
        )
        tmem_offset += tcgen05.find_tmem_tensor_col_offset(tCtO)

        assert tmem_offset <= self.tmem_alloc_cols

        ##############################
        # Exit early
        ##############################
        if exit_early:
            noop = None  # early return not supported # noqa: F841

        ##############################
        # TMA Q Dispatch
        ##############################
        elif warp_idx == self.tma_qo_warp_id:
            # Free registers
            if cutlass.const_expr(self.use_reg_reconfig):
                cute.arch.setmaxregister_decrease(self.mma_tma_regs)

            gQ = cute.local_tile(
                mQ,
                tiler=(blk_tile_h, blk_tile_d),
                coord=(coord_hr, 0, coord_hb),
            )
            gQ_mma = cute.flat_divide(gQ, (mma_tile_n, mma_tile_k))
            tBgQ = thrblk_mma_kq.partition_B(gQ_mma)
            tGSsQ, tGSgQ = cute.nvgpu.cpasync.tma_partition(
                tma_atom_q,
                mcast_coord,
                mcast_layout,
                smem_tensor=cute.group_modes(tBsQ, 0, 3),
                gmem_tensor=cute.group_modes(tBgQ, 0, 3),
            )
            for dk in cutlass.range_constexpr(tiles_dk):
                q_handle = q_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_q,
                    tGSgQ[None, 0, dk],
                    tGSsQ[None, dk],
                    tma_bar_ptr=q_handle.barrier,
                )

        ##############################
        # Convert Dispatch
        ##############################
        elif warpgroup_idx in self.cvt_warpgroup_ids:
            # Free registers
            if cutlass.const_expr(self.use_reg_reconfig):
                cute.arch.setmaxregister_decrease(self.cvt_regs)

            # Intermediate convert type
            cvt_type = cutlass.Float32
            if cutlass.const_expr(
                mma_dtype is cutlass.BFloat16
                and k_dtype in (cutlass.Int4, cutlass.Int8)
            ):
                cvt_type = mma_dtype

            # Initialize for multiple warpgroups if necessary
            convert_phase = 0
            if cutlass.const_expr(self.convert_warpgroups > 1):
                assert tiles_dk % self.convert_warpgroups == 0
                assert (tiles_dm * tiles_sk) % self.convert_warpgroups == 0
                convert_phase = warpgroup_idx % self.convert_warpgroups
                for _ in cutlass.range(convert_phase):
                    cvt_producer.advance()
            cvt_load_nbar = pipeline.NamedBarrier(
                barrier_id=4 + convert_phase, num_threads=warpgroup_threads
            )

            # Construct tiled copy and partition K
            mma_k_bits = mma_tile_k * mma_dtype.width
            tmem_store_atom_k = cute.make_copy_atom(
                tcgen05.St16x256bOp(tcgen05.Repetition(mma_k_bits // 256)),
                mma_dtype,
            )
            smem_load_atom_k = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x16x8bOp(
                    num_matrices=4,
                    unpack_bits=(k_dtype.width if k_dtype.width < 8 else None),
                ),
                kv_smem_dtype,
            )

            tmem_store_k = tcgen05.make_tmem_copy(
                tmem_store_atom_k, tAtK_cvt[mma_dice + (0,)]
            )
            thr_store_k = tmem_store_k.get_slice(warpgroup_tidx)
            tKrK_cvt_shape = thr_store_k.partition_S(tAtK_cvt).shape[:-1]
            tKtK_cvt = thr_store_k.partition_D(tAtK_cvt)

            smem_load_k = cute.make_tiled_copy_S(smem_load_atom_k, tmem_store_k)
            thr_load_k = smem_load_k.get_slice(warpgroup_tidx)
            tKsK = thr_load_k.partition_S(tAsK)
            tKrK_shape = thr_load_k.partition_D(tAsK).shape[:-1]

            # Construct tiled copy and partition V
            tmem_store_atom_v = cute.make_copy_atom(
                tcgen05.St16x256bOp(tcgen05.Repetition(mma_k_bits // 256)), mma_dtype
            )
            smem_load_op_v = cute.nvgpu.warp.LdMatrix16x16x8bOp(
                transpose=True,
                num_matrices=2,
                unpack_bits=(v_dtype.width if v_dtype.width < 8 else None),
            )
            smem_load_atom_v = cute.make_copy_atom(smem_load_op_v, kv_smem_dtype)

            tmem_store_v = tcgen05.make_tmem_copy(
                tmem_store_atom_v, tAtV_cvt[mma_dice + (0,)]
            )
            thr_store_v = tmem_store_v.get_slice(warpgroup_tidx)
            tVrV_cvt_shape = thr_store_v.partition_S(tAtV_cvt).shape[:-1]
            tVtV_cvt = thr_store_v.partition_D(tAtV_cvt)

            smem_load_v = cute.make_tiled_copy_S(smem_load_atom_v, tmem_store_v)
            thr_load_v = smem_load_v.get_slice(warpgroup_tidx)
            tVsV = thr_load_v.partition_S(tAsV)
            tVrV_shape = thr_load_v.partition_D(tAsV).shape[:-1]

            # Prime the unified K-or-V ring with K0. Each conversion
            # warpgroup owns one 128-D half of every logical ring item.
            self._issue_sparse_tile(
                k_iter,
                selected_slots,
                coord_hb,
                kv_split_idx,
                convert_phase * mma_tile_k,
                sKV_vector_address,
                convert_phase,
                warpgroup_tidx,
                lane_idx,
                kv_smem_dtype,
            )

            #
            # Sequence loop
            #
            for s in cutlass.range(prefetch_iters + iters_s):
                if s < iters_s:
                    # Convert and scale K
                    for _ in cutlass.range(
                        tiles_dk // self.convert_warpgroups, unroll=2
                    ):
                        # Issue the next logical K-or-V item before
                        # consuming K_s. One pending cp.async group overlaps
                        # its latency with conversion and UMMA.
                        if s == 0:
                            self._issue_sparse_tile(
                                k_iter,
                                selected_slots,
                                coord_hb,
                                kv_split_idx + kv_splits,
                                convert_phase * mma_tile_k,
                                sKV_vector_address,
                                self.convert_warpgroups + convert_phase,
                                warpgroup_tidx,
                                lane_idx,
                                kv_smem_dtype,
                            )
                        else:
                            self._issue_sparse_tile(
                                v_iter,
                                selected_slots,
                                coord_hb,
                                kv_split_idx + (s - 1) * kv_splits,
                                convert_phase * mma_tile_m,
                                sKV_vector_address,
                                convert_phase,
                                warpgroup_tidx,
                                lane_idx,
                                kv_smem_dtype,
                            )
                        cute.arch.cp_async_wait_group(1)
                        k_smem_stage = convert_phase
                        if s > 0:
                            k_smem_stage = self.convert_warpgroups + convert_phase
                        cvt_load_nbar.arrive_and_wait()

                        tKrK = cute.make_rmem_tensor(tKrK_shape, kv_smem_dtype)
                        tKrK_cvt = cute.make_rmem_tensor(tKrK_cvt_shape, mma_dtype)

                        cute.copy(
                            thr_load_k,
                            tKsK[cpy_dice + (k_smem_stage,)],
                            tKrK,
                        )
                        cute.arch.fence_view_async_shared()

                        # Sign extend unpacked int4 to int8
                        if cutlass.const_expr(k_dtype is cutlass.Int4):
                            tKrK_unpacked_i4_vec = tKrK.load().maybe_downcast()
                            tKrK_i8_vec = cute.arch.sext_unpacked_i4_i8_intrinsic(
                                tKrK_unpacked_i4_vec, cute.size(tKrK_shape)
                            )
                            tKrK.store(
                                cute.TensorSSA(tKrK_i8_vec, tKrK_shape, cutlass.Int8)
                            )

                        tKrK_ssa = tKrK.load().to(cvt_type).to(mma_dtype)
                        tKrK_cvt.store(tKrK_ssa.reshape(tKrK_cvt_shape))

                        cvt_handle = cvt_producer.acquire_and_advance()
                        cute.copy(
                            thr_store_k,
                            tKrK_cvt,
                            tKtK_cvt[cpy_dice + (cvt_handle.index,)],
                        )
                        cute.arch.fence_view_async_tmem_store()
                        cvt_handle.commit()

                        # Advance again for multiple warpgroups
                        for _ in cutlass.range_constexpr(self.convert_warpgroups - 1):
                            cvt_producer.advance()

                if s >= prefetch_iters:
                    # Convert and scale V
                    for _ in cutlass.range(
                        tiles_dm * tiles_sk // self.convert_warpgroups, unroll=2
                    ):
                        # V_(s-1) is resident/in flight. Refill the other
                        # ring slot with K_(s+1), or the final V item.
                        if s < iters_s:
                            if s + 1 < iters_s:
                                self._issue_sparse_tile(
                                    k_iter,
                                    selected_slots,
                                    coord_hb,
                                    kv_split_idx + (s + 1) * kv_splits,
                                    convert_phase * mma_tile_k,
                                    sKV_vector_address,
                                    self.convert_warpgroups + convert_phase,
                                    warpgroup_tidx,
                                    lane_idx,
                                    kv_smem_dtype,
                                )
                            else:
                                self._issue_sparse_tile(
                                    v_iter,
                                    selected_slots,
                                    coord_hb,
                                    kv_split_idx + s * kv_splits,
                                    convert_phase * mma_tile_m,
                                    sKV_vector_address,
                                    self.convert_warpgroups + convert_phase,
                                    warpgroup_tidx,
                                    lane_idx,
                                    kv_smem_dtype,
                                )
                            cute.arch.cp_async_wait_group(1)
                        else:
                            cute.arch.cp_async_wait_group(0)
                        v_smem_stage = convert_phase
                        if s == iters_s:
                            v_smem_stage = self.convert_warpgroups + convert_phase
                        cvt_load_nbar.arrive_and_wait()

                        tVrV = cute.make_rmem_tensor(tVrV_shape, kv_smem_dtype)
                        tVrV_cvt = cute.make_rmem_tensor(tVrV_cvt_shape, mma_dtype)

                        cute.copy(
                            thr_load_v,
                            tVsV[cpy_dice + (v_smem_stage,)],
                            tVrV,
                        )
                        cute.arch.fence_view_async_shared()

                        # Sign extend unpacked int4 to int8
                        if cutlass.const_expr(v_dtype is cutlass.Int4):
                            tVrV_unpacked_i4_vec = tVrV.load().maybe_downcast()
                            tVrV_i8_vec = cute.arch.sext_unpacked_i4_i8_intrinsic(
                                tVrV_unpacked_i4_vec, cute.size(tVrV_shape)
                            )
                            tVrV.store(
                                cute.TensorSSA(tVrV_i8_vec, tVrV_shape, cutlass.Int8)
                            )

                        tVrV_ssa = tVrV.load().to(cvt_type).to(mma_dtype)
                        tVrV_cvt.store(tVrV_ssa.reshape(tVrV_cvt_shape))

                        cvt_handle = cvt_producer.acquire_and_advance()
                        cute.copy(
                            thr_store_v,
                            tVrV_cvt,
                            tVtV_cvt[cpy_dice + (cvt_handle.index,)],
                        )
                        cute.arch.fence_view_async_tmem_store()
                        cvt_handle.commit()

                        # Advance again for multiple warpgroups
                        for _ in cutlass.range_constexpr(self.convert_warpgroups - 1):
                            cvt_producer.advance()

        ##############################
        # MMA KQ Dispatch
        ##############################
        elif warp_idx == self.mma_kq_warp_id:
            # Free registers
            if cutlass.const_expr(self.use_reg_reconfig):
                cute.arch.setmaxregister_decrease(self.mma_tma_regs)

            # Setup mma descriptors
            tBsQ_desc = thrblk_mma_kq.make_fragment_B(tBsQ)

            # Wait for Q
            for dk in cutlass.range_constexpr(tiles_dk):
                q_consumer.wait_and_advance()

            # Sequence loop
            s_token = True  # Producer always acquires first
            for s in cutlass.range(iters_s):
                # BMM1
                k_token = cvt_consumer.try_wait()
                tiled_mma_kq.set(tcgen05.Field.ACCUMULATE, False)
                s_handle = s_producer.acquire_and_advance(s_token)
                for dk in cutlass.range_constexpr(tiles_dk):
                    is_last_iter = dk == tiles_dk - 1
                    k_handle = cvt_consumer.wait_and_advance(k_token)
                    # Signal BMM2 to start
                    if is_last_iter:
                        mma_kq_nbar.arrive()
                    for mma_k in cutlass.range_constexpr(tAtK_cvt.shape[2]):
                        cute.gemm(
                            tiled_mma_kq,
                            tCtS[mma_dice + (s_handle.index,)],
                            tAtK_cvt[None, None, mma_k, k_handle.index],
                            tBsQ_desc[None, None, mma_k, dk],
                            tCtS[mma_dice + (s_handle.index,)],
                        )
                        if dk == 0 and mma_k == 0:
                            tiled_mma_kq.set(tcgen05.Field.ACCUMULATE, True)
                    k_handle.release()
                    if not is_last_iter:
                        k_token = cvt_consumer.try_wait()
                s_handle.commit()

                # Advance and wait for BMM 2
                if s > 0:
                    for _ in cutlass.range_constexpr(tiles_dm * tiles_sk):
                        cvt_consumer.advance()
                    mma_vp_nbar.arrive_and_wait()
                    s_token = s_producer.try_acquire()

        ##############################
        # MMA VP Dispatch
        ##############################
        elif warp_idx == self.mma_vp_warp_id:
            # Free registers
            if cutlass.const_expr(self.use_reg_reconfig):
                cute.arch.setmaxregister_decrease(self.mma_tma_regs)

            # Setup mma descriptors
            tiled_mma_vp.set(tcgen05.Field.ACCUMULATE, True)
            tBsP_desc = thrblk_mma_vp.make_fragment_B(tBsP_nk)

            # Advance and wait for BMM1
            for _ in cutlass.range_constexpr(tiles_dk):
                cvt_consumer.advance()
            mma_kq_nbar.arrive_and_wait()

            # Sequence loop
            p_token = False
            o_token = True  # Producer always acquires first
            for s in cutlass.range(iters_s):
                # Advance and wait for BMM1
                if s < iters_s - 1:
                    for _ in cutlass.range_constexpr(tiles_dk):
                        cvt_consumer.advance()
                    mma_kq_nbar.arrive_and_wait()
                    p_token = p_consumer.try_wait()

                # BMM2
                v_token = cvt_consumer.try_wait()
                p_handle = p_consumer.wait_and_advance(p_token)
                o_handle = o_producer.acquire_and_advance(o_token)
                for sk in cutlass.range_constexpr(tiles_sk):
                    for dm in cutlass.range_constexpr(tiles_dm):
                        is_last_iter = sk == tiles_sk - 1 and dm == tiles_dm - 1
                        v_handle = cvt_consumer.wait_and_advance(v_token)
                        # Signal BMM1 to start
                        if is_last_iter:
                            mma_vp_nbar.arrive()
                        for mma_k in cutlass.range_constexpr(tAtV_cvt.shape[2]):
                            cute.gemm(
                                tiled_mma_vp,
                                tCtO[mma_dice + (dm, 0)],
                                tAtV_cvt[None, None, mma_k, v_handle.index],
                                tBsP_desc[None, None, mma_k, sk, p_handle.index],
                                tCtO[mma_dice + (dm, 0)],
                            )
                        v_handle.release()
                        if not is_last_iter:
                            v_token = cvt_consumer.try_wait()
                p_handle.release()
                o_handle.commit()
                o_token = o_producer.try_acquire()

            # Wait for signal to dealloc tmem, then dealloc
            o_producer.tail()
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(tmem_ptr, self.tmem_alloc_cols)

        ##############################
        # Softmax + Correction Dispatch
        ##############################
        elif warpgroup_idx == self.softmax_warpgroup_id:
            # Alloc registers
            if cutlass.const_expr(self.use_reg_reconfig):
                cute.arch.setmaxregister_increase(self.softmax_regs)

            # Construct tiled copies
            tmem_op_width = 32
            tmem_op_repeat = tcgen05.Repetition(
                mma_tile_n * acc_dtype.width // tmem_op_width
            )
            tmem_load_atom_s = cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tmem_op_repeat), acc_dtype
            )
            tmem_load_s = tcgen05.make_tmem_copy(
                tmem_load_atom_s, tCtS[mma_dice + (0,)]
            )
            thr_load_s = tmem_load_s.get_slice(warpgroup_tidx)

            tmem_store_atom_o = cute.make_copy_atom(
                tcgen05.St32x32bOp(tmem_op_repeat), o_dtype
            )
            tmem_store_o = tcgen05.make_tmem_copy(
                tmem_store_atom_o, tCtO[mma_dice + (0, 0)]
            )
            thr_store_o = tmem_store_o.get_slice(warpgroup_tidx)

            # Partition S and P
            tStS = thr_load_s.partition_S(
                tCtS
            )  # (CPY, #CPY_MMA, #CPY_M, #CPY_N, stages_sp)
            cS = cute.make_identity_tensor((mma_tile_m, mma_tile_n))
            tOcS = thrblk_mma_kq.partition_C(cS)
            tScS = thr_load_s.partition_D(tOcS)
            tSsP = thr_load_s.partition_D(
                tCsP
            )  # (CPY, #CPY_MMA, #CPY_M, #CPY_N, stages_sp)

            # Partition O
            tStO = thr_load_s.partition_S(
                tCtO
            )  # (CPY, #CPY_MMA, #CPY_M, #CPY_N, #TILE_DM, #TILE_HN)
            tSsO = thr_load_s.partition_D(
                tCsO
            )  # (CPY, #CPY_MMA, #CPY_M, #CPY_N, #TILE_DM, #TILE_HN)
            tSrO = cute.make_rmem_tensor_like(tSsO)
            cO = cute.make_identity_tensor((blk_tile_d, blk_tile_h))
            tOcO = thrblk_mma_vp.partition_C(cO)
            tScO = thr_load_s.partition_D(tOcO)

            # Partition colmax and initialize in RF
            tSsM = thr_load_s.partition_D(tCsM)  # (CPY, #CPY_MMA, #CPY_M, #CPY_N)
            tSrM_prev = cute.make_rmem_tensor_like(tSsM)
            tSrM_prev.fill(-cutlass.Float32.inf)

            # Partition colsum and initialize in RF
            # Each thread maintains a local colsum in RF, smem reduction happens after loop
            tSsL = thr_load_s.partition_D(
                tCsL
            )  # (CPY, #CPY_MMA, #CPY_M, #CPY_N, WARPS)
            tSrL = cute.make_rmem_tensor_like(tSsL[cpy_dice + (0,)])
            tSrL.fill(cutlass.Float32(0))

            assert warp_threads >= cute.size(tSsM)

            # Initialize O
            tSrO.fill(cutlass.Float32(0))
            cute.copy(thr_store_o, tSrO, tStO)

            # Initialize colsum and colmax in smem and wait
            if warpgroup_widx == 0 and lane_store_max:
                tSsM[lane_idx] = -cutlass.Float32.inf
            if warpgroup_widx == 1 and lane_store_max:
                tSsL[lane_idx] = cutlass.Float32(0)
            softmax_nbar.arrive_and_wait()

            #
            # Sequence loop
            #
            for s in cutlass.range(iters_s):
                # Load S from tmem
                s_handle = s_consumer.wait_and_advance()
                tSrS = cute.make_rmem_tensor(tSsP.shape[:-1], acc_dtype)
                cute.copy(tmem_load_s, tStS[cpy_dice + (s_handle.index,)], tSrS)
                cute.arch.fence_view_async_tmem_load()
                s_handle.release()

                # Gather4 maps invalid sparse entries to slot zero for memory
                # safety.  Mask those logits again here so slot zero never
                # contributes to the softmax.
                tSrValid = cute.make_rmem_tensor(tSrS.shape, cutlass.Int32)
                selected_tile = kv_split_idx + s * kv_splits
                for i in cutlass.range_constexpr(cute.size(tSrS)):
                    score_coord = tScS[i]
                    selected_idx = selected_tile * blk_tile_s + score_coord[0]
                    valid = cutlass.Boolean(False)
                    if selected_idx < selected_slots.shape[1] and score_coord[1] < 6:
                        valid = selected_slots[coord_hb, selected_idx] > 0
                    tSrValid[i] = cutlass.Int32(1) if valid else cutlass.Int32(0)
                    if not valid:
                        tSrS[i] = cutlass.Float32(-1.0e30)

                # Reduce colmax in warp RF
                tSrM = cute.make_rmem_tensor_like(tSsM)
                tSrM_lane = cutlass.Float32(0)  # Avoid dynamic register indexing
                for i in cutlass.range_constexpr(cute.size(tSrS)):
                    tSrM[i] = warp_fmax(tSrS[i])
                    if i == lane_idx:
                        tSrM_lane = tSrM[i]

                # Reduce colmax in smem
                if lane_store_max:
                    smem_fmax(tSsM.iterator + tSsM.layout(lane_idx), tSrM_lane)

                # Wait for colmax then load
                softmax_nbar.arrive_and_wait()
                cute.autovec_copy(tSsM, tSrM)

                # Compute online softmax
                tSrP = cute.make_rmem_tensor(tSsP.shape[:-1], mma_dtype)
                tSrP_f32 = cute.make_rmem_tensor(tSrS.shape, acc_dtype)
                for i in cutlass.range_constexpr(0, cute.size(tSrS), 2):
                    p_f32x2 = fadd2((tSrS[i], tSrS[i + 1]), (-tSrM[i], -tSrM[i + 1]))
                    p_f32x2 = fmul2(p_f32x2, (scale_qs_log2_e, scale_qs_log2_e))
                    tSrP_f32[i] = (
                        exp2(p_f32x2[0]) if tSrValid[i] != 0 else cutlass.Float32(0.0)
                    )
                    tSrP_f32[i + 1] = (
                        exp2(p_f32x2[1])
                        if tSrValid[i + 1] != 0
                        else cutlass.Float32(0.0)
                    )
                tSrP.store(tSrP_f32.load().to(mma_dtype))

                # Store P to smem
                p_handle = p_producer.acquire_and_advance()
                cute.autovec_copy(tSrP, tSsP[cpy_dice + (p_handle.index,)])
                cute.arch.fence_view_async_shared()
                p_handle.commit()

                # Compute correction and correct colsum
                correction = cute.make_rmem_tensor_like(tSrM)
                for i in cutlass.range_constexpr(0, cute.size(tSrM), 2):
                    c_f32x2 = fadd2(
                        (tSrM_prev[i], tSrM_prev[i + 1]), (-tSrM[i], -tSrM[i + 1])
                    )
                    c_f32x2 = fmul2(c_f32x2, (scale_qs_log2_e, scale_qs_log2_e))
                    c_f32x2 = (exp2(c_f32x2[0]), exp2(c_f32x2[1]))
                    correction[i] = c_f32x2[0]
                    correction[i + 1] = c_f32x2[1]
                    l_f32x2 = ffma2(
                        c_f32x2,
                        (tSrL[i], tSrL[i + 1]),
                        (tSrP_f32[i], tSrP_f32[i + 1]),
                    )
                    tSrL[i] = l_f32x2[0]
                    tSrL[i + 1] = l_f32x2[1]

                # Correct O
                if s > 0:
                    # Wait for O
                    o_handle = o_consumer.wait_and_advance()

                    # Apply correction
                    for dm in cutlass.range_constexpr(tiles_dm):
                        tSrO_dm = cute.make_rmem_tensor_like(tSsO[cpy_dice + (0, 0)])
                        cute.copy(thr_load_s, tStO[cpy_dice + (dm, 0)], tSrO_dm)

                        for i in cutlass.range_constexpr(0, cute.size(tSrO_dm), 2):
                            o_f32x2 = fmul2(
                                (tSrO_dm[i], tSrO_dm[i + 1]),
                                (correction[i], correction[i + 1]),
                            )
                            tSrO_dm[i] = o_f32x2[0]
                            tSrO_dm[i + 1] = o_f32x2[1]

                        cute.copy(thr_store_o, tSrO_dm, tStO[cpy_dice + (dm, 0)])

                    # Notify MMA
                    cute.arch.fence_view_async_tmem_store()
                    o_handle.release()

                # Update colmax
                tSrM_prev.store(tSrM.load())

            #
            # Softmax Epilogue
            #

            # Reduce colsum in warp RF
            tSrL_lane = cutlass.Float32(0.0)
            for i in cutlass.range_constexpr(cute.size(tSrL)):
                tSrL[i] = cute.arch.warp_reduction_sum(tSrL[i])
                if i == lane_idx:
                    tSrL_lane = tSrL[i]

            # Store partial colsum in smem
            if lane_store_max:
                tSsL[cpy_dice + (warpgroup_widx,)][lane_idx] = tSrL_lane

            # Wait for colsum
            softmax_nbar.arrive_and_wait()

            if warpgroup_widx == 0 and lane_store_max:
                # Load colsum and colmax
                sL_lane_wg = sL[0, lane_idx, None]
                sL_lane = sL_lane_wg[0] + sL_lane_wg[1] + sL_lane_wg[2] + sL_lane_wg[3]
                sM_lane = sM[0, lane_idx]

                # Scale colmax
                sM_lane = sM_lane * scale_qs

                # Keep final statistics local for the DSM merge.
                sL[0, lane_idx, 0] = sL_lane
                sM[0, lane_idx] = sM_lane

            o_handle = o_consumer.wait_and_advance()
            cute.copy(thr_load_s, tStO, tSrO)
            cute.arch.fence_view_async_tmem_load()
            o_handle.release()  # Final release signals tmem dealloc

            # Persist this CTA's numerator directly in its own completed KV
            # staging allocation.  The coordinate partition is exactly the
            # one paired with the TMEM fragment, avoiding swizzle inversion.
            local_partial = cute.make_tensor(
                cute.recast_ptr(tAsK.iterator, dtype=acc_dtype),
                cute.make_layout((6, blk_tile_d + 2), stride=(blk_tile_d + 2, 1)),
            )
            for i in cutlass.range_constexpr(cute.size(tSrO)):
                coord = tScO[i]
                dim = coord[0]
                head = coord[1]
                if head < 6:
                    local_partial[head, dim] = tSrO[i]
            if warpgroup_widx == 0 and lane_idx < 6:
                local_partial[lane_idx, blk_tile_d] = sM[0, lane_idx]
                local_partial[lane_idx, blk_tile_d + 1] = sL[0, lane_idx, 0]

        # All split-local numerators now reside at the same offset in each
        # CTA's DSM.  The eight-way small-row specialization has enough ranks
        # to assign one output head to each of ranks 0..5.  Those ranks reduce
        # their head directly from peer DSM, avoiding the gather into rank zero
        # and one cluster barrier.  Keep the compact gather/reduce path for the
        # four-way large-row specialization.
        cute.arch.sync_threads()
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        cluster_splits = self.kv_splits
        reduction_heads = 6
        local_partial = cute.make_tensor(
            cute.recast_ptr(tAsK.iterator, dtype=acc_dtype),
            cute.make_layout(
                (reduction_heads, blk_tile_d + 2), stride=(blk_tile_d + 2, 1)
            ),
        )
        cta_rank = cute.arch.block_idx_in_cluster()
        if cutlass.const_expr(self.kv_splits == 8):
            if cta_rank < reduction_heads:
                head = cta_rank
                if tidx == 0:
                    global_max = -cutlass.Float32.inf
                    for split in cutlass.range_constexpr(cluster_splits):
                        max_ptr = cute.domain_offset(
                            (head, blk_tile_d), local_partial
                        ).iterator
                        split_max = ld_shared_remote_f32(set_block_rank(max_ptr, split))
                        global_max = cute.arch.fmax(global_max, split_max)
                    denominator = cutlass.Float32(0.0)
                    for split in cutlass.range_constexpr(cluster_splits):
                        max_ptr = cute.domain_offset(
                            (head, blk_tile_d), local_partial
                        ).iterator
                        sum_ptr = cute.domain_offset(
                            (head, blk_tile_d + 1), local_partial
                        ).iterator
                        split_max = ld_shared_remote_f32(set_block_rank(max_ptr, split))
                        split_sum = ld_shared_remote_f32(set_block_rank(sum_ptr, split))
                        correction = exp2(log2_e * (split_max - global_max))
                        denominator += correction * split_sum
                    sM[0, head] = global_max
                    sL[0, head, 0] = denominator

            cute.arch.sync_threads()
            if cta_rank < reduction_heads and tidx < blk_tile_d:
                head = cta_rank
                dim = tidx
                numerator = cutlass.Float32(0.0)
                for split in cutlass.range_constexpr(cluster_splits):
                    max_ptr = cute.domain_offset(
                        (head, blk_tile_d), local_partial
                    ).iterator
                    value_ptr = cute.domain_offset((head, dim), local_partial).iterator
                    split_max = ld_shared_remote_f32(set_block_rank(max_ptr, split))
                    split_value = ld_shared_remote_f32(set_block_rank(value_ptr, split))
                    correction = exp2(log2_e * (split_max - sM[0, head]))
                    numerator += correction * split_value
                mOut[dim, head, coord_hb] = mOut.element_type(
                    scale_o * numerator / sL[0, head, 0]
                    if sL[0, head, 0] > 0.0
                    else 0.0
                )
        else:
            ranked_partial = cute.make_tensor(
                cute.recast_ptr(tAsK.iterator, dtype=acc_dtype),
                cute.make_layout(
                    (cluster_splits, reduction_heads, blk_tile_d + 2),
                    stride=(
                        reduction_heads * (blk_tile_d + 2),
                        blk_tile_d + 2,
                        1,
                    ),
                ),
            )
            elems = reduction_heads * blk_tile_d
            elems_per_thread = cute.ceil_div(elems, self.threads_per_cta)
            for i in cutlass.range_constexpr(elems_per_thread):
                elem = tidx + i * self.threads_per_cta
                if elem < elems:
                    head = elem // blk_tile_d
                    dim = elem - head * blk_tile_d
                    dst = cute.domain_offset(
                        (cta_rank, head, dim), ranked_partial
                    ).iterator
                    st_shared_remote_f32(
                        set_block_rank(dst, cutlass.Int32(0)),
                        local_partial[head, dim],
                    )
            if tidx < reduction_heads:
                for stat in cutlass.range_constexpr(2):
                    dim = blk_tile_d + stat
                    dst = cute.domain_offset(
                        (cta_rank, tidx, dim), ranked_partial
                    ).iterator
                    st_shared_remote_f32(
                        set_block_rank(dst, cutlass.Int32(0)),
                        local_partial[tidx, dim],
                    )

            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

            if cta_rank == 0:
                if tidx < reduction_heads:
                    global_max = -cutlass.Float32.inf
                    for split in cutlass.range_constexpr(cluster_splits):
                        global_max = cute.arch.fmax(
                            global_max, ranked_partial[split, tidx, blk_tile_d]
                        )
                    denominator = cutlass.Float32(0.0)
                    for split in cutlass.range_constexpr(cluster_splits):
                        correction = exp2(
                            log2_e
                            * (ranked_partial[split, tidx, blk_tile_d] - global_max)
                        )
                        denominator += (
                            correction * ranked_partial[split, tidx, blk_tile_d + 1]
                        )
                    sM[0, tidx] = global_max
                    sL[0, tidx, 0] = denominator

                cute.arch.sync_threads()
                for i in cutlass.range_constexpr(elems_per_thread):
                    elem = tidx + i * self.threads_per_cta
                    if elem < elems:
                        head = elem // blk_tile_d
                        dim = elem - head * blk_tile_d
                        numerator = cutlass.Float32(0.0)
                        for split in cutlass.range_constexpr(cluster_splits):
                            correction = exp2(
                                log2_e
                                * (
                                    ranked_partial[split, head, blk_tile_d]
                                    - sM[0, head]
                                )
                            )
                            numerator += correction * ranked_partial[split, head, dim]
                        mOut[dim, head, coord_hb] = mOut.element_type(
                            scale_o * numerator / sL[0, head, 0]
                            if sL[0, head, 0] > 0.0
                            else 0.0
                        )

        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        return


_NUM_HEADS = 6
_HEAD_DIM = 256
_SELECTED_WIDTH = 2051
_SMALL_MAX_ROWS = 8
_SMALL_NUM_SPLITS = 8
_LARGE_NUM_SPLITS = 4
_COMPILED_KERNELS = {}


def _to_tvm_meta(tensor: torch.Tensor, assumed_align: int = 16) -> cute.Tensor:
    """Create one compile-time tensor descriptor for direct Torch TVM FFI."""

    storage = tensor
    fp8 = tensor.dtype is torch.float8_e4m3fn
    if fp8:
        storage = tensor.view(torch.uint8)
    result = from_dlpack(
        storage.detach(),
        assumed_align=assumed_align,
        enable_tvm_ffi=True,
    )
    if fp8:
        result.element_type = cutlass.Float8E4M3FN
    return result


def _scalar(value: float | torch.Tensor | None, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("QSA FP8 descales must be scalar")
        return float(value.item())
    return float(value)


def _compile_kernel(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    selected_slots: torch.Tensor,
    output: torch.Tensor,
):
    problem_shape = (
        query.shape[0],
        _NUM_HEADS,
        1,
        key_cache.shape[0],
        _HEAD_DIM,
    )
    small_batch = query.shape[0] <= _SMALL_MAX_ROWS
    kv_splits = _SMALL_NUM_SPLITS if small_batch else _LARGE_NUM_SPLITS
    fmha = MixedInputFusedMultiHeadAttentionDecode(
        headdim=_HEAD_DIM,
        block_scaledim=_HEAD_DIM,
        grouped_head_tile=8,
        convert_warpgroups=2,
        kv_splits=kv_splits,
    )
    fmha.problem_shape = problem_shape
    return cute.compile(
        fmha,
        _to_tvm_meta(query),
        _to_tvm_meta(key_cache),
        _to_tvm_meta(value_cache),
        _to_tvm_meta(selected_slots),
        _to_tvm_meta(output),
        1.0,
        1.0,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --opt-level 2",
    )


def kernel(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    selected_slots: torch.Tensor,
    *,
    scale: float,
    max_seqlen_q: int = 1,
    k_scale: float | torch.Tensor | None = None,
    v_scale: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Run direct-slot QSA attention with adaptive SM100 CTA clusters."""

    if query.ndim != 3 or query.shape[1:] != (_NUM_HEADS, _HEAD_DIM):
        raise ValueError("candidate requires query shape [tokens, 6, 256]")
    if key_cache.ndim != 3 or key_cache.shape[1:] != (1, _HEAD_DIM):
        raise ValueError("candidate requires key cache shape [slots, 1, 256]")
    if value_cache.shape != key_cache.shape:
        raise ValueError("candidate requires matching K/V cache shapes")
    if selected_slots.shape != (query.shape[0], _SELECTED_WIDTH):
        raise ValueError("candidate requires selected slots shape [tokens, 2051]")
    if query.dtype is not torch.bfloat16:
        raise TypeError("candidate requires BF16 query")
    if (
        key_cache.dtype is not torch.float8_e4m3fn
        or value_cache.dtype is not torch.float8_e4m3fn
    ):
        raise TypeError("candidate requires FP8 E4M3FN K/V cache")
    if selected_slots.dtype is not torch.int32:
        raise TypeError("candidate requires int32 selected slots")
    if max_seqlen_q < 1 or query.shape[0] % max_seqlen_q:
        raise ValueError(
            "candidate requires positive max_seqlen_q dividing the query rows"
        )
    if query.shape[0] == 0:
        return torch.empty_like(query)

    # The tuned production layout is compact [token, head, dim]. Keep the
    # independent stride check correct without penalizing that hot path.
    if not query.is_contiguous():
        query = query.contiguous()
    output = torch.empty(
        query.shape,
        dtype=query.dtype,
        device=query.device,
    )
    scale_qs = float(scale) * _scalar(k_scale, 1.0)
    scale_o = _scalar(v_scale, 1.0)
    problem_shape = (
        query.shape[0],
        _NUM_HEADS,
        1,
        key_cache.shape[0],
        _HEAD_DIM,
    )
    cache_key = (
        query.device.index,
        problem_shape,
        max_seqlen_q,
        _SMALL_NUM_SPLITS if query.shape[0] <= _SMALL_MAX_ROWS else _LARGE_NUM_SPLITS,
        tuple(selected_slots.stride()),
    )
    compiled = _COMPILED_KERNELS.get(cache_key)
    if compiled is None:
        compiled = _compile_kernel(
            query,
            key_cache,
            value_cache,
            selected_slots,
            output,
        )
        _COMPILED_KERNELS[cache_key] = compiled
    compiled(
        query,
        key_cache.view(torch.uint8),
        value_cache.view(torch.uint8),
        selected_slots,
        output,
        scale_qs,
        scale_o,
    )
    return output

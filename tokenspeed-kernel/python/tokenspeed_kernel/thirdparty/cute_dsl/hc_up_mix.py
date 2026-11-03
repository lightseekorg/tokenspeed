# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Literal

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.gemm.sm100 as gemm_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait


@cute.jit
def mix_epilogue(
    gemm,
    tidx,
    acc_base,
    gate_base,
    coord_base,
    out_base,
    rows,
    tile_coord,
    state,
    acc_pipeline,
    sx,
    x_state,
    x_pipeline,
):
    epi_tile = (gemm.x_tile[0], 32)
    pieces = gemm.x_tile[1] // epi_tile[1]
    acc = gemm_utils.transform_partitioned_tensor_layout(acc_base)
    gates = gemm_utils.transform_partitioned_tensor_layout(gate_base)
    coords = gemm_utils.transform_partitioned_tensor_layout(coord_base)
    out = gemm_utils.transform_partitioned_tensor_layout(out_base)
    copy_t2r, acc_parts, reg = gemm_utils.epilogue_tmem_copy_and_partition(
        gemm, tidx, acc, gates, epi_tile, gemm.use_2cta_instrs
    )
    thr = copy_t2r.get_slice(tidx)
    c = thr.partition_D(cute.flat_divide(coords, epi_tile))[
        (None, None, None, None, None, *tile_coord)
    ]
    oo = thr.partition_D(cute.flat_divide(out, epi_tile))[
        (None, None, None, None, None, *tile_coord)
    ]
    ss = thr.partition_D(cute.flat_divide(sx, epi_tile))
    a = acc_parts[(None, None, None, None, None, state.index)]
    mixed = cute.make_rmem_tensor(reg.shape, cutlass.Float32)
    value = cute.make_rmem_tensor((*reg.shape, gemm.hc_count), gemm.c_dtype)
    result = cute.make_rmem_tensor(reg.shape, gemm.c_dtype)
    atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), gemm.c_dtype, num_bits_per_copy=128
    )
    pred = cute.make_rmem_tensor((1, *reg.shape[1:]), cutlass.Boolean)
    barrier = pipeline.NamedBarrier(barrier_id=gemm.epilog_sync_bar_id, num_threads=128)

    acc_pipeline.consumer_wait(state)
    x_pipeline.consumer_wait(x_state)
    for piece in range(pieces):
        mixed.fill(0.0)
        cc = c[(None, None, None, 0, piece)]
        for mi in cutlass.range_constexpr(reg.shape[1]):
            for ni in cutlass.range_constexpr(reg.shape[2]):
                pred[0, mi, ni] = cc[0, mi, ni][0] < rows
        for branch in cutlass.range_constexpr(gemm.hc_count):
            cute.copy(
                atom,
                ss[(None, None, None, 0, branch * pieces + piece)],
                value[(None, None, None, branch)],
            )
        for branch in cutlass.range_constexpr(gemm.hc_count):
            vx = value[(None, None, None, branch)]
            cute.copy(copy_t2r, a[(None, None, None, 0, branch * pieces + piece)], reg)
            for i in cutlass.range_constexpr(cute.size(reg)):
                # Preserve the unfused GEMM store rounding before sigmoid.
                gate = reg[i].to(gemm.c_dtype).to(cutlass.Float32)
                weight = cute.arch.rcp_approx(1.0 + cute.exp(-gate, fastmath=True))
                mixed[i] = mixed[i] + weight * vx[i].to(cutlass.Float32)
        result.store((mixed.load() * (1.0 / gemm.hc_count)).to(gemm.c_dtype))
        cute.copy(atom, result, oo[(None, None, None, 0, piece)], pred=pred)
    cute.arch.fence_view_async_tmem_load()
    # All SMEM reads have been consumed by the math before these arrivals.
    x_pipeline.consumer_release(x_state)
    x_state.advance()
    with cute.arch.elect_one():
        acc_pipeline.consumer_release(state)
    state.advance()
    barrier.arrive_and_wait()
    return state, x_state


class HcUpMixKernel:
    """SM107 HC4/H2560/R320: 2-CTA M256/N256/K64 Up GEMM and Mix.

    Adapted from CUTLASS dense_gemm_persistent.py (CuTe DSL 4.8).
    Five A/B stages and two independent X slots use about 288 KiB per CTA;
    this configuration exceeds Blackwell's shared-memory capacity. Four
    epilogue warps consume two accumulator slots, alongside one MMA warp and
    independent A/B and X TMA warps. Packing groups 64 columns from each branch
    so every GEMM tile produces complete four-branch Mix outputs.
    """

    def __init__(self, *, use_pdl: bool):
        self.use_pdl = use_pdl
        self.acc_dtype = cutlass.Float32
        self.use_2cta_instrs = True
        self.cluster_shape_mn = (2, 1)
        self.swizzle_size = 1
        self.raster_along = "m"
        self.mma_tiler = (256, 256, 64)
        self.hc_count = 4
        self.hidden_size = 2560
        # Each CTA loads one X plane per branch for its share of the GEMM tile.
        self.x_tile = (
            self.mma_tiler[0] // self.cluster_shape_mn[0],
            self.mma_tiler[1] // self.hc_count,
            1,
        )
        self.arch = "sm_100"
        self.cta_group = tcgen05.CtaGroup.TWO
        self.epilogue_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.x_tma_warp_id = 6
        self.num_x_stage = 2
        self.num_acc_stage = 2
        self.num_ab_stage = 5
        self.threads_per_cta = 224
        self.epilog_sync_bar_id = 1
        self.tmem_alloc_sync_bar_id = 2
        # These layout-dependent fields are filled during JIT setup.
        self.a_dtype = self.b_dtype = self.c_dtype = None
        self.a_major_mode = self.b_major_mode = self.c_layout = None
        self.cta_tile_shape_mnk = self.cluster_layout_vmnk = None
        self.num_mcast_ctas_a = self.num_mcast_ctas_b = None
        self.is_a_mcast = self.is_b_mcast = None
        self.a_smem_layout_staged = self.b_smem_layout_staged = None
        self.num_tmem_alloc_cols = self.num_tma_load_bytes = None
        self.tile_sched_params = None

    def _create_tiled_mma(self):
        return utils.sm100.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )

    def _setup_attributes(self):
        # Configure tiled mma
        tiled_mma = self._create_tiled_mma()

        # Compute mma/cluster/tile shapes
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # Compute A/B shared memory layout
        self.a_smem_layout_staged = utils.sm100.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage
        )
        self.b_smem_layout_staged = utils.sm100.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage
        )

        # Compute the number of tensor memory allocation columns
        self.num_tmem_alloc_cols = self._compute_num_tmem_alloc_cols(
            tiled_mma, self.mma_tiler, self.num_acc_stage, self.arch
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        x: cute.Tensor,
        out: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        # Virtual gate tensor supplies only GEMM shape/coordinate metadata.
        # No full gate output is allocated or stored.
        c = cute.make_tensor(
            out.iterator,
            cute.make_layout((a.shape[0], b.shape[0], 1), stride=(b.shape[0], 1, 0)),
        )
        # Setup static attributes before smem/grid/tma computation
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.a_major_mode = cutlass.tensor_utils.LayoutEnum.from_tensor(
            a
        ).mma_major_mode()
        self.b_major_mode = cutlass.tensor_utils.LayoutEnum.from_tensor(
            b
        ).mma_major_mode()
        self.c_layout = cutlass.tensor_utils.LayoutEnum.from_tensor(c)

        # Check if input data types are compatible with MMA instruction
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        tiled_mma = self._create_tiled_mma()

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()

        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Setup TMA load for A
        a_op = utils.sm100.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.select(
            self.a_smem_layout_staged,
            mode=list(range(cute.rank(self.a_smem_layout_staged) - 1)),
        )
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(
                cutlass.TFloat32 if a.element_type is cutlass.Float32 else None
            ),
        )

        # Setup TMA load for B
        b_op = utils.sm100.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.select(
            self.b_smem_layout_staged,
            mode=list(range(cute.rank(self.b_smem_layout_staged) - 1)),
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(
                cutlass.TFloat32 if b.element_type is cutlass.Float32 else None
            ),
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size

        # One descriptor loads each original-layout 128x64 X branch plane.
        # The shared-pointer swizzle is expressed in bytes.
        x_plane_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout(self.x_tile, stride=(self.x_tile[1], 1, 0)),
        )
        tma_atom_x, tma_tensor_x = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), x, x_plane_layout, cta_tiler=self.x_tile
        )
        # Compute grid size
        self.tile_sched_params, grid = self._compute_grid(
            c,
            self.cta_tile_shape_mnk,
            self.cluster_shape_mn,
            self.swizzle_size,
            self.raster_along,
            max_active_clusters,
        )

        # Launch on the caller's stream.
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            c,
            tma_tensor_x,
            out,
            tma_atom_x,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            use_pdl=self.use_pdl,
        )
        return

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        mC_mnl: cute.Tensor,
        mX: cute.Tensor,
        mOut: cute.Tensor,
        tma_atom_x: cute.CopyAtom,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

        # Prefetch tma desc

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
        if warp_idx == self.x_tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_x)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        # Setup cta/thread coordinates

        # Coords inside cluster
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier

        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_acc_stage * 2
            ]
            x_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_x_stage * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Initialize mainloop ab_pipeline (barrier) and states
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Warp)
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            enable_multicast_signaling=True,
            defer_sync=True,
        ).make_participants()

        # X is CTA-local; each of the 128 consumers releases after its last read.
        x_rows, x_cols, _ = self.x_tile
        x_plane_elements = x_rows * x_cols
        x_slot_elements = x_plane_elements * self.hc_count
        branch_tiles = self.hidden_size // x_cols
        x_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.x_mbar_ptr.data_ptr(),
            num_stages=self.num_x_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
            tx_count=x_slot_elements * self.c_dtype.width // 8,
            enable_multicast_signaling=True,
            defer_sync=True,
        )

        # Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilogue_warp_id) * (
            2 if use_2cta_instrs else 1
        )
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_id)),
        )
        # Tensor memory dealloc barrier init
        tmem = cutlass.memory.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # Setup smem tensors A/B/X

        # (MMA, MMA_M, MMA_K, STAGE)
        sA = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )

        # X slots are independent of the A/B and accumulator rings.
        sX = smem.allocate_tensor(
            element_type=self.c_dtype,
            layout=cute.make_layout(
                (x_rows, (x_cols, self.hc_count), self.num_x_stage),
                stride=(x_cols, (1, x_plane_elements), x_slot_elements),
            ),
            byte_alignment=128,
            swizzle=cute.make_swizzle(3, 4, 3),
        )

        # Compute multicast mask for A/B buffer full

        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        # Local_tile partition global tensors

        # (bM, bK, RestM, RestK, RestL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bM, bN, RestM, RestN, RestL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        # Partition global tensor for TiledMMA_A/B/C

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
        tCgC = thr_mma.partition_C(gC_mnl)

        # All four branches share output coordinates; only branch zero stores.
        out_view = cute.make_tensor(
            mOut.iterator,
            cute.make_layout(
                (mOut.shape[0], (x_cols, self.hc_count, branch_tiles), 1),
                stride=(self.hidden_size, (1, 0, x_cols), 0),
            ),
        )
        tCgOut = thr_mma.partition_C(
            cute.local_tile(out_view, self.mma_tiler[:2], (None, None, None))
        )

        # Coordinate tensor used to predicate SIMT stores from partial clusters.
        idC = cute.make_identity_tensor(mC_mnl.shape)
        cC_mnl = cute.local_tile(
            idC, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        tCcC = thr_mma.partition_C(cC_mnl)

        # Partition global/shared tensor for TMA load A/B

        # TMA load A partition_S/D
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        # Partition shared/tensor memory tensor for TiledMMA_A/B/C

        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        # Cluster wait before tensor memory alloc

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # Construct the scheduler

        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
        )
        work_tile = tile_sched.initial_work_tile_info()

        # Specialized TMA load warp

        if warp_idx == self.tma_warp_id:

            # Persistent tile scheduling loop

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                # Slice to per mma tile index

                # ((atom_v, rest_v), RestK)
                tAgA_slice = tAgA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]
                # ((atom_v, rest_v), RestK)
                tBgB_slice = tBgB[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]

                # Check availability of the first A/B stage.
                ab_producer.reset()
                peek_ab_empty_status = ab_producer.try_acquire()

                # Tma load loop

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # Conditionally wait for AB buffer empty
                    handle = ab_producer.acquire_and_advance(peek_ab_empty_status)

                    # TMA load A/B
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, handle.count)],
                        tAsA[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, handle.count)],
                        tBsB[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=b_full_mcast_mask,
                    )

                    # Check availability of the next A/B stage.
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if handle.count + 1 < k_tile_cnt:
                        peek_ab_empty_status = ab_producer.try_acquire()

                # Advance to next tile

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            # Wait A/B buffer empty

            ab_producer.tail()

        # X has its own producer so waiting for an X slot never blocks A/B.
        if warp_idx == self.x_tma_warp_id:
            x_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_x_stage
            )
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                cluster_m = cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape)
                x_pipeline.producer_acquire(x_producer_state)
                x_barrier = x_pipeline.producer_get_barrier(x_producer_state)
                for branch in cutlass.range_constexpr(self.hc_count):
                    gx = cute.local_tile(
                        mX,
                        self.x_tile,
                        (
                            cluster_m * self.cluster_shape_mn[0] + mma_tile_coord_v,
                            cur_tile_coord[1] + branch * branch_tiles,
                            0,
                        ),
                    )
                    sx = cute.make_tensor(
                        sX.iterator
                        + x_producer_state.index * x_slot_elements
                        + branch * x_plane_elements,
                        cute.make_layout(self.x_tile, stride=(x_cols, 1, 0)),
                    )
                    sx_part, gx_part = cpasync.tma_partition(
                        tma_atom_x,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sx, 0, 3),
                        cute.group_modes(gx, 0, 3),
                    )
                    cute.copy(tma_atom_x, gx_part, sx_part, tma_bar_ptr=x_barrier)
                x_producer_state.advance()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            x_pipeline.producer_tail(x_producer_state)

        # Specialized MMA warp

        if warp_idx == self.mma_warp_id:

            # Retrieving tensor memory ptr and make accumulator tensor

            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            # Persistent tile scheduling loop

            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                # Set tensor memory buffer for current tile
                # (MMA, MMA_M, MMA_N)
                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]

                # Peek (try_wait) AB buffer full for k_tile = 0
                ab_consumer.reset()
                peek_ab_full_status = cutlass.Boolean(1)
                if is_leader_cta:
                    peek_ab_full_status = ab_consumer.try_wait()

                # Wait for accumulator buffer empty

                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                # Mma mainloop

                for k_tile in range(k_tile_cnt):
                    if is_leader_cta:
                        # Conditionally wait for AB buffer full
                        handle = ab_consumer.wait_and_advance(peek_ab_full_status)

                        # tCtAcc += tCrA * tCrB
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                        tile_crd = (None, None, None, handle.index)
                        cute.gemm(
                            tiled_mma, tCtAcc, tCrA[tile_crd], tCrB[tile_crd], tCtAcc
                        )

                        # Async arrive AB buffer empty
                        handle.release()

                        # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
                        peek_ab_full_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_full_status = ab_consumer.try_wait()

                # Async arrive accumulator buffer full

                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                # Advance to next tile

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            # Wait for accumulator buffer empty

            acc_pipeline.producer_tail(acc_producer_state)

        # Specialized epilogue warps

        if warp_idx < self.mma_warp_id:

            # Alloc tensor memory buffer

            tmem.allocate(self.num_tmem_alloc_cols)

            # Retrieving tensor memory ptr and make accumulator tensor

            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            # Persistent tile scheduling loop for epilogue

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            x_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_x_stage
            )
            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                # Pre-advance to next tile

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

                acc_consumer_state, x_consumer_state = mix_epilogue(
                    self,
                    tidx,
                    tCtAcc_base,
                    tCgC,
                    tCcC,
                    tCgOut,
                    mX.shape[0],
                    mma_tile_coord_mnl,
                    acc_consumer_state,
                    acc_pipeline,
                    sX[(None, None, x_consumer_state.index)],
                    x_consumer_state,
                    x_pipeline,
                )

            # Dealloc the tensor memory buffer

            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        cta_tile_shape_mnk: tuple[int, int, int],
        cluster_shape_mn: tuple[int, int],
        swizzle_size: int,
        raster_along: Literal["m", "n"],
        max_active_clusters: cutlass.Constexpr,
    ) -> tuple[utils.PersistentTileSchedulerParams, tuple[int, int, int]]:
        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl, swizzle_size, raster_along == "m"
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        return tile_sched_params, grid

    @staticmethod
    def _compute_num_tmem_alloc_cols(
        tiled_mma: cute.TiledMma,
        mma_tiler: tuple[int, int, int],
        num_acc_stage: int,
        arch: str,
    ) -> int:
        acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, num_acc_stage))
        num_tmem_alloc_cols = cutlass.memory.get_num_tmem_alloc_cols(
            tCtAcc_fake, arch=arch
        )

        return num_tmem_alloc_cols

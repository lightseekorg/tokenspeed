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

"""Fused FP8 DP projections for a fully resident GPU grid.

Derived from NVIDIA CUTLASS blockwise_gemm.py at revision
0b55a2f691d69981583568fd9eb69687b1f0de8a; retain its BSD license above.

One launch quantizes owned BF16 rows with FP32 1x128 scales, pushes FP8 rows
and scales to peers, computes the column-sharded GEMM, and stores each
output tile into its row owner's symmetric receive buffer. A final local
copy preserves ordinary caller-owned output tensors. Row-parallel TP2/4
projections send input-channel shards and reduce the received BF16 partials
in FP32 into caller-owned outputs. Input publication and
output consumption use 64-bit device epochs; no flag reset occurs on replay.

Row buckets of at least 128 rows divide scaling and accumulation across two
warp groups. Each group keeps half the output columns in registers and stores
the rounded BF16 tile into shared memory for the epilogue. This leaves all
four TMEM stages available for partial GEMM accumulators. Received partials
are contiguous per producer, and output owners sum them in fixed rank order.

Output producers acquire consumption acknowledgements before reusing receive
storage. Some row buckets consume every prepared partial, including padding;
that completion also retires reads of the owner's input. Those buckets can
reuse the same input layout without another peer join. Layout changes and
kernels that skip padding retain the acknowledgement join.

Node-local GB300 GPUs use prepared row buckets, aligned widths, and a fixed,
fully resident grid on idle GPUs. Column and row projections use their own
bucket ladders; remote stores split GEMM tiles at owner boundaries.
Input staging zero-fills each owner's row tail, and final stores write only
that owner's physical rows. Empty owners still compute their weight shard
and participate in every epoch.
Concurrent workspace use and arbitrary competing collectives are unsupported.

Prepared tensor strides stay fixed while a runtime row bound skips whole
padding tiles and their reduction jobs.
"""

import math
from typing import Tuple, Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait


@dsl_user_op
def _tma_l2_policy(*, keep: bool, loc=None, ip=None):
    priority = "evict_last" if keep else "evict_first"
    return cutlass.Int64(
        llvm.inline_asm(
            cutlass.Int64.mlir_type,
            [],
            f"createpolicy.fractional.L2::{priority}.L2::evict_unchanged.b64 $0, 1.0;",
            "=l",
            has_side_effects=False,
            is_align_stack=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _publish_input_epoch(
    address: cutlass.Int64, value: cutlass.Int64, *, loc=None, ip=None
):
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip), value.ir_value(loc=loc, ip=ip)],
        "multimem.st.release.sys.global.b64 [$0], $1;",
        "l,l",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _queue_poll_backoff(*, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [],
        "nanosleep.u32 64;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _load_payload_128(address: cutlass.Int64, *, l2_only: bool, loc=None, ip=None):
    # Receive buffers are local; TP2 polls their payloads through L2.
    scalar = cutlass.Uint32.mlir_type
    result = llvm.inline_asm(
        llvm.StructType.get_literal([scalar] * 4),
        [address.ir_value(loc=loc, ip=ip)],
        (
            "ld.global.cg.v4.b32 {$0, $1, $2, $3}, [$4];"
            if l2_only
            else "ld.volatile.global.v4.b32 {$0, $1, $2, $3}, [$4];"
        ),
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return tuple(
        cutlass.Uint32(llvm.extractvalue(scalar, result, [i], loc=loc, ip=ip))
        for i in range(4)
    )


@dsl_user_op
def _clear_payload_streaming(address: cutlass.Int64, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip)],
        "{ .reg .b32 sentinel; mov.b32 sentinel, 0x80008000; st.global.cs.v4.b32 [$0], {sentinel, sentinel, sentinel, sentinel}; }",
        "l",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _canonicalize_zero_pair(value: cutlass.Uint32, *, loc=None, ip=None):
    result = llvm.inline_asm(
        cutlass.Uint32.mlir_type,
        [value.ir_value(loc=loc, ip=ip)],
        "{ .reg .b32 nonzero; and.b32 nonzero, $1, 0x7fff7fff; add.u32 nonzero, nonzero, 0x7fff7fff; or.b32 nonzero, nonzero, 0x7fff7fff; and.b32 $0, $1, nonzero; }",
        "=r,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return cutlass.Uint32(result)


@dsl_user_op
def _publish_epoch_async(
    address: cutlass.Int64, value: cutlass.Int64, *, loc=None, ip=None
):
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip), value.ir_value(loc=loc, ip=ip)],
        "multimem.st.async.release.sys.global.b64 [$0], $1;",
        "l,l",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


class FusedTPProjectionKernel:
    """Fuse owner quantization, row/column-sharded FP8 GEMM and output exchange.

    Use 64x128 or 128x128 tiles on node-local GB300 peers.

    Selected row buckets pair adjacent N tiles to share TMA loads of A.
    A fully resident grid and warp specialization overlap TMA,
    tensor-core MMA, FP32 block scaling, and the output epilogue.
    """

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        *,
        row_parallel: bool,
        accumulator_groups: int,
        use_input_multicast: bool,
    ):
        """Initializes the configuration for a Blackwell blockwise dense GEMM kernel.

        This configuration includes several key aspects:

        1.  MMA Instruction Settings (tcgen05):
            - acc_dtype: Data types for MMA accumulator.
            - mma_tiler_mn: The (M, N) shape of the MMA instruction tiler.
            - use_2cta_instrs: Boolean indicating if the tcgen05 MMA variant
              with cta_group=2 should be used.

        2.  Cluster Shape:
            - cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster.

        :param acc_dtype: Data type of the accumulator.
        :type acc_dtype: type[cutlass.Numeric]
        :param mma_tiler_mn: Tuple (M, N) shape of the MMA instruction.
        :type mma_tiler_mn: Tuple[int, int]
        :param use_2cta_instrs: Boolean, True to use cta_group=2 MMA variant.
        :type use_2cta_instrs: bool
        :param cluster_shape_mn: Tuple (ClusterM, ClusterN) shape of the cluster.
        :type cluster_shape_mn: Tuple[int, int]
        :param accumulator_groups: One or two warp groups for FP32 accumulation.
        :param use_input_multicast: Publish input readiness through multicast memory.
        """

        supported_tiles = ((64, 128), (128, 128)) if row_parallel else ((128, 128),)
        if (
            mma_tiler_mn not in supported_tiles
            or cluster_shape_mn not in (((1, 1), (1, 2)) if row_parallel else ((1, 1),))
            or use_2cta_instrs
        ):
            raise ValueError(
                "Fused TP projection requires a supported 128-column tile and cluster shape"
            )
        self.row_parallel = row_parallel
        if accumulator_groups not in (1, 2):
            raise ValueError("Projection requires one or two accumulator warp groups")
        self.split_accumulator = accumulator_groups == 2
        self.local_flag_scope = "gpu" if row_parallel else "sys"
        self.acc_dtype: Type[cutlass.Numeric] = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        self.num_ctas = 0
        self.gemm_ctas = 0
        self.num_cta_flags = 0
        self.large_row_bucket = False
        self.reduction_rows = 32
        self.use_input_multicast = use_input_multicast
        self.shared_mma_completion = False
        self.dynamic_gemm = False
        self.tmem_before_scales = False
        self.tile_owner_input = False
        self.independent_operand_stages = False
        self.tma_scales = False
        self.input_phase_clear = False
        self.concurrent_reduction = False
        self.reduce_sync_barrier = pipeline.NamedBarrier(barrier_id=4, num_threads=128)
        self.acc_store_sync_barrier = pipeline.NamedBarrier(
            barrier_id=5, num_threads=256
        )
        self.vector_output = False
        self.cache_group_m = 0
        self.skip_padded_tiles = False
        self.scale_k_total = 0
        self.scale_b_stride = 0
        self.bulk_scale_stages = 0
        self.sfa_bulk_layout = None
        self.sfb_bulk_layout = None
        # K dimension is deferred in _setup_attributes
        self.mma_tiler = (*mma_tiler_mn, 1)

        self.cta_group = (
            tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        self.occupancy = 1
        # Set specialized warp ids
        self.acc_update_warp_id = (
            tuple(range(8)) if self.split_accumulator else (0, 1, 2, 3)
        )
        self.epilog_warp_id = (8, 9, 10, 11) if self.split_accumulator else (4, 5, 6, 7)
        self.mma_warp_id = 12 if self.split_accumulator else 8
        self.tma_warp_id = 13 if self.split_accumulator else 9
        self.scale_warp_id = 14 if self.split_accumulator else 10
        self.sched_warp_id = 15 if self.split_accumulator else 11
        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (
                *self.acc_update_warp_id,
                *self.epilog_warp_id,
                self.mma_warp_id,
                self.tma_warp_id,
                self.scale_warp_id,
                self.sched_warp_id,
            )
        )
        self.threads_wo_sched = self.threads_per_warp * len(
            (
                *self.acc_update_warp_id,
                *self.epilog_warp_id,
                self.mma_warp_id,
                self.tma_warp_id,
                self.scale_warp_id,
            )
        )
        self.num_regs_uniform_warps = 64
        self.num_regs_sched_warps = 64
        self.num_regs_epilogue_warps = 64 if self.split_accumulator else 128
        self.num_regs_acc_update_warps = 192 if self.split_accumulator else 256

        # Set barrier for epilogue sync and tmem ptr sync
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=32 * len(self.epilog_warp_id),
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=32
            * len((self.mma_warp_id, *self.epilog_warp_id, *self.acc_update_warp_id)),
        )
        self.sched_sync_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=self.threads_per_warp,
        )
        self.num_smem_capacity = cutlass.memory.get_smem_capacity_in_bytes("sm_100")
        # TMEM offset for final accumulator
        self.tmem_final_offset = 384

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B
        - Computing epilogue subtile
        - Setting up A/B/C stage counts in shared memory
        - Computing A/B/C shared memory layout
        - Computing tensor memory allocation columns
        """
        # Configure tiled mma
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )

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

        self.scale_granularity_m = 1
        self.scale_granularity_n = 128
        self.scale_granularity_k = 128
        self.scale_m_per_tile = self.cta_tile_shape_mnk[0] // self.scale_granularity_m
        self.scale_n_per_tile = self.cta_tile_shape_mnk[1] // self.scale_granularity_n
        self.scale_k_per_tile = self.cta_tile_shape_mnk[2] // self.scale_granularity_k

        if self.scale_k_per_tile != 1:
            raise ValueError("scale_k_per_tile must be 1")
        if self.scale_m_per_tile != self.cta_tile_shape_mnk[0]:
            raise ValueError("scale_m_per_tile must be cta_tile_m")
        if self.scale_n_per_tile != 1:
            raise ValueError("scale_n_per_tile must be 1")

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # Compute epilogue subtile
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )

        self.epi_tile = (self.mma_tiler[0], 64)

        # Setup A/B/C/Scale stage count in shared memory and ACC stage count in tensor memory
        (
            self.num_acc_stage,
            self.num_ab_stage,
            self.num_c_stage,
            self.num_scale_stage,
            self.num_tile_stage,
        ) = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.sfa_dtype,
            self.sfb_dtype,
            self.scale_m_per_tile * self.scale_k_per_tile,
            self.scale_n_per_tile * self.scale_k_per_tile,
            self.num_smem_capacity,
            self.occupancy,
            self.large_row_bucket,
        )

        if self.shared_mma_completion:
            self.num_ab_stage = self.num_acc_stage

        if self.tma_scales:
            self.num_scale_stage = self.scale_k_total * self.bulk_scale_stages
        # Compute A/B/C/Scale shared memory layout
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.num_c_stage,
        )
        self.sfa_smem_layout_staged = cute.make_layout(
            (
                (self.scale_granularity_m, self.scale_m_per_tile),
                (self.scale_granularity_k, self.scale_k_per_tile),
                self.num_scale_stage,
            ),
            stride=(
                (0, self.scale_k_per_tile),
                (0, 1),
                self.scale_k_per_tile * self.scale_m_per_tile,
            ),
        )
        self.sfb_smem_layout_staged = cute.make_layout(
            (
                (self.scale_granularity_n, self.scale_n_per_tile),
                (self.scale_granularity_k, self.scale_k_per_tile),
                self.num_scale_stage,
            ),
            stride=(
                (0, self.scale_k_per_tile),
                (0, 1),
                self.scale_k_per_tile * self.scale_n_per_tile,
            ),
        )

        if self.tma_scales:
            self.sfa_smem_layout_staged = cute.make_layout(
                (
                    (1, self.scale_m_per_tile),
                    (128, 1),
                    (self.scale_k_total, self.bulk_scale_stages),
                ),
                stride=(
                    (0, self.scale_k_total),
                    (0, 0),
                    (1, self.scale_m_per_tile * self.scale_k_total),
                ),
            )
            self.sfb_smem_layout_staged = cute.make_layout(
                ((128, 1), (128, 1), (self.scale_k_total, self.bulk_scale_stages)),
                stride=((0, 0), (0, 0), (1, self.scale_b_stride)),
            )
        self.sfa_bulk_layout = cute.make_layout(
            (self.mma_tiler[0], self.scale_k_total), stride=(self.scale_k_total, 1)
        )
        self.sfb_bulk_layout = cute.make_layout(
            (1, self.scale_k_total), stride=(self.scale_k_total, 1)
        )

        # Compute the number of tensor memory allocation columns
        self.num_tmem_alloc_cols = 512

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        resident_ctas: cutlass.Constexpr,
        stream: cuda.CUstream,
        output: cute.Tensor,
        peers: tuple,
        ready: cute.Tensor,
        peer_ready: tuple,
        ack: cute.Tensor,
        peer_ack: tuple,
        rank: cutlass.Int32,
        input_source: cute.Tensor,
        input_scales: Union[cute.Tensor, None],
        input_peers: tuple,
        scale_peers: tuple,
        input_flags: cute.Tensor,
        peer_input_flags: tuple,
        input_multicast_ptr: cutlass.Int64,
        live_rows: cutlass.Int32,
        epilogue_op: cutlass.Constexpr,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes before smem/grid/tma computation
        - Setup TMA load/store atoms and tensors
        - Compute grid size with regard to hardware constraints
        - Define shared storage for kernel
        - Launch the kernel on the supplied stream

        :param a: Input tensor A
        :type a: cute.Tensor
        :param b: Input tensor B
        :type b: cute.Tensor
        :param c: Output tensor C
        :type c: cute.Tensor
        :param sfa: Scale factor tensor A
        :type sfa: cute.Tensor
        :param sfb: Scale factor tensor B
        :type sfb: cute.Tensor
        :param resident_ctas: Number of resident CTAs in the persistent grid
        :type resident_ctas: cutlass.Constexpr
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        :param live_rows: Maximum physical row count across owners for this call.
        :type live_rows: cutlass.Int32 runtime input; prepared strides stay fixed.
        :param epilogue_op: Elementwise output function; pass identity for no transform
        :type epilogue_op: cutlass.Constexpr
        :raises TypeError: If input data types are incompatible with the MMA instruction.
        """
        # Setup static attributes before smem/grid/tma computation
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b.element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.sfa_dtype: Type[cutlass.Numeric] = sfa.element_type
        self.sfb_dtype: Type[cutlass.Numeric] = sfb.element_type
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

        # Setup attributes that dependent on gemm inputs
        self.num_cta_flags = ready.shape[0] - 16
        self.input_phase_clear = (
            self.row_parallel
            and len(peer_ready) == 4
            and input_source.shape[1] == 12288
            and output.shape[1] == 7168
        )
        # Prepared buckets redistribute registers between accumulation and
        # data movement within the same per-CTA budget.
        self.num_regs_uniform_warps = 64
        self.num_regs_sched_warps = 64
        self.num_regs_acc_update_warps = 192 if self.split_accumulator else 256
        self.num_regs_epilogue_warps = 64 if self.split_accumulator else 128
        if cutlass.const_expr(self.input_phase_clear and peers[1].shape[0] == 256):
            # Leave more registers for control and data movement at this bucket.
            self.num_regs_acc_update_warps = 176
            self.num_regs_epilogue_warps = 80
            self.num_regs_uniform_warps = 80
            self.num_regs_sched_warps = 80
        if cutlass.const_expr(
            self.input_phase_clear and peers[1].shape[0] in (960, 1024)
        ):
            self.num_regs_acc_update_warps = 184
            self.num_regs_epilogue_warps = 72
            self.num_regs_uniform_warps = 72
            self.num_regs_sched_warps = 72
        # These prepared buckets publish partials from accumulator warps
        # while epilogue warps reduce them.
        self.concurrent_reduction = self.input_phase_clear and peers[1].shape[0] in (
            2048,
            2304,
            4096,
            4608,
        )
        if cutlass.const_expr(self.cluster_shape_mn != (1, 1)):
            assert self.concurrent_reduction
            assert resident_ctas % 2 == 0 and output.shape[1] % 256 == 0
        if cutlass.const_expr(self.concurrent_reduction):
            self.num_regs_acc_update_warps = 192
            self.num_regs_epilogue_warps = 64
            self.num_regs_uniform_warps = 64
            self.num_regs_sched_warps = 64
        self.vector_output = self.input_phase_clear and (
            peers[1].shape[0] == 256 or peers[1].shape[0] >= 640
        )
        self.skip_padded_tiles = self.input_phase_clear and peers[1].shape[0] > 2048
        self.cache_group_m = (
            min(4, peers[1].shape[0] // self.mma_tiler[0])
            if (
                (self.vector_output and peers[1].shape[0] >= 4096)
                or (
                    self.row_parallel
                    and len(peer_ready) == 2
                    and input_source.shape[1] == 12288
                    and output.shape[1] == 7168
                    and peers[1].shape[0] >= 4608
                )
            )
            else 0
        )
        self.large_row_bucket = self.row_parallel and peers[1].shape[0] >= 128
        # Multicast operands need completion from every consuming CTA.
        self.shared_mma_completion = (
            self.row_parallel
            and peers[1].shape[0] >= 160
            and self.cluster_shape_mn == (1, 1)
        )
        self.dynamic_gemm = self.shared_mma_completion and len(peer_ready) == 2
        self.skip_padded_tiles = self.skip_padded_tiles or (
            self.dynamic_gemm and self.cache_group_m > 0
        )
        self.tmem_before_scales = self.row_parallel and (
            peers[1].shape[0] >= 512
            or (self.input_phase_clear and peers[1].shape[0] == 256)
        )
        self.tile_owner_input = self.row_parallel and peers[1].shape[0] >= 512
        self.scale_k_total = sfa.shape[1]
        # Every TMA destination begins on a 128-byte boundary.
        self.scale_b_stride = (self.scale_k_total + 31) // 32 * 32
        self.bulk_scale_stages = 1
        # Prepared TP2 and TP4 buckets amortize one whole-tile scale transfer.
        self.tma_scales = self.row_parallel and (
            (
                (
                    peers[1].shape[0] >= 640
                    or peers[1].shape[0] == 576
                    or (self.input_phase_clear and peers[1].shape[0] == 256)
                )
                and len(peer_ready) == 4
                and self.scale_k_total == 24
            )
            or (
                peers[1].shape[0] >= 512
                and len(peer_ready) == 2
                and self.scale_k_total == 48
                and output.shape[1] == 7168
            )
        )
        self.independent_operand_stages = self.row_parallel and (
            (len(peer_ready) == 2 and peers[1].shape[0] >= 512)
            or (len(peer_ready) == 4 and 256 <= peers[1].shape[0] <= 1536)
        )
        if cutlass.const_expr(self.independent_operand_stages):
            self.shared_mma_completion = False
        self._setup_attributes()
        if cutlass.const_expr(self.dynamic_gemm):
            self.num_tile_stage = 1

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Setup TMA load for A
        a_op = self._get_tma_atom_kind(atom_thr_size, self.is_a_mcast)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
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
        b_op = self._get_tma_atom_kind(atom_thr_size, self.is_b_mcast)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
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

        # Setup TMA store for C
        tma_atom_c = None
        tma_tensor_c = None
        c_cta_v_layout = cute.composition(
            cute.make_identity_layout(c.shape), self.epi_tile
        )
        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            c,
            epi_smem_layout,
            c_cta_v_layout,
        )

        remote_atoms = ()
        remote_tensors = ()
        # Keep the whole MMA tile for the TMEM transfer, then split shared memory
        # at owner boundaries for remote stores.
        remote_epi_tile = (
            math.gcd(peers[1].shape[0], self.mma_tiler[0]),
            self.epi_tile[1],
        )
        remote_smem_layout = cute.composition(epi_smem_layout, remote_epi_tile)
        for peer in cutlass.range_constexpr(len(peer_ready)):
            remote = peers[peer + 2]
            remote_atom, remote_tensor = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                remote,
                remote_smem_layout,
                cute.composition(
                    cute.make_identity_layout(remote.shape), remote_epi_tile
                ),
            )
            remote_atoms = (*remote_atoms, remote_atom)
            remote_tensors = (*remote_tensors, remote_tensor)

        scale_atom_a = scale_atom_b = None
        scale_tensor_a = scale_tensor_b = None
        if cutlass.const_expr(self.tma_scales):
            scale_atom_a, scale_tensor_a = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                sfa,
                self.sfa_bulk_layout,
                cute.composition(
                    cute.make_identity_layout(sfa.shape),
                    (self.mma_tiler[0], self.scale_k_total),
                ),
            )
            scale_atom_b, scale_tensor_b = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                sfb,
                self.sfb_bulk_layout,
                cute.composition(
                    cute.make_identity_layout(sfb.shape), (1, self.scale_k_total)
                ),
            )

        tensor_sfa = cute.make_tensor(
            sfa.iterator,
            cute.make_layout(
                (
                    (self.scale_granularity_m, sfa.shape[0]),
                    (self.scale_granularity_k, sfa.shape[1]),
                    sfa.shape[2],
                ),
                stride=(
                    (0, sfa.layout.stride[0]),
                    (0, sfa.layout.stride[1]),
                    sfa.layout.stride[2],
                ),
            ),
        )
        tensor_sfb = cute.make_tensor(
            sfb.iterator,
            cute.make_layout(
                (
                    (self.scale_granularity_n, sfb.shape[0]),
                    (self.scale_granularity_k, sfb.shape[1]),
                    sfb.shape[2],
                ),
                stride=(
                    (0, sfb.layout.stride[0]),
                    (0, sfb.layout.stride[1]),
                    sfb.layout.stride[2],
                ),
            ),
        )

        # Compute grid size
        self.num_ctas = resident_ctas
        self.gemm_ctas = resident_ctas - (
            (4 if peers[1].shape[0] >= 4096 else 8)
            if self.input_phase_clear and peers[1].shape[0] >= 640
            else (4 if self.dynamic_gemm and self.cache_group_m > 0 else 0)
        )
        # These GEMM strides keep M fixed across successive output-channel tiles.
        if cutlass.const_expr(self.input_phase_clear and peers[1].shape[0] == 224):
            self.gemm_ctas = resident_ctas - 5
        if cutlass.const_expr(self.input_phase_clear and peers[1].shape[0] == 576):
            self.gemm_ctas = resident_ctas - 8
        if cutlass.const_expr(self.input_phase_clear and peers[1].shape[0] == 640):
            self.gemm_ctas = resident_ctas - 12
        if cutlass.const_expr(self.concurrent_reduction):
            self.gemm_ctas = resident_ctas
        # Keep roughly two reduction jobs per resident CTA without making
        # the row count itself a new specialization.
        reduction_rows_needed = (
            peers[1].shape[0] * (output.shape[1] // 128) + 2 * resident_ctas - 1
        ) // (2 * resident_ctas)
        self.reduction_rows = max(32, 1 << (reduction_rows_needed - 1).bit_length())
        if cutlass.const_expr(self.row_parallel and peers[1].shape[0] >= 512):
            self.reduction_rows = min(128, self.reduction_rows)
        if cutlass.const_expr(self.concurrent_reduction):
            self.reduction_rows = 32
        if cutlass.const_expr(self.input_phase_clear and peers[1].shape[0] == 256):
            # Wider channel tiles preserve the number of reduction jobs.
            self.reduction_rows = 32
        self.tile_sched_params, grid = self._compute_grid(
            c, self.cta_tile_shape_mnk, self.cluster_shape_mn, resident_ctas
        )

        self.buffer_align_bytes = 1024

        c_smem_size = cute.cosize(self.c_smem_layout_staged.outer)

        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            # (bidx, bidy, bidz, valid)
            sInfo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 4 * self.num_tile_stage],
                # 1 byte alignment
                1,
            ]
            reduce_job: cute.struct.MemRange[
                cutlass.Int32, 1 if self.concurrent_reduction else 0
            ]
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            scale_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64,
                (self.bulk_scale_stages if self.tma_scales else self.num_scale_stage)
                * 2,
            ]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tile_info_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_tile_stage * 2
            ]
            epi_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1 * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    c_smem_size,
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
            # (granularity_m, repeat_m), (granularity_k, repeat_k), num_scale_stage)
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sfa_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            # (granularity_n, repeat_n), (granularity_k, repeat_k), num_scale_stage)
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sfb_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel on the supplied stream
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            tensor_sfa,
            tensor_sfb,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            epilogue_op,
            output,
            peers,
            ready,
            peer_ready,
            ack,
            peer_ack,
            rank,
            input_source,
            input_scales,
            input_peers,
            scale_peers,
            input_flags,
            peer_input_flags,
            input_multicast_ptr,
            remote_atoms,
            remote_tensors,
            scale_atom_a,
            scale_atom_b,
            scale_tensor_a,
            scale_tensor_b,
            live_rows,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )
        return

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        mSFA_mkl: cute.Tensor,
        mSFB_nkl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
        output: cute.Tensor,
        peers: tuple,
        ready: cute.Tensor,
        peer_ready: tuple,
        ack: cute.Tensor,
        peer_ack: tuple,
        rank: cutlass.Int32,
        input_source: cute.Tensor,
        input_scales: Union[cute.Tensor, None],
        input_peers: tuple,
        scale_peers: tuple,
        input_flags: cute.Tensor,
        peer_input_flags: tuple,
        input_multicast_ptr: cutlass.Int64,
        remote_atoms: tuple,
        remote_tensors: tuple,
        scale_atom_a,
        scale_atom_b,
        scale_tensor_a,
        scale_tensor_b,
        live_rows: cutlass.Int32,
    ):
        """
        GPU device kernel performing the Persistent batched GEMM computation.
        """
        # The retired tile-info buffer broadcasts reduction jobs after all
        # GEMM roles have joined. Only CTA zero initializes the device queue.
        if cutlass.const_expr(self.row_parallel):
            if (
                cute.arch.block_idx()[0] == 0
                and cute.arch.block_idx()[1] == 0
                and cute.arch.block_idx()[2] == 0
                and cute.arch.thread_idx()[0] == 0
            ):
                queue_epoch = cute.arch.load(
                    input_flags.iterator + 16, cutlass.Int64, sem="acquire", scope="gpu"
                ) + cutlass.Int64(1)
                cute.arch.store(
                    ready.iterator + 1, cutlass.Int64(0), sem="relaxed", scope="gpu"
                )
                if cutlass.const_expr(self.dynamic_gemm):
                    cute.arch.store(
                        ready.iterator + 3,
                        cutlass.Int64(self.gemm_ctas),
                        sem="relaxed",
                        scope="gpu",
                    )
                cute.arch.store(
                    ready.iterator + 2, queue_epoch, sem="release", scope="gpu"
                )
        if cutlass.const_expr(not (self.large_row_bucket and len(peer_ready) == 2)):
            self.stage_input(
                input_source,
                input_scales,
                peers[1].shape[0],
                input_peers,
                scale_peers,
                input_flags,
                peer_input_flags,
                rank,
                peer_ack,
                input_multicast_ptr,
                peers[1],
            )
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        lane_idx = cute.arch.lane_idx()

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)

        # Overlap TP2 descriptor fetches with quantization in large row buckets.
        if cutlass.const_expr(self.large_row_bucket and len(peer_ready) == 2):
            self.stage_input(
                input_source,
                input_scales,
                peers[1].shape[0],
                input_peers,
                scale_peers,
                input_flags,
                peer_input_flags,
                rank,
                peer_ack,
                input_multicast_ptr,
                peers[1],
            )

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, bidy, bidz = cute.arch.block_idx()
        bidz = self.cta_index()
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

        #
        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Initialize mainloop ab_pipeline (barrier) and states
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Warp)
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            enable_multicast_signaling=True,
            defer_sync=True,
        )

        # Initialize mainloop scale_pipeline (barrier) and states
        scale_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * 1,
        )
        scale_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * len(self.acc_update_warp_id),
        )
        scale_pipeline = None
        if cutlass.const_expr(self.tma_scales):
            scale_pipeline = pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.scale_mbar_ptr.data_ptr(),
                num_stages=self.bulk_scale_stages,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=scale_pipeline_consumer_group,
                tx_count=(self.mma_tiler[0] + 1) * self.scale_k_total * 4,
                defer_sync=True,
                enable_multicast_signaling=True,
            )
        else:
            scale_pipeline = pipeline.PipelineCpAsync.create(
                barrier_storage=storage.scale_mbar_ptr.data_ptr(),
                num_stages=self.num_scale_stage,
                producer_group=scale_pipeline_producer_group,
                consumer_group=scale_pipeline_consumer_group,
                defer_sync=True,
            )

        # Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.acc_update_warp_id) * (
            2 if use_2cta_instrs else 1
        )
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Initialize epilogue pipeline (barrier) and states
        epi_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * len(self.acc_update_warp_id),
        )
        epi_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * len(self.epilog_warp_id),
        )
        epi_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.epi_mbar_ptr.data_ptr(),
            num_stages=1,
            producer_group=epi_pipeline_producer_group,
            consumer_group=epi_pipeline_consumer_group,
            defer_sync=True,
        )

        # Initialize tile info pipeline (barrier) and states
        tile_info_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * 1,
        )
        tile_info_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_wo_sched - (128 if self.concurrent_reduction else 0),
        )
        tile_info_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.tile_info_mbar_ptr.data_ptr(),
            num_stages=self.num_tile_stage,
            producer_group=tile_info_pipeline_producer_group,
            consumer_group=tile_info_pipeline_consumer_group,
            defer_sync=True,
        )

        # Tensor memory dealloc barrier init
        tmem = cutlass.memory.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        #
        # Setup smem tensor A/B/C/Scale
        #
        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        sC = storage.sC.get_tensor(
            c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
        )
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        # (granularity_m, repeat_m), (granularity_k, repeat_k), num_scale_stage)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        # (granularity_n, repeat_n), (granularity_k, repeat_k), num_scale_stage)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        # (bidx, bidy, bidz, valid)
        info_layout = cute.make_layout((4, self.num_tile_stage), stride=(1, 4))
        sInfo = storage.sInfo.get_tensor(info_layout)
        reduce_job = None
        if cutlass.const_expr(self.concurrent_reduction):
            reduce_job = storage.reduce_job.get_tensor(
                cute.make_layout((1, 1), stride=(1, 1))
            )

        #
        # Compute multicast mask for A/B buffer full
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        #
        # Local_tile partition global tensors
        #
        # (bM, bK, loopM, loopK, loopL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, loopN, loopK, loopL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bM, bN, loopM, loopN, loopL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        # (bM, bK, loopM, loopK, loopL)
        gSFA_mkl = cute.local_tile(
            mSFA_mkl,
            cute.slice_(self.cta_tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        # (bN, bK, loopN, loopK, loopL)
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            cute.slice_(self.cta_tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        # coordinate
        cSFA_mkl = cute.make_identity_tensor(cute.shape(mSFA_mkl))
        cSFB_nkl = cute.make_identity_tensor(cute.shape(mSFB_nkl))
        # (bM, bK, loopM, loopK, loopL)
        cSFA = cute.local_tile(
            cSFA_mkl,
            cute.slice_(self.cta_tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        # (bN, bK, loopN, loopK, loopL)
        cSFB = cute.local_tile(
            cSFB_nkl,
            cute.slice_(self.cta_tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        #
        # Partition global tensor for TiledMMA_A/B/C
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, loopM, loopK, loopL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, loopN, loopK, loopL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_N, loopM, loopN, loopL)
        tCgC = thr_mma.partition_C(gC_mnl)

        # scale viewed as C tensor
        sSFA_view_as_C_layout = cute.make_layout(
            (
                (self.scale_granularity_m, self.scale_m_per_tile),
                self.cta_tile_shape_mnk[1],
                (
                    (self.scale_k_total, self.bulk_scale_stages)
                    if self.tma_scales
                    else self.num_scale_stage
                ),
            ),
            stride=(
                (0, self.scale_k_total if self.tma_scales else 1),
                0,
                (
                    (1, self.scale_m_per_tile * self.scale_k_total)
                    if self.tma_scales
                    else self.scale_m_per_tile
                ),
            ),
        )
        sSFB_view_as_C_layout = cute.make_layout(
            (
                self.cta_tile_shape_mnk[0],
                (self.scale_granularity_n, self.scale_n_per_tile),
                (
                    (self.scale_k_total, self.bulk_scale_stages)
                    if self.tma_scales
                    else self.num_scale_stage
                ),
            ),
            stride=(
                0,
                (0, 1),
                (1, self.scale_b_stride) if self.tma_scales else self.scale_n_per_tile,
            ),
        )
        sSFA_view_as_C = cute.make_tensor(sSFA.iterator, sSFA_view_as_C_layout)
        sSFB_view_as_C = cute.make_tensor(sSFB.iterator, sSFB_view_as_C_layout)

        #
        # Partition global/shared tensor for TMA load A/B
        #
        # TMA load A partition_S/D
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), loopM, loopK, loopL)
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
        # ((atom_v, rest_v), loopM, loopK, loopL)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        #
        # Partition global/shared tensor for TMA load A/B
        #
        # load scaleA/scaleB
        atom_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mSFA_mkl.element_type,
            num_bits_per_copy=mSFA_mkl.element_type.width,
        )
        tiled_copy_sfa = cute.make_tiled_copy_tv(
            atom_copy, cute.make_layout((32,)), cute.make_layout((1,))
        )
        tiled_copy_sfb = cute.make_tiled_copy_tv(
            atom_copy, cute.make_layout((32,)), cute.make_layout((1,))
        )
        thr_copy_sfa = tiled_copy_sfa.get_slice(lane_idx)
        thr_copy_sfb = tiled_copy_sfb.get_slice(lane_idx)
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), loopM, loopK, loopL)
        tAgSFA_mkl = thr_copy_sfa.partition_S(gSFA_mkl)
        tAsSFA = thr_copy_sfa.partition_D(sSFA)
        tAcSFA = thr_copy_sfa.partition_S(cSFA)
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), loopN, loopK, loopL)
        tBgSFB_nkl = thr_copy_sfb.partition_S(gSFB_nkl)
        tBsSFB = thr_copy_sfb.partition_D(sSFB)
        tBcSFB = thr_copy_sfb.partition_S(cSFB)

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        #
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

        #
        # Cluster wait before tensor memory alloc
        #
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # Reserve a small initial group for reduction while the rest
        # compute. GEMM CTAs join the same queue when their tiles retire.
        early_reducer = cutlass.Boolean(False)
        if cutlass.const_expr(self.gemm_ctas < self.num_ctas):
            early_reducer = bidz >= self.gemm_ctas
        if early_reducer:
            # Tail clearing is distributed across the entire input grid.
            # Wait for this owner's published input epoch before polling any
            # payload, or an uncleared previous-call value could appear ready.
            if warp_idx == 0:
                self.await_tile_input_owners(
                    (
                        rank * peers[1].shape[0] // self.mma_tiler[0]
                        if cutlass.const_expr(peers[1].shape[0] in (576, 960))
                        else rank * (peers[1].shape[0] // self.mma_tiler[0])
                    ),
                    peers[1].shape[0],
                    input_flags,
                    peer_input_flags,
                    rank,
                    cutlass.Int32(0),
                )
            cute.arch.barrier()
            early_epoch = cute.arch.load(
                input_flags.iterator + 16 + bidz,
                cutlass.Int64,
                sem="acquire",
                scope="gpu",
            )
            early_copy = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.c_dtype,
                num_bits_per_copy=128,
            )
            self.reduce_output_inband(
                output, peers, ready, sInfo, early_epoch, early_copy, live_rows, False
            )
        else:
            #
            # Specialized Schedule warp
            #
            if warp_idx == self.sched_warp_id:
                cute.arch.setmaxregister_decrease(self.num_regs_sched_warps)
                if cutlass.const_expr(self.dynamic_gemm):
                    epoch = cute.arch.load(
                        input_flags.iterator + 16 + bidz,
                        cutlass.Int64,
                        sem="acquire",
                        scope="gpu",
                    )
                    if lane_idx == 0:
                        arrival = cute.arch.load(
                            ready.iterator + 2,
                            cutlass.Int64,
                            sem="acquire",
                            scope="gpu",
                        )
                        while arrival < epoch:
                            _queue_poll_backoff()
                            arrival = cute.arch.load(
                                ready.iterator + 2,
                                cutlass.Int64,
                                sem="acquire",
                                scope="gpu",
                            )
                    self.sched_sync_barrier.arrive_and_wait()
                    m_tiles = cute.ceil_div(
                        len(peer_ready) * peers[1].shape[0], self.cta_tile_shape_mnk[0]
                    )
                    n_tiles = output.shape[1] // self.cta_tile_shape_mnk[1]
                    tile_count = m_tiles * n_tiles
                    is_valid_tile = bidz < tile_count
                    tile_info_producer_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Producer, self.num_tile_stage
                    )
                    while is_valid_tile:
                        tile_info_pipeline.producer_acquire(tile_info_producer_state)
                        with cute.arch.elect_one():
                            next_tile = cute.arch.atomic_add(
                                ready.iterator + 3,
                                cutlass.Int64(1),
                                sem="relaxed",
                                scope="gpu",
                            ).to(cutlass.Int32)
                            next_coord = self.cache_tile_coord(
                                (
                                    next_tile % m_tiles,
                                    next_tile // m_tiles,
                                    cutlass.Int32(0),
                                ),
                                peers[1].shape[0],
                                len(peer_ready),
                                output.shape[1],
                            )
                            if cutlass.const_expr(self.skip_padded_tiles):
                                # Initial GEMM CTAs all lie in the first cache
                                # group, below the smallest live row bound for
                                # these prepared buckets. Only subsequent
                                # atomic tickets can reach whole padding tiles.
                                if live_rows <= peers[1].shape[0] - self.mma_tiler[0]:
                                    while (
                                        next_tile < tile_count
                                        and (
                                            next_coord[0]
                                            % (peers[1].shape[0] // self.mma_tiler[0])
                                        )
                                        * self.mma_tiler[0]
                                        >= live_rows
                                    ):
                                        next_tile = cute.arch.atomic_add(
                                            ready.iterator + 3,
                                            cutlass.Int64(1),
                                            sem="relaxed",
                                            scope="gpu",
                                        ).to(cutlass.Int32)
                                        next_coord = self.cache_tile_coord(
                                            (
                                                next_tile % m_tiles,
                                                next_tile // m_tiles,
                                                cutlass.Int32(0),
                                            ),
                                            peers[1].shape[0],
                                            len(peer_ready),
                                            output.shape[1],
                                        )
                            sInfo[(0, tile_info_producer_state.index)] = next_coord[0]
                            sInfo[(1, tile_info_producer_state.index)] = next_coord[1]
                            sInfo[(2, tile_info_producer_state.index)] = next_coord[2]
                            sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(
                                next_tile < tile_count
                            )
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.sched_sync_barrier.arrive_and_wait()
                        is_valid_tile = sInfo[(3, tile_info_producer_state.index)] == 1
                        tile_info_pipeline.producer_commit(tile_info_producer_state)
                        tile_info_producer_state.advance()
                    tile_info_pipeline.producer_tail(tile_info_producer_state)
                else:
                    #
                    # Persistent tile scheduling loop
                    #
                    tile_sched = utils.StaticPersistentTileScheduler.create(
                        tile_sched_params,
                        cute.arch.block_idx(),
                        (
                            cute.arch.grid_dim()
                            if self.gemm_ctas == self.num_ctas
                            else (
                                cutlass.Int32(1),
                                cutlass.Int32(1),
                                cutlass.Int32(self.gemm_ctas),
                            )
                        ),
                    )
                    # First tile
                    work_tile = tile_sched.initial_work_tile_info()
                    if cutlass.const_expr(self.skip_padded_tiles):
                        if live_rows <= peers[1].shape[0] - self.mma_tiler[0]:
                            live_coord = self.cache_tile_coord(
                                work_tile.tile_idx,
                                peers[1].shape[0],
                                len(peer_ready),
                                output.shape[1],
                            )
                            while (
                                work_tile.is_valid_tile
                                and (
                                    live_coord[0]
                                    % (peers[1].shape[0] // self.mma_tiler[0])
                                )
                                * self.mma_tiler[0]
                                >= live_rows
                            ):
                                tile_sched.advance_to_next_work()
                                work_tile = tile_sched.get_current_work()
                                live_coord = self.cache_tile_coord(
                                    work_tile.tile_idx,
                                    peers[1].shape[0],
                                    len(peer_ready),
                                    output.shape[1],
                                )

                    tile_info_producer_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Producer, self.num_tile_stage
                    )

                    while work_tile.is_valid_tile:
                        # query next tile
                        tile_sched.advance_to_next_work()
                        work_tile = tile_sched.get_current_work()
                        if cutlass.const_expr(self.skip_padded_tiles):
                            if live_rows <= peers[1].shape[0] - self.mma_tiler[0]:
                                live_coord = self.cache_tile_coord(
                                    work_tile.tile_idx,
                                    peers[1].shape[0],
                                    len(peer_ready),
                                    output.shape[1],
                                )
                                while (
                                    work_tile.is_valid_tile
                                    and (
                                        live_coord[0]
                                        % (peers[1].shape[0] // self.mma_tiler[0])
                                    )
                                    * self.mma_tiler[0]
                                    >= live_rows
                                ):
                                    tile_sched.advance_to_next_work()
                                    work_tile = tile_sched.get_current_work()
                                    live_coord = self.cache_tile_coord(
                                        work_tile.tile_idx,
                                        peers[1].shape[0],
                                        len(peer_ready),
                                        output.shape[1],
                                    )

                        # acquire tile info pipeline
                        tile_info_pipeline.producer_acquire(tile_info_producer_state)

                        # store the tile info
                        cur_tile_coord = self.cache_tile_coord(
                            work_tile.tile_idx,
                            peers[1].shape[0],
                            len(peer_ready),
                            output.shape[1],
                        )
                        with cute.arch.elect_one():
                            sInfo[(0, tile_info_producer_state.index)] = cur_tile_coord[
                                0
                            ]
                            sInfo[(1, tile_info_producer_state.index)] = cur_tile_coord[
                                1
                            ]
                            sInfo[(2, tile_info_producer_state.index)] = cur_tile_coord[
                                2
                            ]
                            sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(
                                work_tile.is_valid_tile
                            )

                        # fence view async shared
                        cute.arch.fence_proxy(
                            "async.shared",
                            space="cta",
                        )
                        self.sched_sync_barrier.arrive_and_wait()
                        # commit tile info pipeline
                        tile_info_pipeline.producer_commit(tile_info_producer_state)
                        tile_info_producer_state.advance()

                    tile_info_pipeline.producer_tail(tile_info_producer_state)

            #
            # Specialized TMA load warp
            #
            if warp_idx == self.tma_warp_id:
                cute.arch.setmaxregister_decrease(self.num_regs_uniform_warps)
                #
                # Persistent tile scheduling loop
                #
                tile_sched = utils.StaticPersistentTileScheduler.create(
                    tile_sched_params,
                    cute.arch.block_idx(),
                    (
                        cute.arch.grid_dim()
                        if self.gemm_ctas == self.num_ctas
                        else (
                            cutlass.Int32(1),
                            cutlass.Int32(1),
                            cutlass.Int32(self.gemm_ctas),
                        )
                    ),
                )
                # First tile
                work_tile = tile_sched.initial_work_tile_info()
                if cutlass.const_expr(self.skip_padded_tiles):
                    if live_rows <= peers[1].shape[0] - self.mma_tiler[0]:
                        live_coord = self.cache_tile_coord(
                            work_tile.tile_idx,
                            peers[1].shape[0],
                            len(peer_ready),
                            output.shape[1],
                        )
                        while (
                            work_tile.is_valid_tile
                            and (
                                live_coord[0] % (peers[1].shape[0] // self.mma_tiler[0])
                            )
                            * self.mma_tiler[0]
                            >= live_rows
                        ):
                            tile_sched.advance_to_next_work()
                            work_tile = tile_sched.get_current_work()
                            live_coord = self.cache_tile_coord(
                                work_tile.tile_idx,
                                peers[1].shape[0],
                                len(peer_ready),
                                output.shape[1],
                            )

                ab_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_ab_stage
                )

                tile_info_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_tile_stage
                )

                # get the first tile info
                tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
                # Get tile coord from tile scheduler
                cur_tile_coord = self.cache_tile_coord(
                    work_tile.tile_idx,
                    peers[1].shape[0],
                    len(peer_ready),
                    output.shape[1],
                )
                # initialize the tile info
                tile_info[0] = cur_tile_coord[0]
                tile_info[1] = cur_tile_coord[1]
                tile_info[2] = cur_tile_coord[2]
                tile_info[3] = work_tile.is_valid_tile

                is_valid_tile = cutlass.Boolean(1)
                is_valid_tile = tile_info[3] == 1

                # Larger buckets can move this CTA between owners during the loop.
                ready_input_owners = cutlass.Int32(0)
                if cutlass.const_expr(not self.tile_owner_input):
                    self.await_input_owner(
                        tile_info[0],
                        peers[1].shape[0],
                        input_flags,
                        peer_input_flags,
                        rank,
                    )

                ab_reuse_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_acc_stage
                )
                ab_issued = cutlass.Int32(0)
                # Keep reused weight tiles in L2 during the grouped traversal.
                b_cache_policy = None
                if cutlass.const_expr(
                    self.input_phase_clear and peers[1].shape[0] >= 4096
                ):
                    b_cache_policy = _tma_l2_policy(keep=True)
                while is_valid_tile:
                    if cutlass.const_expr(self.tile_owner_input):
                        ready_input_owners = self.await_tile_input_owners(
                            tile_info[0],
                            peers[1].shape[0],
                            input_flags,
                            peer_input_flags,
                            rank,
                            ready_input_owners,
                        )
                    mma_tile_coord_mnl = (
                        tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                        tile_info[1],
                        tile_info[2],
                    )
                    #
                    # Slice to per mma tile index
                    #
                    # ((atom_v, rest_v), loopK)
                    tAgA_slice = tAgA[
                        (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                    ]
                    # ((atom_v, rest_v), loopK)
                    tBgB_slice = tBgB[
                        (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                    ]

                    # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt
                    ab_producer_state.reset_count()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if (
                        cutlass.const_expr(not self.shared_mma_completion)
                        and ab_producer_state.count < k_tile_cnt
                    ):
                        if cutlass.const_expr(self.independent_operand_stages):
                            peek_ab_empty_status = (
                                ab_pipeline.sync_object_empty.test_wait(
                                    ab_producer_state.index, ab_producer_state.phase
                                )
                            )
                        else:
                            peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                                ab_producer_state
                            )
                    #
                    # Tma load loop
                    #
                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        tAgA_k = tAgA_slice[(None, ab_producer_state.count)]
                        tBgB_k = tBgB_slice[(None, ab_producer_state.count)]
                        tAsA_pipe = tAsA[(None, ab_producer_state.index)]
                        tBsB_pipe = tBsB[(None, ab_producer_state.index)]

                        tma_bar = ab_pipeline.producer_get_barrier(ab_producer_state)

                        # Conditionally wait for AB buffer empty
                        if cutlass.const_expr(self.shared_mma_completion):
                            # The accumulator completion also proves MMA has
                            # finished reading this A/B stage. Reuse that signal.
                            if ab_issued >= self.num_ab_stage:
                                acc_pipeline.consumer_wait(ab_reuse_state)
                                ab_reuse_state.advance()
                            ab_pipeline.producer_acquire(
                                ab_producer_state, cutlass.Boolean(True)
                            )
                            ab_issued += 1
                        else:
                            ab_pipeline.producer_acquire(
                                ab_producer_state, peek_ab_empty_status
                            )

                        # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt + k_tile + 1
                        ab_producer_state.advance()
                        peek_ab_empty_status = cutlass.Boolean(1)
                        if (
                            cutlass.const_expr(not self.shared_mma_completion)
                            and ab_producer_state.count < k_tile_cnt
                        ):
                            if cutlass.const_expr(self.independent_operand_stages):
                                peek_ab_empty_status = (
                                    ab_pipeline.sync_object_empty.test_wait(
                                        ab_producer_state.index, ab_producer_state.phase
                                    )
                                )
                            else:
                                peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                                    ab_producer_state
                                )

                        # TMA load A/B
                        cute.copy(
                            tma_atom_a,
                            tAgA_k,
                            tAsA_pipe,
                            tma_bar_ptr=tma_bar,
                            mcast_mask=a_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_b,
                            tBgB_k,
                            tBsB_pipe,
                            tma_bar_ptr=tma_bar,
                            mcast_mask=b_full_mcast_mask,
                            cache_policy=b_cache_policy,
                        )

                    #
                    # Advance to next tile
                    #
                    tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                    for idx in cutlass.range(4, unroll_full=True):
                        tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                    is_valid_tile = tile_info[3] == 1
                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    tile_info_pipeline.consumer_release(tile_info_consumer_state)
                    tile_info_consumer_state.advance()

                #
                # Wait A/B buffer empty
                #
                if cutlass.const_expr(not self.shared_mma_completion):
                    ab_pipeline.producer_tail(ab_producer_state)

            #
            # Specialized Scale load warp
            #
            if warp_idx == self.scale_warp_id:
                cute.arch.setmaxregister_decrease(self.num_regs_uniform_warps)
                #
                # Persistent tile scheduling loop
                #
                tile_sched = utils.StaticPersistentTileScheduler.create(
                    tile_sched_params,
                    cute.arch.block_idx(),
                    (
                        cute.arch.grid_dim()
                        if self.gemm_ctas == self.num_ctas
                        else (
                            cutlass.Int32(1),
                            cutlass.Int32(1),
                            cutlass.Int32(self.gemm_ctas),
                        )
                    ),
                )
                # First tile
                work_tile = tile_sched.initial_work_tile_info()
                if cutlass.const_expr(self.skip_padded_tiles):
                    if live_rows <= peers[1].shape[0] - self.mma_tiler[0]:
                        live_coord = self.cache_tile_coord(
                            work_tile.tile_idx,
                            peers[1].shape[0],
                            len(peer_ready),
                            output.shape[1],
                        )
                        while (
                            work_tile.is_valid_tile
                            and (
                                live_coord[0] % (peers[1].shape[0] // self.mma_tiler[0])
                            )
                            * self.mma_tiler[0]
                            >= live_rows
                        ):
                            tile_sched.advance_to_next_work()
                            work_tile = tile_sched.get_current_work()
                            live_coord = self.cache_tile_coord(
                                work_tile.tile_idx,
                                peers[1].shape[0],
                                len(peer_ready),
                                output.shape[1],
                            )

                scale_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer,
                    self.bulk_scale_stages if self.tma_scales else self.num_scale_stage,
                )

                tile_info_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_tile_stage
                )

                # get the first tile info
                tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
                # Get tile coord from tile scheduler
                cur_tile_coord = self.cache_tile_coord(
                    work_tile.tile_idx,
                    peers[1].shape[0],
                    len(peer_ready),
                    output.shape[1],
                )
                # initialize the tile info
                tile_info[0] = cur_tile_coord[0]
                tile_info[1] = cur_tile_coord[1]
                tile_info[2] = cur_tile_coord[2]
                tile_info[3] = work_tile.is_valid_tile

                is_valid_tile = cutlass.Boolean(1)
                is_valid_tile = tile_info[3] == 1

                # Larger buckets can move this CTA between owners during the loop.
                ready_input_owners = cutlass.Int32(0)
                if cutlass.const_expr(not self.tile_owner_input):
                    self.await_input_owner(
                        tile_info[0],
                        peers[1].shape[0],
                        input_flags,
                        peer_input_flags,
                        rank,
                    )

                while is_valid_tile:
                    if cutlass.const_expr(self.tile_owner_input):
                        ready_input_owners = self.await_tile_input_owners(
                            tile_info[0],
                            peers[1].shape[0],
                            input_flags,
                            peer_input_flags,
                            rank,
                            ready_input_owners,
                        )
                    if cutlass.const_expr(self.tma_scales):
                        scale_pipeline.producer_acquire(scale_producer_state)
                        barrier = scale_pipeline.producer_get_barrier(
                            scale_producer_state
                        )
                        ga = cute.local_tile(
                            scale_tensor_a,
                            (self.mma_tiler[0], self.scale_k_total),
                            (tile_info[0], 0, 0),
                        )
                        gb = cute.local_tile(
                            scale_tensor_b,
                            (1, self.scale_k_total),
                            (tile_info[1], 0, 0),
                        )
                        sa = cute.make_tensor(
                            sSFA.iterator
                            + scale_producer_state.index
                            * self.mma_tiler[0]
                            * self.scale_k_total,
                            cute.make_layout(
                                (self.mma_tiler[0], self.scale_k_total),
                                stride=(self.scale_k_total, 1),
                            ),
                        )
                        sb = cute.make_tensor(
                            sSFB.iterator
                            + scale_producer_state.index * self.scale_b_stride,
                            cute.make_layout(
                                (1, self.scale_k_total), stride=(self.scale_k_total, 1)
                            ),
                        )
                        psa, pga = cpasync.tma_partition(
                            scale_atom_a,
                            0,
                            cute.make_layout(1),
                            cute.group_modes(sa, 0, 2),
                            cute.group_modes(ga, 0, 2),
                        )
                        psb, pgb = cpasync.tma_partition(
                            scale_atom_b,
                            0,
                            cute.make_layout(1),
                            cute.group_modes(sb, 0, 2),
                            cute.group_modes(gb, 0, 2),
                        )
                        cute.copy(scale_atom_a, pga, psa, tma_bar_ptr=barrier)
                        cute.copy(scale_atom_b, pgb, psb, tma_bar_ptr=barrier)
                        scale_producer_state.advance()
                    else:
                        #
                        # Prepare the mask for scaleA/scaleB
                        #
                        tApSFA = cute.make_rmem_tensor(
                            cute.make_layout(
                                cute.filter_zeros(
                                    cute.slice_(tAsSFA, (None, None, None, 0))
                                ).shape
                            ),
                            cutlass.Boolean,
                        )
                        tBpSFB = cute.make_rmem_tensor(
                            cute.make_layout(
                                cute.filter_zeros(
                                    cute.slice_(tBsSFB, (None, None, None, 0))
                                ).shape
                            ),
                            cutlass.Boolean,
                        )

                        # Peek (try_wait) SCALE buffer empty
                        scale_producer_state.reset_count()
                        peek_scale_empty_status = cutlass.Boolean(1)
                        if scale_producer_state.count < k_tile_cnt:
                            peek_scale_empty_status = (
                                scale_pipeline.producer_try_acquire(
                                    scale_producer_state
                                )
                            )

                        #
                        # load loop
                        #
                        for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                            #
                            # Slice to per mma tile index
                            #
                            tAsSFA_pipe = cute.filter_zeros(
                                tAsSFA[(None, None, None, scale_producer_state.index)]
                            )
                            tBsSFB_pipe = cute.filter_zeros(
                                tBsSFB[(None, None, None, scale_producer_state.index)]
                            )
                            tAgSFA_k = cute.filter_zeros(
                                tAgSFA_mkl[
                                    (
                                        None,
                                        None,
                                        None,
                                        tile_info[0],
                                        scale_producer_state.count,
                                        tile_info[2],
                                    )
                                ]
                            )
                            tBgSFB_k = cute.filter_zeros(
                                tBgSFB_nkl[
                                    (
                                        None,
                                        None,
                                        None,
                                        tile_info[1],
                                        scale_producer_state.count,
                                        tile_info[2],
                                    )
                                ]
                            )

                            tAcSFA_compact = cute.filter_zeros(
                                cute.slice_(
                                    tAcSFA,
                                    (
                                        None,
                                        None,
                                        None,
                                        tile_info[0],
                                        scale_producer_state.count,
                                        tile_info[2],
                                    ),
                                )
                            )
                            tBcSFB_compact = cute.filter_zeros(
                                cute.slice_(
                                    tBcSFB,
                                    (
                                        None,
                                        None,
                                        None,
                                        tile_info[1],
                                        scale_producer_state.count,
                                        tile_info[2],
                                    ),
                                )
                            )
                            for i in cutlass.range_constexpr(
                                cute.size(tApSFA, mode=[1])
                            ):
                                tApSFA[((0, 0), i, (0, 0))] = cute.elem_less(
                                    tAcSFA_compact[(i)][0], mSFA_mkl.shape[0]
                                )
                            for i in cutlass.range_constexpr(
                                cute.size(tBpSFB, mode=[1])
                            ):
                                tBpSFB[((0, 0), i, (0, 0))] = cute.elem_less(
                                    tBcSFB_compact[(i)][0], mSFB_nkl.shape[0]
                                )

                            # Conditionally wait for Scale buffer empty
                            scale_pipeline.producer_acquire(
                                scale_producer_state, peek_scale_empty_status
                            )

                            # load scaleA/scaleB
                            cute.copy(
                                tiled_copy_sfa, tAgSFA_k, tAsSFA_pipe, pred=tApSFA
                            )
                            cute.copy(
                                tiled_copy_sfb, tBgSFB_k, tBsSFB_pipe, pred=tBpSFB
                            )

                            scale_pipeline.producer_commit(scale_producer_state)

                            # Peek (try_wait) Scale buffer empty
                            scale_producer_state.advance()
                            peek_scale_empty_status = cutlass.Boolean(1)
                            if scale_producer_state.count < k_tile_cnt:
                                peek_scale_empty_status = (
                                    scale_pipeline.producer_try_acquire(
                                        scale_producer_state
                                    )
                                )

                    #
                    # Advance to next tile
                    #
                    tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                    for idx in cutlass.range(4, unroll_full=True):
                        tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                    is_valid_tile = tile_info[3] == 1
                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    tile_info_pipeline.consumer_release(tile_info_consumer_state)
                    tile_info_consumer_state.advance()

                #
                # Wait Scale buffer empty
                #
                scale_pipeline.producer_tail(scale_producer_state)

            #
            # Specialized MMA warp
            #
            if warp_idx == self.mma_warp_id:
                cute.arch.setmaxregister_decrease(self.num_regs_uniform_warps)
                #
                # Bar sync for retrieve tensor memory ptr from shared mem
                #
                tmem.wait_for_alloc()

                #
                # Retrieving tensor memory ptr and make accumulator tensor
                #
                tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
                # (MMA, MMA_M, MMA_N, STAGE)
                tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

                #
                # Persistent tile scheduling loop
                #
                tile_sched = utils.StaticPersistentTileScheduler.create(
                    tile_sched_params,
                    cute.arch.block_idx(),
                    (
                        cute.arch.grid_dim()
                        if self.gemm_ctas == self.num_ctas
                        else (
                            cutlass.Int32(1),
                            cutlass.Int32(1),
                            cutlass.Int32(self.gemm_ctas),
                        )
                    ),
                )
                work_tile = tile_sched.initial_work_tile_info()
                if cutlass.const_expr(self.skip_padded_tiles):
                    if live_rows <= peers[1].shape[0] - self.mma_tiler[0]:
                        live_coord = self.cache_tile_coord(
                            work_tile.tile_idx,
                            peers[1].shape[0],
                            len(peer_ready),
                            output.shape[1],
                        )
                        while (
                            work_tile.is_valid_tile
                            and (
                                live_coord[0] % (peers[1].shape[0] // self.mma_tiler[0])
                            )
                            * self.mma_tiler[0]
                            >= live_rows
                        ):
                            tile_sched.advance_to_next_work()
                            work_tile = tile_sched.get_current_work()
                            live_coord = self.cache_tile_coord(
                                work_tile.tile_idx,
                                peers[1].shape[0],
                                len(peer_ready),
                                output.shape[1],
                            )

                ab_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_ab_stage
                )
                acc_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_acc_stage
                )

                tile_info_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_tile_stage
                )

                # get the first tile info
                tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
                # Get tile coord from tile scheduler
                cur_tile_coord = self.cache_tile_coord(
                    work_tile.tile_idx,
                    peers[1].shape[0],
                    len(peer_ready),
                    output.shape[1],
                )
                # initialize the tile info
                tile_info[0] = cur_tile_coord[0]
                tile_info[1] = cur_tile_coord[1]
                tile_info[2] = cur_tile_coord[2]
                tile_info[3] = work_tile.is_valid_tile

                is_valid_tile = cutlass.Boolean(1)
                is_valid_tile = tile_info[3] == 1

                while is_valid_tile:
                    # Peek (try_wait) AB buffer full for k_tile = 0
                    ab_consumer_state.reset_count()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                        if cutlass.const_expr(self.row_parallel):
                            # A peek must not suspend before current MMA completion
                            # is published to the accumulator warps.
                            peek_ab_full_status = (
                                ab_pipeline.sync_object_full.test_wait(
                                    ab_consumer_state.index, ab_consumer_state.phase
                                )
                            )
                        else:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(
                                ab_consumer_state
                            )

                    # Peek (try_wait) Acc buffer empty for k_tile = 0
                    acc_producer_state.reset_count()
                    peek_acc_empty_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                        peek_acc_empty_status = acc_pipeline.producer_try_acquire(
                            acc_producer_state
                        )

                    #
                    # Mma mainloop
                    #
                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        # Set tensor memory buffer for current tile
                        # (MMA, MMA_M, MMA_N)
                        tCtAcc = tCtAcc_base[
                            (None, None, None, acc_producer_state.index)
                        ]

                        #
                        # Wait for accumulator buffer empty
                        #
                        if is_leader_cta:
                            acc_pipeline.producer_acquire(
                                acc_producer_state, peek_acc_empty_status
                            )

                        #
                        # Reset the ACCUMULATE field for each tile
                        #
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                        if is_leader_cta:
                            # Conditionally wait for AB buffer full
                            ab_pipeline.consumer_wait(
                                ab_consumer_state, peek_ab_full_status
                            )

                            # tCtAcc += tCrA * tCrB
                            tile_crd = (None, None, None, ab_consumer_state.index)
                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[tile_crd],
                                tCrB[tile_crd],
                                tCtAcc,
                            )

                            # Async arrive AB buffer empty
                            if cutlass.const_expr(not self.shared_mma_completion):
                                ab_pipeline.consumer_release(ab_consumer_state)

                        # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
                        ab_consumer_state.advance()
                        peek_ab_full_status = cutlass.Boolean(1)
                        if ab_consumer_state.count < k_tile_cnt:
                            if is_leader_cta:
                                if cutlass.const_expr(self.row_parallel):
                                    # A peek must not suspend before current MMA completion
                                    # is published to the accumulator warps.
                                    peek_ab_full_status = (
                                        ab_pipeline.sync_object_full.test_wait(
                                            ab_consumer_state.index,
                                            ab_consumer_state.phase,
                                        )
                                    )
                                else:
                                    peek_ab_full_status = ab_pipeline.consumer_try_wait(
                                        ab_consumer_state
                                    )

                        #
                        # Async arrive accumulator buffer full(each kblock)
                        #
                        if is_leader_cta:
                            acc_pipeline.producer_commit(acc_producer_state)

                        # Peek (try_wait) Acc buffer empty for k_tile = k_tile + 1
                        acc_producer_state.advance()
                        if acc_producer_state.count < k_tile_cnt:
                            if is_leader_cta:
                                peek_acc_empty_status = (
                                    acc_pipeline.producer_try_acquire(
                                        acc_producer_state
                                    )
                                )

                    #
                    # Advance to next tile
                    #
                    tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                    for idx in cutlass.range(4, unroll_full=True):
                        tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                    is_valid_tile = tile_info[3] == 1
                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    tile_info_pipeline.consumer_release(tile_info_consumer_state)
                    tile_info_consumer_state.advance()

                #
                # Wait for accumulator buffer empty
                #
                acc_pipeline.producer_tail(acc_producer_state)

            #
            # Specialized acc update warps
            #
            if warp_idx <= self.acc_update_warp_id[-1]:
                cute.arch.setmaxregister_increase(self.num_regs_acc_update_warps)
                #
                # Bar sync for retrieve tensor memory ptr from shared memory
                #
                tmem.wait_for_alloc()

                #
                # Retrieving tensor memory ptr and make accumulator tensor
                #
                tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
                # (MMA, MMA_M, MMA_N, STAGE)
                tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
                tCtAcc_final = cute.make_tensor(
                    tCtAcc_base.iterator + self.tmem_final_offset, tCtAcc_base.layout
                )

                #
                # Partition for epilogue
                #
                epi_tidx = tidx % 128
                (
                    tiled_copy_t2r,
                    tiled_copy_r2t,
                    tTR_tAcc_base,
                    tTR_rAcc,
                    tTR_rAcc_final,
                    tTR_sSFA,
                    tTR_sSFB,
                    tRT_rAcc,
                    tRT_tAcc_base,
                ) = self.acc_update_tmem_copy_and_partition(
                    epi_tidx,
                    tCtAcc_base,
                    tCtAcc_final,
                    tCgC,
                    sSFA_view_as_C,
                    sSFB_view_as_C,
                    (self.mma_tiler[0], 32),
                )

                # Row epilogues consume the rounded tile directly from shared memory.
                if cutlass.const_expr(self.row_parallel):
                    handoff_smem = cute.flat_divide(sC, (self.mma_tiler[0], 32))
                    handoff_smem = cute.group_modes(
                        handoff_smem, 2, cute.rank(handoff_smem)
                    )
                    handoff_values = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
                    handoff_copy, handoff_regs, handoff_shared = (
                        self.epilog_smem_copy_and_partition(
                            tiled_copy_t2r, handoff_values, epi_tidx, handoff_smem
                        )
                    )

                #
                # Persistent tile scheduling loop
                #
                tile_sched = utils.StaticPersistentTileScheduler.create(
                    tile_sched_params,
                    cute.arch.block_idx(),
                    (
                        cute.arch.grid_dim()
                        if self.gemm_ctas == self.num_ctas
                        else (
                            cutlass.Int32(1),
                            cutlass.Int32(1),
                            cutlass.Int32(self.gemm_ctas),
                        )
                    ),
                )
                work_tile = tile_sched.initial_work_tile_info()
                if cutlass.const_expr(self.skip_padded_tiles):
                    if live_rows <= peers[1].shape[0] - self.mma_tiler[0]:
                        live_coord = self.cache_tile_coord(
                            work_tile.tile_idx,
                            peers[1].shape[0],
                            len(peer_ready),
                            output.shape[1],
                        )
                        while (
                            work_tile.is_valid_tile
                            and (
                                live_coord[0] % (peers[1].shape[0] // self.mma_tiler[0])
                            )
                            * self.mma_tiler[0]
                            >= live_rows
                        ):
                            tile_sched.advance_to_next_work()
                            work_tile = tile_sched.get_current_work()
                            live_coord = self.cache_tile_coord(
                                work_tile.tile_idx,
                                peers[1].shape[0],
                                len(peer_ready),
                                output.shape[1],
                            )

                acc_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_acc_stage
                )

                scale_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer,
                    self.bulk_scale_stages if self.tma_scales else self.num_scale_stage,
                )

                epi_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, 1
                )

                tile_info_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_tile_stage
                )

                # get the first tile info
                tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
                # Get tile coord from tile scheduler
                cur_tile_coord = self.cache_tile_coord(
                    work_tile.tile_idx,
                    peers[1].shape[0],
                    len(peer_ready),
                    output.shape[1],
                )
                # initialize the tile info
                tile_info[0] = cur_tile_coord[0]
                tile_info[1] = cur_tile_coord[1]
                tile_info[2] = cur_tile_coord[2]
                tile_info[3] = work_tile.is_valid_tile

                is_valid_tile = cutlass.Boolean(1)
                is_valid_tile = tile_info[3] == 1

                if cutlass.const_expr(self.concurrent_reduction):
                    # Every destination must acknowledge the previous call
                    # before accumulator warps write new partials.
                    if warp_idx == 0:
                        previous_epoch = cute.arch.load(
                            ready.iterator + 16 + bidz,
                            cutlass.Int64,
                            sem="relaxed",
                            scope="gpu",
                        )
                        address = peer_ack[0].iterator
                        for peer in cutlass.range_constexpr(len(peer_ack)):
                            if lane_idx == peer:
                                address = peer_ack[peer].iterator
                        if lane_idx > 0 and lane_idx < len(peer_ack):
                            arrival = cute.arch.load(
                                address, cutlass.Int64, sem="acquire", scope="sys"
                            )
                            while arrival < previous_epoch:
                                arrival = cute.arch.load(
                                    address, cutlass.Int64, sem="acquire", scope="sys"
                                )
                    self.acc_store_sync_barrier.arrive_and_wait()

                while is_valid_tile:
                    # initialize the final accumulator
                    tTR_rAcc_final.fill(0.0)

                    tTR_rSFA = cute.make_rmem_tensor(
                        cute.slice_(tTR_sSFA, (None, None, None, 0, None, 0)).shape,
                        self.acc_dtype,
                    )
                    tTR_rSFB = cute.make_rmem_tensor(
                        cute.slice_(tTR_sSFB, (None, None, None, 0, None, 0)).shape,
                        self.acc_dtype,
                    )

                    scale_consumer_state.reset_count()
                    peek_scale_full_status = cutlass.Boolean(1)
                    if (
                        cutlass.const_expr(not self.tma_scales)
                        and scale_consumer_state.count < k_tile_cnt
                    ):
                        peek_scale_full_status = scale_pipeline.consumer_try_wait(
                            scale_consumer_state
                        )

                    acc_consumer_state.reset_count()
                    peek_acc_full_status = cutlass.Boolean(1)
                    if acc_consumer_state.count < k_tile_cnt:
                        peek_acc_full_status = acc_pipeline.consumer_try_wait(
                            acc_consumer_state
                        )

                    if cutlass.const_expr(self.tma_scales):
                        scale_pipeline.consumer_wait(scale_consumer_state)
                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        # Set tensor memory buffer for current tile
                        # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                        tTR_tAcc = tTR_tAcc_base[
                            (None, None, None, None, None, acc_consumer_state.index)
                        ]

                        if cutlass.const_expr(self.tmem_before_scales):
                            # Release this K block's TMEM before waiting for its
                            # scales, allowing subsequent MMA work to proceed.
                            acc_pipeline.consumer_wait(
                                acc_consumer_state, peek_acc_full_status
                            )
                            tTR_tAcc = cute.group_modes(
                                tTR_tAcc, 3, cute.rank(tTR_tAcc)
                            )
                            current_partial = cute.make_rmem_tensor(
                                tTR_rAcc_final.shape, self.acc_dtype
                            )
                            for preload_idx in cutlass.range_constexpr(
                                cute.size(tTR_tAcc.shape, mode=[3])
                            ):
                                cute.copy(
                                    tiled_copy_t2r,
                                    tTR_tAcc[(None, None, None, preload_idx)],
                                    current_partial[(None, None, None, preload_idx)],
                                )
                            cute.arch.fence_view_async_tmem_load()
                            with cute.arch.elect_one():
                                acc_pipeline.consumer_release(acc_consumer_state)

                        #
                        # Wait for scale buffer full
                        #
                        if cutlass.const_expr(not self.tma_scales):
                            scale_pipeline.consumer_wait(
                                scale_consumer_state, peek_scale_full_status
                            )

                        tTR_sSFA_slice = cute.slice_(
                            tTR_sSFA,
                            (
                                None,
                                None,
                                None,
                                0,
                                None,
                                (
                                    (k_tile, scale_consumer_state.index)
                                    if self.tma_scales
                                    else scale_consumer_state.index
                                ),
                            ),
                        )
                        tTR_sSFB_slice = cute.slice_(
                            tTR_sSFB,
                            (
                                None,
                                None,
                                None,
                                0,
                                None,
                                (
                                    (k_tile, scale_consumer_state.index)
                                    if self.tma_scales
                                    else scale_consumer_state.index
                                ),
                            ),
                        )

                        scale_atom_copy = cute.make_copy_atom(
                            cute.nvgpu.CopyUniversalOp(),
                            self.acc_dtype,
                            num_bits_per_copy=self.acc_dtype.width,
                        )

                        cute.copy(scale_atom_copy, tTR_sSFA_slice, tTR_rSFA)
                        cute.copy(scale_atom_copy, tTR_sSFB_slice, tTR_rSFB)
                        if cutlass.const_expr(not self.tma_scales):
                            scale_pipeline.consumer_release(scale_consumer_state)

                        #
                        # Wait for accumulator buffer full
                        #

                        if cutlass.const_expr(not self.tmem_before_scales):
                            acc_pipeline.consumer_wait(
                                acc_consumer_state, peek_acc_full_status
                            )
                            tTR_tAcc = cute.group_modes(
                                tTR_tAcc, 3, cute.rank(tTR_tAcc)
                            )

                        #
                        # Update accumulator by scale factor in subtiles
                        #
                        subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                        # Keep the FP32 sum in registers and apply scales in
                        # the original 32-column subtiles.
                        for subtile_idx in cutlass.range_constexpr(subtile_cnt):
                            if cutlass.const_expr(self.tmem_before_scales):
                                tTR_rAcc.store(
                                    current_partial[
                                        (None, None, None, subtile_idx)
                                    ].load()
                                )
                            else:
                                cute.copy(
                                    tiled_copy_t2r,
                                    tTR_tAcc[(None, None, None, subtile_idx)],
                                    tTR_rAcc,
                                )
                                if cutlass.const_expr(subtile_idx == subtile_cnt - 1):
                                    cute.arch.fence_view_async_tmem_load()
                                    with cute.arch.elect_one():
                                        acc_pipeline.consumer_release(
                                            acc_consumer_state
                                        )

                            #
                            # Update accumulator by scale factor
                            #
                            tTR_rAcc_subtile = tTR_rAcc_final[
                                (None, None, None, subtile_idx)
                            ]
                            tTR_rSFA_subtile = tTR_rSFA[(None, None, None, subtile_idx)]
                            tTR_rSFB_subtile = tTR_rSFB[(None, None, None, subtile_idx)]

                            acc_vec = tTR_rAcc.load()
                            final_vec = tTR_rAcc_subtile.load()
                            scale_a = tTR_rSFA_subtile.load()
                            scale_b = tTR_rSFB_subtile.load()
                            scale = scale_a * scale_b
                            final_vec = acc_vec * scale + final_vec
                            tTR_rAcc_subtile.store(final_vec.to(self.acc_dtype))

                        #
                        # Async arrive accumulator buffer empty
                        #
                        if cutlass.const_expr(not self.tma_scales):
                            scale_consumer_state.advance()

                        peek_scale_full_status = cutlass.Boolean(1)
                        if (
                            cutlass.const_expr(not self.tma_scales)
                            and scale_consumer_state.count < k_tile_cnt
                        ):
                            peek_scale_full_status = scale_pipeline.consumer_try_wait(
                                scale_consumer_state
                            )
                        #
                        # Async arrive accumulator buffer empty
                        #
                        acc_consumer_state.advance()

                        peek_acc_full_status = cutlass.Boolean(1)
                        if acc_consumer_state.count < k_tile_cnt:
                            peek_acc_full_status = acc_pipeline.consumer_try_wait(
                                acc_consumer_state
                            )

                    if cutlass.const_expr(self.tma_scales):
                        scale_pipeline.consumer_release(scale_consumer_state)
                        scale_consumer_state.advance()
                    if cutlass.const_expr(self.row_parallel):
                        if cutlass.const_expr(not self.concurrent_reduction):
                            epi_pipeline.producer_acquire(epi_producer_state)
                        for subtile_idx in cutlass.range_constexpr(
                            cute.size(tTR_rAcc_final.shape, mode=[3])
                        ):
                            final_subtile = tTR_rAcc_final[
                                (None, None, None, subtile_idx)
                            ]
                            handoff_regs.store(
                                epilogue_op(
                                    handoff_copy.retile(final_subtile)
                                    .load()
                                    .to(self.c_dtype)
                                )
                            )
                            # The FP32 reduction starts at +0, so replacing BF16
                            # -0 partials by +0 preserves its result and reserves
                            # -0 to mean that an incoming value is not ready yet.
                            handoff_bits = cute.recast_tensor(
                                handoff_regs, cutlass.Uint32
                            )
                            for value_index in cutlass.range_constexpr(
                                cute.size(handoff_bits)
                            ):
                                handoff_bits[value_index] = _canonicalize_zero_pair(
                                    handoff_bits[value_index]
                                )
                            cute.copy(
                                handoff_copy,
                                handoff_regs,
                                handoff_shared[
                                    (
                                        None,
                                        None,
                                        None,
                                        (
                                            subtile_idx + (tidx // 128) * 2
                                            if self.split_accumulator
                                            else subtile_idx
                                        ),
                                    )
                                ],
                            )
                        cute.arch.fence_proxy("async.shared", space="cta")
                        if cutlass.const_expr(self.concurrent_reduction):
                            # Both accumulator groups share this staging tensor.
                            # Join before publishing it and before reusing it.
                            self.acc_store_sync_barrier.arrive_and_wait()
                            self.store_partial_vector(
                                sC, peers, rank, tile_info[0], tile_info[1], tidx // 128
                            )
                            self.acc_store_sync_barrier.arrive_and_wait()

                    else:
                        tRT_tAcc = tRT_tAcc_base[(None, None, None, None, None, 0)]
                        tRT_tAcc = cute.group_modes(tRT_tAcc, 3, cute.rank(tRT_tAcc))

                        #
                        # Wait for epilogue buffer empty
                        #
                        epi_pipeline.producer_acquire(epi_producer_state)

                        # copy the accumulator to tensor memory buffer
                        cute.copy(tiled_copy_r2t, tTR_rAcc_final, tRT_tAcc)
                        cute.arch.fence_view_async_tmem_store()

                    #
                    # Async arrive epilogue buffer full
                    #
                    if cutlass.const_expr(not self.concurrent_reduction):
                        epi_pipeline.producer_commit(epi_producer_state)
                        epi_producer_state.advance()

                    #
                    # Advance to next tile
                    #
                    tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                    for idx in cutlass.range(4, unroll_full=True):
                        tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                    is_valid_tile = tile_info[3] == 1
                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    tile_info_pipeline.consumer_release(tile_info_consumer_state)
                    tile_info_consumer_state.advance()

            #
            # Specialized epilogue warps
            #
            if (
                warp_idx <= self.epilog_warp_id[-1]
                and warp_idx >= self.epilog_warp_id[0]
            ):
                cute.arch.setmaxregister_decrease(self.num_regs_epilogue_warps)
                #
                # Alloc tensor memory buffer
                #
                if cutlass.const_expr(self.concurrent_reduction):
                    # Accumulator warps store partials, leaving this group free
                    # to reduce completed output while GEMM keeps advancing.
                    tmem.allocate(self.num_tmem_alloc_cols)
                    tmem.wait_for_alloc()
                    if warp_idx == self.epilog_warp_id[0]:
                        self.await_tile_input_owners(
                            (
                                rank * peers[1].shape[0] // self.mma_tiler[0]
                                if cutlass.const_expr(peers[1].shape[0] in (576, 960))
                                else rank * (peers[1].shape[0] // self.mma_tiler[0])
                            ),
                            peers[1].shape[0],
                            input_flags,
                            peer_input_flags,
                            rank,
                            cutlass.Int32(0),
                        )
                    self.reduce_sync_barrier.arrive_and_wait()
                    early_epoch = cute.arch.load(
                        input_flags.iterator + 16 + bidz,
                        cutlass.Int64,
                        sem="acquire",
                        scope="gpu",
                    )
                    early_copy = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        self.c_dtype,
                        num_bits_per_copy=128,
                    )
                    self.reduce_output_inband(
                        output,
                        peers,
                        ready,
                        reduce_job,
                        early_epoch,
                        early_copy,
                        live_rows,
                        True,
                    )
                else:
                    tmem.allocate(self.num_tmem_alloc_cols)

                    #
                    # Bar sync for retrieve tensor memory ptr from shared memory
                    #
                    tmem.wait_for_alloc()

                    #
                    # Retrieving tensor memory ptr and make accumulator tensor
                    #
                    tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
                    # (MMA, MMA_M, MMA_N, STAGE)
                    tCtAcc_base_ = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
                    tCtAcc_final = cute.make_tensor(
                        tCtAcc_base_.iterator + self.tmem_final_offset,
                        tCtAcc_base_.layout,
                    )

                    #
                    # Partition for epilogue
                    #
                    epi_tidx = tidx % 128
                    (
                        tiled_copy_t2r,
                        tTR_tAcc_base,
                        tTR_rAcc,
                    ) = self.epilog_tmem_copy_and_partition(
                        epi_tidx, tCtAcc_final, tCgC, epi_tile, use_2cta_instrs
                    )

                    tTR_rC = None
                    tiled_copy_r2s = None
                    tRS_rC = None
                    tRS_sC = None
                    bSG_sC = None
                    bSG_gC_partitioned = None
                    tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
                    tiled_copy_r2s, tRS_rC, tRS_sC = (
                        self.epilog_smem_copy_and_partition(
                            tiled_copy_t2r, tTR_rC, epi_tidx, sC
                        )
                    )
                    (
                        tma_atom_c,
                        bSG_sC,
                        bSG_gC_partitioned,
                    ) = self.epilog_gmem_copy_and_partition(
                        epi_tidx, tma_atom_c, tCgC, epi_tile, sC
                    )

                    remote_partitions = ()
                    remote_epi_tile = (
                        math.gcd(peers[1].shape[0], self.mma_tiler[0]),
                        epi_tile[1],
                    )
                    for peer in cutlass.range_constexpr(len(remote_atoms)):
                        remote_global = cute.local_tile(
                            remote_tensors[peer],
                            cute.slice_(self.mma_tiler, (None, None, 0)),
                            (None, None, None),
                        )
                        remote_tCgC = thr_mma.partition_C(remote_global)
                        remote_atom, remote_shared, remote_partition = (
                            self.epilog_gmem_copy_and_partition(
                                epi_tidx,
                                remote_atoms[peer],
                                remote_tCgC,
                                remote_epi_tile,
                                sC,
                            )
                        )
                        remote_partitions = (*remote_partitions, remote_partition)

                    #
                    # Persistent tile scheduling loop
                    #
                    tile_sched = utils.StaticPersistentTileScheduler.create(
                        tile_sched_params,
                        cute.arch.block_idx(),
                        (
                            cute.arch.grid_dim()
                            if self.gemm_ctas == self.num_ctas
                            else (
                                cutlass.Int32(1),
                                cutlass.Int32(1),
                                cutlass.Int32(self.gemm_ctas),
                            )
                        ),
                    )
                    work_tile = tile_sched.initial_work_tile_info()
                    if cutlass.const_expr(self.skip_padded_tiles):
                        if live_rows <= peers[1].shape[0] - self.mma_tiler[0]:
                            live_coord = self.cache_tile_coord(
                                work_tile.tile_idx,
                                peers[1].shape[0],
                                len(peer_ready),
                                output.shape[1],
                            )
                            while (
                                work_tile.is_valid_tile
                                and (
                                    live_coord[0]
                                    % (peers[1].shape[0] // self.mma_tiler[0])
                                )
                                * self.mma_tiler[0]
                                >= live_rows
                            ):
                                tile_sched.advance_to_next_work()
                                work_tile = tile_sched.get_current_work()
                                live_coord = self.cache_tile_coord(
                                    work_tile.tile_idx,
                                    peers[1].shape[0],
                                    len(peer_ready),
                                    output.shape[1],
                                )

                    epi_consumer_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer, 1
                    )

                    c_pipeline = None
                    # Threads/warps participating in tma store pipeline
                    c_producer_group = pipeline.CooperativeGroup(
                        pipeline.Agent.Thread,
                        32 * len(self.epilog_warp_id),
                    )
                    c_pipeline = pipeline.PipelineTmaStore.create(
                        num_stages=self.num_c_stage,
                        producer_group=c_producer_group,
                    )

                    tile_info_consumer_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer, self.num_tile_stage
                    )

                    # get the first tile info
                    tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
                    # Get tile coord from tile scheduler
                    cur_tile_coord = self.cache_tile_coord(
                        work_tile.tile_idx,
                        peers[1].shape[0],
                        len(peer_ready),
                        output.shape[1],
                    )
                    # initialize the tile info
                    tile_info[0] = cur_tile_coord[0]
                    tile_info[1] = cur_tile_coord[1]
                    tile_info[2] = cur_tile_coord[2]
                    tile_info[3] = work_tile.is_valid_tile

                    is_valid_tile = cutlass.Boolean(1)
                    is_valid_tile = tile_info[3] == 1

                    num_prev_subtiles = cutlass.Int32(0)

                    # Other owners may still be reducing the preceding call. Wait only
                    # for destinations this CTA can overwrite before issuing TMA stores.
                    if cutlass.const_expr(self.row_parallel):
                        if warp_idx == self.epilog_warp_id[0]:
                            bx, by, bz = cute.arch.block_idx()
                            gx, gy, gz = cute.arch.grid_dim()
                            previous_epoch = cute.arch.load(
                                ready.iterator + 16 + bx + gx * (by + gy * bz),
                                cutlass.Int64,
                                sem="relaxed",
                                scope="gpu",
                            )
                            first_owner = (
                                tile_info[0]
                                * self.cta_tile_shape_mnk[0]
                                // peers[1].shape[0]
                            )
                            owners_per_tile = max(
                                1, self.cta_tile_shape_mnk[0] // peers[1].shape[0]
                            )
                            # A GEMM tile can intersect two owners at these sizes.
                            if cutlass.const_expr(peers[1].shape[0] in (224, 576, 960)):
                                owners_per_tile = (
                                    (
                                        (tile_info[0] + 1) * self.cta_tile_shape_mnk[0]
                                        - 1
                                    )
                                    // peers[1].shape[0]
                                    - first_owner
                                    + 1
                                )
                            m_tiles = cute.ceil_div(
                                len(peer_ack) * peers[1].shape[0],
                                self.cta_tile_shape_mnk[0],
                            )
                            if cutlass.const_expr(
                                self.dynamic_gemm or self.gemm_ctas % m_tiles != 0
                            ):
                                first_owner = cutlass.Int32(0)
                                owners_per_tile = len(peer_ack)
                            owner = (rank + lane_idx) % len(peer_ack)
                            address = peer_ack[0].iterator
                            for peer in cutlass.range_constexpr(len(peer_ack)):
                                if lane_idx == peer:
                                    address = peer_ack[peer].iterator
                            if (
                                lane_idx < len(peer_ack)
                                and owner != rank
                                and owner >= first_owner
                                and owner < first_owner + owners_per_tile
                            ):
                                arrival = cute.arch.load(
                                    address, cutlass.Int64, sem="acquire", scope="sys"
                                )
                                while arrival < previous_epoch:
                                    arrival = cute.arch.load(
                                        address,
                                        cutlass.Int64,
                                        sem="acquire",
                                        scope="sys",
                                    )
                        self.epilog_sync_barrier.arrive_and_wait()
                        if warp_idx == self.epilog_warp_id[0]:
                            cute.arch.fence_proxy("async.global")

                    while is_valid_tile:
                        mma_tile_coord_mnl = (
                            tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                            tile_info[1],
                            tile_info[2],
                        )
                        #
                        # Slice to per mma tile index
                        #
                        bSG_gC = None
                        # ((ATOM_V, REST_V), EPI_M, EPI_N)
                        bSG_gC = bSG_gC_partitioned[
                            (
                                None,
                                None,
                                None,
                                mma_tile_coord_mnl[0],
                                mma_tile_coord_mnl[1],
                                mma_tile_coord_mnl[2],
                            )
                        ]

                        # Set tensor memory buffer for current tile
                        # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                        tTR_tAcc = tTR_tAcc_base[
                            (None, None, None, None, None, epi_consumer_state.index)
                        ]

                        #
                        # Wait for accumulator buffer full
                        #
                        epi_pipeline.consumer_wait(epi_consumer_state)

                        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                        bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                        #
                        # Store accumulator to global memory in subtiles
                        #
                        subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                        for subtile_idx in cutlass.range(subtile_cnt):
                            if cutlass.const_expr(self.row_parallel):
                                c_buffer = subtile_idx

                            else:
                                #
                                # Load accumulator from tensor memory buffer to register
                                #
                                tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                                cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)
                                # Return the TMEM stage immediately after its last read.
                                if subtile_idx == subtile_cnt - 1:
                                    cute.arch.fence_view_async_tmem_load()
                                    epi_pipeline.consumer_release(epi_consumer_state)

                                #
                                # Convert to C type
                                #
                                acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                                acc_vec = epilogue_op(acc_vec.to(self.c_dtype))
                                tRS_rC.store(acc_vec)

                                #
                                # Store C to shared memory
                                #
                                num_prev_subtiles = num_prev_subtiles + 1
                                c_buffer = num_prev_subtiles % self.num_c_stage
                                cute.copy(
                                    tiled_copy_r2s,
                                    tRS_rC,
                                    tRS_sC[(None, None, None, c_buffer)],
                                )
                                # Fence and barrier to make sure shared memory store is visible to TMA store
                                cute.arch.fence_proxy(
                                    "async.shared",
                                    space="cta",
                                )
                                self.epilog_sync_barrier.arrive_and_wait()

                            #
                            # TMA store C to global memory
                            #
                            if cutlass.const_expr(self.vector_output):
                                self.store_partial_vector(
                                    sC,
                                    peers,
                                    rank,
                                    mma_tile_coord_mnl[0],
                                    mma_tile_coord_mnl[1],
                                    subtile_idx,
                                )
                            else:
                                if warp_idx == self.epilog_warp_id[0]:
                                    owner_rows = peers[1].shape[0]
                                    remote_rows = remote_epi_tile[0]
                                    for owner_slice in cutlass.range_constexpr(
                                        self.mma_tiler[0] // remote_rows
                                    ):
                                        global_row = (
                                            mma_tile_coord_mnl[0] * self.mma_tiler[0]
                                            + owner_slice * remote_rows
                                        )
                                        owner = global_row // owner_rows
                                        local_row = global_row % owner_rows
                                        for peer in cutlass.range_constexpr(
                                            len(remote_atoms)
                                        ):
                                            if owner == (rank + peer) % len(
                                                remote_atoms
                                            ):
                                                remote_dest = remote_partitions[peer][
                                                    (
                                                        None,
                                                        (local_row % self.mma_tiler[0])
                                                        // remote_rows,
                                                        subtile_idx,
                                                        local_row // self.mma_tiler[0],
                                                        mma_tile_coord_mnl[1],
                                                        mma_tile_coord_mnl[2],
                                                    )
                                                ]
                                                cute.copy(
                                                    remote_atoms[peer],
                                                    remote_shared[
                                                        (None, owner_slice, 0, c_buffer)
                                                    ],
                                                    remote_dest,
                                                )
                                    # Fence and barrier to make sure shared memory store is visible to TMA store
                                    c_pipeline.producer_commit()
                                    c_pipeline.producer_acquire()
                            self.epilog_sync_barrier.arrive_and_wait()

                        if cutlass.const_expr(self.row_parallel):
                            if warp_idx == self.epilog_warp_id[0]:
                                c_pipeline.producer_tail()
                            self.epilog_sync_barrier.arrive_and_wait()
                            epi_pipeline.consumer_release(epi_consumer_state)

                        #
                        # Async arrive accumulator buffer empty
                        #
                        epi_consumer_state.advance()

                        #
                        # Advance to next tile
                        #
                        tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                        for idx in cutlass.range(4, unroll_full=True):
                            tile_info[idx] = sInfo[
                                (idx, tile_info_consumer_state.index)
                            ]
                        is_valid_tile = tile_info[3] == 1
                        cute.arch.fence_proxy(
                            "async.shared",
                            space="cta",
                        )
                        tile_info_pipeline.consumer_release(tile_info_consumer_state)
                        tile_info_consumer_state.advance()

                    #
                    # Dealloc the tensor memory buffer
                    #
                    tmem.relinquish_alloc_permit()
                    self.epilog_sync_barrier.arrive_and_wait()
                    tmem.free(tmem_ptr)
                    #
                    # Wait for C store complete
                    #
                    c_pipeline.producer_tail()
                    if warp_idx == self.epilog_warp_id[0]:
                        cute.arch.cp_async_bulk_wait_group(0, read=False)
                        cute.arch.fence_proxy("async.global")

        # Join all roles before collective completion. Any remaining reduction
        # work reuses the whole resident grid.
        cute.arch.barrier()
        if cutlass.const_expr(self.concurrent_reduction):
            if (
                warp_idx >= self.epilog_warp_id[0]
                and warp_idx <= self.epilog_warp_id[-1]
            ):
                tmem.relinquish_alloc_permit()
                self.epilog_sync_barrier.arrive_and_wait()
                if warp_idx == self.epilog_warp_id[0]:
                    cute.arch.dealloc_tmem(
                        tmem.retrieve_ptr(self.acc_dtype),
                        self.num_tmem_alloc_cols,
                        is_two_cta=use_2cta_instrs,
                    )
            cute.arch.barrier()
        # Return to a uniform register budget for the remaining reduction and
        # collective completion.
        if warp_idx < 8:
            cute.arch.setmaxregister_decrease(128)
        cute.arch.barrier()
        if warp_idx >= 8:
            cute.arch.setmaxregister_increase(128)
        if cutlass.const_expr(self.input_phase_clear and peers[1].shape[0] > 512):
            # A CTA may skip every padded GEMM tile or compute only other
            # owners' rows. Acquire this owner's receive clear before reducing,
            # just like the initially reserved reducers.
            if not early_reducer and warp_idx == 0:
                self.await_tile_input_owners(
                    (
                        rank * peers[1].shape[0] // self.mma_tiler[0]
                        if cutlass.const_expr(peers[1].shape[0] in (576, 960))
                        else rank * (peers[1].shape[0] // self.mma_tiler[0])
                    ),
                    peers[1].shape[0],
                    input_flags,
                    peer_input_flags,
                    rank,
                    cutlass.Int32(0),
                )
        cute.arch.barrier()
        self.phase_exchange(
            output,
            peers,
            ready,
            peer_ready,
            ack,
            peer_ack,
            rank,
            sInfo,
            input_multicast_ptr,
            live_rows,
        )

    @cute.jit
    def cta_index(self):
        # Communication flags index physical CTAs; scheduling indexes clusters.
        if cutlass.const_expr(self.cluster_shape_mn != (1, 1)):
            bx, by, bz = cute.arch.block_idx()
            return bx + self.cluster_shape_mn[0] * (by + self.cluster_shape_mn[1] * bz)
        else:
            return cute.arch.block_idx()[2]

    @cute.jit
    def stage_input(
        self,
        source: cute.Tensor,
        source_scales: Union[cute.Tensor, None],
        rows: cutlass.Constexpr,
        input_peers: tuple,
        scale_peers: tuple,
        flags: cute.Tensor,
        peer_flags: tuple,
        rank: cutlass.Int32,
        peer_ack: tuple,
        notify_address: cutlass.Int64,
        receive: cute.Tensor,
    ):
        # Quantize BF16 once, or forward the supplied FP8 values and scales.
        # Both formats use the same publication and workspace-reuse protocol.
        tid = cute.arch.thread_idx()[0]
        block = self.cta_index()
        blocks = self.num_ctas
        warp = tid // 32
        lane = tid % 32
        valid_rows, width = source.shape
        groups_per_row = width // 128
        epoch = cute.arch.load(
            flags.iterator + 16 + block,
            cutlass.Int64,
            sem="acquire",
            scope=self.local_flag_scope,
        ) + cutlass.Int64(1)
        if cutlass.const_expr(self.row_parallel):
            previous_rows = cute.arch.load(
                flags.iterator + 1, cutlass.Int64, sem="relaxed", scope="gpu"
            )
            previous_valid_rows = cute.arch.load(
                flags.iterator + 2, cutlass.Int64, sem="relaxed", scope="gpu"
            )
            if cutlass.const_expr(
                self.input_phase_clear
                and rows in (224, 256, 576, 640)
                and not self.skip_padded_tiles
            ):
                # These buckets consume all prepared partials, including padding.
                # Completion therefore covers every read of this owner's input.
                # A changed stride still needs the cross-rank layout join.
                join_previous = previous_rows != rows
            else:
                join_previous = (
                    previous_rows != rows or previous_valid_rows != previous_rows
                )
            if join_previous:
                address = peer_ack[0].iterator
                for peer in cutlass.range_constexpr(len(peer_ack)):
                    if tid == peer:
                        address = peer_ack[peer].iterator
                if tid < len(peer_ack):
                    arrival = cute.arch.load(
                        address, cutlass.Int64, sem="acquire", scope="sys"
                    )
                    while arrival < epoch - cutlass.Int64(1):
                        arrival = cute.arch.load(
                            address, cutlass.Int64, sem="acquire", scope="sys"
                        )
                cute.arch.barrier()
        # Eight lanes quantize one 128-channel group in large row buckets.
        # Wider loads reduce loop overhead without mixing neighboring scales.
        values_per_lane = 16 if self.large_row_bucket else 4
        lanes_per_group = 128 // values_per_lane
        groups_per_warp = 32 // lanes_per_group
        input_values = cute.make_rmem_tensor((values_per_lane,), source.element_type)
        input_copy = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            source.element_type,
            num_bits_per_copy=min(128, values_per_lane * source.element_type.width),
        )
        fragment = cute.make_rmem_tensor((values_per_lane,), cutlass.Float32)
        quantized = cute.make_rmem_tensor((values_per_lane,), self.a_dtype)
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.a_dtype,
            num_bits_per_copy=values_per_lane * 8,
        )
        first_group = (
            block * (self.threads_per_cta // lanes_per_group)
            + warp * groups_per_warp
            + lane // lanes_per_group
        )
        group_stride = blocks * (self.threads_per_cta // lanes_per_group)
        # Direct FP8 copies avoid lookahead bookkeeping at this bucket.
        prefetch_input = (
            self.row_parallel
            and rows >= 160
            and not (
                self.input_phase_clear and rows == 256 and source_scales is not None
            )
        )
        pending_input = cute.make_rmem_tensor((values_per_lane,), source.element_type)
        if cutlass.const_expr(prefetch_input):
            pending_input.fill(0)
            if first_group < rows * groups_per_row:
                if first_group // groups_per_row < valid_rows:
                    first_input = cute.make_tensor(
                        source.iterator
                        + cute.assume(
                            first_group * 128
                            + (lane % lanes_per_group) * values_per_lane,
                            divby=values_per_lane,
                        ),
                        cute.make_layout((values_per_lane,), stride=(1,)),
                    )
                    cute.copy(input_copy, first_input, pending_input)
        for group in cutlass.range(first_group, rows * groups_per_row, group_stride):
            row = group // groups_per_row
            k_group = group % groups_per_row
            input_tile = cute.make_tensor(
                source.iterator
                + cute.assume(
                    row * width
                    + k_group * 128
                    + (lane % lanes_per_group) * values_per_lane,
                    divby=values_per_lane,
                ),
                cute.make_layout((values_per_lane,), stride=(1,)),
            )
            input_values.fill(0)
            if cutlass.const_expr(prefetch_input):
                input_values.store(pending_input.load())
                # Start the next independent read before processing this group.
                pending_input.fill(0)
                next_group = group + group_stride
                if next_group < rows * groups_per_row:
                    if next_group // groups_per_row < valid_rows:
                        next_input = cute.make_tensor(
                            source.iterator
                            + cute.assume(
                                next_group * 128
                                + (lane % lanes_per_group) * values_per_lane,
                                divby=values_per_lane,
                            ),
                            cute.make_layout((values_per_lane,), stride=(1,)),
                        )
                        cute.copy(input_copy, next_input, pending_input)
            else:
                if row < valid_rows:
                    cute.copy(input_copy, input_tile, input_values)
            if cutlass.const_expr(self.input_phase_clear and rows > 512):
                # Clear the owner's receive storage before publishing its input.
                # All output producers await that owner's input epoch, so no
                # new partial can overwrite a clear still in flight.
                # The first 512 physical rows stay clear after reduction,
                # covering every smaller bucket without transition cleanup.
                prefix_vectors = 512 * receive.shape[1] // 8
                receive_vectors = cute.size(receive) // 8
                quant_vectors = rows * groups_per_row * lanes_per_group
                clear_groups = cute.ceil_div(
                    receive_vectors - prefix_vectors, quant_vectors
                )
                for clear_group in cutlass.range_constexpr(clear_groups):
                    clear_vector = (
                        prefix_vectors
                        + group * lanes_per_group
                        + lane % lanes_per_group
                        + clear_group * quant_vectors
                    )
                    if clear_vector < receive_vectors:
                        _clear_payload_streaming(
                            receive.iterator.toint() + cutlass.Int64(clear_vector) * 16
                        )
            if cutlass.const_expr(source_scales is not None):
                # Preserve FP8 bytes and scales; do not dequantize or requantize.
                quantized.store(input_values.load())
                inverse = cutlass.Float32(1.0)
                if row < valid_rows:
                    inverse = source_scales[row, k_group]
            else:
                values = input_values.load().to(cutlass.Float32)
                # The reference one-shot AllGather removes its BF16 -0 sentinel
                # up to 128 rows. Larger NCCL gathers preserve source signed zeros.
                if cutlass.const_expr(rows <= 128):
                    zeros = cute.zeros_like(values)
                    fragment.store(cute.where(values == zeros, zeros, values))
                else:
                    fragment.store(values)
                amax = cutlass.Float32(0.0)
                for i in cutlass.range_constexpr(values_per_lane):
                    value = fragment[i]
                    amax = cute.arch.fmax(amax, cute.arch.fmax(value, -value))
                if cutlass.const_expr(self.large_row_bucket):
                    # XOR distances below eight keep each group's maximum separate.
                    for delta in cutlass.range_constexpr(3):
                        amax = cute.arch.fmax(
                            amax, cute.arch.shuffle_sync_bfly(amax, 1 << delta)
                        )
                else:
                    amax = cute.arch.warp_redux_sync(
                        amax,
                        kind="fmax",
                        mask_and_clamp=0xFFFFFFFF,
                        abs=False,
                        nan=False,
                    )
                # Exact FP32 representation of BF16(1e-10), bits 0x2edc0000.
                # A compile-time BF16 literal followed by .to(Float32) preserves
                # the Python literal in this DSL and does not round it to BF16.
                amax = cute.arch.fmax(amax, cutlass.Float32(1.000444171950221e-10))
                multiplier = cutlass.Float32(448.0) / amax
                inverse = cutlass.Float32(1.0) / multiplier
                quantized.store((fragment.load() * multiplier).to(self.a_dtype))
            for peer in cutlass.range_constexpr(len(input_peers)):
                peer_width = width
                peer_group = k_group
                send = cutlass.Boolean(True)
                if cutlass.const_expr(self.row_parallel):
                    peer_width = width // len(input_peers)
                    peer_group = k_group % (peer_width // 128)
                    send = k_group // (peer_width // 128) == (rank + peer) % len(
                        input_peers
                    )
                if send:
                    target = cute.make_tensor(
                        input_peers[peer].iterator
                        + cute.assume(
                            (rank * rows + row) * peer_width
                            + peer_group * 128
                            + (lane % lanes_per_group) * values_per_lane,
                            divby=values_per_lane,
                        ),
                        cute.make_layout((values_per_lane,), stride=(1,)),
                    )
                    cute.copy(copy_atom, quantized, target)
                    if lane % lanes_per_group == 0:
                        scale_peers[peer][rank * rows + row, peer_group, 0] = inverse
        # The CTA barrier orders every writer before its leader's release.
        # Row projection joins these device-local arrivals in CTA zero, then
        # publishes one system-release epoch for remote input consumers.
        if cutlass.const_expr(not self.row_parallel):
            cute.arch.fence_acq_rel_sys()
        cute.arch.barrier()
        if tid == 0:
            cute.arch.store(
                flags.iterator + 16 + block,
                epoch,
                sem="release",
                scope=self.local_flag_scope,
            )
        if block == 0:
            if cutlass.const_expr(self.row_parallel and blocks < self.num_cta_flags):
                # Keep inactive CTA epochs current when prepared buckets use
                # different resident grids. These slots own no work this time.
                if tid >= blocks and tid < self.num_cta_flags:
                    cute.arch.store(
                        flags.iterator + 16 + tid,
                        epoch,
                        sem="relaxed",
                        scope="gpu",
                    )
            if tid < blocks:
                arrival = cute.arch.load(
                    flags.iterator + 16 + tid,
                    cutlass.Int64,
                    sem="acquire",
                    scope=self.local_flag_scope,
                )
                while arrival < epoch:
                    arrival = cute.arch.load(
                        flags.iterator + 16 + tid,
                        cutlass.Int64,
                        sem="acquire",
                        scope=self.local_flag_scope,
                    )
            cute.arch.barrier()
            if tid == 0:
                if cutlass.const_expr(self.row_parallel):
                    cute.arch.store(
                        flags.iterator + 1,
                        cutlass.Int64(rows),
                        sem="relaxed",
                        scope="gpu",
                    )
                    cute.arch.store(
                        flags.iterator + 2,
                        cutlass.Int64(valid_rows),
                        sem="relaxed",
                        scope="gpu",
                    )
                if cutlass.const_expr(
                    self.row_parallel and rows >= 160 and self.use_input_multicast
                ):
                    cute.arch.store(flags.iterator, epoch, sem="relaxed", scope="gpu")
                    if cutlass.const_expr(len(input_peers) == 2):
                        _publish_epoch_async(
                            notify_address + cutlass.Int64((176 + rank * 16) * 8), epoch
                        )
                    else:
                        _publish_input_epoch(
                            notify_address + cutlass.Int64((176 + rank * 16) * 8), epoch
                        )
                else:
                    cute.arch.store(flags.iterator, epoch, sem="release", scope="sys")
            cute.arch.barrier()
        # Only the TMA and scale producers read the communicated activation.
        cute.arch.barrier()

    @cute.jit
    def cache_tile_coord(
        self,
        coord,
        rows: cutlass.Constexpr,
        world: cutlass.Constexpr,
        width: cutlass.Constexpr,
    ):
        result = coord
        if cutlass.const_expr(self.cache_group_m > 0):
            local_m_tiles = rows // self.mma_tiler[0]
            # Remap each N pair together, then restore the CTA-local N offset.
            cluster_n = self.cluster_shape_mn[1]
            n_tiles = width // (self.mma_tiler[1] * cluster_n)
            linear = coord[0] + (coord[1] // cluster_n) * (world * local_m_tiles)
            group_area = self.cache_group_m * world * n_tiles
            group = linear // group_area
            within = linear % group_area
            active_m = cutlass.Int32(self.cache_group_m)
            if cutlass.const_expr(local_m_tiles % self.cache_group_m != 0):
                if group >= local_m_tiles // self.cache_group_m:
                    active_m = cutlass.Int32(local_m_tiles % self.cache_group_m)
            owner = within % world
            local_m = group * self.cache_group_m + (within // world) % active_m
            physical_n = ((within // world) // active_m) * cluster_n + coord[
                1
            ] % cluster_n
            result = (owner * local_m_tiles + local_m, physical_n, coord[2])
        return result

    @cute.jit
    def await_tile_input_owners(
        self,
        tile_m: cutlass.Int32,
        rows: cutlass.Constexpr,
        flags: cute.Tensor,
        peer_flags: tuple,
        rank: cutlass.Int32,
        ready_mask: cutlass.Int32,
    ):
        first_owner = tile_m * self.cta_tile_shape_mnk[0] // rows
        owners_per_tile = max(1, self.cta_tile_shape_mnk[0] // rows)
        # Cache readiness for every owner touched by the tile.
        if cutlass.const_expr(self.row_parallel and rows in (576, 960)):
            owners_per_tile = (
                ((tile_m + 1) * self.cta_tile_shape_mnk[0] - 1) // rows
                - first_owner
                + 1
            )
            required = ((cutlass.Int32(1) << owners_per_tile) - 1) << first_owner
        else:
            required = cutlass.Int32((1 << owners_per_tile) - 1) << first_owner
        if (ready_mask & required) != required:
            block = self.cta_index()
            epoch = cute.arch.load(
                flags.iterator + 16 + block,
                cutlass.Int64,
                sem="acquire",
                scope=self.local_flag_scope,
            )
            lane = cute.arch.lane_idx()
            address = peer_flags[0].iterator
            for peer in cutlass.range_constexpr(len(peer_flags)):
                if lane == peer:
                    address = peer_flags[peer].iterator
            owner = (rank + lane) % len(peer_flags)
            if (
                lane < len(peer_flags)
                and owner >= first_owner
                and owner < first_owner + owners_per_tile
                and (ready_mask & (cutlass.Int32(1) << owner)) == 0
            ):
                arrival = cute.arch.load(
                    address, cutlass.Int64, sem="acquire", scope="sys"
                )
                while arrival < epoch:
                    arrival = cute.arch.load(
                        address, cutlass.Int64, sem="acquire", scope="sys"
                    )
            cute.arch.sync_warp()
            cute.arch.fence_proxy("async.global")
        return ready_mask | required

    @cute.jit
    def await_input_owner(
        self,
        tile_m: cutlass.Int32,
        rows: cutlass.Constexpr,
        flags: cute.Tensor,
        peer_flags: tuple,
        rank: cutlass.Int32,
    ):
        tid = cute.arch.thread_idx()[0]
        block = self.cta_index()
        epoch = cute.arch.load(
            flags.iterator + 16 + block,
            cutlass.Int64,
            sem="acquire",
            scope=self.local_flag_scope,
        )
        first_owner = tile_m * self.cta_tile_shape_mnk[0] // rows
        owners_per_tile = max(1, self.cta_tile_shape_mnk[0] // rows)
        # A GEMM tile can intersect two owners at these sizes.
        if cutlass.const_expr(self.row_parallel and rows in (224, 576, 960)):
            owners_per_tile = (
                ((tile_m + 1) * self.cta_tile_shape_mnk[0] - 1) // rows
                - first_owner
                + 1
            )
        # M is the contiguous scheduler coordinate. A CTA retains its M tile
        # only when the grid stride is a multiple of the number of M tiles.
        # Otherwise later iterations may read any owner, regardless of TP size.
        m_tiles = cute.ceil_div(len(peer_flags) * rows, self.cta_tile_shape_mnk[0])
        if cutlass.const_expr(self.dynamic_gemm or self.gemm_ctas % m_tiles != 0):
            first_owner = cutlass.Int32(0)
            owners_per_tile = len(peer_flags)
        if cutlass.const_expr(self.row_parallel):
            # One lane per peer issues the acquire in parallel. The warp join
            # below orders every TMA/scale producer after all required owners.
            lane = cute.arch.lane_idx()
            address = peer_flags[0].iterator
            for peer in cutlass.range_constexpr(len(peer_flags)):
                if lane == peer:
                    address = peer_flags[peer].iterator
            owner = (rank + lane) % len(peer_flags)
            if (
                lane < len(peer_flags)
                and owner >= first_owner
                and owner < first_owner + owners_per_tile
            ):
                arrival = cute.arch.load(
                    address, cutlass.Int64, sem="acquire", scope="sys"
                )
                while arrival < epoch:
                    arrival = cute.arch.load(
                        address, cutlass.Int64, sem="acquire", scope="sys"
                    )
        else:
            if cute.arch.lane_idx() == 0:
                for peer in cutlass.range_constexpr(len(peer_flags)):
                    owner = (rank + peer) % len(peer_flags)
                    if owner >= first_owner and owner < first_owner + owners_per_tile:
                        arrival = cute.arch.load(
                            peer_flags[peer].iterator,
                            cutlass.Int64,
                            sem="acquire",
                            scope="sys",
                        )
                        while arrival < epoch:
                            arrival = cute.arch.load(
                                peer_flags[peer].iterator,
                                cutlass.Int64,
                                sem="acquire",
                                scope="sys",
                            )
        cute.arch.sync_warp()
        cute.arch.fence_proxy("async.global")

    @cute.jit
    def phase_exchange(
        self,
        output: cute.Tensor,
        peers: tuple,
        ready: cute.Tensor,
        peer_ready: tuple,
        ack: cute.Tensor,
        peer_ack: tuple,
        rank: cutlass.Int32,
        queue_shared: cute.Tensor,
        notify_address: cutlass.Int64,
        live_rows: cutlass.Int32,
    ):
        tid = cute.arch.thread_idx()[0]
        block = self.cta_index()
        blocks = self.num_ctas
        width = output.shape[1]
        rows = peers[1].shape[0]
        valid_rows = output.shape[0]
        epoch = cute.arch.load(
            ready.iterator + 16 + block,
            cutlass.Int64,
            sem="acquire",
            scope=self.local_flag_scope,
        ) + cutlass.Int64(1)
        cute.arch.barrier()
        if tid == 0:
            cute.arch.store(
                ready.iterator + 16 + block,
                epoch,
                sem="relaxed" if self.row_parallel else "release",
                scope=self.local_flag_scope,
            )
        copy_in = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            self.c_dtype,
            num_bits_per_copy=128,
            load_cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL,
        )
        copy_out = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self.c_dtype, num_bits_per_copy=128
        )
        if cutlass.const_expr(self.row_parallel):
            if cutlass.const_expr(not self.concurrent_reduction):
                if cutlass.const_expr(self.gemm_ctas == self.num_ctas):
                    self.reduce_output_inband(
                        output,
                        peers,
                        ready,
                        queue_shared,
                        epoch,
                        copy_out,
                        live_rows,
                        False,
                    )
                elif block < self.gemm_ctas:
                    self.reduce_output_inband(
                        output,
                        peers,
                        ready,
                        queue_shared,
                        epoch,
                        copy_out,
                        live_rows,
                        False,
                    )
        else:
            values = cute.make_rmem_tensor((8,), self.c_dtype)
            # M is the contiguous scheduler coordinate. Each copy CTA handles a
            # weight shard and strided N tiles. Smaller row buckets put multiple
            # owners in one GEMM tile, so those N tiles can have different producer
            # CTAs. Acquire each distinct producer before reading its output.
            # Small buckets divide one M tile; larger owners contain whole M tiles.
            # This keeps owner boundaries aligned for remote stores and producer waits.
            assert rows in (32, 64) or (rows >= 128 and rows % 128 == 0)
            world = len(peer_ready)
            shard_width = width // world
            n_tiles = shard_width // 128
            producer_groups = blocks // world
            producer = block // producer_groups
            first_column_tile = block % producer_groups
            tile_count = (
                n_tiles - first_column_tile + producer_groups - 1
            ) // producer_groups
            # TP2 at the 32-row bucket has one partially filled M tile.
            m_tiles = cute.ceil_div(world * rows, self.cta_tile_shape_mnk[0])
            # For whole-tile owners, advancing by producer_groups N tiles shifts
            # the producer by blocks * (rows / 128), hence wraps to the same CTA.
            # Small buckets repeat after max(1, 128 / rows) such steps.
            # Owners larger than a GEMM tile also need each local M-tile producer.
            for row_tile in cutlass.range_constexpr(max(1, rows // 128)):
                if row_tile * 128 < valid_rows:
                    for step in cutlass.range_constexpr(max(1, 128 // rows)):
                        if step < tile_count:
                            column_tile = first_column_tile + step * producer_groups
                            producer_block = (
                                rank * rows // 128 + row_tile + m_tiles * column_tile
                            ) % blocks
                            for peer in cutlass.range_constexpr(len(peer_ready)):
                                if producer == (rank + peer) % world:
                                    if tid == 0:
                                        count = cute.arch.load(
                                            peer_ready[peer].iterator
                                            + 16
                                            + producer_block,
                                            cutlass.Int64,
                                            sem="acquire",
                                            scope="sys",
                                        )
                                        while count < epoch:
                                            count = cute.arch.load(
                                                peer_ready[peer].iterator
                                                + 16
                                                + producer_block,
                                                cutlass.Int64,
                                                sem="acquire",
                                                scope="sys",
                                            )
            cute.arch.barrier()
            vectors_per_tile = valid_rows * 16
            total_vectors = vectors_per_tile * tile_count
            fragments = tuple(
                cute.make_rmem_tensor((8,), self.c_dtype) for _ in range(4)
            )
            for base in cutlass.range(tid, total_vectors, self.threads_per_cta * 4):
                for item in cutlass.range_constexpr(4):
                    vector = base + item * self.threads_per_cta
                    if vector < total_vectors:
                        row = (vector // 16) % valid_rows
                        column_tile = (
                            first_column_tile
                            + (vector // vectors_per_tile) * producer_groups
                        )
                        column = (
                            producer * shard_width
                            + column_tile * 128
                            + (vector % 16) * 8
                        )
                        offset = cute.assume(row * width + column, divby=8)
                        source = cute.make_tensor(
                            peers[1].iterator + offset,
                            cute.make_layout((8,), stride=(1,)),
                        )
                        cute.copy(copy_in, source, fragments[item])
                for item in cutlass.range_constexpr(4):
                    vector = base + item * self.threads_per_cta
                    if vector < total_vectors:
                        row = (vector // 16) % valid_rows
                        column_tile = (
                            first_column_tile
                            + (vector // vectors_per_tile) * producer_groups
                        )
                        column = (
                            producer * shard_width
                            + column_tile * 128
                            + (vector % 16) * 8
                        )
                        offset = cute.assume(row * width + column, divby=8)
                        target = cute.make_tensor(
                            output.iterator + offset,
                            cute.make_layout((8,), stride=(1,)),
                        )
                        cute.copy(copy_out, fragments[item], target)
        # CTA zero acquires each per-CTA arrival in parallel and publishes
        # one rank epoch. The barrier completes every reader before the local
        # release; CTA zero forwards that ordering with a system-release ack.
        if cutlass.const_expr(not self.row_parallel):
            cute.arch.fence_acq_rel_sys()
        cute.arch.barrier()
        if tid == 0:
            cute.arch.store(
                ack.iterator + 16 + block,
                epoch,
                sem="release",
                scope=self.local_flag_scope,
            )
        if block == 0:
            if cutlass.const_expr(self.row_parallel and blocks < self.num_cta_flags):
                if tid >= blocks and tid < self.num_cta_flags:
                    cute.arch.store(
                        ready.iterator + 16 + tid,
                        epoch,
                        sem="relaxed",
                        scope="gpu",
                    )
                    cute.arch.store(
                        ack.iterator + 16 + tid,
                        epoch,
                        sem="relaxed",
                        scope="gpu",
                    )
            if tid < blocks:
                arrival = cute.arch.load(
                    ack.iterator + 16 + tid,
                    cutlass.Int64,
                    sem="acquire",
                    scope=self.local_flag_scope,
                )
                while arrival < epoch:
                    arrival = cute.arch.load(
                        ack.iterator + 16 + tid,
                        cutlass.Int64,
                        sem="acquire",
                        scope=self.local_flag_scope,
                    )
            cute.arch.barrier()
            if tid == 0:
                # The local consumption join also implies every local CTA
                # already published its ready epoch.
                if cutlass.const_expr(self.row_parallel):
                    # This counter is diagnostic; consumers acquire per-CTA ready
                    # flags or the following system-release acknowledgement.
                    cute.arch.store(ready.iterator, epoch, sem="relaxed", scope="sys")
                else:
                    cute.arch.store(ready.iterator, epoch, sem="release", scope="sys")
                if cutlass.const_expr(self.row_parallel and self.use_input_multicast):
                    cute.arch.store(ack.iterator, epoch, sem="relaxed", scope="gpu")
                    _publish_epoch_async(
                        notify_address + cutlass.Int64((176 + rank * 16 + 1) * 8), epoch
                    )
                else:
                    cute.arch.store(ack.iterator, epoch, sem="release", scope="sys")
            cute.arch.barrier()
        # One lane per peer acquires its acknowledgement in CTA zero.
        # Other warps may retire: stream order waits for these polling lanes.
        if cutlass.const_expr(not self.row_parallel):
            if block == 0:
                address = peer_ack[0].iterator
                for step in cutlass.range_constexpr(len(peer_ack)):
                    if tid == step:
                        address = peer_ack[step].iterator
                if tid < len(peer_ack):
                    count = cute.arch.load(
                        address, cutlass.Int64, sem="acquire", scope="sys"
                    )
                    while count < epoch:
                        count = cute.arch.load(
                            address, cutlass.Int64, sem="acquire", scope="sys"
                        )

    @cute.jit
    def store_partial_vector(self, shared, peers, rank, tile_m, tile_n, subtile):
        if cutlass.const_expr(peers[1].shape[0] == 960):
            # A 128-row tile can intersect two owners in the 960-row bucket.
            # Route 64-row pieces; each piece remains contiguous within its owner.
            rows = peers[1].shape[0]
            width = peers[0].shape[1]
            world = len(peers) - 2
            owner = tile_m * self.mma_tiler[0] // rows
            row_base = tile_m * self.mma_tiler[0] % rows
            first_destination = peers[2].iterator
            next_destination = peers[2].iterator
            for peer in cutlass.range_constexpr(1, world):
                if owner == (rank + peer) % world:
                    first_destination = peers[peer + 2].iterator
                if owner + 1 == (rank + peer) % world:
                    next_destination = peers[peer + 2].iterator
            source = shared[(None, None, subtile)]
            atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self.c_dtype, num_bits_per_copy=128
            )
            vector_copy = cute.make_tiled_copy_tv(
                atom,
                cute.make_layout((16, 8), stride=(8, 1)),
                cute.make_layout((1, 8), stride=(8, 1)),
            )
            thread_copy = vector_copy.get_slice(cute.arch.thread_idx()[0] % 128)
            for piece in cutlass.range_constexpr(2):
                local_row = row_base + piece * 64
                destination = first_destination
                if local_row >= rows:
                    local_row = local_row - rows
                    destination = next_destination
                offset = (
                    local_row * width
                    + tile_n * self.mma_tiler[1]
                    + subtile * self.epi_tile[1]
                )
                target = cute.make_tensor(
                    destination + cute.assume(offset, divby=8),
                    cute.make_layout((64, self.epi_tile[1]), stride=(width, 1)),
                )
                source_piece = cute.local_tile(
                    source, (64, self.epi_tile[1]), (piece, 0)
                )
                source_fragment = thread_copy.partition_S(source_piece)
                target_fragment = thread_copy.partition_D(target)
                values = cute.make_fragment_like(source_fragment)
                cute.copy(vector_copy, source_fragment, values)
                cute.copy(vector_copy, values, target_fragment)
        else:
            rows = peers[1].shape[0]
            width = peers[0].shape[1]
            world = len(peers) - 2
            owner = tile_m * self.mma_tiler[0] // rows
            destination = peers[2].iterator
            for peer in cutlass.range_constexpr(1, world):
                if owner == (rank + peer) % world:
                    destination = peers[peer + 2].iterator
            row_base = tile_m * self.mma_tiler[0] % rows
            offset = (
                row_base * width
                + tile_n * self.mma_tiler[1]
                + subtile * self.epi_tile[1]
            )
            target = cute.make_tensor(
                destination + cute.assume(offset, divby=8),
                cute.make_layout(
                    (self.mma_tiler[0], self.epi_tile[1]), stride=(width, 1)
                ),
            )
            source = shared[(None, None, subtile)]
            atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self.c_dtype, num_bits_per_copy=128
            )
            vector_copy = cute.make_tiled_copy_tv(
                atom,
                cute.make_layout((16, 8), stride=(8, 1)),
                cute.make_layout((1, 8), stride=(8, 1)),
            )
            thread_copy = vector_copy.get_slice(cute.arch.thread_idx()[0] % 128)
            source_fragment = thread_copy.partition_S(source)
            target_fragment = thread_copy.partition_D(target)
            values = cute.make_fragment_like(source_fragment)
            cute.copy(vector_copy, source_fragment, values)
            cute.copy(vector_copy, values, target_fragment)

    @cute.jit
    def reduction_barrier(self, warp_reduction: cutlass.Constexpr):
        if cutlass.const_expr(warp_reduction):
            self.reduce_sync_barrier.arrive_and_wait()
        else:
            cute.arch.barrier()

    @cute.jit
    def reduce_output_inband(
        self,
        output,
        peers,
        ready,
        queue_shared,
        epoch,
        copy_out,
        live_rows: cutlass.Int32,
        warp_reduction: cutlass.Constexpr,
    ):
        tid = cute.arch.thread_idx()[0]
        group_threads = 128 if warp_reduction else self.threads_per_cta
        vector_count = 1 if warp_reduction else 2
        if cutlass.const_expr(warp_reduction):
            tid = tid % 128
        rows = peers[1].shape[0]
        width = output.shape[1]
        valid_rows = output.shape[0]
        world = len(peers) - 2
        reduction_columns = (
            128
            if self.concurrent_reduction
            else (64 if self.reduction_rows == 64 else 128)
        )
        vectors_per_row = reduction_columns // 8
        column_tiles = width // reduction_columns
        reduction_rows = self.reduction_rows
        # One GEMM row tile spans 128/reduction_rows reduction row tiles.
        # Cover the same producer row band before advancing output columns.
        reduction_group_m = self.cache_group_m * (
            self.mma_tiler[0] // reduction_rows if self.concurrent_reduction else 1
        )
        row_tiles = cute.ceil_div(rows, reduction_rows)
        fragment = cute.make_rmem_tensor((8,), self.c_dtype)
        accum = cute.make_rmem_tensor((8,), cutlass.Float32)
        if tid == 0:
            queue_epoch = cute.arch.load(
                ready.iterator + 2, cutlass.Int64, sem="acquire", scope="gpu"
            )
            while queue_epoch < epoch:
                _queue_poll_backoff()
                queue_epoch = cute.arch.load(
                    ready.iterator + 2, cutlass.Int64, sem="acquire", scope="gpu"
                )
        self.reduction_barrier(warp_reduction)
        if tid == 0:
            queue_shared[0, 0] = cute.arch.atomic_add(
                ready.iterator + 1, cutlass.Int64(1), sem="relaxed", scope="gpu"
            ).to(cutlass.Int32)
        self.reduction_barrier(warp_reduction)
        tile = queue_shared[0, 0]
        while tile < row_tiles * column_tiles:
            row_tile = tile % row_tiles
            column_tile = tile // row_tiles
            if cutlass.const_expr(reduction_group_m > 0):
                group_area = reduction_group_m * column_tiles
                group = tile // group_area
                within = tile % group_area
                active_m = cutlass.Int32(reduction_group_m)
                if cutlass.const_expr(row_tiles % reduction_group_m != 0):
                    if group >= row_tiles // reduction_group_m:
                        active_m = cutlass.Int32(row_tiles % reduction_group_m)
                row_tile = group * reduction_group_m + within % active_m
                column_tile = within // active_m
            live_tile = cutlass.Boolean(True)
            if cutlass.const_expr(self.skip_padded_tiles):
                live_tile = row_tile * reduction_rows < live_rows
            # No GEMM writes a skipped tile. It stays at the receive sentinel
            # established by the previous prefix clear and this input-tail clear.
            if live_tile:
                fragments = tuple(
                    tuple(
                        cute.make_rmem_tensor((8,), self.c_dtype) for _ in range(world)
                    )
                    for _ in range(vector_count)
                )
                for base in cutlass.range(
                    tid, reduction_rows * vectors_per_row, group_threads * vector_count
                ):
                    for item in cutlass.range_constexpr(vector_count):
                        vector = base + item * group_threads
                        row = row_tile * reduction_rows + vector // vectors_per_row
                        column = (
                            column_tile * reduction_columns
                            + (vector % vectors_per_row) * 8
                        )
                        if vector < reduction_rows * vectors_per_row and row < rows:
                            for producer in cutlass.range_constexpr(world):
                                offset = producer * rows * width + row * width + column
                                packed = _load_payload_128(
                                    peers[1].iterator.toint()
                                    + cutlass.Int64(offset) * 2,
                                    l2_only=world == 2,
                                )
                                packed_fragment = cute.recast_tensor(
                                    fragments[item][producer], cutlass.Uint32
                                )
                                for word in cutlass.range_constexpr(4):
                                    packed_fragment[word] = packed[word]
                    pending = cutlass.Boolean(True)
                    while pending:
                        pending = cutlass.Boolean(False)
                        for item in cutlass.range_constexpr(vector_count):
                            vector = base + item * group_threads
                            row = row_tile * reduction_rows + vector // vectors_per_row
                            column = (
                                column_tile * reduction_columns
                                + (vector % vectors_per_row) * 8
                            )
                            if vector < reduction_rows * vectors_per_row and row < rows:
                                for producer in cutlass.range_constexpr(world):
                                    packed_fragment = cute.recast_tensor(
                                        fragments[item][producer], cutlass.Uint32
                                    )
                                    unfinished = cutlass.Boolean(False)
                                    for word in cutlass.range_constexpr(4):
                                        # Detect a negative-zero halfword using the
                                        # standard zero-halfword bit test.
                                        value = packed_fragment[word] ^ cutlass.Uint32(
                                            0x80008000
                                        )
                                        unfinished = unfinished | (
                                            (
                                                (value - cutlass.Uint32(0x00010001))
                                                & ~value
                                                & cutlass.Uint32(0x80008000)
                                            )
                                            != 0
                                        )
                                    if unfinished:
                                        offset = (
                                            producer * rows * width
                                            + row * width
                                            + column
                                        )
                                        packed = _load_payload_128(
                                            peers[1].iterator.toint()
                                            + cutlass.Int64(offset) * 2,
                                            l2_only=world == 2,
                                        )
                                        for word in cutlass.range_constexpr(4):
                                            packed_fragment[word] = packed[word]
                                        pending = cutlass.Boolean(True)
                        if pending:
                            _queue_poll_backoff()
                    for item in cutlass.range_constexpr(vector_count):
                        vector = base + item * group_threads
                        row = row_tile * reduction_rows + vector // vectors_per_row
                        column = (
                            column_tile * reduction_columns
                            + (vector % vectors_per_row) * 8
                        )
                        if vector < reduction_rows * vectors_per_row and row < rows:
                            accum.fill(0)
                            for producer in cutlass.range_constexpr(world):
                                accum.store(
                                    accum.load()
                                    + fragments[item][producer]
                                    .load()
                                    .to(cutlass.Float32)
                                )
                                source = cute.make_tensor(
                                    peers[1].iterator
                                    + cute.assume(
                                        producer * rows * width + row * width + column,
                                        divby=8,
                                    ),
                                    cute.make_layout((8,), stride=(1,)),
                                )
                                if cutlass.const_expr(
                                    self.input_phase_clear and rows > 512
                                ):
                                    # Receive storage is producer-major. Preserve
                                    # a clean prefix large enough for all small views.
                                    clear_rows = max(
                                        0, min(rows, 512 * world - producer * rows)
                                    )
                                    if cutlass.const_expr(clear_rows == rows):
                                        _clear_payload_streaming(
                                            source.iterator.toint()
                                        )
                                    elif cutlass.const_expr(clear_rows > 0):
                                        if row < clear_rows:
                                            _clear_payload_streaming(
                                                source.iterator.toint()
                                            )
                                else:
                                    _clear_payload_streaming(source.iterator.toint())
                            if row < valid_rows:
                                fragment.store(accum.load().to(self.c_dtype))
                                target = cute.make_tensor(
                                    output.iterator
                                    + cute.assume(row * width + column, divby=8),
                                    cute.make_layout((8,), stride=(1,)),
                                )
                                cute.copy(copy_out, fragment, target)
            self.reduction_barrier(warp_reduction)
            if tid == 0:
                queue_shared[0, 0] = cute.arch.atomic_add(
                    ready.iterator + 1, cutlass.Int64(1), sem="relaxed", scope="gpu"
                ).to(cutlass.Int32)
            self.reduction_barrier(warp_reduction)
            tile = queue_shared[0, 0]

    def acc_update_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        tAcc_final: cute.Tensor,
        gC_mnl: cute.Tensor,
        sSFA: cute.Tensor,
        sSFB: cute.Tensor,
        epi_tile: cute.Tile,
    ) -> Tuple[
        cute.TiledCopy,
        cute.TiledCopy,
        cute.Tensor,
        cute.Tensor,
        cute.Tensor,
        cute.Tensor,
        cute.Tensor,
        cute.Tensor,
        cute.Tensor,
    ]:
        """
        Make tiledCopy for tensor memory load, then use it to partition tensor memory (source) and register array (destination).
        Make tiledCopy for tensor memory store, then use it to partition register array (source) and tensor memory (destination).
        Partition the scale factor tensor for related copy operations.

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param tAcc: The accumulator tensor to be copied and partitioned
        :type tAcc: cute.Tensor
        :param tAcc_final: The final accumulator tensor to be copied and partitioned
        :type tAcc_final: cute.Tensor
        :param gC_mnl: The global tensor C
        :type gC_mnl: cute.Tensor
        :param sSFA: The scale factor tensor for A
        :type sSFA: cute.Tensor
        :param sSFB: The scale factor tensor for B
        :type sSFB: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile

        :return: A tuple containing (tiled_copy_t2r, tTR_tAcc, tTR_rAcc) where:
            - tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
            - tiled_copy_r2t: The tiled copy operation for register to tmem copy(r2t)
            - tTR_tAcc: The partitioned accumulator tensor
            - tTR_rAcc: The accumulated tensor in register used to hold t2r results
            - tTR_rAcc_final: The accumulated tensor in register used to hold all t2r results
            - tTR_sSFA: The partitioned tensor SFA by tiled_copy_t2r
            - tTR_sSFB: The partitioned tensor SFB by tiled_copy_t2r
            - tRT_rAcc_final: The accumulated tensor in register used to hold all r2t results
            - tRT_tAcc_final: The partitioned accumulator tensor by tiled_copy_r2t
        :rtype: Tuple[cute.TiledCopy, cute.TiledCopy, cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor]
        """
        # Make tiledCopy for tensor memory load
        tmem_load_atom = None
        tmem_store_atom = None
        if cutlass.const_expr(self.mma_tiler[0] == 64):
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(4)),
                self.acc_dtype,
            )
        else:
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
                self.acc_dtype,
            )
        if cutlass.const_expr(self.mma_tiler[0] == 64):
            tmem_store_atom = cute.make_copy_atom(
                tcgen05.copy.St16x256bOp(tcgen05.copy.Repetition(4)),
                self.acc_dtype,
            )
        else:
            tmem_store_atom = cute.make_copy_atom(
                tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)),
                self.acc_dtype,
            )

        acc_matrix = tAcc[((None, None), 0, 0, None)]
        final_matrix = tAcc_final[((None, None), 0, 0, None)]
        group = cute.arch.thread_idx()[0] // 128
        if cutlass.const_expr(self.split_accumulator):
            acc_matrix = cute.local_tile(acc_matrix, (128, 64), (0, group, None))
            final_matrix = cute.local_tile(final_matrix, (128, 64), (0, group, None))
        tAcc_epi = cute.flat_divide(acc_matrix, epi_tile)
        tAcc_final_epi = cute.flat_divide(final_matrix, epi_tile)

        tiled_copy_t2r = tcgen05.make_tmem_copy(
            tmem_load_atom, tAcc_epi[(None, None, 0, 0, 0)]
        )
        tiled_copy_r2t = tcgen05.make_tmem_copy(
            tmem_store_atom, tAcc_final_epi[(None, None, 0, 0, 0)]
        )

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        thr_copy_r2t = tiled_copy_r2t.get_slice(tidx)

        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, loopM, loopN, loopL)
        c_matrix = gC_mnl[((None, None), 0, 0, None, None, None)]
        if cutlass.const_expr(self.split_accumulator):
            c_matrix = cute.local_tile(
                c_matrix, (128, 64), (0, group, None, None, None)
            )
            sSFA = cute.local_tile(sSFA, (128, 64), (0, group, None))
            sSFB = cute.local_tile(sSFB, (128, 64), (0, group, None))
        gC_mnl_epi = cute.flat_divide(c_matrix, epi_tile)
        sSFA_epi = cute.flat_divide(sSFA, epi_tile)
        sSFB_epi = cute.flat_divide(sSFB, epi_tile)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, loopM, loopN, loopL)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        tTR_sSFA = thr_copy_t2r.partition_D(sSFA_epi)
        tTR_sSFB = thr_copy_t2r.partition_D(sSFB_epi)
        # (T2R, T2R_M, T2R_N)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
        tTR_rAcc_final_ = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, None, None, 0, 0, 0)].shape, self.acc_dtype
        )
        tTR_rAcc_final = cute.group_modes(
            tTR_rAcc_final_, 3, cute.rank(tTR_rAcc_final_)
        )

        tRT_gC = thr_copy_r2t.partition_S(gC_mnl_epi)
        tRT_tAcc_final = thr_copy_r2t.partition_D(tAcc_final_epi)
        # (R2T, R2T_M, R2T_N, EPI_M, EPI_N, loopM, loopN, loopL)
        tRT_rAcc_final_ = cute.make_rmem_tensor(
            tRT_gC[(None, None, None, None, None, 0, 0, 0)].shape, self.acc_dtype
        )
        # (R2T, R2T_M, R2T_N, (EPI_M, EPI_N))
        tRT_rAcc_final = cute.group_modes(
            tRT_rAcc_final_, 3, cute.rank(tRT_rAcc_final_)
        )

        return (
            tiled_copy_t2r,
            tiled_copy_r2t,
            tTR_tAcc,
            tTR_rAcc,
            tTR_rAcc_final,
            tTR_sSFA,
            tTR_sSFB,
            tRT_rAcc_final,
            tRT_tAcc_final,
        )

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for tensor memory load, then use it to partition tensor memory (source) and register array (destination).

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param tAcc: The accumulator tensor to be copied and partitioned
        :type tAcc: cute.Tensor
        :param gC_mnl: The global tensor C
        :type gC_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param use_2cta_instrs: Whether use_2cta_instrs is enabled
        :type use_2cta_instrs: bool

        :return: A tuple containing (tiled_copy_t2r, tTR_tAcc, tTR_rAcc) where:
            - tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
            - tTR_tAcc: The partitioned accumulator tensor
            - tTR_rAcc: The accumulated tensor in register used to hold t2r results
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # Make tiledCopy for tensor memory load
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        tAcc_epi = cute.flat_divide(
            tAcc[((None, None), 0, 0, None)],
            epi_tile,
        )
        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
        )

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_M, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, loopM, loopN, loopL)
        gC_mnl_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, loopM, loopN, loopL)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)

        # (T2R, T2R_M, T2R_N)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )

        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for shared memory store, then use it to partition register array (source) and shared memory (destination).

        :param tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
        :type tiled_copy_t2r: cute.TiledCopy
        :param tTR_rC: The partitioned accumulator tensor
        :type tTR_rC: cute.Tensor
        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param sC: The shared memory tensor to be copied and partitioned
        :type sC: cute.Tensor
        :type sepi: cute.Tensor

        :return: A tuple containing (tiled_copy_r2s, tRS_rC, tRS_sC) where:
            - tiled_copy_r2s: The tiled copy operation for register to smem copy(r2s)
            - tRS_rC: The partitioned tensor C (register source)
            - tRS_sC: The partitioned tensor C (smem destination)
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        # (R2S, R2S_M, R2S_N)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    def epilog_gmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        atom: Union[cute.CopyAtom, cute.TiledCopy],
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        sC: cute.Tensor,
    ) -> Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        """Make tiledCopy for global memory store, then use it to:
        - partition register array (source) and global memory (destination) for none TMA store version;
        - partition shared memory (source) and global memory (destination) for TMA store version.

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param atom: The copy_atom_c to be used for TMA store version, or tiled_copy_t2r for none TMA store version
        :type atom: cute.CopyAtom or cute.TiledCopy
        :param gC_mnl: The global tensor C
        :type gC_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param sC: The shared memory tensor to be copied and partitioned
        :type sC: cute.Tensor

        :return: A tuple containing :
            - For TMA store: (tma_atom_c, bSG_sC, bSG_gC) where:
                - tma_atom_c: The TMA copy atom
                - bSG_sC: The partitioned shared memory tensor C
                - bSG_gC: The partitioned global tensor C
        :rtype: Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]
        """
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, loopM, loopN, loopL)
        gC_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )
        tma_atom_c = atom
        sC_for_tma_partition = cute.group_modes(cute.flat_divide(sC, epi_tile), 0, 2)
        gC_for_tma_partition = cute.group_modes(gC_epi, 0, 2)
        # ((ATOM_V, REST_V), EPI_M, EPI_N)
        # ((ATOM_V, REST_V), EPI_M, EPI_N, loopM, loopN, loopL)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sC_for_tma_partition,
            gC_for_tma_partition,
        )
        return tma_atom_c, bSG_sC, bSG_gC

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        epi_tile: cute.Tile,
        c_dtype: Type[cutlass.Numeric],
        c_layout: cutlass.tensor_utils.LayoutEnum,
        sfa_dtype: Type[cutlass.Numeric],
        sfb_dtype: Type[cutlass.Numeric],
        sfa_count: int,
        sfb_count: int,
        num_smem_capacity: int,
        occupancy: int,
        large_row_bucket: bool,
    ) -> Tuple[int, int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics.

        :param tiled_mma: The tiled MMA object defining the core computation.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler_mnk: The shape (M, N, K) of the MMA tiler.
        :type mma_tiler_mnk: tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param epi_tile: The epilogue tile shape.
        :type epi_tile: cute.Tile
        :param c_dtype: Data type of operand C (output).
        :type c_dtype: type[cutlass.Numeric]
        :param c_layout: Layout of operand C.
        :type c_layout: cutlass.tensor_utils.LayoutEnum
        :param num_smem_capacity: Total available shared memory capacity in bytes.
        :type num_smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (ACC stages, A/B operand stages, C stages)
        :rtype: tuple[int, int, int]
        """
        # Row epilogues use shared memory, freeing the fourth TMEM stage.
        num_acc_stage = 4 if large_row_bucket else (3 if mma_tiler_mnk[0] == 128 else 6)

        # Default C stages
        num_c_stage = 2

        # Default ScaleA/B stages
        num_scale_stage = 4 if large_row_bucket else 2

        # Default Tile info stages
        num_tile_stage = 2

        # Calculate smem layout and size for one stage of A, B, and C
        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            1,  # a tmp 1 stage is provided
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            1,  # a tmp 1 stage is provided
        )
        c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            1,
        )

        ab_bytes_per_stage = cute.size_in_bytes(
            a_dtype, a_smem_layout_stage_one
        ) + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
        # 1024B alignment
        mbar_helpers_bytes = 1024
        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
        c_bytes = c_bytes_per_stage * num_c_stage
        sfa_bytes = sfa_count * (sfa_dtype.width // 8) * num_scale_stage
        sfb_bytes = sfb_count * (sfb_dtype.width // 8) * num_scale_stage
        scale_bytes = math.ceil((sfa_bytes + sfb_bytes) / 1024) * 1024

        # Calculate A/B stages:
        # Start with total smem per CTA (capacity / occupancy)
        # Subtract reserved bytes and initial C stages bytes
        # Divide remaining by bytes needed per A/B stage
        num_ab_stage = (
            num_smem_capacity // occupancy
            - (mbar_helpers_bytes + c_bytes + scale_bytes)
        ) // ab_bytes_per_stage

        if large_row_bucket:
            # Reserve space for more scales instead of a sixth operand stage.
            num_ab_stage = min(num_ab_stage, 5)
        if mma_tiler_mnk[0] == 64:
            # Keep six operand stages for the 32-row TP4 bucket.
            num_ab_stage = min(num_ab_stage, 6)
        # Refine epilogue stages:
        # Calculate remaining smem after allocating for A/B stages and reserved bytes
        # Add remaining unused smem to epilogue
        num_c_stage += (
            num_smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (mbar_helpers_bytes + c_bytes + scale_bytes)
        ) // (occupancy * c_bytes_per_stage)
        if mma_tiler_mnk[0] == 64 or large_row_bucket:
            # Two 64-column stores cover the tile.
            num_c_stage = 2
        return num_acc_stage, num_ab_stage, num_c_stage, num_scale_stage, num_tile_stage

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        resident_ctas: cutlass.Constexpr,
    ) -> Tuple[utils.PersistentTileSchedulerParams, Tuple[int, int, int]]:
        """Use persistent tile scheduler to compute the grid size for the output tensor C.

        :param c: The output tensor C
        :type c: cute.Tensor
        :param cta_tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type cta_tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]
        :param resident_ctas: Number of resident CTAs in the persistent grid.
        :type resident_ctas: cutlass.Constexpr

        :return: A tuple containing:
            - tile_sched_params: Parameters for the persistent tile scheduler.
            - grid: Grid shape for kernel launch.
        :rtype: Tuple[utils.PersistentTileSchedulerParams, tuple[int, int, int]]
        """
        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl
        )
        # Keep the selected resident grid, including CTAs without GEMM work.
        # CTA zero advances flag slots omitted by a smaller prepared grid.
        grid = (
            *cluster_shape_mn,
            resident_ctas // (cluster_shape_mn[0] * cluster_shape_mn[1]),
        )

        return tile_sched_params, grid

    @staticmethod
    def _get_tma_atom_kind(
        atom_sm_cnt: cutlass.Int32, mcast: cutlass.Boolean
    ) -> Union[
        cpasync.CopyBulkTensorTileG2SMulticastOp, cpasync.CopyBulkTensorTileG2SOp
    ]:
        """
        Select the appropriate TMA copy atom based on the number of SMs and the multicast flag.

        :param atom_sm_cnt: The number of SMs
        :type atom_sm_cnt: cutlass.Int32
        :param mcast: The multicast flag
        :type mcast: cutlass.Boolean

        :return: The appropriate TMA copy atom kind
        :rtype: cpasync.CopyBulkTensorTileG2SMulticastOp or cpasync.CopyBulkTensorTileG2SOp

        :raise ValueError: If the atom_sm_cnt is invalid
        """
        if atom_sm_cnt == 2 and mcast:
            return cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.TWO)
        elif atom_sm_cnt == 2 and not mcast:
            return cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.TWO)
        elif atom_sm_cnt == 1 and mcast:
            return cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.ONE)
        elif atom_sm_cnt == 1 and not mcast:
            return cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)

        raise ValueError(f"Invalid atom_sm_cnt: {atom_sm_cnt} and {mcast}")

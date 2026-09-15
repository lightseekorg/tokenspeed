# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
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


"""Throughput-oriented SM100 up projection into caller-owned symmetric output.

Adapted from CUTLASS blackwell/dense_gemm_persistent.py, retaining its
one/two-CTA MMA, double-buffered accumulators and phase-continuous pipelines.
Shared and replicated residual owner slices are fused with two BF16 roundings.
The output is the final distributed allocation, not an intermediate mailbox.
"""

from typing import Optional, Tuple, Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.common import CacheEvictionPriority
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.primitives import (
    bf16x8_to_packed_u32x4,
    packed_u32x4_to_bf16x8,
)


@cute.jit
def _copy_addend_async(destination, source, active):
    """Use CuTe's swizzle-aware copy, explicitly zeroing invalid vectors.

    Converting a swizzled shared pointer to an integer before handwritten PTX
    can drop the swizzle metadata. Keep typed pointers through the copy atoms.
    The inactive path issues no global read and overwrites all eight values.
    """
    # The restricted coalesced loader starts every BF16x8 vector at a 16-byte
    # boundary: global row strides are 896/7168 BF16 and SMEM rows are 64 BF16.
    # Dynamic partition offsets can lose this fact. align() restores metadata
    # while retaining the shared pointer's swizzle; no address is rounded.
    source_tensor = cute.make_tensor(source.align(16), cute.make_layout(8))
    destination_tensor = cute.make_tensor(destination.align(16), cute.make_layout(8))
    atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        source_tensor.element_type,
        num_bits_per_copy=128,
    )
    if active:
        cute.copy(atom, source_tensor, destination_tensor)
    else:
        zeros = cute.make_rmem_tensor(8, source_tensor.element_type)
        zeros.fill(0)
        cute.autovec_copy(zeros, destination_tensor)


@cute.jit
def _issue_coalesced_addends(
    gemm_kernel: cutlass.Constexpr,
    shared_source,
    residual_source,
    source_coords,
    shared_destination,
    residual_destination,
    subtile_index: cutlass.Int32,
    stage_index: cutlass.Int32,
    output_shape,
):
    """All 128 epilogue threads issue contiguous row vectors, then commit."""
    source_s = shared_source[(None, None, None, subtile_index)]
    source_r = residual_source[(None, None, None, subtile_index)]
    coords = source_coords[(None, None, None, subtile_index)]
    target_s = shared_destination[(None, None, None, stage_index)]
    target_r = residual_destination[(None, None, None, stage_index)]
    source_s = cute.group_modes(source_s, 0, cute.rank(source_s))
    source_r = cute.group_modes(source_r, 0, cute.rank(source_r))
    coords = cute.group_modes(coords, 0, cute.rank(coords))
    target_s = cute.group_modes(target_s, 0, cute.rank(target_s))
    target_r = cute.group_modes(target_r, 0, cute.rank(target_r))
    assert cute.size(source_s) % 8 == 0
    # unroll_full is a backend hint on a runtime loop, not a compile-time
    # iteration. The optional subclass hook is a Python configuration object,
    # so enumerate this fixed-size vector pack entirely during specialization.
    if cutlass.const_expr(gemm_kernel.residual_issue_order != "after_reduce"):
        gemm_kernel.issue_residual_first_addends(
            source_r, coords, target_s, target_r, output_shape
        )
    elif cutlass.const_expr(gemm_kernel.fused_reduce_vectors > 1):
        assert gemm_kernel.fused_shared_rs
        group_size = gemm_kernel.fused_reduce_vectors
        assert cute.size(source_s) % (8 * group_size) == 0
        for vector_group in cutlass.range_constexpr(
            cute.size(source_s) // (8 * group_size)
        ):
            destinations, coordinates, predicates = (), (), ()
            for vector_index in cutlass.range_constexpr(group_size):
                offset = (vector_group * group_size + vector_index) * 8
                active = cute.elem_less(coords[offset], output_shape) & cute.elem_less(
                    coords[offset + 7], output_shape
                )
                destinations += (target_s.iterator + target_s.layout(offset),)
                coordinates += (coords[offset],)
                predicates += (active,)
            gemm_kernel.load_shared_vectors(destinations, coordinates, predicates)
            for vector_index in cutlass.range_constexpr(group_size):
                offset = (vector_group * group_size + vector_index) * 8
                _copy_addend_async(
                    target_r.iterator + target_r.layout(offset),
                    source_r.iterator + source_r.layout(offset),
                    predicates[vector_index],
                )
    else:
        for vector_index in cutlass.range_constexpr(cute.size(source_s) // 8):
            offset = vector_index * 8
            active = cute.elem_less(coords[offset], output_shape) & cute.elem_less(
                coords[offset + 7], output_shape
            )
            if cutlass.const_expr(gemm_kernel.fused_shared_rs):
                gemm_kernel.load_shared_vector(
                    target_s.iterator + target_s.layout(offset), coords[offset], active
                )
            else:
                _copy_addend_async(
                    target_s.iterator + target_s.layout(offset),
                    source_s.iterator + source_s.layout(offset),
                    active,
                )
            if cutlass.const_expr(not gemm_kernel.preadded_owner):
                _copy_addend_async(
                    target_r.iterator + target_r.layout(offset),
                    source_r.iterator + source_r.layout(offset),
                    active,
                )
    cute.arch.cp_async_commit_group()


@dsl_user_op
def _load_addend_pair(
    shared_address,
    residual_address,
    active,
    cache_policy,
    load_residual,
    *,
    loc=None,
    ip=None,
):
    """Issue independent shared/residual vector loads before consuming either."""
    opcode = (
        "ld.global.L1::no_allocate.v4.u32"
        if cache_policy == "no_allocate"
        else "ld.global.v4.u32"
    )
    instructions = [
        "{",
        ".reg .pred p_valid;",
        ".reg .u64 shared_ptr, residual_ptr;",
        "mov.u64 shared_ptr, $8;",
        "mov.u64 residual_ptr, $9;",
        "setp.ne.u32 p_valid, $10, 0;",
        *[f"mov.b32 ${index}, 0;" for index in range(8)],
        f"@p_valid {opcode} {{$0, $1, $2, $3}}, [shared_ptr];",
    ]
    if load_residual:
        # One asm block keeps shared conversions from moving ahead of this
        # independent residual request. Legacy preadded-owner omits it entirely.
        instructions.append(f"@p_valid {opcode} {{$4, $5, $6, $7}}, [residual_ptr];")
    instructions.append("}")
    loaded = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 8),
        [
            shared_address.ir_value(loc=loc, ip=ip),
            residual_address.ir_value(loc=loc, ip=ip),
            cutlass.Uint32(active).ir_value(loc=loc, ip=ip),
        ],
        "\n".join(instructions),
        "=r,=r,=r,=r,=r,=r,=r,=r,l,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    addends = []
    for addend_index in range(2):
        packed = vector.from_elements(
            ir.VectorType.get([4], T.i32(), loc=loc),
            [
                llvm.extractvalue(
                    T.i32(), loaded, [addend_index * 4 + index], loc=loc, ip=ip
                )
                for index in range(4)
            ],
            loc=loc,
            ip=ip,
        )
        addends.append(
            packed_u32x4_to_bf16x8(
                cute.TensorSSA(packed, 4, cutlass.Uint32), loc=loc, ip=ip
            )
        )
    return addends[0], addends[1]


@dsl_user_op
def _timeline_timestamp(address, *, loc=None, ip=None):
    """Write an instruction-side global timer sample; no completion fence."""
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip)],
        """{
            .reg .u64 timestamp;
            mov.u64 timestamp, %globaltimer;
            st.global.u64 [$0], timestamp;
        }""",
        "l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _timeline_event(
    gemm_kernel,
    tile_index: cutlass.Int32,
    event_index: cutlass.Int32,
    writer_warp: cutlass.Constexpr,
):
    """Record one bounded event from one elected lane of the selected CTA."""
    if cutlass.const_expr(gemm_kernel.timeline_address != 0):
        bx, by, bz = cute.arch.block_idx()
        warp_idx = cute.arch.warp_idx()
        selected = (
            (bx == gemm_kernel.timeline_cta[0])
            & (by == gemm_kernel.timeline_cta[1])
            & (bz == gemm_kernel.timeline_cta[2])
            & (warp_idx == writer_warp)
        )
        # The last allocated row holds only the final drain begin/end events.
        tile_valid = (tile_index >= 0) & (
            (tile_index < gemm_kernel.timeline_max_tiles)
            | (
                (tile_index == gemm_kernel.timeline_max_tiles)
                & ((event_index == 7) | (event_index == 8))
            )
        )
        if selected & tile_valid & (event_index >= 0) & (event_index < 64):
            with cute.arch.elect_one():
                address = (
                    cutlass.Int64(gemm_kernel.timeline_address)
                    + (cutlass.Int64(tile_index) * 64 + cutlass.Int64(event_index)) * 8
                )
                _timeline_timestamp(address)


@cute.jit
def _epilogue_timeline_event(gemm_kernel, num_tiles_executed, event_index):
    # Epilogue pre-advances its scheduler before entering the tile helper,
    # unlike the MMA warp. Subtract exactly once to match their tile ordinals.
    if cutlass.const_expr(gemm_kernel.timeline_address != 0):
        _timeline_event(
            gemm_kernel,
            num_tiles_executed - 1,
            cutlass.Int32(event_index),
            gemm_kernel.epilogue_warp_id[0],
        )


@dsl_user_op
def _store_vector(address, packed, active, multicast, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            address.ir_value(loc=loc, ip=ip),
            *[packed[i].ir_value(loc=loc, ip=ip) for i in range(4)],
            cutlass.Uint32(active).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(multicast).ir_value(loc=loc, ip=ip),
        ],
        """{
            .reg .pred p_valid, p_multi, p_local;
            setp.ne.u32 p_valid, $5, 0;
            setp.ne.u32 p_multi, $6, 0;
            setp.eq.u32 p_local, $6, 0;
            and.pred p_multi, p_multi, p_valid;
            and.pred p_local, p_local, p_valid;
            @p_multi multimem.st.relaxed.sys.global.v4.f32 [$0], {$1, $2, $3, $4};
            @p_local st.global.v4.u32 [$0], {$1, $2, $3, $4};
        }""",
        "l,r,r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _fence_proxy_alias(*, loc=None, ip=None) -> None:
    """Order multicast virtual-address stores before ordinary-alias loads."""

    llvm.inline_asm(
        None,
        [],
        "fence.proxy.alias;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _epilogue_shared_residual(
    gemm_kernel,
    epi_tidx: cutlass.Int32,
    warp_idx: cutlass.Int32,
    tma_atom_c: cute.CopyAtom,
    tCtAcc_base: cute.Tensor,
    sC: cute.Tensor,
    sAddShared,
    sAddResidual,
    tCgC_base: cute.Tensor,
    tCgShared_base: cute.Tensor,
    tCgResidual_base: cute.Tensor,
    tCcC_base: cute.Tensor,
    mC_mnl: cute.Tensor,
    output_address: cutlass.Int64,
    multicast: cutlass.Boolean,
    epi_tile: cute.Tile,
    num_tiles_executed: cutlass.Int32,
    mma_tile_coord_mnl,
    acc_consumer_state: pipeline.PipelineState,
    acc_pipeline: pipeline.PipelineAsync,
    c_pipeline: pipeline.PipelineTmaStore,
) -> pipeline.PipelineState:
    """Two BF16 roundings and a direct TMA store to the final symmetric output."""
    sm100 = utils.gemm.sm100
    tCgC = sm100.transform_partitioned_tensor_layout(tCgC_base)
    tCgShared = sm100.transform_partitioned_tensor_layout(tCgShared_base)
    tCgResidual = sm100.transform_partitioned_tensor_layout(tCgResidual_base)
    tCcC = sm100.transform_partitioned_tensor_layout(tCcC_base)
    tCtAcc = sm100.transform_partitioned_tensor_layout(tCtAcc_base)

    tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = sm100.epilogue_tmem_copy_and_partition(
        gemm_kernel,
        epi_tidx,
        tCtAcc,
        tCgC,
        epi_tile,
        gemm_kernel.use_2cta_instrs,
    )
    tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, gemm_kernel.c_dtype)
    tTR_rShared = cute.make_rmem_tensor(tTR_rAcc.shape, gemm_kernel.c_dtype)
    tTR_rResidual = cute.make_rmem_tensor(tTR_rAcc.shape, gemm_kernel.c_dtype)
    tiled_copy_r2s, tRS_rC, tRS_sC = sm100.epilogue_smem_copy_and_partition(
        gemm_kernel, tiled_copy_t2r, tTR_rC, epi_tidx, sC
    )

    tCgC_epi = cute.flat_divide(tCgC, epi_tile)
    bSG_sC, bSG_gC_partitioned = cpasync.tma_partition(
        tma_atom_c,
        0,
        cute.make_layout(1),
        cute.group_modes(sC, 0, 2),
        cute.group_modes(tCgC_epi, 0, 2),
    )
    epilog_sync_barrier = pipeline.NamedBarrier(
        barrier_id=gemm_kernel.epilog_sync_bar_id,
        num_threads=32 * len(gemm_kernel.epilogue_warp_id),
    )

    bSG_gC = bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)]
    tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_consumer_state.index)]
    thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
    tTR_gShared_partitioned = thr_copy_t2r.partition_D(
        cute.flat_divide(tCgShared, epi_tile)
    )
    tTR_gResidual_partitioned = thr_copy_t2r.partition_D(
        cute.flat_divide(tCgResidual, epi_tile)
    )
    tTR_gResidual = tTR_gResidual_partitioned[
        (None, None, None, None, None, *mma_tile_coord_mnl)
    ]
    tTR_cC_partitioned = thr_copy_t2r.partition_D(cute.flat_divide(tCcC, epi_tile))
    tTR_gShared = tTR_gShared_partitioned[
        (None, None, None, None, None, *mma_tile_coord_mnl)
    ]
    tTR_cC = tTR_cC_partitioned[(None, None, None, None, None, *mma_tile_coord_mnl)]

    exemplar = tTR_gShared_partitioned[(None, None, None, 0, 0, 0, 0, 0)]
    mcl_r = cute.max_common_layout(tTR_rShared.layout, exemplar.layout)
    shared_copy_bits = min(
        exemplar.iterator.alignment * 8,
        cute.size(mcl_r) * gemm_kernel.c_dtype.width,
        128,
    )
    shared_g2r_atom = cute.make_copy_atom(
        cute.nvgpu.CopyG2ROp(),
        gemm_kernel.c_dtype,
        num_bits_per_copy=shared_copy_bits,
        l1c_evict_priority=(
            CacheEvictionPriority.NO_ALLOCATE
            if gemm_kernel.addend_cache_policy == "no_allocate"
            else CacheEvictionPriority.EVICT_NORMAL
        ),
    )

    tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
    tTR_gShared = cute.group_modes(tTR_gShared, 3, cute.rank(tTR_gShared))
    tTR_cC = cute.group_modes(tTR_cC, 3, cute.rank(tTR_cC))
    tTR_gResidual = cute.group_modes(tTR_gResidual, 3, cute.rank(tTR_gResidual))
    bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

    subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
    if cutlass.const_expr(gemm_kernel.addend_stages > 0):
        copy_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            gemm_kernel.c_dtype,
            num_bits_per_copy=128,
        )
        row_threads = cute.size(epi_tile[1]) // 8
        loader = cute.make_tiled_copy_tv(
            copy_atom,
            cute.make_layout(
                (128 // row_threads, row_threads), stride=(row_threads, 1)
            ),
            cute.make_layout((1, 8)),
        ).get_slice(epi_tidx)
        # Producer and consumer use different thread-value layouts over the
        # same swizzled tile. No global rearrangement or BF16 conversion occurs.
        load_s = loader.partition_S(cute.flat_divide(tCgShared, epi_tile))
        load_r = loader.partition_S(cute.flat_divide(tCgResidual, epi_tile))
        load_c = loader.partition_S(cute.flat_divide(tCcC, epi_tile))
        load_s = load_s[(None, None, None, None, None, *mma_tile_coord_mnl)]
        load_r = load_r[(None, None, None, None, None, *mma_tile_coord_mnl)]
        load_c = load_c[(None, None, None, None, None, *mma_tile_coord_mnl)]
        load_s = cute.group_modes(load_s, 3, cute.rank(load_s))
        load_r = cute.group_modes(load_r, 3, cute.rank(load_r))
        load_c = cute.group_modes(load_c, 3, cute.rank(load_c))
        load_target_s = loader.partition_D(sAddShared)
        load_target_r = loader.partition_D(sAddResidual)
        consume_s = thr_copy_t2r.partition_D(sAddShared)
        consume_r = thr_copy_t2r.partition_D(sAddResidual)
    if cutlass.const_expr(gemm_kernel.timeline_address != 0):
        assert subtile_cnt <= 16, "timeline has room for at most sixteen subtiles"
    previous_subtile_count = num_tiles_executed * subtile_cnt
    if cutlass.const_expr(gemm_kernel.enable_pdl):
        cute.arch.griddepcontrol_wait()
    if cutlass.const_expr(gemm_kernel.addend_stages == 2):
        # Prime stage zero before the accumulator wait. At most one following
        # subtile is prefetched, independently of the output TMA store groups.
        _issue_coalesced_addends(
            gemm_kernel,
            load_s,
            load_r,
            load_c,
            load_target_s,
            load_target_r,
            cutlass.Int32(0),
            cutlass.Int32(0),
            mC_mnl.shape,
        )
    tTR_rAcc_tile = None
    if cutlass.const_expr(gemm_kernel.prefetch_acc_tile):
        # An explicitly requested register reservoir separates TMEM ownership
        # from the entire output ring, not just its last subtile. Unrolling both
        # loops keeps the subtile indices static: dynamic rmem indexing could
        # otherwise turn this reservoir into spilled local memory.
        tTR_rAcc_tile = cute.make_rmem_tensor(
            cute.append(tTR_rAcc.shape, subtile_cnt), gemm_kernel.acc_dtype
        )
        if cutlass.const_expr(gemm_kernel.timeline_address != 0):
            _epilogue_timeline_event(gemm_kernel, num_tiles_executed, 3)
        acc_pipeline.consumer_wait(acc_consumer_state)
        if cutlass.const_expr(gemm_kernel.timeline_address != 0):
            _epilogue_timeline_event(gemm_kernel, num_tiles_executed, 4)
        for load_idx in cutlass.range(subtile_cnt, unroll_full=True):
            cute.copy(
                tiled_copy_t2r,
                tTR_tAcc[(None, None, None, load_idx)],
                tTR_rAcc_tile[(None, None, None, load_idx)],
            )
        cute.arch.fence_view_async_tmem_load()
        if cutlass.const_expr(gemm_kernel.timeline_address != 0):
            _epilogue_timeline_event(gemm_kernel, num_tiles_executed, 5)
        epilog_sync_barrier.arrive_and_wait()
        if cutlass.const_expr(gemm_kernel.release_acc_early):
            with cute.arch.elect_one():
                acc_pipeline.consumer_release(acc_consumer_state)
            if cutlass.const_expr(gemm_kernel.timeline_address != 0):
                _epilogue_timeline_event(gemm_kernel, num_tiles_executed, 6)

    for subtile_idx in cutlass.range(
        subtile_cnt, unroll_full=gemm_kernel.prefetch_acc_tile
    ):
        if cutlass.const_expr(gemm_kernel.addend_stages > 0):
            addend_stage = subtile_idx % gemm_kernel.addend_stages
            if cutlass.const_expr(gemm_kernel.addend_stages == 1):
                _issue_coalesced_addends(
                    gemm_kernel,
                    load_s,
                    load_r,
                    load_c,
                    load_target_s,
                    load_target_r,
                    subtile_idx,
                    cutlass.Int32(0),
                    mC_mnl.shape,
                )
                cute.arch.cp_async_wait_group(0)
            else:
                if subtile_idx + 1 < subtile_cnt:
                    _issue_coalesced_addends(
                        gemm_kernel,
                        load_s,
                        load_r,
                        load_c,
                        load_target_s,
                        load_target_r,
                        subtile_idx + 1,
                        (subtile_idx + 1) % 2,
                        mC_mnl.shape,
                    )
                    cute.arch.cp_async_wait_group(1)
                else:
                    cute.arch.cp_async_wait_group(0)
            # Every thread waits on its own cp.async groups before consumers
            # read data produced by other lanes/warps through the new layout.
            epilog_sync_barrier.arrive_and_wait()
        tTR_gShared_subtile = tTR_gShared[(None, None, None, subtile_idx)]
        tTR_cC_subtile = tTR_cC[(None, None, None, subtile_idx)]
        pred_shape = (1, *tTR_cC_subtile.shape[1:])
        pred = cute.make_rmem_tensor(pred_shape, cutlass.Boolean)
        for m_idx in range(tTR_cC_subtile.shape[1]):
            for n_idx in range(tTR_cC_subtile.shape[2]):
                pred[(0, m_idx, n_idx)] = cute.elem_less(
                    tTR_cC_subtile[(0, m_idx, n_idx)], mC_mnl.shape
                )
        if cutlass.const_expr(gemm_kernel.addend_stages > 0):
            cute.autovec_copy(consume_s[(None, None, None, addend_stage)], tTR_rShared)
            if cutlass.const_expr(not gemm_kernel.preadded_owner):
                cute.autovec_copy(
                    consume_r[(None, None, None, addend_stage)], tTR_rResidual
                )
            else:
                tTR_rResidual.store(
                    cute.zeros_like(tTR_rResidual, dtype=gemm_kernel.c_dtype)
                )
        elif cutlass.const_expr(gemm_kernel.paired_addend_loads):
            assert shared_copy_bits == 128 and cute.size(mcl_r) >= 8
            assert cute.size(tTR_rShared) % 8 == 0
            shared_flat = cute.group_modes(
                tTR_gShared_subtile, 0, cute.rank(tTR_gShared_subtile)
            )
            residual_subtile = tTR_gResidual[(None, None, None, subtile_idx)]
            residual_flat = cute.group_modes(
                residual_subtile, 0, cute.rank(residual_subtile)
            )
            coords_flat = cute.group_modes(tTR_cC_subtile, 0, cute.rank(tTR_cC_subtile))
            for vector_idx in cutlass.range(
                cute.size(tTR_rShared) // 8, unroll_full=True
            ):
                value_index = vector_idx * 8
                # Check both ends of each aligned, row-contiguous BF16x8 vector.
                # Invalid rows/owner columns return zero without issuing loads.
                active = cute.elem_less(
                    coords_flat[value_index], mC_mnl.shape
                ) & cute.elem_less(coords_flat[value_index + 7], mC_mnl.shape)
                shared_address = (
                    shared_flat.iterator + shared_flat.layout(value_index)
                ).toint()
                residual_address = (
                    residual_flat.iterator + residual_flat.layout(value_index)
                ).toint()
                shared_values, residual_values = _load_addend_pair(
                    shared_address,
                    residual_address,
                    active,
                    gemm_kernel.addend_cache_policy,
                    not gemm_kernel.preadded_owner,
                )
                cute.make_tensor(
                    tTR_rShared.iterator + value_index, cute.make_layout(8)
                ).store(shared_values)
                cute.make_tensor(
                    tTR_rResidual.iterator + value_index, cute.make_layout(8)
                ).store(residual_values)
        else:
            tTR_rShared.store(cute.zeros_like(tTR_rShared, dtype=gemm_kernel.c_dtype))
            cute.copy(
                shared_g2r_atom,
                tTR_gShared_subtile,
                tTR_rShared,
                pred=pred,
            )

            tTR_rResidual.store(
                cute.zeros_like(tTR_rResidual, dtype=gemm_kernel.c_dtype)
            )
            if cutlass.const_expr(not gemm_kernel.preadded_owner):
                cute.copy(
                    shared_g2r_atom,
                    tTR_gResidual[(None, None, None, subtile_idx)],
                    tTR_rResidual,
                    pred=pred,
                )

        if cutlass.const_expr(gemm_kernel.prefetch_acc_tile):
            tTR_rAcc.store(tTR_rAcc_tile[(None, None, None, subtile_idx)].load())
        else:
            # Both owner addends are ready before the accumulator wait.
            if subtile_idx == 0:
                if cutlass.const_expr(gemm_kernel.timeline_address != 0):
                    _epilogue_timeline_event(gemm_kernel, num_tiles_executed, 3)
                acc_pipeline.consumer_wait(acc_consumer_state)
                if cutlass.const_expr(gemm_kernel.timeline_address != 0):
                    _epilogue_timeline_event(gemm_kernel, num_tiles_executed, 4)
            tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
            cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)
            if cutlass.const_expr(gemm_kernel.timeline_address != 0):
                if subtile_idx == subtile_cnt - 1:
                    _epilogue_timeline_event(gemm_kernel, num_tiles_executed, 5)

        if cutlass.const_expr(
            gemm_kernel.release_acc_early and not gemm_kernel.prefetch_acc_tile
        ):
            if subtile_idx == subtile_cnt - 1:
                # No later operation reads this TMEM stage. Complete the loads
                # in all epilogue warps before returning it to the MMA producer;
                # the last subtile remains live in registers while TMA drains.
                cute.arch.fence_view_async_tmem_load()
                epilog_sync_barrier.arrive_and_wait()
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                if cutlass.const_expr(gemm_kernel.timeline_address != 0):
                    _epilogue_timeline_event(gemm_kernel, num_tiles_executed, 6)

        # Keep the accumulator in FP32 through the add so the epilogue rounds once.
        gemm_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
        shared_vec = tiled_copy_r2s.retile(tTR_rShared).load()
        residual_vec = tiled_copy_r2s.retile(tTR_rResidual).load()
        if cutlass.const_expr(gemm_kernel.preadded_owner):
            # Matched addmm diagnostic: residual was added by the caller.
            if cutlass.const_expr(gemm_kernel.round_acc_before_add):
                # Isolated legacy specialization mirrors the measured cuBLAS
                # intermediate BF16 rounding. The RSAG contract never takes
                # this branch and retains its FP32 GEMM-plus-shared addition.
                gemm_vec = gemm_vec.to(gemm_kernel.c_dtype).to(cutlass.Float32)
            result = gemm_vec + shared_vec.to(cutlass.Float32)
        elif cutlass.const_expr(gemm_kernel.addends_before_gemm):
            # Diagnostic legacy producer: BF16(shared + residual), then addmm.
            addend = (
                shared_vec.to(cutlass.Float32) + residual_vec.to(cutlass.Float32)
            ).to(gemm_kernel.c_dtype)
            result = gemm_vec + addend.to(cutlass.Float32)
        else:
            fused_vec = (gemm_vec + shared_vec.to(cutlass.Float32)).to(
                gemm_kernel.c_dtype
            )
            # Existing RSAG contract: addmm + BF16 residual add.
            result = fused_vec.to(cutlass.Float32) + residual_vec.to(cutlass.Float32)
        tRS_rC.store(result.to(gemm_kernel.c_dtype))

        if cutlass.const_expr(gemm_kernel.store_kind == "multimem"):
            # rC has the same logical value coordinates as the predicated
            # addend copy. Each eight-value vector covers one contiguous row span.
            assert cute.size(mcl_r) >= 8
            coords = cute.group_modes(tTR_cC_subtile, 0, cute.rank(tTR_cC_subtile))
            for vector_idx in cutlass.range(cute.size(tTR_rC) // 8, unroll_full=True):
                coord = coords[vector_idx * 8]
                values = cute.make_tensor(
                    tTR_rC.iterator + vector_idx * 8, cute.make_layout(8)
                ).load()
                packed = bf16x8_to_packed_u32x4(values)
                address = (
                    output_address + (cutlass.Int64(coord[0]) * 7168 + coord[1]) * 2
                )
                _store_vector(
                    address, packed, cute.elem_less(coord, mC_mnl.shape), multicast
                )
        else:
            c_buffer = (previous_subtile_count + subtile_idx) % gemm_kernel.num_c_stage
            if cutlass.const_expr(gemm_kernel.acquire_before_store):
                # This warp issued every earlier store group. Waiting here,
                # immediately before reuse, permits the next subtile's TMEM
                # load and addend arithmetic to overlap the prior TMA store.
                if warp_idx == gemm_kernel.epilogue_warp_id[0]:
                    if cutlass.const_expr(gemm_kernel.timeline_address != 0):
                        _epilogue_timeline_event(
                            gemm_kernel, num_tiles_executed, 16 + 3 * subtile_idx
                        )
                    c_pipeline.producer_acquire()
                    if cutlass.const_expr(gemm_kernel.timeline_address != 0):
                        _epilogue_timeline_event(
                            gemm_kernel, num_tiles_executed, 17 + 3 * subtile_idx
                        )
                epilog_sync_barrier.arrive_and_wait()
            cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])
            cute.arch.fence_proxy("async.shared", space="cta")
            epilog_sync_barrier.arrive_and_wait()
            if warp_idx == gemm_kernel.epilogue_warp_id[0]:
                cute.copy(
                    tma_atom_c,
                    bSG_sC[(None, c_buffer)],
                    bSG_gC[(None, subtile_idx)],
                )
                c_pipeline.producer_commit()
                if cutlass.const_expr(gemm_kernel.timeline_address != 0):
                    _epilogue_timeline_event(
                        gemm_kernel, num_tiles_executed, 18 + 3 * subtile_idx
                    )
                if cutlass.const_expr(not gemm_kernel.acquire_before_store):
                    if cutlass.const_expr(gemm_kernel.timeline_address != 0):
                        _epilogue_timeline_event(
                            gemm_kernel, num_tiles_executed, 16 + 3 * subtile_idx
                        )
                    c_pipeline.producer_acquire()
                    if cutlass.const_expr(gemm_kernel.timeline_address != 0):
                        _epilogue_timeline_event(
                            gemm_kernel, num_tiles_executed, 17 + 3 * subtile_idx
                        )
            epilog_sync_barrier.arrive_and_wait()

    epilog_sync_barrier.arrive_and_wait()
    if cutlass.const_expr(not gemm_kernel.release_acc_early):
        with cute.arch.elect_one():
            acc_pipeline.consumer_release(acc_consumer_state)
        if cutlass.const_expr(gemm_kernel.timeline_address != 0):
            _epilogue_timeline_event(gemm_kernel, num_tiles_executed, 6)
    acc_consumer_state.advance()
    return acc_consumer_state


def _compute_stages(
    tiled_mma: cute.TiledMma,
    mma_tiler_mnk: Tuple[int, int, int],
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    smem_capacity: int,
    occupancy: int,
    use_tma_store: bool,
    c_smem_layout: Union[cute.Layout, None],
    requested_c_stages: int,
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
    :param c_dtype: Data type of operand C (output).
    :type c_dtype: type[cutlass.Numeric]
    :param smem_capacity: Total available shared memory capacity in bytes.
    :type smem_capacity: int
    :param occupancy: Target number of CTAs per SM (occupancy).
    :type occupancy: int
    :param use_tma_store: Whether TMA store is enabled.
    :type use_tma_store: bool
    :param c_smem_layout: Layout of C operand in shared memory, or None if not using TMA store.
    :type c_smem_layout: Union[cute.Layout, None]
    :param requested_c_stages: Exact C ring depth, or zero for the frozen heuristic.
    :type requested_c_stages: int

    :return: A tuple containing the computed number of stages for:
             (ACC stages, A/B operand stages, C stages)
    :rtype: tuple[int, int, int]
    """
    # Default ACC stages
    num_acc_stage = 2

    # Default C stages
    num_c_stage = (requested_c_stages or 2) if use_tma_store else 0

    # Calculate smem layout and size for one stage of A, B, and C with 1-stage
    a_smem_layout_stage_one = utils.sm100.make_smem_layout_a(
        tiled_mma, mma_tiler_mnk, a_dtype, 1
    )
    b_smem_layout_staged_one = utils.sm100.make_smem_layout_b(
        tiled_mma, mma_tiler_mnk, b_dtype, 1
    )

    ab_bytes_per_stage = cute.size_in_bytes(
        a_dtype, a_smem_layout_stage_one
    ) + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
    mbar_helpers_bytes = 1024

    c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout)
    c_bytes = c_bytes_per_stage * num_c_stage

    # Calculate A/B stages:
    # Start with total smem per CTA (capacity / occupancy)
    # Subtract reserved bytes and initial C stages bytes
    # Divide remaining by bytes needed per A/B stage
    num_ab_stage = (
        smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)
    ) // ab_bytes_per_stage

    # Refine epilogue stages:
    # Calculate remaining smem after allocating for A/B stages and reserved bytes
    # Add remaining unused smem to epilogue
    if use_tma_store and requested_c_stages == 0:
        num_c_stage += (
            smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (mbar_helpers_bytes + c_bytes)
        ) // (occupancy * c_bytes_per_stage)
    if requested_c_stages and num_ab_stage < 2:
        raise ValueError("requested C ring leaves fewer than two A/B stages")
    return num_acc_stage, num_ab_stage, num_c_stage


class SymmetricUpProjectionGemm:
    """This class implements batched matrix multiplication (C = A x B) with support for various data types
    and architectural features specific to Blackwell GPUs with persistent tile scheduling and warp specialization.

    :param acc_dtype: Data type for accumulation during computation
    :type acc_dtype: type[cutlass.Numeric]
    :param use_2cta_instrs: Whether to use CTA group 2 for advanced thread cooperation
    :type use_2cta_instrs: bool
    :param mma_tiler_mn: Shape of the Matrix Multiply-Accumulate (MMA) tile (M,N)
    :type mma_tiler_mn: Tuple[int, int]
    :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
    :type cluster_shape_mn: Tuple[int, int]
    :param use_tma_store: Whether to use Tensor Memory Access (TMA) for storing results
    :type use_tma_store: bool

    :note: In current version, A and B tensor must have the same data type
        - i.e., Float8E4M3FN for A and Float8E5M2 for B is not supported

    :note: Supported A/B data types:
        - TFloat32
        - Float16/BFloat16
        - Int8/Uint8
        - Float8E4M3FN/Float8E5M2

    :note: Supported accumulator data types:
        - Float32 (for all floating point A/B data types)
        - Float16 (only for fp16 and fp8 A/B data types)
        - Int32 (only for uint8/int8 A/B data types)

    :note: Supported C data types:
        - Float32 (for float32 and int32 accumulator data types)
        - Int32 (for float32 and int32 accumulator data types)
        - Float16/BFloat16 (for fp16 and fp8 accumulator data types)
        - Int8/Uint8 (for uint8/int8 accumulator data types)
        - Float8E4M3FN/Float8E5M2 (for float32 accumulator data types)

    :note: Constraints:
        - MMA tiler M must be 64/128 (use_2cta_instrs=False) or 128/256 (use_2cta_instrs=True)
        - MMA tiler N must be 32-256, step 32
        - Cluster shape M must be multiple of 2 if use_2cta_instrs=True
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 16

    **Example:**
        gemm = PersistentDenseGemmKernel(
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=True,
            mma_tiler_mn=(128, 128),
            cluster_shape_mn=(2, 2)
        )
        gemm(a, b, c, max_active_clusters, stream)
    """

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_tma_store: bool,
        enable_pdl: bool,
        store_kind: str,
    ):
        """Initializes the configuration for a Blackwell dense GEMM kernel.

        This configuration includes several key aspects:

        1.  MMA Instruction Settings (tcgen05):
            - acc_dtype: Data types for MMA accumulator.
            - mma_tiler_mn: The (M, N) shape of the MMA instruction tiler.
            - use_2cta_instrs: Boolean indicating if the tcgen05 MMA variant
              with cta_group=2 should be used.

        2.  Cluster Shape:
            - cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster.

        3. Output C tensor store mode:
            - use_tma_store: Boolean indicating whether to use Tensor Memory Access (TMA) for storing results.

        :param acc_dtype: Data type of the accumulator.
        :type acc_dtype: type[cutlass.Numeric]
        :param mma_tiler_mn: Tuple (M, N) shape of the MMA instruction.
        :type mma_tiler_mn: Tuple[int, int]
        :param use_2cta_instrs: Boolean, True to use cta_group=2 MMA variant.
        :type use_2cta_instrs: bool
        :param cluster_shape_mn: Tuple (ClusterM, ClusterN) shape of the cluster.
        :type cluster_shape_mn: Tuple[int, int]
        :param use_tma_store: Use Tensor Memory Access (TMA) or normal store for output C tensor.
        :type use_tma_store: bool
        """

        if not use_tma_store:
            raise ValueError("the fused epilogue requires the TMA layout path")
        self.acc_dtype: Type[cutlass.Numeric] = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        # K dimension is deferred in _setup_attributes
        self.mma_tiler_mn = mma_tiler_mn
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.use_tma_store = use_tma_store
        self.arch = "sm_100"
        self.enable_pdl = enable_pdl
        self.store_kind = store_kind
        # Opt-in output overlap controls. Existing callers retain the original
        # buffer heuristic, accumulator lifetime and TMA acquire placement.
        self.requested_c_stages = 0
        self.release_acc_early = False
        self.acquire_before_store = False
        self.prefetch_acc_tile = False
        self.requested_epilogue_tile = (0, 0)
        self.addend_cache_policy = "no_allocate"
        self.paired_addend_loads = False
        self.addend_stages = 0
        self.addend_smem_bytes = 0
        # Only the independent fused-RS subclass supplies multicast partials.
        self.fused_shared_rs = False
        self.fused_reduce_vectors = 1
        self.residual_issue_order = "after_reduce"
        self.experimental_fused_n64 = False
        # No timeline code is emitted unless explicitly configured before JIT.
        self.timeline_address = 0
        self.timeline_max_tiles = 0
        self.timeline_cta = (0, 0, 0)
        # Overridden only by the isolated old-producer diagnostic subclass.
        self.addends_before_gemm = False
        self.preadded_owner = False
        self.round_acc_before_add = False
        self.shared_leading_dim = 896

        self.cta_group = (
            tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        self.occupancy = 1
        # Set specialized warp ids
        self.epilogue_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 32 * len(
            (self.mma_warp_id, self.tma_warp_id, *self.epilogue_warp_id)
        )
        # Set barrier id for cta sync, epilogue sync and tmem ptr sync
        self.epilog_sync_bar_id = 1
        self.tmem_alloc_sync_bar_id = 2
        self.tmem_dealloc_sync_bar_id = 3

    def configure_output_pipeline(
        self,
        c_stages: int,
        release_acc_early: bool,
        acquire_before_store: bool,
        prefetch_acc_tile: bool,
    ) -> None:
        """Select experimental output scheduling before compilation.

        Args:
            c_stages: Exact shared-memory output ring depth (2 through 8), or
                zero to retain the original A/B-first allocation heuristic.
            release_acc_early: Release TMEM after the last accumulator load,
                before output arithmetic and the potentially blocked TMA wait.
            acquire_before_store: Wait for a reusable output buffer immediately
                before writing shared memory, instead of after the prior commit.
            prefetch_acc_tile: Drain every accumulator subtile into an unrolled
                register reservoir before output work. With early release, this
                frees TMEM before any output wait, at increased register cost.

        Returns:
            None. The object is configured for its next ``cute.compile`` call.

        These controls neither remove the final TMA drain/proxy-alias fence nor
        change output ownership, BF16 rounding or external rank barriers.
        """
        if type(c_stages) is not int or c_stages not in (0, 2, 3, 4, 5, 6, 7, 8):
            raise ValueError("c_stages must be 0 (automatic) or an integer in [2, 8]")
        if any(
            type(value) is not bool
            for value in (release_acc_early, acquire_before_store, prefetch_acc_tile)
        ):
            raise TypeError("output pipeline switches must be bool")
        if self.store_kind != "tma":
            raise ValueError("output overlap tuning requires store_kind='tma'")
        self.requested_c_stages = c_stages
        self.release_acc_early = release_acc_early
        self.acquire_before_store = acquire_before_store
        self.prefetch_acc_tile = prefetch_acc_tile

    def configure_addend_loads(self, cache_policy: str, paired_loads: bool) -> None:
        """Configure experimental shared/residual global loads before JIT.

        Args:
            cache_policy: ``"no_allocate"`` preserves the original L1 no-allocate policy;
                ``"normal"`` allows normal L1 allocation and eviction behavior.
            paired_loads: Issue each shared/residual BF16x8 pair in one assembly
                block before consuming either, preserving two independent
                requests. False keeps the original CuTe copy implementation.

        Returns:
            None. Exact masks, BF16 bits, rounding and output ownership remain
            unchanged. A preadded-owner legacy plan never loads residual.
        """
        if cache_policy not in ("no_allocate", "normal"):
            raise ValueError("addend cache policy must be 'no_allocate' or 'normal'")
        if type(paired_loads) is not bool:
            raise TypeError("paired_loads must be bool")
        self.addend_cache_policy = cache_policy
        self.paired_addend_loads = paired_loads

    def configure_addend_pipeline(self, stages: int) -> None:
        """Select independent coalesced shared/residual SMEM staging before JIT.

        Args:
            stages: Zero retains register G2R loads. One uses coalesced
                BF16x8 cp.async G2S loads and a wait/ready barrier per subtile.
                Two prefetches the following subtile while the current one is
                consumed. Each stage reserves two complete swizzled epilogue
                tiles, separately from output C and A/B buffers.

        Returns:
            None. The stage memory is reserved before A/B/C sizing; configurations
            leaving fewer than two A/B stages are rejected. This first experiment
            supports only a 128x128 per-CTA tile with a full-M 128x64 epilogue,
            no_allocate policy and unpaired loads. The async copies explicitly
            use .cg cache mode; that is not claimed identical to G2R L1 policy.
        """
        if type(stages) is not int or stages not in (0, 1, 2):
            raise ValueError("addend stages must be the explicit integer 0, 1 or 2")
        if stages:
            cta_m = self.mma_tiler_mn[0] // (2 if self.use_2cta_instrs else 1)
            n64 = (
                getattr(self, "experimental_fused_n64", False) is True
                and self.use_2cta_instrs
                and self.cluster_shape_mn == (2, 1)
                and (cta_m, self.mma_tiler_mn[1]) == (128, 64)
                and stages == 1
            )
            if (
                ((cta_m, self.mma_tiler_mn[1]) != (128, 128) and not n64)
                or self.requested_epilogue_tile != (128, 64)
                or self.store_kind != "tma"
                or self.addend_cache_policy != "no_allocate"
                or self.paired_addend_loads
            ):
                raise ValueError(
                    "coalesced addends require CTA128x128, epi128x64, TMA and unpaired no_allocate"
                )
        self.addend_stages = stages

    def configure_epilogue_tile(self, epi_m: int, epi_n: int) -> None:
        """Select a wider output transaction tile before compilation.

        Args:
            epi_m: Rows per epilogue subtile, or zero with ``epi_n=0`` to retain
                CUTLASS's original automatic tile selection.
            epi_n: Columns per epilogue subtile. Explicit dimensions must divide
                the per-CTA accumulator tile, retain its entire M extent, use
                16 or 32 TMEM datapaths per warp, and contain at least 4096
                elements. For a 256x128 two-CTA MMA, supported examples are
                128x64 and 128x128.

        Returns:
            None. Later setup rebuilds the TMEM/SMEM partitions and TMA store
            descriptor from this tile, then recalculates the A/B/C stage budget.

        This modifies transaction granularity only: the two BF16 roundings,
        physical/symmetric output address, wait protocol and rank ownership are
        unchanged. Generated register usage and correctness still require GPU
        validation for every explicitly selected geometry.
        """
        if type(epi_m) is not int or type(epi_n) is not int:
            raise TypeError("epilogue tile dimensions must be integers")
        if (epi_m, epi_n) == (0, 0):
            self.requested_epilogue_tile = (0, 0)
            return
        if self.store_kind != "tma":
            raise ValueError("epilogue tile tuning requires store_kind='tma'")
        cta_m = self.mma_tiler_mn[0] // (2 if self.use_2cta_instrs else 1)
        cta_n = self.mma_tiler_mn[1]
        # A 64x128 epilogue on the 128x128 per-CTA accumulator compiled but
        # failed ordinary-producer correctness in the bounded width sweep.
        # Retain full CTA M until that 16-datapath partition is independently
        # corrected and validated; do not silently expose the failed geometry.
        if epi_m != cta_m:
            raise ValueError(
                "epilogue M must equal CTA M; partial-M partitions are not validated"
            )
        warp_m, warp_n = (2, 2) if cta_m == 64 and self.use_2cta_instrs else (4, 1)
        if (
            epi_m not in (32, 64, 128)
            or epi_n not in (32, 64, 128, 256)
            or cta_m % epi_m
            or cta_n % epi_n
            or epi_m * epi_n < 4096
        ):
            raise ValueError(
                "epilogue tile must divide the CTA tile and contain >=4096 elements"
            )
        if epi_m // warp_m not in (16, 32) or epi_n % warp_n:
            raise ValueError(
                "epilogue tile is incompatible with the TMEM warp/datapath layout"
            )
        self.requested_epilogue_tile = (epi_m, epi_n)

    def configure_timeline(
        self,
        trace_address: int,
        max_tiles: int,
        selected_cta: Tuple[int, int, int],
    ) -> None:
        """Bind a diagnostic-only fixed device timeline allocation before JIT.

        Args:
            trace_address: Address of a caller-owned, contiguous CUDA uint64
                buffer with at least ``(max_tiles + 1) * 64`` elements. The
                binding must validate device, size and nonaliasing and retain
                this allocation for the compiled plan's complete lifetime.
            max_tiles: Maximum per-CTA persistent tile count to record. Later
                tiles are skipped rather than wrapping or writing out of bounds.
            selected_cta: Exactly one launch-coordinate ``(block_x, block_y,
                block_z)`` to record. A two-CTA MMA leader has even block_x.

        Returns:
            None. The pointer becomes fixed in this diagnostic specialization.

        Zero the buffer before one diagnostic launch/replay; subsequent launches
        overwrite the same slots. Concurrent reuse is unsupported. Per-tile
        slots 0..6 are MMA acquire begin/end/commit, epilogue accumulator wait
        begin/end, last TMEM read, accumulator release. Slots 16+3*s through
        18+3*s are subtile-s output acquire begin/end/commit (s < 16). The final
        reserved row uses slots 7/8 for output drain begin/end. Values are
        ``%globaltimer`` samples, not proof of remote visibility or pure network
        time. Instrumented performance is never an acceptance measurement.
        """
        if type(trace_address) is not int or trace_address <= 0 or trace_address % 8:
            raise ValueError("timeline address must be a positive 8-byte aligned int")
        if type(max_tiles) is not int or not 1 <= max_tiles <= 65535:
            raise ValueError("timeline max_tiles must be an integer in [1, 65535]")
        if (
            type(selected_cta) is not tuple
            or len(selected_cta) != 3
            or any(type(value) is not int or value < 0 for value in selected_cta)
        ):
            raise ValueError(
                "selected_cta must be a nonnegative integer coordinate tuple"
            )
        if self.use_2cta_instrs and selected_cta[0] % 2:
            raise ValueError("select the even-x MMA leader CTA for a two-CTA timeline")
        if self.store_kind != "tma":
            raise ValueError("timeline instrumentation requires store_kind='tma'")
        self.timeline_address = trace_address
        self.timeline_max_tiles = max_tiles
        self.timeline_cta = selected_cta

    def _create_tiled_mma(self):
        return utils.sm100.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )

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

        # Compute epilogue subtile
        if cutlass.const_expr(self.use_tma_store):
            if self.requested_epilogue_tile == (0, 0):
                self.epi_tile = utils.sm100.compute_epilogue_tile_shape(
                    self.cta_tile_shape_mnk,
                    self.use_2cta_instrs,
                    self.c_layout,
                    self.c_dtype,
                )
            else:
                epi_m, epi_n = self.requested_epilogue_tile
                # Preserve CUTLASS's warp-column placement. In the special
                # 64-row two-CTA case, the two N warps own distinct halves of
                # the CTA N range; a plain contiguous tiler would mix them.
                warp_n = (
                    2
                    if self.cta_tile_shape_mnk[0] == 64 and self.use_2cta_instrs
                    else 1
                )
                epi_n_layout = cute.make_layout(
                    (epi_n // warp_n, warp_n),
                    stride=(1, self.cta_tile_shape_mnk[1] // warp_n),
                )
                self.epi_tile = (
                    cute.make_layout(epi_m),
                    cute.coalesce(epi_n_layout),
                )
        else:
            self.epi_tile = self.cta_tile_shape_mnk[:2]

        c_smem_layout = None
        if cutlass.const_expr(self.use_tma_store):
            c_smem_layout = utils.sm100.make_smem_layout_epi(
                self.c_dtype, self.c_layout, self.epi_tile, 1
            )

        self.smem_capacity = utils.get_smem_capacity_in_bytes()
        self.addend_smem_layout_staged = None
        self.addend_smem_bytes = 0
        if self.addend_stages:
            self.addend_smem_layout_staged = utils.sm100.make_smem_layout_epi(
                self.c_dtype, self.c_layout, self.epi_tile, self.addend_stages
            )
            self.addend_smem_bytes = 2 * cute.size_in_bytes(
                self.c_dtype, self.addend_smem_layout_staged
            )

        # Setup A/B/C stage count in shared memory and ACC stage count in tensor memory
        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = _compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.c_dtype,
            self.smem_capacity - self.addend_smem_bytes,
            self.occupancy,
            self.use_tma_store,
            c_smem_layout,
            self.requested_c_stages,
        )
        if self.addend_stages and self.num_ab_stage < 2:
            raise ValueError(
                "coalesced addend staging leaves fewer than two A/B stages"
            )

        # Compute A/B/C shared memory layout
        self.a_smem_layout_staged = utils.sm100.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage
        )
        self.b_smem_layout_staged = utils.sm100.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage
        )

        self.c_smem_layout_staged = None
        if self.use_tma_store:
            self.c_smem_layout_staged = utils.sm100.make_smem_layout_epi(
                self.c_dtype, self.c_layout, self.epi_tile, self.num_c_stage
            )

        if self.experimental_fused_n64:
            # Plain compile-time byte counts, retained before MLIR layouts die.
            # 1024 covers SharedStorage and the five 128-byte alignments; this
            # is an explicit upper bound, not an observed launch dynamic size.
            self.n64_a_smem_bytes = cute.size_in_bytes(
                self.a_dtype, self.a_smem_layout_staged
            )
            self.n64_b_smem_bytes = cute.size_in_bytes(
                self.b_dtype, self.b_smem_layout_staged
            )
            self.n64_c_smem_bytes = cute.size_in_bytes(
                self.c_dtype, self.c_smem_layout_staged
            )
            self.n64_smem_budget_bytes = (
                1024
                + self.n64_a_smem_bytes
                + self.n64_b_smem_bytes
                + self.n64_c_smem_bytes
                + self.addend_smem_bytes
            )
            assert self.n64_smem_budget_bytes <= self.smem_capacity

        # Compute the number of tensor memory allocation columns
        self.num_tmem_alloc_cols = self._compute_num_tmem_alloc_cols(
            tiled_mma, self.mma_tiler, self.num_acc_stage, self.arch
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        shared: cute.Tensor,
        residual: cute.Tensor,
        output_address: cutlass.Int64,
        multicast: cutlass.Boolean,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes before smem/grid/tma computation
        - Setup TMA load/store atoms and tensors
        - Compute grid size with regard to hardware constraints
        - Define shared storage for kernel
        - Launch the kernel synchronously

        :param a: Input tensor A
        :type a: cute.Tensor
        :param b: Input tensor B
        :type b: cute.Tensor
        :param c: Output tensor C
        :type c: cute.Tensor
        :param max_active_clusters: Maximum number of active clusters
        :type max_active_clusters: cutlass.Constexpr
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        :param epilogue_op: Optional elementwise lambda function to apply to the output tensor
        :type epilogue_op: cutlass.Constexpr
        :raises TypeError: If input data types are incompatible with the MMA instruction.
        :raises AssertionError: If OOB (Out-Of-Bounds) tiles are present when TMA store is disabled.
        """
        c = cute.make_tensor(
            cute.make_ptr(
                c.element_type, output_address, cute.AddressSpace.gmem, assumed_align=16
            ),
            c.layout,
        )
        # These are fixed K3 row strides, not runtime DLPack stride scalars.
        # Preserve the 16-byte owner-vector alignment through M indexing.
        shared = cute.make_tensor(
            shared.iterator,
            cute.make_layout(c.shape, stride=(self.shared_leading_dim, 1, 0)),
        )
        residual = cute.make_tensor(
            residual.iterator, cute.make_layout(c.shape, stride=(7168, 1, 0))
        )
        # Setup static attributes before smem/grid/tma computation
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b.element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)

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
        b_op = utils.sm100.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
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
        if cutlass.const_expr(self.use_tma_store):
            epi_smem_layout = cute.select(self.c_smem_layout_staged, mode=[0, 1])
            tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(), c, epi_smem_layout, self.epi_tile
            )

        # Compute grid size
        self.tile_sched_params, grid = self._compute_grid(
            c, self.cta_tile_shape_mnk, self.cluster_shape_mn, max_active_clusters
        )

        # Launch the kernel synchronously
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c if self.use_tma_store else c,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.addend_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            shared,
            residual,
            output_address,
            multicast,
            epilogue_op,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            use_pdl=self.enable_pdl,
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
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        addend_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        mShared_mnl: cute.Tensor,
        mResidual_mnl: cute.Tensor,
        output_address: cutlass.Int64,
        multicast: cutlass.Boolean,
        epilogue_op: cutlass.Constexpr,
    ):
        """
        GPU device kernel performing the Persistent batched GEMM computation.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            if cutlass.const_expr(self.use_tma_store):
                cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # Setup cta/thread coordinates
        #
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

        #
        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_acc_stage * 2
            ]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Initialize mainloop ab_pipeline (barrier) and states
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

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
        tmem_dealloc_barrier = None
        if cutlass.const_expr(not self.use_tma_store):
            tmem_dealloc_barrier = pipeline.NamedBarrier(
                barrier_id=self.tmem_dealloc_sync_bar_id,
                num_threads=32 * len(self.epilogue_warp_id),
            )
        # Tensor memory dealloc barrier init
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        #
        # Setup smem tensor A/B/C
        #
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

        #
        # Partition global tensor for TiledMMA_A/B/C
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
        tCgC = thr_mma.partition_C(gC_mnl)
        addend_tile = cute.slice_(self.mma_tiler, (None, None, 0))
        tCgShared = thr_mma.partition_C(
            cute.local_tile(mShared_mnl, addend_tile, (None, None, None))
        )
        tCgResidual = thr_mma.partition_C(
            cute.local_tile(mResidual_mnl, addend_tile, (None, None, None))
        )
        tCcC = thr_mma.partition_C(
            cute.local_tile(
                cute.make_identity_tensor(mC_mnl.shape), addend_tile, (None, None, None)
            )
        )

        #
        # Partition global/shared tensor for TMA load A/B
        #
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
        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        #
        # Construct the scheduler
        #
        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
        )
        work_tile = tile_sched.initial_work_tile_info()

        #
        # Specialized TMA load warp
        #

        if warp_idx == self.tma_warp_id:
            if cutlass.const_expr(self.enable_pdl):
                cute.arch.griddepcontrol_wait()
            #
            # Persistent tile scheduling loop
            #

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                #
                # Slice to per mma tile index
                #
                # ((atom_v, rest_v), RestK)
                tAgA_slice = tAgA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]
                # ((atom_v, rest_v), RestK)
                tBgB_slice = tBgB[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]

                # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt
                ab_producer.reset()
                peek_ab_empty_status = ab_producer.try_acquire()

                #
                # Tma load loop
                #
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

                    # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt + k_tile + 1
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if handle.count + 1 < k_tile_cnt:
                        peek_ab_empty_status = ab_producer.try_acquire()

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # Wait A/B buffer empty
            #
            ab_producer.tail()

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            #
            # Persistent tile scheduling loop
            #

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

                #
                # Wait for accumulator buffer empty
                #
                if is_leader_cta:
                    if cutlass.const_expr(self.timeline_address != 0):
                        _timeline_event(
                            self,
                            tile_sched.num_tiles_executed,
                            cutlass.Int32(0),
                            self.mma_warp_id,
                        )
                    acc_pipeline.producer_acquire(acc_producer_state)
                    if cutlass.const_expr(self.timeline_address != 0):
                        _timeline_event(
                            self,
                            tile_sched.num_tiles_executed,
                            cutlass.Int32(1),
                            self.mma_warp_id,
                        )

                #
                # Reset the ACCUMULATE field for each tile
                #
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                #
                # Mma mainloop
                #
                for k_tile in range(k_tile_cnt):
                    if is_leader_cta:
                        # Conditionally wait for AB buffer full
                        handle = ab_consumer.wait_and_advance(peek_ab_full_status)

                        # tCtAcc += tCrA * tCrB
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblk_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblk_crd = (None, None, kblk_idx, handle.index)

                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblk_crd],
                                tCrB[kblk_crd],
                                tCtAcc,
                            )
                            # Enable accumulate on tCtAcc after first kblock
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # Async arrive AB buffer empty
                        handle.release()

                        # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
                        peek_ab_full_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_full_status = ab_consumer.try_wait()

                #
                # Async arrive accumulator buffer full
                #
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                    if cutlass.const_expr(self.timeline_address != 0):
                        _timeline_event(
                            self,
                            tile_sched.num_tiles_executed,
                            cutlass.Int32(2),
                            self.mma_warp_id,
                        )
                acc_producer_state.advance()

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # Wait for accumulator buffer empty
            #
            acc_pipeline.producer_tail(acc_producer_state)

        sC = None
        if cutlass.const_expr(self.use_tma_store):
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC = smem.allocate_tensor(
                element_type=self.c_dtype,
                layout=c_smem_layout_staged.outer,
                byte_alignment=128,
                swizzle=c_smem_layout_staged.inner,
            )
        sAddShared = None
        sAddResidual = None
        if cutlass.const_expr(self.addend_stages > 0):
            sAddShared = smem.allocate_tensor(
                element_type=self.c_dtype,
                layout=addend_smem_layout_staged.outer,
                byte_alignment=128,
                swizzle=addend_smem_layout_staged.inner,
            )
            sAddResidual = smem.allocate_tensor(
                element_type=self.c_dtype,
                layout=addend_smem_layout_staged.outer,
                byte_alignment=128,
                swizzle=addend_smem_layout_staged.inner,
            )

        #
        # Specialized epilogue warps
        #
        if warp_idx < self.mma_warp_id:
            #
            # Alloc tensor memory buffer
            #
            tmem.allocate(self.num_tmem_alloc_cols)

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            #
            # Persistent tile scheduling loop for epilogue
            #
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            if cutlass.const_expr(self.use_tma_store):
                assert tma_atom_c is not None and sC is not None
                c_producer_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    32 * len(self.epilogue_warp_id),
                )
                c_pipeline = pipeline.PipelineTmaStore.create(
                    num_stages=self.num_c_stage, producer_group=c_producer_group
                )
            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                #
                # Pre-advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

                num_tiles_executed = tile_sched.num_tiles_executed
                if cutlass.const_expr(self.use_tma_store):
                    acc_consumer_state = _epilogue_shared_residual(
                        self,
                        tidx,
                        warp_idx,
                        tma_atom_c,
                        tCtAcc_base,
                        sC,
                        sAddShared,
                        sAddResidual,
                        tCgC,
                        tCgShared,
                        tCgResidual,
                        tCcC,
                        mC_mnl,
                        output_address,
                        multicast,
                        epi_tile,
                        num_tiles_executed,
                        mma_tile_coord_mnl,
                        acc_consumer_state,
                        acc_pipeline,
                        c_pipeline,
                    )
                else:
                    acc_consumer_state = utils.gemm.sm100.epilogue(
                        self,
                        tidx,
                        tCtAcc_base,
                        tCgC,
                        epi_tile,
                        epilogue_op,
                        mma_tile_coord_mnl,
                        acc_consumer_state,
                        acc_pipeline,
                    )

            if cutlass.const_expr(self.use_tma_store):
                # Wait for C store complete
                if cutlass.const_expr(self.timeline_address != 0):
                    _timeline_event(
                        self,
                        cutlass.Int32(self.timeline_max_tiles),
                        cutlass.Int32(7),
                        self.epilogue_warp_id[0],
                    )
                c_pipeline.producer_tail()
                if cutlass.const_expr(self.timeline_address != 0):
                    _timeline_event(
                        self,
                        cutlass.Int32(self.timeline_max_tiles),
                        cutlass.Int32(8),
                        self.epilogue_warp_id[0],
                    )
                _fence_proxy_alias()
            else:
                # Synchronize before TMEM dealloc (done by the caller)
                tmem_dealloc_barrier.arrive_and_wait()

            #
            # Dealloc the tensor memory buffer
            #
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
    ) -> Tuple[utils.PersistentTileSchedulerParams, Tuple[int, int, int]]:
        """Use persistent tile scheduler to compute the grid size for the output tensor C.

        :param c: The output tensor C
        :type c: cute.Tensor
        :param cta_tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type cta_tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]
        :param max_active_clusters: Maximum number of active clusters.
        :type max_active_clusters: cutlass.Constexpr

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
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        return tile_sched_params, grid

    @staticmethod
    def _compute_num_tmem_alloc_cols(
        tiled_mma: cute.TiledMma,
        mma_tiler: Tuple[int, int, int],
        num_acc_stage: int,
        arch: str,
    ) -> int:
        """
        Compute the number of tensor memory allocation columns.

        :param tiled_mma: The tiled MMA object defining the core computation.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler: The shape (M, N, K) of the MMA tile.
        :type mma_tiler: tuple[int, int, int]
        :param num_acc_stage: The stage of the accumulator tensor.
        :type num_acc_stage: int

        :return: The number of tensor memory allocation columns.
        :rtype: int
        """
        acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, num_acc_stage))
        num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake, arch=arch)

        return num_tmem_alloc_cols

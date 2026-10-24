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

"""One adaptive attention kernel with static head-geometry configurations."""

import math
from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05
from tokenspeed_kernel.thirdparty.cute_dsl._qsa_sparse.common import (
    _CONVERT_WARPGROUPS,
    _HEAD_DIM,
    _PADDING_READ_SLOT,
    _WIDE_NUM_SPLITS,
    _prefetch_private_kv_payload,
    _ScratchArena,
    _select_head_tile,
    copy_gmem_to_smem_u128,
    cpy_dice,
    exp2,
    fadd2,
    ffma2,
    fmul2,
    invalidate_mbarrier,
    ld_shared_remote_f32,
    log2_e,
    mma_dice,
    mma_modes,
    prefetch_gmem_l2,
    set_block_rank,
    smem_fmax,
    store_smem_prefix_block,
    warp_fmax,
    warp_threads,
    warpgroup_threads,
    warpgroup_warps,
)


class MixedInputFusedMultiHeadAttentionDecode:
    def __init__(
        self,
        kv_splits,
        enable_pdl,
        query_rows_per_cta,
        bf16_smem_slots,
        bf16_async_ready,
        head_tile,
        problem_shape,
        query_group,
        original_heads_per_kv,
        caller_allocates_scratch,
        reuse_work,
        softmax_warpgroups,
        device_sm_count,
        *,
        register_budget: tuple[int, int, int] | None,
        distinct_bf16_smem_slots: int,
        launch_threads_per_cta: int | None,
        scalar_softmax_stats: bool,
        direct_query: bool,
        bucket_prefix: bool,
        leader_first: bool,
        producer_warpgroups: int,
        private_tail_workers: int,
        cached_bf16_lookahead: bool,
        work_sync_threads: int,
        prefix_in_cta: bool,
        large_query_grid: bool,
    ):
        self.kv_splits = kv_splits
        self.bf16_smem_slots = bf16_smem_slots
        self.bf16_async_ready = bf16_async_ready
        self.enable_pdl = enable_pdl
        self.query_rows_per_cta = query_rows_per_cta
        self.head_tile = head_tile
        self.problem_shape = problem_shape
        self.query_group = query_group
        self.original_heads_per_kv = original_heads_per_kv
        self.caller_allocates_scratch = caller_allocates_scratch
        self.reuse_work = reuse_work
        self.device_sm_count = device_sm_count
        self.register_budget = register_budget
        self.distinct_bf16_smem_slots = distinct_bf16_smem_slots
        self.scalar_softmax_stats = scalar_softmax_stats
        self.direct_query = direct_query
        self.bucket_prefix = bucket_prefix
        self.leader_first = leader_first
        self.cached_bf16_lookahead = cached_bf16_lookahead
        self.producer_warpgroups = producer_warpgroups
        self.q_stages = 0
        self.cvt_stages = 0
        self.kv_stages = 0
        self.tmem_alloc_cols = 0
        self.mbarrier_reserved_bytes = 0
        self.scratch_arena_bytes = 0
        self.softmax_warpgroups = softmax_warpgroups
        self.softmax_head_tile = head_tile // softmax_warpgroups
        self.softmax_head_tiles = (
            (32, 32, 8)
            if head_tile == 72
            else (self.softmax_head_tile,) * softmax_warpgroups
        )
        assert sum(self.softmax_head_tiles) == head_tile
        self.softmax_warpgroup_ids = tuple(range(softmax_warpgroups))
        warpgroup_id = 0
        self.softmax_warpgroup_id = warpgroup_id
        warpgroup_id += self.softmax_warpgroups
        self.cvt_warpgroup_ids = tuple(
            range(warpgroup_id, warpgroup_id + self.producer_warpgroups)
        )
        warpgroup_id += self.producer_warpgroups
        # Separate QK and PV warps keep each MMA role's register lifetime short.
        self.mma_kq_warp_id = warpgroup_id * warpgroup_warps
        self.mma_vp_warp_id = self.mma_kq_warp_id + 1
        self.tma_qo_warp_id = self.mma_kq_warp_id + 3
        warpgroup_id += 1
        self.threads_per_cta = warpgroup_id * warpgroup_threads
        self.private_tail_warp = self.mma_kq_warp_id + 2
        self.private_tail_start_warp = self.private_tail_warp
        self.private_tail_warps = 0
        self.private_tail_extra_warps = 0
        self.private_tail_workers = private_tail_workers
        self.work_sync_threads = work_sync_threads
        self.prefix_in_cta = prefix_in_cta
        self.large_query_grid = large_query_grid
        self.prefetch_private_kv = False
        self.prefetch_private_kv_warp = self.mma_kq_warp_id + 2
        self.overlap_private_tail = False
        if original_heads_per_kv == 12 and problem_shape[2] == 2:
            self.threads_per_cta = (
                max(warpgroup_id, 4) * warpgroup_threads
                if launch_threads_per_cta is None
                else launch_threads_per_cta
            )
            self.work_sync_threads = (
                512 if bucket_prefix and launch_threads_per_cta == 512 else 0
            )
            self.private_tail_start_warp = warpgroup_id * warpgroup_warps
            self.private_tail_warps = max(
                0, self.threads_per_cta // warp_threads - self.private_tail_start_warp
            )
            self.overlap_private_tail = (
                head_tile == 16
                and query_group == 1
                and self.private_tail_warps > 0
                and kv_splits == 1
                and bf16_async_ready
                and bf16_smem_slots == 3
            )
            self.prefetch_private_kv = bf16_async_ready and (
                (query_group in (4, 6, 8) and head_tile == 12 * query_group)
                or (query_group == 2 and head_tile == 32)
            )
        elif original_heads_per_kv == 12:
            self.overlap_private_tail = (
                kv_splits == 1
                and bf16_async_ready
                and (
                    (head_tile == 16 and query_group == 1 and bf16_smem_slots == 3)
                    or (head_tile == 48 and query_group == 4 and bf16_smem_slots == 2)
                )
            )
            self.private_tail_extra_warps = (
                4 if self.overlap_private_tail and head_tile == 48 else 0
            )
            self.private_tail_workers = 1 + self.private_tail_extra_warps
            self.threads_per_cta += self.private_tail_extra_warps * warp_threads
        else:
            self.overlap_private_tail = private_tail_workers > 0
            assert private_tail_workers in (0, 2, 6)
        self.sp_stages = 1 if self.prefetch_private_kv else 2
        self.o_stages = 1
        self.kv_ring_stages = 2
        if leader_first:
            assert caller_allocates_scratch and query_rows_per_cta == 1
            assert query_group in (6, 8)
        if bucket_prefix:
            assert caller_allocates_scratch and query_group == 1 and kv_splits == 1
            assert original_heads_per_kv == 12 and head_tile == 16
            assert bf16_smem_slots == 3 and bf16_async_ready
        if self.work_sync_threads:
            assert self.work_sync_threads == 512 and caller_allocates_scratch
            assert kv_splits == 1 and query_rows_per_cta == 1

    def __extract_mlir_values__(self):
        return []

    def __new_from_mlir_values__(self, values):
        assert not values
        return self

    @cute.jit
    def _map_query_row(
        self,
        query_work: cutlass.Int32,
        kv_heads: cutlass.Constexpr[int],
        prefix_shared: cute.Tensor,
    ):
        row = query_work // kv_heads
        if cutlass.const_expr(self.leader_first):
            # Flags are immutable after the existing PDL wait. This sample
            # chooses a permutation, while each group's own flag owns reuse.
            if cutlass.const_expr(not self.prefix_in_cta):
                if prefix_shared[0] != 0:
                    group_count = cute.arch.grid_dim()[2] // (
                        self.query_group * kv_heads
                    )
                    within_group = row // group_count
                    group = row - within_group * group_count
                    row = group * self.query_group + within_group
        return row

    @cute.jit
    def _shared_prefix_in_cta(
        self,
        selected_slots: cute.Tensor,
        pair_row: cutlass.Int32,
        arena: cute.Pointer,
        tid: cutlass.Int32,
    ):
        if cutlass.const_expr(
            self.original_heads_per_kv == 12 and self.problem_shape[2] == 2
        ):
            warp = cute.arch.make_warp_uniform(tid // 32)
            mismatches = cute.make_tensor(
                cute.recast_ptr(arena, dtype=cutlass.Int32), cute.make_layout(8)
            )
            different = cutlass.Boolean(False)
            if tid < 128:
                anchor = selected_slots[pair_row * self.query_group, tid]
                for other in cutlass.range_constexpr(1, self.query_group):
                    different = (
                        different
                        or anchor
                        != selected_slots[pair_row * self.query_group + other, tid]
                    )
                warp_mismatch = cute.arch.warp_reduction_sum(cutlass.Int32(different))
                if cute.arch.lane_idx() == 0:
                    mismatches[warp] = warp_mismatch
            cute.arch.sync_threads()
            mismatch = cutlass.Int32(0)
            for index in cutlass.range_constexpr(4):
                mismatch += mismatches[index]
            if mismatch == 0:
                if tid < 128:
                    for chunk in cutlass.range_constexpr(1, 16):
                        index = chunk * 128 + tid
                        anchor = selected_slots[pair_row * self.query_group, index]
                        for other in cutlass.range_constexpr(1, self.query_group):
                            different = (
                                different
                                or anchor
                                != selected_slots[
                                    pair_row * self.query_group + other, index
                                ]
                            )
                    warp_mismatch = cute.arch.warp_reduction_sum(
                        cutlass.Int32(different)
                    )
                    if cute.arch.lane_idx() == 0:
                        # Keep the first four entries immutable for readers.
                        mismatches[warp + 4] = warp_mismatch
                cute.arch.sync_threads()
                mismatch = cutlass.Int32(0)
                for index in cutlass.range_constexpr(4):
                    mismatch += mismatches[index + 4]
            shared_prefix = mismatch == 0
            cute.arch.sync_threads()
            return shared_prefix
        else:
            if cutlass.const_expr(self.original_heads_per_kv == 12):
                first = selected_slots[pair_row * self.query_group, 0]
                first_different = cutlass.Boolean(False)
                for other in cutlass.range_constexpr(1, self.query_group):
                    first_different = (
                        first_different
                        or first
                        != selected_slots[pair_row * self.query_group + other, 0]
                    )
                shared_prefix = cutlass.Boolean(False)
                if not first_different:
                    warp = cute.arch.make_warp_uniform(tid // 32)
                    # An empty flag tensor requests exact detection in this CTA. Reuse
                    # the arena header only until every thread has read the decision.
                    mismatches = cute.make_tensor(
                        cute.recast_ptr(arena, dtype=cutlass.Int32), cute.make_layout(8)
                    )
                    different = cutlass.Boolean(False)
                    if tid < 128:
                        anchor = selected_slots[pair_row * self.query_group, tid]
                        for other in cutlass.range_constexpr(1, self.query_group):
                            different = (
                                different
                                or anchor
                                != selected_slots[
                                    pair_row * self.query_group + other, tid
                                ]
                            )
                        warp_mismatch = cute.arch.warp_reduction_sum(
                            cutlass.Int32(different)
                        )
                        if cute.arch.lane_idx() == 0:
                            mismatches[warp] = warp_mismatch
                    cute.arch.sync_threads()
                    mismatch = cutlass.Int32(0)
                    for index in cutlass.range_constexpr(4):
                        mismatch += mismatches[index]
                    if mismatch == 0:
                        if tid < 128:
                            for chunk in cutlass.range_constexpr(1, 16):
                                index = chunk * 128 + tid
                                anchor = selected_slots[
                                    pair_row * self.query_group, index
                                ]
                                for other in cutlass.range_constexpr(
                                    1, self.query_group
                                ):
                                    different = (
                                        different
                                        or anchor
                                        != selected_slots[
                                            pair_row * self.query_group + other, index
                                        ]
                                    )
                            warp_mismatch = cute.arch.warp_reduction_sum(
                                cutlass.Int32(different)
                            )
                            if cute.arch.lane_idx() == 0:
                                # Keep the first four entries immutable for readers.
                                mismatches[warp + 4] = warp_mismatch
                        cute.arch.sync_threads()
                        mismatch = cutlass.Int32(0)
                        for index in cutlass.range_constexpr(4):
                            mismatch += mismatches[index + 4]
                    shared_prefix = mismatch == 0
                    # TMEM allocation and decode_work may now overwrite the header.
                    cute.arch.sync_threads()
                return shared_prefix

    @cute.jit
    def _sync_work_threads(self):
        if cutlass.const_expr(self.work_sync_threads != 0):
            if cutlass.const_expr(
                self.original_heads_per_kv == 12 and self.problem_shape[2] == 2
            ):
                pipeline.NamedBarrier(
                    barrier_id=11, num_threads=self.work_sync_threads
                ).arrive_and_wait()
            else:
                if cutlass.const_expr(self.original_heads_per_kv == 12):
                    pipeline.NamedBarrier(
                        barrier_id=11, num_threads=self.work_sync_threads
                    ).arrive_and_wait()
                else:
                    pipeline.NamedBarrier(
                        barrier_id=13, num_threads=512
                    ).arrive_and_wait()
        else:
            cute.arch.sync_threads()

    @cute.jit
    def _sync_cluster_work(self):
        if cutlass.const_expr(self.work_sync_threads != 0):
            if cutlass.const_expr(
                self.original_heads_per_kv == 12 and self.problem_shape[2] == 2
            ):
                self._sync_work_threads()
            else:
                if cutlass.const_expr(self.original_heads_per_kv == 12):
                    self._sync_work_threads()
                else:
                    pipeline.NamedBarrier(
                        barrier_id=13, num_threads=512
                    ).arrive_and_wait()
        else:
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

    @cute.jit
    def _bucket_prefix_in_cta(
        self,
        selected_slots: cute.Tensor,
        row: cutlass.Int32,
        ordered: cute.Tensor,
        scratch: cute.Tensor,
        bucket_shift: cutlass.Int32,
        tid: cutlass.Int32,
    ):
        first_slot = cutlass.Int32(0)
        if tid < 512:
            first_slot = selected_slots[row, tid * 4]
            aligned = (first_slot > 0) & ((first_slot & 3) == 0)
            for i in cutlass.range_constexpr(1, 4):
                aligned = aligned & (selected_slots[row, tid * 4 + i] == first_slot + i)
            warp_aligned = cute.arch.vote_all_sync(aligned)
            if tid % 32 == 0:
                scratch[tid // 32] = cutlass.Int32(warp_aligned)
        if cutlass.const_expr(
            self.original_heads_per_kv == 12 and self.problem_shape[2] == 2
        ):
            self._sync_work_threads()
        else:
            if cutlass.const_expr(self.original_heads_per_kv == 12):
                self._sync_work_threads()
            else:
                cute.arch.sync_threads()
        if tid < 32:
            value = cutlass.Int32(1)
            if tid < 16:
                value = scratch[tid]
            all_aligned = cute.arch.vote_all_sync(value != 0)
            if tid == 0:
                scratch[576] = cutlass.Int32(all_aligned)
        if cutlass.const_expr(
            self.original_heads_per_kv == 12 and self.problem_shape[2] == 2
        ):
            self._sync_work_threads()
        else:
            if cutlass.const_expr(self.original_heads_per_kv == 12):
                self._sync_work_threads()
            else:
                cute.arch.sync_threads()
        eligible = scratch[576] != 0
        if cutlass.const_expr(
            self.original_heads_per_kv == 12 and self.problem_shape[2] == 2
        ):
            self._sync_work_threads()
        else:
            if cutlass.const_expr(self.original_heads_per_kv == 12):
                self._sync_work_threads()
            else:
                cute.arch.sync_threads()
        if eligible:
            if tid < 512:
                scratch[tid] = cutlass.Int32(0)
                if tid < 64:
                    scratch[512 + tid] = cutlass.Int32(0)
            if cutlass.const_expr(
                self.original_heads_per_kv == 12 and self.problem_shape[2] == 2
            ):
                self._sync_work_threads()
            else:
                if cutlass.const_expr(self.original_heads_per_kv == 12):
                    self._sync_work_threads()
                else:
                    cute.arch.sync_threads()
            bucket = cutlass.Int32(0)
            lane_rank = cutlass.Int32(0)
            if tid < 512:
                bucket = (first_slot >> bucket_shift) & 31
                lane = tid % 32
                matches = cute.arch.match_sync(-1, bucket, kind="any")
                lower_lanes = (
                    cutlass.Uint32(1) << cutlass.Uint32(lane)
                ) - cutlass.Uint32(1)
                lane_rank = cutlass.Int32(cute.arch.popc(matches & lower_lanes))
                if lane_rank == 0:
                    scratch[bucket * 17 + tid // 32] = cutlass.Int32(
                        cute.arch.popc(matches)
                    )
            if cutlass.const_expr(
                self.original_heads_per_kv == 12 and self.problem_shape[2] == 2
            ):
                self._sync_work_threads()
            else:
                if cutlass.const_expr(self.original_heads_per_kv == 12):
                    self._sync_work_threads()
                else:
                    cute.arch.sync_threads()
            if tid < 512:
                # Each half warp scans the sixteen source-warp counts of one
                # bucket. The 17-word row avoids bucket-column bank conflicts.
                scan_bucket = tid // 16
                scan_warp = tid % 16
                count = scratch[scan_bucket * 17 + scan_warp]
                prefix = count
                for log_distance in cutlass.range_constexpr(4):
                    distance = 1 << log_distance
                    peer = cute.arch.shuffle_sync_up(prefix, distance)
                    if scan_warp >= distance:
                        prefix = prefix + peer
                scratch[scan_bucket * 17 + scan_warp] = prefix - count
                if scan_warp == 15:
                    scratch[544 + scan_bucket] = prefix
            if cutlass.const_expr(
                self.original_heads_per_kv == 12 and self.problem_shape[2] == 2
            ):
                self._sync_work_threads()
            else:
                if cutlass.const_expr(self.original_heads_per_kv == 12):
                    self._sync_work_threads()
                else:
                    cute.arch.sync_threads()
            if tid < 32:
                count = scratch[544 + tid]
                prefix = count
                for log_distance in cutlass.range_constexpr(5):
                    distance = 1 << log_distance
                    peer = cute.arch.shuffle_sync_up(prefix, distance)
                    if tid >= distance:
                        prefix = prefix + peer
                scratch[544 + tid] = prefix - count
            if cutlass.const_expr(
                self.original_heads_per_kv == 12 and self.problem_shape[2] == 2
            ):
                self._sync_work_threads()
            else:
                if cutlass.const_expr(self.original_heads_per_kv == 12):
                    self._sync_work_threads()
                else:
                    cute.arch.sync_threads()
            if tid < 512:
                position = (
                    scratch[544 + bucket] + scratch[bucket * 17 + tid // 32] + lane_rank
                )
                store_smem_prefix_block(ordered.iterator + position * 4, first_slot)
        if cutlass.const_expr(
            self.original_heads_per_kv == 12 and self.problem_shape[2] == 2
        ):
            self._sync_work_threads()
        else:
            if cutlass.const_expr(self.original_heads_per_kv == 12):
                self._sync_work_threads()
            else:
                cute.arch.sync_threads()
        return eligible

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
            slot = cutlass.Int32(_PADDING_READ_SLOT)
            if lane_idx % 8 == 0 and selected_idx < selected_slots.shape[1]:
                candidate = selected_slots[batch_idx, selected_idx]
                if candidate > 0:
                    slot = candidate
            slot = cute.arch.shuffle_sync(slot, lane_idx - lane_idx % 8)
            source = (
                cache_iter
                + cutlass.Int64(slot) * self.problem_shape[2] * _HEAD_DIM
                + dim_offset
                + col
            )
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
    def _load_sparse_bf16_slot(
        self,
        selected_slots: cute.Tensor,
        batch_idx: cutlass.Int32,
        selected_tile: cutlass.Int32,
        warpgroup_tidx: cutlass.Int32,
        lane_idx: cutlass.Int32,
    ):
        index_row = (
            (warpgroup_tidx // warp_threads) * 2 + lane_idx % 2 + (lane_idx // 2) * 8
        )
        index_column = selected_tile * 128 + index_row
        raw_slot = cutlass.Int32(_PADDING_READ_SLOT)
        if index_column < selected_slots.shape[1]:
            raw_slot = selected_slots[batch_idx, index_column]
        return raw_slot

    @cute.jit
    def _issue_sparse_tile_bf16_loaded(
        self,
        cache_iter: cute.Pointer,
        raw_slot: cutlass.Int32,
        dim_offset: cutlass.Int32,
        smem_base: cutlass.Int64,
        smem_stage: cutlass.Int32,
        warpgroup_tidx: cutlass.Int32,
        lane_idx: cutlass.Int32,
        kv_smem_dtype: Type[cutlass.Numeric],
    ):
        lane_slot = raw_slot if raw_slot > 0 else cutlass.Int32(_PADDING_READ_SLOT)
        base_row = warpgroup_tidx // 16
        col = (warpgroup_tidx - base_row * 16) * 8
        column_source = cache_iter + dim_offset + col
        logical_base = (
            smem_base
            + base_row * 128
            + (col % 64) * 2
            + (col // 64) * 16384
            + smem_stage * 32768
        )
        swizzled_base = logical_base ^ ((logical_base >> 3) & 0x70)
        if cutlass.const_expr(
            self.original_heads_per_kv == 12 and self.problem_shape[2] == 2
        ):
            for iteration in cutlass.range(
                16, unroll=2 if self.head_tile in (72, 96) else 4
            ):
                slot = cute.arch.shuffle_sync(lane_slot, iteration * 2 + lane_idx // 16)
                source = (
                    column_source
                    + cutlass.Int64(slot) * self.problem_shape[2] * _HEAD_DIM
                )
                destination = cute.make_ptr(
                    kv_smem_dtype,
                    swizzled_base + iteration * 1024,
                    cute.AddressSpace.smem,
                    assumed_align=16,
                )
                copy_gmem_to_smem_u128(destination, source)
        else:
            if cutlass.const_expr(self.original_heads_per_kv == 12):
                for iteration in cutlass.range(16, unroll=2):
                    slot = cute.arch.shuffle_sync(
                        lane_slot, iteration * 2 + lane_idx // 16
                    )
                    source = (
                        column_source
                        + cutlass.Int64(slot) * self.problem_shape[2] * _HEAD_DIM
                    )
                    destination = cute.make_ptr(
                        kv_smem_dtype,
                        swizzled_base + iteration * 1024,
                        cute.AddressSpace.smem,
                        assumed_align=16,
                    )
                    copy_gmem_to_smem_u128(destination, source)
            else:
                for iteration in cutlass.range(16, unroll=4):
                    slot = cute.arch.shuffle_sync(
                        lane_slot, iteration * 2 + lane_idx // 16
                    )
                    source = (
                        column_source
                        + cutlass.Int64(slot) * self.problem_shape[2] * _HEAD_DIM
                    )
                    destination = cute.make_ptr(
                        kv_smem_dtype,
                        swizzled_base + iteration * 1024,
                        cute.AddressSpace.smem,
                        assumed_align=16,
                    )
                    copy_gmem_to_smem_u128(destination, source)
        cute.arch.cp_async_commit_group()

    @cute.jit
    def _issue_sparse_stream_item_bf16_loaded(
        self,
        k_iter: cute.Pointer,
        v_iter: cute.Pointer,
        raw_slot: cutlass.Int32,
        stream_item: cutlass.Constexpr[int],
        stream_length: cutlass.Constexpr[int],
        ring_depth: cutlass.Constexpr[int],
        dim_offset: cutlass.Int32,
        smem_base: cutlass.Int64,
        convert_phase: cutlass.Int32,
        warpgroup_tidx: cutlass.Int32,
        lane_idx: cutlass.Int32,
        kv_smem_dtype: Type[cutlass.Numeric],
    ):
        if cutlass.const_expr(
            stream_item == 0
            or (stream_item % 2 == 1 and stream_item != stream_length - 1)
        ):
            cache_iter = k_iter
        else:
            cache_iter = v_iter
        self._issue_sparse_tile_bf16_loaded(
            cache_iter,
            raw_slot,
            dim_offset,
            smem_base,
            (stream_item % ring_depth) * _CONVERT_WARPGROUPS + convert_phase,
            warpgroup_tidx,
            lane_idx,
            kv_smem_dtype,
        )

    @cute.jit
    def _issue_sparse_tile_bf16_full_loaded(
        self,
        cache_iter: cute.Pointer,
        raw_slot: cutlass.Int32,
        smem_base: cutlass.Int64,
        smem_stage: cutlass.Int32,
        second_smem_stage: cutlass.Int32,
        warpgroup_tidx: cutlass.Int32,
        lane_idx: cutlass.Int32,
        kv_smem_dtype: Type[cutlass.Numeric],
    ):
        if cutlass.const_expr(
            self.original_heads_per_kv == 12 and self.problem_shape[2] == 2
        ):
            """Copy both D128 halves using one cached slot and source address."""
            lane_slot = raw_slot if raw_slot > 0 else cutlass.Int32(_PADDING_READ_SLOT)
            for iteration in cutlass.range(16, unroll=2):
                vector = warpgroup_tidx + iteration * warpgroup_threads
                row = vector // 16
                col = (vector - row * 16) * 8
                slot = cute.arch.shuffle_sync(lane_slot, iteration * 2 + lane_idx // 16)
                source = (
                    cache_iter
                    + cutlass.Int64(slot) * self.problem_shape[2] * _HEAD_DIM
                    + col
                )
                logical_address = (
                    smem_base
                    + row * 128
                    + (col % 64) * 2
                    + (col // 64) * 16384
                    + smem_stage * 32768
                )
                destination = cute.make_ptr(
                    kv_smem_dtype,
                    logical_address ^ ((logical_address >> 3) & 0x70),
                    cute.AddressSpace.smem,
                    assumed_align=16,
                )
                copy_gmem_to_smem_u128(destination, source)
                second_address = (
                    logical_address + (second_smem_stage - smem_stage) * 32768
                )
                second_destination = cute.make_ptr(
                    kv_smem_dtype,
                    second_address ^ ((second_address >> 3) & 0x70),
                    cute.AddressSpace.smem,
                    assumed_align=16,
                )
                copy_gmem_to_smem_u128(second_destination, source + 128)
            cute.arch.cp_async_commit_group()
        else:
            if cutlass.const_expr(self.original_heads_per_kv == 12):
                """Copy both D128 halves using one cached slot and source address."""
                lane_slot = (
                    raw_slot if raw_slot > 0 else cutlass.Int32(_PADDING_READ_SLOT)
                )
                base_row = warpgroup_tidx // 16
                col = (warpgroup_tidx - base_row * 16) * 8
                column_source = cache_iter + col
                logical_base = (
                    smem_base
                    + base_row * 128
                    + (col % 64) * 2
                    + (col // 64) * 16384
                    + smem_stage * 32768
                )
                swizzled_base = logical_base ^ ((logical_base >> 3) & 0x70)
                second_delta = (second_smem_stage - smem_stage) * 32768
                for iteration in cutlass.range(16, unroll=2):
                    slot = cute.arch.shuffle_sync(
                        lane_slot, iteration * 2 + lane_idx // 16
                    )
                    source = (
                        column_source
                        + cutlass.Int64(slot) * self.problem_shape[2] * _HEAD_DIM
                    )
                    # Both offsets are multiples of1024: the swizzle's source/target
                    # bits stay below bit10, so its mask is invariant over rounds/stages.
                    destination = cute.make_ptr(
                        kv_smem_dtype,
                        swizzled_base + iteration * 1024,
                        cute.AddressSpace.smem,
                        assumed_align=16,
                    )
                    copy_gmem_to_smem_u128(destination, source)
                    second_destination = cute.make_ptr(
                        kv_smem_dtype,
                        swizzled_base + iteration * 1024 + second_delta,
                        cute.AddressSpace.smem,
                        assumed_align=16,
                    )
                    copy_gmem_to_smem_u128(second_destination, source + 128)
                cute.arch.cp_async_commit_group()

    @cute.jit
    def _load_staged_query_bf16(
        self,
        query_smem: cute.Pointer,
        head: cutlass.Int32,
        dim: cutlass.Int32,
        heads_per_tile: cutlass.Constexpr[int],
    ):
        # Scalar pointer loads need the explicit SW128 address transform.
        # Tensor-coordinate loads apply it through the composed Q layout.
        index = head * 64 + dim % 64 + (dim // 64) * heads_per_tile * 64
        logical = query_smem.toint() + cutlass.Int64(index * 2)
        address = logical ^ ((logical >> 3) & 0x70)
        pointer = cute.make_ptr(
            cutlass.BFloat16, address, cute.AddressSpace.smem, assumed_align=2
        )
        return pointer[0]

    @cute.jit
    def _prefetch_sparse_stream_item_bf16_full_loaded(
        self,
        k_iter: cute.Pointer,
        v_iter: cute.Pointer,
        raw_slot: cutlass.Int32,
        stream_item: cutlass.Constexpr[int],
        stream_length: cutlass.Constexpr[int],
    ):
        if cutlass.const_expr(
            stream_item == 0
            or (stream_item % 2 == 1 and stream_item != stream_length - 1)
        ):
            cache_iter = k_iter
        else:
            cache_iter = v_iter
        slot = raw_slot if raw_slot > 0 else cutlass.Int32(_PADDING_READ_SLOT)
        source = cache_iter + cutlass.Int64(slot) * self.problem_shape[2] * _HEAD_DIM
        # Four 128-byte lines cover one D256 BF16 row. Copies own readiness.
        prefetch_gmem_l2(source, loc=None, ip=None)
        prefetch_gmem_l2(source + 64, loc=None, ip=None)
        prefetch_gmem_l2(source + 128, loc=None, ip=None)
        prefetch_gmem_l2(source + 192, loc=None, ip=None)

    @cute.jit
    def _issue_sparse_stream_item_bf16_full_loaded(
        self,
        k_iter: cute.Pointer,
        v_iter: cute.Pointer,
        raw_slot: cutlass.Int32,
        stream_item: cutlass.Constexpr[int],
        stream_length: cutlass.Constexpr[int],
        ring_depth: cutlass.Constexpr[int],
        smem_base: cutlass.Int64,
        warpgroup_tidx: cutlass.Int32,
        lane_idx: cutlass.Int32,
        kv_smem_dtype: Type[cutlass.Numeric],
    ):
        if cutlass.const_expr(
            self.original_heads_per_kv == 12 and self.problem_shape[2] == 2
        ):
            if cutlass.const_expr(
                stream_item == 0
                or (stream_item % 2 == 1 and stream_item != stream_length - 1)
            ):
                cache_iter = k_iter
            else:
                cache_iter = v_iter
            first_stage = (stream_item % ring_depth) * _CONVERT_WARPGROUPS
            self._issue_sparse_tile_bf16_full_loaded(
                cache_iter,
                raw_slot,
                smem_base,
                first_stage,
                first_stage + 1,
                warpgroup_tidx,
                lane_idx,
                kv_smem_dtype,
            )
        else:
            if cutlass.const_expr(self.original_heads_per_kv == 12):
                if cutlass.const_expr(
                    stream_item == 0
                    or (stream_item % 2 == 1 and stream_item != stream_length - 1)
                ):
                    cache_iter = k_iter
                else:
                    cache_iter = v_iter
                first_stage = (stream_item % ring_depth) * _CONVERT_WARPGROUPS
                self._issue_sparse_tile_bf16_full_loaded(
                    cache_iter,
                    raw_slot,
                    smem_base,
                    first_stage,
                    first_stage + 1,
                    warpgroup_tidx,
                    lane_idx,
                    kv_smem_dtype,
                )

    @cute.jit
    def _prepare_private_tail(
        self,
        q_source: cute.Pointer,
        k_iter: cute.Pointer,
        v_iter: cute.Pointer,
        selected_slots: cute.Tensor,
        partial: cute.Tensor,
        row_idx: cutlass.Int32,
        q_head_offset: cutlass.Int32,
        warp_idx: cutlass.Int32,
        lane_idx: cutlass.Int32,
        scale_qs: cutlass.Float32,
    ):
        if cutlass.const_expr(
            self.original_heads_per_kv == 12 and self.problem_shape[2] == 2
        ):
            if (
                warp_idx >= self.private_tail_start_warp
                and warp_idx < self.private_tail_start_warp + self.private_tail_warps
            ):
                for head_round in cutlass.range_constexpr(
                    cute.ceil_div(self.original_heads_per_kv, self.private_tail_warps)
                ):
                    head = (
                        warp_idx
                        - self.private_tail_start_warp
                        + head_round * self.private_tail_warps
                    )
                    if head < self.original_heads_per_kv:
                        q_head = q_head_offset + head
                        query_values = cute.make_rmem_tensor(8, cutlass.Float32)
                        numerator = cute.make_rmem_tensor(8, cutlass.Float32)
                        numerator.fill(cutlass.Float32(0.0))
                        for part in cutlass.range_constexpr(8):
                            dim = lane_idx + part * warp_threads
                            query_values[part] = cutlass.Float32(
                                q_source[
                                    (
                                        cutlass.Int64(row_idx) * self.problem_shape[1]
                                        + q_head
                                    )
                                    * _HEAD_DIM
                                    + dim
                                ]
                            )
                        maximum = -cutlass.Float32.inf
                        denominator = cutlass.Float32(0.0)
                        for tail in cutlass.range_constexpr(3):
                            slot = cutlass.Int64(selected_slots[row_idx, 2048 + tail])
                            if slot > 0:
                                dot = cutlass.Float32(0.0)
                                for part in cutlass.range_constexpr(8):
                                    dim = lane_idx + part * warp_threads
                                    dot += query_values[part] * cutlass.Float32(
                                        k_iter[
                                            slot * self.problem_shape[2] * _HEAD_DIM
                                            + dim
                                        ]
                                    )
                                score = cute.arch.warp_reduction_sum(dot) * scale_qs
                                new_maximum = cute.arch.fmax(maximum, score)
                                correction = exp2(log2_e * (maximum - new_maximum))
                                weight = exp2(log2_e * (score - new_maximum))
                                denominator = denominator * correction + weight
                                for part in cutlass.range_constexpr(8):
                                    dim = lane_idx + part * warp_threads
                                    numerator[part] = numerator[
                                        part
                                    ] * correction + weight * cutlass.Float32(
                                        v_iter[
                                            slot * self.problem_shape[2] * _HEAD_DIM
                                            + dim
                                        ]
                                    )
                                maximum = new_maximum
                        for part in cutlass.range_constexpr(8):
                            partial[head, lane_idx + part * warp_threads] = numerator[
                                part
                            ]
                        if lane_idx == 0:
                            partial[head, 256] = maximum
                            partial[head, 257] = denominator
        else:
            if cutlass.const_expr(self.original_heads_per_kv == 12):
                extra_warp_start = (
                    self.threads_per_cta // warp_threads - self.private_tail_extra_warps
                )
                is_worker = warp_idx == self.private_tail_warp
                if cutlass.const_expr(self.private_tail_extra_warps > 0):
                    is_worker = is_worker or warp_idx >= extra_warp_start
                if is_worker:
                    worker = cutlass.Int32(0)
                    if cutlass.const_expr(self.private_tail_extra_warps > 0):
                        if warp_idx >= extra_warp_start:
                            worker = warp_idx - extra_warp_start + 1
                    heads_per_query = (
                        self.problem_shape[1]
                        // self.problem_shape[2]
                        // self.query_group
                    )
                    kv_head = q_head_offset // (heads_per_query * self.query_group)
                    original_q_heads = (
                        self.original_heads_per_kv * self.problem_shape[2]
                    )
                    for head_round in cutlass.range(
                        cute.ceil_div(
                            self.original_heads_per_kv * self.query_group,
                            self.private_tail_workers,
                        ),
                        unroll=1,
                    ):
                        head = worker + head_round * self.private_tail_workers
                        if head < self.original_heads_per_kv * self.query_group:
                            # Each grouped query retains its original Q row and private slots.
                            original_row = (
                                row_idx * self.query_group + head // heads_per_query
                            )
                            original_head = (
                                kv_head * self.original_heads_per_kv
                                + head % heads_per_query
                            )
                            query_values = cute.make_rmem_tensor(8, cutlass.Float32)
                            numerator = cute.make_rmem_tensor(8, cutlass.Float32)
                            numerator.fill(cutlass.Float32(0.0))
                            for part in cutlass.range_constexpr(8):
                                dim = lane_idx + part * warp_threads
                                query_values[part] = cutlass.Float32(
                                    q_source[
                                        (
                                            cutlass.Int64(original_row)
                                            * original_q_heads
                                            + original_head
                                        )
                                        * _HEAD_DIM
                                        + dim
                                    ]
                                )
                            maximum = -cutlass.Float32.inf
                            denominator = cutlass.Float32(0.0)
                            for tail in cutlass.range_constexpr(3):
                                slot = cutlass.Int64(
                                    selected_slots[original_row, 2048 + tail]
                                )
                                if slot > 0:
                                    dot = cutlass.Float32(0.0)
                                    for part in cutlass.range_constexpr(8):
                                        dim = lane_idx + part * warp_threads
                                        dot += query_values[part] * cutlass.Float32(
                                            k_iter[
                                                slot * self.problem_shape[2] * _HEAD_DIM
                                                + dim
                                            ]
                                        )
                                    score = cute.arch.warp_reduction_sum(dot) * scale_qs
                                    new_maximum = cute.arch.fmax(maximum, score)
                                    correction = exp2(log2_e * (maximum - new_maximum))
                                    weight = exp2(log2_e * (score - new_maximum))
                                    denominator = denominator * correction + weight
                                    for part in cutlass.range_constexpr(8):
                                        dim = lane_idx + part * warp_threads
                                        numerator[part] = numerator[
                                            part
                                        ] * correction + weight * cutlass.Float32(
                                            v_iter[
                                                slot * self.problem_shape[2] * _HEAD_DIM
                                                + dim
                                            ]
                                        )
                                    maximum = new_maximum
                            for part in cutlass.range_constexpr(8):
                                partial[head, lane_idx + part * warp_threads] = (
                                    numerator[part]
                                )
                            if lane_idx == 0:
                                partial[head, 256] = maximum
                                partial[head, 257] = denominator
            else:
                worker = warp_idx - self.private_tail_start_warp
                if worker >= 0 and worker < self.private_tail_workers:
                    # Two available warps each own three heads; larger launches keep
                    # one head per warp. A head's arithmetic and private slots agree.
                    for head_round in cutlass.range(
                        self.original_heads_per_kv // self.private_tail_workers,
                        unroll=1,
                    ):
                        head = worker + head_round * self.private_tail_workers
                        q_head = q_head_offset + head
                        query_values = cute.make_rmem_tensor(8, cutlass.Float32)
                        numerator = cute.make_rmem_tensor(8, cutlass.Float32)
                        numerator.fill(cutlass.Float32(0.0))
                        for part in cutlass.range_constexpr(8):
                            dim = lane_idx + part * warp_threads
                            query_values[part] = cutlass.Float32(
                                q_source[
                                    (
                                        cutlass.Int64(row_idx) * self.problem_shape[1]
                                        + q_head
                                    )
                                    * _HEAD_DIM
                                    + dim
                                ]
                            )
                        maximum = -cutlass.Float32.inf
                        denominator = cutlass.Float32(0.0)
                        for tail in cutlass.range_constexpr(3):
                            slot = cutlass.Int64(selected_slots[row_idx, 2048 + tail])
                            if slot > 0:
                                dot = cutlass.Float32(0.0)
                                for part in cutlass.range_constexpr(8):
                                    dim = lane_idx + part * warp_threads
                                    dot += query_values[part] * cutlass.Float32(
                                        k_iter[
                                            slot * self.problem_shape[2] * _HEAD_DIM
                                            + dim
                                        ]
                                    )
                                score = cute.arch.warp_reduction_sum(dot) * scale_qs
                                new_maximum = cute.arch.fmax(maximum, score)
                                correction = exp2(log2_e * (maximum - new_maximum))
                                weight = exp2(log2_e * (score - new_maximum))
                                denominator = denominator * correction + weight
                                for part in cutlass.range_constexpr(8):
                                    dim = lane_idx + part * warp_threads
                                    numerator[part] = numerator[
                                        part
                                    ] * correction + weight * cutlass.Float32(
                                        v_iter[
                                            slot * self.problem_shape[2] * _HEAD_DIM
                                            + dim
                                        ]
                                    )
                                maximum = new_maximum
                        for part in cutlass.range_constexpr(8):
                            partial[head, lane_idx + part * warp_threads] = numerator[
                                part
                            ]
                        if lane_idx == 0:
                            partial[head, 256] = maximum
                            partial[head, 257] = denominator

    @cute.jit
    def _issue_sparse_tile_bf16(
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
        """Gather one 128x128 BF16 tile with aligned 16-byte copies."""
        raw_slot = self._load_sparse_bf16_slot(
            selected_slots, batch_idx, selected_tile, warpgroup_tidx, lane_idx
        )
        self._issue_sparse_tile_bf16_loaded(
            cache_iter,
            raw_slot,
            dim_offset,
            smem_base,
            smem_stage,
            warpgroup_tidx,
            lane_idx,
            kv_smem_dtype,
        )

    @cute.jit
    def _issue_sparse_tile_for_dtype(
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
        if cutlass.const_expr(cache_iter.dtype is cutlass.BFloat16):
            self._issue_sparse_tile_bf16(
                cache_iter,
                selected_slots,
                batch_idx,
                selected_tile,
                dim_offset,
                smem_base,
                smem_stage,
                warpgroup_tidx,
                lane_idx,
                kv_smem_dtype,
            )
        else:
            self._issue_sparse_tile(
                cache_iter,
                selected_slots,
                batch_idx,
                selected_tile,
                dim_offset,
                smem_base,
                smem_stage,
                warpgroup_tidx,
                lane_idx,
                kv_smem_dtype,
            )

    @cute.jit
    def _prefetch_sparse_stream_item(
        self,
        k_iter: cute.Pointer,
        v_iter: cute.Pointer,
        selected_slots: cute.Tensor,
        row_idx: cutlass.Int32,
        kv_split_idx: cutlass.Int32,
        kv_splits: cutlass.Int32,
        stream_item: cutlass.Constexpr[int],
        stream_length: cutlass.Constexpr[int],
        dim_offset: cutlass.Int32,
        warpgroup_tidx: cutlass.Int32,
    ):
        if cutlass.const_expr(stream_item == 0):
            cache_iter = k_iter
            selected_tile = kv_split_idx
        elif cutlass.const_expr(stream_item == stream_length - 1):
            cache_iter = v_iter
            selected_tile = kv_split_idx + (stream_length // 2 - 1) * kv_splits
        elif cutlass.const_expr(stream_item % 2 == 1):
            cache_iter = k_iter
            selected_tile = kv_split_idx + ((stream_item + 1) // 2) * kv_splits
        else:
            cache_iter = v_iter
            selected_tile = kv_split_idx + (stream_item // 2 - 1) * kv_splits
        selected_idx = selected_tile * 128 + warpgroup_tidx
        slot = cutlass.Int32(_PADDING_READ_SLOT)
        if selected_idx < selected_slots.shape[1]:
            candidate = selected_slots[row_idx, selected_idx]
            if candidate > 0:
                slot = candidate
        source = (
            cache_iter
            + cutlass.Int64(slot) * self.problem_shape[2] * _HEAD_DIM
            + dim_offset
        )
        prefetch_gmem_l2(source, loc=None, ip=None)
        prefetch_gmem_l2(source + 64, loc=None, ip=None)

    def _distinct_query_config(self, original_query, key_cache):
        """Keep independent prefixes on the measured single-query geometry."""
        kv_heads = key_cache.shape[1]
        bf16_kv = key_cache.element_type is cutlass.BFloat16
        eligible = (
            self.kv_splits == 1
            and self.query_rows_per_cta == 1
            and self.distinct_bf16_smem_slots == 3
            and self.bf16_async_ready
            and bf16_kv
        )
        bucket_prefix = (
            eligible
            and self.original_heads_per_kv == 12
            and (kv_heads == 1 or self.query_group != 2 or self.large_query_grid)
        )
        single_work512 = (
            eligible
            and self.original_heads_per_kv == 6
            and kv_heads == 1
            and self.query_group in (3, 4)
            and self.threads_per_cta == 640
        )
        native_head_tile = self.head_tile in (48, 72, 96)
        if self.original_heads_per_kv == 12 and kv_heads == 2:
            register_budget = (
                (128, 56, 80)
                if bucket_prefix
                else (128, 48, 80) if native_head_tile else None
            )
            launch_threads = (
                512
                if bucket_prefix
                else self.threads_per_cta if native_head_tile else None
            )
        elif self.original_heads_per_kv == 12:
            register_budget = (
                (128, 48, 72)
                if self.head_tile == 48
                else (
                    (128, 80, 112)
                    if self.head_tile == 32 and self.query_group == 2
                    else None
                )
            )
            launch_threads = None
        else:
            register_budget = None
            launch_threads = None
        return MixedInputFusedMultiHeadAttentionDecode(
            self.kv_splits,
            self.enable_pdl,
            self.query_rows_per_cta,
            self.distinct_bf16_smem_slots,
            self.bf16_async_ready,
            _select_head_tile(self.original_heads_per_kv, bf16_kv, 1),
            (
                None,
                original_query.shape[1],
                kv_heads,
                key_cache.shape[0],
                _HEAD_DIM,
            ),
            1,
            self.original_heads_per_kv,
            True,
            False,
            1,
            self.device_sm_count,
            register_budget=register_budget,
            distinct_bf16_smem_slots=self.distinct_bf16_smem_slots,
            launch_threads_per_cta=launch_threads,
            scalar_softmax_stats=False,
            direct_query=False,
            bucket_prefix=bucket_prefix,
            leader_first=False,
            producer_warpgroups=(
                1
                if eligible and self.original_heads_per_kv == 12
                else _CONVERT_WARPGROUPS
            ),
            private_tail_workers=(
                (2 if single_work512 else 6 if self.threads_per_cta >= 640 else 2)
                if eligible and self.original_heads_per_kv == 6 and kv_heads == 1
                else 0
            ),
            cached_bf16_lookahead=(
                eligible
                and self.query_group == 4
                and self.threads_per_cta >= 640
                and self.original_heads_per_kv == 6
                and kv_heads == 1
            ),
            work_sync_threads=(
                512
                if single_work512
                or (bucket_prefix and kv_heads == 1 and self.head_tile == 48)
                else 0
            ),
            prefix_in_cta=self.prefix_in_cta,
            large_query_grid=self.large_query_grid,
        )

    @cute.jit
    def __call__(
        self,
        q_tensor: cute.Tensor,
        original_query: cute.Tensor,
        k_tensor: cute.Tensor,
        v_tensor: cute.Tensor,
        selected_slots: cute.Tensor,
        prefix_shared: cute.Tensor,
        output_tensor: cute.Tensor,
        scale_qs: cutlass.Float32,
        scale_o: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        work_args = self._work_arguments(
            q_tensor,
            k_tensor,
            v_tensor,
            selected_slots,
            prefix_shared,
            output_tensor,
            scale_qs,
            scale_o,
        )
        if cutlass.const_expr(self.caller_allocates_scratch):
            single_config = self._distinct_query_config(original_query, k_tensor)
            single_args = single_config._work_arguments(
                original_query,
                k_tensor,
                v_tensor,
                selected_slots,
                prefix_shared,
                output_tensor,
                scale_qs,
                scale_o,
            )
            # Both exact-prefix reuse and independent queries borrow this arena.
            self.scratch_arena_bytes = max(
                self.scratch_arena_bytes, single_config.scratch_arena_bytes
            )
            self.decode_adaptive(single_config, *single_args, *work_args).launch(
                grid=(
                    1,
                    1,
                    original_query.shape[0] * self.problem_shape[2],
                ),
                block=[self.threads_per_cta, 1, 1],
                max_number_threads=[self.threads_per_cta, 1, 1],
                cluster=[1, 1, 1],
                stream=stream,
                min_blocks_per_mp=1,
                use_pdl=self.enable_pdl,
            )
        else:
            _, h_q, h_k, _, _ = self.problem_shape
            b = original_query.shape[0] // self.query_group
            self.decode(*work_args).launch(
                grid=(
                    self.kv_splits,
                    cute.ceil_div(h_q // h_k, self.head_tile),
                    cute.ceil_div(b, self.query_rows_per_cta) * h_k,
                ),
                block=[self.threads_per_cta, 1, 1],
                cluster=[self.kv_splits, 1, 1],
                stream=stream,
                min_blocks_per_mp=(
                    2
                    if cutlass.const_expr(
                        k_tensor.element_type is cutlass.BFloat16
                        and self.bf16_smem_slots == 1
                    )
                    else 1
                ),
                use_pdl=self.enable_pdl,
            )

    @cute.jit
    def _work_arguments(
        self,
        q_tensor: cute.Tensor,
        k_tensor: cute.Tensor,
        v_tensor: cute.Tensor,
        selected_slots: cute.Tensor,
        prefix_shared: cute.Tensor,
        output_tensor: cute.Tensor,
        scale_qs: cutlass.Float32,
        scale_o: cutlass.Float32,
    ):
        problem_shape = self.problem_shape
        kv_splits = self.kv_splits
        q_iter = q_tensor.iterator
        k_iter = k_tensor.iterator
        v_iter = v_tensor.iterator
        o_iter = output_tensor.iterator
        mma_dtype = q_iter.dtype
        acc_dtype = cutlass.Float32
        blk_tile_s = 128
        blk_tile_h = self.head_tile
        blk_tile_d = _HEAD_DIM
        blk_tile_shd = (blk_tile_s, blk_tile_h, blk_tile_d)
        mma_tile_m = 128
        mma_tile_n = self.head_tile
        mma_tile_k = 128
        mma_tile_mnk = (mma_tile_m, mma_tile_n, mma_tile_k)
        bf16_kv = k_iter.dtype is cutlass.BFloat16
        dual_cta_bf16 = bf16_kv and self.bf16_smem_slots == 1
        if cutlass.const_expr(self.query_group > 1):
            assert self.kv_splits == 1 and self.bf16_async_ready
        self.kv_ring_stages = 1 if cutlass.const_expr(dual_cta_bf16) else 2
        kv_mma_source = (
            tcgen05.OperandSource.SMEM
            if cutlass.const_expr(bf16_kv)
            else tcgen05.OperandSource.TMEM
        )
        tiled_mma_kq = sm100_utils.make_trivial_tiled_mma(
            mma_dtype,
            mma_dtype,
            OperandMajorMode.K,  # K
            OperandMajorMode.K,  # Q
            acc_dtype,
            tcgen05.CtaGroup.ONE,
            mma_tile_mnk[:2],
            kv_mma_source,
        )
        tiled_mma_vp = sm100_utils.make_trivial_tiled_mma(  #
            mma_dtype,
            mma_dtype,
            (
                OperandMajorMode.MN
                if cutlass.const_expr(bf16_kv)
                else OperandMajorMode.K
            ),  # V
            OperandMajorMode.MN,  # P
            acc_dtype,
            tcgen05.CtaGroup.ONE,
            mma_tile_mnk[:2],
            kv_mma_source,
        )
        self.q_stages = blk_tile_d // mma_tile_k
        tmem_alloc_cols = mma_tile_n * self.sp_stages
        tmem_alloc_cols += mma_tile_n * self.o_stages * (blk_tile_d // mma_tile_m)
        tmem_capacity = 512
        cvt_stage_cols = mma_tile_k * mma_dtype.width // 32
        self.cvt_stages = (tmem_capacity - tmem_alloc_cols) // cvt_stage_cols
        self.cvt_stages = min(self.cvt_stages, 8)
        if cutlass.const_expr(not bf16_kv):
            tmem_alloc_cols += self.cvt_stages * cvt_stage_cols
        self.tmem_alloc_cols = 2 ** math.ceil(
            math.log2(tmem_alloc_cols)
        )  # Tmem alloc must be PO2
        self.mbarrier_reserved_bytes = 768
        smem_alloc_bits = self.mbarrier_reserved_bytes * 8
        smem_alloc_bits += mma_tile_n * acc_dtype.width  # colmax
        smem_alloc_bits += mma_tile_n * warpgroup_warps * acc_dtype.width  # colsum
        smem_alloc_bits += (
            mma_tile_n * mma_tile_k * self.q_stages * mma_dtype.width
        )  # Q
        smem_alloc_bits += (
            mma_tile_m * mma_tile_n * self.sp_stages * mma_dtype.width
        )  # P
        smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        kv_smem_dtype = k_iter.dtype
        self.kv_stages = (smem_capacity * 8 - smem_alloc_bits) // (
            mma_tile_m * mma_tile_k * kv_smem_dtype.width
        )
        self.kv_stages = min(self.kv_stages, 8)
        if cutlass.const_expr(bf16_kv):
            self.kv_stages = self.bf16_smem_slots * _CONVERT_WARPGROUPS
        assert self.kv_stages >= self.kv_ring_stages * _CONVERT_WARPGROUPS
        if cutlass.const_expr(bf16_kv):
            # BF16 tiles are already in the MMA input format. Their shared
            # stages now stay owned until the corresponding MMA completes.
            self.cvt_stages = self.kv_stages
        _, h_q, h_k, s_k, d = problem_shape
        b = output_tensor.shape[0] // self.query_group
        h_r = h_q // h_k
        if cutlass.const_expr(self.direct_query):
            original_hr = self.original_heads_per_kv
            original_hq = original_hr * h_k
            # Keep head and query as separate TMA dimensions. Their source
            # strides select the original Q rows without a global packed copy.
            q = cute.make_tensor(
                q_iter,
                cute.make_layout(
                    ((original_hr, self.query_group), d, (h_k, b)),
                    stride=(
                        (d, original_hq * d),
                        1,
                        (original_hr * d, self.query_group * original_hq * d),
                    ),
                ),
            )
        else:
            q = cute.make_tensor(
                q_iter,
                cute.make_ordered_layout(
                    shape=(h_r, d, (h_k, b)), order=(1, 0, (2, 3))
                ),
            )
        assert v_iter.dtype is k_iter.dtype
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
        q_mma_tile_mnk = mma_tile_mnk
        if cutlass.const_expr(self.direct_query):
            query_head_tile = h_r // self.query_group
            if cutlass.const_expr(query_head_tile != self.original_heads_per_kv):
                if cutlass.const_expr(
                    self.original_heads_per_kv == 12 and self.problem_shape[2] == 2
                ):
                    q_mma_tile_mnk = (
                        mma_tile_mnk[0],
                        (query_head_tile, self.query_group),
                        mma_tile_mnk[2],
                    )
                else:
                    if cutlass.const_expr(self.original_heads_per_kv == 12):
                        q_mma_tile_mnk = (
                            mma_tile_mnk[0],
                            (query_head_tile, self.query_group),
                            mma_tile_mnk[2],
                        )
                    else:
                        q_mma_tile_mnk = (
                            mma_tile_mnk[0],
                            (
                                query_head_tile,
                                cute.ceil_div(mma_tile_n, query_head_tile),
                            ),
                            mma_tile_mnk[2],
                        )
        tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            q,
            cute.select(smem_layout_q, mma_modes),
            q_mma_tile_mnk,
            tiled_mma_kq,
        )
        scale_qs_log2_e = scale_qs * log2_e
        n_tiles = cute.ceil_div(h_r, blk_tile_h)
        l_tiles = cute.ceil_div(b, self.query_rows_per_cta) * h_k
        original_hr = self.original_heads_per_kv
        original_hq = original_hr * h_k
        padded_hr = h_r // self.query_group
        output_layout = cute.make_layout(
            (d, ((padded_hr, self.query_group), h_k), b),
            stride=(
                1,
                ((d, original_hq * d), original_hr * d),
                self.query_group * original_hq * d,
            ),
        )
        o = cute.make_tensor(o_iter, output_layout)
        if cutlass.const_expr(
            self.original_heads_per_kv == 12 and self.problem_shape[2] == 2
        ):
            self.scratch_arena_bytes = (
                ((224 + 20 * mma_tile_n + 127) // 128) * 128
                + 2 * 128 * 128 * self.kv_stages
                + mma_tile_n * (512 + 256 * self.sp_stages)
                + (
                    self.query_group * 3 * _HEAD_DIM * 4
                    if self.prefetch_private_kv
                    else 0
                )
                + (self.head_tile * 264 * 4 if self.overlap_private_tail else 0)
            )
        else:
            self.scratch_arena_bytes = (
                ((224 + 20 * mma_tile_n + 127) // 128) * 128
                + 2 * 128 * 128 * self.kv_stages
                + mma_tile_n * 1024
                + (self.head_tile * 264 * 4 if self.overlap_private_tail else 0)
            )
        return (
            blk_tile_shd,
            mma_tile_mnk,
            tiled_mma_kq,
            tiled_mma_vp,
            q_iter.dtype,
            smem_layout_q,
            tma_atom_q,
            tma_tensor_q,
            q_iter,
            k_iter.dtype,
            smem_layout_k,
            k_iter,
            v_iter.dtype,
            smem_layout_v,
            v_iter,
            selected_slots,
            prefix_shared,
            acc_dtype,
            smem_layout_o,
            o,
            scale_qs,
            scale_qs_log2_e,
            scale_o,
        )

    @cute.kernel
    def decode(
        self,
        blk_tile_shd: cute.Tile,
        mma_tile_mnk: cute.Tile,
        tiled_mma_kq: cute.TiledMma,
        tiled_mma_vp: cute.TiledMma,
        q_dtype: Type[cutlass.Numeric],
        smem_layout_q: cute.ComposedLayout,
        tma_atom_q: cute.CopyAtom,
        mQ: cute.Tensor,
        q_source: cute.Pointer,
        k_dtype: Type[cutlass.Numeric],
        smem_layout_k: cute.ComposedLayout,
        k_iter: cute.Pointer,
        v_dtype: Type[cutlass.Numeric],
        smem_layout_v: cute.ComposedLayout,
        v_iter: cute.Pointer,
        selected_slots: cute.Tensor,
        prefix_shared: cute.Tensor,
        o_dtype: Type[cutlass.Numeric],
        smem_layout_o: cute.ComposedLayout,
        mOut: cute.Tensor,
        scale_qs: cutlass.Float32,
        scale_qs_log2_e: cutlass.Float32,
        scale_o: cutlass.Float32,
    ):
        self.decode_work(
            blk_tile_shd,
            mma_tile_mnk,
            tiled_mma_kq,
            tiled_mma_vp,
            q_dtype,
            smem_layout_q,
            tma_atom_q,
            mQ,
            q_source,
            k_dtype,
            smem_layout_k,
            k_iter,
            v_dtype,
            smem_layout_v,
            v_iter,
            selected_slots,
            prefix_shared,
            o_dtype,
            smem_layout_o,
            mOut,
            scale_qs,
            scale_qs_log2_e,
            scale_o,
            0,
            0,
            q_source,
        )

    @cute.kernel
    def decode_adaptive(
        self,
        single_config: cutlass.Constexpr,
        single_blk_tile_shd: cute.Tile,
        single_mma_tile_mnk: cute.Tile,
        single_tiled_mma_kq: cute.TiledMma,
        single_tiled_mma_vp: cute.TiledMma,
        single_q_dtype: Type[cutlass.Numeric],
        single_smem_layout_q: cute.ComposedLayout,
        single_tma_atom_q: cute.CopyAtom,
        single_mQ: cute.Tensor,
        single_q_source: cute.Pointer,
        single_k_dtype: Type[cutlass.Numeric],
        single_smem_layout_k: cute.ComposedLayout,
        single_k_iter: cute.Pointer,
        single_v_dtype: Type[cutlass.Numeric],
        single_smem_layout_v: cute.ComposedLayout,
        single_v_iter: cute.Pointer,
        single_selected_slots: cute.Tensor,
        single_prefix_shared: cute.Tensor,
        single_o_dtype: Type[cutlass.Numeric],
        single_smem_layout_o: cute.ComposedLayout,
        single_mOut: cute.Tensor,
        single_scale_qs: cutlass.Float32,
        single_scale_qs_log2_e: cutlass.Float32,
        single_scale_o: cutlass.Float32,
        pair_blk_tile_shd: cute.Tile,
        pair_mma_tile_mnk: cute.Tile,
        pair_tiled_mma_kq: cute.TiledMma,
        pair_tiled_mma_vp: cute.TiledMma,
        pair_q_dtype: Type[cutlass.Numeric],
        pair_smem_layout_q: cute.ComposedLayout,
        pair_tma_atom_q: cute.CopyAtom,
        pair_mQ: cute.Tensor,
        pair_q_source: cute.Pointer,
        pair_k_dtype: Type[cutlass.Numeric],
        pair_smem_layout_k: cute.ComposedLayout,
        pair_k_iter: cute.Pointer,
        pair_v_dtype: Type[cutlass.Numeric],
        pair_smem_layout_v: cute.ComposedLayout,
        pair_v_iter: cute.Pointer,
        pair_selected_slots: cute.Tensor,
        pair_prefix_shared: cute.Tensor,
        pair_o_dtype: Type[cutlass.Numeric],
        pair_smem_layout_o: cute.ComposedLayout,
        pair_mOut: cute.Tensor,
        pair_scale_qs: cutlass.Float32,
        pair_scale_qs_log2_e: cutlass.Float32,
        pair_scale_o: cutlass.Float32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
        _, _, query_work = cute.arch.block_idx()
        _, _, kv_heads, _, _ = self.problem_shape
        if cutlass.const_expr(
            self.original_heads_per_kv == 12 and self.problem_shape[2] == 2
        ):
            query_row = self._map_query_row(query_work, kv_heads, pair_prefix_shared)
        else:
            if cutlass.const_expr(not (self.original_heads_per_kv == 12)):
                smem = utils.SmemAllocator()
                prefix_scratch = smem.allocate_array(
                    cutlass.Int32, self.threads_per_cta // 32 + 2, byte_alignment=16
                )
            if cutlass.const_expr(self.original_heads_per_kv == 12):
                query_row = query_work // kv_heads
            else:
                query_row = self._map_query_row_in_cta(
                    pair_selected_slots, query_work, prefix_scratch, tid
                )
        kv_head = query_work % kv_heads
        pair_row = query_row // self.query_group
        if cutlass.const_expr(
            self.original_heads_per_kv == 12 and self.problem_shape[2] == 2
        ):
            arena = utils.SmemAllocator().allocate_array(
                cutlass.Int8, self.scratch_arena_bytes, byte_alignment=128
            )
            if cutlass.const_expr(self.prefix_in_cta):
                shared_prefix = self._shared_prefix_in_cta(
                    pair_selected_slots, pair_row, arena, tid
                )
            else:
                shared_prefix = pair_prefix_shared[pair_row] != 0
            if shared_prefix:
                if query_row % self.query_group == 0:
                    tmem_cell = cute.recast_ptr(arena, dtype=cutlass.Int32)
                    if warp == 1:
                        cute.arch.alloc_tmem(self.tmem_alloc_cols, tmem_cell)
                    cute.arch.sync_threads()
                    self.decode_work(
                        pair_blk_tile_shd,
                        pair_mma_tile_mnk,
                        pair_tiled_mma_kq,
                        pair_tiled_mma_vp,
                        pair_q_dtype,
                        pair_smem_layout_q,
                        pair_tma_atom_q,
                        pair_mQ,
                        pair_q_source,
                        pair_k_dtype,
                        pair_smem_layout_k,
                        pair_k_iter,
                        pair_v_dtype,
                        pair_smem_layout_v,
                        pair_v_iter,
                        pair_selected_slots,
                        pair_prefix_shared,
                        pair_o_dtype,
                        pair_smem_layout_o,
                        pair_mOut,
                        pair_scale_qs,
                        pair_scale_qs_log2_e,
                        pair_scale_o,
                        pair_row,
                        kv_head,
                        arena,
                    )
                    if warp == self.mma_vp_warp_id:
                        tmem_ptr = cute.arch.retrieve_tmem_ptr(
                            cutlass.Int32, 16, tmem_cell
                        )
                        cute.arch.relinquish_tmem_alloc_permit()
                        cute.arch.dealloc_tmem(tmem_ptr, self.tmem_alloc_cols)
            else:
                if cutlass.const_expr(single_config.work_sync_threads != 0):
                    if not shared_prefix and tid >= single_config.work_sync_threads:
                        cute.arch.setmaxregister_decrease(
                            single_config.register_budget[1]
                        )
                tmem_cell = cute.recast_ptr(arena, dtype=cutlass.Int32)
                if warp == 1:
                    cute.arch.alloc_tmem(self.tmem_alloc_cols, tmem_cell)
                cute.arch.sync_threads()
                run_single = cutlass.Boolean(True)
                if cutlass.const_expr(single_config.work_sync_threads != 0):
                    run_single = tid < single_config.work_sync_threads
                if run_single:
                    single_config.decode_work(
                        single_blk_tile_shd,
                        single_mma_tile_mnk,
                        single_tiled_mma_kq,
                        single_tiled_mma_vp,
                        single_q_dtype,
                        single_smem_layout_q,
                        single_tma_atom_q,
                        single_mQ,
                        single_q_source,
                        single_k_dtype,
                        single_smem_layout_k,
                        single_k_iter,
                        single_v_dtype,
                        single_smem_layout_v,
                        single_v_iter,
                        single_selected_slots,
                        single_prefix_shared,
                        single_o_dtype,
                        single_smem_layout_o,
                        single_mOut,
                        single_scale_qs,
                        single_scale_qs_log2_e,
                        single_scale_o,
                        query_row,
                        kv_head,
                        arena,
                    )
                    if cutlass.const_expr(single_config.work_sync_threads != 0):
                        if warp == single_config.mma_vp_warp_id:
                            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                                cutlass.Int32, 16, tmem_cell
                            )
                            cute.arch.relinquish_tmem_alloc_permit()
                            cute.arch.dealloc_tmem(tmem_ptr, self.tmem_alloc_cols)
                if cutlass.const_expr(single_config.work_sync_threads == 0):
                    if warp == self.mma_vp_warp_id:
                        tmem_ptr = cute.arch.retrieve_tmem_ptr(
                            cutlass.Int32, 16, tmem_cell
                        )
                        cute.arch.relinquish_tmem_alloc_permit()
                        cute.arch.dealloc_tmem(tmem_ptr, self.tmem_alloc_cols)
        else:
            if cutlass.const_expr(not (self.original_heads_per_kv == 12)):
                shared_prefix = self._classify_prefix_in_cta(
                    pair_selected_slots, pair_row, prefix_scratch, tid
                )
            if cutlass.const_expr(self.original_heads_per_kv == 12):
                arena = utils.SmemAllocator().allocate_array(
                    cutlass.Int8,
                    self.scratch_arena_bytes,
                    byte_alignment=128,
                )
            else:
                arena = smem.allocate_array(
                    cutlass.Int8, self.scratch_arena_bytes, byte_alignment=128
                )
            if cutlass.const_expr(self.original_heads_per_kv == 12):
                if cutlass.const_expr(self.prefix_in_cta):
                    shared_prefix = self._shared_prefix_in_cta(
                        pair_selected_slots,
                        pair_row,
                        arena,
                        tid,
                    )
                else:
                    # Low-level callers' explicitly supplied flags remain authoritative.
                    shared_prefix = pair_prefix_shared[pair_row] != 0
            if not shared_prefix or query_row % self.query_group == 0:
                tmem_cell = cute.recast_ptr(arena, dtype=cutlass.Int32)
                if cutlass.const_expr(self.original_heads_per_kv == 12):
                    if cutlass.const_expr(single_config.work_sync_threads != 0):
                        if not shared_prefix and tid >= single_config.work_sync_threads:
                            # Idle warps release registers before the existing allocation join.
                            cute.arch.setmaxregister_decrease(
                                single_config.register_budget[1]
                            )
                if warp == 1:
                    cute.arch.alloc_tmem(self.tmem_alloc_cols, tmem_cell)
                cute.arch.sync_threads()
                if shared_prefix:
                    self.decode_work(
                        pair_blk_tile_shd,
                        pair_mma_tile_mnk,
                        pair_tiled_mma_kq,
                        pair_tiled_mma_vp,
                        pair_q_dtype,
                        pair_smem_layout_q,
                        pair_tma_atom_q,
                        pair_mQ,
                        pair_q_source,
                        pair_k_dtype,
                        pair_smem_layout_k,
                        pair_k_iter,
                        pair_v_dtype,
                        pair_smem_layout_v,
                        pair_v_iter,
                        pair_selected_slots,
                        pair_prefix_shared,
                        pair_o_dtype,
                        pair_smem_layout_o,
                        pair_mOut,
                        pair_scale_qs,
                        pair_scale_qs_log2_e,
                        pair_scale_o,
                        pair_row,
                        kv_head,
                        arena,
                    )
                    if warp == self.mma_vp_warp_id:
                        tmem_ptr = cute.arch.retrieve_tmem_ptr(
                            cutlass.Int32, 16, tmem_cell
                        )
                        cute.arch.relinquish_tmem_alloc_permit()
                        cute.arch.dealloc_tmem(tmem_ptr, self.tmem_alloc_cols)
                else:
                    if cutlass.const_expr(self.original_heads_per_kv == 12):
                        if cutlass.const_expr(single_config.work_sync_threads == 0):
                            active_work = cutlass.Boolean(True)
                        else:
                            active_work = tid < single_config.work_sync_threads
                        if active_work:
                            single_config.decode_work(
                                single_blk_tile_shd,
                                single_mma_tile_mnk,
                                single_tiled_mma_kq,
                                single_tiled_mma_vp,
                                single_q_dtype,
                                single_smem_layout_q,
                                single_tma_atom_q,
                                single_mQ,
                                single_q_source,
                                single_k_dtype,
                                single_smem_layout_k,
                                single_k_iter,
                                single_v_dtype,
                                single_smem_layout_v,
                                single_v_iter,
                                single_selected_slots,
                                single_prefix_shared,
                                single_o_dtype,
                                single_smem_layout_o,
                                single_mOut,
                                single_scale_qs,
                                single_scale_qs_log2_e,
                                single_scale_o,
                                query_row,
                                kv_head,
                                arena,
                            )
                            release_warp = (
                                single_config.mma_vp_warp_id
                                if cutlass.const_expr(
                                    single_config.work_sync_threads != 0
                                )
                                else self.mma_vp_warp_id
                            )
                            if warp == release_warp:
                                tmem_ptr = cute.arch.retrieve_tmem_ptr(
                                    cutlass.Int32, 16, tmem_cell
                                )
                                cute.arch.relinquish_tmem_alloc_permit()
                                # The caller owns the full allocation, including unused columns.
                                cute.arch.dealloc_tmem(tmem_ptr, self.tmem_alloc_cols)
                    else:
                        run_single = cutlass.Boolean(True)
                        if cutlass.const_expr(single_config.work_sync_threads != 0):
                            run_single = tid < single_config.work_sync_threads
                        if run_single:
                            single_config.decode_work(
                                single_blk_tile_shd,
                                single_mma_tile_mnk,
                                single_tiled_mma_kq,
                                single_tiled_mma_vp,
                                single_q_dtype,
                                single_smem_layout_q,
                                single_tma_atom_q,
                                single_mQ,
                                single_q_source,
                                single_k_dtype,
                                single_smem_layout_k,
                                single_k_iter,
                                single_v_dtype,
                                single_smem_layout_v,
                                single_v_iter,
                                single_selected_slots,
                                single_prefix_shared,
                                single_o_dtype,
                                single_smem_layout_o,
                                single_mOut,
                                single_scale_qs,
                                single_scale_qs_log2_e,
                                single_scale_o,
                                query_row,
                                kv_head,
                                arena,
                            )
                            if warp == (
                                single_config.mma_vp_warp_id
                                if cutlass.const_expr(
                                    single_config.work_sync_threads != 0
                                )
                                else self.mma_vp_warp_id
                            ):
                                tmem_ptr = cute.arch.retrieve_tmem_ptr(
                                    cutlass.Int32, 16, tmem_cell
                                )
                                cute.arch.relinquish_tmem_alloc_permit()
                                cute.arch.dealloc_tmem(tmem_ptr, self.tmem_alloc_cols)

    @cute.jit
    def _softmax_work(
        self,
        softmax_head_tile: cutlass.Constexpr,
        softmax_head_offset: cutlass.Int32,
        softmax_copy_group: cutlass.Int32,
        acc_dtype: cutlass.Constexpr,
        blk_tile_d: cutlass.Constexpr,
        blk_tile_h: cutlass.Constexpr,
        blk_tile_s: cutlass.Constexpr,
        head_offset,
        heads_per_kv,
        iters_s,
        kv_split_idx,
        kv_splits,
        lane_idx,
        mma_dice: cutlass.Constexpr,
        mma_dtype: cutlass.Constexpr,
        mma_tile_m: cutlass.Constexpr,
        mma_tile_n: cutlass.Constexpr,
        o_consumer,
        o_dtype: cutlass.Constexpr,
        p_producer,
        reduction_heads,
        row_idx,
        sL,
        sM,
        s_consumer,
        scale_qs,
        scale_qs_log2_e,
        selected_slots,
        softmax_nbar,
        tAsK,
        tCsL,
        tCsM,
        tCsO,
        tCsP,
        tCtO,
        tCtS,
        thrblk_mma_kq,
        thrblk_mma_vp,
        tiles_dm: cutlass.Constexpr,
        warpgroup_tidx,
        warpgroup_widx,
    ):
        if cutlass.const_expr(self.register_budget is not None):
            # 128*208 + 384*80 = 57344 registers from the CTA pool.
            cute.arch.setmaxregister_increase(self.register_budget[0])
        head_in_tile = softmax_head_offset + lane_idx
        lane_store_max = lane_idx < softmax_head_tile
        softmax_dice = (None, None, None)
        # Construct tiled copies
        tmem_op_width = 32
        tmem_op_repeat = tcgen05.Repetition(
            softmax_head_tile * acc_dtype.width // tmem_op_width
        )
        tmem_load_atom_s = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tmem_op_repeat), acc_dtype
        )
        tmem_load_s = tcgen05.make_tmem_copy(tmem_load_atom_s, tCtS[mma_dice + (0,)])
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
        tStS = tStS[None, softmax_copy_group, None, None, None]
        cS = cute.make_identity_tensor((mma_tile_m, mma_tile_n))
        tOcS = thrblk_mma_kq.partition_C(cS)
        tScS = thr_load_s.partition_D(tOcS)
        tScS = tScS[None, softmax_copy_group, None, None]
        tSsP = thr_load_s.partition_D(
            tCsP
        )  # (CPY, #CPY_MMA, #CPY_M, #CPY_N, stages_sp)
        tSsP = tSsP[None, softmax_copy_group, None, None, None]

        # Partition O
        tStO = thr_load_s.partition_S(
            tCtO
        )  # (CPY, #CPY_MMA, #CPY_M, #CPY_N, #TILE_DM, #TILE_HN)
        tStO = tStO[None, softmax_copy_group, None, None, None, None]
        tSsO = thr_load_s.partition_D(
            tCsO
        )  # (CPY, #CPY_MMA, #CPY_M, #CPY_N, #TILE_DM, #TILE_HN)
        tSsO = tSsO[None, softmax_copy_group, None, None, None, None]
        tSrO = cute.make_rmem_tensor_like(tSsO)
        cO = cute.make_identity_tensor((blk_tile_d, blk_tile_h))
        tOcO = thrblk_mma_vp.partition_C(cO)
        tScO = thr_load_s.partition_D(tOcO)
        tScO = tScO[None, softmax_copy_group, None, None]
        if cutlass.const_expr(self.scalar_softmax_stats):
            # Eight-head fragments permit a whole warp to skip
            # identity correction without masking TMEM instructions.
            chunk_load_atom = cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tcgen05.Repetition(8)), acc_dtype
            )
            chunk_store_atom = cute.make_copy_atom(
                tcgen05.St32x32bOp(tcgen05.Repetition(8)), o_dtype
            )
            chunk_load = tcgen05.make_tmem_copy(
                chunk_load_atom, tCtO[mma_dice + (0, 0)]
            ).get_slice(warpgroup_tidx)
            chunk_store = tcgen05.make_tmem_copy(
                chunk_store_atom, tCtO[mma_dice + (0, 0)]
            ).get_slice(warpgroup_tidx)
            chunk_tmem = chunk_load.partition_S(tCtO)
            chunk_rf = chunk_load.partition_D(tCsO)

        # Partition colmax and initialize in RF
        tSsM = thr_load_s.partition_D(tCsM)  # (CPY, #CPY_MMA, #CPY_M, #CPY_N)
        tSsM = tSsM[None, softmax_copy_group, None, None]
        if cutlass.const_expr(self.scalar_softmax_stats):
            previous_max_lane = cutlass.Float32(-cutlass.Float32.inf)
        else:
            tSrM_prev = cute.make_rmem_tensor_like(tSsM)
            tSrM_prev.fill(-cutlass.Float32.inf)

        # Partition colsum and initialize in RF
        # Each thread maintains a local colsum in RF, smem reduction happens after loop
        tSsL = thr_load_s.partition_D(tCsL)  # (CPY, #CPY_MMA, #CPY_M, #CPY_N, WARPS)
        tSsL = tSsL[None, softmax_copy_group, None, None, None]
        tSrL = cute.make_rmem_tensor_like(tSsL[softmax_dice + (0,)])
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
            selected_tile = kv_split_idx + s * kv_splits
            active_head_group = cutlass.Boolean(True)
            # Load S from tmem
            s_handle = s_consumer.wait_and_advance()
            tSrS = cute.make_rmem_tensor(tSsP.shape[:-1], acc_dtype)
            if active_head_group:
                cute.copy(thr_load_s, tStS[softmax_dice + (s_handle.index,)], tSrS)
                cute.arch.fence_view_async_tmem_load()
            s_handle.release()

            tSrValid = cute.make_rmem_tensor(tSrS.shape, cutlass.Int32)
            tSrP = cute.make_rmem_tensor(tSsP.shape[:-1], mma_dtype)
            tSrP.fill(cutlass.BFloat16(0))
            tSrP_f32 = cute.make_rmem_tensor(tSrS.shape, acc_dtype)
            if cutlass.const_expr(self.scalar_softmax_stats):
                max_lane = previous_max_lane
                correction_lane = cutlass.Float32(0.0)
            else:
                tSrM = cute.make_rmem_tensor_like(tSsM)
                correction = cute.make_rmem_tensor_like(tSsM)
            if active_head_group:
                # Invalid entries load slot 1, which the caller keeps finite.
                # Mask their logits using the original selection so
                # they never contribute to the softmax.
                for i in cutlass.range_constexpr(cute.size(tSrS)):
                    score_coord = tScS[i]
                    selected_idx = selected_tile * blk_tile_s + score_coord[0]
                    valid = cutlass.Boolean(False)
                    if (
                        selected_idx < 2048
                        and head_offset + score_coord[1] < heads_per_kv
                        and (head_offset + score_coord[1])
                        % (heads_per_kv // self.query_group)
                        < self.original_heads_per_kv
                    ):
                        source_row = row_idx
                        source_column = selected_idx
                        if cutlass.const_expr(self.query_group > 1):
                            source_row = row_idx * self.query_group
                        valid = selected_slots[source_row, source_column] > 0
                    tSrValid[i] = cutlass.Int32(1) if valid else cutlass.Int32(0)
                    if not valid:
                        tSrS[i] = cutlass.Float32(-1.0e30)

                # Reduce colmax in warp RF
                tSrM_lane = cutlass.Float32(0)  # Avoid dynamic register indexing
                for i in cutlass.range_constexpr(cute.size(tSrS)):
                    reduced_max = warp_fmax(tSrS[i])
                    if cutlass.const_expr(not self.scalar_softmax_stats):
                        tSrM[i] = reduced_max
                    if i == lane_idx:
                        tSrM_lane = reduced_max

                # Reduce colmax in smem
                if lane_store_max:
                    smem_fmax(tSsM.iterator + tSsM.layout(lane_idx), tSrM_lane)

                # Wait for colmax then load
                softmax_nbar.arrive_and_wait()
                # One lane owns each head statistic. Broadcast it
                # where used instead of keeping per-head RF arrays.
                if cutlass.const_expr(self.scalar_softmax_stats):
                    if lane_store_max:
                        max_lane = tSsM[lane_idx]
                else:
                    cute.autovec_copy(tSsM, tSrM)

                # Compute online softmax
                for i in cutlass.range_constexpr(0, cute.size(tSrS), 2):
                    if cutlass.const_expr(self.scalar_softmax_stats):
                        max_f32x2 = (
                            cute.arch.shuffle_sync(max_lane, i),
                            cute.arch.shuffle_sync(max_lane, i + 1),
                        )
                    else:
                        max_f32x2 = (tSrM[i], tSrM[i + 1])
                    p_f32x2 = fadd2(
                        (tSrS[i], tSrS[i + 1]),
                        (-max_f32x2[0], -max_f32x2[1]),
                    )
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
            cute.autovec_copy(tSrP, tSsP[softmax_dice + (p_handle.index,)])
            cute.arch.fence_view_async_shared()
            p_handle.commit()

            if active_head_group:
                if cutlass.const_expr(self.scalar_softmax_stats):
                    if lane_store_max:
                        correction_lane = exp2(
                            (previous_max_lane - max_lane) * scale_qs_log2_e
                        )
                        previous_max_lane = max_lane
                for i in cutlass.range_constexpr(0, cute.size(tSsM), 2):
                    if cutlass.const_expr(self.scalar_softmax_stats):
                        c_f32x2 = (
                            cute.arch.shuffle_sync(correction_lane, i),
                            cute.arch.shuffle_sync(correction_lane, i + 1),
                        )
                    else:
                        c_f32x2 = fadd2(
                            (tSrM_prev[i], tSrM_prev[i + 1]),
                            (-tSrM[i], -tSrM[i + 1]),
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
                if cutlass.const_expr(not self.scalar_softmax_stats):
                    tSrM_prev.store(tSrM.load())

            # Correct O
            if s > 0:
                # Wait for O
                o_handle = o_consumer.wait_and_advance()

                if active_head_group:
                    # Apply correction
                    if cutlass.const_expr(self.scalar_softmax_stats):
                        for head_chunk in cutlass.range_constexpr(
                            softmax_head_tile // 8
                        ):
                            begin = head_chunk * 8
                            changed = (
                                lane_idx >= begin
                                and lane_idx < begin + 8
                                and correction_lane != cutlass.Float32(1)
                            )
                            needs_chunk = cute.arch.vote_any_sync(changed)
                            if needs_chunk:
                                copy_group = softmax_head_offset // 8 + head_chunk
                                source = chunk_tmem[
                                    None, copy_group, None, None, None, None
                                ]
                                rf_view = chunk_rf[
                                    None, copy_group, None, None, None, None
                                ]
                                for dm in cutlass.range_constexpr(tiles_dm):
                                    values = cute.make_rmem_tensor_like(
                                        rf_view[softmax_dice + (0, 0)]
                                    )
                                    cute.copy(
                                        chunk_load,
                                        source[softmax_dice + (dm, 0)],
                                        values,
                                    )
                                    for i in cutlass.range_constexpr(
                                        0, cute.size(values), 2
                                    ):
                                        coeff = (
                                            cute.arch.shuffle_sync(
                                                correction_lane, begin + i
                                            ),
                                            cute.arch.shuffle_sync(
                                                correction_lane, begin + i + 1
                                            ),
                                        )
                                        product = fmul2(
                                            (values[i], values[i + 1]), coeff
                                        )
                                        values[i] = product[0]
                                        values[i + 1] = product[1]
                                    cute.copy(
                                        chunk_store,
                                        values,
                                        source[softmax_dice + (dm, 0)],
                                    )
                    else:
                        for dm in cutlass.range_constexpr(tiles_dm):
                            tSrO_dm = cute.make_rmem_tensor_like(
                                tSsO[softmax_dice + (0, 0)]
                            )
                            cute.copy(
                                thr_load_s,
                                tStO[softmax_dice + (dm, 0)],
                                tSrO_dm,
                            )

                            for i in cutlass.range_constexpr(0, cute.size(tSrO_dm), 2):
                                if cutlass.const_expr(self.scalar_softmax_stats):
                                    c_f32x2 = (
                                        cute.arch.shuffle_sync(correction_lane, i),
                                        cute.arch.shuffle_sync(correction_lane, i + 1),
                                    )
                                else:
                                    c_f32x2 = (correction[i], correction[i + 1])
                                o_f32x2 = fmul2((tSrO_dm[i], tSrO_dm[i + 1]), c_f32x2)
                                tSrO_dm[i] = o_f32x2[0]
                                tSrO_dm[i + 1] = o_f32x2[1]

                            cute.copy(
                                thr_store_o,
                                tSrO_dm,
                                tStO[softmax_dice + (dm, 0)],
                            )

                # Notify MMA
                cute.arch.fence_view_async_tmem_store()
                o_handle.release()

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
            tSsL[softmax_dice + (warpgroup_widx,)][lane_idx] = tSrL_lane

        # Wait for colsum
        softmax_nbar.arrive_and_wait()

        if warpgroup_widx == 0 and lane_store_max:
            # Load colsum and colmax
            sL_lane_wg = sL[0, head_in_tile, None]
            sL_lane = sL_lane_wg[0] + sL_lane_wg[1] + sL_lane_wg[2] + sL_lane_wg[3]
            sM_lane = sM[0, head_in_tile]

            # Scale colmax
            sM_lane = sM_lane * scale_qs

            # Keep final statistics local for the DSM merge.
            sL[0, head_in_tile, 0] = sL_lane
            sM[0, head_in_tile] = sM_lane

        o_handle = o_consumer.wait_and_advance()
        cute.copy(thr_load_s, tStO, tSrO)
        cute.arch.fence_view_async_tmem_load()
        o_handle.release()  # Final release signals tmem dealloc

        # Persist this CTA's numerator directly in its own completed KV
        # staging allocation.  The coordinate partition is exactly the
        # one paired with the TMEM fragment, avoiding swizzle inversion.
        local_partial = cute.make_tensor(
            cute.recast_ptr(tAsK.iterator, dtype=acc_dtype),
            cute.make_layout(
                (reduction_heads, blk_tile_d + 2),
                stride=(blk_tile_d + 2, 1),
            ),
        )
        for i in cutlass.range_constexpr(cute.size(tSrO)):
            coord = tScO[i]
            dim = coord[0]
            head = coord[1]
            if head < reduction_heads:
                local_partial[head, dim] = tSrO[i]
        if warpgroup_widx == 0 and head_in_tile < reduction_heads:
            local_partial[head_in_tile, blk_tile_d] = sM[0, head_in_tile]
            local_partial[head_in_tile, blk_tile_d + 1] = sL[0, head_in_tile, 0]

    @cute.jit
    def decode_work(
        self,
        blk_tile_shd: cute.Tile,
        mma_tile_mnk: cute.Tile,
        tiled_mma_kq: cute.TiledMma,
        tiled_mma_vp: cute.TiledMma,
        q_dtype: Type[cutlass.Numeric],
        smem_layout_q: cute.ComposedLayout,
        tma_atom_q: cute.CopyAtom,
        mQ: cute.Tensor,
        q_source: cute.Pointer,
        k_dtype: Type[cutlass.Numeric],
        smem_layout_k: cute.ComposedLayout,
        k_iter: cute.Pointer,
        v_dtype: Type[cutlass.Numeric],
        smem_layout_v: cute.ComposedLayout,
        v_iter: cute.Pointer,
        selected_slots: cute.Tensor,
        prefix_shared: cute.Tensor,
        o_dtype: Type[cutlass.Numeric],
        smem_layout_o: cute.ComposedLayout,
        mOut: cute.Tensor,
        scale_qs: cutlass.Float32,
        scale_qs_log2_e: cutlass.Float32,
        scale_o: cutlass.Float32,
        work_row_base: cutlass.Int32,
        work_kv_head: cutlass.Int32,
        scratch_base: cute.Pointer,
    ):
        if cutlass.const_expr(self.caller_allocates_scratch):
            kv_splits = self.kv_splits
            kv_split_idx = 0
            coord_hr = 0
            coord_hb = work_row_base * self.problem_shape[2] + work_kv_head
        else:
            kv_splits, tiles_hr, tiles_hb = cute.arch.grid_dim()
            kv_split_idx, coord_hr, coord_hb = cute.arch.block_idx()
        _, num_q_heads, num_kv_heads, cache_slots, head_dim = self.problem_shape
        num_rows = selected_slots.shape[0] // self.query_group
        heads_per_kv = num_q_heads // num_kv_heads
        head_tiles = cute.ceil_div(heads_per_kv, self.head_tile)
        row_base = (coord_hb // num_kv_heads) * self.query_rows_per_cta
        kv_head_idx = coord_hb % num_kv_heads
        head_offset = (coord_hr % head_tiles) * self.head_tile
        q_head_offset = kv_head_idx * heads_per_kv + head_offset
        reduction_heads = min(heads_per_kv, self.head_tile)
        k_iter = k_iter + kv_head_idx * head_dim
        v_iter = v_iter + kv_head_idx * head_dim
        tidx, _, _ = cute.arch.thread_idx()
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(tidx // warp_threads)
        warpgroup_idx = cute.arch.make_warp_uniform(tidx // warpgroup_threads)
        warpgroup_tidx = tidx % warpgroup_threads
        warpgroup_widx = warp_idx % warpgroup_warps
        init_warp = 0
        mcast_coord = 0
        mcast_layout = cute.make_layout((1, 1, 1, 1))  # vmnk
        mma_dtype = q_dtype
        acc_dtype = o_dtype
        kv_smem_dtype = k_dtype
        blk_tile_s, blk_tile_h, blk_tile_d = blk_tile_shd
        mma_tile_m, mma_tile_n, mma_tile_k = mma_tile_mnk
        tiles_dm, tiles_sk = cute.ceil_div(
            (blk_tile_d, blk_tile_s), (mma_tile_m, mma_tile_k)
        )
        tiles_dk, tiles_sm = cute.ceil_div(
            (blk_tile_d, blk_tile_s), (mma_tile_k, mma_tile_m)
        )
        tiles_s = selected_slots.shape[1] // blk_tile_s
        assert tiles_s % self.kv_splits == 0
        static_iters_s = tiles_s // self.kv_splits
        iters_s = static_iters_s
        prefetch_iters = self.sp_stages - 1
        if static_iters_s < prefetch_iters:
            prefetch_iters = static_iters_s
        assert tiles_sm == 1
        exit_early = kv_split_idx >= tiles_s
        lane_store_max = mma_tile_n == warp_threads or lane_idx < mma_tile_n
        svector_align = 16
        stensor_align = 128
        if cutlass.const_expr(self.caller_allocates_scratch):
            smem = _ScratchArena(scratch_base, self.scratch_arena_bytes)
        else:
            smem = utils.SmemAllocator()
        if warp_idx == init_warp and not exit_early:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
        init_warp += 1
        tmem_ptr_smem_ptr = smem.allocate_array(cutlass.Int32)
        if cutlass.const_expr(not self.caller_allocates_scratch):
            if warp_idx == init_warp and not exit_early:
                cute.arch.alloc_tmem(self.tmem_alloc_cols, tmem_ptr_smem_ptr)
        init_warp += 1
        q_pipeline_ptr = smem.allocate_array(cutlass.Int64, self.q_stages * 2)
        cvt_pipeline_ptr = smem.allocate_array(cutlass.Int64, self.cvt_stages * 2)
        s_pipeline_ptr = smem.allocate_array(cutlass.Int64, self.sp_stages * 2)
        p_pipeline_ptr = smem.allocate_array(cutlass.Int64, self.sp_stages * 2)
        o_pipeline_ptr = smem.allocate_array(cutlass.Int64, self.o_stages * 2)
        softmax_nbar = pipeline.NamedBarrier(
            barrier_id=(
                (8 + warpgroup_idx) if cutlass.const_expr(self.query_group > 1) else 1
            ),
            num_threads=warpgroup_threads,
        )
        mma_kq_even_nbar = pipeline.NamedBarrier(barrier_id=2, num_threads=64)
        mma_kq_odd_nbar = pipeline.NamedBarrier(barrier_id=6, num_threads=64)
        elect_one_cooperative = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        warpgroup_cooperative = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, warpgroup_threads
        )
        mma_group = elect_one_cooperative
        tma_group = elect_one_cooperative
        cvt_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, warpgroup_threads)
        cvt_groups = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, warpgroup_threads * _CONVERT_WARPGROUPS
        )
        softmax_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, warpgroup_threads * self.softmax_warpgroups
        )
        thrblk_mma_kq = tiled_mma_kq.get_slice(0)
        thrblk_mma_vp = tiled_mma_vp.get_slice(0)
        sM_layout = cute.make_layout(shape=(mma_tile_m, mma_tile_n), stride=(0, 1))
        sM = smem.allocate_tensor(acc_dtype, sM_layout, svector_align)
        tCsM = thrblk_mma_kq.partition_C(sM)
        sL_layout = cute.make_layout(
            shape=(mma_tile_m, mma_tile_n, warpgroup_warps), stride=(0, 1, mma_tile_n)
        )
        sL = smem.allocate_tensor(acc_dtype, sL_layout, svector_align)
        tCsL = thrblk_mma_kq.partition_C(sL)
        tBsQ = smem.allocate_tensor(
            q_dtype, smem_layout_q.outer, stensor_align, smem_layout_q.inner
        )  # (MMA, #MMA_N, #MMA_K, q_stages)
        tAsK = smem.allocate_tensor(
            kv_smem_dtype, smem_layout_k.outer, stensor_align, smem_layout_k.inner
        )  # (MMA, #MMA_M, #MMA_K, kv_stages)
        tAtK_cvt_shape = tiled_mma_kq.partition_shape_A(
            (mma_tile_m, mma_tile_k, self.cvt_stages)
        )  # (MMA, #MMA_M, #MMA_K, cvt_stages)
        if cutlass.const_expr(k_dtype is cutlass.BFloat16):
            tAtK_cvt_fragment = thrblk_mma_kq.make_fragment_A(tAsK)
        else:
            tAtK_cvt_fragment = thrblk_mma_kq.make_fragment_A(tAtK_cvt_shape)
        tAsV_iterator = cute.recast_ptr(
            tAsK.iterator, smem_layout_v.inner, dtype=kv_smem_dtype
        )  # KV share input buffers
        tAsV = cute.make_tensor(
            tAsV_iterator, smem_layout_v.outer
        )  # (MMA, #MMA_M, #MMA_K, kv_stages)
        sKV_vector_ptr = cute.recast_ptr(tAsK.iterator, dtype=kv_smem_dtype)
        sKV_vector_address = sKV_vector_ptr.toint()
        if cutlass.const_expr(
            not (self.original_heads_per_kv == 12 and self.problem_shape[2] == 2)
        ):
            if cutlass.const_expr(not (self.original_heads_per_kv == 12)):
                prefix_ptr = cute.make_ptr(
                    cutlass.Int32,
                    sKV_vector_address,
                    cute.AddressSpace.smem,
                    assumed_align=128,
                )
                ordered_prefix = cute.make_tensor(prefix_ptr, cute.make_layout(2048))
                prefix_bucket_eligible = cutlass.Boolean(False)
                if cutlass.const_expr(
                    self.caller_allocates_scratch
                    and self.query_group == 1
                    and self.original_heads_per_kv == 6
                    and self.problem_shape[2] == 1
                    and k_dtype is cutlass.BFloat16
                ):
                    prefix_histogram = cute.make_tensor(
                        prefix_ptr + 2048, cute.make_layout(577)
                    )
                    # Cache capacity is already part of the parent's fixed geometry;
                    # no per-request value becomes a new specialization parameter.
                    bucket_shift = cutlass.Int32(
                        max(0, (cache_slots - 1).bit_length() - 5)
                    )
                    prefix_bucket_eligible = self._bucket_prefix_in_cta(
                        selected_slots,
                        row_base,
                        ordered_prefix,
                        prefix_histogram,
                        bucket_shift,
                        tidx,
                    )
        tAtV_cvt_shape = tiled_mma_vp.partition_shape_A(
            (mma_tile_m, mma_tile_k, self.cvt_stages)
        )  # (MMA, #MMA_M, #MMA_K, cvt_stages)
        if cutlass.const_expr(v_dtype is cutlass.BFloat16):
            tAtV_cvt_fragment = thrblk_mma_vp.make_fragment_A(tAsV)
        else:
            tAtV_cvt_fragment = thrblk_mma_vp.make_fragment_A(tAtV_cvt_shape)
        tCtS_shape = tiled_mma_kq.partition_shape_C(
            (mma_tile_m, mma_tile_n, self.sp_stages)
        )
        tCtS_fragment = thrblk_mma_kq.make_fragment_C(
            tCtS_shape
        )  # (MMA_MN, #MMA_M=1, #MMA_N=1, sp_stages)
        mma_tile_nm = (None, mma_tile_n, mma_tile_m)
        tBsP_nm_layout = sm100_utils.make_smem_layout_b(
            tiled_mma_vp, mma_tile_nm, mma_dtype, self.sp_stages
        )
        tBsP_nm = smem.allocate_tensor(
            mma_dtype, tBsP_nm_layout.outer, stensor_align, tBsP_nm_layout.inner
        )
        if cutlass.const_expr(
            self.original_heads_per_kv == 12 and self.problem_shape[2] == 2
        ):
            if cutlass.const_expr(self.prefetch_private_kv):
                assert kv_smem_dtype is cutlass.BFloat16
                private_kv = smem.allocate_tensor(
                    kv_smem_dtype,
                    cute.make_layout(
                        (blk_tile_d, 3, self.query_group, 2),
                        stride=(
                            1,
                            blk_tile_d,
                            3 * blk_tile_d,
                            3 * blk_tile_d * self.query_group,
                        ),
                    ),
                    svector_align,
                )
                staged_key = private_kv[None, None, None, 0]
                staged_value = private_kv[None, None, None, 1]
        if cutlass.const_expr(self.overlap_private_tail):
            # The independent tail partial must not alias the active KV ring.
            private_tail = smem.allocate_tensor(
                acc_dtype,
                cute.make_layout((self.head_tile, 264), stride=(264, 1)),
                svector_align,
            )
        tBsP_nk_tile = thrblk_mma_vp.partition_shape_B((mma_tile_n, mma_tile_k))
        tBsP_nk = cute.local_tile(tBsP_nm, tBsP_nk_tile, (0, 0, None, None))
        tCsP_tile = cute.make_ordered_layout(tCtS_shape, order=((2, 0), 3, 1, 4))
        tCsP = cute.composition(tBsP_nm, tCsP_tile)
        sO_iterator = cute.recast_ptr(
            tBsQ.iterator, smem_layout_o.inner, dtype=o_dtype
        )  # Reuse QKV smem for O TMA store
        sO_mma = cute.make_tensor(
            sO_iterator, smem_layout_o.outer
        )  # (MMA_TILE_M, MMA_TILE_N, #TILE_DM, #TILE_HN)
        tCsO = thrblk_mma_vp.partition_C(
            sO_mma
        )  # (MMA, #MMA_M, #MMA_N, #TILE_DM, #TILE_HN)
        tCtO_fragment = thrblk_mma_vp.make_fragment_C(tCsO.shape)
        if cutlass.const_expr(self.enable_pdl and not self.caller_allocates_scratch):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(
            self.original_heads_per_kv == 12 and self.problem_shape[2] == 2
        ):
            prefix_bucket_eligible = cutlass.Boolean(False)
            if cutlass.const_expr(self.bucket_prefix):
                # PDL publication is complete. The first KV stage is still unused;
                # the full-D producer retires it after caching all IDs.
                prefix_ptr = cute.make_ptr(
                    cutlass.Int32,
                    sKV_vector_address,
                    cute.AddressSpace.smem,
                    assumed_align=128,
                )
                ordered_prefix = cute.make_tensor(prefix_ptr, cute.make_layout(2048))
                prefix_histogram = cute.make_tensor(
                    prefix_ptr + 2048, cute.make_layout(577)
                )
                bucket_shift = cutlass.Int32(max(0, (cache_slots - 1).bit_length() - 5))
                prefix_bucket_eligible = self._bucket_prefix_in_cta(
                    selected_slots,
                    row_base,
                    ordered_prefix,
                    prefix_histogram,
                    bucket_shift,
                    tidx,
                )
        else:
            if cutlass.const_expr(self.original_heads_per_kv == 12):
                prefix_bucket_eligible = cutlass.Boolean(False)
                if cutlass.const_expr(self.bucket_prefix):
                    # PDL publication is complete. The first KV stage is still unused;
                    # producers retire this temporary storage after caching all IDs.
                    prefix_ptr = cute.make_ptr(
                        cutlass.Int32,
                        sKV_vector_address,
                        cute.AddressSpace.smem,
                        assumed_align=128,
                    )
                    ordered_prefix = cute.make_tensor(
                        prefix_ptr, cute.make_layout(2048)
                    )
                    prefix_histogram = cute.make_tensor(
                        prefix_ptr + 2048, cute.make_layout(577)
                    )
                    bucket_shift = cutlass.Int32(
                        max(0, (cache_slots - 1).bit_length() - 5)
                    )
                    prefix_bucket_eligible = self._bucket_prefix_in_cta(
                        selected_slots,
                        row_base,
                        ordered_prefix,
                        prefix_histogram,
                        bucket_shift,
                        tidx,
                    )
        for query_step in cutlass.range_constexpr(self.query_rows_per_cta):
            row_idx = row_base + query_step
            coord_hb = row_idx * num_kv_heads + kv_head_idx
            valid_query = True
            if cutlass.const_expr(self.query_rows_per_cta > 1):
                valid_query = row_idx < num_rows
            if valid_query:
                q_producer, q_consumer = pipeline.PipelineTmaAsync.create(
                    num_stages=self.q_stages,
                    producer_group=tma_group,
                    consumer_group=softmax_group,  # Reuse Q consumer mbarriers to sync O store
                    tx_count=cute.size_in_bytes(
                        q_dtype, cute.select(smem_layout_q, mma_modes)
                    ),
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
                cute.arch.mbarrier_init_fence()
                self._sync_work_threads()
                tmem_ptr = cute.arch.retrieve_tmem_ptr(
                    cutlass.Int32, 16, tmem_ptr_smem_ptr
                )
                tmem_offset = 0
                if cutlass.const_expr(k_dtype is not cutlass.BFloat16):
                    tAtK_cvt = cute.make_tensor(
                        cute.recast_ptr(tmem_ptr + tmem_offset, dtype=mma_dtype),
                        tAtK_cvt_fragment.layout,
                    )
                    tAtV_cvt = cute.make_tensor(
                        cute.recast_ptr(tmem_ptr + tmem_offset, dtype=mma_dtype),
                        tAtV_cvt_fragment.layout,
                    )
                    tmem_offset += tcgen05.find_tmem_tensor_col_offset(tAtK_cvt)
                if cutlass.const_expr(k_dtype is cutlass.BFloat16):
                    tAtK_cvt = tAtK_cvt_fragment
                    tAtV_cvt = tAtV_cvt_fragment
                tCtS = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr + tmem_offset, dtype=acc_dtype),
                    tCtS_fragment.layout,
                )
                tmem_offset += tcgen05.find_tmem_tensor_col_offset(tCtS)
                tCtO = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr + tmem_offset, dtype=acc_dtype),
                    tCtO_fragment.layout,
                )
                tmem_offset += tcgen05.find_tmem_tensor_col_offset(tCtO)
                assert tmem_offset <= self.tmem_alloc_cols
                if cutlass.const_expr(
                    self.original_heads_per_kv == 12 and self.problem_shape[2] == 2
                ):
                    if cutlass.const_expr(self.register_budget is not None):
                        if warpgroup_idx not in self.softmax_warpgroup_ids:
                            cute.arch.setmaxregister_decrease(self.register_budget[1])
                else:
                    if cutlass.const_expr(self.original_heads_per_kv == 12):
                        if cutlass.const_expr(self.register_budget is not None):
                            if warpgroup_idx not in self.softmax_warpgroup_ids:
                                cute.arch.setmaxregister_decrease(
                                    self.register_budget[1]
                                )
                if exit_early:
                    noop = None  # early return not supported # noqa: F841
                else:
                    if warp_idx == self.tma_qo_warp_id:
                        q_tile_h = blk_tile_h
                        if cutlass.const_expr(self.direct_query):
                            query_head_tile = heads_per_kv // self.query_group
                            if cutlass.const_expr(
                                query_head_tile != self.original_heads_per_kv
                            ):
                                if cutlass.const_expr(
                                    self.original_heads_per_kv == 12
                                    and self.problem_shape[2] == 2
                                ):
                                    q_tile_h = (query_head_tile, self.query_group)
                                else:
                                    if cutlass.const_expr(
                                        self.original_heads_per_kv == 12
                                    ):
                                        q_tile_h = (query_head_tile, self.query_group)
                                    else:
                                        q_tile_h = (
                                            query_head_tile,
                                            cute.ceil_div(blk_tile_h, query_head_tile),
                                        )
                        gQ = cute.local_tile(
                            mQ,
                            tiler=(q_tile_h, blk_tile_d),
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
                    else:
                        if warpgroup_idx in self.cvt_warpgroup_ids:
                            if cutlass.const_expr(k_dtype is cutlass.BFloat16):
                                if cutlass.const_expr(
                                    self.original_heads_per_kv == 12
                                    and self.problem_shape[2] == 2
                                ):
                                    if cutlass.const_expr(
                                        self.producer_warpgroups == 1
                                    ):
                                        raw_slots = cute.make_rmem_tensor(
                                            static_iters_s, cutlass.Int32
                                        )
                                        if cutlass.const_expr(self.bucket_prefix):
                                            if prefix_bucket_eligible:
                                                index_row = (
                                                    (warpgroup_tidx // warp_threads) * 2
                                                    + lane_idx % 2
                                                    + (lane_idx // 2) * 8
                                                )
                                                for index in cutlass.range_constexpr(
                                                    static_iters_s
                                                ):
                                                    column = index * 128 + index_row
                                                    raw_slots[index] = ordered_prefix[
                                                        column
                                                    ]
                                                # One full-D producer owns both K/V halves.
                                                # Its complete team retires the temporary IDs
                                                # before any thread overwrites their KV stage.
                                                pipeline.NamedBarrier(
                                                    barrier_id=7, num_threads=128
                                                ).arrive_and_wait()
                                            else:
                                                for index in cutlass.range_constexpr(
                                                    static_iters_s
                                                ):
                                                    raw_slots[index] = (
                                                        self._load_sparse_bf16_slot(
                                                            selected_slots,
                                                            row_idx,
                                                            kv_split_idx
                                                            + index * kv_splits,
                                                            warpgroup_tidx,
                                                            lane_idx,
                                                        )
                                                    )
                                        else:
                                            for index in cutlass.range_constexpr(
                                                static_iters_s
                                            ):
                                                raw_slots[index] = (
                                                    self._load_sparse_bf16_slot(
                                                        selected_slots,
                                                        row_idx,
                                                        kv_split_idx
                                                        + index * kv_splits,
                                                        warpgroup_tidx,
                                                        lane_idx,
                                                    )
                                                )
                                        for item in cutlass.range_constexpr(
                                            static_iters_s * 2
                                        ):
                                            if cutlass.const_expr(item == 0):
                                                raw_slot = raw_slots[0]
                                            elif cutlass.const_expr(
                                                item == static_iters_s * 2 - 1
                                            ):
                                                raw_slot = raw_slots[static_iters_s - 1]
                                            elif cutlass.const_expr(item % 2 == 1):
                                                raw_slot = raw_slots[(item + 1) // 2]
                                            else:
                                                raw_slot = raw_slots[item // 2 - 1]
                                            kv_handle = (
                                                cvt_producer.acquire_and_advance()
                                            )
                                            second_kv_handle = (
                                                cvt_producer.acquire_and_advance()
                                            )
                                            self._issue_sparse_stream_item_bf16_full_loaded(
                                                k_iter,
                                                v_iter,
                                                raw_slot,
                                                item,
                                                static_iters_s * 2,
                                                self.kv_stages // _CONVERT_WARPGROUPS,
                                                sKV_vector_address,
                                                warpgroup_tidx,
                                                lane_idx,
                                                kv_smem_dtype,
                                            )
                                            # The same 128 threads publish both phases only
                                            # after their copies of both D halves complete.
                                            cute.arch.cp_async_mbarrier_arrive_noinc(
                                                kv_handle.barrier
                                            )
                                            cute.arch.cp_async_mbarrier_arrive_noinc(
                                                second_kv_handle.barrier
                                            )
                                            if cutlass.const_expr(
                                                self.work_sync_threads != 0
                                            ):
                                                if cutlass.const_expr(
                                                    item + 2 < static_iters_s * 2
                                                ):
                                                    future_item = item + 2
                                                    if cutlass.const_expr(
                                                        future_item
                                                        == static_iters_s * 2 - 1
                                                    ):
                                                        future_slot = raw_slots[
                                                            static_iters_s - 1
                                                        ]
                                                    elif cutlass.const_expr(
                                                        future_item % 2 == 1
                                                    ):
                                                        future_slot = raw_slots[
                                                            (future_item + 1) // 2
                                                        ]
                                                    else:
                                                        future_slot = raw_slots[
                                                            future_item // 2 - 1
                                                        ]
                                                    # Retain both current completion arrivals before hints.
                                                    self._prefetch_sparse_stream_item_bf16_full_loaded(
                                                        k_iter,
                                                        v_iter,
                                                        future_slot,
                                                        future_item,
                                                        static_iters_s * 2,
                                                    )
                                    elif cutlass.const_expr(self.query_group > 1):
                                        convert_phase = (
                                            warpgroup_idx % _CONVERT_WARPGROUPS
                                        )
                                        for _ in cutlass.range(convert_phase):
                                            cvt_producer.advance()
                                        # Each partition ID serves both K and V. Constant indices
                                        # let the compiler keep this small cache in registers.
                                        raw_slots = cute.make_rmem_tensor(
                                            static_iters_s, cutlass.Int32
                                        )
                                        for index in cutlass.range_constexpr(
                                            static_iters_s
                                        ):
                                            raw_slots[index] = (
                                                self._load_sparse_bf16_slot(
                                                    selected_slots,
                                                    row_idx * self.query_group,
                                                    kv_split_idx + index * kv_splits,
                                                    warpgroup_tidx,
                                                    lane_idx,
                                                )
                                            )
                                        for item in cutlass.range_constexpr(
                                            static_iters_s * 2
                                        ):
                                            if cutlass.const_expr(item == 0):
                                                raw_slot = raw_slots[0]
                                            elif cutlass.const_expr(
                                                item == static_iters_s * 2 - 1
                                            ):
                                                raw_slot = raw_slots[static_iters_s - 1]
                                            elif cutlass.const_expr(item % 2 == 1):
                                                raw_slot = raw_slots[(item + 1) // 2]
                                            else:
                                                raw_slot = raw_slots[item // 2 - 1]
                                            kv_handle = (
                                                cvt_producer.acquire_and_advance()
                                            )
                                            self._issue_sparse_stream_item_bf16_loaded(
                                                k_iter,
                                                v_iter,
                                                raw_slot,
                                                item,
                                                iters_s * 2,
                                                self.kv_stages // _CONVERT_WARPGROUPS,
                                                convert_phase * mma_tile_k,
                                                sKV_vector_address,
                                                convert_phase,
                                                warpgroup_tidx,
                                                lane_idx,
                                                kv_smem_dtype,
                                            )
                                            cute.arch.cp_async_mbarrier_arrive_noinc(
                                                kv_handle.barrier
                                            )
                                            for _ in cutlass.range_constexpr(
                                                _CONVERT_WARPGROUPS - 1
                                            ):
                                                cvt_producer.advance()
                                    else:
                                        convert_phase = (
                                            warpgroup_idx % _CONVERT_WARPGROUPS
                                        )
                                        for _ in cutlass.range(convert_phase):
                                            cvt_producer.advance()
                                        # Each partition ID serves both K and V. Constant indices
                                        # let the compiler keep this small cache in registers.
                                        raw_slots = cute.make_rmem_tensor(
                                            static_iters_s, cutlass.Int32
                                        )
                                        for index in cutlass.range_constexpr(
                                            static_iters_s
                                        ):
                                            raw_slots[index] = (
                                                self._load_sparse_bf16_slot(
                                                    selected_slots,
                                                    row_idx,
                                                    kv_split_idx + index * kv_splits,
                                                    warpgroup_tidx,
                                                    lane_idx,
                                                )
                                            )
                                        for item in cutlass.range_constexpr(
                                            static_iters_s * 2
                                        ):
                                            if cutlass.const_expr(item == 0):
                                                raw_slot = raw_slots[0]
                                            elif cutlass.const_expr(
                                                item == static_iters_s * 2 - 1
                                            ):
                                                raw_slot = raw_slots[static_iters_s - 1]
                                            elif cutlass.const_expr(item % 2 == 1):
                                                raw_slot = raw_slots[(item + 1) // 2]
                                            else:
                                                raw_slot = raw_slots[item // 2 - 1]
                                            kv_handle = (
                                                cvt_producer.acquire_and_advance()
                                            )
                                            self._issue_sparse_stream_item_bf16_loaded(
                                                k_iter,
                                                v_iter,
                                                raw_slot,
                                                item,
                                                static_iters_s * 2,
                                                self.kv_stages // _CONVERT_WARPGROUPS,
                                                convert_phase * mma_tile_k,
                                                sKV_vector_address,
                                                convert_phase,
                                                warpgroup_tidx,
                                                lane_idx,
                                                kv_smem_dtype,
                                            )
                                            if cutlass.const_expr(
                                                self.kv_splits < _WIDE_NUM_SPLITS
                                                and not (
                                                    self.bf16_smem_slots == 3
                                                    and self.bf16_async_ready
                                                )
                                                and item + self.bf16_smem_slots
                                                < static_iters_s * 2
                                            ):
                                                self._prefetch_sparse_stream_item(
                                                    k_iter,
                                                    v_iter,
                                                    selected_slots,
                                                    row_idx,
                                                    kv_split_idx,
                                                    kv_splits,
                                                    item + self.bf16_smem_slots,
                                                    static_iters_s * 2,
                                                    convert_phase * mma_tile_k,
                                                    warpgroup_tidx,
                                                )
                                            if cutlass.const_expr(
                                                self.bf16_async_ready
                                            ):
                                                # Each of the 128 producers contributes one asynchronous
                                                # arrival after its copies finish. No immediate ready
                                                # signal and no producer wait on the just-issued copy.
                                                cute.arch.cp_async_mbarrier_arrive_noinc(
                                                    kv_handle.barrier
                                                )
                                            else:
                                                cute.arch.cp_async_wait_group(0)
                                                cute.arch.fence_view_async_shared()
                                                kv_handle.commit()
                                            for _ in cutlass.range_constexpr(
                                                _CONVERT_WARPGROUPS - 1
                                            ):
                                                cvt_producer.advance()
                                else:
                                    if cutlass.const_expr(
                                        self.original_heads_per_kv == 12
                                    ):
                                        if cutlass.const_expr(
                                            self.producer_warpgroups == 1
                                        ):
                                            raw_slots = cute.make_rmem_tensor(
                                                static_iters_s, cutlass.Int32
                                            )
                                            if cutlass.const_expr(self.bucket_prefix):
                                                if prefix_bucket_eligible:
                                                    index_row = (
                                                        (warpgroup_tidx // warp_threads)
                                                        * 2
                                                        + lane_idx % 2
                                                        + (lane_idx // 2) * 8
                                                    )
                                                    for (
                                                        index
                                                    ) in cutlass.range_constexpr(
                                                        static_iters_s
                                                    ):
                                                        column = (
                                                            kv_split_idx
                                                            + index * kv_splits
                                                        ) * 128 + index_row
                                                        raw_slots[index] = (
                                                            ordered_prefix[column]
                                                        )
                                                    # All128 producers cache every ID before overwriting KV bytes.
                                                    pipeline.NamedBarrier(
                                                        barrier_id=7, num_threads=128
                                                    ).arrive_and_wait()
                                                else:
                                                    for (
                                                        index
                                                    ) in cutlass.range_constexpr(
                                                        static_iters_s
                                                    ):
                                                        raw_slots[index] = (
                                                            self._load_sparse_bf16_slot(
                                                                selected_slots,
                                                                row_idx
                                                                * self.query_group,
                                                                kv_split_idx
                                                                + index * kv_splits,
                                                                warpgroup_tidx,
                                                                lane_idx,
                                                            )
                                                        )
                                            else:
                                                for index in cutlass.range_constexpr(
                                                    static_iters_s
                                                ):
                                                    raw_slots[index] = (
                                                        self._load_sparse_bf16_slot(
                                                            selected_slots,
                                                            row_idx * self.query_group,
                                                            kv_split_idx
                                                            + index * kv_splits,
                                                            warpgroup_tidx,
                                                            lane_idx,
                                                        )
                                                    )
                                            for item in cutlass.range_constexpr(
                                                static_iters_s * 2
                                            ):
                                                if cutlass.const_expr(item == 0):
                                                    raw_slot = raw_slots[0]
                                                elif cutlass.const_expr(
                                                    item == static_iters_s * 2 - 1
                                                ):
                                                    raw_slot = raw_slots[
                                                        static_iters_s - 1
                                                    ]
                                                elif cutlass.const_expr(item % 2 == 1):
                                                    raw_slot = raw_slots[
                                                        (item + 1) // 2
                                                    ]
                                                else:
                                                    raw_slot = raw_slots[item // 2 - 1]
                                                kv_handle = (
                                                    cvt_producer.acquire_and_advance()
                                                )
                                                second_kv_handle = (
                                                    cvt_producer.acquire_and_advance()
                                                )
                                                self._issue_sparse_stream_item_bf16_full_loaded(
                                                    k_iter,
                                                    v_iter,
                                                    raw_slot,
                                                    item,
                                                    static_iters_s * 2,
                                                    self.kv_stages
                                                    // _CONVERT_WARPGROUPS,
                                                    sKV_vector_address,
                                                    warpgroup_tidx,
                                                    lane_idx,
                                                    kv_smem_dtype,
                                                )
                                                # Both128-reader phases arrive only after all full-D copies.
                                                cute.arch.cp_async_mbarrier_arrive_noinc(
                                                    kv_handle.barrier
                                                )
                                                cute.arch.cp_async_mbarrier_arrive_noinc(
                                                    second_kv_handle.barrier
                                                )
                                        elif cutlass.const_expr(self.query_group > 1):
                                            convert_phase = (
                                                warpgroup_idx % _CONVERT_WARPGROUPS
                                            )
                                            for _ in cutlass.range(convert_phase):
                                                cvt_producer.advance()
                                            # Each partition ID serves both K and V. Constant indices
                                            # let the compiler keep this small cache in registers.
                                            raw_slots = cute.make_rmem_tensor(
                                                static_iters_s, cutlass.Int32
                                            )
                                            for index in cutlass.range_constexpr(
                                                static_iters_s
                                            ):
                                                raw_slots[index] = (
                                                    self._load_sparse_bf16_slot(
                                                        selected_slots,
                                                        row_idx * self.query_group,
                                                        kv_split_idx
                                                        + index * kv_splits,
                                                        warpgroup_tidx,
                                                        lane_idx,
                                                    )
                                                )
                                            for item in cutlass.range_constexpr(
                                                static_iters_s * 2
                                            ):
                                                if cutlass.const_expr(item == 0):
                                                    raw_slot = raw_slots[0]
                                                elif cutlass.const_expr(
                                                    item == static_iters_s * 2 - 1
                                                ):
                                                    raw_slot = raw_slots[
                                                        static_iters_s - 1
                                                    ]
                                                elif cutlass.const_expr(item % 2 == 1):
                                                    raw_slot = raw_slots[
                                                        (item + 1) // 2
                                                    ]
                                                else:
                                                    raw_slot = raw_slots[item // 2 - 1]
                                                kv_handle = (
                                                    cvt_producer.acquire_and_advance()
                                                )
                                                self._issue_sparse_stream_item_bf16_loaded(
                                                    k_iter,
                                                    v_iter,
                                                    raw_slot,
                                                    item,
                                                    iters_s * 2,
                                                    self.kv_stages
                                                    // _CONVERT_WARPGROUPS,
                                                    convert_phase * mma_tile_k,
                                                    sKV_vector_address,
                                                    convert_phase,
                                                    warpgroup_tidx,
                                                    lane_idx,
                                                    kv_smem_dtype,
                                                )
                                                cute.arch.cp_async_mbarrier_arrive_noinc(
                                                    kv_handle.barrier
                                                )
                                                for _ in cutlass.range_constexpr(
                                                    _CONVERT_WARPGROUPS - 1
                                                ):
                                                    cvt_producer.advance()
                                        else:
                                            convert_phase = (
                                                warpgroup_idx % _CONVERT_WARPGROUPS
                                            )
                                            for _ in cutlass.range(convert_phase):
                                                cvt_producer.advance()
                                            # Each partition ID serves both K and V. Constant indices
                                            # let the compiler keep this small cache in registers.
                                            raw_slots = cute.make_rmem_tensor(
                                                static_iters_s, cutlass.Int32
                                            )
                                            if cutlass.const_expr(self.bucket_prefix):
                                                if prefix_bucket_eligible:
                                                    index_row = (
                                                        (warpgroup_tidx // warp_threads)
                                                        * 2
                                                        + lane_idx % 2
                                                        + (lane_idx // 2) * 8
                                                    )
                                                    for (
                                                        index
                                                    ) in cutlass.range_constexpr(
                                                        static_iters_s
                                                    ):
                                                        column = (
                                                            kv_split_idx
                                                            + index * kv_splits
                                                        ) * 128 + index_row
                                                        raw_slots[index] = (
                                                            ordered_prefix[column]
                                                        )
                                                    # Both 128-thread half-D producers must finish
                                                    # reading every ID before either reuses KV bytes.
                                                    pipeline.NamedBarrier(
                                                        barrier_id=7, num_threads=256
                                                    ).arrive_and_wait()
                                                else:
                                                    for (
                                                        index
                                                    ) in cutlass.range_constexpr(
                                                        static_iters_s
                                                    ):
                                                        raw_slots[index] = (
                                                            self._load_sparse_bf16_slot(
                                                                selected_slots,
                                                                row_idx,
                                                                kv_split_idx
                                                                + index * kv_splits,
                                                                warpgroup_tidx,
                                                                lane_idx,
                                                            )
                                                        )
                                            else:
                                                for index in cutlass.range_constexpr(
                                                    static_iters_s
                                                ):
                                                    raw_slots[index] = (
                                                        self._load_sparse_bf16_slot(
                                                            selected_slots,
                                                            row_idx,
                                                            kv_split_idx
                                                            + index * kv_splits,
                                                            warpgroup_tidx,
                                                            lane_idx,
                                                        )
                                                    )
                                            for item in cutlass.range_constexpr(
                                                static_iters_s * 2
                                            ):
                                                if cutlass.const_expr(item == 0):
                                                    raw_slot = raw_slots[0]
                                                elif cutlass.const_expr(
                                                    item == static_iters_s * 2 - 1
                                                ):
                                                    raw_slot = raw_slots[
                                                        static_iters_s - 1
                                                    ]
                                                elif cutlass.const_expr(item % 2 == 1):
                                                    raw_slot = raw_slots[
                                                        (item + 1) // 2
                                                    ]
                                                else:
                                                    raw_slot = raw_slots[item // 2 - 1]
                                                kv_handle = (
                                                    cvt_producer.acquire_and_advance()
                                                )
                                                self._issue_sparse_stream_item_bf16_loaded(
                                                    k_iter,
                                                    v_iter,
                                                    raw_slot,
                                                    item,
                                                    static_iters_s * 2,
                                                    self.kv_stages
                                                    // _CONVERT_WARPGROUPS,
                                                    convert_phase * mma_tile_k,
                                                    sKV_vector_address,
                                                    convert_phase,
                                                    warpgroup_tidx,
                                                    lane_idx,
                                                    kv_smem_dtype,
                                                )
                                                if cutlass.const_expr(
                                                    self.kv_splits < _WIDE_NUM_SPLITS
                                                    and not (
                                                        self.bf16_smem_slots == 3
                                                        and self.bf16_async_ready
                                                    )
                                                    and item + self.bf16_smem_slots
                                                    < static_iters_s * 2
                                                ):
                                                    self._prefetch_sparse_stream_item(
                                                        k_iter,
                                                        v_iter,
                                                        selected_slots,
                                                        row_idx,
                                                        kv_split_idx,
                                                        kv_splits,
                                                        item + self.bf16_smem_slots,
                                                        static_iters_s * 2,
                                                        convert_phase * mma_tile_k,
                                                        warpgroup_tidx,
                                                    )
                                                if cutlass.const_expr(
                                                    self.bf16_async_ready
                                                ):
                                                    # Each of the 128 producers contributes one asynchronous
                                                    # arrival after its copies finish. No immediate ready
                                                    # signal and no producer wait on the just-issued copy.
                                                    cute.arch.cp_async_mbarrier_arrive_noinc(
                                                        kv_handle.barrier
                                                    )
                                                else:
                                                    cute.arch.cp_async_wait_group(0)
                                                    cute.arch.fence_view_async_shared()
                                                    kv_handle.commit()
                                                for _ in cutlass.range_constexpr(
                                                    _CONVERT_WARPGROUPS - 1
                                                ):
                                                    cvt_producer.advance()
                                    else:
                                        if cutlass.const_expr(self.query_group > 1):
                                            convert_phase = (
                                                warpgroup_idx % _CONVERT_WARPGROUPS
                                            )
                                            for _ in cutlass.range(convert_phase):
                                                cvt_producer.advance()
                                            # Each partition ID serves both K and V. Constant indices
                                            # let the compiler keep this small cache in registers.
                                            raw_slots = cute.make_rmem_tensor(
                                                static_iters_s, cutlass.Int32
                                            )
                                            for index in cutlass.range_constexpr(
                                                static_iters_s
                                            ):
                                                raw_slots[index] = (
                                                    self._load_sparse_bf16_slot(
                                                        selected_slots,
                                                        row_idx * self.query_group,
                                                        kv_split_idx
                                                        + index * kv_splits,
                                                        warpgroup_tidx,
                                                        lane_idx,
                                                    )
                                                )
                                            for item in cutlass.range_constexpr(
                                                static_iters_s * 2
                                            ):
                                                if cutlass.const_expr(item == 0):
                                                    raw_slot = raw_slots[0]
                                                elif cutlass.const_expr(
                                                    item == static_iters_s * 2 - 1
                                                ):
                                                    raw_slot = raw_slots[
                                                        static_iters_s - 1
                                                    ]
                                                elif cutlass.const_expr(item % 2 == 1):
                                                    raw_slot = raw_slots[
                                                        (item + 1) // 2
                                                    ]
                                                else:
                                                    raw_slot = raw_slots[item // 2 - 1]
                                                kv_handle = (
                                                    cvt_producer.acquire_and_advance()
                                                )
                                                self._issue_sparse_stream_item_bf16_loaded(
                                                    k_iter,
                                                    v_iter,
                                                    raw_slot,
                                                    item,
                                                    iters_s * 2,
                                                    self.kv_stages
                                                    // _CONVERT_WARPGROUPS,
                                                    convert_phase * mma_tile_k,
                                                    sKV_vector_address,
                                                    convert_phase,
                                                    warpgroup_tidx,
                                                    lane_idx,
                                                    kv_smem_dtype,
                                                )
                                                cute.arch.cp_async_mbarrier_arrive_noinc(
                                                    kv_handle.barrier
                                                )
                                                for _ in cutlass.range_constexpr(
                                                    _CONVERT_WARPGROUPS - 1
                                                ):
                                                    cvt_producer.advance()
                                        else:
                                            convert_phase = (
                                                warpgroup_idx % _CONVERT_WARPGROUPS
                                            )
                                            for _ in cutlass.range(convert_phase):
                                                cvt_producer.advance()
                                            # Each partition ID serves both K and V. Constant indices
                                            # let the compiler keep this small cache in registers.
                                            raw_slots = cute.make_rmem_tensor(
                                                static_iters_s, cutlass.Int32
                                            )
                                            if prefix_bucket_eligible:
                                                index_row = (
                                                    (warpgroup_tidx // warp_threads) * 2
                                                    + lane_idx % 2
                                                    + (lane_idx // 2) * 8
                                                )
                                                for index in cutlass.range_constexpr(
                                                    static_iters_s
                                                ):
                                                    column = (
                                                        kv_split_idx + index * kv_splits
                                                    ) * 128 + index_row
                                                    raw_slots[index] = ordered_prefix[
                                                        column
                                                    ]
                                                # Both producer warpgroups cache every ID before
                                                # either can overwrite the temporary KV stage.
                                                pipeline.NamedBarrier(
                                                    barrier_id=7, num_threads=256
                                                ).arrive_and_wait()
                                            else:
                                                for index in cutlass.range_constexpr(
                                                    static_iters_s
                                                ):
                                                    raw_slots[index] = (
                                                        self._load_sparse_bf16_slot(
                                                            selected_slots,
                                                            row_idx,
                                                            kv_split_idx
                                                            + index * kv_splits,
                                                            warpgroup_tidx,
                                                            lane_idx,
                                                        )
                                                    )
                                            for item in cutlass.range_constexpr(
                                                static_iters_s * 2
                                            ):
                                                if cutlass.const_expr(item == 0):
                                                    raw_slot = raw_slots[0]
                                                elif cutlass.const_expr(
                                                    item == static_iters_s * 2 - 1
                                                ):
                                                    raw_slot = raw_slots[
                                                        static_iters_s - 1
                                                    ]
                                                elif cutlass.const_expr(item % 2 == 1):
                                                    raw_slot = raw_slots[
                                                        (item + 1) // 2
                                                    ]
                                                else:
                                                    raw_slot = raw_slots[item // 2 - 1]
                                                kv_handle = (
                                                    cvt_producer.acquire_and_advance()
                                                )
                                                self._issue_sparse_stream_item_bf16_loaded(
                                                    k_iter,
                                                    v_iter,
                                                    raw_slot,
                                                    item,
                                                    static_iters_s * 2,
                                                    self.kv_stages
                                                    // _CONVERT_WARPGROUPS,
                                                    convert_phase * mma_tile_k,
                                                    sKV_vector_address,
                                                    convert_phase,
                                                    warpgroup_tidx,
                                                    lane_idx,
                                                    kv_smem_dtype,
                                                )
                                                if cutlass.const_expr(
                                                    self.kv_splits < _WIDE_NUM_SPLITS
                                                    and not (
                                                        self.bf16_smem_slots == 3
                                                        and self.bf16_async_ready
                                                    )
                                                    and item + self.bf16_smem_slots
                                                    < static_iters_s * 2
                                                ):
                                                    self._prefetch_sparse_stream_item(
                                                        k_iter,
                                                        v_iter,
                                                        selected_slots,
                                                        row_idx,
                                                        kv_split_idx,
                                                        kv_splits,
                                                        item + self.bf16_smem_slots,
                                                        static_iters_s * 2,
                                                        convert_phase * mma_tile_k,
                                                        warpgroup_tidx,
                                                    )
                                                if cutlass.const_expr(
                                                    self.bf16_async_ready
                                                ):
                                                    # Each of the 128 producers contributes one asynchronous
                                                    # arrival after its copies finish. No immediate ready
                                                    # signal and no producer wait on the just-issued copy.
                                                    cute.arch.cp_async_mbarrier_arrive_noinc(
                                                        kv_handle.barrier
                                                    )
                                                else:
                                                    cute.arch.cp_async_wait_group(0)
                                                    cute.arch.fence_view_async_shared()
                                                    kv_handle.commit()
                                                for _ in cutlass.range_constexpr(
                                                    _CONVERT_WARPGROUPS - 1
                                                ):
                                                    cvt_producer.advance()
                                                if cutlass.const_expr(
                                                    self.cached_bf16_lookahead
                                                    and item + self.bf16_smem_slots
                                                    < static_iters_s * 2
                                                ):
                                                    # Publish current-copy completion before hinting
                                                    # a future tile in the same cached copy order.
                                                    future_item = (
                                                        item + self.bf16_smem_slots
                                                    )
                                                    if cutlass.const_expr(
                                                        future_item
                                                        == static_iters_s * 2 - 1
                                                    ):
                                                        future_slot = raw_slots[
                                                            static_iters_s - 1
                                                        ]
                                                    elif cutlass.const_expr(
                                                        future_item % 2 == 1
                                                    ):
                                                        future_slot = raw_slots[
                                                            (future_item + 1) // 2
                                                        ]
                                                    else:
                                                        future_slot = raw_slots[
                                                            future_item // 2 - 1
                                                        ]
                                                    self._prefetch_sparse_stream_item_bf16_loaded(
                                                        k_iter,
                                                        v_iter,
                                                        future_slot,
                                                        future_item,
                                                        static_iters_s * 2,
                                                        convert_phase * mma_tile_k,
                                                    )
                            else:
                                cvt_type = cutlass.Float32
                                assert tiles_dk % _CONVERT_WARPGROUPS == 0
                                assert (tiles_dm * tiles_sk) % _CONVERT_WARPGROUPS == 0
                                convert_phase = warpgroup_idx % _CONVERT_WARPGROUPS
                                for _ in cutlass.range(convert_phase):
                                    cvt_producer.advance()
                                cvt_load_nbar = pipeline.NamedBarrier(
                                    barrier_id=4 + convert_phase,
                                    num_threads=warpgroup_threads,
                                )
                                mma_k_bits = mma_tile_k * mma_dtype.width
                                tmem_store_atom_k = cute.make_copy_atom(
                                    tcgen05.St16x256bOp(
                                        tcgen05.Repetition(mma_k_bits // 256)
                                    ),
                                    mma_dtype,
                                )
                                smem_load_op_k = cute.nvgpu.warp.LdMatrix8x16x8bOp(
                                    num_matrices=4,
                                )
                                smem_load_atom_k = cute.make_copy_atom(
                                    smem_load_op_k, kv_smem_dtype
                                )
                                tmem_store_k = tcgen05.make_tmem_copy(
                                    tmem_store_atom_k, tAtK_cvt[mma_dice + (0,)]
                                )
                                thr_store_k = tmem_store_k.get_slice(warpgroup_tidx)
                                tKrK_cvt_shape = thr_store_k.partition_S(
                                    tAtK_cvt
                                ).shape[:-1]
                                tKtK_cvt = thr_store_k.partition_D(tAtK_cvt)
                                smem_load_k = cute.make_tiled_copy_S(
                                    smem_load_atom_k, tmem_store_k
                                )
                                thr_load_k = smem_load_k.get_slice(warpgroup_tidx)
                                tKsK = thr_load_k.partition_S(tAsK)
                                tKrK_shape = thr_load_k.partition_D(tAsK).shape[:-1]
                                tmem_store_atom_v = cute.make_copy_atom(
                                    tcgen05.St16x256bOp(
                                        tcgen05.Repetition(mma_k_bits // 256)
                                    ),
                                    mma_dtype,
                                )
                                smem_load_op_v = cute.nvgpu.warp.LdMatrix16x16x8bOp(
                                    transpose=True,
                                    num_matrices=2,
                                )
                                smem_load_atom_v = cute.make_copy_atom(
                                    smem_load_op_v, kv_smem_dtype
                                )
                                tmem_store_v = tcgen05.make_tmem_copy(
                                    tmem_store_atom_v, tAtV_cvt[mma_dice + (0,)]
                                )
                                thr_store_v = tmem_store_v.get_slice(warpgroup_tidx)
                                tVrV_cvt_shape = thr_store_v.partition_S(
                                    tAtV_cvt
                                ).shape[:-1]
                                tVtV_cvt = thr_store_v.partition_D(tAtV_cvt)
                                smem_load_v = cute.make_tiled_copy_S(
                                    smem_load_atom_v, tmem_store_v
                                )
                                thr_load_v = smem_load_v.get_slice(warpgroup_tidx)
                                tVsV = thr_load_v.partition_S(tAsV)
                                tVrV_shape = thr_load_v.partition_D(tAsV).shape[:-1]
                                self._issue_sparse_tile_for_dtype(
                                    k_iter,
                                    selected_slots,
                                    row_idx,
                                    kv_split_idx,
                                    convert_phase * mma_tile_k,
                                    sKV_vector_address,
                                    convert_phase,
                                    warpgroup_tidx,
                                    lane_idx,
                                    kv_smem_dtype,
                                )
                                for s in cutlass.range(prefetch_iters + static_iters_s):
                                    if s < static_iters_s:
                                        # Convert and scale K
                                        for _ in cutlass.range(
                                            tiles_dk // _CONVERT_WARPGROUPS, unroll=2
                                        ):
                                            # Issue the next logical K-or-V item before
                                            # consuming K_s. One pending cp.async group overlaps
                                            # its latency with conversion and UMMA.
                                            if s == 0:
                                                if static_iters_s > 1:
                                                    self._issue_sparse_tile_for_dtype(
                                                        k_iter,
                                                        selected_slots,
                                                        row_idx,
                                                        kv_split_idx + kv_splits,
                                                        convert_phase * mma_tile_k,
                                                        sKV_vector_address,
                                                        _CONVERT_WARPGROUPS
                                                        + convert_phase,
                                                        warpgroup_tidx,
                                                        lane_idx,
                                                        kv_smem_dtype,
                                                    )
                                                else:
                                                    # With one tile, the next ring item is V0.
                                                    self._issue_sparse_tile_for_dtype(
                                                        v_iter,
                                                        selected_slots,
                                                        row_idx,
                                                        kv_split_idx,
                                                        convert_phase * mma_tile_m,
                                                        sKV_vector_address,
                                                        _CONVERT_WARPGROUPS
                                                        + convert_phase,
                                                        warpgroup_tidx,
                                                        lane_idx,
                                                        kv_smem_dtype,
                                                    )
                                            else:
                                                self._issue_sparse_tile_for_dtype(
                                                    v_iter,
                                                    selected_slots,
                                                    row_idx,
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
                                                k_smem_stage = (
                                                    _CONVERT_WARPGROUPS + convert_phase
                                                )
                                            cvt_load_nbar.arrive_and_wait()

                                            tKrK = cute.make_rmem_tensor(
                                                tKrK_shape, kv_smem_dtype
                                            )
                                            tKrK_cvt = cute.make_rmem_tensor(
                                                tKrK_cvt_shape, mma_dtype
                                            )

                                            cute.copy(
                                                thr_load_k,
                                                tKsK[cpy_dice + (k_smem_stage,)],
                                                tKrK,
                                            )
                                            cute.arch.fence_view_async_shared()
                                            if cutlass.const_expr(static_iters_s > 1):
                                                # A faster warp may otherwise refill this ring
                                                # slot while a peer warp still reads its K tile.
                                                cvt_load_nbar.arrive_and_wait()

                                            tKrK_ssa = (
                                                tKrK.load().to(cvt_type).to(mma_dtype)
                                            )
                                            tKrK_cvt.store(
                                                tKrK_ssa.reshape(tKrK_cvt_shape)
                                            )

                                            cvt_handle = (
                                                cvt_producer.acquire_and_advance()
                                            )
                                            cute.copy(
                                                thr_store_k,
                                                tKrK_cvt,
                                                tKtK_cvt[
                                                    cpy_dice + (cvt_handle.index,)
                                                ],
                                            )
                                            cute.arch.fence_view_async_tmem_store()
                                            cvt_handle.commit()

                                            # Advance again for multiple warpgroups
                                            for _ in cutlass.range_constexpr(
                                                _CONVERT_WARPGROUPS - 1
                                            ):
                                                cvt_producer.advance()

                                    if s >= prefetch_iters:
                                        # Convert and scale V
                                        for _ in cutlass.range(
                                            tiles_dm * tiles_sk // _CONVERT_WARPGROUPS,
                                            unroll=2,
                                        ):
                                            # V_(s-1) is resident/in flight. Refill the other
                                            # ring slot with K_(s+1), or the final V item.
                                            if s < static_iters_s:
                                                if s + 1 < static_iters_s:
                                                    self._issue_sparse_tile_for_dtype(
                                                        k_iter,
                                                        selected_slots,
                                                        row_idx,
                                                        kv_split_idx
                                                        + (s + 1) * kv_splits,
                                                        convert_phase * mma_tile_k,
                                                        sKV_vector_address,
                                                        _CONVERT_WARPGROUPS
                                                        + convert_phase,
                                                        warpgroup_tidx,
                                                        lane_idx,
                                                        kv_smem_dtype,
                                                    )
                                                else:
                                                    self._issue_sparse_tile_for_dtype(
                                                        v_iter,
                                                        selected_slots,
                                                        row_idx,
                                                        kv_split_idx + s * kv_splits,
                                                        convert_phase * mma_tile_m,
                                                        sKV_vector_address,
                                                        _CONVERT_WARPGROUPS
                                                        + convert_phase,
                                                        warpgroup_tidx,
                                                        lane_idx,
                                                        kv_smem_dtype,
                                                    )
                                                cute.arch.cp_async_wait_group(1)
                                            else:
                                                cute.arch.cp_async_wait_group(0)
                                            v_smem_stage = convert_phase
                                            if s == static_iters_s:
                                                v_smem_stage = (
                                                    _CONVERT_WARPGROUPS + convert_phase
                                                )
                                            cvt_load_nbar.arrive_and_wait()

                                            tVrV = cute.make_rmem_tensor(
                                                tVrV_shape, kv_smem_dtype
                                            )
                                            tVrV_cvt = cute.make_rmem_tensor(
                                                tVrV_cvt_shape, mma_dtype
                                            )

                                            cute.copy(
                                                thr_load_v,
                                                tVsV[cpy_dice + (v_smem_stage,)],
                                                tVrV,
                                            )
                                            cute.arch.fence_view_async_shared()
                                            if cutlass.const_expr(static_iters_s > 1):
                                                cvt_load_nbar.arrive_and_wait()

                                            tVrV_ssa = (
                                                tVrV.load().to(cvt_type).to(mma_dtype)
                                            )
                                            tVrV_cvt.store(
                                                tVrV_ssa.reshape(tVrV_cvt_shape)
                                            )

                                            cvt_handle = (
                                                cvt_producer.acquire_and_advance()
                                            )
                                            cute.copy(
                                                thr_store_v,
                                                tVrV_cvt,
                                                tVtV_cvt[
                                                    cpy_dice + (cvt_handle.index,)
                                                ],
                                            )
                                            cute.arch.fence_view_async_tmem_store()
                                            cvt_handle.commit()

                                            # Advance again for multiple warpgroups
                                            for _ in cutlass.range_constexpr(
                                                _CONVERT_WARPGROUPS - 1
                                            ):
                                                cvt_producer.advance()
                        else:
                            if warp_idx == self.mma_kq_warp_id:
                                tBsQ_desc = thrblk_mma_kq.make_fragment_B(tBsQ)
                                for dk in cutlass.range_constexpr(tiles_dk):
                                    q_consumer.wait_and_advance()
                                s_token = True  # Producer always acquires first
                                for s in cutlass.range(iters_s):
                                    k_token = cvt_consumer.try_wait()
                                    tiled_mma_kq.set(tcgen05.Field.ACCUMULATE, False)
                                    s_handle = s_producer.acquire_and_advance(s_token)
                                    for dk in cutlass.range_constexpr(tiles_dk):
                                        is_last_iter = dk == tiles_dk - 1
                                        k_handle = cvt_consumer.wait_and_advance(
                                            k_token
                                        )
                                        if cutlass.const_expr(
                                            k_dtype is cutlass.BFloat16
                                            and self.bf16_async_ready
                                        ):
                                            cute.arch.fence_view_async_shared()
                                        # Signal BMM2 to start
                                        if is_last_iter:
                                            if s & 1:
                                                mma_kq_odd_nbar.arrive()
                                            else:
                                                mma_kq_even_nbar.arrive()
                                        for mma_k in cutlass.range_constexpr(
                                            cute.size(tAtK_cvt, mode=[2])
                                        ):
                                            cute.gemm(
                                                tiled_mma_kq,
                                                tCtS[mma_dice + (s_handle.index,)],
                                                tAtK_cvt[
                                                    None, None, mma_k, k_handle.index
                                                ],
                                                tBsQ_desc[None, None, mma_k, dk],
                                                tCtS[mma_dice + (s_handle.index,)],
                                            )
                                            if dk == 0 and mma_k == 0:
                                                tiled_mma_kq.set(
                                                    tcgen05.Field.ACCUMULATE, True
                                                )
                                        k_handle.release()
                                        if not is_last_iter:
                                            k_token = cvt_consumer.try_wait()
                                    s_handle.commit()
                                    if cutlass.const_expr(
                                        self.original_heads_per_kv == 12
                                        and self.problem_shape[2] == 2
                                    ):
                                        if cutlass.const_expr(self.sp_stages == 1):
                                            if s == 0:
                                                # Only the first S slot starts empty. Reuse must
                                                # wait for its first softmax reader to retire.
                                                s_token = s_producer.try_acquire()
                                    if s > 0:
                                        for _ in cutlass.range_constexpr(
                                            tiles_dm * tiles_sk
                                        ):
                                            cvt_consumer.advance()
                                        pipeline.NamedBarrier(
                                            barrier_id=3 + 9 * ((s - 1) % 2),
                                            num_threads=64,
                                        ).arrive_and_wait()
                                        s_token = s_producer.try_acquire()
                                if cutlass.const_expr(
                                    self.query_rows_per_cta > 1 or self.reuse_work
                                ):
                                    # The final PV tile also arrives on this named barrier.
                                    # Retire its matching QK arrival before the next query.
                                    pipeline.NamedBarrier(
                                        barrier_id=3 + 9 * ((iters_s - 1) % 2),
                                        num_threads=64,
                                    ).arrive_and_wait()
                            else:
                                if warp_idx == self.mma_vp_warp_id:
                                    tiled_mma_vp.set(tcgen05.Field.ACCUMULATE, True)
                                    tBsP_desc = thrblk_mma_vp.make_fragment_B(tBsP_nk)
                                    for _ in cutlass.range_constexpr(tiles_dk):
                                        cvt_consumer.advance()
                                    mma_kq_even_nbar.arrive_and_wait()
                                    p_token = False
                                    o_token = True  # Producer always acquires first
                                    for s in cutlass.range(iters_s):
                                        # Advance and wait for BMM1
                                        if s < iters_s - 1:
                                            for _ in cutlass.range_constexpr(tiles_dk):
                                                cvt_consumer.advance()
                                            # V_s follows K_{s+1} in the shared K/V stream.
                                            if (s + 1) & 1:
                                                mma_kq_odd_nbar.arrive_and_wait()
                                            else:
                                                mma_kq_even_nbar.arrive_and_wait()
                                            p_token = p_consumer.try_wait()

                                        # BMM2
                                        v_token = cvt_consumer.try_wait()
                                        p_handle = p_consumer.wait_and_advance(p_token)
                                        o_handle = o_producer.acquire_and_advance(
                                            o_token
                                        )
                                        for sk in cutlass.range_constexpr(tiles_sk):
                                            for dm in cutlass.range_constexpr(tiles_dm):
                                                is_last_iter = (
                                                    sk == tiles_sk - 1
                                                    and dm == tiles_dm - 1
                                                )
                                                v_handle = (
                                                    cvt_consumer.wait_and_advance(
                                                        v_token
                                                    )
                                                )
                                                if cutlass.const_expr(
                                                    v_dtype is cutlass.BFloat16
                                                    and self.bf16_async_ready
                                                ):
                                                    cute.arch.fence_view_async_shared()
                                                # Signal BMM1 to start
                                                if is_last_iter:
                                                    pipeline.NamedBarrier(
                                                        barrier_id=3 + 9 * (s % 2),
                                                        num_threads=64,
                                                    ).arrive()
                                                for mma_k in cutlass.range_constexpr(
                                                    cute.size(tAtV_cvt, mode=[2])
                                                ):
                                                    cute.gemm(
                                                        tiled_mma_vp,
                                                        tCtO[mma_dice + (dm, 0)],
                                                        tAtV_cvt[
                                                            None,
                                                            None,
                                                            mma_k,
                                                            v_handle.index,
                                                        ],
                                                        tBsP_desc[
                                                            None,
                                                            None,
                                                            mma_k,
                                                            sk,
                                                            p_handle.index,
                                                        ],
                                                        tCtO[mma_dice + (dm, 0)],
                                                    )
                                                v_handle.release()
                                                if not is_last_iter:
                                                    v_token = cvt_consumer.try_wait()
                                        p_handle.release()
                                        o_handle.commit()
                                        o_token = o_producer.try_acquire()
                                    o_producer.tail()
                                    if cutlass.const_expr(
                                        self.query_rows_per_cta == 1
                                        and not self.caller_allocates_scratch
                                    ):
                                        cute.arch.relinquish_tmem_alloc_permit()
                                        cute.arch.dealloc_tmem(
                                            tmem_ptr, self.tmem_alloc_cols
                                        )
                                else:
                                    if warpgroup_idx in self.softmax_warpgroup_ids:
                                        if cutlass.const_expr(
                                            self.original_heads_per_kv == 12
                                            and self.problem_shape[2] == 2
                                        ):
                                            if cutlass.const_expr(self.head_tile == 72):
                                                # N72 uses two32-column consumers and one8-column consumer.
                                                if warpgroup_idx < 2:
                                                    self._softmax_work(
                                                        32,
                                                        warpgroup_idx * 32,
                                                        warpgroup_idx,
                                                        acc_dtype,
                                                        blk_tile_d,
                                                        blk_tile_h,
                                                        blk_tile_s,
                                                        head_offset,
                                                        heads_per_kv,
                                                        iters_s,
                                                        kv_split_idx,
                                                        kv_splits,
                                                        lane_idx,
                                                        mma_dice,
                                                        mma_dtype,
                                                        mma_tile_m,
                                                        mma_tile_n,
                                                        o_consumer,
                                                        o_dtype,
                                                        p_producer,
                                                        reduction_heads,
                                                        row_idx,
                                                        sL,
                                                        sM,
                                                        s_consumer,
                                                        scale_qs,
                                                        scale_qs_log2_e,
                                                        selected_slots,
                                                        softmax_nbar,
                                                        tAsK,
                                                        tCsL,
                                                        tCsM,
                                                        tCsO,
                                                        tCsP,
                                                        tCtO,
                                                        tCtS,
                                                        thrblk_mma_kq,
                                                        thrblk_mma_vp,
                                                        tiles_dm,
                                                        warpgroup_tidx,
                                                        warpgroup_widx,
                                                    )
                                                else:
                                                    self._softmax_work(
                                                        8,
                                                        64,
                                                        8,
                                                        acc_dtype,
                                                        blk_tile_d,
                                                        blk_tile_h,
                                                        blk_tile_s,
                                                        head_offset,
                                                        heads_per_kv,
                                                        iters_s,
                                                        kv_split_idx,
                                                        kv_splits,
                                                        lane_idx,
                                                        mma_dice,
                                                        mma_dtype,
                                                        mma_tile_m,
                                                        mma_tile_n,
                                                        o_consumer,
                                                        o_dtype,
                                                        p_producer,
                                                        reduction_heads,
                                                        row_idx,
                                                        sL,
                                                        sM,
                                                        s_consumer,
                                                        scale_qs,
                                                        scale_qs_log2_e,
                                                        selected_slots,
                                                        softmax_nbar,
                                                        tAsK,
                                                        tCsL,
                                                        tCsM,
                                                        tCsO,
                                                        tCsP,
                                                        tCtO,
                                                        tCtS,
                                                        thrblk_mma_kq,
                                                        thrblk_mma_vp,
                                                        tiles_dm,
                                                        warpgroup_tidx,
                                                        warpgroup_widx,
                                                    )
                                            else:
                                                self._softmax_work(
                                                    self.softmax_head_tiles[0],
                                                    warpgroup_idx
                                                    * self.softmax_head_tiles[0],
                                                    warpgroup_idx,
                                                    acc_dtype,
                                                    blk_tile_d,
                                                    blk_tile_h,
                                                    blk_tile_s,
                                                    head_offset,
                                                    heads_per_kv,
                                                    iters_s,
                                                    kv_split_idx,
                                                    kv_splits,
                                                    lane_idx,
                                                    mma_dice,
                                                    mma_dtype,
                                                    mma_tile_m,
                                                    mma_tile_n,
                                                    o_consumer,
                                                    o_dtype,
                                                    p_producer,
                                                    reduction_heads,
                                                    row_idx,
                                                    sL,
                                                    sM,
                                                    s_consumer,
                                                    scale_qs,
                                                    scale_qs_log2_e,
                                                    selected_slots,
                                                    softmax_nbar,
                                                    tAsK,
                                                    tCsL,
                                                    tCsM,
                                                    tCsO,
                                                    tCsP,
                                                    tCtO,
                                                    tCtS,
                                                    thrblk_mma_kq,
                                                    thrblk_mma_vp,
                                                    tiles_dm,
                                                    warpgroup_tidx,
                                                    warpgroup_widx,
                                                )
                                        else:
                                            if cutlass.const_expr(
                                                self.original_heads_per_kv == 12
                                            ):
                                                if cutlass.const_expr(
                                                    self.register_budget is not None
                                                ):
                                                    cute.arch.setmaxregister_increase(
                                                        self.register_budget[0]
                                                    )
                                            softmax_head_group = warpgroup_idx
                                            head_in_tile = (
                                                softmax_head_group
                                                * self.softmax_head_tile
                                                + lane_idx
                                            )
                                            lane_store_max = (
                                                lane_idx < self.softmax_head_tile
                                            )
                                            softmax_dice = (None, None, None)
                                            tmem_op_width = 32
                                            tmem_op_repeat = tcgen05.Repetition(
                                                self.softmax_head_tile
                                                * acc_dtype.width
                                                // tmem_op_width
                                            )
                                            tmem_load_atom_s = cute.make_copy_atom(
                                                tcgen05.Ld32x32bOp(tmem_op_repeat),
                                                acc_dtype,
                                            )
                                            tmem_load_s = tcgen05.make_tmem_copy(
                                                tmem_load_atom_s, tCtS[mma_dice + (0,)]
                                            )
                                            thr_load_s = tmem_load_s.get_slice(
                                                warpgroup_tidx
                                            )
                                            tmem_store_atom_o = cute.make_copy_atom(
                                                tcgen05.St32x32bOp(tmem_op_repeat),
                                                o_dtype,
                                            )
                                            tmem_store_o = tcgen05.make_tmem_copy(
                                                tmem_store_atom_o,
                                                tCtO[mma_dice + (0, 0)],
                                            )
                                            thr_store_o = tmem_store_o.get_slice(
                                                warpgroup_tidx
                                            )
                                            tStS = thr_load_s.partition_S(
                                                tCtS
                                            )  # (CPY, #CPY_MMA, #CPY_M, #CPY_N, stages_sp)
                                            tStS = tStS[
                                                None,
                                                softmax_head_group,
                                                None,
                                                None,
                                                None,
                                            ]
                                            cS = cute.make_identity_tensor(
                                                (mma_tile_m, mma_tile_n)
                                            )
                                            tOcS = thrblk_mma_kq.partition_C(cS)
                                            tScS = thr_load_s.partition_D(tOcS)
                                            tScS = tScS[
                                                None, softmax_head_group, None, None
                                            ]
                                            tSsP = thr_load_s.partition_D(
                                                tCsP
                                            )  # (CPY, #CPY_MMA, #CPY_M, #CPY_N, stages_sp)
                                            tSsP = tSsP[
                                                None,
                                                softmax_head_group,
                                                None,
                                                None,
                                                None,
                                            ]
                                            tStO = thr_load_s.partition_S(
                                                tCtO
                                            )  # (CPY, #CPY_MMA, #CPY_M, #CPY_N, #TILE_DM, #TILE_HN)
                                            tStO = tStO[
                                                None,
                                                softmax_head_group,
                                                None,
                                                None,
                                                None,
                                                None,
                                            ]
                                            tSsO = thr_load_s.partition_D(
                                                tCsO
                                            )  # (CPY, #CPY_MMA, #CPY_M, #CPY_N, #TILE_DM, #TILE_HN)
                                            tSsO = tSsO[
                                                None,
                                                softmax_head_group,
                                                None,
                                                None,
                                                None,
                                                None,
                                            ]
                                            tSrO = cute.make_rmem_tensor_like(tSsO)
                                            cO = cute.make_identity_tensor(
                                                (blk_tile_d, blk_tile_h)
                                            )
                                            tOcO = thrblk_mma_vp.partition_C(cO)
                                            tScO = thr_load_s.partition_D(tOcO)
                                            tScO = tScO[
                                                None, softmax_head_group, None, None
                                            ]
                                            tSsM = thr_load_s.partition_D(
                                                tCsM
                                            )  # (CPY, #CPY_MMA, #CPY_M, #CPY_N)
                                            tSsM = tSsM[
                                                None, softmax_head_group, None, None
                                            ]
                                            tSrM_prev = cute.make_rmem_tensor_like(tSsM)
                                            tSrM_prev.fill(-cutlass.Float32.inf)
                                            tSsL = thr_load_s.partition_D(
                                                tCsL
                                            )  # (CPY, #CPY_MMA, #CPY_M, #CPY_N, WARPS)
                                            tSsL = tSsL[
                                                None,
                                                softmax_head_group,
                                                None,
                                                None,
                                                None,
                                            ]
                                            tSrL = cute.make_rmem_tensor_like(
                                                tSsL[softmax_dice + (0,)]
                                            )
                                            tSrL.fill(cutlass.Float32(0))
                                            assert warp_threads >= cute.size(tSsM)
                                            tSrO.fill(cutlass.Float32(0))
                                            cute.copy(thr_store_o, tSrO, tStO)
                                            if warpgroup_widx == 0 and lane_store_max:
                                                tSsM[lane_idx] = -cutlass.Float32.inf
                                            if warpgroup_widx == 1 and lane_store_max:
                                                tSsL[lane_idx] = cutlass.Float32(0)
                                            softmax_nbar.arrive_and_wait()
                                            for s in cutlass.range(iters_s):
                                                selected_tile = (
                                                    kv_split_idx + s * kv_splits
                                                )
                                                active_head_group = cutlass.Boolean(
                                                    True
                                                )
                                                # Load S from tmem
                                                s_handle = s_consumer.wait_and_advance()
                                                tSrS = cute.make_rmem_tensor(
                                                    tSsP.shape[:-1], acc_dtype
                                                )
                                                if active_head_group:
                                                    cute.copy(
                                                        thr_load_s,
                                                        tStS[
                                                            softmax_dice
                                                            + (s_handle.index,)
                                                        ],
                                                        tSrS,
                                                    )
                                                    cute.arch.fence_view_async_tmem_load()
                                                s_handle.release()

                                                tSrValid = cute.make_rmem_tensor(
                                                    tSrS.shape, cutlass.Int32
                                                )
                                                tSrM = cute.make_rmem_tensor_like(tSsM)
                                                tSrP = cute.make_rmem_tensor(
                                                    tSsP.shape[:-1], mma_dtype
                                                )
                                                tSrP.fill(cutlass.BFloat16(0))
                                                tSrP_f32 = cute.make_rmem_tensor(
                                                    tSrS.shape, acc_dtype
                                                )
                                                correction = cute.make_rmem_tensor_like(
                                                    tSrM
                                                )
                                                if active_head_group:
                                                    # Invalid entries load slot 1, which the caller keeps finite.
                                                    # Mask their logits using the original selection so
                                                    # they never contribute to the softmax.
                                                    for i in cutlass.range_constexpr(
                                                        cute.size(tSrS)
                                                    ):
                                                        score_coord = tScS[i]
                                                        selected_idx = (
                                                            selected_tile * blk_tile_s
                                                            + score_coord[0]
                                                        )
                                                        valid = cutlass.Boolean(False)
                                                        if (
                                                            selected_idx < 2048
                                                            and head_offset
                                                            + score_coord[1]
                                                            < heads_per_kv
                                                            and (
                                                                head_offset
                                                                + score_coord[1]
                                                            )
                                                            % (
                                                                heads_per_kv
                                                                // self.query_group
                                                            )
                                                            < self.original_heads_per_kv
                                                        ):
                                                            source_row = row_idx
                                                            source_column = selected_idx
                                                            if cutlass.const_expr(
                                                                self.query_group > 1
                                                            ):
                                                                source_row = (
                                                                    row_idx
                                                                    * self.query_group
                                                                )
                                                            valid = (
                                                                selected_slots[
                                                                    source_row,
                                                                    source_column,
                                                                ]
                                                                > 0
                                                            )
                                                        tSrValid[i] = (
                                                            cutlass.Int32(1)
                                                            if valid
                                                            else cutlass.Int32(0)
                                                        )
                                                        if not valid:
                                                            tSrS[i] = cutlass.Float32(
                                                                -1.0e30
                                                            )

                                                    # Reduce colmax in warp RF
                                                    tSrM_lane = cutlass.Float32(
                                                        0
                                                    )  # Avoid dynamic register indexing
                                                    for i in cutlass.range_constexpr(
                                                        cute.size(tSrS)
                                                    ):
                                                        tSrM[i] = warp_fmax(tSrS[i])
                                                        if i == lane_idx:
                                                            tSrM_lane = tSrM[i]

                                                    # Reduce colmax in smem
                                                    if lane_store_max:
                                                        smem_fmax(
                                                            tSsM.iterator
                                                            + tSsM.layout(lane_idx),
                                                            tSrM_lane,
                                                        )

                                                    # Wait for colmax then load
                                                    softmax_nbar.arrive_and_wait()
                                                    cute.autovec_copy(tSsM, tSrM)

                                                    # Compute online softmax
                                                    for i in cutlass.range_constexpr(
                                                        0, cute.size(tSrS), 2
                                                    ):
                                                        p_f32x2 = fadd2(
                                                            (tSrS[i], tSrS[i + 1]),
                                                            (-tSrM[i], -tSrM[i + 1]),
                                                        )
                                                        p_f32x2 = fmul2(
                                                            p_f32x2,
                                                            (
                                                                scale_qs_log2_e,
                                                                scale_qs_log2_e,
                                                            ),
                                                        )
                                                        tSrP_f32[i] = (
                                                            exp2(p_f32x2[0])
                                                            if tSrValid[i] != 0
                                                            else cutlass.Float32(0.0)
                                                        )
                                                        tSrP_f32[i + 1] = (
                                                            exp2(p_f32x2[1])
                                                            if tSrValid[i + 1] != 0
                                                            else cutlass.Float32(0.0)
                                                        )
                                                    tSrP.store(
                                                        tSrP_f32.load().to(mma_dtype)
                                                    )

                                                # Store P to smem
                                                p_handle = (
                                                    p_producer.acquire_and_advance()
                                                )
                                                cute.autovec_copy(
                                                    tSrP,
                                                    tSsP[
                                                        softmax_dice + (p_handle.index,)
                                                    ],
                                                )
                                                cute.arch.fence_view_async_shared()
                                                p_handle.commit()

                                                if active_head_group:
                                                    # Compute correction and correct colsum
                                                    for i in cutlass.range_constexpr(
                                                        0, cute.size(tSrM), 2
                                                    ):
                                                        c_f32x2 = fadd2(
                                                            (
                                                                tSrM_prev[i],
                                                                tSrM_prev[i + 1],
                                                            ),
                                                            (-tSrM[i], -tSrM[i + 1]),
                                                        )
                                                        c_f32x2 = fmul2(
                                                            c_f32x2,
                                                            (
                                                                scale_qs_log2_e,
                                                                scale_qs_log2_e,
                                                            ),
                                                        )
                                                        c_f32x2 = (
                                                            exp2(c_f32x2[0]),
                                                            exp2(c_f32x2[1]),
                                                        )
                                                        correction[i] = c_f32x2[0]
                                                        correction[i + 1] = c_f32x2[1]
                                                        l_f32x2 = ffma2(
                                                            c_f32x2,
                                                            (tSrL[i], tSrL[i + 1]),
                                                            (
                                                                tSrP_f32[i],
                                                                tSrP_f32[i + 1],
                                                            ),
                                                        )
                                                        tSrL[i] = l_f32x2[0]
                                                        tSrL[i + 1] = l_f32x2[1]

                                                    tSrM_prev.store(tSrM.load())
                                                # Correct O
                                                if s > 0:
                                                    # Wait for O
                                                    o_handle = (
                                                        o_consumer.wait_and_advance()
                                                    )

                                                    if active_head_group:
                                                        # Apply correction
                                                        for (
                                                            dm
                                                        ) in cutlass.range_constexpr(
                                                            tiles_dm
                                                        ):
                                                            tSrO_dm = cute.make_rmem_tensor_like(
                                                                tSsO[
                                                                    softmax_dice
                                                                    + (0, 0)
                                                                ]
                                                            )
                                                            cute.copy(
                                                                thr_load_s,
                                                                tStO[
                                                                    softmax_dice
                                                                    + (dm, 0)
                                                                ],
                                                                tSrO_dm,
                                                            )

                                                            for (
                                                                i
                                                            ) in cutlass.range_constexpr(
                                                                0, cute.size(tSrO_dm), 2
                                                            ):
                                                                o_f32x2 = fmul2(
                                                                    (
                                                                        tSrO_dm[i],
                                                                        tSrO_dm[i + 1],
                                                                    ),
                                                                    (
                                                                        correction[i],
                                                                        correction[
                                                                            i + 1
                                                                        ],
                                                                    ),
                                                                )
                                                                tSrO_dm[i] = o_f32x2[0]
                                                                tSrO_dm[i + 1] = (
                                                                    o_f32x2[1]
                                                                )

                                                            cute.copy(
                                                                thr_store_o,
                                                                tSrO_dm,
                                                                tStO[
                                                                    softmax_dice
                                                                    + (dm, 0)
                                                                ],
                                                            )

                                                    # Notify MMA
                                                    cute.arch.fence_view_async_tmem_store()
                                                    o_handle.release()
                                            tSrL_lane = cutlass.Float32(0.0)
                                            for i in cutlass.range_constexpr(
                                                cute.size(tSrL)
                                            ):
                                                tSrL[i] = cute.arch.warp_reduction_sum(
                                                    tSrL[i]
                                                )
                                                if i == lane_idx:
                                                    tSrL_lane = tSrL[i]
                                            if lane_store_max:
                                                tSsL[softmax_dice + (warpgroup_widx,)][
                                                    lane_idx
                                                ] = tSrL_lane
                                            softmax_nbar.arrive_and_wait()
                                            if warpgroup_widx == 0 and lane_store_max:
                                                # Load colsum and colmax
                                                sL_lane_wg = sL[0, head_in_tile, None]
                                                sL_lane = (
                                                    sL_lane_wg[0]
                                                    + sL_lane_wg[1]
                                                    + sL_lane_wg[2]
                                                    + sL_lane_wg[3]
                                                )
                                                sM_lane = sM[0, head_in_tile]

                                                # Scale colmax
                                                sM_lane = sM_lane * scale_qs

                                                # Keep final statistics local for the DSM merge.
                                                sL[0, head_in_tile, 0] = sL_lane
                                                sM[0, head_in_tile] = sM_lane
                                            o_handle = o_consumer.wait_and_advance()
                                            cute.copy(thr_load_s, tStO, tSrO)
                                            cute.arch.fence_view_async_tmem_load()
                                            o_handle.release()  # Final release signals tmem dealloc
                                            local_partial = cute.make_tensor(
                                                cute.recast_ptr(
                                                    tAsK.iterator, dtype=acc_dtype
                                                ),
                                                cute.make_layout(
                                                    (reduction_heads, blk_tile_d + 2),
                                                    stride=(blk_tile_d + 2, 1),
                                                ),
                                            )
                                            for i in cutlass.range_constexpr(
                                                cute.size(tSrO)
                                            ):
                                                coord = tScO[i]
                                                dim = coord[0]
                                                head = coord[1]
                                                if head < reduction_heads:
                                                    local_partial[head, dim] = tSrO[i]
                                            if (
                                                warpgroup_widx == 0
                                                and head_in_tile < reduction_heads
                                            ):
                                                local_partial[
                                                    head_in_tile, blk_tile_d
                                                ] = sM[0, head_in_tile]
                                                local_partial[
                                                    head_in_tile, blk_tile_d + 1
                                                ] = sL[0, head_in_tile, 0]
                                    else:
                                        if cutlass.const_expr(
                                            self.original_heads_per_kv == 12
                                            and self.problem_shape[2] == 2
                                        ):
                                            if cutlass.const_expr(
                                                self.prefetch_private_kv
                                            ):
                                                # The unused MMA warp copies query-private K/V while all
                                                # prefix roles run. The existing reader join below retires it.
                                                _prefetch_private_kv_payload(
                                                    k_iter,
                                                    v_iter,
                                                    selected_slots,
                                                    row_idx,
                                                    staged_key,
                                                    staged_value,
                                                    self.query_group,
                                                    num_kv_heads,
                                                    self.prefetch_private_kv_warp,
                                                )

                                            elif cutlass.const_expr(
                                                self.overlap_private_tail
                                            ):
                                                # One otherwise idle warp computes each head's private
                                                # tail contribution while the prefix roles run concurrently.
                                                self._prepare_private_tail(
                                                    q_source,
                                                    k_iter,
                                                    v_iter,
                                                    selected_slots,
                                                    private_tail,
                                                    row_idx,
                                                    q_head_offset,
                                                    warp_idx,
                                                    lane_idx,
                                                    scale_qs,
                                                )
                                        else:
                                            if cutlass.const_expr(
                                                self.original_heads_per_kv == 12
                                            ):
                                                if cutlass.const_expr(
                                                    self.overlap_private_tail
                                                ):
                                                    # One otherwise idle warp computes each head's private
                                                    # tail contribution while the prefix roles run concurrently.
                                                    self._prepare_private_tail(
                                                        q_source,
                                                        k_iter,
                                                        v_iter,
                                                        selected_slots,
                                                        private_tail,
                                                        row_idx,
                                                        q_head_offset,
                                                        warp_idx,
                                                        lane_idx,
                                                        scale_qs,
                                                    )
                if cutlass.const_expr(
                    self.original_heads_per_kv == 12 and self.problem_shape[2] == 2
                ):
                    if cutlass.const_expr(self.register_budget is not None):
                        if warpgroup_idx in self.softmax_warpgroup_ids:
                            cute.arch.setmaxregister_decrease(self.register_budget[1])
                else:
                    if cutlass.const_expr(self.original_heads_per_kv == 12):
                        if cutlass.const_expr(self.register_budget is not None):
                            if warpgroup_idx in self.softmax_warpgroup_ids:
                                cute.arch.setmaxregister_decrease(
                                    self.register_budget[1]
                                )
                    else:
                        if cutlass.const_expr(self.overlap_private_tail):
                            # The Q warp arrives here after issuing both Q loads. The
                            # remaining tail warps had no prefix role; each owns a head.
                            if warp_idx >= self.private_tail_start_warp:
                                self._prepare_private_tail(
                                    q_source,
                                    k_iter,
                                    v_iter,
                                    selected_slots,
                                    private_tail,
                                    row_idx,
                                    q_head_offset,
                                    warp_idx,
                                    lane_idx,
                                    scale_qs,
                                )
                self._sync_work_threads()
                if cutlass.const_expr(
                    self.original_heads_per_kv == 12 and self.problem_shape[2] == 2
                ):
                    if cutlass.const_expr(self.register_budget is not None):
                        cute.arch.setmaxregister_increase(self.register_budget[2])
                else:
                    if cutlass.const_expr(self.original_heads_per_kv == 12):
                        if cutlass.const_expr(self.register_budget is not None):
                            cute.arch.setmaxregister_increase(self.register_budget[2])
                self._sync_cluster_work()
                cluster_splits = self.kv_splits
                cta_rank = cute.arch.block_idx_in_cluster()
                local_partial = cute.make_tensor(
                    cute.recast_ptr(tAsK.iterator, dtype=acc_dtype),
                    cute.make_layout(
                        (reduction_heads, blk_tile_d + 2), stride=(blk_tile_d + 2, 1)
                    ),
                )
                tail_count = (selected_slots.shape[1] % blk_tile_s) * self.query_group
                tail_start = 2048
                if cutlass.const_expr(self.direct_query):
                    # Valid output heads have the same original row/head
                    # addresses as Q, including each query's private tail.
                    mQuery = cute.make_tensor(q_source, mOut.layout)
                else:
                    mQuery = cute.make_tensor(
                        q_source, cute.make_layout((head_dim, num_q_heads, num_rows))
                    )
                kv_layout = cute.make_layout(
                    (head_dim, cache_slots), stride=(1, num_kv_heads * head_dim)
                )
                mKey = cute.make_tensor(k_iter, kv_layout)
                mValue = cute.make_tensor(v_iter, kv_layout)
                merge_groups = self.threads_per_cta // blk_tile_d
                merge_group = tidx // blk_tile_d
                merge_tidx = tidx % blk_tile_d
                merge_warp = merge_tidx // warp_threads
                merge_scratch_ptr = cute.recast_ptr(
                    tAsK.iterator, dtype=acc_dtype
                ) + self.head_tile * (blk_tile_d + 2)
                merge_sL = cute.make_tensor(
                    merge_scratch_ptr,
                    cute.make_layout(
                        (8, 4 * self.query_group, merge_groups),
                        stride=(1, 8, 32 * self.query_group),
                    ),
                )
                merge_sM = cute.make_tensor(
                    merge_scratch_ptr + 32 * self.query_group * merge_groups,
                    cute.make_layout((2, merge_groups), stride=(1, 2)),
                )
                if cutlass.const_expr(self.kv_splits == 1):
                    for head_round in cutlass.range_constexpr(
                        cute.ceil_div(
                            reduction_heads, self.threads_per_cta // warp_threads
                        )
                    ):
                        head = warp_idx + head_round * (
                            self.threads_per_cta // warp_threads
                        )
                        if (
                            head < reduction_heads
                            and head_offset + head < heads_per_kv
                            and (
                                (head_offset + head)
                                % (heads_per_kv // self.query_group)
                                < self.original_heads_per_kv
                            )
                        ):
                            q_head = q_head_offset + head
                            numerator = cute.make_rmem_tensor(
                                blk_tile_d // warp_threads, cutlass.Float32
                            )
                            query_values = cute.make_rmem_tensor_like(numerator)
                            for part in cutlass.range_constexpr(
                                blk_tile_d // warp_threads
                            ):
                                dim = lane_idx + part * warp_threads
                                numerator[part] = local_partial[head, dim]
                                if cutlass.const_expr(not self.overlap_private_tail):
                                    if cutlass.const_expr(
                                        self.original_heads_per_kv == 12
                                        and self.problem_shape[2] == 2
                                    ):
                                        if cutlass.const_expr(
                                            q_dtype is cutlass.BFloat16
                                            and k_dtype is cutlass.BFloat16
                                        ):
                                            query_values[part] = cutlass.Float32(
                                                self._load_staged_query_bf16(
                                                    tBsQ.iterator, head, dim, mma_tile_n
                                                )
                                            )
                                        else:
                                            query_values[part] = cutlass.Float32(
                                                mQuery[dim, q_head, row_idx]
                                            )
                                    else:
                                        query_values[part] = cutlass.Float32(
                                            mQuery[dim, q_head, row_idx]
                                        )
                            maximum = local_partial[head, blk_tile_d]
                            denominator = local_partial[head, blk_tile_d + 1]
                            if cutlass.const_expr(self.overlap_private_tail):
                                tail_denominator = private_tail[head, 257]
                                if tail_denominator > 0.0:
                                    tail_maximum = private_tail[head, 256]
                                    if denominator > 0.0:
                                        new_maximum = cute.arch.fmax(
                                            maximum, tail_maximum
                                        )
                                        prefix_weight = exp2(
                                            log2_e * (maximum - new_maximum)
                                        )
                                        tail_weight = exp2(
                                            log2_e * (tail_maximum - new_maximum)
                                        )
                                        denominator = (
                                            denominator * prefix_weight
                                            + tail_denominator * tail_weight
                                        )
                                        for part in cutlass.range_constexpr(8):
                                            dim = lane_idx + part * warp_threads
                                            numerator[part] = (
                                                numerator[part] * prefix_weight
                                                + private_tail[head, dim] * tail_weight
                                            )
                                        maximum = new_maximum
                                    else:
                                        denominator = tail_denominator
                                        maximum = tail_maximum
                                        for part in cutlass.range_constexpr(8):
                                            numerator[part] = private_tail[
                                                head, lane_idx + part * warp_threads
                                            ]
                            else:
                                if cutlass.const_expr(
                                    self.original_heads_per_kv == 12
                                    and self.problem_shape[2] == 2
                                ):
                                    source_row = row_idx
                                    if cutlass.const_expr(self.query_group > 1):
                                        source_row = row_idx * self.query_group + (
                                            head_offset + head
                                        ) // (heads_per_kv // self.query_group)
                                    for tail in cutlass.range_constexpr(
                                        selected_slots.shape[1] % blk_tile_s
                                    ):
                                        slot = cutlass.Int64(
                                            selected_slots[
                                                source_row, tail_start + tail
                                            ]
                                        )
                                        if slot > 0:
                                            dot = cutlass.Float32(0.0)
                                            for part in cutlass.range_constexpr(
                                                blk_tile_d // warp_threads
                                            ):
                                                dim = lane_idx + part * warp_threads
                                                if cutlass.const_expr(
                                                    self.prefetch_private_kv
                                                ):
                                                    source_query = (
                                                        head_offset + head
                                                    ) // (
                                                        heads_per_kv // self.query_group
                                                    )
                                                    key_value = cutlass.Float32(
                                                        staged_key[
                                                            dim, tail, source_query
                                                        ]
                                                    )
                                                else:
                                                    key_value = cutlass.Float32(
                                                        mKey[dim, slot]
                                                    )
                                                dot += query_values[part] * key_value
                                            score = (
                                                cute.arch.warp_reduction_sum(dot)
                                                * scale_qs
                                            )
                                            new_maximum = cute.arch.fmax(maximum, score)
                                            correction = exp2(
                                                log2_e * (maximum - new_maximum)
                                            )
                                            weight = exp2(
                                                log2_e * (score - new_maximum)
                                            )
                                            denominator = (
                                                denominator * correction + weight
                                            )
                                            for part in cutlass.range_constexpr(
                                                blk_tile_d // warp_threads
                                            ):
                                                dim = lane_idx + part * warp_threads
                                                if cutlass.const_expr(
                                                    self.prefetch_private_kv
                                                ):
                                                    source_query = (
                                                        head_offset + head
                                                    ) // (
                                                        heads_per_kv // self.query_group
                                                    )
                                                    value = cutlass.Float32(
                                                        staged_value[
                                                            dim, tail, source_query
                                                        ]
                                                    )
                                                else:
                                                    value = cutlass.Float32(
                                                        mValue[dim, slot]
                                                    )
                                                numerator[part] = (
                                                    numerator[part] * correction
                                                    + weight * value
                                                )
                                            maximum = new_maximum
                                else:
                                    for tail in cutlass.range_constexpr(tail_count):
                                        slot = cutlass.Int64(
                                            selected_slots[
                                                row_idx * self.query_group + tail // 3,
                                                tail_start + tail % 3,
                                            ]
                                        )
                                        if slot > 0 and tail // 3 == (
                                            head_offset + head
                                        ) // (heads_per_kv // self.query_group):
                                            dot = cutlass.Float32(0.0)
                                            for part in cutlass.range_constexpr(
                                                blk_tile_d // warp_threads
                                            ):
                                                dim = lane_idx + part * warp_threads
                                                dot += query_values[
                                                    part
                                                ] * cutlass.Float32(mKey[dim, slot])
                                            score = (
                                                cute.arch.warp_reduction_sum(dot)
                                                * scale_qs
                                            )
                                            new_maximum = cute.arch.fmax(maximum, score)
                                            correction = exp2(
                                                log2_e * (maximum - new_maximum)
                                            )
                                            weight = exp2(
                                                log2_e * (score - new_maximum)
                                            )
                                            denominator = (
                                                denominator * correction + weight
                                            )
                                            for part in cutlass.range_constexpr(
                                                blk_tile_d // warp_threads
                                            ):
                                                dim = lane_idx + part * warp_threads
                                                numerator[part] = numerator[
                                                    part
                                                ] * correction + weight * cutlass.Float32(
                                                    mValue[dim, slot]
                                                )
                                            maximum = new_maximum
                            for part in cutlass.range_constexpr(
                                blk_tile_d // warp_threads
                            ):
                                dim = lane_idx + part * warp_threads
                                mOut[dim, q_head, row_idx] = mOut.element_type(
                                    scale_o * numerator[part] / denominator
                                    if denominator > 0.0
                                    else 0.0
                                )
                        if cutlass.const_expr(
                            not (
                                self.original_heads_per_kv == 12
                                and self.problem_shape[2] == 2
                            )
                        ):
                            self._sync_work_threads()
                    if cutlass.const_expr(
                        self.original_heads_per_kv == 12 and self.problem_shape[2] == 2
                    ):
                        self._sync_work_threads()
                else:
                    for head_group in cutlass.range_constexpr(
                        cute.ceil_div(reduction_heads, cluster_splits * merge_groups)
                    ):
                        head = (
                            cta_rank
                            + (head_group * merge_groups + merge_group) * cluster_splits
                        )
                        valid_head = (
                            merge_group < merge_groups
                            and head < reduction_heads
                            and head_offset + head < heads_per_kv
                            and (head_offset + head)
                            % (heads_per_kv // self.query_group)
                            < self.original_heads_per_kv
                        )
                        q_head = q_head_offset + head
                        numerator = cutlass.Float32(0.0)
                        tail_values = cute.make_rmem_tensor(tail_count, cutlass.Float32)
                        tail_values.fill(0.0)
                        tail_dots = cute.make_rmem_tensor(tail_count, cutlass.Float32)
                        tail_dots.fill(0.0)
                        if valid_head:
                            if merge_warp == 0:
                                split_max = -cutlass.Float32.inf
                                split_sum = cutlass.Float32(0.0)
                                if lane_idx < cluster_splits:
                                    max_ptr = cute.domain_offset(
                                        (head, blk_tile_d), local_partial
                                    ).iterator
                                    sum_ptr = cute.domain_offset(
                                        (head, blk_tile_d + 1), local_partial
                                    ).iterator
                                    split_max = ld_shared_remote_f32(
                                        set_block_rank(max_ptr, lane_idx)
                                    )
                                    split_sum = ld_shared_remote_f32(
                                        set_block_rank(sum_ptr, lane_idx)
                                    )
                                global_max = warp_fmax(split_max)
                                correction = exp2(log2_e * (split_max - global_max))
                                denominator = cute.arch.warp_reduction_sum(
                                    correction * split_sum
                                )
                                if lane_idx < cluster_splits:
                                    merge_sL[
                                        lane_idx % 8, lane_idx // 8, merge_group
                                    ] = correction
                                if lane_idx == 0:
                                    merge_sM[0, merge_group] = denominator
                                    merge_sM[1, merge_group] = global_max
                            if merge_group < merge_groups:
                                query_value = cutlass.Float32(
                                    mQuery[merge_tidx, q_head, row_idx]
                                )
                                for tail in cutlass.range_constexpr(tail_count):
                                    slot = cutlass.Int64(
                                        selected_slots[
                                            row_idx * self.query_group + tail // 3,
                                            tail_start + tail % 3,
                                        ]
                                    )
                                    if slot > 0 and tail // 3 == (
                                        head_offset + head
                                    ) // (heads_per_kv // self.query_group):
                                        tail_dots[tail] = query_value * cutlass.Float32(
                                            mKey[merge_tidx, slot]
                                        )
                                        tail_values[tail] = cutlass.Float32(
                                            mValue[merge_tidx, slot]
                                        )
                        self._sync_work_threads()
                        if valid_head and merge_group < merge_groups:
                            for split in cutlass.range_constexpr(cluster_splits):
                                value_ptr = cute.domain_offset(
                                    (head, merge_tidx), local_partial
                                ).iterator
                                split_value = ld_shared_remote_f32(
                                    set_block_rank(value_ptr, split)
                                )
                                numerator += (
                                    merge_sL[split % 8, split // 8, merge_group]
                                    * split_value
                                )
                        self._sync_work_threads()
                        if valid_head and merge_group < merge_groups:
                            for tail in cutlass.range_constexpr(tail_count):
                                dot = cute.arch.warp_reduction_sum(tail_dots[tail])
                                if lane_idx == 0:
                                    merge_sL[merge_warp, tail, merge_group] = dot
                        self._sync_work_threads()
                        if valid_head and merge_warp == 0:
                            tail_score = -cutlass.Float32.inf
                            if lane_idx < tail_count:
                                if (
                                    lane_idx // 3
                                    == (head_offset + head)
                                    // (heads_per_kv // self.query_group)
                                    and selected_slots[
                                        row_idx * self.query_group + lane_idx // 3,
                                        tail_start + lane_idx % 3,
                                    ]
                                    > 0
                                ):
                                    tail_score = cutlass.Float32(0.0)
                                    for warp in cutlass.range_constexpr(8):
                                        tail_score += merge_sL[
                                            warp, lane_idx, merge_group
                                        ]
                                    tail_score *= scale_qs
                            new_max = cute.arch.fmax(
                                merge_sM[1, merge_group], warp_fmax(tail_score)
                            )
                            tail_weight = exp2(log2_e * (tail_score - new_max))
                            main_correction = exp2(
                                log2_e * (merge_sM[1, merge_group] - new_max)
                            )
                            tail_sum = cute.arch.warp_reduction_sum(tail_weight)
                            # The weight stores alias the dot-product scratch, and the
                            # new statistics alias the main maximum read by every lane.
                            # Shuffle reductions do not order shared-memory accesses.
                            cute.arch.sync_warp()
                            if lane_idx < tail_count:
                                merge_sL[lane_idx, 0, merge_group] = tail_weight
                            if lane_idx == 0:
                                merge_sM[0, merge_group] = (
                                    merge_sM[0, merge_group] * main_correction
                                    + tail_sum
                                )
                                merge_sM[1, merge_group] = main_correction
                        self._sync_work_threads()
                        if valid_head and merge_group < merge_groups:
                            numerator *= merge_sM[1, merge_group]
                            for tail in cutlass.range_constexpr(tail_count):
                                numerator += (
                                    merge_sL[tail, 0, merge_group] * tail_values[tail]
                                )
                            mOut[merge_tidx, q_head, row_idx] = mOut.element_type(
                                scale_o * numerator / merge_sM[0, merge_group]
                                if merge_sM[0, merge_group] > 0.0
                                else 0.0
                            )
                        self._sync_work_threads()
                self._sync_cluster_work()
                if cutlass.const_expr(self.query_rows_per_cta > 1 or self.reuse_work):
                    if warp_idx == 0:
                        with cute.arch.elect_one():
                            for barrier in cutlass.range_constexpr(self.q_stages * 2):
                                invalidate_mbarrier(
                                    q_pipeline_ptr + barrier, loc=None, ip=None
                                )
                            for barrier in cutlass.range_constexpr(self.cvt_stages * 2):
                                invalidate_mbarrier(
                                    cvt_pipeline_ptr + barrier, loc=None, ip=None
                                )
                            for barrier in cutlass.range_constexpr(self.sp_stages * 2):
                                invalidate_mbarrier(
                                    s_pipeline_ptr + barrier, loc=None, ip=None
                                )
                                invalidate_mbarrier(
                                    p_pipeline_ptr + barrier, loc=None, ip=None
                                )
                            for barrier in cutlass.range_constexpr(self.o_stages * 2):
                                invalidate_mbarrier(
                                    o_pipeline_ptr + barrier, loc=None, ip=None
                                )
                    self._sync_work_threads()
        if cutlass.const_expr(
            self.query_rows_per_cta > 1 and not self.caller_allocates_scratch
        ):
            if warp_idx == self.mma_vp_warp_id:
                tmem_ptr = cute.arch.retrieve_tmem_ptr(
                    cutlass.Int32, 16, tmem_ptr_smem_ptr
                )
                cute.arch.relinquish_tmem_alloc_permit()
                cute.arch.dealloc_tmem(tmem_ptr, self.tmem_alloc_cols)
        return

    @cute.jit
    def _prefetch_sparse_stream_item_bf16_loaded(
        self,
        k_iter: cute.Pointer,
        v_iter: cute.Pointer,
        raw_slot: cutlass.Int32,
        stream_item: cutlass.Constexpr[int],
        stream_length: cutlass.Constexpr[int],
        dim_offset: cutlass.Int32,
    ):
        if cutlass.const_expr(
            stream_item == 0
            or (stream_item % 2 == 1 and stream_item != stream_length - 1)
        ):
            cache_iter = k_iter
        else:
            cache_iter = v_iter
        slot = raw_slot if raw_slot > 0 else cutlass.Int32(_PADDING_READ_SLOT)
        source = (
            cache_iter
            + cutlass.Int64(slot) * self.problem_shape[2] * _HEAD_DIM
            + dim_offset
        )
        prefetch_gmem_l2(source, loc=None, ip=None)
        prefetch_gmem_l2(source + 64, loc=None, ip=None)

    @cute.jit
    def _classify_prefix_in_cta(self, selected_slots, group_row, scratch, tid):
        """Compare the ordered prefix; each query keeps its private tail."""
        first_row = group_row * self.query_group
        # Most distinct groups fail after one coalesced warp load.
        if tid < 32:
            equal = cutlass.Boolean(True)
            anchor = selected_slots[first_row, tid]
            for other in cutlass.range_constexpr(1, self.query_group):
                equal = equal & (anchor == selected_slots[first_row + other, tid])
            warp_equal = cute.arch.vote_all_sync(equal)
            if tid == 0:
                scratch[0] = cutlass.Int32(warp_equal)
        cute.arch.sync_threads()
        shared_prefix = scratch[0] != 0
        if shared_prefix:
            equal = cutlass.Boolean(True)
            for chunk in cutlass.range_constexpr(
                cute.ceil_div(2048, self.threads_per_cta)
            ):
                column = chunk * self.threads_per_cta + tid
                if column < 2048:
                    anchor = selected_slots[first_row, column]
                    for other in cutlass.range_constexpr(1, self.query_group):
                        equal = equal & (
                            anchor == selected_slots[first_row + other, column]
                        )
            warp_equal = cute.arch.vote_all_sync(equal)
            if tid % 32 == 0:
                scratch[tid // 32 + 1] = cutlass.Int32(warp_equal)
            cute.arch.sync_threads()
            if tid < 32:
                warp_flag = cutlass.Int32(1)
                if tid < self.threads_per_cta // 32:
                    warp_flag = scratch[tid + 1]
                all_equal = cute.arch.vote_all_sync(warp_flag != 0)
                if tid == 0:
                    scratch[0] = cutlass.Int32(all_equal)
            cute.arch.sync_threads()
            shared_prefix = scratch[0] != 0
        return shared_prefix

    @cute.jit
    def _map_query_row_in_cta(self, selected_slots, query_work, scratch, tid):
        """Choose a common permutation; the sample never establishes reuse."""
        kv_heads = self.problem_shape[2]
        query_row = query_work // kv_heads
        if cutlass.const_expr(self.query_group in (3, 4)):
            # Every CTA samples the same first group, so mixed batches still
            # choose one bijection. Exact reuse is checked separately below.
            order_flag = self.threads_per_cta // 32 + 1
            if tid < 32:
                equal = cutlass.Boolean(True)
                anchor = selected_slots[0, tid]
                for other in cutlass.range_constexpr(1, self.query_group):
                    equal = equal & (anchor == selected_slots[other, tid])
                warp_equal = cute.arch.vote_all_sync(equal)
                if tid == 0:
                    scratch[order_flag] = cutlass.Int32(warp_equal)
            cute.arch.sync_threads()
            if scratch[order_flag] != 0:
                leader_work = selected_slots.shape[0] // self.query_group * kv_heads
                query_row = (
                    query_work % leader_work
                ) // kv_heads * self.query_group + query_work // leader_work
        return query_row

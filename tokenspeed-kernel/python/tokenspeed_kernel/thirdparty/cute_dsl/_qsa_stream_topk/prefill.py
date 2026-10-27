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

"""Grouped scoring with an exact register-resident running top-512."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import cute, pipeline, utils
from cutlass._mlir.dialects import llvm as _llvm
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from tokenspeed_kernel.thirdparty.cute_dsl._qsa_stream_topk.common import (
    QueryGroups,
    _copy_u128,
)

_QUERY_GROUP = 8
_HEADS = 4
_HEAD_DIM = 128
_MMA_M = 128
_K_STAGES = 2
_ACC_STAGES = 2
_STREAM_TILE = 4096
_TOPK = 512
_SCORE_STAGES = 1
_SELECTOR_WARPS_PER_ROW = 2
_K_COPY_WARPGROUP = 0
_EPILOGUE_WARPGROUP = 1
_TOPK_FIRST_WARP = 8
_MMA_WARP = _TOPK_FIRST_WARP + _QUERY_GROUP * _SELECTOR_WARPS_PER_ROW


class PrefillTopK(QueryGroups):
    def __init__(self, enable_pdl: bool, group_layout: str):
        self.enable_pdl = enable_pdl
        self.group_layout = group_layout
        self.query_group = _QUERY_GROUP
        self.stream_tile = _STREAM_TILE
        self.selector_warps_per_row = _SELECTOR_WARPS_PER_ROW
        self.selector_threads_per_row = self.selector_warps_per_row * 32
        self.histogram_items_per_thread = 512 // self.selector_threads_per_row
        self.exact_items_per_thread = 256 // self.selector_threads_per_row
        self.mma_n = _QUERY_GROUP * _HEADS
        self.query_copy_iterations = _QUERY_GROUP * _HEADS // 8
        self.mma_warp = _MMA_WARP
        self.epilogue_warpgroup = _EPILOGUE_WARPGROUP
        self.topk_first_warp = _TOPK_FIRST_WARP
        self.threads = (self.mma_warp + 1) * 32

    @cute.jit
    def __call__(
        self,
        query: cute.Tensor,
        keys: cute.Tensor,
        page_table: cute.Tensor,
        request_indices: cute.Tensor,
        complete_blocks: cute.Tensor,
        selected: cute.Tensor,
        q_stride_row: cutlass.Int32,
        q_stride_head: cutlass.Int32,
        queries_per_request: cutlass.Int32,
        page_size: cutlass.Constexpr[int],
        stream: cuda.CUstream,
    ):
        mma_dtype = cutlass.BFloat16
        acc_dtype = cutlass.Float32
        mma_tile_mnk = (_MMA_M, self.mma_n, _HEAD_DIM)
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            mma_dtype,
            mma_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            acc_dtype,
            tcgen05.CtaGroup.ONE,
            mma_tile_mnk[:2],
            tcgen05.OperandSource.SMEM,
        )
        smem_layout_k = sm100_utils.make_smem_layout_a(
            tiled_mma, mma_tile_mnk, mma_dtype, _K_STAGES
        )
        smem_layout_q = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tile_mnk,
            mma_dtype,
            1,
        )
        self._kernel(
            tiled_mma,
            smem_layout_k,
            smem_layout_q,
            query,
            keys,
            page_table,
            request_indices,
            complete_blocks,
            selected,
            q_stride_row,
            q_stride_head,
            queries_per_request,
            page_size,
        ).launch(
            grid=(self._group_count(query.shape[0], queries_per_request), 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.enable_pdl,
        )

    @cute.jit
    def _warp_scan(self, value: cutlass.Int32, lane_idx: cutlass.Int32):
        for offset in cutlass.range_constexpr(1, 32):
            # Only power-of-two offsets participate in the inclusive scan.
            if cutlass.const_expr(offset in (1, 2, 4, 8, 16)):
                other = cute.arch.shuffle_sync_up(
                    value, offset, mask=0xFFFFFFFF, mask_and_clamp=0
                )
                if lane_idx >= offset:
                    value += other
        return value

    @cute.jit
    def _float_bits(self, value: cutlass.Float32):
        return cutlass.Uint32(_llvm.bitcast(cutlass.Uint32.mlir_type, value.ir_value()))

    @cute.jit
    def _candidate_bits_id(
        self,
        candidate: cutlass.Int32,
        running_count: cutlass.Int32,
        score_stage: cutlass.Int32,
        tile_base: cutlass.Int32,
        running_values: cute.Tensor,
        score_buffers: cute.Tensor,
        row_in_group: cutlass.Int32,
    ):
        bits = cutlass.Uint32(0)
        block_id = cutlass.Int32(-1)
        if candidate < running_count:
            packed = cutlass.Int64(0)
            for item in cutlass.range_constexpr(_TOPK // self.selector_threads_per_row):
                if candidate // self.selector_threads_per_row == item:
                    packed = running_values[item]
            bits = cutlass.Uint32((packed >> 32) - 1)
            block_id = cutlass.Int32(packed & cutlass.Int64(0xFFFFFFFF))
        else:
            offset = candidate - running_count
            bits = self._float_bits(score_buffers[score_stage, row_in_group, offset])
            block_id = tile_base + offset
        return bits, block_id

    @cute.jit
    def _ordered(self, bits: cutlass.Uint32):
        # Scores are non-negative after ReLU. Smaller ordered keys represent
        # larger scores, matching an ascending radix prefix scan.
        return (bits ^ cutlass.Uint32(0xFFFFFFFF)) & cutlass.Uint32(0x7FFFFFFF)

    @cute.jit
    def _coarse_ordered(self, bits: cutlass.Uint32):
        value = cutlass.Float32(
            _llvm.bitcast(cutlass.Float32.mlir_type, bits.ir_value())
        )
        half = cutlass.Float16(value)
        half_bits = cutlass.Uint16(
            _llvm.bitcast(cutlass.Uint16.mlir_type, half.ir_value())
        )
        ordered = (half_bits ^ cutlass.Uint16(0xFFFF)) & cutlass.Uint16(0x7FFF)
        return cutlass.Int32((ordered >> cutlass.Uint16(7)) & cutlass.Uint16(0xFF))

    @cute.jit
    def _select_tile_full(
        self,
        row_in_group: cutlass.Int32,
        score_stage: cutlass.Int32,
        tile_base: cutlass.Int32,
        tile_valid: cutlass.Int32,
        running_count: cutlass.Int32,
        running_values: cute.Tensor,
        score_buffers: cute.Tensor,
        histograms: cute.Tensor,
        scalars: cute.Tensor,
        compacted_row: cute.Tensor,
        lane_idx: cutlass.Int32,
        warp_in_row: cutlass.Int32,
        thread_in_row: cutlass.Int32,
        topk_barrier: pipeline.NamedBarrier,
    ):
        candidate_count = running_count + tile_valid
        prefix = cutlass.Uint32(0)
        remaining = candidate_count
        if remaining > _TOPK:
            remaining = cutlass.Int32(_TOPK)

        for radix_round in cutlass.range_constexpr(4):
            shift = cutlass.const_expr(24 - radix_round * 8)
            prefix_bits = cutlass.const_expr(radix_round * 8)
            if cutlass.const_expr(prefix_bits == 0):
                prefix_mask = cutlass.Uint32(0)
            else:
                prefix_mask = cutlass.Uint32(
                    ((1 << prefix_bits) - 1) << (32 - prefix_bits)
                )

            for item in cutlass.range_constexpr(self.exact_items_per_thread):
                histograms[
                    row_in_group,
                    thread_in_row * self.exact_items_per_thread + item,
                ] = cutlass.Int32(0)
            topk_barrier.arrive_and_wait()

            candidate = thread_in_row
            while candidate < candidate_count:
                bits, _ = self._candidate_bits_id(
                    candidate,
                    running_count,
                    score_stage,
                    tile_base,
                    running_values,
                    score_buffers,
                    row_in_group,
                )
                ordered = self._ordered(bits)
                if (ordered & prefix_mask) == prefix:
                    bucket = cutlass.Int32(
                        (ordered >> cutlass.Uint32(shift)) & cutlass.Uint32(0xFF)
                    )
                    cute.arch.atomic_add(
                        histograms.iterator + row_in_group * 512 + bucket,
                        cutlass.Int32(1),
                        sem="relaxed",
                        scope="cta",
                    )
                candidate += self.selector_threads_per_row
            topk_barrier.arrive_and_wait()

            if warp_in_row == 0:
                local_total = cutlass.Int32(0)
                counts = cute.make_rmem_tensor((8,), cutlass.Int32)
                for item in cutlass.range_constexpr(8):
                    count = histograms[row_in_group, lane_idx * 8 + item]
                    counts[item] = count
                    local_total += count
                inclusive = self._warp_scan(local_total, lane_idx)
                lane_base = inclusive - local_total
                local_prefix = cutlass.Int32(0)
                for item in cutlass.range_constexpr(8):
                    count = counts[item]
                    before = lane_base + local_prefix
                    after = before + count
                    if before < remaining and after >= remaining:
                        scalars[row_in_group, 0] = lane_idx * 8 + item
                        scalars[row_in_group, 1] = remaining - before
                    local_prefix += count
            topk_barrier.arrive_and_wait()
            bucket = scalars[row_in_group, 0]
            remaining = scalars[row_in_group, 1]
            prefix |= cutlass.Uint32(bucket) << cutlass.Uint32(shift)

        # Count and compact independently inside each warp. A ballot gives
        # deterministic lane ranks; the row-local 64-thread barrier supplies
        # the second warp's base without a contended shared atomic counter.
        warp_high_count = cutlass.Int32(0)
        warp_equal_count = cutlass.Int32(0)
        candidate_base = warp_in_row * 32
        while candidate_base < candidate_count:
            candidate = candidate_base + lane_idx
            active = candidate < candidate_count
            bits = cutlass.Uint32(0)
            if active:
                bits, _ = self._candidate_bits_id(
                    candidate,
                    running_count,
                    score_stage,
                    tile_base,
                    running_values,
                    score_buffers,
                    row_in_group,
                )
            ordered = self._ordered(bits)
            high = active and ordered < prefix
            equal = active and ordered == prefix
            high_mask = cute.arch.vote_ballot_sync(high)
            equal_mask = cute.arch.vote_ballot_sync(equal)
            warp_high_count += cutlass.Int32(cute.arch.popc(cutlass.Int32(high_mask)))
            warp_equal_count += cutlass.Int32(cute.arch.popc(cutlass.Int32(equal_mask)))
            candidate_base += self.selector_threads_per_row
        if lane_idx == 0:
            histograms[row_in_group, warp_in_row] = warp_high_count
            histograms[
                row_in_group,
                self.selector_warps_per_row + warp_in_row,
            ] = warp_equal_count
        topk_barrier.arrive_and_wait()

        high_base = cutlass.Int32(0)
        equal_base = cutlass.Int32(0)
        for selector_warp in cutlass.range_constexpr(self.selector_warps_per_row):
            equal_base += histograms[row_in_group, selector_warp]
            if selector_warp < warp_in_row:
                high_base += histograms[row_in_group, selector_warp]
                equal_base += histograms[
                    row_in_group,
                    self.selector_warps_per_row + selector_warp,
                ]
        high_offset = cutlass.Int32(0)
        equal_offset = cutlass.Int32(0)
        lower_lane_mask = cutlass.Uint32(0)
        if lane_idx > 0:
            lower_lane_mask = (
                cutlass.Uint32(1) << cutlass.Uint32(lane_idx)
            ) - cutlass.Uint32(1)
        candidate_base = warp_in_row * 32
        while candidate_base < candidate_count:
            candidate = candidate_base + lane_idx
            active = candidate < candidate_count
            bits = cutlass.Uint32(0)
            block_id = cutlass.Int32(-1)
            if active:
                bits, block_id = self._candidate_bits_id(
                    candidate,
                    running_count,
                    score_stage,
                    tile_base,
                    running_values,
                    score_buffers,
                    row_in_group,
                )
            ordered = self._ordered(bits)
            high = active and ordered < prefix
            equal = active and ordered == prefix
            high_mask = cute.arch.vote_ballot_sync(high)
            equal_mask = cute.arch.vote_ballot_sync(equal)
            high_rank = cutlass.Int32(
                cute.arch.popc(cutlass.Int32(high_mask & lower_lane_mask))
            )
            equal_rank = cutlass.Int32(
                cute.arch.popc(cutlass.Int32(equal_mask & lower_lane_mask))
            )
            if high:
                position = high_base + high_offset + high_rank
                if position < _TOPK:
                    compacted_row[position] = (
                        (cutlass.Int64(bits) + 1) << 32
                    ) | cutlass.Int64(block_id)
            if equal:
                position = equal_base + equal_offset + equal_rank
                if position < _TOPK:
                    compacted_row[position] = (
                        (cutlass.Int64(bits) + 1) << 32
                    ) | cutlass.Int64(block_id)
            high_offset += cutlass.Int32(cute.arch.popc(cutlass.Int32(high_mask)))
            equal_offset += cutlass.Int32(cute.arch.popc(cutlass.Int32(equal_mask)))
            candidate_base += self.selector_threads_per_row
        topk_barrier.arrive_and_wait()

    @cute.jit
    def _select_tile(
        self,
        row_in_group: cutlass.Int32,
        score_stage: cutlass.Int32,
        tile_base: cutlass.Int32,
        tile_valid: cutlass.Int32,
        running_count: cutlass.Int32,
        running_values: cute.Tensor,
        score_buffers: cute.Tensor,
        coarse_histograms: cute.Tensor,
        exact_histograms: cute.Tensor,
        fallback_histograms: cute.Tensor,
        scalars: cute.Tensor,
        tie_candidates: cute.Tensor,
        compacted_row: cute.Tensor,
        lane_idx: cutlass.Int32,
        warp_in_row: cutlass.Int32,
        thread_in_row: cutlass.Int32,
        topk_barrier: pipeline.NamedBarrier,
        clear_coarse_histogram: cutlass.Boolean,
    ):
        candidate_count = running_count + tile_valid
        remaining = candidate_count
        if remaining > _TOPK:
            remaining = cutlass.Int32(_TOPK)

        # The direct TMEM epilogue populated the FP16-derived coarse histogram
        # for this tile while each score was still in registers. Later tiles
        # merge the preceding exact running set, so add its 512-or-fewer
        # candidates before scanning the combined threshold.
        if tile_base > 0:
            for item in cutlass.range_constexpr(_TOPK // self.selector_threads_per_row):
                running_candidate = thread_in_row + item * self.selector_threads_per_row
                if running_candidate < running_count:
                    packed = running_values[item]
                    bits = cutlass.Uint32((packed >> 32) - 1)
                    bucket = self._coarse_ordered(bits)
                    cute.arch.atomic_add(
                        coarse_histograms.iterator
                        + row_in_group * 512
                        + ((running_candidate // 32) % 2) * 256
                        + bucket,
                        cutlass.Int32(1),
                        sem="relaxed",
                        scope="cta",
                    )
            topk_barrier.arrive_and_wait()

        # Start selection at the coarse threshold scan instead of rereading
        # all current-tile scores merely to build this first histogram.
        if warp_in_row == 0:
            local_total = cutlass.Int32(0)
            coarse_counts = cute.make_rmem_tensor((8,), cutlass.Int32)
            first_counts = cute.make_rmem_tensor((8,), cutlass.Int32)
            histogram_base = lane_idx * 8
            for item in cutlass.range_constexpr(8):
                first_count = coarse_histograms[row_in_group, histogram_base + item]
                second_count = coarse_histograms[
                    row_in_group, 256 + histogram_base + item
                ]
                first_counts[item] = first_count
                coarse_counts[item] = first_count + second_count
                local_total += first_count + second_count
            first_local_total = cutlass.Int32(0)
            for item in cutlass.range_constexpr(8):
                first_local_total += first_counts[item]
            inclusive = self._warp_scan(local_total, lane_idx)
            first_inclusive = self._warp_scan(first_local_total, lane_idx)
            lane_base = inclusive - local_total
            first_lane_base = first_inclusive - first_local_total
            local_prefix = cutlass.Int32(0)
            first_local_prefix = cutlass.Int32(0)
            for item in cutlass.range_constexpr(8):
                count = coarse_counts[item]
                first_count = first_counts[item]
                before = lane_base + local_prefix
                after = before + count
                if before < remaining and after >= remaining:
                    scalars[row_in_group, 0] = histogram_base + item
                    scalars[row_in_group, 1] = remaining - before
                    scalars[row_in_group, 2] = before
                    scalars[row_in_group, 3] = count
                    scalars[row_in_group, 4] = first_lane_base + first_local_prefix
                    scalars[row_in_group, 5] = first_count
                local_prefix += count
                first_local_prefix += first_count
        topk_barrier.arrive_and_wait()
        coarse_bucket = scalars[row_in_group, 0]
        tie_remaining = scalars[row_in_group, 1]
        better_count = scalars[row_in_group, 2]

        tie_count = scalars[row_in_group, 3]
        use_compact = tie_count <= 256
        if use_compact:
            # Collect definite winners and the coarse threshold bucket in one
            # second full pass. The sharded coarse histogram supplies each warp's
            # exact output base, avoiding a contended atomic allocation counter.
            lower_lane_mask = cutlass.Uint32(0)
            if lane_idx > 0:
                lower_lane_mask = (
                    cutlass.Uint32(1) << cutlass.Uint32(lane_idx)
                ) - cutlass.Uint32(1)
            better_base = cutlass.Int32(0)
            tie_base = cutlass.Int32(0)
            if warp_in_row == 1:
                better_base = scalars[row_in_group, 4]
                tie_base = scalars[row_in_group, 5]
            better_offset = cutlass.Int32(0)
            tie_offset = cutlass.Int32(0)
            candidate_base = warp_in_row * 32
            while candidate_base < candidate_count:
                candidate = candidate_base + lane_idx
                active = candidate < candidate_count
                bits = cutlass.Uint32(0)
                block_id = cutlass.Int32(-1)
                if active:
                    bits, block_id = self._candidate_bits_id(
                        candidate,
                        running_count,
                        score_stage,
                        tile_base,
                        running_values,
                        score_buffers,
                        row_in_group,
                    )
                ordered = self._ordered(bits)
                bucket = cutlass.Int32(0)
                if active:
                    bucket = self._coarse_ordered(bits)
                better = active and bucket < coarse_bucket
                tied = active and bucket == coarse_bucket
                better_mask = cute.arch.vote_ballot_sync(better)
                tie_mask = cute.arch.vote_ballot_sync(tied)
                better_batch = cutlass.Int32(cute.arch.popc(cutlass.Int32(better_mask)))
                tie_batch = cutlass.Int32(cute.arch.popc(cutlass.Int32(tie_mask)))
                packed = ((cutlass.Int64(bits) + 1) << 32) | cutlass.Int64(block_id)
                if better:
                    position = (
                        better_base
                        + better_offset
                        + cutlass.Int32(
                            cute.arch.popc(cutlass.Int32(better_mask & lower_lane_mask))
                        )
                    )
                    if position < _TOPK:
                        compacted_row[position] = packed
                if tied:
                    position = (
                        tie_base
                        + tie_offset
                        + cutlass.Int32(
                            cute.arch.popc(cutlass.Int32(tie_mask & lower_lane_mask))
                        )
                    )
                    if position < 256:
                        tie_candidates[row_in_group, position] = packed
                better_offset += better_batch
                tie_offset += tie_batch
                candidate_base += self.selector_threads_per_row
            topk_barrier.arrive_and_wait()

            prefix = cutlass.Uint32(0)
            for radix_round in cutlass.range_constexpr(4):
                shift = cutlass.const_expr(24 - radix_round * 8)
                if cutlass.const_expr(radix_round == 0):
                    prefix_mask = cutlass.Uint32(0)
                elif cutlass.const_expr(radix_round == 1):
                    prefix_mask = cutlass.Uint32(0xFF000000)
                elif cutlass.const_expr(radix_round == 2):
                    prefix_mask = cutlass.Uint32(0xFFFF0000)
                else:
                    prefix_mask = cutlass.Uint32(0xFFFFFF00)

                for item in cutlass.range_constexpr(self.exact_items_per_thread):
                    exact_histograms[
                        row_in_group,
                        thread_in_row * self.exact_items_per_thread + item,
                    ] = cutlass.Int32(0)
                topk_barrier.arrive_and_wait()

                tie_index = thread_in_row
                while tie_index < tie_count:
                    packed = tie_candidates[row_in_group, tie_index]
                    bits = cutlass.Uint32((packed >> 32) - 1)
                    ordered = self._ordered(bits)
                    if (ordered & prefix_mask) == prefix:
                        bucket = cutlass.Int32(
                            (ordered >> cutlass.Uint32(shift)) & cutlass.Uint32(0xFF)
                        )
                        cute.arch.atomic_add(
                            exact_histograms.iterator
                            + row_in_group * self.stream_tile
                            + bucket,
                            cutlass.Int32(1),
                            sem="relaxed",
                            scope="cta",
                        )
                    tie_index += self.selector_threads_per_row
                topk_barrier.arrive_and_wait()

                if warp_in_row == 0:
                    local_total = cutlass.Int32(0)
                    counts = cute.make_rmem_tensor((8,), cutlass.Int32)
                    for item in cutlass.range_constexpr(8):
                        count = exact_histograms[row_in_group, lane_idx * 8 + item]
                        counts[item] = count
                        local_total += count
                    inclusive = self._warp_scan(local_total, lane_idx)
                    lane_base = inclusive - local_total
                    local_prefix = cutlass.Int32(0)
                    for item in cutlass.range_constexpr(8):
                        count = counts[item]
                        before = lane_base + local_prefix
                        after = before + count
                        if before < tie_remaining and after >= tie_remaining:
                            scalars[row_in_group, 0] = lane_idx * 8 + item
                            scalars[row_in_group, 1] = tie_remaining - before
                        local_prefix += count
                topk_barrier.arrive_and_wait()
                bucket = scalars[row_in_group, 0]
                tie_remaining = scalars[row_in_group, 1]
                prefix |= cutlass.Uint32(bucket) << cutlass.Uint32(shift)

            warp_high_count = cutlass.Int32(0)
            warp_equal_count = cutlass.Int32(0)
            candidate_base = warp_in_row * 32
            while candidate_base < tie_count:
                tie_index = candidate_base + lane_idx
                active = tie_index < tie_count
                ordered = cutlass.Uint32(0)
                if active:
                    packed = tie_candidates[row_in_group, tie_index]
                    bits = cutlass.Uint32((packed >> 32) - 1)
                    ordered = self._ordered(bits)
                high = active and ordered < prefix
                equal = active and ordered == prefix
                high_mask = cute.arch.vote_ballot_sync(high)
                equal_mask = cute.arch.vote_ballot_sync(equal)
                warp_high_count += cutlass.Int32(
                    cute.arch.popc(cutlass.Int32(high_mask))
                )
                warp_equal_count += cutlass.Int32(
                    cute.arch.popc(cutlass.Int32(equal_mask))
                )
                candidate_base += self.selector_threads_per_row
            if lane_idx == 0:
                exact_histograms[row_in_group, warp_in_row] = warp_high_count
                exact_histograms[
                    row_in_group,
                    self.selector_warps_per_row + warp_in_row,
                ] = warp_equal_count
            topk_barrier.arrive_and_wait()

            high_base = better_count
            equal_base = better_count
            for selector_warp in cutlass.range_constexpr(self.selector_warps_per_row):
                equal_base += exact_histograms[row_in_group, selector_warp]
                if selector_warp < warp_in_row:
                    high_base += exact_histograms[row_in_group, selector_warp]
                    equal_base += exact_histograms[
                        row_in_group,
                        self.selector_warps_per_row + selector_warp,
                    ]
            high_offset = cutlass.Int32(0)
            equal_offset = cutlass.Int32(0)
            lower_lane_mask = cutlass.Uint32(0)
            if lane_idx > 0:
                lower_lane_mask = (
                    cutlass.Uint32(1) << cutlass.Uint32(lane_idx)
                ) - cutlass.Uint32(1)
            candidate_base = warp_in_row * 32
            while candidate_base < tie_count:
                tie_index = candidate_base + lane_idx
                active = tie_index < tie_count
                packed = cutlass.Int64(0)
                ordered = cutlass.Uint32(0)
                if active:
                    packed = tie_candidates[row_in_group, tie_index]
                    bits = cutlass.Uint32((packed >> 32) - 1)
                    ordered = self._ordered(bits)
                high = active and ordered < prefix
                equal = active and ordered == prefix
                high_mask = cute.arch.vote_ballot_sync(high)
                equal_mask = cute.arch.vote_ballot_sync(equal)
                high_rank = cutlass.Int32(
                    cute.arch.popc(cutlass.Int32(high_mask & lower_lane_mask))
                )
                equal_rank = cutlass.Int32(
                    cute.arch.popc(cutlass.Int32(equal_mask & lower_lane_mask))
                )
                if high:
                    position = high_base + high_offset + high_rank
                    if position < _TOPK:
                        compacted_row[position] = packed
                if equal:
                    position = equal_base + equal_offset + equal_rank
                    if position < _TOPK:
                        compacted_row[position] = packed
                high_offset += cutlass.Int32(cute.arch.popc(cutlass.Int32(high_mask)))
                equal_offset += cutlass.Int32(cute.arch.popc(cutlass.Int32(equal_mask)))
                candidate_base += self.selector_threads_per_row
            topk_barrier.arrive_and_wait()
        else:
            self._select_tile_full(
                row_in_group,
                score_stage,
                tile_base,
                tile_valid,
                running_count,
                running_values,
                score_buffers,
                fallback_histograms,
                scalars,
                compacted_row,
                lane_idx,
                warp_in_row,
                thread_in_row,
                topk_barrier,
            )

        # The single score-buffer stage prevents the next epilogue from
        # publishing until selectors release it. Clear the per-tile histogram
        # first so the next producer never accumulates into stale buckets.
        if clear_coarse_histogram:
            for item in cutlass.range_constexpr(self.histogram_items_per_thread):
                coarse_histograms[
                    row_in_group,
                    thread_in_row * self.histogram_items_per_thread + item,
                ] = cutlass.Int32(0)
            topk_barrier.arrive_and_wait()

    @cute.kernel
    def _kernel(
        self,
        tiled_mma: cute.TiledMma,
        smem_layout_k: cute.ComposedLayout,
        smem_layout_q: cute.ComposedLayout,
        query: cute.Tensor,
        keys: cute.Tensor,
        page_table: cute.Tensor,
        request_indices: cute.Tensor,
        complete_blocks: cute.Tensor,
        selected: cute.Tensor,
        q_stride_row: cutlass.Int32,
        q_stride_head: cutlass.Int32,
        queries_per_request: cutlass.Int32,
        page_size: cutlass.Constexpr[int],
    ):
        block_idx, _, _ = cute.arch.block_idx()
        if cutlass.const_expr(self.enable_pdl and self.group_layout == "ragged"):
            cute.arch.griddepcontrol_wait()
        row_base, valid_rows = self._query_group(
            request_indices, query.shape[0], queries_per_request, block_idx
        )
        if valid_rows > 0:
            self._run_group(
                tiled_mma,
                smem_layout_k,
                smem_layout_q,
                query,
                keys,
                page_table,
                request_indices,
                complete_blocks,
                selected,
                q_stride_row,
                q_stride_head,
                row_base,
                valid_rows,
                page_size,
            )

    @cute.jit
    def _run_group(
        self,
        tiled_mma: cute.TiledMma,
        smem_layout_k: cute.ComposedLayout,
        smem_layout_q: cute.ComposedLayout,
        query: cute.Tensor,
        keys: cute.Tensor,
        page_table: cute.Tensor,
        request_indices: cute.Tensor,
        complete_blocks: cute.Tensor,
        selected: cute.Tensor,
        q_stride_row: cutlass.Int32,
        q_stride_head: cutlass.Int32,
        row_base: cutlass.Int32,
        valid_rows: cutlass.Int32,
        page_size: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(tidx // 32)
        warpgroup_idx = cute.arch.make_warp_uniform(tidx // 128)
        warpgroup_tidx = tidx % 128
        # Preserve the aligned row arithmetic through the grouped helper.
        if cutlass.const_expr(self.group_layout == "aligned"):
            row_base = block_idx * self.query_group
        num_blocks = page_table.shape[1] * page_size

        if cutlass.const_expr(self.enable_pdl and self.group_layout != "ragged"):
            cute.arch.griddepcontrol_wait()
        frontier = cutlass.Int32(0)
        for row in cutlass.range_constexpr(self.query_group):
            count = self._row_complete(complete_blocks, row_base, row, valid_rows)
            if count > frontier:
                frontier = count
        if frontier > num_blocks:
            frontier = cutlass.Int32(num_blocks)
        if frontier <= _TOPK:
            # Every valid block qualifies; the score ranking has no effect.
            for iteration in cutlass.range_constexpr(
                cute.ceil_div(self.query_group * _TOPK, self.threads)
            ):
                index = tidx + iteration * self.threads
                if index < self.query_group * _TOPK:
                    query_row = index // _TOPK
                    block_id = index % _TOPK
                    count = self._row_complete(
                        complete_blocks, row_base, query_row, valid_rows
                    )
                    value = cutlass.Int32(-1)
                    if block_id < count and block_id < num_blocks:
                        value = block_id
                    self._store_selected(
                        selected,
                        row_base + query_row,
                        block_id,
                        value,
                        row_base,
                        valid_rows,
                    )
            cute.arch.sync_threads()
            if cutlass.const_expr(self.enable_pdl):
                cute.arch.griddepcontrol_launch_dependents()
        else:
            smem = utils.SmemAllocator()
            tmem_ptr_smem = smem.allocate_array(cutlass.Int32)
            k_pipeline_ptr = smem.allocate_array(cutlass.Int64, _K_STAGES * 2)
            acc_pipeline_ptr = smem.allocate_array(cutlass.Int64, _ACC_STAGES * 2)
            score_pipeline_ptr = smem.allocate_array(cutlass.Int64, _SCORE_STAGES * 2)
            sK = smem.allocate_tensor(
                cutlass.BFloat16, smem_layout_k.outer, 128, smem_layout_k.inner
            )
            sQ = smem.allocate_tensor(
                cutlass.BFloat16, smem_layout_q.outer, 128, smem_layout_q.inner
            )
            score_buffers = smem.allocate_tensor(
                cutlass.Float32,
                cute.make_layout(
                    (_SCORE_STAGES, self.query_group, self.stream_tile),
                    stride=(
                        self.query_group * self.stream_tile,
                        self.stream_tile,
                        1,
                    ),
                ),
                128,
            )
            coarse_histograms = smem.allocate_tensor(
                cutlass.Int32,
                cute.make_layout((self.query_group, 512), stride=(512, 1)),
                128,
            )
            # K is dead once the epilogue has published the complete score tile.
            # Selectors reuse one 32 KiB K stage to compact all eight running sets.
            # A tile-boundary barrier protects their loads before the next K copy.
            running_scratch = cute.make_tensor(
                cute.recast_ptr(sK.iterator, dtype=cutlass.Int64),
                cute.make_layout((self.query_group, _TOPK), stride=(_TOPK, 1)),
            )
            reuse_barrier = pipeline.NamedBarrier(
                barrier_id=10,
                num_threads=128 + self.query_group * self.selector_threads_per_row,
            )
            scalars = smem.allocate_tensor(
                cutlass.Int32,
                cute.make_layout((self.query_group, 6), stride=(6, 1)),
                16,
            )
            row_completes = smem.allocate_tensor(
                cutlass.Int32,
                cute.make_layout((self.query_group,), stride=(1,)),
                16,
            )
            # The sharded coarse histogram is dead before tie compaction. Reuse its
            # 16 KiB storage for 256 packed int64 ties per row. Exact and fallback
            # radix scratch remain row-private so rows can choose compact versus
            # full refinement independently.
            tie_candidates = cute.make_tensor(
                cute.recast_ptr(coarse_histograms.iterator, dtype=cutlass.Int64),
                cute.make_layout((self.query_group, 256), stride=(256, 1)),
            )
            exact_histograms = cute.make_tensor(
                cute.recast_ptr(score_buffers.iterator, dtype=cutlass.Int32),
                cute.make_layout((self.query_group, 256), stride=(self.stream_tile, 1)),
            )
            fallback_histograms = cute.make_tensor(
                coarse_histograms.iterator,
                cute.make_layout((self.query_group, 256), stride=(512, 1)),
            )
            if warp_idx == self.mma_warp:
                cute.arch.alloc_tmem(self.mma_n * _ACC_STAGES, tmem_ptr_smem)

            k_producer, k_consumer = pipeline.PipelineAsyncUmma.create(
                num_stages=_K_STAGES,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                barrier_storage=k_pipeline_ptr,
                defer_sync=True,
            ).make_participants()
            acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
                num_stages=_ACC_STAGES,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
                barrier_storage=acc_pipeline_ptr,
                defer_sync=True,
            ).make_participants()
            score_producer, score_consumer = pipeline.PipelineAsync.create(
                num_stages=_SCORE_STAGES,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    self.query_group * self.selector_threads_per_row,
                ),
                barrier_storage=score_pipeline_ptr,
                defer_sync=True,
            ).make_participants()

            cute.arch.mbarrier_init_fence()

            # Producers may release us before publishing Q or the row frontiers.
            # Wait before either read, including the asynchronous Q copy.
            if cutlass.const_expr(self.enable_pdl and self.group_layout != "ragged"):
                cute.arch.griddepcontrol_wait()

            if warpgroup_idx == _K_COPY_WARPGROUP:
                q_base_addr = cute.recast_ptr(
                    sQ.iterator, dtype=cutlass.BFloat16
                ).toint()
                for iteration in cutlass.range_constexpr(self.query_copy_iterations):
                    vector = warpgroup_tidx + iteration * 128
                    q_head = vector // 16
                    col = (vector - q_head * 16) * 8
                    q_row = q_head // _HEADS
                    head = q_head - q_row * _HEADS
                    source = (
                        query.iterator
                        + (row_base + q_row) * q_stride_row
                        + head * q_stride_head
                        + col
                    )
                    logical_byte = (
                        q_head * 128 + (col % 64) * 2 + (col // 64) * self.mma_n * 128
                    )
                    logical_address = q_base_addr + logical_byte
                    swizzled_address = logical_address ^ ((logical_address >> 3) & 0x70)
                    destination = cute.make_ptr(
                        cutlass.BFloat16,
                        swizzled_address,
                        cute.AddressSpace.smem,
                        assumed_align=16,
                    )
                    self._copy_query(destination, source, q_row, valid_rows)
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.fence_view_async_shared()

            # Initialize the first tile's coarse histograms before the epilogue can
            # publish scores into them. The existing CTA synchronization also
            # protects the query copy, so this adds no new barrier.
            if warp_idx >= self.topk_first_warp and warp_idx < self.mma_warp:
                consumer_warp = warp_idx - self.topk_first_warp
                row_in_group = consumer_warp // self.selector_warps_per_row
                warp_in_row = consumer_warp - row_in_group * self.selector_warps_per_row
                thread_in_row = warp_in_row * 32 + lane_idx
                if row_in_group < self.query_group:
                    for item in cutlass.range_constexpr(
                        self.histogram_items_per_thread
                    ):
                        coarse_histograms[
                            row_in_group,
                            thread_in_row * self.histogram_items_per_thread + item,
                        ] = cutlass.Int32(0)
                    if thread_in_row == 0:
                        for item in cutlass.range_constexpr(6):
                            scalars[row_in_group, item] = cutlass.Int32(0)
                        row_complete = self._row_complete(
                            complete_blocks, row_base, row_in_group, valid_rows
                        )
                        if row_complete > num_blocks:
                            row_complete = cutlass.Int32(num_blocks)
                        row_completes[row_in_group] = row_complete
            cute.arch.sync_threads()

            if cutlass.const_expr(self.enable_pdl):
                cute.arch.griddepcontrol_launch_dependents()

            request = request_indices[row_base]
            group_complete = cutlass.Int32(0)
            for query_row in cutlass.range_constexpr(self.query_group):
                count = self._row_complete(
                    complete_blocks, row_base, query_row, valid_rows
                )
                if count > group_complete:  # noqa: PLR1730
                    group_complete = count
            if group_complete > num_blocks:
                group_complete = cutlass.Int32(num_blocks)
            num_stream_tiles = cute.ceil_div(group_complete, self.stream_tile)

            thr_mma = tiled_mma.get_slice(0)
            k_fragment = thr_mma.make_fragment_A(sK)
            q_desc = thr_mma.make_fragment_B(sQ)
            acc_shape = tiled_mma.partition_shape_C((_MMA_M, self.mma_n, _ACC_STAGES))
            acc_fragment = thr_mma.make_fragment_C(acc_shape)
            tmem_ptr = cute.arch.retrieve_tmem_ptr(cutlass.Int32, 16, tmem_ptr_smem)
            tAcc = cute.make_tensor(
                cute.recast_ptr(tmem_ptr, dtype=cutlass.Float32), acc_fragment.layout
            )
            k_base_addr = cute.recast_ptr(sK.iterator, dtype=cutlass.BFloat16).toint()

            if warpgroup_idx == _K_COPY_WARPGROUP:
                for stream_tile in cutlass.range(num_stream_tiles):
                    tile_subtiles = cute.ceil_div(
                        group_complete - stream_tile * self.stream_tile, _MMA_M
                    )
                    if tile_subtiles > self.stream_tile // _MMA_M:
                        tile_subtiles = cutlass.Int32(self.stream_tile // _MMA_M)
                    for subtile in cutlass.range(tile_subtiles):
                        tile_idx = stream_tile * (self.stream_tile // _MMA_M) + subtile
                        k_handle = k_producer.acquire_and_advance()
                        index_row = (
                            (warpgroup_tidx // 32) * 2
                            + lane_idx % 2
                            + (lane_idx // 2) * 8
                        )
                        logical_block = tile_idx * _MMA_M + index_row
                        slot = cutlass.Int32(0)
                        if logical_block < group_complete:
                            page_column = logical_block // page_size
                            offset = logical_block - page_column * page_size
                            page = page_table[request, page_column]
                            slot = page * page_size + offset
                        for iteration in cutlass.range(16, unroll=2):
                            vector = warpgroup_tidx + iteration * 128
                            row = vector // 16
                            vector_in_row = vector - row * 16
                            col = vector_in_row * 8
                            source_slot = cute.arch.shuffle_sync(
                                slot, iteration * 2 + lane_idx // 16
                            )
                            source = keys.iterator + source_slot * _HEAD_DIM + col
                            logical_byte = (
                                row * 128
                                + (col % 64) * 2
                                + (col // 64) * 16384
                                + k_handle.index * 32768
                            )
                            logical_address = k_base_addr + logical_byte
                            swizzled_address = logical_address ^ (
                                (logical_address >> 3) & 0x70
                            )
                            destination = cute.make_ptr(
                                cutlass.BFloat16,
                                swizzled_address,
                                cute.AddressSpace.smem,
                                assumed_align=16,
                            )
                            _copy_u128(destination, source)
                        cute.arch.cp_async_commit_group()
                        cute.arch.cp_async_wait_group(0)
                        cute.arch.fence_view_async_shared()
                        k_handle.commit()
                    reuse_barrier.arrive_and_wait()
                    cute.arch.fence_view_async_shared()

            elif warp_idx == self.mma_warp:
                for stream_tile in cutlass.range(num_stream_tiles):
                    tile_subtiles = cute.ceil_div(
                        group_complete - stream_tile * self.stream_tile, _MMA_M
                    )
                    if tile_subtiles > self.stream_tile // _MMA_M:
                        tile_subtiles = cutlass.Int32(self.stream_tile // _MMA_M)
                    for _ in cutlass.range(tile_subtiles):
                        k_handle = k_consumer.wait_and_advance()
                        acc_handle = acc_producer.acquire_and_advance()
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        for mma_k in cutlass.range_constexpr(
                            cute.size(k_fragment, mode=[2])
                        ):
                            cute.gemm(
                                tiled_mma,
                                tAcc[None, None, None, acc_handle.index],
                                k_fragment[None, None, mma_k, k_handle.index],
                                q_desc[None, None, mma_k, 0],
                                tAcc[None, None, None, acc_handle.index],
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        acc_handle.commit()
                        k_handle.release()
                acc_producer.tail()
                cute.arch.relinquish_tmem_alloc_permit()
                cute.arch.dealloc_tmem(tmem_ptr, self.mma_n * _ACC_STAGES)

            elif warpgroup_idx == self.epilogue_warpgroup:
                tmem_atom = cute.make_copy_atom(
                    tcgen05.Ld32x32bOp(tcgen05.Repetition(self.mma_n)),
                    cutlass.Float32,
                )
                tmem_copy = tcgen05.make_tmem_copy(tmem_atom, tAcc[None, None, None, 0])
                thread_copy = tmem_copy.get_slice(warpgroup_tidx)
                tSrc = thread_copy.partition_S(tAcc)
                identity = cute.make_identity_tensor((_MMA_M, self.mma_n))
                identity_c = thr_mma.partition_C(identity)
                tCoord = thread_copy.partition_D(identity_c)
                register_scores = cute.make_rmem_tensor(tCoord.shape, cutlass.Float32)
                for stream_tile in cutlass.range(num_stream_tiles):
                    score_handle = score_producer.acquire_and_advance()
                    tile_subtiles = cute.ceil_div(
                        group_complete - stream_tile * self.stream_tile, _MMA_M
                    )
                    if tile_subtiles > self.stream_tile // _MMA_M:
                        tile_subtiles = cutlass.Int32(self.stream_tile // _MMA_M)
                    for subtile in cutlass.range(tile_subtiles):
                        acc_handle = acc_consumer.wait_and_advance()
                        cute.copy(
                            tmem_copy,
                            tSrc[None, None, None, None, acc_handle.index],
                            register_scores,
                        )
                        cute.arch.fence_view_async_tmem_load()
                        row_scores = cute.make_rmem_tensor(
                            (_QUERY_GROUP,), cutlass.Float32
                        )
                        for query_row in cutlass.range_constexpr(_QUERY_GROUP):
                            row_scores[query_row] = cutlass.Float32(0.0)
                        for value_idx in cutlass.range_constexpr(
                            cute.size(register_scores)
                        ):
                            coord = tCoord[value_idx]
                            value = register_scores[value_idx]
                            if value > 0.0:
                                for query_row in cutlass.range_constexpr(_QUERY_GROUP):
                                    if (
                                        coord[1] >= query_row * _HEADS
                                        and coord[1] < (query_row + 1) * _HEADS
                                    ):
                                        row_scores[query_row] += value
                        key_row = tCoord[0][0]
                        for query_row in cutlass.range_constexpr(_QUERY_GROUP):
                            score = row_scores[query_row]
                            score_buffers[
                                score_handle.index,
                                query_row,
                                subtile * _MMA_M + key_row,
                            ] = score
                            logical_block = (
                                stream_tile * self.stream_tile
                                + subtile * _MMA_M
                                + key_row
                            )
                            if logical_block < row_completes[query_row]:
                                bits = self._float_bits(score)
                                bucket = self._coarse_ordered(bits)
                                selector_warp = (logical_block // 32) % 2
                                cute.arch.atomic_add(
                                    coarse_histograms.iterator
                                    + query_row * 512
                                    + selector_warp * 256
                                    + bucket,
                                    cutlass.Int32(1),
                                    sem="relaxed",
                                    scope="cta",
                                )
                        acc_handle.release()
                    score_handle.commit()
                score_producer.tail()

            elif warp_idx >= self.topk_first_warp and warp_idx < self.mma_warp:
                consumer_warp = warp_idx - self.topk_first_warp
                row_in_group = consumer_warp // self.selector_warps_per_row
                warp_in_row = consumer_warp - row_in_group * self.selector_warps_per_row
                thread_in_row = warp_in_row * 32 + lane_idx
                if row_in_group < self.query_group:
                    topk_barrier = pipeline.NamedBarrier(
                        barrier_id=row_in_group + 2,
                        num_threads=self.selector_threads_per_row,
                    )
                    row = row_base + row_in_group
                    complete = row_completes[row_in_group]
                    selected_row = selected[row, None]
                    running_values = cute.make_rmem_tensor(
                        (_TOPK // self.selector_threads_per_row,), cutlass.Int64
                    )
                    # Each thread owns eight packed (FP32 score, block id) entries.
                    for item in cutlass.range_constexpr(
                        _TOPK // self.selector_threads_per_row
                    ):
                        running_values[item] = cutlass.Int64(0)
                    for stream_tile in cutlass.range(num_stream_tiles):
                        score_handle = score_consumer.wait_and_advance()
                        tile_base = stream_tile * self.stream_tile
                        tile_valid = cutlass.Int32(0)
                        if tile_base < complete:
                            tile_valid = complete - tile_base
                            if tile_valid > self.stream_tile:
                                tile_valid = cutlass.Int32(self.stream_tile)

                        # Selection participation is CTA-uniform, which keeps the
                        # selector named barrier valid at the 512-block boundary.
                        do_select = group_complete > _TOPK
                        if do_select:
                            running_count = complete
                            if running_count > tile_base:  # noqa: PLR1730
                                running_count = tile_base
                            if running_count > _TOPK:
                                running_count = cutlass.Int32(_TOPK)
                            compacted_row = running_scratch[row_in_group, None]
                            self._select_tile(
                                row_in_group,
                                score_handle.index,
                                tile_base,
                                tile_valid,
                                running_count,
                                running_values,
                                score_buffers,
                                coarse_histograms,
                                exact_histograms,
                                fallback_histograms,
                                scalars,
                                tie_candidates,
                                compacted_row,
                                lane_idx,
                                warp_in_row,
                                thread_in_row,
                                topk_barrier,
                                stream_tile + 1 < num_stream_tiles,
                            )
                            for item in cutlass.range_constexpr(
                                _TOPK // self.selector_threads_per_row
                            ):
                                index = (
                                    thread_in_row + item * self.selector_threads_per_row
                                )
                                value = cutlass.Int64(0)
                                if index < complete:
                                    value = compacted_row[index]
                                running_values[item] = value
                        # The copy warps resume only after every selector has
                        # loaded its new running set from the reused K storage.
                        reuse_barrier.arrive_and_wait()
                        score_handle.release()

                    for item in cutlass.range_constexpr(
                        _TOPK // self.selector_threads_per_row
                    ):
                        index = thread_in_row + item * self.selector_threads_per_row
                        value = cutlass.Int32(-1)
                        if index < complete:
                            if complete <= _TOPK:
                                value = index
                            else:
                                value = cutlass.Int32(
                                    running_values[item] & cutlass.Int64(0xFFFFFFFF)
                                )
                        self._store_row(
                            selected_row, index, value, row_in_group, valid_rows
                        )

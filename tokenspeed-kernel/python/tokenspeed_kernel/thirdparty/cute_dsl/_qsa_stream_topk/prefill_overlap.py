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

"""Double-buffered scoring and cutoff-filtered selection across tiles."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import cute, pipeline, utils
from cutlass._mlir.dialects import llvm as _llvm
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from tokenspeed_kernel.thirdparty.cute_dsl._qsa_stream_topk.prefill import PrefillTopK

_QUERY_GROUP = 8
_HEADS = 4
_HEAD_DIM = 128
_MMA_M = 128
_K_STAGES = 2
_ACC_STAGES = 2
_STREAM_TILE = 1120
_TOPK = 512
_SCORE_STAGES = 2
_SELECTOR_WARPS_PER_ROW = 2
_K_COPY_FIRST_WARP = 8
_K_COPY_WARPS = 2
_EPILOGUE_WARPGROUP = 0
_TOPK_FIRST_WARP = 10
_MMA_WARP = 9
_SELECTOR_END_WARP = _TOPK_FIRST_WARP + _QUERY_GROUP * _SELECTOR_WARPS_PER_ROW


class OverlapPrefillTopK(PrefillTopK):
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
        self.query_copy_iterations = _QUERY_GROUP * _HEADS * 16 // (_K_COPY_WARPS * 32)
        self.mma_warp = _MMA_WARP
        self.epilogue_warpgroup = _EPILOGUE_WARPGROUP
        self.topk_first_warp = _TOPK_FIRST_WARP
        self.threads = _SELECTOR_END_WARP * 32

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
        k_source = cute.make_tensor(
            keys.iterator,
            cute.make_layout((keys.shape[0], _HEAD_DIM), stride=(_HEAD_DIM, 1)),
        )
        k_page_layout = cute.tile_to_shape(
            tcgen05.make_smem_layout_atom(
                tcgen05.SmemLayoutAtomKind.K_SW128, mma_dtype
            ),
            (64, 64),
            order=(1, 0),
        )
        k_tma_atom, k_tma_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            k_source,
            k_page_layout,
            (64, 64),
        )
        load_mma = sm100_utils.make_trivial_tiled_mma(
            mma_dtype,
            mma_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            acc_dtype,
            tcgen05.CtaGroup.ONE,
            (_MMA_M, self.mma_n // 2),
            tcgen05.OperandSource.SMEM,
        )
        self._kernel(
            tiled_mma,
            load_mma,
            smem_layout_k,
            smem_layout_q,
            k_tma_atom,
            k_tma_tensor,
            k_page_layout,
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
    def _candidate_bits_id(
        self,
        candidate: cutlass.Int32,
        running_count: cutlass.Int32,
        score_stage: cutlass.Int32,
        tile_base: cutlass.Int32,
        running_values: cute.Tensor,
        score_buffers: cute.Tensor,
        score_offsets: cute.Tensor,
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
            block_id = tile_base + cutlass.Int32(
                score_offsets[score_stage, row_in_group, offset]
            )
        return bits, block_id

    # A fixed group lets the compact scan read its running set directly
    # instead of searching the per-thread registers as the radix path does.
    @cute.jit
    def _candidate_const_bits_id(
        self,
        group: cutlass.Constexpr[int],
        candidate: cutlass.Int32,
        running_count: cutlass.Int32,
        score_stage: cutlass.Int32,
        tile_base: cutlass.Int32,
        running_values: cute.Tensor,
        score_buffers: cute.Tensor,
        score_offsets: cute.Tensor,
        row_in_group: cutlass.Int32,
    ):
        bits = cutlass.Uint32(0)
        block_id = cutlass.Int32(-1)
        if cutlass.const_expr(group < _TOPK // self.selector_threads_per_row):
            if candidate < running_count:
                packed = running_values[group]
                bits = cutlass.Uint32((packed >> 32) - 1)
                block_id = cutlass.Int32(packed & cutlass.Int64(0xFFFFFFFF))
            else:
                offset = candidate - running_count
                bits = self._float_bits(
                    score_buffers[score_stage, row_in_group, offset]
                )
                block_id = tile_base + cutlass.Int32(
                    score_offsets[score_stage, row_in_group, offset]
                )
        else:
            offset = candidate - running_count
            bits = self._float_bits(score_buffers[score_stage, row_in_group, offset])
            block_id = tile_base + cutlass.Int32(
                score_offsets[score_stage, row_in_group, offset]
            )
        return bits, block_id

    @cute.jit
    def _window_coarse(self, bits: cutlass.Uint32, origin: cutlass.Int32):
        value = cutlass.Float32(
            _llvm.bitcast(cutlass.Float32.mlir_type, bits.ir_value())
        )
        half = cutlass.Float16(value)
        half_bits = cutlass.Uint16(
            _llvm.bitcast(cutlass.Uint16.mlir_type, half.ir_value())
        )
        ordered = (half_bits ^ cutlass.Uint16(0xFFFF)) & cutlass.Uint16(0x7FFF)
        result = cutlass.Int32(ordered >> cutlass.Uint16(7))
        if origin > -1024:
            result = cutlass.Int32(ordered >> cutlass.Uint16(5)) - origin
            if result < 0:
                result = cutlass.Int32(0)
            if result > 255:
                result = cutlass.Int32(255)
        return result

    @cute.jit
    def _half_suffix(self, bits: cutlass.Uint32):
        value = cutlass.Float32(
            _llvm.bitcast(cutlass.Float32.mlir_type, bits.ir_value())
        )
        half = cutlass.Float16(value)
        half_bits = cutlass.Uint16(
            _llvm.bitcast(cutlass.Uint16.mlir_type, half.ir_value())
        )
        return cutlass.Int32((half_bits ^ cutlass.Uint16(0xFFFF)) & cutlass.Uint16(127))

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
        score_offsets: cute.Tensor,
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
                    score_offsets,
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
                    score_offsets,
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
                    score_offsets,
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
        return prefix

    @cute.jit
    def _sort_boundary_bits(
        self, bits: cutlass.Uint32, lane: cutlass.Int32, stages: cutlass.Constexpr[int]
    ):
        for stage in cutlass.range_constexpr(stages):
            width = cutlass.const_expr(1 << (stage + 1))
            for inner in cutlass.range_constexpr(stage + 1):
                distance = cutlass.const_expr(1 << (stage - inner))
                peer = cute.arch.shuffle_sync(bits, lane ^ distance)
                take_max = ((lane & width) == 0) == ((lane & distance) == 0)
                if take_max:
                    if peer > bits:
                        bits = peer
                else:
                    if peer < bits:
                        bits = peer
        return bits

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
        score_offsets: cute.Tensor,
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
        coarse_origin = cutlass.Int32(-32768)
        cutoff_bits = cutlass.Uint32(0)
        encoded_cutoff = scalars[row_in_group, 6]
        if encoded_cutoff > 0:
            cutoff_bits = cutlass.Uint32(encoded_cutoff) - cutlass.Uint32(1)
            cutoff_value = cutlass.Float32(
                _llvm.bitcast(cutlass.Float32.mlir_type, cutoff_bits.ir_value())
            )
            cutoff_half = cutlass.Float16(cutoff_value)
            cutoff_half_bits = cutlass.Uint16(
                _llvm.bitcast(cutlass.Uint16.mlir_type, cutoff_half.ir_value())
            )
            ordered_half = (cutoff_half_bits ^ cutlass.Uint16(0xFFFF)) & cutlass.Uint16(
                0x7FFF
            )
            coarse_origin = cutlass.Int32(ordered_half >> cutlass.Uint16(5)) - 255
        remaining = candidate_count
        if remaining > _TOPK:
            remaining = cutlass.Int32(_TOPK)

        # A selector-owned histogram stays independent of both score stages.
        # Its full scan runs while producers score the following tile.
        offset = thread_in_row
        while offset < tile_valid:
            bits = self._float_bits(score_buffers[score_stage, row_in_group, offset])
            if bits >= cutoff_bits:
                bucket = self._window_coarse(bits, coarse_origin)
                cute.arch.atomic_add(
                    coarse_histograms.iterator
                    + row_in_group * 512
                    + warp_in_row * 256
                    + bucket,
                    cutlass.Int32(1),
                    sem="relaxed",
                    scope="cta",
                )
            offset += self.selector_threads_per_row
        topk_barrier.arrive_and_wait()

        # Merge the preceding exact running set into the tile histogram.
        if running_count > 0:
            for item in cutlass.range_constexpr(_TOPK // self.selector_threads_per_row):
                running_candidate = thread_in_row + item * self.selector_threads_per_row
                if running_candidate < running_count:
                    packed = running_values[item]
                    bits = cutlass.Uint32((packed >> 32) - 1)
                    bucket = self._window_coarse(bits, coarse_origin)
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
        use_compact = tie_count <= 256 and (
            coarse_origin == -32768 or coarse_bucket > 0
        )
        prefix = cutlass.Uint32(0)
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
            # Unroll the bounded scan to keep running-set register indices static.
            for candidate_group in cutlass.range_constexpr(
                cute.ceil_div(_TOPK + self.stream_tile, self.selector_threads_per_row)
            ):
                candidate_base = (
                    warp_in_row * 32 + candidate_group * self.selector_threads_per_row
                )
                if candidate_base < candidate_count:
                    candidate = candidate_base + lane_idx
                    active = candidate < candidate_count
                    bits = cutlass.Uint32(0)
                    block_id = cutlass.Int32(-1)
                    if active:
                        bits, block_id = self._candidate_const_bits_id(
                            candidate_group,
                            candidate,
                            running_count,
                            score_stage,
                            tile_base,
                            running_values,
                            score_buffers,
                            score_offsets,
                            row_in_group,
                        )
                    active = active and bits >= cutoff_bits
                    ordered = self._ordered(bits)
                    bucket = cutlass.Int32(0)
                    if active:
                        bucket = self._window_coarse(bits, coarse_origin)
                    better = active and bucket < coarse_bucket
                    tied = active and bucket == coarse_bucket
                    better_mask = cute.arch.vote_ballot_sync(better)
                    tie_mask = cute.arch.vote_ballot_sync(tied)
                    better_batch = cutlass.Int32(
                        cute.arch.popc(cutlass.Int32(better_mask))
                    )
                    tie_batch = cutlass.Int32(cute.arch.popc(cutlass.Int32(tie_mask)))
                    packed = ((cutlass.Int64(bits) + 1) << 32) | cutlass.Int64(block_id)
                    if better:
                        position = (
                            better_base
                            + better_offset
                            + cutlass.Int32(
                                cute.arch.popc(
                                    cutlass.Int32(better_mask & lower_lane_mask)
                                )
                            )
                        )
                        if position < _TOPK:
                            compacted_row[position] = packed
                    if tied:
                        position = (
                            tie_base
                            + tie_offset
                            + cutlass.Int32(
                                cute.arch.popc(
                                    cutlass.Int32(tie_mask & lower_lane_mask)
                                )
                            )
                        )
                        if position < 256:
                            tie_candidates[row_in_group, position] = packed
                    better_offset += better_batch
                    tie_offset += tie_batch
            topk_barrier.arrive_and_wait()

            # Resolve the remaining FP16 bits only inside the compact bucket.
            # The FP32 threshold still decides every selected score.
            original_remaining = tie_remaining
            for item in cutlass.range_constexpr(128 // self.selector_threads_per_row):
                exact_histograms[
                    row_in_group, thread_in_row + item * self.selector_threads_per_row
                ] = cutlass.Int32(0)
            topk_barrier.arrive_and_wait()
            fine_index = thread_in_row
            while fine_index < tie_count:
                packed = tie_candidates[row_in_group, fine_index]
                bits = cutlass.Uint32((packed >> 32) - 1)
                suffix = self._half_suffix(bits)
                cute.arch.atomic_add(
                    exact_histograms.iterator
                    + row_in_group * self.stream_tile
                    + suffix,
                    cutlass.Int32(1),
                    sem="relaxed",
                    scope="cta",
                )
                fine_index += self.selector_threads_per_row
            topk_barrier.arrive_and_wait()
            if warp_in_row == 0:
                fine_counts = cute.make_rmem_tensor((4,), cutlass.Int32)
                fine_total = cutlass.Int32(0)
                for item in cutlass.range_constexpr(4):
                    fine_counts[item] = exact_histograms[
                        row_in_group, lane_idx * 4 + item
                    ]
                    fine_total += fine_counts[item]
                fine_inclusive = self._warp_scan(fine_total, lane_idx)
                fine_before = fine_inclusive - fine_total
                for item in cutlass.range_constexpr(4):
                    fine_after = fine_before + fine_counts[item]
                    if (
                        fine_before < original_remaining
                        and fine_after >= original_remaining
                    ):
                        scalars[row_in_group, 0] = lane_idx * 4 + item
                        scalars[row_in_group, 1] = original_remaining - fine_before
                        scalars[row_in_group, 2] = fine_counts[item]
                    fine_before = fine_after
            topk_barrier.arrive_and_wait()
            refined_bucket = scalars[row_in_group, 0]
            refined_remaining = scalars[row_in_group, 1]
            refined_count = scalars[row_in_group, 2]
            if refined_count == 1:
                # Exactly one matching FP32 score owns this shared result.
                fine_index = thread_in_row
                while fine_index < tie_count:
                    bits = cutlass.Uint32(
                        (tie_candidates[row_in_group, fine_index] >> 32) - 1
                    )
                    if self._half_suffix(bits) == refined_bucket:
                        scalars[row_in_group, 0] = cutlass.Int32(bits)
                    fine_index += self.selector_threads_per_row
                topk_barrier.arrive_and_wait()
                prefix = self._ordered(cutlass.Uint32(scalars[row_in_group, 0]))
            elif refined_remaining == 1 or refined_remaining == refined_count:
                local_min = cutlass.Uint32(0xFFFFFFFF)
                local_max = cutlass.Uint32(0)
                fine_index = thread_in_row
                while fine_index < tie_count:
                    bits = cutlass.Uint32(
                        (tie_candidates[row_in_group, fine_index] >> 32) - 1
                    )
                    if self._half_suffix(bits) == refined_bucket:
                        if bits < local_min:
                            local_min = bits
                        if bits > local_max:
                            local_max = bits
                    fine_index += self.selector_threads_per_row
                warp_min = cute.arch.warp_redux_sync(local_min, "umin", 0xFFFFFFFF)
                warp_max = cute.arch.warp_redux_sync(local_max, "umax", 0xFFFFFFFF)
                result = warp_min
                if refined_remaining == 1:
                    result = warp_max
                if lane_idx == 0:
                    exact_histograms[row_in_group, warp_in_row] = cutlass.Int32(result)
                topk_barrier.arrive_and_wait()
                if thread_in_row == 0:
                    first = cutlass.Uint32(exact_histograms[row_in_group, 0])
                    second = cutlass.Uint32(exact_histograms[row_in_group, 1])
                    if refined_remaining == 1:
                        if second > first:
                            first = second
                    else:
                        if second < first:
                            first = second
                    scalars[row_in_group, 0] = cutlass.Int32(first)
                topk_barrier.arrive_and_wait()
                prefix = self._ordered(cutlass.Uint32(scalars[row_in_group, 0]))
            elif refined_count <= 32:
                if thread_in_row == 0:
                    scalars[row_in_group, 4] = cutlass.Int32(0)
                topk_barrier.arrive_and_wait()
                fine_index = thread_in_row
                while fine_index < tie_count:
                    packed = tie_candidates[row_in_group, fine_index]
                    bits = cutlass.Uint32((packed >> 32) - 1)
                    if self._half_suffix(bits) == refined_bucket:
                        position = cute.arch.atomic_add(
                            scalars.iterator + row_in_group * 7 + 4,
                            cutlass.Int32(1),
                            sem="relaxed",
                            scope="cta",
                        )
                        exact_histograms[row_in_group, position] = cutlass.Int32(bits)
                    fine_index += self.selector_threads_per_row
                topk_barrier.arrive_and_wait()
                if warp_in_row == 0:
                    sorted_bits = cutlass.Uint32(0)
                    if lane_idx < refined_count:
                        sorted_bits = cutlass.Uint32(
                            exact_histograms[row_in_group, lane_idx]
                        )
                    # Nonnegative FP32 bits have their numerical order. Padding
                    # with zero cannot change a rank inside refined_count.
                    if refined_count <= 8:
                        sorted_bits = self._sort_boundary_bits(sorted_bits, lane_idx, 3)
                    elif refined_count <= 16:
                        sorted_bits = self._sort_boundary_bits(sorted_bits, lane_idx, 4)
                    else:
                        sorted_bits = self._sort_boundary_bits(sorted_bits, lane_idx, 5)
                    cutoff_bits = cute.arch.shuffle_sync(
                        sorted_bits, refined_remaining - 1
                    )
                    if lane_idx == 0:
                        scalars[row_in_group, 0] = cutlass.Int32(cutoff_bits)
                topk_barrier.arrive_and_wait()
                prefix = self._ordered(cutlass.Uint32(scalars[row_in_group, 0]))
            else:
                # Dense equal-half buckets retain the previously validated
                # exact radix over the original coarse candidates.
                tie_remaining = original_remaining
                # Positive normal FP16 buckets have one FP32 high byte except
                # the lower edge of an even-exponent power-of-two bucket.
                boundary_bits = cutlass.Uint32(
                    (tie_candidates[row_in_group, 0] >> 32) - 1
                )
                raw_coarse = 255 - self._coarse_ordered(boundary_bits)
                half_exponent = raw_coarse // 8
                mantissa_group = raw_coarse % 8
                prefix_proven = (
                    half_exponent > 0
                    and half_exponent < 31
                    and (mantissa_group > 0 or half_exponent % 2 == 1)
                    and (coarse_origin == -32768 or coarse_bucket > 0)
                )
                second_origin = cutlass.Int32(0)
                prefix = cutlass.Uint32(0)
                if prefix_proven:
                    exponent = half_exponent + 112
                    prefix = (
                        cutlass.Uint32(0x7F) ^ cutlass.Uint32(exponent // 2)
                    ) << cutlass.Uint32(24)
                    # Rounding can include the preceding FP32 mantissa group,
                    # leaving exactly 17 possible values for the next byte.
                    second_origin = 255 - (
                        exponent % 2 * 128 + (mantissa_group + 1) * 16 - 1
                    )
                for radix_round in cutlass.range_constexpr(4):
                    run_round = cutlass.Boolean(True)
                    if cutlass.const_expr(radix_round == 0):
                        run_round = not prefix_proven
                    narrow_second = cutlass.Boolean(False)
                    if cutlass.const_expr(radix_round == 1):
                        narrow_second = prefix_proven
                    if run_round:
                        shift = cutlass.const_expr(24 - radix_round * 8)
                        if cutlass.const_expr(radix_round == 0):
                            prefix_mask = cutlass.Uint32(0)
                        elif cutlass.const_expr(radix_round == 1):
                            prefix_mask = cutlass.Uint32(0xFF000000)
                        elif cutlass.const_expr(radix_round == 2):
                            prefix_mask = cutlass.Uint32(0xFFFF0000)
                        else:
                            prefix_mask = cutlass.Uint32(0xFFFFFF00)
                        if narrow_second:
                            if thread_in_row < 32:
                                exact_histograms[row_in_group, thread_in_row] = (
                                    cutlass.Int32(0)
                                )
                        else:
                            for item in cutlass.range_constexpr(
                                self.exact_items_per_thread
                            ):
                                histogram_index = (
                                    thread_in_row * self.exact_items_per_thread + item
                                )
                                exact_histograms[row_in_group, histogram_index] = (
                                    cutlass.Int32(0)
                                )
                        topk_barrier.arrive_and_wait()

                        tie_index = thread_in_row
                        while tie_index < tie_count:
                            packed = tie_candidates[row_in_group, tie_index]
                            bits = cutlass.Uint32((packed >> 32) - 1)
                            ordered = self._ordered(bits)
                            if (ordered & prefix_mask) == prefix:
                                bucket = cutlass.Int32(
                                    (ordered >> cutlass.Uint32(shift))
                                    & cutlass.Uint32(0xFF)
                                )
                                if narrow_second:
                                    bucket -= second_origin
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
                            if narrow_second:
                                count = exact_histograms[row_in_group, lane_idx]
                                inclusive = self._warp_scan(count, lane_idx)
                                before = inclusive - count
                                if (
                                    before < tie_remaining
                                    and inclusive >= tie_remaining
                                ):
                                    scalars[row_in_group, 0] = lane_idx + second_origin
                                    scalars[row_in_group, 1] = tie_remaining - before
                            else:
                                local_total = cutlass.Int32(0)
                                counts = cute.make_rmem_tensor((8,), cutlass.Int32)
                                for item in cutlass.range_constexpr(8):
                                    count = exact_histograms[
                                        row_in_group, lane_idx * 8 + item
                                    ]
                                    counts[item] = count
                                    local_total += count
                                inclusive = self._warp_scan(local_total, lane_idx)
                                lane_base = inclusive - local_total
                                local_prefix = cutlass.Int32(0)
                                for item in cutlass.range_constexpr(8):
                                    count = counts[item]
                                    before = lane_base + local_prefix
                                    after = before + count
                                    if (
                                        before < tie_remaining
                                        and after >= tie_remaining
                                    ):
                                        scalars[row_in_group, 0] = lane_idx * 8 + item
                                        scalars[row_in_group, 1] = (
                                            tie_remaining - before
                                        )
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
            prefix = self._select_tile_full(
                row_in_group,
                score_stage,
                tile_base,
                tile_valid,
                running_count,
                running_values,
                score_buffers,
                score_offsets,
                fallback_histograms,
                scalars,
                compacted_row,
                lane_idx,
                warp_in_row,
                thread_in_row,
                topk_barrier,
            )

        # Encode score bits + 1; zero means no complete running set exists.
        # Once 512 entries meet a cutoff, equal future scores are dispensable:
        # the contract accepts any 512 indices at the exact score boundary.
        # A stale cutoff remains conservative and only keeps extra candidates.
        if thread_in_row == 0 and candidate_count >= _TOPK:
            cute.arch.atomic_max(
                scalars.iterator + row_in_group * 7 + 6,
                cutlass.Int32(self._ordered(prefix)) + 1,
                sem="relaxed",
                scope="cta",
            )

        # Only selectors use this histogram; clear it for their next merge.
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
        load_mma: cute.TiledMma,
        smem_layout_k: cute.ComposedLayout,
        smem_layout_q: cute.ComposedLayout,
        k_tma_atom: cute.CopyAtom,
        k_tma_tensor: cute.Tensor,
        k_page_layout: cute.ComposedLayout,
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
                load_mma,
                smem_layout_k,
                smem_layout_q,
                k_tma_atom,
                k_tma_tensor,
                k_page_layout,
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
        load_mma: cute.TiledMma,
        smem_layout_k: cute.ComposedLayout,
        smem_layout_q: cute.ComposedLayout,
        k_tma_atom: cute.CopyAtom,
        k_tma_tensor: cute.Tensor,
        k_page_layout: cute.ComposedLayout,
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

        smem = utils.SmemAllocator()
        tmem_ptr_smem = smem.allocate_array(cutlass.Int32)
        k_pipeline_ptr = smem.allocate_array(cutlass.Int64, _K_STAGES * 2)
        acc_pipeline_ptr = smem.allocate_array(cutlass.Int64, _ACC_STAGES * 2)
        score_pipeline_ptr = smem.allocate_array(cutlass.Int64, _SCORE_STAGES * 4)
        sK = smem.allocate_tensor(
            cutlass.BFloat16, smem_layout_k.outer, 1024, smem_layout_k.inner
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
        score_offsets = smem.allocate_tensor(
            cutlass.Uint16,
            cute.make_layout(
                (_SCORE_STAGES, self.query_group, self.stream_tile),
                stride=(self.query_group * self.stream_tile, self.stream_tile, 1),
            ),
            128,
        )
        score_counts = smem.allocate_tensor(
            cutlass.Int32,
            cute.make_layout(
                (_SCORE_STAGES, self.query_group), stride=(self.query_group, 1)
            ),
            16,
        )
        score_last = smem.allocate_tensor(
            cutlass.Int32,
            cute.make_layout((2, _SCORE_STAGES), stride=(_SCORE_STAGES, 1)),
            16,
        )
        coarse_histograms = smem.allocate_tensor(
            cutlass.Int32,
            cute.make_layout((self.query_group, 512), stride=(512, 1)),
            128,
        )
        # Dedicated compaction scratch is disjoint from the live K stage.
        running_scratch = smem.allocate_tensor(
            cutlass.Int64,
            cute.make_layout((self.query_group, _TOPK), stride=(_TOPK, 1)),
            128,
        )
        scalars = smem.allocate_tensor(
            cutlass.Int32,
            cute.make_layout((self.query_group, 7), stride=(7, 1)),
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
        fallback_histograms = cute.make_tensor(
            coarse_histograms.iterator,
            cute.make_layout((self.query_group, 256), stride=(512, 1)),
        )
        if warp_idx == self.mma_warp:
            cute.arch.alloc_tmem(self.mma_n * _ACC_STAGES, tmem_ptr_smem)

        k_producer, k_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=_K_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            tx_count=32768,
            barrier_storage=k_pipeline_ptr,
            defer_sync=True,
        ).make_participants()
        acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=_ACC_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 256),
            barrier_storage=acc_pipeline_ptr,
            defer_sync=True,
        ).make_participants()
        # Each producer warpgroup and its four query rows share one queue.
        # Copy and MMA warps do not participate in either score queue.
        score_half = cutlass.Int32(0)
        if warp_idx < _K_COPY_FIRST_WARP:
            score_half = warpgroup_idx
        elif warp_idx >= self.topk_first_warp:
            score_half = (warp_idx - self.topk_first_warp) // (
                (_QUERY_GROUP // 2) * self.selector_warps_per_row
            )
        score_producer, score_consumer = pipeline.PipelineAsync.create(
            num_stages=_SCORE_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                (self.query_group // 2) * self.selector_threads_per_row,
            ),
            barrier_storage=score_pipeline_ptr + score_half * (_SCORE_STAGES * 2),
            defer_sync=True,
        ).make_participants()

        # PipelineAsync initializes only the half selected by warp 0.
        # Initialize the other half before the common fence and CTA sync.
        if warp_idx == 0:
            with cute.arch.elect_one():
                for stage in cutlass.range_constexpr(_SCORE_STAGES):
                    cute.arch.mbarrier_init(
                        score_pipeline_ptr + _SCORE_STAGES * 2 + stage, 128
                    )
                    cute.arch.mbarrier_init(
                        score_pipeline_ptr + _SCORE_STAGES * 3 + stage,
                        (self.query_group // 2) * self.selector_threads_per_row,
                    )

        cute.arch.mbarrier_init_fence()

        # Producers may release us before publishing Q or the row frontiers.
        # Wait before either read, including the asynchronous Q copy.
        if cutlass.const_expr(self.enable_pdl and self.group_layout != "ragged"):
            cute.arch.griddepcontrol_wait()

        if (
            warp_idx >= _K_COPY_FIRST_WARP
            and warp_idx < _K_COPY_FIRST_WARP + _K_COPY_WARPS
        ):
            q_base_addr = cute.recast_ptr(sQ.iterator, dtype=cutlass.BFloat16).toint()
            for iteration in cutlass.range_constexpr(self.query_copy_iterations):
                vector = (
                    tidx - _K_COPY_FIRST_WARP * 32 + iteration * (_K_COPY_WARPS * 32)
                )
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
        if warp_idx >= self.topk_first_warp and warp_idx < _SELECTOR_END_WARP:
            consumer_warp = warp_idx - self.topk_first_warp
            row_in_group = consumer_warp // self.selector_warps_per_row
            warp_in_row = consumer_warp - row_in_group * self.selector_warps_per_row
            thread_in_row = warp_in_row * 32 + lane_idx
            if row_in_group < self.query_group:
                for item in cutlass.range_constexpr(self.histogram_items_per_thread):
                    coarse_histograms[
                        row_in_group,
                        thread_in_row * self.histogram_items_per_thread + item,
                    ] = cutlass.Int32(0)
                if thread_in_row == 0:
                    for item in cutlass.range_constexpr(7):
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
            # Selectors published these clamped frontiers before the CTA barrier.
            count = row_completes[query_row]
            if count > group_complete:  # noqa: PLR1730
                group_complete = count
        if group_complete > num_blocks:
            group_complete = cutlass.Int32(num_blocks)
        num_subtiles = cute.ceil_div(group_complete, _MMA_M)

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

        if warp_idx == _K_COPY_FIRST_WARP:
            for tile_idx in cutlass.range(num_subtiles):
                k_handle = k_producer.acquire_and_advance()
                # Two physical pages, each split into the two 64-feature bands
                # of the existing MMA swizzle. All four transactions complete
                # the stage barrier; the next slot can be issued independently.
                for row_half in cutlass.range_constexpr(2):
                    logical_start = tile_idx * _MMA_M + row_half * 64
                    physical_page = cutlass.Int32(0)
                    if logical_start < group_complete:
                        physical_page = page_table[request, logical_start // 64]
                    for dim_half in cutlass.range_constexpr(2):
                        address = (
                            k_base_addr
                            + k_handle.index * 32768
                            + row_half * 8192
                            + dim_half * 16384
                        )
                        pointer = cute.make_ptr(
                            cutlass.BFloat16,
                            address,
                            cute.AddressSpace.smem,
                            assumed_align=128,
                        )
                        destination = cute.make_tensor(pointer, k_page_layout)
                        source = cute.local_tile(
                            k_tma_tensor, (64, 64), (physical_page, dim_half)
                        )
                        tS, tG = cpasync.tma_partition(
                            k_tma_atom,
                            0,
                            cute.make_layout(1),
                            cute.group_modes(destination, 0, 2),
                            cute.group_modes(source, 0, 2),
                        )
                        cute.copy(k_tma_atom, tG, tS, tma_bar_ptr=k_handle.barrier)

        elif warp_idx == self.mma_warp:
            for tile_idx in cutlass.range(num_subtiles):
                k_handle = k_consumer.wait_and_advance()
                cute.arch.fence_view_async_shared()
                acc_handle = acc_producer.acquire_and_advance()
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for mma_k in cutlass.range_constexpr(cute.size(k_fragment, mode=[2])):
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

        elif warpgroup_idx < 2:
            query_half = warpgroup_idx
            load_slice = load_mma.get_slice(0)
            load_fragment = load_slice.make_fragment_C(
                load_mma.partition_shape_C((_MMA_M, self.mma_n // 2, _ACC_STAGES))
            )
            half_layout = cute.make_layout(
                load_fragment.layout.shape,
                stride=(
                    load_fragment.layout.stride[0],
                    load_fragment.layout.stride[1],
                    load_fragment.layout.stride[2],
                    self.mma_n,
                ),
            )
            half_acc = cute.make_tensor(
                tAcc.iterator + query_half * (self.mma_n // 2), half_layout
            )
            tmem_atom = cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tcgen05.Repetition(self.mma_n // 2)), cutlass.Float32
            )
            tmem_copy = tcgen05.make_tmem_copy(tmem_atom, half_acc[None, None, None, 0])
            thread_copy = tmem_copy.get_slice(warpgroup_tidx)
            tSrc = thread_copy.partition_S(half_acc)
            identity = cute.make_identity_tensor((_MMA_M, self.mma_n // 2))
            tCoord = thread_copy.partition_D(load_slice.partition_C(identity))
            register_scores = cute.make_rmem_tensor(tCoord.shape, cutlass.Float32)
            # Frontiers are fixed for this invocation after the common CTA
            # initialization; histogram/count atomics do not change them.
            producer_frontiers = cute.make_rmem_tensor(
                (_QUERY_GROUP // 2,), cutlass.Int32
            )
            for query_row in cutlass.range_constexpr(_QUERY_GROUP // 2):
                producer_frontiers[query_row] = row_completes[
                    query_half * 4 + query_row
                ]
            subtile = cutlass.Int32(0)
            while subtile < num_subtiles:
                score_handle = score_producer.acquire_and_advance()
                # Compact offsets are local to a 64K window. A score stage
                # cannot span windows, even when cutoff filtering keeps few.
                stage_base = (subtile * _MMA_M) & cutlass.Int32(-65536)
                cutoff_bits = cute.make_rmem_tensor((_QUERY_GROUP // 2,), cutlass.Int32)
                for query_row in cutlass.range_constexpr(_QUERY_GROUP // 2):
                    cutoff = cutlass.Int32(0)
                    if lane_idx == 0:
                        cutoff = cute.arch.atomic_add(
                            scalars.iterator + (query_half * 4 + query_row) * 7 + 6,
                            cutlass.Int32(0),
                            sem="relaxed",
                            scope="cta",
                        )
                    cutoff_bits[query_row] = cute.arch.shuffle_sync(cutoff, 0)
                    if warpgroup_tidx == 0:
                        score_counts[score_handle.index, query_half * 4 + query_row] = (
                            cutlass.Int32(0)
                        )
                pipeline.NamedBarrier(
                    barrier_id=11 + query_half, num_threads=128
                ).arrive_and_wait()
                stage_subtiles = cutlass.Int32(0)
                flush = cutlass.Boolean(False)
                while subtile < num_subtiles and not flush:
                    acc_handle = acc_consumer.wait_and_advance()
                    cute.copy(
                        tmem_copy,
                        tSrc[None, None, None, None, acc_handle.index],
                        register_scores,
                    )
                    cute.arch.fence_view_async_tmem_load()
                    row_scores = cute.make_rmem_tensor(
                        (_QUERY_GROUP // 2,), cutlass.Float32
                    )
                    for query_row in cutlass.range_constexpr(_QUERY_GROUP // 2):
                        row_scores[query_row] = cutlass.Float32(0.0)
                    for value_idx in cutlass.range_constexpr(
                        cute.size(register_scores)
                    ):
                        coord = tCoord[value_idx]
                        value = register_scores[value_idx]
                        if value > 0.0:
                            for query_row in cutlass.range_constexpr(_QUERY_GROUP // 2):
                                if (
                                    coord[1] >= query_row * _HEADS
                                    and coord[1] < (query_row + 1) * _HEADS
                                ):
                                    row_scores[query_row] += value
                    key_row = tCoord[0][0]
                    for query_row in cutlass.range_constexpr(_QUERY_GROUP // 2):
                        score = row_scores[query_row]
                        logical_offset = subtile * _MMA_M + key_row
                        logical_block = logical_offset
                        keep = (
                            logical_block < producer_frontiers[query_row]
                            and cutlass.Int32(self._float_bits(score)) + 1
                            > cutoff_bits[query_row]
                        )
                        mask = cute.arch.vote_ballot_sync(keep)
                        count = cutlass.Int32(cute.arch.popc(cutlass.Int32(mask)))
                        compact_base = cutlass.Int32(0)
                        if lane_idx == 0:
                            compact_base = cute.arch.atomic_add(
                                score_counts.iterator
                                + score_handle.index * self.query_group
                                + query_half * 4
                                + query_row,
                                count,
                                sem="relaxed",
                                scope="cta",
                            )
                        compact_base = cute.arch.shuffle_sync(compact_base, 0)
                        lower_mask = cutlass.Uint32(0xFFFFFFFF) >> cutlass.Uint32(
                            31 - lane_idx
                        )
                        lower_mask >>= cutlass.Uint32(1)
                        rank = cutlass.Int32(
                            cute.arch.popc(cutlass.Int32(mask & lower_mask))
                        )
                        if keep:
                            position = compact_base + rank
                            score_buffers[
                                score_handle.index, query_half * 4 + query_row, position
                            ] = score
                            score_offsets[
                                score_handle.index, query_half * 4 + query_row, position
                            ] = cutlass.Uint16(logical_block)
                    acc_handle.release()
                    subtile += 1
                    stage_subtiles += 1
                    # Reserve space for a full 128-key append before advancing.
                    # Flushing above stream_tile - 128 keeps both buffers bounded.
                    pipeline.NamedBarrier(
                        barrier_id=11 + query_half, num_threads=128
                    ).arrive_and_wait()
                    max_count = cutlass.Int32(0)
                    for query_row in cutlass.range_constexpr(_QUERY_GROUP // 2):
                        count = score_counts[
                            score_handle.index, query_half * 4 + query_row
                        ]
                        if count > max_count:
                            max_count = count
                    # All warps must finish this snapshot before another append
                    # can change the counts and disagree on when to flush.
                    pipeline.NamedBarrier(
                        barrier_id=11 + query_half, num_threads=128
                    ).arrive_and_wait()
                    flush = (
                        max_count > self.stream_tile - _MMA_M
                        or subtile % (65536 // _MMA_M) == 0
                    )
                if warpgroup_tidx == 0:
                    # Low bit marks the last stage; remaining bits hold its base.
                    score_last[score_half, score_handle.index] = (
                        stage_base | cutlass.Int32(subtile >= num_subtiles)
                    )
                score_handle.commit()
            score_producer.tail()

        elif warp_idx >= self.topk_first_warp and warp_idx < _SELECTOR_END_WARP:
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
                running_count = cutlass.Int32(0)
                has_more = group_complete > 0
                while has_more:
                    score_handle = score_consumer.wait_and_advance()
                    exact_histograms = cute.make_tensor(
                        cute.recast_ptr(
                            score_buffers[score_handle.index, None, None].iterator,
                            dtype=cutlass.Int32,
                        ),
                        cute.make_layout(
                            (self.query_group, 256), stride=(self.stream_tile, 1)
                        ),
                    )
                    stage_info = score_last[score_half, score_handle.index]
                    tile_base = stage_info & cutlass.Int32(-2)
                    is_last = (stage_info & cutlass.Int32(1)) != 0
                    tile_valid = score_counts[score_handle.index, row_in_group]

                    do_select = complete > _TOPK and tile_valid > 0
                    if do_select:
                        compacted_row = running_scratch[row_in_group, None]
                        self._select_tile(
                            row_in_group,
                            score_handle.index,
                            tile_base,
                            tile_valid,
                            running_count,
                            running_values,
                            score_buffers,
                            score_offsets,
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
                            not is_last,
                        )
                        next_count = running_count + tile_valid
                        if next_count > _TOPK:
                            next_count = cutlass.Int32(_TOPK)
                        for item in cutlass.range_constexpr(
                            _TOPK // self.selector_threads_per_row
                        ):
                            index = thread_in_row + item * self.selector_threads_per_row
                            value = cutlass.Int64(0)
                            if index < next_count:
                                value = compacted_row[index]
                            running_values[item] = value
                        running_count = next_count
                    has_more = not is_last
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

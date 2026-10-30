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

"""Double-buffered scoring with integer coarse/fine bins and bounded carry.

Keep at most 528 full-width candidates when the complete coarse boundary bucket
fits. The initial coarse window uses six mantissa bits; subsequent windows use
eight, tightening surplus retention after a valid lower bound is known. Fine
keys follow the next eight ordered FP32 bits without FP16 rounding;
clipped/dense boundaries retain exact FP32 radix fallback. The actual retained
minimum remains a conservative producer cutoff. Exact FP32
refinement runs on overflow or finalization, including an empty final stage.
The 16 carry entries beyond top-512 reuse dead coarse-histogram storage, which
is cleared only after both selector warps have loaded the retained candidates.
"""

from __future__ import annotations

from typing import NamedTuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import cute, pipeline, utils
from cutlass._mlir.dialects import llvm as _llvm
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cutlass_dsl import dsl_user_op
from tokenspeed_kernel.thirdparty.cute_dsl._qsa_stream_topk.prefill import (
    PrefillTopK,
)
from tokenspeed_kernel.thirdparty.cute_dsl._qsa_stream_topk.score import (
    _paired_head_sum,
)

_QUERY_GROUP = 8
_HEADS = 4
_HEAD_DIM = 128
_MMA_M = 128
_K_STAGES = 2
_ACC_STAGES = 2
_STREAM_TILE = 1120
_FLUSH_COUNT_BIT = 1 << (_STREAM_TILE - _MMA_M).bit_length()
_FLUSH_COUNT_BIAS = _FLUSH_COUNT_BIT - (_STREAM_TILE - _MMA_M + 1)
_FLUSH_PACKED_MASK = _FLUSH_COUNT_BIT | (_FLUSH_COUNT_BIT << 16)
_FLUSH_PACKED_BIAS = _FLUSH_COUNT_BIAS | (_FLUSH_COUNT_BIAS << 16)
_TOPK = 512
_RUNNING_CAPACITY = 528
_SCORE_STAGES = 2
_SELECTOR_WARPS_PER_ROW = 2
_SELECTOR_THREADS = _SELECTOR_WARPS_PER_ROW * 32
_EXACT_ITEMS = 256 // _SELECTOR_THREADS
_QUERY_BAND_BYTES = _QUERY_GROUP * _HEADS * 128
_K_COPY_FIRST_WARP = 8
_K_COPY_WARPS = 2
_K_COPY_THREADS = _K_COPY_WARPS * 32
_TOPK_FIRST_WARP = 10
_MMA_WARP = 9
_SELECTOR_END_WARP = _TOPK_FIRST_WARP + _QUERY_GROUP * _SELECTOR_WARPS_PER_ROW


@dsl_user_op
def _producer_any(
    value: cutlass.Int32, barrier_id: cutlass.Int32, *, loc=None, ip=None
):
    result = _llvm.inline_asm(
        cutlass.Int32.mlir_type,
        [value.ir_value(loc=loc, ip=ip), barrier_id.ir_value(loc=loc, ip=ip)],
        "{ .reg .pred input_pred, result_pred; setp.ne.u32 input_pred, $1, 0; "
        "barrier.red.or.pred result_pred, $2, 128, input_pred; "
        "selp.u32 $0, 1, 0, result_pred; }",
        "=r,r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Int32(result)


@dsl_user_op
def _load_shared_cutoff(pointer: cute.Pointer, *, loc=None, ip=None):
    # This scalar is concurrently published by CTA-scoped atomic_max.
    result = _llvm.inline_asm(
        cutlass.Int32.mlir_type,
        [pointer.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)],
        "{ .reg .u32 address; cvt.u32.u64 address, $1; "
        "ld.relaxed.cta.shared.u32 $0, [address]; }",
        "=r,l,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Int32(result)


class _SharedStorage(NamedTuple):
    """One shared allocation reused after all readers finish each group."""

    tmem_ptr_smem: cute.Pointer
    k_pipeline_ptr: cute.Pointer
    acc_pipeline_ptr: cute.Pointer
    score_pipeline_ptr: cute.Pointer
    keys: cute.Tensor
    queries: cute.Tensor
    score_buffers: cute.Tensor
    score_offsets: cute.Tensor
    score_counts: cute.Tensor
    score_last: cute.Tensor
    coarse_histograms: cute.Tensor
    running_scratch: cute.Tensor
    scalars: cute.Tensor
    row_completes: cute.Tensor


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
        query_vectors = _QUERY_GROUP * _HEADS * (_HEAD_DIM // 8)
        self.query_copy_iterations = query_vectors // _K_COPY_THREADS
        self.mma_warp = _MMA_WARP
        self.topk_first_warp = _TOPK_FIRST_WARP
        self.threads = _SELECTOR_END_WARP * 32

    @cute.jit
    def _reserve_score_pair(
        self,
        counter: cute.Pointer,
        first_count: cutlass.Int32,
        second_count: cutlass.Int32,
        lane: cutlass.Int32,
    ):
        # Each row holds at most 1120 scores, so 16-bit fields cannot carry.
        increment = cutlass.Uint32(first_count) | (
            cutlass.Uint32(
                second_count,
            )
            << 16
        )
        previous = cutlass.Uint32(0)
        if lane == 0:
            previous = cute.arch.atomic_add(
                counter, increment, sem="relaxed", scope="cta"
            )
        previous = cute.arch.shuffle_sync(previous, 0)
        return previous

    @cute.jit
    def _score_pair_overflow(
        self,
        previous: cutlass.Uint32,
        first_count: cutlass.Int32,
        second_count: cutlass.Int32,
    ):
        # Counts<=1120 stay below the next threshold bit after adding31.
        # Both16-bit fields therefore detect >992 without a carry between rows.
        increment = cutlass.Uint32(first_count) | (
            cutlass.Uint32(
                second_count,
            )
            << 16
        )
        end = previous + increment + cutlass.Uint32(_FLUSH_PACKED_BIAS)
        return end & cutlass.Uint32(_FLUSH_PACKED_MASK)

    @cute.jit
    def _score_above_cutoff(
        self,
        score: cutlass.Float32,
        encoded_cutoff: cutlass.Int32,
    ):
        # Scores are nonnegative, including +inf. Comparing their bits directly
        # is equivalent to bits+1>encoded_cutoff and also accepts an unset zero.
        return cutlass.Int32(self._float_bits(score)) >= encoded_cutoff

    @cute.jit
    def _group_count(self, rows: cutlass.Int32, width: cutlass.Int32):
        if cutlass.const_expr(self.group_layout == "aligned"):
            count = rows // _QUERY_GROUP
        elif cutlass.const_expr(self.group_layout == "uniform"):
            count = (rows // width) * cute.ceil_div(width, _QUERY_GROUP)
        else:
            count = cute.ceil_div(rows, _QUERY_GROUP)
        return count

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
            for item in cutlass.range_constexpr(
                cute.ceil_div(_RUNNING_CAPACITY, self.selector_threads_per_row)
            ):
                if candidate // self.selector_threads_per_row == item:
                    packed = running_values[item]
            bits = cutlass.Uint32((packed >> 32) - 1)
            block_id = cutlass.Int32(packed & cutlass.Int64(0xFFFFFFFF))
        else:
            offset = candidate - running_count
            bits = self._float_bits(
                score_buffers[score_stage, row_in_group, offset],
            )
            block_id = tile_base + cutlass.Int32(
                score_offsets[score_stage, row_in_group, offset]
            )
        return bits, block_id

    # A fixed group lets the compact scan read its running set directly
    # instead of searching the per-thread registers as the radix path does.
    @cute.jit
    def _union_candidate_active(
        self,
        group: cutlass.Constexpr[int],
        candidate: cutlass.Int32,
        candidate_count: cutlass.Int32,
    ):
        # Every selection union contains at least 512 real candidates. Keep
        # runtime masks beyond that guaranteed prefix,including partial tails.
        if cutlass.const_expr(group < _TOPK // self.selector_threads_per_row):
            return cutlass.Boolean(True)
        else:
            return candidate < candidate_count

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
        if cutlass.const_expr(
            group
            < cute.ceil_div(
                _RUNNING_CAPACITY,
                self.selector_threads_per_row,
            )
        ):
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
            bits = self._float_bits(
                score_buffers[score_stage, row_in_group, offset],
            )
            block_id = tile_base + cutlass.Int32(
                score_offsets[score_stage, row_in_group, offset]
            )
        return bits, block_id

    @cute.jit
    def _window_coarse(
        self,
        bits: cutlass.Uint32,
        origin: cutlass.Int32,
        coarse_shift: cutlass.Int32,
    ):
        result = (
            cutlass.Int32(
                self._ordered(bits) >> cutlass.Uint32(coarse_shift),
            )
            - origin
        )
        if result < 0:
            result = cutlass.Int32(0)
        if result > 255:
            result = cutlass.Int32(255)
        return result

    @cute.jit
    def _integer_suffix(
        self,
        bits: cutlass.Uint32,
        coarse_shift: cutlass.Int32,
    ):
        return cutlass.Int32(
            (self._ordered(bits) >> cutlass.Uint32(coarse_shift - 6))
            & cutlass.Uint32(0x3F)
        )

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
        # Rows<=512 bypass selection. The first score stage has>=513 valid
        # scores;every later union contains a complete running top-k.
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
                        (
                            ordered
                            >> cutlass.Uint32(
                                shift,
                            )
                        )
                        & cutlass.Uint32(0xFF)
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
            warp_high_count += cutlass.Int32(
                cute.arch.popc(cutlass.Int32(high_mask)),
            )
            warp_equal_count += cutlass.Int32(
                cute.arch.popc(cutlass.Int32(equal_mask)),
            )
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
        for selector_warp in cutlass.range_constexpr(
            self.selector_warps_per_row,
        ):
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
            high_offset += cutlass.Int32(
                cute.arch.popc(cutlass.Int32(high_mask)),
            )
            equal_offset += cutlass.Int32(
                cute.arch.popc(cutlass.Int32(equal_mask)),
            )
            candidate_base += self.selector_threads_per_row
        topk_barrier.arrive_and_wait()
        return prefix

    @cute.jit
    def _sort_boundary_bits(
        self,
        bits: cutlass.Uint32,
        lane: cutlass.Int32,
        stages: cutlass.Constexpr[int],
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
    def _coarse_cdf(
        self,
        row_in_group: cutlass.Int32,
        coarse_histograms: cute.Tensor,
        scalars: cute.Tensor,
        remaining: cutlass.Int32,
        lane_idx: cutlass.Int32,
        warp_in_row: cutlass.Int32,
        thread_in_row: cutlass.Int32,
        topk_barrier: pipeline.NamedBarrier,
    ):
        items_per_thread = cutlass.const_expr(
            256 // self.selector_threads_per_row,
        )
        coarse_counts = cute.make_rmem_tensor(
            (items_per_thread,),
            cutlass.Int32,
        )
        first_counts = cute.make_rmem_tensor((items_per_thread,), cutlass.Int32)
        histogram_base = thread_in_row * items_per_thread
        local_total = cutlass.Int32(0)
        first_local_total = cutlass.Int32(0)
        for item in cutlass.range_constexpr(items_per_thread):
            first = coarse_histograms[row_in_group, histogram_base + item]
            second = coarse_histograms[
                row_in_group,
                256 + histogram_base + item,
            ]
            first_counts[item] = first
            coarse_counts[item] = first + second
            local_total += first + second
            first_local_total += first
        inclusive = self._warp_scan(local_total, lane_idx)
        first_inclusive = self._warp_scan(first_local_total, lane_idx)
        if warp_in_row == 0:
            # Warp0 has cached the lower128 bins. Finish its reads before
            # reusing two dead histogram cells for the cross-warp prefixes.
            cute.arch.sync_warp()
            if lane_idx == 31:
                coarse_histograms[row_in_group, 0] = inclusive
                coarse_histograms[row_in_group, 1] = first_inclusive
        topk_barrier.arrive_and_wait()
        lane_base = inclusive - local_total
        first_lane_base = first_inclusive - first_local_total
        if warp_in_row == 1:
            lane_base += coarse_histograms[row_in_group, 0]
            first_lane_base += coarse_histograms[row_in_group, 1]
        local_prefix = cutlass.Int32(0)
        first_local_prefix = cutlass.Int32(0)
        for item in cutlass.range_constexpr(items_per_thread):
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
        # Every prefix reader finishes before tie compaction reuses this row.
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
        is_last: cutlass.Boolean,
    ):
        candidate_count = running_count + tile_valid
        coarse_origin = cutlass.Int32(0)
        coarse_shift = cutlass.Int32(17)
        cutoff_bits = cutlass.Uint32(0)
        encoded_cutoff = scalars[row_in_group, 6]
        if encoded_cutoff > 0:
            coarse_shift = cutlass.Int32(15)
            cutoff_bits = cutlass.Uint32(encoded_cutoff) - cutlass.Uint32(1)
            # Retained/new scores are at least this conservative minimum, so
            # the last bin contains only its one integer coarse key.
            coarse_origin = (
                cutlass.Int32(
                    self._ordered(cutoff_bits) >> cutlass.Uint32(coarse_shift)
                )
                - 255
            )
        else:
            # The first merge has no useful lower bound. Anchor its window at
            # the actual maximum; a clipped low-score boundary uses exact radix.
            local_max = cutlass.Uint32(0)
            for item in cutlass.range_constexpr(
                cute.ceil_div(_RUNNING_CAPACITY, self.selector_threads_per_row)
            ):
                candidate = thread_in_row + item * self.selector_threads_per_row
                if candidate < running_count:
                    bits = cutlass.Uint32((running_values[item] >> 32) - 1)
                    if bits > local_max:
                        local_max = bits
            offset = thread_in_row
            while offset < tile_valid:
                bits = self._float_bits(
                    score_buffers[score_stage, row_in_group, offset]
                )
                if bits > local_max:
                    local_max = bits
                offset += self.selector_threads_per_row
            warp_max = cute.arch.warp_redux_sync(local_max, "umax", 0xFFFFFFFF)
            if lane_idx == 0:
                scalars[row_in_group, warp_in_row] = cutlass.Int32(warp_max)
            topk_barrier.arrive_and_wait()
            maximum = cutlass.max(
                cutlass.Uint32(scalars[row_in_group, 0]),
                cutlass.Uint32(scalars[row_in_group, 1]),
            )
            coarse_origin = cutlass.Int32(
                self._ordered(maximum) >> cutlass.Uint32(coarse_shift)
            )
        # Rows<=512 bypass selection. The first score stage has>=513 valid
        # scores;every later union contains a complete running top-k.
        remaining = cutlass.Int32(_TOPK)

        # A selector-owned histogram stays independent of both score stages.
        # Its full scan runs while producers score the following tile.
        # Match the combined candidate scan's warp ownership even when the
        # retained prefix is not a multiple of 64. Each warp keeps one shard.
        scan_mask = self.selector_threads_per_row - 1
        offset = (thread_in_row - running_count) & scan_mask
        while offset < tile_valid:
            bits = self._float_bits(
                score_buffers[score_stage, row_in_group, offset],
            )
            if bits >= cutoff_bits:
                bucket = self._window_coarse(bits, coarse_origin, coarse_shift)
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
            for item in cutlass.range_constexpr(
                cute.ceil_div(_RUNNING_CAPACITY, self.selector_threads_per_row)
            ):
                running_step = item * self.selector_threads_per_row
                running_candidate = thread_in_row + running_step
                if running_candidate < running_count:
                    packed = running_values[item]
                    bits = cutlass.Uint32((packed >> 32) - 1)
                    bucket = self._window_coarse(
                        bits,
                        coarse_origin,
                        coarse_shift,
                    )
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

        self._coarse_cdf(
            row_in_group,
            coarse_histograms,
            scalars,
            remaining,
            lane_idx,
            warp_in_row,
            thread_in_row,
            topk_barrier,
        )
        coarse_bucket = scalars[row_in_group, 0]
        tie_remaining = scalars[row_in_group, 1]
        better_count = scalars[row_in_group, 2]

        tie_count = scalars[row_in_group, 3]
        next_count = cutlass.Int32(_TOPK)
        kept_count = better_count + tie_count
        deferred = not is_last and kept_count <= _RUNNING_CAPACITY
        if deferred:
            # Preserve every candidate in the boundary bucket. Since at least
            # 512 candidates survive, no lower bucket can affect the final
            # top-k.
            # The histogram is dead after its prefix metadata was published;
            # its packed tie slots hold the retained candidates beyond top-512.
            output_base = cutlass.Int32(0)
            if warp_in_row == 1:
                output_base = (
                    scalars[
                        row_in_group,
                        4,
                    ]
                    + scalars[row_in_group, 5]
                )
            written = cutlass.Int32(0)
            local_min = cutlass.Uint32(0xFFFFFFFF)
            lower_mask = cutlass.Uint32(0xFFFFFFFF) >> cutlass.Uint32(
                31 - lane_idx,
            )
            lower_mask >>= cutlass.Uint32(1)
            for group in cutlass.range_constexpr(
                cute.ceil_div(
                    _RUNNING_CAPACITY + self.stream_tile,
                    self.selector_threads_per_row,
                )
            ):
                candidate_base = (
                    warp_in_row * 32 + group * self.selector_threads_per_row
                )
                if self._union_candidate_active(
                    group,
                    candidate_base,
                    candidate_count,
                ):
                    candidate = candidate_base + lane_idx
                    active = self._union_candidate_active(
                        group, candidate, candidate_count
                    )
                    bits = cutlass.Uint32(0)
                    block_id = cutlass.Int32(-1)
                    if active:
                        bits, block_id = self._candidate_const_bits_id(
                            group,
                            candidate,
                            running_count,
                            score_stage,
                            tile_base,
                            running_values,
                            score_buffers,
                            score_offsets,
                            row_in_group,
                        )
                    keep = active and bits >= cutoff_bits
                    if keep:
                        keep = (
                            self._window_coarse(
                                bits,
                                coarse_origin,
                                coarse_shift,
                            )
                            <= coarse_bucket
                        )
                    mask = cute.arch.vote_ballot_sync(keep)
                    rank = cutlass.Int32(
                        cute.arch.popc(cutlass.Int32(mask & lower_mask))
                    )
                    if keep:
                        packed = (
                            (
                                cutlass.Int64(
                                    bits,
                                )
                                + 1
                            )
                            << 32
                        ) | cutlass.Int64(block_id)
                        position = output_base + written + rank
                        if position < _TOPK:
                            compacted_row[position] = packed
                        else:
                            tie_candidates[
                                row_in_group,
                                position - _TOPK,
                            ] = packed
                        if bits < local_min:
                            local_min = bits
                    written += cutlass.Int32(
                        cute.arch.popc(cutlass.Int32(mask)),
                    )
            # All scores are consumed before their aliased exact scratch
            # changes.
            topk_barrier.arrive_and_wait()
            warp_min = cute.arch.warp_redux_sync(local_min, "umin", 0xFFFFFFFF)
            if lane_idx == 0:
                exact_histograms[row_in_group, warp_in_row] = cutlass.Int32(
                    warp_min,
                )
            topk_barrier.arrive_and_wait()
            next_count = kept_count
            if thread_in_row == 0 and next_count >= _TOPK:
                minimum = cutlass.min(
                    cutlass.Uint32(exact_histograms[row_in_group, 0]),
                    cutlass.Uint32(exact_histograms[row_in_group, 1]),
                )
                cute.arch.atomic_max(
                    scalars.iterator + row_in_group * 7 + 6,
                    cutlass.Int32(minimum) + 1,
                    sem="relaxed",
                    scope="cta",
                )
        else:
            # A clipped boundary may span several coarse keys. With a
            # published lower bound, bin 255 is known to hold one key.
            use_compact = (
                tie_count <= 256
                and coarse_bucket > 0
                and (encoded_cutoff > 0 or coarse_bucket < 255)
            )
            prefix = cutlass.Uint32(0)
            if use_compact:
                # Collect definite winners and the coarse threshold bucket in
                # one
                # second full pass. The sharded coarse histogram supplies each
                # warp's
                # exact output base, avoiding a contended atomic allocation
                # counter.
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
                # Unroll the bounded scan to keep running-set register indices
                # static.
                for candidate_group in cutlass.range_constexpr(
                    cute.ceil_div(
                        _RUNNING_CAPACITY + self.stream_tile,
                        self.selector_threads_per_row,
                    )
                ):
                    candidate_base = (
                        warp_in_row * 32
                        + candidate_group * self.selector_threads_per_row
                    )
                    if self._union_candidate_active(
                        candidate_group, candidate_base, candidate_count
                    ):
                        candidate = candidate_base + lane_idx
                        active = self._union_candidate_active(
                            candidate_group, candidate, candidate_count
                        )
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
                            bucket = self._window_coarse(
                                bits, coarse_origin, coarse_shift
                            )
                        better = active and bucket < coarse_bucket
                        tied = active and bucket == coarse_bucket
                        better_mask = cute.arch.vote_ballot_sync(better)
                        tie_mask = cute.arch.vote_ballot_sync(tied)
                        better_batch = cutlass.Int32(
                            cute.arch.popc(cutlass.Int32(better_mask))
                        )
                        tie_batch = cutlass.Int32(
                            cute.arch.popc(cutlass.Int32(tie_mask))
                        )
                        packed = (
                            (
                                cutlass.Int64(
                                    bits,
                                )
                                + 1
                            )
                            << 32
                        ) | cutlass.Int64(block_id)
                        if better:
                            position = (
                                better_base
                                + better_offset
                                + cutlass.Int32(
                                    cute.arch.popc(
                                        cutlass.Int32(
                                            better_mask & lower_lane_mask,
                                        )
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
                                        cutlass.Int32(
                                            tie_mask & lower_lane_mask,
                                        )
                                    )
                                )
                            )
                            if position < 256:
                                tie_candidates[row_in_group, position] = packed
                        better_offset += better_batch
                        tie_offset += tie_batch
                topk_barrier.arrive_and_wait()

                original_remaining = tie_remaining
                if tie_count <= 32:
                    # Use the existing exact warp network when the complete
                    # coarse boundary fits in one warp, before fine binning.
                    if warp_in_row == 0:
                        sorted_bits = cutlass.Uint32(0)
                        if lane_idx < tie_count:
                            sorted_bits = cutlass.Uint32(
                                (
                                    tie_candidates[
                                        row_in_group,
                                        lane_idx,
                                    ]
                                    >> 32
                                )
                                - 1
                            )
                        if tie_count <= 4:
                            sorted_bits = self._sort_boundary_bits(
                                sorted_bits, lane_idx, 2
                            )
                        elif tie_count <= 8:
                            sorted_bits = self._sort_boundary_bits(
                                sorted_bits, lane_idx, 3
                            )
                        elif tie_count <= 16:
                            sorted_bits = self._sort_boundary_bits(
                                sorted_bits, lane_idx, 4
                            )
                        else:
                            sorted_bits = self._sort_boundary_bits(
                                sorted_bits, lane_idx, 5
                            )
                        cutoff = cute.arch.shuffle_sync(
                            sorted_bits, original_remaining - 1
                        )
                        if lane_idx == 0:
                            scalars[row_in_group, 0] = cutlass.Int32(cutoff)
                    topk_barrier.arrive_and_wait()
                    prefix = self._ordered(
                        cutlass.Uint32(scalars[row_in_group, 0]),
                    )
                else:
                    # Refine the next six exact FP32 bits within this one coarse
                    # key; the remaining low bits are resolved from actual
                    # scores.
                    for item in cutlass.range_constexpr(
                        64 // self.selector_threads_per_row
                    ):
                        exact_histograms[
                            row_in_group,
                            thread_in_row + item * _SELECTOR_THREADS,
                        ] = cutlass.Int32(0)
                    topk_barrier.arrive_and_wait()
                    fine_index = thread_in_row
                    while fine_index < tie_count:
                        packed = tie_candidates[row_in_group, fine_index]
                        bits = cutlass.Uint32((packed >> 32) - 1)
                        suffix = self._integer_suffix(bits, coarse_shift)
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
                        fine_counts = cute.make_rmem_tensor((2,), cutlass.Int32)
                        fine_total = cutlass.Int32(0)
                        for item in cutlass.range_constexpr(2):
                            fine_counts[item] = exact_histograms[
                                row_in_group, lane_idx * 2 + item
                            ]
                            fine_total += fine_counts[item]
                        fine_inclusive = self._warp_scan(fine_total, lane_idx)
                        fine_before = fine_inclusive - fine_total
                        for item in cutlass.range_constexpr(2):
                            fine_after = fine_before + fine_counts[item]
                            if (
                                fine_before < original_remaining
                                and fine_after >= original_remaining
                            ):
                                scalars[row_in_group, 0] = lane_idx * 2 + item
                                scalars[row_in_group, 1] = (
                                    original_remaining - fine_before
                                )
                                scalars[row_in_group, 2] = fine_counts[item]
                            fine_before = fine_after
                    topk_barrier.arrive_and_wait()
                    refined_bucket = scalars[row_in_group, 0]
                    refine_rank = scalars[row_in_group, 1]
                    refined_count = scalars[row_in_group, 2]
                    if refined_count == 1:
                        # Snapshot the bucket in every lane before its slot
                        # stores
                        # the unique candidate's FP32 score bits.
                        topk_barrier.arrive_and_wait()
                        # Exactly one matching FP32 score owns this shared
                        # result.
                        fine_index = thread_in_row
                        while fine_index < tie_count:
                            bits = cutlass.Uint32(
                                (
                                    tie_candidates[
                                        row_in_group,
                                        fine_index,
                                    ]
                                    >> 32
                                )
                                - 1
                            )
                            if (
                                self._integer_suffix(bits, coarse_shift)
                                == refined_bucket
                            ):
                                scalars[row_in_group, 0] = cutlass.Int32(bits)
                            fine_index += self.selector_threads_per_row
                        topk_barrier.arrive_and_wait()
                        prefix = self._ordered(
                            cutlass.Uint32(scalars[row_in_group, 0]),
                        )
                    elif refine_rank == 1 or refine_rank == refined_count:
                        local_min = cutlass.Uint32(0xFFFFFFFF)
                        local_max = cutlass.Uint32(0)
                        fine_index = thread_in_row
                        while fine_index < tie_count:
                            bits = cutlass.Uint32(
                                (
                                    tie_candidates[
                                        row_in_group,
                                        fine_index,
                                    ]
                                    >> 32
                                )
                                - 1
                            )
                            if (
                                self._integer_suffix(bits, coarse_shift)
                                == refined_bucket
                            ):
                                if bits < local_min:
                                    local_min = bits
                                if bits > local_max:
                                    local_max = bits
                            fine_index += self.selector_threads_per_row
                        warp_min = cute.arch.warp_redux_sync(
                            local_min, "umin", 0xFFFFFFFF
                        )
                        warp_max = cute.arch.warp_redux_sync(
                            local_max, "umax", 0xFFFFFFFF
                        )
                        result = warp_min
                        if refine_rank == 1:
                            result = warp_max
                        if lane_idx == 0:
                            exact_histograms[
                                row_in_group,
                                warp_in_row,
                            ] = cutlass.Int32(result)
                        topk_barrier.arrive_and_wait()
                        if thread_in_row == 0:
                            first = cutlass.Uint32(
                                exact_histograms[row_in_group, 0],
                            )
                            second = cutlass.Uint32(
                                exact_histograms[row_in_group, 1],
                            )
                            if refine_rank == 1:
                                if second > first:
                                    first = second
                            else:
                                if second < first:
                                    first = second
                            scalars[row_in_group, 0] = cutlass.Int32(first)
                        topk_barrier.arrive_and_wait()
                        prefix = self._ordered(
                            cutlass.Uint32(scalars[row_in_group, 0]),
                        )
                    elif refined_count <= 32:
                        if thread_in_row == 0:
                            scalars[row_in_group, 4] = cutlass.Int32(0)
                        topk_barrier.arrive_and_wait()
                        fine_index = thread_in_row
                        while fine_index < tie_count:
                            packed = tie_candidates[row_in_group, fine_index]
                            bits = cutlass.Uint32((packed >> 32) - 1)
                            if (
                                self._integer_suffix(bits, coarse_shift)
                                == refined_bucket
                            ):
                                position = cute.arch.atomic_add(
                                    scalars.iterator + row_in_group * 7 + 4,
                                    cutlass.Int32(1),
                                    sem="relaxed",
                                    scope="cta",
                                )
                                exact_histograms[row_in_group, position] = (
                                    cutlass.Int32(bits)
                                )
                            fine_index += self.selector_threads_per_row
                        topk_barrier.arrive_and_wait()
                        if warp_in_row == 0:
                            sorted_bits = cutlass.Uint32(0)
                            if lane_idx < refined_count:
                                sorted_bits = cutlass.Uint32(
                                    exact_histograms[row_in_group, lane_idx]
                                )
                            # Nonnegative FP32 bits have their numerical order.
                            # Padding
                            # with zero cannot change a rank inside
                            # refined_count.
                            if refined_count <= 8:
                                sorted_bits = self._sort_boundary_bits(
                                    sorted_bits, lane_idx, 3
                                )
                            elif refined_count <= 16:
                                sorted_bits = self._sort_boundary_bits(
                                    sorted_bits, lane_idx, 4
                                )
                            else:
                                sorted_bits = self._sort_boundary_bits(
                                    sorted_bits, lane_idx, 5
                                )
                            cutoff_bits = cute.arch.shuffle_sync(
                                sorted_bits, refine_rank - 1
                            )
                            if lane_idx == 0:
                                scalars[row_in_group, 0] = cutlass.Int32(
                                    cutoff_bits,
                                )
                        topk_barrier.arrive_and_wait()
                        prefix = self._ordered(
                            cutlass.Uint32(scalars[row_in_group, 0]),
                        )
                    else:
                        # Dense equal-integer buckets use exact radix. The
                        # former
                        # FP16 exponent shortcut is inapplicable to integer
                        # bins.
                        tie_remaining = original_remaining
                        prefix_proven = cutlass.Boolean(False)
                        second_origin = cutlass.Int32(0)
                        prefix = cutlass.Uint32(0)
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
                                        exact_histograms[
                                            row_in_group, thread_in_row
                                        ] = cutlass.Int32(0)
                                else:
                                    for item in cutlass.range_constexpr(
                                        self.exact_items_per_thread
                                    ):
                                        hist_base = thread_in_row * _EXACT_ITEMS
                                        histogram_index = hist_base + item
                                        exact_histograms[
                                            row_in_group, histogram_index
                                        ] = cutlass.Int32(0)
                                topk_barrier.arrive_and_wait()

                                tie_index = thread_in_row
                                while tie_index < tie_count:
                                    packed = tie_candidates[
                                        row_in_group,
                                        tie_index,
                                    ]
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
                                        count = exact_histograms[
                                            row_in_group,
                                            lane_idx,
                                        ]
                                        inclusive = self._warp_scan(
                                            count,
                                            lane_idx,
                                        )
                                        before = inclusive - count
                                        if (
                                            before < tie_remaining
                                            and inclusive >= tie_remaining
                                        ):
                                            scalars[row_in_group, 0] = (
                                                lane_idx + second_origin
                                            )
                                            scalars[row_in_group, 1] = (
                                                tie_remaining - before
                                            )
                                    else:
                                        local_total = cutlass.Int32(0)
                                        counts = cute.make_rmem_tensor(
                                            (8,), cutlass.Int32
                                        )
                                        for item in cutlass.range_constexpr(8):
                                            count = exact_histograms[
                                                row_in_group,
                                                lane_idx * 8 + item,
                                            ]
                                            counts[item] = count
                                            local_total += count
                                        inclusive = self._warp_scan(
                                            local_total, lane_idx
                                        )
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
                                                scalars[row_in_group, 0] = (
                                                    lane_idx * 8 + item
                                                )
                                                scalars[row_in_group, 1] = (
                                                    tie_remaining - before
                                                )
                                            local_prefix += count
                                topk_barrier.arrive_and_wait()
                                bucket = scalars[row_in_group, 0]
                                tie_remaining = scalars[row_in_group, 1]
                                prefix |= cutlass.Uint32(
                                    bucket,
                                ) << cutlass.Uint32(shift)

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
                    exact_histograms[
                        row_in_group,
                        warp_in_row,
                    ] = warp_high_count
                    exact_histograms[
                        row_in_group,
                        self.selector_warps_per_row + warp_in_row,
                    ] = warp_equal_count
                topk_barrier.arrive_and_wait()

                high_base = better_count
                equal_base = better_count
                for selector_warp in cutlass.range_constexpr(
                    self.selector_warps_per_row
                ):
                    equal_base += exact_histograms[row_in_group, selector_warp]
                    if selector_warp < warp_in_row:
                        high_base += exact_histograms[
                            row_in_group,
                            selector_warp,
                        ]
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
                        cute.arch.popc(
                            cutlass.Int32(high_mask & lower_lane_mask),
                        )
                    )
                    equal_rank = cutlass.Int32(
                        cute.arch.popc(
                            cutlass.Int32(equal_mask & lower_lane_mask),
                        )
                    )
                    if high:
                        position = high_base + high_offset + high_rank
                        if position < _TOPK:
                            compacted_row[position] = packed
                    if equal:
                        position = equal_base + equal_offset + equal_rank
                        if position < _TOPK:
                            compacted_row[position] = packed
                    high_offset += cutlass.Int32(
                        cute.arch.popc(cutlass.Int32(high_mask))
                    )
                    equal_offset += cutlass.Int32(
                        cute.arch.popc(cutlass.Int32(equal_mask))
                    )
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
            # Once 512 entries meet a cutoff, equal future scores are
            # dispensable:
            # the contract accepts any 512 indices at the exact score boundary.
            # A stale cutoff remains conservative and only keeps extra
            # candidates.
            if thread_in_row == 0 and candidate_count >= _TOPK:
                cute.arch.atomic_max(
                    scalars.iterator + row_in_group * 7 + 6,
                    cutlass.Int32(self._ordered(prefix)) + 1,
                    sem="relaxed",
                    scope="cta",
                )

        for item in cutlass.range_constexpr(
            cute.ceil_div(_RUNNING_CAPACITY, self.selector_threads_per_row)
        ):
            index = thread_in_row + item * self.selector_threads_per_row
            value = cutlass.Int64(0)
            if index < next_count:
                if cutlass.const_expr(
                    item < _TOPK // self.selector_threads_per_row,
                ):
                    value = compacted_row[index]
                else:
                    value = tie_candidates[row_in_group, index - _TOPK]
            running_values[item] = value
        if deferred:
            # Finish every overflow read before clearing the reused histogram.
            topk_barrier.arrive_and_wait()
        # Clear in coalesced four-word vectors rather than stride-eight
        # vectors, preserving complete coverage without shared bank conflicts.
        if not is_last:
            for item in cutlass.range_constexpr(
                self.histogram_items_per_thread,
            ):
                coarse_histograms[
                    row_in_group,
                    thread_in_row * _EXACT_ITEMS
                    + (item // 4) * (self.selector_threads_per_row * 4)
                    + item % 4,
                ] = cutlass.Int32(0)
            topk_barrier.arrive_and_wait()

        return next_count

    @cute.jit
    def _allocate_storage(
        self,
        smem_layout_k: cute.ComposedLayout,
        smem_layout_q: cute.ComposedLayout,
    ):
        """Allocate buffers once, outside request-group branches."""
        smem = utils.SmemAllocator()
        tmem_ptr_smem = smem.allocate_array(cutlass.Int32)
        k_pipeline_ptr = smem.allocate_array(cutlass.Int64, _K_STAGES * 2)
        acc_pipeline_ptr = smem.allocate_array(cutlass.Int64, _ACC_STAGES * 2)
        score_pipeline_ptr = smem.allocate_array(
            cutlass.Int64,
            _SCORE_STAGES * 4,
        )
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
                stride=(
                    self.query_group * self.stream_tile,
                    self.stream_tile,
                    1,
                ),
            ),
            128,
        )
        # Two row counts share a native 32-bit counter; both stay below 1120.
        score_counts = smem.allocate_tensor(
            cutlass.Uint32,
            cute.make_layout(
                (_SCORE_STAGES, self.query_group // 2),
                stride=(self.query_group // 2, 1),
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
        return _SharedStorage(
            tmem_ptr_smem,
            k_pipeline_ptr,
            acc_pipeline_ptr,
            score_pipeline_ptr,
            sK,
            sQ,
            score_buffers,
            score_offsets,
            score_counts,
            score_last,
            coarse_histograms,
            running_scratch,
            scalars,
            row_completes,
        )

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
        # Both ownership branches use one physical shared-memory allocation.
        # Their invocations are mutually exclusive within this CTA.
        storage = self._allocate_storage(smem_layout_k, smem_layout_q)
        if cutlass.const_expr(
            self.enable_pdl and self.group_layout == "ragged",
        ):
            cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(self.group_layout == "ragged"):
            row_base = block_idx * _QUERY_GROUP
            end = cutlass.min(query.shape[0], row_base + _QUERY_GROUP)
            _, first_valid = self._query_group(
                request_indices, query.shape[0], queries_per_request, row_base
            )
            # Most dense eight-row windows belong to one request. Keep their
            # common
            # invocation outside the ownership loop so the compiler sees its
            # fixed row base and complete query validity.
            if first_valid == _QUERY_GROUP:
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
                    cutlass.Int32(_QUERY_GROUP),
                    page_size,
                    storage,
                )
            else:
                while row_base < end:
                    start, valid_rows = self._query_group(
                        request_indices,
                        query.shape[0],
                        queries_per_request,
                        row_base,
                    )
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
                        storage,
                    )
                    row_base += valid_rows
                    if row_base < end:
                        # Every reader must finish before another request reuses
                        # this CTA's shared buffers and pipeline barriers.
                        cute.arch.sync_threads()
        else:
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
                    storage,
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
        storage: _SharedStorage,
    ):
        tmem_ptr_smem = storage.tmem_ptr_smem
        k_pipeline_ptr = storage.k_pipeline_ptr
        acc_pipeline_ptr = storage.acc_pipeline_ptr
        score_pipeline_ptr = storage.score_pipeline_ptr
        sK = storage.keys
        sQ = storage.queries
        score_buffers = storage.score_buffers
        score_offsets = storage.score_offsets
        score_counts = storage.score_counts
        score_last = storage.score_last
        coarse_histograms = storage.coarse_histograms
        running_scratch = storage.running_scratch
        scalars = storage.scalars
        row_completes = storage.row_completes
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
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                256,
            ),
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
        score_barriers = score_pipeline_ptr + score_half * (_SCORE_STAGES * 2)
        score_producer, score_consumer = pipeline.PipelineAsync.create(
            num_stages=_SCORE_STAGES,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                128,
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                (self.query_group // 2) * self.selector_threads_per_row,
            ),
            barrier_storage=score_barriers,
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
        if cutlass.const_expr(
            self.enable_pdl and self.group_layout != "ragged",
        ):
            cute.arch.griddepcontrol_wait()

        if (
            warp_idx >= _K_COPY_FIRST_WARP
            and warp_idx < _K_COPY_FIRST_WARP + _K_COPY_WARPS
        ):
            q_base_addr = cute.recast_ptr(
                sQ.iterator,
                dtype=cutlass.BFloat16,
            ).toint()
            for iteration in cutlass.range_constexpr(
                self.query_copy_iterations,
            ):
                copy_thread = tidx - _K_COPY_FIRST_WARP * 32
                vector = copy_thread + iteration * _K_COPY_THREADS
                q_head = vector // 16
                col = (vector - q_head * 16) * 8
                q_row = q_head // _HEADS
                head = q_head - q_row * _HEADS
                source = (
                    query.iterator
                    + cutlass.Int64(row_base + q_row) * q_stride_row
                    + head * q_stride_head
                    + col
                )
                head_offset = q_head * 128 + (col % 64) * 2
                logical_byte = head_offset + (col // 64) * _QUERY_BAND_BYTES
                logical_address = q_base_addr + logical_byte
                swizzle_bits = (logical_address >> 3) & 0x70
                swizzled_address = logical_address ^ swizzle_bits
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
            warp_in_row = consumer_warp % self.selector_warps_per_row
            thread_in_row = warp_in_row * 32 + lane_idx
            if row_in_group < self.query_group:
                for item in cutlass.range_constexpr(
                    self.histogram_items_per_thread,
                ):
                    coarse_histograms[
                        row_in_group,
                        thread_in_row * _EXACT_ITEMS
                        + (item // 4) * (self.selector_threads_per_row * 4)
                        + item % 4,
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
            # Selectors published these clamped frontiers before the CTA
            # barrier.
            count = row_completes[query_row]
            # Every valid key is already selected when the row fits top-k.
            # Its scores cannot affect output, including in a mixed group.
            if count > _TOPK and count > group_complete:
                group_complete = count
        if group_complete > num_blocks:
            group_complete = cutlass.Int32(num_blocks)
        num_subtiles = cute.ceil_div(group_complete, _MMA_M)

        thr_mma = tiled_mma.get_slice(0)
        k_fragment = thr_mma.make_fragment_A(sK)
        q_desc = thr_mma.make_fragment_B(sQ)
        acc_shape = tiled_mma.partition_shape_C(
            (_MMA_M, self.mma_n, _ACC_STAGES),
        )
        acc_fragment = thr_mma.make_fragment_C(acc_shape)
        tmem_ptr = cute.arch.retrieve_tmem_ptr(cutlass.Int32, 16, tmem_ptr_smem)
        tAcc = cute.make_tensor(
            cute.recast_ptr(tmem_ptr, dtype=cutlass.Float32),
            acc_fragment.layout,
        )
        k_base_addr = cute.recast_ptr(
            sK.iterator,
            dtype=cutlass.BFloat16,
        ).toint()

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
                        cute.copy(
                            k_tma_atom,
                            tG,
                            tS,
                            tma_bar_ptr=k_handle.barrier,
                        )

        elif warp_idx == self.mma_warp:
            for tile_idx in cutlass.range(num_subtiles):
                k_handle = k_consumer.wait_and_advance()
                # TMA writes and UMMA reads K through the same async proxy.
                # The full barrier completes TMA; the pipeline release below
                # waits for the final MMA reader before reusing the K slot.
                # Q's initialization proxy fence remains before this loop.
                acc_handle = acc_producer.acquire_and_advance()
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for mma_k in cutlass.range_constexpr(
                    cute.size(k_fragment, mode=[2]),
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

        elif warpgroup_idx < 2:
            query_half = warpgroup_idx
            load_slice = load_mma.get_slice(0)
            load_fragment = load_slice.make_fragment_C(
                load_mma.partition_shape_C(
                    (_MMA_M, self.mma_n // 2, _ACC_STAGES),
                )
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
                tcgen05.Ld32x32bOp(tcgen05.Repetition(self.mma_n // 2)),
                cutlass.Float32,
            )
            tmem_copy = tcgen05.make_tmem_copy(
                tmem_atom,
                half_acc[None, None, None, 0],
            )
            thread_copy = tmem_copy.get_slice(warpgroup_tidx)
            tSrc = thread_copy.partition_S(half_acc)
            identity = cute.make_identity_tensor((_MMA_M, self.mma_n // 2))
            tCoord = thread_copy.partition_D(load_slice.partition_C(identity))
            register_scores = cute.make_rmem_tensor(
                tCoord.shape,
                cutlass.Float32,
            )
            # Frontiers are fixed for this invocation after the common CTA
            # initialization; histogram/count atomics do not change them.
            producer_frontiers = cute.make_rmem_tensor(
                (_QUERY_GROUP // 2,), cutlass.Int32
            )
            for query_row in cutlass.range_constexpr(_QUERY_GROUP // 2):
                frontier = row_completes[query_half * 4 + query_row]
                if frontier <= _TOPK:
                    frontier = cutlass.Int32(0)
                producer_frontiers[query_row] = frontier
            subtile = cutlass.Int32(0)
            while subtile < num_subtiles:
                score_handle = score_producer.acquire_and_advance()
                # Compact offsets are local to a 64K window. A score stage
                # cannot span windows, even when cutoff filtering keeps few.
                stage_base = (subtile * _MMA_M) & cutlass.Int32(-65536)
                cutoff_bits = cute.make_rmem_tensor(
                    (_QUERY_GROUP // 2,),
                    cutlass.Int32,
                )
                for query_row in cutlass.range_constexpr(_QUERY_GROUP // 2):
                    # Cutoffs only increase and always bound a complete top-k.
                    # Each lane may safely use either concurrent version;the
                    # cutoff does not publish payloads through this load.
                    cutoff_bits[query_row] = _load_shared_cutoff(
                        scalars.iterator + (query_half * 4 + query_row) * 7 + 6
                    )
                if warpgroup_tidx == 0:
                    for pair in cutlass.range_constexpr(_QUERY_GROUP // 4):
                        score_counts[
                            score_handle.index,
                            query_half * 2 + pair,
                        ] = cutlass.Uint32(0)
                pipeline.NamedBarrier(
                    barrier_id=11 + query_half, num_threads=128
                ).arrive_and_wait()
                stage_subtiles = cutlass.Int32(0)
                flush = cutlass.Boolean(False)
                # Seven subtiles append at most 896entries per row, below
                # the 992-entry flush threshold. No cross-warp capacity check
                # is needed yet; the existing release/acquire score queue
                # publishes all writes, including a short or window-ending
                # stage. Starting at subtile8, retain the full flush protocol.
                for scan_part in cutlass.range_constexpr(2):
                    part_end = num_subtiles
                    if cutlass.const_expr(scan_part == 0):
                        part_end = cutlass.min(num_subtiles, subtile + 7)
                    while subtile < part_end and not flush:
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
                        # TMEM provides all four heads in order for each query.
                        # Pair independent queries without changing either chain
                        # of FP32 additions or the treatment of nonpositive
                        # heads.
                        for pair in cutlass.range_constexpr(_QUERY_GROUP // 4):
                            base = cutlass.const_expr(pair * 2 * _HEADS)
                            packed = _paired_head_sum(
                                register_scores[base],
                                register_scores[base + 1],
                                register_scores[base + 2],
                                register_scores[base + 3],
                                register_scores[base + 4],
                                register_scores[base + 5],
                                register_scores[base + 6],
                                register_scores[base + 7],
                            )
                            first_bits = cutlass.Uint32(packed)
                            second_bits = cutlass.Uint32(
                                packed >> cutlass.Uint64(32),
                            )
                            row_scores[pair * 2] = cutlass.Float32(
                                _llvm.bitcast(
                                    cutlass.Float32.mlir_type,
                                    first_bits.ir_value(),
                                )
                            )
                            row_scores[pair * 2 + 1] = cutlass.Float32(
                                _llvm.bitcast(
                                    cutlass.Float32.mlir_type,
                                    second_bits.ir_value(),
                                )
                            )
                        key_row = tCoord[0][0]
                        local_flush_flags = cutlass.Uint32(0)
                        logical_block = subtile * _MMA_M + key_row
                        lower_mask = (
                            cutlass.Uint32(1)
                            << cutlass.Uint32(
                                lane_idx,
                            )
                        ) - 1
                        for pair in cutlass.range_constexpr(_QUERY_GROUP // 4):
                            first_row = cutlass.const_expr(pair * 2)
                            second_row = cutlass.const_expr(pair * 2 + 1)
                            first_keep = logical_block < producer_frontiers[
                                first_row
                            ] and self._score_above_cutoff(
                                row_scores[first_row], cutoff_bits[first_row]
                            )
                            second_keep = logical_block < producer_frontiers[
                                second_row
                            ] and self._score_above_cutoff(
                                row_scores[second_row], cutoff_bits[second_row]
                            )
                            first_mask = cute.arch.vote_ballot_sync(first_keep)
                            second_mask = cute.arch.vote_ballot_sync(
                                second_keep,
                            )
                            first_count = cutlass.Int32(
                                cute.arch.popc(first_mask),
                            )
                            second_count = cutlass.Int32(
                                cute.arch.popc(second_mask),
                            )
                            previous = self._reserve_score_pair(
                                score_counts.iterator
                                + score_handle.index * (self.query_group // 2)
                                + query_half * 2
                                + pair,
                                first_count,
                                second_count,
                                lane_idx,
                            )
                            local_flush_flags |= self._score_pair_overflow(
                                previous,
                                first_count,
                                second_count,
                            )
                            for member in cutlass.range_constexpr(2):
                                query_row = cutlass.const_expr(
                                    pair * 2 + member,
                                )
                                if cutlass.const_expr(member == 0):
                                    keep = first_keep
                                    mask = first_mask
                                    count = first_count
                                else:
                                    keep = second_keep
                                    mask = second_mask
                                    count = second_count
                                rank = cutlass.Int32(
                                    cute.arch.popc(mask & lower_mask),
                                )
                                if keep:
                                    compact_base = cutlass.Int32(
                                        (
                                            previous
                                            >> cutlass.Uint32(
                                                member * 16,
                                            )
                                        )
                                        & cutlass.Uint32(0xFFFF)
                                    )
                                    position = compact_base + rank
                                    score_buffers[
                                        score_handle.index,
                                        query_half * 4 + query_row,
                                        position,
                                    ] = row_scores[query_row]
                                    score_offsets[
                                        score_handle.index,
                                        query_half * 4 + query_row,
                                        position,
                                    ] = cutlass.Uint16(logical_block)
                        acc_handle.release()
                        subtile += 1
                        stage_subtiles += 1
                        # A row's final count is the largest reservation end.
                        # All writes complete before the producer group observes
                        # the OR result. IDs 13/14 are separate from reset
                        # syncs.
                        group_flush = cutlass.Int32(0)
                        if cutlass.const_expr(scan_part == 1):
                            group_flush = _producer_any(
                                cutlass.Int32(local_flush_flags),
                                cutlass.Int32(13) + query_half,
                            )
                        # Both parts must end at the same 64K logical-ID
                        # boundary.
                        window_ends = subtile % (65536 // _MMA_M) == 0
                        flush = group_flush != 0 or window_ends
                if warpgroup_tidx == 0:
                    # Low bit marks the last stage; remaining bits hold its
                    # base.
                    score_last[score_half, score_handle.index] = (
                        stage_base | cutlass.Int32(subtile >= num_subtiles)
                    )
                score_handle.commit()
            score_producer.tail()

        elif warp_idx >= self.topk_first_warp and warp_idx < _SELECTOR_END_WARP:
            consumer_warp = warp_idx - self.topk_first_warp
            row_in_group = consumer_warp // self.selector_warps_per_row
            warp_in_row = consumer_warp % self.selector_warps_per_row
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
                    (
                        cute.ceil_div(
                            _RUNNING_CAPACITY,
                            self.selector_threads_per_row,
                        ),
                    ),
                    cutlass.Int64,
                )
                # Nine full-width slots per thread cover the 528-entry carry;
                # the last 48 physical slots remain inactive.
                for item in cutlass.range_constexpr(
                    cute.ceil_div(
                        _RUNNING_CAPACITY,
                        self.selector_threads_per_row,
                    )
                ):
                    running_values[item] = cutlass.Int64(0)
                running_count = cutlass.Int32(0)
                has_more = group_complete > 0
                while has_more:
                    score_handle = score_consumer.wait_and_advance()
                    exact_histograms = cute.make_tensor(
                        cute.recast_ptr(
                            score_buffers[
                                score_handle.index,
                                None,
                                None,
                            ].iterator,
                            dtype=cutlass.Int32,
                        ),
                        cute.make_layout(
                            (self.query_group, 256),
                            stride=(self.stream_tile, 1),
                        ),
                    )
                    stage_info = score_last[score_half, score_handle.index]
                    tile_base = stage_info & cutlass.Int32(-2)
                    is_last = (stage_info & cutlass.Int32(1)) != 0
                    packed_count = score_counts[
                        score_handle.index,
                        row_in_group // 2,
                    ]
                    tile_valid = cutlass.Int32(
                        (
                            packed_count
                            >> cutlass.Uint32(
                                (row_in_group % 2) * 16,
                            )
                        )
                        & cutlass.Uint32(0xFFFF)
                    )

                    do_select = complete > _TOPK and (
                        tile_valid > 0 or (is_last and running_count > _TOPK)
                    )
                    if do_select:
                        compacted_row = running_scratch[row_in_group, None]
                        running_count = self._select_tile(
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
                            is_last,
                        )
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
                    if cutlass.const_expr(self.group_layout == "aligned"):
                        selected_row[index] = value
                    elif row_in_group < valid_rows:
                        selected_row[index] = value

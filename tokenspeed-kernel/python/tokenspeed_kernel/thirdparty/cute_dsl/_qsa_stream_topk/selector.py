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

"""Exact FP32 radix selection for packed and native score tiles."""

from __future__ import annotations

import cutlass
from cuda.bindings import driver as cuda
from cutlass import cute, utils
from cutlass._mlir.dialects import llvm as _llvm
from tokenspeed_kernel.thirdparty.cute_dsl._qsa_stream_topk.common import (
    SelectorPrimitives,
    _load_global_u32,
    _match_any,
)

_TOPK = 512


class SplitSelector(SelectorPrimitives):
    def __init__(self, threads: int, entries: int, enable_pdl: bool):
        self.enable_pdl = enable_pdl
        self.threads = threads
        self.warps = threads // 32
        self.entries = entries
        self.cache_encoded = entries <= 8

    @cute.jit
    def _entry(
        self,
        values: cute.Tensor,
        item: cutlass.Constexpr[int],
        packed: cutlass.Constexpr[bool],
        count: cutlass.Int32,
        base: cutlass.Int32,
        tidx: cutlass.Int32,
    ):
        index = tidx + item * self.threads
        if cutlass.const_expr(packed):
            # Read each pass from L2 rather than retaining every candidate in
            # registers across all refinement rounds.
            if cutlass.const_expr(self.cache_encoded):
                value = values[item]
            else:
                value = _load_global_u32(values.iterator + item * self.threads * 2)
            bits = value - cutlass.Uint32(1)
            valid = value != 0
        else:
            bits = values[item]
            valid = index < count
        return bits, valid, base + index

    @cute.jit
    def _clear_histogram(self, histogram: cute.Tensor, tidx: cutlass.Int32):
        for item in cutlass.range_constexpr(self.warps * 256 // self.threads):
            offset = tidx + item * self.threads
            histogram[offset // 256, offset % 256] = cutlass.Int32(0)
        cute.arch.sync_threads()

    @cute.jit
    def _block_scan(
        self,
        value: cutlass.Int32,
        warp_sums: cute.Tensor,
        tidx: cutlass.Int32,
        lane: cutlass.Int32,
    ):
        warp = tidx // 32
        inclusive = self._warp_scan(value, lane)
        if lane == 31:
            warp_sums[warp] = inclusive
        cute.arch.sync_threads()
        if warp == 0:
            count = cutlass.Int32(0)
            if lane < self.warps:
                count = warp_sums[lane]
            prefix = self._warp_scan(count, lane)
            if lane < self.warps:
                warp_sums[lane] = prefix
        cute.arch.sync_threads()
        before = cutlass.Int32(0)
        if warp > 0:
            before = warp_sums[warp - 1]
        total = warp_sums[self.warps - 1]
        cute.arch.sync_threads()
        return before + inclusive, total

    @cute.jit
    def _find_bucket(
        self,
        histogram: cute.Tensor,
        scalars: cute.Tensor,
        warp_sums: cute.Tensor,
        rank: cutlass.Int32,
        tidx: cutlass.Int32,
        lane: cutlass.Int32,
    ):
        count = cutlass.Int32(0)
        if tidx < 256:
            for shard in cutlass.range_constexpr(self.warps):
                count += histogram[shard, tidx]
        inclusive, _ = self._block_scan(count, warp_sums, tidx, lane)
        if tidx < 256 and inclusive >= rank and inclusive - count < rank:
            scalars[0] = tidx
            scalars[1] = rank - (inclusive - count)
        cute.arch.sync_threads()
        return scalars[0], scalars[1]

    @cute.jit
    def _select(
        self,
        values: cute.Tensor,
        packed: cutlass.Constexpr[bool],
        count: cutlass.Int32,
        base: cutlass.Int32,
        histogram: cute.Tensor,
        scalars: cute.Tensor,
        warp_sums: cute.Tensor,
        output: cute.Tensor,
        tidx: cutlass.Int32,
        lane: cutlass.Int32,
    ):
        for item in cutlass.range_constexpr(cute.ceil_div(_TOPK, self.threads)):
            if tidx + item * self.threads < _TOPK:
                output[tidx + item * self.threads] = cutlass.Int64(0)
        cute.arch.sync_threads()
        if count > 0:
            if cutlass.const_expr(not packed):
                if count <= _TOPK:
                    for item in cutlass.range_constexpr(
                        cute.ceil_div(_TOPK, self.threads)
                    ):
                        index = tidx + item * self.threads
                        if index < count:
                            bits, _, logical_id = self._entry(
                                values, item, packed, count, base, tidx
                            )
                            output[index] = (
                                (cutlass.Int64(bits) + 1) << 32
                            ) | cutlass.Int64(logical_id)
            if packed or count > _TOPK:
                self._clear_histogram(histogram, tidx)
                for item in cutlass.range_constexpr(cute.size(values)):
                    bits, valid, _ = self._entry(
                        values, item, packed, count, base, tidx
                    )
                    bucket = cutlass.Int32(256)
                    if valid:
                        bucket = self._coarse(bits)
                    self._add_bucket(histogram, bucket, tidx, lane)
                cute.arch.sync_threads()
                coarse, remaining = self._find_bucket(
                    histogram, scalars, warp_sums, cutlass.Int32(_TOPK), tidx, lane
                )
                prefix = cutlass.Uint32(0)
                for round_idx in cutlass.range_constexpr(4):
                    shift = cutlass.const_expr(24 - round_idx * 8)
                    mask = cutlass.const_expr(
                        (0xFFFFFFFF << (32 - round_idx * 8)) & 0xFFFFFFFF
                        if round_idx
                        else 0
                    )
                    self._clear_histogram(histogram, tidx)
                    for item in cutlass.range_constexpr(cute.size(values)):
                        bits, valid, _ = self._entry(
                            values, item, packed, count, base, tidx
                        )
                        ordered = bits ^ cutlass.Uint32(0xFFFFFFFF)
                        bucket = cutlass.Int32(256)
                        if (
                            valid
                            and self._coarse(bits) == coarse
                            and (ordered & cutlass.Uint32(mask)) == prefix
                        ):
                            bucket = cutlass.Int32(
                                (ordered >> cutlass.Uint32(shift)) & cutlass.Uint32(255)
                            )
                        self._add_bucket(histogram, bucket, tidx, lane)
                    cute.arch.sync_threads()
                    digit, remaining = self._find_bucket(
                        histogram, scalars, warp_sums, remaining, tidx, lane
                    )
                    prefix |= cutlass.Uint32(digit) << cutlass.Uint32(shift)
                greater = cutlass.Int32(0)
                equal = cutlass.Int32(0)
                for item in cutlass.range_constexpr(cute.size(values)):
                    bits, valid, _ = self._entry(
                        values, item, packed, count, base, tidx
                    )
                    ordered = bits ^ cutlass.Uint32(0xFFFFFFFF)
                    if valid and ordered < prefix:
                        greater += 1
                    if valid and ordered == prefix:
                        equal += 1
                counts = greater | (equal << 16)
                inclusive, total = self._block_scan(counts, warp_sums, tidx, lane)
                greater_base = (inclusive & 0xFFFF) - greater
                equal_base = ((inclusive >> 16) & 0xFFFF) - equal
                total_greater = total & 0xFFFF
                for item in cutlass.range_constexpr(cute.size(values)):
                    bits, valid, logical_id = self._entry(
                        values, item, packed, count, base, tidx
                    )
                    ordered = bits ^ cutlass.Uint32(0xFFFFFFFF)
                    destination = cutlass.Int32(_TOPK)
                    if valid and ordered < prefix:
                        destination = greater_base
                        greater_base += 1
                    elif valid and ordered == prefix:
                        destination = total_greater + equal_base
                        equal_base += 1
                    if destination < _TOPK:
                        output[destination] = (
                            (cutlass.Int64(bits) + 1) << 32
                        ) | cutlass.Int64(logical_id)
        cute.arch.sync_threads()


class PackedMerge(SplitSelector):
    def __init__(self, splits: int, partial_count: int, enable_pdl: bool):
        super().__init__(1024, (splits * partial_count + 1023) // 1024, enable_pdl)
        self.splits = splits
        self.partial_count = partial_count

    @cute.jit
    def _select_compact(
        self,
        values: cute.Tensor,
        count: cutlass.Int32,
        histogram: cute.Tensor,
        scalars: cute.Tensor,
        warp_sums: cute.Tensor,
        selected: cute.Tensor,
        tidx: cutlass.Int32,
        lane: cutlass.Int32,
    ):
        warp = cute.arch.make_warp_uniform(tidx // 32)
        self._clear_histogram(histogram, tidx)
        for item in cutlass.range_constexpr(self.entries):
            bits, valid, _ = self._entry(values, item, True, count, 0, tidx)
            bucket = cutlass.Int32(256)
            if valid:
                bucket = self._coarse(bits)
            self._add_bucket(histogram, bucket, tidx, lane)
        cute.arch.sync_threads()
        coarse, remaining = self._find_bucket(
            histogram, scalars, warp_sums, cutlass.Int32(_TOPK), tidx, lane
        )
        # The coarse shards already contain each warp's exact scatter sizes.
        local = cutlass.Int32(0)
        for item in cutlass.range_constexpr(8):
            bucket = lane * 8 + item
            if bucket < coarse:
                local += histogram[warp, bucket]
        inclusive = self._warp_scan(local, lane)
        if lane == 31:
            warp_sums[warp] = inclusive | (histogram[warp, coarse] << 16)
        cute.arch.sync_threads()
        if warp == 0:
            inclusive_warps = self._warp_scan(warp_sums[lane], lane)
            warp_sums[lane] = inclusive_warps
        cute.arch.sync_threads()
        before = cutlass.Int32(0)
        if warp > 0:
            before = warp_sums[warp - 1]
        total = warp_sums[self.warps - 1]
        greater_base = before & 0xFFFF
        tie_base = (before >> 16) & 0xFFFF
        total_greater = total & 0xFFFF
        total_ties = (total >> 16) & 0xFFFF
        cute.arch.sync_threads()
        if total_ties <= 256:
            ties = cute.make_tensor(
                cute.recast_ptr(histogram.iterator, dtype=cutlass.Int64),
                cute.make_layout((256,)),
            )
            exact = cute.make_tensor(histogram.iterator + 512, cute.make_layout((256,)))
            lower = (cutlass.Uint32(1) << cutlass.Uint32(lane)) - cutlass.Uint32(1)
            greater_offset = cutlass.Int32(0)
            tie_offset = cutlass.Int32(0)
            for item in cutlass.range_constexpr(self.entries):
                bits, valid, position = self._entry(values, item, True, count, 0, tidx)
                bucket = cutlass.Int32(256)
                if valid:
                    bucket = self._coarse(bits)
                greater_mask = cute.arch.vote_ballot_sync(valid and bucket < coarse)
                tie_mask = cute.arch.vote_ballot_sync(valid and bucket == coarse)
                packed = ((cutlass.Int64(bits) + 1) << 32) | cutlass.Int64(position)
                if valid and bucket < coarse:
                    rank = cutlass.Int32(cute.arch.popc(greater_mask & lower))
                    selected[greater_base + greater_offset + rank] = packed
                if valid and bucket == coarse:
                    rank = cutlass.Int32(cute.arch.popc(tie_mask & lower))
                    ties[tie_base + tie_offset + rank] = packed
                greater_offset += cutlass.Int32(cute.arch.popc(greater_mask))
                tie_offset += cutlass.Int32(cute.arch.popc(tie_mask))
            cute.arch.sync_threads()
            # One warp refines at most 256 ties; other threads have completed
            # their definite winners and wait at the final CTA barrier.
            if warp == 0:
                tie_bits = cute.make_rmem_tensor((8,), cutlass.Uint32)
                for item in cutlass.range_constexpr(8):
                    index = lane + item * 32
                    bits = cutlass.Uint32(0)
                    if index < total_ties:
                        bits = cutlass.Uint32(
                            ties[index] >> cutlass.Int64(32)
                        ) - cutlass.Uint32(1)
                    tie_bits[item] = bits
                prefix = cutlass.Uint32(0)
                rank_remaining = cutlass.Int32(_TOPK) - total_greater
                for round_idx in cutlass.range_constexpr(4):
                    shift = cutlass.const_expr(24 - round_idx * 8)
                    mask = cutlass.const_expr(
                        ((0xFFFFFFFF << (32 - round_idx * 8)) & 0xFFFFFFFF)
                        if round_idx
                        else 0
                    )
                    for item in cutlass.range_constexpr(8):
                        exact[lane * 8 + item] = cutlass.Int32(0)
                    cute.arch.sync_warp()
                    for item in cutlass.range_constexpr(8):
                        ordered = tie_bits[item] ^ cutlass.Uint32(0xFFFFFFFF)
                        bucket = cutlass.Int32(256)
                        if (
                            lane + item * 32 < total_ties
                            and (ordered & cutlass.Uint32(mask)) == prefix
                        ):
                            bucket = cutlass.Int32(
                                (ordered >> cutlass.Uint32(shift)) & cutlass.Uint32(255)
                            )
                        matching = _match_any(bucket)
                        if bucket < 256 and (matching & lower) == 0:
                            cute.arch.atomic_add(
                                exact.iterator + bucket,
                                cute.arch.popc(matching),
                                sem="relaxed",
                                scope="cta",
                            )
                    cute.arch.sync_warp()
                    counts = cute.make_rmem_tensor((8,), cutlass.Int32)
                    local_total = cutlass.Int32(0)
                    for item in cutlass.range_constexpr(8):
                        counts[item] = exact[lane * 8 + item]
                        local_total += counts[item]
                    scan = self._warp_scan(local_total, lane)
                    previous = scan - local_total
                    for item in cutlass.range_constexpr(8):
                        after = previous + counts[item]
                        if previous < rank_remaining and after >= rank_remaining:
                            scalars[0] = lane * 8 + item
                            scalars[1] = rank_remaining - previous
                        previous = after
                    cute.arch.sync_warp()
                    prefix |= cutlass.Uint32(scalars[0]) << cutlass.Uint32(shift)
                    rank_remaining = scalars[1]
                greater_count = cutlass.Int32(0)
                equal_count = cutlass.Int32(0)
                for item in cutlass.range_constexpr(8):
                    ordered = tie_bits[item] ^ cutlass.Uint32(0xFFFFFFFF)
                    if lane + item * 32 < total_ties and ordered < prefix:
                        greater_count += 1
                    if lane + item * 32 < total_ties and ordered == prefix:
                        equal_count += 1
                packed_counts = greater_count | (equal_count << 16)
                inclusive_counts = self._warp_scan(packed_counts, lane)
                all_greater = cute.arch.shuffle_sync(inclusive_counts & 0xFFFF, 31)
                greater_position = (inclusive_counts & 0xFFFF) - greater_count
                equal_position = ((inclusive_counts >> 16) & 0xFFFF) - equal_count
                target = cutlass.Int32(_TOPK) - total_greater
                for item in cutlass.range_constexpr(8):
                    index = lane + item * 32
                    ordered = tie_bits[item] ^ cutlass.Uint32(0xFFFFFFFF)
                    destination = cutlass.Int32(_TOPK)
                    if index < total_ties and ordered < prefix:
                        destination = greater_position
                        greater_position += 1
                    elif index < total_ties and ordered == prefix:
                        destination = all_greater + equal_position
                        equal_position += 1
                    if destination < target:
                        selected[total_greater + destination] = ties[index]
            cute.arch.sync_threads()
        else:
            self._select(
                values,
                True,
                count,
                0,
                histogram,
                scalars,
                warp_sums,
                selected,
                tidx,
                lane,
            )

    @cute.jit
    def __call__(
        self,
        partials: cute.Tensor,
        complete: cute.Tensor,
        output: cute.Tensor,
        capacity: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self._merge(partials, complete, output, capacity).launch(
            grid=(output.shape[0], 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def _merge(
        self,
        partials: cute.Tensor,
        complete: cute.Tensor,
        output: cute.Tensor,
        capacity: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        smem = utils.SmemAllocator()
        histogram = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((self.warps, 256), stride=(256, 1)), 128
        )
        warp_sums = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((self.warps,)), 16
        )
        scalars = smem.allocate_tensor(cutlass.Int32, cute.make_layout((2,)), 16)
        selected = smem.allocate_tensor(cutlass.Int64, cute.make_layout((_TOPK,)), 128)
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()
        valid = complete[row]
        if valid > capacity:
            valid = capacity
        if valid < 0:
            valid = cutlass.Int32(0)
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()
        if valid <= _TOPK:
            if tidx < _TOPK:
                output_id = cutlass.Int32(-1)
                if tidx < valid:
                    output_id = tidx
                output[row, tidx] = output_id
        else:
            words = cute.make_tensor(
                cute.recast_ptr(partials.iterator, dtype=cutlass.Uint32),
                cute.make_layout((partials.shape[0] * partials.shape[1] * 2,)),
            )
            if cutlass.const_expr(self.cache_encoded):
                values = cute.make_rmem_tensor((self.entries,), cutlass.Uint32)
                for item in cutlass.range_constexpr(self.entries):
                    index = tidx + item * self.threads
                    encoded = cutlass.Uint32(0)
                    if index < self.splits * self.partial_count:
                        encoded = cutlass.Uint32(
                            words[(row * partials.shape[1] + index) * 2 + 1]
                        )
                    values[item] = encoded
                self._select(
                    values,
                    True,
                    valid,
                    0,
                    histogram,
                    scalars,
                    warp_sums,
                    selected,
                    tidx,
                    lane,
                )
            else:
                values = cute.make_tensor(
                    words.iterator + row * partials.shape[1] * 2 + tidx * 2 + 1,
                    cute.make_layout((self.entries,), stride=(self.threads * 2,)),
                )
                self._select_compact(
                    values, valid, histogram, scalars, warp_sums, selected, tidx, lane
                )
            if tidx < _TOPK:
                position = cutlass.Int32(selected[tidx] & cutlass.Int64(0xFFFFFFFF))
                if cutlass.const_expr(self.partial_count < _TOPK):
                    output[row, tidx] = position
                else:
                    output[row, tidx] = cutlass.Int32(
                        words[(row * partials.shape[1] + position) * 2]
                    )


class SplitMerge(SplitSelector):
    def __init__(
        self, splits: int, partial_count: int, enable_pdl: bool, linear_bins: bool
    ):
        super().__init__(1024, (splits * partial_count + 1023) // 1024, enable_pdl)
        self.splits = splits
        self.partial_count = partial_count
        self.linear_bins = linear_bins

    @cute.jit
    def _rank_small_ties(
        self,
        ties: cute.Tensor,
        total_ties: cutlass.Int32,
        total_greater: cutlass.Int32,
        selected: cute.Tensor,
        tidx: cutlass.Int32,
        lane: cutlass.Int32,
        items: cutlass.Constexpr[int],
    ):
        warp = cute.arch.make_warp_uniform(tidx // 32)
        # Packed values give a strict exact order even when FP32 scores tie.
        # Each warp owns one target and compares 32 candidates per ballot.
        candidates = cute.make_rmem_tensor((items,), cutlass.Int64)
        for item in cutlass.range_constexpr(items):
            index = lane + item * 32
            value = cutlass.Int64(0)
            if index < total_ties:
                value = ties[index]
            candidates[item] = value
        for group in cutlass.range_constexpr(items):
            target_index = warp + group * 32
            if target_index < total_ties:
                target = ties[target_index]
                rank = cutlass.Int32(0)
                for item in cutlass.range_constexpr(items):
                    index = lane + item * 32
                    matching = cute.arch.vote_ballot_sync(
                        index < total_ties and candidates[item] > target
                    )
                    rank += cutlass.Int32(cute.arch.popc(matching))
                if lane == 0 and rank < cutlass.Int32(_TOPK) - total_greater:
                    selected[total_greater + rank] = target

    @cute.jit
    def _fine_scan(
        self,
        value: cutlass.Int32,
        warp_sums: cute.Tensor,
        tidx: cutlass.Int32,
        lane: cutlass.Int32,
    ):
        warp = tidx // 32
        inclusive = self._warp_scan(value, lane)
        if lane == 31:
            warp_sums[warp] = inclusive
        cute.arch.sync_threads()
        previous_warps = cutlass.Int32(0)
        if lane < warp:
            previous_warps = warp_sums[lane]
        before = cute.arch.warp_redux_sync(previous_warps, "add", 0xFFFFFFFF)
        # The caller's bucket-publication barrier protects warp_sums reuse.
        return before + inclusive

    @cute.jit
    def _fine_coarse(self, bits: cutlass.Uint32):
        value = cutlass.Float32(
            _llvm.bitcast(cutlass.Float32.mlir_type, bits.ir_value())
        )
        if cutlass.const_expr(self.linear_bins):
            # Clamp before integer conversion, including FP32 overflow.
            scaled = value * cutlass.Float32(16)
            index = cutlass.Int32(0)
            if scaled >= cutlass.Float32(1023):
                index = cutlass.Int32(1023)
            elif scaled > cutlass.Float32(0):
                index = cutlass.Int32(scaled)
            bucket = cutlass.Int32(1023) - index
        else:
            half = cutlass.Float16(value)
            half_bits = cutlass.Uint16(
                _llvm.bitcast(cutlass.Uint16.mlir_type, half.ir_value())
            )
            bucket = cutlass.Int32(
                ((half_bits ^ cutlass.Uint16(0xFFFF)) & cutlass.Uint16(0x7FFF))
                >> cutlass.Uint16(5)
            )
        return bucket

    @cute.jit
    def _select_fine(
        self,
        values: cute.Tensor,
        count: cutlass.Int32,
        histogram: cute.Tensor,
        scalars: cute.Tensor,
        warp_sums: cute.Tensor,
        selected: cute.Tensor,
        tidx: cutlass.Int32,
        lane: cutlass.Int32,
    ):
        warp = cute.arch.make_warp_uniform(tidx // 32)
        flat = cute.make_tensor(
            histogram.iterator, cute.make_layout((self.warps * 256,))
        )
        flat[tidx] = cutlass.Int32(0)
        cute.arch.sync_threads()
        lower = (cutlass.Uint32(1) << cutlass.Uint32(lane)) - cutlass.Uint32(1)
        zero_count = cutlass.Int32(0)
        for item in cutlass.range_constexpr(self.entries):
            bits, valid, _ = self._entry(values, item, True, count, 0, tidx)
            bucket = cutlass.Int32(1024)
            if valid:
                bucket = self._fine_coarse(bits)
            if bucket == 1023:
                zero_count += 1
            elif bucket < 1023:
                cute.arch.atomic_add(
                    flat.iterator + bucket, cutlass.Int32(1), sem="relaxed", scope="cta"
                )
        warp_zeros = cute.arch.warp_redux_sync(zero_count, "add", 0xFFFFFFFF)
        if lane == 0 and warp_zeros > 0:
            cute.arch.atomic_add(
                flat.iterator + 1023, warp_zeros, sem="relaxed", scope="cta"
            )
        cute.arch.sync_threads()
        bucket_count = flat[tidx]
        inclusive = self._fine_scan(bucket_count, warp_sums, tidx, lane)
        if inclusive >= _TOPK and inclusive - bucket_count < _TOPK:
            scalars[0] = tidx
            scalars[1] = cutlass.Int32(_TOPK) - (inclusive - bucket_count)
            scalars[2] = bucket_count
        cute.arch.sync_threads()
        coarse = scalars[0]
        remaining = scalars[1]
        total_ties = scalars[2]
        total_greater = cutlass.Int32(_TOPK) - remaining
        if total_ties <= 256:
            # Coarse counts prove bounded output sizes before any reservation.
            ties = cute.make_tensor(
                cute.recast_ptr(histogram.iterator, dtype=cutlass.Int64),
                cute.make_layout((256,)),
            )
            exact = cute.make_tensor(histogram.iterator + 512, cute.make_layout((256,)))
            if tidx == 0:
                scalars[4] = cutlass.Int32(0)
                scalars[5] = cutlass.Int32(0)
            cute.arch.sync_threads()
            for item in cutlass.range_constexpr(self.entries):
                bits, valid, position = self._entry(values, item, True, count, 0, tidx)
                bucket = cutlass.Int32(1024)
                if valid:
                    bucket = self._fine_coarse(bits)
                greater_mask = cute.arch.vote_ballot_sync(valid and bucket < coarse)
                tie_mask = cute.arch.vote_ballot_sync(valid and bucket == coarse)
                greater_count = cutlass.Int32(cute.arch.popc(greater_mask))
                tie_count = cutlass.Int32(cute.arch.popc(tie_mask))
                greater_base = cutlass.Int32(0)
                tie_base = cutlass.Int32(0)
                if lane == 0:
                    if greater_count > 0:
                        greater_base = cute.arch.atomic_add(
                            scalars.iterator + 4,
                            greater_count,
                            sem="relaxed",
                            scope="cta",
                        )
                    if tie_count > 0:
                        tie_base = cute.arch.atomic_add(
                            scalars.iterator + 5, tie_count, sem="relaxed", scope="cta"
                        )
                greater_base = cute.arch.shuffle_sync(greater_base, 0)
                tie_base = cute.arch.shuffle_sync(tie_base, 0)
                packed = ((cutlass.Int64(bits) + 1) << 32) | cutlass.Int64(position)
                if valid and bucket < coarse:
                    rank = cutlass.Int32(cute.arch.popc(greater_mask & lower))
                    selected[greater_base + rank] = packed
                if valid and bucket == coarse:
                    rank = cutlass.Int32(cute.arch.popc(tie_mask & lower))
                    ties[tie_base + rank] = packed
                # Each reservation owns disjoint slots through this pass.
            cute.arch.sync_threads()
            if total_ties <= 32:
                self._rank_small_ties(
                    ties, total_ties, total_greater, selected, tidx, lane, 1
                )
            elif total_ties <= 64:
                self._rank_small_ties(
                    ties, total_ties, total_greater, selected, tidx, lane, 2
                )
            elif total_ties <= 128:
                self._rank_small_ties(
                    ties, total_ties, total_greater, selected, tidx, lane, 4
                )
            else:
                # One warp refines at most 256 ties; other threads have completed
                # their definite winners and wait at the final CTA barrier.
                if warp == 0:
                    tie_bits = cute.make_rmem_tensor((8,), cutlass.Uint32)
                    for item in cutlass.range_constexpr(8):
                        index = lane + item * 32
                        bits = cutlass.Uint32(0)
                        if index < total_ties:
                            bits = cutlass.Uint32(
                                ties[index] >> cutlass.Int64(32)
                            ) - cutlass.Uint32(1)
                        tie_bits[item] = bits
                    prefix = cutlass.Uint32(0)
                    rank_remaining = cutlass.Int32(_TOPK) - total_greater
                    for round_idx in cutlass.range_constexpr(4):
                        shift = cutlass.const_expr(24 - round_idx * 8)
                        mask = cutlass.const_expr(
                            ((0xFFFFFFFF << (32 - round_idx * 8)) & 0xFFFFFFFF)
                            if round_idx
                            else 0
                        )
                        for item in cutlass.range_constexpr(8):
                            exact[lane * 8 + item] = cutlass.Int32(0)
                        cute.arch.sync_warp()
                        for item in cutlass.range_constexpr(8):
                            ordered = tie_bits[item] ^ cutlass.Uint32(0xFFFFFFFF)
                            bucket = cutlass.Int32(256)
                            if (
                                lane + item * 32 < total_ties
                                and (ordered & cutlass.Uint32(mask)) == prefix
                            ):
                                bucket = cutlass.Int32(
                                    (ordered >> cutlass.Uint32(shift))
                                    & cutlass.Uint32(255)
                                )
                            matching = _match_any(bucket)
                            if bucket < 256 and (matching & lower) == 0:
                                cute.arch.atomic_add(
                                    exact.iterator + bucket,
                                    cute.arch.popc(matching),
                                    sem="relaxed",
                                    scope="cta",
                                )
                        cute.arch.sync_warp()
                        counts = cute.make_rmem_tensor((8,), cutlass.Int32)
                        local_total = cutlass.Int32(0)
                        for item in cutlass.range_constexpr(8):
                            counts[item] = exact[lane * 8 + item]
                            local_total += counts[item]
                        scan = self._warp_scan(local_total, lane)
                        previous = scan - local_total
                        for item in cutlass.range_constexpr(8):
                            after = previous + counts[item]
                            if previous < rank_remaining and after >= rank_remaining:
                                scalars[0] = lane * 8 + item
                                scalars[1] = rank_remaining - previous
                            previous = after
                        cute.arch.sync_warp()
                        prefix |= cutlass.Uint32(scalars[0]) << cutlass.Uint32(shift)
                        rank_remaining = scalars[1]
                    greater_count = cutlass.Int32(0)
                    equal_count = cutlass.Int32(0)
                    for item in cutlass.range_constexpr(8):
                        ordered = tie_bits[item] ^ cutlass.Uint32(0xFFFFFFFF)
                        if lane + item * 32 < total_ties and ordered < prefix:
                            greater_count += 1
                        if lane + item * 32 < total_ties and ordered == prefix:
                            equal_count += 1
                    packed_counts = greater_count | (equal_count << 16)
                    inclusive_counts = self._warp_scan(packed_counts, lane)
                    all_greater = cute.arch.shuffle_sync(inclusive_counts & 0xFFFF, 31)
                    greater_position = (inclusive_counts & 0xFFFF) - greater_count
                    equal_position = ((inclusive_counts >> 16) & 0xFFFF) - equal_count
                    target = cutlass.Int32(_TOPK) - total_greater
                    for item in cutlass.range_constexpr(8):
                        index = lane + item * 32
                        ordered = tie_bits[item] ^ cutlass.Uint32(0xFFFFFFFF)
                        destination = cutlass.Int32(_TOPK)
                        if index < total_ties and ordered < prefix:
                            destination = greater_position
                            greater_position += 1
                        elif index < total_ties and ordered == prefix:
                            destination = all_greater + equal_position
                            equal_position += 1
                        if destination < target:
                            selected[total_greater + destination] = ties[index]
            cute.arch.sync_threads()
        else:
            # Large coarse ties (zeros/FP16 overflow) use unchanged exact radix.
            self._select(
                values,
                True,
                count,
                0,
                histogram,
                scalars,
                warp_sums,
                selected,
                tidx,
                lane,
            )

    @cute.jit
    def __call__(
        self,
        partials: cute.Tensor,
        complete: cute.Tensor,
        output: cute.Tensor,
        capacity: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self._merge(partials, complete, output, capacity).launch(
            grid=(output.shape[0], 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.enable_pdl,
        )


class Merge32(SplitMerge):
    def __init__(
        self, splits: int, partial_count: int, enable_pdl: bool, linear_bins: bool
    ):
        super().__init__(splits, partial_count, enable_pdl, linear_bins)
        self.vector_layout = self.cache_encoded and self.entries >= 4

    @cute.jit
    def _entry(
        self,
        values: cute.Tensor,
        item: cutlass.Constexpr[int],
        packed: cutlass.Constexpr[bool],
        count: cutlass.Int32,
        base: cutlass.Int32,
        tidx: cutlass.Int32,
    ):
        index = tidx + item * self.threads
        if cutlass.const_expr(self.vector_layout):
            index = (tidx + (item // 4) * self.threads) * 4 + item % 4
        if cutlass.const_expr(packed):
            if cutlass.const_expr(self.cache_encoded):
                value = values[item]
            else:
                value = _load_global_u32(values.iterator + item * self.threads)
            bits = value - cutlass.Uint32(1)
            valid = value != 0
        else:
            bits = values[item]
            valid = index < count
        return bits, valid, base + index

    @cute.kernel
    def _merge(
        self,
        partials: cute.Tensor,
        complete: cute.Tensor,
        output: cute.Tensor,
        capacity: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        smem = utils.SmemAllocator()
        histogram = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((self.warps, 256), stride=(256, 1)), 128
        )
        warp_sums = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((self.warps,)), 16
        )
        scalars = smem.allocate_tensor(cutlass.Int32, cute.make_layout((6,)), 16)
        selected = smem.allocate_tensor(cutlass.Int64, cute.make_layout((_TOPK,)), 128)
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()
        valid = complete[row]
        if valid > capacity:
            valid = capacity
        if valid < 0:
            valid = cutlass.Int32(0)
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()
        if valid <= _TOPK:
            if tidx < _TOPK:
                output_id = cutlass.Int32(-1)
                if tidx < valid:
                    output_id = tidx
                output[row, tidx] = output_id
        else:
            if cutlass.const_expr(self.cache_encoded):
                values = cute.make_rmem_tensor((self.entries,), cutlass.Uint32)
                if cutlass.const_expr(self.vector_layout):
                    for group in cutlass.range_constexpr(self.entries // 4):
                        source = cute.make_tensor(
                            partials.iterator
                            + row * partials.shape[1]
                            + (tidx + group * self.threads) * 4,
                            cute.make_layout((4,)),
                        )
                        packet = cute.make_rmem_tensor((4,), cutlass.Uint32)
                        cute.autovec_copy(source, packet)
                        for item in cutlass.range_constexpr(4):
                            values[group * 4 + item] = packet[item]
                else:
                    for item in cutlass.range_constexpr(self.entries):
                        index = tidx + item * self.threads
                        encoded = cutlass.Uint32(0)
                        if index < self.splits * self.partial_count:
                            encoded = partials[row, index]
                        values[item] = encoded
                self._select_fine(
                    values, valid, histogram, scalars, warp_sums, selected, tidx, lane
                )
            else:
                values = cute.make_tensor(
                    partials.iterator + row * partials.shape[1] + tidx,
                    cute.make_layout((self.entries,), stride=(self.threads,)),
                )
                self._select_fine(
                    values, valid, histogram, scalars, warp_sums, selected, tidx, lane
                )
            if tidx < _TOPK:
                output[row, tidx] = cutlass.Int32(
                    selected[tidx] & cutlass.Int64(0xFFFFFFFF)
                )

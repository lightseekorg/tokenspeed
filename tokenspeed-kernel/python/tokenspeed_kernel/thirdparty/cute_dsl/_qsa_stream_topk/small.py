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

"""Native warp MMA with exact guarded candidate pruning for small query grids."""

from __future__ import annotations

import cutlass
from cuda.bindings import driver as cuda
from cutlass import cute, utils
from cutlass.cute.nvgpu import warp
from tokenspeed_kernel.thirdparty.cute_dsl._qsa_stream_topk.common import (
    _copy_u128,
    _match_any,
)
from tokenspeed_kernel.thirdparty.cute_dsl._qsa_stream_topk.selector import (
    Merge32,
    SplitMerge,
    SplitSelector,
)

_TOPK = 512


class WarpScore(SplitSelector):
    def __init__(self, splits: int, tile: int, enable_pdl: bool):
        super().__init__(128, (tile + 127) // 128, enable_pdl)
        self.splits = splits
        self.tile = tile
        self.partial_count = min(tile, _TOPK)

    @cute.jit
    def __call__(
        self,
        query: cute.Tensor,
        keys: cute.Tensor,
        table: cute.Tensor,
        requests: cute.Tensor,
        complete: cute.Tensor,
        partials: cute.Tensor,
        q_stride_row: cutlass.Int32,
        q_stride_head: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
        )
        atom = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3), 0, cute.make_layout((8, 64), stride=(64, 1))
        )
        k_layout = cute.tile_to_shape(atom, (64, 128), order=(0, 1))
        q_layout = cute.tile_to_shape(atom, (8, 128), order=(0, 1))
        self._partial_sync(
            mma,
            k_layout,
            q_layout,
            query,
            keys,
            table,
            requests,
            complete,
            partials,
            q_stride_row,
            q_stride_head,
        ).launch(
            grid=(query.shape[0] * self.splits, 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def _partial_sync(
        self,
        mma: cute.TiledMma,
        k_layout: cute.ComposedLayout,
        q_layout: cute.ComposedLayout,
        query: cute.Tensor,
        keys: cute.Tensor,
        table: cute.Tensor,
        requests: cute.Tensor,
        complete: cute.Tensor,
        partials: cute.Tensor,
        q_stride_row: cutlass.Int32,
        q_stride_head: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(tidx // 32)
        row = block // self.splits
        split = block % self.splits
        start = split * self.tile
        smem = utils.SmemAllocator()
        sK = smem.allocate_tensor(
            cutlass.BFloat16, k_layout.outer, 1024, k_layout.inner
        )
        sQ = smem.allocate_tensor(
            cutlass.BFloat16, q_layout.outer, 1024, q_layout.inner
        )
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()
        valid = complete[row]
        capacity = table.shape[1] * 64
        if valid > capacity:
            valid = cutlass.Int32(capacity)
        if valid < 0:
            valid = cutlass.Int32(0)
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()
        if valid > _TOPK:
            request = requests[row]
            thr_mma = mma.get_slice(lane)
            tB = thr_mma.partition_B(sQ)
            rB = thr_mma.make_fragment_B(tB)
            coords = thr_mma.partition_C(cute.make_identity_tensor((16, 8)))
            rC = cute.make_rmem_tensor(coords.shape, cutlass.Float32)
            copy_A = cute.make_tiled_copy_A(
                cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), cutlass.BFloat16),
                mma,
            )
            copy_B = cute.make_tiled_copy_B(
                cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 2), cutlass.BFloat16),
                mma,
            )
            thr_A = copy_A.get_slice(lane)
            thr_B = copy_B.get_slice(lane)
            sB_view = thr_B.partition_S(sQ)
            rB_view = thr_B.retile(rB)
            k_base = cute.recast_ptr(sK.iterator, dtype=cutlass.BFloat16).toint()
            q_base = cute.recast_ptr(sQ.iterator, dtype=cutlass.BFloat16).toint()
            for tile in cutlass.range_constexpr(self.tile // 64):
                tile_start = start + tile * 64
                local = tile * 64 + warp_idx * 16
                logical = start + local
                encoded0 = cutlass.Int64(0)
                encoded1 = cutlass.Int64(0)
                if tile_start < valid:
                    # The validated 64-feature swizzle permits aligned 16B
                    # vectors and ldmatrix row loads.
                    page = table[request, tile_start // 64]
                    for item in cutlass.range_constexpr(8):
                        vector = tidx + item * self.threads
                        key_row = vector // 16
                        col = (vector % 16) * 8
                        logical_key = tile_start + key_row
                        slot = page * 64 + logical_key % 64
                        source = keys.iterator + slot * 128 + col
                        address = (
                            k_base + key_row * 128 + (col % 64) * 2 + (col // 64) * 8192
                        )
                        address ^= (address >> 3) & 0x70
                        destination = cute.make_ptr(
                            cutlass.BFloat16,
                            address,
                            cute.AddressSpace.smem,
                            assumed_align=16,
                        )
                        _copy_u128(destination, source)
                    if tidx < 128:
                        q_head = tidx // 16
                        col = (tidx % 16) * 8
                        source = (
                            query.iterator
                            + row * q_stride_row
                            + (q_head % 4) * q_stride_head
                            + col
                        )
                        address = (
                            q_base + q_head * 128 + (col % 64) * 2 + (col // 64) * 1024
                        )
                        address ^= (address >> 3) & 0x70
                        destination = cute.make_ptr(
                            cutlass.BFloat16,
                            address,
                            cute.AddressSpace.smem,
                            assumed_align=16,
                        )
                        _copy_u128(destination, source)
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.fence_view_async_shared()
                    cute.arch.sync_threads()
                    warp_keys = cute.local_tile(sK, (16, 128), (warp_idx, 0))
                    tA = thr_mma.partition_A(warp_keys)
                    rA = thr_mma.make_fragment_A(tA)
                    sA_view = thr_A.partition_S(warp_keys)
                    rA_view = thr_A.retile(rA)
                    rC.fill(cutlass.Float32(0.0))
                    for k in cutlass.range_constexpr(8):
                        cute.copy(
                            copy_A, sA_view[None, None, k], rA_view[None, None, k]
                        )
                        cute.copy(
                            copy_B, sB_view[None, None, k], rB_view[None, None, k]
                        )
                        cute.gemm(mma, rC, rA[None, None, k], rB[None, None, k], rC)
                    a = cutlass.Float32(0.0)
                    b = cutlass.Float32(0.0)
                    c = cutlass.Float32(0.0)
                    d = cutlass.Float32(0.0)
                    if rC[0] > 0.0:
                        a = rC[0]
                    if rC[1] > 0.0:
                        b = rC[1]
                    if rC[2] > 0.0:
                        c = rC[2]
                    if rC[3] > 0.0:
                        d = rC[3]
                    other_a = cute.arch.shuffle_sync(a, lane ^ 1)
                    other_b = cute.arch.shuffle_sync(b, lane ^ 1)
                    other_c = cute.arch.shuffle_sync(c, lane ^ 1)
                    other_d = cute.arch.shuffle_sync(d, lane ^ 1)
                    score0 = ((a + b) + other_a) + other_b
                    score1 = ((c + d) + other_c) + other_d
                    key0 = logical + coords[0][0]
                    key1 = logical + coords[2][0]
                    if key0 < valid:
                        encoded0 = (
                            (cutlass.Int64(self._float_bits(score0)) + 1) << 32
                        ) | cutlass.Int64(key0)
                    if key1 < valid:
                        encoded1 = (
                            (cutlass.Int64(self._float_bits(score1)) + 1) << 32
                        ) | cutlass.Int64(key1)
                if lane % 4 == 0:
                    partials[row, split * self.partial_count + local + coords[0][0]] = (
                        cutlass.Uint32(encoded0 >> 32)
                    )
                    partials[row, split * self.partial_count + local + coords[2][0]] = (
                        cutlass.Uint32(encoded1 >> 32)
                    )
                cute.arch.sync_threads()


class GuardScore(WarpScore):
    def __init__(
        self,
        splits: int,
        tile: int,
        enable_pdl: bool,
        keep: int,
        compact_ids: bool,
        partition_width: int,
    ):
        super().__init__(splits, tile, enable_pdl)
        self.keep = keep
        self.compact_ids = compact_ids
        self.partition_width = partition_width

    @cute.jit
    def __call__(
        self,
        query: cute.Tensor,
        keys: cute.Tensor,
        table: cute.Tensor,
        requests: cute.Tensor,
        complete: cute.Tensor,
        partials: cute.Tensor,
        candidates: cute.Tensor,
        candidate_ids: cute.Tensor,
        discarded: cute.Tensor,
        q_stride_row: cutlass.Int32,
        q_stride_head: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
        )
        atom = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3), 0, cute.make_layout((8, 64), stride=(64, 1))
        )
        k_layout = cute.tile_to_shape(atom, (64, 128), order=(0, 1))
        q_layout = cute.tile_to_shape(atom, (8, 128), order=(0, 1))
        self._partial_sync(
            mma,
            k_layout,
            q_layout,
            query,
            keys,
            table,
            requests,
            complete,
            partials,
            candidates,
            candidate_ids,
            discarded,
            q_stride_row,
            q_stride_head,
        ).launch(
            grid=(query.shape[0] * self.splits, 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def _partial_sync(
        self,
        mma: cute.TiledMma,
        k_layout: cute.ComposedLayout,
        q_layout: cute.ComposedLayout,
        query: cute.Tensor,
        keys: cute.Tensor,
        table: cute.Tensor,
        requests: cute.Tensor,
        complete: cute.Tensor,
        partials: cute.Tensor,
        candidates: cute.Tensor,
        candidate_ids: cute.Tensor,
        discarded: cute.Tensor,
        q_stride_row: cutlass.Int32,
        q_stride_head: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(tidx // 32)
        row = block // self.splits
        split = block % self.splits
        start = split * self.tile
        smem = utils.SmemAllocator()
        sK = smem.allocate_tensor(
            cutlass.BFloat16, k_layout.outer, 1024, k_layout.inner
        )
        sQ = smem.allocate_tensor(
            cutlass.BFloat16, q_layout.outer, 1024, q_layout.inner
        )
        local_scores = smem.allocate_tensor(
            cutlass.Uint32, cute.make_layout((64,)), 128
        )
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()
        valid = complete[row]
        capacity = table.shape[1] * 64
        if valid > capacity:
            valid = cutlass.Int32(capacity)
        if valid < 0:
            valid = cutlass.Int32(0)
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()
        if valid > _TOPK:
            request = requests[row]
            thr_mma = mma.get_slice(lane)
            tB = thr_mma.partition_B(sQ)
            rB = thr_mma.make_fragment_B(tB)
            coords = thr_mma.partition_C(cute.make_identity_tensor((16, 8)))
            rC = cute.make_rmem_tensor(coords.shape, cutlass.Float32)
            copy_A = cute.make_tiled_copy_A(
                cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), cutlass.BFloat16),
                mma,
            )
            copy_B = cute.make_tiled_copy_B(
                cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 2), cutlass.BFloat16),
                mma,
            )
            thr_A = copy_A.get_slice(lane)
            thr_B = copy_B.get_slice(lane)
            sB_view = thr_B.partition_S(sQ)
            rB_view = thr_B.retile(rB)
            k_base = cute.recast_ptr(sK.iterator, dtype=cutlass.BFloat16).toint()
            q_base = cute.recast_ptr(sQ.iterator, dtype=cutlass.BFloat16).toint()
            for tile in cutlass.range_constexpr(self.tile // 64):
                tile_start = start + tile * 64
                local = tile * 64 + warp_idx * 16
                logical = start + local
                encoded0 = cutlass.Int64(0)
                encoded1 = cutlass.Int64(0)
                if tile_start < valid:
                    # The validated 64-feature swizzle permits aligned 16B
                    # vectors and ldmatrix row loads.
                    page = table[request, tile_start // 64]
                    for item in cutlass.range_constexpr(8):
                        vector = tidx + item * self.threads
                        key_row = vector // 16
                        col = (vector % 16) * 8
                        logical_key = tile_start + key_row
                        slot = page * 64 + logical_key % 64
                        source = keys.iterator + slot * 128 + col
                        address = (
                            k_base + key_row * 128 + (col % 64) * 2 + (col // 64) * 8192
                        )
                        address ^= (address >> 3) & 0x70
                        destination = cute.make_ptr(
                            cutlass.BFloat16,
                            address,
                            cute.AddressSpace.smem,
                            assumed_align=16,
                        )
                        _copy_u128(destination, source)
                    if tidx < 128:
                        q_head = tidx // 16
                        col = (tidx % 16) * 8
                        source = (
                            query.iterator
                            + row * q_stride_row
                            + (q_head % 4) * q_stride_head
                            + col
                        )
                        address = (
                            q_base + q_head * 128 + (col % 64) * 2 + (col // 64) * 1024
                        )
                        address ^= (address >> 3) & 0x70
                        destination = cute.make_ptr(
                            cutlass.BFloat16,
                            address,
                            cute.AddressSpace.smem,
                            assumed_align=16,
                        )
                        _copy_u128(destination, source)
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.fence_view_async_shared()
                    cute.arch.sync_threads()
                    warp_keys = cute.local_tile(sK, (16, 128), (warp_idx, 0))
                    tA = thr_mma.partition_A(warp_keys)
                    rA = thr_mma.make_fragment_A(tA)
                    sA_view = thr_A.partition_S(warp_keys)
                    rA_view = thr_A.retile(rA)
                    rC.fill(cutlass.Float32(0.0))
                    for k in cutlass.range_constexpr(8):
                        cute.copy(
                            copy_A, sA_view[None, None, k], rA_view[None, None, k]
                        )
                        cute.copy(
                            copy_B, sB_view[None, None, k], rB_view[None, None, k]
                        )
                        cute.gemm(mma, rC, rA[None, None, k], rB[None, None, k], rC)
                    a = cutlass.Float32(0.0)
                    b = cutlass.Float32(0.0)
                    c = cutlass.Float32(0.0)
                    d = cutlass.Float32(0.0)
                    if rC[0] > 0.0:
                        a = rC[0]
                    if rC[1] > 0.0:
                        b = rC[1]
                    if rC[2] > 0.0:
                        c = rC[2]
                    if rC[3] > 0.0:
                        d = rC[3]
                    other_a = cute.arch.shuffle_sync(a, lane ^ 1)
                    other_b = cute.arch.shuffle_sync(b, lane ^ 1)
                    other_c = cute.arch.shuffle_sync(c, lane ^ 1)
                    other_d = cute.arch.shuffle_sync(d, lane ^ 1)
                    score0 = ((a + b) + other_a) + other_b
                    score1 = ((c + d) + other_c) + other_d
                    key0 = logical + coords[0][0]
                    key1 = logical + coords[2][0]
                    if key0 < valid:
                        encoded0 = (
                            (cutlass.Int64(self._float_bits(score0)) + 1) << 32
                        ) | cutlass.Int64(key0)
                    if key1 < valid:
                        encoded1 = (
                            (cutlass.Int64(self._float_bits(score1)) + 1) << 32
                        ) | cutlass.Int64(key1)
                if lane % 4 == 0:
                    partials[row, split * self.partial_count + local + coords[0][0]] = (
                        cutlass.Uint32(encoded0 >> 32)
                    )
                    partials[row, split * self.partial_count + local + coords[2][0]] = (
                        cutlass.Uint32(encoded1 >> 32)
                    )
                    local_scores[local + coords[0][0]] = cutlass.Uint32(encoded0 >> 32)
                    local_scores[local + coords[2][0]] = cutlass.Uint32(encoded1 >> 32)
                cute.arch.sync_threads()
                # Partition keys propose candidates; discarded FP32 maxima prove safety.
                if cutlass.const_expr(self.partition_width == 32):
                    if warp_idx < 2:
                        source = warp_idx * 32 + lane
                        value = (
                            local_scores[source] & cutlass.Uint32(0xFFFFFFE0)
                        ) | cutlass.Uint32(lane)
                        for stage in cutlass.range_constexpr(1, 6):
                            size = cutlass.const_expr(1 << stage)
                            for step in cutlass.range_constexpr(stage - 1, -1, -1):
                                distance = cutlass.const_expr(1 << step)
                                if cutlass.const_expr(stage < 5 or distance >= 8):
                                    peer = cute.arch.shuffle_sync(
                                        value, lane ^ distance
                                    )
                                    take_max = ((lane & size) == 0) == (
                                        (lane & distance) == 0
                                    )
                                    if take_max:
                                        if peer > value:
                                            value = peer
                                    else:
                                        if peer < value:
                                            value = peer
                        selected_id = warp_idx * 32 + cutlass.Int32(
                            value & cutlass.Uint32(31)
                        )
                        encoded = local_scores[selected_id]
                        if lane < 8:
                            packed = cutlass.Int64(0)
                            if encoded != 0:
                                packed = (cutlass.Int64(encoded) << 32) | cutlass.Int64(
                                    start + selected_id
                                )
                            candidates[row, split * 16 + warp_idx * 8 + lane] = packed
                        maximum = cutlass.Uint32(0)
                        if lane >= 8:
                            maximum = encoded
                        discarded_max = cute.arch.warp_redux_sync(
                            maximum, "max", 0xFFFFFFFF
                        )
                        if lane == 0:
                            discarded[row, split * 2 + warp_idx] = discarded_max
                elif cutlass.const_expr(self.compact_ids):
                    if warp_idx == 0:
                        first = (
                            local_scores[lane] & cutlass.Uint32(0xFFFFFFC0)
                        ) | cutlass.Uint32(lane)
                        second = (
                            local_scores[lane + 32] & cutlass.Uint32(0xFFFFFFC0)
                        ) | cutlass.Uint32(lane + 32)
                        # Warp-register bitonic partition of 64 score keys.
                        for stage in cutlass.range_constexpr(1, 6):
                            size = cutlass.const_expr(1 << stage)
                            for step in cutlass.range_constexpr(stage - 1, -1, -1):
                                distance = cutlass.const_expr(1 << step)
                                if cutlass.const_expr(distance == 32):
                                    peer_first = second
                                    peer_second = first
                                else:
                                    peer_first = cute.arch.shuffle_sync(
                                        first, lane ^ distance
                                    )
                                    peer_second = cute.arch.shuffle_sync(
                                        second, lane ^ distance
                                    )
                                take_max_first = ((lane & size) == 0) == (
                                    (lane & distance) == 0
                                )
                                take_max_second = (((lane + 32) & size) == 0) == (
                                    ((lane + 32) & distance) == 0
                                )
                                if take_max_first:
                                    if peer_first > first:
                                        first = peer_first
                                else:
                                    if peer_first < first:
                                        first = peer_first
                                if take_max_second:
                                    if peer_second > second:
                                        second = peer_second
                                else:
                                    if peer_second < second:
                                        second = peer_second
                        # Only the retained set matters; its order is unspecified.
                        previous_first = first
                        if second > first:
                            first = second
                        if previous_first < second:
                            second = previous_first
                        for step in cutlass.range_constexpr(
                            4, 3 if self.keep == 16 else 2, -1
                        ):
                            distance = cutlass.const_expr(1 << step)
                            peer = cute.arch.shuffle_sync(first, lane ^ distance)
                            if (lane & distance) == 0:
                                if peer > first:
                                    first = peer
                            else:
                                if peer < first:
                                    first = peer
                        first_id = cutlass.Int32(first & cutlass.Uint32(63))
                        second_id = cutlass.Int32(second & cutlass.Uint32(63))
                        encoded_first = local_scores[first_id]
                        encoded_second = local_scores[second_id]
                        if lane < self.keep:
                            candidates[row, split * self.keep + lane] = encoded_first
                            # The split index supplies the high bits at merge.
                            candidate_ids[row, split * self.keep + lane] = (
                                cutlass.Uint16(first_id)
                            )
                        maximum = encoded_second
                        if lane >= self.keep and encoded_first > maximum:
                            maximum = encoded_first
                        discarded_max = cute.arch.warp_redux_sync(
                            maximum, "max", 0xFFFFFFFF
                        )
                        if lane == 0:
                            discarded[row, split] = discarded_max
                else:
                    if warp_idx == 0:
                        first = (
                            local_scores[lane] & cutlass.Uint32(0xFFFFFFC0)
                        ) | cutlass.Uint32(lane)
                        second = (
                            local_scores[lane + 32] & cutlass.Uint32(0xFFFFFFC0)
                        ) | cutlass.Uint32(lane + 32)
                        # Warp-register bitonic partition of 64 score keys.
                        for stage in cutlass.range_constexpr(1, 7):
                            size = cutlass.const_expr(1 << stage)
                            for step in cutlass.range_constexpr(stage - 1, -1, -1):
                                distance = cutlass.const_expr(1 << step)
                                if cutlass.const_expr(distance == 32):
                                    peer_first = second
                                    peer_second = first
                                else:
                                    peer_first = cute.arch.shuffle_sync(
                                        first, lane ^ distance
                                    )
                                    peer_second = cute.arch.shuffle_sync(
                                        second, lane ^ distance
                                    )
                                take_max_first = ((lane & size) == 0) == (
                                    (lane & distance) == 0
                                )
                                take_max_second = (((lane + 32) & size) == 0) == (
                                    ((lane + 32) & distance) == 0
                                )
                                if take_max_first:
                                    if peer_first > first:
                                        first = peer_first
                                else:
                                    if peer_first < first:
                                        first = peer_first
                                if take_max_second:
                                    if peer_second > second:
                                        second = peer_second
                                else:
                                    if peer_second < second:
                                        second = peer_second
                        first_id = cutlass.Int32(first & cutlass.Uint32(63))
                        second_id = cutlass.Int32(second & cutlass.Uint32(63))
                        encoded_first = local_scores[first_id]
                        encoded_second = local_scores[second_id]
                        if lane < 16:
                            packed = cutlass.Int64(0)
                            if encoded_first != 0:
                                packed = (
                                    cutlass.Int64(encoded_first) << 32
                                ) | cutlass.Int64(start + first_id)
                            candidates[row, split * 16 + lane] = packed
                        maximum = encoded_second
                        if lane >= 16 and encoded_first > maximum:
                            maximum = encoded_first
                        discarded_max = cute.arch.warp_redux_sync(
                            maximum, "max", 0xFFFFFFFF
                        )
                        if lane == 0:
                            discarded[row, split] = discarded_max
                cute.arch.sync_threads()


class GuardCandidateSelector(SplitMerge):
    def __init__(
        self,
        splits: int,
        keep: int,
        enable_pdl: bool,
        linear_bins: bool,
        compact_ids: bool,
        discarded_per_tile: int,
    ):
        super().__init__(splits, keep, enable_pdl, linear_bins)
        self.compact_ids = compact_ids
        self.discarded_per_tile = discarded_per_tile

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
        value = values[item]
        if cutlass.const_expr(self.compact_ids):
            encoded = value
            position = tidx + item * self.threads
        else:
            encoded = cutlass.Uint32(value >> 32)
            position = cutlass.Int32(value & cutlass.Int64(0xFFFFFFFF))
        return encoded - cutlass.Uint32(1), encoded != 0, position

    @cute.jit
    def _fine_lower_bits(self, half_code: cutlass.Int32):
        boundary = cutlass.Uint32(0)
        if half_code > 0:
            if half_code < 1024:
                midpoint = cutlass.Float32(half_code * 2 - 1) * cutlass.Float32(
                    2.0**-25
                )
                boundary = self._float_bits(midpoint)
            elif half_code == 1024:
                boundary = cutlass.Uint32(0x387FE000)
            elif half_code <= 0x7C00:
                boundary = (
                    (cutlass.Uint32(half_code // 1024 + 112) << 23)
                    | (cutlass.Uint32(half_code % 1024) << 13)
                ) - cutlass.Uint32(4096)
            else:
                boundary = cutlass.Uint32(0xFFFFFFFF)
        return boundary

    @cute.jit
    def _select_guarded(
        self,
        values: cute.Tensor,
        count: cutlass.Int32,
        histogram: cute.Tensor,
        scalars: cute.Tensor,
        warp_sums: cute.Tensor,
        selected: cute.Tensor,
        discarded: cute.Tensor,
        row: cutlass.Int32,
        tidx: cutlass.Int32,
        lane: cutlass.Int32,
    ):
        warp = cute.arch.make_warp_uniform(tidx // 32)
        flat = cute.make_tensor(
            histogram.iterator, cute.make_layout((self.warps * 256,))
        )
        flat[tidx] = cutlass.Int32(0)
        maximum = cutlass.Uint32(0)
        if cutlass.const_expr(self.discarded_per_tile == 1):
            if tidx < self.splits:
                maximum = discarded[row, tidx]
        else:
            for group in cutlass.range_constexpr(
                cute.ceil_div(self.splits * self.discarded_per_tile, self.threads)
            ):
                discarded_index = tidx + group * self.threads
                part = cutlass.Uint32(0)
                if discarded_index < self.splits * self.discarded_per_tile:
                    part = discarded[row, discarded_index]
                if part > maximum:
                    maximum = part
        warp_max = cute.arch.warp_redux_sync(maximum, "max", 0xFFFFFFFF)
        if lane == 0:
            flat[1024 + warp] = cutlass.Int32(warp_max)
        cute.arch.sync_threads()
        if warp == 0:
            total_max = cute.arch.warp_redux_sync(
                cutlass.Uint32(flat[1024 + lane]), "max", 0xFFFFFFFF
            )
            if lane == 0:
                scalars[3] = cutlass.Int32(total_max)
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
        if cutlass.const_expr(self.linear_bins):
            lower_encoded = self._float_bits(
                cutlass.Float32(1023 - coarse) * cutlass.Float32(0.0625)
            ) + cutlass.Uint32(1)
        else:
            lower_encoded = self._fine_lower_bits(
                (1023 - coarse) * 32
            ) + cutlass.Uint32(1)
        use_fallback = lower_encoded < cutlass.Uint32(scalars[3])
        if not use_fallback:
            if total_ties <= 256:
                # Coarse counts prove bounded output sizes before any reservation.
                ties = cute.make_tensor(
                    cute.recast_ptr(histogram.iterator, dtype=cutlass.Int64),
                    cute.make_layout((256,)),
                )
                exact = cute.make_tensor(
                    histogram.iterator + 512, cute.make_layout((256,))
                )
                if tidx == 0:
                    scalars[4] = cutlass.Int32(0)
                    scalars[5] = cutlass.Int32(0)
                cute.arch.sync_threads()
                for item in cutlass.range_constexpr(self.entries):
                    bits, valid, position = self._entry(
                        values, item, True, count, 0, tidx
                    )
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
                                scalars.iterator + 5,
                                tie_count,
                                sem="relaxed",
                                scope="cta",
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
                                if (
                                    previous < rank_remaining
                                    and after >= rank_remaining
                                ):
                                    scalars[0] = lane * 8 + item
                                    scalars[1] = rank_remaining - previous
                                previous = after
                            cute.arch.sync_warp()
                            prefix |= cutlass.Uint32(scalars[0]) << cutlass.Uint32(
                                shift
                            )
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
                        all_greater = cute.arch.shuffle_sync(
                            inclusive_counts & 0xFFFF, 31
                        )
                        greater_position = (inclusive_counts & 0xFFFF) - greater_count
                        equal_position = (
                            (inclusive_counts >> 16) & 0xFFFF
                        ) - equal_count
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

        return use_fallback


class GuardMerge(Merge32):
    def __init__(
        self,
        splits: int,
        partial_count: int,
        enable_pdl: bool,
        keep: int,
        compact_ids: bool,
        linear_bins: bool,
        partition_width: int,
    ):
        super().__init__(splits, partial_count, enable_pdl, linear_bins)
        self.candidate_selector = GuardCandidateSelector(
            splits, keep, enable_pdl, linear_bins, compact_ids, 64 // partition_width
        )
        self.candidate_entries = (splits * keep + 1023) // 1024
        self.keep = keep
        self.compact_ids = compact_ids
        self.min_pruned_count = (_TOPK // keep) * 64

    @cute.jit
    def __call__(
        self,
        partials: cute.Tensor,
        candidates: cute.Tensor,
        candidate_ids: cute.Tensor,
        discarded: cute.Tensor,
        complete: cute.Tensor,
        output: cute.Tensor,
        capacity: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self._merge_guard(
            partials, candidates, candidate_ids, discarded, complete, output, capacity
        ).launch(
            grid=(output.shape[0], 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def _merge_guard(
        self,
        partials: cute.Tensor,
        candidates: cute.Tensor,
        candidate_ids: cute.Tensor,
        discarded: cute.Tensor,
        complete: cute.Tensor,
        output: cute.Tensor,
        capacity: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.make_warp_uniform(tidx // 32)
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
            use_fallback = cutlass.Boolean(True)
            # Only try pruning when the complete prefix supplies 512 candidates.
            if valid >= self.min_pruned_count:
                if cutlass.const_expr(self.compact_ids):
                    compact = cute.make_rmem_tensor(
                        (self.candidate_entries,), cutlass.Uint32
                    )
                else:
                    compact = cute.make_rmem_tensor(
                        (self.candidate_entries,), cutlass.Int64
                    )
                for item in cutlass.range_constexpr(self.candidate_entries):
                    index = tidx + item * self.threads
                    if cutlass.const_expr(self.compact_ids):
                        value = cutlass.Uint32(0)
                    else:
                        value = cutlass.Int64(0)
                    if index < self.splits * self.keep:
                        value = candidates[row, index]
                    compact[item] = value
                use_fallback = self.candidate_selector._select_guarded(
                    compact,
                    valid,
                    histogram,
                    scalars,
                    warp_sums,
                    selected,
                    discarded,
                    row,
                    tidx,
                    lane,
                )
            if use_fallback:
                if cutlass.const_expr(self.cache_encoded):
                    values = cute.make_rmem_tensor((self.entries,), cutlass.Uint32)
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
                    self._select_fine(
                        values,
                        valid,
                        histogram,
                        scalars,
                        warp_sums,
                        selected,
                        tidx,
                        lane,
                    )
                else:
                    values = cute.make_tensor(
                        partials.iterator + row * partials.shape[1] + tidx,
                        cute.make_layout((self.entries,), stride=(self.threads,)),
                    )
                    self._select_fine(
                        values,
                        valid,
                        histogram,
                        scalars,
                        warp_sums,
                        selected,
                        tidx,
                        lane,
                    )
            if tidx < _TOPK:
                index = cutlass.Int32(selected[tidx] & cutlass.Int64(0xFFFFFFFF))
                if cutlass.const_expr(self.compact_ids):
                    if not use_fallback:
                        index = (
                            index // self.keep
                        ) * self.partial_count + cutlass.Int32(
                            candidate_ids[row, index]
                        )
                output[row, tidx] = index

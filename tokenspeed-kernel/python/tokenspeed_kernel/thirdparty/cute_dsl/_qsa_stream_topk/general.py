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

"""Workspace-free streaming selection for general NVIDIA QSA layouts."""

from __future__ import annotations

import cutlass
import torch
from cuda.bindings import driver as cuda
from cutlass import cute, utils
from cutlass.cute.nvgpu import warp
from cutlass.cute.runtime import make_fake_tensor
from tokenspeed_kernel.thirdparty.cute_dsl._qsa_stream_topk.common import (
    SelectorPrimitives,
    _copy_u128,
)

# Cache only model/kernel choices; live dimensions, strides and alignment are
# runtime arguments so later requests do not compile on the serving thread.
_COMPILED = {}
_DTYPES = {
    torch.bfloat16: cutlass.BFloat16,
    torch.float16: cutlass.Float16,
    torch.float32: cutlass.Float32,
    torch.int32: cutlass.Int32,
    torch.int64: cutlass.Int64,
}


def _meta(tensor: torch.Tensor) -> cute.Tensor:
    # Every stride is dynamic, including feature/column strides equal to one
    # in the first invocation. Later views must not inherit that assumption.
    return make_fake_tensor(
        _DTYPES[tensor.dtype],
        shape=tuple(cute.sym_int() for _ in tensor.shape),
        stride=tuple(cute.sym_int64() for _ in tensor.shape),
        assumed_align=1,
    )


class GeneralStreamTopK(SelectorPrimitives):
    """One independent row per CTA, with a shared-memory running selection."""

    def __init__(
        self,
        *,
        heads: int,
        head_dim: int,
        topk: int,
        page_size: int,
        enable_pdl: bool,
        use_mma: bool,
    ) -> None:
        self.heads = heads
        self.head_dim = head_dim
        self.topk = topk
        self.page_size = page_size
        self.enable_pdl = enable_pdl
        self.use_mma = use_mma
        self.threads = 256
        self.warps = self.threads // 32

    @cute.jit
    def __call__(
        self,
        query: cute.Tensor,
        key_cache: cute.Tensor,
        page_table: cute.Tensor,
        request_indices: cute.Tensor,
        complete_blocks: cute.Tensor,
        selected_blocks: cute.Tensor,
        vectorized_copy: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
        )
        atom = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3), 0, cute.make_layout((8, 64), stride=(64, 1))
        )
        k_layout = cute.tile_to_shape(atom, (128, 128), order=(0, 1))
        q_layout = cute.tile_to_shape(atom, (8, 128), order=(0, 1))
        self._kernel(
            mma,
            k_layout,
            q_layout,
            query,
            key_cache,
            page_table,
            request_indices,
            complete_blocks,
            selected_blocks,
            vectorized_copy,
        ).launch(
            grid=(query.shape[0], 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.jit
    def _copy_shared_vector(
        self,
        shared: cute.Tensor,
        row: cutlass.Int32,
        col: cutlass.Int32,
        source: cute.Pointer,
    ):
        # Inline cp.async needs a physical byte address; pointer-to-int does
        # not apply the swizzle carried by the CuTe shared-memory iterator.
        address = cute.recast_ptr(shared.iterator, dtype=cutlass.BFloat16).toint()
        address += shared.layout((row, col)) * 2
        # Apply the Swizzle<3, 4, 3> layout above in byte-address units.
        address ^= (address >> 3) & 0x70
        destination = cute.make_ptr(
            cutlass.BFloat16,
            address,
            cute.AddressSpace.smem,
            assumed_align=16,
        )
        _copy_u128(destination, source)

    @cute.jit
    def _score_mma(
        self,
        mma: cute.TiledMma,
        sK: cute.Tensor,
        sQ: cute.Tensor,
        query: cute.Tensor,
        key_cache: cute.Tensor,
        page_table: cute.Tensor,
        pool: cute.Tensor,
        row: cutlass.Int32,
        request: cutlass.Int32 | cutlass.Int64,
        tile_start: cutlass.Int32,
        valid: cutlass.Int32,
        tidx: cutlass.Int32,
        lane: cutlass.Int32,
        vectorized_copy: cutlass.Int32,
    ):
        if vectorized_copy != 0:
            if tidx < 128:
                head = tidx // 16
                col = (tidx % 16) * 8
                source = (
                    query.iterator
                    + row * query.layout.stride[0]
                    + (head % 4) * query.layout.stride[1]
                    + col
                )
                self._copy_shared_vector(sQ, head, col, source)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.fence_view_async_shared()
        else:
            for item in cutlass.range_constexpr(4):
                index = tidx + item * self.threads
                sQ[index // 128, index % 128] = query[
                    row, (index // 128) % 4, index % 128
                ]
        cute.arch.sync_threads()
        thr_mma = mma.get_slice(lane)
        warp_keys = cute.local_tile(sK, (16, 128), (tidx // 32, 0))
        tA = thr_mma.partition_A(warp_keys)
        tB = thr_mma.partition_B(sQ)
        rA = thr_mma.make_fragment_A(tA)
        rB = thr_mma.make_fragment_B(tB)
        coords = thr_mma.partition_C(cute.make_identity_tensor((16, 8)))
        rC = cute.make_rmem_tensor(coords.shape, cutlass.Float32)
        copy_A = cute.make_tiled_copy_A(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), cutlass.BFloat16), mma
        )
        copy_B = cute.make_tiled_copy_B(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 2), cutlass.BFloat16), mma
        )
        sA_view = copy_A.get_slice(lane).partition_S(warp_keys)
        sB_view = copy_B.get_slice(lane).partition_S(sQ)
        rA_view = copy_A.get_slice(lane).retile(rA)
        rB_view = copy_B.get_slice(lane).retile(rB)
        for subtile in cutlass.range_constexpr(self.topk // 128):
            if vectorized_copy != 0:
                for item in cutlass.range_constexpr(128 * 16 // self.threads):
                    index = tidx + item * self.threads
                    key_row = index // 16
                    col = (index % 16) * 8
                    logical = tile_start + subtile * 128 + key_row
                    if logical < valid:
                        page = page_table[request, logical // self.page_size]
                        slot = page * self.page_size + logical % self.page_size
                        source = (
                            key_cache.iterator + slot * key_cache.layout.stride[0] + col
                        )
                        self._copy_shared_vector(sK, key_row, col, source)
                    else:
                        for element in cutlass.range_constexpr(8):
                            sK[key_row, col + element] = cutlass.BFloat16(0)
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.fence_view_async_shared()
            else:
                for item in cutlass.range_constexpr(128 * 128 // self.threads):
                    index = tidx + item * self.threads
                    logical = tile_start + subtile * 128 + index // 128
                    value = cutlass.BFloat16(0)
                    if logical < valid:
                        page = page_table[request, logical // self.page_size]
                        slot = page * self.page_size + logical % self.page_size
                        value = key_cache[slot, 0, index % 128]
                    sK[index // 128, index % 128] = value
            cute.arch.sync_threads()
            rC.fill(cutlass.Float32(0))
            for k in cutlass.range_constexpr(8):
                cute.copy(copy_A, sA_view[None, None, k], rA_view[None, None, k])
                cute.copy(copy_B, sB_view[None, None, k], rB_view[None, None, k])
                cute.gemm(mma, rC, rA[None, None, k], rB[None, None, k], rC)
            scores = cute.make_rmem_tensor((4,), cutlass.Float32)
            for item in cutlass.range_constexpr(4):
                scores[item] = cutlass.Float32(0)
                if rC[item] > 0:
                    scores[item] = rC[item]
            for item in cutlass.range_constexpr(2):
                first = scores[item * 2]
                second = scores[item * 2 + 1]
                other_first = cute.arch.shuffle_sync(first, lane ^ 1)
                other_second = cute.arch.shuffle_sync(second, lane ^ 1)
                score = ((first + second) + other_first) + other_second
                offset = subtile * 128 + (tidx // 32) * 16 + coords[item * 2][0]
                logical = tile_start + offset
                packed = cutlass.Int64(0)
                if logical < valid:
                    packed = (
                        (cutlass.Int64(self._float_bits(score)) + 1) << 32
                    ) | cutlass.Int64(logical)
                if lane % 4 == 0:
                    pool[self.topk + offset] = packed
            cute.arch.sync_threads()

    @cute.jit
    def _score_warp(
        self,
        query: cute.Tensor,
        key_cache: cute.Tensor,
        page_table: cute.Tensor,
        pool: cute.Tensor,
        row: cutlass.Int32,
        request: cutlass.Int32 | cutlass.Int64,
        tile_start: cutlass.Int32,
        valid: cutlass.Int32,
        tidx: cutlass.Int32,
        lane: cutlass.Int32,
    ):
        # Warp lanes cover features of one key. Each warp writes its candidates
        # into the second half of the pool, opposite the retained top-k.
        warp_index = tidx // 32
        for item in cutlass.range_constexpr(self.topk // self.warps):
            offset = warp_index + item * self.warps
            logical = tile_start + offset
            packed = cutlass.Int64(0)
            if logical < valid:
                page = page_table[request, logical // self.page_size]
                slot = page * self.page_size + logical % self.page_size
                score = cutlass.Float32(0)
                for head in cutlass.range_constexpr(self.heads):
                    dot = cutlass.Float32(0)
                    for part in cutlass.range_constexpr(
                        cute.ceil_div(self.head_dim, 32)
                    ):
                        feature = lane + part * 32
                        if feature < self.head_dim:
                            q = cutlass.Float32(query[row, head, feature])
                            k = cutlass.Float32(key_cache[slot, 0, feature])
                            dot += q * k
                    for step in cutlass.range_constexpr(5):
                        dot += cute.arch.shuffle_sync_bfly(dot, 1 << step)
                    if dot > 0:
                        score += dot
                packed = (
                    (cutlass.Int64(self._float_bits(score)) + 1) << 32
                ) | cutlass.Int64(logical)
            if lane == 0:
                pool[self.topk + offset] = packed

    @cute.jit
    def _compare_exchange(
        self,
        values: cute.Tensor,
        pool: cute.Tensor,
        distance: cutlass.Constexpr[int],
        sort_size: cutlass.Constexpr[int],
        tidx: cutlass.Int32,
    ):
        # Only comparisons across warps need shared memory. Warp-local pairs
        # use shuffles; pairs owned by the same thread stay in registers.
        if cutlass.const_expr(32 <= distance < self.threads):
            for item in cutlass.range_constexpr(cute.size(values)):
                index = tidx + item * self.threads
                if index < 2 * self.topk:
                    pool[index] = values[item]
            cute.arch.sync_threads()
        next_values = cute.make_rmem_tensor(values.shape, cutlass.Int64)
        for item in cutlass.range_constexpr(cute.size(values)):
            index = tidx + item * self.threads
            value = values[item]
            peer = cutlass.Int64(0)
            if cutlass.const_expr(distance < 32):
                peer = cute.arch.shuffle_sync_bfly(value, distance)
            elif cutlass.const_expr(distance >= self.threads):
                peer = values[item ^ (distance // self.threads)]
            else:
                if index < 2 * self.topk:
                    peer = pool[index ^ distance]
            active = index < 2 * self.topk
            take_min = (index & distance) != 0
            if cutlass.const_expr(sort_size > 0):
                active = active and index >= self.topk
                local_index = index - self.topk
                take_min = ((local_index & sort_size) == 0) == (
                    (local_index & distance) == 0
                )
            if active:
                if take_min:
                    if peer < value:
                        value = peer
                else:
                    if peer > value:
                        value = peer
            next_values[item] = value
        if cutlass.const_expr(32 <= distance < self.threads):
            cute.arch.sync_threads()
        for item in cutlass.range_constexpr(cute.size(values)):
            values[item] = next_values[item]

    @cute.jit
    def _merge_topk(self, pool: cute.Tensor, tidx: cutlass.Int32):
        # Sort the new tile ascending, opposite to the retained set.
        # Their union is bitonic, so one merge keeps the exact top-k.
        values = cute.make_rmem_tensor(
            (cute.ceil_div(2 * self.topk, self.threads),), cutlass.Int64
        )
        for item in cutlass.range_constexpr(cute.size(values)):
            index = tidx + item * self.threads
            value = cutlass.Int64(0)
            if index < 2 * self.topk:
                value = pool[index]
            values[item] = value
        for stage in cutlass.range_constexpr(1, self.topk.bit_length()):
            size = cutlass.const_expr(1 << stage)
            for step in cutlass.range_constexpr(stage - 1, -1, -1):
                distance = cutlass.const_expr(1 << step)
                self._compare_exchange(values, pool, distance, size, tidx)
        for step in cutlass.range_constexpr(self.topk.bit_length() - 1, -1, -1):
            self._compare_exchange(values, pool, 1 << step, 0, tidx)
        for item in cutlass.range_constexpr(cute.size(values)):
            index = tidx + item * self.threads
            if index < self.topk:
                pool[index] = values[item]
        cute.arch.sync_threads()

    @cute.kernel
    def _kernel(
        self,
        mma: cute.TiledMma,
        k_layout: cute.ComposedLayout,
        q_layout: cute.ComposedLayout,
        query: cute.Tensor,
        key_cache: cute.Tensor,
        page_table: cute.Tensor,
        request_indices: cute.Tensor,
        complete_blocks: cute.Tensor,
        selected_blocks: cute.Tensor,
        vectorized_copy: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        smem = utils.SmemAllocator()
        pool = smem.allocate_tensor(
            cutlass.Int64, cute.make_layout((2 * self.topk,)), 128
        )
        if cutlass.const_expr(self.use_mma):
            sK = smem.allocate_tensor(
                cutlass.BFloat16, k_layout.outer, 1024, k_layout.inner
            )
            sQ = smem.allocate_tensor(
                cutlass.BFloat16, q_layout.outer, 1024, q_layout.inner
            )
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()
        capacity = page_table.shape[1] * self.page_size
        valid = cutlass.Int32(complete_blocks[row])
        if valid > capacity:
            valid = cutlass.Int32(capacity)
        if valid < 0:
            valid = cutlass.Int32(0)
        if valid <= self.topk:
            for item in cutlass.range_constexpr(cute.ceil_div(self.topk, self.threads)):
                index = tidx + item * self.threads
                if index < self.topk:
                    value = cutlass.Int32(-1)
                    if index < valid:
                        value = index
                    selected_blocks[row, index] = value
        else:
            request = request_indices[row]
            # Zero marks invalid candidates. Scoring increments the nonnegative
            # FP32 score bits so valid zero scores rank above this sentinel.
            for item in cutlass.range_constexpr(cute.ceil_div(self.topk, self.threads)):
                index = tidx + item * self.threads
                if index < self.topk:
                    pool[index] = cutlass.Int64(0)
            tile_start = cutlass.Int32(0)
            while tile_start < valid:
                if cutlass.const_expr(self.use_mma):
                    self._score_mma(
                        mma,
                        sK,
                        sQ,
                        query,
                        key_cache,
                        page_table,
                        pool,
                        row,
                        request,
                        tile_start,
                        valid,
                        tidx,
                        lane,
                        vectorized_copy,
                    )
                else:
                    self._score_warp(
                        query,
                        key_cache,
                        page_table,
                        pool,
                        row,
                        request,
                        tile_start,
                        valid,
                        tidx,
                        lane,
                    )
                cute.arch.sync_threads()
                self._merge_topk(pool, tidx)
                tile_start += self.topk
            for item in cutlass.range_constexpr(cute.ceil_div(self.topk, self.threads)):
                index = tidx + item * self.threads
                if index < self.topk:
                    selected_blocks[row, index] = cutlass.Int32(
                        pool[index] & cutlass.Int64(0xFFFFFFFF)
                    )
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()


def kernel(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    page_table: torch.Tensor,
    request_indices: torch.Tensor,
    complete_blocks: torch.Tensor,
    *,
    page_size: int,
    block_topk: int,
    enable_pdl: bool,
) -> torch.Tensor:
    """Select QSA blocks without global scratch or request-group assumptions.

    Tensor dimensions, strides and alignment remain runtime values. Page size,
    head geometry, top-k and PDL are fixed model/kernel choices.
    Empty query batches return without compiling or launching a kernel.

    Args:
        query: BF16, FP16 or FP32 queries shaped ``[rows, heads, head_dim]``.
        key_cache: BF16, FP16 or FP32 keys shaped ``[slots, 1, head_dim]``.
        page_table: Int32 or int64 physical pages shaped ``[requests, pages]``.
        request_indices: Int32 or int64 owning request id per query row.
        complete_blocks: Int32 or int64 valid-block count per query row,
            clamped to the page-table capacity on the device.
        page_size: Number of compressed blocks per page.
        block_topk: Power-of-two number of selected blocks, at least 64.
        enable_pdl: Whether to wait for producers and release dependent grids.

    Returns:
        Int32 logical block ids shaped ``[rows, block_topk]``, padded with ``-1``.
        Equal scores prefer larger logical ids when selection is needed; rows
        fitting entirely within top-k return all their valid ids.
    """

    tensors = (query, key_cache, page_table, request_indices, complete_blocks)
    if not query.is_cuda or any(tensor.device != query.device for tensor in tensors):
        raise ValueError("CuTe QSA stream tensors must share one CUDA device")
    if query.ndim != 3 or key_cache.ndim != 3 or page_table.ndim != 2:
        raise ValueError("CuTe QSA stream expects rank-three Q/K and a rank-two table")
    if key_cache.shape[1:] != (1, query.shape[2]) or min(query.shape[1:]) < 1:
        raise ValueError(
            "CuTe QSA stream requires nonempty heads and matching key width"
        )
    rows = query.shape[0]
    if request_indices.shape != (rows,) or complete_blocks.shape != (rows,):
        raise ValueError("CuTe QSA stream metadata must have one entry per row")
    if page_size < 1 or block_topk < 64 or block_topk & (block_topk - 1):
        raise ValueError(
            "CuTe QSA stream requires positive pages and power-of-two topk >= 64"
        )
    if any(
        tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        for tensor in (query, key_cache)
    ):
        raise TypeError("CuTe QSA stream requires BF16, FP16 or FP32 Q/K")
    if any(
        tensor.dtype not in (torch.int32, torch.int64)
        for tensor in (page_table, request_indices, complete_blocks)
    ):
        raise TypeError("CuTe QSA stream metadata must be int32 or int64")
    selected_blocks = torch.empty(
        (rows, block_topk), device=query.device, dtype=torch.int32
    )
    if rows == 0:
        return selected_blocks
    # Keep alignment a runtime choice so vector and scalar copies share one
    # compiled kernel when feature or projection strides change.
    vectorized_copy = int(
        query.data_ptr() % 16 == 0
        and key_cache.data_ptr() % 16 == 0
        and query.stride(2) == 1
        and key_cache.stride(2) == 1
        and query.stride(0) % 8 == 0
        and query.stride(1) % 8 == 0
        and key_cache.stride(0) % 8 == 0
    )
    cache_key = (
        query.device.index,
        *(tensor.dtype for tensor in tensors),
        query.shape[1],
        query.shape[2],
        block_topk,
        page_size,
        enable_pdl,
    )
    compiled = _COMPILED.get(cache_key)
    if compiled is None:
        tensor_args = [_meta(tensor) for tensor in (*tensors, selected_blocks)]
        use_mma = (
            query.dtype is torch.bfloat16
            and key_cache.dtype is torch.bfloat16
            and query.shape[1:] == (4, 128)
            and block_topk >= 128
        )
        compiled = cute.compile(
            GeneralStreamTopK(
                heads=query.shape[1],
                head_dim=query.shape[2],
                topk=block_topk,
                page_size=page_size,
                enable_pdl=enable_pdl,
                use_mma=use_mma,
            ),
            *tensor_args,
            vectorized_copy,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi --opt-level 3",
        )
        _COMPILED[cache_key] = compiled
    compiled(*tensors, selected_blocks, vectorized_copy)
    return selected_blocks

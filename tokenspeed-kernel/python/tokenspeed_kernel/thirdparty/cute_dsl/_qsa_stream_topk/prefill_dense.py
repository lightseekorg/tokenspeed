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

"""Grouped single-tile TMA scoring and exact selection."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import cute, pipeline, utils
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from tokenspeed_kernel.thirdparty.cute_dsl._qsa_stream_topk.prefill import PrefillTopK

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
_K_COPY_FIRST_WARP = 8
_K_COPY_WARPS = 2
_EPILOGUE_WARPGROUP = 0
_TOPK_FIRST_WARP = 10
_MMA_WARP = 9
_SELECTOR_END_WARP = _TOPK_FIRST_WARP + _QUERY_GROUP * _SELECTOR_WARPS_PER_ROW


class DensePrefillTopK(PrefillTopK):
    def __init__(self, enable_pdl: bool, splits: int, group_layout: str):
        self.splits = splits
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
        page_layout = cute.tile_to_shape(
            tcgen05.make_smem_layout_atom(
                tcgen05.SmemLayoutAtomKind.K_SW128, cutlass.BFloat16
            ),
            (64, 64),
            order=(1, 0),
        )
        source = cute.make_tensor(
            keys.iterator, cute.make_layout((keys.shape[0], 128), stride=(128, 1))
        )
        tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), source, page_layout, (64, 64)
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
            tma_atom,
            tma_tensor,
            page_layout,
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
            grid=(
                self._group_count(query.shape[0], queries_per_request) * self.splits,
                1,
                1,
            ),
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
        row_in_group: cutlass.Int32,
    ):
        bits = self._float_bits(score_buffers[score_stage, row_in_group, candidate])
        block_id = candidate
        return bits, block_id

    @cute.kernel
    def _kernel(
        self,
        tiled_mma: cute.TiledMma,
        load_mma: cute.TiledMma,
        smem_layout_k: cute.ComposedLayout,
        smem_layout_q: cute.ComposedLayout,
        tma_atom: cute.CopyAtom,
        tma_tensor: cute.Tensor,
        page_layout: cute.ComposedLayout,
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
            request_indices,
            query.shape[0],
            queries_per_request,
            block_idx // self.splits,
        )
        if valid_rows > 0:
            self._run_group(
                tiled_mma,
                load_mma,
                smem_layout_k,
                smem_layout_q,
                tma_atom,
                tma_tensor,
                page_layout,
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
        tma_atom: cute.CopyAtom,
        tma_tensor: cute.Tensor,
        page_layout: cute.ComposedLayout,
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
            row_base = (block_idx // self.splits) * self.query_group
        split = block_idx % self.splits
        split_base = split * self.stream_tile
        num_blocks = page_table.shape[1] * page_size

        if cutlass.const_expr(self.enable_pdl and self.group_layout != "ragged"):
            cute.arch.griddepcontrol_wait()
        maximum_complete = cutlass.Int32(0)
        for query_row in cutlass.range_constexpr(_QUERY_GROUP):
            count = self._row_complete(complete_blocks, row_base, query_row, valid_rows)
            if count > maximum_complete:
                maximum_complete = count
        if maximum_complete > num_blocks:
            maximum_complete = cutlass.Int32(num_blocks)
        if maximum_complete <= _TOPK:
            if cutlass.const_expr(self.enable_pdl):
                cute.arch.griddepcontrol_launch_dependents()
            for item in cutlass.range_constexpr(
                cute.ceil_div(_QUERY_GROUP * _TOPK, self.threads)
            ):
                offset = tidx + item * self.threads
                if offset < _QUERY_GROUP * _TOPK:
                    row = row_base + offset // _TOPK
                    index = offset % _TOPK
                    count = self._row_complete(
                        complete_blocks, row_base, row - row_base, valid_rows
                    )
                    if count > num_blocks:
                        count = cutlass.Int32(num_blocks)
                    value = cutlass.Int32(-1)
                    if index < count:
                        value = index
                    self._store_selected(
                        selected, row, index, value, row_base, valid_rows
                    )
        else:
            smem = utils.SmemAllocator()
            tmem_ptr_smem = smem.allocate_array(cutlass.Int32)
            k_pipeline_ptr = smem.allocate_array(cutlass.Int64, _K_STAGES * 2)
            acc_pipeline_ptr = smem.allocate_array(cutlass.Int64, _ACC_STAGES * 2)
            score_pipeline_ptr = smem.allocate_array(cutlass.Int64, _SCORE_STAGES * 2)
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
            score_producer, score_consumer = pipeline.PipelineAsync.create(
                num_stages=_SCORE_STAGES,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 256),
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

            if (
                warp_idx >= _K_COPY_FIRST_WARP
                and warp_idx < _K_COPY_FIRST_WARP + _K_COPY_WARPS
            ):
                q_base_addr = cute.recast_ptr(
                    sQ.iterator, dtype=cutlass.BFloat16
                ).toint()
                for iteration in cutlass.range_constexpr(self.query_copy_iterations):
                    vector = (
                        tidx
                        - _K_COPY_FIRST_WARP * 32
                        + iteration * (_K_COPY_WARPS * 32)
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
                        row_complete -= split_base
                        if row_complete < 0:
                            row_complete = cutlass.Int32(0)
                        if row_complete > self.stream_tile:
                            row_complete = cutlass.Int32(self.stream_tile)
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
            group_complete -= split_base
            if group_complete < 0:
                group_complete = cutlass.Int32(0)
            if group_complete > self.stream_tile:
                group_complete = cutlass.Int32(self.stream_tile)
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
            if warp_idx == _K_COPY_FIRST_WARP:
                for stream_tile in cutlass.range_constexpr(1):
                    tile_subtiles = cute.ceil_div(
                        group_complete - stream_tile * self.stream_tile, _MMA_M
                    )
                    if tile_subtiles > self.stream_tile // _MMA_M:
                        tile_subtiles = cutlass.Int32(self.stream_tile // _MMA_M)
                    if warp_idx == _K_COPY_FIRST_WARP:
                        for subtile in cutlass.range(tile_subtiles):
                            tile_idx = (
                                stream_tile * (self.stream_tile // _MMA_M) + subtile
                            )
                            k_handle = k_producer.acquire_and_advance()
                            for row_half in cutlass.range_constexpr(2):
                                local_key = tile_idx * _MMA_M + row_half * 64
                                page = cutlass.Int32(0)
                                if local_key < group_complete:
                                    page = page_table[
                                        request, (split_base + local_key) // page_size
                                    ]
                                for dim_half in cutlass.range_constexpr(2):
                                    address = (
                                        sK.iterator.toint()
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
                                    destination = cute.make_tensor(pointer, page_layout)
                                    source = cute.local_tile(
                                        tma_tensor, (64, 64), (page, dim_half)
                                    )
                                    tS, tG = cpasync.tma_partition(
                                        tma_atom,
                                        0,
                                        cute.make_layout(1),
                                        cute.group_modes(destination, 0, 2),
                                        cute.group_modes(source, 0, 2),
                                    )
                                    cute.copy(
                                        tma_atom, tG, tS, tma_bar_ptr=k_handle.barrier
                                    )
                    # All 128 copy threads protect the shared K scratch reuse.

            elif warp_idx == self.mma_warp:
                for stream_tile in cutlass.range_constexpr(1):
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
                    tcgen05.Ld32x32bOp(tcgen05.Repetition(self.mma_n // 2)),
                    cutlass.Float32,
                )
                tmem_copy = tcgen05.make_tmem_copy(
                    tmem_atom, half_acc[None, None, None, 0]
                )
                thread_copy = tmem_copy.get_slice(warpgroup_tidx)
                tSrc = thread_copy.partition_S(half_acc)
                identity = cute.make_identity_tensor((_MMA_M, self.mma_n // 2))
                tCoord = thread_copy.partition_D(load_slice.partition_C(identity))
                register_scores = cute.make_rmem_tensor(tCoord.shape, cutlass.Float32)
                for stream_tile in cutlass.range_constexpr(1):
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
                                for query_row in cutlass.range_constexpr(
                                    _QUERY_GROUP // 2
                                ):
                                    if (
                                        coord[1] >= query_row * _HEADS
                                        and coord[1] < (query_row + 1) * _HEADS
                                    ):
                                        row_scores[query_row] += value
                        key_row = tCoord[0][0]
                        for query_row in cutlass.range_constexpr(_QUERY_GROUP // 2):
                            score = row_scores[query_row]
                            score_buffers[
                                score_handle.index,
                                query_half * 4 + query_row,
                                subtile * _MMA_M + key_row,
                            ] = score
                            logical_block = (
                                stream_tile * self.stream_tile
                                + subtile * _MMA_M
                                + key_row
                            )
                            if (
                                logical_block
                                < row_completes[query_half * 4 + query_row]
                            ):
                                bits = self._float_bits(score)
                                bucket = self._coarse_ordered(bits)
                                selector_warp = (logical_block // 32) % 2
                                cute.arch.atomic_add(
                                    coarse_histograms.iterator
                                    + (query_half * 4 + query_row) * 512
                                    + selector_warp * 256
                                    + bucket,
                                    cutlass.Int32(1),
                                    sem="relaxed",
                                    scope="cta",
                                )
                        acc_handle.release()
                    score_handle.commit()

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
                    compacted_row = running_scratch[row_in_group, None]
                    for stream_tile in cutlass.range_constexpr(1):
                        score_handle = score_consumer.wait_and_advance()
                        tile_base = stream_tile * self.stream_tile
                        tile_valid = cutlass.Int32(0)
                        if tile_base < complete:
                            tile_valid = complete - tile_base
                            if tile_valid > self.stream_tile:
                                tile_valid = cutlass.Int32(self.stream_tile)

                        # Selection participation is CTA-uniform, which keeps the
                        # selector named barrier valid at the 512-block boundary.
                        do_select = complete > _TOPK
                        if do_select:
                            running_count = cutlass.Int32(0)
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
                        index = thread_in_row + item * self.selector_threads_per_row
                        value = cutlass.Int32(-1)
                        if index < complete:
                            if complete <= _TOPK:
                                value = index
                            else:
                                value = cutlass.Int32(
                                    compacted_row[index] & cutlass.Int64(0xFFFFFFFF)
                                )
                        self._store_row(
                            selected_row, index, value, row_in_group, valid_rows
                        )

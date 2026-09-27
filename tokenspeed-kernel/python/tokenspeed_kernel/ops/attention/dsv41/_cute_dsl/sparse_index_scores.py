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

"""Hopper sparse MQA index scoring: score only the candidate pool.

DeepGEMM ships a sparse MQA logits kernel for ``sm100`` only, so on ``sm90`` a
Reindex layer scores the whole history through the dense paged kernel and then
throws away everything outside the 2048-block candidate pool. This kernel reads
just the pool.

The output is the compacted score matrix the selector already consumes: column
``c`` holds the score of row ``candidates[token, c // 8] * 8 + c % 8``, matching
:func:`tokenspeed_kernel.ops.attention.dsv41.triton.candidate_scores` applied to
a dense, cleaned score row. The caller maps winners back through that identity.

Scoring follows the DeepGEMM FP8 path rather than the portable BF16 one: E4M3
query and key multiply as-is, the per-head ReLU and weight apply to the raw
accumulator, and the per-row key scale lands after the head reduction. The
query scale is already folded into ``weights`` by the fused quantizer.
"""

import cutlass
from cuda.bindings.driver import CUstream
from cutlass import Float8E4M3FN, Float32, Int32, Int64, cute
from cutlass.cute.nvgpu import cpasync, warp
from tokenspeed_kernel.ops.attention._cute_dsl.utils import (
    EVICT_FIRST,
    mma_sync,
    simple_tma_copy,
)

# One candidate block is eight index rows; the cache page holding it is 64.
BLOCK_ROWS = 8
PAGE_ROWS = 64
BLOCKS_PER_PAGE = PAGE_ROWS // BLOCK_ROWS
HEAD_DIM = 128


class SparseIndexScoreKernel:
    """Score one query's candidate pool against the packed FP8 index cache.

    A CTA owns one query token and a strided slice of its candidate tiles. Each
    tile gathers ``BLOCKS_PER_TILE`` candidate blocks into one contiguous shared
    K tile, so the MMA sees dense rows even though the gather is scattered. The
    four MMA warps split the tile's rows; the gather warp issues one bulk copy
    per candidate block.

    The split over tiles is the launch's grid height, read back through
    ``grid_dim`` rather than compiled in, so one compiled kernel serves every
    batch size.
    """

    TILE_ROWS = 128
    BLOCKS_PER_TILE = TILE_ROWS // BLOCK_ROWS
    ROWS_PER_WARP = 32
    NUM_MMA_WARPS = TILE_ROWS // ROWS_PER_WARP
    MMA_N = 8
    MMA_K = 32
    # A bulk copy is issued by one elected thread, so the single gather warp
    # starts one block at a time. Splitting the issue over four warps measured
    # slower: the barrier ordering the transaction count against their copies,
    # plus the occupancy the extra warps cost, outweighed the issue rate.
    BAR_MMA = 1
    num_stages = 3

    def __init__(self, num_heads: int, enable_pdl: bool):
        self.num_heads = num_heads
        self.enable_pdl = enable_pdl

    @cute.jit
    def __call__(
        self,
        gQ: cute.Tensor,  # [tokens, heads, 128] E4M3
        gWeights: cute.Tensor,  # [tokens, heads] FP32, query scale folded in
        gK: cute.Tensor,  # [pages, 64, 128] E4M3 value plane
        gK_scales: cute.Tensor,  # [pages, 64] FP32 scale plane
        gTable: cute.Tensor,  # [tokens, table_width] int32 page ids
        gVisible: cute.Tensor,  # [tokens] int32 visible row count
        gCandidates: cute.Tensor,  # [tokens, blocks] int32 block ids, -1 padded
        gOut: cute.Tensor,  # [tokens, blocks * 8] FP32
        split_k: Int32,  # CTAs per query token, each owning every split_k-th tile
        stream: CUstream,
    ):
        tokens = gQ.shape[0]
        num_heads = self.num_heads

        # Elements per 128B TMA inner mode; one E4M3 head row is exactly one.
        elems = 128 * 8 // Float8E4M3FN.width
        swizzle_128B = cute.make_swizzle(3, 4, 3)

        sQ_layout = cute.make_layout(
            (1, num_heads, (elems, HEAD_DIM // elems)),
            stride=(0, elems, (1, num_heads * elems)),
        )
        sQ_layout = cute.make_composed_layout(swizzle_128B, 0, sQ_layout)
        Q_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            cute.logical_divide(gQ, (None, None, elems)),
            sQ_layout,
            cta_tiler=(1, num_heads, HEAD_DIM),
        )

        # The K box is one candidate block: eight rows out of a 64-row page.
        # The atom describes that box; the tile it lands in is built below with
        # the same swizzle, whose row period is eight, so a box placed at a
        # multiple of eight rows keeps the phase the MMA side expects.
        sK_box = cute.make_composed_layout(
            swizzle_128B,
            0,
            cute.make_layout(
                (1, BLOCK_ROWS, (elems, HEAD_DIM // elems)),
                stride=(0, elems, (1, BLOCK_ROWS * elems)),
            ),
        )
        K_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            cute.logical_divide(gK, (None, None, elems)),
            sK_box,
            cta_tiler=(1, BLOCK_ROWS, HEAD_DIM),
        )

        self.kernel(
            Q_tma,
            gWeights,
            K_tma,
            gK_scales,
            gTable,
            gVisible,
            gCandidates,
            gOut,
        ).launch(
            grid=(tokens, split_k, 1),
            block=(32 * (self.NUM_MMA_WARPS + 1), 1, 1),
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        Q_tma: cpasync.TmaInfo,
        gWeights: cute.Tensor,
        K_tma: cpasync.TmaInfo,
        gK_scales: cute.Tensor,
        gTable: cute.Tensor,
        gVisible: cute.Tensor,
        gCandidates: cute.Tensor,
        gOut: cute.Tensor,
    ):
        token, split_id, _ = cute.arch.block_idx()
        _, split_k, _ = cute.arch.grid_dim()
        warp_id = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_id = cute.arch.lane_idx()

        num_heads = self.num_heads
        num_stages = self.num_stages
        num_tiles = cute.ceil_div(gCandidates.shape[1], self.BLOCKS_PER_TILE)
        pages = gK_scales.shape[0]
        table_width = gTable.shape[1]

        # The gather writes eight-row boxes into this tile; the box atom and
        # this layout share a swizzle whose row period is eight, so every box
        # lands in the phase the MMA reads it back in.
        tma_elems = 128 * 8 // Float8E4M3FN.width
        sK_pipe = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout(
                (1, self.TILE_ROWS, (tma_elems, HEAD_DIM // tma_elems), num_stages),
                stride=(
                    0,
                    tma_elems,
                    (1, self.TILE_ROWS * tma_elems),
                    self.TILE_ROWS * HEAD_DIM,
                ),
            ),
        )

        smem = cutlass.utils.SmemAllocator()
        sK_all = smem.allocate_tensor(
            Float8E4M3FN,
            sK_pipe.outer,
            byte_alignment=128,
            swizzle=sK_pipe.inner,
        )[0, None, None, None]
        sQ_raw = smem.allocate_tensor(
            Float8E4M3FN,
            Q_tma.smem_layout.outer,
            byte_alignment=128,
            swizzle=Q_tma.smem_layout.inner,
        )
        # Per-row key scales and the resolved block id of each candidate, staged
        # alongside K. The gather warp has already read the page table, so
        # publishing them here keeps the epilogue off global memory entirely.
        sScales = smem.allocate_array(Float32, self.TILE_ROWS * num_stages)
        sBlocks = smem.allocate_array(Int32, self.BLOCKS_PER_TILE * num_stages)
        tma_full_mbar = smem.allocate_array(Int64, num_stages)
        tma_empty_mbar = smem.allocate_array(Int64, num_stages)
        q_full_mbar = smem.allocate_array(Int64, 1)

        if warp_id == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(num_stages):
                    # Two arrivals: the bulk copies' transaction count, and the
                    # gather warp once its plain stores to sScales/sBlocks are
                    # published. The MMA side must not see either half early.
                    cute.arch.mbarrier_init(tma_full_mbar + i, 2)
                    cute.arch.mbarrier_init(tma_empty_mbar + i, 32 * self.NUM_MMA_WARPS)
                cute.arch.mbarrier_init(q_full_mbar, 1)
                cute.arch.mbarrier_init_fence()
        elif warp_id == 1:
            cpasync.prefetch_descriptor(Q_tma.atom)
            cpasync.prefetch_descriptor(K_tma.atom)
        cute.arch.sync_threads()

        # Every global read of another kernel's output sits behind this wait,
        # including the visibility bound the epilogue masks with.
        cute.arch.griddepcontrol_wait()
        visible = gVisible[token]

        if warp_id == self.NUM_MMA_WARPS:
            gQ_tile = cute.local_tile(
                cute.domain_offset((token, 0, 0), Q_tma.tma_tensor),
                tiler=(1, num_heads, HEAD_DIM),
                coord=(0, 0, 0),
            )
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    q_full_mbar, num_heads * HEAD_DIM
                )
            simple_tma_copy(Q_tma.atom, gQ_tile, sQ_raw[0, None, None], q_full_mbar)

            stage = 0
            parity = 1
            for tile in cutlass.range(split_id, num_tiles, split_k):
                # Resolve the tile's blocks one per lane. The page-table read
                # depends on the candidate id, so resolving them inside the
                # issue loop would serialise the whole gather behind sixteen
                # global-load latencies instead of one.
                # A null or out-of-range block still owes the barrier its
                # bytes, so it reads page zero row zero -- always inside the
                # cache -- and the epilogue masks the result.
                lane_page = Int32(0)
                lane_row0 = Int32(0)
                lane_block = Int32(-1)
                if lane_id < self.BLOCKS_PER_TILE:
                    block = gCandidates[token, tile * self.BLOCKS_PER_TILE + lane_id]
                    if block >= 0 and block // BLOCKS_PER_PAGE < table_width:
                        entry = gTable[token, block // BLOCKS_PER_PAGE]
                        if entry >= 0 and entry < pages:
                            # The table and candidate tensors are not required
                            # to be int32, so convert rather than let the branch
                            # change these variables' type.
                            lane_page = Int32(entry)
                            lane_row0 = Int32((block % BLOCKS_PER_PAGE) * BLOCK_ROWS)
                            lane_block = Int32(block)
                # Issued before the bulk copies so the eight scale loads sit
                # under their latency rather than after it.
                lane_scales = cute.make_rmem_tensor(BLOCK_ROWS, Float32)
                # The scale plane spans the whole cache field, which is tens of
                # gigabytes in a real deployment: its element count passes 2^32
                # and a 32-bit offset would wrap. Advance the pointer in 64-bit
                # and index within the page.
                page_scales = cute.make_tensor(
                    gK_scales.iterator + Int64(lane_page) * Int64(gK_scales.stride[0]),
                    cute.make_layout(PAGE_ROWS),
                )
                for r in cutlass.range_constexpr(BLOCK_ROWS):
                    lane_scales[r] = page_scales[lane_row0 + r]

                mbar = tma_full_mbar + stage
                cute.arch.mbarrier_wait(tma_empty_mbar + stage, parity)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        mbar, self.TILE_ROWS * HEAD_DIM
                    )
                # The transaction count must be posted before any copy can
                # arrive against it.
                cute.arch.sync_warp()
                for j in cutlass.range_constexpr(self.BLOCKS_PER_TILE):
                    page = cute.arch.shuffle_sync(lane_page, j)
                    row0 = cute.arch.shuffle_sync(lane_row0, j)
                    simple_tma_copy(
                        K_tma.atom,
                        cute.local_tile(
                            cute.domain_offset((page, row0, 0), K_tma.tma_tensor),
                            tiler=(1, BLOCK_ROWS, HEAD_DIM),
                            coord=(0, 0, 0),
                        ),
                        cute.local_tile(
                            sK_all,
                            (BLOCK_ROWS, HEAD_DIM, 1),
                            (j, 0, stage),
                        ),
                        mbar,
                        cache_policy=EVICT_FIRST,
                    )

                if lane_id < self.BLOCKS_PER_TILE:
                    sBlocks[stage * self.BLOCKS_PER_TILE + lane_id] = lane_block
                    for r in cutlass.range_constexpr(BLOCK_ROWS):
                        sScales[stage * self.TILE_ROWS + lane_id * BLOCK_ROWS + r] = (
                            lane_scales[r]
                        )
                # Publishes the stores above; the copies publish their own
                # bytes through the barrier's transaction count. One arrival,
                # matching the init count of two.
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(mbar)

                stage = (stage + 1) % num_stages
                if stage == 0:
                    parity ^= 1
        else:
            ldsm_elems = 128 // Float8E4M3FN.width  # E4M3 values in 16B
            K_STEPS = HEAD_DIM // self.MMA_K
            N_TILES = cute.ceil_div(num_heads, self.MMA_N)

            # One query token, so the head mode is already the MMA's N mode.
            sQ = sQ_raw[0, None, None]
            sK_warp = cute.local_tile(
                sK_all, (self.ROWS_PER_WARP, HEAD_DIM, num_stages), (warp_id, 0, 0)
            )
            sK_ldsm = cute.zipped_divide(
                sK_warp, (16, cute.make_layout((ldsm_elems, 2)), 1)
            )
            sQ_ldsm = cute.zipped_divide(
                sQ, (self.MMA_N, cute.make_layout((ldsm_elems, 4)))
            )
            sK_ldsm = sK_ldsm[(lane_id % 16, (None, lane_id // 16), 0), None]
            sQ_ldsm = sQ_ldsm[(lane_id % self.MMA_N, (None, lane_id // 8)), None]

            ldsm_atom = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(num_matrices=4), Float8E4M3FN
            )
            rQ = cute.make_rmem_tensor(
                ((ldsm_elems // 2, 2), HEAD_DIM // (self.MMA_K * 2), N_TILES),
                Float8E4M3FN,
            )
            rK = cute.make_rmem_tensor((ldsm_elems, 2, K_STEPS), Float8E4M3FN)
            rC = cute.make_rmem_tensor((4, 2, N_TILES), Float32)
            rW = cute.make_rmem_tensor((2, N_TILES), Float32)

            cute.arch.mbarrier_wait(q_full_mbar, 0)
            cute.arch.barrier(barrier_id=self.BAR_MMA, number_of_threads=128)
            for n in cutlass.range_constexpr(N_TILES):
                cute.copy(ldsm_atom, sQ_ldsm[None, (n, None)], rQ[None, None, n])

            # An MMA lane owns two head columns per N tile, and rows
            # {lane / 4, lane / 4 + 8} of each 16-row half.
            head_lo = (lane_id % 4) * 2
            for n in cutlass.range_constexpr(N_TILES):
                for c in cutlass.range_constexpr(2):
                    head = n * self.MMA_N + head_lo + c
                    rW[c, n] = gWeights[token, head] if head < num_heads else 0.0

            row_lo = lane_id // 4
            stage = 0
            parity = 0
            for tile in cutlass.range(split_id, num_tiles, split_k):
                rC.fill(0.0)
                cute.arch.mbarrier_wait(tma_full_mbar + stage, parity)
                for k in cutlass.range_constexpr(K_STEPS):
                    cute.copy(
                        ldsm_atom, sK_ldsm[None, (None, k, stage)], rK[None, None, k]
                    )
                    for m in cutlass.range_constexpr(2):
                        for n in cutlass.range_constexpr(N_TILES):
                            rC[None, m, n] = mma_sync(
                                rK[None, m, k],
                                rQ[(None, k % 2), k // 2, n],
                                rC[None, m, n],
                            )
                # The staged scales and block ids live in the stage's buffers,
                # so take them into registers before releasing it.
                rScale = cute.make_rmem_tensor((2, 2), Float32)
                rBlock = cute.make_rmem_tensor((2, 2), Int32)
                for m in cutlass.range_constexpr(2):
                    for half in cutlass.range_constexpr(2):
                        local = (
                            warp_id * self.ROWS_PER_WARP + m * 16 + row_lo + half * 8
                        )
                        rScale[m, half] = sScales[stage * self.TILE_ROWS + local]
                        rBlock[m, half] = sBlocks[
                            stage * self.BLOCKS_PER_TILE + local // BLOCK_ROWS
                        ]
                cute.arch.mbarrier_arrive(tma_empty_mbar + stage)

                for m in cutlass.range_constexpr(2):
                    # Weighted ReLU reduction over heads; the key scale lands
                    # after it, matching the DeepGEMM FP8 order.
                    acc_lo = Float32(0.0)
                    acc_hi = Float32(0.0)
                    for n in cutlass.range_constexpr(N_TILES):
                        for c in cutlass.range_constexpr(2):
                            acc_lo += cute.arch.fmax(rC[c, m, n], 0.0) * rW[c, n]
                            acc_hi += cute.arch.fmax(rC[2 + c, m, n], 0.0) * rW[c, n]
                    for i in cutlass.range_constexpr(2):
                        acc_lo += cute.arch.shuffle_sync_bfly(
                            acc_lo, offset=1 << i, mask=-1, mask_and_clamp=31
                        )
                        acc_hi += cute.arch.shuffle_sync_bfly(
                            acc_hi, offset=1 << i, mask=-1, mask_and_clamp=31
                        )
                    if lane_id % 4 == 0:
                        base = warp_id * self.ROWS_PER_WARP + m * 16
                        self._store(
                            gOut,
                            token,
                            tile,
                            base + row_lo,
                            acc_lo,
                            rScale[m, 0],
                            rBlock[m, 0],
                            visible,
                        )
                        self._store(
                            gOut,
                            token,
                            tile,
                            base + row_lo + 8,
                            acc_hi,
                            rScale[m, 1],
                            rBlock[m, 1],
                            visible,
                        )

                stage = (stage + 1) % num_stages
                if stage == 0:
                    parity ^= 1

        # Successors read the scores this kernel writes, so they are released
        # once every warp's stores are in memory, not on entry. The kernel has
        # no early return, so both the gather and the MMA warps reach this.
        cute.arch.sync_threads()
        cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def _store(self, gOut, token, tile, local_row, acc, scale, block, visible):
        """Scale and write one score, masking rows the pool cannot reach.

        ``block`` is the gather warp's resolved candidate id: negative when the
        candidate was null or its page unmapped, so the page-table check is
        already folded in and only the visibility bound remains.
        """
        column = tile * self.TILE_ROWS + local_row
        value = Float32(float("-inf"))
        if block >= 0 and block * BLOCK_ROWS + local_row % BLOCK_ROWS < visible:
            value = acc * scale
        if column < gOut.shape[1]:
            gOut[token, column] = value

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

"""Normal Q @ K.T WGMMA scoring for paged and workspace BF16 keys.

M packs heads from queries sharing a history; N is a 64-key tile. Consumer
warp-groups share K, reduce eight-head stripes, and combine per-query scores.
"""

from functools import cache

import cutlass
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float32, Int32, Int64, cute
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.utils.hopper_helpers import make_trivial_tiled_mma
from quack.compile_utils import make_fake_tensor
from quack.sm90_utils import gemm
from tokenspeed_kernel.ops.attention._cute_dsl.utils import (
    EVICT_FIRST,
    simple_tma_copy,
)
from tokenspeed_kernel.ops.attention.dsa._cute_dsl.bf16_indexer import (
    BF16IndexScoreBase,
)


class BF16IndexScoreNormal(BF16IndexScoreBase):
    def __init__(self, heads, head_dim, stages, initial, recent, paged, query_group):
        super().__init__(heads, head_dim, stages, initial, recent, paged, query_group)
        self.m_tile = (self.heads + 63) // 64 * 64
        self.warpgroups = self.m_tile // 64
        self.consumer_threads = self.warpgroups * 128
        self.producer_warp = self.consumer_threads // 32
        self.key_tile = self.PAGE
        self.pages_per_tile = 1
        # Two eight-row accumulator stripes belong to the same query when H
        # is divisible by 16. Combine them before the warp reduction.
        self.heads_per_partial = 16 if heads % 16 == 0 else 8

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        weights: cute.Tensor,
        keys: cute.Tensor,
        pages: cute.Tensor,
        positions: cute.Tensor,
        lengths: cute.Tensor,
        causal: cute.Tensor,
        starts: cute.Tensor,
        valid: cute.Tensor,
        out: cute.Tensor,
        scale: Float32,
        split: Int32,
        stream: CUstream,
    ):
        q = cute.make_tensor(
            q.iterator,
            cute.make_layout(
                (q.shape[0] // self.query_group, self.heads, self.head_dim),
                stride=(self.heads * self.head_dim, self.head_dim, 1),
            ),
        )
        # Normal orientation: M packs query heads; N covers history keys.
        # Multiple warp-groups share each K tile when M exceeds 64.
        tiled_mma = make_trivial_tiled_mma(
            BFloat16,
            BFloat16,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            Float32,
            (self.warpgroups, 1, 1),
            (64, self.key_tile),
            warpgroup.OperandSource.SMEM,
        )
        elems = 64
        swizzle = cute.make_swizzle(3, 4, 3)
        q_layout = cute.make_composed_layout(
            swizzle,
            0,
            cute.make_layout(
                (1, self.m_tile, (elems, self.head_dim // elems)),
                stride=(0, elems, (1, self.m_tile * elems)),
            ),
        )
        q_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            cute.logical_divide(q, (None, None, elems)),
            q_layout,
            cta_tiler=(1, self.m_tile, self.head_dim),
        )
        k_layout = cute.make_composed_layout(
            swizzle,
            0,
            cute.make_layout(
                (self.PAGE, (elems, self.head_dim // elems)),
                stride=(elems, (1, self.PAGE * elems)),
            ),
        )
        k_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            cute.logical_divide(keys, (None, elems)),
            k_layout,
            cta_tiler=(self.PAGE, self.head_dim),
        )
        self.kernel(
            tiled_mma,
            q_tma,
            weights,
            k_tma,
            pages,
            positions,
            lengths,
            causal,
            starts,
            valid,
            out,
            scale,
        ).launch(
            grid=(q.shape[0], split, 1),
            block=(self.consumer_threads + 32, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        q_tma: cpasync.TmaInfo,
        weights: cute.Tensor,
        k_tma: cpasync.TmaInfo,
        pages: cute.Tensor,
        positions: cute.Tensor,
        lengths: cute.Tensor,
        causal: cute.Tensor,
        starts: cute.Tensor,
        valid: cute.Tensor,
        out: cute.Tensor,
        scale: Float32,
    ):
        token_group, split_id, _ = cute.arch.block_idx()
        _, split, _ = cute.arch.grid_dim()
        token = token_group * self.query_group
        last_token = token + self.query_group - 1
        warp_id = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        stages = self.stages
        s_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout(
                ((self.PAGE, self.pages_per_tile), (64, self.head_dim // 64), stages),
                stride=(
                    (64, self.PAGE * self.head_dim),
                    (1, self.PAGE * 64),
                    self.key_tile * self.head_dim,
                ),
            ),
        )
        alloc = cutlass.utils.SmemAllocator()
        sK = alloc.allocate_tensor(
            BFloat16, s_layout.outer, byte_alignment=128, swizzle=s_layout.inner
        )
        sQ = alloc.allocate_tensor(
            BFloat16,
            q_tma.smem_layout.outer,
            byte_alignment=128,
            swizzle=q_tma.smem_layout.inner,
        )
        sPositions = alloc.allocate_array(Int32, stages * self.pages_per_tile)
        sSources = alloc.allocate_array(Int32, stages * self.pages_per_tile)
        sPartial = alloc.allocate_tensor(
            Float32,
            cute.make_layout(
                (self.key_tile, self.m_tile // self.heads_per_partial),
                stride=(1, self.key_tile),
            ),
            byte_alignment=128,
        )
        full = alloc.allocate_array(Int64, stages)
        empty = alloc.allocate_array(Int64, stages)
        q_full = alloc.allocate_array(Int64, 1)
        if warp_id == 0:
            with cute.arch.elect_one():
                for s in cutlass.range_constexpr(stages):
                    cute.arch.mbarrier_init(full + s, 2)
                    cute.arch.mbarrier_init(empty + s, self.consumer_threads)
                cute.arch.mbarrier_init(q_full, 1)
                cute.arch.mbarrier_init_fence()
        elif warp_id == 1:
            cpasync.prefetch_descriptor(q_tma.atom)
            cpasync.prefetch_descriptor(k_tma.atom)
        cute.arch.sync_threads()
        valid_rows = lengths[last_token]
        tiles = cute.ceil_div(cute.min(valid_rows, out.shape[1]), self.key_tile)
        if warp_id == self.producer_warp:
            q_tile = cute.local_tile(
                cute.domain_offset((token_group, 0, 0), q_tma.tma_tensor),
                tiler=(1, self.m_tile, self.head_dim),
                coord=(0, 0, 0),
            )
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    q_full, self.m_tile * self.head_dim * 2
                )
            simple_tma_copy(q_tma.atom, q_tile, sQ[0, None, None], q_full)
            stage = 0
            phase = 1
            for tile in cutlass.range(split_id, tiles, split):
                cute.arch.mbarrier_wait(empty + stage, phase)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        full + stage, self.key_tile * self.head_dim * 2
                    )
                for p in cutlass.range_constexpr(self.pages_per_tile):
                    page = Int32(0)
                    position = Int32(-1)
                    column = tile * self.pages_per_tile + p
                    if cutlass.const_expr(self.paged):
                        if column < pages.shape[1] and column * self.PAGE < valid_rows:
                            page = pages[last_token, column] * self.PAGE
                            position = positions[last_token, column]
                            if page <= 0 or page >= k_tma.tma_tensor.shape[0]:
                                position = Int32(-1)
                                page = Int32(0)
                    else:
                        page = starts[last_token] + column * self.PAGE
                        position = column
                        if starts[last_token] < 0:
                            position = Int32(-1)
                            page = Int32(0)
                    simple_tma_copy(
                        k_tma.atom,
                        cute.local_tile(
                            cute.domain_offset((page, 0), k_tma.tma_tensor),
                            tiler=(self.PAGE, self.head_dim),
                            coord=(0, 0),
                        ),
                        cute.local_tile(
                            sK, (self.PAGE, self.head_dim, 1), (p, 0, stage)
                        ),
                        full + stage,
                        cache_policy=EVICT_FIRST,
                    )
                    with cute.arch.elect_one():
                        sPositions[stage * self.pages_per_tile + p] = position
                        sSources[stage * self.pages_per_tile + p] = page
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(full + stage)
                stage = (stage + 1) % stages
                if stage == 0:
                    phase ^= 1
        else:
            tid = warp_id * 32 + lane
            thr_mma = tiled_mma.get_slice(tid)
            rA = thr_mma.make_fragment_A(thr_mma.partition_A(sQ[0, None, None]))
            rB = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
            rC = cute.make_rmem_tensor(
                tiled_mma.partition_shape_C((self.m_tile, self.key_tile)), Float32
            )
            flatC = cute.make_tensor(rC.iterator, cute.make_layout(cute.size(rC)))
            # Reduce each warp's head stripes before sharing partial sums.
            # Only partials from the same query are combined.
            head_lo = warp_id * 16 + lane // 4
            head_hi = head_lo + 8
            wlo = Float32(0.0)
            whi = Float32(0.0)
            if head_lo < self.heads:
                wlo = weights[
                    token + head_lo // self.query_heads, head_lo % self.query_heads
                ]
            if head_hi < self.heads:
                whi = weights[
                    token + head_hi // self.query_heads, head_hi % self.query_heads
                ]
            cute.arch.mbarrier_wait(q_full, 0)
            cute.arch.barrier(barrier_id=1, number_of_threads=self.consumer_threads)
            stage = 0
            phase = 0
            for tile in cutlass.range(split_id, tiles, split):
                cute.arch.mbarrier_wait(full + stage, phase)
                gemm(
                    tiled_mma,
                    rC,
                    rA,
                    rB[None, None, None, stage],
                    zero_init=True,
                    wg_wait=0,
                )
                position = sPositions[stage]
                source = sSources[stage]
                # All consumers have finished WGMMA before the producer reuses
                # K. Keep metadata in registers while TMA overlaps the epilogue.
                cute.arch.mbarrier_arrive(empty + stage)
                for n in cutlass.range_constexpr(self.key_tile // 8):
                    for c in cutlass.range_constexpr(2):
                        lo = cute.arch.fmax(flatC[n * 4 + c], 0.0) * wlo
                        hi = cute.arch.fmax(flatC[n * 4 + c + 2], 0.0) * whi
                        if cutlass.const_expr(self.heads_per_partial == 16):
                            lo += hi
                            for i in cutlass.range_constexpr(3):
                                lo += cute.arch.shuffle_sync_bfly(
                                    lo, offset=4 << i, mask=-1, mask_and_clamp=31
                                )
                            if lane < 4:
                                sPartial[n * 8 + lane * 2 + c, warp_id] = lo
                        else:
                            for i in cutlass.range_constexpr(3):
                                lo += cute.arch.shuffle_sync_bfly(
                                    lo, offset=4 << i, mask=-1, mask_and_clamp=31
                                )
                                hi += cute.arch.shuffle_sync_bfly(
                                    hi, offset=4 << i, mask=-1, mask_and_clamp=31
                                )
                            if lane < 4:
                                col = n * 8 + lane * 2 + c
                                sPartial[col, warp_id * 2] = lo
                                sPartial[col, warp_id * 2 + 1] = hi
                cute.arch.barrier(barrier_id=1, number_of_threads=self.consumer_threads)
                for part in cutlass.range_constexpr(
                    cute.ceil_div(
                        self.query_group * self.key_tile, self.consumer_threads
                    )
                ):
                    item = part * self.consumer_threads + tid
                    if item < self.query_group * self.key_tile:
                        g = item // self.key_tile
                        col = item % self.key_tile
                        score = Float32(0.0)
                        for h in cutlass.range_constexpr(
                            self.query_heads // self.heads_per_partial
                        ):
                            score += sPartial[
                                col,
                                g * (self.query_heads // self.heads_per_partial) + h,
                            ]
                        self.store(
                            out,
                            token + g,
                            tile * self.key_tile + col,
                            position * self.PAGE + col % self.PAGE,
                            position,
                            causal[token + g],
                            lengths[token + g],
                            score * scale,
                            source + col % self.PAGE,
                            k_tma.tma_tensor.shape[0],
                            valid,
                        )
                # Do not overwrite shared partials before every query is stored.
                cute.arch.barrier(barrier_id=1, number_of_threads=self.consumer_threads)
                stage = (stage + 1) % stages
                if stage == 0:
                    phase ^= 1
        cute.arch.sync_threads()


@cache
def compile_normal_score(heads, head_dim, stages, initial, recent, paged, query_group):
    t = cute.sym_int()
    cols = cute.sym_int()
    key_rows = cute.sym_int()
    args = [
        cute.runtime.make_fake_tensor(
            BFloat16,
            (t, heads, head_dim),
            (heads * head_dim, head_dim, 1),
            assumed_align=16,
        ),
        cute.runtime.make_fake_tensor(
            Float32, (t, heads), (cute.sym_int(), 1), assumed_align=4
        ),
        cute.runtime.make_fake_tensor(
            BFloat16,
            (key_rows, head_dim),
            (cute.sym_int(divisibility=8), 1),
            assumed_align=16,
        ),
        make_fake_tensor(Int32, (t, cols), divisibility=1),
        make_fake_tensor(Int32, (t, cols), divisibility=1),
        make_fake_tensor(Int32, (t,), divisibility=1),
        make_fake_tensor(Int32, (t,), divisibility=1),
        make_fake_tensor(Int32, (t,), divisibility=1),
        make_fake_tensor(Int32, (cute.sym_int(),), divisibility=1),
        make_fake_tensor(Float32, (t, cute.sym_int()), divisibility=1),
    ]
    return cute.compile(
        BF16IndexScoreNormal(
            heads, head_dim, stages, initial, recent, paged, query_group
        ),
        *args,
        Float32(0.1),
        Int32(1),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )

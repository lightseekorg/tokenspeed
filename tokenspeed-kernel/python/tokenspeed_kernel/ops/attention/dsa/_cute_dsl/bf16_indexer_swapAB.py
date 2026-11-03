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

"""Hopper BF16 K @ Q.T WGMMA scorer: M=64 keys, N=query heads."""

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


class BF16IndexScoreSwapAB(BF16IndexScoreBase):
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
        # Swap the logical Q/K operands: key rows fill physical M=64
        # independently of the number of query heads. Q stays in shared memory
        # across key tiles.
        tiled_mma = make_trivial_tiled_mma(
            BFloat16,
            BFloat16,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            Float32,
            (1, 1, 1),
            (64, self.heads),
            warpgroup.OperandSource.SMEM,
        )
        elems = 64
        swizzle = cute.make_swizzle(3, 4, 3)
        q_layout = cute.make_composed_layout(
            swizzle,
            0,
            cute.make_layout(
                (1, self.heads, (elems, self.head_dim // elems)),
                stride=(0, elems, (1, self.heads * elems)),
            ),
        )
        q_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            cute.logical_divide(q, (None, None, elems)),
            q_layout,
            cta_tiler=(1, self.heads, self.head_dim),
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
        ).launch(grid=(q.shape[0], split, 1), block=(160, 1, 1), stream=stream)

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
                (self.PAGE, (64, self.head_dim // 64), stages),
                stride=(64, (1, self.PAGE * 64), self.PAGE * self.head_dim),
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
        sPositions = alloc.allocate_array(Int32, stages)
        sSources = alloc.allocate_array(Int32, stages)
        full = alloc.allocate_array(Int64, stages)
        empty = alloc.allocate_array(Int64, stages)
        q_full = alloc.allocate_array(Int64, 1)
        if warp_id == 0:
            with cute.arch.elect_one():
                for s in cutlass.range_constexpr(stages):
                    cute.arch.mbarrier_init(full + s, 2)
                    cute.arch.mbarrier_init(empty + s, 128)
                cute.arch.mbarrier_init(q_full, 1)
                cute.arch.mbarrier_init_fence()
        elif warp_id == 1:
            cpasync.prefetch_descriptor(q_tma.atom)
            cpasync.prefetch_descriptor(k_tma.atom)
        cute.arch.sync_threads()
        valid_rows = lengths[last_token]
        tiles = cute.ceil_div(cute.min(valid_rows, out.shape[1]), self.ROWS)
        if warp_id == 4:
            q_tile = cute.local_tile(
                cute.domain_offset((token_group, 0, 0), q_tma.tma_tensor),
                tiler=(1, self.heads, self.head_dim),
                coord=(0, 0, 0),
            )
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    q_full, self.heads * self.head_dim * 2
                )
            simple_tma_copy(q_tma.atom, q_tile, sQ[0, None, None], q_full)
            stage = 0
            phase = 1
            for tile in cutlass.range(split_id, tiles, split):
                page = Int32(0)
                position = Int32(-1)
                if cutlass.const_expr(self.paged):
                    if tile < pages.shape[1] and tile * 64 < valid_rows:
                        page = pages[last_token, tile] * 64
                        position = positions[last_token, tile]
                        if page <= 0 or page >= k_tma.tma_tensor.shape[0]:
                            position = Int32(-1)
                            page = Int32(0)
                else:
                    page = starts[last_token] + tile * 64
                    position = tile
                    if starts[last_token] < 0:
                        position = Int32(-1)
                        page = Int32(0)
                cute.arch.mbarrier_wait(empty + stage, phase)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        full + stage, self.ROWS * self.head_dim * 2
                    )
                simple_tma_copy(
                    k_tma.atom,
                    cute.local_tile(
                        cute.domain_offset((page, 0), k_tma.tma_tensor),
                        tiler=(self.PAGE, self.head_dim),
                        coord=(0, 0),
                    ),
                    sK[None, None, stage],
                    full + stage,
                    cache_policy=EVICT_FIRST,
                )
                with cute.arch.elect_one():
                    sPositions[stage] = position
                    sSources[stage] = page
                    cute.arch.mbarrier_arrive(full + stage)
                stage = (stage + 1) % stages
                if stage == 0:
                    phase ^= 1
        else:
            thr_mma = tiled_mma.get_slice(warp_id * 32 + lane)
            rA = thr_mma.make_fragment_A(thr_mma.partition_A(sK))
            rB = thr_mma.make_fragment_B(thr_mma.partition_B(sQ[0, None, None]))
            rC = cute.make_rmem_tensor(
                tiled_mma.partition_shape_C((64, self.heads)), Float32
            )
            # WGMMA assigns two key rows to each four-lane group. Consecutive
            # groups of four accumulators cover eight heads, with the two row
            # values interleaved. Reduce heads within each query, never across it.
            flatC = cute.make_tensor(rC.iterator, cute.make_layout(cute.size(rC)))
            rW = cute.make_rmem_tensor(
                (self.query_heads // 4, self.query_group), Float32
            )
            for g in cutlass.range_constexpr(self.query_group):
                for j in cutlass.range_constexpr(self.query_heads // 4):
                    rW[j, g] = weights[token + g, (j // 2) * 8 + j % 2 + lane % 4 * 2]
            cute.arch.mbarrier_wait(q_full, 0)
            cute.arch.barrier(barrier_id=1, number_of_threads=128)
            stage = 0
            phase = 0
            for tile in cutlass.range(split_id, tiles, split):
                cute.arch.mbarrier_wait(full + stage, phase)
                gemm(
                    tiled_mma,
                    rC,
                    rA[None, None, None, stage],
                    rB,
                    zero_init=True,
                    wg_wait=0,
                )
                position = sPositions[stage]
                source = sSources[stage]
                cute.arch.mbarrier_arrive(empty + stage)
                for g in cutlass.range_constexpr(self.query_group):
                    lo = Float32(0.0)
                    hi = Float32(0.0)
                    for n in cutlass.range_constexpr(self.query_heads // 8):
                        for c in cutlass.range_constexpr(2):
                            lo += (
                                cute.arch.fmax(
                                    flatC[g * (self.query_heads // 2) + n * 4 + c], 0.0
                                )
                                * rW[n * 2 + c, g]
                            )
                            hi += (
                                cute.arch.fmax(
                                    flatC[g * (self.query_heads // 2) + n * 4 + c + 2],
                                    0.0,
                                )
                                * rW[n * 2 + c, g]
                            )
                    for i in cutlass.range_constexpr(2):
                        lo += cute.arch.shuffle_sync_bfly(
                            lo, offset=1 << i, mask=-1, mask_and_clamp=31
                        )
                        hi += cute.arch.shuffle_sync_bfly(
                            hi, offset=1 << i, mask=-1, mask_and_clamp=31
                        )
                    if lane % 4 == 0:
                        row = warp_id * 16 + lane // 4
                        self.store(
                            out,
                            token + g,
                            tile * self.ROWS + row,
                            position * 64 + row,
                            position,
                            causal[token + g],
                            lengths[token + g],
                            lo * scale,
                            source + row,
                            k_tma.tma_tensor.shape[0],
                            valid,
                        )
                        self.store(
                            out,
                            token + g,
                            tile * self.ROWS + row + 8,
                            position * 64 + row + 8,
                            position,
                            causal[token + g],
                            lengths[token + g],
                            hi * scale,
                            source + row + 8,
                            k_tma.tma_tensor.shape[0],
                            valid,
                        )
                stage = (stage + 1) % stages
                if stage == 0:
                    phase ^= 1
        cute.arch.sync_threads()


@cache
def compile_swap_ab_score(heads, head_dim, stages, initial, recent, paged, query_group):
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
        BF16IndexScoreSwapAB(
            heads, head_dim, stages, initial, recent, paged, query_group
        ),
        *args,
        Float32(0.1),
        Int32(1),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )

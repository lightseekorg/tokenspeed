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

"""CTA-local LongCat P2 E4M3 QK, with the original mma.sync C fragments.

All 384 threads call begin/compute/end uniformly. Warp 8 allocates and issues
MMA; warps 0..7 each consume one query. prepare is a compile-time setup.
Q occupies 32768 B. Each K stage needs tile_n * 128 B, with caller-owned scales after
that range. Both buffers must be 1024 B aligned, K-major SW128. No QK output is
written to global or shared memory by compute.

Evidence: CUTLASS 4.5.2 tcgen05_programming; PTX sections 9.7.18.2.3.1.1
(.16x256b fragments), 9.7.18.8.1 (warp access), 9.7.18.7.1 (allocation).
CUTLASS v4.5.2 (commit db1c288993354c88e551c40c19a8fb93a774a241)
include/cute/atom/copy_traits_sm100.hpp specialization
SM100_TMEM_LOAD_16dp256b4x supplies the exact bit mapping:
  ValID: (1024,16):(1,DP_b)
  DstLayout: ((4,8),(64,2,4)):((64,1024),(1,8192,256))
For FP32 register r in [0,16), a lane l reads row l//4+8*((r//2)%2)
and column 2*(l%4)+(r%2)+8*(r//4). That is rS[r%4,m,r//4] for
an m16 load at row offset m*16. Two loads replace the 128 warp shuffles.
"""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cute.nvgpu import tcgen05
from cutlass.cutlass_dsl import dsl_user_op


@dsl_user_op
def _load_fragment_m16(address, *, loc=None, ip=None):
    # PTX .16x256b.x4: 16 FP32 registers per lane, in mma.sync C order.
    ty = cutlass.Float32.mlir_type
    registers = ", ".join(f"${i}" for i in range(16))
    result = llvm.inline_asm(
        llvm.StructType.get_literal([ty] * 16),
        [cutlass.Int32(address).ir_value(loc=loc, ip=ip)],
        "tcgen05.ld.sync.aligned.16x256b.x4.b32 {" + registers + "}, [$16];",
        ",".join(["=f"] * 16 + ["r", "~{memory}"]),
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )
    values = [llvm.extractvalue(ty, result, [i], loc=loc, ip=ip) for i in range(16)]
    return cute.TensorSSA(
        vector.from_elements(ir.VectorType.get([16], ty, loc=loc), values, loc=loc),
        16,
        cutlass.Float32,
    )


class LongcatQKTile:
    """Two independent M128 K128 dense native-FP8 MMA tiles, N=32 or 64."""

    def __init__(self, tile_n: int = 32):
        if tile_n not in (32, 64):
            raise ValueError("tile_n must be 32 or 64")
        self.tile_n = int(tile_n)

    @cute.jit
    def prepare(self):
        mma = cute.make_tiled_mma(
            tcgen05.MmaF8F6F4Op(
                cutlass.Float8E4M3FN,
                cutlass.Float8E4M3FN,
                cutlass.Float32,
                (128, self.tile_n, 32),
                tcgen05.CtaGroup.ONE,
                tcgen05.OperandSource.SMEM,
                tcgen05.OperandMajorMode.K,
                tcgen05.OperandMajorMode.K,
            )
        )
        # Logical matrices are row-major; pointer swizzle carries SW128 into
        # the SMEM descriptor. The 16-byte copy address formula below agrees.
        q_layout = cute.make_layout((128, 128), stride=(128, 1))
        k_layout = cute.make_layout((self.tile_n, 128), stride=(128, 1))
        acc_layout = mma.make_fragment_C(
            mma.partition_shape_C((128, self.tile_n))
        ).layout
        return mma, q_layout, k_layout, acc_layout

    @cute.jit
    def q_offset(self, args, row, col_bytes):
        """Destination byte offset for a Q row's aligned 16-byte segment."""
        return row * 128 + (col_bytes ^ ((row & 7) * 16))

    @cute.jit
    def k_offset(self, args, row, col_bytes):
        """Destination byte offset within one K stage, for a 16-byte segment."""
        return row * 128 + (col_bytes ^ ((row & 7) * 16))

    @cute.jit
    def begin(self, mbar, tmem_word, tidx):
        """Initialize one completion barrier and allocate 2 * tile_n TMEM columns.

        mbar is an 8-byte SMEM pointer, tmem_word a 4-byte SMEM pointer.
        Named barrier 2 is used with all 384 threads, only during allocation.
        Return the TMEM pointer; the caller starts completion phase at zero.
        """
        if tidx == 0:
            cute.arch.mbarrier_init(mbar, 1)
            cute.arch.mbarrier_init_fence()
        cute.arch.barrier()
        allocator = utils.TmemAllocator(
            tmem_word,
            barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=2, num_threads=384),
            allocator_warp_id=8,
        )
        allocator.allocate(2 * self.tile_n)
        allocator.wait_for_alloc()
        return allocator.retrieve_ptr(cutlass.Float32)

    @cute.jit
    def compute(self, args, sQ_u8_ptr, sK_u8_ptr, mbar, tmem, tidx, phase):
        """Compute one tile and return its first 32 columns as rS (4,2,4).

        Caller completes Q/K copies before entry. All 384 threads enter.
        phase alternates 0/1 for each call; only warp 0..7 returns valid rS.
        A completion wait and a CTA barrier protect subsequent K and TMEM
        reuse. For N64, read(..., half=1) consumes the remaining columns before reuse.
        Scale/ReLU/weighted sum/causal mask remain in the caller.
        """
        mma, q_layout, k_layout, acc_layout = args
        warp = tidx // 32
        swizzle = cute.make_swizzle(3, 4, 3)
        q0 = cute.make_tensor(
            cute.recast_ptr(sQ_u8_ptr, swizzle, cutlass.Float8E4M3FN), q_layout
        )
        q1 = cute.make_tensor(
            cute.recast_ptr(sQ_u8_ptr + 16384, swizzle, cutlass.Float8E4M3FN), q_layout
        )
        key = cute.make_tensor(
            cute.recast_ptr(sK_u8_ptr, swizzle, cutlass.Float8E4M3FN), k_layout
        )
        thread_mma = mma.get_slice(0)
        fa0 = mma.make_fragment_A(thread_mma.partition_A(q0))
        fa1 = mma.make_fragment_A(thread_mma.partition_A(q1))
        fb = mma.make_fragment_B(thread_mma.partition_B(key))
        acc0 = cute.make_tensor(tmem, acc_layout)
        acc1 = cute.make_tensor(tmem + self.tile_n, acc_layout)
        # Q may have been written through the generic proxy; order it before
        # SMEM descriptors are consumed by the MMA asynchronous proxy.
        cute.arch.fence_view_async_shared()
        cute.arch.barrier()
        if warp == 8:
            for k in cutlass.range_constexpr(4):
                mma.set(tcgen05.Field.ACCUMULATE, k != 0)
                cute.gemm(mma, acc0, fa0[None, None, k], fb[None, None, k], acc0)
                cute.gemm(mma, acc1, fa1[None, None, k], fb[None, None, k], acc1)
            with cute.arch.elect_one():
                tcgen05.commit(mbar)
        cute.arch.mbarrier_wait(mbar, phase)
        result = self.read(args, tmem, tidx, half=0)
        cute.arch.barrier()
        return result

    @cute.jit
    def read(self, args, tmem, tidx, half: cutlass.Constexpr = 0):
        """Read a completed 32-column window; caller protects TMEM reuse."""
        warp = tidx // 32
        result = cute.make_rmem_tensor((4, 2, 4), cutlass.Float32)
        result.fill(0.0)
        if warp < 8:
            address = (
                cutlass.Int32(tmem.toint())
                + (warp % 4) * (32 << 16)
                + (warp // 4) * self.tile_n
                + half * 32
            )
            # Each warp can access its 32 TMEM rows. Issue both m16 loads
            # before one wait, then use the returned native C-order registers.
            first = cute.make_rmem_tensor((16,), cutlass.Float32)
            second = cute.make_rmem_tensor((16,), cutlass.Float32)
            first.store(_load_fragment_m16(address))
            second.store(_load_fragment_m16(address + (16 << 16)))
            cute.arch.fence_view_async_tmem_load()
            for n in cutlass.range_constexpr(4):
                for i in cutlass.range_constexpr(4):
                    result[i, 0, n] = first[n * 4 + i]
                    result[i, 1, n] = second[n * 4 + i]
        return result

    @cute.jit
    def end(self, tmem, tidx):
        """Free the P2 allocation uniformly, after all last-tile consumers.

        The indexer invokes P2 once per kernel launch, and later phases do
        not allocate TMEM. Therefore warp 8 relinquishes its allocation
        permit before freeing. Calling begin again within the same kernel
        launch after this end is invalid.
        """
        cute.arch.barrier()
        if tidx // 32 == 8:
            cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)
            cute.arch.dealloc_tmem(tmem, 2 * self.tile_n, is_two_cta=False)
        cute.arch.barrier()

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

"""Opt-in owner-shard reduction inside the up-projection epilogue.

The caller publishes immutable symmetric partials before launch and performs a
complete all-rank output barrier before any rank reuses those partials. No CTA
inside this persistent GEMM attempts a cross-rank rendezvous.
"""

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cutlass_dsl import T, dsl_user_op
from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.primitives import (
    packed_u32x4_to_bf16x8,
)
from tokenspeed_kernel.thirdparty.cute_dsl.symmetric_up_projection.fused_shared_rs_groups import (
    reduce_shared_vectors,
)
from tokenspeed_kernel.thirdparty.cute_dsl.symmetric_up_projection.gemm import (
    SymmetricUpProjectionGemm,
)


@dsl_user_op
def reduce_shared_vector(address, active, *, loc=None, ip=None):
    """Return the exact BF16x8 result of the established FP32 NVLS reduction."""
    words = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [
            address.ir_value(loc=loc, ip=ip),
            cutlass.Uint32(active).ir_value(loc=loc, ip=ip),
        ],
        """{
            .reg .u64 input_address;
            .reg .pred valid;
            mov.u64 input_address, $4;
            setp.ne.u32 valid, $5, 0;
            mov.b32 $0, 0;
            mov.b32 $1, 0;
            mov.b32 $2, 0;
            mov.b32 $3, 0;
            @valid multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2
                {$0, $1, $2, $3}, [input_address];
        }""",
        "=r,=r,=r,=r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    packed = vector.from_elements(
        ir.VectorType.get([4], T.i32(), loc=loc),
        [llvm.extractvalue(T.i32(), words, [i], loc=loc, ip=ip) for i in range(4)],
        loc=loc,
        ip=ip,
    )
    return packed_u32x4_to_bf16x8(
        cute.TensorSSA(packed, 4, cutlass.Uint32), loc=loc, ip=ip
    )


class FusedSharedRsUpProjectionGemm(SymmetricUpProjectionGemm):
    """Canonical wide GEMM with one or two coalesced reduction/addend stages.

    Args:
        raw_multicast_address: Aligned NVLS mapping of replicated-layout
            rank-local BF16 partials [M,7168], kept immutable through the exit.
        rank: This owner's rank in the eight-rank group.
        c_stages: Explicit output ring depth, or zero for the auto budget.
        prefetch_acc_tile: Preserve the independently selectable TMEM reservoir.
        addend_stages: One reuses a single tile; two prefetches the following
            reduction/residual tile before consuming the current tile. Storage
            and readiness follow the ordinary validated addend pipeline.
        reduce_vectors: One retains immediate per-vector consumption; 2/4/8
            issue independent NVLS requests together before any R2S consumption.

    No global shared shard is produced. Hardware reduction BF16 bits become
    the existing shared addend, with the original two epilogue roundings.
    """

    def __init__(
        self,
        raw_multicast_address: int,
        rank: int,
        c_stages: int,
        prefetch_acc_tile: bool,
        addend_stages: int,
        reduce_vectors: int,
    ):
        if (
            type(raw_multicast_address) is not int
            or raw_multicast_address <= 0
            or raw_multicast_address % 16
        ):
            raise ValueError("raw multicast address must be positive and aligned")
        if type(rank) is not int or not 0 <= rank < 8:
            raise ValueError("requires an explicit TP8 owner rank")
        if type(addend_stages) is not int or addend_stages not in (1, 2):
            raise ValueError("fused RS requires one or two explicit addend stages")
        if type(reduce_vectors) is not int or reduce_vectors not in (1, 2, 4, 8):
            raise ValueError("fused RS reduce_vectors must be 1, 2, 4 or 8")
        if reduce_vectors > 1 and addend_stages != 1:
            raise ValueError("grouped reduction currently requires one addend stage")
        super().__init__(cutlass.Float32, True, (256, 128), (2, 1), True, False, "tma")
        self.configure_output_pipeline(c_stages, True, True, prefetch_acc_tile)
        self.configure_epilogue_tile(128, 64)
        self.configure_addend_loads("no_allocate", False)
        self.configure_addend_pipeline(addend_stages)
        self.fused_shared_rs = True
        self.raw_multicast_address = raw_multicast_address
        self.raw_owner_rank = rank
        self.fused_reduce_vectors = reduce_vectors

    @cute.jit
    def load_shared_vectors(self, destinations, coordinates, predicates):
        addresses = ()
        for index in cutlass.range_constexpr(len(destinations)):
            coordinate = coordinates[index]
            addresses += (
                cutlass.Int64(self.raw_multicast_address)
                + (
                    cutlass.Int64(coordinate[0]) * 7168
                    + self.raw_owner_rank * 896
                    + coordinate[1]
                )
                * 2,
            )
        values = reduce_shared_vectors(addresses, predicates)
        for index in cutlass.range_constexpr(len(destinations)):
            registers = cute.make_rmem_tensor(8, cutlass.BFloat16)
            registers.store(values[index])
            target = cute.make_tensor(
                destinations[index].align(16), cute.make_layout(8)
            )
            cute.autovec_copy(registers, target)

    @cute.jit
    def load_shared_vector(self, destination, coordinate, active):
        # coordinate N is owner-local; every raw input has the original 7168
        # row stride. No reduction is repeated across owners or padded vectors.
        address = (
            cutlass.Int64(self.raw_multicast_address)
            + (
                cutlass.Int64(coordinate[0]) * 7168
                + self.raw_owner_rank * 896
                + coordinate[1]
            )
            * 2
        )
        values = reduce_shared_vector(address, active)
        registers = cute.make_rmem_tensor(8, cutlass.BFloat16)
        registers.store(values)
        target = cute.make_tensor(destination.align(16), cute.make_layout(8))
        # Keep the typed pointer so CuTe applies the destination swizzle.
        cute.autovec_copy(registers, target)

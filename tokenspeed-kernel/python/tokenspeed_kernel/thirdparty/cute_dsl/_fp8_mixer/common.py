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

"""Shared CuTe operations for prequantized FP8 mixers."""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T


@cute.experimental.jit
def _decode_residual(codes):
    # Every E4M3 value is exactly representable in BF16 and FP32.
    words = cute.recast_tensor(codes, cutlass.Uint16)
    decoded = cute.make_rmem_tensor(cute.size(codes), cutlass.Float32)
    for i in cutlass.range_constexpr(cute.size(codes) // 2):
        result = llvm.inline_asm(
            llvm.StructType.get_literal([T.f32(), T.f32()]),
            [words[i].ir_value()],
            "{ .reg .b32 pair; .reg .b16 lo, hi;\n"
            "cvt.rn.bf16x2.e4m3x2 pair, $2;\n"
            "mov.b32 {lo, hi}, pair;\n"
            "cvt.f32.bf16 $0, lo; cvt.f32.bf16 $1, hi; }",
            "=f,=f,h",
            has_side_effects=False,
            asm_dialect=0,
        )
        decoded[2 * i] = cutlass.Float32(llvm.extractvalue(T.f32(), result, [0]))
        decoded[2 * i + 1] = cutlass.Float32(llvm.extractvalue(T.f32(), result, [1]))
    return decoded


@cute.experimental.jit
def _copy_residual_16(destination, source, size):
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(destination.toint()).ir_value(),
            source.toint().ir_value(),
            size.ir_value(),
        ],
        "cp.async.ca.shared.global [$0], [$1], 16, $2;",
        "r,l,r,~{memory}",
        has_side_effects=True,
        asm_dialect=0,
    )


@cute.experimental.jit
def _commit_residual():
    llvm.inline_asm(
        None,
        [],
        "cp.async.commit_group;",
        "~{memory}",
        has_side_effects=True,
        asm_dialect=0,
    )


@cute.experimental.jit
def _wait_residual():
    llvm.inline_asm(
        None,
        [],
        "cp.async.wait_group 0;",
        "~{memory}",
        has_side_effects=True,
        asm_dialect=0,
    )


@cute.experimental.jit
def _acquire(ptr):
    return cutlass.Int64(
        llvm.inline_asm(
            T.i64(),
            [ptr.toint().ir_value()],
            "ld.global.acquire.gpu.u64 $0, [$1];",
            "=l,l,~{memory}",
            has_side_effects=True,
            asm_dialect=0,
        )
    )


@cute.experimental.jit
def _release(ptr, value):
    llvm.inline_asm(
        None,
        [ptr.toint().ir_value(), value.ir_value()],
        "st.global.release.gpu.u64 [$0], $1;",
        "l,l,~{memory}",
        has_side_effects=True,
        asm_dialect=0,
    )


@cute.experimental.jit
def _invalidate(barriers, count: cutlass.Constexpr):
    with cute.arch.elect_one():
        for index in cutlass.range_constexpr(count):
            address = cutlass.Int32((barriers + index).toint())
            llvm.inline_asm(
                None,
                [address.ir_value()],
                "mbarrier.inval.shared::cta.b64 [$0];",
                "r,~{memory}",
                has_side_effects=True,
                asm_dialect=0,
            )


@cute.experimental.jit
def _operand_storage(arena, first_layout, second_layout):
    dtype = cutlass.Float8E4M3FN
    first = cute.make_tensor(
        cute.make_ptr(
            dtype,
            arena.iterator.toint(),
            cute.AddressSpace.smem,
            assumed_align=1024,
            swizzle_=cute.get_swizzle_portion(first_layout),
        ),
        cute.get_nonswizzle_portion(first_layout),
    )
    second = cute.make_tensor(
        cute.make_ptr(
            dtype,
            arena.iterator.toint() + cute.cosize(first_layout),
            cute.AddressSpace.smem,
            assumed_align=1024,
            swizzle_=cute.get_swizzle_portion(second_layout),
        ),
        cute.get_nonswizzle_portion(second_layout),
    )
    return first, second


@cute.experimental.jit
def _operand_at(arena, layout, offset):
    return cute.make_tensor(
        cute.make_ptr(
            cutlass.Float8E4M3FN,
            arena.iterator.toint() + offset,
            cute.AddressSpace.smem,
            assumed_align=1024,
            swizzle_=cute.get_swizzle_portion(layout),
        ),
        cute.get_nonswizzle_portion(layout),
    )


@cute.experimental.jit
def _load_f32(ptr):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [ptr.toint().ir_value()],
            "ld.global.cg.f32 $0, [$1];",
            "=f,l,~{memory}",
            has_side_effects=True,
            asm_dialect=0,
        )
    )


@cute.experimental.jit
def _x_scale(scales, row):
    index = cutlass.Int64(0)
    if scales.shape[0] != 1:
        index = cutlass.Int64(row)
    return _load_f32(scales.iterator + index)


@cute.experimental.jit
def _mul_hidden_float_pair(lhs0, lhs1, rhs0, rhs1):
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [lhs0.ir_value(), lhs1.ir_value(), rhs0.ir_value(), rhs1.ir_value()],
        "{ .reg .b64 a,b,c; mov.b64 a, {$2,$3}; mov.b64 b, {$4,$5}; "
        "mul.rn.f32x2 c,a,b; mov.b64 {$0,$1},c; }",
        "=f,=f,f,f,f,f",
        has_side_effects=False,
        asm_dialect=0,
    )
    return cutlass.Float32(llvm.extractvalue(T.f32(), result, [0])), cutlass.Float32(
        llvm.extractvalue(T.f32(), result, [1])
    )


@cute.experimental.jit
def _store_private_epoch(ptr, value):
    llvm.inline_asm(
        None,
        [ptr.toint().ir_value(), value.ir_value()],
        "st.global.cg.u64 [$0], $1;",
        "l,l,~{memory}",
        has_side_effects=True,
        asm_dialect=0,
    )

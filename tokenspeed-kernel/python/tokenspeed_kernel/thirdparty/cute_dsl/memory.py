# MIT License
#
# Copyright (c) 2026 LightSeek Foundation <contact@lightseek.org>
# Copyright (c) 2026 MiniMax
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Aligned global-memory primitives shared by CuTe DSL kernels."""

import cutlass.cute as cute
from cutlass import Float32, Int64
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def load_global_f16x4_as_f32(
    pointer: Int64, is_bf16: bool, *, loc=None, ip=None
) -> cute.TensorSSA:
    """Read four contiguous 16-bit values into a four-element FP32 TensorSSA.

    ``pointer`` is a valid, eight-byte-aligned global byte address. ``is_bf16``
    selects BF16 when true and FP16 otherwise; it is fixed during compilation.
    The caller owns bounds checks. Conversion preserves element order.
    """
    kind = "bf16" if is_bf16 else "f16"
    code = "{ .reg .b32 lo, hi; .reg .b16 e0,e1,e2,e3;\n"
    code += "ld.global.v2.b32 {lo,hi}, [$4];\n"
    code += "mov.b32 {e0,e1},lo; mov.b32 {e2,e3},hi;\n"
    for i in range(4):
        code += f"cvt.f32.{kind} ${i},e{i};\n"
    code += "}"
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [pointer.ir_value(loc=loc, ip=ip)],
        code,
        "=f,=f,=f,=f,l,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    values = [llvm.extractvalue(T.f32(), result, [i], loc=loc, ip=ip) for i in range(4)]
    return cute.TensorSSA(
        vector.from_elements(ir.VectorType.get([4], T.f32()), values, loc=loc, ip=ip),
        4,
        Float32,
    )


@dsl_user_op
def store_global_f32x4_as_fp8_e4m3(
    gmem_ptr: cute.Pointer,
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Convert four FP32 values to E4M3 and write one aligned 32-bit word.

    ``gmem_ptr`` addresses four valid bytes of global memory and is four-byte
    aligned. Values are stored in v0, v1, v2, v3 order, with round-to-nearest-even
    and finite saturation. The caller applies any quantization multiplier and
    owns bounds checks. Returns no value.
    """
    llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [
            gmem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            Float32(v0).ir_value(loc=loc, ip=ip),
            Float32(v1).ir_value(loc=loc, ip=ip),
            Float32(v2).ir_value(loc=loc, ip=ip),
            Float32(v3).ir_value(loc=loc, ip=ip),
        ],
        "{\n"
        ".reg .b16 h0, h1;\n"
        ".reg .b32 p0;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h0, $6, $5;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h1, $8, $7;\n"
        "mov.b32 p0, {h0, h1};\n"
        "st.global.b32 [$4], p0;\n"
        "}\n"
        "mov.f32 $0, 0f00000000; mov.f32 $1, 0f00000000; "
        "mov.f32 $2, 0f00000000; mov.f32 $3, 0f00000000;",
        "=f,=f,=f,=f,l,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

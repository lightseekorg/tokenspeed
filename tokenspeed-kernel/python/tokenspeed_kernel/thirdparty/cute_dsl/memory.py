# MIT License
#
# Copyright (c) 2026 LightSeek Foundation <contact@lightseek.org>
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

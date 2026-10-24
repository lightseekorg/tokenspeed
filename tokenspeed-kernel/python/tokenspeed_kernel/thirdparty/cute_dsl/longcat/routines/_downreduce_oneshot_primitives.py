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

"""Sentinel transport primitives for fused DownReduce.

These primitives provide word sanitization, polling and publication for
the one-shot DownReduce transport.
"""

import cutlass.cute as cute
from cutlass import Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cutlass_dsl import T, dsl_user_op


@cute.jit
def sanitize_oneshot_word(word):
    if word & Uint32(0xFFFF) == Uint32(0x8000):
        word = word & Uint32(0xFFFF0000)
    if word >> Uint32(16) == Uint32(0x8000):
        word = word & Uint32(0x0000FFFF)
    return word


@cute.jit
def oneshot_word_is_dirty(word):
    return word == Uint32(0x80000000)


@dsl_user_op
def store_oneshot_fragment(address: Int64, packed, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            address.ir_value(loc=loc, ip=ip),
            *[packed[i].ir_value(loc=loc, ip=ip) for i in range(4)],
        ],
        "multimem.st.weak.global.v4.bf16x2 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_oneshot_fragment(pointer: cute.Pointer, *, loc=None, ip=None):
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [pointer.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)],
        "ld.volatile.global.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l,~{memory}",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )
    packed = vector.from_elements(
        ir.VectorType.get([4], T.i32(), loc=loc),
        [llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip) for i in range(4)],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(packed, 4, Uint32)

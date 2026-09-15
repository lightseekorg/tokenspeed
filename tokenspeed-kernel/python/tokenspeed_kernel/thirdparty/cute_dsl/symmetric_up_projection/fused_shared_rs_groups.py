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

"""Independent NVLS requests issued together before any shared result is consumed."""

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cutlass_dsl import T, dsl_user_op
from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.primitives import (
    packed_u32x4_to_bf16x8,
)


@dsl_user_op
def reduce_shared_vectors(addresses, predicates, *, loc=None, ip=None):
    count = len(addresses)
    assert count in (2, 4, 8) and len(predicates) == count
    operands = []
    instructions = ["{"]
    # Copy every input before touching outputs, avoiding =r early-clobber
    # hazards even when the register allocator reuses a predicate input.
    for index in range(count):
        operands.extend((addresses[index], cutlass.Uint32(predicates[index])))
        instructions.extend(
            (
                f".reg .u64 address_{index};",
                f".reg .pred valid_{index};",
                f"mov.u64 address_{index}, ${4 * count + 2 * index};",
                f"setp.ne.u32 valid_{index}, ${4 * count + 2 * index + 1}, 0;",
            )
        )
    instructions.extend(f"mov.b32 ${index}, 0;" for index in range(4 * count))
    for index in range(count):
        words = ", ".join(f"${4 * index + word}" for word in range(4))
        instructions.append(
            f"@valid_{index} multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2 "
            f"{{{words}}}, [address_{index}];"
        )
    instructions.append("}")
    loaded = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * (4 * count)),
        [value.ir_value(loc=loc, ip=ip) for value in operands],
        "\n".join(instructions),
        ",".join(["=r"] * (4 * count) + ["l", "r"] * count),
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    result = []
    for index in range(count):
        packed = vector.from_elements(
            ir.VectorType.get([4], T.i32(), loc=loc),
            [
                llvm.extractvalue(T.i32(), loaded, [4 * index + word], loc=loc, ip=ip)
                for word in range(4)
            ],
            loc=loc,
            ip=ip,
        )
        result.append(
            packed_u32x4_to_bf16x8(
                cute.TensorSSA(packed, 4, cutlass.Uint32), loc=loc, ip=ip
            )
        )
    return tuple(result)

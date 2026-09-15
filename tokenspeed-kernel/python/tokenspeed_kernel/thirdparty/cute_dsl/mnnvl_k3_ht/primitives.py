# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tail-predicated MNNVL operations derived from FlashInfer 0.6.18."""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def ldmc_bf16x8_predicated(
    address: Int64,
    predicate: Int32,
    *,
    loc=None,
    ip=None,
):
    """Reduce-load eight BF16 values, returning zeros when predicated off."""

    loaded = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [
            address.ir_value(loc=loc, ip=ip),
            Int32(predicate).ir_value(loc=loc, ip=ip),
        ],
        (
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.s32 p, $5, 0;\n\t"
            "@!p mov.u32 $0, 0;\n\t"
            "@!p mov.u32 $1, 0;\n\t"
            "@!p mov.u32 $2, 0;\n\t"
            "@!p mov.u32 $3, 0;\n\t"
            "@p multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2 "
            "{$0, $1, $2, $3}, [$4];\n\t"
            "}"
        ),
        "=r,=r,=r,=r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    packed = vector.from_elements(
        ir.VectorType.get([4], T.i32(), loc=loc),
        [
            llvm.extractvalue(T.i32(), loaded, [index], loc=loc, ip=ip)
            for index in range(4)
        ],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(packed, 4, Uint32)


@dsl_user_op
def stmc_bf16x8_predicated(
    address: Int64,
    values,
    predicate: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Multicast-store eight BF16 values only when ``predicate`` is true."""

    words = [values[index].ir_value(loc=loc, ip=ip) for index in range(4)]
    llvm.inline_asm(
        None,
        [
            address.ir_value(loc=loc, ip=ip),
            *words,
            Int32(predicate).ir_value(loc=loc, ip=ip),
        ],
        (
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.s32 p, $5, 0;\n\t"
            "@p multimem.st.relaxed.sys.global.v4.bf16x2 "
            "[$0], {$1, $2, $3, $4};\n\t"
            "}"
        ),
        "l,r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


__all__ = ["ldmc_bf16x8_predicated", "stmc_bf16x8_predicated"]

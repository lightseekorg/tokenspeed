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

"""Optional-predicate memory operations for the vendored FlashInfer HT kernel."""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def load_global_u32x4(
    pointer: cute.Pointer,
    predicate: Int32 | None = None,
    *,
    volatile: cutlass.Constexpr[bool] = False,
    loc=None,
    ip=None,
):
    """Load four words; an omitted predicate loads normally, false returns zeros."""

    predicate = Int32(1) if predicate is None else Int32(predicate)
    address = pointer.toint(loc=loc, ip=ip)
    opcode = "ld.volatile.global.v4.u32" if volatile else "ld.global.v4.u32"
    loaded = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [
            address.ir_value(loc=loc, ip=ip),
            predicate.ir_value(loc=loc, ip=ip),
        ],
        (
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.s32 p, $5, 0;\n\t"
            "@!p mov.u32 $0, 0;\n\t"
            "@!p mov.u32 $1, 0;\n\t"
            "@!p mov.u32 $2, 0;\n\t"
            "@!p mov.u32 $3, 0;\n\t"
            f"@p {opcode} {{$0, $1, $2, $3}}, [$4];\n\t"
            "}"
        ),
        "=r,=r,=r,=r,l,r",
        has_side_effects=volatile,
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
def ldmc_bf16x8(
    address: Int64,
    predicate: Int32 | None = None,
    *,
    loc=None,
    ip=None,
):
    """Reduce-load eight BF16s; an omitted predicate loads, false returns zeros."""

    predicate = Int32(1) if predicate is None else Int32(predicate)
    loaded = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [
            address.ir_value(loc=loc, ip=ip),
            predicate.ir_value(loc=loc, ip=ip),
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
def stmc_bf16x8(
    address: Int64,
    values,
    predicate: Int32 | None = None,
    *,
    loc=None,
    ip=None,
) -> None:
    """Multicast-store eight BF16s; an omitted predicate stores, false skips."""

    predicate = Int32(1) if predicate is None else Int32(predicate)
    words = [values[index].ir_value(loc=loc, ip=ip) for index in range(4)]
    llvm.inline_asm(
        None,
        [
            address.ir_value(loc=loc, ip=ip),
            *words,
            predicate.ir_value(loc=loc, ip=ip),
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


__all__ = ["load_global_u32x4", "ldmc_bf16x8", "stmc_bf16x8"]

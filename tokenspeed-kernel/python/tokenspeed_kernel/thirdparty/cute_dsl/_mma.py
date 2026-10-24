# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared register-fragment MMA and FP8 conversion implementations."""

import cutlass.cute as cute
from cutlass import BFloat16, Float8E4M3FN, Float16, Float32, Int32, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cutlass_dsl import T

_CUTE_TO_PTX_DTYPE = {
    BFloat16: "bf16",
    Float16: "f16",
    Float8E4M3FN: "e4m3",
    Float32: "f32",
}


def _mma_sync(a, b, c: cute.Tensor, *, loc, ip, vector_kwargs):
    a_ty = _CUTE_TO_PTX_DTYPE[a.element_type]
    b_ty = _CUTE_TO_PTX_DTYPE[b.element_type]
    c_ty = _CUTE_TO_PTX_DTYPE[c.element_type]
    mlir_ty = c.element_type.mlir_type
    K = 256 // a.element_type.width  # 32B

    # Recast expects tensor-backed fragments; materialize SSA fragments here so
    # callsites can pass converted FP8 fragments directly.
    if isinstance(a, cute.TensorSSA):
        a_ = cute.make_rmem_tensor_like(a)
        a_.store(a, loc=loc, ip=ip)
        a = a_
    if isinstance(b, cute.TensorSSA):
        b_ = cute.make_rmem_tensor_like(b)
        b_.store(b, loc=loc, ip=ip)
        b = b_

    a = cute.recast_tensor(a, Int32, loc=loc, ip=ip)
    b = cute.recast_tensor(b, Int32, loc=loc, ip=ip)
    out = llvm.inline_asm(
        llvm.StructType.get_literal([mlir_ty] * 4),
        [a[i].ir_value(loc=loc, ip=ip) for i in range(4)]
        + [b[i].ir_value(loc=loc, ip=ip) for i in range(2)]
        + [c[i].ir_value(loc=loc, ip=ip) for i in range(4)],
        f"mma.sync.aligned.m16n8k{K}.row.col.{c_ty}.{a_ty}.{b_ty}.{c_ty} "
        "{$0, $1, $2, $3}, "
        "{$4, $5, $6, $7}, "
        "{$8, $9}, "
        "{$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    vec = vector.from_elements(
        ir.VectorType.get([4], mlir_ty, loc=loc),
        [llvm.extractvalue(mlir_ty, out, [i], loc=loc, ip=ip) for i in range(4)],
        loc=loc,
        **vector_kwargs,
    )
    return cute.TensorSSA(vec, 4, c.element_type)


def _fp8x4_to_fp16x4(x: Uint32, *, loc, ip, vector_kwargs) -> cute.TensorSSA:
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [x.ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b16 lo, hi;\n\t"
        "mov.b32 {lo, hi}, $2;\n\t"
        "cvt.rn.f16x2.e4m3x2 $0, lo;\n\t"
        "cvt.rn.f16x2.e4m3x2 $1, hi;\n\t"
        "}\n",
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    vec = vector.from_elements(
        ir.VectorType.get([2], T.i32(), loc=loc),
        [llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip) for i in range(2)],
        loc=loc,
        **vector_kwargs,
    )
    return cute.TensorSSA(vec, 2, Uint32)

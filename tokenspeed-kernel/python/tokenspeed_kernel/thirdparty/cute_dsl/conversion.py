# Copyright (c) 2026 MiniMax
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


from typing import Type

import cutlass
import cutlass.cute as cute
from cutlass import Float32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def cvt_e4m3x2_bf16x2(packed: cutlass.Int16, *, loc=None, ip=None) -> cutlass.Int32:
    """Convert two E4M3 bytes to packed BF16; requires PTX 9.2/SM100+."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [packed.ir_value(loc=loc, ip=ip)],
            "cvt.rn.bf16x2.e4m3x2 $0, $1;",
            "=r,h",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def convert_e4m3_bf16(src: cute.Tensor, dst: cute.Tensor, *, native: cutlass.Constexpr):
    """Convert equal-sized dense register tensors, explicitly selecting native PTX.

    Native conversion consumes adjacent FP8 pairs and preserves all finite
    values, including subnormals and signed zero, without an FP32 temporary.
    The FP32 conversion supports compilers predating PTX 9.2.
    """
    if cutlass.const_expr(native):
        packed_src = cute.recast_tensor(src, cutlass.Int16)
        packed_dst = cute.recast_tensor(dst, cutlass.Int32)
        for i in cutlass.range_constexpr(cute.size(packed_src)):
            # Keep the pair in a 16-bit operand: widening it first emits
            # redundant sign-extension permutations for every FP8 pair.
            packed_dst[i] = cvt_e4m3x2_bf16x2(packed_src[i])
    else:
        values = src.load().to(cutlass.Float32).to(cutlass.BFloat16)
        dst.store(values.reshape(dst.shape))


@dsl_user_op
def cvt_f16x2_f32(
    a: float | Float32, b: float | Float32, to_dtype: Type, *, loc=None, ip=None
) -> cutlass.Int32:
    """Round two FP32 values to BF16/FP16, packing a low and b high in Int32."""
    assert to_dtype in [
        cutlass.BFloat16,
        cutlass.Float16,
    ], "to_dtype must be BFloat16 or Float16"
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            f"cvt.rn.{'bf16x2' if to_dtype is cutlass.BFloat16 else 'f16x2'}.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )

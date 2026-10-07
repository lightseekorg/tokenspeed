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

# Upstream portions from FlashInfer PR #6177, _rubin_helpers.py:
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# BSD 3-Clause License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""SM107 helpers for the MLA FP8 decode kernel in ``mla_decode_fp8_sm107.py``.

Based on FlashInfer PR #6177:
https://github.com/flashinfer-ai/flashinfer/pull/6177
Comparison baseline: commit 4090e370e4d1a82750ef42ad3d21e172f3a1e44f,
``flashinfer/cute_dsl/attention/rubin_mtp/_rubin_helpers.py``.

Local changes:
- Retain only the four packed-FP16 softmax helpers used by the decode kernel.
- Remove unused packing/quantization helpers and their imports. The retained
  functions preserve the upstream PTX and arithmetic unchanged.

H96/Q4/Q8 adaptations and tuning are in ``mla_decode_fp8_sm107.py``; its
module docstring lists the kernel changes. Upstream copyright and license
notices are reproduced above.
"""

from typing import Optional, Tuple

from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cute.typing import Float16, Float32, Uint32
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def pack_f16x2(
    a: Float16,
    b: Float16,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Uint32:
    """Pack two Float16 values into one f16x2 Uint32 register."""
    vec_f16x2_type = ir.VectorType.get([2], Float16.mlir_type, loc=loc)
    a_val = Float16(a).ir_value(loc=loc, ip=ip)
    b_val = Float16(b).ir_value(loc=loc, ip=ip)
    vec = vector.from_elements(vec_f16x2_type, (a_val, b_val), loc=loc, ip=ip)
    return Uint32(llvm.bitcast(T.i32(), vec, loc=loc, ip=ip))


@dsl_user_op
def add_packed_f16x2_u32(
    a: Uint32,
    b: Uint32,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Uint32:
    """Add two packed f16x2 values stored in Uint32 registers."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
            "add.f16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def reduce_sum_packed_f16x2_to_f32(
    packed: Uint32,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Float32:
    """Unpack a f16x2 Uint32 register and return the f32 sum of both lanes."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(packed).ir_value(loc=loc, ip=ip)],
            "{\n\t"
            ".reg .f16 lo, hi;\n\t"
            ".reg .f32 lo_f32, hi_f32;\n\t"
            "mov.b32 {lo, hi}, $1;\n\t"
            "cvt.f32.f16 lo_f32, lo;\n\t"
            "cvt.f32.f16 hi_f32, hi;\n\t"
            "add.f32 $0, lo_f32, hi_f32;\n\t"
            "}\n",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def softmax_f32x4_to_f16x2x2_and_e4m3x4(
    a0: Float32,
    a1: Float32,
    a2: Float32,
    a3: Float32,
    packed_b: Uint32,
    packed_c: Uint32,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Tuple[Uint32, Uint32, Uint32]:
    """Fused f32x4 softmax step returning two f16x2 sums and packed e4m3x4.

    Intended for the non-correction softmax stage where the accumulated error
    from FP16 mantissa rounding (via ``ex2.approx.f16x2``) does not compound.
    """
    a0_val = Float32(a0).ir_value(loc=loc, ip=ip)
    a1_val = Float32(a1).ir_value(loc=loc, ip=ip)
    a2_val = Float32(a2).ir_value(loc=loc, ip=ip)
    a3_val = Float32(a3).ir_value(loc=loc, ip=ip)
    b_val = Uint32(packed_b).ir_value(loc=loc, ip=ip)
    c_val = Uint32(packed_c).ir_value(loc=loc, ip=ip)

    f16x2_0 = llvm.inline_asm(
        T.i32(),
        [a0_val, a1_val, b_val, c_val],
        "{\n\t"
        ".reg .b32 packed_a, fma_result;\n\t"
        "cvt.rn.f16x2.f32 packed_a, $2, $1;\n\t"
        "fma.rn.f16x2 fma_result, packed_a, $3, $4;\n\t"
        "ex2.approx.f16x2 $0, fma_result;\n\t"
        "}\n",
        "=r,f,f,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    f16x2_1 = llvm.inline_asm(
        T.i32(),
        [a2_val, a3_val, b_val, c_val],
        "{\n\t"
        ".reg .b32 packed_a, fma_result;\n\t"
        "cvt.rn.f16x2.f32 packed_a, $2, $1;\n\t"
        "fma.rn.f16x2 fma_result, packed_a, $3, $4;\n\t"
        "ex2.approx.f16x2 $0, fma_result;\n\t"
        "}\n",
        "=r,f,f,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    fp8x4 = llvm.inline_asm(
        T.i32(),
        [f16x2_0, f16x2_1],
        "{\n\t"
        ".reg .b16 e0, e1;\n\t"
        "cvt.rn.satfinite.e4m3x2.f16x2 e0, $1;\n\t"
        "cvt.rn.satfinite.e4m3x2.f16x2 e1, $2;\n\t"
        "mov.b32 $0, {e0, e1};\n\t"
        "}\n",
        "=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )

    return Uint32(f16x2_0), Uint32(f16x2_1), Uint32(fp8x4)

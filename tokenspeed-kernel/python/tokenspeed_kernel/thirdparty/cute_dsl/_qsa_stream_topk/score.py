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

"""Exact FP32 head reductions shared by warp and tensor-memory scoring."""

from __future__ import annotations

import cutlass
from cutlass import cute
from cutlass._mlir.dialects import llvm as _llvm
from cutlass.cutlass_dsl import dsl_user_op


@dsl_user_op
def _positive_f32(value: cutlass.Float32, *, loc=None, ip=None):
    # Numeric max preserves the original suppression of NaN and negative zero.
    result = _llvm.inline_asm(
        cutlass.Float32.mlir_type,
        [value.ir_value(loc=loc, ip=ip)],
        "max.f32 $0, $1, 0f00000000;",
        "=f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Float32(result)


@dsl_user_op
def _add_head_pairs(
    a0: cutlass.Float32,
    a1: cutlass.Float32,
    a2: cutlass.Float32,
    a3: cutlass.Float32,
    b0: cutlass.Float32,
    b1: cutlass.Float32,
    b2: cutlass.Float32,
    b3: cutlass.Float32,
    *,
    loc=None,
    ip=None,
):
    # Each half keeps its original left-to-right FP32 rounding order.
    result = _llvm.inline_asm(
        cutlass.Uint64.mlir_type,
        [v.ir_value(loc=loc, ip=ip) for v in (a0, a1, a2, a3, b0, b1, b2, b3)],
        "{ .reg .b64 sum, value; "
        "mov.b64 sum, {$1, $5}; mov.b64 value, {$2, $6}; "
        "add.rn.f32x2 sum, sum, value; "
        "mov.b64 value, {$3, $7}; add.rn.f32x2 sum, sum, value; "
        "mov.b64 value, {$4, $8}; add.rn.f32x2 $0, sum, value; }",
        "=l,f,f,f,f,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Uint64(result)


@cute.jit
def _paired_head_sum(
    a0: cutlass.Float32,
    a1: cutlass.Float32,
    a2: cutlass.Float32,
    a3: cutlass.Float32,
    b0: cutlass.Float32,
    b1: cutlass.Float32,
    b2: cutlass.Float32,
    b3: cutlass.Float32,
):
    """Sum two independent four-head scores in their original FP32 order."""
    return _add_head_pairs(
        _positive_f32(a0),
        _positive_f32(a1),
        _positive_f32(a2),
        _positive_f32(a3),
        _positive_f32(b0),
        _positive_f32(b1),
        _positive_f32(b2),
        _positive_f32(b3),
    )

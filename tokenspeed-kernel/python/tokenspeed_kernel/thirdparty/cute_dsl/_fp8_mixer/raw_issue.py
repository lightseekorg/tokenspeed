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

"""Issue one dense native FP8 MMA from an explicitly elected thread."""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import tcgen05
from cutlass.experimental.primitives import Tcgen05InstrDesc


@cute.experimental.jit
def fp8_instruction(m: cutlass.Constexpr, n: cutlass.Constexpr):
    return Tcgen05InstrDesc.build(
        sparse_id2=0,
        sparse_flag=0,
        saturate=0,
        sparse_format=0,
        c_dtype=cutlass.Float32,
        a_dtype=cutlass.Float8E4M3FN,
        b_dtype=cutlass.Float8E4M3FN,
        a_negate=0,
        b_negate=0,
        a_major=0,
        b_major=0,
        n_dim=n,
        m_dim=m,
        k_dim=0,
        max_shift=0,
    )


@cute.experimental.jit
def fp8_descriptor_at(fragment, block, stage):
    # Descriptor-view layouts already express offsets in encoded address units.
    # Keep conversion at a static origin; dynamic selection then uses the same
    # canonical layout arithmetic without converting a sliced descriptor view.
    origin = tcgen05.smem_descriptor_to_int(fragment[None, None, 0, 0].iterator)
    return origin + cutlass.Int64(fragment.layout((0, 0, block, stage)))


@cute.experimental.jit
def fp8_mma_encoded(acc, da, db, instruction, accumulate):
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(acc.iterator.toint()).ir_value(),
            da.ir_value(),
            db.ir_value(),
            cutlass.Int32(instruction).ir_value(),
            cutlass.Int32(accumulate).ir_value(),
        ],
        "{ .reg .pred p; setp.ne.u32 p, $4, 0; "
        "tcgen05.mma.cta_group::1.kind::f8f6f4 [$0], $1, $2, $3, p; }",
        "r,l,l,r,r,~{memory}",
        has_side_effects=True,
        asm_dialect=0,
    )

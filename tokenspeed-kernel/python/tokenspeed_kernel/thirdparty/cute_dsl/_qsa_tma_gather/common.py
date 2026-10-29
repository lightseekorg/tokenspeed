# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith


def row_max_for_exp2(total_max):
    """Canonical masked-softmax row-max guard (FlashAttention / cuDNN form)."""
    still_dead = total_max == cutlass.Float32(float("-inf"))
    return cutlass.Float32(
        arith.select(
            still_dead.ir_value(),
            cutlass.Float32(0.0).ir_value(),
            total_max.ir_value(),
        )
    )


def _vec(ptr, n):
    return cute.make_tensor(ptr, cute.make_layout((n,), stride=(1,)))


def _bshd(ptr, batch, seq, heads, d, strides):
    """View BSHD storage with runtime 64-bit outer strides."""
    bs, ss, hs = (
        cutlass.Int64(s) for s in strides
    )  # Int64 leaves: TMA-unit scaling happens in the leaf's width
    return cute.make_tensor(
        ptr, cute.make_layout((batch, seq, heads, d), stride=(bs, ss, hs, 1))
    )

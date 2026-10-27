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

"""Merge normalized FP32 split outputs using their natural-log LSE."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith

_HEAD_DIM = 256


@cute.jit
def _select(condition, yes, no):
    return cutlass.Float32(
        arith.select(condition.ir_value(), yes.ir_value(), no.ir_value())
    )


@cute.kernel
def _combine(
    partial,
    lse,
    output,
    batch,
    queries,
    heads,
    strides,
    splits: cutlass.Constexpr[int],
    enable_pdl: cutlass.Constexpr[bool],
):
    if cutlass.const_expr(enable_pdl):
        cute.arch.griddepcontrol_wait()
    row = cute.arch.block_idx()[0]
    dim = cute.arch.thread_idx()[0]
    b = row // (queries * heads)
    q = row // heads % queries
    h = row % heads
    total_rows = cutlass.Int64(batch) * queries * heads
    maximum = cutlass.Float32(float("-inf"))
    for split in cutlass.range_constexpr(splits):
        value = lse[
            (cutlass.Int64(split) * batch + b) * heads * queries + h * queries + q
        ]
        maximum = cute.math.max(maximum, value)
    # Empty splits store LSE=-inf; avoid exp(-inf - -inf), including all-empty rows.
    maximum = _select(
        maximum == cutlass.Float32(float("-inf")), cutlass.Float32(0), maximum
    )
    numerator = cutlass.Float32(0)
    denominator = cutlass.Float32(0)
    for split in cutlass.range_constexpr(splits):
        value = lse[
            (cutlass.Int64(split) * batch + b) * heads * queries + h * queries + q
        ]
        weight = cute.math.exp2(
            (value - maximum) * cutlass.Float32(1.4426950408889634), fastmath=True
        )
        numerator += weight * partial[(split * total_rows + row) * _HEAD_DIM + dim]
        denominator += weight
    safe_denominator = cute.math.max(denominator, cutlass.Float32(1e-30))
    value = _select(
        denominator > cutlass.Float32(0),
        numerator / safe_denominator,
        cutlass.Float32(0),
    )
    bs, qs, hs = strides
    offset = cutlass.Int64(b) * bs + cutlass.Int64(q) * qs + cutlass.Int64(h) * hs + dim
    output[offset] = cutlass.BFloat16(value)
    if cutlass.const_expr(enable_pdl):
        cute.arch.griddepcontrol_launch_dependents()


@cute.jit
def _launch(
    partial_ptr: cute.Pointer,
    lse_ptr: cute.Pointer,
    output_ptr: cute.Pointer,
    batch: cutlass.Int32,
    queries: cutlass.Int32,
    heads: cutlass.Int32,
    strides: tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    splits: cutlass.Constexpr[int],
    enable_pdl: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    count = batch * queries * heads
    partial = cute.make_tensor(
        partial_ptr,
        cute.make_layout((cutlass.Int64(splits) * count * _HEAD_DIM,), stride=(1,)),
    )
    lse = cute.make_tensor(
        lse_ptr, cute.make_layout((cutlass.Int64(splits) * count,), stride=(1,))
    )
    # The output address uses its own strides; the flat view is only for pointer indexing.
    span = (
        cutlass.Int64(batch - 1) * strides[0]
        + cutlass.Int64(queries - 1) * strides[1]
        + cutlass.Int64(heads - 1) * strides[2]
        + _HEAD_DIM
    )
    output = cute.make_tensor(output_ptr, cute.make_layout((span,), stride=(1,)))
    # Each thread combines one head-dimension element.
    _combine(
        partial, lse, output, batch, queries, heads, strides, splits, enable_pdl
    ).launch(
        grid=(count, 1, 1), block=(_HEAD_DIM, 1, 1), stream=stream, use_pdl=enable_pdl
    )


def compile_combine(*, split_kv: int, enable_pdl: bool):
    gmem = cute.AddressSpace.gmem
    return cute.compile(
        _launch,
        cute.runtime.make_ptr(cutlass.Float32, 16, gmem, assumed_align=16),
        cute.runtime.make_ptr(cutlass.Float32, 16, gmem, assumed_align=16),
        cute.runtime.make_ptr(cutlass.BFloat16, 16, gmem, assumed_align=16),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        (cutlass.Int64(0),) * 3,
        split_kv,
        enable_pdl,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
    )

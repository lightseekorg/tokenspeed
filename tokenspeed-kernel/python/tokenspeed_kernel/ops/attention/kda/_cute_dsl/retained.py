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

"""Retained-history arguments and loads shared by the CuTe verify kernels."""

import cutlass
from cutlass import Float32, Int32, Int64, cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


@cute.jit
def load_element(tensor, index, head, position, channel):
    offset = (
        Int64(index) * tensor.stride[0]
        + Int64(head) * tensor.stride[1]
        + Int64(position) * tensor.stride[2]
        + Int64(channel) * tensor.stride[3]
    )
    return cute.make_tensor(tensor.iterator + offset, cute.make_layout((1,)))[0].to(
        Float32
    )


@cute.jit
def retained_row(history, request):
    row = Int64(request)
    return Int64(history[5][row]), Int64(history[6][row])


@cute.jit
def retained_count(history, slot):
    """The accepted-record count stored for the request in ``slot``.

    Every layer of a request shares the slot's count. A verify applies
    ``applied_length`` of it.
    """
    length = history[4]
    return Int32(
        cute.make_tensor(
            length.iterator + Int64(slot) * length.stride[0], cute.make_layout((1,))
        )[0]
    )


@dsl_user_op
def applied_length(count: Int32, limit: int, *, loc=None, ip=None) -> Int32:
    """The records a verify applies of a stored ``count``.

    ``limit`` is the history capacity minus the verify window: a count
    outside [0, limit] overflowed, and the round's prepare folded its records
    into the live page, so the verify applies none.

    One select in inline PTX, placed by the caller: ptxas keeps it where it
    is written, so a kernel that loads the count early bounds it only where
    every warp uses it, and the loads issued before do not wait for the
    count. The compilers see no bound on the result either, so the record
    loads keep the schedule they have for the stored count.
    """
    if limit < 0:
        return Int32(0)
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(count).ir_value(loc=loc, ip=ip)],
            "{ .reg .pred p; setp.le.u32 p, $1, %d; selp.b32 $0, $1, 0, p; }" % limit,
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def retained_length(history, slot, limit: cutlass.Constexpr[int]):
    """``applied_length`` of the slot's count, for a caller that uses it at once."""
    return applied_length(retained_count(history, slot), limit)


def retained_arguments(state, history):
    return (
        state,
        history.correction,
        history.key,
        history.decay,
        history.length,
        history.live_state_indices,
        history.record_slots,
    )


def fake_retained_arguments(state, correction, key, *, batch, live_dtype, slot_dtype):
    from quack.compile_utils import make_fake_tensor

    decay = make_fake_tensor(Float32, key.shape, leading_dim=None)
    # One accepted-record count per request slot, beside its records.
    length = make_fake_tensor(Int32, (correction.shape[0], 1))
    live = make_fake_tensor(live_dtype, (batch,))
    record_slots = make_fake_tensor(slot_dtype, (batch,), leading_dim=None)
    return (state, correction, key, decay, length, live, record_slots)

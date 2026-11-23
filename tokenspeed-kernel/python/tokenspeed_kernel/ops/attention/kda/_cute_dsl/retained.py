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

from cutlass import Float32, Int32, Int64, cute


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
def retained_length(history, slot):
    """Accepted records the request in ``slot`` holds beyond its dense base."""
    length = history[4]
    return Int32(
        cute.make_tensor(
            length.iterator + Int64(slot) * length.stride[0], cute.make_layout((1,))
        )[0]
    )


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

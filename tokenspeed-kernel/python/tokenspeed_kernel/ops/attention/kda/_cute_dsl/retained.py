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

"""Shared FP32 logical-state initialization for the retained CuTe producers."""

import cutlass
from cutlass import Float32, Int32, Int64, cute


@cute.jit
def load_element(tensor, page, head, position, channel):
    offset = (
        Int64(page) * tensor.stride[0]
        + Int64(head) * tensor.stride[1]
        + Int64(position) * tensor.stride[2]
        + Int64(channel) * tensor.stride[3]
    )
    return cute.make_tensor(tensor.iterator + offset, cute.make_layout((1,)))[0].to(
        Float32
    )


@cute.jit
def retained_page(history, request):
    live = history[5]
    return Int64(
        cute.make_tensor(
            live.iterator + Int64(request) * live.stride[0], cute.make_layout((1,))
        )[0]
    )


@cute.jit
def retained_length(history, page):
    length = history[4]
    return Int32(
        cute.make_tensor(
            length.iterator + Int64(page) * length.stride[0], cute.make_layout((1,))
        )[0]
    )


@cute.jit
def initialize_page_tile(
    tile,
    history,
    page,
    head,
    value_start,
    rows: cutlass.Constexpr[int],
    keys: cutlass.Constexpr[int],
    tidx,
    threads: cutlass.Constexpr[int],
):
    """Load base plus accepted history in FP32 without changing persistent storage."""
    count = retained_length(history, page)
    for item in cutlass.range_constexpr(0, rows * keys, threads):
        offset = item + tidx
        if offset < rows * keys:
            value = value_start + offset // keys
            key = offset % keys
            current = load_element(history[0], page, head, value, key)
            for position in cutlass.range(count):
                u = load_element(history[1], page, head, position, value)
                k = load_element(history[2], page, head, position, key)
                decay = load_element(history[3], page, head, position, key)
                current = current * decay + u * k
            tile[offset] = current
    cute.arch.barrier()


def retained_arguments(state, history):
    return (
        state,
        history.correction,
        history.key,
        history.decay,
        history.length,
        history.live_state_indices,
    )


def fake_retained_arguments(state, correction, key, *, slots, batch, live_dtype):
    from quack.compile_utils import make_fake_tensor

    decay = make_fake_tensor(Float32, key.shape, leading_dim=None)
    length = make_fake_tensor(Int32, (slots, 1))
    live = make_fake_tensor(live_dtype, (batch,))
    return (state, correction, key, decay, length, live)


@cute.jit
def initialize_tile(
    tile,
    history,
    request,
    head,
    value_start,
    rows: cutlass.Constexpr[int],
    keys: cutlass.Constexpr[int],
    tidx,
    threads: cutlass.Constexpr[int],
):
    initialize_page_tile(
        tile,
        history,
        retained_page(history, request),
        head,
        value_start,
        rows,
        keys,
        tidx,
        threads,
    )

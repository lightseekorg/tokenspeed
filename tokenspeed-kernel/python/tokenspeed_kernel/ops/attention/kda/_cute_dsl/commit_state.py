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

"""Tiled CuTe retained-state consolidation at copy and capacity boundaries.

The previous same-round dense commit is now a materialization operation.
Candidate acceptance still uses the common nine-column convolution/length ABI.
"""

from __future__ import annotations

import functools

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float32, Int32, Int64, cute
from quack.compile_utils import make_fake_tensor
from tokenspeed_kernel.ops.attention.kda._cute_dsl.retained import (
    fake_retained_arguments,
    initialize_page_tile,
    retained_arguments,
    retained_length,
)

THREADS = 128
_DTYPES = {
    torch.float32: Float32,
    torch.bfloat16: BFloat16,
    torch.int32: Int32,
    torch.int64: Int64,
}


class ConsolidateHistory:
    def __init__(self, heads, keys, values, rows, capacity):
        self.heads = heads
        self.keys = keys
        self.values = values
        self.rows = rows
        self.capacity = capacity

    @cute.jit
    def __call__(
        self, history: tuple, reads: cute.Tensor, starts: cute.Tensor, stream: CUstream
    ):
        self.kernel(history, reads, starts).launch(
            grid=[self.values // self.rows, reads.shape[0], self.heads],
            block=[THREADS, 1, 1],
            smem=self.rows * self.keys * 4 + 128,
            stream=stream,
        )

    @cute.kernel
    def kernel(self, history: tuple, reads: cute.Tensor, starts: cute.Tensor):
        tile_id, request, head = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        row = Int64(request)
        source = Int64(reads[row])
        target = Int64(history[5][row])
        tokens = Int64(starts[row + 1]) - Int64(starts[row])
        if (source > 0) & (target > 0) & (tokens > 0):
            count = retained_length(history, source)
            if (source != target) | (count + tokens > self.capacity):
                smem = cutlass.memory.SmemAllocator()
                tile = smem.allocate_tensor(
                    Float32,
                    cute.make_layout((self.rows * self.keys,)),
                    byte_alignment=128,
                )
                value_start = tile_id * self.rows
                initialize_page_tile(
                    tile,
                    history,
                    source,
                    head,
                    value_start,
                    self.rows,
                    self.keys,
                    tidx,
                    THREADS,
                )
                state = history[0]
                for item in cutlass.range_constexpr(0, self.rows * self.keys, THREADS):
                    offset = item + tidx
                    if offset < self.rows * self.keys:
                        value = value_start + offset // self.keys
                        key = offset % self.keys
                        dst = (
                            target * state.stride[0]
                            + Int64(head) * state.stride[1]
                            + Int64(value) * state.stride[2]
                            + Int64(key) * state.stride[3]
                        )
                        cute.make_tensor(state.iterator + dst, cute.make_layout((1,)))[
                            0
                        ] = tile[offset].to(state.element_type)


@functools.cache
def _compile(
    heads,
    keys,
    values,
    rows,
    capacity,
    state_dtype,
    read_dtype,
    live_dtype,
    start_dtype,
):
    pages = cute.sym_int()
    batch = cute.sym_int()
    # Every arena axis is addressed through its runtime element stride.
    state = make_fake_tensor(
        _DTYPES[state_dtype], (pages, heads, values, keys), leading_dim=None
    )
    correction = make_fake_tensor(
        Float32, (pages, heads, capacity, values), leading_dim=None
    )
    key = make_fake_tensor(Float32, (pages, heads, capacity, keys), leading_dim=None)
    history = fake_retained_arguments(
        state, correction, key, slots=pages, batch=batch, live_dtype=_DTYPES[live_dtype]
    )
    reads = make_fake_tensor(_DTYPES[read_dtype], (batch,))
    starts = make_fake_tensor(_DTYPES[start_dtype], (cute.sym_int(),))
    return cute.compile(
        ConsolidateHistory(heads, keys, values, rows, capacity),
        history,
        reads,
        starts,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def consolidate_history(state, history, reads, starts):
    """Fold accepted history only for private copies or overflowing candidate windows.

    state is the FP32/BF16 page pool, history owns FP32 U/K/decay, and reads
    and starts give source pages and packed boundaries. Length publication is
    a separate ordered launch after every value tile completes.
    """
    _, heads, values, keys = state.shape
    rows = min(32, values)
    while values % rows:
        rows //= 2
    kernel = _compile(
        heads,
        keys,
        values,
        rows,
        history.correction.shape[2],
        state.dtype,
        reads.dtype,
        history.live_state_indices.dtype,
        starts.dtype,
    )
    kernel(retained_arguments(state, history), reads, starts)

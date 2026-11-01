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

"""State copies cross the NPU launch limit without changing per-layer addressing."""

import pytest
import torch

pytest.importorskip("torch_npu")
from tokenspeed_kernel.ops.kvcache.triton import copy_state_rows


@pytest.mark.parametrize("size", [8, 13, 32])
def test_large_interleaved_state_copy(size):
    if not torch.npu.is_available():
        pytest.skip("requires NPU")
    layers, width, pool = 42, 131072, size + 2
    host = torch.arange(pool * layers * width, dtype=torch.int32).view(
        pool, layers, width
    )
    source = host.npu()
    # Extra trailing values detect writes through a mistaken dense row stride.
    target = torch.empty((layers, pool, width + 8), dtype=torch.int32, device="npu")
    src_ptr = torch.tensor(
        [source[:, i].data_ptr() for i in range(layers)],
        dtype=torch.uint64,
        device="npu",
    )
    dst_ptr = torch.tensor(
        [target[i].data_ptr() for i in range(layers)], dtype=torch.uint64, device="npu"
    )
    src_stride = torch.full((layers,), layers * width, dtype=torch.int64, device="npu")
    dst_stride = torch.full((layers,), width + 8, dtype=torch.int64, device="npu")
    src_rows = torch.empty(layers * size, dtype=torch.int64, device="npu")
    dst_rows = torch.empty_like(src_rows)

    def invoke():
        copy_state_rows(
            src_ptr,
            dst_ptr,
            src_rows,
            dst_rows,
            row_bytes=width * 4,
            src_row_strides=src_stride,
            dst_row_strides=dst_stride,
        )

    src_rows.copy_(torch.arange(size, dtype=torch.int64).repeat(layers))
    dst_rows.copy_(torch.arange(size, dtype=torch.int64).repeat(layers))
    invoke()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        invoke()
    for masked in (False, True):
        sh = torch.arange(1, size + 1).repeat(layers).to(torch.int64)
        dh = torch.arange(size).repeat(layers).to(torch.int64)
        if masked:
            sh[::3] = -1
            dh[1::3] = -1
        target.fill_(-123)
        src_rows.copy_(sh)
        dst_rows.copy_(dh)
        graph.replay()
        actual = target.cpu()
        for layer in range(layers):
            expected = torch.full((pool, width + 8), -123, dtype=torch.int32)
            for row in range(size):
                s, d = int(sh[layer * size + row]), int(dh[layer * size + row])
                if d >= 0:
                    expected[d, :width] = host[s, layer] if s >= 0 else 0
            torch.testing.assert_close(actual[layer], expected, rtol=0, atol=0)

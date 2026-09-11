# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pytest
import torch
from tokenspeed_kernel.ops.kvcache import triton as kvcache


def test_range_staging_size_limits():
    stager = kvcache._ZeroRangeTableStager(capacity=4, min_rows=2)
    for count in (0, 1, 5):
        assert stager.acquire([(0, 1)] * count, torch.device("cuda", 0)) is None
    assert not stager.slots


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_range_staging_reserved_slots_and_streams(monkeypatch):
    stager = kvcache._ZeroRangeTableStager(capacity=4, min_rows=1)
    device = torch.device("cuda", torch.cuda.current_device())
    acquired = [stager.acquire([(i, 1)], device) for i in range(4)]
    assert all(item is not None for item in acquired)
    assert stager.acquire([(8, 1)], device) is None
    for table, index, stream in acquired:
        stager.release(device, index, stream)
    torch.cuda.synchronize()
    monkeypatch.setattr(kvcache, "_ZERO_RANGE_TABLE_STAGER", stager)
    buffers = [
        torch.full((256,), 7, dtype=torch.uint8, device=device) for _ in range(2)
    ]
    streams = [torch.cuda.Stream() for _ in buffers]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    for step in range(12):
        for buffer, stream in zip(buffers, streams):
            with torch.cuda.stream(stream):
                kvcache.zero_byte_ranges(buffer, [(step * 8, 4), (128 + step * 8, 4)])
    torch.cuda.synchronize()
    expected = torch.full((256,), 7, dtype=torch.uint8)
    for step in range(12):
        expected[step * 8 : step * 8 + 4] = 0
        expected[128 + step * 8 : 132 + step * 8] = 0
    assert all(torch.equal(buffer.cpu(), expected) for buffer in buffers)
    assert len(stager.slots[device]) == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_range_staging_capture_bypasses_storage():
    stager = kvcache._ZeroRangeTableStager(capacity=4, min_rows=1)
    device = torch.device("cuda", torch.cuda.current_device())
    graph = torch.cuda.CUDAGraph()
    buffer = torch.zeros(1, device=device)
    with torch.cuda.graph(graph):
        assert stager.acquire([(0, 1)], device) is None
        buffer.add_(1)
    assert not stager.slots
    graph.replay()
    assert buffer.item() == 1

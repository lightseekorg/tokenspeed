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


import pytest
import torch
import torch_npu  # noqa: F401
from tokenspeed_kernel_npu.ops.dsv41_candidate_gather_native import (
    _library,
    gather_candidates,
)


@pytest.fixture(autouse=True)
def npu_device():
    torch.npu.set_device(0)
    torch.set_num_threads(4)
    if _library() is None:
        pytest.skip("Build optional dsv41_candidate_gather kernel")


def reference(c, t, v, cache):
    c, t, v = [x.cpu().long() for x in (c, t, v)]
    col = torch.arange(c.shape[1])[None, :]
    pages = torch.gather(t, 1, (c // 8).clamp(0, t.shape[1] - 1))
    valid = (c >= 0) & (c * 8 < v[:, None]) & (pages > 0) & (pages < cache.shape[0])
    physical = torch.where(valid, pages, 0) * 8 + torch.where(valid, c, col) % 8
    # Gather only the needed blocks; do not copy the multi-GiB arena to CPU.
    return (
        cache.reshape(-1, 1024)
        .view(torch.int16)[physical.to(cache.device)]
        .cpu()
        .reshape(c.shape[0], -1, 128)
    )


@pytest.mark.parametrize(
    "n,width,count", [(1, 64, 128), (32, 1024, 2048), (192, 15936, 2048)]
)
def test_candidate_bits_and_mutable_graph(n, width, count):
    torch.manual_seed(129)
    pages = 65
    bits = torch.arange(pages * 64 * 128, dtype=torch.int32).to(torch.int16)
    cache = bits.view(torch.bfloat16).reshape(pages, 64, 128).npu()
    c = torch.randint(
        -2, min(width * 8, 6401), (n, count + 3), device="npu", dtype=torch.int64
    )[:, :count]
    t = torch.randint(-2, pages + 2, (n, width + 3), device="npu", dtype=torch.int32)[
        :, :width
    ]
    v = torch.full((n * 2,), 51203, device="npu", dtype=torch.int64)[::2]
    out = gather_candidates(c, t, v, cache)
    torch.testing.assert_close(
        out.view(torch.int16).cpu(), reference(c, t, v, cache), atol=0, rtol=0
    )
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = gather_candidates(c, t, v, cache)
    for step in range(3):
        v.fill_(step * 25003)
        c[:, ::7] = -1
        t[:, ::9] = 0
        graph.replay()
        torch.testing.assert_close(
            out.view(torch.int16).cpu(), reference(c, t, v, cache), atol=0, rtol=0
        )


def test_candidate_addresses_above_two_gib():
    pages = 131075
    cache = torch.empty((pages, 64, 128), device="npu", dtype=torch.bfloat16)
    cache[0].fill_(-7)
    cache[-1].copy_(
        torch.arange(64 * 128, device="npu", dtype=torch.int32)
        .to(torch.bfloat16)
        .reshape(64, 128)
    )
    c = torch.arange(128, device="npu", dtype=torch.int32).repeat(2, 1)
    c[:, ::11] = -1
    t = torch.full((2, 64), pages - 1, device="npu", dtype=torch.int32)
    v = torch.full((2,), 4096, device="npu", dtype=torch.int32)
    out = gather_candidates(c, t, v, cache)
    torch.testing.assert_close(
        out.view(torch.int16).cpu(), reference(c, t, v, cache), atol=0, rtol=0
    )


def test_candidate_gather_uses_input_device_stream():
    if torch.npu.device_count() < 2:
        pytest.skip("Requires two NPU devices")
    cache = torch.arange(2 * 64 * 128, dtype=torch.int32, device="npu:0")
    cache = cache.to(torch.bfloat16).reshape(2, 64, 128)
    c = torch.arange(8, dtype=torch.int32, device="npu:0").reshape(1, 8)
    t = torch.ones((1, 1), dtype=torch.int32, device="npu:0")
    v = torch.full((1,), 64, dtype=torch.int32, device="npu:0")
    torch.npu.synchronize(0)
    with torch.npu.device(1):
        out = gather_candidates(c, t, v, cache)
    torch.npu.synchronize(0)
    torch.testing.assert_close(
        out.view(torch.int16).cpu(), reference(c, t, v, cache), atol=0, rtol=0
    )

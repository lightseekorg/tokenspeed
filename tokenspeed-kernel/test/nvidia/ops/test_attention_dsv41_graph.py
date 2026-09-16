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
from tokenspeed_kernel.ops.attention import dsv41

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires an NVIDIA GPU",
)


def _make_index_cache(x):
    cache = torch.zeros(
        ((x.shape[0] + 63) // 64, 64, 68), dtype=torch.uint8, device=x.device
    )
    dsv41.cache_scatter(x, cache, torch.arange(x.shape[0], device=x.device), "index")
    return cache


def test_index_topk_graph_full_candidates_and_reindex():
    device = torch.device("cuda:0")
    torch.manual_seed(43)
    cache = _make_index_cache(
        torch.randn(256, 128, device=device, dtype=torch.bfloat16)
    )
    q = torch.randn(2, 2, 128, device=device, dtype=torch.bfloat16)
    weights = torch.rand(2, 2, device=device, dtype=torch.bfloat16)
    table = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], device=device, dtype=torch.int32)
    visible = torch.tensor([0, 0], device=device, dtype=torch.int32)

    def run():
        full = dsv41.index_topk(
            q, weights, cache, table, visible, None, 16, 4, 8, 2, 64, None, None
        )
        reindex = dsv41.index_topk(
            q, weights, cache, table, visible, full[2], 16, 0, 8, 2, 64, None, None
        )
        return full + reindex

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = run()
    torch.cuda.current_stream().wait_stream(stream)
    for lengths in ([17, 130], [256, 9], [0, 0], [65, 255]):
        q.normal_()
        weights.uniform_()
        visible.copy_(torch.tensor(lengths, device=device, dtype=torch.int32))
        table.copy_(table.flip(1))
        expected = run()
        graph.replay()
        for got, want in zip(output, expected, strict=True):
            torch.testing.assert_close(got, want, rtol=0, atol=0)

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
from tokenspeed_kernel.ops.attention.dsv41 import triton as impl


@pytest.mark.parametrize("n", [48, 64, 96, 97, 128, 160, 192, 256])
@pytest.mark.parametrize("hist", [8192, 131072])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_full_fp4_batched_vs_small_chunks(monkeypatch, n, hist, dtype):
    monkeypatch.setenv("TOKENSPEED_EXPERIMENT_SM120_DG_INDEXER", "1")
    monkeypatch.setenv("TOKENSPEED_EXPERIMENT_SM120_DEEPSELECT", "1")
    torch.manual_seed(53)
    q = torch.randn(n, 32, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.rand(n, 32, device="cuda", dtype=dtype)
    cache = torch.zeros(hist // 64, 64, 68, device="cuda", dtype=torch.uint8)
    dsv41.cache_scatter(
        torch.randn(hist, 128, device="cuda", dtype=torch.bfloat16),
        cache,
        torch.arange(hist, device="cuda"),
        "index",
    )
    table = torch.arange(hist // 64, device="cuda", dtype=torch.int32).repeat(n, 1)
    table[:, 3] = -1
    visible = torch.randint(1, hist, (n,), device="cuda", dtype=torch.int32)
    visible[0] = 0
    visible[1] = hist - 1

    def run(a, b):
        return impl._index_topk(
            q[a:b],
            w[a:b],
            cache,
            table[a:b],
            visible[a:b],
            None,
            512,
            2048,
            8,
            256,
            256,
            None,
            None,
            None,
            scan_num_warps=4,
            dense_selector=None,
        )

    run(0, n)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        got = run(0, n)
    for _ in range(2):
        visible.random_(0, hist)
        visible[0] = 0
        visible[1] = hist - 1
        q.normal_()
        g.replay()
        torch.cuda.synchronize()
        parts = [run(a, min(a + 24, n)) for a in range(0, n, 24)]
        for k in range(4):
            expected = torch.cat([x[k] for x in parts])
            torch.testing.assert_close(got[k], expected, rtol=0, atol=0)

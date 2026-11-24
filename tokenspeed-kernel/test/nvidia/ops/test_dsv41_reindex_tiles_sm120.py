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
from tokenspeed_kernel.ops.attention.dsv41.triton import (
    _index_scan_kernel,
    index_q_quantize,
)
from tokenspeed_kernel.platform import ArchVersion, Platform

pytestmark = pytest.mark.skipif(
    Platform.get().arch_version != ArchVersion(12, 0), reason="SM120 only"
)


@pytest.mark.parametrize("rows", [6, 24, 48, 64, 96, 97, 128, 160, 192, 256])
@pytest.mark.parametrize("nc", [512, 2048])
@pytest.mark.parametrize("wdtype", [torch.bfloat16, torch.float32])
def test_dense_reindex_tiles_exact(rows, nc, wdtype):
    history = 32768
    torch.manual_seed(51)
    q = torch.randn(rows, 32, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(rows, 32, device="cuda", dtype=wdtype)
    keys = torch.randn(history, 128, device="cuda", dtype=torch.bfloat16)
    cache = torch.zeros(history // 64, 64, 68, device="cuda", dtype=torch.uint8)
    dsv41.cache_scatter(keys, cache, torch.arange(history, device="cuda"), "index")
    table = torch.arange(history // 64, device="cuda", dtype=torch.int32).repeat(
        rows, 1
    )
    table[:, 3] = -1
    visible = torch.full((rows,), history - 19, device="cuda", dtype=torch.int32)
    visible[0] = 0
    cand = torch.stack(
        [
            torch.randperm(history // 8, device="cuda")[:nc].sort().values
            for _ in range(rows)
        ]
    ).int()
    cand[:, -1] = -1
    q = index_q_quantize(q, "index", None)
    dummy = torch.empty((rows, 0), device="cuda")
    ids = torch.empty((rows, 0), device="cuda", dtype=torch.int64)
    dense = torch.empty(rows, nc * 8, device="cuda")

    def scan(parts, b, st):
        dense.fill_(-torch.inf)
        _index_scan_kernel[(rows, parts)](
            q,
            w,
            cache,
            table,
            visible,
            cand,
            dense,
            ids,
            dummy,
            ids,
            *q.stride(),
            *w.stride(),
            *cache.stride(),
            *table.stride(),
            visible.stride(0),
            *cand.stride(),
            cache.shape[0],
            table.shape[1],
            nc,
            32,
            32,
            1,
            32,
            parts,
            b,
            b,
            512,
            32,
            False,
            False,
            True,
            nc * 8,
            enable_fp_fusion=False,
            num_warps=4,
            num_stages=st
        )

    scan(16, 256, 3)
    reference = dense.clone()
    scan(64, 128, 2)
    assert torch.equal(dense, reference)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        scan(64, 128, 2)
    for _ in range(2):
        visible.random_(0, history)
        visible[0] = 0
        cand[1, ::3] = -1
        graph.replay()
        actual = dense.clone()
        scan(16, 256, 3)
        assert torch.equal(actual, dense)

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


@pytest.mark.parametrize("width", [1, 6])
@pytest.mark.parametrize("topk", [4, 512])
@pytest.mark.parametrize("full", [False, True])
def test_local_reindex_graph_matches_replicated(width, topk, full):
    from tokenspeed_kernel.ops.attention.dsv41 import deep_gemm as native

    if not torch.cuda.is_available() or not native.is_hopper_indexer_available():
        pytest.skip("Native local Reindex currently requires Hopper")
    torch.manual_seed(37)
    cache = torch.zeros(17, 64, 132, dtype=torch.uint8, device="cuda")
    keys = torch.randn(17 * 64, 128, device="cuda", dtype=torch.bfloat16)
    dsv41.cache_scatter(keys, cache, torch.arange(17 * 64, device="cuda"), "index_v4")
    table = (
        (torch.randperm(16, device="cuda") + 1)
        .int()[None, :]
        .expand(width, -1)
        .contiguous()
    )
    candidates = (
        torch.arange(128, device="cuda", dtype=torch.int32)[None, :]
        .expand(width, -1)
        .clone()
    )
    visible = torch.full((width,), 1024, device="cuda", dtype=torch.int32)
    q = torch.randn(width, 32, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.rand(width, 32, device="cuda", dtype=torch.bfloat16)
    if full:
        candidates = None
    assert dsv41.local_index_supported(q, cache, candidates)

    def local():
        packed = []
        for rank in range(4):
            local_table = table.masked_fill((table - 1) % 4 != rank, -1)
            packed.append(
                dsv41.index_candidates(
                    q, weights, cache, local_table, visible, candidates, topk
                )
            )
        rows = torch.empty(width, topk, device="cuda", dtype=torch.int32)
        lens = torch.empty(width, device="cuda", dtype=torch.int32)
        return dsv41.merge_index_candidates(torch.cat(packed, dim=1), rows, lens)

    local()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rows, lens = local()
    for end in (1024, 65, 0):
        visible.copy_(
            (torch.arange(width, device="cuda") + end - width + 1).clamp_min(0)
        )
        graph.replay()
        expected = native.hopper_index_topk(
            q,
            weights,
            cache,
            table,
            visible,
            candidates,
            topk,
            0,
            8,
            256,
            256,
            None,
            None,
            None,
        )
        torch.testing.assert_close(rows, expected[0], rtol=0, atol=0)
        torch.testing.assert_close(lens, expected[1], rtol=0, atol=0)

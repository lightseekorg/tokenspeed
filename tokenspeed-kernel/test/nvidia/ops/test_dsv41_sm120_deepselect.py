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
from tokenspeed_kernel.ops.attention.dsv41.flashinfer import _indices
from tokenspeed_kernel.thirdparty.deepgemm_sm120_index_select_experiment import (
    select_scores,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="SM120 experiment",
)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_sanitize_slots_graph(monkeypatch, dtype):
    slots = torch.tensor(
        [[-1, 0, 63, 64, 127, 128, 2**30]], device="cuda", dtype=dtype
    ).repeat(2, 2)[:, ::2]
    cache = torch.empty((2, 64, 68), device="cuda", dtype=torch.uint8)
    monkeypatch.setenv("TOKENSPEED_EXPERIMENT_SM120_DEEPSELECT", "1")
    _indices(slots, cache)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        got = _indices(slots, cache)
    for delta in (0, 3):
        slots.add_(delta)
        g.replay()
        torch.cuda.synchronize()
        ref = slots.int().masked_fill((slots < 0) | (slots >= 128), -1)
        assert torch.equal(got, ref)


@pytest.mark.parametrize("n", [6, 48, 64, 96, 97, 128, 160, 192, 256])
@pytest.mark.parametrize("blocks", [0, 64, 2048])
def test_full_selection(monkeypatch, blocks, n):
    torch.manual_seed(51)
    width = 16384
    scores = torch.randn(n, width, device="cuda").round()
    table = torch.arange(width // 64, device="cuda", dtype=torch.int32).repeat(n, 1)
    table[:, 3] = -1
    ends = torch.tensor(
        ([0, 13, 513, 4091, width - 3, width] * ((n + 5) // 6))[:n],
        device="cuda",
        dtype=torch.int32,
    )
    out = (
        torch.empty(n, 512, device="cuda", dtype=torch.int32),
        torch.empty(n, device="cuda", dtype=torch.int32),
        torch.empty(n, blocks, device="cuda", dtype=torch.int32),
        torch.empty(n, device="cuda", dtype=torch.int32),
    )
    monkeypatch.setenv("TOKENSPEED_EXPERIMENT_SM120_DEEPSELECT", "1")

    def run():
        return select_scores(scores, table, ends, width // 64, 512, blocks, out, 32)

    run()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run()
    for lengths in (
        ends.clone(),
        torch.tensor(
            ([width, 0, 31, 8191, 1234, 512] * ((n + 5) // 6))[:n],
            device="cuda",
            dtype=torch.int32,
        ),
    ):
        ends.copy_(lengths)
        g.replay()
        torch.cuda.synchronize()
        clean = scores.masked_fill(
            (torch.arange(width, device="cuda")[None, :] >= ends[:, None])
            | (table.repeat_interleave(64, dim=1) < 0),
            -torch.inf,
        )
        for row in range(n):
            ids = out[0][row, : out[1][row]].long()
            k = min(512, int((clean[row] > -torch.inf).sum()))
            assert len(ids) == k and len(ids.unique()) == k
            assert torch.equal(
                clean[row, ids].sort().values, clean[row].topk(k).values.sort().values
            )
        if blocks:
            maxima = clean.reshape(n, -1, 8).amax(2)
            bid = (ends - 1) // 8
            for row in range(n):
                if ends[row] > 0 and maxima[row, bid[row]] > -torch.inf:
                    maxima[row, bid[row]] = torch.inf
                ids = out[2][row, : out[3][row]].long()
                k = min(blocks, int((maxima[row] > -torch.inf).sum()))
                assert len(ids) == k and len(ids.unique()) == k
                assert torch.equal(
                    maxima[row, ids].sort().values,
                    maxima[row].topk(k).values.sort().values,
                )


@pytest.mark.parametrize("n", [6, 48, 64, 96, 97, 128, 160, 192, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_reindex_graph(monkeypatch, dtype, n):
    torch.manual_seed(52)
    hist, nc = 8192, 512
    q = torch.randn(n, 32, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, 32, device="cuda", dtype=dtype)
    cache = torch.zeros(hist // 64, 64, 68, device="cuda", dtype=torch.uint8)
    dsv41.cache_scatter(
        torch.randn(hist, 128, device="cuda", dtype=torch.bfloat16),
        cache,
        torch.arange(hist, device="cuda"),
        "index",
    )
    table = torch.arange(hist // 64, device="cuda", dtype=torch.int32).repeat(n, 1)
    table[:, 3] = -1
    cand = torch.stack(
        [torch.randperm(hist // 8, device="cuda")[:nc].sort().values for _ in range(n)]
    ).int()
    cand[:, -1] = -1
    visible = torch.tensor(
        ([0, 513, 2345, 8192, 7000, 8191] * ((n + 5) // 6))[:n],
        device="cuda",
        dtype=torch.int32,
    )

    def call():
        return impl._index_topk(
            q,
            w,
            cache,
            table,
            visible,
            cand,
            512,
            0,
            8,
            256,
            256,
            None,
            None,
            None,
            scan_num_warps=4,
            dense_selector=None,
        )

    monkeypatch.setenv("TOKENSPEED_EXPERIMENT_SM120_DEEPSELECT", "1")
    call()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        got = call()
    for lengths in (
        visible.clone(),
        torch.tensor(
            ([8192, 0, 41, 4321, 8191, 1024] * ((n + 5) // 6))[:n],
            device="cuda",
            dtype=torch.int32,
        ),
    ):
        visible.copy_(lengths)
        g.replay()
        torch.cuda.synchronize()
        # Obtain the same score-only scan used by the candidate selector, then
        # compare against independent torch.topk. Ties may choose other IDs.
        pq = impl.index_q_quantize(q, "index", None)
        scores = torch.full((n, nc * 8), -torch.inf, device="cuda")
        empty = torch.empty((n, 0), device="cuda")
        ei = torch.empty((n, 0), device="cuda", dtype=torch.int64)
        impl._index_scan_kernel[(n, 16)](
            pq,
            w,
            cache,
            table,
            visible,
            cand,
            scores,
            ei,
            empty,
            ei,
            *pq.stride(),
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
            16,
            256,
            256,
            512,
            32,
            False,
            False,
            True,
            nc * 8,
            enable_fp_fusion=False,
            num_warps=4
        )
        for row in range(n):
            logical = (
                cand[row, :, None].long() * 8 + torch.arange(8, device="cuda")
            ).flatten()
            valid = logical >= 0
            by_id = torch.full((hist,), -torch.inf, device="cuda")
            by_id[logical[valid]] = scores[row, valid]
            k = min(512, int((scores[row] > -torch.inf).sum()))
            ids = got[0][row, : got[1][row]].long()
            assert len(ids) == k and len(ids.unique()) == k
            assert bool((ids[1:] > ids[:-1]).all())
            assert torch.equal(
                by_id[ids].sort().values, scores[row].topk(k).values.sort().values
            )

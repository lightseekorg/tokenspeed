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

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from utils import is_cdna5

if not is_cdna5():
    pytest.skip("GFX1250 is required", allow_module_level=True)

_OPS = Path(__file__).resolve().parents[3] / "ops"
if str(_OPS) not in sys.path:
    sys.path.insert(0, str(_OPS))

from test_attention_dsv41 import _reference_quantize  # noqa: E402
from tokenspeed_kernel.ops.attention import dsv41
from tokenspeed_kernel_amd.ops.gfx1250.attention.dsv41.indexer import (
    dsv41_index_logits_gfx1250,
)


@pytest.mark.parametrize("seed", [251, 781])
@pytest.mark.parametrize("reindex", [False, True])
@pytest.mark.parametrize("score_chunk", [40, 256])
def test_key_layout_logits_reference_and_graph(seed, reindex, score_chunk):
    g = torch.Generator(device="cuda").manual_seed(seed)
    keys = torch.randn(512, 128, device="cuda", dtype=torch.bfloat16, generator=g) * 0.1
    _, key_reference = _reference_quantize(keys, "index")
    storage = torch.zeros(8, 4352 + 13, device="cuda", dtype=torch.uint8)
    cache = storage[:, :4352].view(8, 64, 68)
    dsv41.cache_scatter(keys, cache, torch.arange(512, device="cuda"), "index")
    q = torch.randn(2, 32, 128, device="cuda", dtype=torch.bfloat16, generator=g) * 0.1
    weights = torch.randn(2, 32, device="cuda", generator=g)
    table = torch.tensor(
        [[3, 1, -1, 2], [7, 5, 4, 6]], device="cuda", dtype=torch.int32
    )
    visible = torch.tensor([193, 221], device="cuda", dtype=torch.int32)
    candidates = (
        torch.tensor(
            [[3, 0, -1, 20, 31], [27, 4, 2, 1, -1]], device="cuda", dtype=torch.int32
        )
        if reindex
        else None
    )
    width = 40 if reindex else 256
    out = torch.empty(2, width, device="cuda")

    def invoke():
        out.fill_(-torch.inf)
        dsv41_index_logits_gfx1250(
            q, weights, cache.flatten(1), table, visible, candidates, out, score_chunk
        )
        return out

    invoke()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        invoke()
    for changed in (False, True):
        if changed:
            q.mul_(-0.75)
            weights.mul_(0.5)
            table.copy_(table.flip(1))
            visible.copy_(torch.tensor([131, 0], device="cuda", dtype=torch.int32))
            if candidates is not None:
                candidates.copy_(candidates.flip(1))
        out.fill_(float("nan"))
        graph.replay()
        expected = torch.full((2, width), -torch.inf)
        for row in range(2):
            logical = torch.arange(width)
            if reindex:
                blocks = candidates[row].cpu()[logical // 8].long()
                logical = torch.where(blocks >= 0, blocks * 8 + logical % 8, -1)
            pages = table[row].cpu()[logical.clamp(min=0) // 64].long()
            valid = (logical >= 0) & (logical < int(visible[row])) & (pages >= 0)
            slots = pages[valid] * 64 + logical[valid] % 64
            dots = q[row].float().cpu() @ key_reference[slots].float().T
            expected[row, valid] = (dots.relu() * weights[row].cpu()[:, None]).sum(0)
        torch.testing.assert_close(out.cpu(), expected, atol=2e-4, rtol=2e-4)
        captured = out.clone()
        torch.testing.assert_close(captured, invoke(), atol=0, rtol=0)

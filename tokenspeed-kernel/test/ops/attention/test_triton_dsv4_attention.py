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

import pytest
import torch
from tokenspeed_kernel.numerics.torch.attention.dsv4 import (
    torch_dsv4_selected_attention,
)
from tokenspeed_kernel.ops.attention.dsv4 import dsv4_prefill


def _reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    slots: torch.Tensor,
    lens: torch.Tensor,
    sink: torch.Tensor,
) -> torch.Tensor:
    return torch_dsv4_selected_attention(q, kv, slots, lens, sink, 512**-0.5)[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("heads", [16, 64, 128])
def test_selected_attention_and_graph_replay(heads: int) -> None:
    torch.manual_seed(74)
    kv = torch.randn(320, 512, device="cuda", dtype=torch.bfloat16)
    q = torch.randn(3, heads, 512, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(heads, device="cuda", dtype=torch.float32)
    selected = torch.arange(192, device="cuda", dtype=torch.int32).repeat(3, 1)
    selected[0, 10] = -1
    lens = torch.tensor([151, 1, 0], device="cuda", dtype=torch.int32)
    output = torch.empty_like(q)

    def run():
        return dsv4_prefill(
            q,
            kv,
            selected,
            lens,
            sink,
            512**-0.5,
            out=output,
            override=None,
            solution="triton",
        )

    expected = _reference(q, kv, selected, lens, sink)
    actual = run()
    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual, expected, atol=8e-3, rtol=8e-3)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    q.mul_(0.5)
    graph.replay()
    expected = _reference(q, kv, selected, lens, sink)
    torch.testing.assert_close(output, expected, atol=8e-3, rtol=8e-3)

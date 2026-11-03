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

"""Kimi-K3 routing into caller-owned buffers, eager and captured."""

import pytest
import torch
from tokenspeed_kernel.ops.moe import _moe_sigmoid_bias_topk
from tokenspeed_kernel.platform import ArchVersion, current_platform

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or current_platform().arch_version not in (ArchVersion(10, 0), ArchVersion(10, 3)),
    reason="requires SM100/SM103",
)


def check_routes(actual, expected):
    aw, ai = actual
    ew, ei = expected
    ai, a_order = ai.sort(-1)
    ei, e_order = ei.sort(-1)
    torch.testing.assert_close(ai, ei, atol=0, rtol=0)
    torch.testing.assert_close(
        aw.gather(1, a_order), ew.gather(1, e_order), atol=2e-7, rtol=2e-6
    )


@pytest.mark.parametrize(
    "rows,pattern",
    [
        (2560, "random"),
        (8193, "random"),
        (16385, "random"),
        (2560, "ties"),
        (2560, "saturated"),
    ],
)
def test_router_output_buffers_and_graph(rows, pattern):
    torch.manual_seed(rows)
    logits = torch.randn(rows, 896, device="cuda")
    bias = torch.randn(896, device="cuda") * 0.01
    if pattern == "ties":
        logits.zero_()
        bias.zero_()
    elif pattern == "saturated":
        logits.copy_(logits.sign() * 100)
        bias.zero_()
    out = (
        torch.empty(rows, 16, device="cuda"),
        torch.empty(rows, 16, device="cuda", dtype=torch.int32),
    )

    def run():
        return _moe_sigmoid_bias_topk(
            logits,
            bias,
            16,
            weights_dtype=torch.bfloat16,
            solution="cuda",
            out=out,
        )

    actual = run()
    assert actual[0] is out[0] and actual[1] is out[1]
    check_routes(actual, _moe_sigmoid_bias_topk(logits, bias, 16, solution="triton"))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    logits.normal_()
    graph.replay()
    check_routes(captured, _moe_sigmoid_bias_topk(logits, bias, 16, solution="triton"))

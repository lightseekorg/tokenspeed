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

"""Per-head native Q/K RMSNorm agrees with the float32 definition."""

import pytest
import torch

pytest.importorskip("torch_npu")
from tokenspeed_kernel_npu.ops.layernorm import qk_rmsnorm


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("q_heads", [0, 1, 4])
@torch.inference_mode()
def test_qk_rmsnorm_eager_and_graph(dtype, q_heads):
    torch.npu.set_device(0)
    torch.manual_seed(87)
    width = 128
    q = torch.randn(5, q_heads * width, device="npu", dtype=dtype)
    k = torch.randn(5, width, device="npu", dtype=dtype)
    qw = torch.randn(width, device="npu", dtype=dtype)
    kw = torch.randn(width, device="npu", dtype=dtype)
    eps = 1e-6

    def reference(x, weight):
        heads = x.reshape(-1, width).float()
        normalized = heads * torch.rsqrt(heads.square().mean(-1, keepdim=True) + eps)
        return (normalized * weight.float()).to(dtype).reshape_as(x)

    def check(outputs):
        tolerance = 0.02 if dtype == torch.bfloat16 else 0.002
        for actual, expected in zip(outputs, (reference(q, qw), reference(k, kw))):
            torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)

    def run():
        return qk_rmsnorm(q, k, qw, kw, eps)

    check(run())
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = run()
    q.add_(0.1)
    k.sub_(0.2)
    graph.replay()
    check(outputs)

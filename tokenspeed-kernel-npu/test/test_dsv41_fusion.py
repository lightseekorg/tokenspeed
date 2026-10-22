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

"""Numerical and changing-input NPUGraph contracts for V4.1 fusion."""

import pytest
import torch
import torch_npu
from tokenspeed_kernel_npu.ops.dsv41_metadata import selection_table
from tokenspeed_kernel_npu.ops.dsv41_mhc import coefficients, mhc_mixes


@pytest.fixture(scope="module", autouse=True)
def device():
    torch.npu.set_device(0)
    torch.set_num_threads(4)
    torch.manual_seed(20261017)


def coefficient_reference(m, s, b, eps, iters):
    pre = torch.sigmoid(m[:, :4] * s[0] + b[:4]) + eps
    post = 2 * torch.sigmoid(m[:, 4:8] * s[1] + b[4:8])
    c = (m[:, 8:] * s[2] + b[8:]).reshape(-1, 4, 4).softmax(-1) + eps
    c = c / (c.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        c = c / (c.sum(-1, keepdim=True) + eps)
        c = c / (c.sum(-2, keepdim=True) + eps)
    return pre, post, c


def check_coefficients(actual, expected):
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a.cpu(), e.cpu(), atol=4e-7, rtol=4e-6)
        assert torch.isfinite(a).all()


@pytest.mark.parametrize(
    "n,iters",
    [(0, 20), (1, 1), (8, 2), (16, 20), (32, 20), (129, 20), (512, 20), (2048, 20)],
)
def test_coefficients(n, iters):
    storage = torch.randn(n, 48, device="npu", dtype=torch.float32)
    m = storage[:, ::2]
    s = torch.tensor([0.5, 0.2, 0.3], device="npu")
    b = torch.randn(24, device="npu")
    check_coefficients(
        coefficients(m, s, b, 1e-6, iters), coefficient_reference(m, s, b, 1e-6, iters)
    )
    check_coefficients(
        coefficients(m, s, b, 1e-6, iters),
        coefficient_reference(m.cpu(), s.cpu(), b.cpu(), 1e-6, iters),
    )


@pytest.mark.parametrize("n", [1, 8, 32, 129])
def test_coefficients_dynamic_graph(n):
    m = torch.randn(n, 24, device="npu")
    s = torch.randn(3, device="npu")
    b = torch.randn(24, device="npu")
    coefficients(m, s, b, 1e-6, 20)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        out = coefficients(m, s, b, 1e-6, 20)
    for scale in (0.0, 1.0, 30.0, 0.01):
        m.copy_(torch.randn_like(m) * scale)
        s.copy_(torch.randn_like(s))
        b.copy_(torch.randn_like(b))
        graph.replay()
        torch.npu.synchronize()
        check_coefficients(out, coefficient_reference(m, s, b, 1e-6, 20))


@pytest.mark.parametrize("n", [1, 32])
def test_mhc_projection_and_runtime_dispatch(n):
    from tokenspeed.runtime.models.deepseek_v41 import v41_hc_mixes

    x = torch.randn(n, 4, 5120, device="npu", dtype=torch.bfloat16)
    w = torch.randn(24, 20480, device="npu") * 0.01
    s = torch.tensor([0.5, 0.2, 0.3], device="npu")
    b = torch.randn(24, device="npu")

    def reference():
        flat = x.flatten(-2).float()
        m = torch.nn.functional.linear(flat, w) * torch.rsqrt(
            flat.square().mean(-1, keepdim=True) + 1e-20
        )
        return coefficient_reference(m, s, b, 1e-6, 20)

    expected = reference()
    check_coefficients(v41_hc_mixes(x, w, s, b, 1e-20, 1e-6, 20), expected)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        out = v41_hc_mixes(x, w, s, b, 1e-20, 1e-6, 20)
    for _ in range(5):
        x.copy_(torch.randn_like(x))
        w.mul_(0.97)
        s.add_(0.01)
        b.mul_(0.99)
        graph.replay()
        torch.npu.synchronize()
        check_coefficients(out, reference())


@pytest.mark.parametrize(
    "n,width,ratio",
    [(0, 7, 1), (1, 0, 2), (8, 17, 2), (32, 8192, 2), (32, 16384, 1), (129, 257, 1)],
)
def test_table_dynamic_graph(n, width, ratio):
    from tokenspeed_kernel.ops.attention.dsv41 import selection_table as dispatch

    backing = torch.randint(
        -1, 100, (10, max(1, width) * 2), device="npu", dtype=torch.int32
    )
    table = backing[:, : width * 2 : 2]
    req = torch.arange(n, device="npu", dtype=torch.int64) % 12 - 1
    pos = torch.arange(n, device="npu", dtype=torch.int64) * 64

    def check(out):
        rr = req.cpu()
        tt = table.cpu()
        p = pos.cpu()
        expect = (
            tt[rr.clamp(0, 9)]
            .clone()
            .masked_fill((rr < 0)[:, None] | (rr >= 10)[:, None], -1)
        )
        expect.masked_fill_(expect <= 0, -1)
        assert torch.equal(out[0].cpu(), expect)
        assert torch.equal(out[1].cpu(), ((p + 1) // ratio).clamp_min(0).int())

    check(dispatch(pos, req, table, ratio))
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        out = dispatch(pos, req, table, ratio)
    for step in range(3):
        req.copy_((req + 3) % 12 - 1)
        pos.copy_(pos // 2 - 3)
        table.copy_(
            torch.randint(-1, 100, table.shape, device="npu", dtype=torch.int32)
        )
        graph.replay()
        torch.npu.synchronize()
        check(out)

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


"""FP32 mHC projection, independent FP64 reference and changing graph inputs."""

import pytest
import torch
import torch_npu  # noqa: F401
from tokenspeed_kernel_npu.ops.dsv41_mhc_mixes import mhc_mixes as candidate

torch.npu.set_device(0)
torch.set_num_threads(4)
torch.manual_seed(812)


def reference(x, w, s, b):
    f = x.cpu().flatten(-2).double()
    w = w.cpu().double()
    s = s.cpu().double()
    b = b.cpu().double()
    m = (f @ w.T) * torch.rsqrt(f.square().mean(-1, keepdim=True) + 1e-20)
    pre = torch.sigmoid(m[:, :4] * s[0] + b[:4]) + 1e-6
    post = 2 * torch.sigmoid(m[:, 4:8] * s[1] + b[4:8])
    c = (m[:, 8:] * s[2] + b[8:]).reshape(-1, 4, 4).softmax(-1) + 1e-6
    c = c / (c.sum(-2, keepdim=True) + 1e-6)
    for _ in range(19):
        c = c / (c.sum(-1, keepdim=True) + 1e-6)
        c = c / (c.sum(-2, keepdim=True) + 1e-6)
    return pre, post, c


def inputs(n):
    x = torch.randn(n, 4, 5120, device="npu", dtype=torch.bfloat16)
    w = torch.randn(24, 20480, device="npu") * 0.01
    s = torch.tensor([0.5, 0.2, 0.3], device="npu")
    b = torch.randn(24, device="npu")
    return x, w, s, b


def check(a, e):
    for x, y in zip(a, e):
        torch.testing.assert_close(x.cpu().double(), y, atol=4e-7, rtol=4e-6)


@pytest.mark.parametrize(
    "n",
    [
        0,
        1,
        2,
        3,
        6,
        7,
        8,
        16,
        31,
        32,
        33,
        80,
        95,
        96,
        97,
        128,
        160,
        191,
        192,
        193,
        512,
        2048,
    ],
)
def test_reference(n):
    x, w, s, b = inputs(n)
    check(candidate(x, w, s, b, 1e-20, 1e-6, 20), reference(x, w, s, b))


@pytest.mark.parametrize("n", [1, 6, 31, 32, 33, 96, 97, 128, 160, 192, 193])
def test_replay(n):
    x, w, s, b = inputs(n)
    candidate(x, w, s, b, 1e-20, 1e-6, 20)
    torch.npu.synchronize()
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        out = candidate(x, w, s, b, 1e-20, 1e-6, 20)
    for scale in [0.0, 1e-3, 1.0, 30.0]:
        x.copy_(torch.randn_like(x) * scale)
        w.mul_(0.97)
        s.add_(0.01)
        b.mul_(0.99)
        g.replay()
        torch.npu.synchronize()
        check(out, reference(x, w, s, b))


@pytest.mark.parametrize("hidden", [8, 129, 1024, 5120])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_projection_tail_and_operand_precision(hidden, dtype):
    x = torch.randn(3, 4, hidden, device="npu", dtype=dtype)
    w = torch.randn(24, 4 * hidden, device="npu", dtype=torch.float32) * 0.01
    s = torch.tensor([0.5, 0.2, 0.3], device="npu")
    b = torch.randn(24, device="npu")
    check(candidate(x, w, s, b, 1e-20, 1e-6, 20), reference(x, w, s, b))


def test_strided_projection_inputs():
    x, w, s, b = inputs(6)
    x = x.transpose(0, 1).contiguous().transpose(0, 1)
    w = w.T.contiguous().T
    check(candidate(x, w, s, b, 1e-20, 1e-6, 20), reference(x, w, s, b))

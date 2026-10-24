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


"""HC4 post tiling preserves FP32 operation order and live graph inputs."""

import pytest
import torch
import torch_npu  # noqa: F401
from tokenspeed_kernel_npu.ops.dsv41_mhc import mhc_post


def reference(x, residual, post, comb):
    outputs = []
    for j in range(4):
        mixed = torch.zeros_like(x, dtype=torch.float32)
        for i in range(4):
            mixed = mixed + residual[:, i].float() * comb[:, i, j, None]
        outputs.append((x.float() * post[:, j] + mixed).bfloat16())
    return torch.stack(outputs, dim=1)


def make_inputs(n, h, strided):
    step = 2 if strided else 1
    x = torch.randn(n, h * step, device="npu", dtype=torch.bfloat16)[:, ::step]
    r = torch.randn(n, 4, h * step, device="npu", dtype=torch.bfloat16)[:, :, ::step]
    p = torch.randn(n, 16, 1, device="npu", dtype=torch.float32)[:, :4]
    c = torch.randn(n, 4, 4, device="npu", dtype=torch.float32)
    if strided:
        c = c.transpose(1, 2)
    return x, r, p, c


@pytest.mark.parametrize(
    "n,h,strided",
    [
        (1, 37, True),
        (32, 5120, False),
        (33, 5120, True),
        (192, 5120, False),
        (193, 5120, True),
        (2048, 5120, False),
    ],
)
def test_post_fp32_order(n, h, strided):
    torch.npu.set_device(0)
    torch.set_num_threads(4)
    torch.manual_seed(1020)
    args = make_inputs(n, h, strided)
    actual = mhc_post(*args)
    expected = reference(*args)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("n", [32, 192])
def test_post_replay(n):
    torch.npu.set_device(0)
    torch.set_num_threads(4)
    torch.manual_seed(1021)
    args = make_inputs(n, 5120, False)
    mhc_post(*args)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = mhc_post(*args)
    for scale in (0.0, 1e-3, 1.0, 30.0):
        for tensor in args:
            tensor.copy_(torch.randn_like(tensor) * scale)
        graph.replay()
        expected = reference(*args)
        torch.npu.synchronize()
        torch.testing.assert_close(out, expected, atol=0, rtol=0)

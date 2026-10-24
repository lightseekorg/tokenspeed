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

"""Grouped Cube projection: independent numerics, strides, and live replay."""

import pytest
import torch
import torch_npu  # noqa: F401
from tokenspeed_kernel.ops.gemm import grouped_bf16_projection


@pytest.mark.parametrize("layout", ["contiguous", "padded", "weight_rows"])
@pytest.mark.parametrize("seed", [42, 361, 20261020])
def test_supported_projection(layout, seed):
    torch.npu.set_device(0)
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    x = torch.randn((48, 2, 4096), dtype=torch.bfloat16)
    w = torch.randn((2, 1024, 4096), dtype=torch.bfloat16) * 0.02
    a, b = x.npu(), w.npu()
    if layout == "padded":
        storage = torch.empty((48, 4, 4096), device="npu", dtype=torch.bfloat16)
        a = storage[:, :2]
        a.copy_(x)
    elif layout == "weight_rows":
        storage = torch.empty((2, 2048, 4096), device="npu", dtype=torch.bfloat16)
        b = storage[:, ::2]
        b.copy_(w)
    expected = torch.einsum("mgk,gnk->mgn", x.double(), w.double()).bfloat16()
    out = torch.empty((48, 2, 1024), device="npu", dtype=torch.bfloat16)
    actual = grouped_bf16_projection(a, b, out, "triton_ascend")
    assert actual is out
    reference = grouped_bf16_projection(a, b, None, "torch")
    # Sum order can perturb a BF16 rounding boundary. Bound both independent
    # FP64 and same-device comparisons, without claiming general bit identity.
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.0079, atol=2e-5)
    delta = actual.float() - reference.float()
    assert float(delta.norm() / reference.float().norm()) < 1e-4
    torch.testing.assert_close(a.cpu(), x, rtol=0, atol=0)
    torch.testing.assert_close(b.cpu(), w, rtol=0, atol=0)


def test_mutable_native_graph_and_cancellation():
    torch.npu.set_device(0)
    x = torch.ones((48, 2, 4096), device="npu", dtype=torch.bfloat16)
    w = torch.ones((2, 1024, 4096), device="npu", dtype=torch.bfloat16)
    grouped_bf16_projection(x, w, None, "triton_ascend")
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(
        graph, stream=torch.npu.Stream(device=x.device), auto_dispatch_capture=True
    ):
        out = grouped_bf16_projection(x, w, None, "triton_ascend")
    for step in range(4):
        x.fill_(step + 1)
        w.fill_(step + 2)
        w[:, :, 1::2].neg_()
        w[:, :, -1].fill_(step + 1)
        graph.replay()
        torch.npu.synchronize()
        reference = torch.einsum(
            "mgk,gnk->mgn", x.cpu().double(), w.cpu().double()
        ).bfloat16()
        torch.testing.assert_close(out.cpu(), reference, atol=0, rtol=0)


@pytest.mark.parametrize("m,groups", [(0, 2), (8, 2), (40, 2), (48, 1), (48, 4)])
def test_unmeasured_geometry_keeps_fallback(m, groups):
    torch.npu.set_device(0)
    x = torch.ones((m, groups, 4096), device="npu", dtype=torch.bfloat16)
    w = torch.ones((groups, 1024, 4096), device="npu", dtype=torch.bfloat16)
    actual = grouped_bf16_projection(x, w, None, None)
    expected = grouped_bf16_projection(x, w, None, "torch")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_nonfinite_propagation():
    torch.npu.set_device(0)
    x = torch.ones((48, 2, 4096), device="npu", dtype=torch.bfloat16)
    w = torch.ones((2, 1024, 4096), device="npu", dtype=torch.bfloat16)
    x[0, 0, 0] = float("nan")
    x[1, 0, 0] = float("inf")
    x[2, 1, 0] = -float("inf")
    actual = grouped_bf16_projection(x, w, None, "triton_ascend")
    expected = grouped_bf16_projection(x, w, None, "torch")
    torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)

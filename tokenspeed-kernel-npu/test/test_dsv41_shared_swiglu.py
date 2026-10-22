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

"""Exact narrow shared-expert activations, including live native graph input."""

import pytest
import torch
import torch_npu  # noqa: F401
from tokenspeed_kernel.ops.activation import silu_and_mul
from tokenspeed_kernel_npu.ops.dsv41_swiglu import swiglu, swiglu_batched


@pytest.mark.parametrize(
    "rows,hidden",
    [
        (1, 32),
        (16, 144 * 2),
        (17, 576),
        (160, 576),
        (192, 576),
        (192, 288),
        (192, 1024),
        (4, 1152),
        (0, 576),
    ],
)
@pytest.mark.parametrize("strided", [False, True])
def test_exact(rows, hidden, strided):
    torch.npu.set_device(0)
    torch.manual_seed(97)
    x = (torch.randn(rows, hidden * (4 if strided else 2), device="npu") * 8).bfloat16()
    if strided:
        x = x[:, ::2]
    expected = swiglu(x, 10.0, block_size=512)
    direct = swiglu_batched(x, 10.0, block_size=1024, rows_per_program=4)
    actual = silu_and_mul(x, limit=10.0)
    for value in (direct, actual):
        torch.testing.assert_close(value, expected, atol=0, rtol=0)


def test_nonfinite_and_signed_zero():
    torch.npu.set_device(0)
    x = torch.zeros(17, 1152, device="npu", dtype=torch.bfloat16)
    special = torch.tensor(
        [float("nan"), float("inf"), -float("inf"), -0.0, 0.0, 10.0, -10.0],
        device="npu",
        dtype=torch.bfloat16,
    )
    x[:, :7] = special
    x[:, 576:583] = special.flip(0)
    expected = swiglu(x, 10.0, block_size=512)
    actual = silu_and_mul(x, limit=10.0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)
    finite = ~torch.isnan(expected)
    assert torch.equal(
        actual.view(torch.int16)[finite], expected.view(torch.int16)[finite]
    )


def test_mutable_native_graph():
    torch.npu.set_device(0)
    x = torch.randn(192, 1152, device="npu", dtype=torch.bfloat16)
    out = torch.empty(192, 576, device="npu", dtype=torch.bfloat16)
    silu_and_mul(x, out, limit=10.0)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = silu_and_mul(x, out, limit=10.0)
    assert actual.data_ptr() == out.data_ptr()
    for scale in (0.0, 1.0, 16.0):
        x.copy_(torch.randn_like(x) * scale)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(out, swiglu(x, 10.0, block_size=512), atol=0, rtol=0)

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

"""Batched GEMM can write the final token-major grouped projection directly."""

import pytest
import torch
from tokenspeed_kernel.ops.gemm import grouped_bf16_projection
from tokenspeed_kernel.ops.gemm.native import torch_grouped_bf16_projection_gfx950


@pytest.mark.parametrize("layout", ["contiguous", "tokens", "groups", "weights"])
@pytest.mark.parametrize("output_mode", ["omitted", "none", "provided"])
def test_direct_grouped_projection_destination_cpu(layout, output_mode):
    tokens, groups, width, rank = 6, 2, 64, 32
    if layout == "tokens":
        x = torch.randn(tokens, groups * width + 32, dtype=torch.bfloat16)[
            :, : groups * width
        ].unflatten(1, (groups, width))
    elif layout == "groups":
        x = torch.randn(tokens, groups + 1, width, dtype=torch.bfloat16)[:, :groups]
    else:
        x = torch.randn(tokens, groups, width, dtype=torch.bfloat16)
    weight = torch.randn(
        groups, rank * 2 if layout == "weights" else rank, width, dtype=torch.bfloat16
    )
    if layout == "weights":
        weight = weight[:, ::2]
    expected = torch.einsum("tgd,grd->tgr", x, weight)
    out = (
        torch.empty(tokens, groups, rank, dtype=torch.bfloat16)
        if output_mode == "provided"
        else None
    )
    if output_mode == "omitted":
        actual = torch_grouped_bf16_projection_gfx950(x, weight)
    else:
        actual = torch_grouped_bf16_projection_gfx950(x, weight, out)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.is_contiguous()
    assert actual.flatten(1).data_ptr() == actual.data_ptr()
    if out is not None:
        assert actual is out


def test_direct_grouped_projection_retains_gradient_reference():
    x = torch.randn(3, 2, 8, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(2, 4, 8, dtype=torch.bfloat16, requires_grad=True)
    actual = torch_grouped_bf16_projection_gfx950(x, weight)
    expected = torch.einsum("tgd,grd->tgr", x, weight)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_grads = torch.autograd.grad(actual.sum(), (x, weight))
    expected_grads = torch.autograd.grad(expected.sum(), (x, weight))
    for value, reference in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)


def test_grouped_projection_cpu_keeps_reference_fallback():
    x = torch.randn(6, 2, 64, dtype=torch.bfloat16)
    weight = torch.randn(2, 32, 64, dtype=torch.bfloat16)
    actual = grouped_bf16_projection(x, weight, None, None)
    torch.testing.assert_close(
        actual, torch.einsum("tgd,grd->tgr", x, weight), rtol=0, atol=0
    )

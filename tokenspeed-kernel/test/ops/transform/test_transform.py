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
from tokenspeed_kernel import hadamard_transform

torch.manual_seed(42)


@pytest.mark.parametrize("solution", ["triton", "fast_hadamard_transform"])
def test_hadamard_transform(device: str, solution: str, require) -> None:
    dtype = torch.bfloat16
    require("transform", "hadamard_transform", solution, dtype, "x")

    x = torch.randn((3, 5, 128), device=device, dtype=dtype)
    scale = 128**-0.5

    out = hadamard_transform(x, scale=scale, solution=solution)

    expected = x.float().reshape(-1, x.shape[-1])
    h = 1
    while h < expected.shape[-1]:
        expected_3d = expected.reshape(
            expected.shape[0], expected.shape[1] // (2 * h), 2 * h
        )
        left = expected_3d[..., :h]
        right = expected_3d[..., h : 2 * h]
        combined = torch.cat((left + right, left - right), dim=-1)
        expected = combined.reshape_as(expected)
        h *= 2
    expected = expected.reshape_as(x) * scale

    assert out.shape == x.shape
    assert out.dtype == x.dtype
    torch.testing.assert_close(
        out.float(),
        expected.to(dtype).float(),
        atol=2.0e-2,
        rtol=2.0e-2,
    )

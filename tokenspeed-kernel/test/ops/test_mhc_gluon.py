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
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required for Gluon mHC tests", allow_module_level=True)

import tokenspeed_kernel  # noqa: E402


@pytest.mark.parametrize("num_tokens", [1, 6])
def test_gluon_mhc_pre_matches_triton(num_tokens: int) -> None:
    torch.manual_seed(0xA950 + num_tokens)
    residual = torch.randn(num_tokens, 4, 7168, dtype=torch.bfloat16, device="cuda")
    fn = torch.randn(24, 4 * 7168, dtype=torch.float32, device="cuda") * 0.01
    hc_scale = torch.randn(3, dtype=torch.float32, device="cuda")
    hc_base = torch.randn(24, dtype=torch.float32, device="cuda")
    args = (residual, fn, hc_scale, hc_base, 1e-6, 1e-6, 20)

    expected = tokenspeed_kernel.mhc_pre(*args, solution="triton")
    actual = tokenspeed_kernel.mhc_pre(*args, solution="gluon")

    torch.testing.assert_close(actual[0], expected[0], rtol=1e-2, atol=1e-2)
    for actual_tensor, expected_tensor in zip(actual[1:], expected[1:], strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=1e-4, atol=1e-5)

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
from tokenspeed_kernel.ops.transform.triton.hadamard import hadamard_transform_128
from tokenspeed_kernel.platform import current_platform


def _reference_hadamard_128(x: torch.Tensor, scale: float) -> torch.Tensor:
    signs = []
    for out_idx in range(128):
        row = []
        for in_idx in range(128):
            parity = (out_idx & in_idx).bit_count() & 1
            row.append(-1.0 if parity else 1.0)
        signs.append(row)
    h = torch.tensor(signs, dtype=torch.float32, device=x.device)
    return (x.float() @ h.t() * scale).to(x.dtype)


def test_hadamard_transform_128_matches_reference(device: str) -> None:
    if not current_platform().is_cdna4:
        pytest.skip("Triton Hadamard fallback is currently exercised on AMD CDNA4")

    torch.manual_seed(4)
    x = torch.randn((7, 128), device=device, dtype=torch.bfloat16)
    scale = 128**-0.5

    actual = hadamard_transform_128(x, scale=scale)
    torch.cuda.synchronize()
    expected = _reference_hadamard_128(x, scale)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

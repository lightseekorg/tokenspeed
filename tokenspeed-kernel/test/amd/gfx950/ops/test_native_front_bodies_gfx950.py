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

"""Numerics of the native decoder layer's front bodies on gfx950, each as a
test kernel (native_body_helpers.py) against an FP32 reference."""

from __future__ import annotations

import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from native_body_helpers import sender_kernel  # noqa: E402


@pytest.mark.parametrize("rows", [8, 16])
def test_sender_matches_fp32_reference(rows: int) -> None:
    """Each sender block's BF16 o_proj columns equal the FP32 product rounded
    once, and repeat bitwise."""
    torch.manual_seed(rows)
    k, n, outputs = 1024, 8192, 64
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.03
    blocks = n // outputs
    out = torch.empty(blocks * 16, outputs, device="cuda", dtype=torch.bfloat16)
    sender_kernel[(blocks,)](
        w, x, out, rows, K=k, OUTPUTS=outputs, VECTOR=2, num_warps=8
    )
    actual = out.view(blocks, 16, outputs)[:, :rows].permute(1, 0, 2).reshape(rows, n)
    expected = x.float() @ w.float().T
    torch.testing.assert_close(actual.float(), expected, atol=1e-2, rtol=2**-7)
    again = torch.empty_like(out)
    sender_kernel[(blocks,)](
        w, x, again, rows, K=k, OUTPUTS=outputs, VECTOR=2, num_warps=8
    )
    assert torch.equal(again, out)

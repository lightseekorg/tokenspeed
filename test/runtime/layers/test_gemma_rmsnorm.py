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

"""GemmaRMSNorm against the eager FP32 Gemma norm, with and without a residual."""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.layers.layernorm import GemmaRMSNorm  # noqa: E402


def _eager(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    values = x.float()
    values = values * torch.rsqrt(values.pow(2).mean(-1, keepdim=True) + eps)
    return (values * (1.0 + weight.float())).to(x.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("hidden", [2560, 10240])
@pytest.mark.parametrize("rows", [1, 4, 37])
def test_gemma_rmsnorm_matches_eager(hidden: int, rows: int) -> None:
    """The norm scales by ``1 + weight`` in FP32 and rounds once. Only the
    order of the FP32 sum of squares may differ from the eager reference,
    which moves a BF16 value by one ulp at most."""
    torch.manual_seed(rows + hidden)
    norm = GemmaRMSNorm(hidden, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
    norm.weight.data.copy_(torch.randn(hidden) * 0.3)
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16) * 4
    residual = torch.randn_like(x) * 30

    before = x.clone()
    out = norm(x)
    assert torch.equal(x, before)
    expected = _eager(x, norm.weight, 1e-6)
    torch.testing.assert_close(out.float(), expected.float(), atol=0, rtol=2**-7)
    assert (out != expected).sum().item() <= max(1, out.numel() // 1000)

    normed, new_residual = norm(x.clone(), residual.clone())
    total = x + residual
    assert torch.equal(new_residual, total)
    torch.testing.assert_close(
        normed.float(), _eager(total, norm.weight, 1e-6).float(), atol=0, rtol=2**-7
    )

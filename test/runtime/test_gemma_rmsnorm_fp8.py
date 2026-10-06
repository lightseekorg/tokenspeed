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
"""GemmaRMSNorm's fused add + norm with a static-FP8 copy for the next projection."""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci
from tokenspeed_kernel.ops.gemm.fp8_utils import static_quant_fp8

from tokenspeed.runtime.layers.layernorm import GemmaRMSNorm

register_cuda_ci(
    est_time=30,
    suite="runtime-1gpu",
    disabled_on_runners=["amd-*"],
    disabled_on_runners_reason="the FP8 copy is produced on NVIDIA only",
)


@pytest.mark.parametrize("rows", [1, 14, 112])
@pytest.mark.parametrize("with_fp8", [False, True])
def test_add_norm_with_fp8_matches_forward(rows: int, with_fp8: bool) -> None:
    torch.manual_seed(rows)
    hidden = 5120
    norm = GemmaRMSNorm(hidden, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
    norm.weight.data.copy_(torch.randn(hidden) * 0.3)
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16) * 4
    residual = torch.randn_like(x) * 30
    scale = torch.tensor([0.02], device="cuda") if with_fp8 else None

    expected, expected_residual = norm(x.clone(), residual.clone())
    normed, normed_fp8, new_residual = norm.add_norm_with_fp8(
        x.clone(), residual.clone(), scale
    )

    assert torch.equal(new_residual, expected_residual)
    # Same FP32 math; the sum of squares may add in another order.
    torch.testing.assert_close(normed, expected, atol=2e-2, rtol=1e-2)
    if with_fp8:
        quantized, _ = static_quant_fp8(normed, scale)
        assert torch.equal(normed_fp8.view(torch.uint8), quantized.view(torch.uint8))
    else:
        assert normed_fp8 is None

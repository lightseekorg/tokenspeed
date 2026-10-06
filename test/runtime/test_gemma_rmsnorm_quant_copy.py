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
"""GemmaRMSNorm's fused add + norm with a static-FP8 or NVFP4 copy for the next projection."""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci
from tokenspeed_kernel.ops.gemm.fp8_utils import static_quant_fp8
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.layers.layernorm import GemmaRMSNorm

register_cuda_ci(
    est_time=30,
    suite="runtime-1gpu",
    disabled_on_runners=["amd-*"],
    disabled_on_runners_reason="the quantized copies are produced on NVIDIA only",
)


_IS_BLACKWELL = current_platform().is_blackwell


def _norm_and_rows(rows: int, hidden: int):
    torch.manual_seed(rows + hidden)
    norm = GemmaRMSNorm(hidden, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
    norm.weight.data.copy_(torch.randn(hidden) * 0.3)
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16) * 4
    return norm, x, torch.randn_like(x) * 30


def _assert_same_norm(normed, residual, expected, expected_residual) -> None:
    assert torch.equal(residual, expected_residual)
    # Same FP32 math; the sum of squares may add in another order, so within one bf16 ulp.
    torch.testing.assert_close(normed, expected, atol=0, rtol=2**-7)
    assert (normed != expected).sum().item() <= normed.numel() // 1000


@pytest.mark.parametrize("rows", [0, 1, 14, 112])
@pytest.mark.parametrize("with_fp8", [False, True])
def test_add_norm_with_fp8_matches_forward(rows: int, with_fp8: bool) -> None:
    norm, x, residual = _norm_and_rows(rows, 5120)
    scale = torch.tensor([0.02], device="cuda") if with_fp8 else None

    expected, expected_residual = norm(x.clone(), residual.clone())
    normed, normed_fp8, new_residual = norm.add_norm_with_fp8(
        x.clone(), residual.clone(), scale
    )

    _assert_same_norm(normed, new_residual, expected, expected_residual)
    if not with_fp8:
        assert normed_fp8 is None
    elif not rows:
        assert normed_fp8.shape == x.shape
    else:
        quantized, _ = static_quant_fp8(normed, scale)
        assert torch.equal(normed_fp8.view(torch.uint8), quantized.view(torch.uint8))


@pytest.mark.parametrize("rows", [0, 1, 14, 130])
@pytest.mark.parametrize("hidden", [5120, 2560])
@pytest.mark.parametrize("with_fp4", [False, True])
def test_add_norm_with_fp4_matches_forward(
    rows: int, hidden: int, with_fp4: bool
) -> None:
    norm, x, residual = _norm_and_rows(rows, hidden)
    scale = torch.tensor([7.5], device="cuda") if with_fp4 else None

    expected, expected_residual = norm(x.clone(), residual.clone())
    normed, normed_fp4, new_residual = norm.add_norm_with_fp4(
        x.clone(), residual.clone(), scale
    )

    _assert_same_norm(normed, new_residual, expected, expected_residual)
    # Off Blackwell and at widths the kernel cannot chunk (2560) the MLP quantizes for itself.
    if not (with_fp4 and _IS_BLACKWELL and hidden == 5120):
        assert normed_fp4 is None
    elif not rows:
        assert normed_fp4[0].shape == (0, hidden // 2)
        assert normed_fp4[1].shape == (0, hidden // 16)
    else:
        from tokenspeed_kernel.ops.quantization.flashinfer import fp4_quantize

        values, scales = fp4_quantize(normed, scale)
        assert torch.equal(normed_fp4[0], values.view(torch.uint8))
        assert torch.equal(normed_fp4[1], scales.view(torch.uint8))


@pytest.mark.skipif(not _IS_BLACKWELL, reason="the NVFP4 copy is made on Blackwell")
@pytest.mark.parametrize(
    "name",
    [
        "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH",
        "TRTLLM_DISABLE_FP4_QUANT_FAST_MATH",
        "FLASHINFER_NVFP4_4OVER6",
    ],
)
@pytest.mark.parametrize("value", ["1", "true"])
def test_add_norm_with_fp4_skips_other_fp4_quantize_recipes(
    monkeypatch, name: str, value: str
) -> None:
    # fp4_quantize leaves its default recipe only when one of these is exactly "1".
    monkeypatch.setenv(name, value)
    norm, x, residual = _norm_and_rows(14, 5120)

    _, normed_fp4, _ = norm.add_norm_with_fp4(
        x, residual, torch.tensor([7.5], device="cuda")
    )

    assert (normed_fp4 is None) == (value == "1")


def test_add_norm_with_fp4_copies_bf16_rows_only() -> None:
    torch.manual_seed(0)
    norm = GemmaRMSNorm(5120, eps=1e-6).to(device="cuda", dtype=torch.float16)
    norm.weight.data.normal_(0, 0.3)
    x = torch.randn(14, 5120, device="cuda", dtype=torch.float16)
    residual = torch.randn_like(x)

    expected, expected_residual = norm(x.clone(), residual.clone())
    normed, normed_fp4, new_residual = norm.add_norm_with_fp4(
        x.clone(), residual.clone(), torch.tensor([7.5], device="cuda")
    )

    _assert_same_norm(normed, new_residual, expected, expected_residual)
    assert normed_fp4 is None

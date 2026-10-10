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

"""The TRT-LLM per-token FP8 quantizer fills its caller's scale buffer (CPU).

``fp8_utils.per_token_quant_fp8`` hands the wrapper an ``[M, 1]`` scale buffer,
``quantize_fp8_with_scale(granularity="token")`` an ``[M]`` one. The wrappers
run as they are; only ``torch.ops.tensorrt_llm.quantize_e4m3_activation`` is
replaced, returning one scale per token as ``[M, 1]`` or as ``[M]``.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform().is_nvidia, reason="the TRT-LLM quantizer is NVIDIA's"
)

_FP8 = torch.float8_e4m3fn


@pytest.fixture(params=["M1", "M"])
def fake_op(request, monkeypatch):
    """The activation quantizer on CPU: amax / 448 per token, its scales
    shaped ``[M, 1]`` or ``[M]``."""

    def quantize(x: torch.Tensor):
        scale = x.float().abs().amax(-1, keepdim=True) / 448.0
        codes = (x.float() / scale).clamp(-448, 448).to(_FP8)
        return codes, scale if request.param == "M1" else scale.squeeze(-1)

    monkeypatch.setattr(
        torch.ops.tensorrt_llm, "quantize_e4m3_activation", quantize, raising=False
    )


def _expected(x: torch.Tensor):
    scale = x.float().abs().amax(-1, keepdim=True) / 448.0
    return (x.float() / scale).clamp(-448, 448).to(_FP8), scale


@pytest.mark.parametrize("m", [1, 3])
def test_per_token_quant_fp8_keeps_its_m_by_1_scales(fake_op, m):
    from tokenspeed_kernel.ops.gemm import fp8_utils

    x = (torch.randn(m, 16, generator=torch.Generator().manual_seed(m)) * 3).to(
        torch.bfloat16
    )
    codes, scale = fp8_utils.per_token_quant_fp8(x)
    expected_codes, expected_scale = _expected(x)
    assert scale.dtype == torch.float32 and tuple(scale.shape) == (m, 1)
    assert torch.equal(scale, expected_scale)
    assert torch.equal(codes.view(torch.uint8), expected_codes.view(torch.uint8))


@pytest.mark.parametrize("m", [1, 3])
def test_registered_token_quantizer_returns_m_by_1_scales(fake_op, m):
    from tokenspeed_kernel.ops.quantization import trtllm

    x = (torch.randn(m, 16, generator=torch.Generator().manual_seed(m)) * 3).to(
        torch.bfloat16
    )
    codes, scale = trtllm.trtllm_quantize_fp8_with_scale(x, granularity="token")
    expected_codes, expected_scale = _expected(x)
    assert scale.dtype == torch.float32 and tuple(scale.shape) == (m, 1)
    assert torch.equal(scale, expected_scale)
    assert torch.equal(codes.view(torch.uint8), expected_codes.view(torch.uint8))
    assert torch.equal(trtllm.trtllm_fp8_token(x), expected_codes.float())

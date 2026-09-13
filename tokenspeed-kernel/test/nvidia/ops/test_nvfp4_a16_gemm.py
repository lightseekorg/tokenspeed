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
from tokenspeed_kernel.thirdparty.msa.cute.quantize import (
    swizzle_nvfp4_scale_to_128x4,
)

_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _requires_flashinfer_nvfp4_a16() -> str | None:
    if not torch.cuda.is_available():
        return "requires CUDA"
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        return "FlashInfer CuTe-DSL W4A16 requires SM100 or SM103"
    import tokenspeed_kernel

    if not tokenspeed_kernel.has_flashinfer_cute_dsl_nvfp4_a16():
        return "FlashInfer CuTe-DSL W4A16 entry points are unavailable"
    return None


_reason = _requires_flashinfer_nvfp4_a16()
requires_flashinfer_nvfp4_a16 = pytest.mark.skipif(
    _reason is not None, reason=str(_reason)
)


def _quantize_nvfp4_torch(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize to packed E2M1 with float8 block-16 scales using only torch."""
    n, k = weight.shape
    assert k % 16 == 0
    blocks = weight.float().reshape(n, k // 16, 16)
    scales = (blocks.abs().amax(dim=-1) / 6.0).clamp_min(2**-9)
    scales = scales.to(torch.float8_e4m3fn)
    normalized = blocks / scales.float().unsqueeze(-1)

    magnitudes = torch.tensor(
        _E2M1_MAGNITUDES, dtype=torch.float32, device=weight.device
    )
    codes = (
        (normalized.abs().unsqueeze(-1) - magnitudes.view(1, 1, 1, -1))
        .abs()
        .argmin(dim=-1)
        .to(torch.uint8)
    )
    codes |= (normalized < 0).to(torch.uint8) << 3
    codes = codes.reshape(n, k)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    swizzled = swizzle_nvfp4_scale_to_128x4(scales, rows=n, cols=k // 16)
    return packed.contiguous(), swizzled, scales


def _dequantize_nvfp4(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    values = torch.tensor(
        _E2M1_MAGNITUDES + tuple(-value for value in _E2M1_MAGNITUDES),
        dtype=torch.float32,
        device=packed.device,
    )
    low = values[(packed & 0x0F).long()]
    high = values[(packed >> 4).long()]
    weight = torch.stack((low, high), dim=-1).reshape(packed.shape[0], -1)
    return weight * scales.float().repeat_interleave(16, dim=1)


@requires_flashinfer_nvfp4_a16
@pytest.mark.parametrize("m", [1, 17])
def test_flashinfer_nvfp4_a16_matches_dequantized_reference(m: int) -> None:
    import tokenspeed_kernel

    torch.manual_seed(20260902 + m)
    n = k = 256
    activation = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.float32) * 0.05
    packed, swizzled_scales, scales = _quantize_nvfp4_torch(weight)
    prepared_weight, prepared_scales, prepared_alpha = (
        tokenspeed_kernel.prepare_nvfp4_a16_weights(packed, swizzled_scales, alpha=None)
    )

    actual = tokenspeed_kernel.mm(
        activation,
        prepared_weight,
        B_scales=prepared_scales,
        alpha=prepared_alpha,
        out_dtype=torch.bfloat16,
        quant="nvfp4_a16",
    )
    expected = (activation.float() @ _dequantize_nvfp4(packed, scales).T).to(
        torch.bfloat16
    )

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=6e-2)


@requires_flashinfer_nvfp4_a16
def test_flashinfer_nvfp4_a16_writes_out_buffer() -> None:
    import tokenspeed_kernel

    torch.manual_seed(7)
    m, n, k = 3, 256, 256
    activation = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    packed, swizzled_scales, _ = _quantize_nvfp4_torch(
        torch.randn(n, k, device="cuda") * 0.05
    )
    prepared_weight, prepared_scales, alpha = (
        tokenspeed_kernel.prepare_nvfp4_a16_weights(packed, swizzled_scales)
    )
    out_storage = torch.empty(m, n + 1, device="cuda", dtype=torch.bfloat16)
    out = out_storage[:, :n]
    assert not out.is_contiguous() and out.stride(-1) == 1

    actual = tokenspeed_kernel.mm(
        activation,
        prepared_weight,
        B_scales=prepared_scales,
        alpha=alpha,
        out=out,
        quant="nvfp4_a16",
    )

    assert actual.data_ptr() == out.data_ptr()


@requires_flashinfer_nvfp4_a16
def test_prepare_nvfp4_a16_normalizes_scalar_alpha() -> None:
    import tokenspeed_kernel

    packed = torch.zeros(128, 64, dtype=torch.uint8, device="cuda")
    scales = swizzle_nvfp4_scale_to_128x4(
        torch.ones(128, 8, dtype=torch.float8_e4m3fn, device="cuda"),
        rows=128,
        cols=8,
    )
    alpha = torch.tensor(0.25, dtype=torch.bfloat16, device="cuda")

    _, prepared_scales, prepared_alpha = tokenspeed_kernel.prepare_nvfp4_a16_weights(
        packed, scales, alpha
    )

    assert prepared_scales.ndim == 6
    assert prepared_alpha is not None
    assert prepared_alpha.shape == (1,)
    assert prepared_alpha.dtype == torch.float32
    assert prepared_alpha.item() == pytest.approx(0.25)


@requires_flashinfer_nvfp4_a16
def test_flashinfer_nvfp4_a16_rejects_unprepared_scales() -> None:
    import tokenspeed_kernel

    activation = torch.zeros(1, 128, dtype=torch.bfloat16, device="cuda")
    packed = torch.zeros(128, 64, dtype=torch.uint8, device="cuda")
    raw_scales = torch.ones(128, 8, dtype=torch.float8_e4m3fn, device="cuda")

    with pytest.raises(ValueError, match="prepare_nvfp4_a16_weights"):
        tokenspeed_kernel.mm(
            activation,
            packed,
            B_scales=raw_scales,
            quant="nvfp4_a16",
        )

    malformed_six_dimensional_scales = torch.ones(
        1, 1, 32, 4, 2, 1, dtype=torch.float8_e4m3fn, device="cuda"
    )
    with pytest.raises(ValueError, match="prepare_nvfp4_a16_weights"):
        tokenspeed_kernel.mm(
            activation,
            packed,
            B_scales=malformed_six_dimensional_scales,
            quant="nvfp4_a16",
        )

    with pytest.raises(ValueError, match="A_scales=None"):
        tokenspeed_kernel.mm(
            activation,
            packed,
            A_scales=torch.ones(1, device="cuda"),
            B_scales=raw_scales,
            quant="nvfp4_a16",
        )

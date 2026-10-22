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
from tokenspeed_kernel.numerics.torch.quantization import (
    dequantize_nvfp4_blocks,
    quantize_nvfp4_blocks,
)
from tokenspeed_kernel.ops.gemm import (
    has_flashinfer_cute_dsl_nvfp4_a16 as kernel_has_flashinfer_cute_dsl_nvfp4_a16,
)
from tokenspeed_kernel.ops.gemm import mm as kernel_mm
from tokenspeed_kernel.ops.gemm import (
    prepare_nvfp4_a16_weights as kernel_prepare_nvfp4_a16_weights,
)
from tokenspeed_kernel.thirdparty.msa.cute.quantize import (
    swizzle_nvfp4_scale_to_128x4,
)


def _requires_flashinfer_nvfp4_a16() -> str | None:
    if not torch.cuda.is_available():
        return "requires CUDA"
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        return "FlashInfer CuTe-DSL W4A16 requires SM100 or SM103"

    if not kernel_has_flashinfer_cute_dsl_nvfp4_a16():
        return "FlashInfer CuTe-DSL W4A16 entry points are unavailable"
    return None


_reason = _requires_flashinfer_nvfp4_a16()
requires_flashinfer_nvfp4_a16 = pytest.mark.skipif(
    _reason is not None, reason=str(_reason)
)


def _quantize_nvfp4_torch(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Packed E2M1 codes, 128x4-swizzled and linear float8 block-16 scales."""
    packed, scales = quantize_nvfp4_blocks(weight)
    n, k = weight.shape
    swizzled = swizzle_nvfp4_scale_to_128x4(scales, rows=n, cols=k // 16)
    return packed, swizzled, scales


@requires_flashinfer_nvfp4_a16
@pytest.mark.parametrize("m", [1, 17])
def test_flashinfer_nvfp4_a16_matches_dequantized_reference(m: int) -> None:

    torch.manual_seed(20260902 + m)
    n = k = 256
    activation = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.float32) * 0.05
    packed, swizzled_scales, scales = _quantize_nvfp4_torch(weight)
    prepared_weight, prepared_scales, prepared_alpha = kernel_prepare_nvfp4_a16_weights(
        packed, swizzled_scales, alpha=None
    )

    actual = kernel_mm(
        activation,
        prepared_weight,
        B_scales=prepared_scales,
        alpha=prepared_alpha,
        out_dtype=torch.bfloat16,
        quant="nvfp4_a16",
    )
    expected = (activation.float() @ dequantize_nvfp4_blocks(packed, scales).T).to(
        torch.bfloat16
    )

    # E2M1 x E4M3 products are exact in bf16, so the kernel and the reference
    # differ by at most one bf16 ulp (2^-7 relative).
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@requires_flashinfer_nvfp4_a16
def test_flashinfer_nvfp4_a16_writes_out_buffer() -> None:

    torch.manual_seed(7)
    m, n, k = 3, 256, 256
    activation = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    packed, swizzled_scales, _ = _quantize_nvfp4_torch(
        torch.randn(n, k, device="cuda") * 0.05
    )
    prepared_weight, prepared_scales, alpha = kernel_prepare_nvfp4_a16_weights(
        packed, swizzled_scales
    )
    out_storage = torch.empty(m, n + 1, device="cuda", dtype=torch.bfloat16)
    out = out_storage[:, :n]
    assert not out.is_contiguous() and out.stride(-1) == 1

    actual = kernel_mm(
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

    packed = torch.zeros(128, 64, dtype=torch.uint8, device="cuda")
    scales = swizzle_nvfp4_scale_to_128x4(
        torch.ones(128, 8, dtype=torch.float8_e4m3fn, device="cuda"),
        rows=128,
        cols=8,
    )
    alpha = torch.tensor(0.25, dtype=torch.bfloat16, device="cuda")

    _, prepared_scales, prepared_alpha = kernel_prepare_nvfp4_a16_weights(
        packed, scales, alpha
    )

    assert prepared_scales.ndim == 6
    assert prepared_alpha is not None
    assert prepared_alpha.shape == (1,)
    assert prepared_alpha.dtype == torch.float32
    assert prepared_alpha.item() == pytest.approx(0.25)


@requires_flashinfer_nvfp4_a16
def test_flashinfer_nvfp4_a16_rejects_unprepared_scales() -> None:

    activation = torch.zeros(1, 128, dtype=torch.bfloat16, device="cuda")
    packed = torch.zeros(128, 64, dtype=torch.uint8, device="cuda")
    raw_scales = torch.ones(128, 8, dtype=torch.float8_e4m3fn, device="cuda")

    with pytest.raises(ValueError, match="prepare_nvfp4_a16_weights"):
        kernel_mm(
            activation,
            packed,
            B_scales=raw_scales,
            quant="nvfp4_a16",
        )

    malformed_six_dimensional_scales = torch.ones(
        1, 1, 32, 4, 2, 1, dtype=torch.float8_e4m3fn, device="cuda"
    )
    with pytest.raises(ValueError, match="prepare_nvfp4_a16_weights"):
        kernel_mm(
            activation,
            packed,
            B_scales=malformed_six_dimensional_scales,
            quant="nvfp4_a16",
        )

    with pytest.raises(ValueError, match="A_scales=None"):
        kernel_mm(
            activation,
            packed,
            A_scales=torch.ones(1, device="cuda"),
            B_scales=raw_scales,
            quant="nvfp4_a16",
        )

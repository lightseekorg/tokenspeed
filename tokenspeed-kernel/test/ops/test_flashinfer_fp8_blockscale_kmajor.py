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

"""FlashInfer FP8 block-scale K-major (copy-free canonical) path tests."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.gemm.flashinfer import gemm_fp8_nt_groupwise
from tokenspeed_kernel.platform import ArchVersion, current_platform
from tokenspeed_kernel.registry import error_fn


def _kmajor_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    if gemm_fp8_nt_groupwise is error_fn:
        return False
    arch = current_platform().arch_version
    return ArchVersion(10, 0) <= arch <= ArchVersion(10, 3)


pytestmark = pytest.mark.skipif(
    not _kmajor_supported(),
    reason="requires SM10x CUDA and FlashInfer FP8 block-scale GEMM",
)


def _quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = x.shape
    grouped = x.view(m, k // 128, 128)
    scales = (grouped.abs().amax(-1).clamp(min=1e-6) / 448.0).float()
    quantized = (grouped / scales[..., None]).to(torch.float8_e4m3fn)
    return quantized.view(m, k), scales


def _make_case(m: int, n: int, k: int, device: str):
    torch.manual_seed(m)
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    weight = (torch.randn(n, k, device=device) * 0.02).to(torch.float8_e4m3fn)
    weight_scales = (
        torch.rand(n // 128, k // 128, device=device, dtype=torch.float32) * 0.02
        + 0.001
    )
    quantized, activation_scales = _quantize(x)
    return quantized, weight, activation_scales, weight_scales


def _mn_reference(
    quantized: torch.Tensor,
    weight: torch.Tensor,
    activation_scales: torch.Tensor,
    weight_scales: torch.Tensor,
) -> torch.Tensor:
    """The pre-K-major canonical path: pad to M%4==0 plus MN transposes."""
    m = quantized.shape[0]
    padded_m = (m + 3) // 4 * 4
    padded = quantized.new_zeros((padded_m, quantized.shape[1]))
    padded[:m] = quantized
    padded_scales = activation_scales.new_ones((padded_m, activation_scales.shape[1]))
    padded_scales[:m] = activation_scales
    return gemm_fp8_nt_groupwise(
        padded,
        weight,
        padded_scales.transpose(0, 1).contiguous(),
        weight_scales.transpose(0, 1).contiguous(),
        scale_major_mode="MN",
        out_dtype=torch.bfloat16,
    )[:m]


@pytest.mark.parametrize("m", [1, 3, 4, 9, 288])
def test_kmajor_canonical_matches_mn_path(device: str, m: int) -> None:
    from tokenspeed_kernel.ops.gemm.flashinfer import flashinfer_mm_fp8_blockscale

    n, k = 256, 512
    quantized, weight, activation_scales, weight_scales = _make_case(m, n, k, device)
    expected = _mn_reference(quantized, weight, activation_scales, weight_scales)
    result = flashinfer_mm_fp8_blockscale(
        quantized,
        weight,
        activation_scales,
        weight_scales,
        torch.bfloat16,
        block_size=[128, 128],
    )
    torch.testing.assert_close(result, expected, atol=0, rtol=0)


def test_kmajor_out_direct_and_strided_fallback(device: str) -> None:
    from tokenspeed_kernel.ops.gemm.flashinfer import flashinfer_mm_fp8_blockscale

    m, n, k = 9, 256, 512
    quantized, weight, activation_scales, weight_scales = _make_case(m, n, k, device)
    expected = _mn_reference(quantized, weight, activation_scales, weight_scales)

    contiguous_out = torch.empty(m, n, device=device, dtype=torch.bfloat16)
    returned = flashinfer_mm_fp8_blockscale(
        quantized,
        weight,
        activation_scales,
        weight_scales,
        torch.bfloat16,
        block_size=[128, 128],
        out=contiguous_out,
    )
    assert returned.data_ptr() == contiguous_out.data_ptr()
    torch.testing.assert_close(contiguous_out, expected, atol=0, rtol=0)

    backing = torch.zeros(m, n + 64, device=device, dtype=torch.bfloat16)
    strided_out = backing[:, :n]
    flashinfer_mm_fp8_blockscale(
        quantized,
        weight,
        activation_scales,
        weight_scales,
        torch.bfloat16,
        block_size=[128, 128],
        out=strided_out,
    )
    torch.testing.assert_close(strided_out, expected, atol=0, rtol=0)
    assert torch.count_nonzero(backing[:, n:]).item() == 0


def test_kmajor_slices_padded_activation_scales(device: str) -> None:
    from tokenspeed_kernel.ops.gemm.flashinfer import flashinfer_mm_fp8_blockscale

    m, n, k = 9, 256, 512
    quantized, weight, activation_scales, weight_scales = _make_case(m, n, k, device)
    expected = _mn_reference(quantized, weight, activation_scales, weight_scales)
    padded_scales = activation_scales.new_ones((12, activation_scales.shape[1]))
    padded_scales[:m] = activation_scales
    result = flashinfer_mm_fp8_blockscale(
        quantized,
        weight,
        padded_scales,
        weight_scales,
        torch.bfloat16,
        block_size=[128, 128],
        original_m=m,
    )
    torch.testing.assert_close(result, expected, atol=0, rtol=0)

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

"""GFX950 group-32 UE8M0 FP8 quantization is bit-exact with the Triton kernel."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.quantization import quantize_fp8
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel_amd.ops.gfx950 import quantization  # noqa: E402

DEVICE = "cuda"


def _quantize(x: torch.Tensor, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    return quantize_fp8(
        x,
        granularity="token_group",
        group_size=32,
        scale_encoding="ue8m0",
        enable_pdl=False,
        override=name,
        solution=None,
    )


def _hard_input(rows: int, k: int, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator(device=DEVICE).manual_seed(rows * k)
    x = torch.randn(rows, k, device=DEVICE, generator=generator)
    # Sweep magnitudes so groups hit tiny, normal and large scales.
    x = x * torch.logspace(-10, 10, k, device=DEVICE)
    x = x.to(dtype)
    x[0, :32] = 0  # The 1e-4 amax floor.
    x[0, 32:64] = 1e-30  # Tiny values below the floor.
    if k >= 128:
        x[0, 64:96] = 448.0 * 2.0**-3  # amax / 448 exactly a power of two.
        x[0, 96:128] = torch.nextafter(
            torch.tensor(448.0 * 2.0**-3), torch.tensor(1e9)
        ).to(dtype)
    return x


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("k", [576, 1280, 2048, 5120])
@pytest.mark.parametrize("rows", [1, 6, 192, 641])
def test_gluon_group32_quantize_bit_exact(rows: int, k: int, dtype) -> None:
    x = _hard_input(rows, k, dtype)
    codes, scales = _quantize(x, "gluon_quantize_fp8_group32_ue8m0_gfx950")
    ref_codes, ref_scales = _quantize(x, "triton_quantize_fp8_group32_ue8m0")
    assert torch.equal(codes.view(torch.uint8), ref_codes.view(torch.uint8))
    assert torch.equal(scales, ref_scales)


def test_gluon_group32_quantize_strided_rows() -> None:
    x = _hard_input(64, 2048, torch.bfloat16)[:, :1024]
    codes, scales = _quantize(x, "gluon_quantize_fp8_group32_ue8m0_gfx950")
    ref_codes, ref_scales = _quantize(x, "triton_quantize_fp8_group32_ue8m0")
    assert torch.equal(codes.view(torch.uint8), ref_codes.view(torch.uint8))
    assert torch.equal(scales, ref_scales)


def test_gluon_group32_quantize_no_recompile_across_batches() -> None:
    x = torch.randn(300, 5120, device=DEVICE, dtype=torch.bfloat16)
    quantization.launch_gluon_quantize_fp8_group32_ue8m0_gfx950(x[:1])
    with assert_no_triton_compile(quantization.gluon_quantize_fp8_group32_ue8m0_gfx950):
        for rows in (2, 6, 16, 48, 129, 192, 300):
            quantization.launch_gluon_quantize_fp8_group32_ue8m0_gfx950(x[:rows])

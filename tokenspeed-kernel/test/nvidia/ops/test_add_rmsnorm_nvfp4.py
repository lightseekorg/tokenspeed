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
"""add_rmsnorm's NVFP4 copy is fp4_quantize of its own normed rows, bit for bit."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.layernorm import add_rmsnorm
from tokenspeed_kernel.ops.quantization.flashinfer import fp4_quantize
from tokenspeed_kernel.platform import current_platform

_platform = current_platform()
pytestmark = pytest.mark.skipif(
    not _platform.is_blackwell, reason="the NVFP4 copy uses Blackwell E2M1 conversions"
)


@pytest.mark.parametrize("rows", [1, 14, 130])
@pytest.mark.parametrize("gemma", [False, True])
@pytest.mark.parametrize("spread", [0, 12])
def test_nvfp4_copy_matches_fp4_quantize(rows: int, gemma: bool, spread: int) -> None:
    torch.manual_seed(rows * 7 + spread)
    cols = 5120
    x = torch.randn(rows, cols, device="cuda")
    if spread:
        # Blocks scaled far apart exercise every E4M3 scale and E2M1 rounding boundary.
        exponents = torch.randint(
            -spread, spread + 1, (rows, cols // 16, 1), device="cuda"
        )
        x = (x.view(rows, cols // 16, 16) * torch.exp2(exponents)).view(rows, cols)
    x = x.bfloat16()
    residual = torch.randn_like(x)
    weight = (torch.randn(cols, device="cuda") * 0.3).bfloat16()
    global_scale = torch.tensor([7.5], device="cuda")
    out = torch.empty_like(x)
    values = torch.empty(rows, cols // 2, dtype=torch.uint8, device="cuda")
    padded = (rows + 127) // 128 * 128
    scales = torch.full((padded, cols // 16), 0xFF, dtype=torch.uint8, device="cuda")

    add_rmsnorm(
        x,
        residual,
        weight,
        1e-6,
        x2=None,
        out=out,
        out_fp8=None,
        fp8_scale=None,
        out_fp4=(values, scales),
        fp4_scale=global_scale,
        gemma=gemma,
    )

    expected_values, expected_scales = fp4_quantize(out, global_scale)
    assert torch.equal(values, expected_values.view(torch.uint8))
    assert torch.equal(scales, expected_scales.view(torch.uint8))


def test_nvfp4_copy_contract() -> None:
    x = torch.randn(4, 5120, device="cuda").bfloat16()
    weight = torch.ones(5120, device="cuda").bfloat16()
    values = torch.empty(4, 2560, dtype=torch.uint8, device="cuda")
    scales = torch.empty(128, 320, dtype=torch.uint8, device="cuda")
    common = dict(x2=None, out=torch.empty_like(x), gemma=False)
    with pytest.raises(ValueError, match="together"):
        add_rmsnorm(
            x,
            x.clone(),
            weight,
            1e-6,
            out_fp8=None,
            fp8_scale=None,
            out_fp4=(values, scales),
            fp4_scale=None,
            **common,
        )
    with pytest.raises(ValueError, match="one quantized copy"):
        add_rmsnorm(
            x,
            x.clone(),
            weight,
            1e-6,
            out_fp8=torch.empty_like(x, dtype=torch.float8_e4m3fn),
            fp8_scale=torch.ones(1, device="cuda"),
            out_fp4=(values, scales),
            fp4_scale=torch.ones(1, device="cuda"),
            **common,
        )
    with pytest.raises(ValueError, match="fp4_quantize"):
        add_rmsnorm(
            x,
            x.clone(),
            weight,
            1e-6,
            out_fp8=None,
            fp8_scale=None,
            out_fp4=(values, scales[:64]),
            fp4_scale=torch.ones(1, device="cuda"),
            **common,
        )

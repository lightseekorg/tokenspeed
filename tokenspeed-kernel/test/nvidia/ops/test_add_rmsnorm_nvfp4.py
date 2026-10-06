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
from tokenspeed_kernel.ops.layernorm import add_rmsnorm, nvfp4_copy_supported
from tokenspeed_kernel.ops.layernorm.triton import _add_rmsnorm_kernel
from tokenspeed_kernel.ops.quantization.flashinfer import fp4_quantize
from tokenspeed_kernel.platform import current_platform, pdl_enabled
from utils import assert_no_triton_compile

_platform = current_platform()
pytestmark = pytest.mark.skipif(
    not _platform.is_blackwell, reason="the NVFP4 copy uses Blackwell E2M1 conversions"
)
_RECIPE_ENV = (
    "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH",
    "TRTLLM_DISABLE_FP4_QUANT_FAST_MATH",
    "FLASHINFER_NVFP4_4OVER6",
)


@pytest.fixture
def pdl_state():
    previous = pdl_enabled()
    yield
    pdl_enabled(overwrite=previous)


def _rows(rows: int, cols: int, spread: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    x = torch.randn(rows, cols, device="cuda")
    if spread:
        # Blocks scaled far apart exercise every E4M3 scale and E2M1 rounding boundary.
        exponents = torch.randint(
            -spread, spread + 1, (rows, cols // 16, 1), device="cuda"
        )
        x = (x.view(rows, cols // 16, 16) * torch.exp2(exponents)).view(rows, cols)
    return x.bfloat16()


def _nvfp4_copy(x, residual, weight, global_scale, *, x2, out, gemma):
    rows, cols = x.shape
    values = torch.empty(rows, cols // 2, dtype=torch.uint8, device="cuda")
    padded = (rows + 127) // 128 * 128
    # Prefilled so the zeroed scales of the padding rows are checked too.
    scales = torch.full((padded, cols // 16), 0xFF, dtype=torch.uint8, device="cuda")
    add_rmsnorm(
        x,
        residual,
        weight,
        1e-6,
        x2=x2,
        out=out,
        out_fp8=None,
        fp8_scale=None,
        out_fp4=(values, scales),
        fp4_scale=global_scale,
        gemma=gemma,
    )
    expected_values, expected_scales = fp4_quantize(out.contiguous(), global_scale)
    assert torch.equal(values, expected_values.view(torch.uint8))
    assert torch.equal(scales, expected_scales.view(torch.uint8))
    return scales


@pytest.mark.parametrize("cols", [512, 3072, 4096, 5120, 6144])
@pytest.mark.parametrize("rows", [1, 14, 129])
@pytest.mark.parametrize("gemma", [False, True])
@pytest.mark.parametrize("spread", [0, 12])
@pytest.mark.parametrize("with_x2", [False, True])
@pytest.mark.parametrize("pdl", [False, True])
def test_nvfp4_copy_matches_fp4_quantize(
    cols: int, rows: int, gemma: bool, spread: int, with_x2: bool, pdl: bool, pdl_state
) -> None:
    pdl_enabled(overwrite=pdl)
    x = _rows(rows, cols, spread, seed=rows * 7 + spread + cols)
    x2 = torch.randn_like(x) if with_x2 else None
    weight = (torch.randn(cols, device="cuda") * 0.3).bfloat16()
    global_scale = torch.tensor([7.5], device="cuda")

    _nvfp4_copy(
        x,
        torch.randn_like(x),
        weight,
        global_scale,
        x2=x2,
        out=torch.empty_like(x),
        gemma=gemma,
    )


@pytest.mark.parametrize("rows", [129, 1025])
def test_nvfp4_copy_of_extreme_rows_in_place(rows: int) -> None:
    # In place in strided rows: NaN, Inf, a flushed subnormal block, scales past E4M3's 448, TMA past 1024 rows.
    cols = 5120
    wide = torch.zeros(rows, 2 * cols, device="cuda").bfloat16()
    x = wide[:, :cols]
    x.copy_(_rows(rows, cols, 16, seed=rows))
    x[0] = float("nan")
    x[1] = torch.randn(cols, device="cuda")
    x[1, 16:32] = 2e-39
    x[64, 3] = float("-inf")
    weight = (torch.randn(cols, device="cuda") * 0.3).bfloat16()
    weight[7] = 3e38
    global_scale = torch.tensor([1000.0], device="cuda")

    scales = _nvfp4_copy(
        x, torch.zeros_like(x), weight, global_scale, x2=None, out=x, gemma=False
    )

    assert torch.equal(wide[:, cols:], torch.zeros_like(x))
    assert x.isinf().any() and (scales[:rows] == 0x7E).any()
    assert 0 < x[1, 16:32].float().abs().max() < torch.finfo(torch.float32).tiny


def test_nvfp4_copy_compiles_once_across_batch_sizes() -> None:
    cols = 5120
    weight = torch.ones(cols, device="cuda").bfloat16()
    global_scale = torch.tensor([7.5], device="cuda")

    def run(rows: int) -> None:
        x = torch.randn(rows, cols, device="cuda").bfloat16()
        _nvfp4_copy(
            x,
            torch.randn_like(x),
            weight,
            global_scale,
            x2=None,
            out=torch.empty_like(x),
            gemma=True,
        )

    run(3)
    with assert_no_triton_compile(_add_rmsnorm_kernel):
        for rows in (1, 100, 128, 129, 300):
            run(rows)


def test_nvfp4_copy_supported_widths_and_recipes(monkeypatch) -> None:
    for cols in (1024, 3072, 4096, 5120, 6144):
        assert nvfp4_copy_supported(cols)
    # 2560's 512-wide chunks are under 1024, so it gets one masked 4096-lane block; 2000 has partial scale groups.
    for cols in (1536, 2000, 2560, 3584):
        assert not nvfp4_copy_supported(cols)
    for name in _RECIPE_ENV:
        monkeypatch.setenv(name, "1")
        assert not nvfp4_copy_supported(5120)
        # Like fp4_quantize, only exactly "1" switches the recipe.
        monkeypatch.setenv(name, "true")
        assert nvfp4_copy_supported(5120)
        monkeypatch.delenv(name)


@pytest.mark.parametrize(
    "cols,scale_rows,match", [(5120, 64, "fp4_quantize"), (2560, 128, "no NVFP4 copy")]
)
def test_nvfp4_copy_rejects_what_it_cannot_write(
    cols: int, scale_rows: int, match: str
) -> None:
    x = torch.randn(4, cols, device="cuda").bfloat16()
    values = torch.empty(4, cols // 2, dtype=torch.uint8, device="cuda")
    scales = torch.empty(scale_rows, cols // 16, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match=match):
        add_rmsnorm(
            x,
            x.clone(),
            torch.ones(cols, device="cuda").bfloat16(),
            1e-6,
            x2=None,
            out=torch.empty_like(x),
            out_fp8=None,
            fp8_scale=None,
            out_fp4=(values, scales),
            fp4_scale=torch.ones(1, device="cuda"),
            gemma=False,
        )

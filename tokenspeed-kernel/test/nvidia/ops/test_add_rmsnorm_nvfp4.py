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


def _fused(x, residual, weight, global_scale, *, x2=None, gemma):
    rows, cols = x.shape
    out = torch.empty_like(x)
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
    return out, values, scales


def _assert_matches_fp4_quantize(out, values, scales, global_scale) -> None:
    expected_values, expected_scales = fp4_quantize(out, global_scale)
    assert torch.equal(values, expected_values.view(torch.uint8))
    assert torch.equal(scales, expected_scales.view(torch.uint8))


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
    residual = torch.randn_like(x)
    weight = (torch.randn(cols, device="cuda") * 0.3).bfloat16()
    global_scale = torch.tensor([7.5], device="cuda")

    out, values, scales = _fused(x, residual, weight, global_scale, x2=x2, gemma=gemma)

    _assert_matches_fp4_quantize(out, values, scales, global_scale)


def test_nvfp4_copy_in_place_at_prefill_rows() -> None:
    # Past 1024 rows fp4_quantize switches to its TMA kernel; the runtime normalizes in place.
    rows, cols = 1025, 5120
    x = _rows(rows, cols, 12, seed=3)
    residual = torch.randn_like(x)
    weight = (torch.randn(cols, device="cuda") * 0.3).bfloat16()
    global_scale = torch.tensor([7.5], device="cuda")
    values = torch.empty(rows, cols // 2, dtype=torch.uint8, device="cuda")
    scales = torch.full((1152, cols // 16), 0xFF, dtype=torch.uint8, device="cuda")

    add_rmsnorm(
        x,
        residual,
        weight,
        1e-6,
        x2=None,
        out=x,
        out_fp8=None,
        fp8_scale=None,
        out_fp4=(values, scales),
        fp4_scale=global_scale,
        gemma=True,
    )

    _assert_matches_fp4_quantize(x, values, scales, global_scale)


@pytest.mark.parametrize("cols", [512, 5120])
def test_nvfp4_copy_of_strided_rows_with_nan_and_inf(cols: int) -> None:
    rows = 129
    torch.manual_seed(cols)
    wide = torch.randn(rows, 2 * cols, device="cuda").bfloat16()
    neighbours = wide[:, cols:].clone()
    x = wide[:, :cols]
    x[0] = float("nan")
    x[64, 7] = float("inf")
    x[128, 3] = float("-inf")
    residual = torch.randn(rows, cols, device="cuda").bfloat16()
    weight = (torch.randn(cols, device="cuda") * 0.3).bfloat16()
    global_scale = torch.tensor([7.5], device="cuda")
    values = torch.empty(rows, cols // 2, dtype=torch.uint8, device="cuda")
    scales = torch.full((256, cols // 16), 0xFF, dtype=torch.uint8, device="cuda")

    add_rmsnorm(
        x,
        residual,
        weight,
        1e-6,
        x2=None,
        out=x,
        out_fp8=None,
        fp8_scale=None,
        out_fp4=(values, scales),
        fp4_scale=global_scale,
        gemma=True,
    )

    assert torch.equal(wide[:, cols:], neighbours)
    _assert_matches_fp4_quantize(x.contiguous(), values, scales, global_scale)


def test_nvfp4_copy_flushes_subnormal_blocks() -> None:
    # fp4_quantize is built flush-to-zero: a block of subnormal values quantizes as an all-zero block.
    rows, cols = 4, 5120
    x = torch.randn(rows, cols, device="cuda").bfloat16()
    x[:, :16] = torch.tensor(2e-39, device="cuda").bfloat16()
    x[:, 16] = -1e-39
    residual = torch.zeros_like(x)
    weight = torch.zeros(cols, device="cuda").bfloat16()
    global_scale = torch.tensor([7.5], device="cuda")

    out, values, scales = _fused(x, residual, weight, global_scale, gemma=True)

    assert out[:, :16].float().abs().max() < torch.finfo(torch.float32).tiny
    _assert_matches_fp4_quantize(out, values, scales, global_scale)


def test_nvfp4_copy_saturates_block_scales() -> None:
    # A served global scale and blocks far above the row's RMS push block scales past E4M3's 448.
    rows, cols = 129, 5120
    x = _rows(rows, cols, 16, seed=5)
    residual = torch.zeros_like(x)
    weight = (torch.randn(cols, device="cuda") * 0.3 + 1).bfloat16()
    weight[7] = 3e38
    global_scale = torch.tensor([1000.0], device="cuda")

    out, values, scales = _fused(x, residual, weight, global_scale, gemma=False)

    assert out.isinf().any()
    assert (scales[:rows] == 0x7E).any()
    _assert_matches_fp4_quantize(out, values, scales, global_scale)


def test_nvfp4_copy_compiles_once_across_batch_sizes() -> None:
    cols = 5120
    weight = torch.ones(cols, device="cuda").bfloat16()
    global_scale = torch.tensor([7.5], device="cuda")

    def run(rows: int) -> None:
        x = torch.randn(rows, cols, device="cuda").bfloat16()
        _fused(x, torch.randn_like(x), weight, global_scale, gemma=True)

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
    assert nvfp4_copy_supported(5120)


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
    narrow = torch.randn(4, 2560, device="cuda").bfloat16()
    with pytest.raises(ValueError, match="no NVFP4 copy"):
        add_rmsnorm(
            narrow,
            narrow.clone(),
            torch.ones(2560, device="cuda").bfloat16(),
            1e-6,
            x2=None,
            out=torch.empty_like(narrow),
            out_fp8=None,
            fp8_scale=None,
            out_fp4=(values[:, :1280], scales[:, :160]),
            fp4_scale=torch.ones(1, device="cuda"),
            gemma=False,
        )

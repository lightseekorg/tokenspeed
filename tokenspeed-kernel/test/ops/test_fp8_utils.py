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

"""Tests for tokenspeed_kernel.ops.gemm.fp8_utils._per_token_group_quant_8bit_raw.

Covers the platform-aware bit8_max/bit8_min selection added in the CDNA3/CDNA4
split patch:
  - CDNA4 (MI350): fp8e4m3fn.max == 448.0  (float8_e4m3fn, IEEE)
  - CDNA3 (MI300): fp8e4m3fn.max == 224.0  (float8_e4m3fnuz, biased-zero)
  - int8 path: always 127 / -128, independent of platform
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.gemm.fp8_utils import _per_token_group_quant_8bit_raw
from tokenspeed_kernel.platform import current_platform

_platform = current_platform()

pytestmark = pytest.mark.skipif(
    not _platform.is_amd,
    reason="_per_token_group_quant_8bit_raw AMD path requires an AMD GPU",
)


# ---------------------------------------------------------------------------
# bit8_max selection (CPU-level, no kernel launch)
# ---------------------------------------------------------------------------


def test_cdna4_fp8_max_is_448() -> None:
    """On CDNA4 (MI350), platform.fp8e4m3fn.max must be 448.0 (IEEE e4m3fn)."""
    assert _platform.is_cdna4, "This test requires a CDNA4 device"
    assert _platform.fp8e4m3fn.max == 448.0
    assert _platform.fp8e4m3fn.min == -448.0
    assert _platform.fp8e4m3fn.dtype == torch.float8_e4m3fn


def test_platform_fp8e4m3fn_dtype_matches_arch() -> None:
    """fp8e4m3fn dtype/max/min must match the current architecture:
      - CDNA3 (MI300): float8_e4m3fnuz, max=224.0
      - CDNA4 (MI350): float8_e4m3fn,   max=448.0
    On this machine (CDNA4) we assert the CDNA4 values; the CDNA3 branch is
    verified by the platform unit tests in test_selection.py via mock fixtures.
    """
    fp8 = _platform.fp8e4m3fn
    if _platform.is_cdna3:
        assert fp8.dtype == torch.float8_e4m3fnuz
        assert fp8.max == 224.0
        assert fp8.min == -224.0
    else:
        assert fp8.dtype == torch.float8_e4m3fn
        assert fp8.max == 448.0
        assert fp8.min == -448.0


def test_int8_max_is_platform_agnostic(device: str) -> None:
    """int8 path must always yield 127/-128 regardless of AMD/NVIDIA."""
    x = torch.randn(16, 128, device=device, dtype=torch.bfloat16)
    x_q, x_s = _per_token_group_quant_8bit_raw(x, group_size=128, dtype=torch.int8)
    assert x_q.dtype == torch.int8
    assert x_q.abs().max().item() <= 127


# ---------------------------------------------------------------------------
# Kernel correctness on the current AMD device
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("group_size", [128, 256])
def test_quantize_output_shape(device: str, group_size: int) -> None:
    rows, cols = 32, group_size * 4
    x = torch.randn(rows, cols, device=device, dtype=torch.bfloat16)
    x_q, x_s = _per_token_group_quant_8bit_raw(x, group_size=group_size)
    assert x_q.shape == x.shape
    assert x_s.shape == (rows, cols // group_size)
    assert x_q.dtype == _platform.fp8e4m3fn.dtype


@pytest.mark.parametrize("group_size", [128, 256])
def test_quantize_values_within_fp8_range(device: str, group_size: int) -> None:
    """All quantized values must lie within [fp8_min, fp8_max]."""
    torch.manual_seed(0)
    rows, cols = 64, group_size * 2
    x = torch.randn(rows, cols, device=device, dtype=torch.bfloat16) * 200
    x_q, _ = _per_token_group_quant_8bit_raw(x, group_size=group_size)
    x_q_f32 = x_q.to(torch.float32)
    fp8_max = _platform.fp8e4m3fn.max
    assert x_q_f32.abs().max().item() <= fp8_max + 1e-3, (
        f"quantized value exceeds fp8_max={fp8_max}"
    )


def test_scale_is_positive(device: str) -> None:
    """Per-group scales must all be positive."""
    torch.manual_seed(1)
    x = torch.randn(16, 256, device=device, dtype=torch.bfloat16)
    _, x_s = _per_token_group_quant_8bit_raw(x, group_size=128)
    assert (x_s > 0).all(), "all group scales must be positive"


def test_dequantize_close_to_input(device: str) -> None:
    """dequant(quant(x)) should be close to x for well-scaled inputs."""
    torch.manual_seed(2)
    group_size = 128
    x = torch.randn(32, group_size * 4, device=device, dtype=torch.bfloat16) * 10
    x_q, x_s = _per_token_group_quant_8bit_raw(x, group_size=group_size)
    # Repeat each scale across its group to reconstruct x.
    x_s_expanded = x_s.repeat_interleave(group_size, dim=-1)
    x_dequant = x_q.to(torch.float32) * x_s_expanded.to(torch.float32)
    # Relative error should be small (fp8 has 3 mantissa bits → ~12% max relative error).
    rel_err = ((x.float() - x_dequant).abs() / (x.float().abs() + 1e-6)).mean()
    assert rel_err.item() < 0.15, f"mean relative dequant error too large: {rel_err:.4f}"

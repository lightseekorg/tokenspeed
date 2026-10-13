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
from tokenspeed_kernel.ops.gemm import mm
from tokenspeed_kernel.platform import current_platform

_platform = current_platform()
_KERNEL = "flashinfer_cute_dsl_mm_nvfp4"
pytestmark = pytest.mark.skipif(
    not _platform.is_blackwell,
    reason="flashinfer_cute_dsl_mm_nvfp4 is selected on SM100, SM103 and SM107 only",
)

# Qwen3.8 MLP down projection, a small-N and a small-K shape, gate_up, and the
# largest K the kernel takes at a large N (where cuBLASLt starts splitting K first).
_SHAPES = [(5120, 17408), (2048, 7168), (7168, 2048), (34816, 5120), (65536, 18432)]

_Operands = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _wide_range(
    rows: int, k: int, spread: int, generator: torch.Generator
) -> torch.Tensor:
    """Normal values, each 16-wide K block scaled by 2**U[-spread, spread].

    Block scales this far apart make the FP32 sum round, so a kernel that adds
    K in another order than cuBLASLt flips some BF16 outputs; with plain normal
    values every order gives the same bits.
    """
    x = torch.randn(rows, k, device="cuda", generator=generator)
    exponent = torch.randint(
        -spread, spread + 1, (rows, k // 16, 1), device="cuda", generator=generator
    )
    return (x.view(rows, k // 16, 16) * torch.exp2(exponent)).view(rows, k).bfloat16()


def _quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from tokenspeed_kernel.ops.quantization.flashinfer import fp4_quantize

    global_scale = (448 * 6) / x.abs().max().float()
    packed, scales = fp4_quantize(x, global_scale.reshape(1))
    return packed, scales, global_scale


def _operands(m: int, n: int, k: int, seed: int) -> _Operands:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    w, w_scales, w_global = _quantize(_wide_range(n, k, 6, generator))
    x, x_scales, x_global = _quantize(_wide_range(m, k, 12, generator))
    alpha = (1.0 / (x_global * w_global)).float()
    return x, x_scales, w, w_scales, alpha


def _mm(
    operands: _Operands,
    override: str | None,
    out_dtype: torch.dtype = torch.bfloat16,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    x, x_scales, w, w_scales, alpha = operands
    return mm(
        x,
        w.T,
        A_scales=x_scales,
        B_scales=w_scales.T,
        out=out,
        out_dtype=out_dtype,
        alpha=alpha,
        quant="nvfp4",
        override=override,
    )


def _assert_same_bits(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("n, k", _SHAPES)
@pytest.mark.parametrize("m", [1, 6, 48, 128])
def test_matches_cublaslt_bit_for_bit(
    m: int, n: int, k: int, out_dtype: torch.dtype
) -> None:
    operands = _operands(m, n, k, seed=m * 131 + n)

    expected = _mm(operands, "cublaslt_mm_nvfp4", out_dtype)
    actual = _mm(operands, _KERNEL, out_dtype)

    _assert_same_bits(actual, expected)


def test_float4_storage_matches_uint8() -> None:
    x, x_scales, w, w_scales, alpha = _operands(6, 2048, 7168, seed=5)
    float4 = torch.float4_e2m1fn_x2
    as_float4 = (x.view(float4), x_scales, w.view(float4), w_scales, alpha)

    _assert_same_bits(
        _mm(as_float4, _KERNEL), _mm((x, x_scales, w, w_scales, alpha), _KERNEL)
    )


def test_unaligned_buffers_receive_the_product() -> None:
    m, n, k = 6, 2048, 7168
    x, x_scales, w, w_scales, alpha = _operands(m, n, k, seed=7)
    expected = _mm((x, x_scales, w, w_scales, alpha), "cublaslt_mm_nvfp4")
    # One element past an allocation: breaks the 16-byte output and 32-byte operand alignment.
    x_storage = torch.empty(m * x.shape[1] + 1, dtype=x.dtype, device=x.device)
    shifted_x = x_storage[1:].view(x.shape).copy_(x)
    storage = torch.zeros(m * n + 1, dtype=torch.bfloat16, device="cuda")
    out = storage[1:].view(m, n)

    returned = _mm((shifted_x, x_scales, w, w_scales, alpha), _KERNEL, out=out)

    assert returned.data_ptr() == out.data_ptr()
    _assert_same_bits(out, expected)
    assert storage[0] == 0

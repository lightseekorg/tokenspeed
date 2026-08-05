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
from tokenspeed_kernel import (
    quantize_fp8,
    quantize_fp8_with_scale,
    quantize_mxfp4,
    quantize_mxfp8,
    quantize_nvfp4,
)
from tokenspeed_kernel.ops.quantization.triton import fp8_quantize

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_FINFO = torch.finfo(_FP8_DTYPE)


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def _e2m1_values(nibbles: torch.Tensor) -> torch.Tensor:
    magnitude_bits = nibbles & 0x7
    exponent = (magnitude_bits >> 1).to(torch.float32)
    mantissa = (magnitude_bits & 0x1).to(torch.float32)
    normal = (1.0 + 0.5 * mantissa) * torch.exp2(exponent - 1.0)
    subnormal = 0.5 * mantissa
    magnitude = torch.where(exponent == 0, subnormal, normal)
    sign = 1.0 - 2.0 * ((nibbles >> 3) & 0x1).to(torch.float32)
    return magnitude * sign


def _dequantize_mxfp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    out = packed.new_empty(
        (*packed.shape[:-1], packed.shape[-1] * 2),
        dtype=torch.float32,
    )
    out[..., 0::2] = _e2m1_values(packed & 0xF)
    out[..., 1::2] = _e2m1_values(packed >> 4)
    scale_values = torch.pow(2.0, scale.to(torch.int32) - 127).to(torch.float32)
    return out * scale_values.repeat_interleave(32, dim=-1)


@pytest.mark.parametrize("solution", ["triton"])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 2880),
        (8, 2880),
        (33, 2880),
        (4, 4096),
        (2, 1),
        (3, 513),
    ],
)
def test_quantize_fp8_pure_cast_bf16(
    device: str,
    solution: str,
    shape: tuple[int, ...],
    require,
) -> None:
    torch.manual_seed(0)
    dtype = torch.bfloat16
    require("quantization", "fp8", solution, dtype, "x")

    x = torch.randn(shape, device=device, dtype=dtype) * 50
    ref = x.to(_FP8_DTYPE)

    out = quantize_fp8(x, solution=solution)
    torch.cuda.synchronize()

    assert out.shape == ref.shape
    assert out.dtype == _FP8_DTYPE
    assert _bitwise_equal(out, ref)


@pytest.mark.parametrize("solution", ["triton"])
def test_quantize_mxfp4_dynamic_scales(
    device: str,
    solution: str,
    require,
) -> None:
    dtype = torch.bfloat16
    require("quantization", "mxfp4", solution, dtype, "x")

    base = torch.tensor(
        [
            0.0,
            0.5,
            -0.5,
            1.0,
            -1.0,
            1.5,
            -1.5,
            2.0,
            -2.0,
            3.0,
            -3.0,
            4.0,
            -4.0,
            6.0,
            -6.0,
            0.0,
        ],
        device=device,
        dtype=dtype,
    )
    row = torch.cat([base, base, base * 0.25, base * 0.25], dim=0)
    x = torch.stack([row, row], dim=0)

    out, scale = quantize_mxfp4(x, scale_layout="linear", solution=solution)
    torch.cuda.synchronize()

    assert out.shape == (2, 32)
    assert out.dtype == torch.uint8
    assert scale.shape == (2, 2)
    assert scale.dtype == torch.uint8
    torch.testing.assert_close(
        scale.cpu(),
        torch.tensor([[127, 125], [127, 125]], dtype=torch.uint8),
    )

    dequant = _dequantize_mxfp4(out.cpu(), scale.cpu())
    torch.testing.assert_close(dequant, x.cpu().to(torch.float32), rtol=0, atol=0)


@pytest.mark.parametrize("solution", ["triton"])
def test_quantize_fp8_strided_slice(
    device: str,
    solution: str,
    require,
) -> None:
    torch.manual_seed(1)
    dtype = torch.bfloat16
    require("quantization", "fp8", solution, dtype, "x")

    s, h, qk_nope, v_head = 4096, 16, 128, 128
    kv = torch.randn(s, h, qk_nope + v_head, device=device, dtype=dtype) * 50
    v = kv[..., qk_nope:]
    assert not v.is_contiguous()

    ref = v.to(_FP8_DTYPE)

    out = quantize_fp8(v, solution=solution)
    torch.cuda.synchronize()

    assert _bitwise_equal(out, ref)


@pytest.mark.parametrize("solution", ["triton"])
@pytest.mark.parametrize("scale", [2.0, 0.5, 7.5])
def test_quantize_fp8_scale_float(
    device: str,
    solution: str,
    scale: float,
    require,
) -> None:
    torch.manual_seed(2)
    dtype = torch.bfloat16
    require("quantization", "fp8", solution, dtype, "x")

    x = torch.randn(2048, 512, device=device, dtype=dtype) * 100
    inv_scale = 1.0 / scale
    ref = (
        (x.to(torch.float32) * inv_scale)
        .clamp(min=_FP8_FINFO.min, max=_FP8_FINFO.max)
        .to(_FP8_DTYPE)
    )

    out = quantize_fp8(x, scale=scale, solution=solution)
    torch.cuda.synchronize()

    assert _bitwise_equal(out, ref)


@pytest.mark.parametrize("solution", ["triton"])
def test_quantize_fp8_scale_tensor(
    device: str,
    solution: str,
    require,
) -> None:
    torch.manual_seed(3)
    dtype = torch.bfloat16
    require("quantization", "fp8", solution, dtype, "x")

    x = torch.randn(8, 2880, device=device, dtype=dtype) * 100
    scale = torch.tensor([0.125], device=device, dtype=torch.float32)
    inv_scale = (1.0 / scale.to(torch.float32)).reshape(())
    ref = (
        (x.to(torch.float32) * inv_scale)
        .clamp(min=_FP8_FINFO.min, max=_FP8_FINFO.max)
        .to(_FP8_DTYPE)
    )

    out = quantize_fp8(x, scale=scale, solution=solution)
    torch.cuda.synchronize()

    assert _bitwise_equal(out, ref)


@pytest.mark.parametrize(
    "n",
    [
        # gpt-oss-120b: H = 2880 (hidden), I/tp = 2880/2 = 1440 (per-rank
        # ispp). Both are non-power-of-2, so the n-axis must be masked
        # both on load and on store for the W4A8 MoE forward path.
        2880,
        1440,
        # ``M`` not divisible by ``BLOCK_M`` exercises the m-axis tail mask
        # while ``N`` is non-pow2, ruling out a simple "round both up" bug.
        7,
        333,
    ],
)
def test_pure_cast_non_pow2_n(device: str, n: int) -> None:
    torch.manual_seed(0)
    x = torch.randn(33, n, device=device, dtype=torch.bfloat16) * 50
    ref = x.to(torch.float8_e4m3fn)
    out = fp8_quantize(x)
    torch.cuda.synchronize()
    assert out.shape == ref.shape
    assert _bitwise_equal(out, ref)


def test_fp8_quantize_rejects_e4m3fnuz(device: str) -> None:
    x = torch.empty(1, device=device, dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="unsupported fp8 dtype"):
        fp8_quantize(x, fp8_dtype=torch.float8_e4m3fnuz)


@pytest.mark.parametrize(
    "solution,granularity",
    [("trtllm", "tensor"), ("trtllm", "token"), ("triton", "token")],
)
def test_quantize_fp8_with_scale_tensor_and_token(
    device: str,
    solution: str,
    granularity: str,
    require,
) -> None:
    torch.manual_seed(4)
    dtype = torch.bfloat16
    require("quantization", "fp8_with_scale", solution, dtype, "x")

    x = torch.randn(16, 128, device=device, dtype=dtype) * 10
    out, scale = quantize_fp8_with_scale(
        x,
        granularity=granularity,
        solution=solution,
    )
    torch.cuda.synchronize()

    assert out.shape == x.shape
    assert out.dtype == _FP8_DTYPE
    assert scale.dtype == torch.float32
    if granularity == "tensor":
        assert scale.shape == (1,)
    else:
        assert scale.shape == (x.shape[0], 1)


@pytest.mark.parametrize(
    "solution,group_size",
    [("trtllm", 128), ("triton", 128), ("triton", 32)],
)
def test_quantize_fp8_with_scale_token_group(
    device: str,
    solution: str,
    group_size: int,
    require,
) -> None:
    torch.manual_seed(5)
    dtype = torch.bfloat16
    require("quantization", "fp8_with_scale", solution, dtype, "x")

    x = torch.randn(16, 256, device=device, dtype=dtype) * 10
    out, scale = quantize_fp8_with_scale(
        x,
        granularity="token_group",
        group_size=group_size,
        solution=solution,
    )
    torch.cuda.synchronize()

    assert out.shape == x.shape
    assert out.dtype == _FP8_DTYPE
    assert scale.dtype == torch.float32
    expected_num_scales = x.shape[0] * (x.shape[1] // group_size)
    assert scale.numel() == expected_num_scales
    if solution == "triton":
        assert scale.shape == (x.shape[0], x.shape[1] // group_size)


@pytest.mark.parametrize(
    "scale_layout,expected_shape,expected_stride",
    [
        ("row_major", (5, 4), (4, 1)),
        ("column_major", (4, 5), (5, 1)),
        ("column_major_tma", (5, 4), (1, 8)),
    ],
)
def test_quantize_fp8_with_scale_layouts(
    device: str,
    scale_layout: str,
    expected_shape: tuple[int, int],
    expected_stride: tuple[int, int],
) -> None:
    torch.manual_seed(6)
    x = torch.randn(5, 512, device=device, dtype=torch.bfloat16)
    out, scales = quantize_fp8_with_scale(
        x,
        granularity="token_group",
        group_size=128,
        scale_layout=scale_layout,
        solution="triton",
    )
    torch.cuda.synchronize()

    assert out.shape == x.shape
    assert scales.shape == expected_shape
    assert scales.stride() == expected_stride
    logical_scales = scales.t() if scale_layout == "column_major" else scales
    restored = out.float().view(5, 4, 128) * logical_scales.float().unsqueeze(-1)
    assert torch.allclose(restored.flatten(1), x.float(), atol=0.15, rtol=0.05)


def test_quantize_fp8_token_flattens_leading_dimensions(device: str) -> None:
    x = torch.randn(2, 3, 128, device=device, dtype=torch.bfloat16)
    out, scales = quantize_fp8_with_scale(
        x,
        granularity="token",
        scale_layout="row_major",
        solution="triton",
    )
    torch.cuda.synchronize()

    assert out.shape == x.shape
    assert scales.shape == (6, 1)


def test_quantize_fp8_token_group_flattens_leading_dimensions(device: str) -> None:
    x = torch.randn(2, 3, 256, device=device, dtype=torch.bfloat16)
    out, scales = quantize_fp8_with_scale(
        x,
        granularity="token_group",
        group_size=128,
        scale_layout="row_major",
        override="triton",
    )
    torch.cuda.synchronize()

    assert out.shape == x.shape
    assert scales.shape == (6, 2)


def test_quantize_fp8_with_packed_ue8m0_scales(device: str) -> None:
    torch.manual_seed(7)
    x = torch.randn(5, 512, device=device, dtype=torch.bfloat16)
    out, packed = quantize_fp8_with_scale(
        x,
        granularity="token_group",
        group_size=128,
        scale_encoding="packed_ue8m0",
        scale_layout="column_major_tma",
        solution="triton",
    )
    torch.cuda.synchronize()

    assert out.shape == x.shape
    assert packed.shape == (5, 1)
    assert packed.dtype == torch.int32
    assert packed.stride() == (1, 8)
    packed64 = packed.to(torch.int64)
    exponents = torch.stack(
        [((packed64 >> (offset * 8)) & 0xFF) for offset in range(4)], dim=-1
    ).flatten(1)
    scales = torch.exp2(exponents.float() - 127.0)
    restored = out.float().view(5, 4, 128) * scales.unsqueeze(-1)
    assert torch.allclose(restored.flatten(1), x.float(), atol=0.25, rtol=0.08)


@pytest.mark.parametrize("solution", ["flashinfer"])
def test_quantize_mxfp8_shape_and_scale(
    device: str,
    solution: str,
    require,
) -> None:
    torch.manual_seed(6)
    dtype = torch.bfloat16
    require("quantization", "mxfp8", solution, dtype, "x")

    x = torch.randn(17, 2880, device=device, dtype=dtype)
    out, scale = quantize_mxfp8(x, solution=solution)
    torch.cuda.synchronize()

    assert out.shape[:-1] == x.shape[:-1]
    assert out.shape[-1] >= x.shape[-1]
    assert scale.numel() > 0


@pytest.mark.parametrize("solution", ["flashinfer"])
def test_quantize_nvfp4_shape_and_scale(
    device: str,
    solution: str,
    require,
) -> None:
    torch.manual_seed(7)
    dtype = torch.bfloat16
    require("quantization", "nvfp4", solution, dtype, "x")

    x = torch.randn(16, 256, device=device, dtype=dtype)
    out, scale = quantize_nvfp4(
        x,
        scale=torch.tensor([0.125], device=device, dtype=torch.float32),
        solution=solution,
    )
    torch.cuda.synchronize()

    assert out.shape[:-1] == x.shape[:-1]
    assert out.shape[-1] == x.shape[-1] // 2
    assert scale.numel() > 0

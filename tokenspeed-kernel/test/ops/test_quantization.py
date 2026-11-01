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
from tokenspeed_kernel._triton import triton
from tokenspeed_kernel.ops.quantization import (
    quantize_fp8,
    quantize_fp8_with_scale,
    quantize_mxfp4,
    quantize_mxfp8,
    quantize_nvfp4,
)
from tokenspeed_kernel.ops.quantization import triton as quant_triton
from tokenspeed_kernel.ops.quantization.triton import fp8_quantize
from tokenspeed_kernel.platform import Platform
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import NoKernelFoundError
from utils import assert_fp8_same_or_adjacent_at_midpoint, kernel_supported

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_FINFO = torch.finfo(_FP8_DTYPE)


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(
        a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8)
    )


def _fp8_token_group_input(
    groups: int, dtype: torch.dtype, *, random_values: bool = False
) -> torch.Tensor:
    if random_values:
        generator = torch.Generator().manual_seed(2718)
        x = 3 * torch.randn((groups, 128), generator=generator)
    else:
        x = torch.linspace(-400.0, 400.0, 128).repeat(groups, 1)
        x *= torch.linspace(0.7, 1.3, groups)[:, None]
    x = x.to(dtype)
    x[0].zero_()
    if groups > 1:
        tiny = 2.0**-40 if dtype == torch.bfloat16 else torch.finfo(dtype).tiny
        x[1].fill_(tiny)
    if groups > 2:
        midpoint = torch.tensor(1.0625, dtype=dtype)
        lower = torch.nextafter(midpoint, torch.tensor(0.0, dtype=dtype))
        upper = torch.nextafter(midpoint, torch.tensor(float("inf"), dtype=dtype))
        x[2].zero_()
        x[2, :8] = torch.tensor(
            [lower, midpoint, upper, -lower, -midpoint, -upper, 448.0, -448.0],
            dtype=dtype,
        )
    if groups > 3:
        peak, boundary = (
            (5.1875, 0.486328125) if dtype == torch.bfloat16 else (2.84375, 0.736328125)
        )
        x[3].zero_()
        x[3, :4] = torch.tensor([boundary, peak, peak / 2, -peak / 2], dtype=dtype)
    return x


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


def test_fp8_roundtrip_rejects_pdl() -> None:
    with pytest.raises(ValueError, match="does not support enable_pdl"):
        quantize_fp8(
            torch.empty((2, 128), dtype=torch.bfloat16),
            granularity="token_group",
            group_size=128,
            scale_encoding="ue8m0",
            dequantize=True,
            enable_pdl=True,
        )


def test_portable_token_quantizer_rejects_pdl(mi350_platform, monkeypatch) -> None:
    monkeypatch.setattr(Platform, "_instance", mi350_platform)
    monkeypatch.setattr(KernelRegistry.get(), "_selection_cache", {})
    with pytest.raises(ValueError, match="does not support enable_pdl"):
        quantize_fp8(
            torch.ones(1, 32, dtype=torch.bfloat16),
            granularity="token",
            enable_pdl=True,
        )


def test_portable_token_quantizer_forwards_pdl(h100_platform, monkeypatch) -> None:
    monkeypatch.setattr(Platform, "_instance", h100_platform)
    monkeypatch.setattr(KernelRegistry.get(), "_selection_cache", {})
    launches = []

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((grid, kwargs))

            return launch

    monkeypatch.setattr(quant_triton, "_fp8_token_quantize_kernel", Kernel())
    quant_triton.triton_quantize_fp8_per_token_portable(
        torch.ones(128, 32, dtype=torch.bfloat16),
        granularity="token",
        group_size=None,
        scale_encoding="float32",
        enable_pdl=True,
    )
    assert len(launches) == 1
    grid, options = launches[0]
    assert grid == (128,)
    assert options["ENABLE_PDL"] is True
    assert options["launch_pdl"] is True


@pytest.mark.parametrize("layout", ["contiguous", "strided", "higher_rank"])
def test_portable_token_quantizer_preserves_layout(device, layout):
    if not torch.cuda.is_available() or not kernel_supported(
        "triton_quantize_fp8_per_token_portable"
    ):
        pytest.skip("portable minimum-scale E4M3 quantization requires gfx950")
    torch.manual_seed(196)
    x = torch.randn(130, 128, device=device, dtype=torch.bfloat16)
    if layout == "strided":
        x = x[:, ::2]
    elif layout == "higher_rank":
        x = x.reshape(2, 65, 128)
    values, scales = quantize_fp8(
        x,
        granularity="token",
        override="triton_quantize_fp8_per_token_portable",
        required_features=frozenset({"token_fp8_min_scaling"}),
    )
    amax = x.float().abs().amax(-1, keepdim=True)
    expected_scales = (amax.double() / 448.0).float().clamp_min(1.0 / (448 * 512))
    multiplier = (448.0 / amax.double()).float().clamp_max(448 * 512)
    expected = (x.float() * multiplier).clamp(-448, 448).to(torch.float8_e4m3fn)
    assert values.shape == x.shape
    assert scales.shape == (*x.shape[:-1], 1)
    torch.testing.assert_close(
        values.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)


@pytest.mark.parametrize(
    "options,override",
    [
        ({}, "triton_quantize_fp8"),
        ({"granularity": "token"}, "triton_quantize_fp8_with_scale"),
        (
            {
                "granularity": "token_group",
                "group_size": 128,
                "scale_encoding": "ue8m0",
                "dequantize": True,
            },
            "triton_quantize_fp8_roundtrip",
        ),
    ],
)
def test_fp8_required_features_cannot_be_bypassed_by_override(options, override):
    with pytest.raises(NoKernelFoundError, match="token_fp8_min_scaling"):
        quantize_fp8(
            torch.empty(2, 128, dtype=torch.bfloat16),
            override=override,
            required_features=frozenset({"token_fp8_min_scaling"}),
            **options,
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("group_size", [64, 128])
def test_quantize_fp8_ue8m0(
    device: str,
    dtype: torch.dtype,
    group_size: int,
    require,
) -> None:
    require("quantization", "fp8", "triton", dtype, "x")
    torch.manual_seed(41)
    base = torch.randn(3, 2, group_size * 3 + 17, device=device, dtype=dtype)
    x = base[..., : group_size * 3]
    assert not x.is_contiguous()
    blocks = x.float().unflatten(-1, (-1, group_size))
    absmax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scales = torch.exp2(torch.ceil(torch.log2(absmax / 448.0)))
    expected = (
        torch.clamp(blocks / scales, -448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .float()
        .mul(scales)
        .flatten(-2)
        .to(dtype)
    )

    actual, returned_scale = quantize_fp8(
        x,
        granularity="token_group",
        group_size=group_size,
        scale_encoding="ue8m0",
        dequantize=True,
        solution="triton",
    )
    torch.cuda.synchronize()

    assert returned_scale is None
    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_quantize_fp8_cuda_graph_replay(device: str, require) -> None:
    dtype = torch.bfloat16
    require("quantization", "fp8", "triton", dtype, "x")
    x = torch.randn(8, 384, device=device, dtype=dtype)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, returned_scale = quantize_fp8(
            x,
            granularity="token_group",
            group_size=128,
            scale_encoding="ue8m0",
            dequantize=True,
            solution="triton",
        )

    x.copy_(torch.linspace(-9.0, 9.0, x.numel(), device=device).reshape_as(x))
    graph.replay()
    torch.cuda.synchronize()

    assert returned_scale is None
    blocks = x.float().unflatten(-1, (-1, 128))
    absmax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scales = torch.exp2(torch.ceil(torch.log2(absmax / 448.0)))
    expected = (
        torch.clamp(blocks / scales, -448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .float()
        .mul(scales)
        .flatten(-2)
        .to(dtype)
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


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

    out, scale = quantize_fp8(x, solution=solution)
    torch.cuda.synchronize()

    assert scale is None
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

    out, scale = quantize_fp8(v, solution=solution)
    torch.cuda.synchronize()

    assert scale is None
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

    out, returned_scale = quantize_fp8(x, scale=scale, solution=solution)
    torch.cuda.synchronize()

    assert returned_scale.item() == scale
    assert _bitwise_equal(out, ref)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured, captured_scale = quantize_fp8(x, scale=scale, solution=solution)
    graph.replay()
    torch.cuda.synchronize()
    assert captured_scale.item() == scale
    assert _bitwise_equal(captured, ref)


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

    out, returned_scale = quantize_fp8(x, scale=scale, solution=solution)
    torch.cuda.synchronize()

    assert returned_scale is scale
    assert _bitwise_equal(out, ref)


@pytest.mark.parametrize(("m", "n"), [(17, 5120), (2049, 6144), (16385, 2048)])
def test_quantize_fp8_scale_tensor_wide_rows(
    device: str,
    m: int,
    n: int,
    require,
) -> None:
    # Wide rows take one to eight rows per program, so odd row counts leave a tail.
    torch.manual_seed(4)
    dtype = torch.bfloat16
    require("quantization", "fp8", "triton", dtype, "x")

    x = torch.randn(m, n, device=device, dtype=dtype) * 100
    scale = torch.tensor([0.125], device=device, dtype=torch.float32)
    ref = (
        (x.to(torch.float32) * (1.0 / scale).reshape(()))
        .clamp(min=_FP8_FINFO.min, max=_FP8_FINFO.max)
        .to(_FP8_DTYPE)
    )

    out, _ = quantize_fp8(x, scale=scale, solution="triton")
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


@pytest.mark.parametrize("solution", ["trtllm"])
@pytest.mark.parametrize("granularity", ["tensor", "token"])
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
    "solution,group_size,dtype,shape",
    [
        ("trtllm", 128, torch.bfloat16, (16, 256)),
        ("triton", 32, torch.bfloat16, (16, 256)),
        ("triton", 128, torch.bfloat16, (16, 256)),
        ("triton", 128, torch.bfloat16, (511, 2048)),
        ("triton", 128, torch.bfloat16, (512, 2048)),
        ("triton", 128, torch.bfloat16, (8193, 128)),
        ("triton", 128, torch.bfloat16, (8192, 2048)),
        ("triton", 128, torch.bfloat16, (8192, 4096)),
        ("triton", 128, torch.float16, (512, 2048)),
    ],
)
def test_quantize_fp8_with_scale_token_group(
    device: str,
    solution: str,
    group_size: int,
    dtype: torch.dtype,
    shape: tuple[int, int],
    require,
) -> None:
    torch.manual_seed(5)
    require("quantization", "fp8_with_scale", solution, dtype, "x")

    if solution == "triton" and group_size == 128:
        groups = shape[0] * shape[1] // group_size
        x = (
            _fp8_token_group_input(groups, dtype, random_values=shape[0] == 8192)
            .reshape(shape)
            .to(device)
        )
    else:
        x = torch.randn(shape, device=device, dtype=dtype) * 10
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
    if solution != "triton" or group_size != 128:
        return

    # Check the public numeric contract around the grouped-launch crossover,
    # including zero, tiny, midpoint, tail, and large random input groups.
    expected_scale = (
        x.float()
        .reshape(shape[0], shape[1] // 128, 128)
        .abs()
        .amax(dim=-1)
        .clamp_min(1.0e-10)
        / _FP8_FINFO.max
    )
    torch.testing.assert_close(scale, expected_scale, rtol=2**-21, atol=0)
    assert torch.isfinite(scale).all()
    assert torch.isfinite(out.float()).all()
    assert (out.float().abs() <= _FP8_FINFO.max).all()
    assert _bitwise_equal(
        out.reshape(-1, 128)[0], torch.zeros(128, device=device).to(_FP8_DTYPE)
    )
    assert _bitwise_equal(out.reshape(-1, 128)[2], x.reshape(-1, 128)[2].to(_FP8_DTYPE))
    reference_out = (
        (x.float() / scale.repeat_interleave(128, dim=-1))
        .clamp(_FP8_FINFO.min, _FP8_FINFO.max)
        .to(_FP8_DTYPE)
    )
    assert_fp8_same_or_adjacent_at_midpoint(x, out, reference_out, scale, 128)
    if shape[0] == 8192:
        reconstructed = out.float() * scale.repeat_interleave(128, dim=-1)
        torch.testing.assert_close(reconstructed, x.float(), rtol=0.08, atol=0.04)


# 16 rows take the Triton per-token quantizer on NVIDIA SM100+, 65 rows the
# portable one (solution="triton") or the vendor one.
@pytest.mark.parametrize("rows", [16, 65])
@pytest.mark.parametrize("solution", [None, "trtllm", "triton"])
def test_quantize_fp8_dynamic_token(
    device: str,
    solution: str | None,
    rows: int,
    require,
) -> None:
    torch.manual_seed(4)
    dtype = torch.bfloat16
    require("quantization", "fp8_with_scale", solution, dtype, "x")

    x = torch.randn(rows, 128, device=device, dtype=dtype) * 10
    out, scale = quantize_fp8(
        x,
        granularity="token",
        solution=solution,
    )
    torch.cuda.synchronize()

    assert out.shape == x.shape
    assert out.dtype == _FP8_DTYPE
    assert scale.dtype == torch.float32
    assert scale.shape == (x.shape[0], 1)
    expected_scales = x.float().abs().amax(-1, keepdim=True) / _FP8_FINFO.max
    # TRT-LLM stores its native scales in the input dtype before the adapter
    # promotes them to FP32. Either representation is valid for auto dispatch.
    rounded_scales = expected_scales.to(dtype).float()
    if solution == "trtllm" or (
        solution is None and torch.equal(scale, rounded_scales)
    ):
        expected_scales = rounded_scales
    torch.testing.assert_close(scale, expected_scales)
    reconstructed = out.float() * scale
    assert torch.norm(reconstructed - x.float()) / torch.norm(x.float()) < 0.04


@pytest.mark.parametrize(
    "solution,group_size",
    [(None, 128), (None, 32), ("trtllm", 128), ("triton", 128), ("triton", 32)],
)
@pytest.mark.parametrize("columns", [256, 5120])
@pytest.mark.parametrize("rows", [1, 13, 16, 17])
def test_quantize_fp8_dynamic_token_group(
    device: str,
    columns: int,
    rows: int,
    solution: str | None,
    group_size: int,
    require,
) -> None:
    torch.manual_seed(5)
    dtype = torch.bfloat16
    require("quantization", "fp8_with_scale", solution, dtype, "x")

    x = torch.randn(rows, columns, device=device, dtype=dtype) * 10
    out, scale = quantize_fp8(
        x,
        granularity="token_group",
        group_size=group_size,
        solution=solution,
    )
    torch.cuda.synchronize()

    assert out.shape == x.shape
    assert out.dtype == _FP8_DTYPE
    assert scale.dtype == torch.float32
    assert scale.shape == (x.shape[0], x.shape[1] // group_size)
    assert scale.is_contiguous()
    expected_scales = x.float().unflatten(-1, (-1, group_size)).abs().amax(-1) / 448
    torch.testing.assert_close(scale, expected_scales)
    reconstructed = out.float() * scale.repeat_interleave(group_size, dim=-1)
    assert torch.norm(reconstructed - x.float()) / torch.norm(x.float()) < 0.04


# 7168 is Kimi K3's hidden size, 128 the DSA index head dimension. Per token,
# 384 pads each group to a 512-wide block, and 33 and 129 leave most of a 64-
# and a 256-wide block masked. The shapes take quantize_fp8 through each of
# its launch tiles. An offset input is contiguous but starts one element into
# its storage; a 3-D input splits each row into its groups.
@pytest.mark.parametrize("layout", ["2d", "offset", "3d"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "granularity,group_size,columns",
    [
        ("token_group", 128, 128),
        ("token_group", 128, 384),
        ("token_group", 128, 7168),
        ("token_group", 32, 128),
        ("token_group", 32, 384),
        ("token_group", 32, 7168),
        ("token", None, 33),
        ("token", None, 128),
        ("token", None, 129),
        ("token", None, 384),
        ("token", None, 7168),
    ],
)
@pytest.mark.parametrize("rows", [0, 1, 7, 128, 4095, 4096])
def test_triton_fp8_token_group_tiles_match_one_group_per_program(
    device: str,
    rows: int,
    columns: int,
    granularity: str,
    group_size: int | None,
    dtype: torch.dtype,
    layout: str,
    require,
) -> None:
    require("quantization", "fp8_with_scale", "triton", dtype, "x")
    torch.manual_seed(6)
    width = columns if group_size is None else group_size
    groups = torch.randn(rows * columns // width, width, device=device) * 10
    groups[3::5] = 0
    groups[1::7] *= 1e-12
    groups[2::11, :2] = (
        torch.tensor([1.0, -1.0], device=device) * torch.finfo(dtype).max
    )
    # A NaN, an all-NaN group and an inf: the tiles below reduce the amax
    # across lanes with different instructions.
    groups[4::13, 5] = float("nan")
    groups[6::17] = float("nan")
    groups[8::19, 3] = float("inf")
    x = groups.to(dtype).flatten()
    if layout == "offset":
        x = torch.cat([x.new_zeros(1), x])[1:]
    if layout == "3d":
        x = x.view(rows, columns // width, width)
    else:
        x = x.view(rows, columns)

    codes, scales = quantize_fp8(
        x,
        granularity=granularity,
        group_size=group_size,
        override="triton_quantize_fp8_with_scale",
    )

    # Tiling invariance: whichever tile quantize_fp8 took, the same kernel
    # launched with each tile -- one group per program as before tiling, 1024
    # and 8192 elements -- gives the same bits. The outputs are the front of
    # sentinel-filled buffers one tile longer, so a partial last tile that
    # writes past the last group shows in the slack.
    block = triton.next_power_of_2(width)
    n = groups.shape[0]
    # Compare tiles under the arithmetic selected by the public call. The
    # gfx950 16-group specialization uses reciprocal normalization.
    use_reciprocal = quant_triton._fp8_token_group_groups_per_cta(x, width, n) == 16
    for groups_per_program in {1, max(1024 // block, 1), max(8192 // block, 1)}:
        code_buffer = torch.full(
            (n + groups_per_program, width), 0x5A, dtype=torch.uint8, device=device
        )
        scale_buffer = torch.full((n + groups_per_program,), -1.0, device=device)
        expected_codes = code_buffer[:n].view(_FP8_DTYPE).view_as(codes)
        expected_scales = scale_buffer[:n].view_as(scales)
        quant_triton._fp8_token_group_quantize_kernel[
            (triton.cdiv(n, groups_per_program),)
        ](
            x,
            expected_codes,
            expected_scales,
            n,
            width,
            1e-10,
            bit8_min=_FP8_FINFO.min,
            bit8_max=_FP8_FINFO.max,
            BLOCK=block,
            GROUPS_PER_PROGRAM=groups_per_program,
            USE_RECIPROCAL=use_reciprocal,
            num_warps=min(max(groups_per_program * block // 256, 1), 8),
            num_stages=1,
        )
        assert _bitwise_equal(codes, expected_codes)
        assert _bitwise_equal(scales, expected_scales)
        assert (code_buffer[n:] == 0x5A).all()
        assert (scale_buffer[n:] == -1.0).all()
    # Against PyTorch: the amax ignoring NaN, floored at 1e-10, over 448, then
    # the clamped quotient of each finite value. Triton's division is within
    # 2 ulp of PyTorch's correctly rounded one, so a code may be one FP8 step
    # off where the quotient is that close to a rounding boundary.
    values = x.float().view(-1, width)
    amax = values.abs().nan_to_num(nan=0.0, posinf=float("inf")).amax(-1)
    torch.testing.assert_close(
        scales.flatten(), amax.clamp_min(1e-10) / _FP8_FINFO.max, rtol=2**-21, atol=0
    )
    reference_codes = (
        (values / scales.view(-1, 1))
        .clamp(_FP8_FINFO.min, _FP8_FINFO.max)
        .to(_FP8_DTYPE)
    )
    # Same-sign E4M3 bit patterns are ordered by magnitude, NaN right after the
    # largest finite value, so a finite input must also not give a NaN code.
    finite = values.isfinite()
    assert not codes.view(-1, width).float()[finite].isnan().any()
    reference_bits = reference_codes.view(torch.uint8).int()
    steps = codes.view(torch.uint8).view(-1, width).int() - reference_bits
    assert (steps[finite].abs() <= 1).all()


def _quantize_fp8_native_scales(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return quantize_fp8(
        x,
        granularity="token_group",
        group_size=128,
        scale_encoding="float32",
        override="trtllm_quantize_fp8_mn_major",
    )


@pytest.mark.parametrize("rows", [1, 13, 32, 128])
@pytest.mark.parametrize("columns", [256, 4096])
def test_native_fp8_scales_match_canonical(
    device: str, rows: int, columns: int, require
) -> None:
    require("quantization", "fp8_with_scale", "trtllm", torch.bfloat16, "x")
    torch.manual_seed(7)
    x = torch.randn(rows, columns, device=device, dtype=torch.bfloat16)
    q, scales = _quantize_fp8_native_scales(x)
    expected_q, expected_scales = quantize_fp8(
        x, granularity="token_group", group_size=128, solution="trtllm"
    )
    torch.testing.assert_close(q.float(), expected_q.float(), rtol=0, atol=0)
    torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)
    # Preserve the producer's stride instead of requiring padding on every ABI.
    _, native_scales = torch.ops.trtllm.fp8_quantize_1x128(x, False)
    if native_scales.ndim == 1:
        expected_stride = (1, (rows + 3) // 4 * 4)
    else:
        expected_stride = native_scales.stride()[::-1]
    assert scales.stride() == expected_stride
    assert scales.stride(0) == 1
    assert scales.stride(1) >= rows
    assert expected_scales.is_contiguous()
    reconstructed = q.float() * scales.repeat_interleave(128, -1)
    assert (reconstructed - x.float()).norm() / x.float().norm() < 0.04

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_q, captured_scales = _quantize_fp8_native_scales(x)
    x.mul_(0.5)
    graph.replay()
    expected_q, expected_scales = quantize_fp8(
        x, granularity="token_group", group_size=128, solution="trtllm"
    )
    torch.testing.assert_close(captured_q.float(), expected_q.float(), rtol=0, atol=0)
    torch.testing.assert_close(captured_scales, expected_scales, rtol=0, atol=0)
    assert captured_scales.stride() == expected_stride


def test_native_fp8_quantization_rejects_pdl(device: str, require) -> None:
    require("quantization", "fp8_with_scale", "trtllm", torch.bfloat16, "x")
    x = torch.zeros(1, 256, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="PDL"):
        quantize_fp8(
            x,
            granularity="token_group",
            group_size=128,
            override="trtllm_quantize_fp8_mn_major",
            enable_pdl=True,
        )


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

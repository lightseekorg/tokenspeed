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
from tokenspeed_kernel.ops.gemm import _online_quantize_mxfp8
from tokenspeed_kernel.ops.gemm.fp8_utils import (
    _per_token_group_quant_8bit_raw,
    per_token_group_quant_fp8,
)
from tokenspeed_kernel.ops.quantization.triton import fp8_quantize
from tokenspeed_kernel.ops.quantization.trtllm import (
    supports_trtllm_fp8_packed_ue8m0,
    trtllm_fp8_packed_ue8m0,
)
from tokenspeed_kernel.platform import ArchVersion, current_platform

FP8_E4M3_FNUZ_MAX = 240.0
_PLATFORM = current_platform()
_IS_SM100 = _PLATFORM.is_nvidia and _PLATFORM.arch_version == ArchVersion(10, 0)


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
    fp8 = current_platform().fp8e4m3fn
    ref = x.to(fp8.dtype)

    out = quantize_fp8(x, solution=solution)
    torch.cuda.synchronize()

    assert out.shape == ref.shape
    assert out.dtype == ref.dtype
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

    fp8 = current_platform().fp8e4m3fn
    ref = v.to(fp8.dtype)

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
    fp8 = current_platform().fp8e4m3fn
    inv_scale = 1.0 / scale
    ref = (
        (x.to(torch.float32) * inv_scale).clamp(min=fp8.min, max=fp8.max).to(fp8.dtype)
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
    fp8 = current_platform().fp8e4m3fn
    inv_scale = (1.0 / scale.to(torch.float32)).reshape(())
    ref = (
        (x.to(torch.float32) * inv_scale).clamp(min=fp8.min, max=fp8.max).to(fp8.dtype)
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


@pytest.mark.skipif(
    not current_platform().is_cdna3,
    reason="float8_e4m3fnuz (tl.float8e4b8) is only supported on AMD CDNA3",
)
def test_pure_cast_e4m3fnuz(device: str) -> None:
    """CDNA3-specific fp8 dtype (bias=8). The Triton cast must saturate to
    ``±240`` to match ``x.to(torch.float8_e4m3fnuz)``."""
    torch.manual_seed(0)
    x = torch.randn(2048, 512, device=device, dtype=torch.bfloat16) * 50
    ref = x.to(torch.float8_e4m3fnuz)
    out = fp8_quantize(x, fp8_dtype=torch.float8_e4m3fnuz)
    torch.cuda.synchronize()
    assert out.dtype == torch.float8_e4m3fnuz
    assert _bitwise_equal(out, ref)


@pytest.mark.skipif(
    not current_platform().is_cdna3,
    reason="float8_e4m3fnuz (tl.float8e4b8) is only supported on AMD CDNA3",
)
@pytest.mark.parametrize("scale", [2.0, 0.5, 7.5])
def test_scaled_cast_e4m3fnuz_matches_reference(device: str, scale: float) -> None:
    torch.manual_seed(0)
    x = torch.randn(2048, 512, device=device, dtype=torch.bfloat16) * 100
    inv_scale = 1.0 / scale
    ref = (
        (x.to(torch.float32) * inv_scale)
        .clamp(-FP8_E4M3_FNUZ_MAX, FP8_E4M3_FNUZ_MAX)
        .to(torch.float8_e4m3fnuz)
    )
    out = fp8_quantize(x, scale=scale, fp8_dtype=torch.float8_e4m3fnuz)
    torch.cuda.synchronize()
    assert _bitwise_equal(out, ref)


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
    fp8 = current_platform().fp8e4m3fn

    out, scale = quantize_fp8_with_scale(
        x,
        granularity=granularity,
        solution=solution,
    )
    torch.cuda.synchronize()

    assert out.shape == x.shape
    assert out.dtype == fp8.dtype
    assert scale.dtype == torch.float32
    if granularity == "tensor":
        assert scale.shape == (1,)
    else:
        assert scale.shape == (x.shape[0], 1)


@pytest.mark.parametrize("solution", ["trtllm"])
def test_quantize_fp8_with_scale_token_group(
    device: str,
    solution: str,
    require,
) -> None:
    torch.manual_seed(5)
    dtype = torch.bfloat16
    require("quantization", "fp8_with_scale", solution, dtype, "x")

    x = torch.randn(16, 256, device=device, dtype=dtype) * 10
    fp8 = current_platform().fp8e4m3fn

    out, scale = quantize_fp8_with_scale(
        x,
        granularity="token_group",
        group_size=128,
        solution=solution,
    )
    torch.cuda.synchronize()

    assert out.shape == x.shape
    assert out.dtype == fp8.dtype
    assert scale.dtype == torch.float32
    assert scale.numel() > 0


@pytest.mark.skipif(
    not _IS_SM100,
    reason="packed UE8M0 TRT-LLM quantization requires SM100",
)
@pytest.mark.parametrize(
    "shape",
    [(1, 128), (2, 2048), (4, 7168), (8, 28672), (17, 512)],
)
def test_quantize_fp8_with_packed_ue8m0(
    device: str,
    shape: tuple[int, int],
    require,
) -> None:
    dtype = torch.bfloat16
    require("quantization", "fp8_with_scale", "trtllm", dtype, "x")
    torch.manual_seed(8)
    x = torch.randn(shape, device=device, dtype=dtype)

    out, scale = quantize_fp8_with_scale(
        x,
        granularity="token_group",
        group_size=128,
        scale_encoding="packed_ue8m0",
        solution="trtllm",
    )
    ref_out, ref_scale = _per_token_group_quant_8bit_raw(
        x,
        128,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )
    torch.cuda.synchronize()

    assert _bitwise_equal(out, ref_out)
    assert scale.dtype == torch.int32
    assert scale.shape == ref_scale.shape
    assert scale.stride() == ref_scale.stride()
    assert torch.equal(scale, ref_scale)


@pytest.mark.skipif(
    not _IS_SM100,
    reason="packed UE8M0 TRT-LLM quantization requires SM100",
)
@pytest.mark.parametrize(
    ("value", "expected_scale_byte"),
    [(0.0, 94), (1e-10, 94), (4.5e-8, 94), (6e-8, 95)],
)
def test_packed_ue8m0_scale_floor_matches_existing_contract(
    device: str,
    value: float,
    expected_scale_byte: int,
) -> None:
    x = torch.full((4, 128), value, device=device, dtype=torch.bfloat16)

    ref_out, ref_scale = _per_token_group_quant_8bit_raw(
        x,
        128,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )
    out, scale = trtllm_fp8_packed_ue8m0(x)
    torch.cuda.synchronize()

    assert _bitwise_equal(out, ref_out)
    assert scale.stride() == ref_scale.stride()
    assert torch.equal(scale, ref_scale)
    physical = torch.as_strided(scale, size=(1, 4), stride=(4, 1))
    scale_bytes = physical.view(torch.uint8).reshape(4, 4)
    assert torch.equal(
        scale_bytes[:, 0],
        torch.full((4,), expected_scale_byte, device=device, dtype=torch.uint8),
    )


@pytest.mark.skipif(
    not _IS_SM100,
    reason="packed UE8M0 TRT-LLM quantization requires SM100",
)
@pytest.mark.parametrize("exponent", [-16, -1, 0, 8, 16])
def test_packed_ue8m0_rounding_boundaries(
    device: str,
    exponent: int,
) -> None:
    boundary = torch.tensor(
        448.0 * 2.0**exponent,
        device=device,
        dtype=torch.bfloat16,
    )
    above = torch.nextafter(
        boundary,
        torch.tensor(float("inf"), device=device, dtype=torch.bfloat16),
    )
    x = torch.stack((boundary, above)).unsqueeze(1).expand(2, 128).contiguous()

    out, scale = trtllm_fp8_packed_ue8m0(x)
    ref_out, ref_scale = _per_token_group_quant_8bit_raw(
        x,
        128,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )
    torch.cuda.synchronize()

    assert _bitwise_equal(out, ref_out)
    assert torch.equal(scale, ref_scale)
    physical = torch.as_strided(scale, size=(1, 4), stride=(4, 1))
    scale_bytes = physical.view(torch.uint8).reshape(4, 4)
    assert scale_bytes[0, 0].item() == exponent + 127
    assert scale_bytes[1, 0].item() == exponent + 128


@pytest.mark.skipif(
    not _IS_SM100,
    reason="packed UE8M0 TRT-LLM quantization requires SM100",
)
def test_packed_ue8m0_padding_is_initialized(device: str) -> None:
    rows, columns = 3, 128
    x = torch.zeros((rows, columns), device=device, dtype=torch.bfloat16)

    out, scale = trtllm_fp8_packed_ue8m0(x)
    torch.cuda.synchronize()

    aligned_rows = 4
    physical = torch.as_strided(
        scale,
        size=(1, aligned_rows),
        stride=(aligned_rows, 1),
    )
    valid_scale_bytes = physical[0, :rows].view(torch.uint8).reshape(rows, 4)
    assert torch.equal(
        valid_scale_bytes[:, 0],
        torch.full((rows,), 94, device=device, dtype=torch.uint8),
    )
    assert torch.count_nonzero(valid_scale_bytes[:, 1:]) == 0
    assert physical[0, rows:].item() == 0
    assert out.untyped_storage().nbytes() == aligned_rows * columns


@pytest.mark.skipif(
    not _IS_SM100,
    reason="packed UE8M0 TRT-LLM quantization requires SM100",
)
@pytest.mark.parametrize("enable_pdl", [False, True])
def test_online_quantization_selects_packed_ue8m0_kernel(
    device: str,
    monkeypatch: pytest.MonkeyPatch,
    enable_pdl: bool,
) -> None:
    x = torch.randn((4, 512), device=device, dtype=torch.bfloat16)
    expected = (
        torch.empty_like(x, dtype=torch.float8_e4m3fn),
        torch.empty((4, 1), device=device, dtype=torch.int32),
    )
    calls = []

    def fake_packed_quant(arg: torch.Tensor, *, enable_pdl: bool):
        calls.append((arg, enable_pdl))
        return expected

    monkeypatch.setattr(
        "tokenspeed_kernel.ops.gemm.fp8_utils._trtllm_fp8_packed_ue8m0",
        fake_packed_quant,
    )
    actual = per_token_group_quant_fp8(
        x,
        128,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
        enable_pdl=enable_pdl,
    )

    assert calls == [(x, enable_pdl)]
    assert actual is expected


@pytest.mark.skipif(
    not _IS_SM100,
    reason="packed UE8M0 TRT-LLM quantization requires SM100",
)
def test_online_quantization_misaligned_input_falls_back(
    device: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows, columns = 4, 512
    storage = torch.randn(
        rows * columns + 1,
        device=device,
        dtype=torch.bfloat16,
    )
    x = storage[1:].view(rows, columns)
    expected = (
        torch.empty_like(x, dtype=torch.float8_e4m3fn),
        torch.empty((rows, 1), device=device, dtype=torch.int32),
    )
    fallback_calls = []

    assert x.is_contiguous()
    assert x.data_ptr() % 32 != 0
    assert not supports_trtllm_fp8_packed_ue8m0(x)

    def fail_fast_path(*args, **kwargs):
        raise AssertionError("misaligned input must not use the fast path")

    def fake_fallback(*args, **kwargs):
        fallback_calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(
        "tokenspeed_kernel.ops.gemm.fp8_utils._trtllm_fp8_packed_ue8m0",
        fail_fast_path,
    )
    monkeypatch.setattr(
        "tokenspeed_kernel.ops.gemm.fp8_utils._per_token_group_quant_8bit_raw",
        fake_fallback,
    )
    actual = per_token_group_quant_fp8(
        x,
        128,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )

    assert len(fallback_calls) == 1
    assert actual is expected


@pytest.mark.skipif(
    not _IS_SM100,
    reason="packed UE8M0 TRT-LLM quantization requires SM100",
)
def test_online_quantization_unavailable_kernel_falls_back(
    device: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = torch.randn((4, 512), device=device, dtype=torch.bfloat16)
    expected = (
        torch.empty_like(x, dtype=torch.float8_e4m3fn),
        torch.empty((4, 1), device=device, dtype=torch.int32),
    )

    monkeypatch.setattr(
        "tokenspeed_kernel.ops.gemm.fp8_utils._supports_trtllm_fp8_packed_ue8m0",
        lambda _x: False,
    )
    monkeypatch.setattr(
        "tokenspeed_kernel.ops.gemm.fp8_utils._trtllm_fp8_packed_ue8m0",
        lambda *args, **kwargs: pytest.fail("unavailable kernel was called"),
    )
    monkeypatch.setattr(
        "tokenspeed_kernel.ops.gemm.fp8_utils._per_token_group_quant_8bit_raw",
        lambda *args, **kwargs: expected,
    )

    actual = per_token_group_quant_fp8(
        x,
        128,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )
    assert actual is expected


@pytest.mark.skipif(
    not _IS_SM100,
    reason="packed UE8M0 TRT-LLM quantization requires SM100",
)
@pytest.mark.parametrize("enable_pdl", [False, True])
def test_online_mxfp8_forwards_pdl_to_quantizer(
    device: str,
    monkeypatch: pytest.MonkeyPatch,
    enable_pdl: bool,
) -> None:
    x = torch.randn((4, 512), device=device, dtype=torch.bfloat16)
    expected = (
        torch.empty_like(x, dtype=torch.float8_e4m3fn),
        torch.empty((4, 1), device=device, dtype=torch.int32),
    )
    calls = []

    def fake_quant(arg, group_size, **kwargs):
        calls.append((arg, group_size, kwargs))
        return expected

    monkeypatch.setattr(
        "tokenspeed_kernel.ops.gemm.fp8_utils.per_token_group_quant_fp8",
        fake_quant,
    )
    actual = _online_quantize_mxfp8(
        x,
        [128, 128],
        "deep_gemm_mm_fp8_blockscale",
        enable_pdl=enable_pdl,
    )

    assert actual is expected
    assert calls[0][0] is x
    assert calls[0][1] == 128
    assert calls[0][2]["enable_pdl"] is enable_pdl


@pytest.mark.skipif(
    not _IS_SM100,
    reason="packed UE8M0 TRT-LLM quantization requires SM100",
)
@pytest.mark.parametrize("enable_pdl", [False, True])
def test_packed_ue8m0_non_default_stream(
    device: str,
    enable_pdl: bool,
) -> None:
    torch.manual_seed(10)
    x = torch.randn((8, 2048), device=device, dtype=torch.bfloat16)
    producer_stream = torch.cuda.current_stream()
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        stream.wait_stream(producer_stream)
        out, scale = trtllm_fp8_packed_ue8m0(x, enable_pdl=enable_pdl)
    stream.synchronize()
    ref_out, ref_scale = _per_token_group_quant_8bit_raw(
        x,
        128,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )
    torch.cuda.synchronize()

    assert _bitwise_equal(out, ref_out)
    assert torch.equal(scale, ref_scale)


@pytest.mark.skipif(
    not _IS_SM100,
    reason="packed UE8M0 TRT-LLM quantization requires SM100",
)
def test_online_packed_ue8m0_quantization_cuda_graph(device: str) -> None:
    torch.manual_seed(9)
    x = torch.randn((16, 7168), device=device, dtype=torch.bfloat16)

    def run():
        return per_token_group_quant_fp8(
            x,
            128,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
            enable_pdl=True,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, scale = run()
    for _ in range(100):
        graph.replay()
    torch.cuda.synchronize()

    replacement = torch.randn_like(x)
    x.copy_(replacement)
    out.view(torch.uint8).fill_(0xFF)
    scale.zero_()
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()

    ref_out, ref_scale = _per_token_group_quant_8bit_raw(
        replacement,
        128,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )
    torch.cuda.synchronize()
    assert _bitwise_equal(out, ref_out)
    assert scale.stride() == ref_scale.stride()
    assert torch.equal(scale, ref_scale)


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

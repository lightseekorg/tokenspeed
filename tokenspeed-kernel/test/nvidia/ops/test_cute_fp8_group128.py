# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Numerical, storage and CUDA Graph contracts for the CuTe quantizer."""

import pytest
import torch
from tokenspeed_kernel.ops.quantization import quantize_fp8

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="requires a Blackwell GPU",
)


def quantize(x, pdl):
    return quantize_fp8(
        x,
        granularity="token_group",
        group_size=128,
        scale_encoding="packed_ue8m0",
        enable_pdl=pdl,
        override="cute_dsl_quantize_fp8_group128_packed",
    )


def reference(x):
    k = x.shape[-1]
    groups = x.float().reshape(-1, k // 128, 128)
    raw = groups.abs().amax(-1).clamp_min(1e-10) * (1.0 / 448)
    bits = raw.view(torch.int32)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0)
    scales = (exponent << 23).view(torch.float32)
    q = (groups / scales[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.reshape(x.shape), exponent


def check(x, q, s):
    k = x.shape[-1]
    m = x.numel() // k
    expected, exponents = reference(x)
    torch.testing.assert_close(
        q.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
    )
    groups = torch.arange(k // 128, device=x.device, dtype=torch.int32)
    actual = (s[:, groups // 4] >> ((groups % 4) * 8)) & 255
    torch.testing.assert_close(actual, exponents, rtol=0, atol=0)
    padded = (m + 3) // 4 * 4
    assert s.shape == (m, (k + 511) // 512)
    assert s.stride() == (1, padded)
    storage = s.as_strided(((k + 511) // 512, padded), (padded, 1))
    assert (storage[:, m:] == 0).all()
    if (k // 128) % 4:
        assert (s[:, -1] >> ((k // 128 % 4) * 8) == 0).all()


@pytest.mark.parametrize(
    "m,k",
    [
        (1, 128),
        (3, 384),
        (5, 640),
        (4, 2560),
        (32, 1536),
        (128, 2560),
        (4096, 1536),
        (16384, 2560),
    ],
)
@pytest.mark.parametrize("pdl", [False, True])
def test_quantization_and_graph_refresh(m, k, pdl):
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    quantize(x, pdl)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q, s = quantize(x, pdl)
    for fill in (None, 0.0, 1e-9, 0.4375):
        if fill is None:
            x.normal_().mul_(0.7)
        else:
            x.fill_(fill)
        graph.replay()
        check(x, q, s)


def test_finite_bf16_patterns():
    x = (
        torch.arange(65536, device="cuda", dtype=torch.int32)
        .to(torch.int16)
        .view(torch.bfloat16)
        .reshape(-1, 128)
    )
    x[~torch.isfinite(x)] = 0
    check(x, *quantize(x, True))


def test_variable_rows_do_not_compile():
    from tokenspeed_kernel.thirdparty.cute_dsl.fp8_group128 import compile_kernel

    for m in (1, 4, 1024):
        quantize(torch.zeros((m, 2560), device="cuda", dtype=torch.bfloat16), True)
    misses = compile_kernel.cache_info().misses
    for m in (2, 3, 17, 33, 127, 257, 513, 4097):
        x = torch.randn((m, 2560), device="cuda", dtype=torch.bfloat16)
        check(x, *quantize(x, True))
    assert compile_kernel.cache_info().misses == misses


@pytest.mark.parametrize("shape", [(0, 2560), (2, 3, 2560), (2560,)])
def test_leading_dimensions_and_empty(shape):
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    q, s = quantize(x, False)
    assert q.shape == x.shape
    if x.numel():
        check(x, q, s)
    else:
        assert s.shape == (0, 5)


@pytest.mark.parametrize(
    "dtype,encoding,expected",
    [
        (torch.bfloat16, "packed_ue8m0", "cute_dsl_quantize_fp8_group128_packed"),
        (torch.float16, "packed_ue8m0", "triton_quantize_fp8_group128_packed"),
        (torch.bfloat16, "float32", "triton_quantize_fp8_group128"),
    ],
)
def test_default_quantizer_selection(dtype, encoding, expected):
    from tokenspeed_kernel.ops.quantization import _select_fp8_dynamic

    selected = _select_fp8_dynamic(
        dtype,
        128,
        "token_group_128",
        encoding,
        solution=None,
        override=None,
    )
    assert selected.name == expected


def test_unaligned_contiguous_slice():
    storage = torch.randn(4 * 2560 + 1, device="cuda", dtype=torch.bfloat16)
    x = storage[1:].view(4, 2560)
    assert x.is_contiguous() and x.data_ptr() % 32
    check(x, *quantize(x, False))


@pytest.mark.parametrize("pdl", [False, True])
def test_each_finite_bf16_value_as_group_maximum(pdl):
    values = (
        torch.arange(65536, device="cuda", dtype=torch.int32)
        .to(torch.int16)
        .view(torch.bfloat16)
    )
    values = values[torch.isfinite(values)]
    x = values[:, None].expand(-1, 128).contiguous()
    check(x, *quantize(x, pdl))


@pytest.mark.parametrize("pdl", [False, True])
def test_nan_values_remain_observable(pdl):
    x = torch.ones((3, 640), device="cuda", dtype=torch.bfloat16)
    x[:, ::17] = float("nan")
    q, s = quantize(x, pdl)
    assert torch.isnan(q.float()[torch.isnan(x)]).all()
    # NaNs do not replace the finite group's maximum or become +448.
    expected_q, expected_s = quantize(torch.ones_like(x), pdl)
    finite = torch.isfinite(x)
    torch.testing.assert_close(
        q.view(torch.uint8)[finite],
        expected_q.view(torch.uint8)[finite],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(s, expected_s, rtol=0, atol=0)

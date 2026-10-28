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

"""Dense Marlin MXFP8 contracts, numerical tests and CUDA graph replay."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.gemm import prepare_mxfp8_linear
from tokenspeed_kernel.ops.gemm.marlin import (
    MarlinMxfp8LinearKernel,
    marlin_mxfp8_padded_shape,
    marlin_mxfp8_permute_scales,
)


def test_minimal_padding():
    assert marlin_mxfp8_padded_shape(2624, 6144) == (2624, 6144)
    assert marlin_mxfp8_padded_shape(70, 96) == (128, 128)
    assert marlin_mxfp8_padded_shape(64, 32) == (64, 128)
    with pytest.raises(ValueError, match="divisible"):
        marlin_mxfp8_padded_shape(64, 33)


def test_raw_scale_permutation_roundtrip():
    original = (torch.arange(96 * 3).reshape(96, 3) % 200 + 1).to(torch.uint8)
    got = marlin_mxfp8_permute_scales(original, padded_n=128, padded_k=128)
    undo_interleave = got.reshape(-1, 4)[:, [0, 2, 1, 3]].reshape(-1, 64)
    perm = torch.tensor([i + 8 * j for i in range(8) for j in range(8)])
    restored = undo_interleave[:, torch.argsort(perm)].reshape(4, 128).T
    assert torch.equal(restored[:96, :3], original)
    assert bool((restored[96:] == 127).all())


def _inputs(n, k):
    generator = torch.Generator(device="cuda").manual_seed(271828 + n + k)
    weight = (
        (torch.randn((n, k), generator=generator, device="cuda") * 3)
        .clamp(-32, 32)
        .to(torch.float8_e4m3fn)
    )
    scales = torch.randint(
        122, 128, (n, k // 32), generator=generator, device="cuda", dtype=torch.uint8
    )
    return weight, scales


def _reference(x, w, scales, bias):
    values = w.double() * torch.exp2(scales.double() - 127).repeat_interleave(32, dim=1)
    output = x.double() @ values.T
    if bias is not None:
        output = output + bias.double()
    return output.to(torch.bfloat16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("n,k", [(64, 128), (128, 64), (70, 96), (2624, 6144)])
@pytest.mark.parametrize("m", [1, 8, 17, 64, 646, 704])
def test_native_dense_against_fp64(m, n, k):
    w, s = _inputs(n, k)
    kernel = prepare_mxfp8_linear(
        w,
        s,
        input_dtype=torch.bfloat16,
        use_atomic_add=False,
        use_fp32_reduce=True,
        solution=None,
        override=None,
    )
    torch.manual_seed(7)
    x = (torch.randn((m, k), device="cuda") * 0.1).to(torch.bfloat16)
    y = kernel.apply(x)
    ref = _reference(x, w, s, None)
    assert bool(torch.isfinite(y).all())
    # Accumulation order can straddle a BF16 rounding boundary. Bound the
    # discrepancy in BF16 ULPs, including near-zero values, and relative L2.
    error = (y.double() - ref.double()).abs()
    ulp = (
        torch.nextafter(ref.abs(), torch.full_like(ref, float("inf"))).double()
        - ref.abs().double()
    ).abs()
    assert bool((error <= 2 * ulp + 2e-5).all())
    assert (
        float(torch.linalg.vector_norm(error) / torch.linalg.vector_norm(ref.double()))
        < 5e-4
    )
    assert torch.equal(y, kernel.apply(x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bias_graph_empty_and_absorption():
    w, s = _inputs(128, 128)
    kernel = prepare_mxfp8_linear(
        w,
        s,
        input_dtype=torch.bfloat16,
        use_atomic_add=False,
        use_fp32_reduce=True,
        solution=None,
        override=None,
    )
    x = torch.zeros((2, 8, 128), device="cuda", dtype=torch.bfloat16)
    x[..., 11] = 1
    bias = torch.arange(128, device="cuda", dtype=torch.bfloat16) / 128
    expected = _reference(x, w, s, bias)
    y = kernel.apply(x, bias)
    torch.testing.assert_close(y, expected, rtol=0, atol=0)
    assert kernel.apply(x[:0]).shape == (0, 8, 128)
    # Capture on a non-default stream after warming both operator and allocator.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            kernel.apply(x, bias)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = kernel.apply(x, bias)
    graph.replay()
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)
    x[..., 11] = 2
    graph.replay()
    torch.testing.assert_close(captured, _reference(x, w, s, bias), rtol=0, atol=0)
    materialized = kernel.dequantize_for_absorption()
    wanted = (
        w.double() * torch.exp2(s.double() - 127).repeat_interleave(32, dim=1)
    ).to(torch.bfloat16)
    torch.testing.assert_close(materialized, wanted, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("n,k", [(128, 64), (70, 96), (64, 32)])
def test_absorption_preserves_padding_boundaries(n, k):
    weight, scale = _inputs(n, k)
    plan = prepare_mxfp8_linear(
        weight,
        scale,
        input_dtype=torch.bfloat16,
        use_atomic_add=False,
        use_fp32_reduce=True,
        solution="marlin",
        override=None,
    )
    expected = (
        weight.double() * torch.exp2(scale.double() - 127).repeat_interleave(32, dim=1)
    ).to(torch.bfloat16)
    torch.testing.assert_close(
        plan.dequantize_for_absorption(), expected, rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_contiguous_offset_inputs_and_weights():
    weight, scale = _inputs(64, 128)
    storage = torch.empty(weight.numel() + 1, dtype=torch.uint8, device="cuda")
    unaligned_weight = storage[1:].view(torch.float8_e4m3fn).reshape_as(weight)
    unaligned_weight.copy_(weight)
    plan = prepare_mxfp8_linear(
        unaligned_weight,
        scale,
        input_dtype=torch.bfloat16,
        use_atomic_add=False,
        use_fp32_reduce=True,
        solution="marlin",
        override=None,
    )
    x = torch.zeros(2 * 128 + 1, dtype=torch.bfloat16, device="cuda")[1:].view(2, 128)
    x[:, 7] = 1
    torch.testing.assert_close(
        plan.apply(x), _reference(x, weight, scale, None), rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_scale_and_dtype_fail_closed():
    w, s = _inputs(64, 128)
    for invalid in (0, 255):
        bad = s.clone()
        bad[0, 0] = invalid
        with pytest.raises(ValueError, match="encoding"):
            MarlinMxfp8LinearKernel(w, bad, use_atomic_add=False, use_fp32_reduce=True)
    with pytest.raises(TypeError, match="E4M3"):
        MarlinMxfp8LinearKernel(
            w.bfloat16(), s, use_atomic_add=False, use_fp32_reduce=True
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "solution,override",
    [
        (None, None),
        ("marlin", None),
        (None, "marlin_mxfp8_linear"),
    ],
)
def test_selected_backend_computes_reference(solution, override):
    weight, scale = _inputs(64, 128)
    plan = prepare_mxfp8_linear(
        weight,
        scale,
        input_dtype=torch.bfloat16,
        use_atomic_add=False,
        use_fp32_reduce=True,
        solution=solution,
        override=override,
    )
    x = torch.ones(2, 128, device=weight.device, dtype=torch.bfloat16)
    torch.testing.assert_close(
        plan.apply(x), _reference(x, weight, scale, None), rtol=5e-3, atol=1e-4
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_unsupported_overrides_and_dtypes_fail(monkeypatch):
    from tokenspeed_kernel.selection import NoKernelFoundError, kernel_override

    weight, scale = _inputs(64, 128)

    kwargs = dict(
        input_dtype=torch.bfloat16,
        use_atomic_add=False,
        use_fp32_reduce=True,
        solution=None,
        override=None,
    )
    with kernel_override("gemm", "mxfp8_w8a16", "missing_mxfp8_backend"):
        with pytest.raises(NoKernelFoundError):
            prepare_mxfp8_linear(weight, scale, **kwargs)
    monkeypatch.setenv(
        "TOKENSPEED_KERNEL_OVERRIDE_GEMM_MXFP8_W8A16", "missing_mxfp8_backend"
    )
    with pytest.raises(NoKernelFoundError):
        prepare_mxfp8_linear(
            weight, scale, **{**kwargs, "override": "marlin_mxfp8_linear"}
        )
    monkeypatch.delenv("TOKENSPEED_KERNEL_OVERRIDE_GEMM_MXFP8_W8A16")
    with pytest.raises(NoKernelFoundError):
        prepare_mxfp8_linear(weight, scale, **{**kwargs, "solution": "torch"})
    with pytest.raises(TypeError, match="BF16"):
        prepare_mxfp8_linear(
            weight,
            scale,
            **{
                **kwargs,
                "input_dtype": torch.float16,
                "override": "marlin_mxfp8_linear",
            },
        )

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

"""Supported model dimensions and fallback behavior share the public HC entry."""

from __future__ import annotations

import math

import pytest
import torch
from tokenspeed_kernel.numerics.torch.residual import torch_hyperconnection_mix
from tokenspeed_kernel.ops.residual import (
    cute_fused,
    gated_residual_mix,
    pad_gated_residual_projection_weight,
)
from tokenspeed_kernel.platform import pdl_enabled


@pytest.fixture(autouse=True)
def fused_device():
    if not torch.cuda.is_available() or not cute_fused.supports_fused_hc(
        torch.device("cuda")
    ):
        pytest.skip("requires supported Blackwell and CuTe DSL")
    previous = pdl_enabled()
    yield
    torch.cuda.synchronize()
    pdl_enabled(previous)


def _inputs(g, h, r, rows, dtype, inject):
    generator = torch.Generator(device="cuda").manual_seed(1729 + g + h + r)
    x = torch.randn(rows, g * h, device="cuda", dtype=dtype, generator=generator)
    projection = torch.randn(
        r + (g if inject else 0), g * h, device="cuda", dtype=dtype, generator=generator
    ) / math.sqrt(g * h)
    up = torch.randn(
        g * h, r, device="cuda", dtype=dtype, generator=generator
    ) / math.sqrt(r)
    projection = pad_gated_residual_projection_weight(
        projection, projection_rows=r + (g if inject else 0)
    )
    return x, projection, up


def _run(values, g, h, r, inject):
    return gated_residual_mix(
        *values,
        g,
        h,
        r,
        projection_rows=r + (g if inject else 0),
        projection_scale=1.0 / g,
        weights_independent=True,
        override="cute_fused_hyperconnection_mix",
        solution=None,
        activation_quantization="none",
    )


def _check(actual, values, g, h, r, inject):
    expected = torch_hyperconnection_mix(
        *values, g, h, r, r + (g if inject else 0), 1.0 / g, True
    )
    for value, reference in zip(actual, expected):
        if reference is None:
            assert value is None
        else:
            atol, rtol = (
                (0.004, 0.01) if value.dtype == torch.bfloat16 else (0.001, 0.002)
            )
            torch.testing.assert_close(value.double(), reference, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "g,h,r,rows,inject,dtype,enable_pdl",
    [
        (4, 8192, 1024, 16, True, torch.bfloat16, True),
        (4, 8192, 1024, 64, True, torch.bfloat16, True),
        (4, 2560, 320, 17, True, torch.bfloat16, False),
        (4, 2560, 320, 97, False, torch.float16, True),
        (4, 2560, 320, 193, True, torch.bfloat16, True),
    ],
)
@torch.inference_mode()
def test_model_dimensions_and_changed_input_graph(
    g, h, r, rows, inject, dtype, enable_pdl
):
    pdl_enabled(enable_pdl)
    values = _inputs(g, h, r, rows, dtype, inject)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        eager = _run(values, g, h, r, inject)
    stream.synchronize()
    _check(eager, values, g, h, r, inject)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = _run(values, g, h, r, inject)
    for factor in (0.75, -0.5, 1.25):
        values[0].mul_(factor)
        values[2].mul_(0.875)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        _check(actual, values, g, h, r, inject)


@torch.inference_mode()
def test_interleaved_supported_model_dimensions():
    cases = [(4, 2560, 320), (4, 8192, 1024)]
    values = [_inputs(g, h, r, 16, torch.bfloat16, False) for g, h, r in cases]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for tensors, (g, h, r) in zip(values, cases, strict=True):
            _run(tensors, g, h, r, False)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = [
            _run(tensors, g, h, r, False)
            for tensors, (g, h, r) in zip(values, cases, strict=True)
        ]
    for _ in range(4):
        for tensors in values:
            tensors[0].mul_(0.75)
        graph.replay()
        torch.cuda.synchronize()
        for result, tensors, (g, h, r) in zip(actual, values, cases, strict=True):
            _check(result, tensors, g, h, r, False)


@pytest.mark.parametrize(
    "g,h,r", [(3, 128, 9), (2, 96, 16), (9, 64, 64), (2, 64, 2056)]
)
@torch.inference_mode()
def test_unsupported_dimensions_fall_back_and_preserve_exact_override(g, h, r):
    values = _inputs(g, h, r, 7, torch.bfloat16, True)
    actual = gated_residual_mix(
        *values,
        g,
        h,
        r,
        projection_rows=r + g,
        projection_scale=1.0 / g,
        weights_independent=True,
        override=None,
        solution=None,
        activation_quantization="none",
    )
    _check(actual, values, g, h, r, True)
    with pytest.raises(ValueError, match="validated model dimensions"):
        _run(values, g, h, r, True)

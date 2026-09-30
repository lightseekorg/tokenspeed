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

"""Numerical and graph-replay checks for the generic packed layout."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel import fp8_linear, prepare_fp8_linear
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp8 import (
    GLUON_BLOCK_FP8_WEIGHT_LAYOUT,
    launch_gluon_mm_fp8_blockscale_largem_gfx950,
    pack_gluon_fp8_blockscale_weight,
)


def _gfx950_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("requires a gfx950 GPU")
    device = torch.device("cuda:0")
    if torch.cuda.get_device_properties(device).gcnArchName.split(":")[0] != "gfx950":
        pytest.skip("requires gfx950")
    return device


def _reference(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scales: torch.Tensor,
    weight_scales: torch.Tensor,
) -> torch.Tensor:
    m, k = activation.shape
    n = weight.shape[0]
    result = torch.zeros((m, n), device=activation.device, dtype=torch.float32)
    for offset in range(0, k, 128):
        index = offset // 128
        partial = (
            activation[:, offset : offset + 128].float()
            @ weight[:, offset : offset + 128].float().T
        )
        row_scale = activation_scales[:, index, None]
        column_scale = weight_scales[:, index].repeat_interleave(128)[None, :]
        result.add_(partial * row_scale * column_scale)
    return result.to(torch.bfloat16)


@pytest.mark.parametrize(
    ("m", "n", "k"),
    [
        (8144, 4096, 512),
        (8144, 6144, 4096),
        (8192, 1024, 4096),
        (8192, 6144, 4096),
    ],
)
def test_block_fp8_numerics_and_graph_replay(
    m: int, n: int, k: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = _gfx950_device()

    torch.manual_seed(42)
    activation = torch.randint(-2, 3, (m, k), device=device).to(torch.float8_e4m3fn)
    weight = torch.randint(-2, 3, (n, k), device=device).to(torch.float8_e4m3fn)
    activation_scales = torch.rand((m, k // 128), device=device) * 0.05 + 0.05
    weight_scales = torch.rand((n // 128, k // 128), device=device) * 0.05 + 0.05
    packed = pack_gluon_fp8_blockscale_weight(weight)
    layout = GLUON_BLOCK_FP8_WEIGHT_LAYOUT
    kernel = launch_gluon_mm_fp8_blockscale_largem_gfx950

    result = kernel(
        activation,
        packed,
        activation_scales,
        weight_scales,
        torch.bfloat16,
        block_size=[128, 128],
        weight_layout=layout,
    )
    expected = _reference(activation, weight, activation_scales, weight_scales)
    torch.testing.assert_close(result, expected, rtol=0.02, atol=0.03)

    monkeypatch.setenv("TOKENSPEED_EXPERIMENTAL_GLUON_FP8_BLOCKSCALE", "1")
    plan = prepare_fp8_linear(weight, weight_scales, (128, 128))
    assert plan.prepared_weight_layout == layout
    public_result = fp8_linear(
        plan,
        activation,
        weight,
        weight_scales,
        input_scales=activation_scales,
        out_dtype=torch.bfloat16,
    )
    torch.testing.assert_close(public_result, result, rtol=0, atol=0)
    online_result = fp8_linear(
        plan,
        activation.to(torch.bfloat16),
        weight,
        weight_scales,
        out_dtype=torch.bfloat16,
    )
    assert torch.isfinite(online_result).all()

    graph_output = torch.empty_like(result)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        kernel(
            activation,
            packed,
            activation_scales,
            weight_scales,
            torch.bfloat16,
            block_size=[128, 128],
            weight_layout=layout,
            out=graph_output,
        )
    graph.replay()
    torch.testing.assert_close(graph_output, result, rtol=0, atol=0)
    activation.zero_()
    graph.replay()
    assert not torch.count_nonzero(graph_output)


def test_block_fp8_handles_broad_magnitudes_and_scales() -> None:
    device = _gfx950_device()
    m, n, k = 8144, 4096, 512
    torch.manual_seed(71)
    padded_activation = torch.empty(
        (m, k + 16), device=device, dtype=torch.float8_e4m3fn
    )
    activation = padded_activation[:, :k]
    activation.copy_((torch.randn((m, k), device=device) * 12).to(torch.float8_e4m3fn))
    weight = (torch.randn((n, k), device=device) * 18).to(torch.float8_e4m3fn)
    padded_scales = torch.empty((m, k // 128 + 1), device=device)
    activation_scales = padded_scales[:, : k // 128]
    activation_scales.copy_(
        torch.exp(torch.randn((m, k // 128), device=device) * 0.6) * 0.02
    )
    weight_scales = (
        torch.exp(torch.randn((n // 128, k // 128), device=device) * 0.6) * 0.02
    )
    output = torch.empty((m, n + 16), device=device, dtype=torch.bfloat16)[:, :n]

    actual = launch_gluon_mm_fp8_blockscale_largem_gfx950(
        activation,
        pack_gluon_fp8_blockscale_weight(weight),
        activation_scales,
        weight_scales,
        torch.bfloat16,
        block_size=[128, 128],
        weight_layout=GLUON_BLOCK_FP8_WEIGHT_LAYOUT,
        out=output,
    )
    assert actual.data_ptr() == output.data_ptr()
    expected = _reference(activation, weight, activation_scales, weight_scales)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.03)


def test_online_bf16_quantization_accepts_padded_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = _gfx950_device()
    m, n, k = 8144, 4096, 512
    torch.manual_seed(73)
    padded = torch.randn((m, k + 16), device=device, dtype=torch.bfloat16)
    activation = padded[:, :k]
    assert not activation.is_contiguous()
    weight = (torch.randn((n, k), device=device) * 4).to(torch.float8_e4m3fn)
    weight_scales = torch.rand((n // 128, k // 128), device=device) * 0.05 + 0.05

    monkeypatch.setenv("TOKENSPEED_EXPERIMENTAL_GLUON_FP8_BLOCKSCALE", "1")
    plan = prepare_fp8_linear(weight, weight_scales, (128, 128))
    assert plan.prepared_weight_layout == GLUON_BLOCK_FP8_WEIGHT_LAYOUT
    actual = fp8_linear(
        plan, activation, weight, weight_scales, out_dtype=torch.bfloat16
    )
    expected = fp8_linear(
        plan, activation.contiguous(), weight, weight_scales, out_dtype=torch.bfloat16
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

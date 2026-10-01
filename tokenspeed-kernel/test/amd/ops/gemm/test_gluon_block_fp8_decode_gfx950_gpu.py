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

"""Numerical, graph, and optional latency checks for packed gfx950 projections."""

from __future__ import annotations

import os
import statistics

import pytest
import torch
from tokenspeed_kernel.ops.gemm.triton import w8a8_block_fp8_matmul_triton
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp8.decode import (
    gluon_mm_fp8_blockscale_decode_reduce_gfx950,
    launch_gluon_mm_fp8_blockscale_decode_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp8.largem import (
    GLUON_BLOCK_FP8_WEIGHT_LAYOUT,
    gluon_mm_fp8_blockscale_largem_gfx950,
    pack_gluon_fp8_blockscale_weight,
)
from tokenspeed_triton.testing import do_bench_cudagraph
from utils import assert_no_triton_compile

_LONG_K_SHAPES = (
    (1024, 4096),
    (6144, 4096),
    (4096, 3072),
    (2048, 4096),
    (4096, 1536),
    (4096, 4096),
)


def _gfx950_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("requires gfx950")
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
    for k_tile in range(k // 128):
        start = k_tile * 128
        end = start + 128
        partial = activation[:, start:end].float() @ weight[:, start:end].float().T
        result.add_(
            partial
            * activation_scales[:, k_tile, None]
            * weight_scales[:, k_tile].repeat_interleave(128)[None, :]
        )
    return result.to(torch.bfloat16)


@pytest.mark.parametrize("m", [1, 2, 4, 8, 16, 64, 65, 128])
@pytest.mark.parametrize("n,k", [(1024, 4096), (4096, 512)])
def test_packed_decode_matches_block_fp8_reference(m: int, n: int, k: int) -> None:
    device = _gfx950_device()
    torch.manual_seed(37)
    activation = (torch.randn((m, k), device=device) * 4).to(torch.float8_e4m3fn)
    weight = (torch.randn((n, k), device=device) * 4).to(torch.float8_e4m3fn)
    activation_scales = torch.rand((m, k // 128), device=device) * 0.04 + 0.02
    weight_scales = torch.rand((n // 128, k // 128), device=device) * 0.04 + 0.02
    actual = launch_gluon_mm_fp8_blockscale_decode_gfx950(
        activation,
        pack_gluon_fp8_blockscale_weight(weight),
        activation_scales,
        weight_scales,
        torch.bfloat16,
        block_size=[128, 128],
        weight_layout=GLUON_BLOCK_FP8_WEIGHT_LAYOUT,
    )
    expected = _reference(activation, weight, activation_scales, weight_scales)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.03)


@pytest.mark.parametrize("n,k", _LONG_K_SHAPES)
@pytest.mark.parametrize("m", [1, 65, 128])
def test_packed_decode_matches_canonical_projection(m: int, n: int, k: int) -> None:
    device = _gfx950_device()
    torch.manual_seed(103)
    activation = (torch.randn((m, k), device=device) * 4).to(torch.float8_e4m3fn)
    weight = (torch.randn((n, k), device=device) * 4).to(torch.float8_e4m3fn)
    activation_scales = torch.rand((m, k // 128), device=device) * 0.04 + 0.02
    weight_scales = torch.rand((n // 128, k // 128), device=device) * 0.04 + 0.02
    canonical = w8a8_block_fp8_matmul_triton(
        activation,
        weight,
        activation_scales,
        weight_scales,
        [128, 128],
        torch.bfloat16,
    )
    packed = launch_gluon_mm_fp8_blockscale_decode_gfx950(
        activation,
        pack_gluon_fp8_blockscale_weight(weight),
        activation_scales,
        weight_scales,
        torch.bfloat16,
        block_size=[128, 128],
        weight_layout=GLUON_BLOCK_FP8_WEIGHT_LAYOUT,
    )
    torch.testing.assert_close(packed, canonical, rtol=0.02, atol=0.03)


@pytest.mark.parametrize("n,k", [(1024, 4096), (4096, 1536)])
def test_packed_decode_row_variation_does_not_compile(n: int, k: int) -> None:
    device = _gfx950_device()
    weight = torch.ones((n, k), device=device, dtype=torch.float8_e4m3fn)
    packed_weight = pack_gluon_fp8_blockscale_weight(weight)
    weight_scales = torch.ones((n // 128, k // 128), device=device)

    def run(m: int) -> torch.Tensor:
        activation = torch.ones((m, k), device=device, dtype=torch.float8_e4m3fn)
        activation_scales = torch.ones((m, k // 128), device=device)
        return launch_gluon_mm_fp8_blockscale_decode_gfx950(
            activation,
            packed_weight,
            activation_scales,
            weight_scales,
            torch.bfloat16,
            block_size=[128, 128],
            weight_layout=GLUON_BLOCK_FP8_WEIGHT_LAYOUT,
        )

    run(1)
    run(64)
    run(65)
    with assert_no_triton_compile(
        gluon_mm_fp8_blockscale_largem_gfx950,
        gluon_mm_fp8_blockscale_decode_reduce_gfx950,
    ):
        for m in (2, 4, 8, 16, 32, 48, 96, 97, 128):
            assert torch.all(run(m) == k)


@pytest.mark.parametrize(
    "m,n,k",
    [(4, 1024, 4096), (1, 4096, 1536), (65, 6144, 4096), (65, 4096, 1536)],
)
def test_packed_decode_graph_replay(m: int, n: int, k: int) -> None:
    device = _gfx950_device()
    activation = torch.ones((m, k), device=device, dtype=torch.float8_e4m3fn)
    packed_weight = pack_gluon_fp8_blockscale_weight(
        torch.ones((n, k), device=device, dtype=torch.float8_e4m3fn)
    )
    activation_scales = torch.ones((m, k // 128), device=device)
    weight_scales = torch.ones((n // 128, k // 128), device=device)
    output = torch.empty((m, n), device=device, dtype=torch.bfloat16)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch_gluon_mm_fp8_blockscale_decode_gfx950(
            activation,
            packed_weight,
            activation_scales,
            weight_scales,
            torch.bfloat16,
            block_size=[128, 128],
            weight_layout=GLUON_BLOCK_FP8_WEIGHT_LAYOUT,
            out=output,
        )
    graph.replay()
    assert torch.all(output == k)
    activation.zero_()
    graph.replay()
    assert not torch.count_nonzero(output)


def test_packed_decode_accepts_padded_rows_and_output() -> None:
    device = _gfx950_device()
    m, n, k = 65, 4096, 4096
    activation = torch.ones((m, k + 16), device=device, dtype=torch.float8_e4m3fn)[
        :, :k
    ]
    weight = torch.ones((n, k), device=device, dtype=torch.float8_e4m3fn)
    activation_scales = torch.ones((m, k // 128 + 1), device=device)[:, : k // 128]
    weight_scales = torch.ones((n // 128, k // 128), device=device)
    output = torch.empty((m, n + 16), device=device, dtype=torch.bfloat16)[:, :n]
    actual = launch_gluon_mm_fp8_blockscale_decode_gfx950(
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
    assert torch.all(actual == k)


@pytest.mark.parametrize(
    "m,n,k",
    [
        (m, n, k)
        for n, k in (*_LONG_K_SHAPES, (4096, 512))
        for m in (1, 4, 16, 64, 65, 128)
    ],
)
def test_packed_decode_latency_comparison(m: int, n: int, k: int) -> None:
    if not os.environ.get("TOKENSPEED_BENCH_PACKED_DECODE"):
        pytest.skip("run explicitly with TOKENSPEED_BENCH_PACKED_DECODE=1")
    device = _gfx950_device()
    torch.manual_seed(79)
    activation = (torch.randn((m, k), device=device) * 4).to(torch.float8_e4m3fn)
    weight = (torch.randn((n, k), device=device) * 4).to(torch.float8_e4m3fn)
    packed_weight = pack_gluon_fp8_blockscale_weight(weight)
    activation_scales = torch.rand((m, k // 128), device=device) * 0.04 + 0.02
    weight_scales = torch.rand((n // 128, k // 128), device=device) * 0.04 + 0.02
    canonical_output = torch.empty((m, n), device=device, dtype=torch.bfloat16)
    packed_output = torch.empty_like(canonical_output)

    def canonical() -> None:
        w8a8_block_fp8_matmul_triton(
            activation,
            weight,
            activation_scales,
            weight_scales,
            [128, 128],
            torch.bfloat16,
            out=canonical_output,
        )

    def packed() -> None:
        launch_gluon_mm_fp8_blockscale_decode_gfx950(
            activation,
            packed_weight,
            activation_scales,
            weight_scales,
            torch.bfloat16,
            block_size=[128, 128],
            weight_layout=GLUON_BLOCK_FP8_WEIGHT_LAYOUT,
            out=packed_output,
        )

    canonical()
    packed()
    torch.testing.assert_close(packed_output, canonical_output, rtol=0.02, atol=0.03)
    baseline_us = 1000 * statistics.median(
        do_bench_cudagraph(canonical, rep=30) for _ in range(5)
    )
    candidate_us = 1000 * statistics.median(
        do_bench_cudagraph(packed, rep=30) for _ in range(5)
    )
    print(
        f"M={m} N={n} K={k} canonical={baseline_us:.2f}us "
        f"packed={candidate_us:.2f}us ratio={baseline_us / candidate_us:.3f}x"
    )

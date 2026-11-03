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

"""GFX1250 group-32 FP8 quantization matches the portable Triton kernel bit for bit."""

from __future__ import annotations

import pytest
import torch
from utils import assert_no_triton_compile, is_cdna5

if not is_cdna5():
    pytest.skip("AMD CDNA5 is required", allow_module_level=True)

from tokenspeed_kernel.ops.quantization import quantize_fp8  # noqa: E402
from tokenspeed_kernel.ops.quantization.triton import (  # noqa: E402
    triton_quantize_fp8_group32_ue8m0,
)
from tokenspeed_kernel.selection import select_kernel  # noqa: E402
from tokenspeed_kernel.signature import (  # noqa: E402
    dense_tensor_format,
    format_signature,
)
from tokenspeed_kernel_amd.ops.gfx1250.quantization import (  # noqa: E402
    gluon_quantize_fp8_group32_ue8m0_gfx1250,
    launch_gluon_quantize_fp8_group32_ue8m0_gfx1250,
)


def _triton(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return triton_quantize_fp8_group32_ue8m0(x, "token_group", 32, "ue8m0", False)


def _assert_bit_exact(actual, expected) -> None:
    assert actual[0].dtype == torch.float8_e4m3fn
    assert actual[1].dtype == torch.uint8
    assert torch.equal(actual[0].view(torch.uint8), expected[0].view(torch.uint8))
    assert torch.equal(actual[1], expected[1])


def _inputs(m: int, k: int, seed: int, dtype=torch.bfloat16) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn((m, k), device="cuda", generator=generator) * 3).to(dtype)


# Rows cover decode batches and prefill chunks on both sides of the launch
# shape switch at 49152 groups; K=576 blocks span several rows.
@pytest.mark.parametrize("m", [0, 1, 96, 848, 8192])
@pytest.mark.parametrize("k", [576, 5120])
def test_quantize_matches_triton(m: int, k: int) -> None:
    x = _inputs(m, k, seed=m + k)
    _assert_bit_exact(launch_gluon_quantize_fp8_group32_ue8m0_gfx1250(x), _triton(x))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_quantize_edge_values_match_triton(dtype: torch.dtype) -> None:
    x = _inputs(16, 2048, seed=7, dtype=torch.float32)
    x[0] = 0
    x[1] = 1e-7
    # One group spans every power-of-two scale boundary up to FP16 overflow.
    x[2, :32] = 448.0 * 2.0 ** torch.arange(-8, 24, device="cuda").float()
    x[3] = 448.0
    x[4] = -448.0 * 2.0**-20
    x[5, ::2] = -1e-30
    x = x.to(dtype)
    _assert_bit_exact(launch_gluon_quantize_fp8_group32_ue8m0_gfx1250(x), _triton(x))


def test_quantize_strided_rows_match_triton() -> None:
    x = _inputs(96, 6144, seed=3)[:, :5120]
    _assert_bit_exact(launch_gluon_quantize_fp8_group32_ue8m0_gfx1250(x), _triton(x))


def test_group32_ue8m0_selects_gluon_quantizer() -> None:
    kernel = select_kernel(
        "quantization",
        "fp8_with_scale",
        format_signature(x=dense_tensor_format(torch.bfloat16)),
        traits={"granularity": "token_group_32", "scale_encoding": "ue8m0"},
    )
    assert kernel.name == "gluon_quantize_fp8_group32_ue8m0_gfx1250"
    x = _inputs(96, 5120, seed=11)
    _assert_bit_exact(
        quantize_fp8(
            x, granularity="token_group", group_size=32, scale_encoding="ue8m0"
        ),
        _triton(x),
    )


def test_quantize_cuda_graph_replay() -> None:
    x = _inputs(32, 5120, seed=1)
    launch_gluon_quantize_fp8_group32_ue8m0_gfx1250(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = launch_gluon_quantize_fp8_group32_ue8m0_gfx1250(x)
    x.copy_(_inputs(32, 5120, seed=2))
    graph.replay()
    _assert_bit_exact(out, _triton(x))


def test_quantize_varying_rows_reuse_compiled_kernels() -> None:
    for m in (1, 8192):
        launch_gluon_quantize_fp8_group32_ue8m0_gfx1250(_inputs(m, 5120, seed=m))
    with assert_no_triton_compile(gluon_quantize_fp8_group32_ue8m0_gfx1250):
        for m in (6, 96, 848, 8191):
            launch_gluon_quantize_fp8_group32_ue8m0_gfx1250(_inputs(m, 5120, seed=m))

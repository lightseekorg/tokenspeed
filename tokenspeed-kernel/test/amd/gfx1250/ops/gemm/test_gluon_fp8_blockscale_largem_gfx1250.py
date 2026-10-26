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

"""Large-M FP8 projections with FP32 ``[128,128]`` block scales on gfx1250."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.numerics.torch.gemm import torch_mm_fp8_blockscale
from utils import assert_no_triton_compile, is_cdna5

if not is_cdna5():
    pytest.skip("AMD CDNA5 is required", allow_module_level=True)

from tokenspeed_kernel.ops.gemm import mm  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx1250.gemm.mxfp8 import (  # noqa: E402
    launch_gluon_mm_fp8_blockscale_largem_gfx1250,
)
from tokenspeed_kernel_amd.ops.gfx1250.gemm.mxfp8 import mm as largem_mod  # noqa: E402

# FP32 accumulation of 128-wide partials, then one BF16 rounding (2**-8).
_BF16_ATOL = 4e-3
_BF16_RTOL = 4e-3


def _operands(
    m: int, n: int, k: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    a = (torch.randn((m, k), device="cuda") * 0.5).to(torch.float8_e4m3fn)
    b = (torch.randn((n, k), device="cuda") * 0.5).to(torch.float8_e4m3fn)
    a_scales = torch.rand((m, k // 128), device="cuda") * 0.02 + 1e-3
    b_scales = torch.rand((n // 128, k // 128), device="cuda") * 0.02 + 1e-3
    return a, b, a_scales, b_scales


def _reference(a, b, a_scales, b_scales) -> torch.Tensor:
    return torch_mm_fp8_blockscale(
        a, b, a_scales, b_scales, torch.float32, block_size=[128, 128]
    )


def _run(a, b, a_scales, b_scales) -> torch.Tensor:
    return launch_gluon_mm_fp8_blockscale_largem_gfx1250(
        a, b, a_scales, b_scales, torch.bfloat16, block_size=[128, 128]
    )


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    scale = expected.abs().max()
    torch.testing.assert_close(
        actual.float() / scale, expected / scale, atol=_BF16_ATOL, rtol=_BF16_RTOL
    )


@pytest.mark.parametrize("m", [65, 96, 128, 129, 848, 2048, 8192])
@pytest.mark.parametrize(
    "n,k",
    [
        (1024, 4096),
        (4096, 512),
        (6144, 4096),
        (4096, 3072),
        (2048, 4096),
        (4096, 1536),
        (4096, 4096),
    ],
)
def test_fp8_blockscale_largem_matches_dequantized_reference(
    m: int, n: int, k: int
) -> None:
    operands = _operands(m, n, k, seed=m + n + k)
    actual = _run(*operands)
    assert actual.dtype == torch.bfloat16
    _assert_close(actual, _reference(*operands))


def test_fp8_blockscale_largem_rejects_partial_k_tiles() -> None:
    a, b, a_scales, b_scales = _operands(65, 128, 640, seed=3)
    with pytest.raises(ValueError, match="K by 512"):
        _run(a, b, a_scales, b_scales)


@pytest.mark.parametrize("m", [96, 4096])
def test_fp8_blockscale_largem_cuda_graph_replay(m: int) -> None:
    n, k = 2048, 4096
    a, b, a_scales, b_scales = _operands(m, n, k, seed=7)
    _run(a, b, a_scales, b_scales)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _run(a, b, a_scales, b_scales)
    fresh_a, _, fresh_scales, _ = _operands(m, n, k, seed=8)
    a.copy_(fresh_a)
    a_scales.copy_(fresh_scales)
    graph.replay()
    _assert_close(out, _reference(a, b, a_scales, b_scales))


def test_fp8_blockscale_largem_varying_rows_reuse_compiled_kernels() -> None:
    n, k = 2048, 4096
    _, b, _, b_scales = _operands(1, n, k, seed=9)

    def run(m: int) -> None:
        a, _, a_scales, _ = _operands(m, 128, k, seed=m)
        _run(a, b, a_scales, b_scales)

    for m in (65, 96, 128, 512, 1024, 2048, 4096, 6000):
        run(m)
    with assert_no_triton_compile(largem_mod.gluon_mm_fp8_blockscale_largem_gfx1250):
        for m in (72, 112, 200, 848, 1000, 3000, 8192):
            run(m)


@pytest.mark.parametrize("m", [65, 848, 2048])
def test_mm_dispatches_fp32_blockscale_above_decode_rows_to_largem_kernel(
    m: int,
) -> None:
    a, b, a_scales, b_scales = _operands(m, 2048, 4096, seed=m)
    kwargs = {
        "A_scales": a_scales,
        "B_scales": b_scales,
        "out_dtype": torch.bfloat16,
        "quant": "mxfp8",
        "block_size": [128, 128],
    }
    expected = mm(a, b, override="gluon_mm_fp8_blockscale_largem_gfx1250", **kwargs)
    torch.testing.assert_close(mm(a, b, **kwargs), expected, atol=0, rtol=0)
    _assert_close(expected, _reference(a, b, a_scales, b_scales))

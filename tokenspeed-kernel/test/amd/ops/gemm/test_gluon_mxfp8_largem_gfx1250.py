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

"""Large-M MXFP8 projection coverage for gfx1250 batches above 16 rows."""

from __future__ import annotations

import pytest
import torch
from utils import assert_no_triton_compile, is_cdna5

if not is_cdna5():
    pytest.skip("AMD CDNA5 is required", allow_module_level=True)

from tokenspeed_kernel.ops.gemm import mm  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx1250.gemm.fp16.mm import (  # noqa: E402
    gluon_wmma_dense_reduce_gfx1250,
)
from tokenspeed_kernel_amd.ops.gfx1250.gemm.mxfp8 import (  # noqa: E402
    launch_gluon_mm_mxfp8_ue8m0_largem_gfx1250,
)
from tokenspeed_kernel_amd.ops.gfx1250.gemm.mxfp8 import mm as largem_mod  # noqa: E402

# Inputs dequantize exactly; leave a small margin above BF16's worst-case
# relative rounding error (2**-8) when comparing against FP32 references.
_BF16_ATOL = 4e-3
_BF16_RTOL = 4e-3


def _operands(
    m: int, n: int, k: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    groups = (k + 31) // 32
    a = (torch.randn((m, k), device="cuda") * 0.25).to(torch.float8_e4m3fn)
    b = (torch.randn((n, k), device="cuda") * 0.25).to(torch.float8_e4m3fn)
    a_scales = torch.randint(122, 128, (m, groups), device="cuda", dtype=torch.uint8)
    b_scales = torch.randint(122, 128, (n, groups), device="cuda", dtype=torch.uint8)
    return a, b, a_scales, b_scales


def _dequantize(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    decoded = scales.view(torch.float8_e8m0fnu).float().repeat_interleave(32, dim=1)
    return values.float() * decoded[:, : values.shape[1]]


def _reference(a, b, a_scales, b_scales) -> torch.Tensor:
    return _dequantize(a, a_scales) @ _dequantize(b, b_scales).T


def _run(a, b, a_scales, b_scales) -> torch.Tensor:
    return launch_gluon_mm_mxfp8_ue8m0_largem_gfx1250(
        a, b, a_scales, b_scales, torch.bfloat16, block_size=[1, 32]
    )


@pytest.mark.parametrize("m", [17, 32, 33, 80, 96, 128, 129, 848, 2048, 8192])
@pytest.mark.parametrize(
    "n,k",
    [(1792, 5120), (4096, 1280), (8192, 1280), (5120, 2048), (1152, 5120), (5120, 576)],
)
def test_mxfp8_largem_matches_dequantized_reference(m: int, n: int, k: int) -> None:
    operands = _operands(m, n, k, seed=m + n + k)
    actual = _run(*operands)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(
        actual.float(), _reference(*operands), atol=_BF16_ATOL, rtol=_BF16_RTOL
    )


@pytest.mark.parametrize("m", [24, 96])
def test_mxfp8_largem_cuda_graph_replay(m: int) -> None:
    n, k = 1792, 5120
    a, b, a_scales, b_scales = _operands(m, n, k, seed=7)
    _run(a, b, a_scales, b_scales)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _run(a, b, a_scales, b_scales)
    fresh_a, _, fresh_scales, _ = _operands(m, n, k, seed=8)
    a.copy_(fresh_a)
    a_scales.copy_(fresh_scales)
    graph.replay()
    torch.testing.assert_close(
        out.float(),
        _reference(a, b, a_scales, b_scales),
        atol=_BF16_ATOL,
        rtol=_BF16_RTOL,
    )


def test_mxfp8_largem_varying_rows_reuse_compiled_kernels() -> None:
    n, k = 1792, 5120
    _, b, _, b_scales = _operands(1, n, k, seed=9)

    def run(m: int) -> None:
        a, _, a_scales, _ = _operands(m, 16, k, seed=m)
        _run(a, b, a_scales, b_scales)

    for m in (17, 24, 32, 48, 64, 96, 128, 512, 1024, 2048, 4096):
        run(m)
    with assert_no_triton_compile(
        largem_mod.gluon_mm_mxfp8_ue8m0_largem_gfx1250, gluon_wmma_dense_reduce_gfx1250
    ):
        for m in (18, 31, 40, 63, 72, 112, 200, 848, 1000, 3000, 8192):
            run(m)


@pytest.mark.parametrize("m", [17, 96, 2048])
def test_mm_dispatches_mxfp8_above_decode_rows_to_largem_kernel(m: int) -> None:
    a, b, a_scales, b_scales = _operands(m, 1792, 5120, seed=m)
    kwargs = {
        "A_scales": a_scales,
        "B_scales": b_scales,
        "out_dtype": torch.bfloat16,
        "quant": "mxfp8",
        "block_size": [1, 32],
    }
    expected = mm(a, b, override="gluon_mm_mxfp8_ue8m0_largem_gfx1250", **kwargs)
    torch.testing.assert_close(mm(a, b, **kwargs), expected, atol=0, rtol=0)
    torch.testing.assert_close(
        expected.float(),
        _reference(a, b, a_scales, b_scales),
        atol=_BF16_ATOL,
        rtol=_BF16_RTOL,
    )

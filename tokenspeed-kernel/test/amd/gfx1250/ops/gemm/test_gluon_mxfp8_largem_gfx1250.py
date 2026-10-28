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

"""Large-M FP8 projection coverage for gfx1250 in both scale modes.

UE8M0 MXFP8 runs above 16 rows and FP32 ``[128,128]`` block scales above 64
rows, both on the same BK512 kernel.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch
from tokenspeed_kernel.numerics.torch.gemm import torch_mm_fp8_blockscale
from utils import assert_no_triton_compile, is_cdna5

if not is_cdna5():
    pytest.skip("AMD CDNA5 is required", allow_module_level=True)

from tokenspeed_kernel.ops.gemm import mm  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx1250.gemm.mxfp8 import (  # noqa: E402
    launch_gluon_mm_fp8_blockscale_largem_gfx1250,
    launch_gluon_mm_mxfp8_ue8m0_largem_gfx1250,
)
from tokenspeed_kernel_amd.ops.gfx1250.gemm.mxfp8 import mm as largem_mod  # noqa: E402

# UE8M0 inputs dequantize exactly, and FP32 scales add one FP32 accumulation of
# 128-wide partials; leave a small margin above BF16's worst-case relative
# rounding error (2**-8) when comparing against FP32 references.
_BF16_ATOL = 4e-3
_BF16_RTOL = 4e-3


def _ue8m0_operands(
    m: int, n: int, k: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    groups = (k + 31) // 32
    a = (torch.randn((m, k), device="cuda") * 0.25).to(torch.float8_e4m3fn)
    b = (torch.randn((n, k), device="cuda") * 0.25).to(torch.float8_e4m3fn)
    a_scales = torch.randint(122, 128, (m, groups), device="cuda", dtype=torch.uint8)
    b_scales = torch.randint(122, 128, (n, groups), device="cuda", dtype=torch.uint8)
    return a, b, a_scales, b_scales


def _fp32_operands(
    m: int, n: int, k: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    a = (torch.randn((m, k), device="cuda") * 0.5).to(torch.float8_e4m3fn)
    b = (torch.randn((n, k), device="cuda") * 0.5).to(torch.float8_e4m3fn)
    a_scales = torch.rand((m, k // 128), device="cuda") * 0.02 + 1e-3
    b_scales = torch.rand((n // 128, k // 128), device="cuda") * 0.02 + 1e-3
    return a, b, a_scales, b_scales


@dataclass(frozen=True)
class _ScaleMode:
    operands: Callable[
        [int, int, int, int],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]
    block_size: tuple[int, int]
    launch: Callable[..., torch.Tensor]
    op: str
    # FP32 scales span orders of magnitude, so compare relative to the
    # output's largest element.
    normalize: bool


_MODES = {
    "ue8m0": _ScaleMode(
        operands=_ue8m0_operands,
        block_size=(1, 32),
        launch=launch_gluon_mm_mxfp8_ue8m0_largem_gfx1250,
        op="gluon_mm_mxfp8_ue8m0_largem_gfx1250",
        normalize=False,
    ),
    "fp32": _ScaleMode(
        operands=_fp32_operands,
        block_size=(128, 128),
        launch=launch_gluon_mm_fp8_blockscale_largem_gfx1250,
        op="gluon_mm_fp8_blockscale_largem_gfx1250",
        normalize=True,
    ),
}


def _reference(mode: str, a, b, a_scales, b_scales) -> torch.Tensor:
    return torch_mm_fp8_blockscale(
        a,
        b,
        a_scales,
        b_scales,
        torch.float32,
        block_size=list(_MODES[mode].block_size),
    )


def _run(mode: str, a, b, a_scales, b_scales) -> torch.Tensor:
    return _MODES[mode].launch(
        a,
        b,
        a_scales,
        b_scales,
        torch.bfloat16,
        block_size=list(_MODES[mode].block_size),
    )


def _assert_close(mode: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    scale = expected.abs().max() if _MODES[mode].normalize else 1.0
    torch.testing.assert_close(
        actual.float() / scale, expected / scale, atol=_BF16_ATOL, rtol=_BF16_RTOL
    )


@pytest.mark.parametrize("m", [17, 32, 33, 80, 96, 128, 129, 848, 2048, 8192])
@pytest.mark.parametrize(
    "n,k",
    [
        (1792, 5120),
        (4096, 1280),
        (8192, 1280),
        (5120, 2048),
        (1152, 5120),
        (5120, 576),
        (25600, 6144),
    ],
)
def test_mxfp8_largem_matches_dequantized_reference(m: int, n: int, k: int) -> None:
    operands = _ue8m0_operands(m, n, k, seed=m + n + k)
    actual = _run("ue8m0", *operands)
    assert actual.dtype == torch.bfloat16
    _assert_close("ue8m0", actual, _reference("ue8m0", *operands))


@pytest.mark.parametrize("m", [17, 96, 848, 2048])
@pytest.mark.parametrize("n,k", [(4096, 512), (2048, 288)])
def test_mxfp8_largem_single_k_tile_matches_dequantized_reference(
    m: int, n: int, k: int
) -> None:
    operands = _ue8m0_operands(m, n, k, seed=m + n + k)
    _assert_close("ue8m0", _run("ue8m0", *operands), _reference("ue8m0", *operands))


@pytest.mark.parametrize("m", [16, 65, 96, 128, 129, 848, 2048, 8192])
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
    operands = _fp32_operands(m, n, k, seed=m + n + k)
    actual = _run("fp32", *operands)
    assert actual.dtype == torch.bfloat16
    _assert_close("fp32", actual, _reference("fp32", *operands))


def test_fp8_blockscale_largem_rejects_partial_k_tiles() -> None:
    operands = _fp32_operands(65, 128, 640, seed=3)
    with pytest.raises(ValueError, match="K by 512"):
        _run("fp32", *operands)


@pytest.mark.parametrize("m", [24, 96, 4096])
@pytest.mark.parametrize("mode", list(_MODES))
def test_largem_cuda_graph_replay(mode: str, m: int) -> None:
    n, k = 1792, 5120
    a, b, a_scales, b_scales = _MODES[mode].operands(m, n, k, 7)
    _run(mode, a, b, a_scales, b_scales)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _run(mode, a, b, a_scales, b_scales)
    fresh_a, _, fresh_scales, _ = _MODES[mode].operands(m, n, k, 8)
    a.copy_(fresh_a)
    a_scales.copy_(fresh_scales)
    graph.replay()
    _assert_close(mode, out, _reference(mode, a, b, a_scales, b_scales))


@pytest.mark.parametrize("mode", list(_MODES))
def test_largem_varying_rows_reuse_compiled_kernels(mode: str) -> None:
    n, k = 1792, 5120
    _, b, _, b_scales = _MODES[mode].operands(1, n, k, 9)

    def run(m: int) -> None:
        a, _, a_scales, _ = _MODES[mode].operands(m, 128, k, m)
        _run(mode, a, b, a_scales, b_scales)

    for m in (17, 24, 32, 48, 64, 96, 128, 512, 1024, 2048, 4096, 6000):
        run(m)
    with assert_no_triton_compile(
        largem_mod.gluon_mm_mxfp8_ue8m0_largem_bk512_gfx1250,
        largem_mod.gluon_mm_mxfp8_ue8m0_largem_bk256_gfx1250,
        largem_mod._mxfp8_split_reduce_gfx1250,
    ):
        for m in (18, 31, 40, 63, 72, 112, 200, 848, 1000, 3000, 8192):
            run(m)


@pytest.mark.parametrize(
    "mode,m",
    [
        ("ue8m0", 17),
        ("ue8m0", 96),
        ("ue8m0", 2048),
        ("fp32", 65),
        ("fp32", 96),
        ("fp32", 2048),
    ],
)
def test_mm_dispatches_above_decode_rows_to_largem_kernel(mode: str, m: int) -> None:
    a, b, a_scales, b_scales = _MODES[mode].operands(m, 1792, 5120, m)
    kwargs = {
        "A_scales": a_scales,
        "B_scales": b_scales,
        "out_dtype": torch.bfloat16,
        "quant": "mxfp8",
        "block_size": list(_MODES[mode].block_size),
    }
    expected = mm(a, b, override=_MODES[mode].op, **kwargs)
    torch.testing.assert_close(mm(a, b, **kwargs), expected, atol=0, rtol=0)
    _assert_close(mode, expected, _reference(mode, a, b, a_scales, b_scales))

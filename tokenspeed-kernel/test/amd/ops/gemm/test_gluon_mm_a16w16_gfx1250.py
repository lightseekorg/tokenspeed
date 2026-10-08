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

"""General BF16 GEMM coverage for gfx1250 with FP32 and BF16 outputs."""

from __future__ import annotations

import pytest
import torch
from utils import assert_no_triton_compile, is_cdna5

if not is_cdna5():
    pytest.skip("AMD CDNA5 is required", allow_module_level=True)

from tokenspeed_kernel.ops.gemm import (  # noqa: E402
    dsv4_linear_fp32,
    grouped_bf16_projection,
)
from tokenspeed_kernel_amd.ops.gfx1250.gemm.fp16 import mm as mm_mod  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx1250.gemm.fp16.mm import (  # noqa: E402
    gluon_mm_a16w16_gfx1250,
)

# FP32 accumulation of exact BF16 products differs from the FP32 reference
# only in summation order.
_ATOL = 2e-4
_RTOL = 2e-4


def _operands(m: int, n: int, k: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) / k**0.5
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    return a, b


def _reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a.double() @ b.double().T


def _mm_fp32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return gluon_mm_a16w16_gfx1250(a, b, torch.float32)


def _assert_bf16_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == torch.bfloat16
    # One BF16 rounding of an FP32 sum: half an ulp is 2**-9 relative.
    torch.testing.assert_close(actual.double(), expected, atol=_ATOL, rtol=2**-8)


@pytest.mark.parametrize(
    "m", [1, 5, 6, 16, 17, 24, 33, 64, 65, 96, 128, 200, 2048, 2049, 8192]
)
@pytest.mark.parametrize("n", [128, 384, 1024])
def test_mm_a16w16_fp32_matches_reference(m: int, n: int) -> None:
    a, b = _operands(m, n, 5120, seed=m * 7 + n)
    actual = _mm_fp32(a, b)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(
        actual.double(), _reference(a, b), atol=_ATOL, rtol=_RTOL
    )


@pytest.mark.parametrize("m", [6, 96])
def test_mm_a16w16_fp32_cuda_graph_replay(m: int) -> None:
    n, k = 384, 5120
    a, b = _operands(m, n, k, seed=11)
    _mm_fp32(a, b)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _mm_fp32(a, b)
    a.copy_(torch.randn_like(a) / k**0.5)
    graph.replay()
    torch.testing.assert_close(out.double(), _reference(a, b), atol=_ATOL, rtol=_RTOL)


@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
def test_mm_a16w16_varying_rows_reuse_compiled_kernels(out_dtype: torch.dtype) -> None:
    n, k = 384, 5120
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    warm = (6, 16, 24, 32, 48, 64, 96, 128, 512, 1024)
    for m in warm:
        gluon_mm_a16w16_gfx1250(
            torch.randn(m, k, device="cuda", dtype=torch.bfloat16), b, out_dtype
        )
    with assert_no_triton_compile(
        mm_mod._wmma_tdm_dense_kernel,
        mm_mod.gluon_wmma_dense_reduce_gfx1250,
    ):
        for m in (1, 5, 12, 15, 17, 30, 40, 63, 80, 112, 300, 1000):
            gluon_mm_a16w16_gfx1250(
                torch.randn(m, k, device="cuda", dtype=torch.bfloat16), b, out_dtype
            )


@pytest.mark.parametrize("m", [6, 96, 8192])
def test_dsv4_linear_fp32_dispatches_to_gfx1250_kernel(m: int) -> None:
    a, b = _operands(m, 384, 5120, seed=3)
    actual = dsv4_linear_fp32(a, b, override="gluon_dsv4_linear_fp32_gfx1250")
    torch.testing.assert_close(
        actual.double(), _reference(a, b), atol=_ATOL, rtol=_RTOL
    )
    torch.testing.assert_close(dsv4_linear_fp32(a, b), actual, atol=0, rtol=0)


@pytest.mark.parametrize("m", [1, 6, 33, 96, 512, 2048, 4096])
def test_mm_a16w16_bf16_matches_reference(m: int) -> None:
    a, b = _operands(m, 1024, 4096, seed=m + 1)
    _assert_bf16_close(gluon_mm_a16w16_gfx1250(a, b, torch.bfloat16), _reference(a, b))


@pytest.mark.parametrize("split_k", [2, 4])
def test_mm_a16w16_bf16_split_k_rounds_reduced_sum(split_k: int) -> None:
    m, n, k = 96, 1024, 4096
    a, b = _operands(m, n, k, seed=split_k + 40)
    block_m, block_n, block_k, num_buffers, warp_bases, _ = mm_mod._mm_a16w16_tiles(
        m, n, k
    )
    outputs = {}
    for dtype in (torch.float32, torch.bfloat16):
        outputs[dtype] = torch.empty((m, n), device="cuda", dtype=dtype)
        mm_mod._launch_wmma_tdm_dense_tiles(
            a,
            b,
            outputs[dtype],
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            num_buffers=num_buffers,
            warp_bases=warp_bases,
            split_k=split_k,
        )
    assert torch.equal(
        outputs[torch.bfloat16], outputs[torch.float32].to(torch.bfloat16)
    )
    _assert_bf16_close(outputs[torch.bfloat16], _reference(a, b))


def _padded_head_view(tokens: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    heads = torch.randn(tokens, 16, 512, device="cuda", dtype=torch.bfloat16) / 64
    return heads[:, :8].reshape(tokens, 1, -1)


@pytest.mark.parametrize("tokens", [1, 6, 96, 2048])
def test_grouped_bf16_projection_single_group_uses_gfx1250_kernel(
    tokens: int,
) -> None:
    x = _padded_head_view(tokens, seed=tokens)
    assert x.is_contiguous() == (tokens == 1)
    weight = torch.randn(1, 1024, 4096, device="cuda", dtype=torch.bfloat16)
    actual = grouped_bf16_projection(x, weight, None, None)
    assert actual.shape == (tokens, 1, 1024)
    direct = gluon_mm_a16w16_gfx1250(x[:, 0], weight[0], torch.bfloat16)
    assert torch.equal(actual[:, 0], direct)
    _assert_bf16_close(actual[:, 0], _reference(x[:, 0], weight[0]))
    out = torch.empty_like(actual)
    assert grouped_bf16_projection(x, weight, out, None).data_ptr() == out.data_ptr()
    assert torch.equal(out, actual)


def test_grouped_bf16_projection_keeps_torch_beyond_tuned_rows() -> None:
    x = _padded_head_view(4096, seed=1)
    weight = torch.randn(1, 1024, 4096, device="cuda", dtype=torch.bfloat16)
    actual = grouped_bf16_projection(x, weight, None, None)
    expected = torch.einsum("tgd,grd->tgr", x, weight)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_grouped_bf16_projection_cuda_graph_replay() -> None:
    x = _padded_head_view(6, seed=2)
    weight = torch.randn(1, 1024, 4096, device="cuda", dtype=torch.bfloat16)
    out = torch.empty((6, 1, 1024), device="cuda", dtype=torch.bfloat16)
    grouped_bf16_projection(x, weight, out, None)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        grouped_bf16_projection(x, weight, out, None)
    x.copy_(_padded_head_view(6, seed=3))
    graph.replay()
    _assert_bf16_close(out[:, 0], _reference(x[:, 0], weight[0]))

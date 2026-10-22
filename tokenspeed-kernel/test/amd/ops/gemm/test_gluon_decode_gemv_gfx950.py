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

from __future__ import annotations

import pytest
import torch
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel.ops.gemm import (  # noqa: E402
    _select_mv,
    _use_mv,
    mm,
    mv,
    mv_weightless_rmsnorm_gated_sigmoid,
    mv_weightless_rmsnorm_gated_sigmoid_supported,
)
from tokenspeed_kernel.ops.layernorm import (  # noqa: E402
    weightless_rmsnorm_gated_sigmoid,
)
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.decode_gemv import (  # noqa: E402
    _PREFERRED_SHAPES,
    gluon_rowcta_bf16_gemv_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm import (  # noqa: E402
    gluon_mm_a16w16_medium_gfx950,
)


# (257, 1000) takes the one-wave short-row config and (1024, 8192) the K rule;
# (160, 4096), (512, 8192) and (7168, 1536) take their measured _CONFIGS entry.
@pytest.mark.parametrize(
    ("n", "k"),
    [(128, 128), (257, 1000), (160, 4096), (512, 8192), (1024, 8192), (7168, 1536)],
)
def test_gluon_rowcta_bf16_gemv_matches_reference(n: int, k: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260915 + n + k)
    x = torch.randn((1, k), dtype=torch.bfloat16, device="cuda", generator=generator)
    weight = torch.randn(
        (n, k),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    actual = torch.empty((1, n), dtype=torch.bfloat16, device="cuda")
    gluon_rowcta_bf16_gemv_gfx950(x, weight, actual)
    expected = x.float() @ weight.float().T
    torch.testing.assert_close(actual.float(), expected, atol=0.5, rtol=8e-3)


def test_gated_gemv_matches_the_gate_norm_then_the_gemv() -> None:
    """One launch reproduces the gated norm and the single-row GEMV bit for bit.

    This is Muse Spark's TP8 output projection; the gate is a strided view of
    a wider projection, as the attention split leaves it. The TP4 projection
    keeps the two launches.
    """
    n, heads = 8192, 16
    k = heads * 64
    generator = torch.Generator(device="cuda").manual_seed(20261022 + k)
    x = torch.randn((1, k), dtype=torch.bfloat16, device="cuda", generator=generator)
    projected = torch.randn(
        (1, 3 * k), dtype=torch.bfloat16, device="cuda", generator=generator
    )
    gate = projected[:, k : 2 * k]
    weight = (
        torch.randn((n, k), dtype=torch.bfloat16, device="cuda", generator=generator)
        / 32
    )
    assert mv_weightless_rmsnorm_gated_sigmoid_supported(x, gate, weight, heads, 64)
    gated = weightless_rmsnorm_gated_sigmoid(x, gate, heads, 64, 1e-6, False)
    expected = mv(gated, weight)
    actual = torch.empty_like(expected)
    mv_weightless_rmsnorm_gated_sigmoid(x, gate, weight, heads, 1e-6, actual)
    assert torch.equal(actual, expected)
    # A row narrower than the weight's K is not one the launch can serve.
    assert not mv_weightless_rmsnorm_gated_sigmoid_supported(
        x[:, : k // 2], gate[:, : k // 2], weight, heads, 64
    )
    wide = torch.empty((1, 2 * k), dtype=torch.bfloat16, device="cuda")
    assert not mv_weightless_rmsnorm_gated_sigmoid_supported(
        wide, wide, weight.new_empty((n, 2 * k)), 2 * heads, 64
    )


def test_decode_gemv_selects_gluon_on_gfx950() -> None:
    assert _select_mv(
        1, 1024, 8192, True, torch.bfloat16, torch.bfloat16, False
    ).__name__ == ("gluon_rowcta_bf16_gemv_gfx950")


def test_mm_takes_the_rowcta_gemv_on_every_listed_single_row_shape() -> None:
    for n, k in sorted(_PREFERRED_SHAPES):
        x = torch.empty((1, k), dtype=torch.bfloat16, device="cuda")
        weight = torch.empty((n, k), dtype=torch.bfloat16, device="cuda")
        assert _use_mv(x, weight), (n, k)
        assert _select_mv(
            1, n, k, True, torch.bfloat16, torch.bfloat16, False
        ).__name__ == ("gluon_rowcta_bf16_gemv_gfx950"), (n, k)


def test_mm_keeps_its_own_path_for_an_unlisted_single_row() -> None:
    x = torch.empty((1, 1536), dtype=torch.bfloat16, device="cuda")
    weight = torch.empty((1536, 1536), dtype=torch.bfloat16, device="cuda")
    assert not _use_mv(x, weight)


def test_mm_routes_a_listed_single_row_through_the_rowcta_gemv() -> None:
    torch.manual_seed(20261017)
    x = torch.randn((1, 3072), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((2560, 3072), dtype=torch.bfloat16, device="cuda") / 3072**0.5
    expected = torch.empty((1, 2560), dtype=torch.bfloat16, device="cuda")
    gluon_rowcta_bf16_gemv_gfx950(x, weight, expected)
    assert torch.equal(mm(x, weight), expected)


def test_qkvg_single_row_dispatch_and_graph_replay() -> None:
    torch.manual_seed(20261007)
    x = torch.randn((1, 8192), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((4608, 8192), dtype=torch.bfloat16, device="cuda") / 8192**0.5
    assert _use_mv(x, weight)
    actual = mv(x, weight)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mv(x, weight, out=actual)
    for scale in (1.0, -0.75):
        x.mul_(scale)
        graph.replay()
        expected = x.float() @ weight.float().T
        torch.testing.assert_close(actual.float(), expected, atol=2e-5, rtol=2**-7)


@pytest.mark.parametrize("k", [2048, 4096])
def test_muse_decode_projection_replay_and_tail_cache(k: int) -> None:
    torch.manual_seed(20261007)
    x = torch.randn((32, k), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((8192, k), dtype=torch.bfloat16, device="cuda") / k**0.5
    out = torch.empty((32, 8192), dtype=torch.bfloat16, device="cuda")
    # The serving ladder warms both integer-alignment classes of this tile.
    for rows in (2, 16):
        mv(x[:rows], weight, out=out[:rows])
    with assert_no_triton_compile(gluon_mm_a16w16_medium_gfx950):
        for rows in (3, 17, 32):
            assert _use_mv(x[:rows], weight)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                mv(x[:rows], weight, out=out[:rows])
            x.mul_(-0.75)
            out.fill_(float("nan"))
            graph.replay()
            expected = x[:rows].float() @ weight.float().T
            torch.testing.assert_close(
                out[:rows].float(), expected, atol=2e-5, rtol=2**-7
            )

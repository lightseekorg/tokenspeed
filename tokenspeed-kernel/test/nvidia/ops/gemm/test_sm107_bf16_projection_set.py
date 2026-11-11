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

"""Tight numerical contracts for the SM107 admitted bf16 projection set."""

import pytest
import torch
from tokenspeed_kernel.ops.gemm import flashinfer as fi
from tokenspeed_kernel.ops.gemm import mm
from tokenspeed_kernel.ops.tuning import autotune
from tokenspeed_kernel.platform import pdl_enabled

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or fi.platform.arch_version != fi.ArchVersion(10, 7)
    or not fi.has_flashinfer_cute_dsl_bf16(),
    reason="SM107 with FlashInfer CuTe DSL required",
)

SHAPES = [
    (2560, 6144),  # attention o_proj, TP1
    (3584, 2560),  # attention qkv gated proj, TP4
    (4096, 2560),  # GDN in_proj, TP4
    (4120, 2560),  # GDN in_proj (padded variant), TP4
    (4160, 2560),  # GDN in_proj (padded variant), TP4
    (2560, 1536),  # attention o_proj, TP4
    (1280, 2560),  # MoE shared-expert gate_up, TP4
    (2560, 640),  # MoE shared-expert down, TP4
    (320, 2560),  # small-N router-class proj, TP4
    (512, 2560),  # small-N gate-up proj, TP4
]

# Must keep the original GEMM: microbenchmark losers (full-N projections, LM
# head) and K misaligned to the 128-element CuTe DSL tile.
NON_ADMITTED = [
    (13312, 2560),  # full-N qkv-class projection
    (16480, 2560),  # full-N GDN in_proj
    (248320, 2560),  # LM head
    (2560, 4104),  # K % 8 == 0 but K % 128 != 0
]


def inputs(m, n, k, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, device="cuda", generator=generator).bfloat16()
    w = (torch.randn(n, k, device="cuda", generator=generator) / k**0.5).bfloat16()
    return x, w


def check_result(got, x, w):
    reference = x.double() @ w.double().T
    error = (got.double() - reference).norm() / reference.norm().clamp_min(1e-12)
    assert error < 0.002
    torch.testing.assert_close(got.double(), reference, rtol=0.008, atol=3e-5)


@pytest.fixture(scope="module", autouse=True)
def tuned():
    if not torch.cuda.is_available() or fi.platform.arch_version != fi.ArchVersion(
        10, 7
    ):
        return
    AutoTuner = pytest.importorskip("flashinfer.autotuner").AutoTuner
    AutoTuner.get().clear_cache()
    with autotune(tune_mode=True, tuning_buckets=None, round_up=None):
        for n, k in SHAPES:
            for m in (1, 2, 3, 4):
                x, w = inputs(m, n, k, 19)
                mm(x, w)


@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("m", [1, 2, 3, 4])
def test_admitted_projection_eager_and_graph(n, k, m):
    x, w = inputs(m, n, k, 7)
    storage = torch.full((m * n + 32,), 42, device="cuda", dtype=torch.bfloat16)
    out = storage[: m * n].view(m, n)
    assert fi.flashinfer_joint_bf16_supported(x, w, out)
    assert mm(x, w, out=out) is out
    check_result(out, x, w)
    # The graph replay contract needs a deterministic runner.
    assert torch.equal(mm(x, w), out)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        mm(x, w, out=out)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        mm(x, w, out=out)
    for scale in [0.125, -8.0]:
        x.mul_(scale)
        expected = mm(x, w).clone()
        out.fill_(float("nan"))
        graph.replay()
        assert torch.equal(out, expected)
        check_result(out, x, w)
        assert torch.all(storage[m * n :] == 42)


@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("m", [5, 32])
def test_larger_rows_keep_original_gemm(n, k, m):
    x, w = inputs(m, n, k, 99)
    out = torch.empty(m, n, dtype=x.dtype, device=x.device)
    with pytest.raises(ValueError, match="Unsupported input"):
        fi.flashinfer_bf16_gemm(x, w, out)
    assert torch.equal(mm(x, w, out=out), x @ w.T)


@pytest.mark.parametrize("n,k", NON_ADMITTED)
def test_non_admitted_shapes_keep_original_gemm(n, k):
    x, w = inputs(4, n, k, 10)
    assert not fi.flashinfer_joint_bf16_supported(x, w, None)
    assert torch.equal(mm(x, w), x @ w.T)


@pytest.mark.parametrize("m", [1, 4])
def test_pdl_and_cancellation(m):
    n, k = 2560, 1536
    x, w = inputs(m, n, k, 91)
    x[:, k // 2 :] = -x[:, : k // 2]
    w[:, k // 2 :] = w[:, : k // 2]
    previous = pdl_enabled()
    try:
        for enabled in [False, True]:
            pdl_enabled(enabled)
            out = mm(x, w)
            torch.testing.assert_close(
                out.float(), torch.zeros_like(out.float()), rtol=0, atol=3e-5
            )
    finally:
        pdl_enabled(previous)

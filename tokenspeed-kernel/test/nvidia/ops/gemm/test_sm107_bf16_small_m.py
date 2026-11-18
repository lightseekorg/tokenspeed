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

"""Tight numerical contracts for SM107 small-M BF16 GEMM admission."""

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

# Admitted by _sm107_cute_dsl_bf16_admits (K % 128 == 0 throughout): N <= 8192,
# or K >= 2048 with at most 96 MiB of weight. The first rows sit on each
# boundary; the rest are TP1 and TP4 projection shapes.
SHAPES = [
    (8192, 1536),  # N boundary, admitted with a short K
    (8192, 8192),  # N boundary, admitted above 96 MiB
    (16480, 2048),  # smallest K admitted for a wide N
    (24576, 2048),  # exactly 96 MiB
    (16480, 2560),
    (13312, 2560),
    (2560, 6144),
    (2560, 2560),
    (4160, 2560),
    (2560, 1536),
    (1280, 2560),
    (2560, 640),
    (640, 2560),
    (320, 2560),
]

# Decode row counts up to the small-M bound, including ones between buckets.
ADMITTED_ROWS = [1, 2, 3, 4, 8, 17, 32]

# Keep the original GEMM, one step past each boundary.
NON_ADMITTED = [
    (8320, 1536),  # N > 8192 with K < 2048
    (16480, 1920),  # K just below 2048 for a wide N
    (24576, 2176),  # wide N, K >= 2048, above 96 MiB
    (248320, 2560),  # an LM head
    (2560, 4104),  # K % 128 != 0
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
            for m in (1, 2, 4, 8, 16, 32):
                x, w = inputs(m, n, k, 19)
                mm(x, w)


@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("m", ADMITTED_ROWS)
def test_admitted_projection_eager_and_graph(n, k, m):
    x, w = inputs(m, n, k, 7)
    storage = torch.full((m * n + 32,), 42, device="cuda", dtype=torch.bfloat16)
    out = storage[: m * n].view(m, n)
    assert fi.flashinfer_joint_bf16_supported(x, w, out)
    assert mm(x, w, out=out) is out
    check_result(out, x, w)
    # The measured small-N projection uses the pinned CuTe DSL tile under PDL.
    if fi.prefer_bf16_small_n_splitk(m, n, k, enable_pdl=pdl_enabled()):
        expected = fi.flashinfer_mm_bf16_small_n_splitk(
            x, w, None, None, x.dtype, enable_pdl=True
        )
    else:
        expected = fi.flashinfer_bf16_gemm(x, w, None)
    assert torch.equal(expected, out)
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
@pytest.mark.parametrize("m", [33, 64])
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


@pytest.mark.parametrize("m", [1, 4, 32])
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

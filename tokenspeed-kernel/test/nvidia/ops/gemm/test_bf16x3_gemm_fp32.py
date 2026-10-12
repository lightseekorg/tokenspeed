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

"""GPU contracts of the BF16x3 FP32-weight GEMM behind ``decode_gemv``."""

from __future__ import annotations

import math

import pytest
import tokenspeed_kernel.ops.gemm  # noqa: F401  (registration side effects)
import torch
from tokenspeed_kernel.ops.gemm import triton_bf16x3
from tokenspeed_kernel.ops.gemm.triton_gemv import (
    _select,
    decode_gemv,
    decode_gemv_weight_split,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import KernelRegistry
from utils import (
    assert_no_triton_compile,
    int_specialization_class,
    warm_specialization_classes,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or current_platform().vendor != "nvidia"
    or not (10, 0) <= torch.cuda.get_device_capability() <= (10, 3),
    reason="the BF16x3 GEMM is registered for SM100 to SM103",
)

U = 2.0**-24


@pytest.fixture(autouse=True)
def _split_kernel_alone():
    """Set aside the FP32 CUDA-core kernel registered ahead of the split kernel
    for rows 17 to 96 of weights up to 256 rows with K a multiple of 512 up to
    8192, so that decode_gemv() runs the split kernel over its whole
    registration here.
    test_decode_gemv_fp32_simt.py checks the two together."""
    reg = KernelRegistry.get()
    name = "gluon_simt_gemm_fp32"
    spec, impl = reg.get_by_name(name), reg.get_impl(name)
    assert spec is not None
    reg._unregister(name)
    _select.cache_clear()
    yield
    reg.register(spec, impl)
    _select.cache_clear()


def _problem(m, n, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, device="cuda", generator=g).to(torch.bfloat16)
    w = torch.randn(n, k, device="cuda", generator=g) * 0.02
    return x, w


def _error_over_magnitude(y, x, w):
    """|y - x.w| in units of 2^-24 sum|x w|, against FP64."""
    exact = x.double() @ w.double().t()
    magnitude = x.double().abs() @ w.double().abs().t()
    return ((y.double() - exact).abs() / (U * magnitude)).max().item()


@pytest.mark.parametrize("m", [17, 33, 64, 65, 200, 513, 2048, 4096])
@pytest.mark.parametrize("n,k", [(384, 2048), (256, 7168)])
def test_split_kernel_matches_fp64(m, n, k):
    x, w = _problem(m, n, k, seed=m)
    pieces = decode_gemv_weight_split(w)
    assert pieces is not None and pieces.shape == (3, n, k)
    assert _select(m, n, k, True, x.dtype, w.dtype, True) is (
        triton_bf16x3.triton_bf16x3_gemm_fp32
    )
    y = decode_gemv(x, w, weight_split=pieces)
    assert y.dtype == torch.float32 and y.shape == (m, n)
    # Draining the MMA accumulator every 64 elements of K keeps the error near
    # one rounding of the largest partial; one accumulator over K reaches
    # tens of units here.
    assert _error_over_magnitude(y, x, w) <= 4


@pytest.mark.parametrize("m", [17, 64, 65, 513, 4096])
@pytest.mark.parametrize("n,k", [(384, 2048), (256, 7168)])
def test_split_kernel_returns_one_hot_rows_exactly(m, n, k):
    """A row with a single 1 picks one column of w. Every product is exact and
    every partial sum of the three pieces is an FP32 value, so the column comes
    back bit for bit, through the drain and split-K. Without one of the pieces
    the row gets the sum of the others instead."""
    _, w = _problem(1, n, k, seed=m + 1)
    g = torch.Generator(device="cuda").manual_seed(m)
    cols = torch.randint(0, k, (m,), device="cuda", generator=g)
    x = torch.zeros(m, k, device="cuda", dtype=torch.bfloat16)
    x[torch.arange(m, device="cuda"), cols] = 1
    pieces = decode_gemv_weight_split(w)
    assert (pieces[2] != 0).any()
    y = decode_gemv(x, w, weight_split=pieces)
    assert torch.equal(y, w[:, cols].t())


@pytest.mark.parametrize("m", [33, 8192])
def test_split_kernel_is_deterministic_and_replays_in_a_graph(m):
    n, k = 256, 7168  # split-K at 33 rows, a single split at 8192
    x, w = _problem(m, n, k, seed=1)
    pieces = decode_gemv_weight_split(w)
    out = torch.empty(m, n, device="cuda")
    eager = decode_gemv(x, w, weight_split=pieces).clone()
    assert torch.equal(decode_gemv(x, w, weight_split=pieces), eager)
    decode_gemv(x, w, out, weight_split=pieces)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        decode_gemv(x, w, out, weight_split=pieces)
    out.zero_()
    graph.replay()
    assert torch.equal(out, eager)


def test_non_finite_values_reach_the_output():
    m, n, k = 32, 256, 2048
    x, w = _problem(m, n, k, seed=2)
    w[0, 5] = math.inf
    w[1, 7] = math.nan
    w[2, 3], w[2, 9] = -math.inf, math.inf
    x[4, 5] = 0.0
    x[6, 9] = math.inf
    pieces = decode_gemv_weight_split(w)
    assert pieces is not None
    y = decode_gemv(x, w, weight_split=pieces)
    expected = x.double() @ w.double().t()
    assert torch.equal(y.isfinite(), expected.isfinite())
    # Non-finite weights act as in an FP32 GEMM. The infinite activation meets
    # pieces of either sign or zero, so its row may be NaN where FP32 is +-inf.
    keep = torch.arange(m, device="cuda") != 6
    y, expected = y[keep], expected[keep]
    assert torch.equal(y.isnan(), expected.isnan())
    assert torch.equal(y.isinf(), expected.isinf())
    assert torch.equal(y[y.isinf()].double(), expected[expected.isinf()])


def test_split_is_only_made_for_weights_a_kernel_takes():
    assert decode_gemv_weight_split(torch.zeros(256, 2048, device="cuda")) is not None
    for weight in (
        torch.zeros(128, 2048, device="cuda"),  # N below 256
        torch.zeros(320, 2048, device="cuda"),  # N not a multiple of 128
        torch.zeros(256, 4000, device="cuda"),  # K not a multiple of 64
        torch.zeros(256, 1984, device="cuda"),  # K below 2048
        torch.zeros(256, 8256, device="cuda"),  # K above 8192
        torch.zeros(256, 2048, device="cuda", dtype=torch.bfloat16),
    ):
        assert decode_gemv_weight_split(weight) is None


def test_row_counts_within_a_tile_bucket_do_not_recompile():
    n, k = 256, 7168
    _, w = _problem(1, n, k, seed=3)
    pieces = decode_gemv_weight_split(w)

    def run(m):
        x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
        y = decode_gemv(x, w, weight_split=pieces)
        assert _error_over_magnitude(y, x, w) <= 4

    def key(m):
        return (triton_bf16x3._bf16x3_config(m, n, k), int_specialization_class(m))

    sweep = (18, 23, 27, 31)
    warm_specialization_classes(run, key, sweep, pool=(17, 32))
    with assert_no_triton_compile(
        triton_bf16x3._bf16x3_gemm_kernel, triton_bf16x3._bf16x3_splitk_reduce_kernel
    ):
        for m in sweep:
            run(m)

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

"""Triton residual kernels."""

import pytest
import torch
from tokenspeed_kernel.numerics.torch.residual import torch_mhc_mixes
from tokenspeed_kernel.ops.residual import mhc_mixes
from tokenspeed_kernel.ops.residual.triton import (
    _mhc_prenorm_gemm_launch_config,
    _mhc_prenorm_gemm_triton_kernel,
    compute_mhc_num_splits,
)
from tokenspeed_kernel.ops.residual.triton_sm120 import _prenorm_gemm
from utils import (
    assert_no_triton_compile,
    int_specialization_class,
    warm_specialization_classes,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12,
    reason="SM120 mHC coefficients",
)


@pytest.fixture(autouse=True)
def select_sm120_mixes(monkeypatch):
    # Exercise this implementation even when the serving environment pins
    # the portable kernel for an older source checkout.
    monkeypatch.setenv(
        "TOKENSPEED_KERNEL_OVERRIDE_RESIDUAL_MHC_MIXES", "triton_sm120_mhc_mixes"
    )


def _operands(tokens, hidden, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(tokens, 4, hidden, device="cuda", generator=g).bfloat16()
    w = torch.randn(24, 4 * hidden, device="cuda", generator=g) * 0.01
    scale = torch.tensor([0.7, 1.1, 0.5], device="cuda")
    base = torch.randn(24, device="cuda", generator=g)
    return x, w, scale, base


@pytest.mark.parametrize(
    "tokens,hidden", [(1, 5120), (128, 5120), (1024, 5120), (2048, 5120), (257, 1000)]
)
def test_mixes_matches_fp64(tokens, hidden):
    x, w, scale, base = _operands(tokens, hidden, tokens + hidden)
    got = mhc_mixes(x, w, scale, base, 1e-6, 1e-6, 20)
    expected = torch_mhc_mixes(
        x, w.double(), scale.double(), base.double(), 1e-6, 1e-6, 20
    )
    for a, b in zip(got, expected, strict=True):
        torch.testing.assert_close(a.double(), b, rtol=1e-5, atol=1e-6)


def test_projection_partials_against_fp64():
    tokens, hidden = 1024, 5120
    x, w, _, _ = _operands(tokens, hidden, 271)
    splits = compute_mhc_num_splits(x.device, 64, 4 * hidden, 16)
    mul = torch.empty(splits, tokens, 24, device="cuda")
    sq = torch.empty(splits, tokens, device="cuda")
    _prenorm_gemm(x.view(tokens, -1), w, mul, sq, splits)
    parts = x.double().view(tokens, splits, -1)
    expected = torch.einsum("tsk,nsk->stn", parts, w.double().view(24, splits, -1))
    torch.testing.assert_close(mul.double(), expected, rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(sq.double(), parts.square().sum(-1).T, rtol=1e-6, atol=0)


def test_graph_replay_refreshes_operands():
    args = _operands(128, 5120, 84)
    mhc_mixes(*args, 1e-6, 1e-6, 20)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = mhc_mixes(*args, 1e-6, 1e-6, 20)
    for seed in [85, 86]:
        for dst, src in zip(args, _operands(128, 5120, seed), strict=True):
            dst.copy_(src)
        graph.replay()
        expected = torch_mhc_mixes(
            args[0], *(a.double() for a in args[1:]), 1e-6, 1e-6, 20
        )
        for got, wanted in zip(output, expected, strict=True):
            torch.testing.assert_close(got.double(), wanted, rtol=1e-5, atol=1e-6)


def test_projection_batch_shapes_do_not_recompile():
    tokens, hidden = 2048, 5120
    x, weight, _, _ = _operands(tokens, hidden, 991)

    def run(rows):
        splits = compute_mhc_num_splits(
            x.device, 64, 4 * hidden, max(1, (rows + 63) // 64)
        )
        mul = torch.empty(splits, rows, 24, device="cuda")
        sq = torch.empty(splits, rows, device="cuda")
        _prenorm_gemm(x[:rows].view(rows, -1), weight, mul, sq, splits)

    def key(rows):
        splits = compute_mhc_num_splits(
            x.device, 64, 4 * hidden, max(1, (rows + 63) // 64)
        )
        config = _mhc_prenorm_gemm_launch_config(rows, 4 * hidden, 24, splits)
        return (
            int_specialization_class(rows),
            int_specialization_class(splits),
            config,
        )

    run(64)
    run(65)
    sweep = (128, 192, 320, 448, 1024)
    warm_specialization_classes(run, key, sweep, range(64, 1025, 64))
    with assert_no_triton_compile(_mhc_prenorm_gemm_triton_kernel):
        for rows in (127, 129, *sweep):
            run(rows)

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

"""GFX950 Gluon kernels behind DeepSeek V4.1's hc=4 mHC helpers."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.residual import mhc_apply_pre, mhc_mixes, mhc_post
from tokenspeed_kernel.ops.residual.triton import (
    triton_mhc_apply_pre,
    triton_mhc_mixes,
    triton_mhc_post,
)
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required for the mHC tests", allow_module_level=True)

from tokenspeed_kernel_amd.ops.gfx950 import mhc  # noqa: E402

DEVICE = "cuda"
HC_EPS = 1e-6
RMS_EPS = 1e-20
SINKHORN_ITERS = 20


def _mixes_args(tokens: int, hidden: int, seed: int) -> tuple:
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    residual = torch.randn(
        tokens, 4, hidden, device=DEVICE, dtype=torch.bfloat16, generator=generator
    )
    weight = torch.randn(24, 4 * hidden, device=DEVICE, generator=generator) * 0.01
    scale = torch.tensor([0.7, 1.1, 0.5], device=DEVICE)
    base = torch.randn(24, device=DEVICE, generator=generator) * 0.1
    return residual, weight, scale, base, RMS_EPS, HC_EPS, SINKHORN_ITERS


def _mixes_reference(residual, weight, scale, base, rms_eps, hc_eps, iters):
    flat = residual.flatten(-2).double()
    mixes = flat @ weight.double().T
    mixes = mixes * torch.rsqrt(flat.square().mean(-1, keepdim=True) + rms_eps)
    pre = torch.sigmoid(mixes[:, :4] * scale[0] + base[:4]) + hc_eps
    post = 2 * torch.sigmoid(mixes[:, 4:8] * scale[1] + base[4:8])
    comb = (mixes[:, 8:] * scale[2] + base[8:]).unflatten(-1, (4, 4))
    comb = comb.softmax(-1) + hc_eps
    comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    return pre, post, comb


@pytest.mark.parametrize("hidden", [4096, 5120])
@pytest.mark.parametrize("tokens", [1, 6, 129, 192, 641, 1025, 2049])
def test_gluon_mhc_mixes_matches_reference(tokens: int, hidden: int) -> None:
    args = _mixes_args(tokens, hidden, seed=tokens)
    actual = mhc.launch_gluon_mhc_mixes_gfx950(*args)
    expected = _mixes_reference(*args)
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            actual_tensor.double(), expected_tensor, rtol=1e-5, atol=1e-5
        )


def test_mhc_mixes_selects_gluon_and_graph_replays() -> None:
    args = _mixes_args(32, 5120, seed=7)
    residual = args[0]
    mhc_mixes(*args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_outputs = mhc_mixes(*args)
    residual.copy_(torch.randn_like(residual))
    graph.replay()
    torch.cuda.synchronize()
    for actual, expected in zip(
        graph_outputs, mhc.launch_gluon_mhc_mixes_gfx950(*args), strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(graph_outputs, triton_mhc_mixes(*args), strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_gluon_mhc_mixes_no_recompile_across_batches() -> None:
    args = _mixes_args(8192, 5120, seed=3)
    residual = args[0]
    # Warm both block buckets and every reduction layout, then sweep batch
    # sizes inside them.
    for tokens in (1, 300, 2048, 8192):
        mhc.launch_gluon_mhc_mixes_gfx950(residual[:tokens], *args[1:])
    with assert_no_triton_compile(
        mhc.gluon_mhc_mixes_project_gfx950, mhc.gluon_mhc_mixes_reduce_gfx950
    ):
        for tokens in (2, 6, 17, 48, 129, 192, 256, 257, 641, 1024, 1025, 4097, 7937):
            mhc.launch_gluon_mhc_mixes_gfx950(residual[:tokens], *args[1:])


def _post_args(tokens: int, hidden: int, seed: int) -> tuple:
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    hidden_states = torch.randn(
        tokens, hidden, device=DEVICE, dtype=torch.bfloat16, generator=generator
    )
    residual = torch.randn(
        tokens, 4, hidden, device=DEVICE, dtype=torch.bfloat16, generator=generator
    )
    post = torch.rand(tokens, 4, 1, device=DEVICE, generator=generator) * 2
    comb = torch.rand(tokens, 4, 4, device=DEVICE, generator=generator)
    return hidden_states, residual, post, comb


def _assert_bf16_rounding_equivalent(actual, expected, reference, magnitude) -> None:
    # Both kernels round the same FP32 math; the RMS reduction order differs,
    # which moves a rare value by one BF16 ulp (plus FP32 rounding of the
    # summed terms, which dominates after cancellation). Neither may be
    # farther from the FP64 reference than that.
    bound = expected.double().abs() * 2**-7 + magnitude * 2**-20
    assert ((actual.double() - expected.double()).abs() <= bound).all()
    assert (actual != expected).double().mean().item() < 1e-3
    assert ((actual.double() - reference).abs() <= bound).all()


@pytest.mark.parametrize("tokens", [1, 6, 192, 641])
@pytest.mark.parametrize("hidden", [16, 4096, 5120, 7168])
def test_gluon_mhc_post_matches_triton_bitwise(tokens: int, hidden: int) -> None:
    hidden_states, residual, post, comb = _post_args(tokens, hidden, seed=tokens)
    torch.testing.assert_close(
        mhc_post(hidden_states, residual, post, comb),
        triton_mhc_post(hidden_states, residual, post, comb),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("tokens", [1, 6, 192, 641])
def test_gluon_mhc_apply_pre_matches_triton(tokens: int) -> None:
    _, residual, _, _ = _post_args(tokens, 5120, seed=tokens)
    pre = torch.rand(tokens, 4, device=DEVICE)
    weight = torch.rand(5120, device=DEVICE, dtype=torch.bfloat16)
    # The plain collapse rounds like the portable kernel, bit for bit.
    torch.testing.assert_close(
        mhc_apply_pre(residual, pre, norm_weight=None, norm_eps=None),
        triton_mhc_apply_pre(residual, pre, norm_weight=None, norm_eps=None),
        rtol=0,
        atol=0,
    )
    actual = mhc_apply_pre(residual, pre, norm_weight=weight, norm_eps=RMS_EPS)
    expected = triton_mhc_apply_pre(residual, pre, norm_weight=weight, norm_eps=RMS_EPS)
    stream_sum = (pre.unsqueeze(-1) * residual.float()).sum(1).bfloat16().double()
    reference = (
        stream_sum
        * torch.rsqrt(stream_sum.square().mean(-1, keepdim=True) + RMS_EPS)
        * weight.double()
    )
    _assert_bf16_rounding_equivalent(actual, expected, reference, reference.abs())


def test_gluon_mhc_streams_no_recompile_across_batches() -> None:
    hidden_states, residual, post, comb = _post_args(300, 5120, seed=1)
    pre = torch.rand(300, 4, device=DEVICE)
    weight = torch.rand(5120, device=DEVICE, dtype=torch.bfloat16)

    def run(tokens: int) -> None:
        mhc.launch_gluon_mhc_post_gfx950(
            hidden_states[:tokens], residual[:tokens], post[:tokens], comb[:tokens]
        )
        for norm_weight, norm_eps in ((None, None), (weight, RMS_EPS)):
            mhc.launch_gluon_mhc_apply_pre_gfx950(
                residual[:tokens],
                pre[:tokens],
                norm_weight=norm_weight,
                norm_eps=norm_eps,
            )

    run(1)
    with assert_no_triton_compile(
        mhc.gluon_mhc_post_gfx950, mhc.gluon_mhc_apply_pre_gfx950
    ):
        for tokens in (2, 6, 16, 48, 129, 192, 300):
            run(tokens)

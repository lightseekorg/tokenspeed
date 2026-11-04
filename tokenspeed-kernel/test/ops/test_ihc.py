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
from tokenspeed_kernel.numerics.torch.residual import (
    torch_ihc_head_mixes,
    torch_ihc_mixes,
    torch_ihc_post,
)
from tokenspeed_kernel.ops.residual import ihc_head_mixes, ihc_mixes, ihc_post
from tokenspeed_kernel.ops.residual.triton import _ihc_gates_kernel, _ihc_post_kernel
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

RMS_EPS = 1e-6
HC_EPS = 1e-6
MAGNITUDE = 2.0


def _mix_inputs(tokens, hidden, *, gate_sets, hc=4, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    residual = torch.randn(
        (tokens, hc, hidden), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    weight = torch.randn(
        (gate_sets * hc, hc * hidden),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    ) * ((hc * hidden) ** -0.5)
    scale = torch.tensor([0.7, 1.3][:gate_sets], device="cuda")
    base = torch.randn(
        gate_sets * hc, device="cuda", dtype=torch.float32, generator=generator
    )
    return residual, weight, scale, base


def _post_inputs(tokens, hc, hidden, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    hidden_states = torch.randn(
        (tokens, hidden), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    residual = torch.randn(
        (tokens, hc, hidden), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    post = 2 * torch.rand(
        (tokens, hc), device="cuda", dtype=torch.float32, generator=generator
    )
    return hidden_states, residual, post


@pytest.mark.parametrize("tokens", [1, 5, 64, 300])
@pytest.mark.parametrize("hidden", [64, 4096])
def test_ihc_mixes_match_fp64_reference(tokens, hidden):
    residual, weight, scale, base = _mix_inputs(tokens, hidden, gate_sets=2)

    pre, post = ihc_mixes(residual, weight, scale, base, RMS_EPS, HC_EPS, MAGNITUDE)
    expected_pre, expected_post = torch_ihc_mixes(
        residual, weight.double(), scale, base, RMS_EPS, HC_EPS, MAGNITUDE
    )

    assert pre.is_contiguous() and post.is_contiguous()
    torch.testing.assert_close(pre, expected_pre.float(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(post, expected_post.float(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("tokens", [1, 300])
@pytest.mark.parametrize("hidden", [64, 4096])
def test_ihc_head_mixes_match_fp64_reference(tokens, hidden):
    residual, weight, scale, base = _mix_inputs(tokens, hidden, gate_sets=1)

    pre = ihc_head_mixes(residual, weight, scale, base, RMS_EPS, HC_EPS)
    expected = torch_ihc_head_mixes(
        residual, weight.double(), scale, base, RMS_EPS, HC_EPS
    )

    torch.testing.assert_close(pre, expected.float(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("hc", [2, 4])
@pytest.mark.parametrize("hidden", [257, 4096])
def test_ihc_post_matches_reference(hc, hidden):
    hidden_states, residual, post = _post_inputs(7, hc, hidden)

    actual = ihc_post(hidden_states, residual, post)
    expected = torch_ihc_post(hidden_states, residual, post)

    # The kernel fuses the multiply-add, so the FP32 sums may round to an
    # adjacent BF16 value.
    torch.testing.assert_close(actual, expected, rtol=2**-7, atol=1e-6)


def test_ihc_runtime_token_counts_do_not_compile():
    ihc_mixes(*_mix_inputs(3, 256, gate_sets=2), RMS_EPS, HC_EPS, MAGNITUDE)
    ihc_post(*_post_inputs(3, 4, 256))
    with assert_no_triton_compile(_ihc_gates_kernel), assert_no_triton_compile(
        _ihc_post_kernel
    ):
        for tokens in [1, 2, 16, 17, 64, 300, 1024]:
            inputs = _mix_inputs(tokens, 256, gate_sets=2, seed=tokens)
            ihc_mixes(*inputs, RMS_EPS, HC_EPS, MAGNITUDE)
            ihc_post(*_post_inputs(tokens, 4, 256, seed=tokens))


def test_ihc_ops_reject_mismatched_layouts():
    residual, weight, scale, base = _mix_inputs(3, 64, gate_sets=2)
    with pytest.raises(ValueError, match=r"weight \[1\*hc"):
        ihc_head_mixes(residual, weight, scale, base, RMS_EPS, HC_EPS)
    with pytest.raises(ValueError, match="hc > 1"):
        ihc_mixes(residual[:, :1], weight, scale, base, RMS_EPS, HC_EPS, MAGNITUDE)

    hidden_states, residual, post = _post_inputs(3, 4, 64)
    with pytest.raises(ValueError, match=r"post \[T,hc\]"):
        ihc_post(hidden_states, residual, post[:, :2])

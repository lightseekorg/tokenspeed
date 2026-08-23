# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Numerical parity for the SiTU (SituGLU) activation used by Kimi-K3.

The reference below expands ``sigmoid`` explicitly and runs in float64 so it is
not structurally identical to the implementation, and is pinned with a couple
of hand-computed scalar checks.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.activation.triton import add3

from tokenspeed.runtime.layers.activation import SituAndMul


def _ref_situ(x: torch.Tensor, beta: float, linear_beta: float | None):
    d = x.shape[-1] // 2
    gate = x[..., :d].to(torch.float64)
    up = x[..., d:].to(torch.float64)
    sigmoid = 1.0 / (1.0 + torch.exp(-gate))
    gate = beta * torch.tanh(gate / beta) * sigmoid
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return gate * up


@pytest.mark.parametrize(
    "beta,linear_beta",
    [(1.0, None), (4.0, None), (4.0, 25.0), (0.5, 2.0)],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_situ_matches_reference(beta, linear_beta, dtype):
    torch.manual_seed(0)
    x = torch.randn(17, 2 * 40, dtype=dtype)
    out = SituAndMul(beta=beta, linear_beta=linear_beta)(x)

    assert out.shape == (17, 40)
    assert out.dtype == dtype

    ref = _ref_situ(x, beta, linear_beta)
    # bf16 carries ~3 significant digits, so use a relative tolerance; the
    # activation output magnitude here reaches O(10) (beta * gate * up).
    atol, rtol = (1e-2, 2e-2) if dtype == torch.bfloat16 else (1e-5, 1e-5)
    torch.testing.assert_close(out.to(torch.float64), ref, atol=atol, rtol=rtol)


def test_situ_zero_gate_is_zero():
    # gate = 0 -> tanh(0) * sigmoid(0) = 0, so the output is 0 regardless of up.
    x = torch.tensor([[0.0, 0.0, 5.0, -3.0]])  # gate=[0,0], up=[5,-3]
    out = SituAndMul(beta=1.0)(x)
    torch.testing.assert_close(out, torch.zeros_like(out))


def test_situ_saturated_gate_passes_up():
    # Large gate -> beta * tanh(gate/beta) * sigmoid(gate) ~= beta (beta=1 here);
    # with no linear_beta the up branch is untouched.
    x = torch.tensor([[30.0, 2.0]])  # gate=30, up=2
    out = SituAndMul(beta=1.0)(x)
    torch.testing.assert_close(out, torch.tensor([[2.0]]), atol=1e-3, rtol=1e-3)


def test_linear_beta_saturates_up_branch():
    # Large up with linear_beta -> linear_beta * tanh(up/linear_beta) ~= linear_beta.
    x = torch.tensor([[30.0, 1000.0]])  # gate=30 (gate_act~=1), up=1000
    out = SituAndMul(beta=1.0, linear_beta=25.0)(x)
    # gate_act ~= 1, up_clip ~= 25 -> out ~= 25.
    torch.testing.assert_close(out, torch.tensor([[25.0]]), atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_add3_supports_row_strided_inputs(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU is unavailable")
    base = torch.arange(72, dtype=torch.float32, device=device).view(3, 24)
    a = base[:, 1:9]
    b = base[:, 9:17]
    c = base[:, 16:24]

    actual = add3(a, b, c)

    torch.testing.assert_close(actual, a + b + c)
    assert actual.is_contiguous()

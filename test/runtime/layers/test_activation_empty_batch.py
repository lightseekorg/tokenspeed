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

"""A DP rank's idle forward runs the model over a 0-token batch; the gated
activations must pass it through instead of launching a kernel over an empty
grid (flashinfer's silu_and_mul fails the launch with cudaErrorInvalidValue,
killing the scheduler mid-lockstep)."""

from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.layers.activation import SiluAndMul

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="SiluAndMul kernels need CUDA"
)


def test_silu_and_mul_passes_through_empty_batch():
    layer = SiluAndMul()
    x = torch.empty(0, 512, dtype=torch.bfloat16, device="cuda")
    out = layer(x)
    assert out.shape == (0, 256)
    assert out.dtype == x.dtype
    assert out.device == x.device


def test_silu_and_mul_fp8_passes_through_empty_batch():
    layer = SiluAndMul()
    x = torch.empty(0, 512, dtype=torch.bfloat16, device="cuda")
    out, scale = layer(x, fp8_out=True)
    assert out.shape == (0, 256)
    assert out.dtype == torch.float8_e4m3fn
    # Mirrors the fused path's TMA-aligned scale collapsed to zero rows.
    assert scale.shape == (0, 256 // 128)
    assert scale.dtype == torch.float32


def test_silu_and_mul_nonempty_still_computes():
    # The guard must not change the computed path: check a 1-token batch
    # against the eager reference.
    layer = SiluAndMul()
    x = torch.randn(1, 512, dtype=torch.bfloat16, device="cuda")
    out = layer(x)
    gate, up = x.chunk(2, dim=-1)
    expected = torch.nn.functional.silu(gate) * up
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)

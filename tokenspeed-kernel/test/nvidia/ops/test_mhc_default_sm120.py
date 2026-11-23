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

"""Run in a separate process with SM120 mHC coefficient opt-in disabled."""

import os

import pytest
import torch
from tokenspeed_kernel.numerics.torch.residual import torch_mhc_mixes, torch_mhc_pre
from tokenspeed_kernel.ops.residual import mhc_mixes, mhc_pre
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 0)
    or os.environ.get("TOKENSPEED_ENABLE_SM120_MHC_MIXES") == "1",
    reason="SM120 default IEEE mHC with optional Tensor Core projection disabled",
)


@pytest.mark.parametrize("tokens", [1, 64])
@pytest.mark.parametrize("mode", ["mhc_mixes", "mhc_pre"])
def test_default_registry_executes_portable_ieee_mhc(tokens, mode, monkeypatch):
    monkeypatch.delenv("TOKENSPEED_KERNEL_OVERRIDE_RESIDUAL_MHC_MIXES", raising=False)
    monkeypatch.delenv("TOKENSPEED_KERNEL_OVERRIDE_RESIDUAL_MHC_PRE", raising=False)
    hidden = 5120
    generator = torch.Generator(device="cuda").manual_seed(tokens + 61)
    residual = torch.randn(
        tokens, 4, hidden, device="cuda", generator=generator
    ).bfloat16()
    weight = torch.randn(24, 4 * hidden, device="cuda", generator=generator) * 0.01
    scale = torch.tensor([0.7, 1.1, 0.5], device="cuda")
    base = torch.randn(24, device="cuda", generator=generator)
    roles = {"residual": dense_tensor_format(residual.dtype)}
    traits = {"buffer_offsets_fit_int32": True, "hidden_size": hidden}
    if mode == "mhc_pre":
        roles.update(
            {
                name: dense_tensor_format(torch.float32)
                for name in ["fn", "hc_scale", "hc_base"]
            }
        )
        traits.update(num_tokens=tokens, hc_mult=4, sinkhorn_iters=20)
    selected = select_kernel("residual", mode, format_signature(**roles), traits=traits)
    assert selected.name == "triton_" + mode
    args = (residual, weight, scale, base, 1e-6, 1e-6, 20)
    if mode == "mhc_mixes":
        actual = mhc_mixes(*args)
        expected = torch_mhc_mixes(
            residual, weight.double(), scale.double(), base.double(), 1e-6, 1e-6, 20
        )
    else:
        actual = mhc_pre(*args, norm_weight=None, norm_eps=None)
        expected = torch_mhc_pre(*args, norm_weight=None, norm_eps=None)
    for index, (got, wanted) in enumerate(zip(actual, expected, strict=True)):
        if mode == "mhc_pre" and index == 0:
            torch.testing.assert_close(got, wanted)
        else:
            torch.testing.assert_close(
                got.double(), wanted.double(), rtol=1e-5, atol=1e-6
            )

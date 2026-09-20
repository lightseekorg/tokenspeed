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


"""The residual-stream reference kernels: registration bands and contracts."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.residual import mhc_pre
from tokenspeed_kernel.registry import KernelRegistry, Priority, load_builtin_kernels
from tokenspeed_kernel.selection import is_ground_truth


@pytest.mark.parametrize(
    "name,band",
    [
        ("torch_attn_res_fwd", Priority.PORTABLE),
        ("torch_mhc_pre", Priority.REFERENCE),
    ],
)
def test_residual_reference_bands(name: str, band: Priority) -> None:
    load_builtin_kernels()
    spec = KernelRegistry.get().get_by_name(name)
    assert spec is not None
    assert spec.solution == "reference"
    assert spec.priority == band
    assert is_ground_truth(spec) == (band == Priority.REFERENCE)


def test_mhc_pre_reference_shapes_and_fused_norm(device: str) -> None:
    torch.manual_seed(0)
    hc_mult, hidden_size = 3, 32
    residual = torch.randn(
        2, 5, hc_mult, hidden_size, device=device, dtype=torch.bfloat16
    )
    fn = torch.randn(2 * hc_mult + hc_mult**2, hc_mult * hidden_size, device=device)
    hc_scale = torch.tensor([0.7, 1.1, 0.5], device=device)
    hc_base = torch.randn(2 * hc_mult + hc_mult**2, device=device)
    norm_weight = torch.rand(hidden_size, device=device, dtype=torch.bfloat16) + 0.5
    args = (residual, fn, hc_scale, hc_base, 1e-6, 1e-5, 3)

    layer_input, post, comb = mhc_pre(
        *args, norm_weight=None, norm_eps=None, solution="reference"
    )
    assert (
        layer_input.shape == (2, 5, hidden_size) and layer_input.dtype == residual.dtype
    )
    assert post.shape == (2, 5, hc_mult, 1) and post.dtype == torch.float32
    assert comb.shape == (2, 5, hc_mult, hc_mult) and comb.dtype == torch.float32
    # Sinkhorn leaves every combine column summing to one (up to hc_eps).
    torch.testing.assert_close(
        comb.sum(dim=-2), torch.ones_like(comb.sum(dim=-2)), rtol=1e-3, atol=1e-3
    )

    normed, post_normed, comb_normed = mhc_pre(
        *args, norm_weight=norm_weight, norm_eps=1e-6, solution="reference"
    )
    torch.testing.assert_close(
        normed, F.rms_norm(layer_input, (hidden_size,), norm_weight, 1e-6)
    )
    torch.testing.assert_close(post_normed, post)
    torch.testing.assert_close(comb_normed, comb)

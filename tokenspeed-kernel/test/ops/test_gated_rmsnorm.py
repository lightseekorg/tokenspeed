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

"""Gated RMSNorm: several rows per program against a float32 reference."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.numerics.torch.layernorm import torch_gated_rmsnorm
from tokenspeed_kernel.ops.layernorm import gated_rmsnorm
from tokenspeed_kernel.ops.layernorm.triton import GATED_RMSNORM_ROWS_PER_PROGRAM
from tokenspeed_kernel.platform import current_platform

platform = current_platform()

pytestmark = pytest.mark.skipif(
    not (platform.is_nvidia or platform.is_amd),
    reason="Triton layernorm tests require an NVIDIA or AMD GPU.",
)

ROWS_PER_PROGRAM = GATED_RMSNORM_ROWS_PER_PROGRAM


# Row counts around the per-program row block, including partial last blocks.
@pytest.mark.parametrize(
    "rows", [1, ROWS_PER_PROGRAM - 1, ROWS_PER_PROGRAM + 1, 527, 8449]
)
@pytest.mark.parametrize("width,group_size", [(128, 128), (512, 128), (1024, 1024)])
@pytest.mark.parametrize("norm_before_gate", [True, False])
@pytest.mark.parametrize("sigmoid_gate", [False, True])
def test_gated_rmsnorm_matches_reference(
    rows, width, group_size, norm_before_gate, sigmoid_gate, device
):
    torch.manual_seed(rows + width)
    x = torch.randn(rows, width, device=device, dtype=torch.bfloat16)
    z = torch.randn(rows, width, device=device, dtype=torch.bfloat16)
    weight = torch.randn(width, device=device, dtype=torch.bfloat16)
    out = gated_rmsnorm(
        x,
        weight,
        z,
        eps=1e-6,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        sigmoid_gate=sigmoid_gate,
        fp8_scale=None,
        weights_independent=True,
    )
    expected = torch_gated_rmsnorm(
        x, weight, z, 1e-6, group_size, norm_before_gate, sigmoid_gate, None, True
    )
    # Two bf16 ulps: the reference rounds once, the kernel may differ in the last bit.
    torch.testing.assert_close(out.float(), expected.float(), rtol=2**-7, atol=2**-7)


def test_gated_rmsnorm_rows_are_independent_of_their_neighbours(device):
    """A row's output must not depend on which other rows share its program."""
    torch.manual_seed(0)
    x = torch.randn(4 * ROWS_PER_PROGRAM + 3, 128, device=device, dtype=torch.bfloat16)
    z = torch.randn_like(x)
    weight = torch.randn(128, device=device, dtype=torch.bfloat16)
    kwargs = dict(
        eps=1e-6,
        group_size=None,
        norm_before_gate=True,
        sigmoid_gate=False,
        fp8_scale=None,
        weights_independent=True,
    )
    full = gated_rmsnorm(x, weight, z, **kwargs)
    for row in (0, ROWS_PER_PROGRAM - 1, ROWS_PER_PROGRAM, x.shape[0] - 1):
        alone = gated_rmsnorm(x[row : row + 1], weight, z[row : row + 1], **kwargs)
        torch.testing.assert_close(full[row : row + 1], alone, rtol=0, atol=0)


def test_one_group_weight_on_whole_rows_matches_one_row_per_group(device):
    """A ``group_size`` weight shared by every group of whole rows, with z a
    row-strided view, normalizes exactly like one row per group."""
    torch.manual_seed(0)
    tokens, heads, dim = 37, 12, 128
    x = torch.randn(tokens, heads * dim, device=device, dtype=torch.bfloat16)
    wide = torch.randn(tokens, (heads + 3) * dim, device=device, dtype=torch.bfloat16)
    z = wide[:, 3 * dim :]
    weight = torch.randn(dim, device=device, dtype=torch.bfloat16)
    kwargs = dict(
        eps=1e-6,
        group_size=dim,
        norm_before_gate=True,
        sigmoid_gate=False,
        fp8_scale=None,
        weights_independent=True,
    )
    rows = gated_rmsnorm(x, weight, z, **kwargs)
    per_group = gated_rmsnorm(
        x.view(-1, dim), weight, z.contiguous().view(-1, dim), **kwargs
    )
    torch.testing.assert_close(rows.view(-1, dim), per_group, rtol=0, atol=0)


def test_gated_rmsnorm_fp8_output_matches_quantized_bf16(device):
    """The FP8 output equals quantizing the activation-dtype output with the same scale."""
    torch.manual_seed(1)
    x = torch.randn(531, 128, device=device, dtype=torch.bfloat16)
    z = torch.randn_like(x)
    weight = torch.randn(128, device=device, dtype=torch.bfloat16)
    scale = torch.tensor([0.05], device=device, dtype=torch.float32)
    kwargs = dict(
        eps=1e-6,
        group_size=None,
        norm_before_gate=True,
        sigmoid_gate=False,
        weights_independent=True,
    )
    bf16 = gated_rmsnorm(x, weight, z, fp8_scale=None, **kwargs)
    fp8 = gated_rmsnorm(x, weight, z, fp8_scale=scale, **kwargs)
    expected = (bf16.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    torch.testing.assert_close(fp8.float(), expected.float(), rtol=0, atol=0)

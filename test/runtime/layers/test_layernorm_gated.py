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

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=20, suite="runtime-1gpu")

from tokenspeed.runtime.layers.attention.linear.layernorm_gated import (  # noqa: E402
    ROWS_PER_PROGRAM,
    rmsnorm_fn,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _reference(x, weight, z, eps, group_size, norm_before_gate, sigmoid_gate):
    x, z = x.float(), z.float()
    gate = torch.sigmoid(z) if sigmoid_gate else F.silu(z)
    if not norm_before_gate:
        x = x * gate
    groups = x.view(*x.shape[:-1], -1, group_size)
    rstd = torch.rsqrt(groups.square().mean(-1, keepdim=True) + eps)
    y = (groups * rstd).view_as(x) * weight.float()
    return y * gate if norm_before_gate else y


# Row counts around the per-program row block, including partial last blocks.
@pytest.mark.parametrize(
    "rows", [1, ROWS_PER_PROGRAM - 1, ROWS_PER_PROGRAM + 1, 527, 8449]
)
@pytest.mark.parametrize("width,group_size", [(128, 128), (512, 128), (1024, 1024)])
@pytest.mark.parametrize("norm_before_gate", [True, False])
@pytest.mark.parametrize("sigmoid_gate", [False, True])
def test_gated_rmsnorm_matches_reference(
    rows, width, group_size, norm_before_gate, sigmoid_gate
):
    torch.manual_seed(rows + width)
    x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
    z = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(width, device="cuda", dtype=torch.bfloat16)
    out = rmsnorm_fn(
        x,
        weight,
        z=z,
        eps=1e-6,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        sigmoid_gate=sigmoid_gate,
        weights_independent=True,
    )
    expected = _reference(
        x, weight, z, 1e-6, group_size, norm_before_gate, sigmoid_gate
    )
    # Two bf16 ulps: the reference rounds once, the kernel may differ in the last bit.
    torch.testing.assert_close(out.float(), expected, rtol=2**-7, atol=2**-7)


def test_gated_rmsnorm_rows_are_independent_of_their_neighbours():
    """A row's output must not depend on which other rows share its program."""
    torch.manual_seed(0)
    x = torch.randn(4 * ROWS_PER_PROGRAM + 3, 128, device="cuda", dtype=torch.bfloat16)
    z = torch.randn_like(x)
    weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    kwargs = dict(eps=1e-6, norm_before_gate=True, weights_independent=True)
    full = rmsnorm_fn(x, weight, z=z, **kwargs)
    for row in (0, ROWS_PER_PROGRAM - 1, ROWS_PER_PROGRAM, x.shape[0] - 1):
        alone = rmsnorm_fn(x[row : row + 1], weight, z=z[row : row + 1], **kwargs)
        torch.testing.assert_close(full[row : row + 1], alone, rtol=0, atol=0)


def test_gated_rmsnorm_fp8_output_matches_quantized_bf16():
    """The FP8 output equals quantizing the activation-dtype output with the same scale."""
    torch.manual_seed(1)
    x = torch.randn(531, 128, device="cuda", dtype=torch.bfloat16)
    z = torch.randn_like(x)
    weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor([0.05], device="cuda", dtype=torch.float32)
    kwargs = dict(eps=1e-6, norm_before_gate=True, weights_independent=True)
    bf16 = rmsnorm_fn(x, weight, z=z, **kwargs)
    fp8 = rmsnorm_fn(x, weight, z=z, fp8_scale=scale, **kwargs)
    expected = (bf16.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    torch.testing.assert_close(fp8.float(), expected.float(), rtol=0, atol=0)

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

"""RMSNorm's per-token FP8 output (``rmsnorm_quantize_fp8``).

One launch writes the BF16 output, bit for bit that of the Triton RMSNorm
without the FP8 output, and its per-token FP8 E4M3 codes and FP32 scales, bit
for bit those of ``fp8_quantize_per_token`` of that output; with the residual
and its BF16 rounding, with and without the x and residual scales. Over many
rows, where summing a row's squares in another order would change some output
bits, also without a residual and with the FP32 residual sum.
"""

import pytest
import torch
from tokenspeed_kernel.ops.layernorm.triton import rmsnorm
from tokenspeed_kernel.ops.quantization import rmsnorm_quantize_fp8
from tokenspeed_kernel.ops.quantization.triton import fp8_quantize_per_token
from tokenspeed_kernel.platform import pdl_enabled
from utils import is_nvidia

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_nvidia(), reason="NVIDIA GPU test"
)

FP8 = torch.float8_e4m3fn


def _rows(rows, hidden, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, hidden, device="cuda", generator=generator)
    x[:, :: max(1, hidden // 7)] *= 20
    return x.to(torch.bfloat16)


@pytest.mark.parametrize("rows", [1, 7, 64])
@pytest.mark.parametrize("scaled", [False, True])
def test_rmsnorm_writes_the_per_token_fp8_of_its_output(rows, scaled):
    hidden = 4096
    x = _rows(rows, hidden, seed=12)
    residual = _rows(rows, hidden, seed=13)
    weight = (1 + 0.1 * torch.randn(hidden, device="cuda")).to(torch.bfloat16)
    kwargs = dict(residual=residual, round_residual_sum_bf16=True)
    if scaled:
        kwargs.update(x_scale=0.75, residual_scale=1.25)
    out, residual_out = rmsnorm(x, weight, 1e-6, enable_pdl=None, **kwargs)
    fused_out, fused_residual, codes, scales = rmsnorm_quantize_fp8(
        x, weight, 1e-6, enable_pdl=pdl_enabled(), **kwargs
    )
    assert torch.equal(out, fused_out) and torch.equal(residual_out, fused_residual)
    assert codes.dtype == FP8 and tuple(codes.shape) == (rows, hidden)
    expected_codes, expected_scales = fp8_quantize_per_token(out)
    assert torch.equal(scales, expected_scales)
    assert torch.equal(codes.view(torch.uint8), expected_codes.view(torch.uint8))


@pytest.mark.parametrize("hidden", [2048, 4096, 7168])
@pytest.mark.parametrize("residual", ["none", "fp32_sum", "bf16_sum", "scaled"])
def test_token_fp8_out_keeps_every_output_bit(hidden, residual):
    """2,048 rows: enough that a launch summing a row's squares in another
    order than the call without token_fp8_out rounds some output elements to
    the other BF16 neighbour."""
    rows = 2048
    x = _rows(rows, hidden, seed=16)
    weight = (1 + 0.1 * torch.randn(hidden, device="cuda")).to(torch.bfloat16)
    kwargs = dict(round_residual_sum_bf16=residual in ("bf16_sum", "scaled"))
    if residual != "none":
        kwargs["residual"] = _rows(rows, hidden, seed=17)
    if residual == "scaled":
        kwargs.update(x_scale=0.75, residual_scale=1.25)
    plain = rmsnorm(x, weight, 1e-6, enable_pdl=None, **kwargs)
    *fused, codes, scales = rmsnorm_quantize_fp8(
        x, weight, 1e-6, enable_pdl=pdl_enabled(), **kwargs
    )
    if residual == "none":
        plain, fused = (plain,), fused[:1]
    for want, got in zip(plain, fused, strict=True):
        differ = (want.view(torch.int16) != got.view(torch.int16)).sum().item()
        assert differ == 0, f"{differ} elements differ from the plain call"
    expected_codes, expected_scales = fp8_quantize_per_token(plain[0])
    assert torch.equal(scales, expected_scales)
    assert torch.equal(codes.view(torch.uint8), expected_codes.view(torch.uint8))

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

"""GemmaRMSNorm on the GPU: the checkpoint offset scales as ``1 + weight``."""

import pytest
import torch

from tokenspeed.runtime.layers.layernorm import GemmaRMSNorm

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GemmaRMSNorm kernels need a GPU"
)


@pytest.mark.parametrize("rows", [1, 33])
def test_gemma_rmsnorm_module_adds_and_normalizes(rows):
    hidden, eps = 5120, 1e-6
    layer = GemmaRMSNorm(hidden, eps=eps).to("cuda", torch.bfloat16)
    offset = (torch.randn(hidden) * 0.5).to("cuda", torch.bfloat16)
    layer.weight.weight_loader(layer.weight, offset)
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    total = x.float() + residual.float()

    def reference(rows_fp32):
        rms = torch.rsqrt(rows_fp32.pow(2).mean(-1, keepdim=True) + eps)
        return rows_fp32 * rms * (1.0 + offset.float())

    expected_first = reference(x.float())
    first = layer(x.clone())
    out, new_residual = layer(x.clone(), residual.clone())

    torch.testing.assert_close(first.float(), expected_first, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(out.float(), reference(total), atol=1e-2, rtol=1e-2)
    assert torch.equal(new_residual, total.to(torch.bfloat16))

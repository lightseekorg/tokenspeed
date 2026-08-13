# Copyright (c) 2024-2026 TokenSpeed Contributors
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
"""``rmsnorm_gated_sigmoid`` with strided gates.

The KDA output path hands the kernel a gate that is a column slice of the
fused projection buffer (row stride > width). The kernel must read the gate
through its row stride; a stride bug would silently pull the neighboring
projection columns into the sigmoid, not crash.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from tokenspeed_kernel.ops.activation.triton import (  # noqa: E402
    rmsnorm_gated_sigmoid,
)

DEV = "cuda"


def _reference(x, gate, weight, eps, num_heads, head_dim):
    xf = x.float().view(-1, num_heads, head_dim)
    var = (xf * xf).mean(-1, keepdim=True)
    y = xf * torch.rsqrt(var + eps) * weight.float()
    y = y * torch.sigmoid(gate.float().reshape(-1, num_heads, head_dim))
    return y.view(x.shape).to(x.dtype)


@pytest.mark.parametrize(
    "tokens,num_heads,head_dim,buf_width",
    [
        (7, 12, 128, 6284),  # K3 TP8 geometry, odd padded projection width
        (1, 12, 128, 6284),  # single token: degenerate row count
        (513, 4, 128, 777),  # odd width, non-block-aligned columns
        (64, 16, 128, 16 * 128 + 1),  # slice one past its own width
    ],
)
def test_strided_gate_matches_contiguous_and_reference(
    tokens, num_heads, head_dim, buf_width
):
    torch.manual_seed(tokens)
    hidden = num_heads * head_dim
    buf = torch.randn(tokens, buf_width, device=DEV, dtype=torch.bfloat16)
    gate = buf[:, buf_width - hidden :]
    x = torch.randn(tokens, hidden, device=DEV, dtype=torch.bfloat16)
    weight = torch.randn(head_dim, device=DEV, dtype=torch.bfloat16)

    out = rmsnorm_gated_sigmoid(x, gate, weight, 1e-6, num_heads, head_dim)
    ref = _reference(x, gate, weight, 1e-6, num_heads, head_dim)
    torch.testing.assert_close(out.float(), ref.float(), atol=3e-2, rtol=3e-2)
    # The strided read must be exactly the packed read, not merely close.
    out_packed = rmsnorm_gated_sigmoid(
        x, gate.contiguous(), weight, 1e-6, num_heads, head_dim
    )
    assert torch.equal(out, out_packed)

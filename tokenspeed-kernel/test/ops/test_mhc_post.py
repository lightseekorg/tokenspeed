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

import pytest
import torch
from tokenspeed_kernel.ops.residual.triton import triton_mhc_post

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("hidden", [257, 5120])
def test_hc4_post_preserves_fma_rounding(hidden):
    # This cancellation crosses a BF16 rounding boundary if the compiler
    # splits or rearranges the FP32 multiply-add chain. Include a partial
    # block and later blocks so all positions obey the same contract.
    x = torch.full((1, hidden), 5.875, dtype=torch.bfloat16, device="cuda")
    residual = (
        torch.tensor(
            [0.2353515625, -1.9140625, -0.89453125, -1.2734375],
            dtype=torch.bfloat16,
            device="cuda",
        )
        .view(1, 4, 1)
        .expand(1, 4, hidden)
        .contiguous()
    )
    post = torch.zeros((1, 4), dtype=torch.float32, device="cuda")
    post[0, 2] = 0.5795707106590271
    comb = torch.zeros((1, 4, 4), dtype=torch.float32, device="cuda")
    comb[0, :, 2] = torch.tensor(
        [
            0.048366911709308624,
            0.5316871404647827,
            0.8416003584861755,
            0.06851339340209961,
        ],
        dtype=torch.float32,
        device="cuda",
    )
    actual = triton_mhc_post(x, residual, post, comb)
    expected = torch.zeros_like(residual)
    expected[:, 2, :] = 1.5625
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

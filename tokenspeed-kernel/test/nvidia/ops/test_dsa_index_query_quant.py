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

"""Parity of ``quantize_index_query_weights`` with the NVIDIA quantizer it fuses.

The DSA indexer's NVIDIA scoring paths used the 1x128 ``quantize_fp8``
followed by ``combine_topk_weights``; the fused kernel reproduces both bit for
bit. Other platforms' ``quantize_fp8`` derives the scale differently.
"""

import pytest
import torch
from tokenspeed_kernel.ops.attention.dsa.triton import (
    combine_topk_weights,
    quantize_index_query_weights,
)
from tokenspeed_kernel.ops.quantization import quantize_fp8
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform().is_nvidia,
    reason="pins parity with the NVIDIA FP8 token-group quantizer",
)


@pytest.mark.parametrize("weights_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tokens,heads", [(1, 32), (8, 32), (37, 64), (513, 32)])
def test_quantize_index_query_weights_matches_quantizer_then_combine(
    weights_dtype, tokens, heads
):
    torch.manual_seed(tokens + heads)
    softmax_scale = 128**-0.5 * heads**-0.5
    q = torch.randn(tokens, heads, 128, device="cuda", dtype=torch.bfloat16) * 3
    q[0, 0] = 0  # an all-zero group takes its scale from the amax floor
    q[-1, -1, 5] = 2.0e4  # a group whose amax dwarfs the rest
    fused = torch.randn(tokens, 128 + heads, device="cuda", dtype=weights_dtype)
    weights = fused.split([128, heads], dim=-1)[1]

    ref_q, ref_scale = quantize_fp8(
        q.view(-1, 128),
        granularity="token_group",
        group_size=128,
        scale_encoding="float32",
    )
    ref_w = combine_topk_weights(weights, ref_scale, softmax_scale)
    got_q, got_w = quantize_index_query_weights(q, weights, softmax_scale)

    assert got_q.shape == (tokens, heads, 128) and got_q.is_contiguous()
    assert torch.equal(got_q.view(torch.uint8), ref_q.view_as(q).view(torch.uint8))
    assert torch.equal(got_w, ref_w)

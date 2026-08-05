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
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and current_platform().is_nvidia),
    reason="NVIDIA CUDA required",
)


class TestRoutingFlash:
    """CUDA fused softmax, top-k, correction bias, and zero-expert masking."""

    NUM_EXPERTS = 384
    NUM_REAL_EXPERTS = 256
    TOPK = 12
    SCALE = 6.0

    def _make_inputs(self, num_tokens=16, seed=42):
        torch.manual_seed(seed)
        inp = torch.randn(
            num_tokens, self.NUM_EXPERTS, device="cuda", dtype=torch.float32
        )
        bias = torch.randn(self.NUM_EXPERTS, device="cuda", dtype=torch.float32)
        idx = torch.empty(num_tokens, self.TOPK, device="cuda", dtype=torch.int32)
        wts = torch.empty(num_tokens, self.TOPK, device="cuda", dtype=torch.float32)
        return inp, bias, idx, wts

    def _torch_ref(self, inp, bias):
        scores = inp.softmax(dim=-1)
        topk_idx = torch.topk(
            scores + bias.unsqueeze(0), k=self.TOPK, dim=-1, sorted=True
        )[1]
        topk_wts = scores.gather(1, topk_idx)
        topk_idx[topk_idx >= self.NUM_REAL_EXPERTS] = -1
        topk_wts *= self.SCALE
        return topk_idx.to(torch.int32), topk_wts

    def test_basic(self):
        from tokenspeed_kernel.ops.other.moe_routing.cuda import routing_flash

        inp, bias, idx, wts = self._make_inputs()
        routing_flash(inp, bias, idx, wts, self.NUM_REAL_EXPERTS, self.SCALE, False)
        assert idx.shape == (16, self.TOPK)
        assert wts.shape == (16, self.TOPK)

    def test_correctness(self):
        from tokenspeed_kernel.ops.other.moe_routing.cuda import routing_flash

        inp, bias, idx, wts = self._make_inputs()
        ref_idx, ref_wts = self._torch_ref(inp.clone(), bias)
        routing_flash(inp, bias, idx, wts, self.NUM_REAL_EXPERTS, self.SCALE, False)
        torch.testing.assert_close(idx, ref_idx)
        torch.testing.assert_close(wts, ref_wts, rtol=1e-3, atol=8e-2)

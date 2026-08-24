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

"""Sqrt-softplus top-k routing, against the tensor-op chain it replaces.

The reference is the chain in ``deepseek_v4_select_experts``: scores are
``sqrt(softplus(logits))``, the correction bias steers selection only, and the
weights gathered for the winners come from the unbiased score. Expert ids are
compared exactly -- a routing kernel that picks a different expert is wrong
however close its weights are -- so the tie-break has to match
``torch.topk(sorted=True)`` as well as the ordering.
"""

from __future__ import annotations

import os
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenspeed_kernel.ops.moe.triton.deepseek_v4_softplus_sqrt_topk import (
    deepseek_v4_softplus_sqrt_topk,
)

TINY = torch.finfo(torch.float32).tiny


def _reference(logits, top_k, renormalize, correction_bias):
    scores = torch.sqrt(F.softplus(logits.float()))
    choice = scores
    if correction_bias is not None:
        choice = choice + correction_bias.to(
            device=scores.device, dtype=scores.dtype
        ).unsqueeze(0)
    ids = torch.topk(choice, k=top_k, dim=-1, sorted=True)[1]
    weights = scores.gather(1, ids)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(TINY)
    return weights.to(torch.float32), ids.to(torch.int32), scores


class TestDeepseekV4RouterTopK(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("GPU is required")
        self.device = torch.device("cuda")

    def _check(self, num_tokens, num_experts, top_k, renormalize, bias, seed=0):
        torch.manual_seed(seed)
        logits = torch.randn(
            num_tokens, num_experts, device=self.device, dtype=torch.float32
        )
        correction = (
            torch.randn(num_experts, device=self.device, dtype=torch.float32) * 0.1
            if bias
            else None
        )
        weights, ids, scores = deepseek_v4_softplus_sqrt_topk(
            logits,
            top_k,
            renormalize,
            correction_bias=correction,
            return_scores=True,
        )
        ref_w, ref_i, ref_s = _reference(logits, top_k, renormalize, correction)

        self.assertTrue(
            torch.equal(ids, ref_i),
            f"expert ids differ: {ids[:2].tolist()} vs {ref_i[:2].tolist()}",
        )
        self.assertLess((weights - ref_w).abs().max().item(), 1e-5)
        self.assertLess((scores - ref_s).abs().max().item(), 1e-5)

    def test_v4_flash_shape(self) -> None:
        """256 experts, 6 per token, renormalized -- what the model runs."""
        for tokens in (1, 2, 8, 32, 64, 160):
            with self.subTest(tokens=tokens):
                self._check(tokens, 256, 6, True, bias=True)

    def test_without_correction_bias(self) -> None:
        self._check(32, 256, 6, True, bias=False)

    def test_without_renormalize(self) -> None:
        self._check(32, 256, 6, False, bias=True)

    def test_other_widths(self) -> None:
        """Expert counts that are not powers of two exercise the padding."""
        for num_experts, top_k in ((384, 6), (128, 4), (192, 8), (64, 1)):
            with self.subTest(num_experts=num_experts, top_k=top_k):
                self._check(16, num_experts, top_k, True, bias=True)

    def test_large_logits_take_the_identity_branch(self) -> None:
        """softplus is the identity above 20; both sides must switch together."""
        torch.manual_seed(3)
        logits = torch.randn(8, 256, device=self.device, dtype=torch.float32) * 30.0
        bias = torch.randn(256, device=self.device, dtype=torch.float32) * 0.1
        weights, ids, scores = deepseek_v4_softplus_sqrt_topk(
            logits, 6, True, correction_bias=bias, return_scores=True
        )
        ref_w, ref_i, ref_s = _reference(logits, 6, True, bias)
        self.assertTrue(torch.equal(ids, ref_i))
        self.assertLess((weights - ref_w).abs().max().item(), 1e-5)
        self.assertLess(
            ((scores - ref_s).abs() / ref_s.clamp_min(1e-6)).max().item(), 1e-5
        )

    def test_tie_break_prefers_the_lower_expert(self) -> None:
        """Equal scores resolve to the lower expert id, deterministically.

        ``torch.topk`` leaves tie order unspecified and in fact returns an
        arbitrary permutation here, so this asserts the kernel's own stronger
        contract rather than agreement with the reference.
        """
        logits = torch.zeros(4, 256, device=self.device, dtype=torch.float32)
        _, ids, _ = deepseek_v4_softplus_sqrt_topk(logits, 6, True, return_scores=True)
        expected = torch.arange(6, device=self.device, dtype=torch.int32)
        for row in ids:
            self.assertTrue(torch.equal(row, expected), f"got {row.tolist()}")

    def test_randomized(self) -> None:
        for seed in range(8):
            with self.subTest(seed=seed):
                self._check(
                    int(torch.randint(1, 48, (1,)).item()), 256, 6, True, True, seed
                )


if __name__ == "__main__":
    unittest.main()

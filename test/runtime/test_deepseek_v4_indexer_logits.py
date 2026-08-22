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

"""DeepSeek V4 sparse indexer MQA logits.

The reference scores each indexer head against the compressed KV, rectifies,
and only then takes the weighted sum over heads::

    index_score = torch.einsum("bshd,btd->bsht", q, kv_cache)
    index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)

The ReLU sits between the per-head scores and the reduction, so it cannot be
folded into the weights -- and it is the detail most likely to be dropped when
reimplementing from a kernel signature. The oracle below keeps that ordering
explicit, and one test fails if the rectification is skipped.
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenspeed_kernel.ops.attention.triton.deepseek_v4 import (
    deepseek_v4_indexer_mqa_logits,
)

INDEX_HEAD_DIM = 128


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    cu_start: torch.Tensor,
    cu_end: torch.Tensor,
    max_len: int,
) -> torch.Tensor:
    num_tokens = q.shape[0]
    out = torch.zeros(num_tokens, max_len, dtype=torch.float32, device=q.device)
    qf = q.to(torch.float32)
    kf = k.to(torch.float32)
    for t in range(num_tokens):
        start = int(cu_start[t])
        end = int(cu_end[t])
        n = min(end - start, max_len)
        if n <= 0:
            continue
        rows = kf[start : start + n]                     # [n, d]
        scores = (qf[t] @ rows.T).relu()                 # [h, n]
        weighted = (scores * weights[t].float()[:, None]).sum(0)
        out[t, :n] = weighted * k_scale[start : start + n].float()
    return out


class TestDeepseekV4IndexerLogits(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("GPU is required")
        self.device = torch.device("cuda")
        torch.manual_seed(0)

    def _make(self, num_tokens: int, num_heads: int, total_keys: int):
        q = torch.randn(
            num_tokens,
            num_heads,
            INDEX_HEAD_DIM,
            device=self.device,
            dtype=torch.bfloat16,
        )
        k = torch.randn(
            total_keys, INDEX_HEAD_DIM, device=self.device, dtype=torch.bfloat16
        )
        k_scale = (
            torch.rand(total_keys, device=self.device, dtype=torch.float32) * 0.5
            + 0.25
        )
        weights = torch.randn(
            num_tokens, num_heads, device=self.device, dtype=torch.float32
        )
        return q, k, k_scale, weights

    def _check(self, q, k, k_scale, weights, cu_start, cu_end, max_len) -> None:
        got = deepseek_v4_indexer_mqa_logits(
            q, k, k_scale, weights, cu_start, cu_end, max_len
        )
        expected = _reference(
            q, k, k_scale, weights, cu_start, cu_end, max_len
        )
        peak = expected.abs().max().item()
        max_abs = (got - expected).abs().max().item()
        self.assertLessEqual(
            max_abs, 3e-2 * peak + 1e-2, f"max_abs={max_abs} peak={peak}"
        )

    def test_uniform_ranges(self) -> None:
        num_tokens, num_heads, total = 16, 8, 512
        q, k, k_scale, weights = self._make(num_tokens, num_heads, total)
        cu_start = torch.zeros(
            num_tokens, device=self.device, dtype=torch.int32
        )
        cu_end = torch.full(
            (num_tokens,), 256, device=self.device, dtype=torch.int32
        )
        self._check(q, k, k_scale, weights, cu_start, cu_end, 256)

    def test_ragged_causal_ranges(self) -> None:
        """Each token sees a different, growing key range."""
        num_tokens, num_heads, total = 32, 8, 1024
        q, k, k_scale, weights = self._make(num_tokens, num_heads, total)
        cu_start = torch.zeros(
            num_tokens, device=self.device, dtype=torch.int32
        )
        cu_end = (
            torch.arange(1, num_tokens + 1, device=self.device, dtype=torch.int32)
            * 8
        )
        self._check(q, k, k_scale, weights, cu_start, cu_end, int(cu_end.max()))

    def test_offset_ranges(self) -> None:
        """Non-zero cu_start must offset into the gathered key buffer."""
        num_tokens, num_heads, total = 8, 8, 1024
        q, k, k_scale, weights = self._make(num_tokens, num_heads, total)
        cu_start = torch.full(
            (num_tokens,), 128, device=self.device, dtype=torch.int32
        )
        cu_end = torch.full(
            (num_tokens,), 128 + 192, device=self.device, dtype=torch.int32
        )
        self._check(q, k, k_scale, weights, cu_start, cu_end, 192)

    def test_relu_is_applied_per_head(self) -> None:
        """Dropping the rectification must change the result.

        With negative per-head scores present, a non-rectified weighted sum
        differs materially -- this test is what catches a missing relu_().
        """
        num_tokens, num_heads, total = 8, 8, 256
        q, k, k_scale, weights = self._make(num_tokens, num_heads, total)
        cu_start = torch.zeros(num_tokens, device=self.device, dtype=torch.int32)
        cu_end = torch.full(
            (num_tokens,), 128, device=self.device, dtype=torch.int32
        )

        got = deepseek_v4_indexer_mqa_logits(
            q, k, k_scale, weights, cu_start, cu_end, 128
        )

        # Same computation without the relu.
        qf, kf = q.to(torch.float32), k.to(torch.float32)
        no_relu = torch.zeros_like(got)
        for t in range(num_tokens):
            scores = qf[t] @ kf[:128].T
            no_relu[t, :128] = (scores * weights[t][:, None]).sum(0) * k_scale[
                :128
            ]

        self.assertGreater(
            (got - no_relu).abs().max().item(),
            0.1 * got.abs().max().item(),
            "relu appears not to be applied",
        )
        # And the rectified oracle agrees.
        self._check(q, k, k_scale, weights, cu_start, cu_end, 128)

    def test_head_padding_masks_extra_rows(self) -> None:
        """Head counts below the dot minimum must not leak padded rows."""
        for num_heads in (1, 4, 8):
            with self.subTest(num_heads=num_heads):
                q, k, k_scale, weights = self._make(8, num_heads, 256)
                cu_start = torch.zeros(8, device=self.device, dtype=torch.int32)
                cu_end = torch.full(
                    (8,), 128, device=self.device, dtype=torch.int32
                )
                self._check(q, k, k_scale, weights, cu_start, cu_end, 128)

    def test_empty_range_leaves_zeros(self) -> None:
        num_tokens, num_heads, total = 4, 8, 256
        q, k, k_scale, weights = self._make(num_tokens, num_heads, total)
        cu_start = torch.zeros(num_tokens, device=self.device, dtype=torch.int32)
        cu_end = torch.zeros(num_tokens, device=self.device, dtype=torch.int32)
        got = deepseek_v4_indexer_mqa_logits(
            q, k, k_scale, weights, cu_start, cu_end, 64
        )
        self.assertEqual(float(got.abs().max().item()), 0.0)


if __name__ == "__main__":
    unittest.main()

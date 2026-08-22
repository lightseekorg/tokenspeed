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

"""DeepSeek V4 sparse latent attention with a learned per-head sink.

The reference gathers top-k latent KV rows, runs an online softmax, and then
adds ``exp(attn_sink[h] - running_max[h])`` to the denominator. The sink has no
value row, so it damps the output without contributing to it -- a detail that a
test comparing only softmax *weights* would miss entirely, which is why the
oracle here is written as a dense masked softmax with an explicit extra
denominator term.
"""

from __future__ import annotations

import math
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenspeed_kernel.ops.attention.triton.deepseek_v4 import (
    deepseek_v4_sparse_attn,
)

HEAD_DIM = 512


def _reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    lens: torch.Tensor,
    attn_sink: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Dense masked softmax with a denominator-only sink term."""
    num_tokens, num_heads, head_dim = q.shape
    topk = indices.shape[-1]
    out = torch.zeros(
        num_tokens, num_heads, head_dim, dtype=torch.float32, device=q.device
    )
    qf = q.to(torch.float32)
    kvf = kv.to(torch.float32)
    for t in range(num_tokens):
        valid = (torch.arange(topk, device=q.device) < lens[t]) & (indices[t] >= 0)
        if not bool(valid.any()):
            continue
        rows = kvf[indices[t][valid].long()]  # [v, d]
        scores = (qf[t] @ rows.T) * scale  # [h, v]
        row_max = scores.max(dim=-1, keepdim=True).values
        exp_scores = torch.exp(scores - row_max)
        denom = exp_scores.sum(dim=-1) + torch.exp(
            attn_sink[:num_heads].to(torch.float32) - row_max[:, 0]
        )
        out[t] = (exp_scores @ rows) / denom[:, None]
    return out


class TestDeepseekV4SparseAttn(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("GPU is required")
        self.device = torch.device("cuda")
        torch.manual_seed(0)
        self.scale = 1.0 / math.sqrt(HEAD_DIM)

    def _make(self, num_tokens: int, num_heads: int, num_rows: int, topk: int):
        q = torch.randn(
            num_tokens,
            num_heads,
            HEAD_DIM,
            device=self.device,
            dtype=torch.bfloat16,
        )
        kv = torch.randn(num_rows, HEAD_DIM, device=self.device, dtype=torch.bfloat16)
        indices = torch.randint(
            0, num_rows, (num_tokens, topk), device=self.device, dtype=torch.int32
        )
        lens = torch.full((num_tokens,), topk, device=self.device, dtype=torch.int32)
        attn_sink = torch.randn(num_heads, device=self.device, dtype=torch.float32)
        return q, kv, indices, lens, attn_sink

    def _check(self, q, kv, indices, lens, attn_sink) -> None:
        got = deepseek_v4_sparse_attn(q, kv, indices, lens, attn_sink, self.scale)
        expected = _reference(q, kv, indices, lens, attn_sink, self.scale)
        peak = expected.abs().max().item()
        max_abs = (got.to(torch.float32) - expected).abs().max().item()
        self.assertLessEqual(
            max_abs, 3e-2 * peak + 1e-2, f"max_abs={max_abs} peak={peak}"
        )

    def test_full_topk(self) -> None:
        self._check(*self._make(num_tokens=8, num_heads=16, num_rows=512, topk=128))

    def test_topk_not_multiple_of_block(self) -> None:
        self._check(*self._make(num_tokens=8, num_heads=16, num_rows=512, topk=100))

    def test_ragged_lengths(self) -> None:
        """Per-token topk_length must truncate the gathered rows."""
        q, kv, indices, lens, sink = self._make(
            num_tokens=16, num_heads=16, num_rows=512, topk=128
        )
        lens = torch.randint(1, 129, (16,), device=self.device, dtype=torch.int32)
        self._check(q, kv, indices, lens, sink)

    def test_negative_indices_are_masked(self) -> None:
        q, kv, indices, lens, sink = self._make(
            num_tokens=8, num_heads=16, num_rows=512, topk=64
        )
        indices[:, ::3] = -1
        self._check(q, kv, indices, lens, sink)

    def test_sink_changes_the_output(self) -> None:
        """A large sink must visibly damp the output; a tiny one must not.

        Guards against the sink being silently dropped -- the failure mode a
        weights-only comparison cannot see.
        """
        q, kv, indices, lens, _ = self._make(
            num_tokens=4, num_heads=16, num_rows=256, topk=64
        )
        small = torch.full((16,), -30.0, device=self.device, dtype=torch.float32)
        large = torch.full((16,), 30.0, device=self.device, dtype=torch.float32)

        out_small = deepseek_v4_sparse_attn(q, kv, indices, lens, small, self.scale).to(
            torch.float32
        )
        out_large = deepseek_v4_sparse_attn(q, kv, indices, lens, large, self.scale).to(
            torch.float32
        )

        # A dominant sink drives the normalized output toward zero.
        self.assertLess(
            out_large.abs().max().item(), 0.2 * out_small.abs().max().item()
        )
        # And the negligible-sink case matches a plain softmax.
        plain = _reference(q, kv, indices, lens, small, self.scale)
        self.assertLessEqual(
            (out_small - plain).abs().max().item(),
            3e-2 * plain.abs().max().item() + 1e-2,
        )

    def test_per_head_sink_is_not_broadcast(self) -> None:
        """Each head must use its own sink value."""
        q, kv, indices, lens, _ = self._make(
            num_tokens=4, num_heads=16, num_rows=256, topk=64
        )
        sink = torch.full((16,), -30.0, device=self.device, dtype=torch.float32)
        sink[0] = 30.0  # only head 0 is damped

        got = deepseek_v4_sparse_attn(q, kv, indices, lens, sink, self.scale).to(
            torch.float32
        )
        head0 = got[:, 0].abs().max().item()
        others = got[:, 1:].abs().max().item()
        self.assertLess(head0, 0.2 * others)
        self._check(q, kv, indices, lens, sink)

    def test_single_token_decode_shape(self) -> None:
        self._check(*self._make(num_tokens=1, num_heads=16, num_rows=256, topk=64))

    def test_prefill_scale_batch(self) -> None:
        """A chunk-sized batch at the model's real topk of 512."""
        self._check(*self._make(num_tokens=256, num_heads=16, num_rows=8192, topk=512))


if __name__ == "__main__":
    unittest.main()

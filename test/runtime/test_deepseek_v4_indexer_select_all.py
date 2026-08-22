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

"""Indexer selection when the top-k is at least as wide as the key range.

A row with no more candidates than the top-k width keeps all of them, so the
scores cannot change the answer. These tests pin that shortcut against the
dense masked ``torch.topk`` it replaces: same set of keys per row, same packing
of the -1 padding. Order inside a row deliberately is not compared -- the
selected indices address cache rows that attention reduces under a softmax, so
the set is the contract.
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenspeed_kernel.ops.attention.triton.deepseek_v4 import (
    deepseek_v4_indexer_select_all,
)


def _dense_reference(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    topk_tokens: int,
    row_starts: torch.Tensor | None,
) -> torch.Tensor:
    """The masked dense top-k the fast path stands in for."""
    num_tokens, max_len = logits.shape
    out = torch.full(
        (num_tokens, topk_tokens), -1, dtype=torch.int32, device=logits.device
    )
    positions = torch.arange(max_len, device=logits.device).unsqueeze(0)
    lo = (
        row_starts.reshape(-1, 1)
        if row_starts is not None
        else torch.zeros_like(lengths).reshape(-1, 1)
    )
    hi = lo + lengths.reshape(-1, 1)
    valid = (positions >= lo) & (positions < hi)
    masked = logits.masked_fill(~valid, float("-inf"))
    selected = min(topk_tokens, max_len)
    values, indices = torch.topk(masked, selected, dim=-1)
    indices = torch.where(torch.isfinite(values), indices, torch.full_like(indices, -1))
    out[:, :selected] = indices.to(torch.int32)
    return out


class TestDeepseekV4IndexerSelectAll(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("GPU is required")
        self.device = torch.device("cuda")
        torch.manual_seed(0)

    def _assert_same_rows(self, fast: torch.Tensor, ref: torch.Tensor) -> None:
        self.assertEqual(fast.shape, ref.shape)
        self.assertEqual(fast.dtype, ref.dtype)
        for row_fast, row_ref in zip(fast.cpu().tolist(), ref.cpu().tolist()):
            keep_fast = [i for i in row_fast if i >= 0]
            keep_ref = [i for i in row_ref if i >= 0]
            self.assertEqual(sorted(keep_fast), sorted(keep_ref))
            # The consumer reports a length, not a mask, so the survivors have
            # to stay packed at the front of the row.
            self.assertEqual(row_fast[: len(keep_fast)], keep_fast)
            self.assertTrue(all(i == -1 for i in row_fast[len(keep_fast) :]))

    def test_matches_dense_topk_for_ragged_rows(self) -> None:
        num_tokens, max_len, topk_tokens = 6, 48, 64
        logits = torch.randn(
            num_tokens, max_len, device=self.device, dtype=torch.float32
        )
        lengths = torch.randint(
            0, max_len + 1, (num_tokens,), device=self.device, dtype=torch.int32
        )
        out = torch.full(
            (num_tokens, topk_tokens), -1, dtype=torch.int32, device=self.device
        )

        fast = deepseek_v4_indexer_select_all(out, lengths)
        ref = _dense_reference(logits, lengths, topk_tokens, None)
        self._assert_same_rows(fast, ref)

    def test_matches_dense_topk_with_row_offsets(self) -> None:
        """Prefill rows start partway into the key range."""
        num_tokens, max_len, topk_tokens = 5, 32, 32
        logits = torch.randn(
            num_tokens, max_len, device=self.device, dtype=torch.float32
        )
        row_starts = torch.randint(
            0, max_len // 2, (num_tokens,), device=self.device, dtype=torch.int32
        )
        lengths = torch.randint(
            0,
            max_len // 2 + 1,
            (num_tokens,),
            device=self.device,
            dtype=torch.int32,
        )
        out = torch.full(
            (num_tokens, topk_tokens), -1, dtype=torch.int32, device=self.device
        )

        fast = deepseek_v4_indexer_select_all(out, lengths, row_starts=row_starts)
        ref = _dense_reference(logits, lengths, topk_tokens, row_starts)
        self._assert_same_rows(fast, ref)

    def test_full_and_empty_rows(self) -> None:
        num_tokens, topk_tokens = 4, 16
        lengths = torch.tensor(
            [0, 1, topk_tokens, topk_tokens], device=self.device, dtype=torch.int32
        )
        out = torch.full(
            (num_tokens, topk_tokens), 7, dtype=torch.int32, device=self.device
        )

        fast = deepseek_v4_indexer_select_all(out, lengths).cpu().tolist()
        self.assertEqual(fast[0], [-1] * topk_tokens)
        self.assertEqual(fast[1], [0] + [-1] * (topk_tokens - 1))
        self.assertEqual(fast[2], list(range(topk_tokens)))

    def test_width_not_a_power_of_two(self) -> None:
        """The engine's top-k width need not be a tile multiple."""
        num_tokens, topk_tokens = 3, 513
        lengths = torch.tensor([0, 300, 513], device=self.device, dtype=torch.int32)
        out = torch.full(
            (num_tokens, topk_tokens), -1, dtype=torch.int32, device=self.device
        )

        fast = deepseek_v4_indexer_select_all(out, lengths).cpu()
        self.assertTrue(torch.all(fast[0] == -1))
        self.assertEqual(fast[1, :300].tolist(), list(range(300)))
        self.assertTrue(torch.all(fast[1, 300:] == -1))
        self.assertEqual(fast[2].tolist(), list(range(513)))


if __name__ == "__main__":
    unittest.main()

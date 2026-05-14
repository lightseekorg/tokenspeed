"""Unit tests for the n-gram (prompt-lookup) speculative drafter.

Covers two layers, both written against the dep-free
``runtime/execution/drafter/ngram_lookup`` module so they can run
without torch / tokenspeed_kernel:

1. The pure-numpy KMP suffix-ngram lookup
   ``find_longest_matched_ngram_and_propose_tokens``.
2. ``propose_batch_into``, the batched in-place row builder that the
   ``NgramDrafter`` wrapper delegates to.
"""

import os
import sys
import unittest

import numpy as np

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.execution.drafter.ngram_lookup import (  # noqa: E402
    find_longest_matched_ngram_and_propose_tokens,
    propose_batch_into,
)


class TestFindLongestMatchedNgram(unittest.TestCase):
    """Pure-function tests for the KMP suffix-ngram lookup."""

    def _propose(self, tokens, *, min_n=1, max_n=3, k=2):
        return find_longest_matched_ngram_and_propose_tokens(
            np.asarray(tokens, dtype=np.int32),
            min_ngram=min_n,
            max_ngram=max_n,
            k=k,
        ).tolist()

    def test_returns_empty_when_context_shorter_than_min(self):
        self.assertEqual(self._propose([], min_n=1, k=3), [])
        self.assertEqual(self._propose([7], min_n=2, k=3), [])

    def test_returns_empty_when_k_is_nonpositive(self):
        self.assertEqual(self._propose([1, 2, 1, 2], k=0), [])

    def test_returns_empty_when_no_ngram_repeats(self):
        self.assertEqual(self._propose([10, 11, 12, 13], min_n=1, k=2), [])

    def test_picks_continuation_after_rightmost_match(self):
        # Sequence repeats "1, 2": the most recent match suffix is the
        # final "1, 2"; drafts come from right after the earlier "1, 2".
        tokens = [1, 2, 9, 1, 2]
        drafts = self._propose(tokens, min_n=2, max_n=2, k=2)
        self.assertEqual(drafts, [9, 1])

    def test_prefers_longest_match_within_bounds(self):
        # Two competing matches: "B C" (len 2) and "A B C" (len 3). With
        # max_ngram >= 3 the longer match wins.
        tokens = [5, 5, 5, 0, 1, 2, 9, 9, 0, 1, 2]
        drafts = self._propose(tokens, min_n=1, max_n=3, k=2)
        self.assertEqual(drafts, [9, 9])

    def test_caps_at_max_ngram(self):
        # Suffix "0 1 2" *would* match as a len-3 ngram. With max_ngram=2
        # we force the shorter match and the draft start position moves.
        tokens = [0, 1, 2, 9, 0, 1, 2]
        drafts = self._propose(tokens, min_n=1, max_n=2, k=2)
        self.assertEqual(drafts, [9, 0])

    def test_returns_fewer_than_k_when_context_exhausted(self):
        # The earlier "1 2" sits at indices 0..1; tokens 2..4 follow it,
        # so with k=10 we still only get those 3 tokens.
        tokens = [1, 2, 9, 1, 2]
        drafts = self._propose(tokens, min_n=2, max_n=2, k=10)
        self.assertEqual(drafts, [9, 1, 2])


class TestProposeBatchInto(unittest.TestCase):
    """Verify the row layout written by the batched proposer."""

    def _new_batch(self, *, max_bs=4, max_context_len=128, spec_num_steps=3):
        history = np.zeros((max_bs, max_context_len), dtype=np.int32)
        history_len = np.zeros((max_bs,), dtype=np.int32)
        out = np.zeros((max_bs, spec_num_steps + 1), dtype=np.int32)
        return history, history_len, out

    def _seed(self, history, history_len, slot, tokens):
        tokens = np.asarray(tokens, dtype=np.int32)
        history[slot, : tokens.size] = tokens
        history_len[slot] = tokens.size

    def test_layout_for_matching_history(self):
        history, history_len, out = self._new_batch(spec_num_steps=3)
        self._seed(history, history_len, slot=0, tokens=[1, 2, 3, 1, 2])
        self._seed(history, history_len, slot=1, tokens=[9, 8, 7])

        pool_indices = np.array([0, 1], dtype=np.int32)
        propose_batch_into(
            history=history,
            history_len=history_len,
            pool_indices=pool_indices,
            out=out[: pool_indices.size],
            min_ngram=1,
            max_ngram=3,
            spec_num_steps=3,
        )

        row0 = out[0].tolist()
        row1 = out[1].tolist()
        # First column is always last_verified (= last history token).
        self.assertEqual(row0[0], 2)
        self.assertEqual(row1[0], 7)
        # Slot 0: KMP picks the rightmost "1 2"; continuation starts with 3.
        self.assertEqual(row0[1], 3)
        # Slot 1: no repeat. Draft columns fall back to ``last_verified``
        # to preserve the fixed verify width without a no-match mask.
        self.assertEqual(row1[1:], [7, 7, 7])

    def test_pads_remaining_columns_with_last_verified(self):
        history, history_len, out = self._new_batch(spec_num_steps=4)
        self._seed(history, history_len, slot=0, tokens=[4, 5, 6, 4, 5])

        pool_indices = np.array([0], dtype=np.int32)
        propose_batch_into(
            history=history,
            history_len=history_len,
            pool_indices=pool_indices,
            out=out[: pool_indices.size],
            min_ngram=1,
            max_ngram=3,
            spec_num_steps=4,
        )

        # Match suffix "4 5"; continuation = [6, 4, 5]. Trailing column
        # padded with last_verified=5.
        self.assertEqual(out[0].tolist(), [5, 6, 4, 5, 5])

    def test_zero_history_row_emits_zeros(self):
        history, history_len, out = self._new_batch(spec_num_steps=2)
        pool_indices = np.array([0], dtype=np.int32)
        propose_batch_into(
            history=history,
            history_len=history_len,
            pool_indices=pool_indices,
            out=out[: pool_indices.size],
            min_ngram=1,
            max_ngram=3,
            spec_num_steps=2,
        )
        self.assertEqual(out[0].tolist(), [0, 0, 0])

    def test_rejects_wrong_out_shape(self):
        history, history_len, _ = self._new_batch(spec_num_steps=3)
        pool_indices = np.array([0], dtype=np.int32)
        bad_out = np.zeros((1, 5), dtype=np.int32)  # spec_num_steps+1 should be 4
        with self.assertRaises(ValueError):
            propose_batch_into(
                history=history,
                history_len=history_len,
                pool_indices=pool_indices,
                out=bad_out,
                min_ngram=1,
                max_ngram=3,
                spec_num_steps=3,
            )

    def test_rejects_wrong_out_batch_size(self):
        history, history_len, _ = self._new_batch(spec_num_steps=3)
        pool_indices = np.array([0, 1], dtype=np.int32)
        bad_out = np.zeros((1, 4), dtype=np.int32)
        with self.assertRaises(ValueError):
            propose_batch_into(
                history=history,
                history_len=history_len,
                pool_indices=pool_indices,
                out=bad_out,
                min_ngram=1,
                max_ngram=3,
                spec_num_steps=3,
            )


if __name__ == "__main__":
    unittest.main()

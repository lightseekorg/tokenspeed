from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

import torch

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.execution.model_executor import ModelExecutor


def _stub(page_ratio: int, page_size: int, columns: int, rows: int = 8):
    return SimpleNamespace(
        drafter=object(),
        _full_history_group_id="full_attention",
        input_buffers=SimpleNamespace(max_bs=rows),
        device="cpu",
        draft_page_table=torch.zeros((rows, columns), dtype=torch.int32),
        _draft_page_ratio=page_ratio,
        _draft_page_size=page_size,
        _logical_page_size=page_size * page_ratio,
    )


def _publish(ex, pool_indices, table):
    ModelExecutor._publish_draft_page_table(
        ex,
        SimpleNamespace(request_pool_indices=pool_indices),
        {"full_attention": table},
    )


class DraftPageTableUnitsTest(unittest.TestCase):
    """draft_page_table holds draft-page ids; publishing expands logical ones.

    The allocator hands out logical pages (Kimi-K3 LCM: 128 tokens) while the
    draft's MLA decode kernel indexes the pool in its own smaller pages. An
    unexpanded id makes the kernel read rows the draft's KV writes never
    touched, which reads back as zeros and collapses EAGLE3 draft logits.
    """

    def test_expands_logical_ids_into_draft_pages(self):
        ex = _stub(page_ratio=2, page_size=64, columns=8)
        _publish(ex, [0], torch.tensor([[3, 5, -1]], dtype=torch.int32))
        expected = torch.tensor([6, 7, 10, 11, 0, 1, 0, 0], dtype=torch.int32)
        self.assertTrue(torch.equal(ex.draft_page_table[0], expected))

    def test_expanded_ids_address_the_same_tokens(self):
        # Draft page d covers tokens [d*64, d*64+64); logical page L covers
        # [L*128, L*128+128). The expansion must preserve the token span.
        ex = _stub(page_ratio=2, page_size=64, columns=4)
        _publish(ex, [0], torch.tensor([[3]], dtype=torch.int32))
        first = int(ex.draft_page_table[0, 0])
        self.assertEqual(first * 64, 3 * 128)

    def test_identity_when_units_match(self):
        ex = _stub(page_ratio=1, page_size=128, columns=4)
        _publish(ex, [0], torch.tensor([[7, 9, -1]], dtype=torch.int32))
        expected = torch.tensor([7, 9, 0, 0], dtype=torch.int32)
        self.assertTrue(torch.equal(ex.draft_page_table[0], expected))

    def test_expansion_truncates_to_table_width(self):
        ex = _stub(page_ratio=2, page_size=64, columns=3)
        _publish(ex, [0], torch.tensor([[3, 5]], dtype=torch.int32))
        expected = torch.tensor([6, 7, 10], dtype=torch.int32)
        self.assertTrue(torch.equal(ex.draft_page_table[0], expected))

    def test_multi_row_batch_ordered(self):
        ex = _stub(page_ratio=2, page_size=64, columns=4)
        _publish(ex, [5, 2], torch.tensor([[3, -1], [7, -1]], dtype=torch.int32))
        self.assertTrue(
            torch.equal(
                ex.draft_page_table[:2],
                torch.tensor([[6, 7, 0, 1], [14, 15, 0, 1]], dtype=torch.int32),
            )
        )


if __name__ == "__main__":
    unittest.main()

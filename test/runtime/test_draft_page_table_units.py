from __future__ import annotations

import os
import sys
import unittest

import torch

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.execution.draft_page_staging import DraftPageStaging


def _staging(page_ratio: int, page_size: int, columns: int, rows: int = 8):
    return DraftPageStaging(
        max_bs=rows,
        max_pages_per_req=columns,
        table_page_size=page_size * page_ratio,
        draft_page_size=page_size,
        full_history_group_id="full_attention",
        enabled=True,
        device="cpu",
    )


def _publish(staging, bs, table):
    staging.publish({"full_attention": table}, bs=bs, padded_bs=staging.table.shape[0])


class DraftPageTableUnitsTest(unittest.TestCase):
    """The staged table holds draft-page ids; publishing expands logical ones.

    The allocator hands out logical pages (Kimi-K3 LCM: 128 tokens) while the
    draft's MLA decode kernel indexes the pool in its own smaller pages. An
    unexpanded id makes the kernel read rows the draft's KV writes never
    touched, which reads back as zeros and collapses EAGLE3 draft logits.
    """

    def test_expands_logical_ids_into_draft_pages(self):
        st = _staging(page_ratio=2, page_size=64, columns=8)
        _publish(st, 1, torch.tensor([[3, 5, -1]], dtype=torch.int32))
        expected = torch.tensor([6, 7, 10, 11, 0, 1, 0, 0], dtype=torch.int32)
        self.assertTrue(torch.equal(st.table[0], expected))

    def test_expanded_ids_address_the_same_tokens(self):
        # Draft page d covers tokens [d*64, d*64+64); logical page L covers
        # [L*128, L*128+128). The expansion must preserve the token span.
        st = _staging(page_ratio=2, page_size=64, columns=4)
        _publish(st, 1, torch.tensor([[3]], dtype=torch.int32))
        first = int(st.table[0, 0])
        self.assertEqual(first * 64, 3 * 128)

    def test_identity_when_units_match(self):
        st = _staging(page_ratio=1, page_size=128, columns=4)
        _publish(st, 1, torch.tensor([[7, 9, -1]], dtype=torch.int32))
        expected = torch.tensor([7, 9, 0, 0], dtype=torch.int32)
        self.assertTrue(torch.equal(st.table[0], expected))

    def test_expansion_truncates_to_table_width(self):
        st = _staging(page_ratio=2, page_size=64, columns=3)
        _publish(st, 1, torch.tensor([[3, 5]], dtype=torch.int32))
        expected = torch.tensor([6, 7, 10], dtype=torch.int32)
        self.assertTrue(torch.equal(st.table[0], expected))

    def test_multi_row_batch_ordered(self):
        st = _staging(page_ratio=2, page_size=64, columns=4)
        _publish(st, 2, torch.tensor([[3, -1], [7, -1]], dtype=torch.int32))
        self.assertTrue(
            torch.equal(
                st.table[:2],
                torch.tensor([[6, 7, 0, 1], [14, 15, 0, 1]], dtype=torch.int32),
            )
        )

    def test_publish_clears_rows_past_the_live_batch(self):
        st = _staging(page_ratio=2, page_size=64, columns=4, rows=4)
        st.table.fill_(99)

        _publish(st, 1, torch.tensor([[3, -1]], dtype=torch.int32))

        self.assertTrue(
            torch.equal(st.table[1:], torch.zeros((3, 4), dtype=torch.int32))
        )

    def test_rejects_misaligned_page_sizes(self):
        with self.assertRaises(ValueError):
            DraftPageStaging(
                max_bs=4,
                max_pages_per_req=4,
                table_page_size=100,
                draft_page_size=64,
                full_history_group_id="full_attention",
                enabled=True,
                device="cpu",
            )


if __name__ == "__main__":
    unittest.main()

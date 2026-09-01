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


def _staging(page_size: int = 128, columns: int = 4, rows: int = 8):
    return DraftPageStaging(
        max_bs=rows,
        max_pages_per_req=columns,
        block_granularity=page_size,
        full_history_group_id="full_attention",
        device="cpu",
    )


def _publish(staging, bs, table):
    staging.publish({"full_attention": table}, bs=bs, padded_bs=staging.table.shape[0])


class DraftPageTableUnitsTest(unittest.TestCase):
    """The staged table holds RAW scheduler page ids; publish is a pure copy.

    Kernel-page expansion happens inside each backend (the same
    ``_expand_history_table`` a wrapper-delivered group table goes through).
    The write-location math is page-size invariant — ``table[i, pos // P] * P
    + pos % P`` addresses the same token for any page size — so the staging
    resolves absolute slots directly over the raw table.
    """

    def test_publish_copies_raw_ids(self):
        st = _staging()
        _publish(st, 1, torch.tensor([[3, 5, -1]], dtype=torch.int32))
        # -1 holes clamp into the null page 0; the tail zero-fills.
        expected = torch.tensor([3, 5, 0, 0], dtype=torch.int32)
        self.assertTrue(torch.equal(st.table[0], expected))

    @unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
    def test_staged_slots_match_the_logical_span(self):
        # Absolute slot of position p in logical page L is L*P + p%P; the
        # staging's uniform resolver must produce exactly that.
        st = DraftPageStaging(
            max_bs=8,
            max_pages_per_req=4,
            block_granularity=128,
            full_history_group_id="full_attention",
            device="cuda",
        )
        _publish(st, 1, torch.tensor([[3]], dtype=torch.int32, device="cuda"))
        out = torch.zeros(1, dtype=torch.int64, device="cuda")
        st.out_cache_loc_uniform(
            out=out,
            cache_start=torch.tensor([5], dtype=torch.int32, device="cuda"),
            num_tokens=1,
        )
        self.assertEqual(int(out[0]), 3 * 128 + 5)

    def test_copy_truncates_to_table_width(self):
        st = _staging(columns=3)
        _publish(st, 1, torch.tensor([[3, 5, 7, 9]], dtype=torch.int32))
        expected = torch.tensor([3, 5, 7], dtype=torch.int32)
        self.assertTrue(torch.equal(st.table[0], expected))

    def test_multi_row_batch_ordered(self):
        st = _staging()
        _publish(st, 2, torch.tensor([[3, -1], [7, -1]], dtype=torch.int32))
        self.assertTrue(
            torch.equal(
                st.table[:2],
                torch.tensor([[3, 0, 0, 0], [7, 0, 0, 0]], dtype=torch.int32),
            )
        )

    def test_publish_clears_rows_past_the_live_batch(self):
        st = _staging(columns=4, rows=4)
        st.table.fill_(99)

        _publish(st, 1, torch.tensor([[3, -1]], dtype=torch.int32))

        self.assertTrue(
            torch.equal(st.table[1:], torch.zeros((3, 4), dtype=torch.int32))
        )

    def test_staging_reports_raw_page_capacity(self):
        st = _staging(page_size=128, columns=4)
        self.assertEqual(st.max_tokens, 4 * 128)
        self.assertEqual(st.block_granularity, 128)


if __name__ == "__main__":
    unittest.main()

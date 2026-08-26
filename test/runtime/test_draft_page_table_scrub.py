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

from tokenspeed.runtime.execution.draft_page_staging import DraftPageStaging
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.model_executor import ModelExecutor
from tokenspeed.runtime.execution.types import DpForwardMetadata


def _staging(rows: int = 8, columns: int = 4, page_ratio: int = 1):
    page_size = 128 // page_ratio
    return DraftPageStaging(
        max_bs=rows,
        max_pages_per_req=columns,
        block_granularity=128,
        draft_kernel_page_size=page_size,
        full_history_group_id="full_attention",
        enabled=True,
        device="cpu",
    )


def _publish(staging, bs, table):
    staging.publish({"full_attention": table}, bs=bs, padded_bs=staging.table.shape[0])


class DraftPageTableScrubTest(unittest.TestCase):
    """Rows past the current batch must never keep a prior batch's page ids.

    CUDA-graph replay reads padded_bs rows straight off this table (row i IS
    batch position i, so the req-pool sink row does not shield it). A stale
    id left by an earlier, larger batch routes the multi-step draft's KV
    writes into another request's pages; the victim then mispredicts
    permanently (#955: M3 EAGLE3 accept 0.665 -> 0.0015).
    """

    def test_smaller_batch_clears_prior_rows(self):
        st = _staging()
        _publish(
            st,
            4,
            torch.tensor([[7, 8], [9, 10], [11, 12], [13, 14]], dtype=torch.int32),
        )
        # Re-publish with a smaller batch: rows [1:] must be inert.
        _publish(st, 1, torch.tensor([[3, 4]], dtype=torch.int32))
        self.assertTrue(torch.equal(st.table[1:], torch.zeros_like(st.table[1:])))

    def test_smaller_batch_keeps_active_rows(self):
        st = _staging()
        _publish(st, 2, torch.tensor([[7, 8], [9, 10]], dtype=torch.int32))
        _publish(st, 1, torch.tensor([[3, 4]], dtype=torch.int32))
        expected = torch.tensor([3, 4, 0, 0], dtype=torch.int32)
        self.assertTrue(torch.equal(st.table[0], expected))

    def test_expanded_path_also_scrubs(self):
        st = _staging(page_ratio=2)
        _publish(st, 2, torch.tensor([[7], [9]], dtype=torch.int32))
        _publish(st, 1, torch.tensor([[3]], dtype=torch.int32))
        self.assertTrue(torch.equal(st.table[1:], torch.zeros_like(st.table[1:])))

    def test_scrub_only_publish_clears_padded_rows(self):
        # The idle path publishes with no tables: scrub must still run.
        st = _staging()
        st.table[:6] = 7
        st.publish(None, bs=0, padded_bs=4)
        self.assertTrue(torch.equal(st.table[:4], torch.zeros_like(st.table[:4])))
        # Rows past padded_bs are untouched (not read by this replay).
        self.assertTrue(bool((st.table[4:6] == 7).all()))

    def test_disabled_staging_still_scrubs(self):
        # cache_group_tables_replace_draft_page_table backends skip the copy
        # but the placeholder must stay inert for its other consumers.
        st = DraftPageStaging(
            max_bs=8,
            max_pages_per_req=4,
            block_granularity=128,
            draft_kernel_page_size=128,
            full_history_group_id="full_attention",
            enabled=False,
            device="cpu",
        )
        st.table[:] = 7
        st.publish(
            {"full_attention": torch.tensor([[3]], dtype=torch.int32)},
            bs=1,
            padded_bs=8,
        )
        self.assertTrue(torch.equal(st.table[1:], torch.zeros_like(st.table[1:])))


class IdleReplayScrubTest(unittest.TestCase):
    """The idle replay path skips the batch publish, so it must scrub on its
    own.

    A DP rank that served a large batch and then goes idle replays the
    captured drafter graph at padded_bs while another rank decodes; without
    the idle-path scrub the stale rows route idle draft KV writes into
    live pages (codex P2 on #955).
    """

    def test_idle_replay_sees_zeroed_rows(self):
        padded_bs = 4
        rows, columns = 8, 4
        captured = {}
        staging = _staging(rows=rows, columns=columns)

        class _Step:
            def can_run(self, bs, ctx):
                return True

            def padded_bs(self, bs, ctx):
                return padded_bs

            def __call__(self, bs, ctx, sampling_info, page_table):
                captured["rows"] = page_table[:padded_bs].clone()

        ex = SimpleNamespace(
            attn_backend=None,
            token_to_kv_pool=None,
            input_buffers=SimpleNamespace(
                req_pool_indices_buf=torch.zeros(rows, dtype=torch.int64),
                fill_dummy_decode_buffers=lambda batch_size, total_tokens: None,
            ),
            runtime_states=SimpleNamespace(
                valid_cache_lengths=torch.zeros(rows, dtype=torch.int32),
                vocab_size=32,
            ),
            device="cpu",
            config=SimpleNamespace(output_length=1),
            capturable_grammar=None,
            _draft_staging=staging,
            draft_page_table=staging.table,
            forward_step=_Step(),
        )
        # Simulate a prior larger batch leaving real ids behind.
        staging.table[:6] = 7
        ModelExecutor.execute_idle_forward(
            ex,
            DpForwardMetadata(
                global_num_tokens=[0],
                global_batch_size=[0],
                global_forward_mode=[int(ForwardMode.IDLE)],
                all_decode_or_idle=True,
                all_extend=False,
                need_idle_forward=True,
            ),
        )
        self.assertIn("rows", captured)
        self.assertTrue(
            torch.equal(captured["rows"], torch.zeros_like(captured["rows"]))
        )


if __name__ == "__main__":
    unittest.main()

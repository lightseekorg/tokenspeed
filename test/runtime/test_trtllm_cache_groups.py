"""TRTLLM MHA leaf metadata under the unified decode path.

The trtllm backend is now a pure ``PagedAttentionBackend`` leaf: the router
owns group routing and write locations, and the leaf owns only its
kernel-facing metadata. What is left to pin here is trtllm-specific: the
two metadata slots (decode vs verify) and which one a refresh publishes,
the verify slot's clamped ``spec_cache_seqlens_buf`` (padded rows replay at
seq_len 1 and would hit an empty causal span -> NaN), pointer stability of
the per-bs views, and the DFLASH block-decode row expansion. Group-routing
coverage lives in test_cache_group_router.py.
"""

from __future__ import annotations

import os
import sys
import unittest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, suite="runtime-1gpu")


def _import_backend():
    from tokenspeed.runtime.layers.attention.backends.paged.trtllm import (
        TRTLLMMHAAttnBackend,
    )

    return TRTLLMMHAAttnBackend


class TRTLLMLeafMetadataTest(unittest.TestCase):
    def setUp(self):
        try:
            self.Backend = _import_backend()
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        import torch

        from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

        self.torch = torch
        self.ForwardMode = ForwardMode

    def _leaf(
        self,
        *,
        page_size=64,
        max_num_pages=8,
        spec_num_tokens=1,
        is_draft=False,
        draft_block_decode=False,
        max_bs=4,
    ):
        # __new__ + attribute shell (workspace pool and kernels stay out);
        # init_cuda_graph_state is the real allocator, so the buffers and
        # per-bs view caches are exactly production's.
        b = self.Backend.__new__(self.Backend)
        b.kernel_page_size = page_size
        b.max_num_pages = max_num_pages
        b.max_context_len = page_size * max_num_pages
        b.device = "cpu"
        b.spec_num_tokens = spec_num_tokens
        b.is_draft = is_draft
        b.draft_block_decode = draft_block_decode
        b.forward_prefill_metadata = None
        b.forward_decode_metadata = None
        b.init_cuda_graph_state(max_bs)
        return b

    def test_plain_decode_publishes_the_single_token_slot(self):
        torch = self.torch
        b = self._leaf()
        seq_lens = torch.tensor([65, 3], dtype=torch.int32)
        page_table = torch.tensor(
            [[11, 12, 0, 0, 0, 0, 0, 0], [13, 0, 0, 0, 0, 0, 0, 0]],
            dtype=torch.int32,
        )
        b.refresh_decode_metadata(2, 2, seq_lens, page_table)
        meta = b.forward_decode_metadata
        self.assertEqual(meta.cache_seqlens_int32.tolist(), [65, 3])
        self.assertEqual(meta.max_seq_len_q, 1)
        self.assertEqual(meta.cu_seqlens_q.tolist(), [0, 1, 2])
        # The metadata views the persistent buffers the graph records.
        self.assertEqual(meta.page_table.data_ptr(), b.page_table_buf.data_ptr())
        self.assertEqual(meta.cache_seqlens_int32.data_ptr(), b.seq_lens_buf.data_ptr())
        # Without speculation the verify slot stays untouched.
        self.assertIsNone(b.forward_prefill_metadata)

    def test_verify_slot_clamps_padded_rows_to_spec_floor(self):
        torch = self.torch
        b = self._leaf(spec_num_tokens=4)
        seq_lens = torch.tensor([65, 1], dtype=torch.int32)  # row 1 is padding
        page_table = torch.zeros((2, 8), dtype=torch.int32)
        b.refresh_decode_metadata(2, 1, seq_lens, page_table, for_graph_replay=True)
        # The decode slot keeps the raw lengths...
        self.assertEqual(
            b.forward_decode_metadata.cache_seqlens_int32.tolist(), [65, 1]
        )
        # ...while the verify slot reads the clamped buffer: q_len=4 against
        # seq_len 1 would be an empty causal span (NaN), so >= spec.
        meta = b.forward_prefill_metadata
        self.assertEqual(
            meta.cache_seqlens_int32.data_ptr(), b.spec_cache_seqlens_buf.data_ptr()
        )
        self.assertEqual(meta.cache_seqlens_int32.tolist(), [65, 4])
        self.assertEqual(meta.max_seq_len_q, 4)
        self.assertEqual(meta.cu_seqlens_q.tolist(), [0, 4, 8])
        self.assertEqual(meta.page_table.data_ptr(), b.page_table_buf.data_ptr())

    def test_per_bs_views_are_pointer_stable_across_refreshes(self):
        torch = self.torch
        b = self._leaf(spec_num_tokens=4)
        seq_lens = torch.tensor([65, 3], dtype=torch.int32)
        page_table = torch.zeros((2, 8), dtype=torch.int32)
        b.refresh_decode_metadata(2, 2, seq_lens, page_table)
        decode_first = b.forward_decode_metadata
        verify_first = b.forward_prefill_metadata
        b.refresh_decode_metadata(
            2,
            1,
            torch.tensor([66, 1], dtype=torch.int32),
            page_table,
            for_graph_replay=True,
        )
        # Same objects, same tensors — a captured graph replays through them.
        self.assertIs(b.forward_decode_metadata, decode_first)
        self.assertIs(b.forward_prefill_metadata, verify_first)
        self.assertEqual(decode_first.cache_seqlens_int32.tolist(), [66, 1])
        self.assertEqual(verify_first.cache_seqlens_int32.tolist(), [66, 4])
        # The capture seeding is the same idle refresh over the same views.
        b.init_forward_metadata_capture_cuda_graph(
            2, torch.ones(2, dtype=torch.int32), page_table
        )
        self.assertIs(b.forward_decode_metadata, decode_first)

    def test_block_decode_expands_rows_and_broadcasts_block_seq_lens(self):
        torch = self.torch
        spec = 4
        b = self._leaf(
            spec_num_tokens=spec, is_draft=True, draft_block_decode=True, max_bs=2
        )
        self.assertTrue(b.block_decode_active)
        # init_cuda_graph_state sized the buffers for max_bs * spec rows.
        self.assertEqual(b.page_table_buf.shape[0], 2 * spec)
        page_table = torch.tensor(
            [[11, 12, 0, 0, 0, 0, 0, 0], [13, 0, 0, 0, 0, 0, 0, 0]],
            dtype=torch.int32,
        )
        block_ends = torch.tensor([66, 2], dtype=torch.int32)
        b.refresh_decode_metadata(2, 2, block_ends, page_table)
        meta = b.forward_decode_metadata
        # Each request's table replicates to its spec rows...
        self.assertEqual(meta.page_table.shape[0], 2 * spec)
        self.assertTrue((meta.page_table[:spec] == page_table[0]).all())
        self.assertTrue((meta.page_table[spec:] == page_table[1]).all())
        # ...sharing the block-end seq_len (clamped to >= spec), one query
        # row each (the non-causal block trick).
        self.assertEqual(meta.cache_seqlens_int32.tolist(), [66] * spec + [4] * spec)
        self.assertEqual(meta.max_seq_len_q, 1)
        self.assertEqual(meta.cu_seqlens_q.tolist(), list(range(2 * spec + 1)))
        # The drafter's in-graph rewrite lands in the same buffer.
        b.fill_block_decode_seq_lens(2, torch.tensor([70, 8], dtype=torch.int32))
        self.assertEqual(meta.cache_seqlens_int32.tolist(), [70] * spec + [8] * spec)

    def test_extend_metadata_reads_the_router_expanded_table(self):
        torch = self.torch
        b = self._leaf()
        seq_lens = torch.tensor([66], dtype=torch.int32)
        page_table = torch.tensor([[5, 6, 0, 0, 0, 0, 0, 0]], dtype=torch.int32)
        b.init_forward_metadata(
            1,
            1,
            seq_lens,
            page_table,
            self.ForwardMode.EXTEND,
            extend_seq_lens=torch.tensor([2], dtype=torch.int32),
            extend_seq_lens_cpu=torch.tensor([2], dtype=torch.int32),
            extend_prefix_lens=torch.tensor([64], dtype=torch.int32),
            extend_prefix_lens_cpu=torch.tensor([64], dtype=torch.int32),
            extend_with_prefix=True,
        )
        meta = b.forward_prefill_metadata
        self.assertEqual(meta.page_table.tolist(), page_table.tolist())
        self.assertEqual(meta.cache_seqlens_int32.tolist(), [66])
        self.assertEqual(meta.cu_seqlens_k.tolist(), [0, 66])
        self.assertEqual(meta.cu_seqlens_q.tolist(), [0, 2])
        self.assertEqual(meta.max_seq_len_q, 2)

    def test_decode_mode_init_is_a_contract_violation(self):
        torch = self.torch
        b = self._leaf()
        no_extends = torch.zeros(0, dtype=torch.int32)
        with self.assertRaisesRegex(RuntimeError, "refresh_decode_metadata"):
            b.init_forward_metadata(
                1,
                0,
                torch.ones(1, dtype=torch.int32),
                torch.zeros((1, 8), dtype=torch.int32),
                self.ForwardMode.DECODE,
                extend_seq_lens=no_extends,
                extend_seq_lens_cpu=no_extends,
                extend_prefix_lens=no_extends,
                extend_prefix_lens_cpu=no_extends,
                extend_with_prefix=False,
            )


if __name__ == "__main__":
    unittest.main()

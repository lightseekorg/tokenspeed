"""Shape regression test for FlashAttention spec-decode page_table buffers.

The topk > 1 (EAGLE) page_table buffers must use ``max_num_pages`` as the
column dimension (page-indexed), not ``max_context_len`` (token-indexed),
matching the non-topk ``decode_cuda_graph_metadata`` buffer's convention.

A regression here silently over-allocates VRAM by a factor of ``page_size``
on the EAGLE configs (~126 MiB at max_bs=128, ctx=128K, page_size=64).
"""

import os
import sys
import unittest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

import torch

from tokenspeed.runtime.layers.attention.backends.flash_attention import (
    FlashAttentionBackend,
)


class TestSpecDecodePageTableShape(unittest.TestCase):
    """Topk page_table buffers share the page-indexed shape contract with the
    non-topk decode buffer in ``init_cuda_graph_state``."""

    def test_topk_page_table_matches_non_topk_shape(self):
        max_bs = 32
        max_context_len = 8192
        page_size = 64
        expected_pages = (max_context_len + page_size - 1) // page_size

        backend = object.__new__(FlashAttentionBackend)
        backend.max_context_len = max_context_len
        backend.page_size = page_size
        backend.topk = 4
        backend.speculative_num_draft_tokens = 8
        backend.speculative_step_id = 0
        backend.has_swa = False
        backend.attention_chunk_size = None
        backend.device = "cpu"

        backend.init_cuda_graph_state(
            max_bs, torch.zeros(max_bs, dtype=torch.int32, device="cpu")
        )

        non_topk = backend.decode_cuda_graph_metadata["page_table"]
        draft = backend.draft_decode_metadata_topk_normal["page_table"]
        target = backend.target_verify_metadata_topk_normal["page_table"]

        self.assertEqual(non_topk.shape[1], expected_pages)
        self.assertEqual(draft.shape[1], expected_pages)
        self.assertEqual(target.shape[1], expected_pages)
        self.assertLess(draft.shape[1], max_context_len)


if __name__ == "__main__":
    unittest.main()

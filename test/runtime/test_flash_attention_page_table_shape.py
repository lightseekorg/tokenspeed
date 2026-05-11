"""Shape regression test for FlashAttention spec-decode page_table buffers.

The topk > 1 (EAGLE) and SWA page_table buffers must use ``max_num_pages`` as
the column dimension (page-indexed), not ``max_context_len`` (token-indexed).
A regression here silently over-allocates VRAM by a factor of ``page_size``
on the EAGLE configs (~126 MiB at max_bs=128, ctx=128K, page_size=64).
"""

import os
import sys
import unittest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

import torch

from tokenspeed.runtime.layers.attention.backends.flash_attention import (
    FlashAttentionBackend,
)


def _make_backend(
    *,
    max_context_len: int,
    page_size: int,
    topk: int,
    speculative_num_draft_tokens: int,
    has_swa: bool,
    device: str = "cpu",
) -> FlashAttentionBackend:
    # Bypass ``__init__`` to avoid pulling a full ServerArgs/ModelConfig.
    # Every attribute touched by ``init_cuda_graph_state`` must be set here;
    # if the method ever grows a new attribute reference, this stub will
    # AttributeError loudly rather than silently miss the regression.
    backend = object.__new__(FlashAttentionBackend)
    backend.max_context_len = max_context_len
    backend.page_size = page_size
    backend.topk = topk
    backend.speculative_num_draft_tokens = speculative_num_draft_tokens
    backend.speculative_step_id = 0
    backend.has_swa = has_swa
    backend.attention_chunk_size = None
    backend.device = device
    return backend


class TestSpecDecodePageTableShape(unittest.TestCase):
    """Regression: spec-decode page_table buffers are page-indexed, not
    token-indexed. See ``init_cuda_graph_state`` in flash_attention.py."""

    MAX_BS = 32
    MAX_CONTEXT_LEN = 8192
    PAGE_SIZE = 64
    TOPK = 4
    SPEC_NUM_DRAFT = 8

    def _seq_lens_buf(self, max_bs: int) -> torch.Tensor:
        return torch.zeros(max_bs, dtype=torch.int32, device="cpu")

    def test_draft_decode_topk_normal_page_table_is_page_indexed(self):
        backend = _make_backend(
            max_context_len=self.MAX_CONTEXT_LEN,
            page_size=self.PAGE_SIZE,
            topk=self.TOPK,
            speculative_num_draft_tokens=self.SPEC_NUM_DRAFT,
            has_swa=False,
        )
        backend.init_cuda_graph_state(self.MAX_BS, self._seq_lens_buf(self.MAX_BS))

        expected_pages = (self.MAX_CONTEXT_LEN + self.PAGE_SIZE - 1) // self.PAGE_SIZE
        page_table = backend.draft_decode_metadata_topk_normal["page_table"]
        self.assertEqual(page_table.shape, (self.MAX_BS, expected_pages))

    def test_target_verify_topk_normal_page_table_is_page_indexed(self):
        backend = _make_backend(
            max_context_len=self.MAX_CONTEXT_LEN,
            page_size=self.PAGE_SIZE,
            topk=self.TOPK,
            speculative_num_draft_tokens=self.SPEC_NUM_DRAFT,
            has_swa=False,
        )
        backend.init_cuda_graph_state(self.MAX_BS, self._seq_lens_buf(self.MAX_BS))

        expected_pages = (self.MAX_CONTEXT_LEN + self.PAGE_SIZE - 1) // self.PAGE_SIZE
        page_table = backend.target_verify_metadata_topk_normal["page_table"]
        self.assertEqual(page_table.shape, (self.MAX_BS, expected_pages))

    def test_target_verify_topk_swa_page_table_is_page_indexed(self):
        # The SWA path requires page_size == 1, so page count == token count
        # and the buffer does not shrink. We still assert the buffer is sized
        # by ``max_num_pages`` to lock in the page-indexed convention in case
        # the page_size == 1 assertion is ever relaxed.
        backend = _make_backend(
            max_context_len=4096,
            page_size=1,
            topk=self.TOPK,
            speculative_num_draft_tokens=self.SPEC_NUM_DRAFT,
            has_swa=True,
        )
        backend.init_cuda_graph_state(self.MAX_BS, self._seq_lens_buf(self.MAX_BS))

        rows = self.MAX_BS * self.SPEC_NUM_DRAFT
        page_table = backend.target_verify_metadata_topk_swa["page_table"]
        self.assertEqual(page_table.shape, (rows, 4096))

    def test_topk_normal_matches_non_topk_decode_convention(self):
        # The non-topk decode buffer uses ``max_num_pages``. The topk
        # variants must use the same shape contract so they can be consumed
        # by the same FlashAttention paged-KV API. We assert both against
        # ``max_num_pages`` directly so a regression that flips both back to
        # ``max_context_len`` is still caught.
        backend = _make_backend(
            max_context_len=self.MAX_CONTEXT_LEN,
            page_size=self.PAGE_SIZE,
            topk=self.TOPK,
            speculative_num_draft_tokens=self.SPEC_NUM_DRAFT,
            has_swa=False,
        )
        backend.init_cuda_graph_state(self.MAX_BS, self._seq_lens_buf(self.MAX_BS))

        expected_pages = (self.MAX_CONTEXT_LEN + self.PAGE_SIZE - 1) // self.PAGE_SIZE
        non_topk = backend.decode_cuda_graph_metadata["page_table"]
        topk_normal = backend.draft_decode_metadata_topk_normal["page_table"]
        self.assertEqual(non_topk.shape[1], expected_pages)
        self.assertEqual(topk_normal.shape[1], expected_pages)
        self.assertLess(topk_normal.shape[1], self.MAX_CONTEXT_LEN)


if __name__ == "__main__":
    unittest.main()

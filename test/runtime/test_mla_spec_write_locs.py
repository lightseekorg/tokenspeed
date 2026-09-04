"""Verify-window widths for MLA under speculative target verify.

Write locations are router-owned (``CacheGroupRouter`` over the stacked
group tables; see test_cache_group_router.py for the slot math). What the
leaf still owns is the verify WIDTH baked into its decode metadata views:
target verify reads a whole window per request, a draft reads one row, and
the refresh clamp keeps every row's length inside the window.
"""

from __future__ import annotations

import os
import sys

import torch

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.layers.attention.backends.paged.mla import (
    MLAAttnBackend,
)

PAGE = 64


def _backend(*, spec_num_tokens: int = 1, is_draft: bool = False) -> MLAAttnBackend:
    backend = MLAAttnBackend.__new__(MLAAttnBackend)
    backend.spec_num_tokens = spec_num_tokens
    backend.is_draft = is_draft
    backend.draft_block_decode = False
    backend.device = torch.device("cpu")
    backend.max_context_len = 4096
    backend.max_num_pages = 8
    backend.kernel_page_size = PAGE
    backend.forward_decode_metadata = None
    backend._decode_views_by_bs = {}
    return backend


# --------------------------------------------------------------------------
# Verify-window width
# --------------------------------------------------------------------------


def test_target_verify_decode_uses_the_full_window() -> None:
    backend = _backend(spec_num_tokens=8)
    backend.init_cuda_graph_state(max_bs=2)
    assert backend.verify_floor == 8
    assert backend._decode_views(2).q_len_per_req == 8


def test_a_chaining_draft_never_takes_the_verify_window() -> None:
    """A chaining draft owns its own per-step write locations."""
    backend = _backend(spec_num_tokens=8, is_draft=True)
    backend.init_cuda_graph_state(max_bs=2)
    assert backend.verify_floor == 1
    assert backend._decode_views(2).q_len_per_req == 1


def test_non_speculative_decode_is_unchanged() -> None:
    backend = _backend(spec_num_tokens=1)
    backend.init_cuda_graph_state(max_bs=2)
    assert backend.verify_floor == 1
    assert backend._decode_views(2).q_len_per_req == 1


# --------------------------------------------------------------------------
# The refresh clamp keeps rows inside the window
# --------------------------------------------------------------------------


def test_refresh_clamps_seq_lens_to_the_window() -> None:
    """A request shorter than the window would resolve read positions before
    its start; the leaf's refresh raises every row to the verify floor."""
    backend = _backend(spec_num_tokens=8)
    backend.init_cuda_graph_state(max_bs=2)
    backend.refresh_decode_metadata(
        2,
        2,
        torch.tensor([3, 200], dtype=torch.int32),
        torch.zeros((2, 8), dtype=torch.int32),
    )
    assert backend.forward_decode_metadata.seq_lens.tolist() == [8, 200]


def test_capture_seeding_clamps_padded_rows_to_the_window() -> None:
    """Capture seeds the graph's idle rows at seq_len 1; the verify clamp must
    keep them inside the window on the same unified refresh path."""
    backend = _backend(spec_num_tokens=8)
    backend.init_cuda_graph_state(max_bs=2)
    backend.init_forward_metadata_capture_cuda_graph(
        bs=2,
        seq_lens=torch.ones(2, dtype=torch.int32),
        page_table=torch.zeros((2, 8), dtype=torch.int32),
    )
    assert backend.forward_decode_metadata.seq_lens.tolist() == [8, 8]


def test_draft_refresh_keeps_short_rows_unclamped() -> None:
    """Floor 1 is the identity: a draft's seq_len 1 rows stay 1."""
    backend = _backend(spec_num_tokens=8, is_draft=True)
    backend.init_cuda_graph_state(max_bs=2)
    backend.refresh_decode_metadata(
        2,
        2,
        torch.tensor([1, 200], dtype=torch.int32),
        torch.zeros((2, 8), dtype=torch.int32),
    )
    assert backend.forward_decode_metadata.seq_lens.tolist() == [1, 200]

"""Non-causal block decode for the MLA draft attention leaf.

DSpark drafts a whole block in one forward, and every block query must see the
whole block -- including the positions after it. TokenSpeed gets that without a
mask kernel: each request expands into ``spec_num_tokens`` single-query decode
rows that share the block-end ``seq_len``, so the causal decode kernel's own
mask admits the entire block. The expansion lives inside the leaf's unified
``refresh_decode_metadata`` (kernel-page tables arrive pre-expanded from the
router). These tests pin the expansion and prove the resulting attention
really is non-causal, using a case where the answer is dominated by a
*future* token so a causal implementation cannot pass by luck.
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.layers.attention.backends.paged.mla import MLAAttnBackend


def _backend(
    *,
    spec_num_tokens: int = 8,
    draft_block_decode: bool = True,
    max_num_pages: int = 4,
    max_context_len: int = 512,
) -> MLAAttnBackend:
    """An MLAAttnBackend shell carrying only the block-decode state."""
    backend = MLAAttnBackend.__new__(MLAAttnBackend)
    backend.draft_block_decode = draft_block_decode
    backend.spec_num_tokens = spec_num_tokens
    backend.is_draft = True
    backend.max_num_pages = max_num_pages
    backend.max_context_len = max_context_len
    backend.kernel_page_size = 64
    backend.device = torch.device("cpu")
    backend.forward_decode_metadata = None
    backend._decode_views_by_bs = {}
    return backend


# --------------------------------------------------------------------------
# Row expansion (the leaf's unified refresh)
# --------------------------------------------------------------------------


def test_block_refresh_keeps_every_block_row() -> None:
    """The block metadata carries one row per block position; nothing is
    sliced away regardless of the round's num_extends discriminator."""
    backend = _backend(spec_num_tokens=8)
    backend.init_cuda_graph_state(max_bs=2)
    page_table = torch.arange(2 * 4, dtype=torch.int32).view(2, 4) + 1

    backend.refresh_decode_metadata(
        2,
        2,
        torch.tensor([100, 200], dtype=torch.int32),
        page_table,
    )

    metadata = backend.forward_decode_metadata
    assert metadata.page_table.shape[0] == 2 * 8
    assert metadata.seq_lens.shape[0] == 2 * 8


def test_block_decode_is_off_without_the_flag() -> None:
    assert not _backend(draft_block_decode=False).block_decode_active


def test_block_decode_is_off_for_a_single_token_window() -> None:
    """spec_num_tokens == 1 is ordinary decode, not a block."""
    assert not _backend(spec_num_tokens=1).block_decode_active


def test_expansion_gives_every_block_row_the_same_length() -> None:
    """The uniform length is what makes the block non-causal."""
    backend = _backend(spec_num_tokens=4)
    backend.init_cuda_graph_state(max_bs=2)
    page_table = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32)
    seq_lens = torch.tensor([37, 51], dtype=torch.int32)

    backend.refresh_decode_metadata(2, 2, seq_lens, page_table)

    metadata = backend.forward_decode_metadata
    assert metadata.page_table.shape == (8, 4)
    assert metadata.seq_lens.tolist() == [37, 37, 37, 37, 51, 51, 51, 51]
    # Each request's rows are contiguous and carry its own page table.
    assert metadata.page_table[:4].eq(page_table[0]).all()
    assert metadata.page_table[4:].eq(page_table[1]).all()


def test_expansion_clamps_to_the_context_limit() -> None:
    """Without the clamp a near-limit request reads past the page table."""
    backend = _backend(spec_num_tokens=4, max_context_len=64)
    backend.init_cuda_graph_state(max_bs=1)

    backend.refresh_decode_metadata(
        1,
        1,
        torch.tensor([9999], dtype=torch.int32),
        torch.zeros((1, 4), dtype=torch.int32),
    )
    assert backend.forward_decode_metadata.seq_lens.tolist() == [64, 64, 64, 64]


def test_expansion_floors_at_the_block_width() -> None:
    backend = _backend(spec_num_tokens=8)
    backend.init_cuda_graph_state(max_bs=1)

    backend.refresh_decode_metadata(
        1,
        1,
        torch.tensor([3], dtype=torch.int32),
        torch.zeros((1, 4), dtype=torch.int32),
    )
    assert backend.forward_decode_metadata.seq_lens.tolist() == [8] * 8


# --------------------------------------------------------------------------
# CUDA-graph buffers (shape/bookkeeping only; no device needed)
# --------------------------------------------------------------------------


def test_graph_buffers_are_sized_by_the_block_decode_expansion() -> None:
    backend = _backend(spec_num_tokens=8)
    backend.init_cuda_graph_state(max_bs=4)

    assert backend.block_decode_expansion == 8
    assert backend.page_table_buf.shape == (4 * backend.block_decode_expansion, 4)
    assert backend.seq_lens_buf.shape == (4 * backend.block_decode_expansion,)
    assert _backend(draft_block_decode=False).block_decode_expansion == 1


def test_graph_capture_records_expanded_metadata() -> None:
    backend = _backend(spec_num_tokens=8)
    backend.init_cuda_graph_state(max_bs=4)
    # Runner contract: capture is the idle refresh over the persistent
    # buffers; capture seq_lens are seeded to spec_num_tokens.
    backend.init_forward_metadata_capture_cuda_graph(
        bs=2,
        seq_lens=torch.tensor([8, 8], dtype=torch.int32),
        page_table=torch.zeros((2, 4), dtype=torch.int32),
    )

    metadata = backend.forward_decode_metadata
    assert metadata.page_table.shape == (16, 4)
    assert metadata.seq_lens.shape == (16,)
    # Seeded with a safe baseline; the real lengths arrive in-graph.
    assert metadata.seq_lens.tolist() == [8] * 16


def test_fill_block_decode_seq_lens_broadcasts_per_request() -> None:
    backend = _backend(spec_num_tokens=4)
    backend.init_cuda_graph_state(max_bs=2)

    backend.fill_block_decode_seq_lens(2, torch.tensor([31, 47], dtype=torch.int32))
    assert backend.seq_lens_buf[:8].tolist() == [31, 31, 31, 31, 47, 47, 47, 47]


def test_fill_block_decode_seq_lens_clamps_to_context() -> None:
    backend = _backend(spec_num_tokens=4, max_context_len=64)
    backend.init_cuda_graph_state(max_bs=1)

    backend.fill_block_decode_seq_lens(1, torch.tensor([9999], dtype=torch.int32))
    assert backend.seq_lens_buf[:4].tolist() == [64] * 4


def test_graph_replay_replicates_the_page_table_across_block_rows() -> None:
    backend = _backend(spec_num_tokens=4, max_num_pages=3)
    backend.init_cuda_graph_state(max_bs=2)
    backend.init_forward_metadata_capture_cuda_graph(
        bs=2,
        seq_lens=torch.tensor([4, 4], dtype=torch.int32),
        page_table=torch.zeros((2, 3), dtype=torch.int32),
    )

    page_table = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
    backend.refresh_decode_metadata(
        2,
        2,
        torch.tensor([10, 20], dtype=torch.int32),
        page_table,
        for_graph_replay=True,
    )

    table = backend.page_table_buf[:8]
    assert table[:4].eq(torch.tensor([1, 2, 3], dtype=torch.int32)).all()
    assert table[4:].eq(torch.tensor([4, 5, 6], dtype=torch.int32)).all()
    # Lengths are the drafter's job (in-graph), still at the capture baseline.
    assert backend.seq_lens_buf[:8].tolist() == [4] * 8


def test_refresh_copies_published_kernel_pages_as_is() -> None:
    """The router hands the leaf kernel pages; the leaf must copy them
    verbatim (identity), never re-expand -- eager and replay alike."""
    backend = _backend(spec_num_tokens=4, max_num_pages=4)
    backend.init_cuda_graph_state(max_bs=1)
    backend.init_forward_metadata_capture_cuda_graph(
        bs=1,
        seq_lens=torch.tensor([4], dtype=torch.int32),
        page_table=torch.zeros((1, 4), dtype=torch.int32),
    )

    # The router already expanded logical page 3 into kernel pages 6 and 7.
    published = torch.tensor([[6, 7, 0, 1]], dtype=torch.int32)
    backend.refresh_decode_metadata(
        1,
        1,
        torch.tensor([10], dtype=torch.int32),
        published,
        for_graph_replay=True,
    )
    assert backend.page_table_buf[:4].eq(published[0]).all()

    backend.refresh_decode_metadata(
        1,
        1,
        torch.tensor([10], dtype=torch.int32),
        published,
    )
    assert backend.forward_decode_metadata.page_table[0].eq(published[0]).all()


# --------------------------------------------------------------------------
# The semantics that matter: the block really is non-causal
# --------------------------------------------------------------------------


def _reference_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """[T,D] queries against [S,D] keys under an explicit [T,S] boolean mask."""
    scores = (q.float() @ k.float().T) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v.float()


def _seqlen_mask(seq_lens: torch.Tensor, total_keys: int) -> torch.Tensor:
    """The mask the decode kernel derives from per-row cache_seqlens."""
    key_positions = torch.arange(total_keys).unsqueeze(0)
    return key_positions < seq_lens.unsqueeze(1)


def _expanded_seq_lens(block: int, total: int) -> torch.Tensor:
    """The block rows' uniform lengths as the refresh publishes them."""
    backend = _backend(spec_num_tokens=block, max_context_len=1024)
    backend.init_cuda_graph_state(max_bs=1)
    backend.refresh_decode_metadata(
        1,
        1,
        torch.tensor([total], dtype=torch.int32),
        torch.zeros((1, 4), dtype=torch.int32),
    )
    return backend.forward_decode_metadata.seq_lens.clone()


def test_uniform_seqlens_reproduce_non_causal_block_attention() -> None:
    """The expansion's mask equals a full non-causal block mask.

    This is the load-bearing claim of the whole approach: uniform block-end
    lengths are equivalent to letting every block query see the entire block.
    """
    prefix, block = 5, 7
    total = prefix + block

    produced = _seqlen_mask(_expanded_seq_lens(block, total), total)

    non_causal = torch.ones(block, total, dtype=torch.bool)
    assert torch.equal(produced, non_causal)


def test_uniform_seqlens_differ_from_the_causal_chain() -> None:
    """The pre-existing causal offsets give a strictly different mask."""
    prefix, block = 5, 7
    total = prefix + block
    seq_lens = torch.tensor([total], dtype=torch.int32)

    uniform = _expanded_seq_lens(block, total)
    # The causal chain the target's verify path uses: offsets 1-N .. 0.
    causal = seq_lens.repeat_interleave(block) + torch.arange(1 - block, 1)

    assert not torch.equal(_seqlen_mask(uniform, total), _seqlen_mask(causal, total))


def test_future_token_dominates_only_under_the_non_causal_mask() -> None:
    """A case a causal implementation cannot pass by coincidence.

    Query 0 is aligned with a key that sits at the *end* of the block. Under the
    non-causal mask the output is that future value; under the causal chain
    query 0 cannot see it at all, so the two answers must differ.
    """
    prefix, block = 3, 7
    total = prefix + block
    # One dimension per key, so a query can select exactly one of them.
    dim = total

    keys = torch.eye(total, dim)
    values = torch.zeros(total, dim)
    for i in range(total):
        values[i, 0] = float(i)
    # Query 0 points hard at the last block position (a strictly future token).
    future_key_idx = total - 1
    queries = torch.zeros(block, dim)
    queries[0, future_key_idx] = 50.0

    seq_lens = torch.tensor([total], dtype=torch.int32)
    uniform = _expanded_seq_lens(block, total)
    causal = seq_lens.repeat_interleave(block) + torch.arange(1 - block, 1)

    non_causal_out = _reference_attention(
        queries, keys, values, _seqlen_mask(uniform, total)
    )
    causal_out = _reference_attention(
        queries, keys, values, _seqlen_mask(causal, total)
    )

    # Non-causal recovers the future token's value; causal cannot reach it.
    assert non_causal_out[0, 0] == pytest.approx(float(future_key_idx), abs=1e-3)
    assert causal_out[0, 0] < float(future_key_idx) - 1.0


# --------------------------------------------------------------------------
# The non-block paths must be untouched
# --------------------------------------------------------------------------


def test_target_verify_keeps_the_unexpanded_causal_path() -> None:
    """draft_block_decode is draft-only; the target's verify must not expand."""
    backend = _backend(draft_block_decode=False, spec_num_tokens=8)
    backend.is_draft = False
    assert not backend.block_decode_active


def test_graph_buffers_are_unexpanded_without_block_decode() -> None:
    backend = _backend(draft_block_decode=False, spec_num_tokens=8)
    backend.init_cuda_graph_state(max_bs=4)
    assert backend.page_table_buf.shape == (4, 4)
    assert backend.seq_lens_buf.shape == (4,)


def test_attn_config_carries_the_block_decode_flag_defaulting_off() -> None:
    """AttnConfig must carry the flag so generate() can set it for the draft."""
    from tokenspeed.runtime.layers.attention.configs.base import AttnConfig

    field = AttnConfig.__dataclass_fields__["draft_block_decode"]
    assert field.default is False

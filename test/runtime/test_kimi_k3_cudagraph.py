"""Kimi-K3 MLA decode CUDA-graph capture/replay core logic.

CPU-only (plain tensors, no real graph capture): exercises the metadata-buffer
capture/replay LOGIC the decode CUDA graph depends on, through the unified
refresh path. The real graph capture/replay parity on the 93-layer serve is
validated on GPU separately.

Coverage:

- the MLA full-attention decode graph: capture binds stable ``page_table``
  buffers, replay refreshes them IN PLACE (same ``data_ptr``) from the
  router-expanded group table;
- padded batch rows resolve to the null page 0 (dummy-page protection,
  the GroupTableStacks fill contract);
- DFLASH/DSpark block decode: row expansion, clamping, in-graph length
  rewrites and the one-query-per-block-row kernel launch.

The KDA multi-group state capture/replay logic lives in
``test_kimi_k3_kda.py``.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest import mock

import torch

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

from tokenspeed.runtime.layers.attention.backends.paged import (
    tokenspeed_mla as tokenspeed_mla_module,
)
from tokenspeed.runtime.layers.attention.backends.paged.group_tables import (
    GroupTableSpec,
    GroupTableStacks,
)
from tokenspeed.runtime.layers.attention.backends.paged.mla import MLAAttnBackend
from tokenspeed.runtime.layers.attention.backends.paged.tokenspeed_mla import (
    CuteDSLMLABackend,
)

register_cuda_ci(est_time=10, suite="runtime-1gpu")

_PAGE_SIZE = 64  # kernel page
_KV_LORA = 4
_ROPE = 4
_LOGICAL_P = 128  # logical block size (ratio 2 kernel pages per logical page)
_MAX_CTX = 256
_FULL = "full_attention"


def _bare_mla_backend(
    *,
    is_draft: bool = False,
    spec_num_tokens: int = 1,
    draft_block_decode: bool = False,
) -> CuteDSLMLABackend:
    """A CuteDSLMLABackend with only the attributes the CUDA-graph metadata
    paths touch — the full ctor JIT-compiles CuteDSL kernels (GPU only)."""
    backend = object.__new__(CuteDSLMLABackend)
    backend.device = "cpu"
    backend.kernel_page_size = _PAGE_SIZE
    backend.max_context_len = _MAX_CTX
    backend.is_draft = is_draft
    backend.spec_num_tokens = spec_num_tokens
    backend.draft_block_decode = draft_block_decode
    backend.kv_lora_rank = _KV_LORA
    backend.qk_rope_head_dim = _ROPE
    backend.kv_cache_dim = _KV_LORA + _ROPE
    backend.data_type = torch.bfloat16
    backend.cutedsl_workspace = None
    backend.forward_decode_metadata = None
    backend.page_table_buf = None
    backend.seq_lens_buf = None
    backend._decode_views_by_bs = {}
    return backend


def _bare_amd_mla_backend(*, spec_num_tokens: int = 1) -> MLAAttnBackend:
    backend = MLAAttnBackend.__new__(MLAAttnBackend)
    backend.device = "cpu"
    backend.kernel_page_size = _PAGE_SIZE
    backend.max_context_len = _MAX_CTX
    backend.max_num_pages = _MAX_CTX // _PAGE_SIZE
    backend.is_draft = False
    backend.spec_num_tokens = spec_num_tokens
    backend.draft_block_decode = False
    backend.forward_decode_metadata = None
    backend.page_table_buf = None
    backend.seq_lens_buf = None
    backend._decode_views_by_bs = {}
    return backend


def _group_table(backend, raw_rows, bs: int, actual_bs: int) -> torch.Tensor:
    """The router-side expansion stage: raw scheduler blocks (logical P) ->
    the leaf's ``[bs, max_num_pages]`` kernel-page table, rows past
    ``actual_bs`` and holes nulled to page 0."""
    stacks = GroupTableStacks(
        [
            GroupTableSpec(
                group_id=_FULL,
                block_granularity=_LOGICAL_P,
                kernel_page_size=backend.kernel_page_size,
                max_num_pages=backend.max_num_pages,
            )
        ],
        max_bs=max(bs, 4),
        max_tokens_per_req=backend.spec_num_tokens,
        device="cpu",
    )
    raw = torch.tensor(raw_rows, dtype=torch.int32)
    stacks.fill(bs, actual_bs, {_FULL: raw})
    return stacks.table(_FULL, bs)


def test_mla_target_verify_width_applies_to_mixed_batches() -> None:
    """A mixed round's decode rows keep the full verify window: the refresh
    publishes the round's extend/decode split without narrowing q_len."""
    backend = _bare_mla_backend(spec_num_tokens=8)
    backend.init_cuda_graph_state(max_bs=2)
    backend.refresh_decode_metadata(
        2,
        2,
        torch.tensor([70, 40], dtype=torch.int32),
        torch.zeros((2, backend.max_num_pages), dtype=torch.int32),
        num_extends=1,
    )
    metadata = backend.forward_decode_metadata
    assert metadata.q_len_per_req == 8
    assert metadata.num_extends == 1


def test_replay_refreshes_buffers_in_place_and_pads_page_zero() -> None:
    backend = _bare_mla_backend()
    backend.init_cuda_graph_state(max_bs=2)
    backend.init_forward_metadata_capture_cuda_graph(
        bs=2,
        seq_lens=torch.tensor([1, 1], dtype=torch.int32),
        page_table=torch.zeros((2, backend.max_num_pages), dtype=torch.int32),
    )
    md = backend.forward_decode_metadata
    captured_kv_ptr = md.page_table.data_ptr()

    # One REAL request (row 0), one padded dummy row (row 1). The raw table
    # carries only the real row; padded rows must land on page 0.
    # Grouped table: real row 0 has two logical pages [3, 5]; page ids > 0.
    table = _group_table(backend, [[3, 5]], bs=2, actual_bs=1)

    backend.refresh_decode_metadata(
        2,  # padded bs
        1,
        torch.tensor([70, 1], dtype=torch.int32),  # real seq 70, pad 1
        table,
        for_graph_replay=True,
    )
    md2 = backend.forward_decode_metadata
    # SAME buffers refreshed in place (no realloc): pointer-stable replay.
    assert md2.page_table.data_ptr() == captured_kv_ptr

    # Real row 0: logical page 3 -> kernel pages [6, 7] (ratio 2), page 5 ->
    # [10, 11]. Expansion: page * ratio + k.
    assert md2.page_table[0].tolist() == [6, 7, 10, 11]

    # Padded row 1: null page 0 everywhere.
    assert torch.all(md2.page_table[1] == 0)


def test_amd_mla_grouped_graph_replay_is_pointer_stable_and_null_padded() -> None:
    backend = _bare_amd_mla_backend()
    backend.init_cuda_graph_state(max_bs=2)
    backend.init_forward_metadata_capture_cuda_graph(
        bs=2,
        seq_lens=torch.ones(2, dtype=torch.int32),
        page_table=torch.zeros((2, backend.max_num_pages), dtype=torch.int32),
    )
    captured = backend.forward_decode_metadata
    page_ptr = captured.page_table.data_ptr()

    table = _group_table(backend, [[3, 5]], bs=2, actual_bs=1)
    backend.refresh_decode_metadata(
        2,
        1,
        torch.tensor([70, 1], dtype=torch.int32),
        table,
        for_graph_replay=True,
    )
    replayed = backend.forward_decode_metadata
    assert replayed.page_table.data_ptr() == page_ptr
    assert replayed.page_table[0].tolist() == [6, 7, 10, 11]
    assert torch.all(replayed.page_table[1] == 0)


def test_amd_mla_eager_decode_consumes_the_expanded_group_table() -> None:
    """Eager decode rides the same refresh: the leaf copies the
    router-expanded table verbatim; a -1 hole is already the null page."""
    backend = _bare_amd_mla_backend()
    backend.init_cuda_graph_state(max_bs=2)
    table = _group_table(backend, [[3, 5], [4, -1]], bs=2, actual_bs=2)
    backend.refresh_decode_metadata(
        2,
        2,
        torch.tensor([70, 40], dtype=torch.int32),
        table,
    )
    decode = backend.forward_decode_metadata
    assert decode.page_table[0].tolist() == [6, 7, 10, 11]
    # The ragged -1 hole collapses onto the null page across its ratio slots.
    assert decode.page_table[1].tolist() == [8, 9, 0, 0]


def test_block_decode_expands_one_row_per_block_position() -> None:
    """A block draft's rows all carry the block-end length, which is what makes
    the block non-causal: the kernel masks each row by its own cache length."""
    spec = 4
    backend = _bare_mla_backend(
        is_draft=True,
        spec_num_tokens=spec,
        draft_block_decode=True,
    )
    assert backend.block_decode_active
    backend.init_cuda_graph_state(max_bs=2)
    table = torch.tensor(
        [[3, 4, 0, 0], [7, 8, 0, 0]], dtype=torch.int32
    )  # kernel pages
    seq_lens = torch.tensor([9, 130], dtype=torch.int32)

    backend.refresh_decode_metadata(2, 2, seq_lens, table)

    metadata = backend.forward_decode_metadata
    assert metadata.page_table.tolist() == [[3, 4, 0, 0]] * spec + [[7, 8, 0, 0]] * spec
    assert metadata.seq_lens_k.tolist() == [9] * spec + [130] * spec


def test_block_decode_clamps_a_length_below_the_block() -> None:
    """A request shorter than the block would ask the kernel for keys it has
    no page for; an overshooting one would walk past the table."""
    spec = 8
    backend = _bare_mla_backend(
        is_draft=True,
        spec_num_tokens=spec,
        draft_block_decode=True,
    )
    backend.init_cuda_graph_state(max_bs=2)
    backend.refresh_decode_metadata(
        2,
        2,
        torch.tensor([1, _MAX_CTX * 4], dtype=torch.int32),
        torch.zeros((2, backend.max_num_pages), dtype=torch.int32),
    )
    lens = backend.forward_decode_metadata.seq_lens_k
    assert lens[:spec].tolist() == [spec] * spec
    assert lens[spec:].tolist() == [_MAX_CTX] * spec


def test_block_decode_graph_buffers_hold_every_block_row() -> None:
    spec, max_bs = 4, 3
    backend = _bare_mla_backend(
        is_draft=True,
        spec_num_tokens=spec,
        draft_block_decode=True,
    )
    backend.init_cuda_graph_state(max_bs)
    assert backend.seq_lens_buf.shape[0] == max_bs * spec
    assert backend.page_table_buf.shape[0] == max_bs * spec


def test_block_decode_replay_broadcasts_pages_and_scrubs_padding() -> None:
    """Padded rows must reach the null page, not another request's pages."""
    spec, max_bs, bs = 4, 3, 3
    backend = _bare_mla_backend(
        is_draft=True,
        spec_num_tokens=spec,
        draft_block_decode=True,
    )
    backend.init_cuda_graph_state(max_bs)
    backend.page_table_buf.fill_(-9)
    # Raw scheduler pages (logical P) expand router-side (ratio 2:
    # page L -> [2L, 2L+1]); the third batch row is graph padding.
    table = _group_table(backend, [[5, 6], [7, 8]], bs=bs, actual_bs=2)

    backend.refresh_decode_metadata(
        bs,
        2,
        torch.tensor([40, 50, 1], dtype=torch.int32),
        table,
        for_graph_replay=True,
    )

    rows = backend.page_table_buf[: bs * spec]
    assert rows[:spec, :4].tolist() == [[10, 11, 12, 13]] * spec
    assert rows[spec : 2 * spec, :4].tolist() == [[14, 15, 16, 17]] * spec
    assert rows[2 * spec :].eq(0).all(), "padded request kept live pages"

    # A table narrower than the persistent buffer scrubs the tail columns.
    backend.page_table_buf.fill_(-9)
    narrow = torch.tensor([[1, 2], [3, 4], [0, 0]], dtype=torch.int32)
    backend.refresh_decode_metadata(
        bs,
        2,
        torch.tensor([40, 50, 1], dtype=torch.int32),
        narrow,
        for_graph_replay=True,
    )
    rows = backend.page_table_buf[: bs * spec]
    assert rows[:, 2:].eq(0).all(), "columns past the table were not scrubbed"


def test_block_decode_lengths_are_rewritten_per_replay() -> None:
    """The drafter calls this inside the captured graph, so two replays with
    different live draft lengths must not share the first one's rows."""
    spec, max_bs = 4, 2
    backend = _bare_mla_backend(
        is_draft=True,
        spec_num_tokens=spec,
        draft_block_decode=True,
    )
    backend.init_cuda_graph_state(max_bs)
    buf = backend.seq_lens_buf

    backend.fill_block_decode_seq_lens(2, torch.tensor([40, 50], dtype=torch.int32))
    assert buf[: 2 * spec].tolist() == [40] * spec + [50] * spec
    backend.fill_block_decode_seq_lens(2, torch.tensor([41, 99], dtype=torch.int32))
    assert buf[: 2 * spec].tolist() == [41] * spec + [99] * spec


def test_block_decode_stays_off_for_target_and_single_token_drafts() -> None:
    """The expansion must be unreachable on every path it was not written for."""
    target = _bare_mla_backend(spec_num_tokens=8)
    assert not target.block_decode_active
    one = _bare_mla_backend(
        is_draft=True,
        spec_num_tokens=1,
        draft_block_decode=True,
    )
    assert not one.block_decode_active


def test_block_decode_hands_the_kernel_one_query_per_block_row() -> None:
    """The expansion is only worth anything if it reaches the kernel: keeping
    the block on the query axis would restore exactly the causal order the
    draft must not have, and every other case here would still pass."""
    spec, bs, heads, dim = 4, 2, 3, 8
    backend = _bare_mla_backend(
        is_draft=True,
        spec_num_tokens=spec,
        draft_block_decode=True,
    )
    backend.init_cuda_graph_state(max_bs=bs)
    backend.refresh_decode_metadata(
        bs,
        bs,
        torch.tensor([40, 50], dtype=torch.int32),
        torch.tensor([[1, 2, 0, 0], [3, 4, 0, 0]], dtype=torch.int32),
    )
    layer = SimpleNamespace(
        tp_q_head_num=heads,
        head_dim=dim,
        layer_id=0,
        scaling=1.0,
        v_head_dim=_KV_LORA,
        k_scale_float=None,
    )
    pool = SimpleNamespace(
        get_key_buffer=lambda _lid: torch.zeros(
            4 * backend.kernel_page_size, backend.kv_cache_dim, dtype=torch.bfloat16
        )
    )

    seen: dict = {}

    def _spy(**kw):
        seen.update(kw)
        return torch.zeros(bs * spec, 1, heads, _KV_LORA)

    # The real workspace sizing wants a CUDA device; the kernel itself is spied.
    with mock.patch.object(
        tokenspeed_mla_module, "tokenspeed_mla_decode", _spy
    ), mock.patch.object(
        type(backend), "_cutedsl_workspace", lambda _self, _q_len: None
    ):
        backend.forward_decode(
            q=torch.zeros(bs * spec, heads, dim, dtype=torch.bfloat16),
            k=None,
            v=None,
            layer=layer,
            out_cache_loc=torch.zeros(bs * spec, dtype=torch.int64),
            token_to_kv_pool=pool,
            bs=bs,
            save_kv_cache=False,
        )

    assert seen["query"].shape == (bs * spec, 1, heads, dim)
    assert seen["block_tables"].shape[0] == bs * spec
    assert seen["seq_lens"].tolist() == [40] * spec + [50] * spec

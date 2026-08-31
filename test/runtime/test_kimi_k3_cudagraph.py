"""Kimi-K3 cache-group CUDA-graph capture/replay core logic.

CPU-only (plain tensors, no real graph capture): exercises the metadata-buffer
capture/replay LOGIC that the decode CUDA graph depends on. The real
graph capture/replay parity on the 93-layer serve is validated on GPU
separately.

Coverage:

- the MLA full-attention decode graph: capture binds stable
  ``block_kv_indices`` + ``group_out_cache_loc`` buffers, replay refreshes them
  IN PLACE (same ``data_ptr``) from a fresh forward op;
- padded batch rows resolve to the null page 0 (dummy-page protection);
- the draft/target structural gate on the MLA
  capture/replay path.

The KDA multi-group state capture/replay logic lives in
``test_kimi_k3_kda.py``.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends import (
    tokenspeed_mla as tokenspeed_mla_module,
)
from tokenspeed.runtime.layers.attention.backends.mla import MLAAttnBackend
from tokenspeed.runtime.layers.attention.backends.mla_cache_groups import (
    MlaCacheGroupMixin,
)
from tokenspeed.runtime.layers.attention.backends.tokenspeed_mla import (
    CuteDSLMLABackend,
    CuteDSLMLADecodeMetadata,
)

register_cuda_ci(est_time=10, suite="runtime-1gpu")

_PAGE_SIZE = 64  # kernel page
_KV_LORA = 4
_ROPE = 4
_LOGICAL_P = 128  # logical block size (ratio 2 kernel pages per logical page)
_MAX_CTX = 256


def _bare_mla_backend(
    *,
    cache_contract: bool,
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
    backend._block_table_aliased = False
    backend._cache_groups_bound = False
    backend.decode_cuda_graph_metadata = {}
    backend.decode_cuda_graph_kv_indices = None
    backend._full_history_group_id = "full_attention"
    backend._history_block_granularity = _LOGICAL_P
    backend.decode_cuda_graph_group_out_cache_loc = None
    backend.forward_decode_metadata = None
    del cache_contract
    return backend


def test_target_verify_mixed_batch_skips_complete_prefill_windows() -> None:
    backend = _bare_mla_backend(cache_contract=False, spec_num_tokens=8)
    backend._cache_groups_bound = True
    locations = torch.arange(16, dtype=torch.int64)
    backend.forward_decode_metadata = CuteDSLMLADecodeMetadata(
        num_extends=1,
        group_out_cache_loc=locations,
        group_q_len_per_req=8,
    )

    selected = backend.select_out_cache_loc(
        SimpleNamespace(layer_id=0),
        torch.full((8,), -1, dtype=torch.int64),
        ForwardMode.DECODE,
    )

    assert selected.tolist() == list(range(8, 16))


def test_mla_target_verify_width_applies_to_mixed_batches() -> None:
    backend = object.__new__(MlaCacheGroupMixin)
    backend.spec_num_tokens = 8
    backend.is_draft = False

    assert backend._verify_q_len(ForwardMode.DECODE) == 8
    assert backend._verify_q_len(ForwardMode.MIXED) == 8


def test_cutedsl_mla_draft_keeps_classic_page_table_contract() -> None:
    # The guarantee is structural now: a draft never allocates the group
    # write-location buffer, and its bind latch stays down unless the runner
    # explicitly dispatches cache groups to it.
    backend = _bare_mla_backend(cache_contract=False, is_draft=True)
    backend.init_cuda_graph_state(max_bs=2)

    backend.bind_decode_views(2)

    assert backend._cache_groups_bound is False
    assert backend.decode_cuda_graph_group_out_cache_loc is None


def _bare_amd_mla_backend(
    *, cache_contract: bool, spec_num_tokens: int = 1
) -> MLAAttnBackend:
    backend = object.__new__(MLAAttnBackend)
    backend.device = "cpu"
    backend.kernel_page_size = _PAGE_SIZE
    backend.max_context_len = _MAX_CTX
    backend.max_num_pages = _MAX_CTX // _PAGE_SIZE
    backend.is_draft = False
    backend.spec_num_tokens = spec_num_tokens
    backend.draft_block_decode = False
    backend._cache_groups_bound = False
    backend.decode_cuda_graph_metadata = {}
    backend.cuda_graph_page_table = None
    backend.cuda_graph_seq_lens = None
    backend.decode_cuda_graph_group_out_cache_loc = None
    backend.forward_decode_metadata = None
    backend._should_use_absorbed_cached_extend = lambda **_: False
    backend._full_history_group_id = "full_attention"
    backend._history_block_granularity = _LOGICAL_P
    del cache_contract
    return backend


def test_replay_refreshes_buffers_in_place_and_pads_page_zero() -> None:
    backend = _bare_mla_backend(cache_contract=True)
    backend.init_cuda_graph_state(max_bs=2)
    backend.init_forward_metadata_capture_cuda_graph(
        bs=2,
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([1, 1], dtype=torch.int32),
        forward_mode=ForwardMode.DECODE,
        cache_group_ids=("full_attention",),
    )
    md = backend.decode_cuda_graph_metadata[2]
    captured_kv_ptr = md.block_kv_indices.data_ptr()
    captured_loc_ptr = md.group_out_cache_loc.data_ptr()

    # One REAL request (row 0), one padded dummy row (row 1). The op-bound
    # table carries only the real row; padded rows must land on page 0.
    # Grouped table: real row 0 has two logical pages [3, 5]; page ids > 0.
    table = torch.tensor([[3, 5]], dtype=torch.int32)

    backend.refresh_decode_metadata(
        2,  # padded bs
        1,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([70, 1], dtype=torch.int32),  # real seq 70, pad 1
        forward_mode=ForwardMode.DECODE,
        page_table=None,
        for_graph_replay=True,
        block_tables={"full_attention": table},
    )
    md2 = backend.forward_decode_metadata
    # SAME buffers refreshed in place (no realloc): pointer-stable replay.
    assert md2.block_kv_indices.data_ptr() == captured_kv_ptr
    assert md2.group_out_cache_loc.data_ptr() == captured_loc_ptr

    # Real row 0: logical page 3 -> kernel pages [6, 7] (ratio 2), page 5 ->
    # [10, 11]. Expansion: page * ratio + k.
    assert md2.block_kv_indices[0].tolist() == [6, 7, 10, 11]
    # Write loc for seq_len 70: pos 69, logical page idx 0 -> page 3, offset 69:
    # 3 * 128 + 69 = 453.
    assert md2.group_out_cache_loc[0].item() == 3 * _LOGICAL_P + 69

    # Padded row 1: null page 0 everywhere.
    assert torch.all(md2.block_kv_indices[1] == 0)
    assert md2.group_out_cache_loc[1].item() == 0


def test_amd_mla_grouped_graph_replay_is_pointer_stable_and_null_padded() -> None:
    backend = _bare_amd_mla_backend(cache_contract=True)
    seq_buf = torch.ones(2, dtype=torch.int32)
    backend.init_cuda_graph_state(max_bs=2)
    backend.init_forward_metadata_capture_cuda_graph(
        bs=2,
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=seq_buf,
        forward_mode=ForwardMode.DECODE,
        cache_group_ids=("full_attention",),
    )
    captured = backend.decode_cuda_graph_metadata[2]
    page_ptr = captured.page_table.data_ptr()
    loc_ptr = captured.group_out_cache_loc.data_ptr()

    table = torch.tensor([[3, 5]], dtype=torch.int32)
    backend.refresh_decode_metadata(
        2,
        1,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([70, 1], dtype=torch.int32),
        forward_mode=ForwardMode.DECODE,
        page_table=None,
        for_graph_replay=True,
        block_tables={"full_attention": table},
    )
    replayed = backend.forward_decode_metadata
    assert replayed.page_table.data_ptr() == page_ptr
    assert replayed.group_out_cache_loc.data_ptr() == loc_ptr
    assert replayed.page_table[0].tolist() == [6, 7, 10, 11]
    assert replayed.group_out_cache_loc[0].item() == 3 * _LOGICAL_P + 69
    assert torch.all(replayed.page_table[1] == 0)
    assert replayed.group_out_cache_loc[1].item() == 0


def test_amd_mla_target_verify_graph_refreshes_all_write_locations() -> None:
    backend = _bare_amd_mla_backend(cache_contract=True, spec_num_tokens=2)
    backend.init_cuda_graph_state(max_bs=2)
    backend.init_forward_metadata_capture_cuda_graph(
        bs=2,
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([2, 2], dtype=torch.int32),
        forward_mode=ForwardMode.DECODE,
        cache_group_ids=("full_attention",),
    )
    captured = backend.decode_cuda_graph_metadata[2]
    page_ptr = captured.page_table.data_ptr()
    loc_ptr = captured.group_out_cache_loc.data_ptr()
    assert captured.group_q_len_per_req == 2
    assert captured.group_out_cache_loc.shape == (4,)

    table = torch.tensor([[3, 5]], dtype=torch.int32)
    backend.refresh_decode_metadata(
        2,
        1,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([70, 1], dtype=torch.int32),
        forward_mode=ForwardMode.DECODE,
        page_table=None,
        for_graph_replay=True,
        block_tables={"full_attention": table},
    )
    replayed = backend.forward_decode_metadata
    assert replayed.page_table.data_ptr() == page_ptr
    assert replayed.group_out_cache_loc.data_ptr() == loc_ptr
    assert replayed.group_out_cache_loc.tolist() == [
        3 * _LOGICAL_P + 68,
        3 * _LOGICAL_P + 69,
        0,
        0,
    ]


def test_amd_mla_eager_decode_uses_group_table_and_refuses_fallback() -> None:
    backend = _bare_amd_mla_backend(cache_contract=True)
    table = torch.tensor([[3, 5], [4, -1]], dtype=torch.int32)
    seq_lens = torch.tensor([70, 40], dtype=torch.int32)
    poisoned = torch.full((8, 8), -99, dtype=torch.int32)
    backend.init_cuda_graph_state(max_bs=2)
    backend.refresh_decode_metadata(
        2,
        2,
        torch.tensor([-99, -99], dtype=torch.int64),
        seq_lens,
        forward_mode=ForwardMode.DECODE,
        page_table=poisoned,
        block_tables={"full_attention": table},
    )
    decode = backend.forward_decode_metadata
    assert decode.page_table[0].tolist() == [6, 7, 10, 11]
    assert decode.page_table[1].tolist() == [8, 9, 0, 1]
    assert decode.group_out_cache_loc.tolist() == [
        3 * _LOGICAL_P + 69,
        4 * _LOGICAL_P + 39,
    ]
    selected = backend.select_out_cache_loc(
        SimpleNamespace(layer_id=0),
        torch.tensor([-1, -1], dtype=torch.int64),
        ForwardMode.DECODE,
    )
    assert torch.equal(selected, decode.group_out_cache_loc)


def test_amd_mla_eager_target_verify_writes_the_full_window() -> None:
    backend = _bare_amd_mla_backend(cache_contract=True, spec_num_tokens=2)
    table = torch.tensor([[3, 5], [4, 6]], dtype=torch.int32)
    backend.init_cuda_graph_state(max_bs=2)
    backend.refresh_decode_metadata(
        2,
        2,
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([70, 130], dtype=torch.int32),
        forward_mode=ForwardMode.DECODE,
        page_table=torch.zeros((2, 4), dtype=torch.int32),
        block_tables={"full_attention": table},
    )

    decode = backend.forward_decode_metadata
    assert decode.group_q_len_per_req == 2
    assert decode.group_out_cache_loc.tolist() == [
        3 * _LOGICAL_P + 68,
        3 * _LOGICAL_P + 69,
        6 * _LOGICAL_P,
        6 * _LOGICAL_P + 1,
    ]
    selected = backend.select_out_cache_loc(
        SimpleNamespace(layer_id=0),
        torch.full((4,), -1, dtype=torch.int64),
        ForwardMode.DECODE,
    )
    assert torch.equal(selected, decode.group_out_cache_loc)


def test_amd_mla_eager_prefill_derives_group_write_locations(monkeypatch) -> None:
    from tokenspeed.runtime.layers.attention.backends import mla as mla_module

    monkeypatch.setattr(
        mla_module,
        "build_chunked_prefill_metadata_arrays",
        lambda *args: (
            1,
            [torch.tensor([0], dtype=torch.int32)],
            torch.tensor([1], dtype=torch.int32),
            torch.tensor([0, 1], dtype=torch.int32),
            [1],
        ),
    )
    backend = _bare_amd_mla_backend(cache_contract=True)
    table = torch.tensor([[3, 5]], dtype=torch.int32)
    backend.init_forward_metadata(
        bs=1,
        num_extends=1,
        req_pool_indices=torch.tensor([-99], dtype=torch.int64),
        seq_lens=torch.tensor([150], dtype=torch.int32),
        page_table=torch.full((4, 4), -99, dtype=torch.int32),
        forward_mode=ForwardMode.EXTEND,
        extend_prefix_lens=torch.tensor([100], dtype=torch.int32),
        extend_prefix_lens_cpu=torch.tensor([100], dtype=torch.int32),
        extend_seq_lens=torch.tensor([50], dtype=torch.int32),
        extend_seq_lens_cpu=torch.tensor([50], dtype=torch.int32),
        block_tables={"full_attention": table},
    )
    expected = torch.cat(
        (
            torch.arange(
                3 * _LOGICAL_P + 100,
                3 * _LOGICAL_P + _LOGICAL_P,
                dtype=torch.int64,
            ),
            torch.arange(5 * _LOGICAL_P, 5 * _LOGICAL_P + 22, dtype=torch.int64),
        )
    )
    prefill = backend.forward_prefill_metadata
    assert torch.equal(prefill.group_out_cache_loc, expected)
    assert prefill.chunked_loop_num > 0
    assert (
        backend.select_out_cache_loc(
            SimpleNamespace(layer_id=0),
            torch.full((50,), -1, dtype=torch.int64),
            ForwardMode.EXTEND,
        ).tolist()
        == expected.tolist()
    )


def test_block_decode_expands_one_row_per_block_position() -> None:
    """A block draft's rows all carry the block-end length, which is what makes
    the block non-causal: the kernel masks each row by its own cache length."""
    spec = 4
    backend = _bare_mla_backend(
        cache_contract=False,
        is_draft=True,
        spec_num_tokens=spec,
        draft_block_decode=True,
    )
    assert backend._block_decode_active
    table = torch.tensor([[3, 4], [7, 8]], dtype=torch.int32)
    seq_lens = torch.tensor([9, 130], dtype=torch.int32)

    rows, lens = backend._expand_block_decode_metadata(table, seq_lens, 2)

    assert rows.tolist() == [[3, 4]] * spec + [[7, 8]] * spec
    assert lens.tolist() == [9] * spec + [130] * spec


def test_block_decode_clamps_a_length_below_the_block() -> None:
    """A request shorter than the block would ask the kernel for keys it has
    no page for."""
    spec = 8
    backend = _bare_mla_backend(
        cache_contract=False,
        is_draft=True,
        spec_num_tokens=spec,
        draft_block_decode=True,
    )
    _, lens = backend._expand_block_decode_metadata(
        torch.zeros((2, 2), dtype=torch.int32),
        torch.tensor([1, _MAX_CTX * 4], dtype=torch.int32),
        2,
    )
    assert lens[:spec].tolist() == [spec] * spec
    assert lens[spec:].tolist() == [_MAX_CTX] * spec


def test_block_decode_graph_buffers_hold_every_block_row() -> None:
    spec, max_bs = 4, 3
    backend = _bare_mla_backend(
        cache_contract=False,
        is_draft=True,
        spec_num_tokens=spec,
        draft_block_decode=True,
    )
    backend.init_cuda_graph_state(max_bs)
    assert backend.cuda_graph_seq_lens.shape[0] == max_bs * spec
    assert backend.decode_cuda_graph_kv_indices.shape[0] == max_bs * spec


def test_block_decode_replay_broadcasts_pages_and_scrubs_padding() -> None:
    """Padded rows must reach the null page, not another request's pages."""
    spec, max_bs, bs = 4, 3, 3
    backend = _bare_mla_backend(
        cache_contract=False,
        is_draft=True,
        spec_num_tokens=spec,
        draft_block_decode=True,
    )
    backend.init_cuda_graph_state(max_bs)
    backend.decode_cuda_graph_kv_indices.fill_(-9)
    # Raw scheduler pages (logical P); the backend expands to kernel pages
    # (ratio 2: page L -> [2L, 2L+1]).
    page_table = torch.tensor([[5, 6], [7, 8]], dtype=torch.int32)

    backend._replay_block_decode_page_table(bs, page_table)

    rows = backend.decode_cuda_graph_kv_indices[: bs * spec]
    assert rows[:spec, :4].tolist() == [[10, 11, 12, 13]] * spec
    assert rows[spec : 2 * spec, :4].tolist() == [[14, 15, 16, 17]] * spec
    assert rows[2 * spec :].eq(0).all(), "padded request kept live pages"
    assert rows[: 2 * spec, 4:].eq(0).all(), "columns past the table were not scrubbed"


def test_block_decode_lengths_are_rewritten_per_replay() -> None:
    """The drafter calls this inside the captured graph, so two replays with
    different live draft lengths must not share the first one's rows."""
    spec, max_bs = 4, 2
    backend = _bare_mla_backend(
        cache_contract=False,
        is_draft=True,
        spec_num_tokens=spec,
        draft_block_decode=True,
    )
    backend.init_cuda_graph_state(max_bs)
    buf = backend.cuda_graph_seq_lens

    backend.fill_block_decode_seq_lens(2, torch.tensor([40, 50], dtype=torch.int32))
    assert buf[: 2 * spec].tolist() == [40] * spec + [50] * spec
    backend.fill_block_decode_seq_lens(2, torch.tensor([41, 99], dtype=torch.int32))
    assert buf[: 2 * spec].tolist() == [41] * spec + [99] * spec


def test_block_decode_stays_off_for_target_and_single_token_drafts() -> None:
    """The expansion must be unreachable on every path it was not written for."""
    target = _bare_mla_backend(cache_contract=False, spec_num_tokens=8)
    assert not target._block_decode_active
    one = _bare_mla_backend(
        cache_contract=False,
        is_draft=True,
        spec_num_tokens=1,
        draft_block_decode=True,
    )
    assert not one._block_decode_active


def test_block_decode_hands_the_kernel_one_query_per_block_row() -> None:
    """The expansion is only worth anything if it reaches the kernel: keeping
    the block on the query axis would restore exactly the causal order the
    draft must not have, and every other case here would still pass."""
    spec, bs, heads, dim = 4, 2, 3, 8
    backend = _bare_mla_backend(
        cache_contract=False,
        is_draft=True,
        spec_num_tokens=spec,
        draft_block_decode=True,
    )
    rows, block_lens = backend._expand_block_decode_metadata(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
        torch.tensor([40, 50], dtype=torch.int32),
        bs,
    )
    backend.forward_decode_metadata = CuteDSLMLADecodeMetadata(
        block_kv_indices=rows,
        max_seq_len_k=_MAX_CTX,
        seq_lens_k=block_lens,
        num_extends=0,
        group_out_cache_loc=None,
        group_q_len_per_req=1,
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

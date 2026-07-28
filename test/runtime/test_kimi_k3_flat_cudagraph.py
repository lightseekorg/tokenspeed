"""Kimi-K3 FlatKV CUDA-graph capture/replay core logic.

CPU-only (plain tensors, no real graph capture): exercises the metadata-buffer
capture/replay LOGIC that the decode CUDA graph depends on. The real
graph capture/replay parity on the 93-layer serve is validated on GPU
separately.

Coverage:

- the MLA full-attention decode graph: capture binds stable
  ``block_kv_indices`` + ``flat_out_cache_loc`` buffers, replay refreshes them
  IN PLACE (same ``data_ptr``) from a fresh forward op;
- padded batch rows resolve to the null page 0 (dummy-page protection);
- the ``mark_flat_contract`` structural gate on the contract-bound MLA
  capture/replay path.

The KDA multi-group state capture/replay logic lives in
``test_kimi_k3_flat_kda.py``.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.mla import MLAAttnBackend
from tokenspeed.runtime.layers.attention.backends.tokenspeed_mla import (
    CuteDSLMLABackend,
)

register_cuda_ci(est_time=10, suite="runtime-1gpu")

_PAGE_SIZE = 64  # kernel page
_FLAT_P = 128  # flat block size (ratio 2 kernel pages per flat page)
_MAX_CTX = 256


def _bare_mla_backend(*, flat_contract: bool) -> CuteDSLMLABackend:
    """A CuteDSLMLABackend with only the attributes the CUDA-graph metadata
    paths touch — the full ctor JIT-compiles CuteDSL kernels (GPU only)."""
    backend = object.__new__(CuteDSLMLABackend)
    backend.device = "cpu"
    backend.page_size = _PAGE_SIZE
    backend.max_context_len = _MAX_CTX
    backend.is_draft = False
    backend.spec_num_tokens = 1
    backend._block_table_aliased = False
    backend._flat_bound = False
    backend._flat_contract_bound = False
    backend.decode_cuda_graph_metadata = {}
    backend.decode_cuda_graph_kv_indices = None
    backend.decode_cuda_graph_flat_out_cache_loc = None
    backend.forward_decode_metadata = None
    if flat_contract:
        backend.mark_flat_contract()
    return backend


def _bare_amd_mla_backend(*, flat_contract: bool) -> MLAAttnBackend:
    backend = object.__new__(MLAAttnBackend)
    backend.device = "cpu"
    backend.page_size = _PAGE_SIZE
    backend.max_context_len = _MAX_CTX
    backend.max_num_pages = _MAX_CTX // _PAGE_SIZE
    backend.is_draft = False
    backend.spec_num_tokens = 1
    backend._flat_bound = False
    backend._flat_contract_bound = False
    backend.decode_cuda_graph_metadata = {}
    backend.cuda_graph_page_table = None
    backend.cuda_graph_seq_lens = None
    backend.decode_cuda_graph_flat_out_cache_loc = None
    backend.forward_decode_metadata = None
    if flat_contract:
        backend.mark_flat_contract()
    return backend


class _StubFullAttnMeta:
    """Minimal stand-in for FlatCacheBatchMetadata's MLA surface: a padded
    full-attention table and the flat block size, freshness-checked by op."""

    full_attention_group_id = "full_attention"

    def __init__(self, table: torch.Tensor, block_size: int, forward_op: object):
        self._table = table
        self.block_size = block_size
        self._forward_op = forward_op

    def require_full_attention_table(self, *, active_forward_op):
        if active_forward_op is not self._forward_op:
            raise RuntimeError("stale forward op")
        return self._table


def test_replay_refreshes_buffers_in_place_and_pads_page_zero() -> None:
    backend = _bare_mla_backend(flat_contract=True)
    seq_buf = torch.ones(2, dtype=torch.int32)
    backend.init_cuda_graph_state(max_bs=2, seq_lens_buf=seq_buf)
    backend.init_forward_metadata_capture_cuda_graph(
        bs=2,
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([1, 1], dtype=torch.int32),
        forward_mode=ForwardMode.DECODE,
        flat_cache_group_ids=("full_attention",),
    )
    md = backend.decode_cuda_graph_metadata[2]
    captured_kv_ptr = md.block_kv_indices.data_ptr()
    captured_loc_ptr = md.flat_out_cache_loc.data_ptr()

    # One REAL request (row 0), one padded dummy row (row 1). The op-bound
    # table carries only the real row; padded rows must land on page 0.
    forward_op = object()
    # Flat table: real row 0 has two flat pages [3, 5]; page ids > 0.
    table = torch.tensor([[3, 5]], dtype=torch.int32)
    meta = _StubFullAttnMeta(table, _FLAT_P, forward_op)

    backend.init_forward_metadata_replay_cuda_graph(
        bs=2,  # padded bs
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([70, 1], dtype=torch.int32),  # real seq 70, pad 1
        forward_mode=ForwardMode.DECODE,
        req_to_page=None,
        flat_cache_metadata=meta,
        flat_cache_forward_op=forward_op,
    )
    md2 = backend.forward_decode_metadata
    # SAME buffers refreshed in place (no realloc): pointer-stable replay.
    assert md2.block_kv_indices.data_ptr() == captured_kv_ptr
    assert md2.flat_out_cache_loc.data_ptr() == captured_loc_ptr

    # Real row 0: flat page 3 -> kernel pages [6, 7] (ratio 2), flat page 5 ->
    # [10, 11]. Expansion: page * ratio + k.
    assert md2.block_kv_indices[0].tolist() == [6, 7, 10, 11]
    # Write loc for seq_len 70: pos 69, flat page idx 0 -> page 3, offset 69:
    # 3 * 128 + 69 = 453.
    assert md2.flat_out_cache_loc[0].item() == 3 * _FLAT_P + 69

    # Padded row 1: null page 0 everywhere.
    assert torch.all(md2.block_kv_indices[1] == 0)
    assert md2.flat_out_cache_loc[1].item() == 0


def test_amd_mla_flat_graph_replay_is_pointer_stable_and_null_padded() -> None:
    backend = _bare_amd_mla_backend(flat_contract=True)
    seq_buf = torch.ones(2, dtype=torch.int32)
    backend.init_cuda_graph_state(max_bs=2, seq_lens_buf=seq_buf)
    backend.init_forward_metadata_capture_cuda_graph(
        bs=2,
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=seq_buf,
        forward_mode=ForwardMode.DECODE,
        flat_cache_group_ids=("full_attention",),
    )
    captured = backend.decode_cuda_graph_metadata[2]
    page_ptr = captured.page_table.data_ptr()
    loc_ptr = captured.flat_out_cache_loc.data_ptr()

    forward_op = object()
    metadata = _StubFullAttnMeta(
        torch.tensor([[3, 5]], dtype=torch.int32),
        _FLAT_P,
        forward_op,
    )
    backend.init_forward_metadata_replay_cuda_graph(
        bs=2,
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([70, 1], dtype=torch.int32),
        forward_mode=ForwardMode.DECODE,
        req_to_page=None,
        flat_cache_metadata=metadata,
        flat_cache_forward_op=forward_op,
    )
    replayed = backend.forward_decode_metadata
    assert replayed.page_table.data_ptr() == page_ptr
    assert replayed.flat_out_cache_loc.data_ptr() == loc_ptr
    assert replayed.page_table[0].tolist() == [6, 7, 10, 11]
    assert replayed.flat_out_cache_loc[0].item() == 3 * _FLAT_P + 69
    assert torch.all(replayed.page_table[1] == 0)
    assert replayed.flat_out_cache_loc[1].item() == 0


def test_amd_mla_eager_decode_uses_flat_table_and_refuses_fallback() -> None:
    backend = _bare_amd_mla_backend(flat_contract=True)
    forward_op = object()
    metadata = _StubFullAttnMeta(
        torch.tensor([[3, 5], [4, -1]], dtype=torch.int32),
        _FLAT_P,
        forward_op,
    )
    seq_lens = torch.tensor([70, 40], dtype=torch.int32)
    poisoned = torch.full((8, 8), -99, dtype=torch.int32)
    backend.init_forward_metadata(
        bs=2,
        num_extends=0,
        req_pool_indices=torch.tensor([-99, -99], dtype=torch.int64),
        seq_lens=seq_lens,
        req_to_page=poisoned,
        forward_mode=ForwardMode.DECODE,
        flat_cache_metadata=metadata,
        flat_cache_forward_op=forward_op,
    )
    decode = backend.forward_decode_metadata
    assert decode.page_table[0].tolist() == [6, 7, 10, 11]
    assert decode.page_table[1].tolist() == [8, 9, 0, 1]
    assert decode.flat_out_cache_loc.tolist() == [
        3 * _FLAT_P + 69,
        4 * _FLAT_P + 39,
    ]
    selected = backend.select_out_cache_loc(
        SimpleNamespace(layer_id=0),
        torch.tensor([-1, -1], dtype=torch.int64),
        ForwardMode.DECODE,
    )
    assert torch.equal(selected, decode.flat_out_cache_loc)

    with pytest.raises(RuntimeError, match="received no flat cache metadata"):
        backend.init_forward_metadata(
            bs=2,
            num_extends=0,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            seq_lens=seq_lens,
            req_to_page=torch.zeros((2, 4), dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
        )


def test_amd_mla_eager_prefill_derives_flat_write_locations(monkeypatch) -> None:
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
    backend = _bare_amd_mla_backend(flat_contract=True)
    forward_op = object()
    metadata = _StubFullAttnMeta(
        torch.tensor([[3, 5]], dtype=torch.int32),
        _FLAT_P,
        forward_op,
    )
    backend.init_forward_metadata(
        bs=1,
        num_extends=1,
        req_pool_indices=torch.tensor([-99], dtype=torch.int64),
        seq_lens=torch.tensor([150], dtype=torch.int32),
        req_to_page=torch.full((4, 4), -99, dtype=torch.int32),
        forward_mode=ForwardMode.EXTEND,
        extend_prefix_lens=torch.tensor([100], dtype=torch.int32),
        extend_prefix_lens_cpu=torch.tensor([100], dtype=torch.int32),
        extend_seq_lens=torch.tensor([50], dtype=torch.int32),
        extend_seq_lens_cpu=torch.tensor([50], dtype=torch.int32),
        flat_cache_metadata=metadata,
        flat_cache_forward_op=forward_op,
    )
    expected = torch.cat(
        (
            torch.arange(
                3 * _FLAT_P + 100,
                3 * _FLAT_P + _FLAT_P,
                dtype=torch.int64,
            ),
            torch.arange(5 * _FLAT_P, 5 * _FLAT_P + 22, dtype=torch.int64),
        )
    )
    prefill = backend.forward_prefill_metadata
    assert torch.equal(prefill.flat_out_cache_loc, expected)
    assert prefill.chunked_loop_num > 0
    assert (
        backend.select_out_cache_loc(
            SimpleNamespace(layer_id=0),
            torch.full((50,), -1, dtype=torch.int64),
            ForwardMode.EXTEND,
        ).tolist()
        == expected.tolist()
    )

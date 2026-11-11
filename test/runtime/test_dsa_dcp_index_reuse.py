# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""DCP top-k builds its index metadata once per forward, with fakes on CPU.

The first indexer layer of a forward builds the local page table, the row
geometry and the compact owned pages; later layers of the same forward reuse
them through the shared decode top-k record. Drafts, other tables, other
verify widths and other row counts rebuild. The decode top-k scores in one
tile.

A KVP prefill does the same through the shared prefill record, at the owned width the DSA leaf plans from the
host page table; the owned-width pages are planned once per forward only
within the memory cap (otherwise every tile plans its own).
"""

import os
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed_kernel.ops.attention.dsa import (  # noqa: E402
    DSAIndexCandidatePages,
)

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode  # noqa: E402
from tokenspeed.runtime.layers.attention.backends.paged import dsa  # noqa: E402
from tokenspeed.runtime.layers.attention.dcp import indexer  # noqa: E402
from tokenspeed.runtime.layers.attention.dcp.indexer import (  # noqa: E402
    DCPIndexMetadata,
)
from tokenspeed.runtime.layers.attention.dcp.placement import (  # noqa: E402
    CachePlacement,
)
from tokenspeed.runtime.models import glm5  # noqa: E402

_TOPK = 16
_PLACEMENT = CachePlacement(128, 64, (0, 1, 2, 3), 2)


@pytest.fixture
def recorded(monkeypatch):
    """Count metadata builds; record what every select receives."""
    builds, selects = [], []

    def build(
        q, weights, index_k_cache, page_table, requests, causal, *, placement, **kw
    ):
        metadata = DCPIndexMetadata(
            page_table,
            page_table.clone(),
            requests.contiguous(),
            causal.contiguous(),
            None,
            placement,
        )
        builds.append(metadata)
        return metadata

    def select(q, weights, index_k_cache, page_table, requests, causal, **kw):
        metadata = kw["metadata"]
        assert requests is metadata.query_requests
        assert causal is metadata.causal_lens
        selects.append((metadata, kw["max_logits_bytes"]))
        rows = q.shape[0]
        return (
            torch.zeros((rows, _TOPK), dtype=torch.int32),
            torch.zeros(rows, dtype=torch.int32),
        )

    monkeypatch.setattr(glm5, "build_dcp_index_metadata", build)
    monkeypatch.setattr(glm5, "select_dsa_topk", select)
    monkeypatch.setattr(glm5, "local_topk_to_global_slots", lambda **kw: None)
    return builds, selects


def _layer():
    layer = object.__new__(glm5.GlmDsaSparseAttentionMixin)
    layer.attn_mqa = SimpleNamespace(layer_id=3)
    layer.indexer = SimpleNamespace(weights_softmax_scale=0.25)
    layer._decode_topk_indices_buffer = None
    layer._decode_topk_lens_buffer = None
    layer._retired_decode_workspaces = []
    return layer


def _ctx(seq_lens, q_len, *, is_draft=False, placement=_PLACEMENT):
    """The router surface the decode top-k reads, for one forward."""
    metadata = SimpleNamespace(
        _dsa_seq_lens_2d=seq_lens.repeat_interleave(q_len).unsqueeze(1)
    )
    backend = SimpleNamespace(
        forward_decode_metadata=metadata,
        cache_placement=lambda layer: placement,
        sparse_topk=SimpleNamespace(prefill=None, decode=None),
        is_draft=is_draft,
    )
    pool = SimpleNamespace(
        get_index_k_buffer=lambda layer_id: torch.zeros(128, 132, dtype=torch.uint8),
        index_k_page_size=64,
    )
    return SimpleNamespace(attn_backend=backend, token_to_kv_pool=pool, num_extends=0)


def _decode_layer(layer, ctx, page_table, seq_lens, q_len):
    """One indexer layer: compute the decode top-k and publish it, as
    ``forward_dsa_core`` does."""
    rows = seq_lens.numel() * q_len
    query = torch.zeros(rows, 4, 128, dtype=torch.bfloat16)
    output = SimpleNamespace(
        query_rows=lambda start, end: query[start:end],
        weights=torch.zeros(rows, 4),
        quantized_query=None,
    )
    record = layer._compute_decode_topk_indices_portable(
        indexer_output=output,
        ctx=ctx,
        seq_lens=seq_lens,
        page_table=page_table,
        q_len_per_req=q_len,
        decode_start=0,
        num_tokens=rows,
        num_decode_tokens=rows,
        topk=_TOPK,
    )
    ctx.attn_backend.sparse_topk.decode = record
    return record


def _batch(requests=3, q_len=8):
    seq_lens = torch.tensor([9, 300, 1100][:requests], dtype=torch.int32)
    page_table = torch.arange(2, 2 + requests * 20, dtype=torch.int32).reshape(
        requests, 20
    )
    return seq_lens, page_table, q_len


def test_consecutive_indexer_layers_build_once(recorded):
    builds, selects = recorded
    seq_lens, page_table, q_len = _batch()
    ctx = _ctx(seq_lens, q_len)
    first = _decode_layer(_layer(), ctx, page_table, seq_lens, q_len)
    # The next layer slices the same decode table again: a new view object.
    second = _decode_layer(_layer(), ctx, page_table[:], seq_lens, q_len)
    third = _decode_layer(_layer(), ctx, page_table[:3], seq_lens, q_len)

    assert len(builds) == 1
    assert all(metadata is builds[0] for metadata, _ in selects)
    assert first.dcp_index is second.dcp_index is third.dcp_index is builds[0]
    assert first.dcp_source is ctx.attn_backend.forward_decode_metadata
    assert first.dcp_q_len == q_len
    assert first.dcp_capturing is False
    # Verify row j of a request sees its history up to its own position.
    requests = torch.arange(3 * q_len, dtype=torch.int32) // q_len
    causal = seq_lens[requests.long()] - (q_len - 1) + torch.arange(3 * q_len) % q_len
    torch.testing.assert_close(builds[0].query_requests, requests)
    torch.testing.assert_close(builds[0].causal_lens, causal)
    assert builds[0].page_table is page_table


@pytest.mark.parametrize(
    "change",
    [
        "draft",
        "table_values",
        "table_rows",
        "decode_metadata",
        "q_len",
        "rows",
        "placement",
        "no_placement_record",
        "capture_state",
    ],
)
def test_other_forward_inputs_rebuild(recorded, change):
    builds, _ = recorded
    seq_lens, page_table, q_len = _batch()
    ctx = _ctx(seq_lens, q_len)
    record = _decode_layer(_layer(), ctx, page_table, seq_lens, q_len)
    share = ctx.attn_backend.sparse_topk
    if change == "draft":
        # A drafter's lengths advance in place, so its record cannot be tied to them.
        ctx.attn_backend.is_draft = True
    elif change == "table_values":
        page_table = page_table.clone()
    elif change == "table_rows":
        seq_lens, page_table = seq_lens[:2], page_table[:2]
    elif change == "decode_metadata":
        ctx.attn_backend.forward_decode_metadata = SimpleNamespace(
            **vars(ctx.attn_backend.forward_decode_metadata)
        )
    elif change == "q_len":
        share.decode = replace(record, dcp_q_len=q_len - 1)
    elif change == "rows":
        index = record.dcp_index
        share.decode = replace(
            record,
            dcp_index=replace(
                index,
                query_requests=index.query_requests[:-1],
                causal_lens=index.causal_lens[:-1],
            ),
        )
    elif change == "placement":
        ctx.attn_backend.cache_placement = lambda layer: replace(_PLACEMENT, rank=1)
    elif change == "no_placement_record":
        share.decode = replace(record, dcp_index=None, dcp_source=None)
    elif change == "capture_state":
        # A record from the other capture state (eager warm-up) is not reused.
        share.decode = replace(record, dcp_capturing=True)
    second = _decode_layer(_layer(), ctx, page_table, seq_lens, q_len)

    assert len(builds) == 2
    assert second.dcp_index is builds[1]


def test_decode_scores_every_row_in_one_tile(recorded):
    _, selects = recorded
    seq_lens, page_table, q_len = _batch(requests=1, q_len=1)
    _decode_layer(_layer(), _ctx(seq_lens, q_len), page_table, seq_lens, q_len)

    [(_, max_logits_bytes)] = selects
    assert max_logits_bytes is None


def test_unsharded_decode_keeps_no_dcp_record(monkeypatch):
    seq_lens, page_table, q_len = _batch(requests=1, q_len=1)
    ctx = _ctx(seq_lens, q_len, placement=None)
    calls = []

    def decode_topk(q, weights, seq_lens, page_table, **kw):
        calls.append(kw["q_len_per_req"])
        return kw["out"], kw["lens_out"]

    monkeypatch.setattr(glm5, "dsa_decode_topk", decode_topk)
    monkeypatch.setattr(
        glm5, "build_dcp_index_metadata", lambda *a, **kw: pytest.fail("built")
    )
    layer = _layer()
    layer._scorer_index_query = lambda q, cache: (q, None)
    ctx.attn_backend.forward_decode_metadata._dsa_plan = None
    monkeypatch.setitem(glm5.global_server_args_dict, "dsa_slot_order", "selection")
    record = _decode_layer(layer, ctx, page_table, seq_lens, q_len)

    assert calls == [q_len]
    assert record.dcp_index is None and record.dcp_source is None


# --- KVP prefill: owned-width index metadata ----------------------------------

_PREFIX = 100


@pytest.fixture
def owned_prefill(monkeypatch):
    """Run the real metadata builder with the kernel package's leaf choice
    and page planning stood in for (no GPU); record the plans and what every
    select receives."""
    plans, selects = [], []

    def plan(q, weights, index_k_cache, local_table, requests, causal, **kw):
        assert local_table.shape[1] >= kw["columns"]
        plans.append((q.shape[0], kw["columns"]))
        empty = torch.zeros((0, 0), dtype=torch.int32)
        return DSAIndexCandidatePages("fake", empty, empty, empty, empty)

    def select(q, weights, index_k_cache, page_table, requests, causal, **kw):
        selects.append((kw["metadata"], requests, causal))
        rows = q.shape[0]
        return (
            torch.zeros((rows, _TOPK), dtype=torch.int32),
            torch.zeros(rows, dtype=torch.int32),
        )

    monkeypatch.setattr(
        indexer, "dsa_index_candidates_accepts_pages", lambda *a, **kw: True
    )
    monkeypatch.setattr(indexer, "plan_dsa_index_candidates", plan)
    monkeypatch.setattr(glm5, "select_dsa_topk", select)
    return plans, selects


def _prefill_ctx(columns):
    """The router surface the KVP prefill top-k reads, for one forward."""
    backend = SimpleNamespace(
        cache_placement=lambda layer: _PLACEMENT,
        sparse_topk=SimpleNamespace(prefill=None, decode=None),
        prefill_index_columns=lambda: columns,
    )
    pool = SimpleNamespace(
        get_index_k_buffer=lambda layer_id: torch.zeros(128, 132, dtype=torch.uint8),
        index_k_page_size=64,
    )
    return SimpleNamespace(attn_backend=backend, token_to_kv_pool=pool, num_extends=1)


def _prefill_table(columns):
    return torch.arange(2, 2 + columns, dtype=torch.int32).unsqueeze(0)


def _first_prefill_layer(ctx, page_table, rows):
    """The forward's first indexer layer: one request continuing a cached
    prefix, chunk row ``j`` seeing ``_PREFIX + j + 1`` positions. Publishes
    its record, as ``forward_dsa_core`` does."""
    layer = _layer()
    layer.index_topk = _TOPK
    seq_len = _PREFIX + rows
    row_starts = torch.zeros(rows, dtype=torch.int32)
    row_ends = torch.arange(_PREFIX + 1, seq_len + 1, dtype=torch.int32)
    output = glm5.GlmDsaIndexerOutput(
        query=torch.zeros(rows, 1, 8, dtype=torch.bfloat16),
        key=torch.zeros(rows, 8, dtype=torch.bfloat16),
        weights=torch.zeros(rows, 1),
    )
    record = layer._compute_prefill_topk_indices_portable(
        indexer_output=output,
        ctx=ctx,
        seq_lens=torch.tensor([seq_len], dtype=torch.int32),
        page_table=page_table,
        kv_workspace_slots=torch.arange(seq_len),
        row_starts=row_starts,
        row_ends=row_ends,
        candidate_lens_cpu=(row_ends - row_starts).long(),
        kv_seq_lens=row_ends - row_starts,
        max_seq_len=seq_len,
        num_prefill_tokens=rows,
        topk=_TOPK,
        dcp_index=None,
    )
    ctx.attn_backend.sparse_topk.prefill = record
    return layer, output, record


def test_prefill_owned_width_pages_built_once_per_forward(owned_prefill):
    plans, selects = owned_prefill
    rows = 28
    ctx = _prefill_ctx(32)
    page_table = _prefill_table(64)
    layer, output, first = _first_prefill_layer(ctx, page_table, rows)
    # Later indexer layers of the forward take the cached prefill path.
    records = [first]
    for _ in range(2):
        record = layer._compute_prefill_topk_indices(output, ctx, rows)
        ctx.attn_backend.sparse_topk.prefill = record
        records.append(record)

    metadata = first.dcp_index
    assert plans == [(rows, 32)]
    assert metadata.columns == 32 and metadata.pages is not None
    assert all(record.dcp_index is metadata for record in records)
    for selected, requests, causal in selects:
        assert selected is metadata
        assert requests is metadata.query_requests
        assert causal is metadata.causal_lens
    torch.testing.assert_close(
        metadata.query_requests, torch.zeros(rows, dtype=torch.int32)
    )
    torch.testing.assert_close(
        metadata.causal_lens,
        torch.arange(_PREFIX + 1, _PREFIX + rows + 1, dtype=torch.int32),
    )
    assert metadata.page_table is page_table

    # A bound wider than the table clamps to its width; another forward's table replans.
    ctx = _prefill_ctx(96)
    _, _, record = _first_prefill_layer(ctx, _prefill_table(64), rows)
    assert record.dcp_index.columns == 64
    assert plans == [(rows, 32), (rows, 64)]


@pytest.mark.parametrize("columns,planned", [(2048, True), (2064, False)])
def test_prefill_owned_width_cache_respects_memory_cap(owned_prefill, columns, planned):
    plans, selects = owned_prefill
    rows = 16384
    # Pages and positions take 8 bytes per row and column, so this is the 256 MiB cap.
    assert rows * 2048 * 8 == 256 << 20
    ctx = _prefill_ctx(columns)
    _, _, record = _first_prefill_layer(ctx, _prefill_table(4096), rows)

    metadata = record.dcp_index
    # Past the cap the table and row geometry stay shared; each tile plans its pages.
    assert metadata.columns == columns
    assert (metadata.pages is not None) == planned
    assert plans == ([(rows, columns)] if planned else [])
    [(selected, _, _)] = selects
    assert selected is metadata


def test_prefill_keeps_the_table_width_without_owned_columns(owned_prefill):
    plans, selects = owned_prefill
    rows = 28
    ctx = _prefill_ctx(None)
    _, _, record = _first_prefill_layer(ctx, _prefill_table(64), rows)

    assert record.dcp_index is None and not plans
    [(selected, requests, causal)] = selects
    assert selected is None
    torch.testing.assert_close(requests, torch.zeros(rows, dtype=torch.int32))
    torch.testing.assert_close(
        causal, torch.arange(_PREFIX + 1, _PREFIX + rows + 1, dtype=torch.int32)
    )


# --- the DSA leaf's owned width ----------------------------------------------


def _kvp_leaf(rank, *, degree=4):
    """A GPU DSA leaf over KVP pages, as far as its extend metadata reads."""
    backend = object.__new__(dsa.DSABackend)
    backend.kernel_page_size = 64
    backend.index_topk = 2048
    backend.is_draft = False
    backend.kvp_group = tuple(range(degree))
    backend.kvp_rank = rank
    backend.kvp_block_granularity = 128
    backend.kvp_virtual_block_count = 1 + 64 * degree
    backend.query_shard_metadata = None
    backend._prefill_page_table = None
    backend._prefill_index_columns = None
    backend._dense_backend = SimpleNamespace(
        init_forward_metadata=lambda *args, **kwargs: None,
        chunked_prefill_metadata=None,
    )
    return backend


def _owned_width_table():
    """Two requests of 128-token blocks (two 64-token pages each): 6400
    positions whose first 40 blocks are rank 0's and next 10 rank 1's, and a
    cold 300-position chunk on rank 2's blocks."""
    long = [1 + 4 * i for i in range(40)] + [2 + 4 * i for i in range(10)]
    cold = [3 + 4 * i for i in range(3)]
    table = torch.zeros((2, 104), dtype=torch.int32)
    for row, blocks in enumerate((long, cold)):
        pages = [2 * block + sub for block in blocks for sub in range(2)]
        table[row, : len(pages)] = torch.tensor(pages, dtype=torch.int32)
    return table, [6000, 0], [400, 300]


def _init_extend(backend, table, prefix, extend, *, forward_mode, page_table_cpu):
    bs = len(prefix)
    lengths = torch.tensor([p + e for p, e in zip(prefix, extend)], dtype=torch.int32)
    backend.init_forward_metadata(
        bs,
        bs,
        lengths,
        table,
        forward_mode,
        extend_seq_lens=torch.tensor(extend, dtype=torch.int32),
        extend_seq_lens_cpu=torch.tensor(extend, dtype=torch.int32),
        extend_prefix_lens=torch.tensor(prefix, dtype=torch.int32),
        extend_prefix_lens_cpu=torch.tensor(prefix, dtype=torch.int32),
        extend_with_prefix=any(prefix),
        query_shard=None,
        page_table_cpu=page_table_cpu,
    )


@pytest.mark.parametrize("rank", range(4))
def test_owned_index_columns_follow_the_host_table(rank):
    table, prefix, extend = _owned_width_table()
    backend = _kvp_leaf(rank)
    _init_extend(
        backend,
        table,
        prefix,
        extend,
        forward_mode=ForwardMode.EXTEND,
        page_table_cpu=table,
    )
    # Rank 0 owns 80 pages of the long request: every rank tiles by that.
    assert backend.prefill_index_columns() == 80

    # A short forward still spans topk positions (32 pages of 64).
    _init_extend(
        backend,
        table[1:],
        [0],
        [300],
        forward_mode=ForwardMode.EXTEND,
        page_table_cpu=table[1:],
    )
    assert backend.prefill_index_columns() == 32

    # The width is per forward: an idle forward clears it.
    empty = torch.zeros((0, 104), dtype=torch.int32)
    _init_extend(
        backend, empty, [], [], forward_mode=ForwardMode.IDLE, page_table_cpu=None
    )
    assert backend.prefill_index_columns() is None


@pytest.mark.parametrize("case", ["no_host_table", "unsharded"])
def test_owned_index_columns_need_kvp_and_the_host_table(case):
    table, prefix, extend = _owned_width_table()
    backend = _kvp_leaf(0, degree=1 if case == "unsharded" else 4)
    _init_extend(
        backend,
        table,
        prefix,
        extend,
        forward_mode=ForwardMode.EXTEND,
        page_table_cpu=None if case == "no_host_table" else table,
    )
    assert backend.prefill_index_columns() is None

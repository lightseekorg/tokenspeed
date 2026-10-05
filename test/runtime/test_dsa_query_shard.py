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

"""The GPU DSA leaf's sharded extend arm, with fakes on CPU.

Covers the per-forward plan (request groups against the gather workspace,
this rank's query slice of each group, the gather split), the gathered-buffer
attention call shape (``return_lse=False``, group-relative top-k rows, every
group's gather run even without local rows), the index-K history gather in
either plane format (FP8 bytes with scales, bf16 keys), the dense delegate's
view of the forward, the decode arm's head-replicated combine, and the
workspace reservation agreeing with the recipe's plan in either format.
"""

from __future__ import annotations

from test.runtime.dsa_index_k_test_utils import (
    expected_index_k_rows,
    index_k_pool,
    write_index_k_plane,
)
from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.query_shard import QueryShardPlan
from tokenspeed.runtime.layers.attention.backends.paged import dsa
from tokenspeed.runtime.layers.attention.configs.dsa import (
    INDEX_K_FORMATS,
    dsa_history_gather_workspace_bytes,
    dsa_index_k_row_bytes,
    index_k_row_bytes,
)
from tokenspeed.runtime.layers.attention.kv_cache.dsa import split_index_k_rows
from tokenspeed.runtime.layers.attention.page_table import (
    build_prefill_kv_workspace_slots,
)

PAGE = 2
KV_DIM = 8
INDEX_HEAD_DIM = 128
WORLD = 4
# Three requests: prefix + chunk rows. The chunk rows (extend lengths) make a
# 10-row span sharded [3, 3, 2, 2] over four ranks.
EXTEND = [4, 1, 5]
PREFIX = [2, 4, 0]
HISTORY = [p + e for p, e in zip(PREFIX, EXTEND)]  # [6, 5, 5]


def _backend(
    rank: int,
    *,
    workspace_rows: int,
    index_k_format: str,
    qcp: bool = True,
) -> dsa.DSABackend:
    backend = object.__new__(dsa.DSABackend)
    backend.kernel_page_size = PAGE
    backend.kernel_solution = None
    backend.slot_order = "selection"
    backend.data_type = torch.bfloat16
    backend.q_data_type = torch.bfloat16
    backend.kv_lora_rank = KV_DIM - 2
    backend.qk_nope_head_dim = 4
    backend.qk_rope_head_dim = 2
    backend.kv_cache_dim = KV_DIM
    backend.index_head_dim = INDEX_HEAD_DIM
    backend.index_k_format = index_k_format
    backend.index_topk = 4
    backend.max_context_len = 64
    # Two heads in the model, one per attention-TP slice: a query of two
    # heads carries every head, of one head the slice.
    backend.num_attention_heads = 2
    backend.num_local_heads = 1
    backend.device = "cpu"
    backend.is_draft = False
    backend.spec_num_tokens = 1
    backend.step_counter = None
    backend.kpool_runtime = None
    backend.dcp_group = (rank,)
    backend.dcp_rank = 0
    backend.dcp_block_granularity = None
    backend.dcp_virtual_block_count = None
    backend.qcp_group = tuple(range(WORLD)) if qcp else (0,)
    backend.qcp_rank = rank if qcp else 0
    backend.query_shard_metadata = None
    backend._prefill_page_table = None
    backend._history_workspace = None
    backend._dense_backend = SimpleNamespace(
        init_forward_metadata=lambda *args, **kwargs: None,
        chunked_prefill_metadata=None,
        forward_decode_metadata=None,
    )
    if workspace_rows:
        backend.preallocate_history_gather_workspace(workspace_rows)
    return backend


def _page_table() -> torch.Tensor:
    """Kernel pages per request (page 0 is the hole); history lengths [6, 5, 5]."""
    return torch.tensor([[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0]], dtype=torch.int32)


def _plan(rank: int) -> QueryShardPlan:
    return QueryShardPlan.from_forward(
        total_tokens=sum(EXTEND), input_lengths=EXTEND, size=WORLD, rank=rank
    )


def _init(backend: dsa.DSABackend, plan: QueryShardPlan) -> None:
    table = _page_table()
    backend.init_forward_metadata(
        3,
        3,
        torch.tensor(HISTORY, dtype=torch.int32),
        table,
        ForwardMode.EXTEND,
        extend_seq_lens=torch.tensor(EXTEND, dtype=torch.int32),
        extend_seq_lens_cpu=torch.tensor(EXTEND, dtype=torch.int32),
        extend_prefix_lens=torch.tensor(PREFIX, dtype=torch.int32),
        extend_prefix_lens_cpu=torch.tensor(PREFIX, dtype=torch.int32),
        extend_with_prefix=True,
        query_shard=plan,
        page_table_cpu=table,
    )


@pytest.mark.parametrize("rank", range(WORLD))
def test_the_plan_groups_requests_and_slices_this_ranks_queries(rank):
    backend = _backend(
        rank, workspace_rows=11, index_k_format="fp8_scaled"
    )  # 6 + 5 fit, 6 + 5 + 5 do not
    plan = _plan(rank)
    delegate_calls = []
    backend._dense_backend.init_forward_metadata = (
        lambda *a, **kw: delegate_calls.append(kw)
    )
    _init(backend, plan)
    # The dense delegate sees the whole span and no shard.
    assert delegate_calls and delegate_calls[0]["query_shard"] is None
    assert delegate_calls[0]["page_table_cpu"] is None
    meta = backend.require_query_shard_metadata()
    assert meta.plan is plan
    assert [g.requests for g in meta.groups] == [slice(0, 2), slice(2, 3)]
    assert [g.rows for g in meta.groups] == [11, 5]
    assert [g.row_base for g in meta.groups] == [0, 11]
    # Query rows 0-4 belong to the first group, 5-9 to the second; each
    # rank's slice is its shard intersected with the group's span.
    start, end = plan.local_start, plan.local_end
    for group, (lo, hi) in zip(meta.groups, ((0, 5), (5, 10))):
        expected = slice(
            min(max(lo, start), end) - start, min(max(hi, start), end) - start
        )
        assert group.local_query == expected
    # One owner (no DCP): the gather is local and holds every history row.
    for group in meta.groups:
        assert group.gather.group == (0,)
        assert group.gather.owned_rows_per_rank == (group.rows,)
        assert group.gather.virtual_slots.numel() == group.rows


def test_a_history_over_the_workspace_is_refused_and_an_unsharded_init_clears():
    backend = _backend(
        0, workspace_rows=3, index_k_format="fp8_scaled"
    )  # four rows once padded to pages
    with pytest.raises(RuntimeError, match="exceeds"):
        _init(backend, _plan(0))
    backend = _backend(0, workspace_rows=11, index_k_format="fp8_scaled")
    _init(backend, _plan(0))
    assert backend.query_shard_metadata is not None
    backend.init_forward_metadata(
        3,
        3,
        torch.tensor(HISTORY, dtype=torch.int32),
        _page_table(),
        ForwardMode.EXTEND,
        extend_seq_lens=torch.tensor(EXTEND, dtype=torch.int32),
        extend_seq_lens_cpu=torch.tensor(EXTEND, dtype=torch.int32),
        extend_prefix_lens=torch.tensor(PREFIX, dtype=torch.int32),
        extend_prefix_lens_cpu=torch.tensor(PREFIX, dtype=torch.int32),
        extend_with_prefix=True,
        query_shard=None,
        page_table_cpu=None,
    )
    assert backend.query_shard_metadata is None
    with pytest.raises(RuntimeError, match="not a sharded"):
        backend.require_query_shard_metadata()


def test_a_sharded_init_needs_the_host_table_and_the_workspace():
    backend = _backend(0, workspace_rows=11, index_k_format="fp8_scaled")
    with pytest.raises(RuntimeError, match="host page table"):
        backend.init_forward_metadata(
            3,
            3,
            torch.tensor(HISTORY, dtype=torch.int32),
            _page_table(),
            ForwardMode.EXTEND,
            extend_seq_lens=torch.tensor(EXTEND, dtype=torch.int32),
            extend_seq_lens_cpu=torch.tensor(EXTEND, dtype=torch.int32),
            extend_prefix_lens=torch.tensor(PREFIX, dtype=torch.int32),
            extend_prefix_lens_cpu=torch.tensor(PREFIX, dtype=torch.int32),
            extend_with_prefix=True,
            query_shard=_plan(0),
            page_table_cpu=None,
        )
    backend = _backend(0, workspace_rows=0, index_k_format="fp8_scaled")
    with pytest.raises(RuntimeError, match="workspace"):
        _init(backend, _plan(0))


@pytest.mark.parametrize("rank", range(WORLD))
def test_the_sharded_arm_attends_gathered_groups_with_full_heads(monkeypatch, rank):
    backend = _backend(rank, workspace_rows=11, index_k_format="fp8_scaled")
    plan = _plan(rank)
    _init(backend, plan)
    meta = backend.require_query_shard_metadata()

    # The pool: a flat latent plane whose row v is v (position-identifying).
    plane = torch.arange(40, dtype=torch.float32).unsqueeze(1).expand(40, KV_DIM)
    plane = plane.to(torch.bfloat16).unsqueeze(1).contiguous()  # [slots, 1, dim]
    pool = SimpleNamespace(quant_method=None, get_key_buffer=lambda layer_id: plane)
    layer = SimpleNamespace(
        layer_id=0,
        tp_q_head_num=2,
        head_dim=KV_DIM,
        v_head_dim=KV_DIM - 2,
        scaling=0.5,
        logit_cap=0.0,
    )
    local_rows = plan.local_rows
    q = torch.randn(local_rows, 2 * KV_DIM, dtype=torch.bfloat16)
    # Workspace rows: history rows numbered request-major over all requests.
    topk = torch.full((local_rows, 4), -1, dtype=torch.int32)
    for j in range(local_rows):
        topk[j, 0] = plan.local_start + j  # some absolute workspace row
    topk_lens = torch.ones(local_rows, dtype=torch.int32)
    kv_seq_lens = torch.full((local_rows,), 3, dtype=torch.int32)

    calls = []

    def fake_prefill(**kwargs):
        # The buffer is a workspace view the next group's gather overwrites.
        calls.append({**kwargs, "kv_cache": kwargs["kv_cache"].clone()})
        rows = kwargs["q"].shape[0]
        return torch.full(
            (rows, 2, KV_DIM - 2), float(len(calls)), dtype=torch.bfloat16
        )

    gathers = []
    real_gather = backend.gather_history_kv

    def counting_gather(layer, pool, group):
        gathers.append(group)
        return real_gather(layer, pool, group)

    monkeypatch.setattr(dsa, "dsa_prefill", fake_prefill)
    monkeypatch.setattr(backend, "gather_history_kv", counting_gather)
    out = backend.forward_sparse_prefill(
        q=q,
        layer=layer,
        token_to_kv_pool=pool,
        kv_seq_lens=kv_seq_lens,
        topk_slots=topk,
        topk_lens=topk_lens,
        max_seq_len=6,
    )
    # Every group's gather ran on every rank, attention only where rows exist.
    assert gathers == list(meta.groups)
    attended = [g for g in meta.groups if g.local_query.stop > g.local_query.start]
    assert len(calls) == len(attended)
    assert out.shape == (local_rows, 2 * (KV_DIM - 2))
    for call, group in zip(calls, attended):
        rows = group.local_query
        assert call["return_lse"] is False
        # The buffer is a whole number of kernel pages (every dsa_prefill
        # solution's flat view holds) holding the group's history in
        # position order in its leading rows.
        padded = -(-group.rows // PAGE) * PAGE
        assert call["kv_cache"].shape == (padded, KV_DIM)
        assert padded % PAGE == 0 and group.rows <= padded < group.rows + PAGE
        expected = plane[group.gather.virtual_slots, 0]
        torch.testing.assert_close(
            call["kv_cache"][: group.rows], expected, rtol=0, atol=0
        )
        # Top-k rows are re-based to the group's buffer, -1 stays -1.
        expected_slots = topk[rows].clone()
        expected_slots[:, 0] -= group.row_base
        assert torch.equal(call["topk_slots"], expected_slots)
        assert torch.equal(call["topk_lens"], topk_lens[rows])
        assert torch.equal(call["kv_seq_lens"], kv_seq_lens[rows])
        assert call["q"].shape[0] == rows.stop - rows.start
    if local_rows == 0:
        assert not calls


def test_the_sharded_arm_refuses_a_query_carrying_the_tp_slice():
    """The gathered history is attended with every head: a query carrying the
    attention-TP slice (the head-sharded form without the exchange in front
    of it) is a layout bug, refused before any gather -- whatever head count
    the layer declares."""
    backend = _backend(1, workspace_rows=11, index_k_format="fp8_scaled")
    plan = _plan(1)
    _init(backend, plan)
    layer = SimpleNamespace(
        layer_id=0,
        tp_q_head_num=2,
        head_dim=KV_DIM,
        v_head_dim=KV_DIM - 2,
        scaling=0.5,
        logit_cap=0.0,
    )
    rows = plan.local_rows
    with pytest.raises(RuntimeError, match="every head"):
        backend.forward_sparse_prefill(
            q=torch.randn(rows, KV_DIM, dtype=torch.bfloat16),
            layer=layer,
            token_to_kv_pool=SimpleNamespace(quant_method=None),
            kv_seq_lens=None,
            topk_slots=torch.full((rows, 4), -1, dtype=torch.int32),
            topk_lens=torch.ones(rows, dtype=torch.int32),
            max_seq_len=6,
        )


def test_the_dense_delegate_is_refused_under_a_shard():
    backend = _backend(0, workspace_rows=11, index_k_format="fp8_scaled")
    _init(backend, _plan(0))
    with pytest.raises(RuntimeError, match="forward_sparse_prefill"):
        backend.forward_extend_chunked(
            None,
            None,
            None,
            0.5,
            0.0,
            cum_seq_lens_q=None,
            cum_seq_lens_kv=None,
            max_q_len=1,
            max_kv_len=1,
            seq_lens=None,
            batch_size=1,
            causal=True,
        )


def _written_index_k_pool(index_k_format: str, page_size: int, pages: int):
    """A pool over a plane of ``pages`` pages written through the production
    write path, slot ``v`` holding the key ``v + i / 4`` (``i`` the element),
    with the rows the read side must hand back for every slot."""
    slots = pages * page_size
    keys = (
        torch.arange(slots, dtype=torch.float32).unsqueeze(1)
        + torch.arange(INDEX_HEAD_DIM, dtype=torch.float32) / 4
    ).to(torch.bfloat16)
    plane = write_index_k_plane(
        index_k_format,
        head_dim=INDEX_HEAD_DIM,
        page_size=page_size,
        slots=slots,
        loc=torch.arange(slots, dtype=torch.int64),
        keys=keys,
    )
    pool = index_k_pool(plane, head_dim=INDEX_HEAD_DIM, page_size=page_size)
    return pool, expected_index_k_rows(index_k_format, keys)


@pytest.mark.parametrize("index_k_format", INDEX_K_FORMATS)
def test_the_index_k_gather_reads_back_what_the_pool_wrote(index_k_format):
    """``gather_index_k_rows`` is the read side of ``set_index_k_buffer``: the
    packed row of a slot is the bytes the write stored for it (FP8 bytes then
    the scale, or the bf16 key), and ``split_index_k_rows`` views them back
    as the rows ``dsa_prefill_topk`` takes."""
    head_dim, page_size, pages = INDEX_HEAD_DIM, 4, 3
    pool, (expected_keys, expected_scale) = _written_index_k_pool(
        index_k_format, page_size, pages
    )
    slots = torch.tensor([0, 5, 11, 6], dtype=torch.int64)
    packed = pool.gather_index_k_rows(0, slots, index_k_format=index_k_format)
    row_bytes = index_k_row_bytes(head_dim, index_k_format)
    assert packed.shape == (4, row_bytes) and packed.dtype == torch.uint8
    keys, scale = split_index_k_rows(
        packed, index_head_dim=head_dim, index_k_format=index_k_format
    )
    assert keys.shape == (4, head_dim)
    assert torch.equal(keys, expected_keys[slots])
    if index_k_format == "fp8_scaled":
        assert row_bytes == dsa_index_k_row_bytes(head_dim)
        assert keys.dtype == torch.uint8 and scale.shape == (4, 1)
        assert torch.equal(scale, expected_scale[slots])
    else:
        assert row_bytes == 2 * head_dim
        assert keys.dtype == torch.bfloat16 and scale is None
        assert keys.data_ptr() == packed.data_ptr()
    # The format is explicit, never guessed from the bytes: the other
    # format's width is refused on the split, and the other format's plane
    # on the read.
    other = "bf16" if index_k_format == "fp8_scaled" else "fp8_scaled"
    with pytest.raises(ValueError, match="bytes wide"):
        split_index_k_rows(packed, index_head_dim=head_dim, index_k_format=other)
    with pytest.raises(ValueError, match="bytes wide"):
        split_index_k_rows(
            packed[:, :-1], index_head_dim=head_dim, index_k_format=index_k_format
        )
    with pytest.raises(ValueError, match=f"not the .* of index_k_format={other!r}"):
        pool.gather_index_k_rows(0, slots, index_k_format=other)


@pytest.mark.parametrize("index_k_format", INDEX_K_FORMATS)
def test_the_index_k_history_is_one_gather_per_group(monkeypatch, index_k_format):
    """``gather_history_index_k`` moves the packed rows in one collective and
    hands the indexer views of the workspace in the plane's format: FP8 bytes
    and fp32 scales, or bf16 keys and no scale."""
    backend = _backend(0, workspace_rows=11, index_k_format=index_k_format)
    plan = _plan(0)
    _init(backend, plan)
    group = backend.require_query_shard_metadata().groups[0]
    pool, (expected_keys, expected_scale) = _written_index_k_pool(
        index_k_format, PAGE, 8
    )
    gathers = []
    real = dsa.gather_history_rows

    def counting(plan_, local, *, out):
        gathers.append((local.dtype, tuple(local.shape), out))
        return real(plan_, local, out=out)

    monkeypatch.setattr(dsa, "gather_history_rows", counting)
    keys, scale = backend.gather_history_index_k(0, pool, group)
    assert len(gathers) == 1
    dtype, shape, out = gathers[0]
    row_bytes = index_k_row_bytes(INDEX_HEAD_DIM, index_k_format)
    assert dtype == torch.uint8 and shape == (group.rows, row_bytes)
    assert out is backend._history_workspace.index_k
    # Views of the workspace in position order.
    slots = group.gather.virtual_slots
    assert keys.shape == (group.rows, INDEX_HEAD_DIM)
    assert keys.data_ptr() == backend._history_workspace.index_k.data_ptr()
    assert torch.equal(keys, expected_keys[slots])
    if index_k_format == "fp8_scaled":
        assert keys.dtype == torch.uint8 and scale.shape == (group.rows, 1)
        assert torch.equal(scale, expected_scale[slots])
    else:
        assert keys.dtype == torch.bfloat16 and scale is None


def _decode_arm_backend(*, num_attention_heads: int, attn_tp_size: int):
    backend = _backend(1, workspace_rows=0, index_k_format="fp8_scaled")
    backend.kernel_page_size = 64
    backend.dcp_group = (0, 1, 2, 3)
    backend.dcp_rank = 1
    backend.dcp_block_granularity = 64
    backend.dcp_virtual_block_count = 5
    backend.kv_lora_rank = 128
    backend.qk_nope_head_dim = 128
    backend.qk_rope_head_dim = 0
    backend.index_topk = 512
    backend.max_context_len = 512
    backend.num_attention_heads = num_attention_heads
    backend.num_local_heads = num_attention_heads // attn_tp_size
    backend._dense_backend = SimpleNamespace(
        forward_decode_metadata=SimpleNamespace(
            num_extends=0, seq_lens_k=torch.tensor([128]), max_seq_len_k=128
        )
    )
    return backend


def _decode_layer(heads: int):
    return SimpleNamespace(
        layer_id=0,
        tp_q_head_num=heads,
        head_dim=128,
        v_head_dim=128,
        scaling=0.1,
        logit_cap=0.0,
    )


@pytest.mark.parametrize("layer_heads,keep_all_heads", [(8, True), (2, False)])
def test_the_dcp_combine_form_follows_the_querys_heads(
    monkeypatch, layer_heads, keep_all_heads
):
    """The decode arm's combine is decided by the heads the query carries,
    not by the layer's declared count or the mapping: every head
    (head-replicated weights, the drafter's steps on a query-sharding engine
    without head TP) keeps all heads -- no query-head gather, an all-reduce
    combine; the attention-TP slice (plain attention TP, or the drafter's
    steps under head TP over the query shards, which exchange nothing and
    hand the one core layer the slice) gathers the group's heads in and
    reduce-scatters its own back -- the DCP arm as it stands. The layer here
    declares every head either way, as a head-TP layer does."""
    backend = _decode_arm_backend(num_attention_heads=8, attn_tp_size=4)
    query = torch.zeros(1, layer_heads, 128, dtype=torch.bfloat16)
    slots = torch.full((1, 512), -1, dtype=torch.int32)
    slots[0, :4] = torch.tensor([64, 128, 192, 256])
    pool = SimpleNamespace(
        quant_method=None, get_key_buffer=lambda layer_id: torch.empty(320, 128)
    )
    gathered = []

    def gather(q, group):
        gathered.append(q.shape)
        return q.repeat(1, len(group), 1)

    def decode(**kwargs):
        assert kwargs["return_lse"] is True
        heads = kwargs["q"].shape[1]
        assert heads == (layer_heads if keep_all_heads else layer_heads * 4)
        return torch.full((1, heads, 128), 7.0), torch.zeros(1, heads)

    def combine(out, lse, *, group, rank, sink, keep_all_heads=None):
        assert keep_all_heads is keep_all_heads_expected and sink is None
        assert group == backend.dcp_group and rank == 1
        return out if keep_all_heads else out[:, :layer_heads]

    keep_all_heads_expected = keep_all_heads
    monkeypatch.setattr(dsa, "gather_query_heads", gather)
    monkeypatch.setattr(dsa, "dsa_decode", decode)
    monkeypatch.setattr(dsa, "combine_attention_partials", combine)
    out = backend.forward_sparse_decode(
        q=query,
        layer=_decode_layer(8),
        token_to_kv_pool=pool,
        bs=1,
        topk_indices=slots,
        topk_lens=None,
    )
    assert out.shape == (1, layer_heads * 128) and (out == 7).all()
    assert gathered == ([] if keep_all_heads else [(1, layer_heads, 128)])


def test_a_query_with_neither_head_layout_is_refused():
    backend = _decode_arm_backend(num_attention_heads=8, attn_tp_size=4)
    layer = _decode_layer(8)
    with pytest.raises(ValueError, match="neither the attention-TP slice"):
        backend._query_heads(torch.zeros(1, 3, 128), layer)
    # A flat query splits by the layer's head_dim; a ragged one is refused.
    assert backend._query_heads(torch.zeros(1, 2 * 128), layer) == 2
    assert backend._query_heads(torch.zeros(0, 8, 128), layer) == 8
    with pytest.raises(ValueError, match="does not split"):
        backend._query_heads(torch.zeros(1, 2 * 128 + 1), layer)
    with pytest.raises(ValueError, match="does not split"):
        backend._query_heads(torch.zeros(1, 2, 64), layer)
    assert backend._query_holds_every_head(8) and not backend._query_holds_every_head(2)
    # Without DCP the form is moot and the arm never asks.
    backend.dcp_group = (1,)
    backend.dcp_rank = 0
    assert not (len(backend.dcp_group) > 1)


@pytest.mark.parametrize("index_k_format", INDEX_K_FORMATS)
def test_the_workspace_reservation_matches_the_recipe_plan(index_k_format):
    """The recipe's plan (``dsa_history_gather_workspace_bytes``) and the
    leaf's allocation are the same bytes in either index-K format: the
    index-K rows are packed in the plane's own row width."""
    backend = _backend(0, workspace_rows=0, index_k_format=index_k_format)
    max_model_len = 37
    allocated = backend.preallocate_history_gather_workspace(max_model_len)
    config = SimpleNamespace(
        kv_cache_dtype=torch.bfloat16,
        kernel_page_size=PAGE,
        component=lambda cls: SimpleNamespace(
            kv_cache_dim=KV_DIM,
            index_head_dim=INDEX_HEAD_DIM,
            index_k_format=index_k_format,
        ),
    )
    assert allocated == dsa_history_gather_workspace_bytes(
        config, max_model_len=max_model_len
    )
    # One whole history, padded to kernel pages.
    rows = 38
    row_bytes = index_k_row_bytes(INDEX_HEAD_DIM, index_k_format)
    assert row_bytes == (132 if index_k_format == "fp8_scaled" else 256)
    assert allocated == rows * (KV_DIM * 2 + row_bytes)
    workspace = backend.history_gather_workspace()
    assert workspace.rows == rows and workspace.nbytes == allocated
    assert workspace.kv.shape == (rows, KV_DIM)
    assert workspace.index_k.shape == (rows, row_bytes)
    assert workspace.index_k.dtype == torch.uint8
    assert workspace.index_k_format == index_k_format
    # Without an override the plan follows the sparse kernels' fixed page.
    config.kernel_page_size = None
    assert dsa_history_gather_workspace_bytes(
        config, max_model_len=max_model_len
    ) == 64 * (KV_DIM * 2 + row_bytes)


def test_build_prefill_slots_match_the_group_history():
    table = _page_table()
    slots = build_prefill_kv_workspace_slots(
        page_table=table[0:2],
        seq_lens=torch.tensor(HISTORY[:2]),
        max_seq_len=6,
        page_size=PAGE,
        device=torch.device("cpu"),
        num_tokens=11,
    )
    assert slots.tolist() == [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


def _router(leaf: dsa.DSABackend, *, is_draft: bool):
    """A CacheGroupRouter over one DSA leaf, the shape the registry builds."""
    from tokenspeed.runtime.layers.attention.backends.paged.cache_group_geometry import (
        CacheGroupGeometry,
    )
    from tokenspeed.runtime.layers.attention.backends.paged.router import (
        CacheGroupRouter,
    )

    router = CacheGroupRouter(
        None,
        is_draft=is_draft,
        spec_num_tokens=1,
        device="cpu",
        consumed_group_ids=None,
    )
    router.bind(
        CacheGroupGeometry(
            granularities={"full": PAGE},
            families={"full": "history"},
            full_history_group_id="full",
            row_geometry={"full": (PAGE, 1)},
            retentions={"full": ("full_history", None)},
        ),
        {"full": leaf},
    )
    return router


def test_the_registry_allocates_the_workspace_once_and_the_draft_shares_it():
    """The serve path: ``_prepare_fixed_workspaces`` allocates the history
    gather workspace on the target tree against the recipe's plan and the
    draft tree gathers into the same buffers, so a sharded draft extend finds
    its workspace without a second reservation."""
    from tokenspeed.runtime.layers.attention.registry import _prepare_fixed_workspaces

    target = _backend(0, workspace_rows=0, index_k_format="fp8_scaled")
    draft = _backend(0, workspace_rows=0, index_k_format="fp8_scaled")
    draft.is_draft = True
    target_router = _router(target, is_draft=False)
    draft_router = _router(draft, is_draft=True)
    max_model_len = 37
    config = SimpleNamespace(
        qcp_size=WORLD,
        context_len=max_model_len,
        max_bs=4,
        kv_cache_dtype=torch.bfloat16,
        kernel_page_size=PAGE,
        component=lambda cls: SimpleNamespace(
            kv_cache_dim=KV_DIM,
            index_head_dim=INDEX_HEAD_DIM,
            index_k_format="fp8_scaled",
        ),
    )
    planned = dsa_history_gather_workspace_bytes(config, max_model_len=max_model_len)
    kwargs = dict(
        server_args=SimpleNamespace(speculative_num_draft_tokens=2),
        config=config,
        backend=target_router,
        draft_backend=draft_router,
        uses_paged_state_verify=False,
        is_inkling=False,
    )
    _prepare_fixed_workspaces(**kwargs, expected_bytes=planned)
    workspace = target_router.history_gather_workspace()
    assert workspace is not None and workspace.nbytes == planned
    assert workspace.rows == 38 and workspace.rows % PAGE == 0
    assert draft.history_gather_workspace() is workspace
    assert draft_router.history_gather_workspace() is workspace
    # Both leaves plan a sharded extend against it.
    for leaf in (target, draft):
        _init(leaf, _plan(0))
        assert leaf.require_query_shard_metadata().groups
    with pytest.raises(RuntimeError, match="does not match allocated"):
        _prepare_fixed_workspaces(**kwargs, expected_bytes=planned + 1)
    # Off: nothing is allocated and nothing is checked.
    fresh = _backend(0, workspace_rows=0, qcp=False, index_k_format="fp8_scaled")
    config.qcp_size = 1
    _prepare_fixed_workspaces(
        **{**kwargs, "backend": _router(fresh, is_draft=False), "draft_backend": None},
        expected_bytes=0,
    )
    assert fresh.history_gather_workspace() is None


def test_the_draft_refuses_a_workspace_of_another_geometry():
    target = _backend(0, workspace_rows=11, index_k_format="fp8_scaled")
    workspace = target.history_gather_workspace()
    draft = _backend(0, workspace_rows=0, index_k_format="fp8_scaled")
    draft.kv_cache_dim = KV_DIM + 2
    with pytest.raises(ValueError, match="geometry mismatch"):
        draft.adopt_history_gather_workspace(workspace)
    draft.kv_cache_dim = KV_DIM
    draft.kernel_page_size = 5  # 12 rows are not whole pages of five
    with pytest.raises(ValueError, match="geometry mismatch"):
        draft.adopt_history_gather_workspace(workspace)
    draft.kernel_page_size = PAGE
    # The index-K format is part of the geometry too (the recipe refuses a
    # draft of another format before any leaf exists; this is the leaf's own
    # invariant over the buffer it gathers into).
    draft.index_k_format = "bf16"
    with pytest.raises(ValueError, match="geometry mismatch"):
        draft.adopt_history_gather_workspace(workspace)
    draft.index_k_format = "fp8_scaled"
    draft.adopt_history_gather_workspace(workspace)
    assert draft.history_gather_workspace() is workspace


def _register_rows_leaf(name: str, *, index_k_format: str, takes_rows: bool):
    """A ``dsa_prefill_topk`` leaf of solution ``name`` for the format, with
    or without the workspace-rows feature; registered on any platform."""
    from tokenspeed_kernel.ops.attention.dsa import INDEX_K_WORKSPACE_ROWS_FEATURE
    from tokenspeed_kernel.registry import KernelRegistry, KernelSpec, Priority
    from tokenspeed_kernel.signature import dense_tensor_format, format_signature

    features = {"batch_invariant"}
    if takes_rows:
        features.add(INDEX_K_WORKSPACE_ROWS_FEATURE)
    spec = KernelSpec(
        name=name,
        family="attention",
        mode="dsa_prefill_topk",
        solution=name,
        format_signatures=frozenset(
            {
                format_signature(
                    q=dense_tensor_format(torch.bfloat16),
                    weights=dense_tensor_format(torch.float32),
                )
            }
        ),
        traits={
            "index_k_format": frozenset({index_k_format}),
            "index_k_layout": frozenset({"packed"}),
            "page_size": frozenset({PAGE}),
        },
        features=frozenset(features),
        priority=Priority.PORTABLE,
    )
    KernelRegistry.get().register(spec, lambda **kwargs: None)
    return spec.name


@pytest.mark.parametrize("index_k_format", INDEX_K_FORMATS)
def test_construction_probes_the_rows_leaf_the_sharded_prefill_will_need(
    index_k_format,
):
    """Under query context parallelism the leaf selects, at construction, the
    ``dsa_prefill_topk`` leaf that scores gathered index-K rows of its format
    (the ``index_k_workspace_rows`` feature): a platform without one fails at
    startup with the format named, not in the first sharded prefill."""
    from tokenspeed_kernel.registry import KernelRegistry
    from tokenspeed_kernel.selection import NoKernelFoundError

    other = "bf16" if index_k_format == "fp8_scaled" else "fp8_scaled"
    rows = _register_rows_leaf(
        "unit_rows", index_k_format=index_k_format, takes_rows=True
    )
    plane = _register_rows_leaf(
        "unit_plane", index_k_format=index_k_format, takes_rows=False
    )
    foreign = _register_rows_leaf("unit_foreign", index_k_format=other, takes_rows=True)
    try:
        backend = _backend(0, workspace_rows=0, index_k_format=index_k_format)
        backend.batch_invariant = True
        spec = SimpleNamespace(index_n_heads=16)
        backend.kernel_solution = "unit_rows"
        backend._probe_history_gather_topk_leaf(spec)
        # A leaf of the format that reads planes only, or a rows leaf of the
        # other format, is no leaf for these rows.
        for solution in ("unit_plane", "unit_foreign"):
            backend.kernel_solution = solution
            with pytest.raises(
                NoKernelFoundError, match=f"index_k_format={index_k_format!r}"
            ):
                backend._probe_history_gather_topk_leaf(spec)
    finally:
        for name in (rows, plane, foreign):
            KernelRegistry.get()._unregister(name)

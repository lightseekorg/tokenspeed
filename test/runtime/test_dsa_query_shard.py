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
group's gather run even without local rows), the dense delegate's view of the
forward, the decode arm's head-replicated combine, and the workspace
reservation agreeing with the recipe's plan.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.query_shard import QueryShardPlan
from tokenspeed.runtime.layers.attention.backends.paged import dsa
from tokenspeed.runtime.layers.attention.configs.dsa import (
    dsa_history_gather_workspace_bytes,
    dsa_index_k_row_bytes,
)
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


def _backend(rank: int, *, workspace_rows: int, qcp: bool = True) -> dsa.DSABackend:
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
    backend.index_topk = 4
    backend.max_context_len = 64
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
    backend._history_workspace_rows = 0
    backend._history_kv_workspace = None
    backend._history_index_k_fp8_workspace = None
    backend._history_index_k_scale_workspace = None
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
    backend = _backend(rank, workspace_rows=11)  # 6 + 5 fit, 6 + 5 + 5 do not
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
    backend = _backend(0, workspace_rows=5)
    with pytest.raises(RuntimeError, match="exceeds"):
        _init(backend, _plan(0))
    backend = _backend(0, workspace_rows=11)
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
    backend = _backend(0, workspace_rows=11)
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
    backend = _backend(0, workspace_rows=0)
    with pytest.raises(RuntimeError, match="workspace"):
        _init(backend, _plan(0))


@pytest.mark.parametrize("rank", range(WORLD))
def test_the_sharded_arm_attends_gathered_groups_with_full_heads(monkeypatch, rank):
    backend = _backend(rank, workspace_rows=11)
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
        assert call["kv_cache"].shape == (group.rows, KV_DIM)
        # The gathered buffer holds the group's history in position order.
        expected = plane[group.gather.virtual_slots, 0]
        torch.testing.assert_close(call["kv_cache"], expected, rtol=0, atol=0)
        # Top-k rows are re-based to the group's buffer, -1 stays -1.
        expected_slots = topk[rows].clone()
        expected_slots[:, 0] -= group.row_base
        assert torch.equal(call["topk_slots"], expected_slots)
        assert torch.equal(call["topk_lens"], topk_lens[rows])
        assert torch.equal(call["kv_seq_lens"], kv_seq_lens[rows])
        assert call["q"].shape[0] == rows.stop - rows.start
    if local_rows == 0:
        assert not calls


def test_the_dense_delegate_is_refused_under_a_shard():
    backend = _backend(0, workspace_rows=11)
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


def test_the_index_k_gather_reads_the_block_split_plane():
    from tokenspeed.runtime.layers.attention.kv_cache.dsa import DSATokenToKVPool

    head_dim, page_size, pages = 128, 4, 3
    groups = head_dim // 128
    row_bytes = head_dim + groups * 4
    buf = torch.zeros(pages * page_size * row_bytes, dtype=torch.uint8)
    # Page-planar layout: fp8 rows then fp32 scales per page.
    for page in range(pages):
        base = page * page_size * row_bytes
        fp8 = buf[base : base + page_size * head_dim].view(page_size, head_dim)
        fp8[:] = (torch.arange(page_size) + 10 * page).unsqueeze(1).to(torch.uint8)
        scales = (
            buf[base + page_size * head_dim : base + page_size * row_bytes]
            .view(torch.float32)
            .view(page_size, groups)
        )
        scales[:] = (torch.arange(page_size) + 100 * page).unsqueeze(1).float()
    pool = SimpleNamespace(
        get_index_k_buffer=lambda layer_id: buf.view(pages * page_size, row_bytes),
        arena=SimpleNamespace(kv_page_size=page_size),
        index_head_dim=head_dim,
    )
    slots = torch.tensor([0, 5, 11, 6], dtype=torch.int64)
    fp8, scale = DSATokenToKVPool.gather_index_k_rows(pool, 0, slots)
    assert fp8[:, 0].tolist() == [0, 11, 23, 12]
    assert scale[:, 0].tolist() == [0.0, 101.0, 203.0, 102.0]


def test_the_decode_arm_keeps_every_head_under_a_query_shard(monkeypatch):
    backend = _backend(1, workspace_rows=0)
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
    backend._dense_backend = SimpleNamespace(
        forward_decode_metadata=SimpleNamespace(
            num_extends=0, seq_lens_k=torch.tensor([128]), max_seq_len_k=128
        )
    )
    query = torch.zeros(1, 2, 128, dtype=torch.bfloat16)
    slots = torch.full((1, 512), -1, dtype=torch.int32)
    slots[0, :4] = torch.tensor([64, 128, 192, 256])
    pool = SimpleNamespace(
        quant_method=None, get_key_buffer=lambda layer_id: torch.empty(320, 128)
    )
    layer = SimpleNamespace(
        layer_id=0,
        tp_q_head_num=2,
        head_dim=128,
        v_head_dim=128,
        scaling=0.1,
        logit_cap=0.0,
    )

    def gather(q, group):
        raise AssertionError("head-replicated attention gathers no query heads")

    def decode(**kwargs):
        assert kwargs["return_lse"] is True
        assert kwargs["q"].shape == (1, 2, 128)
        return torch.full((1, 2, 128), 7.0), torch.zeros(1, 2)

    def combine(out, lse, *, group, rank, sink, keep_all_heads):
        assert keep_all_heads is True and sink is None
        assert group == backend.dcp_group and rank == 1
        return out

    monkeypatch.setattr(dsa, "gather_query_heads", gather)
    monkeypatch.setattr(dsa, "dsa_decode", decode)
    monkeypatch.setattr(dsa, "combine_attention_partials", combine)
    out = backend.forward_sparse_decode(
        q=query,
        layer=layer,
        token_to_kv_pool=pool,
        bs=1,
        topk_indices=slots,
        topk_lens=None,
    )
    assert out.shape == (1, 256) and (out == 7).all()


def test_the_workspace_reservation_matches_the_recipe_plan():
    backend = _backend(0, workspace_rows=0)
    rows = 37
    allocated = backend.preallocate_history_gather_workspace(rows)
    config = SimpleNamespace(
        kv_cache_dtype=torch.bfloat16,
        component=lambda cls: SimpleNamespace(
            kv_cache_dim=KV_DIM,
            index_head_dim=INDEX_HEAD_DIM,
            index_k_format="fp8_scaled",
        ),
    )
    assert allocated == dsa_history_gather_workspace_bytes(config, max_model_len=rows)
    assert allocated == rows * (KV_DIM * 2 + dsa_index_k_row_bytes(INDEX_HEAD_DIM))
    assert backend._history_kv_workspace.shape == (rows, KV_DIM)
    assert backend._history_index_k_fp8_workspace.shape == (rows, INDEX_HEAD_DIM)
    assert backend._history_index_k_scale_workspace.shape == (rows, 1)


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

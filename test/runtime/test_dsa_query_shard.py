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
from tokenspeed.runtime.layers.attention.kv_cache.dsa import (
    DSATokenToKVPool,
    split_index_k_rows,
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
    backend = _backend(0, workspace_rows=3)  # four rows once padded to pages
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


def _index_k_plane(head_dim: int, page_size: int, pages: int) -> torch.Tensor:
    """A block-split index-K plane: fp8 rows then fp32 scales per page; row
    ``r`` of page ``p`` holds byte ``r + 10 p`` and scale ``r + 100 p``."""
    groups = head_dim // 128
    row_bytes = head_dim + groups * 4
    buf = torch.zeros(pages * page_size * row_bytes, dtype=torch.uint8)
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
    return buf.view(pages * page_size, row_bytes)


def test_the_index_k_gather_reads_the_block_split_plane_packed_per_row():
    head_dim, page_size, pages = 128, 4, 3
    pool = SimpleNamespace(
        get_index_k_buffer=lambda layer_id: _index_k_plane(head_dim, page_size, pages),
        arena=SimpleNamespace(kv_page_size=page_size),
        index_head_dim=head_dim,
    )
    slots = torch.tensor([0, 5, 11, 6], dtype=torch.int64)
    packed = DSATokenToKVPool.gather_index_k_rows(pool, 0, slots)
    assert packed.shape == (4, dsa_index_k_row_bytes(head_dim))
    assert packed.dtype == torch.uint8
    fp8, scale = split_index_k_rows(packed, index_head_dim=head_dim)
    assert fp8.shape == (4, head_dim) and scale.shape == (4, 1)
    assert fp8[:, 0].tolist() == [0, 11, 23, 12]
    assert scale[:, 0].tolist() == [0.0, 101.0, 203.0, 102.0]
    with pytest.raises(ValueError, match="bytes wide"):
        split_index_k_rows(packed[:, :-1], index_head_dim=head_dim)


def test_the_index_k_history_is_one_gather_per_group(monkeypatch):
    """``gather_history_index_k`` moves the packed rows in one collective and
    hands the indexer views of the workspace: FP8 bytes and fp32 scales."""
    backend = _backend(0, workspace_rows=11)
    plan = _plan(0)
    _init(backend, plan)
    group = backend.require_query_shard_metadata().groups[0]
    pool = SimpleNamespace(
        get_index_k_buffer=lambda layer_id: _index_k_plane(INDEX_HEAD_DIM, PAGE, 8),
        arena=SimpleNamespace(kv_page_size=PAGE),
        index_head_dim=INDEX_HEAD_DIM,
    )
    pool.gather_index_k_rows = lambda layer_id, slots: (
        DSATokenToKVPool.gather_index_k_rows(pool, layer_id, slots)
    )
    gathers = []
    real = dsa.gather_history_rows

    def counting(plan_, local, *, out):
        gathers.append((local.dtype, tuple(local.shape), out))
        return real(plan_, local, out=out)

    monkeypatch.setattr(dsa, "gather_history_rows", counting)
    fp8, scale = backend.gather_history_index_k(0, pool, group)
    assert len(gathers) == 1
    dtype, shape, out = gathers[0]
    assert dtype == torch.uint8 and shape == (group.rows, 132)
    assert out is backend._history_workspace.index_k
    # Views of the workspace in position order: slot v is page v // 2, row v % 2.
    slots = group.gather.virtual_slots
    assert fp8.shape == (group.rows, INDEX_HEAD_DIM) and scale.shape == (group.rows, 1)
    assert fp8[:, 0].tolist() == ((slots % PAGE) + 10 * (slots // PAGE)).tolist()
    assert scale[:, 0].tolist() == ((slots % PAGE) + 100 * (slots // PAGE)).tolist()
    assert fp8.data_ptr() == backend._history_workspace.index_k.data_ptr()


def _decode_arm_backend(*, num_attention_heads: int, attn_tp_size: int):
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
def test_the_dcp_combine_form_follows_the_layers_head_layout(
    monkeypatch, layer_heads, keep_all_heads
):
    """The decode arm's combine is decided by the layer's heads, not by the
    mapping: a layer holding every head (head-replicated weights, a query
    shard's drafter steps) keeps all heads -- no query-head gather, an
    all-reduce combine; a layer holding the attention-TP slice gathers the
    group's heads in and reduce-scatters its own back."""
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
        layer=_decode_layer(layer_heads),
        token_to_kv_pool=pool,
        bs=1,
        topk_indices=slots,
        topk_lens=None,
    )
    assert out.shape == (1, layer_heads * 128) and (out == 7).all()
    assert gathered == ([] if keep_all_heads else [(1, layer_heads, 128)])


def test_a_layer_with_neither_head_layout_is_refused():
    backend = _decode_arm_backend(num_attention_heads=8, attn_tp_size=4)
    with pytest.raises(ValueError, match="neither the attention-TP slice"):
        backend._layer_holds_every_head(_decode_layer(3))
    # Without DCP the form is moot and the arm never asks.
    backend.dcp_group = (1,)
    backend.dcp_rank = 0
    assert not (len(backend.dcp_group) > 1)


def test_the_workspace_reservation_matches_the_recipe_plan():
    backend = _backend(0, workspace_rows=0)
    max_model_len = 37
    allocated = backend.preallocate_history_gather_workspace(max_model_len)
    config = SimpleNamespace(
        kv_cache_dtype=torch.bfloat16,
        kernel_page_size=PAGE,
        component=lambda cls: SimpleNamespace(
            kv_cache_dim=KV_DIM,
            index_head_dim=INDEX_HEAD_DIM,
            index_k_format="fp8_scaled",
        ),
    )
    assert allocated == dsa_history_gather_workspace_bytes(
        config, max_model_len=max_model_len
    )
    # One whole history, padded to kernel pages.
    rows = 38
    assert allocated == rows * (KV_DIM * 2 + dsa_index_k_row_bytes(INDEX_HEAD_DIM))
    workspace = backend.history_gather_workspace()
    assert workspace.rows == rows and workspace.nbytes == allocated
    assert workspace.kv.shape == (rows, KV_DIM)
    assert workspace.index_k.shape == (
        rows,
        dsa_index_k_row_bytes(INDEX_HEAD_DIM),
    )
    # Without an override the plan follows the sparse kernels' fixed page.
    config.kernel_page_size = None
    assert dsa_history_gather_workspace_bytes(
        config, max_model_len=max_model_len
    ) == 64 * (KV_DIM * 2 + dsa_index_k_row_bytes(INDEX_HEAD_DIM))


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

    target = _backend(0, workspace_rows=0)
    draft = _backend(0, workspace_rows=0)
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
    fresh = _backend(0, workspace_rows=0, qcp=False)
    config.qcp_size = 1
    _prepare_fixed_workspaces(
        **{**kwargs, "backend": _router(fresh, is_draft=False), "draft_backend": None},
        expected_bytes=0,
    )
    assert fresh.history_gather_workspace() is None


def test_the_draft_refuses_a_workspace_of_another_geometry():
    target = _backend(0, workspace_rows=11)
    workspace = target.history_gather_workspace()
    draft = _backend(0, workspace_rows=0)
    draft.kv_cache_dim = KV_DIM + 2
    with pytest.raises(ValueError, match="geometry mismatch"):
        draft.adopt_history_gather_workspace(workspace)
    draft.kv_cache_dim = KV_DIM
    draft.kernel_page_size = 5  # 12 rows are not whole pages of five
    with pytest.raises(ValueError, match="geometry mismatch"):
        draft.adopt_history_gather_workspace(workspace)
    draft.kernel_page_size = PAGE
    draft.adopt_history_gather_workspace(workspace)
    assert draft.history_gather_workspace() is workspace

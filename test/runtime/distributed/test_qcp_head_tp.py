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

"""Attention head TP over the query shards of a prefill engine, on CPU over gloo.

Four ranks, attention TP 4 with ``--prefill-context-parallel-size 4`` and
``--attn-head-tp-size 4``: the head group is the shard group. The reference
is the same mapping without head TP -- head-replicated q_b / kv_b / o_proj,
QCP's default -- so the exchanges, the shard row tables, the prologue's
gathered write and both o_proj tails are what is tested, with the core
stubbed per row and head as in ``test_decode_tp_layouts`` (the shared fakes
live in ``_tp_layout_fakes``). The stub stands in for the DSA backend: its
sparse prefill reads the head count from the query, attends every head of
the local rows against a history it gathers over the shard group on every
rank -- so a rank that skipped the core on its empty shard would hang the
others -- and its decode core attends whatever heads the query carries.

Covered:
* the sharded extend through the sequence a DSA model threads
  (``forward_absorb_qkv_proj`` -> ``sparse_prefill_attn_v_proj`` ->
  ``project_output``): ``head_tp_leg_row_counts`` reads the shard plan
  (input and collective legs), the legs run with an uneven split and with a
  rank whose shard is empty (it joins every collective, the core's history
  gather included, and attends nothing);
* the o_proj forms: row-parallel + reduce-scatter to the shard rows, and
  ``--tp-batch-invariant attn`` (all-gather heads, column-parallel GEMM,
  all-to-all back);
* the drafter's decode step (replicated rows, no shard): no exchange, this
  rank's head slice through the one core layer, the all-reduce / all-gather
  tail, through the hooks and through the dense ``forward``;
* the resolver refuses a replicated-row forward's table under query sharding.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from _tp_layout_fakes import (
    HIDDEN,
    KV_LORA,
    NUM_HEADS,
    V_DIM,
    attention_weights,
    build_attention,
    init_gloo,
)

from tokenspeed.runtime.distributed.comm_manager import (
    CommManager,
    head_tp_row_counts,
)
from tokenspeed.runtime.distributed.comm_ops import token_all_gather
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.query_shard import QueryShardPlan

WORLD = 4


def _mapping(rank: int, *, head_tp: bool) -> Mapping:
    return Mapping(
        rank=rank,
        world_size=WORLD,
        attn_tp_size=WORLD,
        attn_qcp_size=WORLD,
        attn_head_tp_size=WORLD if head_tp else None,
        dense_tp_size=1,
        moe_tp_size=1,
        moe_ep_size=WORLD,
        nprocs_per_node=WORLD,
    )


class _QcpStubCore:
    """``PagedAttention`` plus the DSA backend's sparse prefill, on CPU.

    The prologue asserts the one-row-count contract and, under a query shard,
    the QCP write: the span's slots on every rank and the rotated latent
    gathered over the shard group (the collective an empty shard still
    joins). The cores take the head count from the query, as the DSA backend
    does: every head (``every_heads``, what the layer declares under head TP)
    or the attention-TP slice (``slice_heads``). Each maps a (token, head)
    query through that token's own latent, so a head's output for a token
    depends on exactly the inputs the real kernel reads. The sparse prefill
    mirrors the sharded arm: every rank gathers the group's history before
    attending its own rows, with or without rows.
    """

    def __init__(
        self, layer_id: int, *, slice_heads: int, every_heads: int, group: tuple
    ):
        self.layer_id = layer_id
        self.tp_q_head_num = every_heads
        self.slice_heads = slice_heads
        self.every_heads = every_heads
        self.group = group
        self.calls_by_heads: dict[int, int] = {slice_heads: 0, every_heads: 0}
        self.gathers = 0
        self.history_gathers = 0
        self.latent: torch.Tensor | None = None

    def latent_prologue(
        self, query, q_pe, latent_cache, positions, ctx, *, slots, expanded, key_rows
    ):
        assert expanded is None
        assert query.shape[0] == q_pe.shape[0] == latent_cache.shape[0]
        assert query.shape[0] == positions.shape[0]
        assert query.shape[1] in (self.slice_heads, self.every_heads)
        if key_rows is None:
            assert slots.shape[0] == query.shape[0]
            latent = latent_cache
        else:
            plan = key_rows.plan
            assert latent_cache.shape[0] == plan.local_rows
            assert slots.shape[0] == plan.total_rows
            gathered = token_all_gather(
                latent_cache.contiguous(), key_rows.group, list(plan.row_counts)
            )
            self.gathers += 1
            latent = gathered[plan.local_slice]
        rotated = query.clone()
        rotated[..., KV_LORA:] = q_pe * (positions.to(query.dtype) + 1.0)[:, None, None]
        self.latent = latent[:, :KV_LORA].clone()
        return SimpleNamespace(query=rotated)

    def _attend(self, Q: torch.Tensor) -> torch.Tensor:
        heads = Q.shape[1]
        assert heads in (self.slice_heads, self.every_heads)
        self.calls_by_heads[heads] += 1
        kv_gain = 1.0 + self.latent.sum(dim=-1)  # [T]
        out = Q[..., :KV_LORA] * kv_gain[:, None, None]
        out = out + 0.01 * Q[..., KV_LORA:].sum(dim=-1, keepdim=True)
        # An empty shard attends nothing; keep the reshape well-defined.
        return out.reshape(Q.shape[0], heads * KV_LORA)

    def __call__(self, Q, k=None, v=None, positions=None, ctx=None, **kwargs):
        """The decode core (``PagedAttention.forward``)."""
        assert k is None and v is None
        return self._attend(Q)

    def sparse_prefill(self, plan: QueryShardPlan, *, q, layer, **kwargs):
        """The backend's ``forward_sparse_prefill`` for ``layer`` (this)."""
        assert layer is self
        assert q.shape[0] == plan.local_rows
        # The sharded arm attends with every head and gathers the group's
        # history on every rank before the empty-query return.
        assert q.shape[1] == self.every_heads
        token_all_gather(self.latent.contiguous(), self.group, list(plan.row_counts))
        self.history_gathers += 1
        return self._attend(q)


def _build(mapping: Mapping, weights: dict[str, torch.Tensor]):
    attn = build_attention(mapping, weights)
    # The decode-layout stub went in for attn_mqa; swap in the QCP-aware one
    # with the heads the core may see: every head after an exchange (what
    # the layer declares), the slice on a replicated-row forward.
    attn.attn_mqa = _QcpStubCore(
        0,
        slice_heads=attn.num_local_heads,
        every_heads=attn.num_heads if attn.has_head_tp else attn.num_local_heads,
        group=mapping.attn.qcp_group,
    )
    return attn


def _extend_ctx(plan: QueryShardPlan, lengths: list[int]) -> ForwardContext:
    return ForwardContext(
        attn_backend=SimpleNamespace(
            spec_num_tokens=1,
            # Dispatches to the layer's stub, as the router dispatches to the
            # leaf serving the layer.
            forward_sparse_prefill=lambda **kw: kw["layer"].sparse_prefill(plan, **kw),
        ),
        token_to_kv_pool=None,
        bs=len(lengths),
        num_extends=len(lengths),
        input_num_tokens=plan.total_rows,
        forward_mode=ForwardMode.EXTEND,
        output_layout=None,
        gather_ids=torch.tensor(lengths).cumsum(0) - 1,
        query_shard=plan,
    )


def _decode_ctx(bs: int) -> ForwardContext:
    return ForwardContext(
        attn_backend=SimpleNamespace(
            spec_num_tokens=1,
            supports_mla_projected_value_decode=False,
            write_locations=lambda layer, mode: torch.arange(bs, dtype=torch.int64),
        ),
        token_to_kv_pool=None,
        bs=bs,
        num_extends=0,
        input_num_tokens=bs,
        forward_mode=ForwardMode.DECODE,
        output_layout=None,
    )


def _absorbed_forward(attn, positions, hidden, ctx, comm, span_slots):
    """The hooks a sparse-attention model threads: the q latent projection
    (token gather under head TP), the absorption and prologue (exchange, the
    QCP write), the core and value projection (the sparse prefill on extend
    rows, the decode core otherwise; exchange back), the o_proj tail. The
    same calls on every layout; the hooks decide."""
    q, latent = attn._project_q_latent(hidden, ctx, comm, None)
    Q = attn.forward_absorb_qkv_proj(q, latent, positions, ctx, span_slots)
    own = hidden.shape[0]
    rows = (
        sum(attn.head_tp_leg_row_counts(ctx, own, collective=True))
        if attn.head_tp_exchanges(ctx)
        else own
    )
    output = q.new_empty(rows, attn.num_local_heads * V_DIM)
    if ctx.num_extends > 0:
        attn.sparse_prefill_attn_v_proj(
            Q,
            ctx,
            output,
            kv_seq_lens=None,
            topk_slots=torch.empty(Q.shape[0], 0, dtype=torch.int32),
            topk_lens=torch.zeros(Q.shape[0], dtype=torch.int32),
            max_seq_len=0,
        )
    else:
        attn.forward_absorb_attn_v_proj(Q, ctx, output)
    return attn.project_output(output, ctx, own)


def _worker(rank: int, rendezvous: str, tp_batch_invariant: str, lengths: list[int]):
    from tokenspeed.runtime.utils.env import global_server_args_dict

    mapping = _mapping(rank, head_tp=True)
    init_gloo(rank, rendezvous, mapping)
    try:
        weights = attention_weights(HIDDEN)
        total = sum(lengths)
        plan = QueryShardPlan.from_forward(
            total_tokens=total, input_lengths=lengths, size=WORLD, rank=rank
        )
        gen = torch.Generator().manual_seed(4)
        hidden_full = torch.randn(total, HIDDEN, generator=gen)
        positions_full = torch.arange(total, dtype=torch.int64) * 3
        span_slots = torch.arange(total, dtype=torch.int64)
        slice_heads = NUM_HEADS // WORLD

        # The head-replicated reference: QCP's default layout.
        global_server_args_dict["tp_batch_invariant"] = "none"
        reference = _build(_mapping(rank, head_tp=False), weights)
        assert not reference.has_head_tp and reference.num_local_heads == NUM_HEADS

        global_server_args_dict["tp_batch_invariant"] = tp_batch_invariant
        sharded = _build(mapping, weights)
        assert sharded.has_head_tp and sharded.num_local_heads == slice_heads
        assert sharded.head_tp_group == mapping.attn.qcp_group
        assert sharded.attn_mqa.tp_q_head_num == NUM_HEADS
        assert type(sharded.o_proj).__name__ == (
            "ColumnParallelLinear"
            if tp_batch_invariant == "attn"
            else "RowParallelLinear"
        )
        comm = CommManager(
            mapping,
            layer_id=0,
            is_moe=False,
            prev_is_moe=False,
            dense_batch_invariant=False,
            query_sharded=True,
        )

        # --- The sharded extend -------------------------------------------
        ctx = _extend_ctx(plan, lengths)
        own = plan.local_slice
        hidden = hidden_full[own].contiguous()
        positions = positions_full[own].contiguous()
        assert sharded.head_tp_exchanges(ctx)
        counts = list(plan.row_counts)
        assert (
            sharded.head_tp_leg_row_counts(ctx, plan.local_rows, collective=False)
            == counts
        )
        assert (
            sharded.head_tp_leg_row_counts(ctx, plan.local_rows, collective=True)
            == counts
        )
        with pytest.raises(ValueError, match="holds"):
            sharded.head_tp_leg_row_counts(ctx, plan.local_rows + 1, collective=True)

        expected = _absorbed_forward(
            reference, positions, hidden, ctx, comm, span_slots
        )
        actual = _absorbed_forward(sharded, positions, hidden, ctx, comm, span_slots)
        assert tuple(actual.shape) == (plan.local_rows, HIDDEN)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
        # Every rank ran the prologue's gather and the core's history gather,
        # the empty shard included; the core attended every head.
        for core in (sharded.attn_mqa, reference.attn_mqa):
            assert core.gathers == 1 and core.history_gathers == 1
        assert sharded.attn_mqa.calls_by_heads == {slice_heads: 0, NUM_HEADS: 1}

        # --- The drafter's decode step: replicated rows, no shard ---------
        bs = 2
        gen = torch.Generator().manual_seed(5)
        rows = torch.randn(bs, HIDDEN, generator=gen)  # the same rows everywhere
        decode_positions = torch.tensor([7, 11], dtype=torch.int64)
        decode_ctx = _decode_ctx(bs)
        assert not sharded.head_tp_exchanges(decode_ctx)
        with pytest.raises(ValueError, match="no query shard"):
            sharded.head_tp_leg_row_counts(decode_ctx, bs, collective=True)
        expected = _absorbed_forward(
            reference, decode_positions, rows, decode_ctx, comm, span_slots[:bs]
        )
        actual = _absorbed_forward(
            sharded, decode_positions, rows, decode_ctx, comm, span_slots[:bs]
        )
        assert tuple(actual.shape) == (bs, HIDDEN)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
        # The one layer attended this rank's head slice of every row.
        assert sharded.attn_mqa.calls_by_heads == {slice_heads: 1, NUM_HEADS: 1}
        # The dense forward takes the same path for a decode step.
        expected = reference(decode_positions, rows, decode_ctx, comm)
        actual = sharded(decode_positions, rows, decode_ctx, comm)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
        assert sharded.attn_mqa.calls_by_heads == {slice_heads: 2, NUM_HEADS: 1}
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("tp_batch_invariant", ["attn", "none"])
@pytest.mark.parametrize(
    "lengths",
    [
        # 7 rows over 4 shards: [2, 2, 2, 1].
        [4, 3],
        # 3 rows over 4 shards: [1, 1, 1, 0]; rank 3's shard is empty.
        [2, 1],
    ],
    ids=["uneven", "empty-shard"],
)
def test_head_tp_over_the_query_shards_matches_head_replicated(
    tmp_path, tp_batch_invariant, lengths
):
    mp.spawn(
        _worker,
        args=((tmp_path / "rv").as_uri(), tp_batch_invariant, lengths),
        nprocs=WORLD,
        join=True,
    )


# ---------------------------------------------------------------------------
# Host-side: the resolver (no distributed)
# ---------------------------------------------------------------------------


def test_head_tp_row_counts_follow_the_layout():
    """One resolver: the DP tables under attention DP, the shard plan under
    query sharding (input rows, or the collective rows a narrowing drafter
    reported); this rank's count is checked either way."""
    dp = Mapping(
        rank=2, world_size=4, attn_tp_size=1, attn_dp_size=4, attn_head_tp_size=4
    )
    ctx = SimpleNamespace(
        global_num_tokens=[2, 0, 3, 1],
        collective_global_num_tokens=[1, 0, 1, 1],
        query_shard=None,
        collective_num_tokens=None,
    )
    assert head_tp_row_counts(ctx, dp, 3, collective=False) == [2, 0, 3, 1]
    assert head_tp_row_counts(ctx, dp, 1, collective=True) == [1, 0, 1, 1]
    with pytest.raises(ValueError, match="holds 2 rows"):
        head_tp_row_counts(ctx, dp, 2, collective=False)

    qcp = Mapping(
        rank=2, world_size=4, attn_tp_size=4, attn_qcp_size=4, attn_head_tp_size=4
    )
    plan = QueryShardPlan.from_forward(
        total_tokens=7, input_lengths=[4, 3], size=4, rank=2
    )
    sharded = SimpleNamespace(
        global_num_tokens=None,
        collective_global_num_tokens=None,
        query_shard=plan,
        collective_num_tokens=None,
    )
    assert head_tp_row_counts(sharded, qcp, 2, collective=False) == [2, 2, 2, 1]
    assert head_tp_row_counts(sharded, qcp, 2, collective=True) == [2, 2, 2, 1]
    # A narrowing drafter reports the sampled rows; the output legs follow
    # (the requests' last rows 3 and 6 sit on ranks 1 and 3).
    assert plan.sampled_rows_per_rank == (0, 1, 0, 1)
    narrowed = SimpleNamespace(**{**vars(sharded), "collective_num_tokens": 2})
    assert head_tp_row_counts(narrowed, qcp, 0, collective=True) == [0, 1, 0, 1]
    with pytest.raises(ValueError, match="holds 1 rows"):
        head_tp_row_counts(narrowed, qcp, 1, collective=True)
    with pytest.raises(ValueError, match="holds 3 rows"):
        head_tp_row_counts(sharded, qcp, 3, collective=False)
    replicated = SimpleNamespace(**{**vars(sharded), "query_shard": None})
    with pytest.raises(ValueError, match="no query shard"):
        head_tp_row_counts(replicated, qcp, 2, collective=False)

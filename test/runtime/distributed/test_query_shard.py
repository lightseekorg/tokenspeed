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

"""The query shard plan: host arithmetic every rank derives identically."""

from __future__ import annotations

import pathlib
import re

import pytest
import torch

from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.query_shard import QueryShardPlan, scatter_count

RUNTIME = (
    pathlib.Path(__file__).resolve().parents[3] / "python" / "tokenspeed" / "runtime"
)


def _plans(total: int, lengths: list[int], size: int) -> list[QueryShardPlan]:
    return [
        QueryShardPlan.from_forward(
            total_tokens=total, input_lengths=lengths, size=size, rank=rank
        )
        for rank in range(size)
    ]


def _gather_rows(lengths: list[int]) -> list[int]:
    return (torch.cumsum(torch.tensor(lengths), 0) - 1).tolist()


def test_scatter_count_matches_the_rsag_split():
    assert scatter_count(10, 4) == [3, 3, 2, 2]
    assert scatter_count(2, 4) == [1, 1, 0, 0]
    assert scatter_count(0, 3) == [0, 0, 0]
    assert CommManager._scatter_count(10, 4) == scatter_count(10, 4)


@pytest.mark.parametrize(
    "lengths,size",
    [
        ([5, 3, 7], 4),
        ([1, 1, 1], 4),
        ([16], 8),
        ([2, 2, 2, 2, 2, 2, 2, 2, 2], 3),
        ([9, 1, 1, 9], 2),
    ],
)
def test_shard_ranges_partition_the_span_and_count_the_sampled_rows(lengths, size):
    total = sum(lengths)
    plans = _plans(total, lengths, size)
    ranges = [(p.local_start, p.local_end) for p in plans]
    assert ranges[0][0] == 0 and ranges[-1][1] == total
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))
    assert all(p.row_counts == tuple(scatter_count(total, size)) for p in plans)
    assert all(p.local_rows == p.local_end - p.local_start for p in plans)
    # The sampled rows of every request land in exactly one shard, and the
    # plan counts them per shard in request order.
    rows = _gather_rows(lengths)
    expected = [sum(1 for row in rows if start <= row < end) for start, end in ranges]
    assert list(plans[0].sampled_rows_per_rank) == expected
    assert plans[0].sampled_rows_total == len(lengths)
    # Rank order is request order: the local sampled rows are a contiguous
    # run of gather_ids starting at local_sampled_first.
    for plan in plans:
        first = plan.local_sampled_first
        local = rows[first : first + plan.local_sampled_rows]
        assert all(plan.local_start <= row < plan.local_end for row in local)
        assert [row - plan.local_start for row in local] == sorted(
            row - plan.local_start for row in local
        )


def test_zero_row_ranks_join_with_empty_shards():
    plans = _plans(2, [2], 4)
    assert [p.local_rows for p in plans] == [1, 1, 0, 0]
    assert plans[0].sampled_rows_per_rank == (0, 1, 0, 0)
    assert plans[3].local_slice == slice(2, 2)
    assert plans[1].local_sampled_first == 0 and plans[1].local_sampled_rows == 1


def test_rows_for_collective_names_the_shard_or_the_sampled_rows():
    plan = _plans(12, [5, 7], 4)[1]
    assert plan.rows_for_collective(None) == plan.row_counts
    assert plan.rows_for_collective(12) == plan.row_counts
    assert plan.rows_for_collective(2) == plan.sampled_rows_per_rank
    with pytest.raises(ValueError, match="matches neither"):
        plan.rows_for_collective(7)


def test_plan_rejects_inconsistent_inputs():
    with pytest.raises(ValueError, match="sum to"):
        QueryShardPlan.from_forward(
            total_tokens=5, input_lengths=[2, 2], size=2, rank=0
        )
    with pytest.raises(ValueError, match="outside"):
        QueryShardPlan(size=2, rank=2, row_counts=(1, 1), sampled_rows_per_rank=(1, 0))
    with pytest.raises(ValueError, match="name every rank"):
        QueryShardPlan(size=2, rank=0, row_counts=(2,), sampled_rows_per_rank=(1, 0))


def test_mapping_exposes_the_query_shard_group():
    mapping = Mapping(rank=5, world_size=8, attn_tp_size=8, attn_qcp_size=8)
    assert mapping.attn.has_qcp
    assert mapping.attn.qcp_rank == 5
    assert mapping.attn.qcp_group == tuple(range(8))
    assert "qcp=8" in repr(mapping)
    off = Mapping(rank=5, world_size=8, attn_tp_size=8)
    assert not off.attn.has_qcp and off.attn.qcp_group == (5,)
    with pytest.raises(ValueError, match="divisible"):
        Mapping(rank=0, world_size=8, attn_tp_size=8, attn_qcp_size=3)
    with pytest.raises(ValueError, match="positive"):
        Mapping(rank=0, world_size=8, attn_tp_size=8, attn_qcp_size=0)


def test_comm_manager_flag_must_agree_with_the_mapping():
    sharded = Mapping(rank=0, world_size=2, attn_tp_size=2, attn_qcp_size=2)
    plain = Mapping(rank=0, world_size=2, attn_tp_size=2)
    for mapping, flag in ((sharded, False), (plain, True)):
        with pytest.raises(ValueError, match="disagrees"):
            CommManager(
                mapping=mapping,
                layer_id=0,
                is_moe=False,
                prev_is_moe=False,
                dense_batch_invariant=False,
                query_sharded=flag,
            )


def test_sharded_models_read_their_rows_from_tensors_not_the_chunk_count():
    """``ctx.input_num_tokens`` is the scheduler's whole chunk; a model under
    query context parallelism sizes its rows from its tensors or
    ``ctx.query_shard``. Only collective sizing may read the chunk count."""
    allowed = {
        RUNTIME / "distributed" / "comm_manager.py",
        RUNTIME / "models" / "base" / "comm_ops.py",
    }
    for path in (
        RUNTIME / "models" / "longcat_flash.py",
        RUNTIME / "models" / "base" / "causal_lm.py",
    ):
        assert path not in allowed
        hits = re.findall(r"input_num_tokens", path.read_text())
        assert not hits, f"{path.name} reads ctx.input_num_tokens: {hits}"


def test_the_model_exit_gathers_sampled_rows_and_keeps_the_shard_capture(monkeypatch):
    """``BaseCausalLM.exit_logits`` under a shard: the LM head sees the
    gathered ``[bs, H]`` rows, the context marks them selected, and a FULL
    capture for the drafter stays the shard's own rows."""
    from types import SimpleNamespace

    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.forward_batch_info import (
        CaptureHiddenMode,
        ForwardMode,
    )
    from tokenspeed.runtime.models.base import causal_lm

    lengths = [4, 1, 5]
    plan = QueryShardPlan.from_forward(
        total_tokens=10, input_lengths=lengths, size=4, rank=1
    )
    shard = torch.arange(plan.local_rows * 2, dtype=torch.float32).reshape(-1, 2)
    gathered = torch.full((3, 2), 9.0)
    seen = {}

    def fake_gather(hidden_states, ctx, *, group):
        seen["gather"] = (hidden_states, group)
        return gathered

    class Processor:
        def __call__(self, input_ids, hidden_states, lm_head, metadata, aux):
            seen["processor"] = (hidden_states, metadata, aux)
            return SimpleNamespace(hidden_states=hidden_states)

    monkeypatch.setattr(causal_lm, "gather_sampled_rows", fake_gather)
    model = causal_lm.BaseCausalLM.__new__(causal_lm.BaseCausalLM)
    model.mapping = SimpleNamespace(attn=SimpleNamespace(qcp_group=(0, 1, 2, 3)))
    model.logits_processor = Processor()
    model.lm_head = object()
    for mode in (CaptureHiddenMode.FULL, CaptureHiddenMode.LAST):
        ctx = ForwardContext(
            attn_backend=None,
            token_to_kv_pool=None,
            bs=3,
            num_extends=3,
            input_num_tokens=10,
            forward_mode=ForwardMode.EXTEND,
            output_layout=None,
            capture_hidden_mode=mode,
            gather_ids=torch.cumsum(torch.tensor(lengths), 0) - 1,
            query_shard=plan,
        )
        out = model.exit_logits(None, shard, None, ctx)
        assert seen["gather"] == (shard, (0, 1, 2, 3))
        assert seen["processor"][0] is gathered
        assert ctx.logits_rows_selected and seen["processor"][1].logits_rows_selected
        assert out.hidden_states is (
            shard if mode is CaptureHiddenMode.FULL else gathered
        )
    # Unsharded forwards do not gather and leave the rows as given.
    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=3,
        num_extends=3,
        input_num_tokens=10,
        forward_mode=ForwardMode.EXTEND,
        output_layout=None,
        gather_ids=torch.cumsum(torch.tensor(lengths), 0) - 1,
    )
    seen.clear()
    model.exit_logits(None, shard, None, ctx)
    assert "gather" not in seen and seen["processor"][0] is shard
    assert not ctx.logits_rows_selected


def test_the_executor_hands_the_model_its_shard_of_the_row_inputs():
    """``_model_input_kwargs`` slices per-row inputs by the shard and gives
    the per-request history view the shard's row offset."""
    from types import SimpleNamespace

    from tokenspeed.runtime.execution.model_executor import ModelExecutor

    executor = ModelExecutor.__new__(ModelExecutor)
    history = torch.arange(10).unsqueeze(1)
    executor.input_buffers = SimpleNamespace(
        ngram_model_kwargs=lambda n: {"engram_previous_tokens": history[:n]},
        req_pool_indices_buf=torch.arange(3),
        input_start_offsets_buf=torch.tensor([0, 4, 5, 10]),
        active_request_mask_buf=torch.ones(3, dtype=torch.bool),
    )
    views = {}
    executor.runtime_states = SimpleNamespace(
        has_request_token_history=True,
        request_token_history_view=lambda **kw: views.update(kw) or "view",
    )
    kwargs = executor._model_input_kwargs(10, 3, slice(3, 6))
    assert torch.equal(kwargs["engram_previous_tokens"], history[3:6])
    assert kwargs["request_token_history"] == "view"
    assert views["row_offset"] == 3
    assert views["input_start_offsets"].tolist() == [0, 4, 5, 10]
    kwargs = executor._model_input_kwargs(10, 3, slice(0, 10))
    assert torch.equal(kwargs["engram_previous_tokens"], history)
    assert views["row_offset"] == 0


def test_the_drafters_first_step_reads_its_shard_and_keeps_full_gather_ids(
    monkeypatch,
):
    """Eagle's step-0 extend under a shard: the shifted prefill ids are the
    shard's slice (after the last-token patch over the whole span) while the
    draft's ``gather_ids`` stay the batch's full layout, exactly as on the
    target's context -- one convention, so the model exit's
    ``gather_sampled_rows`` is the one place that cuts them to the shard."""
    from types import SimpleNamespace

    from tokenspeed.runtime.distributed import comm_manager
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.drafter.eagle import Eagle
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

    lengths = [4, 1, 5]  # sampled rows 3, 4, 9 over shards [3, 3, 2, 2]
    total = sum(lengths)
    rows = (torch.cumsum(torch.tensor(lengths), 0) - 1).tolist()
    shifted = torch.arange(100, 100 + total)
    shifted[torch.tensor(rows)] = -1  # the last token of every request
    drafter = Eagle.__new__(Eagle)
    drafter.spec_num_tokens = 1
    drafter.input_buffers = SimpleNamespace(
        shifted_prefill_ids_buf=shifted.clone(),
        input_lengths_buf=torch.tensor(lengths),
    )
    sampled = torch.tensor([7, 8, 9])
    # The draft's final hidden rows on every rank: row r of the span is r.
    hidden = torch.arange(total, dtype=torch.float32).unsqueeze(1)
    gathered_by_rank = {}
    for rank in range(4):
        plan = QueryShardPlan.from_forward(
            total_tokens=total, input_lengths=lengths, size=4, rank=rank
        )
        draft_input = SimpleNamespace(
            num_extends=3,
            base_model_output=sampled,
            accept_lengths=torch.ones(3, dtype=torch.int64),
            query_shard=plan,
        )
        ids, gather_ids = drafter._get_first_step_input(draft_input, 3, total)
        assert ids.shape[0] == plan.local_rows
        patched = shifted.clone()
        patched[torch.tensor(rows)] = sampled
        assert torch.equal(ids, patched[plan.local_slice])
        assert gather_ids.tolist() == rows  # full layout, not re-based
        # The exit selects this rank's sampled rows from its shard.
        local = plan.local_sampled_ids(gather_ids)
        expected = [
            r - plan.local_start for r in rows if plan.local_start <= r < plan.local_end
        ]
        assert local.tolist() == expected
        assert len(expected) == plan.local_sampled_rows == [0, 2, 0, 1][rank]
        ctx = ForwardContext(
            attn_backend=None,
            token_to_kv_pool=None,
            bs=3,
            num_extends=3,
            input_num_tokens=total,
            forward_mode=ForwardMode.EXTEND,
            output_layout=None,
            gather_ids=gather_ids,
            query_shard=plan,
        )
        monkeypatch.setattr(
            comm_manager,
            "token_all_gather",
            lambda t, g, counts, r=rank: gathered_by_rank.setdefault(
                r, (t.clone(), counts)
            ),
        )
        comm_manager.gather_sampled_rows(
            hidden[plan.local_slice], ctx, group=(0, 1, 2, 3)
        )
        contributed, counts = gathered_by_rank[rank]
        assert counts == [0, 2, 0, 1]
        # Rank 1 contributes rows 3 and 4, rank 3 row 9; the others none.
        assert contributed.flatten().tolist() == [
            r for r in rows if plan.local_start <= r < plan.local_end
        ]
    # The concatenation in rank order is the batch's sampled rows in request order.
    assert (
        torch.cat([gathered_by_rank[r][0] for r in range(4)]).flatten().tolist() == rows
    )
    # A shard's slice must not be handed in as the full layout.
    with pytest.raises(ValueError, match="full layout"):
        QueryShardPlan.from_forward(
            total_tokens=total, input_lengths=lengths, size=4, rank=1
        ).local_sampled_ids(torch.tensor([0, 1]))

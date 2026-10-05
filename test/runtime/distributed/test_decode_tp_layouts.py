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

"""Decode-side TP layouts under attention DP, on CPU over gloo.

Four ranks, attention TP 1 / DP 4, with heads, the dense MLP and the LM head
sharded over the whole group. Rank 1 owns no rows (an idle DP rank), so every
exchange runs with an uneven, zero-including row split.

Covered:
* ``all_to_all_transpose`` / ``all_to_all_head_scatter`` round trip;
* ``CommManager`` row-count helpers and the batch-invariant dense tail
  (``pre_dense_comm`` -> column-parallel ``down_proj`` -> ``post_dense_comm``)
  against a replicated dense MLP;
* ``LogitsProcessor`` with the LM head vocab-sharded over DP ranks against a
  replicated head;
* ``DeepseekV3AttentionMLA`` under head TP with the batch-invariant o_proj
  against the TP1 replicated layer, with core attention stubbed per row and
  head (every head of a token sees that token's own KV, as the real kernel
  does), so the exchanges, the prologue's one-row-count contract and the
  weight sharding are what is tested.

The GPU test the stubs stand in for: a head-TP + ``--tp-batch-invariant attn``
decode engine must produce the TP1 replicated engine's bits under the aok
GEMMs (``--numerics rl-bitwise``).
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from _tp_layout_fakes import (
    HIDDEN,
    NUM_HEADS,
    attention_weights,
    build_attention,
    init_gloo,
)

from tokenspeed.runtime.distributed.comm_manager import (
    CommManager,
    dp_group_row_counts,
    forward_collective_row_table,
    forward_input_row_table,
)
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

WORLD = 4
# Rows each DP rank owns; rank 1 is idle.
ROW_COUNTS = [2, 0, 3, 1]


def _mapping(
    rank: int, *, head_tp: bool, head_tp_size: int = WORLD, dense_tp_size: int = WORLD
) -> Mapping:
    return Mapping(
        rank=rank,
        world_size=WORLD,
        attn_tp_size=1,
        attn_dp_size=WORLD,
        attn_head_tp_size=head_tp_size if head_tp else None,
        lm_head_tp_size=WORLD,
        dense_tp_size=dense_tp_size,
        nprocs_per_node=WORLD,
    )


def _ctx(attn_backend, rank: int, row_counts: list[int] = ROW_COUNTS) -> ForwardContext:
    return ForwardContext(
        attn_backend=attn_backend,
        token_to_kv_pool=None,
        bs=row_counts[rank],
        num_extends=0,
        input_num_tokens=row_counts[rank],
        forward_mode=ForwardMode.DECODE if row_counts[rank] else ForwardMode.IDLE,
        output_layout=None,
        global_num_tokens=list(row_counts),
        global_bs=list(row_counts),
        all_decode_or_idle=True,
    )


def _own_rows(rank: int, row_counts: list[int] = ROW_COUNTS) -> slice:
    start = sum(row_counts[:rank])
    return slice(start, start + row_counts[rank])


# ---------------------------------------------------------------------------
# Comm primitives and CommManager
# ---------------------------------------------------------------------------


def _worker_comm(rank: int, rendezvous: str) -> None:
    from tokenspeed.runtime.distributed.comm_ops import (
        all_to_all_head_scatter,
        all_to_all_transpose,
    )

    mapping = _mapping(rank, head_tp=True)
    init_gloo(rank, rendezvous, mapping)
    try:
        group = mapping.attn.head_tp_group
        heads_local, dim = 3, 5
        width = heads_local * dim
        rows_full = sum(ROW_COUNTS)
        full = torch.arange(rows_full * WORLD * width, dtype=torch.float32).view(
            rows_full, WORLD * width
        )
        # This rank's feature shard of every rank's rows ...
        shard = full[:, rank * width : (rank + 1) * width].contiguous()
        own = all_to_all_transpose(shard, group, input_split_sizes=ROW_COUNTS)
        # ... becomes this rank's rows with every shard, in rank order.
        torch.testing.assert_close(own, full[_own_rows(rank)])
        back = all_to_all_head_scatter(
            own.view(-1, WORLD * heads_local, dim), group, output_split_sizes=ROW_COUNTS
        )
        torch.testing.assert_close(back, shard.view(rows_full, heads_local, dim))

        with pytest.raises(ValueError, match="sum to"):
            all_to_all_transpose(shard, group, input_split_sizes=[1] * WORLD)

        # The row tables by forward phase: the input rows, and the collective
        # rows a narrowing drafter reports; a rank's own count is checked.
        ctx = _ctx(SimpleNamespace(), rank)
        own = ROW_COUNTS[rank]
        assert dp_group_row_counts(forward_input_row_table(ctx), group, rank, own) == (
            ROW_COUNTS
        )
        narrowed = replace(ctx, collective_global_num_tokens=[1, 0, 1, 1])
        assert forward_collective_row_table(ctx) is ctx.global_num_tokens
        assert dp_group_row_counts(
            forward_collective_row_table(narrowed), group, rank, 1 if own else 0
        ) == [1, 0, 1, 1]
        with pytest.raises(ValueError, match="holds"):
            dp_group_row_counts(forward_input_row_table(ctx), group, rank, own + 1)

        # CommManager: the batch-invariant dense tail returns each rank's
        # rows of the full hidden.
        cm = CommManager(
            mapping,
            layer_id=1,
            is_moe=False,
            prev_is_moe=False,
            dense_batch_invariant=True,
            query_sharded=False,
        )

        hidden_full = torch.randn(
            rows_full, HIDDEN, generator=torch.Generator().manual_seed(7)
        )
        gathered = cm.pre_dense_comm(hidden_full[_own_rows(rank)].contiguous(), ctx)
        torch.testing.assert_close(gathered, hidden_full)
        shard_w = HIDDEN // WORLD
        down_out = hidden_full[:, rank * shard_w : (rank + 1) * shard_w].contiguous()
        back, _ = cm.post_dense_comm(down_out, None, ctx)
        torch.testing.assert_close(back, hidden_full[_own_rows(rank)])
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_transpose_round_trip_and_dense_tail(tmp_path):
    mp.spawn(_worker_comm, args=((tmp_path / "rv").as_uri(),), nprocs=WORLD, join=True)


# ---------------------------------------------------------------------------
# Dense MLP: batch-invariant TP == replicated
# ---------------------------------------------------------------------------


def _worker_dense(rank: int, rendezvous: str) -> None:
    from tokenspeed.runtime.models.deepseek_v3 import DeepseekV3MLP

    mapping = _mapping(rank, head_tp=True)
    init_gloo(rank, rendezvous, mapping)
    try:
        intermediate = 48
        gen = torch.Generator().manual_seed(11)
        gate_up_w = torch.randn(2 * intermediate, HIDDEN, generator=gen)
        down_w = torch.randn(HIDDEN, intermediate, generator=gen)
        rows_full = sum(ROW_COUNTS)
        x_full = torch.randn(rows_full, HIDDEN, generator=gen)

        sharded = DeepseekV3MLP(
            HIDDEN,
            intermediate,
            "silu",
            mapping,
            None,
            "mlp",
            False,
            batch_invariant=True,
        )
        # The merged loader takes gate and up separately; the column loader narrows.
        gate_w, up_w = gate_up_w.split(intermediate, dim=0)
        sharded.gate_up_proj.weight_loader(sharded.gate_up_proj.weight, gate_w, 0)
        sharded.gate_up_proj.weight_loader(sharded.gate_up_proj.weight, up_w, 1)
        sharded.down_proj.weight_loader(sharded.down_proj.weight, down_w)

        replicated_mapping = Mapping(
            rank=rank,
            world_size=WORLD,
            attn_tp_size=1,
            attn_dp_size=WORLD,
            dense_tp_size=1,
        )
        replicated = DeepseekV3MLP(
            HIDDEN,
            intermediate,
            "silu",
            replicated_mapping,
            None,
            "mlp",
            False,
            batch_invariant=False,
        )
        replicated.gate_up_proj.weight_loader(replicated.gate_up_proj.weight, gate_w, 0)
        replicated.gate_up_proj.weight_loader(replicated.gate_up_proj.weight, up_w, 1)
        replicated.down_proj.weight_loader(replicated.down_proj.weight, down_w)

        cm = CommManager(
            mapping,
            layer_id=1,
            is_moe=False,
            prev_is_moe=False,
            dense_batch_invariant=True,
            query_sharded=False,
        )
        ctx = _ctx(SimpleNamespace(), rank)
        own = x_full[_own_rows(rank)].contiguous()
        hidden = cm.pre_dense_comm(own, ctx)
        hidden = sharded(hidden)
        assert tuple(hidden.shape) == (rows_full, HIDDEN // WORLD)
        hidden, _ = cm.post_dense_comm(hidden, None, ctx)
        expected = replicated(own)
        assert (
            tuple(hidden.shape) == tuple(expected.shape) == (ROW_COUNTS[rank], HIDDEN)
        )
        torch.testing.assert_close(hidden, expected, atol=1e-4, rtol=1e-4)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_batch_invariant_dense_matches_replicated(tmp_path):
    mp.spawn(_worker_dense, args=((tmp_path / "rv").as_uri(),), nprocs=WORLD, join=True)


# ---------------------------------------------------------------------------
# LM head TP under DP == replicated head
# ---------------------------------------------------------------------------


def _worker_lm_head(rank: int, rendezvous: str) -> None:
    from unittest import mock

    from tokenspeed.runtime.layers import logits_processor as lp_module
    from tokenspeed.runtime.layers.logits_processor import (
        LogitsMetadata,
        LogitsProcessor,
    )

    mapping = _mapping(rank, head_tp=False)
    init_gloo(rank, rendezvous, mapping)
    try:
        vocab, padded_vocab = 30, 32
        gen = torch.Generator().manual_seed(5)
        weight = torch.randn(padded_vocab, HIDDEN, generator=gen)
        rows_full = sum(ROW_COUNTS)
        hidden_full = torch.randn(rows_full, HIDDEN, generator=gen)
        config = SimpleNamespace(model_type="test", vocab_size=vocab)

        processor = LogitsProcessor(
            config,
            skip_all_gather=True,
            tp_rank=mapping.lm_head.tp_rank,
            tp_size=mapping.lm_head.tp_size,
            tp_group=mapping.lm_head.tp_group,
            dp_lm_head_tp=True,
        )
        shard = padded_vocab // WORLD
        lm_head = SimpleNamespace(weight=weight[rank * shard : (rank + 1) * shard])
        own = hidden_full[_own_rows(rank)].contiguous()
        expected = (own @ weight.T)[:, :vocab]

        # The decode path reads the row counts from the host tables: no
        # count exchange (a device sync) is issued.
        decode = LogitsMetadata.from_forward_context(_ctx(SimpleNamespace(), rank))
        with mock.patch.object(
            lp_module, "all_gather", wraps=lp_module.all_gather
        ) as ag:
            logits = processor._get_logits(
                own, lm_head, decode, require_full_vocab=False
            )
            assert ag.call_count == 0
        assert tuple(logits.shape) == (ROW_COUNTS[rank], vocab)
        torch.testing.assert_close(logits, expected, atol=1e-5, rtol=1e-5)

        # A narrowing drafter's live rows follow its collective table.
        live = [1, 0, 1, 1]
        narrowed = LogitsMetadata.from_forward_context(
            replace(_ctx(SimpleNamespace(), rank), collective_global_num_tokens=live)
        )
        logits = processor._get_logits(
            own[: live[rank]], lm_head, narrowed, require_full_vocab=False
        )
        torch.testing.assert_close(logits, expected[: live[rank]], atol=1e-5, rtol=1e-5)

        # A shape with no table (not every rank decodes) exchanges the counts.
        mixed = LogitsMetadata(
            ForwardMode.DECODE, all_decode_or_idle=False, query_shard=None
        )
        with mock.patch.object(
            lp_module, "all_gather", wraps=lp_module.all_gather
        ) as ag:
            logits = processor._get_logits(
                own, lm_head, mixed, require_full_vocab=False
            )
            assert ag.call_count == 1
        torch.testing.assert_close(logits, expected, atol=1e-5, rtol=1e-5)

        # The full forward: a rank whose model selected no logits rows still
        # joins the group's exchange instead of returning early.
        selected = replace(
            LogitsMetadata.from_forward_context(_ctx(SimpleNamespace(), rank)),
            logits_rows_selected=True,
        )
        rows = own if rank != 1 else own[:0]
        out = processor.forward(None, rows, lm_head, selected)
        assert tuple(out.next_token_logits.shape) == (ROW_COUNTS[rank], vocab)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_lm_head_tp_under_dp_matches_replicated(tmp_path):
    mp.spawn(
        _worker_lm_head, args=((tmp_path / "rv").as_uri(),), nprocs=WORLD, join=True
    )


# ---------------------------------------------------------------------------
# MLA attention under head TP + batch-invariant o_proj == TP1 replicated
# ---------------------------------------------------------------------------


def _replicated_mapping(rank: int) -> Mapping:
    return Mapping(
        rank=rank,
        world_size=WORLD,
        attn_tp_size=1,
        attn_dp_size=WORLD,
        dense_tp_size=1,
    )


def _worker_attention(
    rank: int,
    rendezvous: str,
    tp_batch_invariant: str,
    row_counts: list[int],
    head_tp_size: int,
) -> None:
    from tokenspeed.runtime.utils.env import global_server_args_dict

    mapping = _mapping(rank, head_tp=True, head_tp_size=head_tp_size)
    init_gloo(rank, rendezvous, mapping)
    try:
        weights = attention_weights(HIDDEN)
        rows_full = sum(row_counts)
        gen = torch.Generator().manual_seed(4)
        hidden_full = torch.randn(rows_full, HIDDEN, generator=gen)
        positions_full = torch.arange(rows_full, dtype=torch.int64) * 3
        own = _own_rows(rank, row_counts)

        # TP1 replicated reference on this rank's rows.
        global_server_args_dict["tp_batch_invariant"] = "none"
        reference = build_attention(_replicated_mapping(rank), weights)
        assert not reference.has_head_tp and reference.num_local_heads == NUM_HEADS

        # Head TP over the DP ranks of the group: the batch-invariant o_proj
        # (column-parallel + transpose) or the row-parallel one whose head
        # partials are reduce-scattered.
        global_server_args_dict["tp_batch_invariant"] = tp_batch_invariant
        sharded = build_attention(mapping, weights)
        assert sharded.has_head_tp
        assert sharded.num_local_heads == NUM_HEADS // head_tp_size
        assert type(sharded.o_proj).__name__ == (
            "ColumnParallelLinear"
            if tp_batch_invariant == "attn"
            else "RowParallelLinear"
        )
        assert sharded.attn_mqa.layer_id == 0

        backend = SimpleNamespace(
            spec_num_tokens=1,
            write_locations=lambda layer, mode: torch.arange(
                row_counts[rank], dtype=torch.int64
            ),
        )
        ctx = _ctx(backend, rank, row_counts)
        comm = CommManager(
            mapping,
            layer_id=0,
            is_moe=False,
            prev_is_moe=False,
            dense_batch_invariant=False,
            query_sharded=False,
        )

        hidden = hidden_full[own].contiguous()
        positions = positions_full[own].contiguous()
        expected = reference(positions, hidden, ctx, comm)
        actual = sharded(positions, hidden, ctx, comm)
        assert tuple(actual.shape) == (row_counts[rank], HIDDEN)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
        # The idle rank ran the exchanges but attended nothing.
        assert sharded.attn_mqa.calls == (1 if row_counts[rank] else 0)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("tp_batch_invariant", ["attn", "none"])
def test_head_tp_attention_matches_tp1(tmp_path, tp_batch_invariant):
    mp.spawn(
        _worker_attention,
        args=((tmp_path / "rv").as_uri(), tp_batch_invariant, ROW_COUNTS, WORLD),
        nprocs=WORLD,
        join=True,
    )


@pytest.mark.parametrize("tp_batch_invariant", ["attn", "none"])
def test_wholly_idle_head_group_skips_its_collectives(tmp_path, tp_batch_invariant):
    """Two head groups of two ranks; ranks 0 and 1 have no rows at all, so
    their group moves nothing and must not wait on a collective."""
    mp.spawn(
        _worker_attention,
        args=((tmp_path / "rv").as_uri(), tp_batch_invariant, [0, 0, 3, 1], 2),
        nprocs=WORLD,
        join=True,
    )


# ---------------------------------------------------------------------------
# Decoder layers on a narrowing draft step with an idle rank
# ---------------------------------------------------------------------------

# Verify-window rows per request on the draft's first step.
VERIFY = 2
# Live rows (one per request) after the step narrows; rank 1 is idle.
LIVE_COUNTS = [2, 0, 3, 1]
INPUT_COUNTS = [VERIFY * n for n in LIVE_COUNTS]


class _Narrowing:
    def publish_accepted_prefix(self) -> None:
        pass


def _narrowing_ctx(rank: int) -> ForwardContext:
    """The draft's first step: ``INPUT_COUNTS`` verify rows in, the
    ``LIVE_COUNTS`` live rows reported as the collective sizing; the idle
    rank runs the plain IDLE forward with the same tables."""
    live, rows = LIVE_COUNTS[rank], INPUT_COUNTS[rank]
    gather_ids = (
        torch.arange(live, dtype=torch.int64) * VERIFY + (VERIFY - 1) if live else None
    )
    backend = SimpleNamespace(
        spec_num_tokens=VERIFY,
        forward_write_locations=lambda layer, mode: torch.arange(
            rows, dtype=torch.int64
        ),
        write_locations=lambda layer, mode: torch.arange(rows, dtype=torch.int64),
        override_num_extends=lambda n: nullcontext(),
    )
    return ForwardContext(
        attn_backend=backend,
        token_to_kv_pool=None,
        bs=live,
        num_extends=0,
        input_num_tokens=rows,
        forward_mode=ForwardMode.DECODE if live else ForwardMode.IDLE,
        output_layout=None,
        global_num_tokens=list(INPUT_COUNTS),
        global_bs=list(LIVE_COUNTS),
        all_decode_or_idle=True,
        collective_num_tokens=live,
        collective_global_num_tokens=list(LIVE_COUNTS),
        gather_ids=gather_ids,
        draft_narrowing=_Narrowing() if live else None,
    )


def _norm_stub(hidden, residual=None):
    return hidden if residual is None else (hidden, residual)


def _draft_layer(mapping: Mapping, attn):
    """``DeepseekV3DraftDecoderLayer`` around the given attention with an
    identity MLP and identity norms; dense TP 1 keeps the MLP comm a no-op so
    the layer's output is the attention's."""
    from torch import nn

    from tokenspeed.runtime.models.deepseek_nextn import DeepseekV3DraftDecoderLayer

    layer = DeepseekV3DraftDecoderLayer.__new__(DeepseekV3DraftDecoderLayer)
    nn.Module.__init__(layer)
    layer.mapping = mapping
    layer.layer_id = 0
    layer.hidden_size = HIDDEN
    layer.self_attn = attn
    layer.is_moe_layer = False
    layer.mlp = lambda hidden: hidden
    layer.input_layernorm = _norm_stub
    layer.post_attention_layernorm = _norm_stub
    layer.comm_manager = CommManager(
        mapping,
        layer_id=0,
        is_moe=False,
        prev_is_moe=False,
        dense_batch_invariant=False,
        query_sharded=False,
        input_layernorm=_norm_stub,
        post_attn_layernorm=_norm_stub,
    )
    return layer


def _eagle3_layer(mapping: Mapping, attn):
    """``Eagle3MlaDecoderLayer`` around the given attention (its
    ``fused_qkv_a_proj_with_mqa`` takes ``[embeds || hidden]``)."""
    from torch import nn

    from tokenspeed.runtime.models.deepseek_v3 import Eagle3MlaDecoderLayer

    layer = Eagle3MlaDecoderLayer.__new__(Eagle3MlaDecoderLayer)
    nn.Module.__init__(layer)
    layer.mapping = mapping
    layer.layer_id = 0
    layer.hidden_size = HIDDEN
    layer.self_attn = attn
    layer.mlp = lambda hidden: hidden

    def fused_input_hidden_norm(*, input_q_a, input_kv_a, output_q_a, output_kv_a):
        output_q_a.copy_(input_q_a)
        output_kv_a.copy_(input_kv_a)

    layer.fused_input_hidden_norm = fused_input_hidden_norm
    layer.comm_manager = CommManager(
        mapping,
        layer_id=0,
        is_moe=False,
        prev_is_moe=False,
        dense_batch_invariant=False,
        query_sharded=False,
        post_attn_layernorm=_norm_stub,
    )
    return layer


def _worker_draft_layers(rank: int, rendezvous: str) -> None:
    from tokenspeed.runtime.models.deepseek_v3 import DeepseekV3DraftAttentionMLA
    from tokenspeed.runtime.utils.env import global_server_args_dict

    mapping = _mapping(rank, head_tp=True, dense_tp_size=1)
    init_gloo(rank, rendezvous, mapping)
    try:
        global_server_args_dict["tp_batch_invariant"] = "none"
        rows_full = sum(INPUT_COUNTS)
        gen = torch.Generator().manual_seed(9)
        hidden_full = torch.randn(rows_full, HIDDEN, generator=gen)
        embeds_full = torch.randn(rows_full, HIDDEN, generator=gen)
        positions_full = torch.arange(rows_full, dtype=torch.int64) * 3
        own = _own_rows(rank, INPUT_COUNTS)
        hidden = hidden_full[own].contiguous()
        embeds = embeds_full[own].contiguous()
        positions = positions_full[own].contiguous()
        live = LIVE_COUNTS[rank]
        ctx = _narrowing_ctx(rank)

        # NextN-style layer: [N, H] in, [bs, H] out on the narrowing step.
        weights = attention_weights(HIDDEN)
        reference = _draft_layer(
            _replicated_mapping(rank),
            build_attention(
                _replicated_mapping(rank), weights, DeepseekV3DraftAttentionMLA
            ),
        )
        sharded = _draft_layer(
            mapping, build_attention(mapping, weights, DeepseekV3DraftAttentionMLA)
        )
        expected, expected_residual = reference(positions, hidden, ctx, None)
        actual, residual = sharded(positions, hidden, ctx, None)
        assert tuple(actual.shape) == (live, HIDDEN)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
        if live:
            torch.testing.assert_close(residual, expected_residual)
            torch.testing.assert_close(residual, hidden[ctx.gather_ids])
        else:
            assert residual is None
        assert sharded.self_attn.attn_mqa.calls == (1 if live else 0)

        # Eagle3 layer: [embeds || hidden] in, [bs, H] out.
        weights2 = attention_weights(2 * HIDDEN)
        reference = _eagle3_layer(
            _replicated_mapping(rank),
            build_attention(
                _replicated_mapping(rank), weights2, DeepseekV3DraftAttentionMLA
            ),
        )
        sharded = _eagle3_layer(
            mapping, build_attention(mapping, weights2, DeepseekV3DraftAttentionMLA)
        )
        expected, _ = reference(positions, embeds, hidden, ctx, None)
        actual, residual = sharded(positions, embeds, hidden, ctx, None)
        assert tuple(actual.shape) == (live, HIDDEN)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
        assert sharded.self_attn.attn_mqa.calls == (1 if live else 0)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_draft_layers_narrow_with_an_idle_rank(tmp_path):
    """The draft's first step narrows to the live rows while rank 1 runs an
    IDLE forward of the same layer: the input and collective row tables
    differ, every rank sizes the exchanges by the right one, and the idle
    layer joins the head group's collectives through the attention."""
    mp.spawn(
        _worker_draft_layers,
        args=((tmp_path / "rv").as_uri(),),
        nprocs=WORLD,
        join=True,
    )


# ---------------------------------------------------------------------------
# Host-side helpers (no distributed)
# ---------------------------------------------------------------------------


def test_dp_group_row_counts_reads_the_group_and_checks_this_rank():
    table = [5, 0, 2, 7, 1, 1, 0, 3]
    assert dp_group_row_counts(table, (4, 5, 6, 7), 6, 0) == [1, 1, 0, 3]
    with pytest.raises(ValueError, match="holds 4 rows"):
        dp_group_row_counts(table, (4, 5, 6, 7), 6, 4)
    with pytest.raises(ValueError, match="attention DP"):
        dp_group_row_counts(None, (0, 1), 0, 1)


def test_dense_batch_invariant_needs_a_token_scatter_tail():
    same_tp = Mapping(rank=0, world_size=8, attn_tp_size=8)
    with pytest.raises(ValueError, match="dense TP group"):
        CommManager(
            same_tp,
            layer_id=0,
            is_moe=False,
            prev_is_moe=False,
            dense_batch_invariant=True,
            query_sharded=False,
        )
    dp_dense_tp = Mapping(
        rank=0,
        world_size=8,
        attn_tp_size=1,
        attn_dp_size=8,
        dense_tp_size=8,
    )
    cm = CommManager(
        dp_dense_tp,
        layer_id=0,
        is_moe=False,
        prev_is_moe=False,
        dense_batch_invariant=True,
        query_sharded=False,
    )
    assert cm.dense_batch_invariant

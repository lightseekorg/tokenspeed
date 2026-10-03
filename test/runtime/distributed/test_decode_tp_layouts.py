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

from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tokenspeed.runtime.distributed.comm_manager import (
    CommManager,
    dp_group_row_counts,
)
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

WORLD = 4
# Rows each DP rank owns; rank 1 is idle.
ROW_COUNTS = [2, 0, 3, 1]
HIDDEN = 32


def _init_gloo(rank: int, rendezvous: str, mapping: Mapping) -> None:
    """One gloo world; every group the layouts use is registered under the
    device-backend key as well, so the NCCL-named backend path runs on CPU."""
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as pg_manager,
    )
    from tokenspeed.runtime.utils.env import global_server_args_dict

    pg_manager.init_distributed(
        mapping,
        distributed_init_method=rendezvous,
        backend="gloo",
        timeout=60,
    )
    # Same creation order on every rank; a size-1 group needs no collective.
    for group in sorted(
        {mapping.attn.head_tp_group, mapping.dense.tp_group, mapping.lm_head.tp_group}
    ):
        if len(group) == 1:
            continue
        pg_manager.init_process_group(group, backend="gloo")
        pg_manager.register_process_group(
            "nccl", group, pg_manager.get_process_group("gloo", group)
        )
    # NCCL (here gloo) collectives instead of symmetric-memory kernels.
    global_server_args_dict["force_deterministic_rsag"] = True
    global_server_args_dict["mapping"] = mapping


def _mapping(rank: int, *, head_tp: bool) -> Mapping:
    return Mapping(
        rank=rank,
        world_size=WORLD,
        attn_tp_size=1,
        attn_cp_size=1,
        attn_dp_size=WORLD,
        attn_head_tp_size=WORLD if head_tp else None,
        lm_head_tp_size=WORLD,
        dense_tp_size=WORLD,
        nprocs_per_node=WORLD,
    )


def _ctx(attn_backend, rank: int) -> ForwardContext:
    return ForwardContext(
        attn_backend=attn_backend,
        token_to_kv_pool=None,
        bs=ROW_COUNTS[rank],
        num_extends=0,
        input_num_tokens=ROW_COUNTS[rank],
        forward_mode=ForwardMode.DECODE if ROW_COUNTS[rank] else ForwardMode.IDLE,
        output_layout=None,
        global_num_tokens=list(ROW_COUNTS),
        global_bs=list(ROW_COUNTS),
        all_decode_or_idle=True,
    )


def _own_rows(rank: int) -> slice:
    start = sum(ROW_COUNTS[:rank])
    return slice(start, start + ROW_COUNTS[rank])


# ---------------------------------------------------------------------------
# Comm primitives and CommManager
# ---------------------------------------------------------------------------


def _worker_comm(rank: int, rendezvous: str) -> None:
    from tokenspeed.runtime.distributed.comm_ops import (
        all_to_all_head_scatter,
        all_to_all_transpose,
    )

    mapping = _mapping(rank, head_tp=True)
    _init_gloo(rank, rendezvous, mapping)
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

        # CommManager: the batch-invariant dense tail returns each rank's
        # rows of the full hidden, and the row-count helpers read the tables.
        cm = CommManager(
            mapping,
            layer_id=1,
            is_moe=False,
            prev_is_moe=False,
            dense_batch_invariant=True,
        )
        ctx = _ctx(SimpleNamespace(), rank)
        assert cm.head_tp_group_scattered_input_num_tokens(ctx, ROW_COUNTS[rank]) == (
            ROW_COUNTS
        )
        assert cm.head_tp_group_scattered_num_tokens(ctx, ROW_COUNTS[rank]) == (
            ROW_COUNTS
        )
        narrowed = replace(ctx, collective_global_num_tokens=[1, 0, 1, 1])
        assert cm.head_tp_group_scattered_num_tokens(narrowed, 1 if rank != 1 else 0) == [
            1,
            0,
            1,
            1,
        ]
        with pytest.raises(ValueError, match="holds"):
            cm.head_tp_group_scattered_num_tokens(ctx, ROW_COUNTS[rank] + 1)

        hidden_full = torch.randn(rows_full, HIDDEN, generator=torch.Generator().manual_seed(7))
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
    mp.spawn(
        _worker_comm, args=((tmp_path / "rv").as_uri(),), nprocs=WORLD, join=True
    )


# ---------------------------------------------------------------------------
# Dense MLP: batch-invariant TP == replicated
# ---------------------------------------------------------------------------


def _worker_dense(rank: int, rendezvous: str) -> None:
    from tokenspeed.runtime.models.deepseek_v3 import DeepseekV3MLP

    mapping = _mapping(rank, head_tp=True)
    _init_gloo(rank, rendezvous, mapping)
    try:
        intermediate = 48
        gen = torch.Generator().manual_seed(11)
        gate_up_w = torch.randn(2 * intermediate, HIDDEN, generator=gen)
        down_w = torch.randn(HIDDEN, intermediate, generator=gen)
        rows_full = sum(ROW_COUNTS)
        x_full = torch.randn(rows_full, HIDDEN, generator=gen)

        sharded = DeepseekV3MLP(
            HIDDEN, intermediate, "silu", mapping, None, "mlp", False, batch_invariant=True
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
            attn_cp_size=1,
            attn_dp_size=WORLD,
            dense_tp_size=1,
        )
        replicated = DeepseekV3MLP(
            HIDDEN, intermediate, "silu", replicated_mapping, None, "mlp", False
        )
        replicated.gate_up_proj.weight_loader(replicated.gate_up_proj.weight, gate_w, 0)
        replicated.gate_up_proj.weight_loader(replicated.gate_up_proj.weight, up_w, 1)
        replicated.down_proj.weight_loader(replicated.down_proj.weight, down_w)

        cm = CommManager(
            mapping, layer_id=1, is_moe=False, prev_is_moe=False, dense_batch_invariant=True
        )
        ctx = _ctx(SimpleNamespace(), rank)
        own = x_full[_own_rows(rank)].contiguous()
        hidden = cm.pre_dense_comm(own, ctx)
        hidden = sharded(hidden)
        assert tuple(hidden.shape) == (rows_full, HIDDEN // WORLD)
        hidden, _ = cm.post_dense_comm(hidden, None, ctx)
        expected = replicated(own)
        assert tuple(hidden.shape) == tuple(expected.shape) == (ROW_COUNTS[rank], HIDDEN)
        torch.testing.assert_close(hidden, expected, atol=1e-4, rtol=1e-4)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_batch_invariant_dense_matches_replicated(tmp_path):
    mp.spawn(
        _worker_dense, args=((tmp_path / "rv").as_uri(),), nprocs=WORLD, join=True
    )


# ---------------------------------------------------------------------------
# LM head TP under DP == replicated head
# ---------------------------------------------------------------------------


def _worker_lm_head(rank: int, rendezvous: str) -> None:
    from tokenspeed.runtime.layers.logits_processor import (
        LogitsMetadata,
        LogitsProcessor,
    )

    mapping = _mapping(rank, head_tp=False)
    _init_gloo(rank, rendezvous, mapping)
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
        logits = processor._get_logits(own, lm_head, LogitsMetadata(ForwardMode.DECODE))
        expected = (own @ weight.T)[:, :vocab]
        assert tuple(logits.shape) == (ROW_COUNTS[rank], vocab)
        torch.testing.assert_close(logits, expected, atol=1e-5, rtol=1e-5)
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

NUM_HEADS = 8
QK_NOPE, QK_ROPE, V_DIM = 8, 4, 6
Q_LORA, KV_LORA = 16, 12


class _StubCoreAttention:
    """Stands in for ``PagedAttention`` on CPU.

    The prologue asserts the one-row-count contract and marks the RoPE
    channels with the row's position; core attention maps each (token, head)
    query through that token's own latent ("KV"), so the output of a head
    for a token depends on exactly the inputs the real kernel reads.
    """

    def __init__(self, layer_id: int):
        self.layer_id = layer_id
        self.calls = 0

    def latent_prologue(self, query, q_pe, latent_cache, positions, ctx, *, slots, expanded):
        assert expanded is None
        assert query.shape[0] == q_pe.shape[0] == latent_cache.shape[0]
        assert query.shape[0] == positions.shape[0] == slots.shape[0]
        # q_pe is the query's own RoPE channels (head TP, after the exchange)
        # or a view of the q_b output sharing no element with the query.
        rotated = query.clone()
        rotated[..., KV_LORA:] = q_pe * (positions.to(query.dtype) + 1.0)[:, None, None]
        self.latent = latent_cache[:, :KV_LORA].clone()
        return SimpleNamespace(query=rotated)

    def __call__(self, Q, k=None, v=None, positions=None, ctx=None, **kwargs):
        assert k is None and v is None
        self.calls += 1
        kv_gain = 1.0 + self.latent.sum(dim=-1)  # [T]
        out = Q[..., :KV_LORA] * kv_gain[:, None, None]
        out = out + 0.01 * Q[..., KV_LORA:].sum(dim=-1, keepdim=True)
        return out.reshape(Q.shape[0], -1)


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


def _build_attention(mapping: Mapping, weights: dict[str, torch.Tensor]):
    from tokenspeed.runtime.models.deepseek_v3 import (
        DeepseekV3AttentionMLA,
        _prepare_mla_kv_b_proj_weights,
    )

    attn = DeepseekV3AttentionMLA(
        config=SimpleNamespace(rms_norm_eps=1e-6),
        mapping=mapping,
        hidden_size=HIDDEN,
        num_heads=NUM_HEADS,
        qk_nope_head_dim=QK_NOPE,
        qk_rope_head_dim=QK_ROPE,
        v_head_dim=V_DIM,
        q_lora_rank=Q_LORA,
        kv_lora_rank=KV_LORA,
        rope_theta=10000.0,
        rope_scaling=None,
        max_position_embeddings=128,
        quant_config=None,
        layer_id=0,
        prefix="layers.0.self_attn",
        reduce_attn_results=False,
    )
    attn.fused_qkv_a_proj_with_mqa.weight.data.copy_(weights["qkv_a"])
    attn.q_b_proj.weight_loader(attn.q_b_proj.weight, weights["q_b"])
    attn.kv_b_proj.weight_loader(attn.kv_b_proj.weight, weights["kv_b"])
    attn.o_proj.weight_loader(attn.o_proj.weight, weights["o"])
    attn.q_a_layernorm.weight.data.copy_(weights["q_norm"])
    attn.kv_a_layernorm.weight.data.copy_(weights["kv_norm"])
    attn.w_kc, attn.w_vc = _prepare_mla_kv_b_proj_weights(attn.kv_b_proj.weight, attn)

    def fused_norm(input_q_a, input_kv_a, output_q_a):
        output_q_a.copy_(_rms_norm(input_q_a, weights["q_norm"]))
        input_kv_a.copy_(_rms_norm(input_kv_a, weights["kv_norm"]))

    # Replace the CUDA-only submodules with CPU stand-ins (plain attributes).
    del attn.fused_qk_layernorm
    attn.fused_qk_layernorm = fused_norm
    del attn.attn_mqa
    attn.attn_mqa = _StubCoreAttention(layer_id=0)
    return attn


def _worker_attention(rank: int, rendezvous: str, tp_batch_invariant: str) -> None:
    from tokenspeed.runtime.utils.env import global_server_args_dict

    mapping = _mapping(rank, head_tp=True)
    _init_gloo(rank, rendezvous, mapping)
    try:
        gen = torch.Generator().manual_seed(3)
        weights = {
            "qkv_a": torch.randn(Q_LORA + KV_LORA + QK_ROPE, HIDDEN, generator=gen),
            "q_b": torch.randn(NUM_HEADS * (QK_NOPE + QK_ROPE), Q_LORA, generator=gen),
            "kv_b": torch.randn(NUM_HEADS * (QK_NOPE + V_DIM), KV_LORA, generator=gen),
            "o": torch.randn(HIDDEN, NUM_HEADS * V_DIM, generator=gen),
            "q_norm": torch.rand(Q_LORA, generator=gen) + 0.5,
            "kv_norm": torch.rand(KV_LORA, generator=gen) + 0.5,
        }
        rows_full = sum(ROW_COUNTS)
        hidden_full = torch.randn(rows_full, HIDDEN, generator=gen)
        positions_full = torch.arange(rows_full, dtype=torch.int64) * 3
        own = _own_rows(rank)

        # TP1 replicated reference on this rank's rows.
        global_server_args_dict["tp_batch_invariant"] = "none"
        replicated_mapping = Mapping(
            rank=rank,
            world_size=WORLD,
            attn_tp_size=1,
            attn_cp_size=1,
            attn_dp_size=WORLD,
            dense_tp_size=1,
        )
        reference = _build_attention(replicated_mapping, weights)
        assert not reference.has_head_tp and reference.num_local_heads == NUM_HEADS

        # Head TP over the four DP ranks: the batch-invariant o_proj
        # (column-parallel + transpose) or the row-parallel one whose head
        # partials are reduce-scattered.
        global_server_args_dict["tp_batch_invariant"] = tp_batch_invariant
        sharded = _build_attention(mapping, weights)
        assert sharded.has_head_tp and sharded.num_local_heads == NUM_HEADS // WORLD
        assert type(sharded.o_proj).__name__ == (
            "ColumnParallelLinear" if tp_batch_invariant == "attn" else "RowParallelLinear"
        )
        assert sharded.attn_mqa.layer_id == 0

        backend = SimpleNamespace(
            spec_num_tokens=1,
            write_locations=lambda layer, mode: torch.arange(
                ROW_COUNTS[rank], dtype=torch.int64
            ),
        )
        ctx = _ctx(backend, rank)
        comm = CommManager(mapping, layer_id=0, is_moe=False, prev_is_moe=False)

        hidden = hidden_full[own].contiguous()
        positions = positions_full[own].contiguous()
        expected = reference(positions, hidden, ctx, comm)
        actual = sharded(positions, hidden, ctx, comm)
        assert tuple(actual.shape) == (ROW_COUNTS[rank], HIDDEN)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
        # The idle rank ran the exchanges but attended nothing.
        assert sharded.attn_mqa.calls == (1 if ROW_COUNTS[rank] else 0)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("tp_batch_invariant", ["attn", "none"])
def test_head_tp_attention_matches_tp1(tmp_path, tp_batch_invariant):
    mp.spawn(
        _worker_attention,
        args=((tmp_path / "rv").as_uri(), tp_batch_invariant),
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
        CommManager(same_tp, layer_id=0, is_moe=False, prev_is_moe=False, dense_batch_invariant=True)
    dp_dense_tp = Mapping(
        rank=0, world_size=8, attn_tp_size=1, attn_cp_size=1, attn_dp_size=8, dense_tp_size=8
    )
    cm = CommManager(
        dp_dense_tp, layer_id=0, is_moe=False, prev_is_moe=False, dense_batch_invariant=True
    )
    assert cm.dense_batch_invariant

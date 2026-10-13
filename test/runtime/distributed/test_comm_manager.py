"""
Multi-GPU test for CommManager.

Spawns real distributed workers, initializes torch.distributed + NCCL,
and runs CommManager's communication cycle with real GPU tensors:

    pre_attn(AG) → attn → post_attn(RS) → pre_dense(AG) → dense → post_dense(RS)

Verifies that with attn_tp ≠ dense_tp and uneven token counts, each rank
recovers its original hidden states after the full cycle.

The query-sharded layout (``CommManager(query_sharded=True)``) is covered on
CPU over gloo at the end of this file: identity around attention on both the
sharded and the replicated forward, the MoE all-gather / reduce-scatter round
trip over the shard's row table, the final gather off, the logits processor's
``gather_sampled_rows`` with an idle rank, and the vocab-parallel embedding's
gather / reduce-scatter of a shard's ids.
"""

import socket
from typing import List, Optional

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.query_shard import QueryShardPlan


def get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def build_scattered(dp_tokens: List[int], tp_size: int) -> List[int]:
    """Build per-rank scattered token counts from per-DP-rank token counts.

    For each DP rank, divides its tokens across tp_size ranks:
      e.g. dp_tokens=[1, 100], tp_size=2 → [1, 0, 50, 50]
    """
    scattered = []
    for tokens in dp_tokens:
        base, rem = divmod(tokens, tp_size)
        scattered.extend([base + 1] * rem)
        scattered.extend([base] * (tp_size - rem))
    return scattered


def make_batch(mapping, dp_tokens: List[int]) -> ForwardContext:
    """The forward context CommManager sizes its collectives from: every
    rank's token count indexed by global rank (dp stride tp_size)."""
    global_num_tokens = [
        dp_tokens[rank // mapping.attn.tp_size] for rank in range(mapping.world_size)
    ]
    return ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=1,
        num_extends=1,
        input_num_tokens=dp_tokens[mapping.attn.dp_rank],
        forward_mode=ForwardMode.EXTEND,
        output_layout=None,
        global_num_tokens=global_num_tokens,
    )


# ---------------------------------------------------------------------------
# Worker: runs on each GPU
# ---------------------------------------------------------------------------


def worker_fn(
    rank, world_size, port, attn_tp, dense_tp, dp_tokens, hidden_size, error_dict
):
    try:
        _worker_main(rank, world_size, port, attn_tp, dense_tp, dp_tokens, hidden_size)
    except Exception as e:
        import traceback

        error_dict[rank] = traceback.format_exc()


def _worker_main(rank, world_size, port, attn_tp, dense_tp, dp_tokens, hidden_size):
    import sys

    def dbg(msg):
        print(f"[Rank {rank}] {msg}", flush=True, file=sys.stderr)

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as pg_manager,
    )

    mapping = Mapping(
        rank=rank,
        world_size=world_size,
        attn_tp_size=attn_tp,
        dense_tp_size=dense_tp,
    )

    # --- Initialize distributed via ProcessGroupManager ---
    pg_manager.init_distributed(
        mapping=mapping,
        distributed_init_method=f"tcp://localhost:{port}",
        backend="nccl",
    )

    # Pre-create all process groups needed by CommManager.
    # Order is fixed across all ranks; _make_all_groups ensures identical new_group calls.
    for group in [
        mapping.attn.tp_group,
        mapping.dense.tp_group,
        mapping.moe.tp_ep_group,
    ]:
        if len(group) > 1:
            pg_manager.init_process_group(group)

    # --- Set up global state that CommManager depends on ---
    from tokenspeed.runtime.utils.env import global_server_args_dict

    max_tokens = max(sum(dp_tokens), 1)
    global_server_args_dict["chunked_prefill_size"] = max_tokens * 2
    global_server_args_dict["max_prefill_tokens"] = max_tokens * 2
    global_server_args_dict["max_model_len"] = 4096
    global_server_args_dict["enable_allreduce_fusion"] = False
    global_server_args_dict["force_deterministic_rsag"] = True
    global_server_args_dict["mapping"] = mapping

    from tokenspeed.runtime.distributed.comm_manager import CommManager

    cm = CommManager(
        mapping=mapping,
        layer_id=1,
        is_moe=False,
        prev_is_moe=False,
        dense_batch_invariant=False,
        query_sharded=False,
    )

    # --- Token distribution ---
    scattered = build_scattered(dp_tokens, attn_tp)
    batch = make_batch(mapping, dp_tokens)
    # In all-reduce mode, all ranks in a TP group hold the same replicated tokens,
    # so each rank has dp_tokens[dp_rank] tokens (not the scattered per-rank count).
    is_all_reduce = cm.use_all_reduce(cm.is_moe)
    if is_all_reduce:
        my_num_tokens = dp_tokens[mapping.attn.dp_rank]
    else:
        my_num_tokens = scattered[rank]

    dist.barrier()

    # --- Create input tensor on GPU ---
    torch.manual_seed(42 if is_all_reduce else 42 + rank)
    original = torch.randn(
        my_num_tokens, hidden_size, dtype=torch.bfloat16, device=device
    )
    hidden = original.clone()
    residual = original.clone()

    # === Full communication cycle ===

    dbg(f"tokens={my_num_tokens}, pre_attn_comm")
    hidden = cm.pre_attn_comm(hidden, batch)
    dbg(f"pre_attn_comm done, shape={hidden.shape}")

    hidden = hidden / attn_tp

    dbg("post_attn_comm")
    hidden, residual = cm.post_attn_comm(hidden, residual, batch)
    dbg(f"post_attn_comm done, shape={hidden.shape}")

    dbg("pre_dense_comm")
    hidden = cm.pre_dense_comm(hidden, batch)
    dbg(f"pre_dense_comm done, shape={hidden.shape}")

    hidden = hidden / dense_tp

    dbg("post_dense_comm")
    hidden, residual = cm.post_dense_comm(hidden, residual, batch)
    dbg(f"post_dense_comm done, shape={hidden.shape}")

    # === Verify ===
    assert (
        hidden.shape == original.shape
    ), f"Rank {rank}: shape {hidden.shape} != {original.shape}"
    torch.testing.assert_close(hidden, original, atol=0.01, rtol=0.01)

    dist.destroy_process_group()


def _run(world_size, attn_tp, dense_tp, dp_tokens, hidden_size=256):
    if world_size > torch.cuda.device_count():
        pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")

    attn_dp = world_size // attn_tp
    assert (
        len(dp_tokens) == attn_dp
    ), f"dp_tokens length {len(dp_tokens)} != attn_dp {attn_dp}"

    port = get_open_port()
    error_dict = mp.Manager().dict()

    mp.spawn(
        worker_fn,
        args=(world_size, port, attn_tp, dense_tp, dp_tokens, hidden_size, error_dict),
        nprocs=world_size,
        join=True,
    )

    if error_dict:
        raise RuntimeError("\n".join(f"Rank {r}: {e}" for r, e in error_dict.items()))


# ---------------------------------------------------------------------------
# Test configs
# ---------------------------------------------------------------------------

PARALLELISM_CONFIGS = [
    pytest.param(4, 2, 2, id="ws4_atp2_dtp2"),
    pytest.param(8, 2, 2, id="ws8_atp2_dtp2"),
    pytest.param(8, 2, 4, id="ws8_atp2_dtp4"),
    pytest.param(8, 4, 2, id="ws8_atp4_dtp2"),
    pytest.param(8, 4, 4, id="ws8_atp4_dtp4"),
]

# Token distributions keyed by attn_dp count.
# Each entry: (name, dp_tokens)
TOKEN_DISTS = {
    2: [
        pytest.param([100, 100], id="even"),
        pytest.param([1, 131], id="uneven"),
        pytest.param([0, 200], id="extreme_skew"),
        pytest.param([0, 500], id="all_on_single_dp"),
    ],
    4: [
        pytest.param([100, 100, 100, 100], id="even"),
        pytest.param([1, 100, 2, 50], id="uneven"),
        pytest.param([0, 200, 0, 1], id="extreme_skew"),
        pytest.param([0, 0, 0, 500], id="all_on_single_dp"),
    ],
}


def _make_test_params():
    params = []
    for pc in PARALLELISM_CONFIGS:
        ws, atp, dtp = pc.values
        attn_dp = ws // atp
        for td in TOKEN_DISTS[attn_dp]:
            dp_tokens = td.values[0]
            test_id = f"{pc.id}-{td.id}"
            params.append(pytest.param(ws, atp, dtp, dp_tokens, id=test_id))
    return params


class TestCommManager:

    @pytest.mark.parametrize(
        "world_size,attn_tp,dense_tp,dp_tokens", _make_test_params()
    )
    def test_comm_cycle(self, world_size, attn_tp, dense_tp, dp_tokens):
        _run(world_size, attn_tp, dense_tp, dp_tokens)


# ---------------------------------------------------------------------------
# The query-sharded layout, on CPU over gloo
# ---------------------------------------------------------------------------

QCP_WORLD = 4
QCP_HIDDEN = 16
# Three requests over 10 rows: shard rows [3, 3, 2, 2]; the sampled rows
# 3, 4 and 9 land on ranks 1, 1 and 3, so rank 0 and rank 2 gather none.
QCP_LENGTHS = [4, 1, 5]


def _qcp_plan(rank: int) -> QueryShardPlan:
    return QueryShardPlan.from_forward(
        total_tokens=sum(QCP_LENGTHS),
        input_lengths=QCP_LENGTHS,
        size=QCP_WORLD,
        rank=rank,
    )


def _qcp_ctx(plan: QueryShardPlan) -> ForwardContext:
    return ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=len(QCP_LENGTHS),
        num_extends=len(QCP_LENGTHS),
        input_num_tokens=plan.total_rows,
        forward_mode=ForwardMode.EXTEND,
        output_layout=None,
        gather_ids=torch.cumsum(torch.tensor(QCP_LENGTHS), 0) - 1,
        query_shard=plan,
    )


def _qcp_worker(rank, port, error_dict):
    try:
        _qcp_main(rank, port)
    except Exception:
        import traceback

        error_dict[rank] = traceback.format_exc()


def _qcp_main(rank: int, port: int) -> None:
    from tokenspeed.runtime.distributed.comm_manager import CommManager
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as pg_manager,
    )
    from tokenspeed.runtime.utils.env import global_server_args_dict

    mapping = Mapping(
        rank=rank,
        world_size=QCP_WORLD,
        attn_tp_size=QCP_WORLD,
        attn_qcp_size=QCP_WORLD,
        dense_tp_size=QCP_WORLD,
        moe_tp_size=1,
        moe_ep_size=QCP_WORLD,
    )
    pg_manager.init_distributed(
        mapping, distributed_init_method=f"tcp://127.0.0.1:{port}", backend="gloo"
    )
    for group in (
        mapping.attn.tp_group,
        mapping.dense.tp_group,
        mapping.moe.tp_ep_group,
    ):
        if not pg_manager.has_process_group("nccl", group):
            pg_manager.init_process_group(group, backend="gloo")
            pg_manager.register_process_group(
                "nccl", group, pg_manager.get_process_group("gloo", group)
            )
    global_server_args_dict["force_deterministic_rsag"] = True
    global_server_args_dict["enable_allreduce_fusion"] = True
    global_server_args_dict["comm_fusion_max_num_tokens"] = 1 << 20
    global_server_args_dict["mapping"] = mapping

    plan = _qcp_plan(rank)
    ctx = _qcp_ctx(plan)
    moe = CommManager(
        mapping=mapping,
        layer_id=1,
        is_moe=True,
        prev_is_moe=False,
        query_sharded=True,
        dense_batch_invariant=False,
    )
    dense = CommManager(
        mapping=mapping,
        layer_id=2,
        is_moe=False,
        prev_is_moe=True,
        query_sharded=True,
        dense_batch_invariant=False,
    )

    torch.manual_seed(7)
    full = torch.randn(plan.total_rows, QCP_HIDDEN)  # every rank: the same batch
    mine = full[plan.local_slice].clone()

    # The shard is the scattered table; attention needs no collective.
    assert moe.scattered_num_tokens(ctx) == list(plan.row_counts)
    assert moe.get_num_tokens(ctx) == (plan.total_rows, max(plan.row_counts))
    assert moe.pre_attn_comm(mine, ctx) is mine
    assert moe.gather_residual(mine, ctx) is mine
    assert moe.post_attn_comm(mine, mine, ctx) == (mine, mine)
    assert dense.post_final_norm_comm(mine, mine, ctx) == (mine, mine)
    assert not moe.use_all_reduce_norm_fusion() and not moe.should_fuse(4)

    # MoE: gather the shards to every rank of the EP group, apply this
    # rank's "experts" (a per-rank scale), reduce-scatter the partials back.
    gathered = moe.pre_moe_comm(mine, ctx)
    torch.testing.assert_close(gathered, full, rtol=0, atol=0)
    partial = gathered * float(rank + 1)
    out, _ = moe.post_moe_comm(partial, mine, ctx)
    expected = (full * float(sum(range(1, QCP_WORLD + 1))))[plan.local_slice]
    torch.testing.assert_close(out, expected)

    # Dense TP over the same group: the same round trip.
    gathered = dense.pre_dense_comm(mine, ctx)
    torch.testing.assert_close(gathered, full, rtol=0, atol=0)
    out, _ = dense.post_dense_comm(gathered * float(rank + 1), mine, ctx)
    torch.testing.assert_close(out, expected)

    # A narrowed draft step reports the sampled-row count; the table follows.
    ctx.collective_num_tokens = len(QCP_LENGTHS)
    assert moe.scattered_num_tokens(ctx) == list(plan.sampled_rows_per_rank)
    ctx.collective_num_tokens = None

    # The logits processor gathers the sampled rows in request order over its
    # TP group (the query shard group), idle ranks contributing none; the
    # gather is byte-preserving, so fp32 rows of an odd width travel too.
    from tokenspeed.runtime.distributed.comm_manager import gather_sampled_rows

    sampled = gather_sampled_rows(
        mine, plan, ctx.gather_ids, group=mapping.attn.tp_group
    )
    torch.testing.assert_close(sampled, full[ctx.gather_ids], rtol=0, atol=0)
    narrow = full[:, :3].contiguous()  # 12-byte fp32 rows
    sampled = gather_sampled_rows(
        narrow[plan.local_slice], plan, ctx.gather_ids, group=mapping.attn.tp_group
    )
    torch.testing.assert_close(sampled, narrow[ctx.gather_ids], rtol=0, atol=0)
    if rank in (0, 2):
        assert plan.local_sampled_rows == 0

    # A forward without a shard (the drafter's decode steps) keeps the
    # replicated layout on the same managers.
    plain = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=2,
        num_extends=0,
        input_num_tokens=2,
        forward_mode=ForwardMode.DECODE,
        output_layout=None,
    )
    assert moe.scattered_num_tokens(plain) == [1, 1, 0, 0]
    rows = torch.ones(2, QCP_HIDDEN)
    # The replicated MoE combine is an in-place all-reduce of the partials.
    out, _ = moe.post_moe_comm(rows.clone(), rows, plain)
    torch.testing.assert_close(out, rows * QCP_WORLD)

    # The prefill preset: attention TP 4, dense TP 1, EP 4. The attention
    # weights are head-replicated under the query-sharding mapping, so the
    # attention legs are identity on EVERY forward -- the replicated decode
    # step must not reduce-scatter complete rows (x4, scattered) nor gather
    # rows nobody scattered; the final norm gathers nothing either.
    preset = Mapping(
        rank=rank,
        world_size=QCP_WORLD,
        attn_tp_size=QCP_WORLD,
        attn_qcp_size=QCP_WORLD,
        dense_tp_size=1,
        moe_tp_size=1,
        moe_ep_size=QCP_WORLD,
    )
    assert not preset.dense.has_tp and preset.moe.tp_ep_size == QCP_WORLD
    for layer_id, is_moe, prev_is_moe in ((0, True, False), (1, False, True)):
        manager = CommManager(
            mapping=preset,
            layer_id=layer_id,
            is_moe=is_moe,
            prev_is_moe=prev_is_moe,
            dense_batch_invariant=False,
            query_sharded=True,
        )
        assert not manager.use_all_reduce(is_moe=False)  # the RSAG dense layout
        assert not manager.needs_pre_attn_all_gather()
        assert not manager.needs_final_all_gather()
        for forward, hidden in ((ctx, mine), (plain, rows)):
            assert manager.pre_attn_comm(hidden, forward) is hidden
            assert manager.gather_residual(hidden, forward) is hidden
            out, res = manager.post_attn_comm(hidden, hidden, forward)
            assert out is hidden and res is hidden
            out, res = manager.post_final_norm_comm(hidden, hidden, forward)
            assert out is hidden and res is hidden
            # Dense TP 1: the dense legs are identity on both layouts.
            assert manager.pre_dense_comm(hidden, forward) is hidden
            out, _ = manager.post_dense_comm(hidden, hidden, forward)
            assert out is hidden
    # Row-layout conversions (a model with MoE and dense MLPs on different
    # patterns re-lays rows between them): identity on a sharded forward,
    # whose rows are the scattered share already; the replicated forward's
    # slice / gather round-trip stands.
    assert dense.slice_scattered_rows(mine, ctx) is mine
    assert dense.gather_scattered_rows(mine, ctx) is mine
    share = dense.slice_scattered_rows(rows, plain)
    assert share.shape[0] == [1, 1, 0, 0][rank]
    torch.testing.assert_close(dense.gather_scattered_rows(share, plain), rows)

    # The vocab-parallel embedding under a shard: every rank's ids are its
    # own rows, so the lookup gathers the ids to the span for its vocab
    # shard, sums the shards and reduce-scatters the rows back -- the same
    # rows the replicated all-reduce lookup gives for the whole span.
    from tokenspeed.runtime.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

    embedding = VocabParallelEmbedding(
        num_embeddings=256,
        embedding_dim=QCP_HIDDEN,
        params_dtype=torch.bfloat16,
        tp_rank=rank,
        tp_size=QCP_WORLD,
        tp_group=mapping.attn.tp_group,
        padding_size=64,
    )
    shard_rows = embedding.weight.shape[0]
    embedding.weight.data.copy_(
        (
            torch.arange(shard_rows, dtype=torch.float32).unsqueeze(1)
            + 1000 * rank
            + torch.arange(QCP_HIDDEN, dtype=torch.float32) / 16
        ).to(torch.bfloat16)
    )
    torch.manual_seed(11)
    ids = torch.randint(0, 256, (plan.total_rows,))  # the span, every rank
    replicated = embedding(ids)  # the whole span through the all-reduce path
    sharded = embedding(ids[plan.local_slice], query_shard=plan)
    assert sharded.shape == (plan.local_rows, QCP_HIDDEN)
    torch.testing.assert_close(sharded, replicated[plan.local_slice], rtol=0, atol=0)
    with pytest.raises(ValueError, match="reduces its rows"):
        embedding(ids[plan.local_slice], reduce_results=False, query_shard=plan)
    with pytest.raises(ValueError, match="embeds"):
        embedding(ids, query_shard=plan)

    # A dense or MoE group narrower than attention TP has no rows to gather
    # on the replicated forward, so the layout is refused.
    with pytest.raises(ValueError, match="1 or the attention TP width"):
        CommManager(
            mapping=Mapping(
                rank=rank,
                world_size=QCP_WORLD,
                attn_tp_size=QCP_WORLD,
                attn_qcp_size=QCP_WORLD,
                dense_tp_size=2,
            ),
            layer_id=0,
            is_moe=False,
            prev_is_moe=False,
            dense_batch_invariant=False,
            query_sharded=True,
        )

    dist.barrier()
    dist.destroy_process_group()


def test_query_sharded_layout_over_gloo():
    port = get_open_port()
    error_dict = mp.Manager().dict()
    mp.spawn(_qcp_worker, args=(port, error_dict), nprocs=QCP_WORLD, join=True)
    if error_dict:
        raise RuntimeError("\n".join(f"Rank {r}: {e}" for r, e in error_dict.items()))


def test_query_sharded_manager_refuses_a_sharded_forward_it_did_not_declare():
    from tokenspeed.runtime.distributed.comm_manager import CommManager
    from tokenspeed.runtime.distributed.mapping import Mapping

    plain = CommManager(
        mapping=Mapping(rank=0, world_size=QCP_WORLD, attn_tp_size=QCP_WORLD),
        layer_id=0,
        is_moe=False,
        prev_is_moe=False,
        dense_batch_invariant=False,
        query_sharded=False,
    )
    with pytest.raises(RuntimeError, match="did not declare"):
        plain.scattered_num_tokens(_qcp_ctx(_qcp_plan(0)))

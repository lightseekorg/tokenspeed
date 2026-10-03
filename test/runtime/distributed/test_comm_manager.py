"""
Multi-GPU test for CommManager.

Spawns real distributed workers, initializes torch.distributed + NCCL,
and runs CommManager's communication cycle with real GPU tensors:

    pre_attn(AG) → attn → post_attn(RS) → pre_dense(AG) → dense → post_dense(RS)

Verifies that with attn_tp ≠ dense_tp and uneven token counts, each rank
recovers its original hidden states after the full cycle.
"""

import socket
from typing import List, Optional

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


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


class FakeBatch:
    def __init__(self, attn_tl, dense_tl, moe_tl):
        self.attn_tp_group_scattered_num_tokens = attn_tl
        self.dense_tp_group_scattered_num_tokens = dense_tl
        self.moe_tp_ep_group_scattered_num_tokens = moe_tl


def make_batch(mapping, scattered):
    a_s = mapping.attn.tp_size * mapping.attn.dp_rank
    d_s = mapping.dense.tp_size * mapping.dense.dp_rank
    m_s = mapping.moe.tp_ep_size * mapping.moe.dp_rank
    return FakeBatch(
        scattered[a_s : a_s + mapping.attn.tp_size],
        scattered[d_s : d_s + mapping.dense.tp_size],
        scattered[m_s : m_s + mapping.moe.tp_ep_size],
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
        attn_cp_size=1,
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
    )

    # --- Token distribution ---
    scattered = build_scattered(dp_tokens, attn_tp)
    batch = make_batch(mapping, scattered)
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

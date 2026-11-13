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

"""Plain all-reduces past the trtllm workspace's row capacity on a real
4-rank group: two workspace calls up to twice the capacity, NCCL beyond."""

import socket
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HIDDEN = 2048


def get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def ill_conditioned_payload(
    rank_in_group: int, world_size: int, rows: int, device
) -> torch.Tensor:
    """This rank's slice of a payload whose group sum exposes the order: per
    element one rank holds ``+B``, another ``-B`` and the rest values below
    bf16's resolution at ``B``."""
    gen = torch.Generator(device=device).manual_seed(11)
    holder = torch.rand(world_size, rows, HIDDEN, generator=gen, device=device)
    holder = holder.argsort(dim=0)
    big = 2.0 ** torch.randint(4, 12, (rows, HIDDEN), generator=gen, device=device)
    small = torch.randn(world_size, rows, HIDDEN, generator=gen, device=device)
    mine = small[rank_in_group] * big * 2.0**-9
    mine = torch.where(holder[0] == rank_in_group, big, mine)
    mine = torch.where(holder[1] == rank_in_group, -big, mine)
    return mine.to(torch.bfloat16)


def worker_fn(rank, world_size, port, error_dict):
    try:
        worker_main(rank, world_size, port)
    except Exception:
        error_dict[rank] = traceback.format_exc()


def worker_main(rank: int, world_size: int, port: int) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    from tokenspeed_kernel.ops.communication.trtllm import MNNVL_TWOSHOT_MAX_TOKEN

    from tokenspeed.runtime.distributed.comm_backend import get_global_backend
    from tokenspeed.runtime.distributed.comm_ops import all_reduce
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as pg_manager,
    )
    from tokenspeed.runtime.utils.env import global_server_args_dict

    mapping = Mapping(
        rank=rank,
        world_size=world_size,
        attn_tp_size=world_size,
        dense_tp_size=world_size,
    )
    pg_manager.init_distributed(
        mapping=mapping,
        distributed_init_method=f"tcp://localhost:{port}",
        backend="nccl",
    )
    for group in (
        mapping.world_group,
        mapping.attn.tp_group,
        mapping.dense.tp_group,
        mapping.moe.tp_ep_group,
    ):
        if len(group) > 1:
            pg_manager.init_process_group(group)

    capacity = MNNVL_TWOSHOT_MAX_TOKEN
    global_server_args_dict["mapping"] = mapping
    global_server_args_dict["chunked_prefill_size"] = 4 * capacity
    global_server_args_dict["max_prefill_tokens"] = 4 * capacity
    global_server_args_dict["max_model_len"] = 4 * capacity
    global_server_args_dict["max_num_seqs"] = 64
    global_server_args_dict["batch_invariant_collectives"] = False
    global_server_args_dict["force_deterministic_rsag"] = False
    try:
        backend = get_global_backend()
        group = mapping.attn.tp_group
        device_group = pg_manager.get_process_group("nccl", group)
        if not backend.trtllm_ar.configure_group(
            rank=group.index(rank),
            group=group,
            max_token_num=capacity,
            hidden_dim=HIDDEN,
        ):
            return

        def nccl_sum(x: torch.Tensor) -> torch.Tensor:
            x = x.clone()
            dist.all_reduce(x, group=device_group)
            return x

        rows = capacity + capacity // 2
        payload = ill_conditioned_payload(group.index(rank), len(group), rows, device)
        half = rows // 2
        # Each half fits the workspace, so these are workspace sums.
        halves = torch.cat(
            [
                all_reduce(payload[:half].clone(), group),
                all_reduce(payload[half:].clone(), group),
            ]
        )
        # The payload tells the workspace's order from NCCL's.
        assert not torch.equal(halves, nccl_sum(payload))

        # Past the capacity, the 4-rank group still sums on the workspace.
        assert torch.equal(all_reduce(payload.clone(), group), halves)

        # Past twice the capacity it takes NCCL.
        wide = ill_conditioned_payload(
            group.index(rank), len(group), 2 * capacity + 1, device
        )
        assert torch.equal(all_reduce(wide.clone(), group), nccl_sum(wide))
    finally:
        dist.barrier()
        dist.destroy_process_group()


def test_split_all_reduce_world4():
    world_size = 4
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the multi-GPU all-reduce")
    if world_size > torch.cuda.device_count():
        pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")
    port = get_open_port()
    error_dict = mp.Manager().dict()
    mp.spawn(
        worker_fn,
        args=(world_size, port, error_dict),
        nprocs=world_size,
        join=True,
    )
    if error_dict:
        raise RuntimeError("\n".join(f"Rank {r}: {e}" for r, e in error_dict.items()))

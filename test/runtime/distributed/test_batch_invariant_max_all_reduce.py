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

"""Under --batch-invariant-collectives, MAX/MIN all-reduces take NCCL instead
of the SUM-only ordered fold: they are exact in any reduction order."""

import socket
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD_SIZE = 2


def _open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _worker(rank: int, port: int, error_dict) -> None:
    try:
        _worker_main(rank, port)
    except Exception:
        error_dict[rank] = traceback.format_exc()


def _worker_main(rank: int, port: int) -> None:
    torch.cuda.set_device(rank)

    from tokenspeed.runtime.distributed.comm_ops import all_reduce
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as pg_manager,
    )
    from tokenspeed.runtime.utils.env import global_server_args_dict

    mapping = Mapping(
        rank=rank,
        world_size=WORLD_SIZE,
        attn_tp_size=WORLD_SIZE,
        dense_tp_size=WORLD_SIZE,
    )
    pg_manager.init_distributed(
        mapping=mapping,
        distributed_init_method=f"tcp://localhost:{port}",
        backend="nccl",
    )
    group = mapping.attn.tp_group
    pg_manager.init_process_group(group)
    global_server_args_dict["mapping"] = mapping
    global_server_args_dict["batch_invariant_collectives"] = True
    global_server_args_dict["force_deterministic_rsag"] = False
    try:
        values = torch.tensor(
            [[rank, -rank], [-(1 << 62), 7 - rank]], dtype=torch.int64, device="cuda"
        )
        maxima = all_reduce(values.clone(), group, op=dist.ReduceOp.MAX)
        minima = all_reduce(values.clone(), group, op=dist.ReduceOp.MIN)

        assert maxima.tolist() == [[1, 0], [-(1 << 62), 7]]
        assert minima.tolist() == [[0, -1], [-(1 << 62), 6]]
    finally:
        dist.barrier()
        dist.destroy_process_group()


def test_max_and_min_all_reduce_under_batch_invariant_collectives():
    if torch.cuda.device_count() < WORLD_SIZE:
        pytest.skip(f"Need {WORLD_SIZE} GPUs")
    error_dict = mp.Manager().dict()
    mp.spawn(_worker, args=(_open_port(), error_dict), nprocs=WORLD_SIZE, join=True)
    if error_dict:
        raise RuntimeError("\n".join(f"Rank {r}: {e}" for r, e in error_dict.items()))

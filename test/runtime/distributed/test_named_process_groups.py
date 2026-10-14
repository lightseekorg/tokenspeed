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

"""Named process groups, on CPU over gloo: a named group is a second process
group over the same ranks, and a backend on it runs its collectives there."""

import os
import socket
import sys

import pytest
import torch
import torch.multiprocessing as mp

# CPU-only tests scheduled in runtime-1gpu because they import the full runtime.
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=20, suite="runtime-1gpu")

WORLD = 2


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _run(rank: int, port: int) -> None:
    from tokenspeed.runtime.distributed.comm_backend.nccl import NcclBackend
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as pg_manager,
    )

    mapping = Mapping(rank=rank, world_size=WORLD, attn_tp_size=WORLD)
    pg_manager.init_distributed(
        mapping, distributed_init_method=f"tcp://127.0.0.1:{port}", backend="gloo"
    )
    group = mapping.attn.tp_group
    pg_manager.init_process_group(group, backend="gloo")
    pg_manager.init_process_group(group, backend="gloo", name="copy")
    named = pg_manager.get_process_group("gloo", group, name="copy")
    assert named is not pg_manager.get_process_group("gloo", group)
    pg_manager.init_process_group(group, backend="gloo", name="copy")
    assert pg_manager.get_process_group("gloo", group, name="copy") is named

    # gloo stands in for NCCL; the named backend resolves the named group.
    pg_manager.register_process_group("nccl", group, named, name="copy")
    backend = NcclBackend().for_named_groups("copy")
    gathered = torch.empty(2 * WORLD)
    work = backend.all_gather_single_async(
        gathered, torch.full((2,), float(rank)), group
    )
    work.wait()
    assert gathered.tolist() == [0.0, 0.0, 1.0, 1.0]

    with pytest.raises(KeyError, match="named 'missing'"):
        NcclBackend().for_named_groups("missing").all_gather_single(
            gathered, torch.zeros(2), group
        )


def _worker(rank: int, port: int, errors) -> None:
    try:
        _run(rank, port)
    except Exception:  # pragma: no cover - reported to the parent
        import traceback

        errors[rank] = traceback.format_exc()


def test_named_group_is_a_separate_process_group_its_backend_runs_on():
    port = _get_open_port()
    errors = mp.Manager().dict()
    mp.spawn(_worker, args=(port, errors), nprocs=WORLD, join=True)
    if errors:
        raise RuntimeError("\n".join(f"rank {r}: {e}" for r, e in errors.items()))

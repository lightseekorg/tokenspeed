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

"""Device memory a one-rank world's NCCL/RCCL communicators take."""

import os
import sys

import pytest
import torch
import torch.multiprocessing as mp

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")

# A one-rank communicator's staging buffers alone took ~2 GiB each before.
MAX_COMMUNICATOR_BYTES = 1 << 30


def _single_rank_world(port: int, result) -> None:
    import torch.distributed as dist

    from tokenspeed.runtime.distributed.process_group_manager import (
        ProcessGroupManager,
    )

    # NCCL reads the buffer size once per process: start from the default.
    os.environ.pop("NCCL_BUFFSIZE", None)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.cuda.synchronize()
    free_before, _ = torch.cuda.mem_get_info()

    manager = ProcessGroupManager()
    manager._init_world(
        world_size=1,
        rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="nccl",
        timeout=60,
        device_id=device,
    )
    # The world plus one more communicator over the same rank, as startup makes.
    manager.init_process_group((0,), backend="nccl", name="copy")
    group = manager.get_process_group("nccl", (0,), name="copy")
    tensor = torch.arange(4.0, device=device)
    dist.all_reduce(tensor)
    dist.all_reduce(tensor, group=group)
    torch.cuda.synchronize()
    free_after, _ = torch.cuda.mem_get_info()
    result.put((free_before - free_after, tensor.cpu().tolist()))
    dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_single_rank_communicators_skip_peer_staging_buffers():
    from tokenspeed.runtime.utils.network import get_free_port

    ctx = mp.get_context("spawn")
    result = ctx.Queue()
    process = ctx.Process(target=_single_rank_world, args=(get_free_port(), result))
    process.start()
    used, values = result.get(timeout=120)
    process.join(timeout=60)
    assert process.exitcode == 0
    assert values == [0.0, 1.0, 2.0, 3.0]
    assert used < 2 * MAX_COMMUNICATOR_BYTES


def test_one_rank_buffer_default_applies_only_when_unset():
    from tokenspeed.runtime.distributed.process_group_manager import (
        SINGLE_RANK_NCCL_BUFFSIZE,
        default_single_rank_nccl_buffsize,
    )

    environ: dict[str, str] = {}
    assert default_single_rank_nccl_buffsize(environ) == SINGLE_RANK_NCCL_BUFFSIZE
    assert environ["NCCL_BUFFSIZE"] == str(SINGLE_RANK_NCCL_BUFFSIZE)


def test_operator_buffer_size_wins_and_is_not_reported_as_applied():
    from tokenspeed.runtime.distributed.process_group_manager import (
        default_single_rank_nccl_buffsize,
    )

    # An explicit size (e.g. for RL weight sync over the weight-update group)
    # stands, and the engine records that it applied nothing.
    environ = {"NCCL_BUFFSIZE": "4194304"}
    assert default_single_rank_nccl_buffsize(environ) is None
    assert environ == {"NCCL_BUFFSIZE": "4194304"}

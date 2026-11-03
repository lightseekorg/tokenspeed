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


"""Bit-preserving TP4/TP8 broadcast under noncontiguous groups and graph replay."""

import socket
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform().is_cdna4, reason="TP4/TP8 broadcast requires gfx950"
)


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _skip_if_unsupported(world_size: int, reason_prefix: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip(f"CUDA/ROCm is required for {reason_prefix}")
    if world_size > torch.cuda.device_count():
        pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")
    if not current_platform().is_amd:
        pytest.skip(f"{reason_prefix} only targets AMD ROCm")
    try:
        import iris  # noqa: F401
    except ImportError:
        pytest.skip("iris is not installed")


def _spawn_and_collect(worker_fn, args, world_size: int) -> None:
    error_dict = mp.Manager().dict()
    mp.spawn(
        worker_fn,
        args=args + (error_dict,),
        nprocs=world_size,
        join=True,
    )

    if error_dict:
        raise RuntimeError("\n".join(f"Rank {r}: {e}" for r, e in error_dict.items()))


def _communication_worker(rank, world_size, port, check, backend, error_dict):
    try:
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        dist.init_process_group(
            backend,
            init_method=f"tcp://localhost:{port}",
            rank=rank,
            world_size=world_size,
        )
        check(rank, world_size, device)
    except Exception:
        error_dict[rank] = traceback.format_exc()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _broadcast_subgroup_worker(rank, world_size, device):
    _broadcast_worker(rank, world_size, device, True)


def _broadcast_world_worker(rank, world_size, device):
    _broadcast_worker(rank, world_size, device, False)


def _broadcast_worker(rank, world_size, device, subgroup):
    ranks = (
        (tuple(range(0, world_size, 2)), tuple(range(1, world_size, 2)))
        if subgroup
        else (tuple(range(world_size)),)
    )
    group_index = rank % 2 if subgroup else 0
    groups = tuple(dist.new_group(r) for r in ranks)
    group_rank = ranks[group_index].index(rank)
    from tokenspeed_kernel.ops.communication._iris import (
        IrisGroup,
        get_or_create_context,
    )
    from tokenspeed_kernel.ops.communication._iris.broadcast import IrisBroadcast
    from utils import assert_no_triton_compile

    ctx = get_or_create_context(heap_size=256 << 20)
    peers = IrisGroup(ctx, groups[group_index], group_rank, device)
    state = IrisBroadcast(peers, 8192)
    peers.finish_prepare()
    counts = (1, 63, 64, 65, 8191, 8192, 8193, 16385, 1)
    inputs = [torch.zeros(n, dtype=torch.int32, device=device) for n in counts]
    outputs = [torch.empty_like(t) for t in inputs]
    state.run(inputs[0])
    state.run(inputs[3])

    def broadcast_sequence():
        for source, output in zip(inputs, outputs):
            output.copy_(source)
            state.run(output)

    with assert_no_triton_compile():
        broadcast_sequence()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        broadcast_sequence()
    extrema = torch.tensor(
        [-2147483648, 2147483647, -1, 0, 65535, 65536],
        dtype=torch.int32,
        device=device,
    )
    for step in range(4):
        expected = []
        for n, value in zip(counts, inputs):
            torch.manual_seed(8000 + n + 17 * step + group_index)
            reference = torch.randint(
                -2147483648,
                2147483647,
                (n,),
                dtype=torch.int32,
                device=device,
            )
            if n >= extrema.numel():
                reference[: extrema.numel()].copy_(extrema.roll(step % 6))
            expected.append(reference)
            value.copy_(reference if group_rank == 0 else reference ^ group_rank)
        if group_rank == step % peers.world_size:
            torch.cuda._sleep(10000)
        graph.replay()
        for actual, reference in zip(outputs, expected):
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    torch.cuda.synchronize()


@pytest.mark.parametrize("check", [_broadcast_subgroup_worker, _broadcast_world_worker])
def test_iris_broadcast_mailbox_replay_and_tails(check):
    _skip_if_unsupported(8, "Iris TP4/TP8 broadcast subgroup tests")
    _spawn_and_collect(
        _communication_worker,
        (8, _get_open_port(), check, "gloo"),
        8,
    )

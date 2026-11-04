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

"""The Iris hidden-shard all-gather on gfx950 and gfx1250 GPUs."""

import socket
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not (current_platform().is_cdna4 or current_platform().is_cdna5),
    reason="The Iris column all-gather targets gfx950 and gfx1250",
)


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _skip_if_unsupported(world_size: int) -> None:
    if world_size > torch.cuda.device_count():
        pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")
    try:
        import tokenspeed_kernel.ops.communication._iris  # noqa: F401
    except ImportError:
        pytest.skip("iris is not installed")


def _spawn_and_collect(worker_fn, args, world_size: int) -> None:
    error_dict = mp.Manager().dict()
    mp.spawn(worker_fn, args=args + (error_dict,), nprocs=world_size, join=True)
    if error_dict:
        raise RuntimeError("\n".join(f"Rank {r}: {e}" for r, e in error_dict.items()))


def _column_shards(rows: int, world_size: int, local_hidden: int, device):
    """Seeded random gathered rows, the same on every rank."""
    generator = torch.Generator().manual_seed(rows)
    # Integers within 64, times the scales below, stay exact in bf16.
    values = torch.randint(
        -64, 65, (rows, world_size * local_hidden), generator=generator
    )
    return values.to(device=device, dtype=torch.bfloat16)


def _check_column_all_gather(rank: int, world_size: int, device) -> None:
    from tokenspeed_kernel.ops.communication._iris.all_gather import (
        IrisAllGather,
        iris_pull_allgather_columns,
    )
    from tokenspeed_kernel.ops.communication.iris import (
        create_iris_state,
        iris_acquire_outputs,
    )
    from utils import assert_no_triton_compile

    hidden = 7168
    local_hidden = hidden // world_size
    columns = slice(rank * local_hidden, (rank + 1) * local_hidden)
    state = create_iris_state(
        group=dist.group.WORLD,
        rank_in_group=rank,
        staged_max_numel=0,
        producer_direct_max_numel=16 * hidden,
        dtype=torch.bfloat16,
        heap_size=None,
        device=device,
    )
    gather = IrisAllGather(state.producer)
    state.peers.finish_prepare()
    overwrite = torch.full(
        (1, local_hidden), -(rank + 1), dtype=torch.bfloat16, device=device
    )
    # A row the producer published in place, ordinary rows, and 33 rows that
    # exceed the 16-row capacity and gather in three chunks.
    for rows in (1, 4, 16, 33):
        if rows == 1:
            local = iris_acquire_outputs(state, ((1, hidden),))[0][:, columns]
            # The kernel skips its input copy only for this rank's published slot.
            assert local.data_ptr() == (
                state.producer.data.data_ptr() + rank * local_hidden * 2
            )
        else:
            local = torch.empty(
                (rows, local_hidden), dtype=torch.bfloat16, device=device
            )
        expected = _column_shards(rows, world_size, local_hidden, device)
        assert gather.can_run(local)

        def run():
            gathered = gather.run(local)
            if rows <= 16:
                # Results within capacity borrow storage the next gather reuses.
                gathered = gathered.clone()
            gather.run(overwrite)
            return gathered

        local.copy_(expected[:, columns])
        torch.testing.assert_close(run(), expected, atol=0, rtol=0)
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gathered = run()
        dist.barrier()
        for scale in (2, -1, -4):
            local.copy_(expected[:, columns] * scale)
            torch.cuda._sleep((rank + 1) * 20000)
            graph.replay()
            torch.testing.assert_close(gathered, expected * scale, atol=0, rtol=0)
        dist.barrier()

    with assert_no_triton_compile(iris_pull_allgather_columns):
        for rows in (3, 7):
            expected = _column_shards(rows, world_size, local_hidden, device)
            gathered = gather.run(expected[:, columns].contiguous())
            torch.testing.assert_close(gathered, expected, atol=0, rtol=0)


def _column_all_gather_worker_fn(rank, world_size, port, error_dict):
    try:
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://localhost:{port}",
            rank=rank,
            world_size=world_size,
        )
        _check_column_all_gather(rank, world_size, device)
    except Exception:
        error_dict[rank] = traceback.format_exc()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [4, 8])
def test_iris_column_all_gather_concatenates_hidden_shards(world_size):
    """TP4/TP8 hidden-shard gathers: published, chunked, and graph-replayed."""
    _skip_if_unsupported(world_size)
    _spawn_and_collect(
        _column_all_gather_worker_fn, (world_size, _get_open_port()), world_size
    )

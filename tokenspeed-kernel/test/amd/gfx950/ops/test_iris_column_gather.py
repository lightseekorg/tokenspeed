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

"""The Iris column all-gather concatenates every rank's feature shard."""

import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.platform import current_platform
from utils import is_amd

pytestmark = pytest.mark.skipif(not is_amd(), reason="Iris requires AMD ROCm")


@pytest.fixture(autouse=True)
def _require_iris():
    pytest.importorskip("tokenspeed_kernel.ops.communication._iris")


def test_large_tp8_shards_take_two_subgroups_per_program():
    from tokenspeed_kernel.ops.communication._iris.all_gather import (
        column_gather_num_warps,
    )

    assert column_gather_num_warps(256 * 896 * 2, 8) == 1
    assert column_gather_num_warps(848 * 288 * 2, 8) == 1
    assert column_gather_num_warps(848 * 512 * 2, 8) == 2
    assert column_gather_num_warps(8192 * 896 * 2, 8) == 2
    assert column_gather_num_warps(8192 * 896 * 2, 4) == 1


@pytest.mark.parametrize("world_size", [4, 8])
def test_warmup_prepares_every_subgroup_count_a_payload_picks(world_size):
    from tokenspeed_kernel.ops.communication._iris.all_gather import (
        _column_gather_variants,
        column_gather_num_warps,
    )

    launched = {"BLOCK_WORDS": 256, "NUM_WARPS": 1, "WORDS_PER_LANE": 4}
    sets = _column_gather_variants(launched | {"WORLD_SIZE": world_size})
    warmed = {
        (warps, block)
        for choices in sets
        for warps in choices["NUM_WARPS"]
        for block in choices["BLOCK_WORDS"]
    }
    picked = {
        column_gather_num_warps(rows * 896 * 2, world_size) for rows in (1, 848, 8192)
    }
    assert {(warps, 256 * warps) for warps in picked} <= warmed


def _column_gather_worker(rank: int, port: int) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=180),
    )
    group = dist.new_group(backend="nccl", timeout=timedelta(seconds=180))
    from tokenspeed_kernel.ops.communication._iris.all_gather import IrisAllGather
    from tokenspeed_kernel.ops.communication.iris import create_iris_state

    state = create_iris_state(
        group=group,
        rank_in_group=rank,
        staged_max_numel=0,
        producer_direct_max_numel=8192 * 7168,
        dtype=torch.bfloat16,
        heap_size=None,
        device=device,
    )
    gather = IrisAllGather(state.producer)
    state.peers.finish_prepare()
    gen = torch.Generator(device=device)
    # One and two subgroups per program, including a partial last tile.
    for rows, width in (
        (4, 896),
        (256, 896),
        (848, 896),
        (8192, 896),
        (848, 288),
        (2047, 512),
    ):
        gen.manual_seed(rows * width + rank)
        shard = torch.randn((rows, width), device=device, generator=gen).to(
            torch.bfloat16
        )
        parts = [torch.empty_like(shard) for _ in range(8)]
        dist.all_gather(parts, shard, group=group)
        expected = torch.cat(parts, dim=1)
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 5:
            torch.cuda._sleep(2_000_000)
        result = gather.run(shard)
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
        torch.cuda.synchronize()
        dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(
    not current_platform().is_cdna4 or torch.cuda.device_count() < 8,
    reason="requires eight CDNA4 GPUs",
)
def test_column_gather_matches_the_concatenated_shards():
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_column_gather_worker, args=(port,), nprocs=8, join=True)

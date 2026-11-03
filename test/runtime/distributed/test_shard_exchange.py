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

"""Prepared even shard exchanges (``prepare_shard_exchange``) on real GPUs.

``all_to_all_transpose`` / ``all_to_all_head_scatter`` with an even row split
and a prepared width run the backend's dedicated exchange; every other call
keeps the generic all-to-all. Both must move every bit unchanged: the
prepared path is compared with the NCCL backend on arbitrary bf16 bit
patterns (NaN payloads, infinities, signed zeros), eager and under CUDA
graph replay with new inputs.
"""

import sys
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(
    est_time=120,
    suite="runtime-2gpu",
    disabled_on_runners=["amd-*"],
    disabled_on_runners_reason="the prepared shard exchange is an NVIDIA NVLink path",
)

_HEADS_LOCAL = 3
_DIM = 64
_MAX_ROWS = 16


def _bits(rows: int, width: int, seed: int, device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randint(-(2**15), 2**15, (rows, width), generator=generator)
    return raw.to(torch.int16).view(torch.bfloat16).to(device)


def _same_bits(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def _worker(rank: int, degree: int, rendezvous: str) -> None:
    torch.cuda.set_device(rank)
    from tokenspeed.runtime.distributed.comm_backend.auto import AutoBackend
    from tokenspeed.runtime.distributed.comm_backend.registry import set_global_backend
    from tokenspeed.runtime.distributed.comm_ops import (
        all_to_all_head_scatter,
        all_to_all_transpose,
        prepare_shard_exchange,
    )
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager,
    )

    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=degree,
        device_id=device,
        timeout=timedelta(seconds=90),
    )
    try:
        group = tuple(range(degree))
        process_group_manager.register_process_group("nccl", group, dist.group.WORLD)
        backend = AutoBackend()
        set_global_backend(backend)
        width = _HEADS_LOCAL * _DIM
        channels = degree * width
        assert prepare_shard_exchange(group, channels, _MAX_ROWS, torch.bfloat16)
        # A second layer of the same width shares the prepared exchange.
        assert prepare_shard_exchange(group, channels, _MAX_ROWS, torch.bfloat16)
        nccl = backend.nccl

        def round_trip(rows: list[int], seed: int) -> None:
            shard = _bits(sum(rows), width, seed + rank, device)
            own = all_to_all_transpose(shard, group, input_split_sizes=rows)
            _same_bits(own, all_to_all_transpose(shard, group, rows, backend=nccl))
            heads = own.view(-1, degree * _HEADS_LOCAL, _DIM)
            back = all_to_all_head_scatter(heads, group, output_split_sizes=rows)
            _same_bits(back, all_to_all_head_scatter(heads, group, rows, backend=nccl))
            _same_bits(back.view(-1, width), shard)

        # Even splits take the prepared exchange (the backend serves them) ...
        even = [5] * degree
        assert (
            backend.try_shard_exchange(
                _bits(5, channels, 1, device), group, inverse=False
            )
            is not None
        )
        round_trip(even, seed=10)
        round_trip([1] * degree, seed=20)
        round_trip([_MAX_ROWS] * degree, seed=30)
        # ... uneven splits, oversized splits and unprepared widths do not.
        round_trip([3] + [7] * (degree - 1), seed=40)
        round_trip([_MAX_ROWS + 1] * degree, seed=50)
        assert (
            backend.try_shard_exchange(
                _bits(5, channels + 2 * degree, 2, device), group, inverse=False
            )
            is None
        )

        # CUDA graph replay of the prepared exchange with new inputs.
        shard = torch.empty(sum(even), width, dtype=torch.bfloat16, device=device)
        shard.copy_(_bits(sum(even), width, 60 + rank, device))
        torch.cuda.synchronize()
        warm = all_to_all_transpose(shard, group, input_split_sizes=even)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = all_to_all_transpose(shard, group, input_split_sizes=even)
        for seed in (70, 80):
            shard.copy_(_bits(sum(even), width, seed + rank, device))
            torch.cuda.synchronize()
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize()
            _same_bits(captured, all_to_all_transpose(shard, group, even, backend=nccl))
        del warm
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("degree", [2, 4])
def test_prepared_exchange_moves_every_bit(degree, tmp_path):
    if torch.cuda.device_count() < degree:
        pytest.skip(f"needs {degree} GPUs")
    rendezvous = f"file://{tmp_path / 'rendezvous'}"
    mp.spawn(_worker, args=(degree, rendezvous), nprocs=degree, join=True)

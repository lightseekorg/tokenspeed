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

"""Gather and scatter projections of owner rows on real process groups."""

import gc
import socket
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=150, suite="runtime-2gpu")

_HIDDEN, _SHARD, _MAX_ROWS = 256, 64, 1024


def _integers(generator, shape, low, high, device):
    # Small integers multiply and sum exactly in FP32, so every GEMM order agrees.
    values = torch.randint(low, high + 1, shape, generator=generator)
    return values.to(device=device, dtype=torch.bfloat16)


def _gathered_reference(inputs, counts, rows, weight):
    """Every owner's zero-padded rows through ``weight`` in FP32, rounded once."""
    padded = inputs.new_zeros((rows, inputs.shape[1]))
    padded[: inputs.shape[0]] = inputs
    parts = [torch.empty_like(padded) for _ in counts]
    dist.all_gather(parts, padded)
    return (torch.cat(parts).float() @ weight.float().T).to(torch.bfloat16)


def _worker(rank: int, port: int, world_size: int, backend_name: str) -> None:
    from tokenspeed.runtime.distributed.comm_backend.auto import AutoBackend
    from tokenspeed.runtime.distributed.comm_backend.nccl import NcclBackend
    from tokenspeed.runtime.distributed.comm_backend.projection import ProjectionSpec
    from tokenspeed.runtime.distributed.comm_ops import (
        prepare_projection_collectives,
        projection_gather_mm,
        projection_mm_reduce_scatter,
    )
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager,
    )

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        group = tuple(range(world_size))
        process_group_manager.init_process_group(group)
        backend = AutoBackend() if backend_name == "auto" else NcclBackend()
        device = torch.device("cuda", rank)
        gather = prepare_projection_collectives(
            ProjectionSpec(
                group=group,
                kind="gather",
                input_size=_HIDDEN,
                output_size=_SHARD * world_size,
                max_tokens=_MAX_ROWS,
                dtype=torch.bfloat16,
                device=device,
            ),
            backend,
        )
        scatter = prepare_projection_collectives(
            ProjectionSpec(
                group=group,
                kind="scatter",
                input_size=64 * world_size,
                output_size=_HIDDEN,
                max_tokens=_MAX_ROWS,
                dtype=torch.bfloat16,
                device=device,
            ),
            backend,
        )

        def gathered(rows, seed):
            # Owners hold different row counts, as data-parallel owners do.
            counts = [max(rows - owner, 1) for owner in range(world_size)]
            generator = torch.Generator().manual_seed(seed)
            weight = _integers(generator, (_SHARD, _HIDDEN), -2, 2, device)
            inputs = _integers(generator, (counts[rank], _HIDDEN), -2, 2, device)
            out = torch.empty(
                world_size * rows, _SHARD, dtype=torch.bfloat16, device=device
            )
            projection_gather_mm(inputs, rows, weight, out, gather, backend)
            return out, _gathered_reference(inputs, counts, rows, weight)

        # One row, a Lamport-sized batch and a copy-engine-sized one.
        for seed, rows in enumerate((1, 100, 700)):
            out, expected = gathered(rows, seed)
            torch.testing.assert_close(out, expected, atol=0, rtol=0)

        for seed, rows in enumerate((1, 100, 700)):
            generator = torch.Generator().manual_seed(100 + seed + rank)
            # Every partial and sum stays an exact BF16 integer, so any order agrees.
            # The last owner's padding rows are left for the projection to fill.
            inputs = _integers(generator, (world_size * rows - 1, 64), -1, 1, device)
            weight = _integers(generator, (_HIDDEN, 64), -1, 1, device)
            reduced = projection_mm_reduce_scatter(
                inputs, rows, weight, scatter, backend
            )
            inputs = torch.nn.functional.pad(inputs, (0, 0, 0, 1))
            parts = [torch.empty_like(inputs) for _ in range(world_size)]
            dist.all_gather(parts, inputs)
            weights = [torch.empty_like(weight) for _ in range(world_size)]
            dist.all_gather(weights, weight)
            expected = sum(
                part.float() @ shard.float().T for part, shard in zip(parts, weights)
            ).to(torch.bfloat16)
            torch.testing.assert_close(
                reduced, expected[rank * rows : (rank + 1) * rows], atol=0, rtol=0
            )

        # Three captured gathers replay on new rows: no state carries between calls.
        rows = 700
        counts = [max(rows - owner, 1) for owner in range(world_size)]
        generator = torch.Generator().manual_seed(7)
        weight = _integers(generator, (_SHARD, _HIDDEN), -2, 2, device)
        inputs = [
            _integers(generator, (counts[rank], _HIDDEN), -2, 2, device)
            for _ in range(3)
        ]
        outs = [
            torch.empty(world_size * rows, _SHARD, dtype=torch.bfloat16, device=device)
            for _ in inputs
        ]
        for value, out in zip(inputs, outs):
            projection_gather_mm(value, rows, weight, out, gather, backend)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        # Destructors of discarded CUDA objects can invalidate an active capture.
        gc.collect()
        gc.disable()
        try:
            with torch.cuda.graph(graph):
                for value, out in zip(inputs, outs):
                    projection_gather_mm(value, rows, weight, out, gather, backend)
        finally:
            gc.enable()
        for replay in range(2):
            for index, value in enumerate(inputs):
                value.copy_(
                    _integers(
                        torch.Generator().manual_seed(20 + 3 * replay + index + rank),
                        value.shape,
                        -2,
                        2,
                        device,
                    )
                )
            graph.replay()
            for value, out in zip(inputs, outs):
                torch.testing.assert_close(
                    out,
                    _gathered_reference(value, counts, rows, weight),
                    atol=0,
                    rtol=0,
                )
        # A live graph holding NCCL work blocks the process group's teardown.
        graph.reset()
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _spawn(world_size: int, backend_name: str) -> None:
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} GPUs")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(
        _worker, args=(port, world_size, backend_name), nprocs=world_size, join=True
    )


@pytest.mark.parametrize("backend_name", ["auto", "nccl"])
@pytest.mark.parametrize("world_size", [2, 4])
def test_owner_row_projections_match_gathered_and_reduced_rows(
    world_size, backend_name
) -> None:
    _spawn(world_size, backend_name)

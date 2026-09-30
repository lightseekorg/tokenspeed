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

"""Small checkpoint-free DP linear tests using real collectives and GEMMs."""

import os
import sys
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, suite="runtime-2gpu")


def _worker(rank, size, rendezvous, a2a_backend, rs_backend):
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as pg_manager,
    )
    from tokenspeed.runtime.layers.dp_linear_communication import (
        DPColumnParallelCommunication,
        DPRowParallelCommunication,
        initialize_projection_group,
        projection_mapping,
    )
    from tokenspeed.runtime.layers.linear import (
        DPColumnParallelLinear,
        DPRowParallelLinear,
    )

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=size,
        timeout=timedelta(seconds=180),
        device_id=device,
    )
    parallel = projection_mapping(rank, size, size)
    initialize_projection_group(parallel)
    width, stored, logical, capacity = 512, 256, 239, 129
    with torch.device(device):
        column = DPColumnParallelLinear(
            width,
            logical,
            padded_output_size=stored,
            parallel=parallel,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix="qkv_proj",
        )
        row = DPRowParallelLinear(
            width,
            stored,
            parallel=parallel,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix="o_proj",
        )
    # Sparse integer weights make BF16 results exact, while every rank's
    # channel shard contributes. Distinct rows expose owner/channel permutations.
    weight = torch.zeros(stored, width, device=device, dtype=torch.bfloat16)
    indices = torch.arange(stored, device=device)
    for shard in range(size):
        weight[indices, (indices * 7) % (width // size) + shard * (width // size)] = 1
    column_weight = weight.clone()
    column_weight[logical:] = 0
    column.weight.weight_loader(column.weight, column_weight)
    row.weight.weight_loader(row.weight, weight)
    column.quant_method.process_weights_after_loading(column)
    row.quant_method.process_weights_after_loading(row)
    column.communication = DPColumnParallelCommunication(
        parallel,
        width,
        stored,
        capacity,
        torch.bfloat16,
        device,
        "nccl",
        a2a_backend,
    )
    row.communication = DPRowParallelCommunication(
        capacity, width, torch.bfloat16, device
    )
    row.communication.initialize_a2a(parallel, [width], a2a_backend)
    row.communication.initialize_reduce_scatter(parallel, [stored], rs_backend)
    patterns = (
        [1] * size,
        [3] * size,
        [0] + [3] * (size - 1),
        [5] + [0] * (size - 1),
        [0] * size,
        [128] * size,
        [129] * size,
    )
    for counts in patterns:
        values = torch.arange(counts[rank] * width, device=device).view(
            counts[rank], width
        )
        x = ((values % 11 - 5 + rank) / 8).to(torch.bfloat16)
        for linear, full_weight, output_width in (
            (column, column_weight, logical),
            (row, weight, stored),
        ):
            expected = (x.float() @ full_weight.float().T)[:, :output_width].to(x.dtype)
            actual, bias = linear(x, counts)
            assert bias is None
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            saved = actual.clone()
            linear(-x, counts)
            torch.testing.assert_close(actual, saved, rtol=0, atol=0)
            if not max(counts):
                continue  # All-empty eager path has no kernels to capture.
            for _ in range(2):
                linear(x, counts)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured, _ = linear(x, counts)
            original = x.clone()
            for factor in (0.5, -1, 0):
                x.copy_(original * factor)
                graph.replay()
                torch.testing.assert_close(captured, expected * factor, rtol=0, atol=0)
            x.copy_(original)
            torch.cuda.synchronize()
            del graph, captured
    column.communication.close()
    row.communication.close()
    dist.destroy_process_group()


@pytest.mark.parametrize("size", [2, 4], ids=["tp2", "tp4"])
@pytest.mark.parametrize(
    "a2a_backend,rs_backend",
    [
        ("nccl", "nccl"),
        ("tokenspeed_a2a_lamport", "triton_peer"),
        ("tokenspeed_a2a_lamport", "trtllm_lamport"),
    ],
)
def test_dp_linears(size, a2a_backend, rs_backend, tmp_path):
    if not torch.cuda.is_available() or torch.cuda.device_count() < size:
        pytest.skip(f"requires {size} GPUs")
    if torch.version.hip is not None and a2a_backend != "nccl":
        pytest.skip("Lamport fast paths require NVIDIA CUDA")
    mp.spawn(
        _worker,
        args=(size, f"file://{tmp_path / 'rendezvous'}", a2a_backend, rs_backend),
        nprocs=size,
        join=True,
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

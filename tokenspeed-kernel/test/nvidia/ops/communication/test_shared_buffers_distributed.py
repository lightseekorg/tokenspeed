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

"""Native shared collectives on local GPUs or a multi-host NVLink group.

Run with torchrun over two or four ranks; ranks may span NVLink-connected hosts.
"""

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.communication.fabric import gather_fabric_map
from tokenspeed_kernel.ops.communication.trtllm import (
    TrtllmAllGatherState,
    TrtllmReduceScatterState,
    trtllm_allgather,
    trtllm_reduce_scatter,
)

pytestmark = pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) not in (2, 4),
    reason="launch with torchrun over two or four GPUs",
)


def test_shared_collectives_survive_graph_replay_and_reallocation():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=180), device_id=device)
    try:
        gather_fabric_map()
        group = dist.group.WORLD
        rank, size = group.rank(), group.size()
        for generation in range(2):
            gather = TrtllmAllGatherState(group, 128, 7168, device, True)
            reduction = TrtllmReduceScatterState(group, 128, 7168, device)
            try:
                for rows in (1, 31, 128, 64, 1):
                    inputs = torch.empty(
                        (rows, 7168), device=device, dtype=torch.bfloat16
                    )
                    partial = torch.empty(
                        (size * rows, 7168), device=device, dtype=torch.bfloat16
                    )
                    inputs.fill_(rank + 1)
                    partial.fill_(rank + 1)
                    trtllm_allgather(gather, inputs)
                    trtllm_reduce_scatter(reduction, partial, rows)
                    torch.cuda.synchronize()
                    dist.barrier()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        gathered = trtllm_allgather(gather, inputs)
                        reduced = trtllm_reduce_scatter(reduction, partial, rows)
                    for iteration in range(12):
                        value = (rank + 1 + generation + iteration) / 16
                        inputs.fill_(value)
                        partial.fill_(value)
                        graph.replay()
                        expected_gather = torch.cat(
                            [
                                torch.full_like(
                                    inputs, (peer + 1 + generation + iteration) / 16
                                )
                                for peer in range(size)
                            ]
                        )
                        expected_sum = (
                            size * (size + 1) / 2 + size * (generation + iteration)
                        ) / 16
                        torch.testing.assert_close(
                            gathered, expected_gather, atol=0, rtol=0
                        )
                        torch.testing.assert_close(
                            reduced,
                            torch.full_like(inputs, expected_sum),
                            atol=0,
                            rtol=0,
                        )
                    del graph, gathered, reduced
            finally:
                gather.close()
                reduction.close()
    finally:
        dist.destroy_process_group()

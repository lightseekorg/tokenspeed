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

"""Exact FP8/scales and ring reuse checks against gather then native quantize."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.ops.communication.trtllm import (
    TrtllmAllGatherQuantState,
    trtllm_allgather_fp8_quantize,
)
from tokenspeed_kernel.ops.gemm.fp8_utils import (
    flashinfer_fp8_blockscale_quantize_prepacked,
)


def _worker(rank, rendezvous):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=600),
        device_id=device,
    )
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    torch.manual_seed(120 + rank)
    groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
    for size, group in ((4, dist.group.WORLD), (2, groups[rank // 2])):
        for hidden in (128, 1536, 7168):
            state = TrtllmAllGatherQuantState(group, 128, hidden, device, sms)
            for rows in (1, 3, 32, 64, 128, 3, 1):
                inputs = torch.randn(rows, hidden, dtype=torch.bfloat16, device=device)
                # Signed zeros are canonicalized by both Lamport paths. Cover
                # a zero group, rank-distinct data, padded/empty owners and tails.
                inputs[0, :128] = -0.0 if rank % 2 else 0.0
                if rows > 1:
                    inputs[1, :128] *= 1e-10
                    inputs[-1, -128:] *= 1e10
                valid = rows * (rank % size) // (size - 1)
                if rows == 3:
                    inputs[valid:] = 0
                if rows == 64:
                    inputs.zero_()

                def reference():
                    # Exercise interchange between the fused and ordinary
                    # consumers of the same rotating Lamport workspace.
                    gathered = state.gather(inputs)
                    return flashinfer_fp8_blockscale_quantize_prepacked(gathered, 128)

                for _ in range(4):
                    expected, expected_scales = reference()
                    actual, actual_scales = trtllm_allgather_fp8_quantize(state, inputs)
                    torch.testing.assert_close(
                        actual.view(torch.uint8),
                        expected.view(torch.uint8),
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        actual_scales, expected_scales, rtol=0, atol=0
                    )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual, actual_scales = trtllm_allgather_fp8_quantize(state, inputs)
                for iteration in range(5):
                    inputs.mul_(0.75).add_((rank + 1) * (iteration + 1) * 0.03125)
                    expected, expected_scales = reference()
                    graph.replay()
                    torch.testing.assert_close(
                        actual.view(torch.uint8),
                        expected.view(torch.uint8),
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        actual_scales, expected_scales, rtol=0, atol=0
                    )
                del graph, actual, actual_scales
            state.close()
    dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 4, reason="Four NVLink CUDA GPUs required"
)
def test_trtllm_allgather_quant(tmp_path):
    mp.spawn(_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=4, join=True)

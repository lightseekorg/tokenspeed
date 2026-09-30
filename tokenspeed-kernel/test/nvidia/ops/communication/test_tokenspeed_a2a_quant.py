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

"""Exact FP8/scales and mixed BF16/quantized Lamport A2A graph reuse."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.ops.communication.cuda import (
    TokenSpeedA2ALamportState,
    tokenspeed_a2a_lamport,
    tokenspeed_a2a_lamport_fp8_quantize,
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
    torch.manual_seed(731 + rank)
    for channels in (512, 12288, 16384):
        state = TokenSpeedA2ALamportState(
            dist.group.WORLD,
            512,
            channels,
            device,
            min(128, torch.cuda.get_device_properties(device).multi_processor_count),
        )
        state.prepare_chunk_exchange(8 * 2**20 + 1)
        state.prepare_fp8_quantization()
        inputs = []
        for rows in (1, 3, 32, 64, 128, 129, 256, 512, 3):
            x = torch.randn(rows, channels, dtype=torch.bfloat16, device=device)
            x[0, :128] = -0.0 if rank % 2 else 0.0
            x[0, 128:256] *= 1e-10
            x[0, 256:384] *= 1e10
            if rows == 64:
                x.zero_()
            if rows == 3:
                x[rank % 3 :] = 0  # Padded/empty logical owners still communicate.
            inputs.append(x)

        def reference(x):
            exchanged = tokenspeed_a2a_lamport(state, x, inverse=False, out=None)
            return flashinfer_fp8_blockscale_quantize_prepacked(exchanged, 128)

        def check(actual, expected):
            torch.testing.assert_close(
                actual[0].view(torch.uint8),
                expected[0].view(torch.uint8),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)

        for x in inputs:
            for _ in range(3):
                expected = reference(x)
                check(tokenspeed_a2a_lamport_fp8_quantize(state, x), expected)
        torch.cuda.synchronize()
        dist.barrier()
        # Quantized and ordinary consumers must share unsigned generations,
        # including transitions across the signed-int32 boundary.
        state.control[0] = 2**31 - 2
        state.chunk_control[0] = 2**31 - 2
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            retained = []
            for x in inputs:
                if rank == 1:
                    torch.cuda._sleep(10000)
                values, scales = tokenspeed_a2a_lamport_fp8_quantize(state, x)
                retained.append((values.clone(), scales.clone()))
                # Interchange quantized and ordinary consumers, including the
                # inverse layout, without separate generations or stale payloads.
                inverse_input = x.view(4 * x.shape[0], channels // 4)
                tokenspeed_a2a_lamport(state, inverse_input, inverse=True, out=None)
        for iteration in range(5):
            for x in inputs:
                x.mul_(0.75).add_((rank + 1) * (iteration + 1) * 0.03125)
            graph.replay()
            for x, actual in zip(inputs, retained):
                check(actual, reference(x))
        torch.cuda.synchronize()
        dist.barrier()
        del graph, retained, state
    dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 4, reason="Four NVLink CUDA GPUs required"
)
def test_tokenspeed_a2a_quant(tmp_path):
    mp.spawn(_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=4, join=True)

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

"""Projection TP kernel correctness at TP4 with C128 per rank."""

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
from tokenspeed_kernel.ops.communication.triton import (
    ProjectionPeerState,
    triton_pack_channel_shards_for_a2a,
    triton_projection_reduce_scatter,
)
from tokenspeed_kernel.ops.communication.trtllm import (
    TrtllmAllGatherQuantState,
    trtllm_allgather_fp8_quantize,
)
from tokenspeed_kernel.ops.gemm.fp8_utils import (
    flashinfer_fp8_blockscale_quantize_prepacked,
)

ROWS = 128
HIDDEN = 7168
KDA_WIDTH = 16384


def _check_quantized(actual, expected):
    torch.testing.assert_close(
        actual[0].view(torch.uint8), expected[0].view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


def _check_pack_and_reduce(rank, device):
    inputs = torch.randn(ROWS, KDA_WIDTH, dtype=torch.bfloat16, device=device)
    workspace = torch.empty(
        (4, ROWS, KDA_WIDTH // 4), dtype=inputs.dtype, device=device
    )
    packed = triton_pack_channel_shards_for_a2a(inputs, workspace)
    expected = inputs.view(ROWS, 4, KDA_WIDTH // 4).transpose(0, 1).contiguous()
    torch.testing.assert_close(packed.view_as(expected), expected, rtol=0, atol=0)

    state = ProjectionPeerState(dist.group.WORLD, ROWS, HIDDEN, device)
    partial = state.input_buffer(ROWS)
    values = (
        torch.arange(partial.numel(), device=device, dtype=torch.float32).view_as(
            partial
        )
        % 31
    )
    partial.copy_((values * 0.015625 + (rank + 1) * 0.125).to(partial.dtype))

    peers = [torch.empty_like(partial) for _ in range(4)]
    dist.all_gather(peers, partial)
    expected = sum(peer[rank * ROWS : (rank + 1) * ROWS].float() for peer in peers).to(
        partial.dtype
    )
    actual = triton_projection_reduce_scatter(state, partial, ROWS)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = triton_projection_reduce_scatter(state, partial, ROWS)
    partial.add_(0.03125)
    dist.all_gather(peers, partial)
    expected = sum(peer[rank * ROWS : (rank + 1) * ROWS].float() for peer in peers).to(
        partial.dtype
    )
    graph.replay()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    del graph, state


def _check_a2a_quant(rank, device):
    state = TokenSpeedA2ALamportState(
        dist.group.WORLD,
        ROWS,
        KDA_WIDTH,
        device,
        min(128, torch.cuda.get_device_properties(device).multi_processor_count),
    )
    state.prepare_fp8_quantization()
    inputs = torch.randn(ROWS, KDA_WIDTH, dtype=torch.bfloat16, device=device)

    def reference():
        exchanged = tokenspeed_a2a_lamport(state, inputs, inverse=False, out=None)
        return flashinfer_fp8_blockscale_quantize_prepacked(exchanged, 128)

    _check_quantized(tokenspeed_a2a_lamport_fp8_quantize(state, inputs), reference())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = tokenspeed_a2a_lamport_fp8_quantize(state, inputs)
    inputs.mul_(0.75).add_((rank + 1) * 0.03125)
    expected = reference()
    graph.replay()
    _check_quantized(actual, expected)
    torch.cuda.synchronize(device)
    dist.barrier()
    del graph, state


def _check_allgather_quant(rank, device):
    state = TrtllmAllGatherQuantState(
        dist.group.WORLD,
        ROWS,
        HIDDEN,
        device,
        torch.cuda.get_device_properties(device).multi_processor_count,
    )
    inputs = torch.randn(ROWS, HIDDEN, dtype=torch.bfloat16, device=device)

    def reference():
        gathered = state.gather(inputs)
        return flashinfer_fp8_blockscale_quantize_prepacked(gathered, 128)

    _check_quantized(trtllm_allgather_fp8_quantize(state, inputs), reference())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = trtllm_allgather_fp8_quantize(state, inputs)
    inputs.mul_(0.75).add_((rank + 1) * 0.03125)
    expected = reference()
    graph.replay()
    _check_quantized(actual, expected)
    del graph
    state.close()


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
    torch.manual_seed(120 + rank)
    _check_pack_and_reduce(rank, device)
    _check_a2a_quant(rank, device)
    _check_allgather_quant(rank, device)
    dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 4, reason="Four NVLink CUDA GPUs required"
)
def test_projection_tp(tmp_path):
    mp.spawn(_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=4, join=True)

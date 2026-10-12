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

"""Projection exchange correctness for TP2/4/8 at KDA channel widths."""

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
    triton_pack_channel_shards_for_a2a,
)
from tokenspeed_kernel.ops.communication.trtllm import (
    TrtllmAllGatherQuantState,
    trtllm_allgather_fp8_quantize,
)
from tokenspeed_kernel.ops.gemm.flashinfer import (
    flashinfer_fp8_blockscale_quantize_prepacked,
)
from tokenspeed_kernel.ops.quantization import quantize_fp8

ROWS = 128
HIDDEN = 7168
KDA_WIDTH = 16384


def _check_quantized(actual, expected):
    torch.testing.assert_close(
        actual[0].view(torch.uint8), expected[0].view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


def _check_pack(device, size):
    inputs = torch.randn(ROWS, KDA_WIDTH, dtype=torch.bfloat16, device=device)
    workspace = torch.empty(
        (size, ROWS, KDA_WIDTH // size), dtype=inputs.dtype, device=device
    )
    packed = triton_pack_channel_shards_for_a2a(inputs, workspace)
    expected = inputs.view(ROWS, size, KDA_WIDTH // size).transpose(0, 1).contiguous()
    torch.testing.assert_close(packed.view_as(expected), expected, rtol=0, atol=0)


def _check_a2a(rank, device, size):
    state = TokenSpeedA2ALamportState(
        dist.group.WORLD,
        257,
        KDA_WIDTH,
        device,
        min(128, torch.cuda.get_device_properties(device).multi_processor_count),
    )
    state.prepare_fp8_quantization()
    state.prepare_chunk_exchange(threshold_bytes=8 * 2**20 + 1)
    # Small packets, the paired-packet launch, chunk exchange, then back to
    # packets. Odd M also exercises TP2's prepared-FP8 row padding.
    for rows in (1, 128, 257, 1):
        inputs = torch.empty(rows, KDA_WIDTH, dtype=torch.bfloat16, device=device)
        gathered = torch.empty(
            size * rows, KDA_WIDTH, dtype=inputs.dtype, device=device
        )
        shard = torch.empty(
            size * rows, KDA_WIDTH // size, dtype=inputs.dtype, device=device
        )
        restored = torch.empty_like(inputs)

        def exchange():
            tokenspeed_a2a_lamport(state, inputs, inverse=False, out=shard)
            tokenspeed_a2a_lamport(state, shard, inverse=True, out=restored)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            exchange()
        for capture in (False, True):
            # Include signed zero, infinities and NaN payloads, not just finite
            # numbers. Compare integers so all BF16 bit patterns are checked.
            inputs.view(torch.int16).random_(-32768, 32768)
            dist.all_gather_into_tensor(gathered, inputs)
            expected = gathered[
                :, rank * (KDA_WIDTH // size) : (rank + 1) * (KDA_WIDTH // size)
            ]
            if capture:
                graph.replay()
            else:
                exchange()
            torch.testing.assert_close(
                shard.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
            )
            torch.testing.assert_close(
                restored.view(torch.int16), inputs.view(torch.int16), rtol=0, atol=0
            )
        del graph

        retained = restored.clone()
        inputs.normal_()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = tokenspeed_a2a_lamport_fp8_quantize(state, inputs)
        for capture in (False, True):
            inputs.mul_(0.75).add_((rank + 1) * 0.03125)
            inputs[:, :128] = 0
            dist.all_gather_into_tensor(gathered, inputs)
            exchanged = gathered[
                :, rank * (KDA_WIDTH // size) : (rank + 1) * (KDA_WIDTH // size)
            ].contiguous()
            # The fused kernel pads the gathered shard to a multiple of four
            # rows with zero/one padding; the prepacked reference reproduces
            # that layout exactly, including for TP2's odd row counts.
            expected = flashinfer_fp8_blockscale_quantize_prepacked(exchanged, 128)
            if capture:
                graph.replay()
            else:
                actual = tokenspeed_a2a_lamport_fp8_quantize(state, inputs)
            _check_quantized(actual, expected)
            # Fused/borrowed calls must not overwrite caller-owned BF16 output.
            torch.testing.assert_close(
                restored.view(torch.int16), retained.view(torch.int16), rtol=0, atol=0
            )
        del graph
    torch.cuda.synchronize(device)
    dist.barrier()
    del state


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
        values, scales = quantize_fp8(
            gathered, granularity="token_group", group_size=128, solution="trtllm"
        )
        return values, scales.t().contiguous()

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


def _worker(rank, size, rendezvous):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=size,
        timeout=timedelta(seconds=600),
        device_id=device,
    )
    torch.manual_seed(120 + rank)
    _check_pack(device, size)
    _check_a2a(rank, device, size)
    _check_allgather_quant(rank, device)
    dist.destroy_process_group()


@pytest.mark.parametrize("size", [2, 4, 8], ids=["tp2", "tp4", "tp8"])
def test_projection_tp(size, tmp_path):
    if torch.cuda.device_count() < size:
        pytest.skip(f"requires {size} NVLink CUDA GPUs on one host")
    mp.spawn(
        _worker,
        args=(size, f"file://{tmp_path / 'rendezvous'}"),
        nprocs=size,
        join=True,
    )

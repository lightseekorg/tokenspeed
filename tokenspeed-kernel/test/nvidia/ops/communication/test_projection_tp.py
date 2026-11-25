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

import ctypes
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.ops.communication.cuda import (
    TokenSpeedA2ALamportState,
    tokenspeed_a2a_lamport,
    tokenspeed_a2a_lamport_fp8,
    tokenspeed_a2a_lamport_fp8_quantize,
)
from tokenspeed_kernel.ops.communication.triton import (
    triton_pack_channel_shards_for_a2a,
)
from tokenspeed_kernel.ops.communication.trtllm import (
    TrtllmAllGatherQuantState,
    TrtllmAllGatherState,
    TrtllmReduceScatterState,
    trtllm_allgather,
    trtllm_allgather_fp8_quantize,
    trtllm_reduce_scatter,
)
from tokenspeed_kernel.ops.gemm._flashinfer.fp8_quant import (
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
    for step, rows in enumerate((1, 128, 257, 1)):
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
        # Advance only the active counter and only once. Rewinding a chunk
        # counter onto flags already published by earlier calls creates ABA.
        if step == 0:
            state.control[0] = -2
        elif rows == 257:
            state.chunk_control[0] = 2**32 - 2
        for capture in (False, True, True, True):
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
    _check_prequantized_a2a(state, rank, device, size)
    torch.cuda.synchronize(device)
    dist.barrier()
    del state
    _check_epoch_boundaries(rank, device, size)


def _check_epoch_boundaries(rank, device, size):
    rows = 128  # At KDA width the full input exercises the 4 MiB paired launch.
    state = TokenSpeedA2ALamportState(dist.group.WORLD, rows, KDA_WIDTH, device, 8)
    full = torch.empty(rows, KDA_WIDTH, dtype=torch.bfloat16, device=device)
    output = torch.empty_like(full).view(size * rows, -1)

    def verify(x, call, skew=False):
        x.view(torch.int16).random_(-32768, 32768)
        gathered = x.new_empty((size * len(x), KDA_WIDTH))
        dist.all_gather_into_tensor(gathered, x)
        if skew and rank == 1:
            torch.cuda._sleep(50000)
        result = call()
        expected = gathered.chunk(size, dim=1)[rank]
        assert torch.equal(result.view(torch.int16), expected.view(torch.int16))

    def exchange():
        return tokenspeed_a2a_lamport(state, full, False, out=output)

    for _ in range(3):
        verify(full, exchange)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        small = tokenspeed_a2a_lamport(state, full[:1], False, out=None)

    def replay():
        graph.replay()
        return small

    state.control[0] = -1
    verify(full[:1], replay)
    assert state.control.tolist() == [1, 0, 0, 1]
    assert state.scratch[: 3 * state.words].count_nonzero().item() == 0
    verify(full, exchange, skew=True)
    state.control[0] = -1
    verify(full, exchange)
    assert state.control.tolist() == [1, 0, 0, 2]
    state.control[0] = 2**31 - 2
    for _ in range(4):
        verify(full[:1], replay)
    assert state.control[0].item() == 2 - 2**31
    graph.reset()

    state.prepare_chunk_exchange(1)
    flags = state.chunk_flags.view(3, size, state.blocks)
    for seed, count in ((2**32 - 2, 5), (2**63 - 2, 4)):
        state.chunk_control[0] = seed
        for _ in range(count):
            verify(full, exchange)
        last = seed + count - 1
        signed = last if last < 2**63 else last - 2**64
        assert state.chunk_control.tolist() == [signed + 1, 0]
        assert (flags[last % 3] == signed).all().item()

    if rank == 0:
        packet = (output, state.peers, state.control, state.scratch)
        args = (rows, KDA_WIDTH, rank, state.blocks, False)
        with pytest.raises(RuntimeError, match="numel"):
            state.module.exchange(full, *packet, state.words + 1, *args)
        misaligned = full.new_empty(full.numel() + 1)[1:].view_as(full)
        with pytest.raises(RuntimeError, match="alignment"):
            state.module.exchange(misaligned, *packet, state.words, *args)
        chunk = (
            output,
            state.chunk_peers,
            state.chunk_control,
            state.chunk_scratch,
            state.chunk_flags,
            state.chunk_capacity,
            rows,
            KDA_WIDTH,
            rank,
        )
        with pytest.raises(RuntimeError, match="numel"):
            state.chunk_module.exchange_chunk(
                state.chunk_flag_peers, full, *chunk, state.blocks + 1, False
            )
        with pytest.raises(RuntimeError, match="alignment"):
            state.chunk_module.exchange_chunk(
                state.chunk_flag_peers, misaligned, *chunk, state.blocks, False
            )
        with pytest.raises(RuntimeError, match="numel"):
            state.module.exchange(
                full[:1],
                output[:size],
                *packet[1:],
                state.words - 1,
                1,
                KDA_WIDTH,
                rank,
                state.blocks,
                False,
            )
        with pytest.raises(RuntimeError, match="numel"):
            state.chunk_module.exchange_chunk(
                state.chunk_flag_peers,
                full[:1],
                output[:size],
                *chunk[1:5],
                state.chunk_capacity - 2,
                1,
                KDA_WIDTH,
                rank,
                state.blocks,
                False,
            )
        with pytest.raises(RuntimeError, match="numel"):
            state.chunk_module.exchange_chunk(
                state.chunk_flag_peers, full, *chunk, state.blocks - 1, False
            )

    _check_dlpack_offset(state, full[:1], rank, size)
    if size in (2, 4):
        narrow = TokenSpeedA2ALamportState(dist.group.WORLD, 1, 8, device, 1)
        with pytest.raises(ValueError, match="alias"):
            tokenspeed_a2a_lamport(
                narrow, narrow.control.view(torch.bfloat16).view(1, 8), False, out=None
            )
        view = narrow.scratch[narrow.words : narrow.words + 2]
        alias = torch.from_dlpack(torch.utils.dlpack.to_dlpack(view))
        assert (
            alias.untyped_storage().data_ptr()
            != narrow.scratch.untyped_storage().data_ptr()
        )
        alias = alias.view(torch.bfloat16).view(1, 8)
        with pytest.raises(ValueError, match="alias"):
            tokenspeed_a2a_lamport(narrow, alias, False, out=None)
    torch.cuda.synchronize(device)
    dist.barrier()


def _check_dlpack_offset(state, inputs, rank, size):
    from flashinfer.comm.dlpack_utils import DLTensor
    from tvm_ffi import from_dlpack

    # Torch normally folds a slice's offset into data. Export the same view
    # with a separate DLPack byte_offset to exercise native address arithmetic.
    def output_view(storage, offset):
        view = storage[offset : offset + inputs.numel()].view(size, -1)
        capsule = torch.utils.dlpack.to_dlpack(view)
        get_pointer = ctypes.PYFUNCTYPE(
            ctypes.POINTER(DLTensor), ctypes.py_object, ctypes.c_char_p
        )(("PyCapsule_GetPointer", ctypes.pythonapi))
        tensor = get_pointer(capsule, b"dltensor").contents
        tensor.data = storage.data_ptr()
        tensor.byte_offset = offset * storage.element_size()
        return from_dlpack(capsule)

    inputs.fill_(rank + 1)
    for chunk in (False, True):
        storage = inputs.new_full((8 + inputs.numel(),), -1)

        def exchange(out):
            if chunk:
                state.chunk_module.exchange_chunk(
                    state.chunk_flag_peers,
                    inputs,
                    out,
                    state.chunk_peers,
                    state.chunk_control,
                    state.chunk_scratch,
                    state.chunk_flags,
                    state.chunk_capacity,
                    1,
                    state.channels,
                    rank,
                    state.blocks,
                    False,
                )
            else:
                state.module.exchange(
                    inputs,
                    out,
                    state.peers,
                    state.control,
                    state.scratch,
                    state.words,
                    1,
                    state.channels,
                    rank,
                    state.blocks,
                    False,
                )

        with pytest.raises(RuntimeError, match="alignment"):
            exchange(output_view(storage, 1))
        exchange(output_view(storage, 8))
        assert (storage[:8] == -1).all().item()
        for peer, row in enumerate(storage[8:].view(size, -1)):
            assert (row == peer + 1).all().item()


def _check_prequantized_a2a(state, rank, device, size):
    # Small and pipelined packet reads; the last owner stays empty. Reuse the
    # BF16 communicator so format changes exercise the same generation ring.
    for rows in (1, 33):
        local_rows = max(rows - rank, 0) if rank != size - 1 else 0
        values = torch.empty(
            (local_rows, KDA_WIDTH), dtype=torch.float8_e4m3fn, device=device
        )
        scales = torch.empty(
            (local_rows, KDA_WIDTH // 128), dtype=torch.float32, device=device
        )
        bf16 = torch.randn(rows, KDA_WIDTH, dtype=torch.bfloat16, device=device)
        bf16_out = torch.empty(
            size * rows, KDA_WIDTH // size, dtype=torch.bfloat16, device=device
        )

        def exchange():
            tokenspeed_a2a_lamport(state, bf16, inverse=False, out=bf16_out)
            return tokenspeed_a2a_lamport_fp8(state, values, scales, rows)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = exchange()
        state.control[0] = -2
        for capture in (True, True, False):
            # Preserve arbitrary payloads, including signed zero and NaNs in
            # either format; compare integer views instead of float values.
            values.view(torch.uint8).random_(0, 256)
            scales.view(torch.int32).random_(-(2**31), 2**31)
            expected = []
            for source in (values.view(torch.uint8), scales.view(torch.int32)):
                padded = source.new_zeros(rows, source.shape[1])
                padded[:local_rows].copy_(source)
                parts = [torch.empty_like(padded) for _ in range(size)]
                dist.all_gather(parts, padded)
                width = source.shape[1] // size
                expected.append(torch.cat(parts)[:, rank * width : (rank + 1) * width])
            torch.cuda._sleep(100000 * rank)
            if capture:
                graph.replay()
                actual = captured
            else:
                actual = exchange()
            for tensor, ref in zip(actual, expected):
                torch.testing.assert_close(tensor.view(ref.dtype), ref, rtol=0, atol=0)
        graph.reset()


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


def _check_shared_collectives(rank, device, size):
    rows = 256
    group = dist.group.WORLD
    gather = TrtllmAllGatherState(group, rows, HIDDEN, device, True)
    reduction = TrtllmReduceScatterState(group, rows, HIDDEN, device)
    try:
        inputs = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device=device)
        partial = torch.zeros(size * rows, HIDDEN, dtype=inputs.dtype, device=device)
        trtllm_allgather(gather, inputs)
        trtllm_reduce_scatter(reduction, partial, rows)
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gathered = trtllm_allgather(gather, inputs)
            reduced = trtllm_reduce_scatter(reduction, partial, rows)
        for value in (rank + 1, rank + 2):
            inputs.fill_(value)
            partial.fill_(value)
            expected_gather = torch.empty_like(partial)
            expected_reduced = torch.empty_like(inputs)
            dist.all_gather_single(expected_gather, inputs)
            dist.reduce_scatter_single(expected_reduced, partial)
            graph.replay()
            torch.testing.assert_close(gathered, expected_gather, rtol=0, atol=0)
            torch.testing.assert_close(reduced, expected_reduced, rtol=0, atol=0)
        del graph
    finally:
        gather.close()
        reduction.close()


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
    _check_shared_collectives(rank, device, size)
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

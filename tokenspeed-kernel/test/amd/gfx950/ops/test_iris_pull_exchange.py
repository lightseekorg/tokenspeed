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

"""Iris pull gathers, all-to-alls and records combines match NCCL's (and the
Triton combine's), back to back and replayed."""

import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.platform import current_platform
from utils import is_amd

pytestmark = pytest.mark.skipif(not is_amd(), reason="Iris requires AMD ROCm")

MAX_BYTES = 8 << 20
# K3.1 KVP8 decode shares: score rows riding their FP8 query, then the
# attention records each rank sends every peer.
SIZES = [(rows * 15104, 8 * rows * 12 * 257 * 4) for rows in (1, 4, 16, 48, 64)]


@pytest.fixture(autouse=True)
def _require_iris():
    pytest.importorskip("tokenspeed_kernel.ops.communication._iris")


def _exchange_worker(rank: int, port: int) -> None:
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
    from tokenspeed_kernel.ops.communication._iris import (
        IrisGroup,
        get_or_create_context,
    )
    from tokenspeed_kernel.ops.communication._iris.pull_exchange import (
        IrisPullExchange,
    )

    ctx = get_or_create_context(heap_size=1 << 28)
    exchange = IrisPullExchange(IrisGroup(ctx, group, rank, device), MAX_BYTES)
    exchange.peers.finish_prepare()
    gen = torch.Generator(device=device)

    def produce(nbytes: int, seed: int) -> torch.Tensor:
        gen.manual_seed(8 * seed + rank)
        return torch.randint(
            0, 256, (nbytes,), dtype=torch.uint8, device=device, generator=gen
        )

    def reference(data: torch.Tensor, gather: bool) -> torch.Tensor:
        if gather:
            out = torch.empty((8 * data.numel(),), dtype=torch.uint8, device=device)
            dist.all_gather_into_tensor(out, data, group=group)
        else:
            out = torch.empty_like(data)
            dist.all_to_all_single(out, data, group=group)
        return out

    def run(data: torch.Tensor, gather: bool, out: torch.Tensor) -> None:
        send = exchange.input(data.numel())
        send.copy_(data)
        if gather:
            assert exchange.can_all_gather(out, send)
            exchange.all_gather(out, send)
        else:
            assert exchange.can_all_to_all(out, send)
            exchange.all_to_all(out, send)

    # Every decode size; the slow rank makes peers wait for its producer.
    for call, sizes in enumerate(SIZES):
        for gather, nbytes in zip((True, False), sizes):
            data = produce(nbytes, call)
            expected = reference(data, gather)
            torch.cuda.synchronize()
            dist.barrier()
            if rank == 3:
                torch.cuda._sleep(5_000_000)
            out = torch.empty_like(expected)
            run(data, gather, out)
            assert torch.equal(out, expected)
            torch.cuda.synchronize()
            dist.barrier()

    # Back to back, as a DSA layer's gather and records exchange: each call's
    # producer overwrites the buffer a slow peer would still pull without the
    # exit barrier. Rank 3's side-stream copies slow its pulls.
    calls = []
    for call in range(40):
        gather = call % 2 == 0
        data = produce(SIZES[call // 2 % len(SIZES)][call % 2], 100 + call)
        calls.append((data, gather, reference(data, gather)))
    hog = torch.empty((1 << 28,), dtype=torch.uint8, device=device)
    hog_stream = torch.cuda.Stream(device)
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 3:
        with torch.cuda.stream(hog_stream):
            for _ in range(200):
                hog.copy_(hog.flip(0))
    results = []
    for call, (data, gather, expected) in enumerate(calls):
        if rank == 3 and call % 4 == 1:
            torch.cuda._sleep(200_000)
        results.append(torch.empty_like(expected))
        run(data, gather, results[-1])
    torch.cuda.synchronize()
    for result, (_, _, expected) in zip(results, calls):
        assert torch.equal(result, expected)
    dist.barrier()

    # Replayed graphs read each replay's epoch from the device.
    gather_bytes, records_bytes = SIZES[2]
    sources = (
        torch.empty((gather_bytes,), dtype=torch.uint8, device=device),
        torch.empty((records_bytes,), dtype=torch.uint8, device=device),
    )
    outputs = (
        torch.empty((8 * gather_bytes,), dtype=torch.uint8, device=device),
        torch.empty((records_bytes,), dtype=torch.uint8, device=device),
    )
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        for _ in range(3):
            run(sources[0], True, outputs[0])
            run(sources[1], False, outputs[1])
    for replay in range(10):
        for index, source in enumerate(sources):
            source.copy_(produce(source.numel(), 200 + 2 * replay + index))
        expected = [
            reference(source, index == 0) for index, source in enumerate(sources)
        ]
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 3 and replay % 2:
            torch.cuda._sleep(2_000_000)
        graph.replay()
        torch.cuda.synchronize()
        for output, wanted in zip(outputs, expected):
            assert torch.equal(output, wanted)
    dist.barrier()

    assert exchange.input(MAX_BYTES + 16) is None
    send = exchange.input(160)
    gathered = torch.empty((8 * 160,), dtype=torch.uint8, device=device)
    assert exchange.can_all_gather(gathered, send)
    assert not exchange.can_all_gather(gathered[:-16], send)
    assert not exchange.can_all_gather(gathered, torch.empty_like(send))
    assert not exchange.can_all_gather(
        gathered, exchange.buffer.view(torch.uint8)[16:176]
    )
    assert not exchange.can_all_gather(gathered[: 8 * 20], exchange.input(20))
    assert exchange.can_all_to_all(gathered[:1280], exchange.input(1280))
    assert not exchange.can_all_to_all(gathered[:160], send)
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()


HEADS, HEAD_DIM = 12, 512


def _records(tokens: int, rank: int, device: torch.device, seed: int) -> torch.Tensor:
    """Rank ``rank``'s K3.1 KVP8 decode records for every rank: BF16 partials,
    LSEs with empty (-inf, and +inf as the combine also reads it) and NaN
    entries, and a head no rank attended."""
    gen = torch.Generator(device=device).manual_seed(8 * seed + rank)
    values = torch.randn((8, tokens, HEADS, HEAD_DIM), device=device, generator=gen).to(
        torch.bfloat16
    )
    lse = torch.randn((8, tokens, HEADS), device=device, generator=gen) * 4
    lse.view(-1)[::7] = -float("inf")
    lse.view(-1)[3::29] = float("inf")
    if tokens >= 4:
        lse[rank, 3, 5] = float("nan")
        lse[:, 2, 1] = -float("inf")
    records = torch.empty(
        (8, tokens, HEADS, HEAD_DIM // 2 + 1), dtype=torch.int32, device=device
    )
    records[..., : HEAD_DIM // 2] = values.view(torch.int32).view(
        8, tokens, HEADS, HEAD_DIM // 2
    )
    records[..., HEAD_DIM // 2] = lse.view(torch.int32)
    return records


def _same_bits(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    return bool(
        (
            (actual.view(torch.int16) == expected.view(torch.int16))
            | (actual.isnan() & expected.isnan())
        ).all()
    )


def _combine_worker(rank: int, port: int) -> None:
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
    from tokenspeed_kernel.ops.attention.dsv4._triton.dcp import (
        combine_dcp_attention_partials,
    )
    from tokenspeed_kernel.ops.communication._iris import (
        IrisGroup,
        get_or_create_context,
    )
    from tokenspeed_kernel.ops.communication._iris.pull_exchange import (
        IrisPullExchange,
    )

    ctx = get_or_create_context(heap_size=1 << 28)
    exchange = IrisPullExchange(IrisGroup(ctx, group, rank, device), MAX_BYTES)
    exchange.peers.finish_prepare()
    dtypes = {"partial_dtype": torch.bfloat16, "output_dtype": torch.bfloat16}

    def reference(records: torch.Tensor) -> torch.Tensor:
        received = torch.empty_like(records)
        dist.all_to_all_single(received.view(-1), records.view(-1), group=group)
        return combine_dcp_attention_partials(
            received, head_dim=HEAD_DIM, sink=None, **dtypes
        )

    def lend(records: torch.Tensor) -> torch.Tensor:
        send = exchange.input(records.nbytes).view(torch.int32).view(records.shape)
        send.copy_(records)
        assert exchange.can_combine_records(send, head_dim=HEAD_DIM, **dtypes)
        return send

    # One-subgroup and four-subgroup programs; the slow rank makes peers wait.
    for tokens in (1, 4, 16, 48, 64):
        records = _records(tokens, rank, device, tokens)
        expected = reference(records)
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 3:
            torch.cuda._sleep(5_000_000)
        combined = exchange.combine_records(lend(records), head_dim=HEAD_DIM)
        assert combined.shape == (tokens, HEADS, HEAD_DIM)
        assert combined.dtype == torch.bfloat16 and combined.is_contiguous()
        assert _same_bits(combined, expected)
        torch.cuda.synchronize()
        dist.barrier()

    # Between gathers, as a DSA layer's records follow its score gather: each
    # producer overwrites the buffer the previous call's slow peer pulled.
    calls = []
    for call in range(12):
        records = _records(16 + call, rank, device, 100 + call)
        gathered = torch.randint(
            0, 256, (4096,), dtype=torch.uint8, device=device
        ).roll(rank)
        wanted = torch.empty((8 * 4096,), dtype=torch.uint8, device=device)
        dist.all_gather_into_tensor(wanted, gathered, group=group)
        calls.append((records, gathered, reference(records), wanted))
    torch.cuda.synchronize()
    dist.barrier()
    results = []
    for call, (records, gathered, _, _) in enumerate(calls):
        if rank == 3 and call % 3 == 1:
            torch.cuda._sleep(200_000)
        send = exchange.input(gathered.numel())
        send.copy_(gathered)
        out = torch.empty((8 * 4096,), dtype=torch.uint8, device=device)
        exchange.all_gather(out, send)
        results.append(
            (out, exchange.combine_records(lend(records), head_dim=HEAD_DIM))
        )
    torch.cuda.synchronize()
    for (out, combined), (_, _, expected, wanted) in zip(results, calls):
        assert torch.equal(out, wanted)
        assert _same_bits(combined, expected)
    dist.barrier()

    # Replayed graphs read each replay's epoch from the device.
    source = torch.empty_like(_records(48, rank, device, 0))
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        for _ in range(3):
            replayed = exchange.combine_records(lend(source), head_dim=HEAD_DIM)
    for replay in range(6):
        source.copy_(_records(48, rank, device, 200 + replay))
        expected = reference(source)
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 3 and replay % 2:
            torch.cuda._sleep(2_000_000)
        graph.replay()
        torch.cuda.synchronize()
        assert _same_bits(replayed, expected)
    dist.barrier()

    empty = exchange.input(0).view(torch.int32).view(_records(0, rank, device, 0).shape)
    assert not exchange.can_combine_records(empty, head_dim=HEAD_DIM, **dtypes)
    records = lend(_records(4, rank, device, 0))
    assert not exchange.can_combine_records(
        records.clone(), head_dim=HEAD_DIM, **dtypes
    )
    assert not exchange.can_combine_records(
        records.view(torch.float32), head_dim=HEAD_DIM, **dtypes
    )
    assert not exchange.can_combine_records(records[:4], head_dim=HEAD_DIM, **dtypes)
    assert not exchange.can_combine_records(
        records[:, :, :6], head_dim=HEAD_DIM, **dtypes
    )
    assert not exchange.can_combine_records(records, head_dim=HEAD_DIM - 2, **dtypes)
    assert not exchange.can_combine_records(
        records,
        head_dim=HEAD_DIM,
        partial_dtype=torch.float32,
        output_dtype=torch.bfloat16,
    )
    assert not exchange.can_combine_records(
        records,
        head_dim=HEAD_DIM,
        partial_dtype=torch.bfloat16,
        output_dtype=torch.float16,
    )
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(
    not current_platform().is_cdna4 or torch.cuda.device_count() < 8,
    reason="requires eight CDNA4 GPUs",
)
def test_pull_exchange_matches_nccl_gathers_and_all_to_alls():
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_exchange_worker, args=(port,), nprocs=8, join=True)


@pytest.mark.skipif(
    not current_platform().is_cdna4 or torch.cuda.device_count() < 8,
    reason="requires eight CDNA4 GPUs",
)
def test_pull_combine_matches_nccl_all_to_all_then_combine():
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_combine_worker, args=(port,), nprocs=8, join=True)

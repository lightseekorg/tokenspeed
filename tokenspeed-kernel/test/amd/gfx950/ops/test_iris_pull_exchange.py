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
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.platform import current_platform
from utils import assert_no_triton_compile, is_amd

pytestmark = pytest.mark.skipif(not is_amd(), reason="Iris requires AMD ROCm")

MAX_BYTES = 8 << 20
# K3.1 KVP8 decode shares: score rows riding their FP8 query, then the
# attention records each rank sends every peer.
ROWS = (1, 4, 16, 48, 64)
SIZES = [(rows * 15104, 8 * rows * 12 * 257 * 4) for rows in ROWS]
HEADS, HEAD_DIM = 12, 512


@pytest.fixture(autouse=True)
def _require_iris():
    pytest.importorskip("tokenspeed_kernel.ops.communication._iris")


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


def _pull_worker(rank: int, port: int) -> None:
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
        _iris_pull_combine_records,
        _iris_pull_exchange,
    )

    ctx = get_or_create_context(heap_size=1 << 28)
    exchange = IrisPullExchange(IrisGroup(ctx, group, rank, device), MAX_BYTES)
    exchange.peers.finish_prepare()
    gen = torch.Generator(device=device)
    dtypes = {"partial_dtype": torch.bfloat16, "output_dtype": torch.bfloat16}

    def produce(nbytes: int, seed: int) -> torch.Tensor:
        gen.manual_seed(8 * seed + rank)
        return torch.randint(
            0, 256, (nbytes,), dtype=torch.uint8, device=device, generator=gen
        )

    # ``gather`` is None for a records combine.
    def reference(data: torch.Tensor, gather: bool | None) -> torch.Tensor:
        if gather:
            out = torch.empty((8 * data.numel(),), dtype=torch.uint8, device=device)
            dist.all_gather_into_tensor(out, data, group=group)
            return out
        out = torch.empty_like(data)
        dist.all_to_all_single(out.view(-1), data.view(-1), group=group)
        if gather is None:
            return combine_dcp_attention_partials(
                out, head_dim=HEAD_DIM, sink=None, **dtypes
            )
        return out

    def lend(records: torch.Tensor) -> torch.Tensor:
        send = exchange.input(records.nbytes).view(torch.int32).view(records.shape)
        send.copy_(records)
        assert exchange.can_combine_records(send, head_dim=HEAD_DIM, **dtypes)
        return send

    def run(data: torch.Tensor, gather: bool | None) -> torch.Tensor:
        if gather is None:
            return exchange.combine_records(lend(data), head_dim=HEAD_DIM)
        send = exchange.input(data.numel())
        send.copy_(data)
        out = torch.empty(
            ((8 if gather else 1) * data.numel(),), dtype=torch.uint8, device=device
        )
        if gather:
            assert exchange.can_all_gather(out, send)
            exchange.all_gather(out, send)
        else:
            assert exchange.can_all_to_all(out, send)
            exchange.all_to_all(out, send)
        return out

    def matches(out: torch.Tensor, expected: torch.Tensor, gather: bool | None) -> bool:
        if gather is None:
            return _same_bits(out, expected)
        return torch.equal(out, expected)

    # Every decode size, through both kernels' one-subgroup and wider
    # programs; the slow rank makes peers wait for its producer.
    for call, (rows, sizes) in enumerate(zip(ROWS, SIZES)):
        inputs = [produce(nbytes, call) for nbytes in sizes]
        inputs.append(_records(rows, rank, device, rows))
        for data, gather in zip(inputs, (True, False, None)):
            expected = reference(data, gather)
            torch.cuda.synchronize()
            dist.barrier()
            if rank == 3:
                torch.cuda._sleep(5_000_000)
            out = run(data, gather)
            if gather is None:
                assert out.shape == (rows, HEADS, HEAD_DIM)
                assert out.dtype == torch.bfloat16 and out.is_contiguous()
            assert matches(out, expected, gather)
            torch.cuda.synchronize()
            dist.barrier()

    # Those sizes warmed every launch shape; other batches' shares reuse them.
    with assert_no_triton_compile(_iris_pull_exchange, _iris_pull_combine_records):
        for rows in (3, 5, 40):
            inputs = (
                produce(rows * 15104, 50 + rows),
                produce(rows * 98688, 50 + rows),
                _records(rows, rank, device, 50 + rows),
            )
            for data, gather in zip(inputs, (True, False, None)):
                expected = reference(data, gather)
                assert matches(run(data, gather), expected, gather)
    torch.cuda.synchronize()
    dist.barrier()

    # Back to back, as a DSA layer's score gather, records exchange and
    # records combine: each call's producer overwrites the buffer a slow peer
    # would still pull without the exit barrier. Rank 3's side-stream copies
    # slow its pulls.
    calls = []
    for call in range(48):
        gather = (True, False, True, None)[call % 4]
        if gather is None:
            data = _records(16 + call // 4, rank, device, 100 + call)
        else:
            data = produce(
                SIZES[call // 2 % len(SIZES)][0 if gather else 1], 100 + call
            )
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
    for call, (data, gather, _) in enumerate(calls):
        if rank == 3 and call % 3 == 1:
            torch.cuda._sleep(200_000)
        results.append(run(data, gather))
    torch.cuda.synchronize()
    for result, (_, gather, expected) in zip(results, calls):
        assert matches(result, expected, gather)
    dist.barrier()

    # Replayed graphs read each replay's epoch from the device.
    gather_bytes, records_bytes = SIZES[2]
    kinds = (True, False, None)
    sources = (
        torch.empty((gather_bytes,), dtype=torch.uint8, device=device),
        torch.empty((records_bytes,), dtype=torch.uint8, device=device),
        torch.empty_like(_records(48, rank, device, 0)),
    )
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        for _ in range(3):
            outputs = [run(source, gather) for source, gather in zip(sources, kinds)]
    for replay in range(3):
        for index, source in enumerate(sources[:2]):
            source.copy_(produce(source.numel(), 200 + 2 * replay + index))
        sources[2].copy_(_records(48, rank, device, 200 + replay))
        expected = [reference(source, gather) for source, gather in zip(sources, kinds)]
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 3 and replay % 2:
            torch.cuda._sleep(2_000_000)
        graph.replay()
        torch.cuda.synchronize()
        for output, wanted, gather in zip(outputs, expected, kinds):
            assert matches(output, wanted, gather)
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
    empty = exchange.input(0).view(torch.int32).view(_records(0, rank, device, 0).shape)
    assert not exchange.can_combine_records(empty, head_dim=HEAD_DIM, **dtypes)
    records = lend(_records(4, rank, device, 0))
    assert not exchange.can_combine_records(
        records.clone(), head_dim=HEAD_DIM, **dtypes
    )
    assert not exchange.can_combine_records(records, head_dim=HEAD_DIM - 2, **dtypes)
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(
    not current_platform().is_cdna4 or torch.cuda.device_count() < 8,
    reason="requires eight CDNA4 GPUs",
)
def test_pull_exchange_matches_nccl_gathers_all_to_alls_and_combines():
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_pull_worker, args=(port,), nprocs=8, join=True)


def _heap_room_worker(rank: int, port: int) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=180),
    )
    group = dist.new_group(backend="nccl", timeout=timedelta(seconds=180))
    from tokenspeed_kernel.ops.communication._iris import (
        MIN_HEAP_BYTES,
        get_or_create_context,
    )
    from tokenspeed_kernel.ops.communication._iris.kimi_k3 import (
        prepare_k3_communication,
    )

    prepare_k3_communication(
        group=group,
        rank_in_group=rank,
        staged_max_numel=1024,
        producer_direct_max_numel=1024,
        attnres_max_numel=0,
        attnres_max_rows=0,
        enable_lamport=False,
        sharded_max_rows=0,
        dtype=torch.bfloat16,
        device=device,
        all_reduce=None,
        pull_exchange_max_bytes=200 << 20,
    )
    # Collectives created later (a drafter's fused all-reduce) allocate from
    # the room the base heap leaves, which the exchange must not take.
    get_or_create_context(heap_size=0).zeros((MIN_HEAP_BYTES // 2,), dtype=torch.uint8)
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(
    not current_platform().is_cdna4 or torch.cuda.device_count() < 2,
    reason="requires two CDNA4 GPUs",
)
def test_a_pull_exchange_leaves_the_heap_room_for_other_collectives():
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_heap_room_worker, args=(port,), nprocs=2, join=True)


@pytest.mark.skipif(not current_platform().is_cdna4, reason="requires CDNA4")
@pytest.mark.parametrize("world, largest", [(2, 2**31 - 16), (8, 2**30 - 16)])
def test_pull_exchange_refuses_buffers_past_its_32bit_offsets(world, largest):
    from tokenspeed_kernel.ops.communication._iris.pull_exchange import (
        IrisPullExchange,
    )

    # A stand-in for the Iris group: the bound holds before any heap
    # allocation, and the buffer here is a meta tensor.
    peers = SimpleNamespace(
        world_size=world,
        device="cpu",
        ctx=SimpleNamespace(
            zeros=lambda shape, dtype: torch.empty(shape, dtype=dtype, device="meta")
        ),
    )
    assert IrisPullExchange(peers, largest).max_bytes == largest
    with pytest.raises(ValueError, match="32-bit offsets"):
        IrisPullExchange(peers, largest + 16)

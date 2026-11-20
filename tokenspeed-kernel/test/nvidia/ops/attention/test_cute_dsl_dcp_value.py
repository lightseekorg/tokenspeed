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

"""The DCP value tail (``DcpValue``) keeps the bits of the unfused chain.

Its output equals the records all-to-all, ``combine_dcp_attention_partials``
and ``mla_value``, call after call and under CUDA graph replay with new inputs;
a peer that never calls traps the kernel at the timeout, and every refusal is
voted on.
"""

import os
import socket
import time
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HEADS, LATENT, VALUE = 12, 512, 128
CAP = 128
BATCHES = (1, 8, 17, 64, 128)
TIMEOUT_NS = 60 * 10**9


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _partials(batch, degree, rank, device, seed):
    """This rank's partials of every gathered head, in padded storage.

    BF16 rows carry negative zeros and NaNs; natural-log LSEs carry empty (-inf,
    and +inf as the combine also reads it) and NaN entries.
    """
    gen = torch.Generator(device=device).manual_seed(1000 * seed + rank)
    gathered = degree * HEADS
    rows = torch.randn((batch, gathered + 16, LATENT), device=device, generator=gen)
    rows = rows.to(torch.bfloat16)
    rows.view(-1)[5::97] = -0.0
    rows.view(-1)[11::4099] = float("nan")
    lse = torch.randn((batch, gathered + 4), device=device, generator=gen) * 4
    lse.view(-1)[::7] = -float("inf")
    lse.view(-1)[3::29] = float("inf")
    lse.view(-1)[13::211] = float("nan")
    # A row whose every rank is empty.
    lse[batch // 2, :gathered] = -float("inf")
    return rows[:, :gathered], lse[:, :gathered]


def _reference(rows, lse, w_uv, gate, degree):
    from tokenspeed_kernel.ops.attention.dsv4._triton.dcp import (
        combine_dcp_attention_partials,
        dispatch_dcp_attention_partials,
    )
    from tokenspeed_kernel.ops.attention.mla.cute_dsl import cute_dsl_mla_project_value

    records = dispatch_dcp_attention_partials(rows, lse, degree=degree)
    received = torch.empty_like(records)
    dist.all_to_all_single(received.flatten(), records.flatten())
    latent = combine_dcp_attention_partials(
        received,
        head_dim=LATENT,
        partial_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        sink=None,
    )
    out = torch.empty(
        (rows.shape[0], HEADS * VALUE), dtype=torch.bfloat16, device=rows.device
    )
    cute_dsl_mla_project_value(latent, w_uv, gate, out)
    return out


def _same_bits(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def _worker(rank, degree, port):
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=degree,
        device_id=device,
        timeout=timedelta(seconds=120),
    )
    try:
        from tokenspeed_kernel.ops.attention.mla.cute_dsl import create_dcp_value_tail
        from tokenspeed_kernel.ops.communication.fabric import gather_fabric_map

        gather_fabric_map()
        tail = create_dcp_value_tail(
            dist.new_group(backend="gloo"),
            device,
            heads=HEADS,
            rows=CAP,
            timeout_ns=TIMEOUT_NS,
            eligible=True,
        )
        assert tail is not None
        gen = torch.Generator(device=device).manual_seed(77 + rank)
        w_uv = torch.randn((HEADS, VALUE, LATENT), device=device, generator=gen)
        w_uv = (w_uv / 16).to(torch.bfloat16).transpose(1, 2)
        failures = []
        # Back-to-back calls of changing sizes alternate the parities.
        for seed, batch in enumerate(BATCHES * 2):
            rows, lse = _partials(batch, degree, rank, device, seed)
            gate = None
            if seed % 2:
                gate = torch.randn((batch, HEADS * VALUE), device=device, generator=gen)
                gate = gate.to(torch.bfloat16)
            expected = _reference(rows, lse, w_uv, gate, degree)
            out = torch.full_like(expected, 7)
            assert tail.serves(rows, lse, w_uv, gate, out)
            tail(rows, lse, w_uv, gate, out)
            torch.cuda.synchronize()
            if not _same_bits(out, expected):
                failures.append(("eager", batch, seed))
        rows, lse = _partials(CAP + 1, degree, rank, device, 40)
        out = torch.empty((CAP + 1, HEADS * VALUE), dtype=torch.bfloat16, device=device)
        assert not tail.serves(rows, lse, w_uv, None, out)
        # Graph replays with new inputs.
        batch = 24
        rows, lse = _partials(batch, degree, rank, device, 50)
        gate = torch.randn((batch, HEADS * VALUE), device=device, generator=gen).to(
            torch.bfloat16
        )
        out = torch.empty((batch, HEADS * VALUE), dtype=torch.bfloat16, device=device)
        side = torch.cuda.Stream(device)
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            assert tail.serves(rows, lse, w_uv, gate, out)
            tail(rows, lse, w_uv, gate, out)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(3):
                assert tail.serves(rows, lse, w_uv, gate, out)
            tail(rows, lse, w_uv, gate, out)
        for seed in (60, 61):
            fresh_rows, fresh_lse = _partials(batch, degree, rank, device, seed)
            rows.copy_(fresh_rows)
            lse.copy_(fresh_lse)
            torch.cuda.synchronize()
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize()
            if not _same_bits(out, _reference(rows, lse, w_uv, gate, degree)):
                failures.append(("graph", batch, seed))
        graph.reset()
        flag = torch.tensor([len(failures)], device=device)
        dist.all_reduce(flag)
        assert int(flag) == 0, f"rank {rank}: {failures}"
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("degree", [2, 4])
def test_dcp_value_is_the_combine_and_projection_bit_for_bit(degree):
    if torch.cuda.device_count() < degree:
        pytest.skip(f"needs {degree} GPUs")
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("DcpValue needs SM100 or SM103")
    mp.spawn(_worker, args=(degree, _port()), nprocs=degree, join=True)


def _init(rank, degree, port):
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=degree,
        device_id=device,
        timeout=timedelta(seconds=120),
    )
    from tokenspeed_kernel.ops.communication.fabric import gather_fabric_map

    gather_fabric_map()
    return device


def _refusal_worker(rank, degree, port):
    device = _init(rank, degree, port)
    try:
        from tokenspeed_kernel.ops.attention.mla.cute_dsl import create_dcp_value_tail

        group = dist.new_group(backend="gloo")
        common = dict(heads=HEADS, rows=CAP, timeout_ns=TIMEOUT_NS, eligible=True)
        # One rank's refusal leaves every rank without a tail.
        mine = dict(common, eligible=rank != 1)
        assert create_dcp_value_tail(group, device, **mine) is None
        assert (
            create_dcp_value_tail(group, device, **dict(common, timeout_ns=0)) is None
        )
        # A mailbox of 2**31 elements would wrap the merge's int32 offsets.
        rows = -(-(2**31) // (2 * degree * HEADS * LATENT))
        assert create_dcp_value_tail(group, device, **dict(common, rows=rows)) is None
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _timeout_worker(rank, degree, port):
    """Rank 1 never calls: rank 0's tail traps once its polls pass the timeout."""
    device = _init(rank, degree, port)
    from tokenspeed_kernel.ops.attention.mla.cute_dsl import create_dcp_value_tail

    group = dist.new_group(backend="gloo")
    timeout_ns = 2 * 10**9
    tail = create_dcp_value_tail(
        group, device, heads=HEADS, rows=CAP, timeout_ns=timeout_ns, eligible=True
    )
    assert tail is not None
    verdict = torch.zeros(1, dtype=torch.int64)
    if rank == 0:
        rows, lse = _partials(8, degree, rank, device, 0)
        w_uv = torch.zeros((HEADS, VALUE, LATENT), device=device)
        w_uv = w_uv.to(torch.bfloat16).transpose(1, 2)
        out = torch.empty((8, HEADS * VALUE), dtype=torch.bfloat16, device=device)
        start = time.monotonic()
        try:
            tail(rows, lse, w_uv, None, out)
            torch.cuda.synchronize()
        except RuntimeError:
            waited = time.monotonic() - start
            verdict[0] = int(timeout_ns / 1e9 <= waited < 60)
    # The trap ends rank 0's CUDA context: report over gloo and exit at once.
    dist.all_reduce(verdict, group=group)
    os._exit(0 if int(verdict) == 1 else 1)


@pytest.mark.parametrize("degree", [2])
def test_dcp_value_refusals_are_voted(degree):
    if torch.cuda.device_count() < degree:
        pytest.skip(f"needs {degree} GPUs")
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("DcpValue needs SM100 or SM103")
    mp.spawn(_refusal_worker, args=(degree, _port()), nprocs=degree, join=True)


def test_dcp_value_traps_when_a_peer_never_calls():
    if torch.cuda.device_count() < 2:
        pytest.skip("needs 2 GPUs")
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("DcpValue needs SM100 or SM103")
    mp.spawn(_timeout_worker, args=(2, _port()), nprocs=2, join=True)

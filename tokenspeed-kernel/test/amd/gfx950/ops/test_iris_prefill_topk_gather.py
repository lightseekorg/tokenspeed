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

"""K3.1 DSA prefill selection rows all-gathered through the K3 Iris state."""

import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.platform import current_platform
from utils import is_amd

pytestmark = pytest.mark.skipif(not is_amd(), reason="Iris requires AMD ROCm")


@pytest.fixture(autouse=True)
def _require_iris():
    pytest.importorskip("tokenspeed_kernel.ops.communication._iris")


def _topk_gather_worker(rank: int, port: int) -> None:
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
    from tokenspeed_kernel.ops.communication._iris.kimi_k3 import (
        prepare_k3_communication,
    )

    k3_state = prepare_k3_communication(
        group=group,
        rank_in_group=rank,
        staged_max_numel=0,
        producer_direct_max_numel=8192 * 10752,
        attnres_max_numel=0,
        attnres_max_rows=0,
        enable_lamport=False,
        sharded_max_rows=8192,
        dtype=torch.bfloat16,
        device=device,
        all_reduce=None,
    )
    gen = torch.Generator(device=device)

    def reference(selection, lens):
        rows = selection.shape[0] * 8
        gathered = torch.empty((rows, 2048), dtype=torch.int16, device=device)
        gathered_lens = torch.empty((rows,), dtype=torch.int32, device=device)
        dist.all_gather_into_tensor(
            gathered.view(torch.bfloat16),
            selection.to(torch.int16).view(torch.bfloat16),
            group=group,
        )
        dist.all_gather_into_tensor(gathered_lens, lens, group=group)
        return gathered, gathered_lens

    # Full, partial, tail and minimal chunks; peers must wait for a slow rank's stores.
    for local_rows in (1024, 96, 112, 8, 1024):
        gen.manual_seed(31 * local_rows + rank)
        selection = torch.randint(
            -1,
            0xFFFF,
            (local_rows, 2048),
            dtype=torch.int32,
            device=device,
            generator=gen,
        )
        selection[:, -7:] = -1
        lens = torch.randint(
            0, 2049, (local_rows,), dtype=torch.int32, device=device, generator=gen
        )
        expected = reference(selection, lens)
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 3:
            torch.cuda._sleep(5_000_000)
        result = k3_state.gather_prefill_topk(selection, lens)
        assert result is not None
        torch.testing.assert_close(result[0], expected[0], atol=0, rtol=0)
        torch.testing.assert_close(result[1], expected[1], atol=0, rtol=0)
        # The values above 0x7FFF keep their unsigned 16-bit slots.
        mine = result[0][rank * local_rows : (rank + 1) * local_rows]
        assert torch.equal(mine.to(torch.int32) & 0xFFFF, selection & 0xFFFF)
        torch.cuda.synchronize()
        dist.barrier()

    # Back to back, as DSA layers: a rank may publish while a slow peer still pulls.
    calls = []
    for call in range(40):
        gen.manual_seed(1000 + 31 * call + rank)
        selection = torch.randint(
            -1,
            0xFFFF,
            (1024, 2048),
            dtype=torch.int32,
            device=device,
            generator=gen,
        )
        lens = torch.randint(
            0, 2049, (1024,), dtype=torch.int32, device=device, generator=gen
        )
        calls.append((selection, lens, reference(selection, lens)))
    hog = torch.empty((1 << 28,), dtype=torch.uint8, device=device)
    hog_stream = torch.cuda.Stream(device)
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 3:
        with torch.cuda.stream(hog_stream):
            for _ in range(200):
                hog.copy_(hog.flip(0))
    results = []
    for call, (selection, lens, _) in enumerate(calls):
        if rank == 3 and call % 2 == 0:
            torch.cuda._sleep(200_000)
        results.append(k3_state.gather_prefill_topk(selection, lens))
    torch.cuda.synchronize()
    for result, (_, _, expected) in zip(results, calls):
        torch.testing.assert_close(result[0], expected[0], atol=0, rtol=0)
        torch.testing.assert_close(result[1], expected[1], atol=0, rtol=0)
    dist.barrier()

    selection = torch.zeros((16, 2048), dtype=torch.int32, device=device)
    lens = torch.zeros((16,), dtype=torch.int32, device=device)
    assert k3_state.gather_prefill_topk(selection[:12], lens[:12]) is None
    assert k3_state.gather_prefill_topk(selection[:, :1024].contiguous(), lens) is None
    assert k3_state.gather_prefill_topk(selection.to(torch.int16), lens) is None
    assert k3_state.gather_prefill_topk(selection, lens[:8]) is None
    too_many = torch.zeros((1032, 2048), dtype=torch.int32, device=device)
    assert (
        k3_state.gather_prefill_topk(
            too_many, torch.zeros((1032,), dtype=torch.int32, device=device)
        )
        is None
    )
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(
    not current_platform().is_cdna4 or torch.cuda.device_count() < 8,
    reason="requires eight CDNA4 GPUs",
)
def test_prefill_topk_rows_gather_matches_the_nccl_gather():
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_topk_gather_worker, args=(port,), nprocs=8, join=True)

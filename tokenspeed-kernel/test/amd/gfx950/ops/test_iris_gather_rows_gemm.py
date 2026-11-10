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

"""The KDA input projection that gathers its row-sharded input inside the GEMM."""

import os
import socket
from contextlib import nullcontext
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.platform import current_platform
from utils import assert_no_triton_compile, is_amd

pytestmark = pytest.mark.skipif(not is_amd(), reason="Iris requires AMD ROCm")


@pytest.fixture(autouse=True)
def _require_iris():
    pytest.importorskip("tokenspeed_kernel.ops.communication._iris")


def _gather_rows_worker(rank: int, port: int) -> None:
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
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.largem import (
        gluon_mm_a16w16_prefill_gather_rows_gfx950,
        launch_gluon_mm_a16w16_prefill_gfx950,
    )

    os.environ["TOKENSPEED_K31_GATHER_ROWS_GEMM"] = "1"
    k3_state = prepare_k3_communication(
        group=group,
        rank_in_group=rank,
        staged_max_numel=0,
        producer_direct_max_numel=8192 * 10752,
        attnres_max_numel=0,
        attnres_max_rows=0,
        enable_lamport=True,
        sharded_max_rows=8192,
        dtype=torch.bfloat16,
        device=device,
        all_reduce=None,
    )
    sharded = k3_state.sharded
    generator = torch.Generator(device=device).manual_seed(4127)
    # The KDA fused q/k/v/gate/f_a/beta projection.
    weight = (torch.randn((6288, 7168), device=device, generator=generator) / 85).to(
        torch.bfloat16
    )
    fence = torch.zeros(1, device=device)
    for step, rows in enumerate((8192, 2048, 6144, 4096, 8192, 2048)):
        local_rows = rows // 8
        generator.manual_seed(611 + 8 * step + rank)
        previous, mixed = (
            torch.randn((local_rows, 7168), device=device, generator=generator).to(
                torch.bfloat16
            )
            for _ in range(2)
        )
        gathered = torch.empty((rows, 7168), device=device, dtype=torch.bfloat16)
        dist.all_gather_into_tensor(gathered, mixed, group=group)
        expected = launch_gluon_mm_a16w16_prefill_gfx950(
            gathered, weight, torch.bfloat16
        )
        output = sharded.output[:rows]
        # The first step compiles; later row counts reuse that binary.
        no_compile = (
            assert_no_triton_compile(gluon_mm_a16w16_prefill_gather_rows_gfx950)
            if step
            else nullcontext()
        )
        with no_compile:
            # Cache the buffer's previous rows, then delay one rank's call.
            assert sharded._gather_rows_project(previous, output, weight) is not None
            dist.all_reduce(fence, group=group)
            if rank == step % 8:
                torch.cuda._sleep(200_000)
            result = sharded._gather_rows_project(mixed, output, weight)
        assert result is not None and result.shape == (rows, 6288)
        dist.all_reduce(fence, group=group)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            result.view(torch.int16), expected.view(torch.int16), atol=0, rtol=0
        )
        torch.testing.assert_close(
            output.view(torch.int16), gathered.view(torch.int16), atol=0, rtol=0
        )

    # Unsupported shapes launch nothing and leave the expected counts alone.
    counts = list(sharded._gather_row_counts)
    ticket_base = sharded._gather_ticket_base
    assert sharded.gather_row_tickets.item() == ticket_base
    for mixed, rows in (
        (torch.zeros((100, 7168), device=device, dtype=torch.bfloat16), 800),
        (torch.zeros((1280, 7168), device=device, dtype=torch.bfloat16), 10240),
        (torch.zeros((256, 7168), device=device, dtype=torch.float16), 2048),
    ):
        assert (
            sharded._gather_rows_project(mixed, sharded.output[:rows], weight) is None
        )
    assert sharded._gather_row_counts == counts
    assert sharded._gather_ticket_base == ticket_base
    torch.cuda.synchronize()
    with torch.cuda.graph(torch.cuda.CUDAGraph()):
        captured = torch.zeros((256, 7168), device=device, dtype=torch.bfloat16)
        assert (
            sharded._gather_rows_project(captured, sharded.output[:2048], weight)
            is None
        )
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(
    not current_platform().is_cdna4 or torch.cuda.device_count() < 8,
    reason="requires eight CDNA4 GPUs",
)
def test_gather_rows_projection_matches_gather_then_gemm():
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_gather_rows_worker, args=(port,), nprocs=8, join=True)

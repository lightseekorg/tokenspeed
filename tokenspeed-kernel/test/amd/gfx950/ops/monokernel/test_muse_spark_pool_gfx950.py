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

"""Muse Spark pooler (pre-expert projection) across the 8 ranks."""

from __future__ import annotations

import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from monokernel_testing import fill, fill_words, payloads
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel.ops.communication._iris import (  # noqa: E402
    IrisGroup,
    get_or_create_context,
)
from tokenspeed_kernel_amd._triton import gl, gluon  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel import mailbox, schedule  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import (  # noqa: E402
    moe_pool,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import (  # noqa: E402
    Plan,
    launch_tag,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.weights import (  # noqa: E402
    bf16_tiles,
)

WORLD, HIDDEN, POOLED, ROWS = 8, 8192, 4096, 512
FLAGS = WORLD * 8  # the attention tail's flags a row: [rank][task]


@gluon.jit
def pool_kernel(scratch, step, bases, w, PLAN: gl.constexpr, RANK: gl.constexpr):
    peers = mailbox.load_peers(bases, RANK, gl.load(step) & 1, 8)
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, None, peers)
    task, active = schedule.stage_task(PLAN, "pool")
    if active:
        moe_pool.pool_stage(mb, task, w, RANK, "pool")


def _worker(rank: int, port: int, rows: int) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=WORLD,
        timeout=timedelta(seconds=300),
    )
    try:
        device = torch.device(f"cuda:{rank}")
        ctx = get_or_create_context(heap_size=256 << 20)
        group = IrisGroup(ctx, dist.group.WORLD, rank, device)
        plan = Plan(
            grid=256,
            stages=moe_pool.stages({"pool": 0}),
            regions=(("xin", HIDDEN // 4), ("xflag", FLAGS)) + moe_pool.regions(),
            rows=rows,
            chain=rows,
            external=("xin", "xflag"),
            world=WORLD,
            peers=("xin", "xflag") + moe_pool.PEERS,
        )
        scratch = ctx.zeros((plan.scratch_pairs,), dtype=torch.int64)
        group.finish_prepare()
        for step in (6, 7):
            g = torch.Generator().manual_seed(step)
            x = torch.randn(rows, HIDDEN, generator=g).to(torch.bfloat16)
            w = (torch.randn(POOLED, HIDDEN, generator=g) / HIDDEN**0.5).to(
                torch.bfloat16
            )
            tag = launch_tag(step, 0)
            # The attention tail's words, then its flags.
            for row in range(rows):
                fill_words(
                    scratch,
                    plan,
                    "xin",
                    x[row].view(torch.int32).to(device),
                    copy=step & 1,
                    row=row,
                )
                fill(
                    scratch,
                    plan,
                    "xflag",
                    torch.ones(FLAGS, dtype=torch.int32, device=device),
                    tag,
                    copy=step & 1,
                    row=row,
                )
            pool_kernel[(plan.grid,)](
                scratch,
                torch.full((1,), step, dtype=torch.int32, device=device),
                group.bases,
                bf16_tiles(w[rank * ROWS : (rank + 1) * ROWS].contiguous()).to(device),
                plan.key,
                rank,
                num_warps=8,
            )
            torch.cuda.synchronize()
            dist.barrier()
            want = (x.double() @ w.double().T).to(torch.bfloat16)
            for row in range(rows):
                got = payloads(
                    scratch, plan, "cg", POOLED // 2, tag, row=row, copy=step & 1
                )
                torch.testing.assert_close(
                    got.view(torch.bfloat16).float().cpu(),
                    want[row].float(),
                    rtol=2**-7,
                    atol=1e-3,
                )
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("rows", [1, 8])
def test_pooler_gathers_every_ranks_columns(rows: int):
    if torch.cuda.device_count() < WORLD:
        pytest.skip(f"requires {WORLD} GPUs")
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port, rows), nprocs=WORLD, join=True)

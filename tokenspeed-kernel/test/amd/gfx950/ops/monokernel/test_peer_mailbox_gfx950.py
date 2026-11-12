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

"""Peer mailbox regions across the ranks of a tensor-parallel group.

Every rank runs the same two-stage plan in one persistent launch:

* ``push``: task ``t`` puts its slice of this rank's partial into every
  rank's copy of peer region ``part`` (``Mailbox.put_peer``).
* ``sum``: task ``t`` polls the slices of all ranks from its own copy and
  adds them in rank order, so every rank gets the same bits.

Launches follow each other without a host sync, with new inputs each time, so
a rank can start its next launch while a slower rank still reads this one: the
two copies of the peer region (by launch parity) must keep them apart.
"""

from __future__ import annotations

import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel.ops.communication._iris import (  # noqa: E402
    IrisGroup,
    get_or_create_context,
)
from tokenspeed_kernel_amd._triton import gl, gluon  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel import mailbox, schedule  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import Plan, Stage  # noqa: E402

WORLD = 8
N = gl.constexpr(8192)  # values each rank contributes
SLICE = gl.constexpr(64)  # values per task
TASKS = int(N) // int(SLICE)
LAUNCHES = 6


def peer_plan() -> Plan:
    # ``sum`` sits on other CTAs than ``push``, so it polls while peers push.
    return Plan(
        grid=2 * TASKS,
        stages=(
            Stage("push", TASKS, 0, writes=("part",)),
            Stage("sum", TASKS, TASKS, reads=("part",)),
        ),
        regions=(("part", WORLD * int(N)),),
        rows=1,
        chain=1,
        world=WORLD,
        peers=("part",),
    )


@gluon.jit
def peer_sum_kernel(
    scratch, step, bases, x, out, PLAN: gl.constexpr, RANK: gl.constexpr
):
    W: gl.constexpr = schedule.world(PLAN)
    peers = mailbox.load_peers(bases, RANK, gl.load(step) & 1, W)
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, None, peers)
    lanes: gl.constexpr = mailbox.lane_layout(gl.num_warps())
    task, active = schedule.stage_task(PLAN, "push")
    if active:
        e = task * SLICE + gl.arange(0, SLICE, lanes)
        value = gl.load(x + e)
        for p in gl.static_range(W):
            mb.put_peer("part", p, RANK * N + e, value)
    task, active = schedule.stage_task(PLAN, "sum")
    if active:
        e = task * SLICE + gl.arange(0, SLICE, lanes)
        indices = ()
        regions: gl.constexpr = ("part",) * W
        for p in gl.static_range(W):
            indices += (p * N + e,)
        parts = mb.poll_many(regions, indices)
        total = parts[0].to(gl.float32, bitcast=True)
        for p in gl.static_range(1, W):
            total += parts[p].to(gl.float32, bitcast=True)
        gl.store(out + e, total)


def _inputs(rank: int, launch: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(1000 * launch + rank)
    return torch.randn(int(N), generator=g)


def _worker(rank: int, port: int) -> None:
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
        plan = peer_plan()
        # Nothing clears the buffer: each launch has a new tag.
        scratch = ctx.zeros((plan.scratch_pairs,), dtype=torch.int64)
        xs = [_inputs(rank, k).to(device) for k in range(LAUNCHES)]
        outs = [torch.empty(int(N), device=device) for _ in range(LAUNCHES)]
        steps = [
            torch.full((1,), k + 1, dtype=torch.int32, device=device)
            for k in range(LAUNCHES)
        ]
        group.finish_prepare()
        for k in range(LAUNCHES):
            peer_sum_kernel[(plan.grid,)](
                scratch,
                steps[k],
                group.bases,
                xs[k],
                outs[k],
                plan.key,
                rank,
                num_warps=1,
            )
        torch.cuda.synchronize()
        for k in range(LAUNCHES):
            want = _inputs(0, k)
            for r in range(1, WORLD):
                want = want + _inputs(r, k)
            assert torch.equal(outs[k].cpu(), want), f"launch {k}"
    finally:
        dist.destroy_process_group()


def test_peer_regions_sum_every_ranks_slices_across_launches():
    if torch.cuda.device_count() < WORLD:
        pytest.skip(f"requires {WORLD} GPUs")
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port,), nprocs=WORLD, join=True)

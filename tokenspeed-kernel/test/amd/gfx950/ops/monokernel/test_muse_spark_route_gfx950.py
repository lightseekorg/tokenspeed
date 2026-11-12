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

"""Muse Spark expert routing across the 8 ranks (column-sharded router)."""

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

from monokernel_testing import fill, payloads  # noqa: E402
from tokenspeed_kernel.ops.communication._iris import (  # noqa: E402
    IrisGroup,
    get_or_create_context,
)
from tokenspeed_kernel_amd._triton import gl, gluon  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel import mailbox, schedule  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import (  # noqa: E402
    moe_route,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import (  # noqa: E402
    Plan,
    launch_tag,
)

WORLD, HIDDEN, OWNED, EXPERTS, TOP_K, TAIL_TASKS = 8, 8192, 1024, 256, 8, 8
SUMS = 4


def _tail_sums(b, h, alpha, beta) -> torch.Tensor:
    """The attention tail's four partial sums (``b^2``, ``(beta b)^2``,
    ``(alpha h)(beta b)``, ``(alpha h)^2``), FP32 ``[rank * task, sum]``."""
    bb, ah = beta.double() * b.double(), alpha.double() * h.double()
    parts = torch.stack([b.double().pow(2), bb * bb, ah * bb, ah * ah], dim=1)
    return parts.view(WORLD * TAIL_TASKS, -1, SUMS).sum(dim=1).float().contiguous()


@gluon.jit
def route_kernel(
    scratch,
    step,
    bases,
    w,
    pre_gain,
    h,
    alpha,
    beta,
    bias,
    PLAN: gl.constexpr,
    RANK: gl.constexpr,
):
    peers = mailbox.load_peers(bases, RANK, gl.load(step) & 1, 8)
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, None, peers)
    task, active = schedule.stage_task(PLAN, "router")
    if active:
        moe_route.router_stage(mb, task, w, pre_gain, h, alpha, beta, RANK, "router")
    task, active = schedule.stage_task(PLAN, "topk")
    if active:
        moe_route.topk_stage(mb, task, bias, "topk")


def _inputs(seed: int, rows: int):
    """The attention tail's ``b`` (BF16 values) and ``h`` of each row
    (``[rows, 8192]``), ``alpha``, ``beta``, the residual ``h' = beta b inv +
    alpha h`` they give, and the router's weights."""
    g = torch.Generator().manual_seed(seed)
    b = torch.randn(rows, HIDDEN, generator=g).to(torch.bfloat16).float()
    h = torch.randn(rows, HIDDEN, generator=g) * 3.0
    alpha = torch.rand(HIDDEN, generator=g) + 0.5
    beta = torch.rand(HIDDEN, generator=g) + 0.5
    inv = torch.rsqrt(b.double().pow(2).mean(-1, keepdim=True) + 1e-8)
    hp = beta.double() * b.double() * inv + alpha.double() * h.double()
    pre_gain = (torch.randn(HIDDEN, generator=g) * 0.1 + 1.0).to(torch.bfloat16).float()
    w = torch.randn(EXPERTS, HIDDEN, generator=g) / HIDDEN**0.5
    bias = torch.randn(EXPERTS, generator=g) * 0.1
    if seed == 3:
        # An exact tie at the top: experts 5 and 9 have the same weights and
        # bias, so the same score + bias. The lower id must come first.
        w[9] = w[5]
        bias[5] = bias[9] = 2.0
    return (b, h, alpha, beta), hp, pre_gain, w, bias


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
            stages=moe_route.stages({"router": 0, "topk": 16}, rows),
            regions=(("red", WORLD * OWNED // 2), ("ss1", WORLD * TAIL_TASKS * SUMS))
            + moe_route.regions(),
            rows=rows,
            chain=rows,
            external=("red", "ss1"),
            world=WORLD,
            peers=moe_route.PEERS,
        )
        scratch = ctx.zeros((plan.scratch_pairs,), dtype=torch.int64)
        group.finish_prepare()
        own = slice(rank * OWNED, (rank + 1) * OWNED)
        for k, seed in enumerate((1, 2, 3)):
            step = 4 + k
            (b, h, alpha, beta), hp, pre_gain, w, bias = _inputs(seed, rows)
            tag = launch_tag(step, 0)
            for row in range(rows):
                # The ranks' o_proj partials that sum to b: rank 1's holds
                # b - bf16 of half of it, rank 0's that half, the rest zero
                # (the router sums them in rank order and rounds to BF16).
                half = (b[row, own] / 2).to(torch.bfloat16)
                rest = (b[row, own] - half.float()).to(torch.bfloat16)
                assert torch.equal((half.float() + rest.float()), b[row, own])
                parts = torch.zeros(WORLD, OWNED, dtype=torch.bfloat16)
                parts[0], parts[1] = half, rest
                fill(
                    scratch,
                    plan,
                    "red",
                    parts.view(torch.int32).to(device),
                    tag,
                    row=row,
                )
                # The attention tail's partial sums: [rank][task][sum].
                sums = _tail_sums(b[row], h[row], alpha, beta)
                fill(
                    scratch,
                    plan,
                    "ss1",
                    sums.view(torch.int32).to(device),
                    tag,
                    row=row,
                )
            route_kernel[(plan.grid,)](
                scratch,
                torch.full((1,), step, dtype=torch.int32, device=device),
                group.bases,
                w[:, own].contiguous().to(device),
                pre_gain[own].to(device),
                h[:, own].contiguous().to(device),
                alpha[own].to(device),
                beta[own].to(device),
                bias.to(device),
                plan.key,
                rank,
                num_warps=8,
            )
            torch.cuda.synchronize()
            for row in range(rows):
                got = payloads(scratch, plan, "route", 2 * TOP_K, tag, row=row).cpu()
                u = hp[row] * torch.rsqrt(hp[row].pow(2).mean() + 1e-5) * pre_gain
                logit = (w.double() @ u) * 0.17677669529663687
                score = torch.sigmoid(logit)
                # Largest first, a tie to the lowest id.
                ids = torch.sort(
                    score + bias.double(), descending=True, stable=True
                ).indices[:TOP_K]
                weights = score[ids] / score[ids].sum()
                assert torch.equal(got[:TOP_K].long(), ids), row
                if seed == 3:
                    assert ids[:2].tolist() == [5, 9]
                torch.testing.assert_close(
                    got[TOP_K:].view(torch.float32).double(),
                    weights,
                    rtol=1e-5,
                    atol=1e-6,
                )
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("rows", [1, 8])
def test_routing_sums_the_ranks_partial_logits_and_picks_the_top_8(rows: int):
    if torch.cuda.device_count() < WORLD:
        pytest.skip(f"requires {WORLD} GPUs")
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port, rows), nprocs=WORLD, join=True)

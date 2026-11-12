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

"""Muse Spark MXFP4 experts across the 8 ranks, on the regular path's
preprocessed expert weights."""

from __future__ import annotations

import socket
from datetime import timedelta

import muse_spark_reference as ref
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.benchmark import muse_spark_monokernel_gluon as gluon_inputs
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

import muse_spark_standins as standins  # noqa: E402
from monokernel_testing import fill, payloads  # noqa: E402
from tokenspeed_kernel.ops.communication._iris import (  # noqa: E402
    IrisGroup,
    get_or_create_context,
)
from tokenspeed_kernel_amd._extern import extern_compile_options  # noqa: E402
from tokenspeed_kernel_amd._triton import gl, gluon  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel import mailbox, schedule  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import (  # noqa: E402
    moe_experts,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import (  # noqa: E402
    Plan,
    launch_tag,
)

# 128 experts, as many as 16 rows can pick. Random picks come from the first
# 48: 8 rows share about 37 of them, as many as a verify's 8 consecutive
# tokens, so the slot lanes run several slots. Picks of a set number of
# distinct experts (_picks) cover the slot counts a lane can have past 8
# rows: up to 64 experts (8 slots a lane, no second ex2 batch), 65 (one lane
# with 9), and up to 128 (16).
WORLD, POOLED, INTER, EXPERTS, TOP_K, OWNED = 8, 4096, 512, 128, 8, 512
RANDOM_EXPERTS = 48
ALL = tuple(range(WORLD))


@gluon.jit
def experts_kernel(
    scratch,
    step,
    bases,
    w13,
    w13s,
    w2,
    w2s,
    s13_we,
    s13_se,
    s13_lin,
    s13_nb,
    s2_we,
    s2_se,
    s2_lin,
    s2_nb,
    pre_gain,
    PLAN: gl.constexpr,
    RANK: gl.constexpr,
):
    peers = mailbox.load_peers(bases, RANK, gl.load(step) & 1, 8)
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, None, peers)
    # A CTA's ex1 and ex2 tasks run in parts (see moe_experts): both weight
    # streams once routing names the experts, then the A operand (xq).
    task, active = schedule.stage_task(PLAN, "slt")
    if active:
        moe_experts.slot_table_stage(mb, "slt")
    task, active = schedule.stage_task(PLAN, "xq")
    if active:
        moe_experts.xq_stage(mb, task, pre_gain, "xq")
    task1, active1 = schedule.stage_task(PLAN, "ex1")
    task1 = gl.where(active1, task1, 0)
    task2, active2 = schedule.stage_task(PLAN, "ex2")
    task2 = gl.where(active2, task2, 0)
    ids, e1, e2 = moe_experts.ex1_poll_routes(mb, task1, task2, active1, "ex1")
    loads1 = moe_experts.ex1_issue(
        task1, active1, e1, w13, w13s, s13_we, s13_se, s13_lin, s13_nb
    )
    loads2 = moe_experts.ex2_issue(
        mb, task2, active2, e2, w2, w2s, s2_we, s2_se, s2_lin, s2_nb, "ex2"
    )
    union = moe_experts.union_of(ids)
    table = moe_experts.cta_slots(PLAN, union)
    pre1 = moe_experts.ex1_prefetch(PLAN, task1, active1, table, union, w13, s13_we)
    a1 = moe_experts.ex1_xq(mb, task1, active1, "ex1")
    if active1:
        moe_experts.ex1_finish(
            mb,
            task1,
            a1,
            loads1,
            e1,
            union,
            table,
            pre1,
            w13,
            w13s,
            s13_we,
            s13_se,
            s13_lin,
            s13_nb,
            "ex1",
        )
    task, active = schedule.stage_task(PLAN, "exr")
    if active:
        moe_experts.exr_stage(mb, task, union, table, "exr")
    if active2:
        moe_experts.ex2_finish(
            mb,
            task2,
            loads2,
            e2,
            union,
            table,
            w2,
            w2s,
            s2_we,
            s2_se,
            s2_lin,
            s2_nb,
            RANK,
            "ex2",
        )


def _raw_experts(rank: int, device: torch.device, present):
    """Random MXFP4 expert shards of ``rank``, made on ``device`` (the same on
    every GPU; on the host they would take seconds a rank): w13 rows
    interleave gate / up. A rank outside ``present`` has zero shards (E2M1
    code 0 is +0 whatever the scale), so it puts exactly zero partials."""
    g = torch.Generator(device).manual_seed(500 + rank)
    kw = dict(dtype=torch.uint8, generator=g, device=device)
    w13 = torch.randint(0, 256, (EXPERTS, 2 * INTER, POOLED // 2), **kw)
    s13 = torch.randint(122, 127, (EXPERTS, 2 * INTER, POOLED // 32), **kw)
    w2 = torch.randint(0, 256, (EXPERTS, POOLED, INTER // 2), **kw)
    s2 = torch.randint(118, 123, (EXPERTS, POOLED, INTER // 32), **kw)
    if rank not in present:
        w13, w2 = torch.zeros_like(w13), torch.zeros_like(w2)
    return w13, s13, w2, s2


def _preprocessed(rank: int, device: torch.device, present):
    return gluon_inputs.preprocess_experts(_raw_experts(rank, device, present))


def _inputs(launch: int, rows: int, distinct: int | None):
    """Each row's pooled input and 8 picks (rows ``[rows, ...]``), and the
    pre-expert gain."""
    g = torch.Generator().manual_seed(77 + launch)
    pooled = torch.randn(rows, POOLED, generator=g).to(torch.bfloat16)
    gain = (1.0 + 0.1 * torch.randn(POOLED, generator=g)).to(torch.bfloat16).float()
    return pooled, gain, _picks(g, rows, distinct)


def _picks(g: torch.Generator, rows: int, distinct: int | None) -> torch.Tensor:
    """Each row's 8 picks, ``int32 [rows, 8]``. ``distinct`` None: random
    picks among the first ``RANDOM_EXPERTS`` experts. Otherwise exactly
    ``distinct`` experts over the launch (``distinct`` slots, lane ``l``
    ``ceil((distinct - l) / 8)`` of them): random distinct experts, dealt to
    the rows' picks in turn so each row's 8 differ and every one is picked."""
    if distinct is None:
        return torch.stack(
            [torch.randperm(RANDOM_EXPERTS, generator=g)[:TOP_K] for _ in range(rows)]
        ).to(torch.int32)
    assert TOP_K <= distinct <= min(rows * TOP_K, EXPERTS)
    chosen = torch.randperm(EXPERTS, generator=g)[:distinct]
    deal = torch.arange(rows * TOP_K).view(rows, TOP_K) % distinct
    order = torch.argsort(torch.rand(rows, TOP_K, generator=g), dim=1)
    return chosen[deal.gather(1, order)].to(torch.int32)


def _worker(
    me: int,
    port: int,
    rows: int,
    exr: int,
    exr_parts: int,
    distinct: int | None,
    present,
) -> None:
    """``present``: None for the whole TP8 group, one process a rank; or the
    two rank programs that run (process ``me`` runs ``present[me]``) with
    the other ranks' expert shards zero, whose partials (exactly zero) the
    host writes before each launch."""
    torch.cuda.set_device(me)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://localhost:{port}",
        rank=me,
        world_size=WORLD if present is None else len(present),
        timeout=timedelta(seconds=600),
    )
    try:
        device = torch.device(f"cuda:{me}")
        ctx = get_or_create_context(heap_size=256 << 20)
        group = IrisGroup(ctx, dist.group.WORLD, me, device)
        plan = Plan(
            grid=256,
            # The layer's placements of exr (layer.BASES, CHAINED_BASES and
            # NEXT_BASES).
            stages=moe_experts.stages(
                {"slt": 255, "xq": 240, "ex1": 0, "exr": exr, "ex2": 0},
                rows,
                exr_parts,
            ),
            regions=(("route", 2 * TOP_K), ("cg", POOLED // 2)) + moe_experts.regions(),
            rows=rows,
            chain=rows,
            external=("route", "cg"),
            world=WORLD,
            peers=("cg",) + moe_experts.PEERS,
        )
        scratch = ctx.zeros((plan.scratch_pairs,), dtype=torch.int64)
        if present is None:
            rank, model, bases = me, ALL, group.bases
        else:
            rank, model = present[me], present
            ghost = torch.zeros_like(scratch)
            bases = standins.peer_bases(group.bases, present, me, scratch, ghost)
        w13, s13, w2, s2 = _preprocessed(rank, device, model)
        group.finish_prepare()
        own = slice(rank * OWNED, (rank + 1) * OWNED)
        for launch in range(2):
            step = 10 + launch
            tag = launch_tag(step, 0)
            pooled, gain, ids = _inputs(launch, rows, distinct)
            if distinct is not None:
                assert ids.unique().numel() == distinct
            for row in range(rows):
                route = torch.cat(
                    [ids[row], torch.full((TOP_K,), 0.125).view(torch.int32)]
                )
                fill(scratch, plan, "route", route.to(device), tag, row=row)
                fill(
                    scratch,
                    plan,
                    "cg",
                    pooled[row].view(torch.int32).to(device),
                    tag,
                    copy=step & 1,
                    row=row,
                )
            if present is not None:
                standins.standin_fill(
                    scratch,
                    plan,
                    present,
                    [(0, t) for t in range(rows)],
                    step,
                    0,
                    step & 1,
                    True,
                )
            experts_kernel[(plan.grid,)](
                scratch,
                torch.full((1,), step, dtype=torch.int32, device=device),
                bases,
                w13,
                s13,
                w2,
                s2,
                w13.stride(0),
                s13.stride(0),
                s13.stride(1),
                s13.stride(2),
                w2.stride(0),
                s2.stride(0),
                s2.stride(1),
                s2.stride(2),
                gain.to(device),
                plan.key,
                rank,
                num_warps=8,
                **extern_compile_options(),
            )
            torch.cuda.synchronize()
            dist.barrier()
            # Each source rank's partials in FP64 and a bound on the kernel's
            # (its two MXFP4 quantizations can step a code where the FP32 error
            # reaches a rounding boundary: ref.expert_partials_bounded).
            want = [
                tuple(
                    t.cpu()
                    for t in ref.expert_partials_bounded(
                        _raw_experts(src, device, model),
                        pooled.to(device),
                        gain.to(device),
                        ids.long().to(device),
                        own,
                    )
                )
                for src in range(WORLD)
            ]
            for row in range(rows):
                got = payloads(
                    scratch,
                    plan,
                    "rr",
                    WORLD * TOP_K * OWNED // 2,
                    tag,
                    row=row,
                    copy=step & 1,
                )
                got = got.view(torch.bfloat16).float().view(WORLD, TOP_K, OWNED).cpu()
                for src in range(WORLD):
                    exact, bound = (t[row] for t in want[src])
                    err = (got[src].double() - exact).abs()
                    assert torch.all(err <= bound), float((err / bound).max())
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "rows, exr, exr_parts, distinct",
    [
        (1, 192, 1, None),
        (8, 0, 4, None),
        (8, 128, 2, None),
        (8, 0, 4, 64),
        (16, 0, 4, None),
        (16, 128, 2, None),
        # Past 8 rows, by the lanes' slot counts: 1 each (the next slot's
        # expert is -1 from the first), 1 or 2, 8 each (no second ex2 batch),
        # 9 on lane 0 only, 9 each, 12 or 13, 16 each.
        (16, 0, 4, 8),
        (16, 0, 4, 13),
        (16, 0, 4, 64),
        (16, 0, 4, 65),
        (16, 128, 2, 65),
        (16, 0, 4, 72),
        (16, 0, 4, 100),
        (16, 128, 2, 128),
    ],
)
def test_experts_match_the_regular_path_math_on_its_weight_layout(
    rows: int, exr: int, exr_parts: int, distinct: int | None
):
    if torch.cuda.device_count() < WORLD:
        pytest.skip(f"requires {WORLD} GPUs")
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(
        _worker,
        args=(port, rows, exr, exr_parts, distinct, None),
        nprocs=WORLD,
        join=True,
    )


@pytest.mark.parametrize("present", standins.PAIRS, ids=standins.PAIR_IDS)
@pytest.mark.parametrize(
    "rows, exr, exr_parts, distinct",
    [(1, 192, 1, None), (8, 0, 4, None), (16, 0, 4, None), (16, 128, 2, 128)],
)
def test_experts_match_the_regular_path_math_on_two_gpus(
    rows: int, exr: int, exr_parts: int, distinct: int | None, present
):
    """As :func:`test_experts_match_the_regular_path_math_on_its_weight_layout`
    with the production rank programs ``present`` on two GPUs and the other
    six ranks' expert shards zero (the host writes their zero partials): the
    present ranks' partials, as received from both, within the reference's
    bound, and the absent ranks' exactly zero. It cannot check the values
    the present ranks put into the absent ranks' segments (see
    muse_spark_standins)."""
    if torch.cuda.device_count() < 2:
        pytest.skip("requires 2 GPUs")
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(
        _worker,
        args=(port, rows, exr, exr_parts, distinct, present),
        nprocs=2,
        join=True,
    )

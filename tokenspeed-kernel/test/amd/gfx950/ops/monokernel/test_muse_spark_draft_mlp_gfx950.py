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

"""The DFlash2 draft's MLP half (Muse Spark) across the 8 ranks."""

from __future__ import annotations

import socket
from datetime import timedelta

import dflash2_draft_reference as ref
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.benchmark import muse_spark_monokernel as inputs
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

import muse_spark_standins as standins  # noqa: E402
from monokernel_testing import payloads  # noqa: E402
from tokenspeed_kernel.ops.communication._iris import (  # noqa: E402
    IrisGroup,
    get_or_create_context,
)
from tokenspeed_kernel_amd._triton import gl, gluon  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel import (  # noqa: E402
    mailbox,
    schedule,
    trace,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import (  # noqa: E402
    draft_mlp,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import (  # noqa: E402
    Plan,
    launch_tag,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.weights import (  # noqa: E402
    bf16_tiles,
)

WORLD, HIDDEN, OWNED = 8, ref.HIDDEN, 1024
LAUNCHES = 2
ALL = tuple(range(WORLD))


def _shared_inputs(launch: int, rows: int, present):
    """The half's replicated inputs (``inputs.draft_mlp_shared``) of a
    zero-shard model for ``present``: the partials of the ranks outside it
    are zero, and so are every partial's and the residual's columns those
    ranks own, so they put exactly zero (red, ss and hin)."""
    parts, residual, gain, coef_w, base = inputs.draft_mlp_shared(launch, rows)
    absent = standins.absent_columns(HIDDEN, OWNED, present)
    parts = [p if r in present else torch.zeros_like(p) for r, p in enumerate(parts)]
    for p in parts:
        p[:, absent] = 0
    residual[:, absent] = 0
    return parts, residual, gain, coef_w, base


@gluon.jit
def draft_mlp_kernel(
    scratch,
    step,
    bases,
    z,
    residual,
    residual_out,
    gain,
    coef_w,
    base,
    gu_w,
    down_w,
    out,
    trace_buf,
    PLAN: gl.constexpr,
    RANK: gl.constexpr,
):
    table = trace.allocate(trace_buf)
    peers = mailbox.load_peers(bases, RANK, gl.load(step) & 1, 8)
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, table, peers)
    G: gl.constexpr = schedule.grid(PLAN)
    task, active = schedule.stage_task(PLAN, "scat")
    if active:
        draft_mlp.scat_stage(mb, task, z, RANK, "red", "scat")
    task, active = schedule.stage_task(PLAN, "sum1")
    if active:
        draft_mlp.sum_norm_stages(
            mb,
            task,
            residual,
            residual_out,
            gain,
            RANK,
            "red",
            "ss",
            "hin",
            "sum1",
            "norm",
            True,
        )
    task, active = schedule.stage_task(PLAN, "coef")
    if active:
        draft_mlp.proj_task(
            mb,
            task,
            coef_w,
            "hin",
            "cpart",
            draft_mlp.COEFS,
            draft_mlp.HIDDEN,
            draft_mlp.COEF_ROWS,
            draft_mlp.COEF_KS,
            "coef",
        )
    # The CTAs that host no conv0 task start their first gate/up task (its
    # weight loads) right after coef. vmcnt counts loads in order, so on the
    # CTAs that do host one, conv0's polls run first, with no weight loads in
    # flight.
    gu_task, rounds = schedule.stage_rounds(PLAN, "gu")
    _, has_conv0 = schedule.stage_task(PLAN, "conv0")
    free = not has_conv0
    if free:
        draft_mlp.proj_task(
            mb,
            gu_task,
            gu_w,
            "hc",
            "gpart",
            draft_mlp.GU,
            draft_mlp.HIDDEN,
            draft_mlp.GU_ROWS,
            draft_mlp.GU_KS,
            "gu",
        )
    task, active = schedule.stage_task(PLAN, "conv0")
    if active:
        draft_mlp.conv0_stage(mb, task, base, "hin", "cpart", "hc", "conv0")
    for i in range(gl.where(free, 1, 0), rounds):
        draft_mlp.proj_task(
            mb,
            gu_task + i * G,
            gu_w,
            "hc",
            "gpart",
            draft_mlp.GU,
            draft_mlp.HIDDEN,
            draft_mlp.GU_ROWS,
            draft_mlp.GU_KS,
            "gu",
        )
    # An act host issues its first down task's weight loads before act: act
    # waits for the last gate/up tasks anyway, and the down weights stream
    # meanwhile. (vmcnt counts in order: act's poll returns after them.)
    d_task, d_rounds = schedule.stage_rounds(PLAN, "down")
    a_task, has_act = schedule.stage_task(PLAN, "act")
    H: gl.constexpr = draft_mlp.HIDDEN
    I: gl.constexpr = draft_mlp.INTER
    if has_act:
        tiles = draft_mlp.proj_issue(
            d_task, down_w, H, I, draft_mlp.DOWN_ROWS, draft_mlp.DOWN_KS
        )
        draft_mlp.act_stage(mb, a_task, "act")
        x = draft_mlp.proj_poll(mb, d_task, "act", I, draft_mlp.DOWN_KS, "down")
        draft_mlp.proj_compute(
            mb,
            d_task,
            tiles,
            x,
            "dpart",
            H,
            I,
            draft_mlp.DOWN_ROWS,
            draft_mlp.DOWN_KS,
            "down",
        )
    for i in range(gl.where(has_act, 1, 0), d_rounds):
        draft_mlp.proj_task(
            mb,
            d_task + i * G,
            down_w,
            "act",
            "dpart",
            H,
            I,
            draft_mlp.DOWN_ROWS,
            draft_mlp.DOWN_KS,
            "down",
        )
    task, active = schedule.stage_task(PLAN, "fin")
    if active:
        draft_mlp.fin_stage(
            mb,
            task,
            base,
            out,
            "dpart",
            draft_mlp.DOWN_KPARTS,
            "cpart",
            draft_mlp.FIN_COLS,
            "fin",
        )
    trace.flush(table, trace_buf)


BASES = {
    "scat": 0,
    "sum": 128,
    "coef": 0,
    "conv0": 128,
    "gu": 0,
    "act": 144,
    "down": 0,
    "fin": 0,
}


def _plan(rows: int) -> Plan:
    return Plan(
        grid=256,
        stages=draft_mlp.stages(BASES, rows),
        regions=draft_mlp.regions(),
        rows=rows,
        chain=rows,
        world=WORLD,
        peers=draft_mlp.PEERS,
    )


def _worker(me: int, port: int, rows: int, present) -> None:
    """``present``: None for the whole TP8 group, one process a rank; or the
    two rank programs that run (process ``me`` runs ``present[me]``) in a
    zero-shard model (:func:`_shared_inputs`) whose other ranks' puts
    (exactly zero) the host writes before each launch."""
    torch.cuda.set_device(me)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://localhost:{port}",
        rank=me,
        world_size=WORLD if present is None else len(present),
        timeout=timedelta(seconds=300),
    )
    try:
        device = torch.device(f"cuda:{me}")
        ctx = get_or_create_context(heap_size=256 << 20)
        group = IrisGroup(ctx, dist.group.WORLD, me, device)
        plan = _plan(rows)
        scratch = ctx.zeros((plan.scratch_pairs,), dtype=torch.int64)
        if present is None:
            rank, model, bases = me, ALL, group.bases
        else:
            rank, model = present[me], present
            ghost = torch.zeros_like(scratch)
            bases = standins.peer_bases(group.bases, present, me, scratch, ghost)
        own = slice(rank * OWNED, (rank + 1) * OWNED)
        group.finish_prepare()
        # Each launch has new inputs and a new step (the peer regions'
        # copies alternate).
        for k in range(LAUNCHES):
            step = 11 + k
            parts, residual, gain, coef_w, base = _shared_inputs(k, rows, model)
            gu_w, down_w = inputs.draft_mlp_rank(k, rank)
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
            residual_out = torch.zeros(
                rows, HIDDEN, dtype=torch.bfloat16, device=device
            )
            out = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device=device)
            draft_mlp_kernel[(plan.grid,)](
                scratch,
                torch.full((1,), step, dtype=torch.int32, device=device),
                bases,
                parts[rank].to(device),
                residual.to(device),
                residual_out,
                gain[own].float().to(device),
                bf16_tiles(coef_w).to(device),
                base.to(device),
                bf16_tiles(gu_w).to(device),
                bf16_tiles(down_w).to(device),
                out,
                None,
                plan.key,
                rank,
                num_warps=8,
            )
            torch.cuda.synchronize()
            tag = launch_tag(step, 0)

            def words(name: str, width: int, copy: int = 0) -> torch.Tensor:
                return torch.stack(
                    [
                        payloads(
                            scratch, plan, name, width, tag, row=row, copy=copy
                        ).cpu()
                        for row in range(rows)
                    ]
                )

            def region(name: str, width: int, copy: int = 0) -> torch.Tensor:
                return words(name, width, copy).view(torch.bfloat16)

            want_h, want_res = ref.tp_sum_norm(parts, residual, gain, round_tp_sum=True)
            # The same additions in the same order: the residual is exact.
            assert torch.equal(residual_out[:, own].cpu(), want_res[:, own])
            # Every stage is checked on the kernel's own input, so a mismatch
            # names the stage. The norm scale sums its squares in another
            # order; the projections sum their K parts in another order.
            h = region("hin", HIDDEN // 2, copy=step & 1)
            ref.assert_bf16_close(h, want_h)
            if present is not None:
                # The model is what the stand-ins claim: the absent ranks'
                # columns of the normed input are zero, in both.
                absent = standins.absent_columns(HIDDEN, OWNED, present)
                assert not want_h[:, absent].float().any()
                assert not h[:, absent].float().any()
            # The convolutions read a coefficient as its two K parts' sum,
            # rounded to BF16.
            cpart = words("cpart", 2 * ref.COEFFICIENTS).view(torch.float32)
            dl = (cpart[:, : ref.COEFFICIENTS] + cpart[:, ref.COEFFICIENTS :]).to(
                torch.bfloat16
            )
            ref.assert_bf16_close(dl, (h.float() @ coef_w.float().T).to(torch.bfloat16))
            delta = dl.view(rows, 2, ref.TAPS, ref.GROUPS)
            hc = region("hc", HIDDEN // 2)
            ref.assert_bf16_close(hc, ref.grouped_conv(h, delta[:, 0], base[0]))
            act = region("act", ref.INTER_RANK // 2)
            gate, up = (
                (hc.float() @ gu_w.float().T)
                .to(torch.bfloat16)
                .float()
                .split(ref.INTER_RANK, dim=-1)
            )
            ref.assert_bf16_close(
                act, (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
            )
            partial = (act.float() @ down_w.float().T).to(torch.bfloat16)
            ref.assert_bf16_close(
                out.cpu(), ref.grouped_conv(partial, delta[:, 1], base[1])
            )
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("rows", [8, 16])
def test_draft_mlp_half_matches_the_reference_stage_by_stage(rows: int):
    if torch.cuda.device_count() < WORLD:
        pytest.skip(f"requires {WORLD} GPUs")
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port, rows, None), nprocs=WORLD, join=True)


@pytest.mark.parametrize("present", standins.PAIRS, ids=standins.PAIR_IDS)
@pytest.mark.parametrize("rows", [8, 16])
def test_draft_mlp_half_matches_the_reference_stage_by_stage_on_two_gpus(
    rows: int, present
):
    """As :func:`test_draft_mlp_half_matches_the_reference_stage_by_stage`
    with the production rank programs ``present`` on two GPUs, in a
    zero-shard model (:func:`_shared_inputs`) whose six other ranks put
    exactly zero (the host writes it): every stage against the reference on
    the kernel's own input, and the absent ranks' columns of the normed
    input stay zero. Two non-zero terms per sum cannot check the rank order
    of the eight-rank all-reduce (see muse_spark_standins)."""
    if torch.cuda.device_count() < 2:
        pytest.skip("requires 2 GPUs")
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port, rows, present), nprocs=2, join=True)

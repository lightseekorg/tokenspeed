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
from monokernel_testing import payloads
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

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


def _shared(launch: int, rows: int):
    """Every rank's BF16 attention partial, the residual, and the replicated
    weights: the norm gain, the coefficient projection and base_kernel."""
    g = torch.Generator().manual_seed(4100 + launch)
    parts = [
        (torch.randn(rows, HIDDEN, generator=g) * 0.5).to(torch.bfloat16)
        for _ in range(WORLD)
    ]
    residual = (torch.randn(rows, HIDDEN, generator=g) * 2).to(torch.bfloat16)
    gain = (1 + 0.1 * torch.randn(HIDDEN, generator=g)).to(torch.bfloat16)
    coef_w = (torch.randn(ref.COEFFICIENTS, HIDDEN, generator=g) * 0.01).to(
        torch.bfloat16
    )
    base = (0.5 + 0.1 * torch.randn(2, ref.TAPS, HIDDEN, generator=g)).to(
        torch.bfloat16
    )
    return parts, residual, gain, coef_w, base


def _rank_weights(launch: int, rank: int):
    """This rank's gate/up rows and down columns."""
    g = torch.Generator().manual_seed(5200 + 10 * launch + rank)
    gu = (torch.randn(2 * ref.INTER_RANK, HIDDEN, generator=g) * 0.02).to(
        torch.bfloat16
    )
    down = (torch.randn(HIDDEN, ref.INTER_RANK, generator=g) * 0.02).to(torch.bfloat16)
    return gu, down


def _close(got: torch.Tensor, want: torch.Tensor) -> None:
    """Within one BF16 step, or a step of the row's largest value where a
    sum cancels to near zero."""
    torch.testing.assert_close(
        got.float(),
        want.float(),
        rtol=2**-7,
        atol=want.float().abs().max().item() * 2**-9,
    )


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
        plan = _plan(rows)
        scratch = ctx.zeros((plan.scratch_pairs,), dtype=torch.int64)
        own = slice(rank * OWNED, (rank + 1) * OWNED)
        group.finish_prepare()
        # Each launch has new inputs and a new step (the peer regions'
        # copies alternate).
        for k in range(LAUNCHES):
            step = 11 + k
            parts, residual, gain, coef_w, base = _shared(k, rows)
            gu_w, down_w = _rank_weights(k, rank)
            residual_out = torch.zeros(
                rows, HIDDEN, dtype=torch.bfloat16, device=device
            )
            out = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device=device)
            draft_mlp_kernel[(plan.grid,)](
                scratch,
                torch.full((1,), step, dtype=torch.int32, device=device),
                group.bases,
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
            _close(h, want_h)
            # The convolutions read a coefficient as its two K parts' sum,
            # rounded to BF16.
            cpart = words("cpart", 2 * ref.COEFFICIENTS).view(torch.float32)
            dl = (cpart[:, : ref.COEFFICIENTS] + cpart[:, ref.COEFFICIENTS :]).to(
                torch.bfloat16
            )
            _close(dl, (h.float() @ coef_w.float().T).to(torch.bfloat16))
            delta = dl.view(rows, 2, ref.TAPS, ref.GROUPS)
            hc = region("hc", HIDDEN // 2)
            _close(hc, ref.grouped_conv(h, delta[:, 0], base[0]))
            act = region("act", ref.INTER_RANK // 2)
            gate, up = (
                (hc.float() @ gu_w.float().T)
                .to(torch.bfloat16)
                .float()
                .split(ref.INTER_RANK, dim=-1)
            )
            _close(act, (torch.nn.functional.silu(gate) * up).to(torch.bfloat16))
            partial = (act.float() @ down_w.float().T).to(torch.bfloat16)
            _close(out.cpu(), ref.grouped_conv(partial, delta[:, 1], base[1]))
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
    mp.spawn(_worker, args=(port, rows), nprocs=WORLD, join=True)

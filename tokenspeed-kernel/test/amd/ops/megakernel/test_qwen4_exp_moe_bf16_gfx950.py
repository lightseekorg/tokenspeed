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

"""Qwen3.8-Flash-Next MoE stages with BF16 experts (the MTP draft layer's),
on TokenSpeed's own unquantized expert weights."""

from __future__ import annotations

import functools

import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from megakernel_testing import fill  # noqa: E402
from tokenspeed_kernel_amd._scheduling import sched_compile_options  # noqa: E402
from tokenspeed_kernel_amd._triton import gl, gluon  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.megakernel import (  # noqa: E402
    mailbox,
    schedule,
    trace,
)
from tokenspeed_kernel_amd.ops.gfx950.megakernel.plan import (  # noqa: E402
    launch_tag,
)
from tokenspeed_kernel_amd.ops.gfx950.megakernel.qwen4_exp import (  # noqa: E402
    moe,
    moe_bf16,
)
from tokenspeed_kernel_amd.ops.gfx950.megakernel.qwen4_exp.layer import (  # noqa: E402
    qsa_plan,
)

HIDDEN, EXPERTS, INTER = 2560, 512, 640
STEP = 13
TAG = launch_tag(STEP, 0)


def _plan(rows: int):
    return qsa_plan(False, rows, 1, "bf16", False).only(("m3", "m4", "m5"))


@gluon.jit
def _moe_kernel(scratch, step, experts, out, trace_buf, PLAN: gl.constexpr):
    table = trace.allocate(trace_buf)
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, table)
    weights = moe_bf16.Weights(*experts)
    task, active = schedule.stage_task(PLAN, "m3")
    if active:
        moe.router_stage(mb, task, weights, "m_", "m3")
    first, rounds = schedule.stage_rounds(PLAN, "m4")
    for i in range(rounds):
        moe_bf16.up_stage(mb, first + i * schedule.grid(PLAN), weights, "m_")
    task, active = schedule.stage_task(PLAN, "m5")
    if active:
        moe_bf16.down_stage(mb, task, weights, out, "m_", "w", "")
    trace.flush(table, trace_buf)


@functools.cache
def _weights():
    """Random BF16 experts (about 6.7 GB), shared by the tests."""
    g = torch.Generator(device="cuda").manual_seed(0)

    def bf16(*shape, scale):
        return (torch.randn(*shape, generator=g, device="cuda") * scale).bfloat16()

    return {
        "router_w": bf16(EXPERTS, HIDDEN, scale=0.02),
        "gate_w": bf16(HIDDEN, scale=0.02),
        "w13": bf16(EXPERTS, 2 * INTER, HIDDEN, scale=0.02),
        "w2": bf16(EXPERTS, HIDDEN, INTER, scale=0.04),
        "shared_gu": bf16(2 * INTER, HIDDEN, scale=0.02),
        "shared_dn": bf16(HIDDEN, INTER, scale=0.04),
    }


def _run(x, residual, inject):
    """The stages on ``rows`` rows. Returns the output streams and the
    router logits that M3 published."""
    rows = x.shape[0]
    plan = _plan(rows)
    w = _weights()
    scratch = torch.zeros(plan.scratch_pairs, dtype=torch.int64, device="cuda")
    for r in range(rows):
        at = r * plan.row_pairs
        fill(
            scratch[at:],
            plan,
            "m_xbf",
            x[r].view(-1, 2).view(torch.int32).flatten(),
            TAG,
        )
        fill(scratch[at:], plan, "m_inj", inject[r].flatten(), TAG)
        fill(
            scratch[at:],
            plan,
            "w",
            residual[r].view(-1, 2).view(torch.int32).flatten(),
            TAG,
        )
    out = torch.empty(rows, 4 * HIDDEN, dtype=torch.bfloat16, device="cuda")
    step = torch.full((1,), STEP, dtype=torch.int32, device="cuda")
    _moe_kernel[(plan.grid,)](
        scratch,
        step,
        (w["router_w"], w["gate_w"], w["w13"], w["w2"], w["shared_gu"], w["shared_dn"]),
        out,
        None,
        plan.key,
        num_warps=8,
        **sched_compile_options(),
    )
    torch.cuda.synchronize()
    logits = []
    for r in range(rows):
        at = plan.offset("lg") + r * plan.row_pairs
        words = (scratch[at : at + 256] & 0xFFFFFFFF).to(torch.int32)
        logits.append(words.view(torch.bfloat16))
    return out, torch.stack(logits)


def _inputs(rows, seed):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(rows, HIDDEN, generator=g) * 0.5).bfloat16().cuda()
    residual = torch.randn(rows, 4 * HIDDEN, generator=g).bfloat16().cuda()
    inject = torch.randn(rows, 2, 4, generator=g).cuda()
    return x, residual, inject


def _bf(t):
    return t.bfloat16().float()


@pytest.mark.parametrize("rows", [1, 2, 4])
def test_moe_bf16_matches_tokenspeed_math(rows):
    """TokenSpeed's BF16 MoE at one row: FP32 gate / up and down of a routed
    expert (the routed sum rounded once), BF16 gate / up and down of the
    shared expert."""
    x, residual, inject = _inputs(rows, rows)
    out, logits = _run(x, residual, inject)
    w = _weights()
    torch.testing.assert_close(
        logits.float(), _bf(x.float() @ w["router_w"].float().T), rtol=1e-2, atol=1e-3
    )
    for r in range(rows):
        xr = x[r].float()
        top_w, top_e = torch.topk(torch.softmax(logits[r].float(), 0), 10)
        top_w = top_w / top_w.sum()

        def act(gate, up):
            return _bf(gate * torch.sigmoid(gate) * up)

        routed = 0.0
        for k, e in enumerate(top_e.tolist()):
            gu = w["w13"][e].float() @ xr
            routed = routed + top_w[k] * (
                w["w2"][e].float() @ act(gu[:INTER], gu[INTER:])
            )
        gu = _bf(w["shared_gu"].float() @ xr)
        shared = _bf(w["shared_dn"].float() @ act(gu[:INTER], gu[INTER:]))
        shared_gate = torch.sigmoid(w["gate_w"].float() @ xr)
        block = _bf(_bf(routed) + shared_gate * shared)
        mix = 2 * torch.sigmoid(_bf(inject[r].sum(0)))
        ref = _bf(residual[r].float().view(4, HIDDEN) + block[None, :] * mix[:, None])
        torch.testing.assert_close(
            out[r].view(4, HIDDEN).float(), ref, rtol=2e-2, atol=2e-2
        )


@pytest.mark.parametrize("rows", [2, 3, 4])
def test_moe_bf16_rows_equal_one_row_launches(rows):
    """Each row of a multi-row launch is bitwise its own one-row launch."""
    x, residual, inject = _inputs(rows, 10 + rows)
    together, _ = _run(x, residual, inject)
    for r in range(rows):
        alone, _ = _run(x[r : r + 1], residual[r : r + 1], inject[r : r + 1])
        assert torch.equal(together[r], alone[0]), r

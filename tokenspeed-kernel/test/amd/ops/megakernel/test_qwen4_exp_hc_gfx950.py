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

"""Qwen3.8-Flash-Next hyper-connection mixer stages of the decode megakernel."""

from __future__ import annotations

import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from megakernel_testing import (  # noqa: E402
    fill,
    mxfp4,
    payloads,
)
from tokenspeed_kernel_amd._triton import gl, gluon  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.megakernel import (  # noqa: E402
    mailbox,
    schedule,
    trace,
)
from tokenspeed_kernel_amd.ops.gfx950.megakernel.plan import (  # noqa: E402
    launch_tag,
)
from tokenspeed_kernel_amd.ops.gfx950.megakernel.qwen4_exp import hc  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.megakernel.qwen4_exp.layer import (  # noqa: E402
    delta_plan,
)
from tokenspeed_kernel_amd.ops.gfx950.megakernel.qwen4_exp.weights import (  # noqa: E402
    pack_mixer,
)

HIDDEN, WIDTH, LOWRANK = 2560, 10240, 320
STEP = 5
TAG = launch_tag(STEP, 0)


@gluon.jit
def _mixer_kernel(
    scratch,
    step,
    h,
    h_stride,
    mixer,
    mixed_out,
    trace_buf,
    PLAN: gl.constexpr,
    MLP: gl.constexpr,
):
    """The attention mixer (A1, A2: global streams, BF16 out) or the MLP
    mixer (M1, M2: streams in region ``w``, BF16 and MXFP4 out)."""
    table = trace.allocate(trace_buf)
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, table)
    weights = hc.Mixer(*mixer)
    if MLP:
        streams = hc.Streams(h, h_stride, "w")
        task, active = schedule.stage_task(PLAN, "m1")
        if active:
            hc.reduce_stage(mb, task, streams, weights, "m_", "m1")
        task, active = schedule.stage_task(PLAN, "m2")
        if active:
            hc.mix_stage(mb, task, streams, weights, "m_", mixed_out, True, "m2")
    else:
        streams = hc.Streams(h, h_stride, "")
        task, active = schedule.stage_task(PLAN, "a1")
        if active:
            hc.reduce_stage(mb, task, streams, weights, "a_", "a1")
        task, active = schedule.stage_task(PLAN, "a2")
        if active:
            hc.mix_stage(mb, task, streams, weights, "a_", mixed_out, False, "a2")
    trace.flush(table, trace_buf)


def _bf16(x: torch.Tensor) -> torch.Tensor:
    return x.bfloat16().float()


# ``one_row``: the input is one row that stands for all four streams (the
# first layer's embedding). The kernel reads it with stream stride 0.
@pytest.mark.parametrize(
    "mlp, rows, one_row",
    [(False, 1, False), (False, 1, True), (False, 3, False), (True, 1, False)]
    + [(True, 4, False)],
)
def test_mixer_matches_tokenspeed_math(mlp, rows, one_row):
    generator = torch.Generator().manual_seed(rows)
    h = (torch.randn(rows, WIDTH, generator=generator) * 3).bfloat16().cuda()
    if one_row:
        h = h[:, :HIDDEN].repeat(1, 4)
    norm_w = (torch.randn(WIDTH, generator=generator) * 0.1).bfloat16().cuda()
    down = (torch.randn(LOWRANK, WIDTH, generator=generator) * 0.01).bfloat16()
    inject = (torch.randn(4, WIDTH, generator=generator) * 0.01).bfloat16()
    up = (torch.randn(WIDTH, LOWRANK, generator=generator) * 0.05).bfloat16().cuda()
    # TokenSpeed pads the projection's storage to whole 8-row groups.
    pad = torch.zeros(4, WIDTH, dtype=torch.bfloat16)
    w = pack_mixer(norm_w, torch.cat([down, inject, pad]).cuda(), up)
    names = ("m1", "m2") if mlp else ("a1", "a2")
    plan = delta_plan(False, rows, 1, "mxfp4").only(names)
    P = "m_" if mlp else "a_"
    scratch = torch.zeros(plan.scratch_pairs, dtype=torch.int64, device="cuda")
    if mlp:
        for r in range(rows):
            words = h[r].view(torch.int32)
            start = plan.offset("w") + r * plan.row_pairs
            scratch[start : start + words.numel()] = (TAG << 32) | (
                words.to(torch.int64) & 0xFFFFFFFF
            )
        assert "w" in plan.external
    step = torch.full((1,), STEP, dtype=torch.int32, device="cuda")
    mixed_out = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device="cuda")
    _mixer_kernel[(plan.grid,)](
        scratch,
        step,
        h[:, :HIDDEN].contiguous() if one_row else h,
        0 if one_row else HIDDEN,
        w.kernel_args(),
        mixed_out,
        None,
        plan.key,
        mlp,
        num_warps=8,
    )
    torch.cuda.synchronize()

    # Reference: TokenSpeed's two-launch mix
    # (gluon_hyperconnection_mix_gfx950 and its up kernel).
    streams = h.float().view(rows, 4, HIDDEN)
    inv = torch.rsqrt(streams.square().mean(-1) + 1e-6)
    y = _bf16(streams * inv[..., None] * (1.0 + norm_w.float().view(4, HIDDEN)))
    y = y.view(rows, WIDTH)
    for r in range(rows):
        got_inj = payloads(scratch, plan, P + "inj", 8, TAG, r).view(torch.float32)
        ref_inj = (y[r].view(2, 2 * HIDDEN) * inject.cuda().float().view(4, 2, -1)).sum(
            -1
        )
        torch.testing.assert_close(got_inj.view(2, 4), ref_inj.T, rtol=1e-4, atol=1e-5)
        got_lr = payloads(scratch, plan, P + "lr", 4 * LOWRANK, TAG, r)
        got_lr = got_lr.view(torch.float32).view(4, LOWRANK)
        ref_lr = torch.einsum(
            "bk,nbk->bn", y[r].view(4, HIDDEN), down.cuda().float().view(-1, 4, HIDDEN)
        )
        torch.testing.assert_close(
            got_lr, ref_lr, rtol=1e-4, atol=1e-4 * ref_lr.abs().max().item()
        )
        # The mix stage from the kernel's own low-rank partials.
        value = _bf16(got_lr.sum(0))
        activated = _bf16(value * torch.sigmoid(value))
        gate = _bf16(up.float() @ activated)
        ref_mixed = (torch.sigmoid(gate).view(4, HIDDEN) * y[r].view(4, HIDDEN)).mean(0)
        got_mixed = payloads(scratch, plan, P + "xbf", 1280, TAG, r).view(
            torch.bfloat16
        )
        torch.testing.assert_close(
            got_mixed.float(), _bf16(ref_mixed), rtol=1e-2, atol=1e-2
        )
        # The optional global copy of the mixed row (the final mixer's output).
        assert torch.equal(mixed_out[r], got_mixed)
        if mlp:
            # The MXFP4 copy is TokenSpeed's quantization of the BF16 row.
            ref_q, ref_e = mxfp4(got_mixed[None])
            x4 = payloads(scratch, plan, "m_x4", 320, TAG, r)
            xe = payloads(scratch, plan, "m_xe", 80, TAG, r)
            assert torch.equal(x4.view(torch.uint8), ref_q.view(-1))
            assert torch.equal(xe.to(torch.uint8), ref_e.view(-1))

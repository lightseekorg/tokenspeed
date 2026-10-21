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
    grid_weight,
    payloads,
    round_mxfp8,
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
PLAN = delta_plan(False, 1, 1).only(("a1", "a2"))


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
):
    table = trace.allocate(trace_buf)
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, table)
    streams = hc.Streams(h, h_stride, "")
    weights = hc.Mixer(*mixer)
    task, active = schedule.stage_task(PLAN, "a1")
    if active:
        hc.reduce_stage(mb, task, streams, weights, "a_", "a1")
    task, active = schedule.stage_task(PLAN, "a2")
    if active:
        hc.mix_stage(mb, task, streams, weights, "a_", mixed_out, "a2")
    trace.flush(table, trace_buf)


# ``one_row``: the input is one row that stands for all four streams (the
# first layer's embedding). The kernel reads it with stream stride 0.
@pytest.mark.parametrize("one_row", [False, True])
def test_mixer_matches_tokenspeed_math(one_row):
    generator = torch.Generator().manual_seed(0)
    h = (torch.randn(WIDTH, generator=generator) * 3).bfloat16().cuda()
    row = h[:HIDDEN].clone()
    if one_row:
        h = row.repeat(4)
    norm_w = (torch.randn(WIDTH, generator=generator) * 0.1).bfloat16().cuda()
    down = grid_weight(LOWRANK, WIDTH, generator)
    inject = (torch.randn(4, WIDTH, generator=generator) * 0.01).bfloat16().cuda()
    up = grid_weight(WIDTH, LOWRANK, generator)
    w = pack_mixer(norm_w, torch.cat([down, inject]), up)
    scratch = torch.zeros(PLAN.scratch_pairs, dtype=torch.int64, device="cuda")
    step = torch.full((1,), STEP, dtype=torch.int32, device="cuda")
    mixed_out = torch.zeros(HIDDEN, dtype=torch.bfloat16, device="cuda")
    _mixer_kernel[(PLAN.grid,)](
        scratch,
        step,
        row if one_row else h,
        0 if one_row else HIDDEN,
        w.kernel_args(),
        mixed_out,
        None,
        PLAN.key,
        num_warps=8,
    )
    torch.cuda.synchronize()

    # Reference: TokenSpeed's mixer, with the megakernel's MXFP8 activations.
    streams = h.float().view(4, HIDDEN)
    inv = torch.rsqrt(streams.square().mean(-1) + 1e-6)
    y = (
        (streams * inv[:, None] * (1.0 + norm_w.float().view(4, HIDDEN)))
        .bfloat16()
        .float()
        .flatten()
    )
    halves = [slice(0, 2 * HIDDEN), slice(2 * HIDDEN, WIDTH)]
    got_inj = payloads(scratch, PLAN, "a_inj", 8, TAG).view(torch.float32).view(2, 4)
    for half, cols in enumerate(halves):
        ref_inj = inject.float()[:, cols] @ y[cols]
        torch.testing.assert_close(got_inj[half], ref_inj, rtol=1e-3, atol=1e-4)

    y8 = round_mxfp8(y)
    got_lr = (
        payloads(scratch, PLAN, "a_lr", 640, TAG).view(torch.float32).view(2, LOWRANK)
    )
    for half, cols in enumerate(halves):
        ref = down.float()[:, cols] @ y8[cols]
        torch.testing.assert_close(
            got_lr[half], ref, rtol=1e-4, atol=1e-4 * ref.abs().max().item()
        )
    projected = got_lr.sum(0)
    lr = projected * torch.sigmoid(projected)

    # The mix stage from the kernel's own low-rank activation.
    gate = up.float() @ round_mxfp8(lr)
    ref_mixed = (
        (torch.sigmoid(gate).view(4, HIDDEN) * y.view(4, HIDDEN)).mean(0).bfloat16()
    )
    xbf = payloads(scratch, PLAN, "a_xbf", 1280, TAG)
    got_mixed = xbf.view(torch.bfloat16)
    torch.testing.assert_close(
        got_mixed.float(), ref_mixed.float(), rtol=1e-2, atol=1e-2
    )
    x8 = (
        payloads(scratch, PLAN, "a_x8", 640, TAG)
        .view(torch.uint8)
        .view(torch.float8_e4m3fn)
        .float()
    )
    xe = payloads(scratch, PLAN, "a_xe", 80, TAG)
    got8 = x8 * torch.exp2(xe.float() - 127).repeat_interleave(32)
    torch.testing.assert_close(got8, round_mxfp8(got_mixed), rtol=0, atol=0)
    # The optional global copy of the mixed row (the final mixer's output).
    assert torch.equal(mixed_out, got_mixed)

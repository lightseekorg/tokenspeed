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

"""Qwen3.8-Flash-Next Gated DeltaNet stages of the decode megakernel."""

from __future__ import annotations

import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from megakernel_testing import (  # noqa: E402
    fill,
    payloads,
    softplus,
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
from tokenspeed_kernel_amd.ops.gfx950.megakernel.qwen4_exp import gdn  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.megakernel.qwen4_exp.layer import (  # noqa: E402
    delta_plan,
)
from tokenspeed_kernel_amd.ops.gfx950.megakernel.weights import (  # noqa: E402
    bf16_tiles,
)

HIDDEN, QKV, VALUE, HEAD = 2560, 10240, 6144, 128
STEP = 9
TAG = launch_tag(STEP, 0)
PLAN = delta_plan(False, 1, 1, "mxfp4").only(("b1", "b3"))


@gluon.jit
def _gdn_kernel(
    scratch,
    step,
    gdn_weights,
    gdn_state,
    trace_buf,
    PLAN: gl.constexpr,
):
    table = trace.allocate(trace_buf)
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, table)
    weights = gdn.Weights(*gdn_weights)
    state = gdn.State(*gdn_state)
    task, active = schedule.stage_task(PLAN, "b1")
    if active:
        gdn.projection_stage(mb, task, weights, state, "a_", "b1")
    task, active = schedule.stage_task(PLAN, "b3")
    if active:
        gdn.recurrence_stage(mb, 0, task, weights, state, "b3")
    trace.flush(table, trace_buf)


def test_gated_deltanet_matches_tokenspeed_math():
    torch.manual_seed(0)
    dev = "cuda"
    mixed = torch.randn(HIDDEN, device=dev).bfloat16()
    # in_proj_qkvzba: q | k | v, z, then b | a.
    in_proj = (torch.randn(QKV + VALUE + 96, HIDDEN, device=dev) * 0.02).bfloat16()
    conv_w = (torch.randn(QKV, 4, device=dev) * 0.5).bfloat16()
    conv_state = torch.randn(3, QKV, 3, device=dev).bfloat16()
    a_log = torch.randn(48, device=dev) * 0.5
    dt_bias = torch.randn(48, device=dev) * 0.5
    norm_w = (1 + 0.1 * torch.randn(HEAD, device=dev)).bfloat16()
    ssm = torch.randn(3, 48, HEAD, HEAD, device=dev) * 0.1
    slots = torch.tensor([1, 2], dtype=torch.int32, device=dev)
    conv_before, ssm_before = conv_state.clone(), ssm.clone()

    scratch = torch.zeros(PLAN.scratch_pairs, dtype=torch.int64, device=dev)
    fill(
        scratch,
        PLAN,
        "a_xbf",
        mixed.view(-1, 2).contiguous().view(torch.int32).flatten(),
        TAG,
    )
    step = torch.full((1,), STEP, dtype=torch.int32, device=dev)
    _gdn_kernel[(PLAN.grid,)](
        scratch,
        step,
        (
            bf16_tiles(in_proj),
            conv_w,
            a_log,
            dt_bias,
            norm_w,
        ),
        # Decode: the window after the token goes to the output page.
        (
            conv_state,
            conv_state.stride(0),
            conv_state,
            slots[1:],
            conv_state.stride(0),
            ssm,
            ssm.stride(0),
            None,
            0,
            slots[:1],
            slots[1:],
        ),
        None,
        PLAN.key,
        num_warps=8,
    )
    torch.cuda.synchronize()

    # Projection and conv.
    proj = (in_proj.float() @ mixed.float()).bfloat16()
    window = torch.cat([conv_before[1].float(), proj[:QKV, None].float()], 1)
    conv = window * conv_w.float()
    conv = conv[:, 0] + conv[:, 1] + conv[:, 2] + conv[:, 3]
    qkv = (conv * torch.sigmoid(conv)).bfloat16()
    got = payloads(scratch, PLAN, "qkvz", 8192, TAG).view(torch.bfloat16)
    torch.testing.assert_close(got[:QKV].float(), qkv.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(
        got[QKV:].float(), proj[QKV : QKV + VALUE].float(), rtol=1e-2, atol=1e-2
    )
    new_conv = torch.cat([conv_before[1][:, 1:], proj[:QKV, None]], 1)
    torch.testing.assert_close(
        conv_state[2].float(), new_conv.float(), rtol=1e-2, atol=1e-2
    )
    assert torch.equal(conv_state[1], conv_before[1])

    ba = proj[QKV + VALUE :].float()
    got_ba = payloads(scratch, PLAN, "ba", 96, TAG).view(torch.float32)
    torch.testing.assert_close(got_ba, ba, rtol=1e-2, atol=1e-3)

    # Recurrence and gated norm on the kernel's own projections.
    q, k, v = got[:QKV].float().split([2048, 2048, VALUE])
    z = got[QKV:].float()
    b, a = got_ba.split(48)
    ref_state = ssm_before[1].clone()
    outputs = []
    for vh in range(48):
        kh = vh // 3
        qn = q[kh * HEAD : (kh + 1) * HEAD]
        kn = k[kh * HEAD : (kh + 1) * HEAD]
        qn = qn / torch.sqrt((qn * qn).sum() + 1e-6)
        kn = kn / torch.sqrt((kn * kn).sum() + 1e-6)
        g = -torch.exp(a_log[vh]) * softplus(a[vh] + dt_bias[vh])
        st = ref_state[vh] * torch.exp(g)  # [V, K]
        delta = torch.sigmoid(b[vh]) * (v[vh * HEAD : (vh + 1) * HEAD] - st @ kn)
        st = st + delta[:, None] * kn[None, :]
        ref_state[vh] = st
        o = ((st @ qn) * HEAD**-0.5).bfloat16().float()
        o = o * torch.rsqrt(o.square().mean() + 1e-6)
        zz = z[vh * HEAD : (vh + 1) * HEAD]
        outputs.append((o * torch.sigmoid(zz) * norm_w.float()).bfloat16().float())
    torch.testing.assert_close(ssm[2], ref_state, rtol=1e-4, atol=1e-5)
    assert torch.equal(ssm[1], ssm_before[1])
    y = torch.cat(outputs)
    obf = payloads(scratch, PLAN, "obf", 3072, TAG).view(torch.bfloat16)
    torch.testing.assert_close(obf.float(), y, rtol=1e-2, atol=1e-2)

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

"""Qwen3.8-Flash-Next MoE stages of the decode megakernel, on TokenSpeed's
own N16 expert banks."""

from __future__ import annotations

import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from megakernel_testing import (  # noqa: E402
    fill,
    mxfp4,
    random_mxfp4,
    round_mxfp4,
)
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
)
from tokenspeed_kernel_amd.ops.gfx950.megakernel.qwen4_exp.layer import (  # noqa: E402
    delta_plan,
)
from tokenspeed_kernel_amd.ops.gfx950.megakernel.qwen4_exp.weights import (  # noqa: E402
    alternate_gate_up,
)
from tokenspeed_kernel_amd.ops.gfx950.megakernel.weights import (  # noqa: E402
    dequantize_mxfp4,
    mxfp4_tiles,
    n32_scales,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.n16_weights import (  # noqa: E402
    preprocess_n16_mxfp4_weights,
)

HIDDEN, EXPERTS, INTER = 2560, 512, 640
STEP = 11
TAG = launch_tag(STEP, 0)
PLAN = delta_plan(False, 1, 1, "mxfp4").only(("m3", "m4", "m5"))


@gluon.jit
def _moe_kernel(
    scratch,
    step,
    experts,
    out,
    trace_buf,
    PLAN: gl.constexpr,
):
    table = trace.allocate(trace_buf)
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, table)
    weights = moe.Weights(*experts)
    task, active = schedule.stage_task(PLAN, "m3")
    if active:
        moe.router_stage(mb, task, weights, "m_", "m3")
    task, active = schedule.stage_task(PLAN, "m4")
    if active:
        moe.up_stage(mb, task, weights, "m_")
    task, active = schedule.stage_task(PLAN, "m5")
    if active:
        moe.down_stage(mb, task, weights, out, "m_", "w", "")
    trace.flush(table, trace_buf)


@gluon.jit
def _routing_kernel(scratch, step, experts_out, weights_out, PLAN: gl.constexpr):
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, None)
    experts, weights = moe.routing(mb)
    R: gl.constexpr = schedule.rows(PLAN)
    lay: gl.constexpr = experts.type.layout
    r = gl.arange(0, experts.shape[0], gl.SliceLayout(1, lay))[:, None]
    k = gl.arange(0, 16, gl.SliceLayout(0, lay))[None, :]
    gl.store(experts_out + r * 16 + k, experts, mask=r < R)
    gl.store(weights_out + r * 16 + k, weights, mask=r < R)


@pytest.mark.parametrize("rows", [1, 2, 3, 4])
def test_routing_picks_top_experts(rows):
    """Every row's top 10 experts in rank order (ties: the lower expert) and
    their renormalized softmax weights, from BF16 logits that include
    negative values and ties across the cut."""
    g = torch.Generator().manual_seed(rows)
    logits = (torch.randn(rows, EXPERTS, generator=g) * 2).bfloat16()
    # Ties: equal logits on both sides of the 10th place, and at the top.
    logits[:, 7:15] = logits[:, 7].max() + 4
    logits[:, 300] = logits[:, 7]
    logits[:, 500] = logits[:, 7] + 1
    plan = delta_plan(False, rows, 1, "mxfp4")
    scratch = torch.zeros(plan.scratch_pairs, dtype=torch.int64, device="cuda")
    for r in range(rows):
        start = plan.offset("lg") + r * plan.row_pairs
        words = logits[r].view(-1, 2).view(torch.int32).flatten().to(torch.int64)
        scratch[start : start + words.numel()] = (TAG << 32) | (words & 0xFFFFFFFF)
    experts = torch.zeros(rows, 16, dtype=torch.int32, device="cuda")
    weights = torch.zeros(rows, 16, dtype=torch.float32, device="cuda")
    step = torch.full((1,), STEP, dtype=torch.int32, device="cuda")
    _routing_kernel[(1,)](scratch, step, experts, weights, plan.key, num_warps=8)
    torch.cuda.synchronize()

    for r in range(rows):
        values = logits[r].float()
        # Logit descending, then expert ascending.
        order = sorted(range(EXPERTS), key=lambda e: (-values[e].item(), e))[:10]
        assert experts[r, :10].tolist() == order
        assert experts[r, 10:].tolist() == [0] * 6
        top = values[order]
        ref = torch.exp(top - top[0])
        ref = ref / ref.sum()
        torch.testing.assert_close(weights[r, :10].cpu(), ref, rtol=1e-6, atol=1e-7)
        assert weights[r, 10:].tolist() == [0.0] * 6


class _Bank(torch.nn.Module):
    def __init__(self, w13, s13, w2, s2):
        super().__init__()
        for name, value in zip(
            ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"),
            (w13, s13, w2, s2),
        ):
            self.register_parameter(
                name, torch.nn.Parameter(value, requires_grad=False)
            )


def test_n32_scales_match_tokenspeed_banks():
    g = torch.Generator().manual_seed(1)
    w13, s13 = random_mxfp4((2, 2 * 64, 512), g)
    w2, s2 = random_mxfp4((2, 512, 64), g)
    bank = _Bank(w13, s13, w2, s2)
    preprocess_n16_mxfp4_weights(bank)
    assert torch.equal(n32_scales(s2[1], 64), bank.w2_weight_scale[1].flatten())
    # Gate / up scales: the bank interleaves 16-row gate and up blocks.
    s13_interleaved = s13[1].view(2, 4, 16, -1).transpose(0, 1).reshape(128, -1)
    assert torch.equal(
        n32_scales(s13_interleaved, 512), bank.w13_weight_scale[1].flatten()
    )


def test_moe_matches_tokenspeed_math():
    g = torch.Generator().manual_seed(0)
    dev = "cuda"
    x = (torch.randn(HIDDEN, generator=g) * 0.5).bfloat16().cuda()
    router_w = (torch.randn(EXPERTS, HIDDEN, generator=g) * 0.02).bfloat16().cuda()
    gate_w = (torch.randn(HIDDEN, generator=g) * 0.02).bfloat16().cuda()
    w13, s13 = random_mxfp4((EXPERTS, 2 * INTER, HIDDEN), g)
    w2, s2 = random_mxfp4((EXPERTS, HIDDEN, INTER), g)
    gu, gu_s = random_mxfp4((2 * INTER, HIDDEN), g)
    dn, dn_s = random_mxfp4((HIDDEN, INTER), g)
    residual = torch.randn(4 * HIDDEN, generator=g).bfloat16().cuda()
    inject = (torch.randn(2, 4, generator=g)).cuda()

    bank = _Bank(w13.clone(), s13.clone(), w2.clone(), s2.clone())
    preprocess_n16_mxfp4_weights(bank)

    scratch = torch.zeros(PLAN.scratch_pairs, dtype=torch.int64, device=dev)
    codes, exponents = mxfp4(x[None])
    fill(scratch, PLAN, "m_xbf", x.view(-1, 2).view(torch.int32).flatten(), TAG)
    fill(scratch, PLAN, "m_x4", codes.view(torch.int32).flatten(), TAG)
    fill(scratch, PLAN, "m_xe", exponents.int().flatten(), TAG)
    fill(scratch, PLAN, "m_inj", inject.flatten(), TAG)
    fill(scratch, PLAN, "w", residual.view(-1, 2).view(torch.int32).flatten(), TAG)
    out = torch.empty(4 * HIDDEN, dtype=torch.bfloat16, device=dev)
    step = torch.full((1,), STEP, dtype=torch.int32, device=dev)
    _moe_kernel[(PLAN.grid,)](
        scratch,
        step,
        (
            router_w,
            gate_w,
            bank.w13_weight,
            bank.w13_weight_scale,
            alternate_gate_up(mxfp4_tiles(gu, HIDDEN)),
            n32_scales(alternate_gate_up(gu_s), HIDDEN),
            bank.w2_weight,
            bank.w2_weight_scale,
            mxfp4_tiles(dn, INTER),
            n32_scales(dn_s, INTER),
        ),
        out,
        None,
        PLAN.key,
        num_warps=8,
        **sched_compile_options(),
    )
    torch.cuda.synchronize()

    # Router.
    logits = (router_w.float() @ x.float()).bfloat16()
    lg = scratch[PLAN.offset("lg") : PLAN.offset("lg") + 256] & 0xFFFFFFFF
    got_logits = lg.to(torch.int32).view(torch.bfloat16)
    torch.testing.assert_close(got_logits.float(), logits.float(), rtol=1e-2, atol=1e-3)

    # Experts, on the kernel's own logits.
    probs = torch.softmax(got_logits.float(), 0)
    top_w, top_e = torch.topk(probs, 10)
    top_w = top_w / top_w.sum()
    x4 = round_mxfp4(x[None])[0]

    def bf(t):
        return t.bfloat16().float()

    def expert(wgu, sgu, wdn, sdn, routed):
        h = bf(dequantize_mxfp4(wgu, sgu) @ x4)
        silu = torch.nn.functional.silu(h[:INTER])
        # TokenSpeed's routed epilogue rounds silu(gate); the shared
        # expert's does not.
        h = bf((bf(silu) if routed else silu) * h[INTER:])
        return bf(dequantize_mxfp4(wdn, sdn) @ round_mxfp4(h[None])[0])

    routed = bf(
        sum(
            top_w[k] * expert(w13[e], s13[e], w2[e], s2[e], True)
            for k, e in enumerate(top_e.tolist())
        )
    )
    shared = expert(gu, gu_s, dn, dn_s, False)
    shared_gate = torch.sigmoid(gate_w.float() @ x.float())
    block = (routed + shared_gate * shared).bfloat16().float()
    mix = 2 * torch.sigmoid(inject.sum(0).bfloat16().float())
    ref = (residual.float().view(4, HIDDEN) + block[None, :] * mix[:, None]).bfloat16()
    torch.testing.assert_close(
        out.view(4, HIDDEN).float(), ref.float(), rtol=2e-2, atol=2e-2
    )

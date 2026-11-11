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

"""Muse Spark QKVG and gate stages: the fused q | k | v | gate projection
of one rank, in two K halves."""

from __future__ import annotations

import muse_spark_reference as ref
import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from monokernel_testing import payloads  # noqa: E402
from tokenspeed_kernel_amd._triton import gl, gluon  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel import mailbox, schedule  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import qkvg  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import (  # noqa: E402
    Plan,
    launch_tag,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.weights import (  # noqa: E402
    bf16_tiles,
)

HIDDEN, QKVG, GATE_ROW = 8192, 2304, 1280


@gluon.jit
def qkvg_kernel(scratch, step, w, a, a_stride, PLAN: gl.constexpr):
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, None)
    task, active = schedule.stage_task(PLAN, "qkvg")
    if active:
        qkvg.qkvg_stage(mb, task, w, a, a_stride, "qkvg")
    task, active = schedule.stage_task(PLAN, "gate")
    if active:
        qkvg.gate_stage(mb, task, w, a, a_stride, "gate")


def _plan(rows: int) -> Plan:
    return Plan(
        grid=256,
        # The launch projects its own q | k | v (no previous launch's qnext).
        stages=qkvg.stages({"qkvg": 0, "gate": 0}, True),
        regions=qkvg.REGIONS,
        rows=rows,
        chain=1,
    )


# 3 rows: the GEMV pads to 4 rows (the padding row repeats the last one).
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 8, 16])
def test_qkvg_halves_sum_to_the_bf16_rounded_fp32_projection(rows):
    g = torch.Generator().manual_seed(rows)
    w = (torch.randn(QKVG, HIDDEN, generator=g) / HIDDEN**0.5).to(torch.bfloat16)
    # A row stride wider than the row checks the row addressing.
    a_full = torch.randn(rows, HIDDEN + 64, generator=g).to(torch.bfloat16)
    a = a_full[:, :HIDDEN]
    plan = _plan(rows)
    scratch = torch.zeros(plan.scratch_pairs, dtype=torch.int64, device="cuda")
    step = 3
    qkvg_kernel[(plan.grid,)](
        scratch,
        torch.full((1,), step, dtype=torch.int32, device="cuda"),
        bf16_tiles(w).cuda(),
        a_full.cuda(),
        a_full.stride(0),
        plan.key,
        num_warps=8,
    )
    torch.cuda.synchronize()
    want, bound = ref.bf16_gemm(a, w)
    tag = launch_tag(step, 0)
    # Every pair the tasks put, in each row's copy of both regions; payloads
    # checks the values' tags. No other pair may be written.
    written = torch.zeros(plan.scratch_pairs, dtype=torch.bool)
    for r in range(rows):
        for region, n in (("qkvg", GATE_ROW), ("gate", QKVG - GATE_ROW)):
            start = plan.offset(region) + r * plan.row_pairs
            written[start : start + 2 * n] = True
    assert not torch.any(scratch.cpu()[~written])
    for r in range(rows):
        halves = []
        for region, n in (("qkvg", GATE_ROW), ("gate", QKVG - GATE_ROW)):
            got = payloads(scratch, plan, region, 2 * n, tag, row=r)
            halves.append(got.view(torch.float32).cpu().view(2, n))
        # The consumers' value: the FP32 sum of the halves, rounded to BF16.
        got = torch.cat(halves, dim=1).sum(0).to(torch.bfloat16)
        # FP32 sums in another order (the halves' sum one more), then one
        # BF16 rounding.
        err = (got.double() - want[r]).abs()
        assert torch.all(err <= bound[r]), float((err / bound[r]).max())

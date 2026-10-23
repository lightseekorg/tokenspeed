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

"""The smallest megakernel built from the shared building blocks, as an
example for a new model's kernels (see docs/design/megakernel.md).

Two stages over the launch's token rows compute ``out = x @ w``:

* ``partial``, ``TASKS`` tasks: task ``t`` reads columns ``CHUNK t ..`` of
  every row (as a weight-streaming stage serves every row in each task) and
  publishes each row's partial sum in mailbox region ``part``.
* ``total``, one task per row: polls its row's partial sums and stores
  their total.
"""

from __future__ import annotations

import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel_amd._triton import aggregate, gl, gluon, triton  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.megakernel import (  # noqa: E402
    mailbox,
    schedule,
    trace,
)
from tokenspeed_kernel_amd.ops.gfx950.megakernel.plan import (  # noqa: E402
    Plan,
    Stage,
    placement,
)

K = gl.constexpr(4096)
TASKS = gl.constexpr(4)
CHUNK = gl.constexpr(1024)  # K // TASKS
NUM_WARPS = 4

# The stage module's part: its mailbox regions (pairs per row) and stages.
REGIONS = (("part", int(TASKS)),)


def stages(bases: dict[str, int], rows: int) -> tuple[Stage, ...]:
    partial, total = placement(bases, ("partial", "total"))
    return (
        Stage("partial", int(TASKS), partial, writes=("part",)),
        Stage("total", rows, total, reads=("part",)),
    )


@aggregate
class Operands:
    """The stages' operands, seen from one row (:meth:`row`)."""

    x: gl.tensor  # FP32 [rows, K]
    w: gl.tensor  # FP32 [K]
    out: gl.tensor  # FP32 [rows]

    @gluon.jit
    def row(self, r):
        return Operands(self.x + r * K, self.w, self.out + r)


@gluon.jit
def partial_stage(mb, task, ops, STAGE: gl.constexpr):
    """Task ``task`` of ``partial`` for every row: rows are the leading
    dimension, so one put publishes all of them."""
    mb.stamp(STAGE, trace.START)
    R: gl.constexpr = schedule.rows(mb.plan)
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 4], [1, 64], [1, gl.num_warps()], [1, 0]
    )
    r = gl.arange(0, triton.next_power_of_2(R), gl.SliceLayout(1, layout))
    c = task * CHUNK + gl.arange(0, CHUNK, gl.SliceLayout(0, layout))
    w = gl.load(ops.w + c)
    x = gl.load(ops.x + r[:, None] * K + c[None, :], mask=(r < R)[:, None], other=0.0)
    sums = gl.sum(x * w[None, :], axis=1)
    mb.stamp(STAGE, trace.COMPUTED)
    # Rows past the launch's rows are padding: a put masks them.
    mb.put("part", mb.row_offset(r) + task, sums, mask=r < R)
    mb.stamp(STAGE, trace.END)


@gluon.jit
def total_stage(mb, ops, STAGE: gl.constexpr):
    """The ``total`` task of the row that ``mb`` and ``ops`` view."""
    mb.stamp(STAGE, trace.START)
    lanes: gl.constexpr = mailbox.lane_layout(gl.num_warps())
    parts = mb.poll("part", gl.arange(0, TASKS, lanes)).to(gl.float32, bitcast=True)
    mb.stamp(STAGE, trace.READY)
    total = gl.sum(parts, axis=0)
    one = gl.arange(0, 1, lanes)
    gl.store(ops.out + one, gl.full([1], 0.0, gl.float32, lanes) + total)
    mb.stamp(STAGE, trace.END)


@gluon.jit
def example_kernel(scratch, step, operands, trace_buf, PLAN: gl.constexpr):
    """Every CTA walks the stages in plan order and runs its tasks of each."""
    table = trace.allocate(trace_buf)
    mb = mailbox.Mailbox(scratch, mailbox.launch_tag(step, 0), PLAN, table)
    ops = Operands(*operands)
    task, active = schedule.stage_task(PLAN, "partial")
    if active:
        partial_stage(mb, task, ops, "partial")
    task, active = schedule.stage_task(PLAN, "total")
    if active:
        row, _, _ = schedule.row_task(PLAN, task, 1)
        total_stage(mb.row(row), ops.row(row), "total")
    trace.flush(table, trace_buf)


def example_plan(rows: int) -> Plan:
    # ``total`` sits on CTAs that ``partial`` does not use, so it polls
    # while ``partial`` computes.
    return Plan(
        grid=8,
        stages=stages({"partial": 0, "total": 4}, rows),
        regions=REGIONS,
        rows=rows,
        chain=1,
    )


@pytest.mark.parametrize("rows", [1, 3])
def test_example_megakernel_computes_each_rows_dot_product(rows):
    g = torch.Generator().manual_seed(rows)
    plan = example_plan(rows)
    # One scratch buffer serves every launch. Nothing clears it, since a new
    # step gives the launch a new tag.
    scratch = torch.zeros(plan.scratch_pairs, dtype=torch.int64, device="cuda")
    for step in (1, 2):
        x = torch.randn(rows, int(K), generator=g).cuda()
        w = torch.randn(int(K), generator=g).cuda()
        out = torch.zeros(rows, device="cuda")
        example_kernel[(plan.grid,)](
            scratch,
            torch.full((1,), step, dtype=torch.int32, device="cuda"),
            (x, w, out),
            None,
            plan.key,
            num_warps=NUM_WARPS,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(out, x @ w, rtol=1e-5, atol=1e-4)

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

"""The device-side view of a launch's :class:`plan.Plan`: its sizes, and
which tasks of each stage this CTA runs.

Every function takes the kernel's ``PLAN`` (a :class:`plan.PlanKey`). Each
result resolves at compile time, except the task indices, which follow from
the CTA's program id.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.megakernel import trace


@gluon.constexpr_function
def grid(plan):
    """CTAs of a launch."""
    return plan.plan.grid


@gluon.constexpr_function
def rows(plan):
    """Token rows a launch serves (``Plan.rows``)."""
    return plan.plan.rows


@gluon.constexpr_function
def chain(plan):
    """Consecutive rows that are one request's tokens (``Plan.chain``)."""
    return plan.plan.chain


@gluon.constexpr_function
def row_pairs(plan):
    """Mailbox pairs between two rows' copies of a region."""
    return plan.plan.row_pairs


@gluon.constexpr_function
def region(plan, name):
    """Pair offset of row 0 of mailbox region ``name``."""
    return plan.plan.offset(name)


@gluon.constexpr_function
def has_stage(plan, name):
    """Whether the plan has stage ``name``."""
    return plan.plan.has_stage(name)


@gluon.constexpr_function
def stage_event(plan, name, point):
    """Trace slot of ``point`` (``trace.START`` .. ``trace.END``)
    of stage ``name``."""
    index = [stage.name for stage in plan.plan.stages].index(name)
    assert 4 * index + point < trace.EVENTS
    return 4 * index + point


@gluon.constexpr_function
def _stage_base(plan, name):
    return plan.plan.stage(name).base


@gluon.constexpr_function
def _stage_tasks(plan, name):
    return plan.plan.stage(name).tasks


@gluon.jit
def stage_task(plan: gl.constexpr, name: gl.constexpr):
    """``(task, active)``: this CTA's task of stage ``name``, a stage of at
    most ``grid`` tasks (one round, see :func:`stage_rounds`)."""
    base: gl.constexpr = _stage_base(plan, name)
    tasks: gl.constexpr = _stage_tasks(plan, name)
    G: gl.constexpr = grid(plan)
    gl.static_assert(tasks <= G, "a stage of more tasks than CTAs runs in rounds")
    task = (gl.program_id(0) + G - base) % G
    return task, task < tasks


@gluon.jit
def stage_rounds(plan: gl.constexpr, name: gl.constexpr):
    """``(task, rounds)``: this CTA runs tasks ``task + i * grid`` of stage
    ``name`` for ``i < rounds`` (0 rounds when it has no task)."""
    base: gl.constexpr = _stage_base(plan, name)
    tasks: gl.constexpr = _stage_tasks(plan, name)
    G: gl.constexpr = grid(plan)
    task = (gl.program_id(0) + G - base) % G
    return task, (tasks - task + G - 1) // G


@gluon.jit
def row_task(plan: gl.constexpr, task, PER_ROW: gl.constexpr):
    """``(row, task, tok)`` of task ``task`` of a stage that runs ``PER_ROW``
    tasks per row: the row, the task within it, and the row's token index in
    its request."""
    row = task // PER_ROW
    return row, task % PER_ROW, row % chain(plan)


@gluon.jit
def head_sleep(
    plan: gl.constexpr, first: gl.constexpr, second: gl.constexpr, units: gl.constexpr
):
    """Delay CTAs with no task in stage ``first`` or ``second`` by ``units``
    ``s_sleep`` units (64 clocks each).

    The first stages of a launch are its critical path. Every other CTA
    issues the later stages' weight streams at once, and these streams would
    compete with the loads of the first stages.
    """
    _, a = stage_rounds(plan, first)
    _, b = stage_rounds(plan, second)
    if not ((a > 0) or (b > 0)):
        for i in gl.static_range(units // 127):
            gl.inline_asm("s_sleep 127")
        if units % 127:
            gl.inline_asm(f"s_sleep {units % 127}")


__all__ = [
    "chain",
    "grid",
    "has_stage",
    "head_sleep",
    "region",
    "row_pairs",
    "row_task",
    "rows",
    "stage_event",
    "stage_rounds",
    "stage_task",
]

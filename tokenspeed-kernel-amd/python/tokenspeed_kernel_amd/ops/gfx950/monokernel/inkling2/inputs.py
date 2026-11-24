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

"""Layer inputs that an earlier kernel computed (``plan.later_pass_plan``).

In a looped core's later passes the regular path gates the previous pass's
message into a layer's inputs and normalizes them between the two launches
of the layer's visit. Their first stages publish those BF16 rows as they are:

* ``in_attn``, one task a (row, source): the Q (0), K (1) or V (2) source
  row into region ``x_attn``, ``x_k`` or ``x_v``;
* ``in_mlp``, one task a row: the MoE input row into region ``x_mlp``.

The rows are in global memory, written by a kernel before the launch, so a
plain load sees them. A region pair ``p`` holds values ``2 p`` and
``2 p + 1``, the row's own 32-bit words.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.monokernel import schedule, trace


@gluon.jit
def _publish_row(mb, x, region: gl.constexpr):
    """Publish the BF16 row ``x`` (global) as the pairs of ``region``."""
    D: gl.constexpr = schedule.dims(mb.plan)
    PAIRS: gl.constexpr = D.hidden // 2
    N: gl.constexpr = triton.next_power_of_2(PAIRS)
    layout: gl.constexpr = gl.BlockedLayout(
        [N // (64 * gl.num_warps())], [64], [gl.num_warps()], [0]
    )
    p = gl.arange(0, N, layout)
    live = p < PAIRS
    words = gl.load(x.to(gl.pointer_type(gl.int32), bitcast=True) + p, mask=live)
    mb.put(region, p, words, mask=live)


@gluon.jit
def attention_input_stage(mb, task, sources, STAGE: gl.constexpr):
    """Task ``task`` = (row, source) of ``in_attn``: ``sources`` holds the Q,
    K and V source rows (BF16 ``[rows, hidden]`` each)."""
    D: gl.constexpr = schedule.dims(mb.plan)
    row = task // 3
    j = task % 3
    view = mb.row(row)
    view.stamp(STAGE, trace.START)
    offset = row.to(gl.int64) * D.hidden
    if j == 0:
        _publish_row(view, sources[0] + offset, "x_attn")
    elif j == 1:
        _publish_row(view, sources[1] + offset, "x_k")
    else:
        _publish_row(view, sources[2] + offset, "x_v")
    view.stamp(STAGE, trace.END)


@gluon.jit
def mlp_input_stage(mb, row, source, STAGE: gl.constexpr):
    """Row ``row`` of ``in_mlp``: ``source`` is the MoE input (BF16
    ``[rows, hidden]``)."""
    D: gl.constexpr = schedule.dims(mb.plan)
    view = mb.row(row)
    view.stamp(STAGE, trace.START)
    _publish_row(view, source + row.to(gl.int64) * D.hidden, "x_mlp")
    view.stamp(STAGE, trace.END)

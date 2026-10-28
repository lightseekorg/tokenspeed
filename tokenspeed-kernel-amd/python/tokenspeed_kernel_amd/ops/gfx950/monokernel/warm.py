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

"""Warm the next launch's weights in the memory-side cache.

A layer's first projection streams its weights from the start of the
launch, but on the mixer's CTAs only after the mixer: those tasks end last.
The previous launch has phases with little HBM traffic. A warm task in such
a phase reads the weights of those late tasks, and the next launch then
reads them from the memory-side cache (MALL), which keeps them through the
rest of the launch's reads.

A warm task loads into an LDS buffer that no stage reads, so the loads take
no registers. Loads complete in order: a warm task delays the loads of the
CTA's next task. Put it on CTAs whose next task starts late.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import cdna4_async_copy, gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.monokernel import trace

# int32 words a lane loads per step (16 bytes).
_LANE_WORDS = gl.constexpr(4)


@gluon.jit
def warm_stage(mb, task, words_ptr, quads, TASKS: gl.constexpr, STAGE: gl.constexpr):
    """Warm task ``task`` of ``TASKS``: read its share of the ``4 * quads``
    int32 words at ``words_ptr`` (the shares are even, the last one shorter)
    and wait until the reads land."""
    mb.stamp(STAGE, trace.START)
    NW: gl.constexpr = gl.num_warps()
    BLOCK: gl.constexpr = NW * 64 * _LANE_WORDS
    layout: gl.constexpr = gl.BlockedLayout([_LANE_WORDS], [64], [NW], [0])
    sink = gl.allocate_shared_memory(
        gl.int32, [BLOCK], gl.SwizzledSharedLayout(1, 1, 1, [0])
    )
    words = quads * _LANE_WORDS
    share = (words + TASKS * BLOCK - 1) // (TASKS * BLOCK) * BLOCK
    offsets = task * share + gl.arange(0, BLOCK, layout)
    for i in range(0, share, BLOCK):
        o = offsets + i
        # The mask holds for a lane's whole 16-byte load. Default caching: a
        # non-temporal load would not fill the MALL.
        cdna4_async_copy.buffer_load_to_shared(
            sink, words_ptr, o, mask=o // _LANE_WORDS < quads
        )
    mb.stamp(STAGE, trace.READY)
    cdna4_async_copy.wait_group(0)
    mb.stamp(STAGE, trace.COMPUTED)
    mb.stamp(STAGE, trace.END)


__all__ = ["warm_stage"]

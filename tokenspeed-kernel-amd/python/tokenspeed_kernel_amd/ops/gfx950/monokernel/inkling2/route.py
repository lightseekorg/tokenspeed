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

"""Expert selection of a row (TokenSpeed ``_topk`` of the Inkling2 MoE).

Every rank routes every row over all ``E`` experts (the plan's
``dims.routed``: 384 for Inkling 2 Small, 128 at half its width):

* selection score ``sigmoid(l_e) + bias_e`` (``-0.0`` as ``+0.0``); the top 8
  in descending score, a tie to the lower expert id;
* weight ``8 sigmoid(l_k) / sum_j sigmoid(l_j)`` over the selected experts,
  in log space from the raw logits (the bias does not enter):
  ``ls = min(l, 0) - log1p(exp(-|l|))``, ``w = 8 exp(ls - logsumexp(ls))``.

The route region of a row holds the 8 expert ids, then the 8 FP32 weights,
the bits of ``_topk`` (TokenSpeed's decode top-k) on the same logits. The
selection is exact (a max of keys), and the selected logit is the one
nonzero of a sum with +0.0s; the weights reduce over the 8 slots, which
``_topk`` lays out the same at any ``E`` (one slot a lane).
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.monokernel import schedule, trace, unfused

TOP_K = gl.constexpr(8)
ROUTE_SCALE = gl.constexpr(8.0)


@gluon.jit
def _ordered(score):
    """int32 of FP32 ``score`` whose signed order is the values' order."""
    bits = score.to(gl.int32, bitcast=True)
    return gl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)


@gluon.jit
def topk_stage(
    mb, bias, logits: gl.constexpr, route: gl.constexpr, STAGE: gl.constexpr
):
    """The top-k task of the row that ``mb`` views (``bias``: FP32 ``[E]``,
    ``E`` the plan's routed experts)."""
    mb.stamp(STAGE, trace.START)
    EXPERTS: gl.constexpr = schedule.dims(mb.plan).routed
    # Keys of the experts in one tile; past E they never win.
    BLOCK: gl.constexpr = triton.next_power_of_2(EXPERTS)
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    e = gl.arange(0, BLOCK, lanes)
    live = e < EXPERTS
    b = gl.load(bias + e, mask=live, other=0.0)
    got = mb.poll(logits, gl.minimum(e, EXPERTS - 1))
    mb.stamp(STAGE, trace.READY)
    logit = got.to(gl.float32, bitcast=True)
    # The transcendentals of ``_topk`` (a launch without FP fusion), bit for
    # bit: libdevice's exp and log, and log1p in its unfused steps.
    exp: gl.constexpr = gl.extra.libdevice.exp
    score = gl.div_rn(1.0, 1.0 + exp(-logit)) + b + 0.0
    key = (_ordered(score).to(gl.int64) << 32) | (BLOCK - e).to(gl.int64)
    key = gl.where(live, key, -(2**62))
    k = gl.arange(0, TOP_K, lanes)
    ids = gl.zeros([TOP_K], gl.int32, lanes)
    raw = gl.zeros([TOP_K], gl.float32, lanes)
    for i in gl.static_range(TOP_K):
        best = gl.max(key, axis=0)
        expert = BLOCK - (best & 0xFFFFFFFF).to(gl.int32)
        chosen = gl.sum(gl.where(e == expert, logit, 0.0), axis=0)
        ids = gl.where(k == i, expert, ids)
        raw = gl.where(k == i, chosen, raw)
        key = gl.where(e == expert, -(2**62), key)
    ls = gl.minimum(raw, 0.0) - unfused.log1p(exp(-gl.abs(raw)))
    top = gl.max(ls, axis=0)
    lse = top + gl.extra.libdevice.log(gl.sum(exp(ls - top), axis=0))
    weight = exp(ls - lse) * ROUTE_SCALE
    mb.put(route, k, ids)
    mb.put(route, TOP_K + k, weight.to(gl.int32, bitcast=True))
    mb.stamp(STAGE, trace.END)

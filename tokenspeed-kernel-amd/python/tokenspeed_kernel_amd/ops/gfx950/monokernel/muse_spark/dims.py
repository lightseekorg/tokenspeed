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

"""Model dimensions of one rank of eight (TP8)."""

from tokenspeed_kernel_amd._triton import gl

POST_EPS = gl.constexpr(1e-8)
PRE_EPS = gl.constexpr(1e-5)

WORLD = gl.constexpr(8)
HIDDEN = gl.constexpr(8192)
OWNED = gl.constexpr(HIDDEN // WORLD)  # hidden columns a rank owns
HEAD_DIM = gl.constexpr(64)
Q_HEADS = gl.constexpr(16)  # 128 / 8
KV_HEADS = gl.constexpr(2)  # 16 / 8
Q = gl.constexpr(Q_HEADS * HEAD_DIM)  # the q, output gate and attention widths
KV = gl.constexpr(KV_HEADS * HEAD_DIM)
# The fused projection's rows, in QKVOGateParallelLinear's order: q | k | v | gate.
K_ROW = gl.constexpr(Q)
V_ROW = gl.constexpr(K_ROW + KV)
GATE_ROW = gl.constexpr(V_ROW + KV)
QKVG = gl.constexpr(GATE_ROW + Q)
POOLED = gl.constexpr(4096)  # the pooled (pre-expert) width
EXPERTS = gl.constexpr(256)  # routed experts
# The DFlash2 draft model, a rank of its TP8 group.
DRAFT_HEAD_DIM = gl.constexpr(128)
DRAFT_Q_HEADS = gl.constexpr(8)  # query heads a rank; one KV head
DRAFT_BLOCK = gl.constexpr(8)  # rows of a draft block: one request's verify

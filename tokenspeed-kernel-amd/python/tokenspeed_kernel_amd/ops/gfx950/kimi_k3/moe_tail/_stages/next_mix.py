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


"""The next layer's attention input, from this layer's output rows.

The arithmetic matches the regular path's AttnRes combine
(``attnres_combine``): two 4096-value chunks a row, its reduction order and
its roundings. The final reducers stay the only consumers of the exchange
words: each mirrors its output rows into scratch and publishes a ready word.
"""

from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.common import (
    descriptor,
    thread_ids,
    wave_pending,
)
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.polls import poll_pairs


@gluon.jit
def _split_eight(values):
    even, odd = gl.split(gl.reshape(values, [512, 4, 2]))
    e0, e2 = gl.split(gl.reshape(even, [512, 2, 2]))
    o1, o3 = gl.split(gl.reshape(odd, [512, 2, 2]))
    x0, x4 = gl.split(e0)
    x2, x6 = gl.split(e2)
    x1, x5 = gl.split(o1)
    x3, x7 = gl.split(o3)
    return x0, x1, x2, x3, x4, x5, x6, x7


@gluon.jit
def _combine_order_dot(x, y):
    # ``attnres_combine`` contracts the first register-pair additions.
    # Explicit FMAs keep those roundings, since the tail compiles with
    # floating-point fusion disabled.
    x0, x1, x2, x3, x4, x5, x6, x7 = _split_eight(x)
    y0, y1, y2, y3, y4, y5, y6, y7 = _split_eight(y)
    even = gl.fma(x0, y0, x2 * y2) + gl.fma(x4, y4, x6 * y6)
    odd = gl.fma(x1, y1, x3 * y3) + gl.fma(x5, y5, x7 * y7)
    local = gl.convert_layout(even + odd, gl.BlockedLayout([1], [64], [8], [0]))
    return gl.sum(local, 0)


@gluon.jit
def stage_next_mix(
    row_buffer,
    ready,
    mix_wp,
    mix_nw,
    mix_m,
    mix_s,
    mix_acc,
    mixed,
    row,
    tag,
    MIX_EPS: gl.constexpr,
    guard,
):
    layout: gl.constexpr = gl.BlockedLayout([8], [64], [8], [0])
    col0 = gl.arange(0, 4096, layout=layout)
    col1 = 4096 + col0
    mask0 = col0 < 7168
    mask1 = col1 < 7168
    offset = row * 7168
    # These operands are complete before the tail starts; overlap their loads with
    # the existing final reduction instead of extending earlier live ranges.
    wp0 = gl.amd.cdna3.buffer.load(descriptor(mix_wp), col0, mask0, 0.0)
    wp1 = gl.amd.cdna3.buffer.load(descriptor(mix_wp), col1, mask1, 0.0)
    ow0 = gl.amd.cdna3.buffer.load(descriptor(mix_nw), col0, mask0, 0.0)
    ow1 = gl.amd.cdna3.buffer.load(descriptor(mix_nw), col1, mask1, 0.0)
    m_b = gl.load(mix_m + row)
    s_b = gl.load(mix_s + row)
    a0 = gl.amd.cdna3.buffer.load(descriptor(mix_acc), offset + col0, mask0, 0.0)
    a1 = gl.amd.cdna3.buffer.load(descriptor(mix_acc), offset + col1, mask1, 0.0)
    tid = thread_ids()
    # Only the first wave polls. A lane mask alone does not narrow EXEC.
    # System coherence forces sc0 sc1 on both the initial and retry loads.
    if wave_pending(tid < 56):
        poll_pairs(ready, (gl.minimum(tid, 55),), tag, tid < 56, "sys", guard)
    # Join the seven non-polling waves before any of them reads the mirror.
    gl.barrier()
    v0 = gl.amd.cdna4.buffer_load(
        row_buffer.to(gl.pointer_type(gl.bfloat16)),
        offset + col0,
        mask=mask0,
        other=0.0,
        cache=".cv",
    ).to(gl.float32)
    v1 = gl.amd.cdna4.buffer_load(
        row_buffer.to(gl.pointer_type(gl.bfloat16)),
        offset + col1,
        mask=mask1,
        other=0.0,
        cache=".cv",
    ).to(gl.float32)
    square_sum = _combine_order_dot(v0, v0) + _combine_order_dot(v1, v1)
    dot = _combine_order_dot(v0, wp0.to(gl.float32)) + _combine_order_dot(
        v1, wp1.to(gl.float32)
    )
    logit = dot * gl.rsqrt(square_sum / 7168 + MIX_EPS)
    maximum = gl.maximum(m_b, logit)
    correction = gl.exp(m_b - maximum)
    weight = gl.exp(logit - maximum)
    inverse_sum = 1.0 / gl.fma(s_b, correction, weight)
    mix0 = (gl.fma(a0, correction, weight * v0) * inverse_sum).to(gl.bfloat16)
    mix1 = (gl.fma(a1, correction, weight * v1) * inverse_sum).to(gl.bfloat16)
    mix0 = mix0.to(gl.float32)
    mix1 = mix1.to(gl.float32)
    mix_square_sum = _combine_order_dot(mix0, mix0) + _combine_order_dot(mix1, mix1)
    inverse_rms = gl.rsqrt(gl.div_rn(mix_square_sum, 7168) + MIX_EPS)
    gl.amd.cdna3.buffer.store(
        (mix0 * inverse_rms * ow0.to(gl.float32)).to(gl.bfloat16),
        descriptor(mixed),
        offset + col0,
        mask=mask0,
    )
    gl.amd.cdna3.buffer.store(
        (mix1 * inverse_rms * ow1.to(gl.float32)).to(gl.bfloat16),
        descriptor(mixed),
        offset + col1,
        mask=mask1,
    )

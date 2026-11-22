# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright 2026 SGLang contributors
#
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


"""The quantized latent rows' acquisition and the latent RMS norm.

Modified from sgl-amd-mono (https://github.com/sgl-project/sgl-amd-mono, Apache
License 2.0) at b9f40eb562f449ce08c030df200f97c7ea716511:
``kimi_k3/gfx950/stages/moe.py``.
"""

from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.common import (
    bf16_round,
    butterfly,
    descriptor,
    publish,
    put_pair,
    thread_ids,
)
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.final import bf16_pair
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.lds import (
    load_f32,
    store_mask_i32,
)
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.polls import poll_pairs
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.sentinel import (
    rearm_sent,
    sent_ix,
    sent_sum,
)


@gluon.jit
def load_latq(region, workspace, tag, S: gl.constexpr, guard):
    tid = thread_ids()
    indices = ()
    for i in gl.static_range(gl.cdiv(S * 560, 512)):
        indices += (gl.minimum(tid + 512 * i, S * 560 - 1),)
    got = poll_pairs(
        region,
        indices,
        tag,
        gl.full([512], True, gl.int1, tid.type.layout),
        "sys",
        guard,
    )
    for i in gl.static_range(gl.cdiv(S * 560, 512)):
        item = tid + 512 * i
        clamped = gl.minimum(item, S * 560 - 1)
        token, word = clamped // 560, clamped % 560
        at = gl.where(
            word < 448, token * 448 + word, S * 448 + token * 112 + word - 448
        )
        store_mask_i32(workspace, at, got[i].to(gl.int32, bitcast=True), item < S * 560)
    gl.barrier()


@gluon.jit
def stage_lnorm(
    latr,
    ln_w,
    ln,
    lmsq,
    lnrdy,
    red,
    group,
    tag,
    epoch,
    own,
    peers,
    rank,
    S: gl.constexpr,
    EPS: gl.constexpr,
    FLAGS_OFFSET: gl.constexpr,
):
    tid = thread_ids()
    live = tid < S * 64
    token, lane = gl.minimum(tid // 64, S - 1), tid % 64
    col = group * 128 + lane * 2
    lo, hi = sent_sum(
        latr,
        col,
        token,
        live,
        epoch,
        own,
        peers,
        rank,
        S,
        3584,
        FLAGS_OFFSET,
        13,
        False,
    )
    lo, hi = bf16_round(lo), bf16_round(hi)
    square = butterfly(lo * lo + hi * hi, (32, 16, 8, 4, 2, 1))
    put_pair(lmsq, group * S + token, square, tag, live & (lane == 0), "gpu")
    gl.barrier()
    value = poll_pairs(
        lmsq,
        (lane * S + token,),
        tag,
        live & (lane < 28),
        "gpu",
        (epoch, own, peers, rank, FLAGS_OFFSET, 15),
    )[0]
    store_mask_i32(
        red, token * 32 + lane, value.to(gl.int32, bitcast=True), live & (lane < 28)
    )
    gl.barrier()
    total = load_f32(red, token * 32)
    for q in gl.static_range(1, 28):
        total = total + load_f32(red, token * 32 + q)
    inverse = gl.rsqrt(total * (1.0 / 3584.0) + EPS)
    desc = descriptor(ln_w)
    w0 = gl.amd.cdna3.buffer.load(desc, col).to(gl.float32)
    w1 = gl.amd.cdna3.buffer.load(desc, col + 1).to(gl.float32)
    word = bf16_pair((lo * inverse) * w0, (hi * inverse) * w1)
    gl.amd.cdna3.buffer.store(
        word,
        descriptor(ln.to(gl.pointer_type(gl.uint32))),
        (token * 3584 + col) // 2,
        mask=live,
        scope="gpu",
    )
    publish(
        lnrdy,
        gl.full([512], 0, gl.int32, tid.type.layout) + group,
        gl.full([512], 1, gl.int32, tid.type.layout),
        tag,
        tid == 0,
    )
    indices = ()
    for src in gl.static_range(8):
        indices += (sent_ix(src, col, token, S, 3584),)
    rearm_sent(latr, indices, live)
    gl.barrier()

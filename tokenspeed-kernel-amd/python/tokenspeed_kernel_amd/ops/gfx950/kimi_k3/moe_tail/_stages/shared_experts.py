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


"""The shared expert's SiTU activation and down projection.

Modified from sgl-amd-mono (https://github.com/sgl-project/sgl-amd-mono, Apache
License 2.0) at b9f40eb562f449ce08c030df200f97c7ea716511:
``kimi_k3/gfx950/stages/moe.py``.
"""

from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.common import (
    bf16_round,
    descriptor,
    gemv_dot,
    gemv_loads,
    gemv_reduce,
    publish,
    put_pair,
    thread_ids,
)
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.final import (
    bf16_pair,
    put_final,
)
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.lds import store_mask_i32
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.polls import poll_pairs


@gluon.jit
def tanh_libm(value):
    exponential = gl.exp2(gl.abs(value) * -2.8853900817779268)
    magnitude = gl.div_rn(1.0 - exponential, 1.0 + exponential)
    return gl.where(value < 0.0, -magnitude, magnitude)


@gluon.jit
def situ(gate, up):
    value = 2.0 * tanh_libm(gate * 0.25)
    value = value * (1.0 + tanh_libm(gate * 0.5))
    return value * (25.0 * tanh_libm(up * (1.0 / 25.0)))


@gluon.jit
def stage_h(sgu, h, token, tag, guard):
    tid = thread_ids()
    got = poll_pairs(
        sgu, (token * 768 + tid, token * 768 + 384 + tid), tag, tid < 384, "gpu", guard
    )
    gate = got[0].to(gl.uint32, bitcast=True)
    up = got[1].to(gl.uint32, bitcast=True)
    g0 = (gate << 16).to(gl.float32, bitcast=True)
    g1 = (gate & 0xFFFF0000).to(gl.float32, bitcast=True)
    u0 = (up << 16).to(gl.float32, bitcast=True)
    u1 = (up & 0xFFFF0000).to(gl.float32, bitcast=True)
    lo = situ(g0, u0).to(gl.bfloat16).to(gl.uint16, bitcast=True).to(gl.uint32)
    hi = situ(g1, u1).to(gl.bfloat16).to(gl.uint16, bitcast=True).to(gl.uint32)
    put_pair(h, token * 384 + tid, lo | (hi << 16), tag, tid < 384, "gpu")
    gl.barrier()


@gluon.jit
def load_h(h, xl, tag, S: gl.constexpr, guard):
    tid = thread_ids()
    N: gl.constexpr = S * 384
    indices = ()
    for i in gl.static_range(gl.cdiv(N, 512)):
        indices += (gl.minimum(tid + 512 * i, N - 1),)
    got = poll_pairs(
        h, indices, tag, gl.full([512], True, gl.int1, tid.type.layout), "gpu", guard
    )
    for i in gl.static_range(gl.cdiv(N, 512)):
        item = tid + 512 * i
        clamped = gl.minimum(item, N - 1)
        at = clamped // 384 * 388 + clamped % 384
        store_mask_i32(xl, at, got[i].to(gl.int32, bitcast=True), item < N)
    gl.barrier()


@gluon.jit
def stage_sdown(
    xl, red, shp, shrdy, final_regions, rank, group, ops, tag, S: gl.constexpr
):
    acc = gemv_dot(xl, ops, 768, S, False)
    gemv_reduce(acc, red, S)
    tid = thread_ids()
    token, pair = gl.minimum(tid // 8, S - 1), tid % 8
    # Keep adjacent columns together and preserve the wave-by-wave fold order.
    PAIRS: gl.constexpr = gl.BlockedLayout([2], [64], [8], [0])
    item = gl.arange(0, 1024, layout=PAIRS)
    pair_tid = item // 2
    pair_token = gl.minimum(pair_tid // 8, S - 1)
    column = (pair_tid % 8) * 2
    base = gl.multiple_of(column // 4 * 64 + pair_token * 4 + column % 4, 2)
    partials = ()
    for wave in gl.static_range(8):
        indices = gl.max_contiguous(base + wave * 256 + item % 2, 2)
        partials += (red.gather(indices, 0),)
    values = gl.full([1024], 0.0, gl.float32, PAIRS)
    for wave in gl.static_range(8):
        values = values + partials[wave]
    lo, hi = gl.split(gl.reshape(values, [512, 2]))
    lo = bf16_round(gl.convert_layout(lo, tid.type.layout, assert_trivial=True))
    hi = bf16_round(gl.convert_layout(hi, tid.type.layout, assert_trivial=True))
    mine = (group >= rank * 56) & (group < rank * 56 + 56)
    live = tid < S * 8
    if mine:
        at = (token * 896 + (group - rank * 56) * 16 + pair * 2) // 2
        gl.amd.cdna3.buffer.store(
            bf16_pair(lo, hi),
            descriptor(shp.to(gl.pointer_type(gl.uint32))),
            at,
            mask=live,
            scope="gpu",
        )
    else:
        put_final(final_regions, rank, group * 16 + pair * 2, token, lo, hi, live, S)
    publish(
        shrdy,
        gl.full([512], 0, gl.int32, tid.type.layout)
        + gl.minimum(gl.maximum(group - rank * 56, 0), 55),
        gl.full([512], 1, gl.int32, tid.type.layout),
        tag,
        (tid == 0) & mine,
    )
    gl.barrier()


@gluon.jit
def run_sdown_first(
    w_sd, h, shp, shrdy, xl, red, final_regions, rank, bid, tag, S: gl.constexpr, guard
):
    if bid >= 224:
        k0 = bid - 224
        k1 = gl.minimum(k0 + 32, 55)
        j0, j1 = (k0 + rank * 56) % 448, (k1 + rank * 56) % 448
        ops0 = gemv_loads(w_sd, j0 * 16, 768, True, False)
        ops1 = gemv_loads(w_sd, j1 * 16, 768, True, False)
        load_h(h, xl, tag, S, guard)
        stage_sdown(xl, red, shp, shrdy, final_regions, rank, j0, ops0, tag, S)
        if k0 + 32 < 56:
            stage_sdown(xl, red, shp, shrdy, final_regions, rank, j1, ops1, tag, S)


@gluon.jit
def run_sdown_rest(
    w_sd, h, shp, shrdy, xl, red, final_regions, rank, bid, tag, S: gl.constexpr, guard
):
    if bid >= 84:
        load_h(h, xl, tag, S, guard)
        k = 56 + bid - 84
        while k < 448:
            group = (k + rank * 56) % 448
            ops = gemv_loads(w_sd, group * 16, 768, True, False)
            stage_sdown(xl, red, shp, shrdy, final_regions, rank, group, ops, tag, S)
            k += 172

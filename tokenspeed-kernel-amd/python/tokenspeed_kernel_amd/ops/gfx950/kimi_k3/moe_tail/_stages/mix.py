# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright 2026 SGLang contributors
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (C) 2026, Advanced Micro Devices, Inc.
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


"""The MLP attention-residual mix and the post-attention norm.

Each of 56 workgroups owns a 128-column slice. It adds the reduced attention
output to the running prefix, then publishes, for every row and source (the
completed blocks and the prefix), the slice's sum of squares and its dot
product with the residual norm and projection weights. The first workgroups
turn them into each row's softmax weights; every slice then mixes its
columns and applies the post-attention RMS norm.

Modified from sgl-amd-mono (https://github.com/sgl-project/sgl-amd-mono, Apache
License 2.0) at b9f40eb562f449ce08c030df200f97c7ea716511: ``common/ops.py`` and
``kimi_k3/gfx950/layer.py``, whose device primitives are adapted from ROCm/ATOM
(https://github.com/ROCm/ATOM, MIT License) at
97359d6df46c59736468168b7be8bc50b482a1a9.
"""

from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.common import (
    bf16_round,
    butterfly,
    descriptor,
    publish,
    put_pair,
    raw_bf4,
    store_bf4,
    thread_ids,
    unpack_bf4,
)
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.polls import poll_pairs


@gluon.jit
def slice_loads(
    prefix_in,
    blocks,
    ares_nw,
    ares_qk,
    task,
    blk_sm,
    blk_sr,
    S: gl.constexpr,
    NB: gl.constexpr,
):
    """Issue the slice's block, prefix and weight loads, kept packed until
    the reduced attention output is ready."""
    tid = thread_ids()
    blks, pres = (), ()
    for i in gl.static_range(gl.cdiv(S * (NB + 1) * 32, 512)):
        item = gl.minimum(tid + 512 * i, S * (NB + 1) * 32 - 1)
        row = item // 32
        t, src = row // (NB + 1), row % (NB + 1)
        col = task * 128 + item % 32 * 4
        blks += (
            raw_bf4(
                blocks,
                t.to(gl.int64) * blk_sm
                + gl.minimum(src, NB - 1).to(gl.int64) * blk_sr
                + col,
            ),
        )
        pres += (raw_bf4(prefix_in, t.to(gl.int64) * 7168 + col),)
    first = gl.minimum(tid, S * (NB + 1) * 32 - 1)
    col = task * 128 + first % 32 * 4
    return blks, pres, raw_bf4(ares_nw, col), raw_bf4(ares_qk, col)


@gluon.jit
def stage_slice_delta(
    prefix,
    part,
    vals,
    delta_lds,
    task,
    tag,
    S: gl.constexpr,
    NB: gl.constexpr,
    RESET: gl.constexpr,
    preloads,
):
    """Update the slice's prefix with the reduced attention output, then
    publish its partials.

    ``delta_lds`` holds S by 128 FP32, already rounded through BF16, and
    ``preloads`` is :func:`slice_loads`'s packed tuple. A ``RESET`` layer's
    prefix restarts from the attention output.
    """
    tid = thread_ids()
    NS: gl.constexpr = NB + 1
    N: gl.constexpr = S * NS * 32
    L: gl.constexpr = gl.BlockedLayout([1, 2], [64, 1], [8, 1], [0, 1])
    wn, qk = unpack_bf4(preloads[2]), unpack_bf4(preloads[3])
    for i in gl.static_range(gl.cdiv(N, 512)):
        item = gl.minimum(tid + 512 * i, N - 1)
        row = item // 32
        t, src = row // NS, row % NS
        col = task * 128 + item % 32 * 4
        live = tid + 512 * i < N
        bv, pv = unpack_bf4(preloads[0][i]), unpack_bf4(preloads[1][i])
        v, raw, pre = (), (), ()
        for j in gl.static_range(4):
            dv = delta_lds.gather(t * 128 + item % 32 * 4 + j, 0)
            p = dv if RESET else bf16_round(pv[j] + dv)
            pre += (p,)
            value = gl.where(src == NB, p, bv[j])
            raw += (value,)
            v += (gl.where(live, value, 0),)
        p0 = pre[0].to(gl.bfloat16).to(gl.uint16, bitcast=True).to(gl.uint32) | (
            pre[1].to(gl.bfloat16).to(gl.uint16, bitcast=True).to(gl.uint32) << 16
        )
        p1 = pre[2].to(gl.bfloat16).to(gl.uint16, bitcast=True).to(gl.uint32) | (
            pre[3].to(gl.bfloat16).to(gl.uint16, bitcast=True).to(gl.uint32) << 16
        )
        words = gl.convert_layout(gl.join(p0, p1), L)
        at = gl.convert_layout(t * 3584 + col // 2, gl.SliceLayout(1, L))
        dd = gl.arange(0, 2, layout=gl.SliceLayout(0, L))
        mask = gl.convert_layout(live & (src == NB), gl.SliceLayout(1, L))
        gl.amd.cdna3.buffer.store(
            words,
            descriptor(prefix.to(gl.pointer_type(gl.uint32))),
            at[:, None] + dd[None, :],
            mask=mask[:, None],
            scope="gpu",
        )
        packed = gl.reshape(
            gl.join(gl.join(raw[0], raw[2]), gl.join(raw[1], raw[3])), [2048]
        )
        vals.slice(i * 2048, 2048).store(packed)
        sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]
        dt = (
            v[0] * (wn[0] * qk[0])
            + v[1] * (wn[1] * qk[1])
            + v[2] * (wn[2] * qk[2])
            + v[3] * (wn[3] * qk[3])
        )
        sq, dt = butterfly(sq, (16, 8, 4, 2, 1)), butterfly(dt, (16, 8, 4, 2, 1))
        at = ((task * S + t) * NS + src) * 2
        put_pair(part, at, sq, tag, live & (tid % 32 == 0), "gpu")
        put_pair(part, at + 1, dt, tag, live & (tid % 32 == 0), "gpu")
    gl.barrier()


@gluon.jit
def stage_gate(
    part,
    wgt,
    red,
    token,
    tag,
    S: gl.constexpr,
    NB: gl.constexpr,
    EPS: gl.constexpr,
    guard,
):
    """Turn a row's partials into its sources' softmax weights and their sum."""
    tid = thread_ids()
    NS: gl.constexpr = NB + 1
    NQ: gl.constexpr = NS * 2
    g, q = tid // NQ, tid % NQ
    idx = ()
    for k in gl.static_range(7):
        idx += ((((g * 7 + k) * S + token) * NS) * 2 + q,)
    got = poll_pairs(part, idx, tag, tid < 8 * NQ, "gpu", guard)
    total = got[0]
    for k in gl.static_range(1, 7):
        total = total + got[k]
    red.scatter(total, gl.where(tid < 8 * NQ, tid, 1024 + tid), 0)
    gl.barrier()
    lane = gl.arange(0, 64, layout=gl.BlockedLayout([1], [64], [8], [0]))
    src = gl.minimum(lane, NS - 1)
    sq = red.gather(src * 2, 0)
    dt = red.gather(src * 2 + 1, 0)
    for g in gl.static_range(1, 8):
        sq = sq + red.gather(g * NQ + src * 2, 0)
        dt = dt + red.gather(g * NQ + src * 2 + 1, 0)
    rs = gl.rsqrt(sq * (1.0 / 7168) + EPS)
    logit = gl.where(lane < NS, dt * rs * 1.4426950408889634, -3.0e38)
    maximum = gl.max(logit, 0)
    ex = gl.where(lane < NS, gl.exp2(logit - maximum), 0)
    den = ex
    for off in gl.static_range(6):
        den = den + gl.gather(den, lane ^ (32 >> off), 0)
    put_pair(wgt, token * (NS + 1) + lane, ex, tag, lane < NS, "gpu")
    put_pair(
        wgt,
        token * (NS + 1) + NS + gl.full([64], 0, gl.int32, lane.type.layout),
        den,
        tag,
        lane == 0,
        "gpu",
    )
    gl.barrier()


@gluon.jit
def stage_mix(
    wgt,
    msq,
    xrdy,
    vals,
    red,
    in_nw,
    xbuf,
    task,
    tag,
    S: gl.constexpr,
    NB: gl.constexpr,
    OUT_EPS: gl.constexpr,
    xt,
    guard,
):
    """Mix the slice's columns by the source weights, then apply the
    post-attention RMS norm; write the rows as BF16 (``xbuf``) and as tagged
    pairs (``xt``)."""
    tid = thread_ids()
    NS: gl.constexpr = NB + 1
    got = poll_pairs(wgt, (tid,), tag, tid < S * (NS + 1), "gpu", guard)[0]
    red.scatter(got, gl.where(tid < S * (NS + 1), tid, 1024 + tid), 0)
    gl.barrier()
    token = gl.minimum(tid // 32, S - 1)
    col = tid % 32 * 4
    mixed = ()
    for j in gl.static_range(4):
        value = gl.full([512], 0, gl.float32, tid.type.layout)
        for src in gl.static_range(NS):
            value = value + red.gather(token * (NS + 1) + src, 0) * vals.gather(
                (token * NS + src) * 128 + col + j, 0
            )
        mixed += (gl.div_rn(value, red.gather(token * (NS + 1) + NS, 0)),)
    sq = (
        mixed[0] * mixed[0]
        + mixed[1] * mixed[1]
        + mixed[2] * mixed[2]
        + mixed[3] * mixed[3]
    )
    sq = butterfly(sq, (16, 8, 4, 2, 1))
    put_pair(msq, task * S + token, sq, tag, (tid < S * 32) & (tid % 32 == 0), "gpu")
    gl.barrier()
    g, tt = tid // S, tid % S
    idx = ()
    for k in gl.static_range(7):
        idx += (((g * 7 + k) * S + tt),)
    got = poll_pairs(msq, idx, tag, tid < 8 * S, "gpu", guard)
    total = got[0]
    for k in gl.static_range(1, 7):
        total = total + got[k]
    red.scatter(total, gl.where(tid < 8 * S, 128 + tid, 1024 + tid), 0)
    gl.barrier()
    total = red.gather(128 + token, 0)
    for g in gl.static_range(1, 8):
        total = total + red.gather(128 + g * S + token, 0)
    rs = gl.rsqrt(total * (1.0 / 7168) + OUT_EPS)
    xvalues = ()
    norms = unpack_bf4(raw_bf4(in_nw, task * 128 + col))
    for j in gl.static_range(4):
        x = mixed[j] * rs * norms[j]
        xvalues += (x.to(gl.bfloat16).to(gl.uint16, bitcast=True).to(gl.uint32),)
    word0, word1 = xvalues[0] | (xvalues[1] << 16), xvalues[2] | (xvalues[3] << 16)
    base = token * 7168 + task * 128 + col
    at = base // 2
    put_pair(xt, at, word0, tag, tid < S * 32, "gpu")
    put_pair(xt, at + 1, word1, tag, tid < S * 32, "gpu")
    store_bf4(xbuf, base, word0, word1, tid < S * 32, "gpu")
    publish(
        xrdy,
        gl.full([512], 0, gl.int32, tid.type.layout) + task,
        gl.full([512], 1, gl.int32, tid.type.layout),
        tag,
        tid == 0,
    )
    gl.barrier()

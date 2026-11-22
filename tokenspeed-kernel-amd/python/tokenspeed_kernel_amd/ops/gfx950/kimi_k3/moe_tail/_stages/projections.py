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


"""The router, shared-expert gate/up and latent projections.

They read the tagged post-attention-norm rows, in quarters of 1792 columns,
with zero-seeded wave sums and BF16 handoffs. The latent rows are quantized
to MXFP4 with the ceil(amax / 6) E8M0 scale rule.

Modified from sgl-amd-mono (https://github.com/sgl-project/sgl-amd-mono, Apache
License 2.0) at b9f40eb562f449ce08c030df200f97c7ea716511:
``kimi_k3/gfx950/layer.py`` and ``stages/moe.py``.
"""

from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.common import (
    bf16_round,
    gemv_dot,
    gemv_loads_strided,
    gemv_reduce,
    put_pair,
    thread_ids,
    wave_pending,
)
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.lds import store_mask_i32
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.polls import poll_pairs
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.staging import (
    poll_rows_to_lds,
    rows_to_lds,
)


@gluon.jit
def row_sum(red, row, token):
    value = gl.full(row.shape, 0, gl.float32, row.type.layout)
    for wave in gl.static_range(8):
        value = value + red.gather(wave * 256 + row // 4 * 64 + token * 4 + row % 4, 0)
    return value


@gluon.jit
def router(
    w_gate,
    xt,
    xl,
    red,
    regions,
    rank,
    group,
    q,
    tag,
    S: gl.constexpr,
    guard,
):
    weights = gemv_loads_strided(
        w_gate, rank * 112 + group * 16, 1792, True, False, 7168, q * 1792
    )
    poll_rows_to_lds(xt, xl, q, tag, S, guard)
    gl.barrier()
    acc = gemv_dot(xl, weights, 1792, S, False)
    gemv_reduce(acc, red, S)
    tid = thread_ids()
    token, row = gl.minimum(tid // 16, S - 1), tid % 16
    value = row_sum(red, row, token)
    at = ((rank * 4 + q) * S + token) * 112 + group * 16 + row
    for peer in gl.static_range(8):
        put_pair(regions[peer], at, value, tag, tid < 16 * S, "sys")
    gl.barrier()
    # This second barrier is not needed for correctness; removing it changes
    # the compiled code.
    gl.barrier()


@gluon.jit
def shared_gate_up(w_sgu, xbuf, xrdy, sgu, xl, red, group, tag, S: gl.constexpr, guard):
    weights = gemv_loads_strided(w_sgu, group * 16, 7168, True, False, 7168, 0)
    tid = thread_ids()
    poll_pairs(xrdy, (tid,), tag, tid < 56, "gpu", guard)
    gl.barrier()
    rows_to_lds(xbuf, xl, 7168, S, 7168, 0, "gpu")
    acc = gemv_dot(xl, weights, 7168, S, False)
    gemv_reduce(acc, red, S)
    token, pair = gl.minimum(tid // 8, S - 1), tid % 8
    lo = bf16_round(row_sum(red, pair * 2, token))
    hi = bf16_round(row_sum(red, pair * 2 + 1, token))
    word = lo.to(gl.bfloat16).to(gl.uint16, bitcast=True).to(gl.uint32) | (
        hi.to(gl.bfloat16).to(gl.uint16, bitcast=True).to(gl.uint32) << 16
    )
    put_pair(sgu, token * 768 + group * 8 + pair, word, tag, tid < 8 * S, "gpu")
    gl.barrier()


@gluon.jit
def quantize_latent(av, regions, rank, group, tag, S: gl.constexpr):
    tid = thread_ids()
    # One lane per row, each holding the row's 32 values.
    if wave_pending(tid < S):
        L: gl.constexpr = gl.BlockedLayout([1, 32], [64, 1], [8, 1], [1, 0])
        ids = gl.arange(0, 512, layout=gl.SliceLayout(1, L))
        token = gl.minimum(ids, S - 1)
        col = gl.arange(0, 32, layout=gl.SliceLayout(0, L))
        values = gl.reshape(
            av.gather(gl.reshape(token[:, None] * 32 + col[None, :], [16384]), 0),
            [512, 32],
        )
        values = gl.convert_layout(values, L)
        bits = values.to(gl.uint32, bitcast=True) & 0x7FFFFFFF
        amax = gl.max(bits, 1).to(gl.float32, bitcast=True)
        code = gl.minimum(
            ((amax * (1.0 / 6.0)).to(gl.uint32, bitcast=True) + 0x7FFFFF) >> 23 & 255,
            254,
        )
        packed = gl.amd.cdna4.scaled_downcast(
            values.to(gl.bfloat16), code[:, None].to(gl.uint8), "e2m1", axis=1
        )
        words = gl.reshape(packed, [512, 4, 4]).to(gl.uint32)
        b0, b1 = gl.split(gl.reshape(words, [512, 4, 2, 2]))
        # Pack four sequential bytes, retaining even-value low nibbles.
        pairs = b0 | (b1 << 8)
        p0, p1 = gl.split(pairs)
        payload = p0 | (p1 << 16)
        word = gl.arange(0, 4, layout=gl.SliceLayout(0, payload.type.layout))
        out_token = gl.arange(0, 512, layout=gl.SliceLayout(1, payload.type.layout))
        at = out_token[:, None] * 560 + (rank * 14 + group) * 4 + word[None, :]
        for peer in gl.static_range(8):
            put_pair(regions[peer], at, payload, tag, out_token[:, None] < S, "sys")
        for peer in gl.static_range(8):
            put_pair(
                regions[peer],
                token * 560 + 448 + rank * 14 + group,
                code,
                tag,
                ids < S,
                "sys",
            )


@gluon.jit
def latent(
    w_ld, xt, lpart, xl, red, av, regions, rank, group, q, tag, S: gl.constexpr, guard
):
    weights0 = gemv_loads_strided(
        w_ld, rank * 448 + group * 32, 1792, True, False, 7168, q * 1792
    )
    weights1 = gemv_loads_strided(
        w_ld, rank * 448 + group * 32 + 16, 1792, True, False, 7168, q * 1792
    )
    poll_rows_to_lds(xt, xl, q, tag, S, guard)
    acc0 = gemv_dot(xl, weights0, 1792, S, False)
    acc1 = gemv_dot(xl, weights1, 1792, S, False)
    tid = thread_ids()
    token, row = gl.minimum(tid // 16, S - 1), tid % 16
    for h in gl.static_range(2):
        gemv_reduce(acc0 if h == 0 else acc1, red, S)
        # Only the waves holding live rows run this. Every workgroup barrier
        # stays outside this wave guard; the masked LDS store adds none.
        if wave_pending(tid < S * 16):
            value = row_sum(red, row, token)
            if q > 0:
                put_pair(
                    lpart,
                    ((group * 4 + q) * S + token) * 32 + h * 16 + row,
                    value,
                    tag,
                    tid < S * 16,
                    "gpu",
                )
            if q == 0:
                indices = ()
                for qq in gl.static_range(1, 4):
                    indices += (((group * 4 + qq) * S + token) * 32 + h * 16 + row,)
                got = poll_pairs(lpart, indices, tag, tid < S * 16, "gpu", guard)
                for qq in gl.static_range(3):
                    value = value + got[qq]
                store_mask_i32(
                    av,
                    token * 32 + h * 16 + row,
                    bf16_round(value).to(gl.int32, bitcast=True),
                    tid < S * 16,
                )
        gl.barrier()
    if q == 0:
        quantize_latent(av, regions, rank, group, tag, S)
    gl.barrier()

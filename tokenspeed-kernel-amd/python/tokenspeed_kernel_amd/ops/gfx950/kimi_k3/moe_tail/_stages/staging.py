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


"""Staging rows into shared memory (LDS) for the narrow GEMVs.

Each copy issues all of its global loads first, then stores them to LDS with
one workgroup barrier at the end.

Modified from sgl-amd-mono (https://github.com/sgl-project/sgl-amd-mono, Apache
License 2.0) at b9f40eb562f449ce08c030df200f97c7ea716511:
``kimi_k3/gfx950/stages/gemv.py``.
"""

from tokenspeed_kernel_amd._extern import shared_store_i64_masked
from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.common import (
    descriptor,
    thread_ids,
)
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.lds import shared_address
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.polls import poll_pairs


@gluon.jit
def store_row_words(xl, words, first, K: gl.constexpr):
    L64: gl.constexpr = gl.BlockedLayout([1, 2], [64, 1], [8, 1], [1, 0])
    N: gl.constexpr = words.shape[0] // 4
    lo, hi = gl.split(gl.reshape(words, [N, 2, 2]))
    packed = lo.to(gl.uint64) | (hi.to(gl.uint64) << 32)
    packed = gl.convert_layout(
        packed.to(gl.int64, bitcast=True), L64, assert_trivial=True
    )
    quad = gl.arange(0, N, layout=gl.SliceLayout(1, L64)) + first // 8
    pair = gl.arange(0, 2, layout=gl.SliceLayout(0, L64))
    at = ((quad + quad // (K // 8)) * 2)[:, None] + pair[None, :]
    # Every lane stores, the zero padding of a partial last load included, so
    # the store takes no mask. The allocation is 16-byte aligned; ``& -16``
    # tells LLVM so.
    shared_store_i64_masked(
        shared_address(xl) & -16, at, packed, gl.full([N, 2], 1, gl.int32, L64)
    )


@gluon.jit
def gated_rows_loads(core, gate, norm_w, S: gl.constexpr, GATE_STRIDE: gl.constexpr):
    """Issue :func:`gated_rows_store`'s loads: the norm weight, then each
    32-head-row chunk's attention output and gate logits."""
    # _rmsnorm_gated_kernel's layout over a head's 128 values, so each head
    # reduces in its order; a lane's 8 values are the 8 that rows_to_lds
    # gives it, so the words go to store_row_words unmoved.
    NL: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [8, 1], [1, 0])
    HEADS: gl.constexpr = 1536 // 128
    CHUNKS: gl.constexpr = S * HEADS // 32
    gl.static_assert(S * HEADS % 32 == 0)
    d = gl.arange(0, 128, layout=gl.SliceLayout(0, NL))
    w = gl.load(norm_w + d)
    xs = ()
    gs = ()
    for i in gl.static_range(CHUNKS):
        head_row = i * 32 + gl.arange(0, 32, layout=gl.SliceLayout(1, NL))
        t = head_row // HEADS
        h = head_row % HEADS
        xs += (gl.load(core + (t * 1536 + h * 128)[:, None] + d[None, :]),)
        gs += (gl.load(gate + (t * GATE_STRIDE + h * 128)[:, None] + d[None, :]),)
    return w, xs, gs


@gluon.jit
def gated_rows_store(loaded, xl, S: gl.constexpr, EPS: gl.constexpr):
    """Stage ``rmsnorm_gated_sigmoid(core, gate, norm_w)``'s rows as
    ``rows_to_lds`` stages rows, bit for bit, from the raw 1536-column
    attention output and its gate logits that :func:`gated_rows_loads`
    loaded."""
    CHUNKS: gl.constexpr = S * (1536 // 128) // 32
    w, xs, gs = loaded
    w = w.to(gl.float32)
    for i in gl.static_range(CHUNKS):
        x = xs[i].to(gl.float32)
        g = gs[i].to(gl.float32)
        rsig = gl.rsqrt(gl.sum(x * x, axis=1) / 128 + EPS)
        y = x * rsig[:, None] * w[None, :] * (1 / (1 + gl.exp(-g)))
        bits = y.to(gl.bfloat16).to(gl.uint16, bitcast=True).to(gl.uint32)
        lo, hi = gl.split(gl.reshape(bits, [32, 64, 2]))
        words = gl.convert_layout(
            gl.reshape(lo | (hi << 16), [2048]),
            gl.BlockedLayout([4], [64], [8], [0]),
            assert_trivial=True,
        )
        store_row_words(xl, words, i * 4096, 1536)
    gl.barrier()


@gluon.jit
def rows_to_lds(
    src,
    xl,
    K: gl.constexpr,
    S: gl.constexpr,
    LDK: gl.constexpr,
    k0,
    SCOPE: gl.constexpr,
):
    L: gl.constexpr = gl.BlockedLayout([1, 4], [64, 1], [8, 1], [1, 0])
    desc = descriptor(src.to(gl.pointer_type(gl.uint32)))
    loads = ()
    for i in gl.static_range(gl.cdiv(S * K, 4096)):
        e = gl.arange(
            0,
            min(512, triton.next_power_of_2(S * K // 8 - i * 512)),
            layout=gl.SliceLayout(1, L),
        )
        item = (e + i * 512) * 8
        t, k = item // K, item % K
        word = gl.arange(0, 4, layout=gl.SliceLayout(0, L))
        at = ((t * LDK + k0 + k) // 2)[:, None] + word[None, :]
        at = gl.max_contiguous(gl.multiple_of(at, [1, 4]), [1, 4])
        value = gl.amd.cdna3.buffer.load(
            desc, at, mask=(item < S * K)[:, None], scope=SCOPE
        )
        loads += (
            gl.convert_layout(
                gl.reshape(value, [e.shape[0] * 4]),
                gl.BlockedLayout([4], [64], [8], [0]),
                assert_trivial=True,
            ),
        )
    # Issue every chunk's loads before any chunk's LDS store: interleaved,
    # the loads would wait behind the stores' barriers.
    for i in gl.static_range(gl.cdiv(S * K, 4096)):
        store_row_words(xl, loads[i], i * 4096, K)
    gl.barrier()


@gluon.jit
def poll_rows_to_lds(xt, xl, q, tag, S: gl.constexpr, guard):
    """Wait for one quarter of the tagged post-attention-norm rows and stage
    them in LDS as packed BF16."""
    tid = thread_ids()
    K: gl.constexpr = 1792
    PER: gl.constexpr = gl.cdiv(S * K // 4, 512)
    gl.static_assert(PER <= 14)
    indices = ()
    for i in gl.static_range(PER):
        item = gl.minimum(tid + 512 * i, S * K // 4 - 1)
        token, col = item // (K // 4), item % (K // 4) * 4
        at = (token * 7168 + q * K + col) // 2
        indices += (at, at + 1)
    got = poll_pairs(
        xt, indices, tag, gl.full([512], True, gl.int1, tid.type.layout), "gpu", guard
    )
    for i in gl.static_range(PER):
        item = gl.minimum(tid + 512 * i, S * K // 4 - 1)
        token, col = item // (K // 4), item % (K // 4) * 4
        # Dead lanes have a disjoint tail, so no producer races a real row.
        at = gl.where(
            tid + 512 * i < S * K // 4,
            (token * (K + 8) + col) // 4,
            S * (K + 8) // 4 + tid,
        )
        packed = got[2 * i].to(gl.uint32, bitcast=True).to(gl.uint64) | (
            got[2 * i + 1].to(gl.uint32, bitcast=True).to(gl.uint64) << 32
        )
        # Dead lanes store too, into their disjoint tail, so the store takes
        # no mask.
        shared_store_i64_masked(
            shared_address(xl) & -16,
            at,
            packed.to(gl.int64, bitcast=True),
            gl.full([512], 1, gl.int32, tid.type.layout),
        )
    gl.barrier()

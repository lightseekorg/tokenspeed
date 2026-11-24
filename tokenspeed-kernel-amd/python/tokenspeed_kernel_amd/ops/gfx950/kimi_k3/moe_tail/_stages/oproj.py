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


"""The attention output projection and its reduction across ranks.

Every rank projects its 1536 attention columns to all 7168 hidden columns
and sends each peer its slices as four-byte exchange words; each slice's
owner sums the eight ranks' words in rank order.

Modified from sgl-amd-mono (https://github.com/sgl-project/sgl-amd-mono, Apache
License 2.0) at b9f40eb562f449ce08c030df200f97c7ea716511:
``kimi_k3/gfx950/stages/oproj.py`` and ``common/ops.py``, whose device
primitives are adapted from ROCm/ATOM (https://github.com/ROCm/ATOM, MIT
License) at 97359d6df46c59736468168b7be8bc50b482a1a9.
"""

from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.common import (
    bf16_round,
    gemv_dot,
    gemv_loads,
    gemv_reduce,
    thread_ids,
    wave_pending,
)
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.sentinel import (
    put_sent,
    rearm_sent,
    sent_item,
    sent_ix,
    sent_sum,
)
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.staging import (
    gated_rows_loads,
    gated_rows_store,
    rows_to_lds,
)


@gluon.jit
def stage_o(xl, red, peer_regions, rank, group, weights, S: gl.constexpr):
    acc = gemv_dot(xl, weights, 1536, S, False)
    # Only the stored accumulators are used here, not the returned sum: the
    # helper stores them in wave order and joins the writing waves.
    gemv_reduce(acc, red, S)
    tid = thread_ids()
    # Only the sending waves read the accumulators.
    # The barrier remains outside this per-wave guard.
    if wave_pending(tid < S * 8):
        token = gl.minimum(tid // 8, S - 1)
        pair = tid % 8
        r0, r1 = pair * 2, pair * 2 + 1
        lo = gl.full([512], 0, gl.float32, tid.type.layout)
        hi = gl.full([512], 0, gl.float32, tid.type.layout)
        for wave in gl.static_range(8):
            lo = lo + red.gather(wave * 256 + r0 // 4 * 64 + token * 4 + r0 % 4, 0)
            hi = hi + red.gather(wave * 256 + r1 // 4 * 64 + token * 4 + r1 % 4, 0)
        lo, hi = bf16_round(lo), bf16_round(hi)
        word = lo.to(gl.bfloat16).to(gl.uint16, bitcast=True).to(gl.uint32) | (
            hi.to(gl.bfloat16).to(gl.uint16, bitcast=True).to(gl.uint32) << 16
        )
        index = sent_ix(rank, group * 16 + pair * 2, token, S, 7168)
        for peer in gl.static_range(8):
            put_sent(peer_regions[peer], index, word, tid < S * 8)
    gl.barrier()


@gluon.jit
def _stage_oproj_groups(
    core,
    w_o,
    xl,
    red,
    peer_regions,
    rank,
    group,
    S: gl.constexpr,
    TWO: gl.constexpr,
    gate,
    norm_w,
    GATED_NORM: gl.constexpr,
    GATE_STRIDE: gl.constexpr,
    NORM_EPS: gl.constexpr,
):
    # The gated norm's rows are loaded before the weights: loads complete in
    # issue order, so behind the weights its arithmetic would wait for all of
    # them. The scheduling barriers keep that order.
    if GATED_NORM:
        gated = gated_rows_loads(core, gate, norm_w, S, GATE_STRIDE)
        gl.amd.hint.sched_barrier()
    # Load every group's weights before the attention rows are staged or any
    # matrix instruction runs.
    weights0 = gemv_loads(w_o, group * 16, 1536, True, False, NONTEMPORAL=False)
    if TWO:
        weights1 = gemv_loads(
            w_o, (group + 200) * 16, 1536, True, False, NONTEMPORAL=False
        )
    if GATED_NORM:
        gl.amd.hint.sched_barrier()
        gated_rows_store(gated, xl, S, NORM_EPS)
    else:
        rows_to_lds(core, xl, 1536, S, 1536, 0, None)
    stage_o(xl, red, peer_regions, rank, group, weights0, S)
    if TWO:
        stage_o(xl, red, peer_regions, rank, group + 200, weights1, S)


@gluon.jit
def stage_oproj(
    core,
    w_o,
    xl,
    red,
    peer_regions,
    rank,
    first_group,
    S: gl.constexpr,
    gate,
    norm_w,
    GATED_NORM: gl.constexpr,
    GATE_STRIDE: gl.constexpr,
    NORM_EPS: gl.constexpr,
):
    """Output groups ``first_group`` and ``first_group + 200``, of 448."""
    _stage_oproj_groups(
        core,
        w_o,
        xl,
        red,
        peer_regions,
        rank,
        first_group,
        S,
        True,
        gate,
        norm_w,
        GATED_NORM,
        GATE_STRIDE,
        NORM_EPS,
    )


@gluon.jit
def stage_oproj_third(
    core,
    w_o,
    xl,
    red,
    peer_regions,
    rank,
    group,
    S: gl.constexpr,
    gate,
    norm_w,
    GATED_NORM: gl.constexpr,
    GATE_STRIDE: gl.constexpr,
    NORM_EPS: gl.constexpr,
):
    """Output group ``group``, one of the 48 past the two rounds of 200."""
    _stage_oproj_groups(
        core,
        w_o,
        xl,
        red,
        peer_regions,
        rank,
        group,
        S,
        False,
        gate,
        norm_w,
        GATED_NORM,
        GATE_STRIDE,
        NORM_EPS,
    )


@gluon.jit
def stage_reduce(
    base,
    delta_lds,
    task,
    epoch,
    own,
    peers,
    rank,
    S: gl.constexpr,
    FLAGS_OFFSET: gl.constexpr,
):
    tid = thread_ids()
    live = tid < S * 64
    # Only the waves with live rows run this; the guard is per wave, so it
    # adds no barrier.
    if wave_pending(live):
        token, pair = sent_item(gl.minimum(tid, S * 64 - 1), S)
        col = task * 128 + pair * 2
        lo, hi = sent_sum(
            base,
            col,
            token,
            live,
            epoch,
            own,
            peers,
            rank,
            S,
            7168,
            FLAGS_OFFSET,
            1,
            False,
        )
        # ``live`` covers whole waves (S * 64 lanes), so in a wave that runs
        # this every lane is live. The select only keeps a dead lane, should
        # there be one, on an index of its own.
        at = gl.where(live, token * 128 + pair * 2, tid * 2)
        # One paired scatter avoids an alias-analysis barrier between two
        # stores. A compiler-inserted CTA barrier inside this wave guard
        # would leave the inactive waves unable to rendezvous.
        paired = delta_lds.reinterpret(
            gl.float32,
            [512, 2],
            gl.SwizzledSharedLayout(1, 1, 1, [1, 0]),
        )
        PAIRS: gl.constexpr = gl.BlockedLayout([1, 2], [64, 1], [8, 1], [0, 1])
        values = gl.convert_layout(
            gl.join(bf16_round(lo), bf16_round(hi)), PAIRS, assert_trivial=True
        )
        rows = gl.convert_layout(at // 2, gl.SliceLayout(1, PAIRS), assert_trivial=True)
        paired.scatter(values, rows[:, None], 0)
    gl.barrier()


@gluon.jit
def rearm_ar(base, task, S: gl.constexpr):
    tid = thread_ids()
    token, pair = sent_item(gl.minimum(tid, S * 64 - 1), S)
    indices = ()
    for src in gl.static_range(8):
        indices += (sent_ix(src, task * 128 + pair * 2, token, S, 7168),)
    rearm_sent(base, indices, tid < S * 64)

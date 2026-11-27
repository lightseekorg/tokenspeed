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


"""The latent up projection, the exchange of the output slices and the
final residual add.

Modified from sgl-amd-mono (https://github.com/sgl-project/sgl-amd-mono, Apache
License 2.0) at b9f40eb562f449ce08c030df200f97c7ea716511:
``kimi_k3/gfx950/stages/moe.py``.
"""

from tokenspeed_kernel_amd._extern import release_wait_vmem
from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.common import (
    bf16_round,
    descriptor,
    gemv_dot,
    gemv_loads,
    gemv_reduce,
    put_pair,
    thread_ids,
    wave_pending,
)
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.lds import load_f32
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.polls import (
    poll_pairs,
    poll_regions,
)
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.sentinel import (
    put_sent,
    rearm_sent,
    sent_item,
    sent_ix,
    sent_sum,
)
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.staging import (
    rows_to_lds,
)


@gluon.jit
def bf16_pair(lo, hi):
    return (
        lo.to(gl.bfloat16).to(gl.uint16, bitcast=True).to(gl.uint32)
        | hi.to(gl.bfloat16).to(gl.uint16, bitcast=True).to(gl.uint32) << 16
    )


@gluon.jit
def put_final(regions, rank, column, token, lo, hi, live, S: gl.constexpr):
    index = sent_ix(rank, column, token, S, 7168)
    word = bf16_pair(lo, hi)
    for peer in gl.static_range(8):
        put_sent(regions[peer], index, word, live)


@gluon.jit
def sum_waves(red, row, token):
    value = gl.full(row.shape, 0.0, gl.float32, row.type.layout)
    for wave in gl.static_range(8):
        value = value + load_f32(red, wave * 256 + row // 4 * 64 + token * 4 + row % 4)
    return value


@gluon.jit
def stage_up(
    w_up,
    ln,
    shp,
    lnrdy,
    shrdy,
    xl,
    red,
    final_regions,
    rank,
    group,
    tag,
    S: gl.constexpr,
    guard,
):
    weights = gemv_loads(w_up, group * 16, 3584, True, False)
    tid = thread_ids()
    # Thread zero waits for both flags together; the other live lanes wait
    # only for their latent norm flag.
    if wave_pending(tid == 0):
        poll_regions(
            (lnrdy, shrdy),
            (tid * 0, tid * 0 + group),
            (tid == 0, tid == 0),
            tag,
            "gpu",
            guard,
        )
    if wave_pending((tid > 0) & (tid < 28)):
        poll_pairs(lnrdy, (tid,), tag, (tid > 0) & (tid < 28), "gpu", guard)
    gl.barrier()
    token, pair = gl.minimum(tid // 8, S - 1), tid % 8
    at = (token * 896 + group * 16 + pair * 2) // 2
    # Issued ahead of the rows' loads, so its latency overlaps theirs.
    word = gl.amd.cdna3.buffer.load(
        descriptor(shp.to(gl.pointer_type(gl.uint32))), at, scope="gpu"
    )
    rows_to_lds(ln, xl, 3584, S, 3584, 0, "gpu")
    acc = gemv_dot(xl, weights, 3584, S, False)
    gemv_reduce(acc, red, S)
    lo = bf16_round(
        (word << 16).to(gl.float32, bitcast=True) + sum_waves(red, pair * 2, token)
    )
    hi = bf16_round(
        (word & 0xFFFF0000).to(gl.float32, bitcast=True)
        + sum_waves(red, pair * 2 + 1, token)
    )
    put_final(
        final_regions,
        rank,
        rank * 896 + group * 16 + pair * 2,
        token,
        lo,
        hi,
        tid < S * 8,
        S,
    )
    gl.barrier()


@gluon.jit
def stage_final(
    final_own,
    prefix,
    output,
    group,
    epoch,
    own,
    peers,
    rank,
    S: gl.constexpr,
    FLAGS_OFFSET: gl.constexpr,
    mixrow,
    mixrdy,
    tag,
    NEXT_MIX: gl.constexpr,
):
    tid = thread_ids()
    live = tid < S * 64
    token, pair = sent_item(gl.minimum(tid, S * 64 - 1), S)
    col = group * 128 + pair * 2
    # The prefix this launch's slices wrote is in flight across the wait.
    pw = gl.amd.cdna3.buffer.load(
        descriptor(prefix.to(gl.pointer_type(gl.uint32))),
        (token * 7168 + col) // 2,
        scope="gpu",
    )
    lo, hi = sent_sum(
        final_own,
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
        14,
        False,
    )
    lo = bf16_round(bf16_round(lo) + (pw << 16).to(gl.float32, bitcast=True))
    hi = bf16_round(bf16_round(hi) + (pw & 0xFFFF0000).to(gl.float32, bitcast=True))
    gl.amd.cdna3.buffer.store(
        bf16_pair(lo, hi),
        descriptor(output.to(gl.pointer_type(gl.uint32))),
        (token * 7168 + col) // 2,
        mask=live,
    )
    if NEXT_MIX:
        # Mirror exactly the same rounded BF16 words, beyond every cache.
        # System coherence emits sc0 sc1 even for this local-only handoff.
        gl.amd.cdna3.buffer.atomic_store(
            bf16_pair(lo, hi),
            descriptor(mixrow.to(gl.pointer_type(gl.uint32))),
            (token * 7168 + col) // 2,
            mask=live,
            sem="relaxed",
            scope="sys",
        )
    indices = ()
    for src in gl.static_range(8):
        indices += (sent_ix(src, col, token, S, 7168),)
    rearm_sent(final_own, indices, live)
    if NEXT_MIX:
        # Every producing wave drains its stores before thread zero publishes.
        release_wait_vmem()
    gl.barrier()
    if NEXT_MIX:
        put_pair(mixrdy, tid * 0 + group, tid * 0 + 1, tag, tid == 0, "sys")

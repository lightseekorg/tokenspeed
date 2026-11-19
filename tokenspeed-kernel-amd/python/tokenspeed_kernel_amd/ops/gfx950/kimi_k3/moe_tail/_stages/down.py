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


"""The routed experts' down projection and the top-k BF16 folds.

Five experts' weights are in flight at once. The LDS region replaces the
gate/up region only after this workgroup has finished its gate/up tasks.

Modified from sgl-amd-mono (https://github.com/sgl-project/sgl-amd-mono, Apache
License 2.0) at b9f40eb562f449ce08c030df200f97c7ea716511:
``kimi_k3/gfx950/stages/moe.py``.
"""

from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.common import (
    bf16_round,
    descriptor,
    thread_ids,
    wave_pending,
)
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.expert_weights import (
    expert_addresses,
)
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.experts import (
    matrix_layout,
    word_bytes,
)
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.lds import (
    load_f32,
    load_four_words,
    load_i32,
    store_four_words,
    store_mask_i32,
)
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.polls import poll_pairs
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.sentinel import (
    put_sent,
    sent_ix,
)


@gluon.jit
def load_inter(inter, workspace, tag, S: gl.constexpr, guard):
    tid = thread_ids()
    NW: gl.constexpr = S * 16 * 48
    NC: gl.constexpr = S * 16 * 12
    PER: gl.constexpr = gl.cdiv(NW, 512)
    PERC: gl.constexpr = gl.cdiv(NC, 512)
    # The intermediates and their scales are adjacent scratch regions.
    indices = ()
    for i in gl.static_range(PER):
        indices += (gl.minimum(tid + 512 * i, NW - 1),)
    for i in gl.static_range(PERC):
        indices += (NW + gl.minimum(tid + 512 * i, NC - 1),)
    got = poll_pairs(
        inter,
        indices,
        tag,
        gl.full([512], True, gl.int1, tid.type.layout),
        "gpu",
        guard,
    )
    # Masked stores (Gluon shared stores take no mask); the barrier below
    # orders them.
    for i in gl.static_range(PER):
        store_mask_i32(
            workspace,
            tid + 512 * i,
            got[i].to(gl.int32, bitcast=True),
            tid + 512 * i < NW,
        )
    for i in gl.static_range(PERC):
        store_mask_i32(
            workspace,
            NW + tid + 512 * i,
            got[PER + i].to(gl.int32, bitcast=True),
            tid + 512 * i < NC,
        )
    gl.barrier()


@gluon.jit
def down_loads(w2, w2s, experts, group, nu, b0):
    ML: gl.constexpr = matrix_layout()
    WORD: gl.constexpr = gl.DotOperandLayout(0, ML, 4)
    SCALE: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(
        gl.DotOperandLayout(0, ML, 16), [8, 16, 4]
    )
    wave = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, WORD)))
    row = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, WORD)))
    kw = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(1, WORD)))
    swave = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, SCALE)))
    srow = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, SCALE)))
    sk = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(1, SCALE)))
    wd = descriptor(gl.multiple_of(w2, 16).to(gl.pointer_type(gl.uint32)))
    sd = descriptor(w2s.to(gl.pointer_type(gl.uint32)))
    base = gl.convert_layout(b0, wave.type.layout, assert_trivial=True)
    operands, scales = (), ()
    for j in gl.static_range(5):
        slot = gl.minimum(
            base + 8 * j + gl.zeros([8], gl.int32, wave.type.layout), nu - 1
        )
        e = load_i32(experts, slot)
        es = gl.convert_layout(e, swave.type.layout)
        for st in gl.static_range(3):
            lane = row[None, :, None] + (kw[None, None, :] // 4) * 16
            at, _, _ = expert_addresses(
                e[:, None, None],
                group,
                st,
                lane,
                kw[None, None, :] % 4,
                3584,
                384,
            )
            at = gl.max_contiguous(gl.multiple_of(at, [1, 1, 4]), [1, 1, 4])
            operands += (
                gl.amd.cdna3.buffer.load(
                    wd, at, cache_policy=gl.amd.cdna3.CachePolicy(nontemporal=True)
                ),
            )
            _, sat, shift = expert_addresses(
                es[:, None, None],
                group,
                st,
                srow[None, :, None] + sk[None, None, :] * 16,
                0,
                3584,
                384,
            )
            scales += (
                ((gl.amd.cdna3.buffer.load(sd, sat) >> shift) & 255).to(gl.uint8),
            )
    return operands + scales


@gluon.jit
def stage_down(
    w2,
    w2s,
    workspace,
    red,
    route,
    experts,
    kof,
    group,
    nu,
    ops,
    latr_regions,
    rank,
    S: gl.constexpr,
):
    ML: gl.constexpr = matrix_layout()
    A: gl.constexpr = gl.DotOperandLayout(0, ML, 16)
    B: gl.constexpr = gl.DotOperandLayout(1, ML, 16)
    BW: gl.constexpr = gl.DotOperandLayout(1, ML, 4)
    BS: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(B, [8, 16, 4])
    wave = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, BW)))
    kw = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, BW)))
    token = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(1, BW)))
    stoken = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, BS)))
    sk = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(1, BS)))
    cwave = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, ML)))
    crow = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, ML)))
    ctoken = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(1, ML)))
    NW: gl.constexpr = S * 16 * 48
    CONTRIB: gl.constexpr = S * 16 * (48 + 12)
    b0 = wave
    # Each wave walks its own down tasks and leaves after a different number
    # of rounds, so the LDS accesses in this loop (and in down_loads) use the
    # barrier-free helpers: a Gluon LDS op would add a barrier not all waves
    # reach.
    while wave_pending(b0 < nu):
        nb = b0 + 40
        next_ops = down_loads(w2, w2s, experts, group, nu, gl.minimum(nb, nu - 1))
        for j in gl.static_range(5):
            slot = gl.minimum(b0 + 8 * j, nu - 1)
            k = load_i32(
                kof, slot[:, None, None] * 8 + gl.minimum(token[None, None, :], S - 1)
            )
            picked = (k >= 0) & (token[None, None, :] < S)
            pick = gl.minimum(token[None, None, :], S - 1) * 16 + gl.maximum(k, 0)
            ks = gl.convert_layout(gl.permute(k, [0, 2, 1]), BS, assert_trivial=True)
            spicked = (ks >= 0) & (stoken[None, :, None] < S)
            spick = gl.minimum(stoken[None, :, None], S - 1) * 16 + gl.maximum(ks, 0)
            acc = gl.full([8, 16, 16], 0.0, gl.float32, ML)
            for st in gl.static_range(3):
                bw = load_four_words(workspace, pick * 48 + st * 16 + kw[None, :, None])
                bw = gl.where(picked, bw, 0)
                sb = load_i32(workspace, NW + spick * 12 + st * 4 + sk[None, None, :])
                sb = gl.where(spicked, sb, 127).to(gl.uint8)
                a = gl.convert_layout(
                    gl.reshape(word_bytes(ops[j * 3 + st]), [8, 16, 64]),
                    A,
                    assert_trivial=True,
                )
                b = gl.convert_layout(
                    gl.reshape(gl.permute(word_bytes(bw), [0, 1, 3, 2]), [8, 64, 16]),
                    B,
                    assert_trivial=True,
                )
                acc = gl.amd.cdna4.mfma_scaled(
                    a, ops[15 + j * 3 + st], "e2m1", b, sb, "e2m1", acc
                )
            ck = gl.convert_layout(k, ML, assert_trivial=True)
            wt = load_f32(
                route,
                gl.minimum(ctoken[None, None, :], S - 1) * 32 + 16 + gl.maximum(ck, 0),
            )
            contribution = bf16_round(acc * wt)
            at = (
                CONTRIB
                + (gl.minimum(ctoken[None, None, :], S - 1) * 16 + gl.maximum(ck, 0))
                * 16
                + crow[None, :, None]
            )
            valid = (
                (gl.convert_layout(b0, cwave.type.layout)[:, None, None] + 8 * j < nu)
                & (ck >= 0)
                & (ctoken[None, None, :] < S)
            )
            store_four_words(
                workspace, at, contribution.to(gl.int32, bitcast=True), valid
            )
        b0 = nb
        ops = next_ops
    gl.barrier()
    tid = thread_ids()
    # Only the live rows' lanes publish, all in wave 0. Keep both
    # workgroup barriers outside this wave-uniform guard.
    if wave_pending(tid < S * 8):
        output_token, output_pair = gl.minimum(tid // 8, S - 1), tid % 8
        y0 = gl.full([512], 0.0, gl.float32, tid.type.layout)
        y1 = gl.full([512], 0.0, gl.float32, tid.type.layout)
        # Two adjacent columns stay together in one lane. Both top-k terms
        # are loaded together; the BF16 additions keep the top-k order.
        FOLD: gl.constexpr = gl.BlockedLayout([4], [64], [8], [0])
        fold_item = gl.arange(0, 2048, layout=FOLD)
        fold_tid = fold_item // 4
        fold_token = gl.minimum(fold_tid // 8, S - 1)
        fold_pair = fold_tid % 8
        for output_k in gl.static_range(0, 16, 2):
            index = (
                CONTRIB
                + (fold_token * 16 + output_k + (fold_item % 4) // 2) * 16
                + fold_pair * 2
                + fold_item % 2
            )
            words = workspace.gather(gl.max_contiguous(index, 2), 0)
            lo, hi = gl.split(
                gl.reshape(words.to(gl.float32, bitcast=True), [512, 2, 2])
            )
            lo0, lo1 = gl.split(lo)
            hi0, hi1 = gl.split(hi)
            lo0 = gl.convert_layout(lo0, tid.type.layout, assert_trivial=True)
            lo1 = gl.convert_layout(lo1, tid.type.layout, assert_trivial=True)
            hi0 = gl.convert_layout(hi0, tid.type.layout, assert_trivial=True)
            hi1 = gl.convert_layout(hi1, tid.type.layout, assert_trivial=True)
            y0 = bf16_round(y0 + lo0)
            y1 = bf16_round(y1 + hi0)
            y0 = bf16_round(y0 + lo1)
            y1 = bf16_round(y1 + hi1)
        output_word = (
            y0.to(gl.bfloat16).to(gl.uint16, bitcast=True).to(gl.uint32)
            | y1.to(gl.bfloat16).to(gl.uint16, bitcast=True).to(gl.uint32) << 16
        )
        output_at = sent_ix(rank, group * 16 + output_pair * 2, output_token, S, 3584)
        for peer in gl.static_range(8):
            put_sent(latr_regions[peer], output_at, output_word, tid < S * 8)
    gl.barrier()

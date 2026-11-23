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


"""The routed experts' gate/up projection, SiTU activation and FP4
intermediate handoff. The tanh and SiTU forms are AITER's.

Modified from sgl-amd-mono (https://github.com/sgl-project/sgl-amd-mono, Apache
License 2.0) at b9f40eb562f449ce08c030df200f97c7ea716511:
``kimi_k3/gfx950/stages/moe.py``.
"""

from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.common import (
    descriptor,
    put_pair,
    thread_ids,
)
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.expert_weights import (
    expert_addresses,
    gate_up_row_group,
)
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.lds import (
    load_f32,
    load_i32,
    store_f32,
)


@gluon.constexpr_function
def matrix_layout():
    return gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 128], transposed=False, warps_per_cta=[8, 1, 1]
    )


@gluon.jit
def word_bytes(words):
    """Each register remains a packed dword until its four bytes are needed."""
    b0 = words.to(gl.uint8)
    b1 = (words >> 8).to(gl.uint8)
    b2 = (words >> 16).to(gl.uint8)
    b3 = (words >> 24).to(gl.uint8)
    return gl.reshape(gl.join(gl.join(b0, b2), gl.join(b1, b3)), [8, 16, 16, 4])


@gluon.jit
def ug_loads(w13, w13s, expert_table, task):
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
    expert = load_i32(
        expert_table, task // 12 + gl.zeros([8], gl.int32, wave.type.layout)
    )
    sexpert = gl.convert_layout(expert, swave.type.layout)
    rg = gate_up_row_group(task % 12, wave)
    srg = gate_up_row_group(task % 12, swave)
    wd = descriptor(gl.multiple_of(w13, 16).to(gl.pointer_type(gl.uint32)))
    sd = descriptor(w13s.to(gl.pointer_type(gl.uint32)))
    operands, scales = (), ()
    for i in gl.static_range(14):
        step = (wave // 4) * 14 + i
        lane = row[None, :, None] + (kw[None, None, :] // 4) * 16
        at, _, _ = expert_addresses(
            expert[:, None, None],
            rg[:, None, None],
            step[:, None, None],
            lane,
            kw[None, None, :] % 4,
            768,
            3584,
        )
        at = gl.max_contiguous(gl.multiple_of(at, [1, 1, 4]), [1, 1, 4])
        operands += (
            gl.amd.cdna3.buffer.load(
                wd, at, cache_policy=gl.amd.cdna3.CachePolicy(nontemporal=True)
            ),
        )
        if i % 2 == 0:
            sstep = (swave // 4) * 14 + i
            _, sat, _ = expert_addresses(
                sexpert[:, None, None],
                srg[:, None, None],
                sstep[:, None, None],
                srow[None, :, None] + sk[None, None, :] * 16,
                0,
                768,
                3584,
            )
            scales += (gl.amd.cdna3.buffer.load(sd, sat),)
    return operands + scales


@gluon.jit
def tanh_aiter(x):
    e = gl.exp2(gl.abs(x) * -2.8853900817779268)
    value = (1.0 - e) * gl.extra.libdevice.fast_dividef(1.0, 1.0 + e)
    return gl.where(x > 0.0, value, -value)


@gluon.jit
def situ_aiter(gate, up):
    sigmoid = gl.extra.libdevice.fast_dividef(
        1.0, 1.0 + gl.exp2(gate * -1.4426950408889634)
    )
    value = (4.0 * tanh_aiter(gate * 0.25)) * sigmoid
    return (value * 25.0) * tanh_aiter(up * 0.04)


@gluon.jit
def pair_wave_sum(red, row, token, first_wave):
    value = gl.full(row.shape, 0.0, gl.float32, row.type.layout)
    at = row // 4 * 64 + token * 4 + row % 4
    value = value + load_f32(red, first_wave * 256 + at)
    return value + load_f32(red, (first_wave + 4) * 256 + at)


@gluon.jit
def prefetch_ug_inputs(
    workspace, wave, kw, token, sw, stoken, sk, STEP: gl.constexpr, S: gl.constexpr
):
    step = (wave // 4) * 14 + STEP
    native_offsets = (
        gl.minimum(token[None, None, :], S - 1) * 448
        + step[:, None, None] * 16
        + kw[None, :, None]
    )
    # Load the four adjacent K words each lane owns together.
    native_indices = gl.reshape(gl.permute(native_offsets, [0, 2, 1]), [2048])
    native_indices = gl.max_contiguous(gl.multiple_of(native_indices, 4), 4)
    native_words = workspace.gather(native_indices, 0)
    bw = gl.permute(gl.reshape(native_words, [8, 16, 16]), [0, 2, 1]).to(gl.uint32)
    native_scale_at = (
        S * 448
        + gl.minimum(stoken[None, :, None], S - 1) * 112
        + ((sw[:, None, None] // 4) * 14 + STEP) * 4
        + sk[None, None, :]
    )
    sb = gl.reshape(
        workspace.gather(gl.reshape(native_scale_at, [512]), 0), [8, 16, 4]
    ).to(gl.uint8)
    return bw, sb


@gluon.jit
def ug_compute(workspace, red, ops, S: gl.constexpr):
    ML: gl.constexpr = matrix_layout()
    A: gl.constexpr = gl.DotOperandLayout(0, ML, 16)
    B: gl.constexpr = gl.DotOperandLayout(1, ML, 16)
    BW: gl.constexpr = gl.DotOperandLayout(1, ML, 4)
    AS: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(A, [8, 16, 4])
    BS: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(B, [8, 16, 4])
    wave = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, BW)))
    kw = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, BW)))
    token = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(1, BW)))
    sw = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, BS)))
    stoken = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, BS)))
    sk = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(1, BS)))
    awave = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, AS)))
    acc = gl.full([8, 16, 16], 0.0, gl.float32, ML)
    bw, sb = prefetch_ug_inputs(workspace, wave, kw, token, sw, stoken, sk, 0, S)
    for i in gl.static_range(14):
        if i < 13:
            next_bw, next_sb = prefetch_ug_inputs(
                workspace, wave, kw, token, sw, stoken, sk, i + 1, S
            )
            gl.amd.hint.sched_barrier()
        a = gl.convert_layout(
            gl.reshape(word_bytes(ops[i]), [8, 16, 64]), A, assert_trivial=True
        )
        b = gl.convert_layout(
            gl.reshape(gl.permute(word_bytes(bw), [0, 1, 3, 2]), [8, 64, 16]),
            B,
            assert_trivial=True,
        )
        # The row group's parity selects the up rows, whatever the group.
        shift = (i % 2) * 16 + ((awave[:, None, None] % 4) // 2) * 8
        sa = ((ops[14 + i // 2] >> shift) & 255).to(gl.uint8)
        acc = gl.amd.cdna4.mfma_scaled(a, sa, "e2m1", b, sb, "e2m1", acc)
        if i < 13:
            bw, sb = next_bw, next_sb
    batch = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, ML)))
    row = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, ML)))
    col = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(1, ML)))
    at = (
        batch[:, None, None] * 256
        + row[None, :, None] // 4 * 64
        + col[None, None, :] * 4
        + row[None, :, None] % 4
    )
    store_f32(red, at, acc)
    gl.barrier()


@gluon.jit
def stage_ug(
    workspace, red, av, expert_kof, inter, isc, task, ops, tag, S: gl.constexpr
):
    ug_compute(workspace, red, ops, S)
    tid = thread_ids()
    token, column = gl.minimum(tid // 32, S - 1), tid % 32
    row = column % 16
    gv = pair_wave_sum(red, row, token, gl.where(column < 16, 0, 1))
    uv = pair_wave_sum(red, row, token, gl.where(column < 16, 2, 3))
    value = situ_aiter(gv, uv)
    maximum = gl.reshape(gl.abs(value), [16, 32])
    columns = gl.reshape(tid, [16, 32]) % 32
    for distance in gl.static_range(5):
        maximum = gl.maximum(maximum, gl.gather(maximum, columns ^ (16 >> distance), 1))
    maximum = gl.convert_layout(gl.reshape(maximum, [512]), tid.type.layout)
    code = gl.minimum(
        (((maximum * (1.0 / 6.0)).to(gl.uint32, bitcast=True) + 0x7FFFFF) >> 23) & 255,
        254,
    )
    gl.barrier()
    store_f32(av, S * 560 + tid, value)
    gl.barrier()
    kof = load_i32(expert_kof, task // 12 * 8 + token)
    # Four lanes a row each quantize eight consecutive columns.
    PACK: gl.constexpr = gl.BlockedLayout([1, 8], [64, 1], [8, 1], [1, 0])
    ids = gl.arange(0, 512, layout=gl.SliceLayout(1, PACK))
    inner = gl.arange(0, 8, layout=gl.SliceLayout(0, PACK))
    native_value_at = (
        S * 560
        + gl.minimum(ids[:, None] // 32, S - 1) * 32
        + (ids[:, None] % 32) * 8
        + inner[None, :]
    )
    native_value_indices = gl.reshape(native_value_at, [4096])
    native_value_indices = gl.max_contiguous(gl.multiple_of(native_value_indices, 8), 8)
    values = gl.reshape(av.gather(native_value_indices, 0), [512, 8])
    codes = gl.convert_layout(code, ids.type.layout)
    packed = gl.amd.cdna4.scaled_downcast(
        values, codes[:, None].to(gl.uint8), "e2m1", axis=1
    )
    byte0, byte1 = gl.split(gl.reshape(packed.to(gl.uint32), [512, 2, 2]))
    pair = byte0 | (byte1 << 8)
    p0, p1 = gl.split(pair)
    word = gl.convert_layout(p0 | (p1 << 16), tid.type.layout, assert_trivial=True)
    pick = token * 16 + gl.maximum(kof, 0)
    put_pair(
        inter,
        pick * 48 + task % 12 * 4 + column,
        word,
        tag,
        (tid < S * 32) & (column < 4) & (kof >= 0),
        "gpu",
    )
    put_pair(
        isc,
        pick * 12 + task % 12,
        code,
        tag,
        (tid < S * 32) & (column == 0) & (kof >= 0),
        "gpu",
    )
    gl.barrier()

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


"""Tagged handoffs, wave predicates and the narrow BF16 GEMVs of the tail.

A tagged pair is one eight-byte word: a 32-bit payload and the step's tag, so
one atomic load returns the value and its readiness.

Modified from sgl-amd-mono (https://github.com/sgl-project/sgl-amd-mono, Apache
License 2.0) at b9f40eb562f449ce08c030df200f97c7ea716511:
``common/{ops,sync}.py`` and ``kimi_k3/gfx950/stages/gemv.py``, whose device
primitives are adapted from ROCm/ATOM (https://github.com/ROCm/ATOM, MIT
License) at 97359d6df46c59736468168b7be8bc50b482a1a9.
"""

from tokenspeed_kernel_amd._extern import release_wait_vmem, wave_all_equal
from tokenspeed_kernel_amd._triton import gl, gluon


@gluon.jit
def thread_ids():
    return gl.arange(0, 512, layout=gl.BlockedLayout([1], [64], [8], [0]))


@gluon.jit
def descriptor(ptr, nbytes=0xFFFFFFFF):
    return gl.amd.cdna3.buffer.make_buffer_descriptor(ptr, nbytes)


@gluon.jit
def put_pair(ptr, index, value, tag, mask, SCOPE: gl.constexpr):
    if value.dtype == gl.float32:
        bits = value.to(gl.uint32, bitcast=True)
    else:
        bits = value.to(gl.uint32)
    word = bits.to(gl.uint64) | (tag.to(gl.uint64) << 32)
    gl.amd.cdna3.buffer.atomic_store(
        word,
        descriptor(ptr.to(gl.pointer_type(gl.uint64))),
        index,
        mask=mask,
        sem="relaxed",
        scope=SCOPE,
    )


@gluon.jit
def wave_pending(pending):
    ready = wave_all_equal(pending.to(gl.float32), 0.0)
    # asm: an empty template (no instruction), the same type bridge as
    # moe/mxfp4/decode_stage2.py's _issue_here. The value is already one
    # scalar register per wave, but Gluon types it as a block tensor, which it
    # rejects as a loop or branch condition. A loop on it must use no LDS.
    return gl.inline_asm("", "=s,0", (ready,), gl.int32, is_pure=True) == 0


@gluon.jit
def publish(ptr, index, value, tag, mask):
    release_wait_vmem()
    gl.barrier()
    put_pair(ptr, index, value, tag, mask, "gpu")


@gluon.jit
def butterfly(value, offsets: gl.constexpr):
    # Keep the gather axis inside one wave; a 512-element gather uses LDS.
    width: gl.constexpr = 64 if offsets[0] == 32 else 32
    wave_values = gl.reshape(value, [512 // width, width])
    lane = gl.reshape(thread_ids(), [512 // width, width]) % width
    for off in gl.static_range(len(offsets)):
        wave_values = wave_values + gl.gather(wave_values, lane ^ offsets[off], 1)
    return gl.convert_layout(gl.reshape(wave_values, [512]), value.type.layout)


@gluon.jit
def bf16_round(value):
    return value.to(gl.bfloat16, fp_downcast_rounding="rtne").to(gl.float32)


@gluon.jit
def step_tag(epoch, slot):
    value = gl.load(epoch)
    return (value % (1 << 22)) * 256 + slot + 1


@gluon.jit
def raw_bf4(ptr, base):
    L: gl.constexpr = gl.BlockedLayout([1, 2], [64, 1], [8, 1], [0, 1])
    # Buffer loads take 32-bit offsets, so ``base`` is cut back to 32 bits.
    # The launcher bounds the residual blocks' offsets, which grow with their
    # runtime strides (_check_blocks); the other offsets are within a row.
    at = gl.convert_layout(base.to(gl.int32), gl.SliceLayout(1, L))
    word = gl.arange(0, 2, layout=gl.SliceLayout(0, L))
    offsets = gl.max_contiguous(
        gl.multiple_of(at[:, None] // 2 + word[None, :], [1, 2]), [1, 2]
    )
    return gl.amd.cdna3.buffer.load(
        descriptor(ptr.to(gl.pointer_type(gl.uint32))), offsets
    )


@gluon.jit
def unpack_bf4(words):
    lo, hi = gl.split(words)
    lo = gl.convert_layout(lo, gl.BlockedLayout([1], [64], [8], [0]))
    hi = gl.convert_layout(hi, gl.BlockedLayout([1], [64], [8], [0]))
    return (
        lo.to(gl.uint16).to(gl.bfloat16, bitcast=True).to(gl.float32),
        (lo >> 16).to(gl.uint16).to(gl.bfloat16, bitcast=True).to(gl.float32),
        hi.to(gl.uint16).to(gl.bfloat16, bitcast=True).to(gl.float32),
        (hi >> 16).to(gl.uint16).to(gl.bfloat16, bitcast=True).to(gl.float32),
    )


@gluon.jit
def store_bf4(ptr, base, word0, word1, mask, SCOPE: gl.constexpr):
    L: gl.constexpr = gl.BlockedLayout([1, 2], [64, 1], [8, 1], [0, 1])
    words = gl.convert_layout(gl.join(word0, word1), L)
    at = gl.convert_layout((base // 2).to(gl.int32), gl.SliceLayout(1, L))
    dd = gl.arange(0, 2, layout=gl.SliceLayout(0, L))
    live = gl.convert_layout(mask, gl.SliceLayout(1, L))
    gl.amd.cdna3.buffer.store(
        words,
        descriptor(ptr.to(gl.pointer_type(gl.uint32))),
        at[:, None] + dd[None, :],
        mask=live[:, None],
        scope=SCOPE,
    )


@gluon.jit
def _gemv_loads_range(
    wptr,
    row0,
    K: gl.constexpr,
    live,
    WAVE_ROWS: gl.constexpr,
    LDK: gl.constexpr,
    k0,
    FIRST_CHUNK: gl.constexpr,
    LAST_CHUNK: gl.constexpr,
):
    """Issue every wave's weight loads, in chunk order."""
    ML: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=False, warps_per_cta=[8, 1, 1]
    )
    # Raw dwordx4 loads, two BF16 values in each register.
    L: gl.constexpr = gl.DotOperandLayout(0, ML, 4)
    batch = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, L)))
    row = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, L)))
    kk = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(1, L)))
    desc = descriptor(
        gl.multiple_of(wptr, 16).to(gl.pointer_type(gl.uint32)),
        gl.where(live, 0xFFFFFFFF, 0).to(gl.uint32),
    )
    out = ()
    for i in gl.static_range(FIRST_CHUNK, LAST_CHUNK):
        ch = (
            gl.full([8], i, gl.int32, batch.type.layout)
            if WAVE_ROWS
            else gl.minimum(batch + 8 * i, K // 128 - 1)
        )
        for q in gl.static_range(4):
            offsets = (
                (
                    row0
                    + row[None, :, None]
                    + (batch[:, None, None] * 16 if WAVE_ROWS else 0)
                )
                * (LDK // 2)
                + k0 // 2
                + ch[:, None, None] * 64
                + q * 16
                + kk[None, None, :]
            )
            offsets = gl.max_contiguous(gl.multiple_of(offsets, [1, 1, 4]), [1, 1, 4])
            out += (
                gl.amd.cdna3.buffer.load(
                    desc,
                    offsets,
                    cache_policy=gl.amd.cdna3.CachePolicy(nontemporal=True),
                ),
            )
    return out


@gluon.jit
def gemv_loads(wptr, row0, K: gl.constexpr, live, WAVE_ROWS: gl.constexpr):
    return _gemv_loads_range(
        wptr,
        row0,
        K,
        live,
        WAVE_ROWS,
        K,
        0,
        0,
        gl.cdiv(K // 128, 1 if WAVE_ROWS else 8),
    )


@gluon.jit
def gemv_loads_strided(
    wptr, row0, K: gl.constexpr, live, WAVE_ROWS: gl.constexpr, LDK: gl.constexpr, k0
):
    """Issue a K part from a wider row-major weight matrix."""
    return _gemv_loads_range(
        wptr,
        row0,
        K,
        live,
        WAVE_ROWS,
        LDK,
        k0,
        0,
        gl.cdiv(K // 128, 1 if WAVE_ROWS else 8),
    )


@gluon.jit
def gemv_dot(xl, weights, K: gl.constexpr, S: gl.constexpr, WAVE_ROWS: gl.constexpr):
    """One independent 16x16 matrix product per wave, split across K chunks."""
    ML: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=False,
        warps_per_cta=[8, 1, 1],
    )
    XL: gl.constexpr = gl.DotOperandLayout(1, ML, 8)
    batch = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, XL)))
    kk = gl.arange(0, 32, layout=gl.SliceLayout(0, gl.SliceLayout(2, XL)))
    token = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(1, XL)))
    acc = gl.full([8, 16, 16], 0, gl.float32, ML)
    for i in gl.static_range(gl.cdiv(K // 128, 1 if WAVE_ROWS else 8)):
        ch = (
            gl.full([8], i, gl.int32, batch.type.layout) if WAVE_ROWS else batch + 8 * i
        )
        for q in gl.static_range(4):
            offsets = (
                gl.minimum(token[None, None, :], S - 1) * (K + 8)
                + gl.minimum(ch[:, None, None], K // 128 - 1) * 128
                + q * 32
                + kk[None, :, None]
            )
            x = gl.convert_layout(
                gl.reshape(xl.gather(gl.reshape(offsets, [4096]), 0), [8, 32, 16]), XL
            )
            if not WAVE_ROWS and gl.cdiv(K // 128, 8) * 8 != K // 128:
                x = gl.where(ch[:, None, None] < K // 128, x, 0)
            words = weights[i * 4 + q]
            lo = words.to(gl.uint16).to(gl.bfloat16, bitcast=True)
            hi = (words >> 16).to(gl.uint16).to(gl.bfloat16, bitcast=True)
            a = gl.convert_layout(
                gl.reshape(gl.join(lo, hi), [8, 16, 32]), gl.DotOperandLayout(0, ML, 8)
            )
            b = gl.convert_layout(x, gl.DotOperandLayout(1, ML, 8))
            acc = gl.amd.cdna3.mfma(a, b, acc)
    return acc


@gluon.jit
def gemv_reduce(acc, red, S: gl.constexpr):
    """Store each wave's lanes and registers in order, then add waves 0 to 7."""
    L: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=False, warps_per_cta=[8, 1, 1]
    )
    batch = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, L)))
    row = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, L)))
    token = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(1, L)))
    at = (
        batch[:, None, None] * 256
        + (row[None, :, None] // 4) * 64
        + token[None, None, :] * 4
        + row[None, :, None] % 4
    )
    red.scatter(
        gl.reshape(gl.convert_layout(acc, L), [2048]), gl.reshape(at, [2048]), 0
    )
    gl.barrier()
    tid = thread_ids()
    row = tid % 16
    token = gl.minimum(tid // 16, S - 1)
    value = gl.full([512], 0, gl.float32, tid.type.layout)
    for wave in gl.static_range(8):
        value = value + red.gather(wave * 256 + row // 4 * 64 + token * 4 + row % 4, 0)
    return bf16_round(value)

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


"""Wave-local shared-memory (LDS) accesses through raw LDS addresses.

Gluon's shared-memory accesses insert workgroup barriers, which waves that
diverge cannot join. These helpers load and store through the allocation's
raw LDS address instead and add no barrier: the caller orders them against
other waves. Pass the allocation itself to every access, which keeps it
alive.

Modified from sgl-amd-mono (https://github.com/sgl-project/sgl-amd-mono, Apache
License 2.0) at b9f40eb562f449ce08c030df200f97c7ea716511: ``common/ops.py`` and
``kimi_k3/gfx950/stages/moe.py``, whose device primitives are adapted from
ROCm/ATOM (https://github.com/ROCm/ATOM, MIT License) at
97359d6df46c59736468168b7be8bc50b482a1a9.
"""

from tokenspeed_kernel_amd._extern import (
    shared_load_i32_masked,
    shared_load_i64_masked,
    shared_store_i32_masked,
    shared_store_i64_masked,
)
from tokenspeed_kernel_amd._triton import gl, gluon


@gluon.jit
def shared_address(descriptor):
    # asm: an empty template (no instruction), the same type bridge as
    # moe/mxfp4/decode_stage2.py's _issue_here. Gluon has no op that exposes a
    # shared-memory allocation's LDS address. Keep the allocation as an
    # operand at every use, which keeps it alive.
    return gl.inline_asm("", "=s,0", (descriptor,), gl.int32, is_pure=False)


@gluon.jit
def load_i32(descriptor, index):
    at = index.to(gl.int32)
    return shared_load_i32_masked(
        shared_address(descriptor), at, (at == at).to(gl.int32)
    )


@gluon.jit
def load_f32(descriptor, index):
    return load_i32(descriptor, index).to(gl.float32, bitcast=True)


@gluon.jit
def store_i32(descriptor, index, value):
    at = index.to(gl.int32)
    shared_store_i32_masked(
        shared_address(descriptor), at, value.to(gl.int32), (at == at).to(gl.int32)
    )


@gluon.jit
def store_f32(descriptor, index, value):
    store_i32(descriptor, index, value.to(gl.int32, bitcast=True))


@gluon.jit
def store_mask_i32(descriptor, index, value, live):
    shared_store_i32_masked(
        shared_address(descriptor),
        index.to(gl.int32),
        value.to(gl.int32),
        live.to(gl.int32),
    )


@gluon.jit
def lane_words(value):
    # In [wave, row, token], a lane owns rows 4*(lane//16)..+3.
    value = gl.reshape(value, [8, 4, 4, 16])
    value = gl.reshape(gl.permute(value, [0, 1, 3, 2]), [512, 4])
    return gl.convert_layout(
        value, gl.BlockedLayout([1, 4], [64, 1], [8, 1], [1, 0]), assert_trivial=True
    )


@gluon.jit
def restore_words(value, LAYOUT: gl.constexpr):
    value = gl.reshape(value, [8, 4, 16, 4])
    value = gl.reshape(gl.permute(value, [0, 1, 3, 2]), [8, 16, 16])
    return gl.convert_layout(value, LAYOUT, assert_trivial=True)


@gluon.jit
def first_word(value):
    first_pair, _ = gl.split(gl.reshape(lane_words(value), [512, 2, 2]))
    first, _ = gl.split(first_pair)
    return first


@gluon.jit
def pair_indices(index):
    L64: gl.constexpr = gl.BlockedLayout([1, 2], [64, 1], [8, 1], [1, 0])
    first = gl.convert_layout(
        first_word(index) // 2, gl.SliceLayout(1, L64), assert_trivial=True
    )
    offset = gl.arange(0, 2, layout=gl.SliceLayout(0, L64))
    return gl.max_contiguous(
        gl.multiple_of(first[:, None] + offset[None, :], [1, 2]), [1, 2]
    )


# A down-projection tile: four adjacent LDS words owned by each lane.


@gluon.jit
def load_four_words(workspace, index):
    L64: gl.constexpr = gl.BlockedLayout([1, 2], [64, 1], [8, 1], [1, 0])
    packed = shared_load_i64_masked(
        shared_address(workspace) & -16,
        pair_indices(index),
        gl.full([512, 2], 1, gl.int32, L64),
    ).to(gl.uint64, bitcast=True)
    words = gl.reshape(
        gl.join(packed.to(gl.uint32), (packed >> 32).to(gl.uint32)), [512, 4]
    )
    return restore_words(words, index.type.layout)


@gluon.jit
def store_four_words(workspace, index, value, live):
    L64: gl.constexpr = gl.BlockedLayout([1, 2], [64, 1], [8, 1], [1, 0])
    lo, hi = gl.split(gl.reshape(lane_words(value), [512, 2, 2]))
    packed = lo.to(gl.uint32).to(gl.uint64) | (hi.to(gl.uint32).to(gl.uint64) << 32)
    packed = gl.convert_layout(
        packed.to(gl.int64, bitcast=True), L64, assert_trivial=True
    )
    live = live & gl.full([8, 16, 16], True, gl.int1, value.type.layout)
    mask = gl.convert_layout(
        first_word(live).to(gl.int32), gl.SliceLayout(1, L64), assert_trivial=True
    )
    shared_store_i64_masked(
        shared_address(workspace) & -16, pair_indices(index), packed, mask[:, None]
    )

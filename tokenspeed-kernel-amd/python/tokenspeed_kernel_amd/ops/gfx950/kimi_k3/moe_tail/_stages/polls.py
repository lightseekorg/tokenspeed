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


"""Bounded waits on tagged pairs.

One atomic load returns both the payload and its tag. After every 64 polls a
wait checks its deadline and the stop flags (:func:`sentinel.guard_tick`), so a
failed rank stops every rank instead of hanging.

Modified from sgl-amd-mono (https://github.com/sgl-project/sgl-amd-mono, Apache
License 2.0) at b9f40eb562f449ce08c030df200f97c7ea716511: ``common/sync.py``,
whose device primitives are adapted from ROCm/ATOM
(https://github.com/ROCm/ATOM, MIT License) at
97359d6df46c59736468168b7be8bc50b482a1a9.
"""

from tokenspeed_kernel_amd._extern import sleep_one
from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.common import (
    descriptor,
    wave_pending,
)
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.sentinel import (
    guard_tick,
)


@gluon.jit
def poll_pairs(ptr, indices, tag, mask, SCOPE: gl.constexpr, guard):
    """Wait until every masked pair at ``indices`` carries ``tag``; return
    their payloads as FP32 bits."""
    epoch, own, peers, rank, FLAGS_OFFSET, SITE = guard
    desc = descriptor(ptr.to(gl.pointer_type(gl.uint64)))
    words = ()
    pending = gl.full(mask.shape, False, gl.int1, mask.type.layout)
    for i in gl.static_range(len(indices)):
        word = gl.amd.cdna3.buffer.atomic_load(
            desc,
            indices[i],
            mask=mask,
            sem="relaxed",
            scope=SCOPE,
        )
        words += (word,)
        pending = pending | (mask & ((word >> 32).to(gl.int32) != tag))
    polls = 0
    first_check = gl.full((), 0, gl.uint64, gl.BlockedLayout([1], [64], [8], [0]))
    # Each wave polls until its own words arrive; s_sleep has no Gluon op.
    while wave_pending(pending):
        sleep_one()
        next_words = ()
        pending = gl.full(mask.shape, False, gl.int1, mask.type.layout)
        for i in gl.static_range(len(indices)):
            word = gl.amd.cdna3.buffer.atomic_load(
                desc,
                indices[i],
                mask=mask,
                sem="relaxed",
                scope=SCOPE,
            )
            next_words += (word,)
            pending = pending | (mask & ((word >> 32).to(gl.int32) != tag))
        words = next_words
        polls += 1
        if polls % 64 == 0:
            first_check = guard_tick(
                epoch,
                own,
                peers,
                rank,
                polls,
                first_check,
                FLAGS_OFFSET,
                SITE,
            )
    values = ()
    for i in gl.static_range(len(indices)):
        values += (words[i].to(gl.uint32).to(gl.float32, bitcast=True),)
    return values


@gluon.jit
def poll_regions(regions, indices, masks, tag, SCOPE: gl.constexpr, guard):
    """:func:`poll_pairs` over one pair in each of several regions."""
    epoch, own, peers, rank, FLAGS_OFFSET, SITE = guard
    words = ()
    pending = gl.full(masks[0].shape, False, gl.int1, masks[0].type.layout)
    for i in gl.static_range(len(indices)):
        word = gl.amd.cdna3.buffer.atomic_load(
            descriptor(regions[i].to(gl.pointer_type(gl.uint64))),
            indices[i],
            mask=masks[i],
            sem="relaxed",
            scope=SCOPE,
        )
        words += (word,)
        pending = pending | (masks[i] & ((word >> 32).to(gl.int32) != tag))
    polls = 0
    first_check = gl.full((), 0, gl.uint64, gl.BlockedLayout([1], [64], [8], [0]))
    # Each wave polls until its own words arrive; s_sleep has no Gluon op.
    while wave_pending(pending):
        sleep_one()
        next_words = ()
        pending = gl.full(masks[0].shape, False, gl.int1, masks[0].type.layout)
        for i in gl.static_range(len(indices)):
            word = gl.amd.cdna3.buffer.atomic_load(
                descriptor(regions[i].to(gl.pointer_type(gl.uint64))),
                indices[i],
                mask=masks[i],
                sem="relaxed",
                scope=SCOPE,
            )
            next_words += (word,)
            pending = pending | (masks[i] & ((word >> 32).to(gl.int32) != tag))
        words = next_words
        polls += 1
        if polls % 64 == 0:
            first_check = guard_tick(
                epoch, own, peers, rank, polls, first_check, FLAGS_OFFSET, SITE
            )
    values = ()
    for i in gl.static_range(len(indices)):
        values += (words[i].to(gl.uint32).to(gl.float32, bitcast=True),)
    return values

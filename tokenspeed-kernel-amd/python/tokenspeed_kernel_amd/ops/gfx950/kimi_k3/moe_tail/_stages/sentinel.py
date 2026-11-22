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


"""Four-byte exchange words between ranks, and the bounded waits on them.

Each word is two BF16 values a rank sends to one peer; 0xFFFFFFFF marks a
word not sent yet. Each word has exactly one consumer, which re-arms it after
reading. The caller's layer slot selects the parity half.

Every wait of the tail, on these words or on tagged pairs, is bounded. After
every 64 unsuccessful polls it runs :func:`guard_tick`:

- The time limit is per wait and counts from the wait's 64th poll. It is
  step words 1-2, in 100 MHz clock ticks (low word first), compared in units
  of 256 ticks as int32, so it must be below 2**39 ticks (about 91 minutes).
- A wait past its limit writes its site number plus one to step word 3, sets
  its rank's stop flag on every rank, and stops the launch with a volatile
  store to the null page: the process aborts with a GPU memory access fault.
- Any wait also stops that way when it finds step word 3, 4 or 5 nonzero, or
  any rank's stop flag set. Only a timed-out wait writes word 3; nothing in
  this package writes words 4 and 5, which are reserved for a caller that
  wants to stop the waits (they start at zero).
- No host-readable record is kept: the step words are device memory, and the
  process ends with the fault.

The site numbers (2 and 3 are unused):

- 1: the output projection's reduction across ranks (``oproj``);
- 4, 5: the attention-residual source weights and the mix (``mix``);
- 6, 7, 8: the router, latent and shared gate/up projections reading the
  post-attention-norm rows (``projections``);
- 9: the router logits from every rank (``route``);
- 10: the shared expert's gate/up outputs (``shared_experts``);
- 11: the quantized latent rows from every rank (``latent``);
- 12: the rows' experts and weights (``route``);
- 13, 15: the latent norm's reduction across ranks and its sums of squares
  (``latent``);
- 14: the layer output's reduction across ranks (``final``);
- 16: the routed experts' intermediates (``down``);
- 17: the shared expert's activations (``shared_experts``);
- 18: the latent norm and shared down outputs, for the latent up projection
  (``final``);
- 19: the layer's output rows, for the next layer's mix (``next_mix``).

Modified from sgl-amd-mono (https://github.com/sgl-project/sgl-amd-mono, Apache
License 2.0) at b9f40eb562f449ce08c030df200f97c7ea716511:
``common/{sync,guard}.py``, whose device primitives are adapted from ROCm/ATOM
(https://github.com/ROCm/ATOM, MIT License) at
97359d6df46c59736468168b7be8bc50b482a1a9.
"""

from tokenspeed_kernel_amd._extern import (
    realtime_clock,
    release_wait_vmem,
    sleep_one,
    stop_on_fault,
)
from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.common import (
    descriptor,
    wave_pending,
)
from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail._stages.extern import (
    load_peer_base,
)


@gluon.jit
def sent_word(word):
    # A payload equal to the not-sent marker (two BF16 NaNs) is sent as two
    # canonical NaNs instead.
    return gl.where(word.to(gl.uint32) == 0xFFFFFFFF, 0xFFC0FFC0, word.to(gl.uint32))


@gluon.jit
def sent_ix(src, col, token, S: gl.constexpr, WIDTH: gl.constexpr):
    return ((src * (WIDTH // 16) + col // 16) * S + token) * 8 + col % 16 // 2


@gluon.jit
def sent_item(item, S: gl.constexpr):
    group = item // (8 * S)
    remainder = item % (8 * S)
    return remainder // 8, group * 8 + remainder % 8


@gluon.jit
def put_sent(base, index, word, live):
    gl.amd.cdna3.buffer.atomic_store(
        sent_word(word),
        descriptor(base.to(gl.pointer_type(gl.uint32))),
        index,
        mask=live,
        sem="relaxed",
        scope="sys",
    )


@gluon.jit
def rearm_sent(base, indices, live):
    desc = descriptor(base.to(gl.pointer_type(gl.uint32)))
    for i in gl.static_range(len(indices)):
        gl.amd.cdna3.buffer.atomic_store(
            gl.full(live.shape, 0xFFFFFFFF, gl.uint32, live.type.layout),
            desc,
            indices[i],
            mask=live,
            sem="relaxed",
            scope="sys",
        )


@gluon.jit
def guard_tick(
    epoch,
    own,
    peers,
    rank,
    polls,
    first_check,
    FLAGS_OFFSET: gl.constexpr,
    SITE: gl.constexpr,
):
    """Check the wait's deadline and the stop flags, after every 64
    unsuccessful polls; on a missed deadline set every rank's stop flag
    and stop."""
    # The 100 MHz clock with its wait visible to the compiler (Triton's helper
    # is inline asm).
    now = realtime_clock()
    first_check = gl.where(polls == 64, now, first_check)
    # The deadline does not change during a launch, so this read may be
    # cached. The error and stop-flag words below are read coherently.
    deadline = gl.load(epoch + 1).to(gl.uint32).to(gl.uint64) | (
        gl.load(epoch + 2).to(gl.uint32).to(gl.uint64) << 32
    )
    late = ((now - first_check) >> 8).to(gl.int32) > (deadline >> 8).to(gl.int32)
    zeros = gl.full([512], 0, gl.int32, gl.BlockedLayout([1], [64], [8], [0]))
    if late:
        gl.amd.cdna3.buffer.atomic_store(
            zeros + SITE + 1,
            descriptor(epoch),
            zeros + 3,
            sem="relaxed",
            scope="gpu",
        )
        for peer in gl.static_range(8):
            # Scalar peer base, so the flag store below needs no lane loop.
            address = load_peer_base(peers + peer)
            flags = (address + FLAGS_OFFSET).to(gl.pointer_type(gl.int32))
            gl.amd.cdna3.buffer.atomic_store(
                zeros + 1,
                descriptor(flags),
                zeros + rank,
                sem="relaxed",
                scope="sys",
            )
        # Gluon has no trap that keeps the code after it reachable, so a null-page
        # store stops the launch.
        release_wait_vmem()
        stop_on_fault()
    stopped = zeros
    epoch_desc = descriptor(epoch)
    for word in gl.static_range(3, 6):
        stopped = stopped | gl.amd.cdna3.buffer.atomic_load(
            epoch_desc,
            zeros + word,
            sem="relaxed",
            scope="gpu",
        )
    flags_desc = descriptor((own + FLAGS_OFFSET).to(gl.pointer_type(gl.int32)))
    for peer in gl.static_range(8):
        stopped = stopped | gl.amd.cdna3.buffer.atomic_load(
            flags_desc,
            zeros + peer,
            sem="relaxed",
            scope="sys",
        )
    # Waves reach this check in different waits, so the decision is per wave:
    # a workgroup-wide reduction would add a barrier they cannot all reach.
    if wave_pending(stopped != 0):
        stop_on_fault()
    return first_check


@gluon.jit
def poll_sent(
    base,
    indices,
    live,
    epoch,
    own,
    peers,
    rank,
    FLAGS_OFFSET: gl.constexpr,
    SITE: gl.constexpr,
    REARM: gl.constexpr,
):
    """Acquire a batch with the payload in the same load that checks readiness."""
    desc = descriptor(base.to(gl.pointer_type(gl.uint32)))
    words = ()
    pending = gl.full(live.shape, False, gl.int1, live.type.layout)
    for i in gl.static_range(len(indices)):
        word = gl.amd.cdna3.buffer.atomic_load(
            desc,
            indices[i],
            mask=live,
            sem="relaxed",
            scope="sys",
        )
        words += (word,)
        pending = pending | (live & (word == 0xFFFFFFFF))
    polls = 0
    first_check = gl.full((), 0, gl.uint64, gl.BlockedLayout([1], [64], [8], [0]))
    # Each wave polls until its own words arrive; s_sleep has no Gluon op.
    while wave_pending(pending):
        sleep_one()
        next_words = ()
        pending = gl.full(live.shape, False, gl.int1, live.type.layout)
        for i in gl.static_range(len(indices)):
            word = gl.amd.cdna3.buffer.atomic_load(
                desc,
                indices[i],
                mask=live,
                sem="relaxed",
                scope="sys",
            )
            next_words += (word,)
            pending = pending | (live & (word == 0xFFFFFFFF))
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
    if REARM:
        rearm_sent(base, indices, live)
    return words


@gluon.jit
def sent_sum(
    base,
    col,
    token,
    live,
    epoch,
    own,
    peers,
    rank,
    S: gl.constexpr,
    WIDTH: gl.constexpr,
    FLAGS_OFFSET: gl.constexpr,
    SITE: gl.constexpr,
    REARM: gl.constexpr,
):
    indices = ()
    for src in gl.static_range(8):
        indices += (sent_ix(src, col, token, S, WIDTH),)
    words = poll_sent(
        base, indices, live, epoch, own, peers, rank, FLAGS_OFFSET, SITE, REARM
    )
    lo = (words[0] << 16).to(gl.float32, bitcast=True)
    hi = (words[0] & 0xFFFF0000).to(gl.float32, bitcast=True)
    for src in gl.static_range(1, 8):
        lo = lo + (words[src] << 16).to(gl.float32, bitcast=True)
        hi = hi + (words[src] & 0xFFFF0000).to(gl.float32, bitcast=True)
    return lo, hi

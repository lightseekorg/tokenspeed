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

"""Exchange reduce fused with the sublayer output's short conv
(TokenSpeed ``inkling_ring_sconv``, decode kernel).

The output conv is per channel, so a reduce task that owns ``COLUMNS``
channels of every row also runs their conv. For the row at position ``p``,
taps ``i = 0 .. 3`` are the raw inputs at ``p - 3 + i`` (tap 3 is the row
itself):

    acc = fma(x3, w3, fma(x2, w2, fma(x1, w1, fma(x0, w0, 0))))
    out = bf16((acc + x3) * 5 ** -0.5)

A tap of an earlier row of the request in this launch is that row's reduced
input: the task folds the partials of the shifted rows again, which gives the
same bits. An older tap comes from the conv ring, row ``position % rows`` (the ring's rows a slot) of
the request's slot (0 before position 0, or for a padded request, slot -1).
Every row writes its raw input to its ring row. The server sizes the ring
as ``3 + K`` rows for verify blocks of ``K`` tokens (11 at ``K = 8``, 7 at
``K = 4``), so in a verify of ``K`` rows row ``j`` reads only positions
before the request's first row from the ring, and the ring rows of this
launch's positions are not among them.

A ring row is the request's channels ``K | V | attention output | MLP
output`` (``inkling_conv_stream_layout``: ``dims.ring_width`` =
``2 kv_dim + 2 hidden``, the attention stream at ``dims.attn_channel0``,
the MLP stream at ``dims.mlp_channel0``). The math is per channel, so the
width only places the channels; the reduce tasks own ``COLUMNS`` channels
each (``hidden / COLUMNS`` tasks).
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import aggregate, gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.monokernel import mx, schedule, trace
from tokenspeed_kernel_amd.ops.gfx950.monokernel.inkling2 import projection

TAPS = gl.constexpr(4)
OUTPUT_SCALE = gl.constexpr(0.4472135954999579)  # 5 ** -0.5


@aggregate
class Ring:
    """A launch's conv ring and row positions (one ring a layer)."""

    ring: gl.tensor  # BF16 [slots, rows, width]
    slots: gl.tensor  # int32 [requests] ring slot (-1: padded request)
    positions: gl.tensor  # int64 [rows] position of each row's token
    chain: gl.constexpr  # rows a request (the plan's chain)
    rows: gl.constexpr  # ring rows a slot (the server's, fixed per server)
    width: gl.constexpr  # channels a ring row (``dims.ring_width``)

    @gluon.constexpr_function
    def __init__(self, ring, slots, positions, chain, rows, width):
        self.ring = ring
        self.slots = slots
        self.positions = positions
        self.chain = gl.constexpr(chain)
        self.rows = gl.constexpr(rows)
        self.width = gl.constexpr(width)


@gluon.jit
def ring_slot(ring, rows, back: gl.constexpr):
    """``(slot, position)`` of the token ``back`` positions before each row's
    (``rows``: a row index or a tensor of them). The chain of the request
    that a row belongs to is the plan's: ``ring.slots`` has one entry a
    request."""
    CHAIN: gl.constexpr = ring.chain
    slot = gl.load(ring.slots + rows // CHAIN)
    position = gl.load(ring.positions + rows) - back
    return slot, position


@gluon.jit
def ring_offset(ring, slot, position):
    """Element offset into ``ring.ring`` (:class:`Ring`) of ring row
    ``position % ring.rows`` of ``slot``."""
    ROWS: gl.constexpr = ring.rows
    WIDTH: gl.constexpr = ring.width
    return slot.to(gl.int64) * (ROWS * WIDTH) + (position % ROWS) * WIDTH


@gluon.jit
def _ring_offsets(ring, rows, channels, back: gl.constexpr):
    """``(offsets, valid)`` into ``ring.ring`` of channels ``channels`` at
    ``back`` positions before each row's, ``[RP, C]``."""
    slot, position = ring_slot(ring, rows, back)
    valid = (slot >= 0) & (position >= 0)
    return (
        ring_offset(ring, slot, position)[:, None] + channels[None, :],
        valid[:, None],
    )


@gluon.jit
def reduce_conv_stage(
    mb,
    task,
    part: gl.constexpr,
    flag: gl.constexpr,
    GATHERED: gl.constexpr,
    out: gl.constexpr,
    term,
    taps,
    ring,
    CHANNEL0: gl.constexpr,
    N: gl.constexpr,
    COLUMNS: gl.constexpr,
    STAGE: gl.constexpr,
):
    """Reduce-and-conv task ``task``: channels ``COLUMNS * task ..`` of every
    row.

    Args:
        part, flag: The peer partials and their flags
            (``projection.partial_stage``), or with ``GATHERED`` the folded
            rows and their flags (``projection.scatter_fold_stage``).
        out: Region of the conv output's BF16 pairs.
        term: Global BF16 ``[rows, N]`` that receives the conv output (the
            next layer's pending term).
        taps: BF16 ``[N, 4]`` conv weights, tap 3 the current row.
        ring: :class:`Ring`; this stream's channels start at ``CHANNEL0``.
    """
    mb.stamp(STAGE, trace.START)
    R: gl.constexpr = schedule.rows(mb.plan)
    RP: gl.constexpr = triton.next_power_of_2(R)
    CHAIN: gl.constexpr = schedule.chain(mb.plan)
    HALF: gl.constexpr = COLUMNS // 2
    layout: gl.constexpr = projection.word_layout(gl.num_warps())
    r = gl.arange(0, RP, gl.SliceLayout(1, layout))
    row = gl.minimum(r, R - 1)
    tok = row % CHAIN
    c = gl.arange(0, HALF, gl.SliceLayout(0, layout))
    channel = task * COLUMNS + 2 * c  # even channel of each pair
    # The weights and the ring taps do not depend on the partials: load them
    # before the poll.
    w_lo = ()
    w_hi = ()
    for i in gl.static_range(TAPS):
        w_lo += (gl.load(taps + channel * TAPS + i).to(gl.float32),)
        w_hi += (gl.load(taps + (channel + 1) * TAPS + i).to(gl.float32),)
    old_lo = ()
    old_hi = ()
    for i in gl.static_range(TAPS - 1):
        offsets, valid = _ring_offsets(ring, row, CHANNEL0 + channel, 3 - i)
        old_lo += (gl.load(ring.ring + offsets, valid, other=0.0).to(gl.float32),)
        old_hi += (gl.load(ring.ring + offsets + 1, valid, other=0.0).to(gl.float32),)
    # Tap 3 is the row itself: a placeholder keeps the tuples indexable by tap.
    old_lo += (old_lo[0],)
    old_hi += (old_hi[0],)
    if GATHERED:
        lanes: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
        mb.poll_flags(flag, task + gl.arange(0, 1, lanes))
        gl.barrier()
    else:
        projection.poll_partials(mb, task, flag, N // COLUMNS)
    mb.stamp(STAGE, trace.READY)
    acc_lo = gl.zeros([RP, HALF], gl.float32, layout)
    acc_hi = gl.zeros([RP, HALF], gl.float32, layout)
    x_lo = acc_lo
    x_hi = acc_hi
    for i in gl.static_range(TAPS):
        back = TAPS - 1 - i
        if GATHERED:
            got = mb.load_words(
                part,
                (2 * mb.row_offset(gl.maximum(row - back, 0)) + task * HALF)[:, None]
                + c[None, :],
            )
            x_lo = (got << 16).to(gl.float32, bitcast=True)
            x_hi = (got & -65536).to(gl.float32, bitcast=True)
        else:
            x_lo, x_hi = projection.fold_rows(
                mb, gl.maximum(row - back, 0), task, part, N, HALF
            )
        if back > 0:
            # Earlier rows of the request come from this launch, older ones
            # from the ring.
            here = (tok >= back)[:, None]
            x_lo = gl.where(here, x_lo, old_lo[i])
            x_hi = gl.where(here, x_hi, old_hi[i])
        acc_lo = gl.fma(x_lo, w_lo[i][None, :], acc_lo)
        acc_hi = gl.fma(x_hi, w_hi[i][None, :], acc_hi)
    # x_lo / x_hi now hold the current rows (tap 3).
    y_lo = (acc_lo + x_lo) * OUTPUT_SCALE
    y_hi = (acc_hi + x_hi) * OUTPUT_SCALE
    live = (r < R)[:, None]
    offsets, valid = _ring_offsets(ring, row, CHANNEL0 + channel, 0)
    gl.store(ring.ring + offsets, x_lo.to(gl.bfloat16), valid & live)
    gl.store(ring.ring + offsets + 1, x_hi.to(gl.bfloat16), valid & live)
    word = gl.convert_layout(
        mx.pack_bf16_pairs(gl.join(y_lo.to(gl.bfloat16), y_hi.to(gl.bfloat16))),
        layout,
    )
    stride: gl.constexpr = schedule.row_pairs(mb.plan)
    pair = task * HALF + c
    mb.put(out, (row * stride)[:, None] + pair[None, :], word, mask=live)
    term_words = term.to(gl.pointer_type(gl.int32), bitcast=True)
    gl.store(term_words + (row * (N // 2))[:, None] + pair[None, :], word, mask=live)
    mb.stamp(STAGE, trace.END)

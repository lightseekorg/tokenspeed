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

"""Attention output projection and residual combine (C1).

160 tasks, each 16 hidden columns. A task computes ``out = o @ W_o.T``
(MXFP4 ``[HIDDEN, VALUE]`` on the attention output's MXFP8 row, K split over
8 waves). Then it runs TokenSpeed's ``gated_residual_combine`` for the 4
streams of those columns: ``h_b + bf16(out) * 2 sigmoid(bf16(inject_b))``.
The combine uses the attention mixer's inject logits (sums of the two stream
halves). The new streams go to the ``w`` region for the MLP mixer.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.megakernel import mx, schedule, trace
from tokenspeed_kernel_amd.ops.gfx950.megakernel.plan import Stage, placement
from tokenspeed_kernel_amd.ops.gfx950.megakernel.qwen4_exp.dims import HIDDEN

VALUE = gl.constexpr(6144)
TASKS = gl.constexpr(160)

REGIONS = (("w", 5120),)  # BF16 pairs of the 4 combined streams


def stages(bases: dict[str, int], P: str, src: tuple[str, ...]) -> tuple[Stage, ...]:
    """C1, placed by ``bases``. It reads the attention output (``o8`` /
    ``oe``) and the inject logits of the mixer with prefix ``P``. It also
    reads the mailbox region ``src`` of the input streams, if any."""
    (c1,) = placement(bases, ("c1",))
    return (
        Stage("c1", int(TASKS), c1, reads=("o8", "oe", P + "inj") + src, writes=("w",)),
    )


@gluon.constexpr_function
def _stream_columns(num_warps):
    """``[rows, 4, 16]`` (row, stream, column) with a BF16 pair per lane."""
    return gl.BlockedLayout([1, 1, 2], [1, 8, 8], [1, num_warps, 1], [2, 1, 0])


@gluon.constexpr_function
def _extra_regions(P, SRC):
    """Regions C1 polls with its MXFP8 rows: the inject logits, and the
    input streams when they come from a mailbox."""
    return (P + "inj",) if SRC == "" else (P + "inj", SRC)


@gluon.jit
def out_stage(
    mb,
    task,
    w,
    ws,
    streams,
    O: gl.constexpr,  # noqa: E741
    P: gl.constexpr,
    STAGE: gl.constexpr,
):
    """C1 task ``task``: hidden columns ``16 task .. 16 task + 15`` of every
    row.

    Args:
        w, ws: ``o_proj`` / ``out_proj`` ``[HIDDEN, VALUE]`` as N16 tiles and
            scale dwords.
        streams: The layer's input :class:`hc.Streams`.
        O: Region prefix of the attention output's MXFP8 row (``o8``/``oe``).
        P: Region prefix of the attention mixer (its ``inj`` logits).
    """
    R: gl.constexpr = schedule.rows(mb.plan)
    SRC: gl.constexpr = streams.src
    mb.stamp(STAGE, trace.START)
    tiles, scales = mx.load_weight_tiles(
        w,
        ws,
        nb0=task,
        K=VALUE,
        kit0=0,
        KITS=48,
        SPLIT=8,
        NW=1,
        GROUP=1,
        GROUP_STRIDE=0,
        SCALES="k4",
    )
    RP: gl.constexpr = triton.next_power_of_2(R)
    # [rows, 4 streams, 16 columns], a BF16 pair per lane.
    sc: gl.constexpr = _stream_columns(gl.num_warps())
    r = gl.arange(0, RP, gl.SliceLayout(1, gl.SliceLayout(2, sc)))[:, None, None]
    b = gl.arange(0, 4, gl.SliceLayout(0, gl.SliceLayout(2, sc)))[None, :, None]
    c = gl.arange(0, 16, gl.SliceLayout(0, gl.SliceLayout(1, sc)))[None, None, :]
    column = b * HIDDEN + task * 16 + c
    if SRC == "":
        residual = gl.load(
            streams.element(r, column),
            mask=r < R,
            other=0.0,
        ).to(gl.float32)
    gl.amd.hint.sched_barrier()
    # One poll: every row's MXFP8 attention output, inject half-logits
    # ([rows, 4 streams, 2 halves]) and, from a mailbox, residual columns.
    il: gl.constexpr = gl.BlockedLayout(
        [1, 1, 2], [1, 64, 1], [1, gl.num_warps(), 1], [2, 1, 0]
    )
    ri = gl.arange(0, RP, gl.SliceLayout(1, gl.SliceLayout(2, il)))[:, None, None]
    bi = gl.arange(0, 4, gl.SliceLayout(0, gl.SliceLayout(2, il)))[None, :, None]
    hi = gl.arange(0, 2, gl.SliceLayout(0, gl.SliceLayout(1, il)))[None, None, :]
    indices = (mb.row_offset(ri) + hi * 4 + bi,)
    if SRC != "":
        indices = indices + (mb.row_offset(r) + column // 2,)
    x_smem, xs_smem, extras = mx.load_mxfp8_rows(
        mb, O + "o8", O + "oe", 48, R, _extra_regions(P, SRC), indices
    )
    if SRC != "":
        residual = mx.bf16_half(extras[1], column)
    logit = gl.sum(extras[0].to(gl.float32, bitcast=True), axis=2)
    gate = 2.0 * mx.sigmoid(logit.to(gl.bfloat16).to(gl.float32))
    gate = gl.convert_layout(gate, gl.SliceLayout(2, sc))[:, :, None]
    mb.stamp(STAGE, trace.READY)
    outs = mx.gemv(
        x_smem,
        xs_smem,
        tiles,
        scales,
        SPLIT=8,
        NW=1,
        SEG=0,
        SEG_STRIDE=0,
        FIRST=0,
        ROWS=R,
    )
    mb.stamp(STAGE, trace.COMPUTED)
    out = gl.convert_layout(outs.to(gl.bfloat16).to(gl.float32), gl.SliceLayout(1, sc))
    streams = (residual + out[:, None, :] * gate).to(gl.bfloat16)
    word = mx.pack_bf16_pairs(gl.reshape(streams, [RP, 4, 8, 2]))
    wl: gl.constexpr = word.type.layout
    rw = gl.arange(0, RP, gl.SliceLayout(1, gl.SliceLayout(2, wl)))[:, None, None]
    sw = gl.arange(0, 4, gl.SliceLayout(0, gl.SliceLayout(2, wl)))[None, :, None]
    pw = gl.arange(0, 8, gl.SliceLayout(0, gl.SliceLayout(1, wl)))[None, None, :]
    mb.put(
        "w",
        mb.row_offset(rw) + sw * (HIDDEN // 2) + task * 8 + pw,
        word,
        mask=rw < R,
    )
    mb.stamp(STAGE, trace.END)

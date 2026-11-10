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

"""QKVG: the fused q | k | v | output-gate projection of one rank.

The rank's BF16 weight is ``[2304, 8192]``: 144 blocks of 16 rows. A task
takes one block and one half of K (128 KiB): one CU cannot stream 256 KiB
as fast as the memory can, so the q | k | v blocks spread over 160 CTAs.

* qkvg, 160 tasks: the q | k | v blocks (rows 0 .. 1279).
* gate, 128 tasks: the output-gate blocks (rows 1280 .. 2303), on CTAs
  that run a qkvg task first. Attention does not wait for the gate, so its
  weights load after the q | k | v weights.
* qnext, 160 tasks (a launch whose next layer is fused too, see
  ``layer.layer_plan``): the next layer's qkvg tasks at the end of this
  launch. Their input is this launch's output rows, which the out tasks
  publish as plain words (``moe_tail``); their partial sums go to global
  memory, where the next launch's head and attention tasks read them with
  plain loads instead of polling ``qkvg``. That launch has no qkvg stage:
  its attention starts at once.

A task issues its weight loads first, then reads its K half of the
attention input rows that the previous launch wrote, and publishes its 16
FP32 partial sums per row. A consumer adds the two halves and rounds the
sum to BF16, as the regular path rounds its FP32 sums.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.monokernel import bf16, schedule, trace
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.dims import (
    GATE_ROW,
    HIDDEN,
    QKVG,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import Stage, placement

KSPLIT = gl.constexpr(2)
QKV_BLOCKS = gl.constexpr(GATE_ROW // 16)  # 80
GATE_BLOCKS = gl.constexpr((QKVG - GATE_ROW) // 16)  # 64

# FP32 partial sums [K half][row]: of q | k | v, and of the gate.
REGIONS = (
    ("qkvg", int(KSPLIT) * int(GATE_ROW)),
    ("gate", int(KSPLIT) * (int(QKVG) - int(GATE_ROW))),
)


# A projection row in global memory (qnext's output, the next launch's input):
# the qkvg region's FP32 partial sums of one token row, [K half][q | k | v].
PROJ_WORDS = gl.constexpr(int(KSPLIT) * int(GATE_ROW))


def stages(bases: dict[str, int], own: bool) -> tuple[Stage, ...]:
    """QKVG (when the launch computes its own projection: ``own``) and gate,
    placed by ``bases``."""
    qkv, gate = placement(bases, ("qkvg", "gate"))
    return (
        (Stage("qkvg", int(KSPLIT * QKV_BLOCKS), qkv, writes=("qkvg",)),) if own else ()
    ) + (
        Stage("gate", int(KSPLIT * GATE_BLOCKS), gate, writes=("gate",)),
    )


def next_stages(bases: dict[str, int]) -> tuple[Stage, ...]:
    """The next layer's qkvg tasks (qnext), placed by ``bases``."""
    (qnext,) = placement(bases, ("qnext",))
    return (Stage("qnext", int(KSPLIT * QKV_BLOCKS), qnext, reads=("anx", "anxf")),)


@gluon.jit
def _block_stage(mb, task, w, a, a_stride, FIRST: gl.constexpr, N: gl.constexpr):
    """Task ``task`` of a stage of ``N`` rows from block ``FIRST``: block
    ``FIRST + task // 2``, K half ``task % 2``."""
    R: gl.constexpr = schedule.rows(mb.plan)
    blk = task // KSPLIT
    half = task % KSPLIT
    KITS: gl.constexpr = HIDDEN // 32 // KSPLIT
    tiles = bf16.load_weight_tiles(
        w,
        nb0=FIRST + blk,
        K=HIDDEN,
        kit0=half * KITS,
        KITS=KITS,
        SPLIT=gl.num_warps(),
        NW=1,
        GROUP=1,
        GROUP_STRIDE=0,
    )
    x_smem = bf16.load_global_rows(
        a, a_stride, half * (HIDDEN // KSPLIT), HIDDEN // KSPLIT, R
    )
    out = bf16.gemv(x_smem, tiles, FIRST=0, ROWS=R)  # FP32 [RP, 16]
    return out, half * N + blk * 16


@gluon.jit
def _put_rows(mb, region: gl.constexpr, out, at):
    R: gl.constexpr = schedule.rows(mb.plan)
    rows = gl.arange(0, out.shape[0], gl.SliceLayout(1, out.type.layout))[:, None]
    for r in gl.static_range(R):
        v = gl.sum(gl.where(rows == r, out, 0.0), axis=0)
        i = at + gl.arange(0, 16, v.type.layout)
        mb.row(r).put(region, i, v.to(gl.int32, bitcast=True))


@gluon.jit
def qkvg_stage(mb, task, w, a, a_stride, STAGE: gl.constexpr):
    """QKVG task ``task``: half ``task % 2`` of K of q | k | v rows
    ``16 (task // 2) ..`` of every token row.

    Args:
        w: The rank's fused weight, BF16 ``[2304, 8192]`` (``weights.bf16_tiles``).
        a: The attention input, BF16 ``[rows, 8192]`` with row stride
            ``a_stride`` (the previous launch's output).
    """
    mb.stamp(STAGE, trace.START)
    out, at = _block_stage(mb, task, w, a, a_stride, 0, GATE_ROW)
    mb.stamp(STAGE, trace.COMPUTED)
    _put_rows(mb, "qkvg", out, at)
    mb.stamp(STAGE, trace.END)


@gluon.jit
def gate_stage(mb, task, w, a, a_stride, STAGE: gl.constexpr):
    """Gate task ``task``: half ``task % 2`` of K of gate rows ``16 (task //
    2) ..`` (rows ``1280 + 16 (task // 2) ..`` of ``w``), as :func:`qkvg_stage`."""
    mb.stamp(STAGE, trace.START)
    out, at = _block_stage(mb, task, w, a, a_stride, QKV_BLOCKS, QKVG - GATE_ROW)
    mb.stamp(STAGE, trace.COMPUTED)
    _put_rows(mb, "gate", out, at)
    mb.stamp(STAGE, trace.END)


@gluon.jit
def qnext_stage(mb, task, w, proj, OUT_PER_ROW: gl.constexpr, STAGE: gl.constexpr):
    """Qnext task ``task``: :func:`qkvg_stage` of the next layer, from this
    launch's output rows.

    It issues its weight loads, polls the flags of the out tasks of its K
    half in every row, reads their words (``anx``), and stores its 16 FP32
    partial sums per row at ``proj`` + row * ``PROJ_WORDS``, as the qkvg
    region holds them.

    Args:
        w: The next layer's fused weight, BF16 ``[2304, 8192]``.
        proj: int32 ``[rows, PROJ_WORDS]``: the next launch's projection.
        OUT_PER_ROW: Out tasks a row; each publishes ``8192 / OUT_PER_ROW``
            values of its row and raises flag ``task`` of ``anxf``.
    """
    mb.stamp(STAGE, trace.START)
    tiles = qnext_tiles(w, task)
    qnext_finish(mb, task, tiles, proj, OUT_PER_ROW, STAGE)


@gluon.jit
def qnext_tiles(w, task):
    """Issue qnext task ``task``'s weight loads (128 KiB; nothing waits)."""
    KITS: gl.constexpr = HIDDEN // 32 // KSPLIT
    return bf16.load_weight_tiles(
        w,
        nb0=task // KSPLIT,
        K=HIDDEN,
        kit0=(task % KSPLIT) * KITS,
        KITS=KITS,
        SPLIT=gl.num_warps(),
        NW=1,
        GROUP=1,
        GROUP_STRIDE=0,
    )


@gluon.jit
def qnext_finish(mb, task, tiles, proj, OUT_PER_ROW: gl.constexpr, STAGE: gl.constexpr):
    """:func:`qnext_stage` after its weight loads (:func:`qnext_tiles`)."""
    R: gl.constexpr = schedule.rows(mb.plan)
    blk = task // KSPLIT
    half = task % KSPLIT
    # The out tasks of this K half, in every row: one flag each.
    F: gl.constexpr = OUT_PER_ROW // KSPLIT
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    n = gl.arange(0, 64 * gl.num_warps(), lanes)
    flag = mb.row_offset(gl.minimum(n // F, R - 1)) + half * F + n % F
    mb.poll_flags("anxf", flag)
    gl.barrier()
    mb.stamp(STAGE, trace.READY)
    x_smem = bf16.load_local_word_rows(
        mb, "anx", half * (HIDDEN // KSPLIT), HIDDEN // KSPLIT, R
    )
    out = bf16.gemv(x_smem, tiles, FIRST=0, ROWS=R)  # FP32 [RP, 16]
    mb.stamp(STAGE, trace.COMPUTED)
    ol: gl.constexpr = out.type.layout
    r = gl.arange(0, out.shape[0], gl.SliceLayout(1, ol))[:, None]
    i = gl.arange(0, 16, gl.SliceLayout(0, ol))[None, :]
    at = r * PROJ_WORDS + half * GATE_ROW + blk * 16 + i
    # (i >= 0) is always true: it broadcasts the row mask to the tile's shape.
    gl.store(proj + at, out.to(gl.int32, bitcast=True), mask=(r < R) & (i >= 0))
    mb.stamp(STAGE, trace.END)

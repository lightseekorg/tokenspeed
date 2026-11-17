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

"""Pre-expert projection (the pooler) of one Muse Spark rank.

pool, 32 tasks a group of up to 8 rows: one 16-row block of the rank's 512
pooler rows (BF16 ``[512, 8192]``, 256 KiB a task) times the whole BF16 MoE
input ``xin`` of the group's rows, rounded to BF16 like the regular path. A
task puts its 16 values of each row into every rank's ``cg``, so every rank
gets the whole pooled rows (4096 each, rank-major). Past 8 rows the two row
groups run on separate tasks, each streaming the same weights, rather than
one task running the groups one after the other: the rows of K sit in LDS
(16 KiB a row), so a task holds 8 rows at a time.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.monokernel import (
    bf16,
    mailbox,
    mx,
    schedule,
    trace,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.attn_tail import TAIL_TASKS
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.dims import (
    HIDDEN,
    POOLED,
    WORLD,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import Stage, placement

POOL_ROWS = gl.constexpr(POOLED // WORLD)  # pooled columns a rank computes
POOL_TASKS = gl.constexpr(32)

PEERS = ("cg",)


def regions() -> tuple[tuple[str, int], ...]:
    return (("cg", int(POOLED) // 2),)  # BF16 pairs of the pooled row


def stages(bases: dict[str, int], rows: int) -> tuple[Stage, ...]:
    """The pool stage for ``rows`` rows (``POOL_TASKS`` a group of up to 8
    rows), placed by ``bases``."""
    (pool,) = placement(bases, ("pool",))
    tasks = int(POOL_TASKS) * ((rows + 7) // 8)
    return (Stage("pool", tasks, pool, reads=("xin", "xflag"), writes=("cg",)),)


@gluon.jit
def pool_stage(mb, task, w, RANK: gl.constexpr, STAGE: gl.constexpr):
    """Pool task ``task``: pooled columns ``512 RANK + 16 (task % 32) ..`` of
    row group ``task // 32``.

    Args:
        w: This rank's pooler rows, BF16 ``[512, 8192]`` (``weights.bf16_tiles``).
    """
    R: gl.constexpr = schedule.rows(mb.plan)
    mb.stamp(STAGE, trace.START)
    if R > 8:
        # Past 8 rows: row group ``task // 32``, weight block ``task % 32``.
        group = task // POOL_TASKS
        task = task % POOL_TASKS
    tiles = bf16.load_weight_tiles(
        w,
        nb0=task,
        K=HIDDEN,
        kit0=0,
        KITS=HIDDEN // 32,
        SPLIT=gl.num_warps(),
        NW=1,
        GROUP=1,
        GROUP_STRIDE=0,
    )
    # One wave spins on the group's last row's flags first: the attention
    # tail publishes every row at about the same time.
    lanes = gl.arange(0, 64 * gl.num_warps(), mailbox.lane_layout(gl.num_warps()))
    if R > 8:
        mb.row(gl.minimum(8 * group + 7, R - 1)).pre_poll(
            "xflag", lanes % (WORLD * TAIL_TASKS)
        )
        if group == 0:
            _pool_pass(mb, tiles, 0, 8, task, RANK)
        else:
            _pool_pass(mb, tiles, 8, R - 8, task, RANK)
    else:
        mb.row(R - 1).pre_poll("xflag", lanes % (WORLD * TAIL_TASKS))
        _pool_pass(mb, tiles, 0, R, task, RANK)
    mb.stamp(STAGE, trace.COMPUTED)
    mb.stamp(STAGE, trace.END)


@gluon.jit
def _pool_pass(mb, tiles, p0: gl.constexpr, n: gl.constexpr, task, RANK: gl.constexpr):
    """Rows ``p0 .. p0 + n - 1`` of pool task ``task``: their whole rows of
    ``xin`` in LDS times the task's weight ``tiles``, BF16, into every
    rank's ``cg``."""
    x_smem = bf16.load_word_rows(
        mb.row(p0), "xin", "xflag", WORLD * TAIL_TASKS, HIDDEN, n
    )
    out = bf16.gemv(x_smem, tiles, FIRST=0, ROWS=n)  # FP32 [RP, 16]
    RP: gl.constexpr = out.shape[0]
    pairs: gl.constexpr = gl.BlockedLayout([1, 2], [1, 64], [gl.num_warps(), 1], [1, 0])
    values = gl.convert_layout(out.to(gl.bfloat16), pairs)
    word = gl.reshape(mx.pack_bf16_pairs(gl.reshape(values, [RP, 8, 2])), [RP, 8])
    wl: gl.constexpr = word.type.layout
    row = gl.arange(0, RP, gl.SliceLayout(1, wl))[:, None]
    p = gl.arange(0, 8, gl.SliceLayout(0, wl))[None, :]
    at = mb.row_offset(p0 + row) + RANK * (POOL_ROWS // 2) + task * 8 + p
    for r in gl.static_range(WORLD):
        # The padding rows past n are masked; (p >= 0) is always true and
        # broadcasts the row mask to the words' shape.
        mb.put_peer("cg", r, at, word, mask=(row < n) & (p >= 0))

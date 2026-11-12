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

"""Stand-ins for the absent ranks of a Muse Spark TP8 launch on two GPUs.

Two processes run two of the eight rank programs (the production TP8
kernels) and exchange their peer regions over the GPU links. Before each
launch the host writes, into every absent rank's segment of every peer
region of each present rank's mailbox, what that rank would have put:

* :func:`standin_fill` writes deterministic finite values keyed by (region,
  request, row of the request, source rank, data index), so two launches
  that hold the same requests in other rows (one launch of two requests,
  or a launch each) read the same stand-ins. Kernel-vs-kernel tests stay
  bit for bit with them.
* With ``zero``, it writes zeros instead: the true payloads of a zero-shard
  model, in which every weight shard of the absent ranks, and every column
  they own, is zero.

What two GPUs cannot check: the rank order of a sum over all eight ranks
(two non-zero terms add the same way in either order), and the values the
present ranks put into the absent ranks' segments (they land in a local
buffer that nothing reads).
"""

from __future__ import annotations

import zlib

import torch
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import moe_tail
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import launch_tag

WORLD = 8
# The two-GPU tests run each pair of rank programs; together the pairs run
# all eight. ut-tokenspeed-kernel-amd-2gpu selects them by their "pair" ids.
PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]
PAIR_IDS = [f"pair{a}{b}" for a, b in PAIRS]


def absent_columns(n: int, width: int, present) -> torch.Tensor:
    """Which of ``n`` columns, ``width`` a rank in rank order, ranks outside
    ``present`` own."""
    return ~torch.isin(torch.arange(n) // width, torch.tensor(present))


def _word_regions(plan) -> set[str]:
    """Peer regions that producers fill with plain words (two a pair) under
    a flag, rather than tagged pairs."""
    words = {"xin"}
    if moe_tail.ffn_words(plan.rows):
        words.add("ffn")
    return words


def _segments(plan, region: str):
    """``(start, n)`` of each rank-ordered block of ``region`` in data units
    a row: words for a word region, else pairs. Rank ``r`` owns data
    indices ``start + r n / 8 ..`` of every block."""
    size = dict(plan.regions)[region]
    blocks = moe_tail.rank_blocks(region, size)
    # xin's blocks count pairs of two words; ffn's count its tagged indices,
    # which its word form keeps as word indices.
    if region == "xin":
        return [(2 * start, 2 * n) for start, n in blocks]
    return list(blocks)


def _values(key: str, n: int) -> torch.Tensor:
    """``n`` int32 words for ``key``: two BF16 halves in [0.5, 2) each, so
    the word reads as a finite positive FP32 or as two such BF16 values."""
    g = torch.Generator().manual_seed(zlib.crc32(key.encode()))
    halves = 0x3F00 + torch.randint(0, 0x100, (n, 2), generator=g, dtype=torch.int32)
    return (halves[:, 1] << 16) | halves[:, 0]


def _absent_ranks(present) -> list[int]:
    return [r for r in range(WORLD) if r not in present]


def standin_fill(
    scratch: torch.Tensor,
    plan,
    present: tuple[int, int],
    rows: list[tuple[int, int]],
    step: int,
    layer: int,
    copy: int,
    zero: bool,
) -> None:
    """Write every absent rank's segment of every peer region of
    ``scratch`` (copy ``copy``) for the launch of ``layer`` at ``step``, as
    that rank would have put it.

    Args:
        scratch: The present rank's int64 mailbox (any device).
        plan: The launch's ``Plan``.
        present: The two rank programs that run.
        rows: ``(request, row of the request)`` of each of the plan's rows.
        step: The launch's step (the tag's step).
        layer: The launch's layer (the tag's layer).
        copy: The peer regions' copy the launch uses (a layer launch's
            ``layer & 1``; a stage test's ``step & 1``).
        zero: Zeros (a zero-shard model's true payloads) instead of keyed
            stand-in values.
    """
    assert len(rows) == plan.rows
    tag = launch_tag(step, layer)
    words = _word_regions(plan)
    idx, val = [], []
    flat32 = scratch.view(torch.int32)
    for region in plan.peers:
        if region in plan.external:
            continue  # the host fills a stage test's external inputs itself
        base = plan.offset(region) + copy * plan.peer_copy(region)
        for t, (request, row) in enumerate(rows):
            at = base + t * plan.row_pairs
            for start, n in _segments(plan, region):
                seg = n // WORLD
                for a in _absent_ranks(present):
                    lo = start + a * seg
                    if plan.is_flag(region):
                        pairs = torch.arange(at + lo, at + lo + seg)
                        idx.append(pairs)
                        val.append(torch.full((seg,), (tag << 32) | 1))
                        continue
                    key = f"{region}/{request}/{row}/{a}/{start}"
                    w = (
                        torch.zeros(seg, dtype=torch.int32)
                        if zero
                        else _values(key, seg)
                    )
                    if region in words:
                        # Plain words: no tag; word i of the row at 2 at + i.
                        first = 2 * at + lo
                        flat32[first : first + seg] = w.to(flat32.device)
                    else:
                        idx.append(torch.arange(at + lo, at + lo + seg))
                        val.append((tag << 32) | (w.to(torch.int64) & 0xFFFFFFFF))
    if idx:
        index = torch.cat(idx).to(scratch.device)
        scratch[index] = torch.cat(val).to(scratch.device)


def peer_bases(
    group_bases: torch.Tensor,
    present: tuple[int, int],
    me: int,
    scratch: torch.Tensor,
    ghost: torch.Tensor,
) -> torch.Tensor:
    """``int64 [8]`` heap bases by rank program: the present ranks' from the
    two-process Iris group (``group_bases``, in process order), every absent
    rank's at this process's ``ghost`` buffer (``scratch``'s size, on the
    same heap), so their puts land there and nothing reads them.

    Args:
        me: This process's index in ``present``.
    """
    bases = torch.empty(WORLD, dtype=torch.int64, device=scratch.device)
    ghost_base = int(group_bases[me]) + (ghost.data_ptr() - scratch.data_ptr())
    for r in range(WORLD):
        bases[r] = ghost_base
    for i, r in enumerate(present):
        bases[r] = int(group_bases[i])
    return bases

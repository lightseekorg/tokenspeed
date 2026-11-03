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

"""SM-driven copy of a fixed list of byte ranges, such as one layer's peer weights.

When every GPU pulls peer weights at once, copy-engine reads slow to about
420 GB/s per GPU and slow further with the rate of compute kernel launches.
About 32 CTAs of 512 threads issuing loads reach about 680 GB/s.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tokenspeed_kernel._triton import tl, triton

_ALIGN = 16
_WORD = 8
# 128 KiB per CTA iteration with 16 warps keeps enough loads in flight.
_BLOCK_WORDS = 16384
_NUM_WARPS = 16


@dataclass(frozen=True)
class FetchPlan:
    """Device tile table for ``launch_peer_fetch`` plus copy-engine leftovers.

    ``tiles`` is int64 ``[n, 3]``: destination and source offsets from
    ``base`` and the length, all in 8-byte words. ``leftovers`` are
    ``(destination, source, bytes)`` ranges that are not 16-byte co-aligned.
    """

    tiles: torch.Tensor
    leftovers: tuple[tuple[int, int, int], ...]


def build_fetch_plan(
    copies: list[tuple[int, int, int]],
    *,
    base: int,
    tile_bytes: int,
    device: torch.device,
) -> FetchPlan:
    """Split ``(destination, source, bytes)`` ranges into interleaved tiles.

    Tiles alternate between the ranges so concurrent CTAs read from all
    sources at once instead of one peer at a time.
    """
    if tile_bytes <= 0 or tile_bytes % _ALIGN or base % _ALIGN:
        raise ValueError("Tiles and the base address must be 16-byte aligned")
    per_copy: list[list[tuple[int, int, int]]] = []
    leftovers = []
    for dst, src, size in copies:
        if size <= 0:
            continue
        if (dst - src) % _ALIGN:
            leftovers.append((dst, src, size))
            continue
        head = min((-dst) % _ALIGN, size)
        body = (size - head) // _ALIGN * _ALIGN
        tail = size - head - body
        if head:
            leftovers.append((dst, src, head))
        if tail:
            leftovers.append((dst + head + body, src + head + body, tail))
        tiles = []
        for offset in range(0, body, tile_bytes):
            length = min(tile_bytes, body - offset)
            d, s = dst + head + offset, src + head + offset
            tiles.append(((d - base) // _WORD, (s - base) // _WORD, length // _WORD))
        per_copy.append(tiles)
    order = []
    for i in range(max((len(t) for t in per_copy), default=0)):
        order.extend(t[i] for t in per_copy if i < len(t))
    tiles = torch.tensor(order, dtype=torch.int64).reshape(-1, 3).to(device)
    return FetchPlan(tiles, tuple(leftovers))


@triton.jit(do_not_specialize=["num_tiles"])
def _peer_fetch_kernel(base, tiles, counter, num_tiles, BLOCK: tl.constexpr):
    # Tiles are claimed from a counter so CTAs that start late, behind other
    # kernels, take fewer tiles instead of delaying the whole copy.
    g = tl.atomic_add(counter, 1)
    while g < num_tiles:
        dst = tl.multiple_of(tl.load(tiles + g * 3), 2)
        src = tl.multiple_of(tl.load(tiles + g * 3 + 1), 2)
        words = tl.multiple_of(tl.load(tiles + g * 3 + 2), 2)
        for start in range(0, words, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < words
            value = tl.load(base + src + offs, mask=mask, cache_modifier=".cg")
            tl.store(base + dst + offs, value, mask=mask, cache_modifier=".cs")
        g = tl.atomic_add(counter, 1)


def launch_peer_fetch(
    plan: FetchPlan,
    *,
    base: torch.Tensor,
    counter: torch.Tensor,
    ctas: int,
) -> None:
    """Copy every tile of ``plan`` on the current stream with ``ctas`` CTAs.

    Args:
        plan: Tiles from ``build_fetch_plan`` with the same ``base`` address.
        base: int64 tensor whose data pointer is the plan's base address.
        counter: One int32 element, reset here; reuse only in stream order.
        ctas: Number of copying CTAs; fixed for the store lifetime.
    """
    num_tiles = plan.tiles.shape[0]
    if not num_tiles:
        return
    counter.zero_()
    _peer_fetch_kernel[(ctas,)](
        base, plan.tiles, counter, num_tiles, BLOCK=_BLOCK_WORDS, num_warps=_NUM_WARPS
    )

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

"""Sizing of the compact Host tiers: the L2 prefix tier and the snapshot pool.

Both are sized the same way, in LCM blocks of the rank's transfer layout: an
explicit size in gigabytes wins (``--kvstore-size``,
``--retraction-snapshot-host-gb``), else a ratio of this rank's Device KV
capacity (``--kvstore-ratio``, ``--retraction-snapshot-ratio``). The snapshot
pool adds two rules of its own. An explicit ratio ``0`` disables it, so every
capacity block aborts its victim. And when neither knob is given, the pool is
derived from what it has to hold: whole images when there is no Host L2 tier
to take the hash-complete pages (the Device KV once), or just the tails of
``max_retracted_requests`` images when there is one. The tail of an image,
per cache group, is one page for a group that publishes to L2 (its unaligned
last page; a state group's live block) and a request's worst-case pages for a
group that never publishes (a replayable sliding group) -- page counts the
scheduler's ``CapacityModel`` answers, folded to LCM blocks by the same
model, so no page arithmetic lives here. The slot-state arena
(``max_retracted_requests`` x blob bytes) is a separate allocation on top of
the pool.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


def gigabytes_to_lcm_blocks(gigabytes: float, *, host_lcm_block_bytes: int) -> int:
    """Whole LCM blocks in ``gigabytes`` (decimal, like the KVStore's sizes)."""
    return int(gigabytes * 1e9 // host_lcm_block_bytes)


@dataclass(frozen=True)
class RetractionPoolRequest:
    """The operator's pool knobs, as ``ServerArgs`` resolved them.

    Attributes:
        host_gb: ``--retraction-snapshot-host-gb``; 0 is not set.
        ratio: ``--retraction-snapshot-ratio``; None is not set, 0 is no pool.
        max_retracted_requests: Slot-state rows (explicit, or the rank's
            running requests); 0 only with no pool.
        tail_lcm_blocks_per_request: LCM blocks one image's tail needs beside
            Host L2 (``tail_lcm_blocks_per_request``), the derived size's
            unit when an L2 tier exists.
    """

    host_gb: float
    ratio: float | None
    max_retracted_requests: int
    tail_lcm_blocks_per_request: int

    @property
    def disabled(self) -> bool:
        """No pool was asked for: a capacity block aborts its victim."""
        return self.host_gb == 0 and self.ratio == 0


@dataclass(frozen=True)
class RetractionPoolSizing:
    """The resolved pool: its LCM blocks, request cap and how they were chosen.

    Attributes:
        lcm_blocks: Usable LCM blocks of the pool; 0 means no pool.
        max_retracted_requests: Slot-state rows; 0 exactly when there is no
            pool.
        source: Which rule sized the pool, for the startup log.
    """

    lcm_blocks: int
    max_retracted_requests: int
    source: str

    def describe(
        self, *, host_lcm_block_bytes: int, blob_bytes: int, l2_tier: bool
    ) -> str:
        """The one startup line stating what the engine does under pressure."""
        if self.lcm_blocks == 0:
            return (
                f"Retraction snapshot pool: none ({self.source}); a capacity block "
                "aborts its victim"
            )
        pool_gb = self.lcm_blocks * host_lcm_block_bytes / 1e9
        arena_mb = self.max_retracted_requests * blob_bytes / 1e6
        images = (
            "hash-complete pages go to Host L2, tails to the pool"
            if l2_tier
            else "no Host L2 tier, whole images go to the pool"
        )
        return (
            f"Retraction snapshot pool: {pool_gb:.2f} GB ({self.lcm_blocks} LCM "
            f"blocks of {host_lcm_block_bytes} bytes; {self.source}), up to "
            f"{self.max_retracted_requests} retracted requests ({arena_mb:.2f} MB "
            f"slot-state arena); {images}"
        )


NO_POOL = RetractionPoolSizing(0, 0, "--retraction-snapshot-ratio 0")


def tail_lcm_blocks_per_request(model, specs: Sequence, token_limit: int) -> int:
    """LCM blocks one request's image tail needs beside Host L2.

    Per group: one page when the group publishes to L2 (its last, unaligned
    page -- a state group's live block), else the group's single-request
    worst-case pages at ``token_limit`` (a replayable group is never
    published, so every page of it is tail). Both counts and the fold to LCM
    blocks come from the scheduler's ``CapacityModel``.

    Args:
        model: The ``CapacityModel`` over the engine's cache groups.
        specs: The groups' ``CacheGroupSpec``s in the model's group order.
        token_limit: The longest request the engine admits.
    """
    worst_case = model.single_request_group_pages(token_limit)
    if len(worst_case) != len(specs):
        raise ValueError(
            f"the capacity model answers {len(worst_case)} groups for "
            f"{len(specs)} specs"
        )
    tail_pages = [
        pages if spec.replayable else 1 for spec, pages in zip(specs, worst_case)
    ]
    return int(model.lcm_blocks_needed_for(tail_pages))


def resolve_retraction_pool(
    request: RetractionPoolRequest,
    *,
    l2_tier: bool,
    device_lcm_blocks: int,
    host_lcm_block_bytes: int,
) -> RetractionPoolSizing:
    """Size the pool from the knobs and the rank's page geometry.

    Args:
        request: The knobs ``ServerArgs`` resolved, plus the tail unit.
        l2_tier: Whether the Host L2 tier exists (``--enable-kvstore``).
        device_lcm_blocks: This rank's usable Device LCM blocks, the base
            ``--kvstore-ratio`` scales too.
        host_lcm_block_bytes: Bytes of one Host LCM block.

    Raises:
        ValueError: A pool that would hold no whole block, or one without
            slot-state rows.
    """
    if request.disabled:
        return NO_POOL
    if request.host_gb > 0:
        blocks = gigabytes_to_lcm_blocks(
            request.host_gb, host_lcm_block_bytes=host_lcm_block_bytes
        )
        source = f"--retraction-snapshot-host-gb {request.host_gb}"
    elif request.ratio is not None:
        blocks = int(device_lcm_blocks * request.ratio)
        source = (
            f"--retraction-snapshot-ratio {request.ratio} of {device_lcm_blocks} "
            "Device LCM blocks"
        )
    elif not l2_tier:
        blocks = device_lcm_blocks
        source = "derived: the Device KV once, no Host L2 tier"
    else:
        blocks = request.max_retracted_requests * request.tail_lcm_blocks_per_request
        source = (
            f"derived: {request.max_retracted_requests} image tails x "
            f"{request.tail_lcm_blocks_per_request} LCM blocks beside Host L2"
        )
    if blocks <= 0:
        raise ValueError(
            f"the retraction snapshot pool ({source}) holds no whole LCM block of "
            f"{host_lcm_block_bytes} bytes; raise --retraction-snapshot-host-gb or "
            "--retraction-snapshot-ratio, or pass --retraction-snapshot-ratio 0 to "
            "run without a pool"
        )
    if request.max_retracted_requests <= 0:
        raise ValueError(
            "a retraction snapshot pool needs slot-state rows: "
            "--retraction-snapshot-max-requests, or --max-num-seqs to derive them"
        )
    return RetractionPoolSizing(blocks, request.max_retracted_requests, source)

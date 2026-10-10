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

"""The Host cache's per-request wire operations, runtime side.

A retraction image is split by publishability. Its bulk -- every block under
a hash-complete prefix page -- goes to Host L2 as today's stream-ordered
``WriteBackOp`` and is pinned there by the retracted request; its tail -- the
unaligned tail pages, every block of a group that is never published, the
slot-state blob -- goes to the small request-private snapshot pool through a
``SnapshotOp``. The way back is one ``RestoreOp`` whose rows name their
source tier, so both Host buffers land in the request's fresh Device pages
under one completion event and the scheduler sees one ``RestoreDone``.

An L3 prefetch (``PrefetchOp``) fills a waiting request's freshly allocated
Host pages from the L3 store before its admission, in prefix order, so the
admission that follows sees a plain Host hit; its ACK carries the pages that
landed (a prefix length, MIN-reduced across the replica).

The C++ scheduler emits these batched per plan (bound as ``Cache.SnapshotOp``
/ ``Cache.RestoreOp`` / ``Cache.PrefetchOp``, lists-of-lists like the L2
batches); ``engine/scheduler_utils.cache_ops_from_plan`` maps every row of a
batch onto one of these per-request dataclasses, which is the Host cache
executor's unit of work: one slot exported or imported, one prefetch job, one
ACK. The ACKs are the binding's ``Cache.SnapshotDoneEvent`` /
``Cache.RestoreDoneEvent`` / ``Cache.PrefetchDoneEvent``; they join
``WriteBackDoneEvent`` / ``LoadBackDoneEvent`` on the cache-result poll and are
replica-intersected by the same hooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class HostTier(IntEnum):
    """Which pinned Host buffer a restore row reads (``Cache.HostTier`` on the wire)."""

    L2 = 0
    SNAPSHOT_POOL = 1


@dataclass(frozen=True, slots=True)
class CacheTransfer:
    """One CacheBlock copy of a snapshot op (the wire's ``CacheTransfer``).

    Attributes:
        group_id: Cache group index, in the scheduler's group order (the
            transfer layout's group order).
        source_page: Scheduler (virtual) block id of the copy's source: a
            Device block for a store, a Host block for a restore.
        destination_page: Scheduler (virtual) block id of the destination:
            a snapshot-pool block for a store, a Device block for a restore.
        content_hash: The prefix key of an L2-tier restore row, so the
            scheduler can publish the restored Device block; empty on every
            other row.
        page_offset: The key's page offset; 0 where the key is empty.
    """

    group_id: int
    source_page: int
    destination_page: int
    content_hash: str
    page_offset: int


@dataclass(frozen=True, slots=True)
class SnapshotOp:
    """Store a retracted request's tail pages and slot state to the snapshot pool.

    Attributes:
        op_id: The scheduler's ticket; the ACK carries it back.
        request_id: The victim.
        request_pool_index: The victim's slot, whose state is exported. The
            FSM frees the slot in the same plan build, and a new owner first
            writes it in a forward behind the store's fence, so the export
            reads the victim's bytes.
        snapshot_slot: The row of the slot-state arena the image lands in.
        transfers: Device block to snapshot-pool block, for every block Host
            L2 does not take; empty when every page went to L2 and the image
            is the slot state alone.
    """

    op_id: int
    request_id: str
    request_pool_index: int
    snapshot_slot: int
    transfers: tuple[CacheTransfer, ...]


@dataclass(frozen=True, slots=True)
class RestoreOp:
    """Copy a retracted request's image back into fresh Device pages.

    Attributes:
        op_id: The scheduler's ticket; the ACK carries it back.
        request_id: The request resuming.
        request_pool_index: The request's NEW slot, whose state is imported.
        snapshot_slot: The arena row the store imaged into.
        transfers: Host block to Device block, both tiers.
        source_tier: ``transfers[i]`` reads ``source_tier[i]``'s buffer.
    """

    op_id: int
    request_id: str
    request_pool_index: int
    snapshot_slot: int
    transfers: tuple[CacheTransfer, ...]
    source_tier: tuple[HostTier, ...]


@dataclass(frozen=True, slots=True)
class PrefetchRow:
    """One Host page of a prefetch op: a prefix page's block of one group.

    Attributes:
        group_id: Cache group index, in the scheduler's group order.
        host_page: Scheduler (virtual) block id of the Host L2 block the
            scheduler allocated for the page.
        content_hash: The prefix page's key; the L3 object name derives from
            it.
        page_offset: The key's page offset within the prefix.
        page_index: The prefix page the row belongs to; rows come in
            non-decreasing page order.
    """

    group_id: int
    host_page: int
    content_hash: str
    page_offset: int
    page_index: int


@dataclass(frozen=True, slots=True)
class PrefetchOp:
    """Fill a waiting request's Host pages from L3 before its admission.

    Attributes:
        op_id: The scheduler's ticket; the ACK carries it back with the
            pages landed.
        request_id: The request waiting in ``Prefetching``.
        first_page: The prefix page the fill starts at (the Host hit's end).
        num_pages: The prefix pages the rows cover; ``landed_pages`` of the
            ACK is the largest ``n <= num_pages`` whose rows all landed.
        rows: The pages to fetch in prefix-page order (every group's rows of
            a page before the next page's); a page may have no rows at all
            (a sliding group's older pages), which lands trivially.
    """

    op_id: int
    request_id: str
    first_page: int
    num_pages: int
    rows: tuple[PrefetchRow, ...]

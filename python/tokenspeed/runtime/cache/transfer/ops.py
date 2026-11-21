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

The C++ scheduler emits these batched per plan (bound as ``Cache.SnapshotOp``
/ ``Cache.RestoreOp``, lists-of-lists like the L2
batches); ``engine/scheduler_utils.cache_ops_from_plan`` maps every row of a
batch onto one of these per-request dataclasses, which is the Host cache
executor's unit of work: one slot exported or imported, one
ACK. The ACKs are the binding's ``Cache.SnapshotDoneEvent`` /
``Cache.RestoreDoneEvent``; they join
``WriteBackDoneEvent`` / ``LoadBackDoneEvent`` on the cache-result poll and are
replica-intersected by the same hooks.

Host-only ``Cache.StorageQueryOp`` / ``StorageGetOp`` / ``StoragePutOp`` /
``StorageCancelOp`` are the engine host's L3 executor's (``ts-l3``); the
adapter sets them aside. They use ``StorageDone`` retirement facts,
independently of GPU copy completion.
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
        content_hash: The prefix key of an L2-tier restore row the scheduler
            publishes the restored Device block under; empty on every other
            row, including an L2 row that restores a private copy of a
            published tail block.
        page_offset: The key's page offset; 0 where the key is empty.
    """

    group_id: int
    source_page: int
    destination_page: int
    content_hash: str
    page_offset: int


@dataclass(frozen=True, slots=True)
class PrefixLoadTelemetry:
    """Immutable admission and operation identity supplied by the scheduler.

    Attributes:
        admission_id: Identity of the request admission, retained across retraction.
        op_generation: Wrap generation of the scheduler operation ticket.
        request_pool_index: Request slot captured when the prefix load was scheduled.
        slot_generation: Allocation generation of that request-pool slot.
    """

    admission_id: int
    op_generation: int
    request_pool_index: int
    slot_generation: int


@dataclass(frozen=True, slots=True)
class RestoreTelemetry:
    """Scheduler-owned restore frontier and slot ownership at scheduling.

    Attributes:
        admission_id: Identity of the same request admission held before retraction.
        op_generation: Wrap generation of the scheduler operation ticket.
        previous_slot: Request slot whose state the retraction image captured.
        previous_slot_generation: Allocation generation of previous_slot at retraction.
        slot_generation: Allocation generation of the restore's new destination slot.
        slot_previous_admission_id: Admission last bound to the destination slot,
            or 0 when no prior admission is recorded.
        restore_sequence: One-based restore scheduling sequence for this admission.
        committed_tokens: Complete prompt plus accepted output tokens retained at
            retraction, independent of client delivery.
        computed_tokens: Computed prefix length at retraction: the prefill
            window's end, or ``committed_tokens - 1`` while decoding.
        prompt_tokens: Full submitted prompt length, including any uncomputed suffix.
    """

    admission_id: int
    op_generation: int
    previous_slot: int
    previous_slot_generation: int
    slot_generation: int
    slot_previous_admission_id: int
    restore_sequence: int
    committed_tokens: int
    computed_tokens: int
    prompt_tokens: int


@dataclass(frozen=True, slots=True)
class PrefixLoadOp:
    """One identified Host prefix load, with the original wire transfer rows.

    Attributes:
        op_id: The scheduler's ticket; the ACK carries it back.
        request_id: The request loading its prefix.
        telemetry: Frozen scheduler admission, operation and slot identity.
        transfers: Host L2 block to Device block rows, before local-owner translation.
    """

    op_id: int
    request_id: str
    telemetry: PrefixLoadTelemetry
    transfers: tuple[CacheTransfer, ...]


@dataclass(frozen=True, slots=True)
class SnapshotOp:
    """Store a retracted request's tail pages and slot state to the snapshot pool.

    Attributes:
        op_id: The scheduler's ticket; the ACK carries it back.
        request_id: The victim.
        admission_id: The immutable identity of this request admission.
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
    admission_id: int
    request_pool_index: int
    snapshot_slot: int
    transfers: tuple[CacheTransfer, ...]


@dataclass(frozen=True, slots=True)
class RestoreOp:
    """Copy a retracted request's image back into fresh Device pages.

    Attributes:
        op_id: The scheduler's ticket; the ACK carries it back.
        request_id: The request resuming.
        admission_id: The same identity the request had before retraction.
        request_pool_index: The request's NEW slot, whose state is imported.
        snapshot_slot: The arena row the store imaged into.
        transfers: Host block to Device block, both tiers.
        source_tier: ``transfers[i]`` reads ``source_tier[i]``'s buffer.
        telemetry: Frozen scheduler restore context; None means this operation
            carries no opt-in scheduler observation.
    """

    op_id: int
    request_id: str
    admission_id: int
    request_pool_index: int
    snapshot_slot: int
    transfers: tuple[CacheTransfer, ...]
    source_tier: tuple[HostTier, ...]
    telemetry: RestoreTelemetry | None = None

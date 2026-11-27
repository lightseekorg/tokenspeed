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

"""Bounded CPU observations attached to the same lifetime as cache-copy ACKs.

These records describe completed local physical copies, not request success or
replica-wide completion. Export, request cancellation, and admission ownership
remain control-plane responsibilities.
"""

from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, NamedTuple

from tokenspeed.runtime.cache.transfer.ops import PrefixLoadTelemetry, RestoreTelemetry

if TYPE_CHECKING:
    from tokenspeed.runtime.cache.transfer.layout import CacheGroupLayout

MAX_REQUEST_ID_BYTES = 256


def cache_observation_capacity_from_env(environ: Mapping[str, str]) -> int:
    """Resolve the public startup opt-in; it is never reread while serving."""
    value = environ.get("TOKENSPEED_CACHE_EVENT_LOG", "0")
    if value not in ("0", "1"):
        raise ValueError("TOKENSPEED_CACHE_EVENT_LOG must be 0 or 1")
    return 4096 if value == "1" else 0


@dataclass(frozen=True, slots=True)
class CopyCounts:
    """Local copy totals for one Host tier.

    Attributes:
        rows: Number of CacheBlock transfer rows after ownership translation.
        bytes: Sum of all field payload bytes for those rows, excluding
            block-stride padding.
    """

    rows: int
    bytes: int


def count_owned_rows(
    rows: Iterable[tuple[int, int, int]], groups: Sequence["CacheGroupLayout"]
) -> CopyCounts:
    """Count local CacheBlocks and the payload bytes that their fields copy."""
    count = 0
    nbytes = 0
    for group, _device, _host in rows:
        if not 0 <= group < len(groups):
            raise ValueError(f"Unknown cache group {group}")
        count += 1
        nbytes += sum(field.payload_bytes for field in groups[group].fields)
    return CopyCounts(count, nbytes)


@dataclass(frozen=True, slots=True)
class CopyBatchTiming:
    """One local copy-stream sample shared by every operation in the batch.

    Group counts identify the State/history mix through the run's cache layout.
    An unsupported device reports a missing duration, never CPU ACK latency.
    """

    batch_id: int
    backend: str
    group_ids: tuple[str, ...]
    rows_by_group: tuple[int, ...]
    bytes: int
    elapsed_ms: float | None
    status: str


class PendingCopyTiming:
    """Timing events owned by the normal copy ACK, without an extra wait."""

    def __init__(
        self, *, batch_id, backend, groups, rows, device_kind, device_module, stream
    ):
        counts = [0] * len(groups)
        for group, _device, _host in rows:
            counts[group] += 1
        self.batch_id = batch_id
        self.backend = backend
        self.group_ids = tuple(group.group_id for group in groups)
        self.rows_by_group = tuple(counts)
        self.bytes = count_owned_rows(rows, groups).bytes
        supported = device_kind in ("cuda", "npu")
        self.start = device_module.Event(enable_timing=True) if supported else None
        self.finish = device_module.Event(enable_timing=True) if supported else None
        if self.start is not None:
            self.start.record(stream)

    def complete(self, batch: "ObservationBatch") -> "ObservationBatch":
        """Read a retired sample once; the ACK queue has already queried finish."""
        timing = CopyBatchTiming(
            self.batch_id,
            self.backend,
            self.group_ids,
            self.rows_by_group,
            self.bytes,
            self.start.elapsed_time(self.finish) if self.start is not None else None,
            "measured" if self.start is not None else "unsupported",
        )
        return ObservationBatch(
            tuple(
                replace(
                    record,
                    copy_batch_id=self.batch_id,
                    h2d_batch_timing=timing if i == 0 else None,
                )
                for i, record in enumerate(batch.records)
            ),
            batch.dropped + (not batch.records),
        )


@dataclass(frozen=True, slots=True)
class CopyObservation:
    """Immutable local physical-copy metadata carried by one operation's ACK.

    Attributes:
        kind: Copy kind, ``prefix_load`` or ``snapshot_restore``.
        op_id: Scheduler operation ticket, joined with kind and
            telemetry.op_generation.
        request_id: External request ID captured for this operation; admission
            identity distinguishes later reuse of the same ID.
        request_pool_index: Request slot captured when the operation was scheduled.
        telemetry: Frozen scheduler identity and, for restores, recovery context.
        l2: Local Host L2 transfer-row and field-payload byte counts.
        pool: Local snapshot-pool transfer-row and field-payload byte counts.
        slot_state_image_bytes: Declared slot-state blob span, not a copied-byte
            count. An unprepared exporter may import only its marker and no
            state rows.
    """

    kind: str
    op_id: int
    request_id: str
    request_pool_index: int
    telemetry: PrefixLoadTelemetry | RestoreTelemetry
    l2: CopyCounts
    pool: CopyCounts
    slot_state_image_bytes: int
    copy_batch_id: int | None = None
    h2d_batch_timing: CopyBatchTiming | None = None

    def payload(self) -> dict[str, object]:
        """Return scalar CPU metadata for the host's bounded export queue."""
        return {
            "kind": self.kind,
            "op_id": self.op_id,
            "request_id": self.request_id,
            "request_pool_index": self.request_pool_index,
            **asdict(self.telemetry),
            "l2_rows": self.l2.rows,
            "l2_bytes": self.l2.bytes,
            "pool_rows": self.pool.rows,
            "pool_bytes": self.pool.bytes,
            "slot_state_image_bytes": self.slot_state_image_bytes,
            "copy_batch_id": self.copy_batch_id,
            "h2d_batch_timing": (
                asdict(self.h2d_batch_timing)
                if self.h2d_batch_timing is not None
                else None
            ),
        }


class ObservationBatch(NamedTuple):
    """Frozen per-ACK records and loss count; neither retains device storage.

    Attributes:
        records: Local copy observations captured for the completed operation.
        dropped: Observations refused while capturing this batch.
    """

    records: tuple[CopyObservation, ...] = ()
    dropped: int = 0


def observe_copy(
    *,
    kind: str,
    op_id: int,
    request_id: str,
    request_pool_index: int,
    telemetry: PrefixLoadTelemetry | RestoreTelemetry | None,
    l2: CopyCounts,
    pool: CopyCounts,
    slot_state_image_bytes: int,
) -> ObservationBatch:
    """Capture a bounded identity, refusing to truncate it into another request."""
    if telemetry is None or (
        len(request_id) > MAX_REQUEST_ID_BYTES
        or len(request_id.encode("utf-8")) > MAX_REQUEST_ID_BYTES
    ):
        return ObservationBatch(dropped=1)
    return ObservationBatch(
        (
            CopyObservation(
                kind=kind,
                op_id=op_id,
                request_id=request_id,
                request_pool_index=request_pool_index,
                telemetry=telemetry,
                l2=l2,
                pool=pool,
                slot_state_image_bytes=slot_state_image_bytes,
            ),
        )
    )


class CompletedObservations:
    """Completed CPU records only; the caller holds the executor's ACK lock.

    Zero capacity disables diagnostics. Overflow discards new observations and
    reports their loss without affecting the scheduler ACKs or copying work.

    Attributes:
        capacity: Maximum number of retained completed records; zero disables
            recording.
        enabled: Whether the configured capacity permits recording observations.
    """

    def __init__(self, capacity: int) -> None:
        if type(capacity) is not int or capacity < 0:
            raise ValueError("Observation capacity must be a nonnegative integer")
        self.capacity: int = capacity
        self._records: deque[CopyObservation] = deque()
        self._dropped: int = 0

    @property
    def enabled(self) -> bool:
        """Whether diagnostics retain completed observations."""
        return self.capacity > 0

    def extend(self, batch: ObservationBatch) -> None:
        """Append within capacity and accumulate batch and overflow losses.

        Keep queued records and drop newer ones beyond the remaining capacity,
        adding both overflow and ``batch.dropped`` to the dropped count.
        A disabled queue ignores the batch.
        """
        if not self.enabled:
            return
        self._dropped += batch.dropped
        for record in batch.records:
            if len(self._records) == self.capacity:
                self._dropped += 1
            else:
                self._records.append(record)

    def drain(self) -> tuple[list[dict[str, object]], int]:
        """Return ``(payloads, dropped)`` and clear the records and loss counter.

        The payloads are the completed records as CPU dictionaries.
        """
        records = [record.payload() for record in self._records]
        dropped = self._dropped
        self.clear()
        return records, dropped

    def clear(self) -> None:
        """Discard retained observations and reset the recorded loss count."""
        self._records.clear()
        self._dropped = 0

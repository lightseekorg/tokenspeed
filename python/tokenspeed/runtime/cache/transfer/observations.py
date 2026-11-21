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
replica-wide completion. Export, request cancellation and admission ownership
remain control-plane responsibilities.
"""

from collections import deque
from dataclasses import asdict, dataclass
from typing import NamedTuple

from tokenspeed.runtime.cache.transfer.ops import PrefixLoadTelemetry, RestoreTelemetry

MAX_REQUEST_ID_BYTES = 256


def cache_observation_capacity_from_env(environ):
    """Resolve the public startup opt-in; it is never reread while serving."""
    value = environ.get("TOKENSPEED_CACHE_EVENT_LOG", "0")
    if value not in ("0", "1"):
        raise ValueError("TOKENSPEED_CACHE_EVENT_LOG must be 0 or 1")
    return 4096 if value == "1" else 0


@dataclass(frozen=True, slots=True)
class CopyCounts:
    rows: int
    bytes: int


def count_owned_rows(rows, groups):
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
class CopyObservation:
    kind: str
    op_id: int
    request_id: str
    request_pool_index: int
    telemetry: PrefixLoadTelemetry | RestoreTelemetry
    l2: CopyCounts
    pool: CopyCounts
    # The declared blob span is not a count of bytes copied by every exporter:
    # an unprepared owner may import only its marker and no state rows.
    slot_state_image_bytes: int

    def payload(self):
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
        }


class ObservationBatch(NamedTuple):
    """Frozen per-ACK records and loss count; neither retains device storage."""

    records: tuple[CopyObservation, ...] = ()
    dropped: int = 0


def observe_copy(
    *,
    kind,
    op_id,
    request_id,
    request_pool_index,
    telemetry,
    l2,
    pool,
    slot_state_image_bytes,
):
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
    """

    def __init__(self, capacity):
        if type(capacity) is not int or capacity < 0:
            raise ValueError("Observation capacity must be a nonnegative integer")
        self.capacity = capacity
        self._records = deque()
        self._dropped = 0

    @property
    def enabled(self):
        return self.capacity > 0

    def extend(self, batch):
        if not self.enabled:
            return
        self._dropped += batch.dropped
        for record in batch.records:
            if len(self._records) == self.capacity:
                self._dropped += 1
            else:
                self._records.append(record)

    def drain(self):
        records = [record.payload() for record in self._records]
        dropped = self._dropped
        self.clear()
        return records, dropped

    def clear(self):
        self._records.clear()
        self._dropped = 0

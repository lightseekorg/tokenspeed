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

"""Storage scheduling limits and Python worker settings, parsed once at startup."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tokenspeed_scheduler import StorageConfig


@dataclass(frozen=True, slots=True)
class StorageRuntimeConfig:
    """Own worker settings and the shared, read-only scheduler contract.

    Scheduling limits live only in ``scheduler``. The executor and replica
    hooks read that same contract instead of keeping a Python mirror.
    Backend connection settings remain inside the selected storage backend.
    """

    scheduler: StorageConfig
    query_workers: int
    read_workers: int
    write_workers: int
    failure_threshold: int
    recovery_interval_ms: int

    @classmethod
    def from_extra_config(
        cls, raw: str | None, *, host_blocks: int, prefix_granularity: int
    ) -> StorageRuntimeConfig:
        from tokenspeed_scheduler import StorageConfig, StoragePressure

        knobs = parse_extra_config(raw).get("async", {})
        if not isinstance(knobs, dict):
            raise ValueError("L3 async config must be an object")
        # Tuning values within the selected asynchronous backend, not switches
        # between schedulers or memory-safety implementations.
        values = dict(
            max_operations=32,
            max_keys_per_operation=256,
            max_host_blocks=max(1, min(256, host_blocks // 2)),
            metadata_entries=max(1, host_blocks * 4),
            max_prefetch_tokens=8192,
            min_prefetch_tokens=prefix_granularity,
            wait_timeout_ms=2000,
            ready_timeout_ms=1000,
            query_workers=2,
            read_workers=2,
            write_workers=1,
            failure_threshold=1,
            recovery_interval_ms=30000,
            queue_high_operations=8,
            queue_low_operations=4,
            queue_high_bytes=256 * 1024 * 1024,
            queue_low_bytes=128 * 1024 * 1024,
            operation_high_age_ms=500,
            operation_low_age_ms=250,
        )
        unknown = knobs.keys() - values.keys()
        if unknown:
            raise ValueError(f"Unsupported L3 async options: {sorted(unknown)}")
        values.update(knobs)
        for name, value in values.items():
            minimum = (
                0
                if name
                in {"queue_low_operations", "queue_low_bytes", "operation_low_age_ms"}
                else 1
            )
            if type(value) is not int or value < minimum or value > (2**63 - 1) // 1000:
                raise ValueError(
                    f"L3 {name} must be an integer in [{minimum}, {(2**63 - 1) // 1000}]"
                )
        if "queue_high_operations" not in knobs:
            values["queue_high_operations"] = min(8, values["max_operations"])
        if "queue_low_operations" not in knobs:
            values["queue_low_operations"] = values["queue_high_operations"] // 2
        for high, low in (
            ("queue_high_operations", "queue_low_operations"),
            ("queue_high_bytes", "queue_low_bytes"),
            ("operation_high_age_ms", "operation_low_age_ms"),
        ):
            if values[low] >= values[high]:
                raise ValueError(f"L3 {low} must be smaller than {high}")
        # Match the scheduler's int32 contract before crossing the binding,
        # rather than exposing a nanobind conversion error at startup.
        for name in (
            "max_operations",
            "max_keys_per_operation",
            "max_host_blocks",
            "metadata_entries",
            "max_prefetch_tokens",
            "min_prefetch_tokens",
        ):
            if values[name] > 2**31 - 1:
                raise ValueError(f"L3 {name} exceeds the scheduler's int32 limit")
        if values["max_host_blocks"] > host_blocks:
            raise ValueError("L3 max_host_blocks exceeds usable Host parents")
        if values["min_prefetch_tokens"] > values["max_prefetch_tokens"]:
            raise ValueError("L3 minimum prefetch must fit the maximum")
        return cls(
            scheduler=StorageConfig(
                values["max_operations"],
                values["max_keys_per_operation"],
                values["max_host_blocks"],
                values["metadata_entries"],
                values["max_prefetch_tokens"],
                values["min_prefetch_tokens"],
                values["wait_timeout_ms"] * 1000,
                values["ready_timeout_ms"] * 1000,
                StoragePressure(
                    values["queue_high_operations"],
                    values["queue_high_bytes"],
                    values["operation_high_age_ms"] * 1000,
                ),
                StoragePressure(
                    values["queue_low_operations"],
                    values["queue_low_bytes"],
                    values["operation_low_age_ms"] * 1000,
                ),
            ),
            query_workers=values["query_workers"],
            read_workers=values["read_workers"],
            write_workers=values["write_workers"],
            failure_threshold=values["failure_threshold"],
            recovery_interval_ms=values["recovery_interval_ms"],
        )


def parse_extra_config(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "--kvstore-storage-backend-extra-config must be a JSON object"
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError("--kvstore-storage-backend-extra-config must be a JSON object")
    return parsed

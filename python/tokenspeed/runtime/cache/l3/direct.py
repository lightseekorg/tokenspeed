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

"""Startup-only settings for bounded, in-process L3 restoration; no Device allocations."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from tokenspeed.runtime.cache.l3.config import L3Settings


@dataclass(frozen=True, slots=True)
class DirectSourceSettings:
    """The existing L3 identity/layout plus bounded broker settings."""

    config: str
    l3: L3Settings
    prefix_granularity: int

    @classmethod
    def resolve(
        cls,
        raw: str | None,
        l3: L3Settings,
        *,
        prefix_granularity: int,
        max_model_len: int,
    ) -> DirectSourceSettings:
        values = json.loads(raw) if raw is not None else {}
        if not isinstance(values, dict):
            raise ValueError("Direct L3 settings must be a JSON object")
        if (
            l3.backend != "mooncake"
            or l3.mooncake is None
            or l3.runtime.scheduler.restore_to_host
        ):
            raise ValueError(
                "Direct L3 requires Mooncake PUTs without Host restore jobs"
            )
        if prefix_granularity <= 0 or max_model_len <= 1:
            raise ValueError(
                "Direct L3 needs a positive prefix grain and reusable model context"
            )
        store = l3.mooncake
        defaults = {
            "library": os.environ.get("TOKENSPEED_MOONCAKE_OFFLOAD_LIBRARY")
            or str(Path(store.library).with_name("libtokenspeed_mooncake_offload.so")),
            "master": store.master_server_address,
            "tenant": store.tenant_id,
            "rpc_timeout_ms": 2000,
            "query_timeout_ms": 500,
            "record_lifetime_ms": 15000,
            "read_budget_ms": 5000,
            "max_query_keys": 1024,
            "max_candidates": 8,
            "max_prepare_bytes": 256 * 1024**2,
            "max_source_batch_bytes": 128 * 1024**2,
            "max_source_batches": 8,
            "source_budget_bytes": 256 * 1024**2,
            "max_records": 4,
            "min_fetch_tokens": 4096,
            "max_restore_tokens": min(max_model_len - 1, 65536),
        }
        if set(values) - defaults.keys():
            raise ValueError("Unknown direct L3 override")
        values = {**defaults, **values}
        if (
            values["master"] != store.master_server_address
            or values["tenant"] != store.tenant_id
        ):
            raise ValueError("Direct L3 and Store must share master and tenant")
        strings = {"library", "master", "tenant"}
        numbers = {
            "query_timeout_ms",
            "min_fetch_tokens",
            "rpc_timeout_ms",
            "max_query_keys",
            "max_candidates",
            "max_prepare_bytes",
            "max_source_batch_bytes",
            "max_source_batches",
            "source_budget_bytes",
            "max_records",
            "record_lifetime_ms",
            "read_budget_ms",
            "max_restore_tokens",
        }
        if set(values) != strings | numbers:
            raise ValueError(
                "Direct L3 requires exactly the documented broker settings"
            )
        for name in strings:
            if (
                not isinstance(values[name], str)
                or not values[name]
                or "\0" in values[name]
            ):
                raise ValueError(f"Direct L3 {name} must be a nonempty string")
        for name in numbers:
            if type(values[name]) is not int or not 0 < values[name] <= 2**63 - 1:
                raise ValueError(f"Direct L3 {name} must be a positive int64")
        if values["max_query_keys"] > 1024 or values["rpc_timeout_ms"] > 10000:
            raise ValueError("Direct L3 exceeds native RPC key/timeout limits")
        if values["max_source_batch_bytes"] > 2**31:
            raise ValueError("Direct L3 prepare batches cannot exceed 2 GiB")
        if values["max_prepare_bytes"] > values["source_budget_bytes"]:
            raise ValueError("Direct L3 per-request bytes exceed its worker budget")
        if values["read_budget_ms"] >= values["record_lifetime_ms"]:
            raise ValueError("Direct L3 read budget must fit inside record lifetime")
        if values["max_candidates"] > 1024 or values["max_records"] > 1024:
            raise ValueError(
                "Direct L3 candidate/record counts exceed bounded metadata capacity"
            )
        if values["record_lifetime_ms"] > 3_600_000 or values[
            "max_restore_tokens"
        ] > min(max_model_len - 1, 65536):
            raise ValueError(
                "Direct L3 lifetime/token bound exceeds the supported range"
            )
        if values["query_timeout_ms"] > values["record_lifetime_ms"]:
            raise ValueError("Direct L3 query deadline exceeds metadata lifetime")
        return cls(json.dumps(values), l3, prefix_granularity)

    @property
    def scheduler_limits(self) -> dict[str, int]:
        values = json.loads(self.config)
        return {
            "storage_min_fetch_tokens": values["min_fetch_tokens"],
            "storage_max_tokens": values["max_restore_tokens"],
            "storage_lookup_timeout_us": values["query_timeout_ms"] * 1000,
        }

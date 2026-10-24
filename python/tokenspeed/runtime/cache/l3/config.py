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

"""The L3 tier's settings, resolved once at startup: the scheduler's storage
limits, the executor's workers, the store client and what the engine host
starts its executor from (``L3Settings``)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tokenspeed_scheduler import StorageConfig


@dataclass(frozen=True, slots=True)
class StorageRuntimeConfig:
    """Own worker settings and the shared, read-only scheduler contract.

    Scheduling limits live only in ``scheduler``; the engine host's executor
    and replica agreement read that same contract. Store connection settings
    are the store's (``MooncakeStoreSettings``).
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


_DEFAULT_LOCAL_BUFFER_SIZE = 16 * 1024 * 1024
_DEFAULT_GLOBAL_SEGMENT_SIZE = 4 * 1024 * 1024 * 1024
_DEFAULT_TENANT_ID = "default"


def _parse_global_segment_size(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("global_segment_size must be an int or size string")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text.endswith("gb"):
            number = text[:-2].strip()
            if not number:
                raise ValueError(
                    "Invalid global_segment_size: missing number before 'gb'"
                )
            return int(number) * 1024 * 1024 * 1024
        if text.endswith("mb"):
            number = text[:-2].strip()
            if not number:
                raise ValueError(
                    "Invalid global_segment_size: missing number before 'mb'"
                )
            return int(number) * 1024 * 1024
        return int(text)
    return int(value)


@dataclass(frozen=True, slots=True)
class MooncakeStoreConfig:
    local_hostname: str
    metadata_server: str
    global_segment_size: int
    protocol: str
    device_name: str
    master_server_address: str
    client_server_address: str
    # Required so a missed argument cannot silently join the shared default tenant.
    tenant_id: str

    @staticmethod
    def from_mapping(extra_config: dict[str, Any] | None) -> MooncakeStoreConfig:
        extra_config = extra_config or {}
        master = extra_config.get("master_server_address") or os.environ.get(
            "MOONCAKE_MASTER", ""
        )
        client = extra_config.get("client_server_address") or os.environ.get(
            "MOONCAKE_CLIENT", ""
        )
        if not master:
            master = client
        if not master:
            raise ValueError(
                "Mooncake Store requires master_server_address or client_server_address "
                "in --kvstore-storage-backend-extra-config, or MOONCAKE_MASTER / "
                "MOONCAKE_CLIENT in the environment"
            )
        local_hostname = (
            extra_config.get("local_hostname")
            or os.environ.get("MOONCAKE_LOCAL_HOSTNAME")
            or os.environ.get("LOCAL_HOSTNAME")
            or "localhost"
        )
        return MooncakeStoreConfig(
            local_hostname=str(local_hostname),
            metadata_server=str(
                extra_config.get(
                    "metadata_server",
                    os.environ.get("MOONCAKE_TE_META_DATA_SERVER", "P2PHANDSHAKE"),
                )
            ),
            global_segment_size=_parse_global_segment_size(
                extra_config.get(
                    "global_segment_size",
                    os.environ.get(
                        "MOONCAKE_GLOBAL_SEGMENT_SIZE", _DEFAULT_GLOBAL_SEGMENT_SIZE
                    ),
                )
            ),
            protocol=str(
                extra_config.get("protocol", os.environ.get("MOONCAKE_PROTOCOL", "tcp"))
            ),
            device_name=str(
                extra_config.get("device_name", os.environ.get("MOONCAKE_DEVICE", ""))
            ),
            master_server_address=str(master),
            client_server_address=str(client),
            tenant_id=str(extra_config.get("tenant_id") or _DEFAULT_TENANT_ID),
        )


@dataclass(frozen=True)
class MooncakeStoreSettings:
    """A Mooncake Store client as the engine host connects it, for one
    cache-owning rank.

    Attributes:
        library: ``libmooncake_store.so``.
        local_hostname: This client's host name.
        metadata_server: The transfer engine's metadata server
            (``P2PHANDSHAKE`` for none).
        global_segment_size: The bytes this rank mounts into the store: its
            share of the configured total; 0 mounts none.
        local_buffer_size: The client's local transfer buffer.
        protocol: ``tcp`` or ``rdma``.
        device_name: The RDMA devices; empty for all.
        master_server_address: The master's address.
        tenant_id: The tenant whose objects this client sees.
    """

    library: str
    local_hostname: str
    metadata_server: str
    global_segment_size: int
    local_buffer_size: int
    protocol: str
    device_name: str
    master_server_address: str
    tenant_id: str

    @classmethod
    def resolve(
        cls,
        extra_config: dict[str, Any] | None,
        *,
        library: str,
        tp_size: int,
        pp_size: int,
    ) -> MooncakeStoreSettings:
        """The settings of one of ``tp_size * pp_size`` cache-owning ranks,
        which mount equal shares of the configured segment, loading the
        store through ``library``."""
        config = MooncakeStoreConfig.from_mapping(extra_config)
        tp_size = max(int(tp_size), 1)
        pp_size = max(int(pp_size), 1)
        per_rank_segment = config.global_segment_size // (tp_size * pp_size)
        # Zero explicitly selects a client backed by independently mounted
        # store segments, so inference restarts need not remove cached objects.
        if config.global_segment_size < 0 or (
            config.global_segment_size > 0 and per_rank_segment == 0
        ):
            raise ValueError(
                "Mooncake global_segment_size "
                f"{config.global_segment_size} is too small to split "
                f"across tp_size={tp_size} pp_size={pp_size}"
            )
        return cls(
            library=library,
            local_hostname=config.local_hostname,
            metadata_server=config.metadata_server,
            global_segment_size=per_rank_segment,
            local_buffer_size=_DEFAULT_LOCAL_BUFFER_SIZE,
            protocol=config.protocol,
            device_name=config.device_name,
            master_server_address=config.master_server_address,
            tenant_id=config.tenant_id,
        )


@dataclass(frozen=True)
class L3Settings:
    """What the engine host starts the rank's L3 executor from (``ts-l3``'s
    ``Settings``): the limits and workers, the namespace, the Host pool the
    executor reads and writes by address, and the store.

    Attributes:
        runtime: The scheduler's storage contract and the executor's workers.
        deployment_id: The identity of everything that shapes the stored
            bytes (``storage_deployment_id``).
        has_draft: The draft model's cache is stored too; its weights follow
            the target's version.
        weight_version: The weight version objects are first named under.
        host_buffer: The pinned Host pool's address; its owner keeps it for
            the process.
        host_buffer_bytes: Its size.
        groups: Per cache group, in the scheduler's order: cache blocks per
            LCM block, bytes per cache block, and the owners its blocks are
            dealt to.
        lcm_block_bytes: Bytes of one LCM block.
        num_lcm_blocks: LCM blocks in the pool, the null block excluded.
        owner_rank: This rank among a sharded group's owners (KV parallel).
        tp_rank: This rank's attention-TP rank.
        backend: ``memory`` (in process, for tests) or ``mooncake``.
        mooncake: The Mooncake client, with that backend; else None.
    """

    runtime: StorageRuntimeConfig
    deployment_id: str
    has_draft: bool
    weight_version: str
    host_buffer: int
    host_buffer_bytes: int
    groups: tuple[tuple[int, int, int], ...]
    lcm_block_bytes: int
    num_lcm_blocks: int
    owner_rank: int
    tp_rank: int
    backend: str
    mooncake: MooncakeStoreSettings | None

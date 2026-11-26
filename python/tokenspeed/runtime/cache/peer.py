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


"""Peer prefix sharing: a replica reads a prefix another replica caches.

``docs/rfcs/0004-peer-prefix-sharing.md`` has the design. The router names a
holder with a request (``peer_holder``); the reader's scheduler asks it how
much of the prompt it holds, pins those blocks at admission, and the reader's
host READs them from the holder's Device or Host memory into its own Device
blocks over the transfer engine. Everything here is fixed at startup: the
scheduler's limits (``peer_scheduler_config``), and what the engine host
starts its peer endpoint and reads from (``PeerSharingSettings``).

Holder and reader must share the namespace (``deployment_id`` and weight
version), which every answer carries: the same checkpoints, cache layout and weight version, and
the same parallel layout, member by member.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tokenspeed_engine.scheduler import PeerConfig

    from tokenspeed.runtime.cache.host_storage import HostCacheStorage
    from tokenspeed.runtime.cache.transfer.layout import CacheGroupLayout
    from tokenspeed.runtime.plugins.registry import PeerHostRegionPlanner

# A holder holds at most this many leases (soft and hard pins) at once.
MAX_LEASES = 4096
# How long a lookup's soft pin keeps the holder's blocks warm: long enough
# for the reader to admit the request and ask for the hard pin.
SOFT_PIN_S = 2.0
# How long a hard pin holds the blocks: the reader's reads end, or fail at
# their deadline, long before.
HARD_PIN_S = 30.0
# A reader whose read failed holds the landing's blocks this long before
# reusing them, so an in-flight READ cannot land in another request's block.
QUARANTINE_S = 5.0
# A holder forgets a lease's record this long after its last request.
RECORD_LIFETIME_S = HARD_PIN_S + 30.0
# How long a holder may take to push its Disk pages into a reader's landing
# (docs/rfcs/0006-disk-tier.md); it posts no write past it, and a reader
# that stops waiting cancels the push first.
PUSH_TIMEOUT_S = 10.0
# Host landings: the share of the Host pool's blocks (in its densest group)
# landings hold at once, and how long a landed prefix is held for its
# request's admission.
HOST_LANDING_BLOCKS_FRACTION = 0.5
HOST_READY_S = 5.0


def peer_incarnation() -> int:
    """This process's incarnation: nonzero and 32-bit, as the scheduler's
    lease ids count up from it shifted left 32 bits. A restart draws a new
    one, so neither side mistakes the restarted process for the old one."""
    return secrets.randbits(32) or 1


def peer_endpoint_name(host: str, port: int, dp_rank: int = 0) -> str:
    """The name a replica is known by: the endpoint other replicas and the
    router reach it at."""
    from tokenspeed.runtime.utils.network import join_host_port

    address = join_host_port(host, port)
    return f"{address}/{dp_rank}" if dp_rank else address


def peer_scheduler_config(
    *,
    name: str,
    incarnation: int,
    answer_timeout_ms: int,
    min_fetch_tokens: int,
    landing_tier: str,
    num_host_blocks: int,
    versioned_shards: bool,
) -> PeerConfig:
    """The scheduler's peer sharing limits.

    Args:
        name: This replica's endpoint name (``peer_endpoint_name``).
        incarnation: This process's incarnation (``peer_incarnation``).
        answer_timeout_ms: How long a lookup waits for the holder's answer
            before the request is admitted without it.
        min_fetch_tokens: The fewest tokens worth reading from a peer past
            the local hit.
        landing_tier: ``--peer-prefix-landing-tier``: ``"device"`` lands a
            prefix into the Device pages the request's admission allocates,
            ``"host"`` into Host blocks before its admission.
        num_host_blocks: The Host pool's blocks in its densest group (its
            parents, the null block excluded, times that group's packing);
            Host landings hold a share of them.

    Returns:
        The ``SchedulerConfig.peer_config`` value.
    """
    from tokenspeed_engine.scheduler import PeerConfig, PeerLandingTier

    host = landing_tier == "host"
    if not host and landing_tier != "device":
        raise ValueError(f"unknown peer landing tier {landing_tier!r}")
    if answer_timeout_ms <= 0:
        raise ValueError(
            f"peer answer_timeout_ms must be positive, got {answer_timeout_ms}"
        )
    answer_timeout_s = answer_timeout_ms / 1000.0
    soft_pin_s = min(
        HARD_PIN_S,
        max(SOFT_PIN_S, (2.0 if host else 4.0) * answer_timeout_s + 2.0),
    )
    host_ready_s = max(HOST_READY_S, answer_timeout_s + 5.0)
    return PeerConfig(
        name=name,
        incarnation=incarnation,
        max_leases=MAX_LEASES,
        soft_pin_us=int(soft_pin_s * 1e6),
        hard_pin_us=int(HARD_PIN_S * 1e6),
        lookup_timeout_us=answer_timeout_ms * 1000,
        min_fetch_tokens=min_fetch_tokens,
        landing_tier=PeerLandingTier.Host if host else PeerLandingTier.Device,
        max_host_landing_blocks=(
            max(1, int(num_host_blocks * HOST_LANDING_BLOCKS_FRACTION)) if host else 0
        ),
        host_ready_timeout_us=int(host_ready_s * 1e6) if host else 0,
        versioned_shards=versioned_shards,
    )


@dataclass(frozen=True, slots=True)
class PeerDeviceField:
    """One Device field: page ``p`` sits at ``address + p * page_stride`` and
    holds ``bytes``. ``group`` is the cache group's index in the scheduler's
    order."""

    group: int
    field_id: str
    address: int
    page_stride: int
    bytes: int


@dataclass(frozen=True, slots=True)
class PeerHostGroup:
    """One cache group's Host blocks: ``packing`` blocks of ``block_bytes``
    per parent of ``parent_bytes``, each laid out field after field at
    ``field_offsets``."""

    parent_bytes: int
    packing: int
    block_bytes: int
    field_offsets: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class PeerHostPool:
    """The Host pool: its address and every cache group, in the scheduler's
    order."""

    address: int
    groups: tuple[PeerHostGroup, ...]


@dataclass(frozen=True, slots=True)
class PeerGeometry:
    """Where this rank's cache blocks live, for a peer to read them or land
    into them, and the memory the transfer engine registers for that."""

    device: tuple[PeerDeviceField, ...]
    host: PeerHostPool
    regions: tuple[tuple[int, int], ...]
    transfer_contract: bytes = b""
    semantic_id: bytes = b""
    host_virtual_bounds: tuple[int, ...] = ()


def peer_host_regions(
    address: int,
    length: int,
    parent_bytes: int,
    planner: PeerHostRegionPlanner | None,
) -> tuple[tuple[int, int], ...]:
    """Map the planner's lengths onto the pool, checking coverage and READ boundaries."""
    if planner is None:
        return ((address, length),)
    regions = []
    offset = 0
    for size in planner(length, parent_bytes):
        if type(size) is not int or size <= 0 or size > length - offset:
            raise ValueError("Invalid peer Host region length")
        regions.append((address + offset, size))
        offset += size
        if offset < length and offset % parent_bytes:
            raise ValueError("A peer Host region boundary splits a compact parent")
    if offset != length:
        raise ValueError("Peer Host regions must cover the entire pool")
    return tuple(regions)


def build_peer_geometry(
    *,
    scheduler_group_ids: Sequence[str],
    groups: Sequence[CacheGroupLayout],
    field_addresses: Mapping[str, int],
    device_regions: Sequence[tuple[int, int]],
    host_storage: HostCacheStorage,
    host_region_planner: PeerHostRegionPlanner | None = None,
    transfer_contract: bytes = b"",
    semantic_id: bytes = b"",
    host_virtual_bounds: tuple[int, ...] = (),
) -> PeerGeometry:
    """This rank's geometry: the fields the Host tier stores (the target's
    and the draft's), at their Device addresses and their offsets in the
    Host pool.

    Args:
        scheduler_group_ids: The cache groups in the scheduler's order, whose
            index the scheduler names a group by.
        groups: The Host tier's layout groups (``HostCacheResources.groups``).
        field_addresses: Each field's Device page 0, by field id
            (``CacheArena.field_address``).
        device_regions: The Device ranges the cache lives in
            (``CacheArena.registration_regions``).
        host_storage: The Host pool.
        host_region_planner: Optional device plugin supplying consecutive region sizes.

    Returns:
        The geometry, registering the Device ranges and the Host pool.
    """
    # The scheduler names a group by its index; the Host lists every group,
    # in the same order.
    if tuple(host_storage.group_ids) != tuple(scheduler_group_ids):
        raise ValueError(
            "peer sharing needs every cache group in the Host layout, in the "
            f"scheduler's order: {host_storage.group_ids} != "
            f"{tuple(scheduler_group_ids)}"
        )
    device = []
    host_groups = []
    for index, group in enumerate(groups):
        offsets = host_storage.host_field_offsets[index]
        for field in group.fields:
            device.append(
                PeerDeviceField(
                    group=index,
                    field_id=field.field_id,
                    address=int(field_addresses[field.field_id]),
                    page_stride=int(field.block_stride_bytes),
                    bytes=int(field.payload_bytes),
                )
            )
        host_groups.append(
            PeerHostGroup(
                parent_bytes=int(host_storage.host_lcm_block_bytes),
                packing=int(host_storage.group_packing[index]),
                block_bytes=int(host_storage.host_cache_block_bytes[index]),
                field_offsets=tuple(
                    (field.field_id, int(offset))
                    for field, offset in zip(group.fields, offsets, strict=True)
                ),
            )
        )
    host_address = int(host_storage.host_buffer.data_ptr())
    host_bytes = int(host_storage.host_buffer_bytes)
    host_regions = peer_host_regions(
        host_address,
        host_bytes,
        host_storage.host_lcm_block_bytes,
        host_region_planner,
    )
    return PeerGeometry(
        device=tuple(device),
        host=PeerHostPool(address=host_address, groups=tuple(host_groups)),
        transfer_contract=transfer_contract,
        semantic_id=semantic_id,
        host_virtual_bounds=host_virtual_bounds,
        regions=(
            *((int(address), int(length)) for address, length in device_regions),
            *host_regions,
        ),
    )


@dataclass(frozen=True, slots=True)
class PeerSharingSettings:
    """What the engine host starts this rank's peer sharing from.

    Attributes:
        name: The advertised endpoint, including the nonzero DP rank suffix.
        node_leaders: (global rank, DP rank) of each local replica leader.
        bind_host: The host the replica's first member binds its endpoint
            on.
        port: The endpoint's port.
        incarnation: This process's incarnation.
        deployment_id: The cache namespace's deployment id
            (``cache_deployment_id``): only a replica of the same id and
            weight version is read.
        weight_version: The weight version the namespace starts at; a
            flushed weight update moves it.
        answer_timeout_s: How long a landing waits for each answer.
        quarantine_s: How long a failed landing's blocks stay unused.
        push_timeout_s: How long a holder may take to push its Disk pages.
        record_lifetime_s: How long a holder keeps a lease's record.
        library: The transfer engine library.
        local_ip: The address the transfer engine starts on.
        push_host: The host other replicas reach this rank's push endpoint
            at (``peer::push``).
        protocol: The transfer engine's transport.
        ib_device: The RDMA devices, comma-separated; empty for all.
        geometry: This rank's geometry.
    """

    name: str
    node_leaders: tuple[tuple[int, int], ...]
    bind_host: str
    port: int
    incarnation: int
    deployment_id: str
    weight_version: str
    answer_timeout_s: float
    quarantine_s: float
    push_timeout_s: float
    record_lifetime_s: float
    library: str
    local_ip: str
    push_host: str
    protocol: str
    ib_device: str
    geometry: PeerGeometry
    semantic_id: bytes = b""
    source_role: str = ""


def peer_semantic_id(
    model_namespace: str,
    revision: str | None,
    weight_version: str | None,
    state_abi: dict | None,
) -> bytes:
    """Identify the model generation and the recipe's cross-layout state encoding."""
    revision = (revision or "").lower()
    pinned = len(revision) == 40 and all(c in "0123456789abcdef" for c in revision)
    version = (weight_version or "").strip()
    if version == "default":
        version = ""
    if not model_namespace or not state_abi or not (pinned or version):
        return b""
    identity = {
        "format": 2,
        "model": model_namespace,
        "revision": revision if pinned else "",
        "weight_version": version,
        "state_abi": state_abi,
    }
    payload = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode()).digest()

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

import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tokenspeed_engine.scheduler import PeerConfig

    from tokenspeed.runtime.cache.host_storage import HostCacheStorage
    from tokenspeed.runtime.cache.transfer.layout import CacheGroupLayout

# A holder holds at most this many leases (soft and hard pins) at once.
MAX_LEASES = 4096
# How long a lookup's soft pin keeps the holder's blocks warm: long enough
# for the reader to admit the request and ask for the hard pin.
SOFT_PIN_S = 2.0
# How long a hard pin holds the blocks: the reader's reads end, or fail at
# their deadline, long before.
HARD_PIN_S = 30.0
# A reader whose read failed holds the landing's blocks this long before
# reusing them, so a READ still in flight cannot land in another request's
# block.
QUARANTINE_S = 5.0
# A holder forgets a lease's record this long after its last request.
RECORD_LIFETIME_S = HARD_PIN_S + 30.0


def peer_incarnation() -> int:
    """This process's incarnation: nonzero and 32-bit, as the scheduler's
    lease ids count up from it shifted left 32 bits. A restart draws a new
    one, so neither side mistakes the restarted process for the old one."""
    return secrets.randbits(32) or 1


def peer_endpoint_name(host: str, port: int) -> str:
    """The name a replica is known by: the endpoint other replicas and the
    router reach it at."""
    from tokenspeed.runtime.utils.network import join_host_port

    return join_host_port(host, port)


def peer_scheduler_config(
    *,
    name: str,
    incarnation: int,
    answer_timeout_ms: int,
    min_fetch_tokens: int,
) -> PeerConfig:
    """The scheduler's peer sharing limits.

    Args:
        name: This replica's endpoint name (``peer_endpoint_name``).
        incarnation: This process's incarnation (``peer_incarnation``).
        answer_timeout_ms: How long a lookup waits for the holder's answer
            before the request is admitted without it.
        min_fetch_tokens: The fewest tokens worth reading from a peer past
            the local hit.

    Returns:
        The ``SchedulerConfig.peer_config`` value.
    """
    from tokenspeed_engine.scheduler import PeerConfig

    return PeerConfig(
        name=name,
        incarnation=incarnation,
        max_leases=MAX_LEASES,
        soft_pin_us=int(SOFT_PIN_S * 1e6),
        hard_pin_us=int(HARD_PIN_S * 1e6),
        lookup_timeout_us=answer_timeout_ms * 1000,
        min_fetch_tokens=min_fetch_tokens,
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


def build_peer_geometry(
    *,
    scheduler_group_ids: Sequence[str],
    groups: Sequence[CacheGroupLayout],
    field_addresses: Mapping[str, int],
    device_regions: Sequence[tuple[int, int]],
    host_storage: HostCacheStorage,
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
    return PeerGeometry(
        device=tuple(device),
        host=PeerHostPool(address=host_address, groups=tuple(host_groups)),
        regions=(
            *((int(address), int(length)) for address, length in device_regions),
            (host_address, int(host_storage.host_buffer_bytes)),
        ),
    )


@dataclass(frozen=True, slots=True)
class PeerSharingSettings:
    """What the engine host starts this rank's peer sharing from.

    Attributes:
        name: The replica's endpoint name, ``host:port``.
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
        record_lifetime_s: How long a holder keeps a lease's record.
        library: The transfer engine library.
        local_ip: The address the transfer engine starts on.
        protocol: The transfer engine's transport.
        ib_device: The RDMA devices, comma-separated; empty for all.
        geometry: This rank's geometry.
    """

    name: str
    bind_host: str
    port: int
    incarnation: int
    deployment_id: str
    weight_version: str
    answer_timeout_s: float
    quarantine_s: float
    record_lifetime_s: float
    library: str
    local_ip: str
    protocol: str
    ib_device: str
    geometry: PeerGeometry

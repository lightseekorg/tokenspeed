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

"""Peer-sharded GPU HBM memory mapped into one contiguous virtual address range.

Each peer GPU creates one physical CUDA VMM shard of a large read-mostly table
(such as DeepSeek V4.1 Engram) in its own HBM, exports that shard through
either a FABRIC handle (MNNVL across hosts or single-node IMEX) or a POSIX file
descriptor (single-node without IMEX), and maps every peer's shard back-to-back
into one reserved virtual address range. GPU kernels then index the resulting
tensor directly across peers over NVLink without an all-reduce or host memory.

In intra-job mode (``DistributedPeerExchange``), supports memory-saver
``release_storage`` / ``resume_storage``: release copies only this peer's local
shard to CPU (when ``backup=True``), unmaps and releases all physical handles
while keeping the virtual address reserved, and resume remaps fresh handles at
the exact same address so captured CUDA graphs stay valid.

In cross-process mode (``TCPStorePeerExchange``), independent engine processes
connect as clients to a standalone ``TCPStore`` (started via
:func:`serve_peer_rendezvous` or ``python -m tokenspeed_kernel.peer_memory
--serve host:port``) using a per-launch ``rendezvous_id``. Cross-process peer
groups are immutable serving units: per-engine ``release_storage`` /
``resume_storage`` is rejected, and recovering or updating a group restarts all
peers with a fresh ``rendezvous_id``.
"""

from __future__ import annotations

import argparse
import ctypes
import functools
import logging
import os
import pickle
import signal
import socket
import struct
import time
import uuid
import weakref
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import timedelta
from typing import Protocol, runtime_checkable

import torch
import torch.distributed as dist
from tokenspeed_kernel.platform import current_platform

__all__ = [
    "DistributedPeerExchange",
    "LocalPeerExchange",
    "PeerDeviceLayout",
    "PeerDeviceMemory",
    "PeerMemoryExchange",
    "TCPStorePeerExchange",
    "create_peer_device_memory",
    "parse_host_port",
    "plan_peer_device_layout",
    "plan_peer_row_range",
    "platform_supported",
    "probe_peer_device_memory",
    "query_peer_fabric_domain",
    "serve_peer_rendezvous",
    "validate_peer_handle_type",
]

logger = logging.getLogger(__name__)

_FD_EXCHANGE_TIMEOUT_S = 300.0
_SUPPORTED_HANDLE_TYPES = frozenset({"fabric", "posix_fd"})
_NVML_GPU_FABRIC_STATE_COMPLETED = 3


def platform_supported() -> bool:
    """Return True on NVIDIA platforms where CUDA VMM peer memory exists."""
    return current_platform().is_nvidia


def _vmm():
    if not platform_supported():
        return None
    try:
        from tokenspeed_kernel.thirdparty.cuda.dwdp import vmm

        return vmm
    except Exception as exc:
        logger.info(f"CUDA VMM unavailable for peer device memory: {exc!s}")
        return None


def _require_vmm():
    vmm = _vmm()
    if vmm is None:
        raise RuntimeError(
            "Peer-sharded GPU VMM memory requires an NVIDIA GPU and cuda.bindings"
        )
    return vmm


def _ensure_driver_context(device_index: int) -> None:
    """Ensure a CUDA driver context is current on ``device_index``."""
    from cuda.bindings import driver as cu

    err, *_ = cu.cuInit(0)
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuInit failed: {err.name}")
    err, current = cu.cuCtxGetCurrent()
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuCtxGetCurrent failed: {err.name}")
    if int(current):
        return
    err, context = cu.cuDevicePrimaryCtxRetain(device_index)
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuDevicePrimaryCtxRetain failed: {err.name}")
    err, *_ = cu.cuCtxSetCurrent(context)
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuCtxSetCurrent failed: {err.name}")


@dataclass(frozen=True)
class PeerDeviceLayout:
    """Contiguous byte-granule partitioning of one table across peer GPUs.

    ``chunks`` holds one ``(offset, length, owner_rank)`` tuple per peer rank
    in rank order ``0 .. peer_world_size - 1``. Peers past the end of a tiny
    table (fewer granules than peers) have ``length == 0``.
    """

    nbytes: int
    granularity: int
    peer_world_size: int
    peer_rank: int
    chunks: tuple[tuple[int, int, int], ...]

    @property
    def padded_bytes(self) -> int:
        return sum(length for _, length, _ in self.chunks)

    @property
    def local_offset(self) -> int:
        return self.chunks[self.peer_rank][0]

    @property
    def local_length(self) -> int:
        return self.chunks[self.peer_rank][1]

    @property
    def local_live_bytes(self) -> int:
        """Bytes of the unpadded ``nbytes`` payload backed by this peer."""
        if self.local_length == 0 or self.local_offset >= self.nbytes:
            return 0
        return min(self.local_length, self.nbytes - self.local_offset)


def plan_peer_device_layout(
    nbytes: int,
    granularity: int,
    peer_rank: int,
    peer_world_size: int,
) -> PeerDeviceLayout:
    """Partition ``nbytes`` into contiguous granularity-aligned peer shards.

    Args:
        nbytes: Logical byte size of the contiguous table.
        granularity: CUDA VMM allocation granularity in bytes (e.g. 2 MiB).
        peer_rank: This peer's index in ``[0, peer_world_size)``.
        peer_world_size: Total number of peer GPUs sharing the table.

    Returns:
        A :class:`PeerDeviceLayout` describing every peer's offset and length.
    """
    if nbytes < 1 or granularity < 1:
        raise ValueError(
            f"Peer device layout needs positive nbytes and granularity; "
            f"got nbytes={nbytes}, granularity={granularity}"
        )
    if peer_world_size < 1 or not 0 <= peer_rank < peer_world_size:
        raise ValueError(
            f"Invalid peer rank {peer_rank} for peer_world_size={peer_world_size}"
        )
    total_granules = (nbytes + granularity - 1) // granularity
    base, remainder = divmod(total_granules, peer_world_size)
    chunks: list[tuple[int, int, int]] = []
    offset = 0
    for rank in range(peer_world_size):
        granules = base + (1 if rank < remainder else 0)
        length = granules * granularity
        chunks.append((offset, length, rank))
        offset += length
    return PeerDeviceLayout(
        nbytes=nbytes,
        granularity=granularity,
        peer_world_size=peer_world_size,
        peer_rank=peer_rank,
        chunks=tuple(chunks),
    )


def plan_peer_row_range(
    num_rows: int,
    row_bytes: int,
    layout: PeerDeviceLayout,
) -> tuple[int, int]:
    """Return disjoint, complete ``[row_start, row_end)`` owned by this peer.

    Every row in ``[0, num_rows)`` is assigned to exactly one peer rank so
    parallel checkpoint loading and in-place canonicalization cover the table
    without overlap or gaps, even when ``row_bytes`` does not divide
    ``layout.granularity`` or ``num_rows * row_bytes`` exceeds 2 GiB.

    Args:
        num_rows: Total rows in the 2D table.
        row_bytes: Byte stride of one packed row.
        layout: Byte-granule layout from :func:`plan_peer_device_layout`.

    Returns:
        ``(row_start, row_end)`` inclusive/exclusive row bounds for this peer.
    """
    if num_rows < 0 or row_bytes < 1:
        raise ValueError(
            f"Peer row range requires num_rows >= 0 and row_bytes >= 1; "
            f"got num_rows={num_rows}, row_bytes={row_bytes}"
        )
    if num_rows == 0:
        return (0, 0)
    if num_rows * row_bytes > layout.nbytes:
        raise ValueError(
            f"Table byte size {num_rows * row_bytes} exceeds layout nbytes {layout.nbytes}"
        )

    def _boundary_row(byte_offset: int) -> int:
        if byte_offset <= 0:
            return 0
        return min(num_rows, (byte_offset + row_bytes - 1) // row_bytes)

    if layout.local_length == 0:
        boundary = _boundary_row(layout.local_offset)
        return (boundary, boundary)

    row_start = 0 if layout.peer_rank == 0 else _boundary_row(layout.local_offset)
    if layout.peer_rank == layout.peer_world_size - 1:
        row_end = num_rows
    else:
        row_end = _boundary_row(layout.local_offset + layout.local_length)
    return (row_start, row_end)


def parse_host_port(endpoint: str) -> tuple[str, int]:
    """Parse ``host:port``, ``[ipv6]:port``, or ``tcp://...`` into ``(host, port)``."""
    raw = endpoint.strip()
    if raw.startswith("tcp://"):
        raw = raw[len("tcp://") :]
    if not raw:
        raise ValueError("Peer rendezvous endpoint must not be empty")
    if raw.startswith("["):
        closing = raw.find("]")
        if closing == -1 or closing + 1 >= len(raw) or raw[closing + 1] != ":":
            raise ValueError(
                f"Invalid bracketed IPv6 rendezvous endpoint: {endpoint!r}"
            )
        host = raw[1:closing]
        port_text = raw[closing + 2 :]
    else:
        if ":" not in raw:
            raise ValueError(
                f"Peer rendezvous endpoint must include :port, got {endpoint!r}"
            )
        host, port_text = raw.rsplit(":", 1)
        if ":" in host:
            raise ValueError(
                f"IPv6 peer rendezvous endpoint must bracket the host: {endpoint!r}"
            )
    if not host:
        raise ValueError(f"Missing host in peer rendezvous endpoint: {endpoint!r}")
    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValueError(
            f"Invalid port in peer rendezvous endpoint: {endpoint!r}"
        ) from exc
    if not 1 <= port <= 65535:
        raise ValueError(
            f"Peer rendezvous port must be in 1..65535, got {port} from {endpoint!r}"
        )
    return host, port


def serve_peer_rendezvous(
    endpoint: str,
    *,
    timeout_s: float = 300.0,
) -> dist.TCPStore:
    """Start a standalone ``TCPStore`` server independent of any engine worker.

    Args:
        endpoint: ``host:port`` or ``[ipv6]:port`` bind address.
        timeout_s: Store socket timeout in seconds.

    Returns:
        The live master :class:`torch.distributed.TCPStore`. Keep the returned
        object alive for the duration of the peer group.
    """
    host, port = parse_host_port(endpoint)
    return dist.TCPStore(
        host_name=host,
        port=port,
        world_size=None,
        is_master=True,
        timeout=timedelta(seconds=timeout_s),
        wait_for_workers=False,
    )


@runtime_checkable
class PeerMemoryExchange(Protocol):
    """Control-plane exchange used to share VMM handles and barriers across peers."""

    @property
    def rank(self) -> int: ...

    @property
    def world_size(self) -> int: ...

    @property
    def is_cross_process(self) -> bool: ...

    def scoped(self, namespace: str) -> PeerMemoryExchange: ...

    def all_gather_object(self, key: str, value: object) -> list[object]: ...

    def barrier(self, key: str) -> None: ...


class LocalPeerExchange:
    """Single-rank peer exchange with no process group (world_size == 1)."""

    def __init__(self, *, namespace: str = "") -> None:
        self._namespace = namespace
        self._used_scopes: set[str] = set()

    @property
    def rank(self) -> int:
        return 0

    @property
    def world_size(self) -> int:
        return 1

    @property
    def is_cross_process(self) -> bool:
        return False

    def scoped(self, namespace: str) -> LocalPeerExchange:
        if namespace in self._used_scopes:
            raise ValueError(
                f"Peer memory scope {namespace!r} has already been created on "
                f"{self._namespace!r}"
            )
        self._used_scopes.add(namespace)
        prefix = f"{self._namespace}/{namespace}" if self._namespace else namespace
        return LocalPeerExchange(namespace=prefix)

    def all_gather_object(self, key: str, value: object) -> list[object]:
        del key
        return [value]

    def barrier(self, key: str) -> None:
        del key


class DistributedPeerExchange:
    """Intra-job peer exchange over an existing ``torch.distributed`` group."""

    def __init__(
        self,
        group: dist.ProcessGroup | None,
        *,
        namespace: str = "",
    ) -> None:
        if not dist.is_initialized():
            raise RuntimeError(
                "DistributedPeerExchange requires an initialized process group"
            )
        self._group = group
        self._namespace = namespace
        self._used_scopes: set[str] = set()
        self._rank = dist.get_rank(group=group)
        self._world_size = dist.get_world_size(group=group)

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def is_cross_process(self) -> bool:
        return False

    def scoped(self, namespace: str) -> DistributedPeerExchange:
        if namespace in self._used_scopes:
            raise ValueError(
                f"Peer memory scope {namespace!r} has already been created on "
                f"{self._namespace!r}"
            )
        self._used_scopes.add(namespace)
        prefix = f"{self._namespace}/{namespace}" if self._namespace else namespace
        return DistributedPeerExchange(self._group, namespace=prefix)

    def all_gather_object(self, key: str, value: object) -> list[object]:
        del key
        gathered: list[object] = [None] * self._world_size
        dist.all_gather_object(gathered, value, group=self._group)
        return gathered

    def barrier(self, key: str) -> None:
        del key
        dist.barrier(group=self._group)


class TCPStorePeerExchange:
    """Cross-process peer exchange over an externally hosted ``TCPStore``.

    Every peer connects as a client (``is_master=False``) to a standalone store
    created via :func:`serve_peer_rendezvous`, isolating the rendezvous server
    from individual worker restarts. Every key carries the per-launch
    ``rendezvous_id`` and a per-scope monotonic call sequence so repeated
    exchanges or barriers never collide with earlier keys. Each scope name
    may be opened only once per parent exchange so sequence numbers stay unique.
    """

    def __init__(
        self,
        store: dist.Store,
        *,
        rank: int,
        world_size: int,
        rendezvous_id: str,
        namespace: str,
    ) -> None:
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError(
                f"Invalid TCPStore peer rank {rank} for world_size={world_size}"
            )
        if not rendezvous_id or "/" in rendezvous_id:
            raise ValueError(
                "TCPStorePeerExchange requires a non-empty rendezvous_id without '/'; "
                f"got {rendezvous_id!r}"
            )
        self._store = store
        self._rank = rank
        self._world_size = world_size
        self._rendezvous_id = rendezvous_id
        self._namespace = namespace
        self._seq = 0
        self._used_scopes: set[str] = set()

    @classmethod
    def connect(
        cls,
        endpoint: str,
        *,
        rank: int,
        world_size: int,
        rendezvous_id: str,
        timeout_s: float = 300.0,
        namespace: str = "engram-peer",
    ) -> TCPStorePeerExchange:
        """Connect as a client to an externally hosted standalone ``TCPStore``."""
        if not rendezvous_id or "/" in rendezvous_id:
            raise ValueError(
                "TCPStorePeerExchange requires a non-empty rendezvous_id without '/'; "
                f"got {rendezvous_id!r}"
            )
        host, port = parse_host_port(endpoint)
        store = dist.TCPStore(
            host_name=host,
            port=port,
            world_size=None,
            is_master=False,
            timeout=timedelta(seconds=timeout_s),
            wait_for_workers=False,
        )
        return cls(
            store,
            rank=rank,
            world_size=world_size,
            rendezvous_id=rendezvous_id,
            namespace=namespace,
        )

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def is_cross_process(self) -> bool:
        return True

    @property
    def rendezvous_id(self) -> str:
        return self._rendezvous_id

    def scoped(self, namespace: str) -> TCPStorePeerExchange:
        if namespace in self._used_scopes:
            raise ValueError(
                f"Peer memory scope {namespace!r} has already been created on "
                f"{self._namespace!r}"
            )
        self._used_scopes.add(namespace)
        prefix = f"{self._namespace}/{namespace}" if self._namespace else namespace
        return TCPStorePeerExchange(
            self._store,
            rank=self._rank,
            world_size=self._world_size,
            rendezvous_id=self._rendezvous_id,
            namespace=prefix,
        )

    def _next_seq(self) -> int:
        seq = self._seq
        self._seq += 1
        return seq

    def _full_key(self, key: str, seq: int, peer: int) -> str:
        return f"{self._namespace}/{self._rendezvous_id}/{key}/s{seq}/{peer}"

    def all_gather_object(self, key: str, value: object) -> list[object]:
        seq = self._next_seq()
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        self._store.set(self._full_key(key, seq, self._rank), payload)
        keys = [self._full_key(key, seq, peer) for peer in range(self._world_size)]
        self._store.wait(keys)
        return [pickle.loads(self._store.get(item)) for item in keys]

    def barrier(self, key: str) -> None:
        seq = self._next_seq()
        self._store.set(self._full_key(f"barrier/{key}", seq, self._rank), b"1")
        keys = [
            self._full_key(f"barrier/{key}", seq, peer)
            for peer in range(self._world_size)
        ]
        self._store.wait(keys)


@functools.cache
def query_peer_fabric_domain(device_index: int) -> tuple[str, int]:
    """Return ``(cluster_uuid_hex, clique_id)`` from NVML GPU fabric info.

    Queries ``nvmlDeviceGetGpuFabricInfoV`` using ``c_nvmlGpuFabricInfo_v2_t``
    so cross-slice fabric mismatches fail before VMM handle import. Fails
    closed with ``RuntimeError`` whenever the NVML query is unavailable,
    incomplete, or reports an empty cluster UUID.
    """
    if not platform_supported():
        raise RuntimeError(
            "NVLink fabric domain query requires an NVIDIA CUDA platform"
        )
    nvml_initialized = False
    try:
        import pynvml

        pynvml.nvmlInit()
        nvml_initialized = True
        props = torch.cuda.get_device_properties(device_index)
        pci_bus_id = (
            f"{props.pci_domain_id:08X}:{props.pci_bus_id:02X}:"
            f"{props.pci_device_id:02X}.0"
        )
        handle = pynvml.nvmlDeviceGetHandleByPciBusId(pci_bus_id)
        info = pynvml.c_nvmlGpuFabricInfo_v2_t()
        info.version = pynvml.nvmlGpuFabricInfo_v2
        pynvml.nvmlDeviceGetGpuFabricInfoV(handle, ctypes.byref(info))
        cluster_uuid = bytes(info.clusterUuid)
        if not any(cluster_uuid):
            raise RuntimeError(
                f"GPU {device_index} reported an empty NVLink fabric clusterUuid"
            )
        state = int(info.state)
        status = int(info.status)
        if status != 0 or state != _NVML_GPU_FABRIC_STATE_COMPLETED:
            raise RuntimeError(
                f"GPU {device_index} NVLink fabric is not ready "
                f"(state={state}, status={status})"
            )
        return (cluster_uuid.hex(), int(info.cliqueId))
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"NVML GPU fabric domain query failed on device {device_index}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    finally:
        if nvml_initialized:
            pynvml.nvmlShutdown()


@functools.cache
def probe_peer_device_memory(device_index: int, handle_type: str) -> str | None:
    """Return ``None`` if a shareable VMM device granule can be created and exported.

    Args:
        device_index: CUDA device ordinal to probe.
        handle_type: Explicit shareable handle type (``"fabric"`` or ``"posix_fd"``).

    Returns:
        ``None`` when creation and export succeeded, else a diagnostic reason.
    """
    if handle_type not in _SUPPORTED_HANDLE_TYPES:
        raise ValueError(
            f"handle_type must be one of {sorted(_SUPPORTED_HANDLE_TYPES)}; "
            f"got {handle_type!r}"
        )
    vmm = _vmm()
    if vmm is None:
        return "peer GPU VMM memory needs an NVIDIA GPU and cuda.bindings"
    try:
        _ensure_driver_context(device_index)
        if handle_type == "fabric":
            query_peer_fabric_domain(device_index)
        gran = vmm.granularity(device_index, handle_type=handle_type)
        handle = vmm.allocate(gran, device_index, handle_type=handle_type)
        try:
            exported = vmm.export(handle, handle_type=handle_type)
            if handle_type == "posix_fd":
                os.close(int(exported))
        finally:
            vmm.release_handle(handle)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def validate_peer_handle_type(
    handle_type: str,
    peer_hosts: Sequence[int],
    peer_probes: Sequence[str | None],
    peer_fabric_domains: Sequence[tuple[str, int] | None] | None = None,
) -> str:
    """Validate that every peer supports the explicitly requested VMM handle type.

    Args:
        handle_type: Explicit shareable handle type (``"fabric"`` or ``"posix_fd"``).
        peer_hosts: Per-peer host identity integers.
        peer_probes: Per-peer :func:`probe_peer_device_memory` results for
            ``handle_type`` (``None`` means supported).
        peer_fabric_domains: Optional per-peer ``(cluster_uuid, clique_id)``
            tuples from :func:`query_peer_fabric_domain`. When ``handle_type``
            is ``"fabric"`` and any peer reports a domain, all peers must share
            the exact same domain.

    Returns:
        The validated ``handle_type``.
    """
    if handle_type not in _SUPPORTED_HANDLE_TYPES:
        raise ValueError(
            f"Peer handle_type must be fabric or posix_fd; got {handle_type!r}"
        )
    if not peer_hosts or len(peer_hosts) != len(peer_probes):
        raise ValueError("Peer host and probe lists must be nonempty and equal length")
    if peer_fabric_domains is not None and len(peer_fabric_domains) != len(peer_hosts):
        raise ValueError(
            "Peer fabric domain list must have the same length as peer_hosts"
        )
    errors = [
        f"peer {rank}: {err}" for rank, err in enumerate(peer_probes) if err is not None
    ]
    if handle_type == "posix_fd":
        distinct_hosts = len(set(peer_hosts))
        if distinct_hosts != 1:
            raise RuntimeError(
                "Peer GPU VMM posix_fd handles require all peers on one host "
                f"(found {distinct_hosts} distinct hosts); use fabric handles for MNNVL"
            )
        if errors:
            raise RuntimeError(
                "Peer GPU VMM posix_fd handles unavailable: " + "; ".join(errors)
            )
        return "posix_fd"
    if errors:
        raise RuntimeError(
            "Peer GPU VMM fabric handles unavailable: " + "; ".join(errors)
        )
    if peer_fabric_domains is None:
        raise RuntimeError(
            "Peer GPU VMM fabric handles require reported NVLink fabric domains "
            "from every peer"
        )
    domains = set(peer_fabric_domains)
    if None in domains or len(domains) != 1:
        summary = ", ".join(
            f"peer {rank}={domain}" for rank, domain in enumerate(peer_fabric_domains)
        )
        raise RuntimeError(
            "Peer GPU VMM fabric handles require all peers in the same NVLink "
            f"fabric cluster and clique: {summary}"
        )
    return "fabric"


def _check_peer_uid(connection: socket.socket, uid: int) -> None:
    _, peer_uid, _ = struct.unpack(
        "iII",
        connection.getsockopt(
            socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("iII")
        ),
    )
    if peer_uid != uid:
        raise PermissionError(
            f"Peer descriptor exchange UID {peer_uid} differs from {uid}"
        )


def _send_single_fd(
    server: socket.socket, fd: int, num_receivers: int, *, uid: int
) -> None:
    for _ in range(num_receivers):
        connection, _ = server.accept()
        with connection:
            connection.settimeout(_FD_EXCHANGE_TIMEOUT_S)
            _check_peer_uid(connection, uid)
            socket.send_fds(connection, [b"1"], [fd])


def _receive_single_fd(address: str, *, uid: int) -> int:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(_FD_EXCHANGE_TIMEOUT_S)
        client.connect(address)
        _check_peer_uid(client, uid)
        _, fds, flags, _ = socket.recv_fds(client, 1, 1)
        if flags & socket.MSG_CTRUNC:
            for fd in fds:
                os.close(fd)
            raise RuntimeError("Peer descriptor message was truncated")
        if len(fds) != 1:
            for fd in fds:
                os.close(fd)
            raise ConnectionError(
                f"Expected 1 peer descriptor from {address!r}, got {len(fds)}"
            )
        return int(fds[0])


@dataclass
class _VmmResources:
    """Mutable VMM resource box detached from ``PeerDeviceMemory`` for weakref cleanup."""

    vmm: object
    va: int
    padded_bytes: int
    mapped_ranges: list[tuple[int, int]]
    handles: list[int]
    freed: bool = False


def _cleanup_vmm(resources: _VmmResources) -> None:
    if resources.freed:
        return
    resources.freed = True
    vmm = resources.vmm
    try:
        for address, length in reversed(resources.mapped_ranges):
            vmm.unmap(address, length)
        resources.mapped_ranges.clear()
        if resources.va:
            vmm.free_address(resources.va, resources.padded_bytes)
            resources.va = 0
    finally:
        for handle in reversed(resources.handles):
            vmm.release_handle(handle)
        resources.handles.clear()


class PeerDeviceMemory:
    """Contiguous CUDA tensor backed by peer-sharded CUDA VMM allocations.

    ``tensor`` is a 1D ``uint8[nbytes]`` CUDA tensor on ``device_index`` whose
    virtual address stays constant across :meth:`release_storage` and
    :meth:`resume_storage`. All physical GPU memory lives in CUDA VMM handles
    (0 bytes in PyTorch's ordinary ``(0, 0)`` caching-allocator pool).
    """

    def __init__(
        self,
        *,
        layout: PeerDeviceLayout,
        device_index: int,
        handle_type: str,
        exchange: PeerMemoryExchange,
        va: int,
        resources: _VmmResources,
        view_owner: object,
        tensor: torch.Tensor,
        finalizer: weakref.finalize,
    ) -> None:
        self.layout = layout
        self.device_index = device_index
        self.handle_type = handle_type
        self._exchange = exchange
        self._va = va
        self._resources = resources
        self._view_owner = view_owner
        self.tensor = tensor
        self._finalizer = finalizer
        self._backup: torch.Tensor | None = None
        self._epoch = 0
        self._released = False
        self._closed = False

    @property
    def is_cross_process(self) -> bool:
        """Whether this peer allocation spans independent engine processes."""
        return self._exchange.is_cross_process

    @property
    def released(self) -> bool:
        """Whether physical GPU VMM handles are currently released."""
        return self._released

    @property
    def closed(self) -> bool:
        """Whether virtual address and handles have been permanently freed."""
        return self._closed

    @property
    def local_physical_bytes(self) -> int:
        """Physical GPU HBM bytes currently allocated by this peer."""
        return 0 if (self._released or self._closed) else self.layout.local_length

    def _map_shards(self, epoch_key: str) -> None:
        vmm = self._resources.vmm
        _ensure_driver_context(self.device_index)
        local_length = self.layout.local_length
        local_handle: int | None = None
        local_handles_created: list[int] = []
        mapped_ranges: list[tuple[int, int]] = []
        try:
            with ExitStack() as cleanup:
                exported_payload: bytes | None = None
                exported_fd: int | None = None
                server: socket.socket | None = None
                socket_address: str | None = None
                if local_length > 0:
                    local_handle = vmm.allocate(
                        local_length,
                        self.device_index,
                        handle_type=self.handle_type,
                    )
                    local_handles_created.append(local_handle)
                    exported = vmm.export(local_handle, handle_type=self.handle_type)
                    if self.handle_type == "fabric":
                        exported_payload = bytes(exported)
                    else:
                        exported_fd = int(exported)
                        cleanup.callback(os.close, exported_fd)
                        if self.layout.peer_world_size > 1:
                            server = cleanup.enter_context(
                                socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                            )
                            socket_address = f"\0tokenspeed-peermem-{uuid.uuid4().hex}"
                            server.bind(socket_address)
                            server.listen(self.layout.peer_world_size)
                            server.settimeout(_FD_EXCHANGE_TIMEOUT_S)

                announced = self._exchange.all_gather_object(
                    f"handles-{epoch_key}",
                    (
                        local_length,
                        (
                            exported_payload
                            if self.handle_type == "fabric"
                            else socket_address
                        ),
                    ),
                )
                uid = os.geteuid()
                for offset, length, owner in self.layout.chunks:
                    announced_len, peer_data = announced[owner]
                    if announced_len != length:
                        raise RuntimeError(
                            f"Peer {owner} announced {announced_len} bytes, "
                            f"expected {length}"
                        )
                    if length == 0:
                        continue
                    if owner == self.layout.peer_rank:
                        assert local_handle is not None
                        handle = local_handle
                        if self.handle_type == "posix_fd" and server is not None:
                            assert exported_fd is not None
                            _send_single_fd(
                                server,
                                exported_fd,
                                self.layout.peer_world_size - 1,
                                uid=uid,
                            )
                    elif self.handle_type == "fabric":
                        if not isinstance(peer_data, bytes):
                            raise RuntimeError(
                                f"Peer {owner} sent invalid fabric handle payload"
                            )
                        handle = vmm.import_handle(
                            peer_data, handle_type=self.handle_type
                        )
                        local_handles_created.append(handle)
                    else:
                        if not isinstance(peer_data, str):
                            raise RuntimeError(
                                f"Peer {owner} sent invalid POSIX socket address"
                            )
                        received_fd = _receive_single_fd(peer_data, uid=uid)
                        try:
                            handle = vmm.import_handle(
                                received_fd, handle_type=self.handle_type
                            )
                        finally:
                            os.close(received_fd)
                        local_handles_created.append(handle)
                    vmm.map_handle(self._va + offset, length, handle)
                    mapped_ranges.append((self._va + offset, length))
                vmm.set_access(self._va, self.layout.padded_bytes, self.device_index)
        except BaseException:
            for addr, length in reversed(mapped_ranges):
                vmm.unmap(addr, length)
            for handle in reversed(local_handles_created):
                vmm.release_handle(handle)
            raise
        self._resources.mapped_ranges = mapped_ranges
        self._resources.handles = local_handles_created

    def release_storage(self, *, backup: bool) -> None:
        """Unmap and release physical GPU HBM handles while keeping the VA reserved.

        Only supported for intra-job peer groups where every rank enters
        release together. Cross-process peer groups cannot release per engine
        because other independent processes hold live VMM mappings to this
        peer's physical memory.

        Args:
            backup: When True, copies this peer's owned byte slice to CPU RAM
                in a single allocation before unmapping so :meth:`resume_storage`
                restores it.
        """
        if self._exchange.is_cross_process:
            raise RuntimeError(
                "Cross-process peer GPU memory does not support per-engine "
                "release_storage; restart the peer group with a new rendezvous_id"
            )
        if self._closed:
            raise RuntimeError("Cannot release closed PeerDeviceMemory")
        if self._released:
            return
        torch.cuda.synchronize(self.device_index)
        self._exchange.barrier(f"release-pre-{self._epoch}")
        live = self.layout.local_live_bytes
        if backup and live > 0:
            start = self.layout.local_offset
            # .cpu() on a CUDA view allocates one owning host buffer and copies
            # once without duplicating peak host RAM.
            self._backup = self.tensor[start : start + live].detach().cpu()
            torch.cuda.synchronize(self.device_index)
        else:
            self._backup = None
        vmm = self._resources.vmm
        for address, length in reversed(self._resources.mapped_ranges):
            vmm.unmap(address, length)
        self._resources.mapped_ranges.clear()
        for handle in reversed(self._resources.handles):
            vmm.release_handle(handle)
        self._resources.handles.clear()
        self._released = True
        self._exchange.barrier(f"release-post-{self._epoch}")

    def resume_storage(self, *, backup: bool) -> None:
        """Reallocate and remap peer VMM handles at the exact same virtual address.

        Args:
            backup: When True, restores this peer's CPU backup into its local
                slice before completing the peer barrier.
        """
        if self._exchange.is_cross_process:
            raise RuntimeError(
                "Cross-process peer GPU memory does not support per-engine "
                "resume_storage; restart the peer group with a new rendezvous_id"
            )
        if self._closed:
            raise RuntimeError("Cannot resume closed PeerDeviceMemory")
        if not self._released:
            return
        self._epoch += 1
        self._map_shards(f"epoch-{self._epoch}")
        live = self.layout.local_live_bytes
        if backup and live > 0:
            if self._backup is None:
                raise RuntimeError(
                    "PeerDeviceMemory resume requested backup restore without a backup"
                )
            start = self.layout.local_offset
            self.tensor[start : start + live].copy_(self._backup)
            torch.cuda.synchronize(self.device_index)
        self._backup = None
        self._released = False
        self._exchange.barrier(f"resume-post-{self._epoch}")

    def barrier(self, tag: str) -> None:
        """Synchronize all peers sharing this allocation on ``tag``."""
        if self._closed:
            raise RuntimeError("Cannot barrier on closed PeerDeviceMemory")
        self._exchange.barrier(tag)

    def close(self) -> None:
        """Unmap, free virtual address, and release local physical handles.

        Never blocks on a cross-rank peer barrier during teardown so shutdown
        cannot hang if another peer has already exited or crashed. The CUDA
        driver reference-counts exported physical allocations until all peers
        unmap or exit.
        """
        if self._closed:
            return
        sync_error: Exception | None = None
        if not self._released:
            try:
                torch.cuda.synchronize(self.device_index)
            except Exception as exc:
                sync_error = exc
        try:
            self._backup = None
            self._finalizer.detach()
            _cleanup_vmm(self._resources)
        finally:
            self._closed = True
        if sync_error is not None:
            raise sync_error


def create_peer_device_memory(
    nbytes: int,
    *,
    device_index: int,
    handle_type: str,
    exchange: PeerMemoryExchange,
) -> PeerDeviceMemory:
    """Allocate, exchange, and map peer-sharded GPU HBM memory at one VA.

    Args:
        nbytes: Logical byte size of the contiguous 1D uint8 tensor.
        device_index: CUDA device ordinal mapping the full virtual range.
        handle_type: Explicit shareable VMM handle type (``"fabric"`` or
            ``"posix_fd"``).
        exchange: Peer exchange providing rank, world_size, all_gather_object,
            and barrier (intra-job or cross-process TCPStore).

    Returns:
        A live :class:`PeerDeviceMemory` whose ``tensor`` spans all peers' HBM.
    """
    if handle_type not in _SUPPORTED_HANDLE_TYPES:
        raise ValueError(
            f"handle_type must be one of {sorted(_SUPPORTED_HANDLE_TYPES)}; "
            f"got {handle_type!r}"
        )
    vmm = _require_vmm()
    _ensure_driver_context(device_index)
    local_granularity = vmm.granularity(device_index, handle_type=handle_type)
    granularities = exchange.all_gather_object("granularity", local_granularity)
    granularity = max(int(g) for g in granularities)
    layout = plan_peer_device_layout(
        nbytes, granularity, exchange.rank, exchange.world_size
    )
    va = vmm.reserve(layout.padded_bytes, granularity)
    resources = _VmmResources(
        vmm=vmm,
        va=va,
        padded_bytes=layout.padded_bytes,
        mapped_ranges=[],
        handles=[],
    )
    view_owner = object()
    finalizer: weakref.finalize | None = None
    try:
        tensor = vmm.tensor_view(
            va,
            (nbytes,),
            torch.uint8,
            torch.device("cuda", device_index),
            view_owner,
        )
        finalizer = weakref.finalize(view_owner, _cleanup_vmm, resources)
        finalizer.atexit = False
        memory = PeerDeviceMemory(
            layout=layout,
            device_index=device_index,
            handle_type=handle_type,
            exchange=exchange,
            va=va,
            resources=resources,
            view_owner=view_owner,
            tensor=tensor,
            finalizer=finalizer,
        )
        memory._map_shards("epoch-0")
        return memory
    except BaseException:
        if finalizer is not None:
            finalizer.detach()
        _cleanup_vmm(resources)
        raise


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a standalone TCPStore rendezvous server for peer GPU VMM memory."
    )
    parser.add_argument(
        "--serve",
        required=True,
        help="Bind host:port or [ipv6]:port for the standalone TCPStore.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="TCPStore socket timeout in seconds.",
    )
    args = parser.parse_args()
    store = serve_peer_rendezvous(args.serve, timeout_s=args.timeout)
    stop = False

    def _handle_signal(_signum, _frame) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    logger.info(f"Serving peer GPU VMM TCPStore rendezvous on {args.serve}")
    try:
        while not stop:
            time.sleep(0.5)
    finally:
        del store


if __name__ == "__main__":
    _main()

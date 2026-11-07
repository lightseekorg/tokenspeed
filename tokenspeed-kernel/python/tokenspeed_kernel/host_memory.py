# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
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
#
# Adapted from vLLM's Engram host tables
# (vllm/models/deepseek_v41/nvidia/ops/engram_host_table.py, Apache-2.0,
# Copyright contributors to the vLLM project). Changed here: a standalone
# module, chunks placed on caller-chosen NUMA nodes, a per-node reserve check
# before creation as well as after each chunk, and optional export.

"""Pinned host memory that GPUs map with 2 MiB pages, shareable within a host.

GPU kernels that gather random rows of a large host table are bound by address
translation rather than by bytes. A ``cudaHostRegister``ed mapping is
translated with the pages that back it, a tmpfs file's 4-64 KiB base pages, so
a 100 GB table spans at least 1.5 M GPU pages and almost every random row walks
the page table. CUDA virtual memory management backs host memory with 2 MiB
pages instead: ``cuMemCreate`` places pinned memory on a host NUMA node, and
one virtual address range maps it for the CPU and for this GPU. FABRIC handles
let the other processes of the host map the same memory.

Memory is created in chunks, each on a NUMA node the caller chooses, and the
creator stops before a node drops below a reserve. The CPU tensor's storage
owns the mapping: any view keeps it alive, and the last one unmaps it.

NVIDIA only (``cuda.bindings``). ``platform_supported`` says whether this
platform has such memory at all and ``probe_host_memory`` why a GPU cannot get
it; callers keep their other kind of host memory.
"""

from __future__ import annotations

import ctypes
import functools
import logging
import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from tokenspeed_kernel.platform import current_platform

__all__ = [
    "ExportedHostMemory",
    "HostMemory",
    "NodeMemory",
    "create_host_memory",
    "host_numa_node",
    "import_host_memory",
    "node_memory",
    "platform_supported",
    "probe_host_memory",
]

logger = logging.getLogger(__name__)

_NODE_ROOT = Path("/sys/devices/system/node")
_GIB = 1 << 30


@dataclass(frozen=True)
class NodeMemory:
    """One NUMA node's memory in bytes, from its sysfs meminfo."""

    total: int
    free: int
    # Active(file) + Inactive(file): clean page cache is reclaimed for pinned
    # allocations, so it counts as available.
    page_cache: int
    # KReclaimable, or SReclaimable on kernels without it.
    reclaimable: int

    @property
    def available(self) -> int:
        return self.free + self.page_cache + self.reclaimable

    def describe(self) -> str:
        return (
            f"{self.available / _GIB:.1f} GiB available = {self.free / _GIB:.1f} "
            f"free + {self.page_cache / _GIB:.1f} page cache + "
            f"{self.reclaimable / _GIB:.1f} reclaimable, of {self.total / _GIB:.1f}"
        )


def node_memory(numa_node: int) -> NodeMemory:
    """Read ``/sys/devices/system/node/node<N>/meminfo``.

    MemFree alone undercounts: creating pinned memory reclaims clean page cache
    (such as checkpoint files just read), so free memory sits near the
    watermark while much more is available.
    """
    fields = {}
    meminfo = _NODE_ROOT / f"node{numa_node}" / "meminfo"
    for line in meminfo.read_text().splitlines():
        # "Node 0 MemFree:   123456 kB"
        parts = line.split()
        if len(parts) == 5 and parts[4] == "kB":
            fields[parts[2].rstrip(":")] = int(parts[3]) * 1024
    return NodeMemory(
        total=fields["MemTotal"],
        free=fields["MemFree"],
        page_cache=fields["Active(file)"] + fields["Inactive(file)"],
        reclaimable=fields.get("KReclaimable", fields.get("SReclaimable", 0)),
    )


@dataclass(frozen=True)
class ExportedHostMemory:
    """What another process of the host needs to map a ``HostMemory``.

    ``chunks`` holds ``(offset, length, numa_node)`` per chunk and ``handles``
    the chunks' FABRIC handles, in the same order.
    """

    nbytes: int
    granularity: int
    chunks: tuple[tuple[int, int, int], ...]
    handles: tuple[bytes, ...]


@dataclass(frozen=True, eq=False)
class HostMemory:
    """Pinned host memory mapped at one address for the CPU and one GPU.

    ``tensor`` is a CPU uint8 ``[nbytes]`` view whose ``data_ptr()`` GPU
    kernels may dereference. Its storage owns the mapping: the mapping lasts
    while any view of ``tensor`` lives, and the last view to go unmaps it and
    releases the chunks. ``handles`` are FABRIC handles when the memory was
    created exportable, else empty.
    """

    tensor: torch.Tensor
    granularity: int
    chunks: tuple[tuple[int, int, int], ...]
    handles: tuple[bytes, ...]
    _finalizer: weakref.finalize

    @property
    def numa_nodes(self) -> tuple[int, ...]:
        return tuple(sorted({node for _, _, node in self.chunks}))

    def export(self) -> ExportedHostMemory:
        """Describe the memory for ``import_host_memory`` in another process."""
        if not self.handles:
            raise ValueError("Host memory was not created exportable")
        return ExportedHostMemory(
            self.tensor.numel(), self.granularity, self.chunks, self.handles
        )

    def close(self) -> None:
        """Unmap now, before the views go away.

        Only for memory no view of which escaped, such as a setup that failed:
        a surviving view would point at unmapped memory.
        """
        self._finalizer()


class _CudaDriver:
    """The CUDA virtual memory management calls this module makes."""

    def __init__(self) -> None:
        from cuda.bindings import driver

        self._cu = driver
        self._call(driver.cuInit, 0)

    def _call(self, fn, *args):
        error, *result = fn(*args)
        if error != self._cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"{fn.__name__} failed: {error.name}")
        return result[0] if result else None

    def ensure_context(self, device_index: int) -> None:
        """Make a context current if none is: the VMM calls need one."""
        cu = self._cu
        if int(self._call(cu.cuCtxGetCurrent)):
            return
        # Torch's runtime uses the same primary context, so it stays retained.
        context = self._call(cu.cuDevicePrimaryCtxRetain, device_index)
        self._call(cu.cuCtxSetCurrent, context)

    def host_numa_node(self, device_index: int) -> int | None:
        cu = self._cu
        node = self._call(
            cu.cuDeviceGetAttribute,
            cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID,
            device_index,
        )
        return node if node >= 0 else None

    def _properties(self, numa_node: int, exportable: bool):
        cu = self._cu
        prop = cu.CUmemAllocationProp()
        prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
        prop.location.id = numa_node
        if exportable:
            prop.requestedHandleTypes = (
                cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
            )
        return prop

    def granularity(self, numa_node: int, exportable: bool) -> int:
        cu = self._cu
        return int(
            self._call(
                cu.cuMemGetAllocationGranularity,
                self._properties(numa_node, exportable),
                cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            )
        )

    def create(self, nbytes: int, numa_node: int, exportable: bool):
        return self._call(
            self._cu.cuMemCreate, nbytes, self._properties(numa_node, exportable), 0
        )

    def export(self, handle) -> bytes:
        cu = self._cu
        shareable = self._call(
            cu.cuMemExportToShareableHandle,
            handle,
            cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC,
            0,
        )
        return bytes(shareable.data)

    def import_handle(self, data: bytes):
        cu = self._cu
        return self._call(
            cu.cuMemImportFromShareableHandle,
            data,
            cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC,
        )

    def release(self, handle) -> None:
        self._call(self._cu.cuMemRelease, handle)

    def reserve(self, nbytes: int, alignment: int) -> int:
        return int(self._call(self._cu.cuMemAddressReserve, nbytes, alignment, 0, 0))

    def free(self, address: int, nbytes: int) -> None:
        self._call(self._cu.cuMemAddressFree, address, nbytes)

    def map(self, address: int, nbytes: int, handle) -> None:
        self._call(self._cu.cuMemMap, address, nbytes, 0, handle, 0)

    def unmap(self, address: int, nbytes: int) -> None:
        self._call(self._cu.cuMemUnmap, address, nbytes)

    def set_access(
        self, address: int, nbytes: int, device_index: int, numa_node: int
    ) -> None:
        cu = self._cu
        descriptors = []
        for location, location_id in (
            (cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE, device_index),
            (cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA, numa_node),
        ):
            descriptor = cu.CUmemAccessDesc()
            descriptor.location.type = location
            descriptor.location.id = location_id
            descriptor.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            descriptors.append(descriptor)
        self._call(cu.cuMemSetAccess, address, nbytes, descriptors, len(descriptors))


def platform_supported() -> bool:
    """Whether this platform has host-NUMA cuMem memory at all.

    Unlike ``probe_host_memory``, this asks no driver, so the ranks of a job,
    which run on one platform, agree on it without a collective.

    Returns:
        True on NVIDIA, else False.
    """
    return current_platform().is_nvidia


@functools.cache
def _driver() -> _CudaDriver | None:
    if not platform_supported():
        return None
    try:
        return _CudaDriver()
    except Exception as exc:
        # cuda.bindings is missing, or the driver cannot initialize.
        logger.info(f"CUDA driver unavailable for host memory: {exc!s}")
        return None


def _require_driver() -> _CudaDriver:
    driver = _driver()
    if driver is None:
        raise RuntimeError("host-NUMA cuMem memory needs an NVIDIA GPU and its driver")
    return driver


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def host_numa_node(device_index: int) -> int | None:
    """The host NUMA node of a GPU (``CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID``).

    Returns:
        The node id, or None off NVIDIA or when the GPU reports none.
    """
    driver = _driver()
    return None if driver is None else driver.host_numa_node(device_index)


@functools.cache
def probe_host_memory(device_index: int, exportable: bool) -> str | None:
    """Say why host memory cannot be created for a GPU, or None if it can.

    Creates one granule on the GPU's host NUMA node, exports its FABRIC handle
    when ``exportable`` (that needs the IMEX channel the importers share as
    well), and releases it. Cached per arguments.

    Args:
        device_index: CUDA device that would map the memory.
        exportable: Whether other processes would import the memory.

    Returns:
        None when the probe allocation succeeded, else the reason it failed.
    """
    driver = _driver()
    if driver is None:
        return "host-NUMA cuMem memory needs an NVIDIA GPU and its driver"
    try:
        driver.ensure_context(device_index)
        node = driver.host_numa_node(device_index)
        if node is None:
            return f"CUDA device {device_index} reports no host NUMA node"
        handle = driver.create(driver.granularity(node, exportable), node, exportable)
        try:
            if exportable:
                driver.export(handle)
        finally:
            driver.release(handle)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def create_host_memory(
    nbytes: int,
    numa_nodes: Sequence[int],
    device_index: int,
    exportable: bool,
    chunk_bytes: int,
    reserve_fraction: float,
) -> HostMemory:
    """Create pinned host memory for a GPU, in chunks placed on NUMA nodes.

    Args:
        nbytes: Size of the returned tensor. The mapping is rounded up to the
            allocation granularity (2 MiB for host NUMA memory).
        numa_nodes: Host NUMA nodes; chunk ``i`` goes to
            ``numa_nodes[i % len(numa_nodes)]``.
        device_index: CUDA device that maps the memory for reads and writes.
        exportable: Request FABRIC handles so ``import_host_memory`` can map
            the memory in other processes of this host.
        chunk_bytes: Chunk size, rounded up to the allocation granularity.
        reserve_fraction: Fraction of each node's MemTotal that must remain
            available. Creation stops with ``MemoryError`` when a node would
            fall below it, before creating anything or after any chunk.

    Returns:
        The mapped memory. Nothing stays allocated when this raises.

    Raises:
        MemoryError: A node would fall below its reserve. The message holds
            the node's measured memory.
        RuntimeError: No NVIDIA driver, or a driver call failed.
    """
    if nbytes < 1 or chunk_bytes < 1 or not numa_nodes:
        raise ValueError("Host memory needs a size, a chunk size and NUMA nodes")
    if not 0.0 <= reserve_fraction < 1.0:
        raise ValueError("The NUMA reserve must be a fraction below 1")
    driver = _require_driver()
    driver.ensure_context(device_index)
    granularity = max(
        driver.granularity(node, exportable) for node in sorted(set(numa_nodes))
    )
    padded = _round_up(nbytes, granularity)
    step = _round_up(chunk_bytes, granularity)
    chunks = tuple(
        (offset, min(step, padded - offset), numa_nodes[index % len(numa_nodes)])
        for index, offset in enumerate(range(0, padded, step))
    )
    planned: dict[int, int] = {}
    for _, length, node in chunks:
        planned[node] = planned.get(node, 0) + length
    reserves = {}
    for node, node_bytes in planned.items():
        memory = node_memory(node)
        reserves[node] = int(memory.total * reserve_fraction)
        if memory.available - node_bytes < reserves[node]:
            raise MemoryError(
                f"NUMA node {node}: {node_bytes / _GIB:.1f} GiB would leave less "
                f"than its {reserves[node] / _GIB:.1f} GiB reserve "
                f"({memory.describe()})"
            )
    handles = []
    try:
        created = 0
        for _, length, node in chunks:
            handles.append(driver.create(length, node, exportable))
            created += length
            memory = node_memory(node)
            if memory.available < reserves[node]:
                raise MemoryError(
                    f"NUMA node {node} fell below its "
                    f"{reserves[node] / _GIB:.1f} GiB reserve after "
                    f"{created / _GIB:.1f} of {padded / _GIB:.1f} GiB "
                    f"({memory.describe()})"
                )
        exported = (
            tuple(driver.export(handle) for handle in handles) if exportable else ()
        )
        return _map(
            driver, nbytes, granularity, chunks, handles, exported, device_index
        )
    except BaseException:
        for handle in handles:
            driver.release(handle)
        raise


def import_host_memory(exported: ExportedHostMemory, device_index: int) -> HostMemory:
    """Map memory another process of this host created exportable.

    Args:
        exported: ``HostMemory.export()`` of the creating process.
        device_index: CUDA device that maps the memory for reads and writes.

    Returns:
        The mapped memory, without FABRIC handles of its own.

    Raises:
        RuntimeError: No NVIDIA driver, or a driver call failed, for example
            when this process lacks the creator's IMEX channel.
    """
    driver = _require_driver()
    driver.ensure_context(device_index)
    handles = []
    try:
        for data in exported.handles:
            handles.append(driver.import_handle(data))
        return _map(
            driver,
            exported.nbytes,
            exported.granularity,
            exported.chunks,
            handles,
            (),
            device_index,
        )
    except BaseException:
        for handle in handles:
            driver.release(handle)
        raise


def _map(
    driver,
    nbytes: int,
    granularity: int,
    chunks: tuple[tuple[int, int, int], ...],
    handles: list,
    exported: tuple[bytes, ...],
    device_index: int,
) -> HostMemory:
    """Map chunks at one reserved range; the returned memory owns the handles."""
    padded = sum(length for _, length, _ in chunks)
    address = driver.reserve(padded, granularity)
    mapped: list[tuple[int, int]] = []
    try:
        for (offset, length, node), handle in zip(chunks, handles, strict=True):
            driver.map(address + offset, length, handle)
            mapped.append((offset, length))
            driver.set_access(address + offset, length, device_index, node)
    except BaseException:
        _unmap(driver, address, padded, tuple(mapped), ())
        raise
    buffer = (ctypes.c_uint8 * nbytes).from_address(address)
    # torch.frombuffer's storage retains the buffer, so the mapping goes away
    # with the last view rather than with any owner object. Not at exit:
    # CUDA may already be torn down by then.
    finalizer = weakref.finalize(
        buffer, _unmap, driver, address, padded, tuple(mapped), tuple(handles)
    )
    finalizer.atexit = False
    return HostMemory(
        torch.frombuffer(buffer, dtype=torch.uint8),
        granularity,
        chunks,
        exported,
        finalizer,
    )


def _unmap(
    driver,
    address: int,
    padded: int,
    mapped: tuple[tuple[int, int], ...],
    handles: tuple,
) -> None:
    try:
        for offset, length in mapped:
            driver.unmap(address + offset, length)
        driver.free(address, padded)
    finally:
        for handle in handles:
            driver.release(handle)

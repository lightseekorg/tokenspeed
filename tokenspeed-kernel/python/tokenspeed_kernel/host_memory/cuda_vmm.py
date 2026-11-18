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

"""NVIDIA host-NUMA VMM allocations with exportable FABRIC handles."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import torch
from tokenspeed_kernel.host_memory import HostMemoryViews, host_memory_views


def _driver():
    if torch.version.cuda is None:
        raise RuntimeError("Host-NUMA VMM requires NVIDIA CUDA")
    from cuda.bindings import driver

    return driver


def _call(function, *args):
    status, *values = function(*args)
    if int(status) != 0:
        raise RuntimeError(f"{function.__name__} failed: {status!s}")
    return values[0] if values else None


def registered_pointer(address: int) -> int:
    """Resolve the CUDA alias of already registered host memory."""
    return int(_call(_driver().cuMemHostGetDevicePointer, address, 0))


def device_host_numa_node(device: int) -> int:
    """Return the host NUMA node exposed by the CUDA device topology."""
    cu = _driver()
    return int(
        _call(
            cu.cuDeviceGetAttribute,
            cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID,
            device,
        )
    )


def _properties(device: int):
    cu = _driver()
    node = device_host_numa_node(device)
    if node < 0:
        raise RuntimeError("CUDA device does not expose a host NUMA node")
    prop = cu.CUmemAllocationProp()
    prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
    prop.location.id = node
    prop.requestedHandleTypes = cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
    granularity = int(_call(cu.cuMemGetAllocationGranularity, prop, 0))
    if granularity < 2 << 20:
        raise RuntimeError(
            f"Host-NUMA allocation granularity is only {granularity} bytes"
        )
    return prop, granularity


def _export(handle) -> bytes:
    cu = _driver()
    value = _call(
        cu.cuMemExportToShareableHandle,
        handle,
        cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC,
        0,
    )
    return bytes(value.data)


@cache
def _probe(device: int) -> None:
    cu = _driver()
    prop, granularity = _properties(device)
    handle = _call(cu.cuMemCreate, granularity, prop, 0)
    try:
        _export(handle)
    finally:
        _call(cu.cuMemRelease, handle)


def _check_node_reserve(node: int, next_bytes: int) -> None:
    values = {}
    for line in (
        Path(f"/sys/devices/system/node/node{node}/meminfo").read_text().splitlines()
    ):
        key, value = line.split(":", 1)
        values[key.split()[-1]] = int(value.split()[0]) * 1024
    available = (
        values["MemFree"]
        + values.get("FilePages", 0)
        + values.get("SReclaimable", 0)
        - values.get("Shmem", 0)
    )
    if available - next_bytes < values["MemTotal"] // 20:
        raise MemoryError(f"Host-NUMA node {node} would fall below its 5% reserve")


@dataclass(frozen=True)
class HostAllocationDescriptor:
    nbytes: int
    numa_node: int
    granularity: int
    lengths: tuple[int, ...]
    handles: tuple[bytes, ...]


class _Mapping:
    def __init__(self, address: int, nbytes: int):
        self.address = address
        self.nbytes = nbytes
        self.ranges: list[tuple[int, int]] = []
        self.closed = False
        self.device = torch.cuda.current_device()

    def close(self) -> None:
        if self.closed:
            return
        cu = _driver()
        with torch.cuda.device(self.device):
            torch.cuda.synchronize(self.device)
            for offset, length in reversed(self.ranges):
                _call(cu.cuMemUnmap, self.address + offset, length)
            _call(cu.cuMemAddressFree, self.address, self.nbytes)
        self.closed = True


class CudaHostAllocation:
    def __init__(self, descriptor: HostAllocationDescriptor, handles: list):
        self.descriptor = descriptor
        self.handles = handles

    @classmethod
    def create(cls, nbytes: int) -> CudaHostAllocation:
        if nbytes <= 0:
            raise ValueError("Host-NUMA allocation size must be positive")
        cu = _driver()
        device = torch.cuda.current_device()
        torch.cuda.set_device(device)
        _probe(device)
        prop, granularity = _properties(device)
        padded = (nbytes + granularity - 1) // granularity * granularity
        chunk = max(2 << 30, granularity)
        chunk = chunk // granularity * granularity
        lengths = tuple(
            min(chunk, padded - offset) for offset in range(0, padded, chunk)
        )
        handles = []
        try:
            for length in lengths:
                _check_node_reserve(prop.location.id, length)
                handles.append(_call(cu.cuMemCreate, length, prop, 0))
            descriptor = HostAllocationDescriptor(
                nbytes,
                prop.location.id,
                granularity,
                lengths,
                tuple(map(_export, handles)),
            )
            return cls(descriptor, handles)
        except BaseException:
            for handle in handles:
                _call(cu.cuMemRelease, handle)
            raise

    @classmethod
    def import_handles(cls, descriptor: HostAllocationDescriptor) -> CudaHostAllocation:
        cu = _driver()
        torch.cuda.set_device(torch.cuda.current_device())
        handles = []
        try:
            for value in descriptor.handles:
                handles.append(
                    _call(
                        cu.cuMemImportFromShareableHandle,
                        value,
                        cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC,
                    )
                )
            return cls(descriptor, handles)
        except BaseException:
            for handle in handles:
                _call(cu.cuMemRelease, handle)
            raise

    def export(self) -> HostAllocationDescriptor:
        return self.descriptor

    def close(self) -> None:
        cu = _driver()
        while self.handles:
            _call(cu.cuMemRelease, self.handles.pop())

    def map(self) -> HostMemoryViews:
        """Map locally; retain handles until every peer has completed import."""
        cu = _driver()
        torch.cuda.set_device(torch.cuda.current_device())
        desc = self.descriptor
        size = sum(desc.lengths)
        mapping = None
        try:
            address = int(_call(cu.cuMemAddressReserve, size, 0, 0, 0))
            mapping = _Mapping(address, size)
            offset = 0
            for length, handle in zip(desc.lengths, self.handles, strict=True):
                _call(cu.cuMemMap, address + offset, length, 0, handle, 0)
                mapping.ranges.append((offset, length))
                offset += length
            access = []
            for kind, index in (
                (
                    cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE,
                    torch.cuda.current_device(),
                ),
                (cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA, desc.numa_node),
            ):
                item = cu.CUmemAccessDesc()
                item.location.type, item.location.id = kind, index
                item.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
                access.append(item)
            _call(cu.cuMemSetAccess, address, size, access, len(access))
            return host_memory_views(
                address, address, desc.nbytes, release=mapping.close
            )
        except BaseException:
            if mapping is not None:
                mapping.close()
            raise

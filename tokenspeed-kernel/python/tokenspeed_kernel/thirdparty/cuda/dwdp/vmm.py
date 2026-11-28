# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CUDA VMM allocations and zero-copy views for immutable expert storage.

Owned storage is exported through an explicit shareable handle type. Fabric
handles carry an opaque payload and need an IMEX channel; POSIX file
descriptors are process-local, so the caller passes them between processes on
one node (for example with ``SCM_RIGHTS``).
"""

from __future__ import annotations

import logging
import math

import torch
from cuda.bindings import driver as cu
from tokenspeed_kernel import host_memory

logger = logging.getLogger(__name__)


def checked(result):
    error, *values = result
    if error != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA VMM operation failed: {error}")
    return values[0] if len(values) == 1 else values


HANDLE_TYPES = {
    "fabric": cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC,
    "posix_fd": cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
}


def _handle_type(name: str):
    if name not in HANDLE_TYPES:
        raise ValueError(f"Unsupported shareable handle type: {name}")
    return HANDLE_TYPES[name]


def device_location(device: int):
    location = cu.CUmemLocation()
    location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    location.id = device
    return location


def allocation_prop(device: int, *, handle_type: str | None):
    """Pinned device memory; ``handle_type=None`` allocates unexported memory."""
    prop = cu.CUmemAllocationProp()
    prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location = device_location(device)
    prop.requestedHandleTypes = (
        _handle_type(handle_type)
        if handle_type is not None
        else cu.CUmemAllocationHandleType(0)
    )
    return prop


def fabric_allocation_properties(device: int, *, host: bool):
    """Return FABRIC allocation properties and the minimum mapping granularity in bytes."""
    prop = allocation_prop(device, handle_type="fabric")
    if host:
        node = host_memory.host_numa_node(device)
        if node is None:
            raise RuntimeError(f"CUDA device {device} has no host NUMA node")
        prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
        prop.location.id = node
    minimum = checked(
        cu.cuMemGetAllocationGranularity(
            prop, cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
        )
    )
    return prop, int(minimum)


def allocate_from_properties(size: int, prop: cu.CUmemAllocationProp) -> int:
    """Allocate size bytes with the given placement and export type."""
    return int(checked(cu.cuMemCreate(size, prop, 0)))


def granularity(device: int, *, handle_type: str) -> int:
    return int(
        checked(
            cu.cuMemGetAllocationGranularity(
                allocation_prop(device, handle_type=handle_type),
                cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
            )
        )
    )


def allocate(size: int, device: int, *, handle_type: str | None) -> int:
    prop = allocation_prop(device, handle_type=handle_type)
    result = cu.cuMemCreate(size, prop, 0)
    if result[0] == cu.CUresult.CUDA_ERROR_OUT_OF_MEMORY:
        # Driver allocations cannot reclaim PyTorch's freed preprocessing buffers.
        before = torch.cuda.mem_get_info(device)[0]
        torch.cuda.empty_cache()
        after = torch.cuda.mem_get_info(device)[0]
        logger.info(
            f"Peer-weight allocation retry: bytes={size}, "
            f"driver_free_before={before}, driver_free_after={after}"
        )
        result = cu.cuMemCreate(size, prop, 0)
    return int(checked(result))


def export(handle: int, *, handle_type: str) -> bytes | int:
    """Fabric returns the opaque payload; POSIX returns a new local descriptor."""
    value = checked(
        cu.cuMemExportToShareableHandle(handle, _handle_type(handle_type), 0)
    )
    return bytes(value.data) if handle_type == "fabric" else int(value)


def import_handle(payload: bytes | int, *, handle_type: str) -> int:
    """Import a fabric payload or a descriptor already received by this process.

    The caller still owns, and closes, a received descriptor.
    """
    return int(
        checked(cu.cuMemImportFromShareableHandle(payload, _handle_type(handle_type)))
    )


def reserve(size: int, alignment: int) -> int:
    return int(checked(cu.cuMemAddressReserve(size, alignment, 0, 0)))


def map_handle(address: int, size: int, handle: int) -> None:
    checked(cu.cuMemMap(address, size, 0, handle, 0))


def set_access(address: int, size: int, device: int) -> None:
    access = cu.CUmemAccessDesc()
    access.location = device_location(device)
    access.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    checked(cu.cuMemSetAccess(address, size, [access], 1))


class _ByteView:
    def __init__(self, address: int, size: int, owner: object) -> None:
        self.address = address
        self.size = size
        self.owner = owner

    @property
    def __cuda_array_interface__(self):
        return {
            "shape": (self.size,),
            "strides": None,
            "typestr": "|u1",
            "data": (self.address, False),
            "version": 3,
        }


def tensor_view(
    address: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    owner: object,
) -> torch.Tensor:
    size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    raw = torch.as_tensor(_ByteView(address, size, owner), device=device)
    if raw.data_ptr() != address:
        raise RuntimeError("CUDA pointer wrapping unexpectedly copied weight storage")
    return raw.view(dtype).reshape(shape)


def copy(destination: int, source: int, size: int, stream: torch.cuda.Stream) -> None:
    if size:
        checked(cu.cuMemcpyDtoDAsync(destination, source, size, stream.cuda_stream))


def unmap(address: int, size: int) -> None:
    checked(cu.cuMemUnmap(address, size))


def free_address(address: int, size: int) -> None:
    checked(cu.cuMemAddressFree(address, size))


def release_handle(handle: int) -> None:
    checked(cu.cuMemRelease(handle))

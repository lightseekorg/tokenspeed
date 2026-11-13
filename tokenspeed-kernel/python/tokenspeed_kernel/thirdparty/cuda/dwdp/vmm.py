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

"""CUDA fabric allocations and zero-copy views for immutable expert storage."""

from __future__ import annotations

import logging
import math

import torch
from cuda.bindings import driver as cu

logger = logging.getLogger(__name__)


def checked(result):
    error, *values = result
    if error != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA VMM operation failed: {error}")
    return values[0] if len(values) == 1 else values


def allocation_prop(device: int, *, fabric: bool):
    location = cu.CUmemLocation()
    location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    location.id = device
    prop = cu.CUmemAllocationProp()
    prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location = location
    prop.requestedHandleTypes = (
        cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
        if fabric
        else cu.CUmemAllocationHandleType(0)
    )
    return prop


def granularity(device: int) -> int:
    return int(
        checked(
            cu.cuMemGetAllocationGranularity(
                allocation_prop(device, fabric=True),
                cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
            )
        )
    )


def allocate(size: int, device: int, *, fabric: bool) -> int:
    prop = allocation_prop(device, fabric=fabric)
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


def export(handle: int) -> bytes:
    value = checked(
        cu.cuMemExportToShareableHandle(
            handle,
            cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC,
            0,
        )
    )
    return bytes(value.data)


def import_handle(payload: bytes) -> int:
    return int(
        checked(
            cu.cuMemImportFromShareableHandle(
                payload,
                cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC,
            )
        )
    )


def reserve(size: int, alignment: int) -> int:
    return int(checked(cu.cuMemAddressReserve(size, alignment, 0, 0)))


def map_handle(address: int, size: int, handle: int) -> None:
    checked(cu.cuMemMap(address, size, 0, handle, 0))


def set_access(address: int, size: int, device: int) -> None:
    access = cu.CUmemAccessDesc()
    access.location = allocation_prop(device, fabric=True).location
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

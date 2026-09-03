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

"""Readiness probe for CUDA fabric-handle (MNNVL) memory.

Multi-node NVLink buffers need the full IMEX stack (nvidia-imex daemon plus
/dev/nvidia-caps-imex-channels), which only exists on NVLink-domain machines
such as GB200 NVL72. Machine-type, device-attribute and NVLS-multicast checks
are all insufficient: drivers report fabric-handle *support* and multicast
support on hosts where the IMEX stack is absent and the allocation itself fails
with CUDA_ERROR_NOT_PERMITTED. Every MNNVL consumer therefore has to probe the
allocation, so the probe lives here instead of being duplicated per consumer.
"""

from __future__ import annotations

import ctypes
import logging
from collections.abc import Sequence

import torch
from tokenspeed_kernel.platform import current_platform

__all__ = [
    "fabric_allocation_supported",
    "gather_fabric_map",
    "group_has_fabric",
]

logger = logging.getLogger(__name__)

# CUDA driver ABI constants used by the fabric probe (stable since CUDA 12).
_CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED = 128
_CU_MEM_ALLOCATION_TYPE_PINNED = 1
_CU_MEM_HANDLE_TYPE_FABRIC = 8
_CU_MEM_LOCATION_TYPE_DEVICE = 1
_CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0

_probe_cache: dict[int, bool] = {}
_fabric_map: list[bool] | None = None


class _CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _CUmemAllocationPropAllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class _CUmemAllocationProp(ctypes.Structure):
    # Mirrors CUmemAllocationProp_st from cuda.h.
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", _CUmemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _CUmemAllocationPropAllocFlags),
    ]


def _probe_fabric_allocation(device_index: int) -> bool:
    """Run one granularity-sized fabric-handle ``cuMemCreate`` and release it."""
    for lib_name in ("libcuda.so.1", "libcuda.so"):
        try:
            libcuda = ctypes.CDLL(lib_name)
            break
        except OSError:
            continue
    else:
        return False

    try:
        if libcuda.cuInit(0) != 0:
            return False
        device = ctypes.c_int()
        if libcuda.cuDeviceGet(ctypes.byref(device), device_index) != 0:
            return False
        supported = ctypes.c_int(0)
        if (
            libcuda.cuDeviceGetAttribute(
                ctypes.byref(supported),
                _CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
                device,
            )
            != 0
            or supported.value == 0
        ):
            return False

        # VMM calls need a current context; reuse the caller's (torch usually
        # made one), else retain the device primary context and restore after.
        prev_ctx = ctypes.c_void_p()
        if libcuda.cuCtxGetCurrent(ctypes.byref(prev_ctx)) != 0:
            return False
        retained = False
        if not prev_ctx.value:
            primary_ctx = ctypes.c_void_p()
            if libcuda.cuDevicePrimaryCtxRetain(ctypes.byref(primary_ctx), device) != 0:
                return False
            retained = True
            libcuda.cuCtxSetCurrent(primary_ctx)
        try:
            prop = _CUmemAllocationProp()
            prop.type = _CU_MEM_ALLOCATION_TYPE_PINNED
            prop.requestedHandleTypes = _CU_MEM_HANDLE_TYPE_FABRIC
            prop.location.type = _CU_MEM_LOCATION_TYPE_DEVICE
            prop.location.id = device.value

            granularity = ctypes.c_size_t(0)
            if (
                libcuda.cuMemGetAllocationGranularity(
                    ctypes.byref(granularity),
                    ctypes.byref(prop),
                    _CU_MEM_ALLOC_GRANULARITY_MINIMUM,
                )
                != 0
            ):
                return False

            handle = ctypes.c_ulonglong(0)
            if (
                libcuda.cuMemCreate(
                    ctypes.byref(handle),
                    ctypes.c_size_t(granularity.value),
                    ctypes.byref(prop),
                    ctypes.c_ulonglong(0),
                )
                != 0
            ):
                return False
            libcuda.cuMemRelease(handle)
            return True
        finally:
            if retained:
                libcuda.cuCtxSetCurrent(prev_ctx)
                libcuda.cuDevicePrimaryCtxRelease(device)
    except (OSError, AttributeError):
        return False


def fabric_allocation_supported(device_index: int) -> bool:
    """Return True iff the driver can serve fabric-handle memory on the device.

    Args:
        device_index: CUDA device ordinal to probe.

    Returns:
        Whether a fabric-handle allocation succeeds, i.e. whether MNNVL buffers
        can be created from this process. The result is cached per device
        because callers must all reach the same decision, and because the probe
        touches the driver.
    """
    cached = _probe_cache.get(device_index)
    if cached is None:
        cached = _probe_fabric_allocation(device_index)
        _probe_cache[device_index] = cached
        logger.info("fabric allocation on device %s: %s", device_index, cached)
    return cached


def gather_fabric_map() -> list[bool]:
    """Gather and cache every world rank's fabric-allocation verdict.

    Fabric handles are an NVIDIA concept, so off NVIDIA the answer is no for
    every rank and is filled in without a collective. Deciding that here rather
    than at the call site keeps the collective out of a lazy path: a caller
    that skipped this on the wrong platform would otherwise trigger the gather
    from a gate, which is the dispatch-time collective this map exists to
    remove.

    That branch is the one thing here the ranks do not agree on by
    construction. ``is_nvidia`` is detected locally, so a job mixing CUDA and
    non-CUDA ranks would have some enter the all_gather and others return, and
    the ones that entered would wait out the NCCL timeout. The caller it
    replaced read a server argument, which was uniform; this reads the machine.
    """
    global _fabric_map

    if _fabric_map is not None:
        return _fabric_map

    if not current_platform().is_nvidia:
        _fabric_map = [False] * torch.distributed.get_world_size()
        return _fabric_map

    device = torch.device("cuda", torch.cuda.current_device())
    local = torch.tensor(
        [fabric_allocation_supported(device.index)], dtype=torch.bool, device=device
    )
    gathered = [
        torch.empty_like(local) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(gathered, local, group=torch.distributed.group.WORLD)
    _fabric_map = [bool(value.item()) for value in gathered]
    logger.info(
        "fabric allocation available on %s/%s ranks",
        sum(_fabric_map),
        len(_fabric_map),
    )
    return _fabric_map


def group_has_fabric(ranks: Sequence[int]) -> bool:
    """Return whether every rank in ``ranks`` reported fabric support.

    Raises if the map has not been gathered, rather than gathering it here.
    This is asked from dispatch, where the ranks present are the group's and
    not the world's, so a lazy gather would run a world collective from a
    stage or a data-parallel subset and hang the ranks that never arrive. A
    missing map means the initialization hook did not run, which is a wiring
    bug; callers outside a server must gather it themselves.
    """
    if _fabric_map is None:
        raise RuntimeError(
            "fabric map was never gathered; call gather_fabric_map() at "
            "distributed initialization before any reachability gate"
        )
    return all(_fabric_map[rank] for rank in ranks)

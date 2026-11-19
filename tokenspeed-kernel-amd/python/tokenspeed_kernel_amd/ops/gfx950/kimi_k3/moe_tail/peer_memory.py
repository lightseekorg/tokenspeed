# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright (C) 2026, Advanced Micro Devices, Inc.
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

"""Uncached memory every rank of a tensor-parallel group can read and write.

The MoE tail's ranks reduce across the group inside the kernel, through each
other's buffers. Each rank allocates one uncached buffer through this
process's HIP runtime and maps every peer's buffer through HIP IPC handles.

The memory is uncached (``hipDeviceMallocUncached``), so a consumer reads each
handoff at its home memory, never through a stale cache line. The Iris
symmetric heap allocates fine-grained memory instead, so the tail does not
use it.

Modified from ROCm/ATOM (https://github.com/ROCm/ATOM, MIT License) at
97359d6df46c59736468168b7be8bc50b482a1a9:
``atom/mono/runtime/peer_memory.py``.
"""

from __future__ import annotations

import ctypes
import functools

import torch
import torch.distributed as dist

_IPC_HANDLE_BYTES = 64
_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 0x1
_DEVICE_MALLOC_UNCACHED = 0x3
# a peer maps an IPC handle in 2 MiB granules, so a partial last granule faults there
_GRANULE = 2 << 20
# closed buffers and their peer mappings, kept until the process exits: on MI355X, pages
# a kernel had used uncached, once freed and reused by ordinary allocations, gave later
# kernels stale cache lines (64-byte runs of zeros or older values)
_RETIRED: list = []


class PeerMemoryError(RuntimeError):
    """Building a peer buffer failed on this or another rank."""


class _IpcHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_byte * _IPC_HANDLE_BYTES)]


@functools.cache
def hip_runtime():
    """The HIP runtime this process loaded (torch's), not a second copy."""
    path = None
    with open("/proc/self/maps") as maps:
        for line in maps:
            if "libamdhip64" in line:
                path = line.split()[-1]
                break
    lib = ctypes.CDLL(path or "libamdhip64.so", mode=ctypes.RTLD_GLOBAL)
    sig = {
        "hipExtMallocWithFlags": [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.c_uint,
        ],
        "hipMemset": [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t],
        "hipIpcGetMemHandle": [ctypes.POINTER(_IpcHandle), ctypes.c_void_p],
        "hipIpcOpenMemHandle": [
            ctypes.POINTER(ctypes.c_void_p),
            _IpcHandle,
            ctypes.c_uint,
        ],
        "hipGetErrorString": [ctypes.c_int],
    }
    for name, args in sig.items():
        fn = getattr(lib, name)
        fn.argtypes = args
        fn.restype = ctypes.c_char_p if name == "hipGetErrorString" else ctypes.c_int
    return lib


def _check(err: int, what: str) -> None:
    if err:
        msg = hip_runtime().hipGetErrorString(err)
        raise RuntimeError(f"{what}: {msg.decode() if msg else err}")


def _alloc_uncached(nbytes: int) -> int:
    size = -(-nbytes // _GRANULE) * _GRANULE
    ptr = ctypes.c_void_p()
    _check(
        hip_runtime().hipExtMallocWithFlags(
            ctypes.byref(ptr), size, _DEVICE_MALLOC_UNCACHED
        ),
        "hipExtMallocWithFlags",
    )
    _check(hip_runtime().hipMemset(ptr, 0, size), "hipMemset")
    return int(ptr.value)


def _handle(ptr: int) -> bytes:
    h = _IpcHandle()
    _check(
        hip_runtime().hipIpcGetMemHandle(ctypes.byref(h), ctypes.c_void_p(ptr)),
        "hipIpcGetMemHandle",
    )
    return bytes(ctypes.string_at(ctypes.byref(h), _IPC_HANDLE_BYTES))


def _open(handle: bytes) -> int:
    h = _IpcHandle()
    ctypes.memmove(ctypes.byref(h), handle, _IPC_HANDLE_BYTES)
    out = ctypes.c_void_p()
    _check(
        hip_runtime().hipIpcOpenMemHandle(
            ctypes.byref(out), h, _IPC_MEM_LAZY_ENABLE_PEER_ACCESS
        ),
        "hipIpcOpenMemHandle",
    )
    return int(out.value)


class _DeviceBytes:
    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


class PeerBuffer:
    """One uncached buffer a rank, and every rank's buffer address
    (``addresses``, int64 on the device). ``group`` is a CPU (gloo) group of
    the ``npes`` ranks.

    Construction is collective: every rank must build its buffers in the same
    order. A failure on any rank, in the allocation, the IPC handle export or
    the mapping of a peer's buffer, raises :class:`PeerMemoryError` on every
    rank. The buffer starts zeroed; the tail's peer buffer then takes the
    start state of :func:`contract.init_peer_bytes`.

    A buffer is never freed: it lives until the process exits, and a
    failed construction keeps its memory in ``_RETIRED``. Pages a kernel had
    used uncached, once freed and reused by ordinary allocations, gave later
    kernels stale cache lines on MI355X.
    """

    def __init__(self, nbytes: int, group, rank: int, npes: int, device: torch.device):
        self.rank = rank
        self.local = 0
        # A view, not an owner of the memory under it.
        self.bytes: torch.Tensor | None = None
        self._opened: list[int] = []
        failure, handle = None, None
        # Any local failure is held until after the agreement below: a rank
        # that raised before the collectives would leave its peers waiting.
        try:
            # HIP allocates and maps on the current device.
            with torch.cuda.device(device):
                self.local = _alloc_uncached(nbytes)
                self.bytes = torch.as_tensor(
                    _DeviceBytes(self.local, nbytes), device=device
                )
                if npes > 1:
                    handle = _handle(self.local)
        except Exception as err:
            failure = err
        addresses = [self.local]
        if npes > 1:
            # Every rank reaches both collectives, also after a failure of its
            # own, so that every rank fails together instead of waiting.
            offers: list = [None] * npes
            dist.all_gather_object(offers, (nbytes, handle), group=group)
            addresses = []
            if failure is None:
                try:
                    with torch.cuda.device(device):
                        addresses = self._map(offers, nbytes)
                except Exception as err:
                    failure = err
            ok = torch.tensor([int(failure is None)], dtype=torch.int32)
            dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=group)
            if not ok.item():
                self.close()
                where = "this" if failure else "another"
                raise PeerMemoryError(
                    f"peer memory setup failed on {where} rank"
                ) from failure
        elif failure is not None:
            self.close()
            raise PeerMemoryError("peer memory setup failed on this rank") from failure
        self.addresses = torch.tensor(addresses, dtype=torch.int64, device=device)

    def _map(self, offers: list, nbytes: int) -> list[int]:
        """Every rank's buffer address: this rank's own, the peers' mapped."""
        addresses = []
        for peer, (size, handle) in enumerate(offers):
            if handle is None:
                raise RuntimeError(f"rank {peer} has no buffer to share")
            if size != nbytes:
                raise RuntimeError(
                    f"rank {peer} offers {size} bytes, this rank {nbytes}: "
                    "the ranks build their buffers in different orders"
                )
            if peer == self.rank:
                addresses.append(self.local)
            else:
                base = _open(handle)
                self._opened.append(base)
                addresses.append(base)
        return addresses

    def close(self) -> None:
        """Drop the buffer after a failed construction, keeping its memory and
        peer mappings in ``_RETIRED`` (idempotent)."""
        if self.local:
            _RETIRED.append((self.local, self._opened))
            self.bytes, self.local, self._opened = None, 0, []

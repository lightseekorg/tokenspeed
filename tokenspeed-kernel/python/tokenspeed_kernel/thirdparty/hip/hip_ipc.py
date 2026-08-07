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

"""Pure-Python bindings for the HIP runtime's memory IPC API.

The runtime is loaded lazily so importing an operation that can fall back from
HIP IPC does not make ``libamdhip64.so`` an unconditional import dependency.
"""

import ctypes
from functools import lru_cache

_HIP_IPC_MEM_HANDLE_SIZE = 64
_HIP_IPC_LAZY_ENABLE_PEER_ACCESS = 1


class _hipIpcMemHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * _HIP_IPC_MEM_HANDLE_SIZE)]


class HipIpcLibrary:
    """Small typed wrapper around the HIP memory IPC runtime functions."""

    def __init__(self, so_file: str = "libamdhip64.so"):
        self.lib = ctypes.CDLL(so_file)
        self.lib.hipIpcGetMemHandle.argtypes = [
            ctypes.POINTER(_hipIpcMemHandle),
            ctypes.c_void_p,
        ]
        self.lib.hipIpcGetMemHandle.restype = ctypes.c_int
        self.lib.hipIpcOpenMemHandle.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            _hipIpcMemHandle,
            ctypes.c_uint,
        ]
        self.lib.hipIpcOpenMemHandle.restype = ctypes.c_int
        self.lib.hipIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
        self.lib.hipIpcCloseMemHandle.restype = ctypes.c_int
        self.lib.hipMemGetAddressRange.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
        ]
        self.lib.hipMemGetAddressRange.restype = ctypes.c_int

    @staticmethod
    def _check(err: int, what: str) -> None:
        if err != 0:
            raise RuntimeError(f"{what} failed with HIP error {err}")

    def hipMemGetAddressRange(self, ptr: int) -> tuple[int, int]:
        """Return the base and size of the allocation containing ``ptr``."""
        base = ctypes.c_void_p()
        size = ctypes.c_size_t()
        self._check(
            self.lib.hipMemGetAddressRange(
                ctypes.byref(base), ctypes.byref(size), ctypes.c_void_p(ptr)
            ),
            "hipMemGetAddressRange",
        )
        return base.value, size.value

    def hipIpcGetMemHandle(self, ptr: int) -> bytes:
        """Export the allocation at ``ptr`` as an opaque 64-byte IPC handle."""
        handle = _hipIpcMemHandle()
        self._check(
            self.lib.hipIpcGetMemHandle(
                ctypes.byref(handle), ctypes.c_void_p(ptr)
            ),
            "hipIpcGetMemHandle",
        )
        return ctypes.string_at(ctypes.byref(handle), _HIP_IPC_MEM_HANDLE_SIZE)

    def hipIpcOpenMemHandle(self, raw: bytes) -> int:
        """Open an opaque IPC handle and return its local mapped base pointer."""
        handle = _hipIpcMemHandle()
        ctypes.memmove(ctypes.byref(handle), raw, _HIP_IPC_MEM_HANDLE_SIZE)
        out = ctypes.c_void_p()
        self._check(
            self.lib.hipIpcOpenMemHandle(
                ctypes.byref(out), handle, _HIP_IPC_LAZY_ENABLE_PEER_ACCESS
            ),
            "hipIpcOpenMemHandle",
        )
        return out.value

    def hipIpcCloseMemHandle(self, ptr: int) -> None:
        """Close a peer mapping, preserving the runtime's best-effort teardown."""
        self.lib.hipIpcCloseMemHandle(ctypes.c_void_p(ptr))


@lru_cache(maxsize=1)
def get_hip_ipc_library() -> HipIpcLibrary:
    """Return the process-wide lazily loaded HIP IPC wrapper."""
    return HipIpcLibrary()

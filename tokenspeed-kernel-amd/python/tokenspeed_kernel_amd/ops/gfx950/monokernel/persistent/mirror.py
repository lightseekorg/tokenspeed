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

"""A host copy of a persistent kernel's failure record that outlives the process.

``fail_stop`` writes the record to the error words in device memory, which
the host cannot read once the trap has ended the queue (the runtime then
aborts the process). With a mirror attached, it also writes the record to a
file-backed page in host memory first, so whoever outlives the process (its
parent, a supervisor) can read why it stopped.
"""

from __future__ import annotations

import ctypes
import mmap
import os

import torch
from tokenspeed_kernel_amd.ops.gfx950.monokernel.persistent.sync import (
    ERROR_MIRROR,
    ERROR_RECORD_FIELDS,
)

_PAGE = 4096
# hipHostRegisterMapped: the page is mapped into the device's address space.
_HIP_HOST_REGISTER_MAPPED = 0x2


def _hip() -> ctypes.CDLL:
    bundled = os.path.join(os.path.dirname(torch.__file__), "lib", "libamdhip64.so")
    return ctypes.CDLL(bundled if os.path.exists(bundled) else "libamdhip64.so")


class FailureMirror:
    """A file-backed host page holding a copy of the failure record.

    Args:
        errors: The kernel's int32 error words (``ERROR_WORDS``) on a GPU;
            the mirror's device address is stored at ``ERROR_MIRROR``.
        path: A file to back the page, such as one under ``/dev/shm``; it is
            created or truncated, and zeroed.
    """

    def __init__(self, errors: torch.Tensor, path: str) -> None:
        self.path = path
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.ftruncate(descriptor, _PAGE)
            self._map = mmap.mmap(descriptor, _PAGE)
        finally:
            os.close(descriptor)
        self._buffer = ctypes.c_char.from_buffer(self._map)
        host = ctypes.addressof(self._buffer)
        hip = _hip()
        with torch.cuda.device(errors.device):
            status = hip.hipHostRegister(
                ctypes.c_void_p(host), ctypes.c_size_t(_PAGE), _HIP_HOST_REGISTER_MAPPED
            )
            if status != 0:
                raise RuntimeError(f"hipHostRegister failed ({status}) for {path}")
            device = ctypes.c_void_p()
            status = hip.hipHostGetDevicePointer(
                ctypes.byref(device), ctypes.c_void_p(host), 0
            )
            if status != 0:
                raise RuntimeError(f"hipHostGetDevicePointer failed ({status})")
        errors.view(torch.int64)[ERROR_MIRROR // 2] = device.value

    @staticmethod
    def read(path: str) -> list[int]:
        """The mirrored record (site, logical workgroup, detail, epoch, label);
        all zero if no failure was recorded."""
        with open(path, "rb") as handle:
            data = handle.read(4 * ERROR_RECORD_FIELDS)
        return [
            int.from_bytes(data[i : i + 4], "little", signed=True)
            for i in range(0, len(data), 4)
        ]

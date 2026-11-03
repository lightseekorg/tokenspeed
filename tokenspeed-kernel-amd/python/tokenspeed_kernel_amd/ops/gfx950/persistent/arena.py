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

"""Uncached device memory for a persistent kernel's cross-XCD flags and small
hand-offs.

Each XCD has its own L2, which does not snoop the others: a line one XCD
cached during a previous launch can still be read there after another XCD
wrote the word. Loads from uncached memory (``hipDeviceMallocUncached``) skip
every cache, so a word another XCD stored is seen without cache bits on the
load, and scalar ``glc`` polls see it too (measured on MI355X).
Keep only words read soon after another XCD writes them here (flags, routed
ids, small activation rows): every access goes to memory.
"""

from __future__ import annotations

import ctypes

import torch
from tokenspeed_kernel_amd.ops.gfx950.persistent.mirror import _hip

# hipExtMallocWithFlags flag: device memory no cache keeps lines of.
_HIP_DEVICE_MALLOC_UNCACHED = 0x3
# hipMemcpyKind of a device-to-host copy.
_HIP_DEVICE_TO_HOST = 2


class UncachedBuffer:
    """A zeroed uncached device allocation, passed to kernels as a pointer.

    Kernels take it like a tensor (``data_ptr()`` and ``dtype``); its
    element type is ``dtype``. Free it with :meth:`free` (it is not
    reference counted by torch).

    Args:
        numel: Elements of ``dtype``.
        dtype: The element type kernels see.
        device: The GPU.
    """

    def __init__(self, numel: int, dtype: torch.dtype, device: torch.device) -> None:
        self.dtype = dtype
        self.device = torch.device(device)
        self.nbytes = numel * torch.empty((), dtype=dtype).element_size()
        self._hip = _hip()
        pointer = ctypes.c_void_p()
        with torch.cuda.device(self.device):
            status = self._hip.hipExtMallocWithFlags(
                ctypes.byref(pointer),
                ctypes.c_size_t(self.nbytes),
                ctypes.c_uint(_HIP_DEVICE_MALLOC_UNCACHED),
            )
            if status != 0:
                raise RuntimeError(
                    f"hipExtMallocWithFlags(uncached) failed ({status}) for "
                    f"{self.nbytes} bytes"
                )
            self._pointer = pointer.value
            status = self._hip.hipMemset(
                ctypes.c_void_p(self._pointer), 0, ctypes.c_size_t(self.nbytes)
            )
            if status != 0:
                raise RuntimeError(f"hipMemset failed ({status})")

    def data_ptr(self) -> int:
        return self._pointer

    def read(self) -> torch.Tensor:
        """A host copy of the buffer (after the device work is synchronized)."""
        host = torch.empty(self.nbytes, dtype=torch.uint8)
        status = self._hip.hipMemcpy(
            ctypes.c_void_p(host.data_ptr()),
            ctypes.c_void_p(self._pointer),
            ctypes.c_size_t(self.nbytes),
            _HIP_DEVICE_TO_HOST,
        )
        if status != 0:
            raise RuntimeError(f"hipMemcpy failed ({status})")
        return host.view(self.dtype)

    def free(self) -> None:
        if self._pointer:
            self._hip.hipFree(ctypes.c_void_p(self._pointer))
            self._pointer = 0

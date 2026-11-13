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

"""Host-memory allocations and zero-copy device aliases, not GPU allocations."""

from __future__ import annotations

import ctypes
import weakref
from collections.abc import Callable
from dataclasses import dataclass

import torch


class _CudaArray:
    def __init__(self, address: int, nbytes: int, owner: object):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (address, False),
            "version": 3,
        }
        self.owner = owner


@dataclass
class HostMemoryViews:
    """CPU storage and an optional CUDA alias of the same physical host bytes."""

    host: torch.Tensor
    device: torch.Tensor | None


def host_memory_views(
    address: int,
    device_address: int | None,
    nbytes: int,
    *,
    release: Callable[[], None],
) -> HostMemoryViews:
    """Wrap existing memory without copying; views retain its cleanup callback.

    Args:
        address: CPU-accessible start of the allocation.
        device_address: GPU-accessible alias, or None for CPU-only users.
        nbytes: Number of accessible bytes.
        release: Idempotent cleanup, called after the last storage view dies.

    Returns:
        Byte tensors sharing the allocation. The device tensor consumes no HBM.
    """
    buffer = (ctypes.c_ubyte * nbytes).from_address(address)
    finalizer = weakref.finalize(buffer, release)
    finalizer.atexit = False
    try:
        host = torch.frombuffer(buffer, dtype=torch.uint8)
        device = None
        if device_address is not None:
            device = torch.as_tensor(
                _CudaArray(device_address, nbytes, buffer),
                device=torch.device("cuda", torch.cuda.current_device()),
            )
        return HostMemoryViews(host, device)
    except BaseException:
        finalizer()
        raise


class HostRegistration:
    """An explicitly owned registration of existing CPU storage."""

    def __init__(self, tensor: torch.Tensor):
        from tokenspeed_kernel.platform import current_platform

        if tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise ValueError("Host registration requires contiguous CPU storage")
        self.address = tensor.data_ptr()
        self.active = False
        self.device_address = 0
        self.device = torch.cuda.current_device()
        platform = current_platform()
        status = torch.cuda.cudart().cudaHostRegister(
            self.address, tensor.numel() * tensor.element_size(), 3
        )
        if int(status) != 0:
            raise RuntimeError(
                f"cudaHostRegister(PORTABLE|MAPPED) failed with {status!s}"
            )
        self.active = True
        try:
            if platform.is_nvidia:
                from tokenspeed_kernel.host_memory.cuda_vmm import registered_pointer

                self.device_address = registered_pointer(self.address)
            else:
                self.device_address = platform.device_visible_data_ptr(tensor)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Unregister while the caller still owns the backing CPU mapping."""
        if self.active:
            with torch.cuda.device(self.device):
                torch.cuda.synchronize(self.device)
                status = torch.cuda.cudart().cudaHostUnregister(self.address)
                if int(status) != 0:
                    raise RuntimeError(f"cudaHostUnregister failed with {status!s}")
            self.active = False


def create_host_allocation(nbytes: int):
    """Create exportable host-NUMA memory, raising if the platform cannot.

    Args:
        nbytes: Logical allocation size.

    Returns:
        An allocation with export(), map(), and close() methods. Exported
        metadata can be passed to import_host_allocation on a same-node peer.
        Close handles after every peer has mapped them; the returned tensor
        views retain the physical allocation independently of those handles.
    """
    from tokenspeed_kernel.host_memory.cuda_vmm import CudaHostAllocation

    return CudaHostAllocation.create(nbytes)


def host_numa_node() -> int | None:
    """Return the current GPU's host NUMA node, or None if unavailable."""
    if torch.version.cuda is None:
        return None
    from tokenspeed_kernel.host_memory.cuda_vmm import device_host_numa_node

    try:
        node = device_host_numa_node(torch.cuda.current_device())
    except (RuntimeError, AttributeError, ImportError):
        return None
    return node if node >= 0 else None


def import_host_allocation(metadata):
    """Import same-node metadata exported by create_host_allocation.

    Args:
        metadata: Pickleable allocation descriptor, never a raw process fd.

    Returns:
        A locally owned allocation handle, ready for map().
    """
    from tokenspeed_kernel.host_memory.cuda_vmm import CudaHostAllocation

    return CudaHostAllocation.import_handles(metadata)

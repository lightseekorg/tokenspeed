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

"""One process-wide Iris heap; operations retain their own group and buffers."""

import inspect
import itertools
import math
import os
import socket
from contextlib import contextmanager

import torch
import torch.distributed as dist
from tokenspeed_kernel._triton import redirect_triton_to_tokenspeed_triton

with redirect_triton_to_tokenspeed_triton():
    import iris
    import iris.host.iris as iris_host

MIN_HEAP_BYTES = 256 << 20
HEAP_HEADROOM_BYTES = 16 << 20
_context = None
_heap_size: int | None = None
_allocator_type: str | None = None
_hostname_exchanges = itertools.count(1)


def configure_iris(*, heap_size: int | None, allocator_type: str | None) -> None:
    """Set heap options before preparation; None picks the allocator.

    None means vmem_chunked when the job spans hosts and Iris's torch default
    otherwise. A name is passed unchanged to Iris, whose IRIS_ALLOCATOR override
    takes precedence. Neither option may change a live heap.
    """
    global _heap_size, _allocator_type
    if _context is not None and (heap_size, allocator_type) != (
        _heap_size,
        _allocator_type,
    ):
        raise RuntimeError("Configure Iris before preparing communication buffers")
    _heap_size = heap_size
    _allocator_type = allocator_type


def allocation_bytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    """Estimate storage using Iris's Torch allocator's 1024-byte alignment."""
    return (math.prod(shape) * dtype.itemsize + 1023) // 1024 * 1024


def _job_spans_hosts() -> bool:
    """Whether the world spans hosts, exchanged through the store as Iris does."""
    world_size = dist.get_world_size()
    if world_size == 1:
        return False
    store = dist.distributed_c10d._get_default_store()
    prefix = f"tokenspeed_iris_hostname_v{next(_hostname_exchanges)}/"
    store.set(f"{prefix}{dist.get_rank()}", socket.gethostname())
    return len({store.get(f"{prefix}{rank}") for rank in range(world_size)}) > 1


@contextmanager
def _heap_on(device: int):
    """Build the heap on ``device``; Iris without ``device=`` uses rank % GPUs.

    The device context is left for the caller to build once ``ctx.device`` is
    corrected, so nothing is allocated on GPU rank % GPUs.
    """
    heap = iris_host.SymmetricHeap
    saved = iris_host.set_device, heap, iris_host.Iris._build_device_context

    def build_heap(size, gpu_id, rank, world_size, allocator_type="torch"):
        allocator = os.environ.get("IRIS_ALLOCATOR", allocator_type).lower()
        if allocator == "vmem" and device != gpu_id:
            # Without device=, VMemAllocator grants access to GPUs 0..world_size-1.
            raise RuntimeError(
                f"Iris's vmem allocator cannot place rank {rank} on GPU {device}: "
                "this Iris has no device= argument. Use the torch or vmem_chunked "
                "allocator, or an Iris that accepts device=."
            )
        return heap(size, device, rank, world_size, allocator_type)

    iris_host.set_device = lambda gpu_id: None
    iris_host.SymmetricHeap = build_heap
    iris_host.Iris._build_device_context = lambda self: None
    try:
        yield
    finally:
        (
            iris_host.set_device,
            iris_host.SymmetricHeap,
            iris_host.Iris._build_device_context,
        ) = saved


def _create_context(size: int, allocator_type: str | None, device: int):
    options = {} if allocator_type is None else {"allocator_type": allocator_type}
    if "device" in inspect.signature(iris.iris).parameters:
        return iris.iris(heap_size=size, device=device, **options)
    with _heap_on(device):
        ctx = iris.iris(heap_size=size, **options)
    ctx.gpu_id, ctx.device = device, f"cuda:{device}"
    ctx._build_device_context()
    return ctx


def get_or_create_context(*, heap_size: int):
    """Create the shared context at its initial capacity, or return it unchanged.

    Configured heap capacity takes precedence over the caller's initial size.
    Iris enforces capacity as states allocate their buffers.
    """
    global _context
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Prepare Iris communication buffers before graph capture")
    if _context is None:
        size = _heap_size if _heap_size is not None else heap_size
        allocator_type = _allocator_type
        if allocator_type is None and _job_spans_hosts():
            # Only vmem_chunked maps peers on other hosts.
            allocator_type = "vmem_chunked"
        _context = _create_context(size, allocator_type, torch.cuda.current_device())
    return _context


class IrisGroup:
    """Retain the context and the ordered heap addresses for one process group."""

    def __init__(self, ctx, group: dist.ProcessGroup, rank: int, device: torch.device):
        if ctx.gpu_id != device.index:
            raise RuntimeError(
                f"The Iris heap is on GPU {ctx.gpu_id}, but this group runs on {device}"
            )
        self.ctx = ctx
        self.group = group
        self.rank = rank
        self.device = device
        self.world_size = group.size()
        ranks = dist.get_process_group_ranks(group)
        assert ranks[rank] == dist.get_rank()
        self.bases = ctx.get_heap_bases()[ranks].contiguous()
        addresses = [int(address) for address in self.bases.tolist()]
        self.addresses = tuple(addresses + [addresses[-1]] * (8 - len(addresses)))

    def finish_prepare(self) -> None:
        """Publish initialized buffers before any peer can use them."""
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)

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

import itertools
import math

import torch
import torch.distributed as dist
from tokenspeed_kernel._triton import redirect_triton_to_tokenspeed_triton
from tokenspeed_kernel.ops.communication.fabric import _host_identity
from tokenspeed_kernel.platform import current_platform

with redirect_triton_to_tokenspeed_triton():
    import iris
    from iris.host.distributed.topology import TopologyDiscovery

MIN_HEAP_BYTES = 256 << 20
HEAP_HEADROOM_BYTES = 16 << 20
_context = None
_heap_size: int | None = None
_allocator_type: str | None = None
_store_exchanges = itertools.count(1)


def configure_iris(*, heap_size: int | None, allocator_type: str | None) -> None:
    """Set heap options before preparation; None picks the allocator.

    None means vmem_chunked when a gfx1250 job spans hosts and Iris's torch
    default otherwise. A name is passed unchanged to Iris, whose IRIS_ALLOCATOR
    override takes precedence. Neither option may change a live heap.
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


def _gather_through_store(name: str, value: str) -> list[str]:
    """Every world rank's ``value``, exchanged through the c10d store as Iris does.

    Every rank must call it, in the same order, since each call takes the next key.
    """
    store = dist.distributed_c10d._get_default_store()
    prefix = f"tokenspeed_iris_{name}_v{next(_store_exchanges)}/"
    store.set(f"{prefix}{dist.get_rank()}", value)
    return [
        store.get(f"{prefix}{rank}").decode() for rank in range(dist.get_world_size())
    ]


def _job_spans_hosts() -> bool:
    """Whether the world spans hosts, told apart by boot id rather than hostname."""
    if dist.get_world_size() == 1:
        return False
    return len(set(_gather_through_store("host", str(_host_identity())))) > 1


def world_in_one_fabric_domain() -> bool:
    """Whether one Iris heap can map every world rank; every rank must call it.

    A world on one host qualifies. Across hosts, every GPU must report the same
    fabric domain (the same pod), which is what vmem_chunked requires.
    """
    if not _job_spans_hosts():
        return True
    topology = TopologyDiscovery().discover()
    domains = {info.fabric_info.domain_key for info in topology.gpu_info.values()}
    return len(domains) == 1 and "" not in domains


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
        if (
            allocator_type is None
            and current_platform().is_cdna5
            and _job_spans_hosts()
        ):
            # Only vmem_chunked maps peers on other hosts, and only gfx1250
            # routes groups across hosts to Iris.
            allocator_type = "vmem_chunked"
        options = {} if allocator_type is None else {"allocator_type": allocator_type}
        _context = iris.iris(
            heap_size=size, device=torch.cuda.current_device(), **options
        )
    return _context


class IrisGroup:
    """Retain the context and the ordered heap addresses for one process group."""

    def __init__(self, ctx, group: dist.ProcessGroup, rank: int, device: torch.device):
        index = torch.cuda.current_device() if device.index is None else device.index
        if ctx.gpu_id != index:
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

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

import logging
import math
import socket

import torch
import torch.distributed as dist
from tokenspeed_kernel._triton import redirect_triton_to_tokenspeed_triton
from tokenspeed_kernel.ops.communication.fabric import host_identity
from tokenspeed_kernel.platform import current_platform

with redirect_triton_to_tokenspeed_triton():
    import iris
    from iris.host.distributed.topology import TopologyDiscovery

logger = logging.getLogger(__name__)

MIN_HEAP_BYTES = 256 << 20
HEAP_HEADROOM_BYTES = 16 << 20
# Heap bytes owners reserve before the heap exists (reserve_heap), by owner:
# get_or_create_context adds them to whatever size its caller planned.
_reserved: dict[str, int] = {}
_context = None
_heap_size: int | None = None
_allocator_type: str | None = None
_spans_hosts: bool | None = None
_unmapped_reason: str | None = None


def configure_iris(*, heap_size: int | None, allocator_type: str | None) -> None:
    """Set heap options before preparation; None picks the allocator.

    None means vmem_chunked when the world spans hosts and Iris's torch default
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
    """Estimate storage using Iris's Torch allocator's 1024-byte alignment.

    vmem_chunked rounds larger allocations up to a power of two and grows the
    heap instead, so the estimate is only its initial capacity.
    """
    return (math.prod(shape) * dtype.itemsize + 1023) // 1024 * 1024


def _gather_hosts() -> list[tuple[str, str]]:
    """Every world rank's (host identity, hostname), through the c10d store.

    Iris exchanges hostnames the same way, without an NCCL collective. The key
    is fixed, so only ``discover_world`` calls this, once per process.
    """
    store = dist.distributed_c10d._get_default_store()
    prefix = "tokenspeed_iris_hosts/"
    store.set(f"{prefix}{dist.get_rank()}", f"{host_identity()} {socket.gethostname()}")
    return [
        tuple(store.get(f"{prefix}{rank}").decode().split(" ", 1))
        for rank in range(dist.get_world_size())
    ]


def _unmapped(hosts: list[tuple[str, str]]) -> str | None:
    """Why one heap cannot map ranks on ``hosts``, or None when it can."""
    pairs = set(hosts)
    identities = {identity for identity, _ in pairs}
    if not len(identities) == len({hostname for _, hostname in pairs}) == len(pairs):
        # Iris tells hosts apart by hostname, so it would map remote peers as
        # local ones, or local peers over the fabric.
        return "hostnames and boot ids disagree on which ranks share a host"
    if len(identities) == 1:
        return None
    if not current_platform().is_cdna5:
        return "only gfx1250 maps peers on other hosts"
    topology = TopologyDiscovery().discover()
    domains = {info.fabric_info.domain_key for info in topology.gpu_info.values()}
    if len(domains) != 1 or "" in domains:
        return "the GPUs are not all in one fabric domain"
    return None


def discover_world() -> bool:
    """Decide once whether one Iris heap can map every world rank.

    The heap spans the world, so a world on one host qualifies; across hosts
    only gfx1250 does, with every GPU in one fabric domain (the same pod). Hosts
    are told apart by boot id, and Iris's hostnames must agree. The first call
    is a world collective; later calls return the cached answer.
    """
    global _spans_hosts, _unmapped_reason
    if _spans_hosts is None:
        if dist.get_world_size() == 1:
            _spans_hosts = False
        else:
            hosts = _gather_hosts()
            # Set together, so a failed discovery leaves the world undecided.
            _unmapped_reason = _unmapped(hosts)
            _spans_hosts = len({identity for identity, _ in hosts}) > 1
        verdict = "yes" if _unmapped_reason is None else f"no, {_unmapped_reason}"
        logger.info(
            f"Iris heap maps the world: {verdict} (spans hosts: {_spans_hosts!s})"
        )
    return _unmapped_reason is None


def maps_world() -> bool | None:
    """``discover_world``'s answer, or None before it ran."""
    return None if _spans_hosts is None else _unmapped_reason is None


def reserve_heap(owner: str, nbytes: int) -> None:
    """Reserve ``nbytes`` of the shared heap for ``owner``, replacing its
    earlier reservation. The heap is created with every reservation on top of
    the size its creator planned, so call this before the heap exists: a live
    heap does not grow."""
    if _context is not None:
        raise RuntimeError(
            f"{owner} reserved {nbytes} bytes of the Iris heap after it was "
            "created; reserve before preparing communication buffers"
        )
    _reserved[owner] = int(nbytes)


def get_or_create_context(*, heap_size: int):
    """Create the shared context at its initial capacity, or return it unchanged.

    The initial capacity is ``heap_size`` plus every reservation
    (``reserve_heap``). A configured heap capacity takes precedence over that
    sum.
    The torch allocator enforces capacity as states allocate their buffers;
    vmem_chunked grows the heap on demand instead. A world left undecided, as
    when the heap is used without distributed initialization, is decided here:
    creating the heap is already a world collective, so this adds no new wait.
    """
    global _context
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Prepare Iris communication buffers before graph capture")
    if _context is None:
        if not discover_world():
            raise RuntimeError(
                f"One Iris heap cannot map this world: {_unmapped_reason}"
            )
        # A configured size is the whole heap, as without reservations: the
        # torch allocator then fails the first allocation past it, while
        # vmem_chunked grows on demand.
        size = (
            _heap_size
            if _heap_size is not None
            else heap_size + sum(_reserved.values())
        )
        if _reserved:
            logger.info(f"Iris heap: {size} bytes, of which reserved {dict(_reserved)}")
        allocator_type = _allocator_type
        if allocator_type is None and _spans_hosts:
            # Only vmem_chunked maps peers on other hosts.
            allocator_type = "vmem_chunked"
        options = {} if allocator_type is None else {"allocator_type": allocator_type}
        _context = iris.iris(
            heap_size=size, device=torch.cuda.current_device(), **options
        )
    return _context


def existing_context():
    """The shared context, which the communication preparation created.

    Raises:
        RuntimeError: The heap does not exist yet. Buffers taken from it are
            reserved beforehand (``reserve_heap``); the heap itself is sized
            only by the preparation that creates it.
    """
    if _context is None:
        raise RuntimeError(
            "The Iris heap does not exist yet: prepare the communication "
            "buffers first"
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

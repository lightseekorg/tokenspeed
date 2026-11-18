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

"""Startup-only coordination of one immutable host table per node and PP stage."""

# Every setup failure must be exchanged before any participant raises.
# ruff: noqa: BLE001

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import math
import mmap
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HostTableGroup:
    rank: int
    ranks: tuple[int, ...]
    local_ranks: tuple[int, ...]
    process_group: dist.ProcessGroup | None

    def __post_init__(self):
        if (
            self.rank not in self.local_ranks
            or not set(self.local_ranks).issubset(self.ranks)
            or tuple(sorted(set(self.ranks))) != self.ranks
            or tuple(sorted(set(self.local_ranks))) != self.local_ranks
        ):
            raise ValueError("Invalid host-table group membership")
        if len(self.ranks) > 1 and self.process_group is None:
            raise ValueError("Shared host tables require a startup CPU process group")

    @property
    def writer(self) -> int:
        return self.local_ranks[0]

    def exchange(self, value):
        if len(self.ranks) == 1:
            return [value]
        output = [None] * len(self.ranks)
        dist.all_gather_object(output, value, group=self.process_group)
        return output

    def errors(self, error: str | None) -> list[str]:
        return [
            f"rank {rank}: {value}"
            for rank, value in zip(self.ranks, self.exchange(error))
            if value
        ]


def ple_host_table_group(mapping) -> HostTableGroup:
    ranks = mapping.attn.world_group
    local_ranks = tuple(
        rank for rank in ranks if rank // mapping.nprocs_per_node == mapping.node_rank
    )
    process_group = None
    if len(ranks) > 1:
        from tokenspeed.runtime.distributed.process_group_manager import (
            process_group_manager,
        )

        process_group = process_group_manager.get_process_group("gloo", ranks)
    return HostTableGroup(mapping.rank, ranks, local_ranks, process_group)


def _available_host_bytes() -> int | None:
    """Bound MemAvailable by this process's cgroup and ancestor headrooms."""
    if sys.platform != "linux":
        return None
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.split()[0]) * 1024
    available = values["MemAvailable"]
    root = Path("/sys/fs/cgroup")
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        _, controllers, relative = line.split(":", 2)
        if not controllers:
            names = ("memory.max", "memory.current")
            base = root
        elif "memory" in controllers.split(","):
            names = ("memory.limit_in_bytes", "memory.usage_in_bytes")
            base = root / "memory"
        else:
            continue
        directory = base / relative.lstrip("/")
        if not directory.is_dir():
            directory = base
        while directory.is_relative_to(base):
            try:
                limit = (directory / names[0]).read_text().strip()
                used = int((directory / names[1]).read_text())
                if limit != "max":
                    available = min(available, max(0, int(limit) - used))
            except FileNotFoundError:
                pass
            if directory == base:
                break
            directory = directory.parent
    return available


def _check_memory(nbytes: int) -> None:
    available = _available_host_bytes()
    if available is not None and nbytes > available * 0.95:
        raise MemoryError(
            f"Shared PLE table requires {nbytes} bytes but only {available} host bytes "
            "are available (including cgroup limits); keep 5% free"
        )


def _error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _prefer_local_numa(address: int, length: int, nodes: tuple[int, ...]) -> None:
    """Place shared pages near their GPU readers before registration faults them."""
    if sys.platform != "linux":
        return
    try:
        library = ctypes.util.find_library("numa")
        if library is None:
            raise OSError("libnuma is unavailable")
        bind = ctypes.CDLL(library, use_errno=True).mbind
        bind.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_uint,
        ]
        bind.restype = ctypes.c_int
        mask = None
        maxnode = 0
        policy = 4  # MPOL_LOCAL when GPU topology is unavailable.
        if nodes:
            maxnode = max(nodes) + 1
            word_bits = ctypes.sizeof(ctypes.c_ulong) * 8
            mask = (ctypes.c_ulong * ((maxnode + word_bits - 1) // word_bits))()
            for node in nodes:
                mask[node // word_bits] |= 1 << (node % word_bits)
            # Linux consumes maxnode - 1 bits from this bitmap.
            maxnode = len(mask) * word_bits + 1
            policy = 1 if len(nodes) == 1 else 3  # PREFERRED / INTERLEAVE.
        if bind(address, length, policy, mask, maxnode, 0) != 0:
            raise OSError(ctypes.get_errno(), f"mbind(policy={policy}) failed")
        logger.info(f"PLE shared-memory NUMA policy={policy}, GPU host nodes={nodes}")
    except (OSError, AttributeError) as exc:
        logger.warning(f"Cannot set PLE shared-memory NUMA policy: {exc!s}")


class _ShmMapping:
    def __init__(self, mapping: mmap.mmap):
        self.mapping = mapping
        self.registration = None
        self.closed = False

    def close(self) -> None:
        if self.closed:
            return
        if self.registration is not None:
            self.registration.close()
        self.mapping.close()
        self.closed = True


class SharedHostTable:
    def __init__(
        self,
        host: torch.Tensor,
        device: torch.Tensor | None,
        *,
        group: HostTableGroup,
        backend: str,
        huge_page_bytes: int,
    ):
        self.host = host
        self.device = device
        self.group = group
        self.backend = backend
        self.huge_page_bytes = huge_page_bytes

    @property
    def is_writer(self) -> bool:
        return self.group.rank == self.group.writer

    @classmethod
    def allocate(
        cls,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        *,
        group: HostTableGroup,
        backend: str,
        directory: str,
        register: bool,
    ) -> SharedHostTable:
        """Allocate one physical table per node, shared by its stage workers.

        Args:
            shape: Nonempty CPU table dimensions, identical across the group.
            dtype: Storage dtype, including FP8.
            group: Stage-wide CPU coordination and this node's participating ranks.
            backend: Explicitly 'auto', 'shm', or 'cuda_vmm'. Auto alone permits fallback.
            directory: Shared-memory filesystem directory, normally /dev/shm.
            register: Whether each rank needs a GPU alias; False is CPU-only shm.

        Returns:
            CPU storage plus a zero-copy device alias. Views keep mappings alive.
        """
        if (
            backend not in ("auto", "shm", "cuda_vmm")
            or not isinstance(shape, tuple)
            or not shape
            or any(
                not isinstance(size, int) or isinstance(size, bool) or size <= 0
                for size in shape
            )
        ):
            raise ValueError("Expected a positive shape and auto/shm/cuda_vmm backend")
        if not register and backend != "shm":
            raise ValueError("CPU-only shared tables require backend='shm'")
        specs = group.exchange(
            (shape, str(dtype), backend, register, group.local_ranks)
        )
        if any(spec[:4] != specs[0][:4] for spec in specs):
            raise ValueError("Host-table ranks disagree on allocation metadata")
        for rank, spec in zip(group.ranks, specs):
            if rank not in spec[4] or any(
                specs[group.ranks.index(peer)][4] != spec[4] for peer in spec[4]
            ):
                raise ValueError("Host-table ranks disagree on node membership")
        nbytes = (
            math.prod(shape) * torch.empty((), dtype=dtype, device="cpu").element_size()
        )
        if backend != "shm":
            candidate = cls._vmm(nbytes, group)
            if isinstance(candidate, list):
                detail = "; ".join(candidate)
                if backend == "cuda_vmm":
                    raise RuntimeError(
                        f"Shared PLE host-NUMA allocation failed: {detail}"
                    )
                if group.rank == group.writer:
                    logger.warning(
                        f"PLE host-NUMA large pages unavailable; falling back to /dev/shm: {detail}"
                    )
            else:
                views, page_bytes = candidate
                return cls(
                    views.host.view(dtype).view(shape),
                    views.device.view(dtype).view(shape),
                    group=group,
                    backend="cuda_vmm",
                    huge_page_bytes=page_bytes,
                )
        views = cls._shm(nbytes, group, directory, register)
        return cls(
            views.host.view(dtype).view(shape),
            views.device.view(dtype).view(shape) if views.device is not None else None,
            group=group,
            backend="shm",
            huge_page_bytes=0,
        )

    @staticmethod
    def _vmm(nbytes: int, group: HostTableGroup):
        allocation = views = metadata = None
        error = None
        if group.rank == group.writer:
            try:
                from tokenspeed_kernel.host_memory import create_host_allocation

                _check_memory(nbytes)
                allocation = create_host_allocation(nbytes)
                metadata = allocation.export()
            except Exception as exc:
                error = _error(exc)
        exported = group.exchange((metadata, error))
        errors = [
            f"rank {rank}: {value[1]}"
            for rank, value in zip(group.ranks, exported)
            if value[1]
        ]
        if errors:
            if allocation is not None:
                allocation.close()
            group.exchange(None)
            return errors
        descriptor = exported[group.ranks.index(group.writer)][0]
        try:
            if allocation is None:
                from tokenspeed_kernel.host_memory import import_host_allocation

                allocation = import_host_allocation(descriptor)
            views = allocation.map()
        except Exception as exc:
            error = _error(exc)
        try:
            errors = group.errors(error)
        finally:
            if allocation is not None:
                allocation.close()
        if errors:
            views = None
            group.exchange(None)
            return errors
        return views, descriptor.granularity

    @staticmethod
    def _shm(nbytes: int, group: HostTableGroup, directory: str, register: bool):
        from tokenspeed_kernel.host_memory import (
            HostRegistration,
            host_memory_views,
            host_numa_node,
        )

        node = None
        if register:
            try:
                node = host_numa_node()
            except Exception as exc:
                logger.warning(f"Cannot query PLE GPU NUMA topology: {exc!s}")
        topology = group.exchange(node)
        local_nodes = [topology[group.ranks.index(rank)] for rank in group.local_ranks]
        nodes = (
            tuple(sorted(set(local_nodes)))
            if all(node is not None for node in local_nodes)
            else ()
        )

        length = (nbytes + mmap.PAGESIZE - 1) // mmap.PAGESIZE * mmap.PAGESIZE
        fd = None
        path = None
        error = None
        if group.rank == group.writer:
            try:
                _check_memory(length)
                usage = os.statvfs(directory)
                if length > usage.f_bavail * usage.f_frsize:
                    raise OSError(
                        f"Not enough space in {directory} for {length} PLE bytes; increase --shm-size"
                    )
                fd, path = tempfile.mkstemp(prefix="ts-ple-", dir=directory)
                os.ftruncate(fd, length)
            except Exception as exc:
                error = _error(exc)
        created = group.exchange((path, error))
        errors = [value[1] for value in created if value[1]]
        if not errors:
            local_path = created[group.ranks.index(group.writer)][0]
            try:
                if fd is None:
                    fd = os.open(local_path, os.O_RDWR)
                if os.fstat(fd).st_size != length:
                    raise ValueError("Shared PLE backing file has the wrong size")
            except Exception as exc:
                error = _error(exc)
            errors = group.errors(error)
        # Every peer owns an fd before the writer unlinks the still-sparse file.
        # A killed worker cannot leave a populated table behind in /dev/shm.
        if group.rank == group.writer and path is not None:
            try:
                os.unlink(path)
            except OSError as exc:
                error = _error(exc)
        errors.extend(group.errors(error))
        if errors:
            if fd is not None:
                os.close(fd)
            raise RuntimeError(f"Shared PLE file setup failed: {'; '.join(errors)}")
        resource = None
        views = None
        try:
            mapping = mmap.mmap(
                fd, length, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE
            )
            resource = _ShmMapping(mapping)
            if sys.platform == "linux":
                try:
                    mapping.madvise(mmap.MADV_HUGEPAGE)
                except OSError:
                    pass
            address = ctypes.addressof(ctypes.c_ubyte.from_buffer(mapping))
            _prefer_local_numa(address, length, nodes)
            device_address = None
            if register:
                temporary = torch.frombuffer(mapping, dtype=torch.uint8)
                try:
                    resource.registration = HostRegistration(temporary)
                    device_address = resource.registration.device_address
                finally:
                    del temporary
            views = host_memory_views(
                address, device_address, nbytes, release=resource.close
            )
        except Exception as exc:
            error = _error(exc)
            if resource is not None:
                resource.close()
        finally:
            os.close(fd)
        errors = group.errors(error)
        if errors:
            views = None
            raise RuntimeError(f"Shared PLE mapping failed: {'; '.join(errors)}")
        return views

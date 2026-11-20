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

import ctypes
import ctypes.util
import logging
import mmap
import os
import sys
import tempfile
import weakref

import torch
import torch.distributed as dist
from tokenspeed_kernel import host_memory
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.distributed.process_group_manager import process_group_manager

logger = logging.getLogger(__name__)


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


class PLESharedTable:
    """GPU-mapped host PLE rows shared by workers on one host."""

    def __init__(self, mapping, source: torch.Tensor, layout: str, alloc: str):
        ranks = mapping.attn.world_group
        local = [r for r in ranks if r // mapping.nprocs_per_node == mapping.node_rank]
        peer = layout == "peer_sharded"
        if peer and len(local) != len(ranks):
            raise ValueError("peer_sharded requires the PLE table group on one node")
        owners = local if peer else local[:1]
        self.is_writer = mapping.rank in owners
        self.group = (
            process_group_manager.get_process_group("gloo", ranks)
            if len(ranks) > 1
            else None
        )
        nbytes = source.numel() * source.element_size()
        device = torch.cuda.current_device()
        try:
            node = host_memory.host_numa_node(device)
        except (RuntimeError, AttributeError, OSError) as exc:
            logger.warning(f"Cannot query PLE GPU NUMA topology: {exc}")
            node = None
        topology = self._exchange(node, len(ranks))
        readers = [mapping.rank] if peer else local
        nodes = [topology[ranks.index(r)] for r in readers]
        nodes = tuple(sorted(set(nodes))) if None not in nodes else ()
        tables = None
        if alloc in ("auto", "cumem"):
            tables, reason = self._cumem_tables(
                mapping.rank, ranks, owners, len(local) > 1, nbytes, nodes, device
            )
            if reason:
                if alloc == "cumem":
                    raise RuntimeError(
                        f"PLE host-NUMA cuMem allocation failed: {reason}"
                    )
                if mapping.rank == ranks[0]:
                    logger.warning(
                        f"PLE cuMem unavailable, falling back to mmap: {reason}"
                    )
        elif alloc != "mmap":
            raise ValueError(f"Unsupported PLE host table allocator: {alloc}")
        self.backend = "cumem" if tables is not None else "mmap"
        if tables is None:
            tables = self._mmap_tables(ranks, owners, nbytes, nodes)
        self.tables = [t.view(source.dtype).view(source.shape) for t in tables]
        self.weight = self.tables[owners.index(mapping.rank) if peer else 0]
        self.peer_ptrs = (
            torch.tensor(
                [current_platform().device_visible_data_ptr(t) for t in self.tables],
                dtype=torch.int64,
                device="cuda",
            )
            if peer
            else None
        )
        if self.is_writer:
            logger.info(
                f"PLE {layout} host table: backend={self.backend}, NUMA nodes={nodes}"
            )

    def _cumem_tables(self, rank, ranks, owners, exportable, nbytes, nodes, device):
        owned = exported = error = None
        mapped = []
        probe = host_memory.probe_host_memory(device, exportable)
        errors = [
            f"rank {r}: {err}"
            for r, err in zip(ranks, self._exchange(probe, len(ranks)))
            if err
        ]
        if errors:
            return None, "; ".join(errors)
        try:
            if self.is_writer:
                owned = host_memory.create_host_memory(
                    nbytes, nodes, device, exportable, 2 << 30, 0.05
                )
                if exportable:
                    exported = owned.export()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        metadata = self._exchange((exported, error), len(ranks))
        errors = [f"rank {r}: {err}" for r, (_, err) in zip(ranks, metadata) if err]
        if not errors:
            try:
                for owner in owners:
                    memory = (
                        owned
                        if owner == rank
                        else host_memory.import_host_memory(
                            metadata[ranks.index(owner)][0], device
                        )
                    )
                    mapped.append(memory)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            errors = [
                f"rank {r}: {err}"
                for r, err in zip(ranks, self._exchange(error, len(ranks)))
                if err
            ]
        if errors:
            for memory in mapped:
                if memory is not owned:
                    memory.close()
            if owned is not None:
                owned.close()
            self.publish()
            return None, "; ".join(errors)
        return [memory.tensor for memory in mapped], None

    def _mmap_tables(self, ranks, owners, nbytes, nodes):
        path = None
        error = None
        try:
            if self.is_writer:
                try:
                    with tempfile.NamedTemporaryFile(
                        prefix="ts-ple-", dir="/dev/shm", delete=False
                    ) as f:
                        path = f.name
                        os.ftruncate(f.fileno(), nbytes)
                        with mmap.mmap(f.fileno(), nbytes) as region:
                            addr = ctypes.addressof(ctypes.c_char.from_buffer(region))
                            _prefer_local_numa(addr, nbytes, nodes)
                        os.posix_fallocate(f.fileno(), 0, nbytes)
                except OSError as exc:
                    error = str(exc)
            locations = self._exchange((path, error), len(ranks))
            errors = [err for _, err in locations if err]
            if errors:
                raise RuntimeError(f"Cannot allocate shared PLE table: {errors}")

            tables = []
            platform = current_platform()
            try:
                for owner in owners:
                    with open(locations[ranks.index(owner)][0], "r+b") as f:
                        backing = mmap.mmap(f.fileno(), nbytes)
                    rows = torch.frombuffer(backing, dtype=torch.uint8)
                    platform.register_host_tensor_for_gpu_access(rows)
                    weakref.finalize(
                        self, platform.unregister_host_tensor_for_gpu_access, rows
                    )
                    tables.append(rows)
            except (OSError, RuntimeError) as exc:
                error = str(exc)
            errors = [err for err in self._exchange(error, len(ranks)) if err]
            if errors:
                raise RuntimeError(f"Cannot map shared PLE table: {errors}")
            return tables
        finally:
            if path is not None:
                os.unlink(path)

    def _exchange(self, value, size):
        if self.group is None:
            return [value]
        values = [None] * size
        dist.all_gather_object(values, value, group=self.group)
        return values

    def publish(self):
        """Wait until all owners have finished writing the checkpoint rows."""
        if self.group is not None:
            dist.barrier(group=self.group)

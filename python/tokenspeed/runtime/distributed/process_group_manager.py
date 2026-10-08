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

"""Helpers for initializing and caching torch distributed process groups."""

from datetime import timedelta

import torch
import torch.distributed as dist

from tokenspeed.runtime.distributed.mapping import Group, Mapping


def _make_all_groups(group: Group) -> list[Group]:
    """Enumerate all groups with the same size and stride pattern as ``group``."""
    size = len(group)
    stride = group[1] - group[0] if len(group) > 1 else 1
    block = size * stride
    world_size = dist.get_world_size()

    groups = []
    for base in range(0, world_size, block):
        for offset in range(stride):
            g = tuple(base + offset + i * stride for i in range(size))
            groups.append(g)
    return groups


class ProcessGroupManager:
    def __init__(self):
        self._process_groups: dict[str, dict[Group, dist.ProcessGroup]] = {}
        self._pg_timeout: timedelta | None = None
        self._device_backend = "nccl"
        # Set by init_emulated_rank_zero: one group per backend holding only
        # this process, standing in for every logical group.
        self._emulated_rank_groups: dict[str, dist.ProcessGroup] | None = None

    def init_distributed(
        self,
        mapping: Mapping,
        distributed_init_method: str = "env://",
        backend: str = "nccl",
        timeout: int | None = None,
        device_id: "torch.device | None" = None,
    ) -> None:
        self._init_world(
            world_size=mapping.world_size,
            rank=mapping.rank,
            distributed_init_method=distributed_init_method,
            backend=backend,
            timeout=timeout,
            device_id=device_id,
        )

    def init_emulated_rank_zero(
        self,
        distributed_init_method: str,
        backend: str,
        timeout: int | None,
        device_id: "torch.device | None",
    ) -> None:
        """Start a one-process world for ``--emulate-rank-zero``.

        Every group later passed to ``init_process_group`` is backed by a group
        holding only this process, so torch.distributed calls on any logical
        group complete locally.
        """
        if dist.is_initialized():
            raise RuntimeError(
                "rank emulation needs its own one-process world, but "
                "torch.distributed is already initialized"
            )
        self._init_world(
            world_size=1,
            rank=0,
            distributed_init_method=distributed_init_method,
            backend=backend,
            timeout=timeout,
            device_id=device_id,
        )
        self._emulated_rank_groups = {}

    def _init_world(
        self,
        *,
        world_size: int,
        rank: int,
        distributed_init_method: str,
        backend: str,
        timeout: int | None,
        device_id: "torch.device | None",
    ) -> None:
        if not dist.is_initialized():
            if distributed_init_method is None:
                raise ValueError(
                    "distributed_init_method must be provided when initializing distributed environment"
                )

            if timeout is not None:
                if not isinstance(timeout, int):
                    raise TypeError("timeout must be a number")
                if timeout <= 0:
                    raise ValueError("timeout must be positive")
                timeout = timedelta(seconds=timeout)

            self._pg_timeout = timeout
            self._device_backend = backend

            dist.init_process_group(
                backend=backend,
                init_method=distributed_init_method,
                world_size=world_size,
                rank=rank,
                timeout=timeout,
                device_id=device_id,
            )

    def register_process_group(
        self, backend: str, group: Group, process_group: dist.ProcessGroup
    ) -> None:
        if backend not in self._process_groups:
            self._process_groups[backend] = {}
        self._process_groups[backend][group] = process_group

    def get_process_group(self, backend: str, group: Group):
        return self._process_groups[backend][group]

    def get_device_process_group(self, group: Group):
        """Return the accelerator collective group for the requested ranks."""
        return self.get_process_group(self._device_backend, group)

    def has_process_group(self, backend: str, group: Group) -> bool:
        if backend not in self._process_groups:
            return False
        return group in self._process_groups[backend]

    def init_process_group(
        self, group: Group, backend: str | list[str] | None = None
    ) -> None:
        if backend is None:
            backends = [self._device_backend, "gloo"]
        elif isinstance(backend, str):
            backends = [backend]
        else:
            backends = backend

        for backend in backends:
            if self.has_process_group(backend, group):
                continue
            if self._emulated_rank_groups is not None:
                if backend not in self._emulated_rank_groups:
                    self._emulated_rank_groups[backend] = dist.new_group(
                        [0], backend=backend, timeout=self._pg_timeout
                    )
                self.register_process_group(
                    backend, group, self._emulated_rank_groups[backend]
                )
                continue
            for g in _make_all_groups(group):
                pg = dist.new_group(g, backend=backend, timeout=self._pg_timeout)
                if g == group:
                    self.register_process_group(backend, g, pg)


process_group_manager = ProcessGroupManager()

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

    def init_distributed(
        self,
        mapping: Mapping,
        distributed_init_method: str = "env://",
        backend: str = "nccl",
        timeout: int | None = None,
        device_id: "torch.device | None" = None,
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

            dist.init_process_group(
                backend=backend,
                init_method=distributed_init_method,
                world_size=mapping.world_size,
                rank=mapping.rank,
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

    def has_process_group(self, backend: str, group: Group) -> bool:
        if backend not in self._process_groups:
            return False
        return group in self._process_groups[backend]

    def init_process_group(
        self, group: Group, backend: str | list[str] | None = None
    ) -> None:
        if backend is None:
            backends = ["nccl", "gloo"]
        elif isinstance(backend, str):
            backends = [backend]
        else:
            backends = backend

        for backend in backends:
            if self.has_process_group(backend, group):
                continue
            for g in _make_all_groups(group):
                pg = dist.new_group(g, backend=backend)
                if g == group:
                    self.register_process_group(backend, g, pg)


process_group_manager = ProcessGroupManager()

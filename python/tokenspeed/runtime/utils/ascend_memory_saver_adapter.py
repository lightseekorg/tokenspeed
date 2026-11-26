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

"""TokenSpeed scopes and startup inventory over the standalone NPU saver."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from threading import local
from typing import TYPE_CHECKING, ClassVar, Self

import torch

try:
    from torch_npu_memory_saver import NpuMemorySaver
    from torch_npu_memory_saver.state import AllocationPolicy, PoolKey
except ModuleNotFoundError as exc:
    if exc.name != "torch_npu_memory_saver":
        raise
    raise ModuleNotFoundError(
        "Ascend Memory Saver requires torch-npu-memory-saver==0.1.5. "
        "See docs/serving/ascend-memory-saver.md for the pinned source install.",
        name=exc.name,
    ) from exc

from tokenspeed.runtime.utils.torch_memory_saver_adapter import (
    _NAMED_REGION_POOLS,
    TorchMemorySaverAdapter,
    _defer_communication_initializers,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.distributed.mapping import Mapping


class _NamedRegion(local):
    active: tuple[int, PoolKey] | None = None


_named_region = _NamedRegion()


class AscendMemorySaverAdapter(NpuMemorySaver, TorchMemorySaverAdapter):
    supports_replica_memory_transitions: ClassVar[bool] = True

    @classmethod
    def configure_node_hccl(
        cls,
        mapping: Mapping,
        *,
        emulate_rank_zero: bool,
        on_startup_failure: ExitStack,
    ) -> Self | None:
        """Bind the A3 saver to an existing or new node-local HCCL group."""
        if emulate_rank_zero or mapping.nprocs_per_node == 1:
            return None
        device_name = torch.npu.get_device_properties(mapping.gpu_id).name
        if not device_name.startswith(("Ascend910_93", "Ascend910C")):
            return None
        from tokenspeed.runtime.distributed.process_group_manager import (
            process_group_manager as pg_manager,
        )

        group = pg_manager.get_node_local_device_group(mapping)
        saver = cls()
        # A failed configure can already own resources in this group.
        on_startup_failure.callback(saver.close_hccl)
        saver.configure_hccl(group)
        return saver

    @contextmanager
    def named_region(self, name: str):
        """Back up persistent owners while reusing their active allocator scope."""
        key = PoolKey("weights", AllocationPolicy.CPU_BACKUP, _NAMED_REGION_POOLS[name])
        scope = (torch.npu.current_device(), key)
        previous = _named_region.active
        if previous == scope:
            yield
            return
        with self.allocation_region(key):
            # Only this worker thread can reuse an already-entered PTA scope.
            _named_region.active = scope
            try:
                yield
            finally:
                _named_region.active = previous

    @contextmanager
    def allocation_region(self, key: PoolKey):
        # HCCL initialization belongs to the runtime, not the allocator library.
        with _defer_communication_initializers():
            with super().allocation_region(key):
                yield

    def validate_startup(self, *, models, kv_regions, graphs) -> None:
        """Check all model and KV storage plus captured graph pool ownership."""
        allocations = []
        for role, model in models:
            found = False
            for name, tensor in model.named_parameters():
                if tensor.device.type != "npu" or not tensor.numel():
                    continue
                storage = tensor.untyped_storage()
                allocations.append(
                    ("weights", f"{role}.{name}", storage.data_ptr(), storage.nbytes())
                )
                found = True
            if not found:
                raise RuntimeError(
                    f"Ascend memory saver found no NPU weights for {role}"
                )
        found = False
        for name, tensor in kv_regions.items():
            if tensor.device.type != "npu" or not tensor.numel():
                continue
            storage = tensor.untyped_storage()
            allocations.append(("kv_cache", name, storage.data_ptr(), storage.nbytes()))
            found = True
        if not found:
            raise RuntimeError("Ascend memory saver found no NPU KV backing")
        self.validate_allocations(allocations, graphs=graphs)

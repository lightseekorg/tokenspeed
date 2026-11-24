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

"""Single-rank Ascend memory saver for PTA 2.9 on CANN 9."""

import ctypes
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, ClassVar, Self

import torch

from tokenspeed.runtime.utils.ascend_memory_saver_state import (
    AllocationPolicy,
    OwnedPool,
    PoolKey,
    PoolState,
)
from tokenspeed.runtime.utils.torch_memory_saver_adapter import (
    _NAMED_REGION_POOLS,
    TorchMemorySaverAdapter,
    _defer_communication_initializers,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.distributed.mapping import Mapping


def _check_caching_allocator() -> None:
    if os.environ.get("PYTORCH_NO_NPU_MEMORY_CACHING", "").strip() == "1":
        raise RuntimeError(
            "Ascend memory saver requires PYTORCH_NO_NPU_MEMORY_CACHING=0 "
            "or unset so managed MemPools receive allocations"
        )


def _library_path() -> str:
    value = os.environ.get("TOKENSPEED_NPU_MEMORY_SAVER_LIBRARY")
    if not value:
        raise RuntimeError(
            "Set TOKENSPEED_NPU_MEMORY_SAVER_LIBRARY to the built library"
        )
    path = Path(value)
    if not path.is_absolute() or not path.is_file():
        raise RuntimeError(
            f"Expected an existing absolute saver library path: {value!r}"
        )
    return str(path)


class NativeBackend:
    def __init__(self, library: str):
        self.library = ctypes.CDLL(library)
        self._trimmed_tags: set[str] = set()
        self._hccl_library: ctypes.CDLL | None = None
        self.library.saver_abi_version.argtypes = []
        self.library.saver_abi_version.restype = ctypes.c_int
        self.library.saver_has_callback_fault.argtypes = []
        self.library.saver_has_callback_fault.restype = ctypes.c_int
        if self.library.saver_abi_version() != 3:
            raise RuntimeError("Rebuild the Ascend memory saver library for ABI 3")
        for name in ("saver_pause", "saver_resume"):
            operation = getattr(self.library, name)
            operation.argtypes = [ctypes.c_char_p]
            operation.restype = ctypes.c_int
        for name in ("saver_last_error", "saver_stats"):
            operation = getattr(self.library, name)
            operation.argtypes = []
            operation.restype = ctypes.c_char_p

    def configure_hccl(self, communicator: int) -> None:
        self._hccl_library = ctypes.CDLL("libhccl.so")
        configure = self.library.saver_configure_hccl
        configure.argtypes = [ctypes.c_void_p] * 5
        configure.restype = ctypes.c_int
        functions = (
            "HcclCommSetMemoryRange",
            "HcclCommUnsetMemoryRange",
            "HcclCommActivateCommMemory",
            "HcclCommDeactivateCommMemory",
        )
        addresses = [
            ctypes.cast(getattr(self._hccl_library, name), ctypes.c_void_p)
            for name in functions
        ]
        if configure(communicator, *addresses) != 0:
            raise RuntimeError(self.library.saver_last_error().decode(errors="replace"))

    def close_hccl(self) -> None:
        close = self.library.saver_close_hccl
        close.argtypes = []
        close.restype = ctypes.c_int
        if close() != 0:
            raise RuntimeError(self.library.saver_last_error().decode(errors="replace"))
        self._hccl_library = None

    def check_health(self) -> None:
        if self.library.saver_has_callback_fault():
            error = self.stats().get("callback_error", "diagnostic unavailable")
            raise RuntimeError(
                f"Ascend allocator callback failed; restart this worker: {error}"
            )

    def _call(self, operation, tag: str | None) -> None:
        self.check_health()
        if operation(None if tag is None else tag.encode()) != 0:
            message = self.library.saver_last_error().decode(errors="replace")
            raise RuntimeError(f"Ascend memory transition failed: {message}")
        self.check_health()

    def pause(self, tag: str | None) -> None:
        self._call(self.library.saver_pause, tag)
        if self._trimmed_tags:
            return
        # Eager workers can have no graph allocation. Use the actual segments.
        blocks = self.stats()["blocks"]
        if blocks and all(block["phase"] == "paused" for block in blocks):
            torch.npu.empty_cache()
            self._trimmed_tags = {block["tag"] for block in blocks}

    def resume(self, tag: str | None) -> None:
        if tag is None or tag in self._trimmed_tags:
            # A failed native resume can still restore some mappings.
            self._trimmed_tags.clear()
        self._call(self.library.saver_resume, tag)

    def stats(self) -> dict:
        return json.loads(self.library.saver_stats().decode())


class DeviceState:
    def __init__(self, device: int):
        _check_caching_allocator()
        # Import only in the worker, after its device has been selected.
        try:
            from torch_npu.npu import memory

            self.pluggable_allocator = memory.NPUPluggableAllocator
            self.mem_pool = memory.MemPool
            self.use_mem_pool = memory.use_mem_pool
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "Ascend memory saver requires the PTA pluggable allocator and "
                "MemPool APIs documented in tokenspeed-kernel-npu/memory-saver/README.md"
            ) from exc
        expandable_segments = []
        for item in os.environ.get("PYTORCH_NPU_ALLOC_CONF", "").split(","):
            key, _, value = item.partition(":")
            if key.strip() == "expandable_segments":
                expandable_segments.append(value.strip())
        if not expandable_segments or any(
            value != "False" for value in expandable_segments
        ):
            raise RuntimeError(
                "Set PYTORCH_NPU_ALLOC_CONF=expandable_segments:False before startup"
            )
        self.device = device
        self.library_path = _library_path()
        self.native = NativeBackend(self.library_path)
        self.hccl_group: "torch.distributed.ProcessGroup | None" = None
        self.state = PoolState(
            backend=self.native, make_pool=self.make_pool, synchronize=self.synchronize
        )

    def synchronize(self) -> None:
        if torch.npu.current_device() != self.device:
            raise RuntimeError("Ascend memory saver cannot switch devices")
        torch.npu.synchronize()

    def make_pool(self, key: PoolKey) -> OwnedPool:
        names = {
            ("weights", AllocationPolicy.CPU_BACKUP): "weights_backup_malloc",
            ("weights", AllocationPolicy.DISCARD): "weights_discard_malloc",
            ("kv_cache", AllocationPolicy.DISCARD): "kv_malloc",
            ("acl_graph", AllocationPolicy.CPU_BACKUP): "graph_backup_malloc",
            ("acl_graph", AllocationPolicy.DISCARD): "graph_discard_malloc",
        }
        try:
            symbol = names[(key.tag, key.policy)]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported Ascend allocation policy for {key.tag}: {key.policy}"
            ) from exc
        allocator = self.pluggable_allocator(self.library_path, symbol, "saver_free")
        return OwnedPool(allocator, self.mem_pool(allocator.allocator()))


# Multiple adapters in a worker must share pool IDs and the pause gate.
_states: dict[int, DeviceState] = {}
_state_lock = RLock()


class AscendMemorySaverAdapter(TorchMemorySaverAdapter):
    supports_replica_memory_transitions: ClassVar[bool] = True

    @classmethod
    def configure_node_hccl(
        cls,
        mapping: "Mapping",
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

    def _state(self) -> DeviceState:
        pid = os.getpid()
        device = torch.npu.current_device()
        with _state_lock:
            if pid not in _states:
                _states[pid] = DeviceState(device)
            state = _states[pid]
        if state.device != device:
            raise RuntimeError(
                "Ascend memory saver supports one NPU device per process"
            )
        state.native.check_health()
        return state

    def configure_hccl(self, group: "torch.distributed.ProcessGroup") -> None:
        """Bind a local HCCL domain before managed allocations can be registered implicitly."""
        import torch.distributed as dist

        state = self._state()
        with _state_lock:
            if state.hccl_group is not None:
                raise RuntimeError("HCCL memory registration is already configured")
            marker = torch.ones(
                1, dtype=torch.float32, device=torch.device("npu", state.device)
            )
            dist.all_reduce(marker, group=group)
            state.synchronize()
            backend = group._get_backend(marker.device)
            state.hccl_group = group  # Keep the communicator alive through cleanup.
            state.native.configure_hccl(backend.get_hccl_comm(state.device))

    def close_hccl(self) -> None:
        """Detach ranges while the communicator is alive, including on failed cleanup."""
        # Cleanup must remain available after an allocator callback fault.
        with _state_lock:
            state = _states.get(os.getpid())
            if state is not None and state.hccl_group is not None:
                state.synchronize()
                state.native.close_hccl()
                state.hccl_group = None

    @contextmanager
    def configure_subprocess(self):
        _check_caching_allocator()
        _library_path()
        yield

    def region(self, tag: str | None = None, enable_cpu_backup: bool = False):
        if tag is None:
            raise ValueError("Ascend allocations require an explicit memory tag")
        policy = (
            AllocationPolicy.CPU_BACKUP
            if enable_cpu_backup
            else AllocationPolicy.DISCARD
        )
        return self.allocation_region(PoolKey(tag, policy, tag))

    @contextmanager
    def named_region(self, name: str):
        """Back up persistent owners while reusing their active allocator scope."""
        key = PoolKey("weights", AllocationPolicy.CPU_BACKUP, _NAMED_REGION_POOLS[name])
        state = self._state()
        # Aliases share one live PTA scope across all adapters in this worker.
        # Keep the gate held so another thread cannot borrow its allocator route.
        with state.state.lock:
            if state.state.active_key == key:
                yield
                return
        with self.allocation_region(key):
            yield

    @contextmanager
    def allocation_region(self, key: PoolKey):
        state = self._state()
        with _defer_communication_initializers():
            with state.state.region(key) as owned:
                route = state.use_mem_pool(owned.pool) if owned else nullcontext()
                with route:
                    yield
            state.native.check_health()

    @contextmanager
    def npu_graph(
        self,
        graph,
        *,
        pool,
        stream,
        tag: str,
        policy: str,
        pool_scope: str,
        auto_dispatch_capture: bool,
    ):
        state = self._state()
        key = PoolKey(tag, AllocationPolicy(policy), pool_scope)
        with state.state.region(key) as owned:
            selected = state.state.graph_pool(owned=owned, requested=pool)
            with torch.npu.graph(
                graph,
                pool=selected,
                stream=stream,
                auto_dispatch_capture=auto_dispatch_capture,
            ):
                yield
        state.native.check_health()

    def pause(self, tag: str | None = None):
        self._state().state.pause(tag)

    def resume(self, tag: str | None = None):
        self._state().state.resume(tag)

    def assert_active(self):
        self._state().state.assert_active()

    def validate_startup(
        self,
        *,
        models: Sequence[tuple[str, torch.nn.Module]],
        kv_regions: Mapping[str, torch.Tensor],
        graphs: Iterable,
    ) -> None:
        """Check named models, cache backing regions and captured graph pools.

        NPU storage must belong to a live VMM segment with its release tag.
        ``graphs`` is empty for eager startup. Raises RuntimeError on bypass.
        """
        state = self._state()
        state.synchronize()
        state.native.check_health()
        blocks = state.native.stats()["blocks"]

        def require_managed(tag: str, name: str, pointer: int, size: int) -> None:
            for block in blocks:
                base = int(block["ptr"], 16)
                if (
                    block["tag"] == tag
                    and block["device"] == state.device
                    and block["phase"] == "active"
                    and block["reserved"]
                    and block["physical_live"]
                    and block["mapped"]
                    and block["contents_valid"]
                    and base <= pointer
                    and pointer + size <= base + block["aligned_bytes"]
                ):
                    return
            raise RuntimeError(
                f"Ascend memory saver {tag} storage {name!r} at "
                f"{pointer:#x} ({size} bytes) bypassed its live managed pool"
            )

        for role, model in models:
            found = False
            for name, tensor in model.named_parameters():
                if tensor.device.type != "npu" or not tensor.numel():
                    continue
                # Storage bounds cover tied, offset and noncontiguous views.
                storage = tensor.untyped_storage()
                require_managed(
                    "weights", f"{role}.{name}", storage.data_ptr(), storage.nbytes()
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
            require_managed("kv_cache", name, storage.data_ptr(), storage.nbytes())
            found = True
        if not found:
            raise RuntimeError("Ascend memory saver found no NPU KV backing")

        graph_pools = {graph.pool() for graph in graphs}
        if graph_pools:
            owned = {
                pool.pool.id
                for key, pool in state.state.pools.items()
                if key.tag == "acl_graph"
            }
            if not graph_pools <= owned:
                raise RuntimeError("ACL Graph capture bypassed its owned managed pool")
            # PTA retains graph scratch in its pool after Python tensors die.
            for segment in torch.npu.memory_snapshot():
                pool = tuple(segment["segment_pool_id"])
                if segment["device"] != state.device or pool not in graph_pools:
                    continue
                require_managed(
                    "acl_graph", str(pool), segment["address"], segment["total_size"]
                )
        state.native.check_health()

    def stats(self):
        return self._state().native.stats()

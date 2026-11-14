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

"""Exercise real adapter/state gates with CPU replacements for absent NPU APIs."""

import builtins
import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Event, local
from types import ModuleType

import pytest
import torch

OWNERS = ("executor_inputs", "runtime_states", "attention_state", "workspace")


@dataclass
class CpuAllocation:
    tensor: torch.Tensor
    tag: str | None
    backup: bool
    saved: torch.Tensor | None = None


class CpuFunction:
    """Represent one missing native ABI entry point used by NativeBackend."""

    def __init__(self, operation):
        self.operation = operation
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.operation(*args)


class CpuNativeLibrary:
    def __init__(self, memory):
        self.memory = memory
        self.fault = False
        self.fail_pause = False
        self.saver_abi_version = CpuFunction(lambda: 2)
        self.saver_has_callback_fault = CpuFunction(lambda: int(self.fault))
        self.saver_last_error = CpuFunction(lambda: b"injected native pause failure")
        self.saver_stats = CpuFunction(
            lambda: json.dumps({"callback_error": "injected callback fault"}).encode()
        )
        self.saver_pause = CpuFunction(self.pause)
        self.saver_resume = CpuFunction(self.resume)

    def pause(self, tag):
        if self.fail_pause:
            return 1
        tag = tag.decode() if tag is not None else None
        for allocation in self.memory.allocations:
            if allocation.tag is not None and (tag is None or allocation.tag == tag):
                if allocation.backup:
                    allocation.saved = allocation.tensor.clone()
                allocation.tensor.zero_()
        return 0

    def resume(self, tag):
        tag = tag.decode() if tag is not None else None
        for allocation in self.memory.allocations:
            if allocation.tag is not None and (tag is None or allocation.tag == tag):
                if allocation.backup:
                    allocation.tensor.copy_(allocation.saved)
                else:
                    allocation.tensor.fill_(-1)
        return 0


class CpuAllocator:
    def __init__(self, library, malloc, free):
        self.library = library
        self.malloc = malloc
        self.free = free

    def allocator(self):
        return self


@dataclass
class CpuPool:
    allocator: CpuAllocator
    id: tuple[int, int]


class CpuNpu:
    """Represent PTA routing and physical storage without replacing state gates."""

    def __init__(self):
        self.device = 0
        self.routes = local()
        self.pools = []
        self.allocations = []
        self.library = CpuNativeLibrary(self)

    def current_device(self):
        return self.device

    def synchronize(self):
        pass

    def mem_pool(self, allocator):
        pool = CpuPool(allocator, (0, len(self.pools) + 1))
        self.pools.append(pool)
        return pool

    @property
    def active_pool(self):
        return self.routes.__dict__.get("pool")

    @contextmanager
    def use_mem_pool(self, pool):
        if self.active_pool is not None:
            raise RuntimeError("PTA cannot nest MemPool routes")
        self.routes.pool = pool
        try:
            yield
        finally:
            self.routes.pool = None

    @contextmanager
    def graph(self, graph, *, pool, stream, auto_dispatch_capture):
        del graph, pool, stream, auto_dispatch_capture
        yield

    def allocate(self, value):
        pool = self.active_pool
        symbols = {
            "weights_backup_malloc": ("weights", True),
            "weights_discard_malloc": ("weights", False),
            "kv_malloc": ("kv_cache", False),
            "graph_backup_malloc": ("acl_graph", True),
            "graph_discard_malloc": ("acl_graph", False),
        }
        tag, backup = symbols[pool.allocator.malloc] if pool else (None, False)
        tensor = torch.tensor([value], dtype=torch.int64)
        self.allocations.append(CpuAllocation(tensor, tag, backup))
        return tensor


@dataclass
class Runtime:
    module: ModuleType
    shared: ModuleType
    memory: CpuNpu
    adapter: object


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    # The unavailable backend is replaced at its device, allocator and native
    # boundaries. DeviceState, NativeBackend and PoolState remain real.
    memory = CpuNpu()
    torch_npu_module = ModuleType("torch_npu")
    npu_module = ModuleType("torch_npu.npu")
    memory_module = ModuleType("torch_npu.npu.memory")
    memory_module.NPUPluggableAllocator = CpuAllocator
    memory_module.MemPool = memory.mem_pool
    memory_module.use_mem_pool = memory.use_mem_pool
    npu_module.memory = memory_module
    torch_npu_module.npu = npu_module
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu_module)
    monkeypatch.setitem(sys.modules, "torch_npu.npu", npu_module)
    monkeypatch.setitem(sys.modules, "torch_npu.npu.memory", memory_module)
    monkeypatch.setitem(sys.modules, "torch_memory_saver", None)
    monkeypatch.setattr(torch, "npu", memory, raising=False)
    monkeypatch.setenv("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:False")
    monkeypatch.delenv("PYTORCH_NO_NPU_MEMORY_CACHING", raising=False)
    library = tmp_path / "libascend_memory_saver.so"
    library.touch()
    monkeypatch.setenv("TOKENSPEED_NPU_MEMORY_SAVER_LIBRARY", str(library))
    original_import = builtins.__import__

    def import_without_cuda_saver(name, *args, **kwargs):
        if name == "torch_memory_saver" or name.startswith("torch_memory_saver."):
            raise AssertionError("NPU allocation imported the CUDA saver")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_cuda_saver)

    source = Path(__file__).resolve().parents[2] / "python/tokenspeed/runtime/utils"
    modules = []
    for name in (
        "ascend_memory_saver_state",
        "torch_memory_saver_adapter",
        "ascend_memory_saver_adapter",
    ):
        module_name = f"tokenspeed.runtime.utils.{name}"
        spec = importlib.util.spec_from_file_location(
            module_name, source / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, module_name, module)
        spec.loader.exec_module(module)
        modules.append(module)
    shared, module = modules[1:]
    monkeypatch.setattr(module.ctypes, "CDLL", lambda path: memory.library)
    return Runtime(module, shared, memory, module.AscendMemorySaverAdapter())


def test_all_owners_share_one_pool_across_adapter_instances(runtime):
    memory, adapter = runtime.memory, runtime.adapter
    first_pool = None
    for owner in OWNERS:
        for current in (adapter, runtime.module.AscendMemorySaverAdapter()):
            with current.named_region(owner):
                first_pool = first_pool or memory.active_pool
                assert memory.active_pool is first_pool
                memory.allocate(11)
    assert len(memory.pools) == 1
    state = adapter._state().state
    key = runtime.module.PoolKey(
        "weights", runtime.module.AllocationPolicy.CPU_BACKUP, "persistent_buffers"
    )
    assert state.pools[key].pool is first_pool


def test_unknown_owner_is_rejected_before_allocator_setup(runtime, monkeypatch):
    monkeypatch.delenv("TOKENSPEED_NPU_MEMORY_SAVER_LIBRARY")
    with pytest.raises(KeyError, match="unregistered"):
        with runtime.adapter.named_region("unregistered"):
            pytest.fail("unknown owner became usable")
    assert runtime.memory.pools == []


def test_nested_aliases_defer_communication_until_outer_exit(runtime):
    memory, adapter = runtime.memory, runtime.adapter
    other = runtime.module.AscendMemorySaverAdapter()
    outputs = []

    def initialize(value):
        # Communication storage must be resident even though it was requested
        # from a managed owner. The allocation gate must have exited as well.
        assert memory.active_pool is None
        adapter.pause("weights")
        adapter.resume("weights")
        outputs.append(memory.allocate(value))

    with adapter.named_region("executor_inputs"):
        persistent = memory.allocate(5)
        for owner, value in zip(OWNERS, (7, 11, 13, 17), strict=True):
            with other.named_region(owner):
                runtime.shared.initialize_communication(
                    lambda value=value: initialize(value)
                )
                assert outputs == []
        assert outputs == []
    assert [tensor.item() for tensor in outputs] == [7, 11, 13, 17]
    adapter.pause("weights")
    adapter.resume("weights")
    assert persistent.item() == 5
    assert [tensor.item() for tensor in outputs] == [7, 11, 13, 17]


def test_failed_outer_scope_drops_nested_callbacks_and_cleans_allocator(runtime):
    outputs = []
    with pytest.raises(ValueError, match="load failed"):
        with runtime.adapter.named_region("runtime_states"):
            with runtime.adapter.named_region("workspace"):
                runtime.shared.initialize_communication(lambda: outputs.append(3))
            raise ValueError("load failed")
    assert outputs == []
    assert runtime.memory.active_pool is None
    runtime.adapter.pause("weights")
    runtime.adapter.resume("weights")
    with runtime.adapter.named_region("attention_state"):
        assert runtime.memory.allocate(19).item() == 19
    runtime.shared.initialize_communication(lambda: outputs.append(5))
    assert outputs == [5]


def test_callback_failure_propagates_after_cleanup_and_next_scope_works(runtime):
    def fail():
        assert runtime.memory.active_pool is None
        runtime.adapter.pause("weights")
        runtime.adapter.resume("weights")
        raise ValueError("communication failed")

    with pytest.raises(ValueError, match="communication failed"):
        with runtime.adapter.named_region("workspace"):
            runtime.shared.initialize_communication(fail)
    with runtime.adapter.named_region("runtime_states"):
        assert runtime.memory.allocate(23).item() == 23


@pytest.mark.parametrize(
    ("tag", "backup"),
    [("weights", False), ("weights", True), ("kv_cache", False), ("acl_graph", False)],
)
@pytest.mark.parametrize("named_outer", [False, True])
def test_named_scope_rejects_other_nested_regions(runtime, tag, backup, named_outer):
    named = runtime.adapter.named_region("executor_inputs")
    ordinary = runtime.adapter.region(tag=tag, enable_cpu_backup=backup)
    outer, inner = (named, ordinary) if named_outer else (ordinary, named)
    with outer:
        with pytest.raises(RuntimeError, match="Cannot nest"):
            with inner:
                pytest.fail("different pool nested")
    with runtime.adapter.named_region("workspace"):
        runtime.memory.allocate(29)


def test_named_scope_rejects_graph_capture_and_active_transitions(runtime):
    adapter = runtime.adapter
    with adapter.named_region("attention_state"):
        pool = runtime.memory.active_pool.id
        with pytest.raises(RuntimeError, match="Cannot nest"):
            with adapter.npu_graph(
                object(),
                pool=pool,
                stream=None,
                tag="acl_graph",
                policy="discard",
                pool_scope="decode",
                auto_dispatch_capture=False,
            ):
                pytest.fail("graph entered a persistent buffer pool")
        for operation in (adapter.pause, adapter.resume):
            with pytest.raises(RuntimeError, match="active allocation"):
                operation("weights")
    # Outside a named scope, graph capture still cannot borrow its pool ID.
    with pytest.raises(ValueError, match="not the managed pool"):
        with adapter.npu_graph(
            object(),
            pool=pool,
            stream=None,
            tag="acl_graph",
            policy="discard",
            pool_scope="decode",
            auto_dispatch_capture=False,
        ):
            pytest.fail("graph borrowed a persistent buffer pool")


def test_weights_sleep_restores_persistent_buffers_beside_discarded_parameters(runtime):
    memory, adapter = runtime.memory, runtime.adapter
    with adapter.named_region("workspace"):
        persistent = memory.allocate(31)
        persistent_pool = memory.active_pool
    with adapter.region(tag="weights", enable_cpu_backup=False):
        weight = memory.allocate(37)
        assert memory.active_pool is not persistent_pool
    with adapter.region(tag="weights", enable_cpu_backup=True):
        backed_weight = memory.allocate(41)
        assert memory.active_pool is not persistent_pool
    with adapter.region(tag="kv_cache", enable_cpu_backup=False):
        kv = memory.allocate(43)
    with adapter.allocation_region(
        runtime.module.PoolKey(
            "acl_graph", runtime.module.AllocationPolicy.DISCARD, "decode"
        )
    ):
        graph = memory.allocate(47)
    addresses = [tensor.data_ptr() for tensor in (persistent, weight, backed_weight)]
    adapter.pause("weights")
    with pytest.raises(RuntimeError, match="paused or needs recovery"):
        with adapter.named_region("executor_inputs"):
            pytest.fail("paused persistent storage was usable")
    adapter.resume("weights")
    assert persistent.item() == 31
    assert weight.item() == -1
    assert backed_weight.item() == 41
    assert kv.item() == 43
    assert graph.item() == 47
    assert [
        tensor.data_ptr() for tensor in (persistent, weight, backed_weight)
    ] == addresses
    with adapter.named_region("runtime_states"):
        assert memory.active_pool is persistent_pool


def test_failed_weight_transition_blocks_named_scope_until_recovery(runtime):
    runtime.memory.library.fail_pause = True
    with pytest.raises(RuntimeError, match="injected native pause failure"):
        runtime.adapter.pause("weights")
    with pytest.raises(RuntimeError, match="paused or needs recovery"):
        with runtime.adapter.named_region("workspace"):
            pytest.fail("failed tag became usable")
    runtime.memory.library.fail_pause = False
    runtime.adapter.resume("weights")
    with runtime.adapter.named_region("workspace"):
        runtime.memory.allocate(43)


@pytest.mark.parametrize("nested", [False, True])
def test_native_callback_fault_blocks_every_named_entry(runtime, nested):
    @contextmanager
    def outer():
        if nested:
            with runtime.adapter.named_region("executor_inputs"):
                yield
        else:
            yield

    with outer():
        runtime.memory.library.fault = True
        with pytest.raises(RuntimeError, match="callback failed; restart"):
            with runtime.adapter.named_region("workspace"):
                pytest.fail("faulted allocator became usable")
        runtime.memory.library.fault = False
    with runtime.adapter.named_region("workspace"):
        runtime.memory.allocate(47)


def test_named_scope_allows_weights_awake_while_kv_and_graph_remain_asleep(runtime):
    runtime.adapter.pause("kv_cache")
    runtime.adapter.pause("acl_graph")
    with runtime.adapter.named_region("runtime_states"):
        assert runtime.memory.allocate(53).item() == 53


def test_named_scope_refuses_device_switch_including_nested_alias(runtime):
    with runtime.adapter.named_region("executor_inputs"):
        runtime.memory.device = 1
        with pytest.raises(RuntimeError, match="one NPU device per process"):
            with runtime.adapter.named_region("workspace"):
                pytest.fail("different device borrowed the active pool")
        runtime.memory.device = 0


def test_other_thread_waits_for_failed_outer_scope_then_uses_own_route(runtime):
    attempting, entered = Event(), Event()

    def allocate_from_other_thread():
        attempting.set()
        with runtime.module.AscendMemorySaverAdapter().named_region("workspace"):
            entered.set()
            return runtime.memory.allocate(59)

    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(ValueError, match="outer failed"):
            with runtime.adapter.named_region("executor_inputs"):
                runtime.memory.allocate(61)
                future = executor.submit(allocate_from_other_thread)
                assert attempting.wait(timeout=5)
                assert not entered.wait(timeout=0.1)
                raise ValueError("outer failed")
        assert future.result(timeout=5).item() == 59
    assert entered.is_set()
    assert runtime.memory.active_pool is None
    assert len(runtime.memory.pools) == 1

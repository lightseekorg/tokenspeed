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

"""Validate real CPU storage addresses with an unavailable NPU represented locally."""

import builtins
import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from threading import Event, local
from types import ModuleType

import pytest
import torch

OWNERS = ("executor_inputs", "runtime_states", "attention_state", "workspace")


def saver_modules(monkeypatch):
    # Import the allocator boundary without loading the accelerator runtime.
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
    return modules


@dataclass(frozen=True)
class NpuDevice:
    type: str = "npu"
    index: int = 0


class NpuParameter(torch.nn.Parameter):
    """Expose an NPU device while retaining actual CPU-backed tensor storage."""

    @property
    def device(self):
        return NpuDevice()


@dataclass
class CpuGraph:
    pool_id: tuple[int, int]

    def pool(self):
        return self.pool_id


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
        self.blocks = []
        self.saver_abi_version = CpuFunction(lambda: 2)
        self.saver_has_callback_fault = CpuFunction(lambda: int(self.fault))
        self.saver_last_error = CpuFunction(lambda: b"injected native pause failure")
        self.saver_stats = CpuFunction(
            lambda: json.dumps(
                {"callback_error": "injected callback fault", "blocks": self.blocks}
            ).encode()
        )
        self.saver_pause = CpuFunction(self.pause)
        self.saver_resume = CpuFunction(self.resume)

    def own(self, tensor, tag):
        storage = tensor.untyped_storage()
        block = {
            "ptr": hex(storage.data_ptr()),
            "aligned_bytes": storage.nbytes(),
            "tag": tag,
            "device": 0,
            "phase": "active",
            "reserved": True,
            "physical_live": True,
            "mapped": True,
            "contents_valid": True,
        }
        self.blocks.append(block)
        return block

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
        self.segments = []
        self.snapshot_available = True
        self.library = CpuNativeLibrary(self)

    def memory_snapshot(self):
        if not self.snapshot_available:
            raise RuntimeError("memory snapshot unavailable")
        return self.segments

    def capture_graph(self, adapter):
        graph = CpuGraph((0, 0))
        with adapter.npu_graph(
            graph,
            pool=None,
            stream=None,
            tag="acl_graph",
            policy="discard",
            pool_scope="decode",
            auto_dispatch_capture=False,
        ):
            pass
        return graph

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
        graph.pool_id = pool
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

    _, shared, module = saver_modules(monkeypatch)
    monkeypatch.setattr(module.ctypes, "CDLL", lambda path: memory.library)
    return Runtime(module, shared, memory, module.AscendMemorySaverAdapter())


def test_all_owners_share_one_pool_across_adapter_instances(runtime, monkeypatch):
    memory, adapter = runtime.memory, runtime.adapter
    with monkeypatch.context() as missing_library:
        missing_library.delenv("TOKENSPEED_NPU_MEMORY_SAVER_LIBRARY")
        with pytest.raises(KeyError, match="unregistered"):
            with adapter.named_region("unregistered"):
                pytest.fail("unknown owner became usable")
    assert memory.pools == []
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


def test_nested_aliases_defer_communication_until_outer_exit(runtime):
    memory, adapter = runtime.memory, runtime.adapter
    other = runtime.module.AscendMemorySaverAdapter()
    outputs = []

    def initialize(value):
        # Communication storage must be resident even though it was requested
        # from a managed owner. The allocation gate must have exited as well.
        assert memory.active_pool is None
        outputs.append(memory.allocate(value))

    with adapter.named_region("executor_inputs"):
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
    assert [tensor.item() for tensor in outputs] == [7, 11, 13, 17]


@pytest.mark.parametrize("failure", ["body", "initializer"])
def test_scope_or_initializer_failure_cleans_nested_allocator(runtime, failure):
    outputs = []

    def initialize():
        outputs.append(3)
        assert runtime.memory.active_pool is None
        runtime.adapter.pause("weights")
        runtime.adapter.resume("weights")
        raise ValueError("load failed")

    with pytest.raises(ValueError, match="load failed"):
        with runtime.adapter.named_region("runtime_states"):
            with runtime.adapter.named_region("workspace"):
                runtime.shared.initialize_communication(initialize)
            if failure == "body":
                raise ValueError("load failed")
    expected = [] if failure == "body" else [3]
    assert outputs == expected
    assert runtime.memory.active_pool is None
    runtime.adapter.pause("weights")
    runtime.adapter.resume("weights")
    with runtime.adapter.named_region("attention_state"):
        assert runtime.memory.allocate(19).item() == 19
    runtime.shared.initialize_communication(lambda: outputs.append(5))
    assert outputs == expected + [5]


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
    with adapter.region(tag="acl_graph", enable_cpu_backup=False):
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

    from tokenspeed.runtime.engine.memory_occupation import memory_regions

    regions = memory_regions(
        adapter._state(),
        adapter,
        graph_tag="acl_graph",
        nccl_suspendable=False,
        synchronize_kv_before_registration=True,
    )
    assert list(regions) == ["kv_cache", "weights", "acl_graph"]
    regions["acl_graph"].release()
    with pytest.raises(RuntimeError, match="unavailable"):
        adapter.assert_active()
    assert (persistent.item(), kv.item(), graph.item()) == (31, 43, 0)
    regions["acl_graph"].resume()
    adapter.assert_active()
    assert (persistent.item(), kv.item(), graph.item()) == (31, 43, -1)


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
    outer = runtime.adapter.named_region("executor_inputs") if nested else nullcontext()
    with outer:
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


@pytest.fixture
def adapter_module(runtime):
    return runtime.module


@pytest.fixture
def startup(runtime):
    adapter, state = runtime.adapter, runtime.memory
    model = torch.nn.Module()
    model.register_parameter(
        "weight", NpuParameter(torch.arange(16).reshape(4, 4).float(), False)
    )
    kv = NpuParameter(torch.zeros(128, dtype=torch.uint8), False)
    state.library.own(model.weight, "weights")
    state.library.own(kv, "kv_cache")
    args = {"models": [("target", model)], "kv_regions": {"arena": kv}, "graphs": []}
    return adapter, state, args


def test_startup_accepts_eager_weights_and_kv_without_graph_snapshot(startup):
    adapter, state, args = startup
    state.snapshot_available = False
    adapter.validate_startup(**args)


def test_startup_accepts_tied_offset_noncontiguous_and_empty_parameters(startup):
    adapter, state, args = startup
    model = args["models"][0][1]
    model.register_parameter("tied", model.weight)
    model.register_parameter("transpose", NpuParameter(model.weight.t(), False))
    model.register_parameter("offset", NpuParameter(model.weight[1:, ::2], False))
    model.register_parameter("empty", NpuParameter(torch.empty(0), False))
    model.register_parameter("host", torch.nn.Parameter(torch.ones(1)))
    draft = torch.nn.Module()
    draft.register_parameter("shared", model.weight)
    draft.register_parameter("independent", NpuParameter(torch.ones(7), False))
    state.library.own(draft.independent, "weights")
    args["models"].append(("draft", draft))
    args["kv_regions"]["host_history"] = torch.zeros(7)
    args["kv_regions"]["empty"] = NpuParameter(torch.empty(0), False)
    adapter.validate_startup(**args)


@pytest.mark.parametrize("tag", ["weights", "kv_cache"])
def test_startup_requires_actual_tensor_backing_even_with_other_native_blocks(
    startup, tag
):
    adapter, state, args = startup
    state.library.blocks[:] = [
        block for block in state.library.blocks if block["tag"] != tag
    ]
    state.library.own(torch.zeros(8), tag)
    with pytest.raises(RuntimeError, match=f"{tag} storage.*bypassed"):
        adapter.validate_startup(**args)


@pytest.mark.parametrize("role", ["target", "draft"])
def test_startup_rejects_individual_parameter_outside_managed_storage(startup, role):
    adapter, _, args = startup
    if role == "draft":
        args["models"].append((role, torch.nn.Module()))
    model = args["models"][-1][1]
    model.register_parameter("unmanaged", NpuParameter(torch.ones(2), False))
    with pytest.raises(RuntimeError, match=f"{role}.unmanaged.*bypassed"):
        adapter.validate_startup(**args)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("device", 1),
        ("phase", "releasing"),
        ("reserved", False),
        ("physical_live", False),
        ("mapped", False),
        ("contents_valid", False),
        ("aligned_bytes", 1),
        ("tag", "acl_graph"),
    ],
)
def test_startup_rejects_wrong_or_incomplete_native_backing(startup, field, value):
    adapter, state, args = startup
    state.library.blocks[0][field] = value
    with pytest.raises(RuntimeError, match="weights storage.*bypassed"):
        adapter.validate_startup(**args)


@pytest.mark.parametrize("missing", ["weights", "kv_cache"])
def test_startup_rejects_empty_required_storage(startup, missing):
    adapter, _, args = startup
    if missing == "weights":
        args["models"] = [("target", torch.nn.Module())]
    else:
        args["kv_regions"] = {"empty": NpuParameter(torch.empty(0), False)}
    with pytest.raises(RuntimeError, match="found no NPU"):
        adapter.validate_startup(**args)


def test_startup_checks_every_pta_segment_of_actual_captured_pool(startup):
    adapter, state, args = startup
    graph = state.capture_graph(adapter)
    args["graphs"] = [graph, graph]
    scratch = torch.zeros(96)
    block = state.library.own(scratch, "acl_graph")
    segment = {
        "device": 0,
        "segment_pool_id": graph.pool(),
        "address": scratch.data_ptr(),
        "total_size": scratch.nbytes,
    }
    # Two PTA segments may be subranges of one physical VMM allocation.
    state.segments = [
        segment | {"total_size": 128},
        segment
        | {"address": scratch.data_ptr() + 128, "total_size": scratch.nbytes - 128},
    ]
    adapter.validate_startup(**args)
    block["tag"] = "weights"
    with pytest.raises(RuntimeError, match="acl_graph storage.*bypassed"):
        adapter.validate_startup(**args)


def test_startup_rejects_captured_pool_with_unmanaged_backing(startup):
    adapter, state, args = startup
    args["graphs"] = [CpuGraph((0, 99))]
    with pytest.raises(RuntimeError, match="owned managed pool"):
        adapter.validate_startup(**args)
    graph = state.capture_graph(adapter)
    args["graphs"] = [graph]
    adapter.validate_startup(**args)
    state.segments = [
        {
            "device": 0,
            "segment_pool_id": graph.pool(),
            "address": 123,
            "total_size": 512,
        }
    ]
    with pytest.raises(RuntimeError, match="acl_graph storage.*bypassed"):
        adapter.validate_startup(**args)


def test_startup_rejects_latched_native_fault(startup):
    adapter, state, args = startup
    state.library.fault = True
    with pytest.raises(RuntimeError, match="restart this worker"):
        adapter.validate_startup(**args)


def test_uncached_mode_rejected_before_launcher_or_device_setup(
    adapter_module, monkeypatch
):
    monkeypatch.setenv("PYTORCH_NO_NPU_MEMORY_CACHING", "1")
    monkeypatch.delenv("TOKENSPEED_NPU_MEMORY_SAVER_LIBRARY", raising=False)
    with pytest.raises(RuntimeError, match="PYTORCH_NO_NPU_MEMORY_CACHING=0"):
        with adapter_module.AscendMemorySaverAdapter().configure_subprocess():
            pass
    with pytest.raises(RuntimeError, match="PYTORCH_NO_NPU_MEMORY_CACHING=0"):
        adapter_module.DeviceState(0)


@pytest.mark.parametrize("caching", [None, "0"])
def test_caching_mode_can_configure_subprocess(
    adapter_module, monkeypatch, tmp_path, caching
):
    if caching is None:
        monkeypatch.delenv("PYTORCH_NO_NPU_MEMORY_CACHING", raising=False)
    else:
        monkeypatch.setenv("PYTORCH_NO_NPU_MEMORY_CACHING", caching)
    library = tmp_path / "libascend_memory_saver.so"
    library.touch()
    monkeypatch.setenv("TOKENSPEED_NPU_MEMORY_SAVER_LIBRARY", str(library))
    with adapter_module.AscendMemorySaverAdapter().configure_subprocess():
        pass

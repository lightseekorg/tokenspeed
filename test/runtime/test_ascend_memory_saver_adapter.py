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

"""TokenSpeed ownership rules over an unavailable NPU allocator."""

import importlib.util
import sys
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from test.ci_system.ci_register import register_cuda_ci
from types import ModuleType, SimpleNamespace

import pytest
import torch

register_cuda_ci(est_time=3, suite="runtime-1gpu")


class AllocationPolicy(StrEnum):
    CPU_BACKUP = "cpu_backup"
    DISCARD = "discard"


@dataclass(frozen=True)
class PoolKey:
    tag: str
    policy: AllocationPolicy
    scope: str


class RecordingNpuSaver:
    """Stand in for the optional device backend at its runtime interface."""

    def __init__(self):
        self.active = False
        self.inventory = None
        self.scopes = []
        self.hccl_group = None

    def configure_hccl(self, group):
        self.hccl_group = group
        group.registered = True
        if group.fail_registration:
            raise RuntimeError("HCCL registration failed after acquiring the buffer")

    def close_hccl(self):
        if self.hccl_group is not None:
            self.hccl_group.registered = False
            self.hccl_group = None

    def region(self, tag, enable_cpu_backup):
        policy = (
            AllocationPolicy.CPU_BACKUP
            if enable_cpu_backup
            else AllocationPolicy.DISCARD
        )
        return self.allocation_region(PoolKey(tag, policy, tag))

    @contextmanager
    def allocation_region(self, key):
        self.scopes.append(key)
        self.active = True
        try:
            yield
        finally:
            self.active = False

    def validate_allocations(self, allocations, *, graphs):
        self.inventory = list(allocations), list(graphs)


@pytest.fixture
def saver_modules(monkeypatch):
    package = ModuleType("torch_npu_memory_saver")
    package.NpuMemorySaver = RecordingNpuSaver
    state = ModuleType("torch_npu_memory_saver.state")
    state.AllocationPolicy = AllocationPolicy
    state.PoolKey = PoolKey
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, state.__name__, state)
    monkeypatch.setitem(sys.modules, "torch_memory_saver", None)

    # Load the adapter in isolation from the rest of the accelerator runtime.
    source = Path(__file__).resolve().parents[2] / "python/tokenspeed/runtime"
    modules = []
    for name in (
        "distributed.mapping",
        "distributed.process_group_manager",
        "utils.torch_memory_saver_adapter",
        "utils.ascend_memory_saver_adapter",
    ):
        module_name = f"tokenspeed.runtime.{name}"
        spec = importlib.util.spec_from_file_location(
            module_name, source / f"{name.replace('.', '/')}.py"
        )
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, module_name, module)
        spec.loader.exec_module(module)
        if name.startswith("utils."):
            modules.append(module)
    return modules


class NpuParameter(torch.nn.Parameter):
    """Represent an NPU parameter with real storage on the CPU test host."""

    @property
    def device(self):
        return NpuDevice()


@dataclass(frozen=True)
class NpuDevice:
    type: str = "npu"


@dataclass
class Startup:
    saver: RecordingNpuSaver
    models: list
    kv_regions: dict
    graphs: list

    def validate(self):
        self.saver.validate_startup(
            models=self.models, kv_regions=self.kv_regions, graphs=self.graphs
        )


@pytest.fixture
def startup(saver_modules):
    _, adapter = saver_modules
    model = torch.nn.Module()
    model.weight = NpuParameter(torch.arange(16).reshape(4, 4).float(), False)
    kv = NpuParameter(torch.zeros(128, dtype=torch.uint8), False)
    return Startup(
        adapter.AscendMemorySaverAdapter(), [("target", model)], {"arena": kv}, []
    )


def test_startup_inventories_storage_for_target_draft_kv_and_graphs(startup):
    target = startup.models[0][1]
    target.tied = target.weight
    target.transpose = NpuParameter(target.weight.t(), False)
    target.offset = NpuParameter(target.weight[1:, ::2], False)
    target.empty = NpuParameter(torch.empty(0), False)
    target.host = torch.nn.Parameter(torch.ones(1))
    draft = torch.nn.Module()
    draft.shared = target.weight
    draft.independent = NpuParameter(torch.ones(7), False)
    startup.models.append(("draft", draft))
    startup.kv_regions["host_history"] = torch.zeros(7)
    startup.kv_regions["empty"] = NpuParameter(torch.empty(0), False)
    startup.graphs.append(object())

    startup.validate()
    weights = target.weight.untyped_storage()
    independent = draft.independent.untyped_storage()
    kv = startup.kv_regions["arena"].untyped_storage()
    assert startup.saver.inventory == (
        [
            ("weights", "target.weight", weights.data_ptr(), weights.nbytes()),
            ("weights", "target.transpose", weights.data_ptr(), weights.nbytes()),
            ("weights", "target.offset", weights.data_ptr(), weights.nbytes()),
            ("weights", "draft.shared", weights.data_ptr(), weights.nbytes()),
            (
                "weights",
                "draft.independent",
                independent.data_ptr(),
                independent.nbytes(),
            ),
            ("kv_cache", "arena", kv.data_ptr(), kv.nbytes()),
        ],
        startup.graphs,
    )


@pytest.mark.parametrize("missing", ["target", "draft", "kv_cache"])
def test_startup_rejects_empty_required_storage(startup, missing):
    if missing == "target":
        startup.models[0] = ("target", torch.nn.Module())
    elif missing == "draft":
        startup.models.append(("draft", torch.nn.Module()))
    else:
        startup.kv_regions["arena"] = NpuParameter(torch.empty(0), False)
    with pytest.raises(RuntimeError, match="found no NPU"):
        startup.validate()


def test_nested_named_owners_share_one_cpu_backed_scope(saver_modules, monkeypatch):
    shared, adapter = saver_modules
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(current_device=lambda: 0), raising=False
    )
    saver = adapter.AscendMemorySaverAdapter()
    other = adapter.AscendMemorySaverAdapter()
    outputs = []

    def initialize():
        assert not saver.active
        outputs.append(3)

    with saver.named_region("executor_inputs"):
        for owner in ("runtime_states", "attention_state", "workspace"):
            with other.named_region(owner):
                shared.initialize_communication(initialize)
            assert outputs == []

    assert outputs == [3, 3, 3]
    assert saver.scopes == [
        PoolKey("weights", AllocationPolicy.CPU_BACKUP, "persistent_buffers")
    ]
    assert other.scopes == []


@pytest.mark.parametrize("fail", [False, True])
def test_weight_load_defers_communication_until_managed_scope_exits(
    saver_modules, fail
):
    shared, adapter = saver_modules
    saver = adapter.AscendMemorySaverAdapter()
    output = []

    def initialize():
        assert not saver.active
        output.append(3)

    outcome = pytest.raises(ValueError, match="load failed") if fail else nullcontext()
    with outcome, saver.region(tag="weights", enable_cpu_backup=False):
        shared.initialize_communication(initialize)
        assert output == []
        if fail:
            raise ValueError("load failed")

    if fail:
        assert output == []
        shared.initialize_communication(initialize)
    assert output == [3]


@dataclass
class LocalHcclGroup:
    fail_registration: bool = False
    registered: bool = False


@pytest.mark.parametrize("reuse_group", [False, True])
@pytest.mark.parametrize("failure", [None, "registration", "startup"])
def test_a3_saver_owns_node_local_registration(
    saver_modules, monkeypatch, reuse_group, failure
):
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.distributed.process_group_manager import ProcessGroupManager

    _, adapter = saver_modules
    mapping = Mapping(rank=5, world_size=8, nprocs_per_node=4)
    manager = ProcessGroupManager()
    monkeypatch.setattr(manager, "_device_backend", "hccl")
    manager_module = sys.modules["tokenspeed.runtime.distributed.process_group_manager"]
    monkeypatch.setattr(manager_module, "process_group_manager", manager)
    group = LocalHcclGroup(fail_registration=failure == "registration")

    def new_group(ranks, *, backend, timeout, use_local_synchronization):
        assert not reuse_group
        assert ranks == (4, 5, 6, 7)
        assert backend == "hccl"
        assert use_local_synchronization
        return group

    def device_properties(device):
        assert device == mapping.gpu_id
        return SimpleNamespace(name="Ascend910_9382")

    if reuse_group:
        manager.register_process_group("hccl", (4, 5, 6, 7), group)
    monkeypatch.setattr(torch.distributed, "new_group", new_group)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(get_device_properties=device_properties),
        raising=False,
    )
    result = pytest.raises(RuntimeError, match=failure) if failure else nullcontext()
    with result, ExitStack() as on_failure:
        saver = adapter.AscendMemorySaverAdapter.configure_node_hccl(
            mapping, emulate_rank_zero=False, on_startup_failure=on_failure
        )
        assert group.registered
        if failure == "startup":
            raise RuntimeError("later startup failed")
        on_failure.pop_all()
    if failure is None:
        assert group.registered
        saver.close_hccl()
    assert not group.registered
    assert manager.get_device_process_group((4, 5, 6, 7)) is group


@pytest.mark.parametrize(
    ("device_name", "local_ranks", "emulate_rank_zero"),
    [("Ascend910B3", 4, False), (None, 1, False), (None, 4, True)],
)
def test_other_npu_workers_do_not_register_hccl(
    saver_modules, monkeypatch, device_name, local_ranks, emulate_rank_zero
):
    from tokenspeed.runtime.distributed.mapping import Mapping

    _, adapter = saver_modules
    mapping = Mapping(rank=0, world_size=8, nprocs_per_node=local_ranks)

    def device_properties(device):
        assert device_name is not None
        return SimpleNamespace(name=device_name)

    def new_group(*args, **kwargs):
        pytest.fail("unexpected HCCL process group creation")

    monkeypatch.setattr(torch.distributed, "new_group", new_group)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(get_device_properties=device_properties),
        raising=False,
    )
    with ExitStack() as on_failure:
        assert (
            adapter.AscendMemorySaverAdapter.configure_node_hccl(
                mapping,
                emulate_rank_zero=emulate_rank_zero,
                on_startup_failure=on_failure,
            )
            is None
        )


def test_startup_exempts_only_declared_peer_storages(saver_modules):
    _, adapter = saver_modules
    saver = adapter.AscendMemorySaverAdapter()
    model = torch.nn.Module()
    model.dense = NpuParameter(torch.arange(16).reshape(4, 4).float(), False)
    model.routed = NpuParameter(torch.arange(32).reshape(4, 8).float(), False)
    kv = NpuParameter(torch.zeros(128, dtype=torch.uint8), False)
    routed_storage = model.routed.untyped_storage()
    saver.validate_startup(
        models=[("target", model)],
        kv_regions={"arena": kv},
        graphs=[],
        peer_storages=frozenset({(routed_storage.data_ptr(), routed_storage.nbytes())}),
    )
    allocations, graphs = saver.inventory
    assert graphs == []
    names = [entry[1] for entry in allocations]
    assert "target.dense" in names
    assert "target.routed" not in names
    assert "arena" in names

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

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch


@pytest.fixture
def adapter_module(monkeypatch):
    # Import the allocator boundary without loading the accelerator runtime.
    source = Path(__file__).resolve().parents[2] / "python/tokenspeed/runtime/utils"
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
    return module


@dataclass(frozen=True)
class NpuDevice:
    type: str = "npu"
    index: int = 0


class NpuParameter(torch.nn.Parameter):
    """Expose an NPU device while retaining actual CPU-backed tensor storage."""

    @property
    def device(self):
        return NpuDevice()


class CpuNativeBackend:
    def __init__(self):
        self.blocks = []
        self.fault = False

    def stats(self):
        return {"blocks": self.blocks}

    def check_health(self):
        if self.fault:
            raise RuntimeError("restart this worker")

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


@dataclass
class CpuPool:
    id: tuple[int, int]


@dataclass
class CpuGraph:
    pool_id: tuple[int, int]

    def pool(self):
        return self.pool_id


class CpuDeviceState:
    def __init__(self, module):
        self.device = 0
        self.native = CpuNativeBackend()
        self.module = module
        self.segments = []
        self.snapshot_available = True
        self.state = module.PoolState(
            backend=self.native,
            make_pool=self.make_pool,
            synchronize=self.synchronize,
        )

    def synchronize(self):
        self.native.check_health()

    def memory_snapshot(self):
        if not self.snapshot_available:
            raise RuntimeError("memory snapshot unavailable")
        return self.segments

    def make_pool(self, key):
        return self.module.OwnedPool(object(), CpuPool((0, len(self.state.pools) + 1)))

    def graph(self):
        key = self.module.PoolKey(
            "acl_graph", self.module.AllocationPolicy.DISCARD, "decode"
        )
        with self.state.region(key) as owned:
            return CpuGraph(owned.pool.id)


@pytest.fixture
def startup(adapter_module, monkeypatch):
    state = CpuDeviceState(adapter_module)
    monkeypatch.setattr(torch, "npu", state, raising=False)
    adapter = adapter_module.AscendMemorySaverAdapter()
    monkeypatch.setattr(adapter, "_state", lambda: state)
    model = torch.nn.Module()
    model.register_parameter(
        "weight", NpuParameter(torch.arange(16).reshape(4, 4).float(), False)
    )
    kv = NpuParameter(torch.zeros(128, dtype=torch.uint8), False)
    state.native.own(model.weight, "weights")
    state.native.own(kv, "kv_cache")
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
    state.native.own(draft.independent, "weights")
    args["models"].append(("draft", draft))
    args["kv_regions"]["host_history"] = torch.zeros(7)
    args["kv_regions"]["empty"] = NpuParameter(torch.empty(0), False)
    adapter.validate_startup(**args)


@pytest.mark.parametrize("tag", ["weights", "kv_cache"])
def test_startup_requires_actual_tensor_backing_even_with_other_native_blocks(
    startup, tag
):
    adapter, state, args = startup
    state.native.blocks[:] = [
        block for block in state.native.blocks if block["tag"] != tag
    ]
    state.native.own(torch.zeros(8), tag)
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
    state.native.blocks[0][field] = value
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
    graph = state.graph()
    args["graphs"] = [graph, graph]
    scratch = torch.zeros(96)
    block = state.native.own(scratch, "acl_graph")
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


def test_startup_accepts_allocation_free_captured_graph(startup):
    adapter, state, args = startup
    args["graphs"] = [state.graph()]
    adapter.validate_startup(**args)


def test_startup_rejects_captured_pool_with_unmanaged_backing(startup):
    adapter, state, args = startup
    args["graphs"] = [CpuGraph((0, 99))]
    with pytest.raises(RuntimeError, match="owned managed pool"):
        adapter.validate_startup(**args)
    graph = state.graph()
    args["graphs"] = [graph]
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
    state.native.fault = True
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

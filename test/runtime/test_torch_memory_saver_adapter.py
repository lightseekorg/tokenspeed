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

"""CPU checks for managed versus resident native allocation scopes."""

from contextlib import contextmanager, nullcontext
from test.ci_system.ci_register import register_cuda_ci
from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.utils import torch_memory_saver_adapter as adapter

register_cuda_ci(est_time=5, suite="runtime-1gpu")


class CpuMemorySaver:
    """Stand in for the unavailable CUDA allocator, retaining region semantics."""

    def __init__(self, allocations):
        self.active = False
        self.tag = None
        self.cpu_backup = False
        self.allocations = allocations
        self.snapshots = {}

    @contextmanager
    def region(self, *, tag, enable_cpu_backup):
        assert not self.active
        self.active = True
        self.tag = tag
        self.cpu_backup = enable_cpu_backup
        try:
            yield
        finally:
            assert self.active
            self.active = False
            self.tag = None

    @contextmanager
    def disable(self):
        # The real CUDA allocator rejects disable() outside an active region.
        assert self.active
        self.active = False
        try:
            yield
        finally:
            self.active = True

    def allocate(self, value):
        tensor = torch.tensor([value], dtype=torch.int32)
        self.allocations.append(
            (self.tag if self.active else None, tensor, self.active and self.cpu_backup)
        )
        return tensor

    def pause(self, *, tag):
        for allocation_tag, tensor, cpu_backup in self.allocations:
            if allocation_tag == tag:
                if cpu_backup:
                    self.snapshots[tensor.data_ptr()] = tensor.clone()
                tensor.zero_()

    def resume(self, *, tag):
        for allocation_tag, tensor, cpu_backup in self.allocations:
            if allocation_tag == tag and cpu_backup:
                tensor.copy_(self.snapshots.pop(tensor.data_ptr()))


@pytest.fixture
def memory(monkeypatch):
    allocations = []
    saver = CpuMemorySaver(allocations)
    monkeypatch.setattr(adapter, "_primary_memory_saver", saver, raising=False)
    monkeypatch.setattr(
        adapter,
        "_buffer_memory_savers",
        {"persistent_buffers": CpuMemorySaver(allocations)},
        raising=False,
    )
    return saver


@pytest.fixture
def buffers(memory):
    return adapter._buffer_memory_savers["persistent_buffers"]


def test_native_storage_initializes_after_weight_region_and_survives_discard(memory):
    saver = adapter.TorchMemorySaverAdapter.create(enable=True)
    communication = []
    with saver.region(tag="weights", enable_cpu_backup=False):
        weight = memory.allocate(5)
        adapter.initialize_communication(
            lambda: communication.append(memory.allocate(7))
        )
        assert communication == []
        derived_weight = memory.allocate(11)
    saver.pause(tag="weights")
    assert weight.item() == derived_weight.item() == 0
    assert communication[0].item() == 7


def test_multiple_preparations_finish_before_region_returns(memory):
    saver = adapter.TorchMemorySaverAdapter.create(enable=True)
    outputs = []
    with saver.region(tag="weights", enable_cpu_backup=False):
        for value in (3, 5, 7):
            adapter.initialize_communication(
                lambda value=value: outputs.append(memory.allocate(value))
            )
        assert outputs == []
    saver.pause(tag="weights")
    assert [tensor.item() for tensor in outputs] == [3, 5, 7]


def test_failed_weight_load_does_not_initialize_communication(memory):
    saver = adapter.TorchMemorySaverAdapter.create(enable=True)
    outputs = []
    with pytest.raises(RuntimeError, match="load failed"):
        with saver.region(tag="weights", enable_cpu_backup=False):
            adapter.initialize_communication(
                lambda: outputs.append(memory.allocate(13))
            )
            raise RuntimeError("load failed")
    assert outputs == []
    adapter.initialize_communication(lambda: outputs.append(memory.allocate(17)))
    assert outputs[0].item() == 17


def test_preparation_failure_fails_loading_after_managed_scope_exits(memory):
    saver = adapter.TorchMemorySaverAdapter.create(enable=True)

    def fail():
        assert not memory.active
        raise RuntimeError("native preparation failed")

    with pytest.raises(RuntimeError, match="native preparation failed"):
        with saver.region(tag="weights", enable_cpu_backup=False):
            adapter.initialize_communication(fail)
    assert not memory.active


def test_disabled_adapter_initializes_communication_immediately(memory):
    saver = adapter.TorchMemorySaverAdapter.create(enable=False)
    outputs = []
    with saver.region(tag="weights", enable_cpu_backup=False):
        adapter.initialize_communication(lambda: outputs.append(memory.allocate(23)))
        assert outputs[0].item() == 23
    memory.pause(tag="weights")
    assert outputs[0].item() == 23


def test_graph_rng_storage_survives_sleep_before_graph_replay(
    memory, buffers, monkeypatch
):
    class CudaGraph:
        """Replace unavailable CUDA capture, retaining its RNG Tensor owners."""

        def __init__(self):
            self.seed = None
            self.offset = None

        def register_generator_state(self, generator):
            self.seed = buffers.allocate(17)
            self.offset = buffers.allocate(29)

    @contextmanager
    def capture(**kwargs):
        assert not buffers.active
        yield

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "default_generators", (object(),))
    monkeypatch.setattr(memory, "cuda_graph", capture, raising=False)
    saver = adapter.TorchMemorySaverAdapter.create(enable=True)
    graph = CudaGraph()
    with saver.cuda_graph(graph, pool=None, stream=None, tag="cuda_graph"):
        temporary = memory.allocate(41)
    tensors = (graph.seed, graph.offset)
    addresses = [tensor.data_ptr() for tensor in tensors]
    for _ in range(3):
        saver.pause(tag="weights")
        assert all(tensor.item() == 0 for tensor in tensors)
        saver.resume(tag="weights")
        assert [tensor.data_ptr() for tensor in tensors] == addresses
        assert [tensor.item() for tensor in tensors] == [17, 29]
        assert temporary.item() == 41


@pytest.mark.parametrize(
    "name", ["executor_inputs", "runtime_states", "attention_state", "workspace"]
)
def test_named_buffers_preserve_contents_and_addresses_across_discard(
    memory, buffers, name
):
    saver = adapter.TorchMemorySaverAdapter.create(enable=True)
    nested_saver = adapter.TorchMemorySaverAdapter.create(enable=True)
    with saver.region(tag="weights", enable_cpu_backup=False):
        weight = memory.allocate(5)
    with saver.region(tag="kv_cache", enable_cpu_backup=False):
        kv = memory.allocate(19)
    with saver.named_region(name):
        constant = buffers.allocate(7)
        nested_name = "workspace" if name != "workspace" else "attention_state"
        with nested_saver.named_region(nested_name):
            index = buffers.allocate(11)
    addresses = (constant.data_ptr(), index.data_ptr())

    saver.pause(tag="kv_cache")
    assert kv.item() == 0
    assert (weight.item(), constant.item(), index.item()) == (5, 7, 11)
    for _ in range(2):
        saver.pause(tag="weights")
        assert weight.item() == constant.item() == index.item() == 0
        saver.resume(tag="weights")
        assert weight.item() == 0
        assert (constant.item(), index.item()) == (7, 11)
        assert (constant.data_ptr(), index.data_ptr()) == addresses


def test_named_buffer_scope_is_restored_after_failure(memory, buffers):
    saver = adapter.TorchMemorySaverAdapter.create(enable=True)
    with pytest.raises(RuntimeError, match="allocation failed"):
        with saver.named_region("executor_inputs"):
            with saver.named_region("workspace"):
                raise RuntimeError("allocation failed")
    with saver.named_region("runtime_states"):
        constant = buffers.allocate(13)
    saver.pause(tag="weights")
    saver.resume(tag="weights")
    assert constant.item() == 13


@pytest.mark.parametrize(
    "name", ["executor_inputs", "runtime_states", "attention_state", "workspace"]
)
def test_disabled_named_region_leaves_allocations_resident(memory, buffers, name):
    saver = adapter.TorchMemorySaverAdapter.create(enable=False)
    with saver.named_region(name):
        constant = buffers.allocate(17)
    memory.pause(tag="weights")
    assert constant.item() == 17


@pytest.mark.parametrize("enabled", [True, False])
def test_unknown_named_region_is_rejected(memory, enabled):
    saver = adapter.TorchMemorySaverAdapter.create(enable=enabled)
    with pytest.raises(KeyError, match="unknown_owner"):
        with saver.named_region("unknown_owner"):
            pytest.fail("unknown region entered its allocation scope")


@pytest.mark.parametrize("enabled", [True, False])
def test_host_transfer_lane_metadata_survives_sleep_and_later_growth(
    memory, buffers, monkeypatch, enabled
):
    from tokenspeed.runtime.cache.transfer import lanes

    class Workspace:
        """CPU-backed stand-in for device allocations at the kernel boundary."""

        def __init__(self):
            self.bindings = None
            self.rows = None
            self.capacity = 0

        def load_block_transfers(self, transfers, *, geometry):
            return len(transfers), ()

        def prepare_backend(self, *args, **kwargs):
            if self.bindings is None:
                self.bindings = buffers.allocate(17)
            return SimpleNamespace(uses_device_tables=True)

        def commit_block_transfers(self, count, device, *, non_blocking):
            if count > self.capacity:
                self.rows = buffers.allocate(count)
                self.capacity = count

    class Event:
        def record(self, stream):
            pass

        def query(self):
            return True

    monkeypatch.setattr(lanes, "HostTransferWorkspace", Workspace)
    monkeypatch.setattr(
        lanes,
        "device_module",
        SimpleNamespace(stream=lambda stream: nullcontext(), Event=Event),
    )
    saver = adapter.TorchMemorySaverAdapter.create(enable=enabled)
    lane = lanes.HostTransferLane(saver)
    for count in (1, 257):
        assert (
            lane._stage(
                [(0, 1, 1)] * count,
                device_buffers=(SimpleNamespace(device="cuda"),),
                host_buffer=object(),
                geometry=object(),
                stream=object(),
                backend="auto",
            )
            == count
        )
        tensors = (lane.workspace.bindings, lane.workspace.rows)
        pointers = [tensor.data_ptr() for tensor in tensors]
        expected = [tensor.clone() for tensor in tensors]
        for tensor in tensors:
            allocation = next(row for row in buffers.allocations if row[1] is tensor)
            assert allocation[0] == ("weights" if enabled else None)
            assert allocation[2] is enabled
        buffers.pause(tag="weights")
        if enabled:
            assert all(tensor.item() == 0 for tensor in tensors)
        buffers.resume(tag="weights")
        assert [tensor.data_ptr() for tensor in tensors] == pointers
        for tensor, reference in zip(tensors, expected):
            torch.testing.assert_close(tensor, reference, rtol=0, atol=0)


def test_cache_arena_geometry_survives_sleep_with_its_original_address(
    memory, buffers, monkeypatch
):
    from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
        CacheFieldSpec,
        pack,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import CacheGroupSpec

    group = CacheGroupSpec(
        group_id="full_attention",
        attn_kind="full_history",
        rows_per_page=8,
        entry_stride_tokens=1,
        transfer_policy="full_suffix",
        replayable=False,
    )
    layout = pack(
        (
            (
                group,
                (CacheFieldSpec("layer.0.kv", "layer.0.kv", (8, 1, 32), "bfloat16"),),
            ),
        ),
        prefix_granularity=8,
        cache_blocks_per_lcm_block={group.group_id: 1},
        alignment=256,
        max_padding_fraction=0.25,
    )
    real_tensor = torch.tensor

    def allocate(*args, **kwargs):
        tensor = real_tensor(*args, **kwargs)
        if buffers.active:
            buffers.allocations.append((buffers.tag, tensor, buffers.cpu_backup))
        return tensor

    # Only the unavailable pausable CUDA allocator is replaced. Arena ownership,
    # geometry construction, Torch storage and adapter dispatch remain real.
    monkeypatch.setattr(torch, "tensor", allocate)
    arena = CacheArena(
        layout.bind(2), "cpu", cache_group_specs=(group,), enable_memory_saver=True
    )
    table = arena._zero_field_tables[group.group_id][1]
    address = table.data_ptr()
    expected = table.clone()
    assert any(
        t.data_ptr() == address and backup for _, t, backup in buffers.allocations
    )
    for _ in range(2):
        arena.memory_saver_adapter.pause(tag="weights")
        assert not table.count_nonzero()
        arena.memory_saver_adapter.resume(tag="weights")
        assert table.data_ptr() == address
        torch.testing.assert_close(table, expected)


@pytest.mark.parametrize("with_draft", [False, True])
def test_capture_metadata_survives_sleep_for_target_and_draft(
    memory, buffers, with_draft
):
    from tokenspeed.runtime.execution.forward_step import ForwardStepRunner

    class CpuAttentionMetadata:
        """Replace GPU metadata construction, retaining the allocated Tensor owner."""

        def __init__(self):
            self.tensor = None

        def init_forward_metadata_capture_cuda_graph(self, batch_size, *args, **kwargs):
            self.tensor = buffers.allocate(batch_size)

    # The capture runner's allocation scope stays real; GPU inputs/backends are
    # replaced because this test verifies CPU allocator ownership and recovery.
    runner = object.__new__(ForwardStepRunner)
    runner.memory_saver_adapter = adapter.TorchMemorySaverAdapter.create(enable=True)
    runner.attn_backend = CpuAttentionMetadata()
    runner.draft_attn_backend = CpuAttentionMetadata() if with_draft else None
    runner.input_buffers = SimpleNamespace(
        req_pool_indices_buf=torch.arange(2),
        seq_lens_buf=torch.ones(2, dtype=torch.int32),
    )
    runner._placeholder_tables = {}
    runner.max_tokens_per_req = 1
    runner.drafter = SimpleNamespace(round_block_tables=None) if with_draft else None
    runner._init_capture_metadata(2)
    tensors = [runner.attn_backend.tensor]
    if with_draft:
        tensors.append(runner.draft_attn_backend.tensor)
        assert runner.drafter.round_block_tables == {}
    addresses = [tensor.data_ptr() for tensor in tensors]
    for _ in range(2):
        runner.memory_saver_adapter.pause(tag="weights")
        assert all(tensor.item() == 0 for tensor in tensors)
        runner.memory_saver_adapter.resume(tag="weights")
        assert [tensor.data_ptr() for tensor in tensors] == addresses
        assert all(tensor.item() == 2 for tensor in tensors)


@pytest.mark.parametrize("enabled", [True, False])
def test_forward_cache_tables_use_restorable_pool_when_retained_between_rounds(
    memory, buffers, monkeypatch, enabled
):
    from tokenspeed.runtime.execution.model_executor import ModelExecutor
    from tokenspeed.runtime.layers.attention.backends import cache_metadata
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
        CacheRuntimeContract,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import CacheGroupSpec

    class ForwardInputsUnavailable(Exception):
        """Stop at the GPU input-fill boundary after real metadata construction."""

    tensors = []

    def pack(op, device, **kwargs):
        # Replace only the GPU table-packing boundary. The executor, metadata
        # validation, shared-storage views and pool dispatch stay real.
        tensor = buffers.allocate(len(op.request_ids))
        tensors.append(tensor)
        return SimpleNamespace(
            tables={"full": tensor.expand(len(op.request_ids), 1)}, tables_cpu={}
        )

    def fill_inputs(**kwargs):
        raise ForwardInputsUnavailable

    monkeypatch.setattr(cache_metadata, "packed_block_tables_from_forward_op", pack)
    group = CacheGroupSpec(
        group_id="full",
        attn_kind="full_history",
        rows_per_page=1,
        entry_stride_tokens=1,
        transfer_policy="full_suffix",
        replayable=False,
    )
    # The full executor needs GPU model construction; retain its real forward
    # prologue and replace the unavailable device/stream and input-fill boundary.
    executor = object.__new__(ModelExecutor)
    executor.log_step = 0
    executor.forward_step = SimpleNamespace(draft_overlap=None)
    executor.device = torch.device("cpu")
    executor.execution_stream = SimpleNamespace(wait_stream=lambda stream: None)
    executor.default_stream = object()
    executor.device_module = SimpleNamespace(stream=lambda stream: nullcontext())
    executor.nan_guard = SimpleNamespace(reset=lambda count: None)
    executor.runtime_states = SimpleNamespace(bind_stop_criteria=lambda *args: None)
    executor.input_buffers = SimpleNamespace(fill_input_buffers=fill_inputs)
    executor.memory_saver_adapter = adapter.TorchMemorySaverAdapter.create(
        enable=enabled
    )
    executor._validate_page_ids = True
    executor._cache_runtime_contract = CacheRuntimeContract(
        prefix_granularity=1,
        num_lcm_blocks=4,
        token_capacity=4,
        group_specs=(group,),
        group_page_counts={"full": 5},
        group_packing={"full": 1},
    )
    for count in (1, 3):
        op = SimpleNamespace(
            num_extends=lambda: 0,
            input_lengths=[1] * count,
            request_ids=list(range(count)),
            request_pool_indices=list(range(count)),
        )
        with pytest.raises(ForwardInputsUnavailable):
            executor.execute_forward_op(
                op,
                [],
                ngram_inputs=None,
                request_history_seeds=None,
                input_logprob_plan=None,
                sampled_top_k=0,
                stop_criteria=[],
                admission_ids=list(range(count)),
            )
        tensor = tensors[-1]
        address = tensor.data_ptr()
        allocation = next(row for row in buffers.allocations if row[1] is tensor)
        assert allocation[0] == ("weights" if enabled else None)
        executor.memory_saver_adapter.pause(tag="weights")
        assert tensor.item() == (0 if enabled else count)
        executor.memory_saver_adapter.resume(tag="weights")
        assert tensor.data_ptr() == address
        assert tensor.item() == count

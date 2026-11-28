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

import subprocess
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from test.ci_system.ci_register import register_cuda_ci
from types import SimpleNamespace
from unittest.mock import sentinel

import numpy as np
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
        self.nccl_resident = True
        self.pending_work = False

    def synchronize(self):
        self.pending_work = False

    def suspend_nccl(self):
        self.nccl_resident = False

    def resume_nccl(self):
        self.nccl_resident = True

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
        assert not self.pending_work, "unmapped storage before pending work completed"
        for allocation_tag, tensor, cpu_backup in self.allocations:
            if allocation_tag == tag:
                if cpu_backup:
                    self.snapshots[tensor.data_ptr()] = tensor.clone()
                tensor.zero_()

    def resume(self, *, tag):
        for allocation_tag, tensor, cpu_backup in self.allocations:
            if allocation_tag == tag and cpu_backup:
                tensor.copy_(self.snapshots.pop(tensor.data_ptr()))


def _set_adapter_build(monkeypatch, backend):
    # The adapter only checks presence. Driver probes must see the real build.
    version = SimpleNamespace(
        cuda=sentinel.cuda_build if backend == "cuda" else None,
        hip=sentinel.hip_build if backend == "hip" else None,
    )
    monkeypatch.setattr(
        adapter, "torch", SimpleNamespace(version=version, cuda=torch.cuda)
    )


@pytest.fixture(params=["cuda", "hip"])
def memory(monkeypatch, request):
    _set_adapter_build(monkeypatch, request.param)
    monkeypatch.setattr(adapter, "_non_expandable_allocation", nullcontext)
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
    saver = adapter.TorchMemorySaverAdapter.create(enable=True, device="cuda")
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
    saver = adapter.TorchMemorySaverAdapter.create(enable=True, device="cuda")
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
    saver = adapter.TorchMemorySaverAdapter.create(enable=True, device="cuda")
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
    saver = adapter.TorchMemorySaverAdapter.create(enable=True, device="cuda")

    def fail():
        assert not memory.active
        raise RuntimeError("native preparation failed")

    with pytest.raises(RuntimeError, match="native preparation failed"):
        with saver.region(tag="weights", enable_cpu_backup=False):
            adapter.initialize_communication(fail)
    assert not memory.active


def test_disabled_adapter_initializes_communication_immediately(memory):
    saver = adapter.TorchMemorySaverAdapter.create(enable=False, device="cuda")
    outputs = []
    with saver.region(tag="weights", enable_cpu_backup=False):
        adapter.initialize_communication(lambda: outputs.append(memory.allocate(23)))
        assert outputs[0].item() == 23
    memory.pause(tag="weights")
    assert outputs[0].item() == 23


def test_enabled_gpu_saver_refuses_a_cpu_torch_build(monkeypatch):
    _set_adapter_build(monkeypatch, "cpu")
    with pytest.raises(RuntimeError, match="requires a CUDA or ROCm PyTorch build"):
        adapter.TorchMemorySaverAdapter.create(enable=True, device="cuda")


@pytest.mark.parametrize("device", ["cuda", "npu"])
def test_disabled_adapter_can_run_without_optional_packages(monkeypatch, device):
    monkeypatch.setitem(sys.modules, "torch_memory_saver", None)
    monkeypatch.setitem(sys.modules, "torch_npu_memory_saver", None)
    monkeypatch.setitem(
        sys.modules, "tokenspeed.runtime.utils.ascend_memory_saver_adapter", None
    )
    with adapter.TorchMemorySaverAdapter.create(enable=False, device=device).region():
        pass


def test_enabled_npu_saver_reports_its_missing_optional_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_npu_memory_saver", None)
    monkeypatch.delitem(
        sys.modules,
        "tokenspeed.runtime.utils.ascend_memory_saver_adapter",
        raising=False,
    )
    with pytest.raises(ModuleNotFoundError, match="torch-npu-memory-saver"):
        adapter.TorchMemorySaverAdapter.create(enable=True, device="npu")


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
    saver = adapter.TorchMemorySaverAdapter.create(enable=True, device="cuda")
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
    "name",
    [
        "executor_inputs",
        "runtime_states",
        "attention_state",
        "cache_layout",
        "cache_transfer",
        "workspace",
    ],
)
def test_named_buffers_preserve_contents_and_addresses_across_discard(
    memory, buffers, name
):
    saver = adapter.TorchMemorySaverAdapter.create(enable=True, device="cuda")
    nested_saver = adapter.TorchMemorySaverAdapter.create(enable=True, device="cuda")
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
    saver = adapter.TorchMemorySaverAdapter.create(enable=True, device="cuda")
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
    "name",
    [
        "executor_inputs",
        "runtime_states",
        "attention_state",
        "cache_layout",
        "cache_transfer",
        "workspace",
    ],
)
def test_disabled_named_region_leaves_allocations_resident(memory, buffers, name):
    saver = adapter.TorchMemorySaverAdapter.create(enable=False, device="cuda")
    with saver.named_region(name):
        constant = buffers.allocate(17)
    memory.pause(tag="weights")
    assert constant.item() == 17


@pytest.mark.parametrize("enabled", [True, False])
def test_unknown_named_region_is_rejected(memory, enabled):
    saver = adapter.TorchMemorySaverAdapter.create(enable=enabled, device="cuda")
    with pytest.raises(KeyError, match="unknown_owner"):
        with saver.named_region("unknown_owner"):
            pytest.fail("unknown region entered its allocation scope")


@pytest.mark.parametrize("nccl_suspendable", [False, True])
def test_gpu_graph_region_routes_to_its_pool(memory, nccl_suspendable):
    from tokenspeed.runtime.engine.memory_occupation import memory_regions

    saver = adapter.TorchMemorySaverAdapter.create(enable=True, device="cuda")
    with saver.region(tag="weights", enable_cpu_backup=False):
        weight = memory.allocate(5)
    with saver.region(tag="cuda_graph", enable_cpu_backup=False):
        scratch = memory.allocate(7)
    regions = memory_regions(
        memory,
        saver,
        graph_tag="cuda_graph",
        nccl_suspendable=nccl_suspendable,
        synchronize_kv_before_registration=False,
    )
    assert list(regions) == ["kv_cache", "weights", "cuda_graph"] + (
        ["nccl"] if nccl_suspendable else []
    )
    memory.pending_work = True
    regions["cuda_graph"].release()
    regions["cuda_graph"].resume()
    assert (weight.item(), scratch.item()) == (5, 0)
    if nccl_suspendable:
        regions["nccl"].release()
        assert not memory.nccl_resident
        regions["nccl"].resume()
        assert memory.nccl_resident


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
    saver = adapter.TorchMemorySaverAdapter.create(enable=enabled, device="cuda")
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


def _ordinary_cuda_bytes() -> int:
    return sum(
        segment["total_size"]
        for segment in torch.cuda.memory_snapshot()
        if segment["device"] == torch.cuda.current_device()
        and tuple(segment["segment_pool_id"]) == (0, 0)
    )


def _check_managed_metadata_on_cuda():
    """Check managed allocations, retained views and zero ordinary bytes on CUDA."""
    from tokenspeed.runtime.execution.forward_step import prepare_capture_metadata
    from tokenspeed.runtime.layers.attention.backends.cache_metadata import (
        build_managed_cache_metadata,
    )
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
    arena = CacheArena(
        layout.bind(2), "cuda", cache_group_specs=(group,), enable_memory_saver=True
    )
    table = arena._zero_field_tables[group.group_id][1]
    address = table.data_ptr()
    expected = table.cpu()
    segments = [
        segment
        for segment in torch.cuda.memory_snapshot()
        if segment["address"] <= address < segment["address"] + segment["total_size"]
    ]
    assert len(segments) == 1
    assert tuple(segments[0]["segment_pool_id"]) != (0, 0)

    class ModelMetadata:
        """Replace model-specific metadata; its CUDA allocations stay native."""

        def __init__(self):
            self.tensor = None

        def init_forward_metadata_capture_cuda_graph(self, batch_size, *args, **kwargs):
            self.tensor = torch.full(
                (batch_size,), batch_size, dtype=torch.int32, device="cuda"
            )

    saver = arena.memory_saver_adapter
    target, draft = ModelMetadata(), ModelMetadata()
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.moe.source_rows import MoESourceRows
    from tokenspeed.runtime.utils.env import global_server_args_dict

    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(global_server_args_dict, "enable_memory_saver", True)
        rows = MoESourceRows(
            Mapping(rank=0, world_size=1), device="cuda", max_rows_per_rank=8
        )
    rows.publish((5,))
    counts = rows.source_token_counts
    retained = [(table, address, expected), (counts, counts.data_ptr(), counts.cpu())]
    for count, host_tables in ((1, False), (257, True), (3, False)):
        arrays = np.ones((count, 2), dtype=np.int32)
        spans = np.tile(np.array([0, 2], dtype=np.int32), (count, 1))
        forward_op = SimpleNamespace(
            block_table_windows=lambda arrays=arrays: {
                group.group_id: (arrays, arrays.shape[1])
            },
            block_table_spans=lambda spans=spans: {group.group_id: spans},
        )
        metadata = build_managed_cache_metadata(
            forward_op,
            memory_saver_adapter=saver,
            device="cuda",
            contract=arena.runtime_contract,
            num_requests=count,
            validate_page_ids=True,
            host_tables=host_tables,
        )
        packed = dict(metadata.tables(active_forward_op=forward_op))
        with saver.named_region("executor_inputs"):
            indices = torch.arange(count, dtype=torch.int32, device="cuda")
            lengths = torch.ones(count, dtype=torch.int32, device="cuda")
        prepare_capture_metadata(
            target,
            draft,
            memory_saver_adapter=saver,
            batch_size=count,
            req_pool_indices=indices,
            seq_lens=lengths,
            block_tables=packed,
            num_tokens=count,
        )
        # Keep old views alive as backends do between forwards and captures.
        retained.extend(
            (tensor, tensor.data_ptr(), tensor.cpu())
            for tensor in (
                packed[group.group_id],
                target.tensor,
                draft.tensor,
                indices,
                lengths,
            )
        )
        for _ in range(3):
            torch.cuda.synchronize()
            saver.pause(tag="weights")
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            ordinary_bytes = sum(
                segment["total_size"]
                for segment in torch.cuda.memory_snapshot()
                if segment["device"] == torch.cuda.current_device()
                and tuple(segment["segment_pool_id"]) == (0, 0)
            )
            assert ordinary_bytes == 0, ordinary_bytes
            saver.resume(tag="weights")
            for tensor, pointer, reference in retained:
                assert tensor.data_ptr() == pointer
                torch.testing.assert_close(tensor.cpu(), reference, rtol=0, atol=0)

    from tokenspeed_kernel.ops.attention.kda import kda_paged_prefill
    from tokenspeed_kernel.ops.attention.kda.cute_dsl import cutedsl_kda_supported

    if cutedsl_kda_supported():
        # Exercise lazy compilation and both workspace routes, including growth.
        for tokens, heads, sequences in (
            (32, 12, 1),
            (257, 96, 1),
            (384, 96, 3),
            (128, 12, 1),
        ):
            host_bounds = torch.arange(sequences + 1, dtype=torch.int64) * (
                tokens // sequences
            )
            with saver.named_region("runtime_states"):
                packed = torch.full(
                    (1, tokens, 3, heads, 128), 0.1, device="cuda", dtype=torch.bfloat16
                )
                gate = torch.zeros(
                    (1, tokens, heads, 128), device="cuda", dtype=torch.bfloat16
                )
                beta = torch.zeros(
                    (1, tokens, heads), device="cuda", dtype=torch.bfloat16
                )
                decay = torch.zeros(heads, device="cuda")
                bias = torch.zeros(heads, 128, device="cuda")
                state = torch.zeros(sequences, heads, 128, 128, device="cuda")
                bounds = host_bounds.to("cuda")

            def scan():
                return kda_paged_prefill(
                    *packed.unbind(2),
                    gate,
                    beta,
                    decay,
                    bias,
                    initial_state=state,
                    cu_seqlens=bounds,
                    cu_seqlens_cpu=host_bounds,
                    capacity=None,
                    lower_bound=-5.0,
                    solution="cutedsl_kda",
                    recurrent_layout="v_major",
                )

            result = scan()
            expected_out, expected_state = result.out.cpu(), result.final_state.cpu()
            del result
            pointer = packed.data_ptr()
            for _ in range(2):
                torch.cuda.synchronize()
                saver.pause(tag="weights")
                torch.cuda.empty_cache()
                ordinary_bytes = sum(
                    segment["total_size"]
                    for segment in torch.cuda.memory_snapshot()
                    if segment["device"] == torch.cuda.current_device()
                    and tuple(segment["segment_pool_id"]) == (0, 0)
                )
                assert ordinary_bytes == 0, ordinary_bytes
                saver.resume(tag="weights")
                assert packed.data_ptr() == pointer
                result = scan()
                torch.testing.assert_close(
                    result.out.cpu(), expected_out, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    result.final_state.cpu(), expected_state, rtol=0, atol=0
                )
                del result


def _run_with_cuda_memory_saver(check, *, timeout=90):
    memory_saver = pytest.importorskip("torch_memory_saver")
    # Preload hooks must be installed before CUDA initializes in the child.
    with memory_saver.configure_subprocess():
        subprocess.run(
            [
                sys.executable,
                "-c",
                f"from test.runtime.test_torch_memory_saver_adapter import "
                f"{check.__name__}; {check.__name__}()",
            ],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            timeout=timeout,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires native CUDA VMM memory-saver hooks",
)
def test_managed_metadata_survives_sleep_and_growth_with_zero_ordinary_bytes():
    _run_with_cuda_memory_saver(_check_managed_metadata_on_cuda)


@torch.inference_mode()
def _check_gated_residual_workspace_on_cuda():
    from tokenspeed_kernel.numerics.torch.residual import torch_hyperconnection_mix
    from tokenspeed_kernel.ops.residual import select_gated_residual_mix_kernel

    from tokenspeed.runtime.layers.hyperconnection import (
        GatedResidualSimple,
        HyperConnectionConfig,
        register_workspace_region,
    )

    saver = adapter.TorchMemorySaverAdapter.create(enable=True, device="cuda")
    register_workspace_region(saver)
    g, h, r = 4, 2048, 256
    with saver.region(tag="weights", enable_cpu_backup=True), torch.device("cuda"):
        mixer = GatedResidualSimple(
            HyperConnectionConfig(hc_count=g, hidden_size=h, hc_lowrank=r)
        ).to(dtype=torch.bfloat16)
        mixer.process_weights_after_loading(mixer)
    with saver.named_region("executor_inputs"):
        normalized = torch.randn(33, g * h, device="cuda", dtype=torch.bfloat16)
    down, up = mixer.mix_inject_proj.weight, mixer.input_mix_weight_up.weight
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graphs = []

    for rows in (1, 33):
        inputs = normalized[:rows]
        with torch.cuda.stream(stream):
            selected, _ = select_gated_residual_mix_kernel(
                inputs,
                down,
                up,
                g,
                h,
                r,
                weights_independent=True,
                override=None,
                solution=None,
            )
            assert selected.name == "cute_fused_hyperconnection_mix"
            mixer.mix(inputs, normalized=inputs)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with saver.cuda_graph(graph, pool=None, stream=stream, tag="cuda_graph"):
            mixed, (_, _, inject) = mixer.mix(inputs, normalized=inputs)
        graphs.append(graph)
        tensors = (inputs, down, up, mixed, inject)
        addresses = tuple(tensor.data_ptr() for tensor in tensors)
        for _ in range(2):
            torch.cuda.synchronize()
            saver.pause(tag="cuda_graph")
            saver.pause(tag="weights")
            torch.cuda.empty_cache()
            ordinary_bytes = _ordinary_cuda_bytes()
            assert ordinary_bytes == 0, ordinary_bytes
            saver.resume(tag="weights")
            saver.resume(tag="cuda_graph")
            assert tuple(tensor.data_ptr() for tensor in tensors) == addresses

            inputs.mul_(-0.5)
            graph.replay()
            torch.cuda.synchronize()
            expected = torch_hyperconnection_mix(
                inputs.cpu(), down.cpu(), up.cpu(), g, h, r, r + g, 1.0, True
            )
            for actual, reference in zip((mixed, inject), expected, strict=True):
                torch.testing.assert_close(
                    actual.cpu().double(), reference, atol=0.004, rtol=0.01
                )


def test_gated_residual_workspace_survives_sleep_and_lazy_allocation():
    from tokenspeed_kernel.ops.residual.cute_fused import supports_fused_hc

    if not torch.cuda.is_available() or not supports_fused_hc(torch.device("cuda")):
        pytest.skip("requires supported Blackwell and CuTe DSL")
    _run_with_cuda_memory_saver(_check_gated_residual_workspace_on_cuda)


@torch.inference_mode()
def _check_bf16_prims_qsa_workspace_on_cuda():
    from test.runtime.cache_pool_test_utils import make_qsa_pool_for_layers

    from tokenspeed_kernel.numerics.torch.attention.qsa import (
        torch_qsa_selected_attention,
    )
    from tokenspeed_kernel.ops.attention.qsa.triton import qwen4_exp_qsa_selected_slots
    from tokenspeed_kernel.workspace import capture_session

    from tokenspeed.runtime.configs.model_config import AttentionArch
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.drafter.eagle import AcceptedPrefixPublisher
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
    from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
    from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
    from tokenspeed.runtime.layers.attention.qsa.metadata import QSALayout, QSASelection
    from tokenspeed.runtime.layers.attention.registry import create_paged_router
    from tokenspeed.runtime.layers.paged_attention import PagedAttention

    page_size, head_dim, q_heads = 256, 256, 12
    pool = make_qsa_pool_for_layers(
        device="cuda",
        layer_ids=(0,),
        field_layer_offset=0,
        num_layers=1,
        num_lcm_blocks=3,
        kv_head_dim=head_dim,
        enable_memory_saver=True,
    )
    saver = pool.arena.memory_saver_adapter
    component = MHAConfig(
        backend_name="qsa",
        num_attention_heads=q_heads,
        num_kv_heads=1,
        head_dim=head_dim,
        attn_tp_size=1,
    )
    config = AttnConfig(
        device="cuda",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        kernel_page_size=page_size,
        context_len=page_size,
        max_bs=257,
        max_resident_requests=257,
        is_draft=True,
        components=(component,),
    )
    router = create_paged_router(config, AttentionArch.MHA, backend_name="qsa")
    router.set_cache_pool(pool)
    backend = router.leaves["full_attention"]
    layer = PagedAttention(
        num_heads=q_heads,
        num_kv_heads=1,
        head_dim=head_dim,
        scaling=head_dim**-0.5,
        layer_id=0,
        rotary_emb=None,
        qk_norm=None,
    )
    torch.manual_seed(42)
    k, v = pool.get_kv_buffer(0)
    k[page_size:].normal_()
    v[page_size:].normal_()
    graphs = []
    retained = []
    previous_capacity = 0
    stream = torch.cuda.Stream()
    for rows in (1, 257):
        with saver.named_region("executor_inputs"):
            q = torch.randn(rows, q_heads, head_dim, device="cuda", dtype=k.dtype)
            requests = torch.arange(rows, device=q.device, dtype=torch.int32)
            page_table = (1 + requests % 2).view(rows, 1)
            verify_width, block_size, block_topk = 3, 4, 64
            first = page_size - verify_width
            positions = torch.arange(first, page_size, device=q.device).repeat(rows)
            verify_requests = requests.repeat_interleave(verify_width)
            complete = ((positions + 1) // block_size).to(torch.int32)
            block_ids = torch.arange(block_topk, device=q.device, dtype=torch.int32)
            blocks = torch.where(block_ids < complete[:, None], block_ids, -1)
            verify_slots = qwen4_exp_qsa_selected_slots(
                blocks,
                complete,
                positions,
                verify_requests,
                page_table,
                page_size,
                block_size,
                block_topk * block_size,
                enable_pdl=False,
            )
            accepted = (requests + 1) % verify_width
            gather_ids = (requests * verify_width + accepted).long()
            accepted_positions = (first + accepted).long()
            seq_lens = (accepted_positions + 1).to(torch.int32)
            slots = verify_slots.index_select(0, gather_ids)
            write_slots = requests[:1]
            selection = QSASelection(verify_slots, blocks, block_size)
            layout = QSALayout(
                seq_lens=seq_lens,
                logical_positions=positions,
                request_indices=verify_requests,
                qsa_locs=write_slots,
                recent_locs=write_slots,
                complete_blocks=complete,
                queries_per_request=verify_width,
                qsa_page_table=page_table,
                full_page_table=page_table,
                full_kernel_page_size=page_size,
                reset_draft_tags=None,
            )
            ctx = ForwardContext(
                attn_backend=router,
                token_to_kv_pool=pool,
                bs=rows,
                num_extends=0,
                input_num_tokens=rows * verify_width,
                forward_mode=ForwardMode.DECODE,
                output_layout=ForwardOutputLayout(0, 0, rows, 1),
                gather_ids=gather_ids,
                draft_narrowing=AcceptedPrefixPublisher(router, seq_lens),
            )

        def forward():
            router.sparse_topk.qsa_selection = selection
            router.sparse_topk.qsa_metadata = layout
            return backend.forward_decode(
                q,
                None,
                None,
                layer,
                write_slots,
                pool,
                rows,
                topk_indices=slots,
                ctx=ctx,
            ).view_as(q)

        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            before = forward().cpu()
            # A real plan proves that draft step zero selected PrimTS.
            runner = backend._qsa_workspace.runners[q.device]
            # Keep the first gather live so the second eager forward uses
            # different addresses while hitting the same plan geometry.
            previous_rows = router.sparse_topk.qsa_draft_rows
            router.sparse_topk.clear_qsa()
            gather_ids.copy_(requests * verify_width + (accepted + 1) % verify_width)
            seq_lens.copy_(positions.index_select(0, gather_ids) + 1)
            slots.copy_(verify_slots.index_select(0, gather_ids))
            expected = forward().cpu()
            reference = torch_qsa_selected_attention(
                q[:1].cpu(),
                k.cpu(),
                v.cpu(),
                slots[:1].cpu(),
                layer.scaling,
                1.0,
                1.0,
            )
            router.sparse_topk.clear_qsa()
            del previous_rows
        assert not torch.equal(before, expected)
        torch.testing.assert_close(expected[:1], reference, atol=4e-3, rtol=1e-2)
        assert runner._workspace.numel() > previous_capacity
        previous_capacity = runner._workspace.numel()
        graph = torch.cuda.CUDAGraph()
        with capture_session(graph), saver.cuda_graph(
            graph, pool=None, stream=stream, tag="cuda_graph"
        ):
            captured = forward()
            router.sparse_topk.clear_qsa()
        graphs.append((graph, captured, expected))
        plan = next(
            plan
            for plan in runner._captured_plans.values()
            if plan.qo_indptr.numel() == rows + 1
        )
        retained.extend(
            (
                q,
                slots,
                blocks,
                page_table,
                gather_ids,
                verify_requests,
                positions,
                plan.qo_indptr,
                runner._workspace,
            )
        )
        for _ in range(2):
            torch.cuda.synchronize()
            snapshots = [(t, t.data_ptr(), t.cpu()) for t in retained]
            saver.pause(tag="cuda_graph")
            saver.pause(tag="weights")
            torch.cuda.empty_cache()
            assert _ordinary_cuda_bytes() == 0
            saver.resume(tag="weights")
            saver.resume(tag="cuda_graph")
            for tensor, address, contents in snapshots:
                assert tensor.data_ptr() == address
                torch.testing.assert_close(tensor.cpu(), contents, atol=0, rtol=0)
            for recorded, output, reference in graphs:
                recorded.replay()
                torch.testing.assert_close(output.cpu(), reference, atol=0, rtol=0)


def test_bf16_prims_draft_workspace_survives_sleep_and_growth():
    from tokenspeed_kernel.platform import ArchVersion, current_platform

    if not torch.cuda.is_available() or current_platform().arch_version not in (
        ArchVersion(10, 0),
        ArchVersion(10, 3),
    ):
        pytest.skip("requires SM100 or SM103 with CUDA VMM memory-saver hooks")
    _run_with_cuda_memory_saver(_check_bf16_prims_qsa_workspace_on_cuda, timeout=300)


def _check_megamoe_buffers_on_cuda():
    import gc
    import tempfile

    import torch.distributed as dist
    from tokenspeed_kernel.thirdparty.flashinfer import kimi_mega_moe

    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager,
    )
    from tokenspeed.runtime.layers.moe import utils as moe_utils
    from tokenspeed.runtime.layers.moe.expert import MoELayer
    from tokenspeed.runtime.layers.moe.topk import StandardTopKOutput
    from tokenspeed.runtime.layers.quantization.mxfp4 import Mxfp4Config
    from tokenspeed.runtime.utils.env import global_server_args_dict

    torch.cuda.set_device(0)
    moe_utils.MOE_BACKEND = moe_utils.MoeBackend.MEGA_MOE
    moe_utils.ALL2ALL_BACKEND = moe_utils.All2AllBackend.NONE
    global_server_args_dict.update(
        enable_memory_saver=True,
        mapping=Mapping(rank=0, world_size=1),
    )
    saver = adapter.TorchMemorySaverAdapter.create(enable=True, device="cuda")
    with tempfile.TemporaryDirectory() as tmp:
        dist.init_process_group(
            "nccl",
            init_method=f"file://{tmp}/group",
            rank=0,
            world_size=1,
            device_id=torch.device("cuda", 0),
        )
        process_group_manager.init_process_group((0,))
        with torch.device("cuda"), saver.region(tag="weights", enable_cpu_backup=True):
            experts = MoELayer(
                top_k=16,
                num_experts=896,
                hidden_size=3584,
                intermediate_size=3072,
                quant_config=Mxfp4Config(is_checkpoint_mxfp4_serialized=True),
                layer_index=0,
                activation="situ",
                activation_situ_beta=4.0,
                activation_situ_linear_beta=25.0,
                routing_mode="precomputed_topk",
                internal_activation_dtype_override="fp8",
                input_dtype=torch.bfloat16,
            )
            with torch.no_grad():
                experts.w13_weight.fill_(0x11)
                experts.w2_weight.fill_(0x11)
                experts.w13_weight_scale.fill_(124)
                experts.w2_weight_scale.fill_(124)
            experts.process_weights_after_loading(experts)

        for rows in (128, 256, 128):
            with saver.named_region("runtime_states"):
                x = torch.full((rows, 3584), 0.01, device="cuda", dtype=torch.bfloat16)
            buffers = experts.input_buffers(rows, rows)
            ids, scores = buffers[2:]
            ids.copy_(torch.arange(16, device="cuda", dtype=torch.int32).expand_as(ids))
            scores.fill_(1 / 16)
            routing = StandardTopKOutput(scores, ids, None)
            expected = experts(x, routing, rows, rows).cpu()
            assert torch.isfinite(expected).all() and expected.abs().sum() > 0
            pointers = tuple(t.data_ptr() for t in (x, *buffers))
            graph = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with saver.cuda_graph(graph, pool=None, stream=stream, tag="cuda_graph"):
                output = experts(x, routing, rows, rows)
            graph.replay()
            torch.cuda.synchronize()
            for _ in range(2):
                saver.pause(tag="weights")
                torch.cuda.empty_cache()
                ordinary_bytes = sum(
                    segment["total_size"]
                    for segment in torch.cuda.memory_snapshot()
                    if segment["device"] == 0
                    and tuple(segment["segment_pool_id"]) == (0, 0)
                )
                assert ordinary_bytes == 0, ordinary_bytes
                saver.resume(tag="weights")
                assert tuple(t.data_ptr() for t in (x, *buffers)) == pointers
                graph.replay()
                torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
            graph.reset()
            del output, graph, routing, ids, scores, buffers, x
        del experts
        for runner in kimi_mega_moe._runners.values():
            for workspace in runner.workspaces.values():
                workspace.destroy()
        kimi_mega_moe._runners.clear()
        gc.collect()
        torch.cuda.empty_cache()
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is not None
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="requires the FlashInfer Kimi MegaMoE CUDA backend",
)
def test_megamoe_buffers_survive_sleep_with_zero_ordinary_bytes():
    memory_saver = pytest.importorskip("torch_memory_saver")
    with memory_saver.configure_subprocess():
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from test.runtime.test_torch_memory_saver_adapter import "
                "_check_megamoe_buffers_on_cuda; _check_megamoe_buffers_on_cuda()",
            ],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            timeout=300,
        )


def _check_checkpoint_metadata_on_cuda():
    """Retain real checkpoint consumers through growth and repeated sleep/wake."""
    from test.runtime.test_gdn_state_checkpoint_in_forward import (
        _checkpointing_state_backend,
    )

    from tokenspeed.runtime.layers.attention.backends.state.checkpoint import (
        in_forward_state_checkpoints,
    )

    saver = adapter.TorchMemorySaverAdapter.create(enable=True, device="cuda")
    retained = []
    consumers = []
    scans = []
    with saver.named_region("attention_state"):
        backend = _checkpointing_state_backend(spec_tokens=1, device="cuda")
    for offsets in ([256, 0, 128], [0, 0], [0, 64, 0, 64, 0, 64, 0, 64, 0], [64]):
        count = len(offsets)
        host_offsets = torch.tensor(offsets, dtype=torch.int32)
        lengths = torch.full((count,), 512, dtype=torch.int32)
        # This is the same region as ForwardStepRunner._init_forward_metadata
        # and prefill graph construction, including the fresh packed upload.
        with saver.named_region("attention_state"):
            base = torch.arange(count * 3, dtype=torch.int32, device="cuda").view(
                count, 3
            )
            tables = {"state0": base, "state1": base + 100}
            if not any(offsets):
                tables = {group: table[:, :2] for group, table in tables.items()}
            checkpoints = in_forward_state_checkpoints(
                host_offsets, lengths, count, tables, tuple(tables)
            )
            if checkpoints is not None:
                scan, scan_rows = backend._scan_state_checkpoints(checkpoints, lengths)
                scan.states.copy_(
                    torch.arange(
                        scan.states.numel(), dtype=torch.float32, device="cuda"
                    ).reshape_as(scan.states)
                )
                scans.append((scan, scan_rows, scan.states[scan_rows].cpu()))
        tensors = list(tables.values())
        if checkpoints is None:
            assert not any(offsets)
        else:
            rows = [row for row, offset in enumerate(offsets) if offset]
            positions = [row * 512 + offsets[row] for row in rows]
            consumers.append((checkpoints, tables, rows, positions))
            tensors.extend(
                (
                    checkpoints.device_rows,
                    checkpoints.device_offsets,
                    checkpoints.token_index,
                    *checkpoints.blocks_by_group.values(),
                )
            )
            tensors.extend((scan.cu_starts, scan_rows, scan.states))
        retained.extend((tensor, tensor.data_ptr(), tensor.cpu()) for tensor in tensors)
        # Keep the older dataclasses and shared-storage views alive, as captured
        # and last-forward metadata can remain live across release and growth.
        for _ in range(3):
            torch.cuda.synchronize()
            saver.pause(tag="weights")
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            ordinary_bytes = sum(
                segment["total_size"]
                for segment in torch.cuda.memory_snapshot()
                if segment["device"] == torch.cuda.current_device()
                and tuple(segment["segment_pool_id"]) == (0, 0)
            )
            assert ordinary_bytes == 0, ordinary_bytes
            saver.resume(tag="weights")
            for tensor, pointer, expected in retained:
                assert tensor.data_ptr() == pointer
                torch.testing.assert_close(tensor.cpu(), expected, rtol=0, atol=0)
            with saver.named_region("attention_state"):
                for scan, scan_rows, expected in scans:
                    torch.testing.assert_close(
                        scan.states[scan_rows].cpu(), expected, rtol=0, atol=0
                    )
                for checkpoints, tables, rows, positions in consumers:
                    actual_positions = (
                        checkpoints.device_rows * 512 + checkpoints.device_offsets
                    )
                    assert actual_positions.cpu().tolist() == positions
                    for table in tables.values():
                        actual_blocks = table[checkpoints.device_rows, 2]
                        expected_blocks = table.cpu()[rows, 2]
                        torch.testing.assert_close(
                            actual_blocks.cpu(), expected_blocks, rtol=0, atol=0
                        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires native CUDA VMM memory-saver hooks",
)
def test_checkpoint_metadata_survives_sleep_and_growth_with_zero_ordinary_bytes():
    memory_saver = pytest.importorskip("torch_memory_saver")
    with memory_saver.configure_subprocess():
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from test.runtime.test_torch_memory_saver_adapter import "
                "_check_checkpoint_metadata_on_cuda; _check_checkpoint_metadata_on_cuda()",
            ],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            timeout=90,
        )


@pytest.mark.parametrize("with_draft", [False, True])
def test_capture_metadata_survives_sleep_for_target_and_draft(
    memory, buffers, with_draft
):
    from tokenspeed.runtime.execution.forward_step import prepare_capture_metadata

    class CpuAttentionMetadata:
        """Replace GPU metadata construction, retaining the allocated Tensor owner."""

        def __init__(self):
            self.tensor = None

        def init_forward_metadata_capture_cuda_graph(self, batch_size, *args, **kwargs):
            self.tensor = buffers.allocate(batch_size)

    target = CpuAttentionMetadata()
    draft = CpuAttentionMetadata() if with_draft else None
    saver = adapter.TorchMemorySaverAdapter.create(enable=True, device="cuda")
    prepare_capture_metadata(
        target,
        draft,
        memory_saver_adapter=saver,
        batch_size=2,
        req_pool_indices=torch.arange(2),
        seq_lens=torch.ones(2, dtype=torch.int32),
        block_tables={},
        num_tokens=2,
    )
    tensors = [target.tensor]
    if draft is not None:
        tensors.append(draft.tensor)
    addresses = [tensor.data_ptr() for tensor in tensors]
    for _ in range(2):
        saver.pause(tag="weights")
        assert all(tensor.item() == 0 for tensor in tensors)
        saver.resume(tag="weights")
        assert [tensor.data_ptr() for tensor in tensors] == addresses
        assert all(tensor.item() == 2 for tensor in tensors)


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("host_tables", [True, False])
def test_forward_cache_tables_use_restorable_pool_when_retained_between_rounds(
    memory, buffers, monkeypatch, enabled, host_tables
):
    from tokenspeed.runtime.layers.attention.backends import cache_metadata
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
        CacheRuntimeContract,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import CacheGroupSpec

    tensors = []

    def pack(op, device, **kwargs):
        # Replace only GPU packing; metadata validation, shared-storage views
        # and pool dispatch stay real.
        tensor = buffers.allocate(len(op.request_ids))
        tensors.append(tensor)
        return SimpleNamespace(
            tables={"full": tensor.expand(len(op.request_ids), 1)},
            tables_cpu=(
                {"full": tensor.expand(len(op.request_ids), 1).clone()}
                if kwargs["host_tables"]
                else {}
            ),
        )

    monkeypatch.setattr(cache_metadata, "packed_block_tables_from_forward_op", pack)
    group = CacheGroupSpec(
        group_id="full",
        attn_kind="full_history",
        rows_per_page=1,
        entry_stride_tokens=1,
        transfer_policy="full_suffix",
        replayable=False,
    )
    saver = adapter.TorchMemorySaverAdapter.create(enable=enabled, device="cuda")
    contract = CacheRuntimeContract(
        prefix_granularity=1,
        num_lcm_blocks=4,
        token_capacity=4,
        group_specs=(group,),
        group_page_counts={"full": 5},
        group_packing={"full": 1},
    )
    for count in (1, 3):
        op = SimpleNamespace(request_ids=list(range(count)))
        metadata = cache_metadata.build_managed_cache_metadata(
            op,
            memory_saver_adapter=saver,
            device="cpu",
            contract=contract,
            num_requests=count,
            validate_page_ids=True,
            host_tables=host_tables,
        )
        table = metadata.tables(active_forward_op=op)["full"]
        tensor = tensors[-1]
        address = tensor.data_ptr()
        allocation = next(row for row in buffers.allocations if row[1] is tensor)
        assert allocation[0] == ("weights" if enabled else None)
        saver.pause(tag="weights")
        assert tensor.item() == (0 if enabled else count)
        saver.resume(tag="weights")
        assert tensor.data_ptr() == address
        assert tensor.item() == count
        assert table.data_ptr() == address
        assert table.tolist() == [[count]] * count
        if host_tables:
            assert (
                metadata.tables_cpu(active_forward_op=op)["full"].tolist()
                == [[count]] * count
            )


def _check_sampled_drafting_on_cuda(sparse: bool):
    """Check the recorded draft q and the sampler's scratch survive a weights release.

    The drafter is the real one and binds through its own
    ``bind_sampling_backend``: base DFLASH allocating the dense ``draft_probs``
    rows plus the shared sampler's scratch, or DFlash2 allocating its sparse
    ``future_draft_q``. Only the verifier is stood in, since constructing one
    needs the sampling kernels' GPU libraries.

    Args:
        sparse: Bind DFlash2 (sparse q) instead of base DFLASH (dense rows).
    """
    from tokenspeed.runtime.execution.drafter.base import BaseDrafter
    from tokenspeed.runtime.execution.drafter.dflash import DFlash
    from tokenspeed.runtime.execution.drafter.dflash2 import DFlash2
    from tokenspeed.runtime.execution.input_buffer import InputBuffers
    from tokenspeed.runtime.execution.model_executor import ModelExecutor
    from tokenspeed.runtime.execution.runtime_states import RuntimeStates
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessor
    from tokenspeed.runtime.sampling.backends.base import SpeculativeSamplingPools

    vocab, pool, width, max_bs, top_k = 64, 6, 4, 4, 3
    saver = adapter.TorchMemorySaverAdapter.create(enable=True, device="cuda")
    executor = ModelExecutor.__new__(ModelExecutor)
    executor.memory_saver_adapter = saver

    with saver.named_region("runtime_states"):
        executor.runtime_states = RuntimeStates(
            req_pool_size=pool,
            input_vocab_size=vocab,
            vocab_size=vocab,
            output_length=width,
            device="cuda",
        )
        buffers = InputBuffers(
            max_bs=max_bs,
            max_num_tokens=max_bs * width,
            state_write_padding_pool_index=pool,
            device="cuda",
        )
        pools = SpeculativeSamplingPools(
            temperature=torch.ones(pool + 1, device="cuda"),
            top_k=torch.ones(pool + 1, dtype=torch.int32, device="cuda"),
            top_p=torch.ones(pool + 1, device="cuda"),
            seed=torch.zeros(pool + 1, dtype=torch.int64, device="cuda"),
        )

    drafter_cls = DFlash2 if sparse else DFlash
    drafter = drafter_cls.__new__(drafter_cls)
    BaseDrafter.__init__(
        drafter,
        spec_num_tokens=width,
        runtime_states=executor.runtime_states,
        input_buffers=buffers,
    )
    # wire_target binds the target's head and processor; DFLASH has no head
    # of its own, and the pairing check only reads the verifier.
    drafter.logits_processor = LogitsProcessor(
        SimpleNamespace(vocab_size=vocab),
        do_argmax=True,
        dp_lm_head_tp=False,
        attn_tp_size=1,
    )
    if sparse:
        drafter.selector_top_k = top_k
    executor.drafter = drafter
    # Constructing a real verifier needs the sampling kernels' GPU libraries;
    # the bind path reads only these three members of it.
    executor.sampling_backend = SimpleNamespace(
        config=SimpleNamespace(
            enable_speculative_sampling=True, sampling_stream="batch"
        ),
        reads_sparse_draft_q=lambda: sparse,
        speculative_sampling_pools=lambda: pools,
    )
    executor._samples_from_vocab_shard = False
    executor._init_sampled_drafting(
        SimpleNamespace(
            enable_speculative_sampling=True,
            spec_draft_tree=False,
            spec_algo="DFLASH",
            spec_reject_draft_prob_threshold=2.0,
        ),
        width,
    )
    sampler = executor.drafter.draft_sampler
    states = executor.runtime_states
    # Sampling the dense rows needs whole-vocabulary logits, so the bind turns
    # the fused argmax off; the DFlash2 walk reads its selector's top-K instead.
    assert drafter.logits_processor.do_argmax is sparse
    if sparse:
        # DFlash2 records no dense rows; its walk writes the sparse tables.
        assert states.draft_probs is None and sampler is None
        draft_q = states.future_draft_q
        assert draft_q is executor.drafter._draft_q
        retained = [draft_q.q, draft_q.q_ids, draft_q.slot, draft_q.pos]
        draft_q.q[3, 1, :] = torch.tensor([0.5, 0.25, 0.25], device="cuda")
        draft_q.q_ids[3, 1, :] = torch.tensor(
            [7, 8, 9], dtype=torch.int32, device="cuda"
        )
        draft_q.slot[3, :] = 1
    else:
        assert states.draft_probs is not None and sampler is not None
        retained = [states.draft_probs, sampler._offsets_pool, sampler._gumbel_out]
        # Persistent contents, not just addresses: a recorded row must come back.
        states.draft_probs[3, 1, :] = torch.arange(vocab, device="cuda") / vocab
    snapshots = [(t, t.data_ptr(), t.cpu()) for t in retained]
    for _ in range(3):
        torch.cuda.synchronize()
        saver.pause(tag="weights")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        ordinary_bytes = sum(
            segment["total_size"]
            for segment in torch.cuda.memory_snapshot()
            if segment["device"] == torch.cuda.current_device()
            and tuple(segment["segment_pool_id"]) == (0, 0)
        )
        assert ordinary_bytes == 0, ordinary_bytes
        saver.resume(tag="weights")
        for tensor, pointer, reference in snapshots:
            assert tensor.data_ptr() == pointer
            torch.testing.assert_close(tensor.cpu(), reference, rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires native CUDA VMM memory-saver hooks",
)
@pytest.mark.parametrize("sparse", [False, True], ids=["dense-q", "dflash2-sparse-q"])
def test_sampled_drafting_state_survives_sleep_with_zero_ordinary_bytes(sparse):
    memory_saver = pytest.importorskip("torch_memory_saver")
    # Preload hooks must be installed before CUDA initializes in the child.
    with memory_saver.configure_subprocess():
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from test.runtime.test_torch_memory_saver_adapter import "
                "_check_sampled_drafting_on_cuda; "
                f"_check_sampled_drafting_on_cuda({sparse})",
            ],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            timeout=90,
        )


def _check_dspark_adaptive_budget_buffers_on_cuda():
    """Keep DeepseekV41DSpark's persistent and lazy calibration buffers out of (0, 0)."""
    import gc

    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.drafter.deepseek_v41_dspark import (
        DeepseekV41DSpark,
    )
    from tokenspeed.runtime.execution.drafter.verify_calibration import (
        VerifyCalibration,
    )
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.execution.input_buffer import InputBuffers
    from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
    from tokenspeed.runtime.execution.runtime_states import RuntimeStates
    from tokenspeed.runtime.utils.env import global_server_args_dict

    vocab, pool_rows, width, steps, max_bs = 64, 8, 6, 5, 4
    saver = adapter.TorchMemorySaverAdapter.create(enable=True, device="cuda")
    mapping = Mapping(rank=0, world_size=1)
    draft_model = SimpleNamespace(
        mapping=mapping,
        model=SimpleNamespace(
            block_size=steps,
            target_layer_ids=(0,),
            hidden_size=128,
            markov_projection=SimpleNamespace(num_embeddings_per_partition=vocab),
            embed_tokens=SimpleNamespace(weight=torch.empty(0, dtype=torch.bfloat16)),
        ),
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(global_server_args_dict, "enable_dsv41_adaptive_dspark", True)
        patch.setitem(global_server_args_dict, "dsv41_dspark_min_batch_size", 2)
        with saver.named_region("runtime_states"):
            states = RuntimeStates(
                req_pool_size=pool_rows,
                input_vocab_size=vocab,
                vocab_size=vocab,
                output_length=width,
                device="cuda",
            )
            ib = InputBuffers(
                max_bs=max_bs,
                max_num_tokens=max_bs * width,
                state_write_padding_pool_index=pool_rows,
                device="cuda",
            )
            drafter = DeepseekV41DSpark(
                spec_num_tokens=width,
                spec_num_steps=steps,
                draft_model_runner=SimpleNamespace(model=draft_model, device="cuda"),
                attn_backend=None,
                token_to_kv_pool=None,
                runtime_states=states,
                input_buffers=ib,
                vocab_size=vocab,
            )
            drafter.bind_memory_saver_adapter(saver)

    drafter.dp_local_graph = True
    drafter.verify_graph_token_step = 4
    drafter._verify_cache_initialized = True
    ib.req_pool_indices_buf[:max_bs].copy_(
        torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="cuda")
    )
    ib.seq_lens_buf[:max_bs].fill_(128)
    drafter.next_tokens_buf.copy_(
        torch.arange(max_bs * width, dtype=torch.int32, device="cuda").view(
            max_bs, width
        )
    )
    drafter.start_pos_buf.copy_(torch.arange(max_bs, dtype=torch.int64, device="cuda"))
    drafter.confidence_scores[1] = 8.0
    drafter.confidence_scores[2:5] = -8.0

    stream = torch.cuda.Stream()
    pool = torch.cuda.graph_pool_handle()
    drafter.capture_round_graphs(4, stream=stream, pool=pool)
    drafter.install_verify_calibration(
        (VerifyCalibration(4, 1, 8192, 6, 100.0, ((12, 1.0),), 8),),
        policy_ms=0.0,
    )
    buffers = drafter._verify_budget_buffers[4]
    buffers.scores.fill_(1.25)
    buffers.counts.fill_(2)
    buffers.valid.fill_(True)
    buffers.device_decision.copy_(
        torch.tensor([12, 1, 8192], dtype=torch.int32, device="cuda")
    )
    retained = [
        drafter.next_tokens_buf,
        drafter.start_pos_buf,
        drafter.confidence_scores,
        drafter._verify_rows,
        drafter._verify_indptr,
        drafter.confidence_features,
        buffers.scores,
        buffers.counts,
        buffers.valid,
        buffers.packet,
        buffers.device_decision,
        buffers.device_cal.ranges,
        buffers.device_cal.native_ms,
        buffers.device_cal.capacities,
        buffers.device_cal.step_ms,
    ]
    snapshots = [(t, t.data_ptr(), t.cpu()) for t in retained]
    for _ in range(3):
        torch.cuda.synchronize()
        saver.pause(tag="weights")
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        ordinary_bytes = sum(
            segment["total_size"]
            for segment in torch.cuda.memory_snapshot()
            if segment["device"] == torch.cuda.current_device()
            and tuple(segment["segment_pool_id"]) == (0, 0)
        )
        assert ordinary_bytes == 0, ordinary_bytes
        saver.resume(tag="weights")
        for tensor, pointer, reference in snapshots:
            assert tensor.data_ptr() == pointer
            torch.testing.assert_close(tensor.cpu(), reference, rtol=0, atol=0)

    # Post-resume execution and calibration replacement keep (0, 0) empty too.
    drafter.install_verify_calibration(
        (
            VerifyCalibration(4, 1, 8192, 6, 100.0, ((12, 1.0),), 8),
            VerifyCalibration(4, 8193, 16384, 6, 100.0, ((16, 1.0),), 8),
        ),
        policy_ms=0.0,
    )
    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=4,
        num_extends=0,
        input_num_tokens=24,
        forward_mode=ForwardMode.DECODE,
        output_layout=ForwardOutputLayout(0, 0, 4, 6),
    )
    drafter.prepare_request_state([101, 102, 103, 104], [1, 2, 3, 4], num_extends=0)
    drafter.confidence_scores[1] = 8.0
    drafter.confidence_scores[2:5] = -8.0
    drafter.finish_round(ctx, graph_bs=4)
    torch.cuda.synchronize()
    assert (
        drafter.prepare_decode_graph_token_layout(ctx, actual_bs=4, padded_bs=4) == 12
    )
    drafter._verify_budget_graphs.clear()
    drafter.release_round_graphs()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires native CUDA VMM memory-saver hooks",
)
def test_dspark_adaptive_budget_buffers_survive_sleep_with_zero_ordinary_bytes():
    memory_saver = pytest.importorskip("torch_memory_saver")
    with memory_saver.configure_subprocess():
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from test.runtime.test_torch_memory_saver_adapter import "
                "_check_dspark_adaptive_budget_buffers_on_cuda; "
                "_check_dspark_adaptive_budget_buffers_on_cuda()",
            ],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            timeout=90,
        )

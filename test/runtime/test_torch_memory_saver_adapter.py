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


@pytest.fixture(params=["cuda", "hip"])
def memory(monkeypatch, request):
    monkeypatch.setattr(
        torch.version, "cuda", "test" if request.param == "cuda" else None
    )
    monkeypatch.setattr(
        torch.version, "hip", "test" if request.param == "hip" else None
    )
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
    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(torch.version, "hip", None)
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
    "name", ["executor_inputs", "runtime_states", "attention_state", "workspace"]
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


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires native CUDA VMM memory-saver hooks",
)
def test_managed_metadata_survives_sleep_and_growth_with_zero_ordinary_bytes():
    memory_saver = pytest.importorskip("torch_memory_saver")
    # Preload hooks must be installed before CUDA initializes in the child.
    with memory_saver.configure_subprocess():
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from test.runtime.test_torch_memory_saver_adapter import "
                "_check_managed_metadata_on_cuda; _check_managed_metadata_on_cuda()",
            ],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            timeout=90,
        )


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
    saver = adapter.TorchMemorySaverAdapter.create(enable=True)
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
    saver = adapter.TorchMemorySaverAdapter.create(enable=True)
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
    saver = adapter.TorchMemorySaverAdapter.create(enable=enabled)
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

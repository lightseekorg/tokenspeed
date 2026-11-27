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

register_cuda_ci(est_time=30, suite="runtime-1gpu")


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
        _check_weights_release_and_resume(saver, retained)


def _check_weights_release_and_resume(saver, snapshots):
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


def _check_v41_prefill_graph_inputs_on_cuda():
    from test.runtime import test_deepseek_v41_cache as cache
    from test.runtime.test_deepseek_v41_model import _ctx

    from tokenspeed_kernel.ops.attention import dsv41

    from tokenspeed.runtime.execution.breakable_cuda_graph import (
        BreakableCapture,
        active_forward,
    )
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.attention.backends.specific.deepseek_v41 import (
        DeepseekV41AttentionBackend,
    )
    from tokenspeed.runtime.layers.attention.configs.deepseek_v41 import (
        DeepseekV41Config,
    )

    recipe = cache._recipe("cuda")
    pool = cache._pool(recipe, "cuda", enable_memory_saver=True)
    saver = pool.arena.memory_saver_adapter
    config = recipe.attn_config
    with saver.named_region("attention_state"):
        backend = DeepseekV41AttentionBackend(
            config, config.component(DeepseekV41Config)
        )
        backend.set_cache_pool(pool)
        tables = cache._tables("cuda")
    pool.swa(0).zero_()
    for rows, prefix in ((4, 2), (64, 32)):
        with saver.named_region("attention_state"):
            meta = cache._extend(
                backend, tables, [rows], [prefix], [0], [rows + prefix]
            )
        ctx = _ctx(backend, rows, ForwardMode.EXTEND)
        capture = BreakableCapture()
        with active_forward(ctx), capture:
            backend.prepare_prefill_stage(ctx, rows, decoder=True)
            current = backend.prefill_stage.swa_current
            current.fill_(rows)
        current.fill_(-1)
        backend.prepare_prefill_stage(ctx, 64, decoder=False)
        stage = backend.prefill_stage
        assert stage.prefix.shape[0] == prefix
        stage.prefix.fill_(rows)
        plan = backend._swa_query_plan(
            meta.positions, meta.request_indices, ctx.forward_mode, query_shard=None
        )
        layout = backend._prefill_layout(plan, -1, ctx.forward_mode)
        tensors = (
            stage.rows,
            stage.pairs.active,
            stage.prefix,
            stage.descriptor,
            plan.prefix_slots,
            layout.batches[0].swa_indices,
            current,
        )
        _check_weights_release_and_resume(
            saver, [(t, t.data_ptr(), t.cpu()) for t in tensors]
        )
        torch.testing.assert_close(stage.positions[:rows], meta.positions)
        dsv41.snapshot_swa_prefix(pool.swa(0), stage.descriptor, backend.swa_format)
        assert torch.count_nonzero(stage.prefix) == 0
        with active_forward(ctx):
            capture.replay()
        assert torch.all(current == rows)

    # The 64-row graph still owns its workspace when the smaller graph captures.
    with saver.named_region("attention_state"):
        cache._extend(backend, tables, [4], [0], [0], [4])
    ctx = _ctx(backend, 4, ForwardMode.EXTEND)
    smaller = BreakableCapture(pool=capture.pool)
    with active_forward(ctx):
        with smaller:
            backend.prepare_prefill_stage(ctx, 4, decoder=True)
            small = backend.prefill_stage.swa_current
            small.fill_(4)
        smaller.replay()
    assert small.data_ptr() == current.data_ptr()
    assert torch.all(current[:4] == 4) and torch.all(current[4:] == 64)

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
                "from test.runtime import test_torch_memory_saver_adapter as checks\n"
                "checks._check_managed_metadata_on_cuda()\n"
                "checks._check_v41_prefill_graph_inputs_on_cuda()",
            ],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            timeout=180,
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


def _check_checkpoint_metadata_on_cuda():
    """Retain real checkpoint consumers through growth and repeated sleep/wake."""
    from test.runtime.test_gdn_state_checkpoint_in_forward import (
        _checkpointing_state_backend,
    )

    from tokenspeed.runtime.layers.attention.backends.state.checkpoint import (
        in_forward_state_checkpoints,
    )

    saver = adapter.TorchMemorySaverAdapter.create(enable=True)
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
    saver = adapter.TorchMemorySaverAdapter.create(enable=True)
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

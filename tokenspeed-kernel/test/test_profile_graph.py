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

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import struct
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
from tokenspeed_kernel.op_profile import (
    CostRegistry,
    OpProfile,
    ProfileSession,
    wrap_op_for_profiling,
)
from tokenspeed_kernel.profile_graph import (
    GraphCapture,
    GraphReplayCollector,
    IntegerSnapshot,
    MetadataSlotPool,
    ProfileStep,
)
from tokenspeed_kernel.profile_utilization import HardwareProfile
from tokenspeed_kernel.registry import KernelSpec


def test_slot_generations_event_fence_and_duplicate_identity():
    pool = MetadataSlotPool(1, identity_capacity=3)
    first, reason = pool.reserve("first")
    assert reason is None
    assert pool.reserve("second")[1] == "ring_full"
    assert pool.reserve("first")[1] == "duplicate_step_identity"
    event = object()
    pool.queue(first, event)
    with pytest.raises(ValueError, match="exact synchronized"):
        pool.ready_after_sync(first, object())
    with pytest.raises(ValueError, match="Persist ready"):
        pool.release(first)
    pool.ready_after_sync(first, event)
    pool.release(first)
    latest, reason = pool.reserve("third")
    assert reason is None and latest.generation == first.generation + 1
    with pytest.raises(ValueError, match="Stale"):
        pool.queue(first, event)
    assert pool.reserve("fourth")[1] == "step_identity_capacity"


class Observer:
    def __init__(self):
        self.scopes = []

    @contextmanager
    def record_func(self, label, *, display_name):
        self.scopes.append(("enter", label))
        try:
            yield
        finally:
            self.scopes.append(("exit", label))

    def snapshot_integer_values(self, source, target, name):
        target[name] = tuple(source.tolist())


def test_public_module_observer_preserves_forward_and_records_real_parent_scope():
    def operation(x):
        return x + 1

    operation = wrap_op_for_profiling(
        operation,
        spec=KernelSpec(name="leaf", family="test", mode="add"),
        profile=OpProfile(cost_rule=None),
    )

    class Layer(torch.nn.Module):
        def forward(self, x):
            return operation(x)

    module = Layer()
    original = module.forward.__func__
    session = ProfileSession(
        run_id="observer", model_id="fixture", rank=0, backend="none"
    )
    observer = Observer()
    with session.watch_module(module, layer_id=3, name="layer3"), session.observe(
        observer
    ):
        assert session.observed_by(observer)
        with session.iteration(iter_id=1, metadata={"logical_batch": 1}):
            result = module(torch.tensor([2]))
    assert result.item() == 3 and module.forward.__func__ is original
    assert session.ops[0].layer_id == session.layers[0].layer_id == 3
    assert session.ops[0].scalar_args["logical_batch"] == 1
    assert len(observer.scopes) == 4
    assert ProfileSession.current() is None and not session.observed_by(observer)
    assert module(torch.tensor([3])).item() == 4 and len(session.ops) == 1


def test_module_exception_is_not_swallowed_and_context_recovers():
    error = ValueError("original")

    class Broken(torch.nn.Module):
        def forward(self, x):
            raise error

    model = Broken()
    session = ProfileSession(
        run_id="exception", model_id="fixture", rank=0, backend="none"
    )
    with session.watch_module(model, layer_id=0, name="broken"):
        with pytest.raises(ValueError) as caught:
            with session.iteration(iter_id=1):
                model(torch.tensor([1]))
    assert caught.value is error
    assert session.layers[0].error == "ValueError"
    assert ProfileSession.current() is None
    with session.iteration(iter_id=2):
        assert ProfileSession.current() is session


def test_explicit_dotted_paths_and_integer_matrix_preserve_arguments():
    from dataclasses import dataclass

    @dataclass
    class Metadata:
        indices: torch.Tensor
        scale: int

    calls = []

    def fn(x, meta):
        calls.append((x, meta))
        return x * meta.scale

    wrapped = wrap_op_for_profiling(
        fn,
        spec=KernelSpec(name="paths", family="test", mode="paths"),
        profile=OpProfile(
            cost_rule=None,
            scalar_args=("meta.scale",),
            integer_tensor_args=("meta.indices",),
            tensor_args=("meta.absent",),
        ),
    )
    matrix = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    metadata = Metadata(matrix, 3)
    session = ProfileSession(run_id="paths", model_id="fixture", rank=0, backend="none")
    x = torch.tensor([2])
    with session.iteration(iter_id=1):
        result = wrapped(x, metadata)
    assert calls[0][0] is x and calls[0][1] is metadata and result.item() == 6
    assert session.ops[0].tensor_values["meta.indices"] == (1, 2, 3, 4)
    assert session.ops[0].scalar_args["meta.scale"] == 3
    assert "meta.absent" not in session.ops[0].inputs
    assert metadata.indices.data_ptr() == matrix.data_ptr()
    with session.iteration(iter_id=2):
        wrapped(x, Metadata(matrix.T, 3))
    assert session.ops[1].tensor_values["meta.indices"] is None


@pytest.fixture
def cpu_gpu_allocations(monkeypatch):
    # Only GPU/pinned allocation unavailable on this CPU host is replaced.
    original_empty, original_zeros = torch.empty, torch.zeros

    def empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        kwargs.pop("device", None)
        return original_empty(*args, **kwargs)

    def zeros(*args, **kwargs):
        kwargs.pop("device", None)
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(torch, "zeros", zeros)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)


def captured(tmp_path):
    cap = GraphCapture(
        session=ProfileSession(
            run_id="capture", model_id="fixture", rank=0, backend="none"
        ),
        costs=CostRegistry(),
        hardware=HardwareProfile("fixture", {}, None, "unit test"),
        capture_id="capture",
        graph_key=("default", 1),
        nodes=None,
        max_metadata_bytes=64,
        max_scopes=4,
        metadata={"run_id": "test"},
        output=tmp_path / "capture.json",
        modules=[],
    )
    cap.used = 16
    cap.snapshots = [IntegerSnapshot("op", "routes", 0, 4, "torch.int32")]
    return cap


def test_index_rows_are_visible_in_order_and_output_is_exclusive(
    tmp_path, cpu_gpu_allocations
):
    output = tmp_path / "replays.jsonl"
    collector = GraphReplayCollector(
        output=output, metadata_slots=1, max_metadata_bytes=64, max_steps=1
    )
    expected = b""
    try:
        with pytest.raises(ValueError, match="already exists"):
            GraphReplayCollector(
                output=output, metadata_slots=1, max_metadata_bytes=64, max_steps=1
            )
        for row in ({"event": "first", "name": "层"}, {"event": "second", "step": 2}):
            collector.emit(row)
            expected += (json.dumps(row, allow_nan=False) + "\n").encode("utf-8")
            assert output.read_bytes() == expected
    finally:
        collector.close()
    collector.close()
    assert output.read_bytes() == expected
    with pytest.raises(ValueError, match="closed file"):
        collector.emit({"event": "after_close"})


def test_ready_result_writer_persists_binary_before_slot_reuse(
    tmp_path, cpu_gpu_allocations
):
    collector = GraphReplayCollector(
        output=tmp_path / "replays.jsonl",
        metadata_slots=1,
        max_metadata_bytes=64,
        max_steps=2,
    )
    step = ProfileStep(
        owner=collector,
        run_id="test",
        rank=0,
        serving_step=1,
        request_ids=("request",),
        pool_indices=(7,),
    )
    step.capture = captured(tmp_path)
    step.lease, reason = collector.slots.reserve(step.step_id)
    collector.host_buffers[0][:16].view(torch.int32).copy_(
        torch.tensor([9, 10, 11, 12], dtype=torch.int32)
    )
    event = object()
    step.bind_result_event(event)
    step.bind_forward_id(42)
    with pytest.raises(ValueError, match="exact result"):
        step.device_results_ready(object())
    step.device_results_ready(event)
    state = collector.flush_ready()
    assert state["writer_error"] is None and state["persisted_payloads"] == 1
    row = json.loads((tmp_path / "replays.jsonl").read_text())
    payload = Path(row["binary_payload"]["path"]).read_bytes()
    assert struct.unpack("4i", payload) == (9, 10, 11, 12)
    assert row["binary_payload"]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert (
        row["forward_id"] == 42
        and row["completion_identity_state"] == "completion_identity_unavailable"
    )
    assert "integer_values" not in row
    replacement, reason = collector.slots.reserve("another")
    assert reason is None and replacement.generation == step.lease.generation + 1
    with pytest.raises(ValueError, match="exact result"):
        step.device_results_ready(event)
    collector.close()
    collector.close()


def test_eager_reservation_is_bounded_and_independent_of_graph_slots(
    tmp_path, cpu_gpu_allocations
):
    collector = GraphReplayCollector(
        output=tmp_path / "replays.jsonl",
        metadata_slots=1,
        max_metadata_bytes=64,
        max_steps=3,
    )

    def step(number):
        return ProfileStep(
            owner=collector,
            run_id="test",
            rank=0,
            serving_step=number,
            request_ids=("request",),
            pool_indices=(7,),
        )

    try:
        first, second, third = (step(number) for number in (1, 2, 3))
        with pytest.raises(RuntimeError, match="forward failed"):
            with collector.reserve_eager_metadata(first) as reserved:
                assert reserved
                assert collector.slots.reserve("graph")[0] is not None
                with collector.reserve_eager_metadata(second) as overflow:
                    assert not overflow
                event = object()
                second.bind_result_event(event)
                second.device_results_ready(event)
                raise RuntimeError("forward failed")
        with collector.reserve_eager_metadata(third) as reserved:
            assert reserved
        state = collector.flush_ready()
        row = json.loads(collector.output.read_text())
        assert row["step_id"] == second.step_id and row["device_ready_sequence"] == 1
        assert row["unavailable_reason"] == "metadata_dropped:eager_postwindow_capacity"
        assert state["dropped_metadata"] == 1 and state["persisted_eager"] == 0
    finally:
        collector.close()


def test_failed_persistence_retains_lease_and_reports_missing(
    tmp_path, cpu_gpu_allocations
):
    collector = GraphReplayCollector(
        output=tmp_path / "replays.jsonl",
        metadata_slots=1,
        max_metadata_bytes=64,
        max_steps=2,
    )
    step = ProfileStep(
        owner=collector,
        run_id="test",
        rank=0,
        serving_step=1,
        request_ids=("request",),
        pool_indices=(7,),
    )
    step.capture = captured(tmp_path)
    step.lease, _ = collector.slots.reserve(step.step_id)
    (tmp_path / "rank-0.step-1.metadata.bin").write_bytes(b"previous evidence")
    event = object()
    step.bind_result_event(event)
    step.device_results_ready(event)
    collector.close()
    row = json.loads((tmp_path / "replays.jsonl").read_text())
    assert "persistence_failed" in row["unavailable_reason"]
    assert collector.slots.reserve("second")[1] == "ring_full"
    assert (
        tmp_path / "rank-0.step-1.metadata.bin"
    ).read_bytes() == b"previous evidence"


def test_index_flush_failure_retains_slots_and_does_not_hang(
    tmp_path, cpu_gpu_allocations
):
    import threading

    collector = GraphReplayCollector(
        output=tmp_path / "index.jsonl",
        metadata_slots=2,
        max_metadata_bytes=64,
        max_steps=3,
    )

    # Keep a real descriptor open, but make writes fail at the OS boundary.
    # TextIOWrapper buffers write(), so only a per-row flush exposes this error.
    readonly = os.open(os.devnull, os.O_RDONLY)
    try:
        os.dup2(readonly, collector._index_stream.fileno())
    finally:
        os.close(readonly)
    for index in range(2):
        step = ProfileStep(
            owner=collector,
            run_id="test",
            rank=0,
            serving_step=index + 1,
            request_ids=("request",),
            pool_indices=(7,),
        )
        step.capture = captured(tmp_path)
        step.lease, _ = collector.slots.reserve(step.step_id)
        event = object()
        step.bind_result_event(event)
        step.device_results_ready(event)
    result = {}

    def flush():
        result.update(collector.flush_ready())

    waiter = threading.Thread(target=flush, daemon=True)
    waiter.start()
    waiter.join(timeout=2)
    assert not waiter.is_alive()
    assert "failure index unavailable" in result["writer_error"]
    assert result["persisted_payloads"] == 0
    assert result["dropped_metadata"] == 2
    assert result["slot_states"]["ready"] == 2
    assert collector.slots.reserve("third")[1] == "ring_full"
    assert collector.output.read_bytes() == b""
    with pytest.raises(OSError):
        collector.close()
    collector.close()


def test_window_is_relative_to_profile_arm_not_warmup():
    from tokenspeed_kernel.profile_graph import ProfileWindow

    window = ProfileWindow(start_step=2, num_steps=2)
    for _ in range(20):
        assert not window.select()
    assert window.observed == window.selected == 0
    window.arm()
    assert [window.select() for _ in range(5)] == [False, True, True, False, False]
    window.stop()
    assert not window.select() and window.observed == 5
    with pytest.raises(ValueError, match="one window"):
        window.arm()


def test_declared_paths_skip_properties_and_snapshot_namedtuple_matrix():
    from typing import NamedTuple

    class Routes(NamedTuple):
        topk_ids: torch.Tensor

    class Metadata:
        @property
        def forbidden(self):
            raise AssertionError("metadata must not execute properties")

    def fn(meta, routes):
        return routes.topk_ids

    wrapped = wrap_op_for_profiling(
        fn,
        spec=KernelSpec(name="paths", family="test", mode="paths"),
        profile=OpProfile(
            cost_rule=None,
            scalar_args=("meta.forbidden",),
            integer_tensor_args=("routes.topk_ids",),
        ),
    )
    matrix = torch.tensor([[1, 2]], dtype=torch.int32)
    session = ProfileSession(
        run_id="declared", model_id="fixture", rank=0, backend="none"
    )
    with session.iteration(iter_id=1):
        assert wrapped(Metadata(), Routes(matrix)) is matrix
    assert session.ops[0].scalar_args["meta.forbidden"] is None
    assert session.ops[0].tensor_values["routes.topk_ids"] == (1, 2)


def test_graph_observer_never_reuses_capture_time_cpu_lengths(
    tmp_path, cpu_gpu_allocations
):
    cap = captured(tmp_path)
    target = {}
    cap.snapshot_integer_values(torch.tensor([7], dtype=torch.int32), target, "lengths")
    assert target["lengths"] is None
    assert (
        cap.unavailable[-1]["reason"]
        == "cpu_integer_argument_not_dynamic_capture_bound"
    )
    assert cap.unavailable[-1]["trace_label"] == str(id(target))


def test_result_exact_sync_gate_and_late_identity_drop_reconciliation(
    tmp_path, cpu_gpu_allocations
):
    import importlib.util
    import sys

    source = (
        Path(__file__).resolve().parents[2]
        / "python/tokenspeed/runtime/execution/types.py"
    )
    spec = importlib.util.spec_from_file_location("graph_test_execution_types", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    class ResultEvent:
        def __init__(self):
            self.waits = 0

        def synchronize(self):
            self.waits += 1

    collector = GraphReplayCollector(
        output=tmp_path / "ready.jsonl",
        metadata_slots=1,
        max_metadata_bytes=64,
        max_steps=1,
    )
    step = ProfileStep(
        owner=collector,
        run_id="late",
        rank=0,
        serving_step=20,
        request_ids=("rid",),
        pool_indices=(2,),
    )
    step.unavailable("prefill_graph_capture_not_observed")
    event = ResultEvent()
    step.bind_result_event(event)
    result = module.ModelExecutionResult(
        torch.tensor([1]), copy_event=event, profile_identity=step
    )
    result.sync()
    step.bind_forward_id(200)
    with pytest.raises(RuntimeError, match="exactly once"):
        result.sync()
    rows = [
        json.loads(row) for row in (tmp_path / "ready.jsonl").read_text().splitlines()
    ]
    state = collector.flush_ready()
    assert (
        event.waits == 1
        and state["dropped_metadata"] == 1
        and state["unresolved_result_metadata"] == 0
    )
    assert rows[0]["forward_id"] is None and rows[1]["forward_id"] == 200
    assert (
        rows[0]["request_ids"] == ["rid"]
        and rows[1]["event"] == "forward_identity_bound"
    )
    assert state["writer_error"] is None and state["persisted_payloads"] == 0
    collector.close()


def test_runtime_window_stop_manifest_and_config_snapshot(
    tmp_path, cpu_gpu_allocations
):
    import importlib.util
    import sys

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "python"))
    from tokenspeed.runtime.execution.utilization_profile import (
        RuntimeUtilizationProfile,
        UtilizationProfileConfig,
    )

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Identity()])

    hardware = tmp_path / "hardware.json"
    hardware.write_text(
        json.dumps(
            dict(
                name="fixture",
                compute_flops_per_second={"fp32_simt": 1e12},
                hbm_bytes_per_second=1e12,
                source="synthetic CPU test",
            )
        )
    )
    config = UtilizationProfileConfig(
        run_id="window",
        output_dir=str(tmp_path / "profile"),
        hardware=str(hardware),
        layers="layers",
        start_step=1,
        num_steps=1,
        max_scopes=16,
        hbm_policy="unavailable",
    )
    profile = RuntimeUtilizationProfile(
        config, model=Model(), model_id="fixture", rank=0, graph_enabled=True
    )
    assert (
        profile.begin_step(step=1, request_ids=("warmup",), pool_indices=(1,)) is None
    )
    profile.arm()
    step = profile.begin_step(step=20, request_ids=("measured",), pool_indices=(2,))
    event = object()
    step.unavailable("prefill_graph_capture_not_observed")
    step.bind_forward_id(200)
    step.bind_result_event(event)
    step.device_results_ready(event)
    state = profile.stop()
    assert (
        state["observed_serving_steps"] == 1
        and state["selected_steps"] == 1
        and state["dropped_metadata"] == 1
    )
    assert state["writer_flushed"] and state["unresolved_result_metadata"] == 0
    assert json.loads((tmp_path / "profile/rank-0.window.json").read_text()) == state
    saved = json.loads((tmp_path / "profile/rank-0.config.json").read_text())
    assert hashlib.sha256(saved["contents"].encode()).hexdigest() == saved["sha256"]
    assert saved["config"]["hardware"] == str(hardware)
    assert (
        profile.begin_step(step=21, request_ids=("after",), pool_indices=(2,)) is None
    )
    profile._replays.close()

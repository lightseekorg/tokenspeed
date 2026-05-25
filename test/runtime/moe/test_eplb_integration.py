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

import inspect
from collections import OrderedDict
from types import SimpleNamespace

import torch


def test_scheduler_utils_builds_eplb_controller_config():
    from tokenspeed.runtime.engine.scheduler_utils import make_eplb_controller_config

    cfg = make_eplb_controller_config(
        SimpleNamespace(
            enable_eplb=True,
            eplb_warmup_steps=3,
            eplb_rebalance_interval=7,
            eplb_max_layers_per_step=2,
            eplb_max_rebalance_period_s=1.5,
            eplb_planner_timeout_s=0.25,
            eplb_max_consecutive_failures=4,
        )
    )
    assert cfg.enabled is True
    assert cfg.warmup_steps == 3
    assert cfg.interval == 7
    assert cfg.max_layers_per_step == 2
    assert cfg.max_rebalance_period_ms == 1500
    assert cfg.planner_timeout_ms == 250
    assert cfg.max_consecutive_failures == 4


def test_eplb_runtime_events_match_scheduler_binding_fields():
    from tokenspeed.runtime.moe.eplb_runtime import _make_event

    event = _make_event(
        "PlanDone",
        op_id=1,
        plan_handle=2,
        layer_ids=[5, 6],
        balancedness=0.5,
        balancedness_before=0.4,
    )
    assert event.op_id == 1
    assert event.plan_handle == 2
    assert list(event.layers_changed) == [5, 6]
    assert event.balancedness_before == 0.4
    assert event.balancedness_pred == 0.5


def test_scheduler_utils_serializes_eplb_events_for_rank_sync():
    from tokenspeed_scheduler import EPLB

    from tokenspeed.runtime.engine.scheduler_utils import (
        eplb_event_from_payload,
        eplb_event_to_payload,
        pop_common_eplb_event_payloads,
    )

    event = EPLB.PlanDone()
    event.op_id = 7
    event.plan_handle = 11
    event.layers_changed = [5, 2]
    event.balancedness_before = 0.25
    event.balancedness_pred = 0.75

    payload = eplb_event_to_payload(event)
    assert payload == {
        "kind": "PlanDone",
        "op_id": 7,
        "plan_handle": 11,
        "layers_changed": [5, 2],
        "balancedness_before": 0.25,
        "balancedness_pred": 0.75,
    }
    rebuilt = eplb_event_from_payload(payload)
    assert rebuilt.op_id == 7
    assert rebuilt.plan_handle == 11
    assert list(rebuilt.layers_changed) == [5, 2]

    assert pop_common_eplb_event_payloads([[payload], []]) == []
    assert pop_common_eplb_event_payloads([[payload], [dict(payload)]]) == [payload]


def test_event_loop_commits_eplb_events_after_all_ep_ranks_report(monkeypatch):
    from tokenspeed_scheduler import EPLB

    from tokenspeed.runtime.engine import event_loop as event_loop_mod
    from tokenspeed.runtime.engine.event_loop import EventLoop

    event = EPLB.PlanDone()
    event.op_id = 7
    event.plan_handle = 11
    event.layers_changed = [2, 5]
    event.balancedness_before = 0.25
    event.balancedness_pred = 0.75

    loop = object.__new__(EventLoop)
    loop.model_executor = SimpleNamespace(drain_eplb_events=lambda: [event])
    loop._pending_eplb_event_payloads = OrderedDict()
    loop._num_inflight_eplb_plan_ops = 1
    loop._num_inflight_eplb_relocate_ops = 0
    loop.eplb_event_pg = object()
    loop.scheduler = object()

    advanced = []
    monkeypatch.setattr(
        event_loop_mod,
        "advance_eplb",
        lambda scheduler, events: advanced.extend(events),
    )
    monkeypatch.setattr(event_loop_mod.dist, "get_world_size", lambda group=None: 2)

    def gather_missing_peer(out, local, group=None):
        out[:] = [list(local), []]

    monkeypatch.setattr(event_loop_mod.dist, "all_gather_object", gather_missing_peer)
    loop._commit_eplb_events()
    assert advanced == []
    assert loop._num_inflight_eplb_plan_ops == 1
    assert len(loop._pending_eplb_event_payloads) == 1

    loop.model_executor = SimpleNamespace(drain_eplb_events=lambda: [])

    def gather_all_peers(out, local, group=None):
        out[:] = [list(local), [dict(local[0])]]

    monkeypatch.setattr(event_loop_mod.dist, "all_gather_object", gather_all_peers)
    loop._commit_eplb_events()
    assert len(advanced) == 1
    assert advanced[0].op_id == 7
    assert loop._num_inflight_eplb_plan_ops == 0
    assert loop._pending_eplb_event_payloads == OrderedDict()


def test_event_loop_participates_while_relocate_is_in_flight(monkeypatch):
    from tokenspeed.runtime.engine import event_loop as event_loop_mod
    from tokenspeed.runtime.engine.event_loop import EventLoop

    loop = object.__new__(EventLoop)
    loop.model_executor = SimpleNamespace(drain_eplb_events=lambda: [])
    loop._pending_eplb_event_payloads = OrderedDict()
    loop._num_inflight_eplb_plan_ops = 0
    loop._num_inflight_eplb_relocate_ops = 1
    loop.eplb_event_pg = object()
    loop.scheduler = object()

    gathered = []
    advanced = []
    monkeypatch.setattr(
        event_loop_mod,
        "advance_eplb",
        lambda scheduler, events: advanced.extend(events),
    )
    monkeypatch.setattr(event_loop_mod.dist, "get_world_size", lambda group=None: 2)

    def gather_no_ready(out, local, group=None):
        gathered.append(list(local))
        out[:] = [[], []]

    monkeypatch.setattr(event_loop_mod.dist, "all_gather_object", gather_no_ready)

    loop._commit_eplb_events()

    assert gathered == [[]]
    assert advanced == []
    assert loop._num_inflight_eplb_relocate_ops == 1


def test_eplb_metrics_skip_forward_collective(monkeypatch):
    from tokenspeed.runtime.moe import distribution_recorder

    reduce_called = False

    def fake_reduce(*args, **kwargs):
        nonlocal reduce_called
        reduce_called = True
        raise AssertionError("EPLB metrics must not reduce on the forward path")

    monkeypatch.setattr(distribution_recorder.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(distribution_recorder.torch.distributed, "reduce", fake_reduce)

    accumulator = distribution_recorder._UtilizationRateAccumulator(
        SimpleNamespace(enable_expert_distribution_metrics=True, enable_eplb=True),
        SimpleNamespace(ep_size=2),
        rank=0,
        pg=object(),
    )
    accumulator._append_utilization_rate(
        1, torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
    )

    assert reduce_called is False


def test_planner_runs_cpu_algorithm_without_materializing_cuda_metadata(monkeypatch):
    from tokenspeed.runtime.moe import eplb_planner

    common = {
        "num_physical_experts": 4,
        "num_local_physical_experts": 4,
        "ep_size": 1,
        "model_config_for_expert_location": SimpleNamespace(
            num_groups=None,
            num_logical_experts=4,
        ),
    }
    monkeypatch.setattr(
        eplb_planner.ExpertLocationMetadata,
        "_init_common",
        staticmethod(lambda server_args, model_config: common),
    )

    def fail_cuda_metadata(*args, **kwargs):
        raise AssertionError("planner thread must not materialize CUDA metadata")

    def fake_rebalance_experts(**kwargs):
        assert kwargs["tokens_per_expert"].device.type == "cpu"
        return (
            torch.tensor([[0, 1, 2, 3]], dtype=torch.int64),
            torch.tensor([[[0], [1], [2], [3]]], dtype=torch.int64),
            torch.ones((1, 4), dtype=torch.int64),
        )

    monkeypatch.setattr(eplb_planner, "_dist_ready", lambda: False)
    monkeypatch.setattr(
        eplb_planner.ExpertLocationMetadata,
        "init_by_eplb",
        staticmethod(fail_cuda_metadata),
    )
    monkeypatch.setattr(
        eplb_planner.eplb_algorithms,
        "rebalance_experts",
        fake_rebalance_experts,
    )

    result = eplb_planner.run_planner_with_broadcast(
        torch.ones((1, 4), dtype=torch.int64),
        rank=0,
        current_metadata=SimpleNamespace(
            physical_to_logical_map_cpu=torch.tensor([[0, 1, 2, 3]])
        ),
        server_args=SimpleNamespace(
            eplb_algorithm="auto",
            mapping=SimpleNamespace(nnodes=1, moe=SimpleNamespace(ep_size=1)),
        ),
        model_config=SimpleNamespace(),
    )

    assert result.changed_layers == []
    assert torch.equal(result.physical_to_logical_map_cpu, torch.tensor([[0, 1, 2, 3]]))


def test_planner_broadcast_uses_control_group(monkeypatch):
    from tokenspeed.runtime.moe import eplb_planner

    control_pg = object()
    nccl_pg = object()
    expected = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
    called = {}

    def fake_broadcast_object_list(payload, src, group):
        called["src"] = src
        called["group"] = group
        payload[0] = expected

    monkeypatch.setattr(eplb_planner, "_dist_ready", lambda: True)
    monkeypatch.setattr(eplb_planner, "_group_rank", lambda group=None: 1)
    monkeypatch.setattr(eplb_planner, "_group_src_rank", lambda group=None: 0)
    monkeypatch.setattr(
        eplb_planner.dist,
        "broadcast_object_list",
        fake_broadcast_object_list,
    )

    result = eplb_planner.run_planner_with_broadcast(
        torch.ones((1, 4), dtype=torch.int64),
        rank=1,
        current_metadata=SimpleNamespace(physical_to_logical_map_cpu=expected),
        server_args=SimpleNamespace(),
        model_config=SimpleNamespace(),
        eplb_pg=nccl_pg,
        eplb_control_pg=control_pg,
    )

    assert called == {"src": 0, "group": control_pg}
    assert torch.equal(result.physical_to_logical_map_cpu, expected)


def test_expert_location_update_reuses_precomputed_dispatch_map(monkeypatch):
    from tokenspeed.runtime.moe import expert_location
    from tokenspeed.runtime.moe.expert_location import ExpertLocationMetadata

    def fail_recompute(*args, **kwargs):
        raise AssertionError("swap update must not recompute every layer dispatch map")

    monkeypatch.setattr(
        expert_location,
        "compute_logical_to_rank_dispatch_physical_map",
        fail_recompute,
    )
    monkeypatch.setattr(expert_location.torch.distributed, "get_world_size", lambda: 2)

    current = ExpertLocationMetadata(
        physical_to_logical_map=torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
        physical_to_logical_map_cpu=torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
        logical_to_all_physical_map=torch.tensor(
            [
                [[0, -1, -1, -1], [1, -1, -1, -1], [2, -1, -1, -1], [3, -1, -1, -1]],
                [[0, -1, -1, -1], [1, -1, -1, -1], [2, -1, -1, -1], [3, -1, -1, -1]],
            ]
        ),
        logical_to_all_physical_map_num_valid=torch.ones((2, 4), dtype=torch.int64),
        logical_to_rank_dispatch_physical_map=torch.tensor(
            [[10, 11, 12, 13], [20, 21, 22, 23]]
        ),
        ep_dispatch_algorithm="static",
    )
    planned = ExpertLocationMetadata(
        physical_to_logical_map=torch.tensor([[3, 2, 1, 0], [3, 2, 1, 0]]),
        physical_to_logical_map_cpu=torch.tensor([[3, 2, 1, 0], [3, 2, 1, 0]]),
        logical_to_all_physical_map=torch.tensor(
            [
                [[3, -1, -1, -1], [2, -1, -1, -1], [1, -1, -1, -1], [0, -1, -1, -1]],
                [[3, -1, -1, -1], [2, -1, -1, -1], [1, -1, -1, -1], [0, -1, -1, -1]],
            ]
        ),
        logical_to_all_physical_map_num_valid=torch.full((2, 4), 2, dtype=torch.int64),
        logical_to_rank_dispatch_physical_map=torch.tensor(
            [[30, 31, 32, 33], [40, 41, 42, 43]]
        ),
        ep_dispatch_algorithm="static",
    )

    current.update(planned, [1])

    assert torch.equal(current.physical_to_logical_map[0], torch.tensor([0, 1, 2, 3]))
    assert torch.equal(current.physical_to_logical_map[1], torch.tensor([3, 2, 1, 0]))
    assert torch.equal(
        current.logical_to_rank_dispatch_physical_map,
        torch.tensor([[10, 11, 12, 13], [40, 41, 42, 43]]),
    )


def test_event_loop_creates_dedicated_eplb_control_group():
    from tokenspeed.runtime.engine.event_loop import EventLoop

    source = inspect.getsource(EventLoop._init_eplb_control_process_group)
    assert "dist.new_group" in source
    assert "pg_manager.get_process_group" not in source


def test_topk_resolves_eplb_dispatch_info_from_current_layer():
    from tokenspeed.runtime.layers.moe.topk import (
        _resolve_expert_location_dispatch_info,
    )
    from tokenspeed.runtime.moe.distribution_recorder import (
        set_global_expert_distribution_recorder,
    )
    from tokenspeed.runtime.moe.eplb_runtime import set_global_eplb_runtime

    sentinel = object()

    class Recorder:
        current_layer_idx = 9

    class Runtime:
        def get_dispatch_info(self, layer_idx):
            assert layer_idx == 9
            return sentinel

    try:
        set_global_expert_distribution_recorder(Recorder())
        set_global_eplb_runtime(Runtime())
        assert _resolve_expert_location_dispatch_info(None) is sentinel
        explicit = object()
        assert _resolve_expert_location_dispatch_info(explicit) is explicit
    finally:
        set_global_expert_distribution_recorder(None)
        set_global_eplb_runtime(None)


def test_trivial_eplb_metadata_matches_checkpoint_loaded_slots(monkeypatch):
    from tokenspeed.runtime.moe.expert_location import (
        ExpertLocationMetadata,
        ModelConfigForExpertLocation,
    )

    monkeypatch.setattr(
        ModelConfigForExpertLocation,
        "from_model_config",
        lambda model_config: ModelConfigForExpertLocation(
            num_layers=1,
            num_logical_experts=8,
            num_groups=None,
        ),
    )
    server_args = SimpleNamespace(
        device="cpu",
        ep_num_redundant_experts=4,
        ep_dispatch_algorithm="static",
        mapping=SimpleNamespace(
            nnodes=1,
            moe=SimpleNamespace(ep_size=2, ep_rank=0),
        ),
    )
    metadata = ExpertLocationMetadata.init_trivial(
        server_args, SimpleNamespace(hf_config=SimpleNamespace())
    )

    assert metadata.physical_to_logical_map_cpu.tolist() == [
        [0, 1, 2, 3, -1, -1, 4, 5, 6, 7, -1, -1]
    ]
    assert metadata.logical_to_all_physical_map[0, 4, 0].item() == 6
    assert torch.all(metadata.logical_to_all_physical_map[0, :, 0] >= 0)


def test_eplb_runtime_persists_snapshots(tmp_path):
    from tokenspeed.runtime.moe.eplb_runtime import EplbRuntime

    runtime = EplbRuntime(
        server_args=SimpleNamespace(eplb_persist_dir=str(tmp_path)),
        model_config=SimpleNamespace(),
        initial_metadata=SimpleNamespace(),
        recorder=SimpleNamespace(),
        rank=0,
        ep_size=1,
    )
    runtime._persist("unit", {"value": torch.tensor([3], dtype=torch.int32)})

    files = list(tmp_path.glob("*_unit.pt"))
    assert len(files) == 1
    payload = torch.load(files[0], weights_only=True)
    assert payload["value"].item() == 3


def test_trtllm_self_routing_expands_logical_logits_to_physical_slots():
    from tokenspeed.runtime.layers.moe.backends.nvfp4.flashinfer_trtllm import (
        _expand_self_routing_logits_for_eplb,
    )

    info = SimpleNamespace(
        ep_dispatch_algorithm="static",
        partial_logical_to_all_physical_map=torch.tensor([[2, -1], [0, 3]]),
        num_physical_experts=4,
    )
    logits = torch.tensor([[10.0, 20.0]], dtype=torch.float32)

    out = _expand_self_routing_logits_for_eplb(logits, info, num_experts=4)

    assert out.shape == (1, 4)
    assert out[0, 2] == 10.0
    assert out[0, 0] == 20.0
    assert out[0, 1] < -1e30
    assert out[0, 3] < -1e30


def test_trtllm_self_routing_records_physical_topk():
    from tokenspeed.runtime.layers.moe.backends.nvfp4.flashinfer_trtllm import (
        _record_self_routing_topk_for_eplb,
    )
    from tokenspeed.runtime.moe.distribution_recorder import (
        set_global_expert_distribution_recorder,
    )

    calls = []

    class Recorder:
        recording = True

        def on_select_experts(self, *, topk_ids, num_experts=None):
            calls.append((topk_ids.cpu(), num_experts))

    try:
        set_global_expert_distribution_recorder(Recorder())
        _record_self_routing_topk_for_eplb(
            torch.tensor([[1.0, 5.0, -3.0, 2.0]], dtype=torch.float32),
            SimpleNamespace(top_k=2, num_fused_shared_experts=0),
            num_experts=4,
        )
    finally:
        set_global_expert_distribution_recorder(None)

    assert len(calls) == 1
    assert calls[0][1] == 4
    assert torch.equal(calls[0][0], torch.tensor([[1, 3]]))


def test_model_executor_wraps_forward_step_with_distribution_recorder():
    import inspect

    from tokenspeed.runtime.execution.model_executor import ModelExecutor

    source = inspect.getsource(ModelExecutor.execute_forward_op)

    assert "expert_distribution_recorder" in source
    assert "with_forward_pass" in source

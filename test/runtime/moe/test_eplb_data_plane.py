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

from types import SimpleNamespace

import torch


class _FakeDoneEvent:
    def __init__(self):
        self.synced = False

    def synchronize(self):
        self.synced = True

    def query(self):
        return True


class _FakeRecorder:
    def __init__(self, snap):
        self.snap = snap
        self.rebound = None

    def snapshot_logical_count(self, reset=True):
        self.reset = reset
        return self.snap

    def rebind(self, metadata):
        self.rebound = metadata


class _FakeMetadata:
    def __init__(self):
        self.updated = []

    def update(self, other, update_layer_ids):
        self.updated.append((other, list(update_layer_ids)))


def test_eplb_logger_rate_limit(monkeypatch):
    from tokenspeed.runtime.moe import eplb_logger

    calls = []
    monkeypatch.setattr(
        eplb_logger._LOG, "info", lambda *args, **kwargs: calls.append(args)
    )
    for _ in range(25):
        eplb_logger.info_rate("skip warmup", every=10)
    assert len(calls) == 2


def test_host_cache_enumerates_expert_dim_params(monkeypatch):
    from tokenspeed.runtime.moe.eplb_host_cache import HostWeightCache

    class FakeLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_index = 3
            self.prefix = "model.layers.3.mlp"
            self.ep_rank = 0
            self.num_local_experts = 2
            self.w13_weight = torch.nn.Parameter(
                torch.arange(16, dtype=torch.float32).view(2, 2, 4)
            )
            self.w2_weight = torch.nn.Parameter(
                torch.arange(16, 32, dtype=torch.float32).view(2, 4, 2)
            )
            self.router = torch.nn.Parameter(torch.ones(4, 4))
            self.w13_weight_scale_inv = torch.nn.Parameter(torch.ones(2, 1))

    layer = FakeLayer()
    monkeypatch.setattr(
        "tokenspeed.runtime.moe.eplb_host_cache._iter_moe_layers", lambda model: [layer]
    )
    cache = HostWeightCache.from_model(torch.nn.Module(), SimpleNamespace())

    entry = cache.get_expert_dim_params(3, 1)
    assert set(entry) == {"w13_weight", "w2_weight", "w13_weight_scale_inv"}
    assert torch.equal(entry["w13_weight"], layer.w13_weight.detach()[1].cpu())
    assert not cache.has(3, 2)


def test_host_cache_skips_redundant_empty_slots(monkeypatch):
    from tokenspeed.runtime.moe.eplb_host_cache import HostWeightCache

    class FakeLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_index = 7
            self.prefix = "model.layers.7.mlp"
            self.ep_rank = 1
            self.ep_size = 2
            self.num_experts = 6
            self.ep_num_redundant_experts = 2
            self.num_local_experts = 3
            self.w13_weight = torch.nn.Parameter(
                torch.arange(18, dtype=torch.float32).view(3, 2, 3)
            )

    layer = FakeLayer()
    monkeypatch.setattr(
        "tokenspeed.runtime.moe.eplb_host_cache._iter_moe_layers", lambda model: [layer]
    )
    cache = HostWeightCache.from_model(torch.nn.Module(), SimpleNamespace())

    assert cache.has(7, 2)
    assert cache.has(7, 3)
    assert not cache.has(7, 4)
    assert torch.equal(
        cache.get_expert_dim_params(7, 3)["w13_weight"],
        layer.w13_weight.detach()[1].cpu(),
    )


def test_host_cache_imports_remote_shared_experts(monkeypatch):
    import os
    import time

    from tokenspeed.runtime.moe.eplb_host_cache import HostWeightCache

    class FakeLayer(torch.nn.Module):
        def __init__(self, ep_rank, weights):
            super().__init__()
            self.layer_index = 9
            self.prefix = f"model.layers.9.mlp.rank{ep_rank}"
            self.ep_rank = ep_rank
            self.ep_size = 2
            self.num_experts = 4
            self.ep_num_redundant_experts = 0
            self.num_local_experts = 2
            self.w = torch.nn.Parameter(torch.tensor(weights, dtype=torch.float32))

    layer0 = FakeLayer(0, [[10.0, 11.0], [20.0, 21.0]])
    layer1 = FakeLayer(1, [[30.0, 31.0], [40.0, 41.0]])
    args = SimpleNamespace()
    base = f"ts_eplb_test_{os.getpid()}_{time.time_ns()}"
    cache0 = HostWeightCache()
    cache1 = HostWeightCache()
    try:
        cache0._store_shared_owner_layer(layer0, args, base, 0)
        cache1._store_shared_owner_layer(layer1, args, base, 1)
        cache0._import_shared_owner_layer(layer0, args, base, 1)

        assert cache0.has(9, 0)
        assert cache0.has(9, 2)
        assert torch.equal(
            cache0.get_expert_dim_params(9, 2)["w"],
            layer1.w.detach()[0].cpu(),
        )
    finally:
        cache0.release()
        cache1.release()


def test_relocator_transfer_plan_is_rank_independent():
    from tokenspeed.runtime.moe.eplb_relocator import WeightRelocator

    old = SimpleNamespace(physical_to_logical_map_cpu=torch.tensor([[0, 1, 2, 3]]))
    new = SimpleNamespace(physical_to_logical_map_cpu=torch.tensor([[2, 1, 0, 3]]))
    plan_a = WeightRelocator.build_transfer_plan(
        old, new, layer_idx=0, ep_size=2, num_local=2
    )
    plan_b = WeightRelocator.build_transfer_plan(
        old, new, layer_idx=0, ep_size=2, num_local=2
    )
    assert plan_a == plan_b
    assert plan_a == [(1, 0, 0, 0, 2), (0, 0, 1, 0, 0)]


def test_relocator_can_fill_slot_that_was_initially_empty():
    from tokenspeed.runtime.moe.eplb_relocator import WeightRelocator

    old = SimpleNamespace(physical_to_logical_map_cpu=torch.tensor([[0, 1, -1, 2]]))
    new = SimpleNamespace(physical_to_logical_map_cpu=torch.tensor([[0, 1, 2, 2]]))

    plan = WeightRelocator.build_transfer_plan(
        old, new, layer_idx=0, ep_size=2, num_local=2
    )

    assert plan == [(1, 1, 1, 0, 2)]


def test_relocator_estimate_transfer_stats_counts_local_bytes():
    from tokenspeed.runtime.moe.eplb_relocator import WeightRelocator

    old = SimpleNamespace(physical_to_logical_map_cpu=torch.tensor([[0, 1, 2, 3]]))
    new = SimpleNamespace(physical_to_logical_map_cpu=torch.tensor([[2, 1, 0, 3]]))
    params = {
        "w": torch.zeros((2, 2), dtype=torch.float32),
        "v": torch.zeros((2, 3), dtype=torch.bfloat16),
    }

    class FakeHostCache:
        def layer_handle(self, layer_id):
            return SimpleNamespace(num_local_experts=2)

        def expert_dim_params(self, layer_id):
            return params

    relocator = WeightRelocator(
        FakeHostCache(),
        {},
        rank=0,
        ep_size=2,
        strategy="host_first",
    )

    assert relocator.estimate_transfer_stats(old, new, [0]) == {
        "transfer_entries": 2,
        "local_dst_entries": 1,
        "local_src_entries": 1,
        "expected_h2d_bytes": 14,
    }


def test_host_first_snapshots_only_local_destinations():
    from tokenspeed.runtime.moe.eplb_relocator import WeightRelocator

    relocator = WeightRelocator(
        host_cache=object(),
        layer_states={},
        rank=0,
        ep_size=2,
        strategy="host_first",
    )
    params = {"w": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}

    remote_destination_plan = [(0, 0, 1, 0, 0)]
    assert relocator._snapshot_local_sources(params, remote_destination_plan) == {}

    local_destination_plan = [(0, 0, 0, 1, 0)]
    sources = relocator._snapshot_local_sources(params, local_destination_plan)
    assert set(sources) == {("w", 0)}
    assert torch.equal(sources[("w", 0)], params["w"][0])


def test_relocator_host_first_uses_shared_cache_without_p2p(monkeypatch):
    from tokenspeed.runtime.moe import eplb_relocator
    from tokenspeed.runtime.moe.eplb_relocator import WeightRelocator

    old = SimpleNamespace(physical_to_logical_map_cpu=torch.tensor([[0, 1, 2, 3]]))
    new = SimpleNamespace(physical_to_logical_map_cpu=torch.tensor([[2, 1, 0, 3]]))
    params = {"w": torch.zeros((2, 2), dtype=torch.float32)}
    shared_remote = {"w": torch.tensor([42.0, 43.0], dtype=torch.float32)}

    class FakeHostCache:
        def layer_handle(self, layer_id):
            return SimpleNamespace(num_local_experts=2)

        def expert_dim_params(self, layer_id):
            return params

        def has(self, layer_id, logical_id):
            return int(layer_id) == 0 and int(logical_id) == 2

        def get_expert_dim_params(self, layer_id, logical_id):
            assert int(layer_id) == 0
            assert int(logical_id) == 2
            return shared_remote

    def fail_p2p(*args, **kwargs):
        raise AssertionError("host_first relocation must not use P2P")

    monkeypatch.setattr(eplb_relocator, "_dist_ready", lambda: True)
    monkeypatch.setattr(eplb_relocator.dist, "irecv", fail_p2p)
    monkeypatch.setattr(eplb_relocator.dist, "isend", fail_p2p)

    relocator = WeightRelocator(
        FakeHostCache(),
        {},
        rank=0,
        ep_size=2,
        strategy="host_first",
        eplb_pg=object(),
    )
    relocator.submit(old, new, [0])

    assert torch.equal(params["w"][0], shared_remote["w"])


def test_relocator_posts_remote_recvs_before_sends(monkeypatch):
    from tokenspeed.runtime.moe import eplb_relocator
    from tokenspeed.runtime.moe.eplb_relocator import WeightRelocator

    old = SimpleNamespace(physical_to_logical_map_cpu=torch.tensor([[0, 1, 2, 3]]))
    new = SimpleNamespace(physical_to_logical_map_cpu=torch.tensor([[2, 3, 0, 1]]))
    params = {"w": torch.tensor([[10.0, 11.0], [20.0, 21.0]], dtype=torch.float32)}
    events = []
    sent = []

    class FakeHostCache:
        def layer_handle(self, layer_id):
            return SimpleNamespace(num_local_experts=2)

        def expert_dim_params(self, layer_id):
            return params

    class FakeWork:
        def __init__(self, kind):
            self.kind = kind

        def wait(self):
            events.append(("wait", self.kind))

    def fake_irecv(tensor, src, group=None, tag=0):
        events.append(("recv", int(src), int(tag)))
        return FakeWork("recv")

    def fake_isend(tensor, dst, group=None, tag=0):
        events.append(("send", int(dst), int(tag)))
        sent.append(tensor.detach().clone())
        return FakeWork("send")

    monkeypatch.setattr(eplb_relocator, "_dist_ready", lambda: True)
    monkeypatch.setattr(eplb_relocator.dist, "irecv", fake_irecv)
    monkeypatch.setattr(eplb_relocator.dist, "isend", fake_isend)

    relocator = WeightRelocator(
        FakeHostCache(),
        {},
        rank=1,
        ep_size=2,
        strategy="p2p",
        eplb_pg=object(),
    )
    relocator.submit(old, new, [0])

    remote_ops = [event[0] for event in events if event[0] in {"recv", "send"}]
    assert remote_ops == ["recv", "recv", "send", "send"]
    assert len(sent) == 2
    assert torch.equal(sent[0], params["w"][0])
    assert torch.equal(sent[1], params["w"][1])
    wait_kinds = [event[1] for event in events if event[0] == "wait"]
    assert wait_kinds == ["recv", "recv", "send", "send"]


def test_runtime_collect_and_swap_emit_events():
    from tokenspeed.runtime.moe.eplb_runtime import EplbRuntime

    events = []
    metadata = _FakeMetadata()
    new_metadata = _FakeMetadata()
    recorder = _FakeRecorder(torch.tensor([[1, 2]], dtype=torch.int32))
    runtime = EplbRuntime(
        server_args=SimpleNamespace(),
        model_config=SimpleNamespace(),
        initial_metadata=metadata,
        recorder=recorder,
        rank=0,
        ep_size=1,
        event_sink=events.append,
    )

    EplbCollectStatsOperation = type("EplbCollectStatsOperation", (), {})
    collect_op = EplbCollectStatsOperation()
    collect_op.op_id = 10
    runtime.handle(collect_op)
    assert type(events[-1]).__name__ == "StatsCollected"
    assert events[-1].stats_handle == 10
    runtime._plan_handles[20] = new_metadata
    runtime.layer_states[0] = SimpleNamespace(relocate_done_event=_FakeDoneEvent())
    EplbSwapOperation = type("EplbSwapOperation", (), {})
    swap_op = EplbSwapOperation()
    swap_op.op_id = 11
    swap_op.plan_handle = 20
    swap_op.layer_ids = [0]
    runtime.handle(swap_op)
    assert type(events[-1]).__name__ == "SwapDone"
    assert metadata.updated == [(new_metadata, [0])]
    assert recorder.rebound is metadata


def test_runtime_relocate_op_finishes_from_background_thread(caplog):
    import logging
    import threading
    import time

    from tokenspeed.runtime.moe.eplb_runtime import EplbRuntime

    caplog.set_level(logging.DEBUG, logger="tokenspeed.eplb")
    metadata = _FakeMetadata()
    new_metadata = _FakeMetadata()
    recorder = _FakeRecorder(torch.tensor([[1, 2]], dtype=torch.int32))
    runtime = EplbRuntime(
        server_args=SimpleNamespace(),
        model_config=SimpleNamespace(),
        initial_metadata=metadata,
        recorder=recorder,
        rank=0,
        ep_size=1,
    )
    runtime._plan_handles[20] = new_metadata
    done_event = _FakeDoneEvent()
    runtime.layer_states[0] = SimpleNamespace(relocate_done_event=done_event)

    entered = threading.Event()
    release = threading.Event()

    class FakeRelocator:
        def submit(self, old_metadata, planned_metadata, layer_ids):
            assert old_metadata is metadata
            assert planned_metadata is new_metadata
            assert layer_ids == [0]
            entered.set()
            release.wait(timeout=1.0)
            return [0]

    runtime._relocator = FakeRelocator()

    EplbRelocateOperation = type("EplbRelocateOperation", (), {})
    relocate_op = EplbRelocateOperation()
    relocate_op.op_id = 12
    relocate_op.plan_handle = 20
    relocate_op.layer_ids = [0]

    worker = threading.Thread(target=runtime.handle, args=(relocate_op,))
    worker.start()
    try:
        assert entered.wait(timeout=1.0)
        worker.join(timeout=0.05)
        assert not worker.is_alive()
        assert runtime.drain_events() == []

        release.set()
        events = []
        deadline = time.time() + 1.0
        while time.time() < deadline:
            events = runtime.drain_events()
            if events:
                break
            time.sleep(0.01)

        assert len(events) == 1
        assert type(events[0]).__name__ == "RelocateDone"
        assert events[0].op_id == 12
        assert events[0].layer_ids == [0]
        assert done_event.synced
        assert "[EPLB Relocate] phase=start op_id=12" in caplog.text
        assert "[EPLB Relocate] phase=done op_id=12" in caplog.text
    finally:
        release.set()
        worker.join(timeout=1.0)


def test_distribution_recorder_buffer_defaults_to_unbounded_when_unspecified():
    from tokenspeed.runtime.moe.distribution_recorder import _Buffer, _InfiniteBuffer

    buffer = _Buffer.init_new((2,), None, torch.int32, "cpu")

    assert isinstance(buffer, _InfiniteBuffer)
    buffer.append(torch.tensor([1, 2], dtype=torch.int32))
    assert torch.equal(buffer.get_all(), torch.tensor([[1, 2]], dtype=torch.int32))


def test_convert_physical_counts_ignores_empty_slots():
    from tokenspeed.runtime.moe.distribution_recorder import (
        _convert_global_physical_count_to_logical_count,
    )

    physical_count = torch.tensor([[[3, 5, 7, 11]]], dtype=torch.int32)
    physical_to_logical = torch.tensor([[0, -1, 1, -1]], dtype=torch.int64)

    logical_count = _convert_global_physical_count_to_logical_count(
        physical_count,
        num_layers=1,
        num_logical_experts=2,
        physical_to_logical_map=physical_to_logical,
    )

    assert torch.equal(logical_count, torch.tensor([[[3, 7]]], dtype=torch.int32))

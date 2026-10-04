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

"""The rebalance hooks: one loop line, named device ops, EP-group agreements.

Drives ``EplbHooks`` with a fake request handler (the FIFO) and a fake
``DeviceHandle`` through a whole rebalance, with ``torch.distributed`` played
by fakes: the snapshot is SUM-reduced only under all-to-all EP and
checksum-checked under replicated-input EP, the committed map is broadcast
from EP rank 0, and every chunk reaches ``apply_expert_placement``.
"""

from __future__ import annotations

from concurrent.futures import Future
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tokenspeed.runtime.engine import eplb_hooks as hooks_mod
from tokenspeed.runtime.engine.eplb_hooks import (
    EplbHooks,
    make_expert_rebalance_controller,
    rebalance_release_refusal,
)
from tokenspeed.runtime.engine.memory_occupation import MemoryOccupationController
from tokenspeed.runtime.engine.pause import PauseController
from tokenspeed.runtime.moe.eplb_algorithms import EplbAlgorithm
from tokenspeed.runtime.moe.expert_location import ExpertLoadSnapshot
from tokenspeed.runtime.moe.expert_rebalance import (
    EplbApplyChunk,
    EplbCommit,
    EplbSnapshot,
    ExpertRebalanceController,
    ExpertRebalanceSpecs,
    RebalancePhase,
)
from tokenspeed.runtime.moe.placement_maps import compute_placement_maps


class _InlineWorker:
    """In-process stand-in for the spawned placement worker."""

    def submit(self, logical_count, **kwargs) -> Future:
        future: Future = Future()
        future.set_result(compute_placement_maps(logical_count, **kwargs))
        return future

    def shutdown(self) -> None:
        pass


SUM = torch.distributed.ReduceOp.SUM
MAX = torch.distributed.ReduceOp.MAX

_TRIVIAL = torch.tensor([[0, 1, 2, 3, 0, 1]] * 3)


def _specs(ep_rank: int, *, all_to_all_ep: bool) -> ExpertRebalanceSpecs:
    return ExpertRebalanceSpecs(
        num_layers=3,
        num_logical_experts=4,
        num_physical_experts=6,
        ep_size=2,
        ep_rank=ep_rank,
        ep_rank_nodes=(0, 0),
        all_to_all_ep=all_to_all_ep,
        num_groups=None,
        num_nodes=1,
    )


def _controller(ep_rank: int = 0, *, all_to_all_ep: bool = False):
    return ExpertRebalanceController(
        _specs(ep_rank, all_to_all_ep=all_to_all_ep),
        rebalance_num_iterations=4,
        layers_per_chunk=2,
        algorithm=EplbAlgorithm.deepseek,
        commit_delay_forwards=2,
        compute_worker=_InlineWorker() if ep_rank == 0 else None,
    )


class _Handler:
    def __init__(self):
        self.queue: list = []
        self.completer = None

    def set_internal_op_completer(self, completer):
        self.completer = completer

    def enqueue_internal_op(self, op):
        self.queue.append(op)


class _Device:
    """A ``DeviceHandle`` fake: the counters and the applied chunks."""

    def __init__(self, load):
        self.load = load
        self.snapshots = 0
        self.applied: list = []

    def snapshot_expert_load(self):
        self.snapshots += 1
        return ExpertLoadSnapshot(
            physical_count=self.load.clone(), physical_to_logical_map=_TRIVIAL.clone()
        )

    def apply_expert_placement(self, layer_ids, new_rows, moves_by_layer):
        self.applied.append((tuple(layer_ids), new_rows.clone(), dict(moves_by_layer)))


class _Collectives:
    """gloo as the hooks see it: records calls; the peer counted ``peer_load``."""

    def __init__(self, *, peer_load=None, peer_checksum_differs=False):
        self.peer_load = peer_load
        self.peer_checksum_differs = peer_checksum_differs
        self.calls: list = []

    def all_reduce(self, buf, *, op, group):
        self.calls.append(("all_reduce", op, group, buf.shape))
        if op == SUM:
            buf.add_(self.peer_load)
        elif op == MAX and self.peer_checksum_differs:
            # A peer with a different checksum c': MAX of (c, c') and (-c, -c')
            # cannot be negatives of each other unless c == c'.
            buf[0] = max(int(buf[0]), int(buf[0]) + 1)
            buf[1] = max(int(buf[1]), -int(buf[0]) + 7)

    def broadcast(self, tensor, *, src, group):
        self.calls.append(("broadcast", src, group, tuple(tensor.shape)))


def _hooks(controller, device, *, ep_group=(4, 5)):
    handler = _Handler()
    hooks = EplbHooks(
        controller,
        handler,
        device,
        ep_cpu_group="ep-gloo",
        ep_group_ranks=ep_group,
    )
    return hooks, handler


def _load():
    load = torch.ones(3, 6, dtype=torch.int64)
    load[:, 3] = 30
    return load


def _drive(hooks, handler, rounds: int):
    """Loop rounds with a forward each; complete every queued op like the gate."""
    for _ in range(rounds):
        hooks.note_round(forwarded=True)
        while handler.queue:
            hooks.complete(handler.queue.pop(0))


def test_replicated_input_ep_checks_agreement_and_runs_the_whole_cycle():
    controller = _controller(ep_rank=0)
    device = _Device(_load())
    hooks, handler = _hooks(controller, device)
    assert handler.completer is hooks
    collectives = _Collectives()

    with (
        mock.patch.object(torch.distributed, "all_reduce", collectives.all_reduce),
        mock.patch.object(torch.distributed, "broadcast", collectives.broadcast),
    ):
        # Three forwards: nothing due. The fourth enqueues the snapshot.
        _drive(hooks, handler, 3)
        assert handler.queue == [] and device.snapshots == 0
        hooks.note_round(forwarded=True)
        assert handler.queue == [EplbSnapshot()]
        hooks.complete(handler.queue.pop(0))
        assert device.snapshots == 1
        assert controller.phase is RebalancePhase.COMPUTING
        # Replicated-input EP: no SUM, one MAX agreement check on a 2-lane buffer.
        assert collectives.calls == [("all_reduce", MAX, "ep-gloo", torch.Size([2]))]
        # Commit two forwards later: the commit and both chunk ops at once.
        hooks.note_round(forwarded=True)
        assert handler.queue == []
        hooks.note_round(forwarded=True)
        assert handler.queue == [
            EplbCommit(),
            EplbApplyChunk((0, 1)),
            EplbApplyChunk((2,)),
        ]
        hooks.complete(handler.queue.pop(0))
        assert collectives.calls[-1] == ("broadcast", 4, "ep-gloo", (3, 6))
        assert controller.is_applying
        hooks.complete(handler.queue.pop(0))
        hooks.complete(handler.queue.pop(0))
    assert controller.is_idle and controller.rebalances_completed == 1
    assert [ids for ids, _, _ in device.applied] == [(0, 1), (2,)]
    # The hot expert 3 gained both redundant slots in every layer.
    rows = torch.cat([device.applied[0][1], device.applied[1][1]])
    assert (rows == 3).sum(-1).tolist() == [3, 3, 3]
    assert set(device.applied[0][2]) == {0, 1} and set(device.applied[1][2]) == {2}


def test_all_to_all_ep_sums_the_load_over_the_ep_group():
    controller = _controller(ep_rank=1, all_to_all_ep=True)
    device = _Device(_load())
    hooks, handler = _hooks(controller, device)
    peer_load = torch.zeros(3, 6, dtype=torch.int64)
    peer_load[:, 1] = 500  # the peer's tokens make expert 1 the hot one
    collectives = _Collectives(peer_load=peer_load)
    seen = {}
    real_on_snapshot = controller.on_snapshot

    def spy(physical, phy2log):
        seen["physical"] = physical.clone()
        real_on_snapshot(physical, phy2log)

    with (
        mock.patch.object(torch.distributed, "all_reduce", collectives.all_reduce),
        mock.patch.object(controller, "on_snapshot", spy),
    ):
        hooks.complete(controller.request_snapshot())
    assert collectives.calls == [("all_reduce", SUM, "ep-gloo", torch.Size([3, 6]))]
    assert torch.equal(seen["physical"], _load() + peer_load)
    # Rank 1 computes nothing; it only takes the broadcast.
    assert controller._future is None


def test_diverging_counters_under_replicated_input_ep_fail_loudly():
    controller = _controller(ep_rank=0)
    hooks, handler = _hooks(controller, _Device(_load()))
    collectives = _Collectives(peer_checksum_differs=True)
    with mock.patch.object(torch.distributed, "all_reduce", collectives.all_reduce):
        with pytest.raises(RuntimeError, match="routed tokens differently"):
            hooks.complete(controller.request_snapshot())


def test_single_rank_ep_group_runs_no_collective():
    controller = _controller(ep_rank=0)
    device = _Device(_load())
    hooks, handler = _hooks(controller, device, ep_group=(0,))
    with (
        mock.patch.object(
            torch.distributed, "all_reduce", side_effect=AssertionError("collective")
        ),
        mock.patch.object(
            torch.distributed, "broadcast", side_effect=AssertionError("collective")
        ),
    ):
        _drive(hooks, handler, 4 + 2)
    assert controller.is_idle and len(device.applied) == 2


def test_manual_trigger_only_while_idle():
    controller = _controller(ep_rank=0)
    device = _Device(_load())
    hooks, handler = _hooks(controller, device, ep_group=(0,))
    ok, message = hooks.begin_manual_rebalance()
    assert ok and "snapshot taken" in message and device.snapshots == 1
    ok, message = hooks.begin_manual_rebalance()
    assert not ok and "in progress (computing)" in message
    assert device.snapshots == 1
    with pytest.raises(TypeError, match="not an internal"):
        hooks.complete("nope")


def test_disabled_hooks_are_a_no_op_and_install_no_completer():
    handler = _Handler()
    hooks = EplbHooks(
        None, handler, _Device(_load()), ep_cpu_group=None, ep_group_ranks=(0,)
    )
    assert handler.completer is None
    hooks.note_round(forwarded=True)
    assert handler.queue == []
    ok, message = hooks.begin_manual_rebalance()
    assert not ok and "--enable-eplb" in message
    with pytest.raises(RuntimeError, match="not enabled"):
        hooks.complete(EplbSnapshot())


def test_controller_factory_follows_the_flag_and_the_specs():
    args = SimpleNamespace(
        enable_eplb=True,
        eplb_rebalance_num_iterations=10,
        eplb_rebalance_layers_per_chunk=3,
        eplb_algorithm="auto",
    )
    # The factory spawns the real worker for EP rank 0; stand it in here.
    with mock.patch.object(hooks_mod, "PlacementComputeWorker", _InlineWorker):
        controller = make_expert_rebalance_controller(
            args, _specs(0, all_to_all_ep=False)
        )
        assert isinstance(controller.compute_worker, _InlineWorker)
        assert (
            make_expert_rebalance_controller(
                args, _specs(1, all_to_all_ep=False)
            ).compute_worker
            is None
        )
    assert isinstance(controller, ExpertRebalanceController)
    assert controller._chunks == [(0, 1, 2)]
    assert controller._commit_delay == hooks_mod.COMMIT_DELAY_FORWARDS == 200
    assert (
        make_expert_rebalance_controller(SimpleNamespace(enable_eplb=False), None)
        is None
    )
    with pytest.raises(RuntimeError, match="built no expert placement updater"):
        make_expert_rebalance_controller(args, None)
    with pytest.raises(RuntimeError, match="without --enable-eplb"):
        make_expert_rebalance_controller(
            SimpleNamespace(enable_eplb=False), _specs(0, all_to_all_ep=False)
        )
    # More layers per chunk than layers is refused at construction.
    with pytest.raises(ValueError, match="layers_per_chunk"):
        make_expert_rebalance_controller(
            SimpleNamespace(**{**vars(args), "eplb_rebalance_layers_per_chunk": 4}),
            _specs(0, all_to_all_ep=False),
        )


class _Sender:
    def __init__(self):
        self.items = []

    def send_pyobj(self, obj):
        self.items.append(obj)


class _Adapter:
    def __init__(self):
        self.paused = []

    def pause(self, tag):
        self.paused.append(tag)


def _drained_scheduler():
    return SimpleNamespace(
        waiting_size=lambda: 0, decoding_size=lambda: 0, prefilling_size=lambda: 0
    )


def test_memory_release_is_refused_mid_rebalance_and_waits_for_chunks():
    from tokenspeed.runtime.engine.io_struct import ReleaseMemoryOccupationReqInput

    controller = _controller(ep_rank=0)
    sender = _Sender()
    pause = PauseController(sender)
    adapter = _Adapter()
    memory = MemoryOccupationController(
        send_func=sender,
        pause_controller=pause,
        adapter=adapter,
        enabled=True,
        reset_caches_fn=lambda: True,
        kv_repair_fn=lambda: None,
        weights_release_refusal_fn=rebalance_release_refusal(controller),
        weights_busy_fn=lambda: controller.is_applying,
    )
    # Not idle: refused outright, nothing drained.
    controller.request_snapshot()
    memory.handle_release(ReleaseMemoryOccupationReqInput(tags=["weights"]))
    assert not sender.items[-1].success
    assert "rebalance is in progress (computing)" in sender.items[-1].message
    assert not pause.is_drain_pending
    # Idle at the request, then chunks pending when the scheduler drains:
    # the release waits for the last chunk, then frees.
    controller.phase = RebalancePhase.IDLE
    memory.handle_release(ReleaseMemoryOccupationReqInput(tags=["weights"]))
    assert pause.is_drain_pending
    controller.phase = RebalancePhase.APPLYING
    pause.maybe_finish_drain(_drained_scheduler())
    assert pause.is_drain_pending and adapter.paused == []
    controller.phase = RebalancePhase.IDLE
    pause.maybe_finish_drain(_drained_scheduler())
    assert not pause.is_drain_pending and adapter.paused == ["weights"]
    assert sender.items[-1].success

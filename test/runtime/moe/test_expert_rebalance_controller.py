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

"""The rebalance controller: trigger at N forwards, commit at +K, chunks in order."""

from __future__ import annotations

from concurrent.futures import Future

import pytest
import torch

from tokenspeed.runtime.moe.eplb_algorithms import EplbAlgorithm
from tokenspeed.runtime.moe.expert_rebalance import (
    EplbApplyChunk,
    EplbCommit,
    EplbSnapshot,
    ExpertRebalanceController,
    ExpertRebalanceSpecs,
    PlacementComputeWorker,
    RebalancePhase,
)
from tokenspeed.runtime.moe.placement_maps import compute_placement_maps


class _InlineWorker:
    """A compute worker that runs in this process: the controller's contract only."""

    def __init__(self):
        self.submitted = 0
        self.shut_down = False

    def submit(self, logical_count, **kwargs) -> Future:
        self.submitted += 1
        future: Future = Future()
        future.set_result(compute_placement_maps(logical_count, **kwargs))
        return future

    def shutdown(self) -> None:
        self.shut_down = True


def _specs(ep_rank: int = 0) -> ExpertRebalanceSpecs:
    # Three layers, four experts on six slots over two ranks.
    return ExpertRebalanceSpecs(
        num_layers=3,
        num_logical_experts=4,
        num_physical_experts=6,
        ep_size=2,
        ep_rank=ep_rank,
        ep_rank_nodes=(0, 0),
        all_to_all_ep=False,
        num_groups=None,
        num_nodes=1,
    )


def _controller(ep_rank: int = 0, *, interval: int = 5, chunk: int = 2, delay: int = 3):
    return ExpertRebalanceController(
        _specs(ep_rank),
        rebalance_num_iterations=interval,
        layers_per_chunk=chunk,
        algorithm=EplbAlgorithm.deepseek,
        commit_delay_forwards=delay,
        compute_worker=_InlineWorker() if ep_rank == 0 else None,
    )


def _rounds(controller, n: int, *, forwarded: bool = True) -> list:
    ops = []
    for _ in range(n):
        ops.extend(controller.note_round(forwarded=forwarded))
    return ops


_TRIVIAL = torch.tensor([[0, 1, 2, 3, 0, 1]] * 3)


def _snapshot(controller, hot: int = 3):
    load = torch.ones(3, 6, dtype=torch.int64)
    load[:, hot] = 40  # a hot single-replica expert
    controller.on_snapshot(load, _TRIVIAL)
    return load


def test_snapshot_triggers_after_n_forwards_and_commit_after_k_more():
    controller = _controller()
    assert controller.is_idle
    assert _rounds(controller, 4) == []
    # Rounds without a forward do not count.
    assert _rounds(controller, 10, forwarded=False) == []
    assert controller.note_round(forwarded=True) == [EplbSnapshot()]
    assert controller.phase is RebalancePhase.COMPUTING
    assert controller.forwards == 5
    # Until the snapshot completes nothing else is due, however many forwards.
    assert _rounds(controller, 2) == []
    _snapshot(controller)
    assert _rounds(controller, 2) == []
    ops = controller.note_round(forwarded=True)  # forwards == 10 == 7 + 3
    assert ops == [EplbCommit(), EplbApplyChunk((0, 1)), EplbApplyChunk((2,))]
    assert controller.phase is RebalancePhase.COMPUTING  # until the commit lands
    assert _rounds(controller, 3) == []


def test_commit_plans_moves_and_chunks_complete_in_order_then_reset():
    controller = _controller()
    _rounds(controller, 5)
    _snapshot(controller)
    _rounds(controller, 3)
    new_map = controller.placement_for_commit()
    assert new_map.dtype == torch.int32 and new_map.shape == (3, 6)
    # Rank 0 computed a placement: both redundant slots replicate the hot expert.
    assert (new_map == 3).sum(-1).tolist() == [3, 3, 3]
    controller.on_commit(new_map)
    assert controller.is_applying
    with pytest.raises(RuntimeError, match="out of order"):
        controller.chunk_payload((2,))
    rows, moves = controller.chunk_payload((0, 1))
    assert torch.equal(rows, new_map[:2]) and set(moves) == {0, 1}
    assert all(not m.is_empty for m in moves.values())
    controller.on_chunk_applied((0, 1))
    assert controller.is_applying
    rows, moves = controller.chunk_payload((2,))
    assert torch.equal(rows, new_map[2:]) and set(moves) == {2}
    controller.on_chunk_applied((2,))
    assert controller.is_idle and controller.rebalances_completed == 1
    # The cadence counts from the snapshot (forward 5): the next one is due
    # at forward 10, two rounds away from the 8 forwards so far.
    assert controller.note_round(forwarded=True) == []
    assert controller.note_round(forwarded=True) == [EplbSnapshot()]


def test_non_zero_ranks_do_not_compute_and_take_the_broadcast_map():
    controller = _controller(ep_rank=1)
    _rounds(controller, 5)
    _snapshot(controller)
    assert controller._future is None
    buffer = controller.placement_for_commit()
    assert buffer.shape == (3, 6) and buffer.dtype == torch.int32
    agreed = torch.tensor([[3, 1, 2, 3, 0, 1]] * 3, dtype=torch.int32)
    controller.on_commit(agreed)
    rows, moves = controller.chunk_payload((0, 1))
    assert torch.equal(rows, agreed[:2])
    # Rank 1 keeps [3, 0, 1]; rank 0's slot 0 turns from 0 into 3, which only
    # rank 1 holds: rank 1 sends its slot 0 to rank 0.
    assert moves[0].recv == () and moves[0].send == ((0, 0),)


def test_manual_request_only_while_idle_and_results_are_phase_checked():
    controller = _controller()
    assert controller.request_snapshot() == EplbSnapshot()
    assert controller.request_snapshot() is None
    assert controller.phase is RebalancePhase.COMPUTING
    with pytest.raises(RuntimeError, match="commit"):
        controller.placement_for_commit()
    with pytest.raises(RuntimeError, match="chunk"):
        controller.chunk_payload((0, 1))
    with pytest.raises(ValueError, match="shape"):
        controller.on_snapshot(torch.ones(2, 6, dtype=torch.int64), _TRIVIAL[:2])
    _snapshot(controller)
    with pytest.raises(RuntimeError, match="snapshot"):
        _snapshot(controller)
    # The commit follows the snapshot by K = 3 forwards, and the manual
    # trigger restarted the cadence from its own round (forward 0).
    assert _rounds(controller, 2) == []
    assert controller.note_round(forwarded=True)[0] == EplbCommit()
    controller.on_commit(controller.placement_for_commit())
    for ids in ((0, 1), (2,)):
        controller.chunk_payload(ids)
        controller.on_chunk_applied(ids)
    assert controller.is_idle
    assert controller.note_round(forwarded=True) == []  # forward 4
    assert controller.note_round(forwarded=True) == [EplbSnapshot()]  # forward 5
    controller.shutdown()


def test_constructor_validates_its_knobs():
    with pytest.raises(ValueError, match="positive"):
        _controller(interval=0)
    with pytest.raises(ValueError, match="layers_per_chunk"):
        _controller(chunk=4)
    with pytest.raises(ValueError, match="non-negative"):
        _controller(delay=-1)
    # The compute worker belongs to EP rank 0 and to no other rank.
    for ep_rank, worker in ((0, None), (1, _InlineWorker())):
        with pytest.raises(ValueError, match="compute worker"):
            ExpertRebalanceController(
                _specs(ep_rank),
                rebalance_num_iterations=5,
                layers_per_chunk=2,
                algorithm=EplbAlgorithm.deepseek,
                commit_delay_forwards=3,
                compute_worker=worker,
            )


def test_rank_zero_submits_to_its_worker_once_per_snapshot_and_shuts_it_down():
    controller = _controller()
    worker = controller.compute_worker
    _rounds(controller, 5)
    _snapshot(controller)
    assert worker.submitted == 1
    _rounds(controller, 3)
    controller.on_commit(controller.placement_for_commit())
    for ids in ((0, 1), (2,)):
        controller.chunk_payload(ids)
        controller.on_chunk_applied(ids)
    controller.shutdown()
    assert worker.shut_down
    assert _controller(ep_rank=1).compute_worker is None


def test_spawned_worker_computes_the_same_maps_off_process():
    """The serving worker: a spawned CPU process (no CUDA context, no GIL
    contention with the forward thread) that returns the maps as host tensors
    identical to the in-process computation."""
    worker = PlacementComputeWorker()
    try:
        worker.wait_ready(timeout=300)
        load = torch.ones(3, 4, dtype=torch.int64)
        load[:, 3] = 40
        kwargs = dict(
            num_physical_experts=6,
            ep_size=2,
            num_groups=None,
            num_nodes=1,
            algorithm=EplbAlgorithm.deepseek,
        )
        phy2log, log2phy = worker.submit(load, **kwargs).result(timeout=300)
        expected = compute_placement_maps(load, **kwargs)
        assert phy2log.device.type == "cpu" and torch.equal(phy2log, expected[0])
        assert torch.equal(log2phy, expected[1])
        with pytest.raises(ValueError, match="host tensors"):
            worker.submit(load.to(torch.device("meta")), **kwargs)
    finally:
        worker.shutdown()

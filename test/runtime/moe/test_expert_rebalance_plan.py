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

"""The slot move plan, simulated on every rank at once with CPU tensors.

``plan_slot_moves`` is a pure function of the old and new placement rows, so
the whole EP group can be played here: every rank plans, the sends and
receives are paired, the copies are applied in the order the updater uses,
and the slots must end up holding their new logical experts.
"""

from __future__ import annotations

import random
import time

import pytest
import torch

from tokenspeed.runtime.moe.expert_rebalance import (
    SlotMoves,
    chunk_layer_ids,
    expected_balancedness,
    plan_slot_moves,
)


def _random_rows(rng: random.Random, num_logical: int, num_physical: int):
    """Two placements of ``num_logical`` experts on ``num_physical`` slots."""

    def row():
        slots = list(range(num_logical)) + [
            rng.randrange(num_logical) for _ in range(num_physical - num_logical)
        ]
        rng.shuffle(slots)
        return slots

    return row(), row()


def _simulate(old, new, ep_rank_nodes, num_local):
    """Play every rank's plan; return the slot contents after the update."""
    ep_size = len(ep_rank_nodes)
    live = {
        r: [old[r * num_local + s] for s in range(num_local)] for r in range(ep_size)
    }
    staging = {r: [None] * num_local for r in range(ep_size)}
    plans = {
        r: plan_slot_moves(
            old, new, ep_rank=r, ep_rank_nodes=ep_rank_nodes, num_local=num_local
        )
        for r in range(ep_size)
    }
    # 1. P2P: every receive must have exactly one mirroring send, carrying the
    #    expert the destination expects, read from the source's LIVE slot.
    sends = []
    for r, plan in plans.items():
        for src_slot, dst_rank in plan.send:
            sends.append((r, dst_rank, live[r][src_slot]))
    recvs = []
    for r, plan in plans.items():
        for dst_slot, src_rank in plan.recv:
            logical = new[r * num_local + dst_slot]
            recvs.append((src_rank, r, logical))
            staging[r][dst_slot] = logical
            # The source holds the expert before the update.
            assert logical in live[src_rank], (src_rank, r, logical)
    assert sorted(sends) == sorted(recvs)
    assert len(set(recvs)) == len(recvs)  # one receive per (pair, expert)
    # 2. Same-GPU copies live -> staging before any live slot is overwritten.
    for r, plan in plans.items():
        for dst_slot, src_slot in plan.local_copy:
            staging[r][dst_slot] = live[r][src_slot]
    # 3. Live writes from staging, then free-riders live -> live in slot order.
    for r, plan in plans.items():
        for dst_slot, _ in plan.recv:
            live[r][dst_slot] = staging[r][dst_slot]
        for dst_slot, _ in plan.local_copy:
            live[r][dst_slot] = staging[r][dst_slot]
        for dst_slot, src_slot in plan.free_rider:
            assert src_slot < dst_slot
            live[r][dst_slot] = live[r][src_slot]
    return plans, live


def _check(old, new, ep_rank_nodes, num_local):
    plans, live = _simulate(old, new, ep_rank_nodes, num_local)
    ep_size = len(ep_rank_nodes)
    for r in range(ep_size):
        assert live[r] == new[r * num_local : (r + 1) * num_local], r
        plan = plans[r]
        touched = (
            [d for d, _ in plan.recv]
            + [d for d, _ in plan.local_copy]
            + [d for d, _ in plan.free_rider]
        )
        # Each changed slot is covered exactly once; unchanged slots never.
        changed = [
            s
            for s in range(num_local)
            if old[r * num_local + s] != new[r * num_local + s]
        ]
        assert sorted(touched) == changed
        assert plan.num_changed_slots == len(changed)
        for dst_slot, src_rank in plan.recv:
            logical = new[r * num_local + dst_slot]
            # A rank never receives an expert it already held (same-GPU case).
            assert logical not in old[r * num_local : (r + 1) * num_local]
            # Same-node source preferred whenever the node has one.
            src_nodes = {p // num_local for p in range(len(old)) if old[p] == logical}
            same_node = [s for s in src_nodes if ep_rank_nodes[s] == ep_rank_nodes[r]]
            if same_node:
                assert src_rank in same_node
        for dst_slot, src_slot in plan.free_rider:
            # Only an earlier slot that itself receives the expert.
            assert new[r * num_local + src_slot] == new[r * num_local + dst_slot]
            assert old[r * num_local + src_slot] != new[r * num_local + src_slot]
    return plans


def test_unchanged_rows_plan_nothing():
    row = [0, 1, 2, 3, 0, 2]
    for r in range(2):
        plan = plan_slot_moves(row, row, ep_rank=r, ep_rank_nodes=(0, 0), num_local=3)
        assert plan.is_empty and plan.num_changed_slots == 0


def test_every_case_of_the_decision_tree():
    #       rank 0 (node 0) | rank 1 (node 0) | rank 2 (node 1)
    old = [0, 1, 2, 3] + [4, 5, 0, 1] + [2, 3, 4, 5]
    new = [0, 2, 1, 5] + [4, 4, 0, 3] + [2, 2, 0, 5]
    nodes = (0, 0, 1)
    plans = _check(old, new, nodes, 4)
    # Rank 0: slots 1/2 swap (same-GPU both ways), slot 3 needs 5 -- rank 1
    # (same node) and rank 2 hold it; the same-node source wins.
    assert plans[0].local_copy == ((1, 2), (2, 1))
    assert plans[0].recv == ((3, 1),)
    assert plans[0].free_rider == ()
    # Rank 1: slot 1 gets 4 from its own slot 0; slot 3 needs 3 from rank 0.
    assert plans[1].local_copy == ((1, 0),)
    assert plans[1].recv == ((3, 0),)
    # Rank 2: slot 1 free-rides on slot 0 (2 is already there, unchanged);
    # slot 2 needs 0 from another node: rank 0 and rank 1 both hold it and the
    # destinations spread over the sources in order (rank 0 first).
    assert plans[2].local_copy == ((1, 0),)
    assert plans[2].recv == ((2, 0),)
    # Sends mirror: rank 0 sends 0 to rank 2 and 3 to rank 1; rank 1 sends 5
    # to rank 0; rank 2 sends nothing.
    assert plans[0].send == ((0, 2), (3, 1))
    assert plans[1].send == ((1, 0),)
    assert plans[2].send == ()


def test_free_rider_follows_the_first_receiving_slot():
    old = [0, 1, 2, 3]
    new = [3, 3, 3, 0]  # rank 0 wants three copies of 3, which only rank 1 held
    plans = _check(old, new, (0, 0), 2)
    assert plans[0].recv == ((0, 1),)
    assert plans[0].free_rider == ((1, 0),)
    # Rank 1 holds [2, 3] -> [3, 0]: slot 0 takes 3 from its slot 1
    # (same-GPU), slot 1 needs 0 from rank 0.
    assert plans[1].local_copy == ((0, 1),)
    assert plans[1].recv == ((1, 0),)
    assert plans[1].send == ((1, 0),)
    assert plans[0].send == ((0, 1),)


def test_destinations_spread_evenly_over_the_sources():
    # Expert 0 is held by ranks 0 and 1 (node 0); ranks 2..5 (node 1) all
    # want it: two cross-node destinations per source, in rank order.
    old = [0, 1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    new = [0, 1, 0, 2, 0, 4, 0, 6, 0, 8, 0, 10]
    nodes = (0, 0, 1, 1, 1, 1)
    plans = _check(old, new, nodes, 2)
    assert [plans[r].recv for r in range(2, 6)] == [
        ((0, 0),),
        ((0, 0),),
        ((0, 1),),
        ((0, 1),),
    ]
    assert plans[0].send == ((0, 2), (0, 3))
    assert plans[1].send == ((0, 4), (0, 5))


def test_same_node_destinations_use_only_same_node_sources():
    # Expert 7 lives on rank 0 (node 0) and rank 3 (node 1). Rank 1 (node 0)
    # and rank 2 (node 1) want it: each takes its own node's source.
    old = [7, 0, 1, 2, 3, 4, 7, 5]
    new = [7, 0, 7, 2, 7, 4, 7, 5]
    plans = _check(old, new, (0, 0, 1, 1), 2)
    assert plans[1].recv == ((0, 0),)
    assert plans[2].recv == ((0, 3),)
    assert plans[0].send == ((0, 1),)
    assert plans[3].send == ((0, 2),)


@pytest.mark.parametrize("seed", range(40))
def test_randomized_group_ends_with_every_slot_holding_its_expert(seed):
    rng = random.Random(seed)
    ep_size = rng.choice([2, 3, 4, 6, 8])
    num_local = rng.choice([1, 2, 3, 5])
    num_physical = ep_size * num_local
    # Between "every slot is a distinct expert" (R = 0, pure permutation)
    # and "half the slots are replicas".
    num_logical = rng.randint(max(1, num_physical // 2), num_physical)
    nodes_count = rng.choice([1, 2, 3])
    nodes = tuple(sorted(rng.randrange(nodes_count) for _ in range(ep_size)))
    old, new = _random_rows(rng, num_logical, num_physical)
    _check(old, new, nodes, num_local)
    # The plan is a function of the rows alone: planning twice is identical.
    for r in range(ep_size):
        assert plan_slot_moves(
            old, new, ep_rank=r, ep_rank_nodes=nodes, num_local=num_local
        ) == plan_slot_moves(
            torch.tensor(old),
            torch.tensor(new),
            ep_rank=r,
            ep_rank_nodes=nodes,
            num_local=num_local,
        )


def test_production_size_layer_plans_in_one_pass():
    """EP128 x 7 slots, 768 experts: the planner indexes the rows once per
    layer (per-expert source and destination rank lists), so a whole layer
    plans in well under a second rather than rescanning the row per slot."""
    rng = random.Random(7)
    ep_size, num_local, num_logical = 128, 7, 768
    nodes = tuple(r // 8 for r in range(ep_size))
    old, new = _random_rows(rng, num_logical, ep_size * num_local)
    started = time.perf_counter()
    plans = [
        plan_slot_moves(old, new, ep_rank=r, ep_rank_nodes=nodes, num_local=num_local)
        for r in range(ep_size)
    ]
    elapsed = time.perf_counter() - started
    assert elapsed < 5.0, f"planning 128 ranks took {elapsed:.1f}s"
    sends = sum(len(p.send) for p in plans)
    recvs = sum(len(p.recv) for p in plans)
    assert sends == recvs > 0


def test_plan_rejects_inconsistent_geometry():
    with pytest.raises(ValueError, match="same length"):
        plan_slot_moves([0, 1], [0], ep_rank=0, ep_rank_nodes=(0,), num_local=2)
    with pytest.raises(ValueError, match="not"):
        plan_slot_moves(
            [0, 1, 2], [0, 1, 2], ep_rank=0, ep_rank_nodes=(0, 0), num_local=2
        )
    with pytest.raises(ValueError, match="outside"):
        plan_slot_moves([0, 1], [1, 0], ep_rank=2, ep_rank_nodes=(0, 0), num_local=1)
    assert isinstance(
        plan_slot_moves([0, 1], [1, 0], ep_rank=0, ep_rank_nodes=(0, 0), num_local=1),
        SlotMoves,
    )


def test_chunks_and_expected_balancedness():
    assert chunk_layer_ids(5, 2) == [(0, 1), (2, 3), (4,)]
    assert chunk_layer_ids(3, 3) == [(0, 1, 2)]
    with pytest.raises(ValueError, match="layers_per_chunk"):
        chunk_layer_ids(3, 4)
    with pytest.raises(ValueError, match="layers_per_chunk"):
        chunk_layer_ids(3, 0)
    # A hot expert replicated across both ranks evens out the rank loads.
    load = torch.tensor([[12, 2, 2, 2]])
    assert expected_balancedness(load, torch.tensor([[0, 1, 2, 3]]), 2).tolist() == [
        pytest.approx(9 / 14)
    ]
    # Two replicas of the hot expert: rank loads 6+2+2 and 2+6+2 (replicas
    # split the expert's load evenly), 9 / 10.
    assert expected_balancedness(
        load, torch.tensor([[0, 1, 2, 3, 0, 3]]), 2
    ).tolist() == [pytest.approx(0.9)]

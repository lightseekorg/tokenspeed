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

"""Expert placement: slot queries, dispatch tables, static maps and load records."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.moe import ExpertDispatch, dispatch_topk_ids_reference

from tokenspeed.runtime.moe import eplb_algorithms
from tokenspeed.runtime.moe.expert_location import (
    ExpertLocationMetadata,
    compute_logical_to_rank_dispatch_physical_map,
)


def _placement(ep_rank: int, dispatch_algorithm: str | None = None):
    # Two layers, four logical experts on six slots over two ranks. Layer 1
    # gives expert 1 three replicas, two of them on rank 1.
    physical_to_logical = torch.tensor([[0, 1, 2, 3, 0, 2], [3, 2, 1, 0, 1, 1]])
    return ExpertLocationMetadata.from_physical_to_logical_map(
        physical_to_logical,
        4,
        ep_size=2,
        ep_rank=ep_rank,
        num_nodes=1,
        dispatch_algorithm=dispatch_algorithm,
    )


def test_placement_queries_follow_the_physical_map():
    placement = _placement(ep_rank=1)
    assert placement.num_local_physical_experts == 3
    assert placement.has_redundancy
    assert placement.local_logical_experts(0, 0) == [0, 1, 2]
    assert placement.local_logical_experts(0, 1) == [3, 0, 2]
    assert placement.local_logical_experts(1, 1) == [0, 1]
    assert placement.local_slot_logical_experts(1, 1) == [0, 1, 1]
    assert placement.local_physical_slots(0, 0, ep_rank=0) == [0]
    assert placement.local_physical_slots(0, 0, ep_rank=1) == [1]
    assert placement.local_physical_slots(1, 1, ep_rank=1) == [1, 2]
    assert placement.local_physical_slots(1, 3, ep_rank=1) == []
    assert placement.dispatch_replicas.tolist() == [
        [[0, 4, -1], [1, -1, -1], [2, 5, -1], [3, -1, -1]],
        [[3, -1, -1], [2, 4, 5], [1, -1, -1], [0, -1, -1]],
    ]
    assert placement.dispatch_num_replicas.tolist() == [[2, 1, 2, 1], [1, 3, 1, 1]]
    # No static map unless a static dispatch algorithm asks for one.
    assert placement.logical_to_rank_dispatch_physical_map is None
    with pytest.raises(ValueError, match="do not divide"):
        ExpertLocationMetadata.from_physical_to_logical_map(
            torch.tensor([[0, 1, 2, 3, 0]]),
            4,
            ep_size=2,
            ep_rank=0,
            num_nodes=1,
            dispatch_algorithm=None,
        )
    with pytest.raises(ValueError, match="no physical slot"):
        ExpertLocationMetadata.from_physical_to_logical_map(
            torch.tensor([[0, 1, 2, 0]]),
            4,
            ep_size=2,
            ep_rank=0,
            num_nodes=1,
            dispatch_algorithm=None,
        )


def test_static_map_prefers_the_local_replica():
    for ep_rank in (0, 1):
        placement = _placement(ep_rank, dispatch_algorithm="static_with_zero_expert")
        static = placement.logical_to_rank_dispatch_physical_map
        assert static.shape == (2, 4)
        local = range(ep_rank * 3, (ep_rank + 1) * 3)
        for layer in range(2):
            for logical in range(4):
                replicas = placement.logical_to_all_physical(layer, logical)
                chosen = int(static[layer, logical])
                assert chosen in replicas
                if any(p in local for p in replicas):
                    assert chosen in local


def test_static_map_prefers_a_same_node_replica_before_a_remote_one():
    # Four ranks on two nodes, two slots each. Expert 0 lives on ranks 0 and
    # 2 (one per node), expert 1 twice on rank 3, experts 2 and 4 once on
    # rank 1, expert 3 on ranks 0 and 2.
    #                                  slot: 0  1  2  3  4  5  6  7
    logical_to_all = torch.tensor([[[0, 4], [6, 7], [2, -1], [1, 5], [3, -1]]])
    maps = [
        compute_logical_to_rank_dispatch_physical_map(
            logical_to_all, num_gpus=4, num_nodes=2, num_physical_experts=8, ep_rank=r
        )
        for r in range(4)
    ]
    # Rank 1 (node 0) has no copy of expert 0; node 0's other rank does.
    assert int(maps[1][0, 0]) == 0
    # Rank 3 (node 1) takes node 1's copy on rank 2.
    assert int(maps[3][0, 0]) == 4
    # Owners dispatch to themselves.
    assert int(maps[0][0, 0]) == 0 and int(maps[2][0, 0]) == 4
    # Single-replica experts resolve to that replica everywhere.
    assert [int(m[0, 2]) for m in maps] == [2, 2, 2, 2]
    # Expert 1 has no copy on node 0: ranks 0 and 1 draw one of its replicas.
    assert int(maps[2][0, 1]) == 6
    assert all(int(maps[r][0, 1]) in (6, 7) for r in (0, 1))
    with pytest.raises(ValueError, match="do not spread evenly"):
        compute_logical_to_rank_dispatch_physical_map(
            logical_to_all, num_gpus=4, num_nodes=3, num_physical_experts=8, ep_rank=0
        )


def test_dispatch_reference_alternates_replicas_and_counts_load():
    placement = _placement(ep_rank=0)
    placement.enable_load_recording()
    dispatch = ExpertDispatch(
        placement.dispatch_replicas[1],
        placement.dispatch_num_replicas[1],
        placement.physical_load[1],
    )
    topk_ids = torch.tensor([[1, 0], [1, 2], [1, 3]], dtype=torch.int32)
    physical = dispatch_topk_ids_reference(topk_ids, dispatch)
    # Expert 1's replicas are physical 2, 4, 5: row r, rank k picks (r + k) % 3.
    assert physical.tolist() == [[2, 3], [4, 1], [5, 0]]
    assert physical.dtype == torch.int32
    assert placement.physical_load.tolist() == [
        [0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1],
    ]
    record = placement.load_record(placement.physical_load)
    assert record["logical_count"].tolist() == [[0, 0, 0, 0], [1, 3, 1, 1]]
    assert record["rank_count"].tolist() == [[0, 0], [3, 3]]
    assert record["balancedness"].tolist() == [0.0, 1.0]
    assert torch.equal(
        record["physical_to_logical_map"], placement.physical_to_logical_map_cpu
    )
    placement.reset_load()
    assert not placement.physical_load.any()


def test_eplb_placement_balances_a_skewed_load():
    torch.manual_seed(0)
    layers, experts, ep = 3, 32, 4
    load = torch.randint(1, 20, (layers, experts)).double()
    load[:, 0] = 300  # one hot expert per layer
    trivial = torch.arange(experts).repeat(layers, 1)

    def busiest_over_mean(p2l):
        replicas = torch.zeros_like(load).scatter_add_(
            1, p2l, torch.ones_like(p2l).double()
        )
        per_slot = load.gather(1, p2l) / replicas.gather(1, p2l)
        per_rank = per_slot.view(layers, ep, -1).sum(-1)
        return (per_rank.max(-1).values / per_rank.mean(-1)).max().item()

    p2l, log2phy, logcnt = eplb_algorithms.deepseek.rebalance_experts(
        load, experts + 8, 1, 1, ep, False
    )
    assert busiest_over_mean(trivial) > 1.5
    assert busiest_over_mean(p2l) < 1.15
    assert logcnt[:, 0].min() >= 2  # the hot expert got replicas
    placement = ExpertLocationMetadata.from_maps(
        p2l,
        log2phy,
        ep_size=ep,
        ep_rank=0,
        num_nodes=1,
        dispatch_algorithm="static",
    )
    assert placement.num_physical_experts == experts + 8
    # Every logical expert is placed somewhere, and every slot is accounted for.
    assert (placement.dispatch_num_replicas >= 1).all()
    assert placement.dispatch_num_replicas.sum(-1).tolist() == [experts + 8] * layers
    assert placement.logical_to_rank_dispatch_physical_map.shape == (layers, experts)

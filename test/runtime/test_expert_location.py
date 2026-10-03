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

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from tokenspeed_kernel.ops.moe import ExpertDispatch, dispatch_topk_ids_reference

from tokenspeed.runtime.moe import eplb_algorithms, expert_location
from tokenspeed.runtime.moe.dispatch_algorithm import (
    EP_DISPATCH_ALGORITHMS,
    STATIC_EP_DISPATCH_ALGORITHMS,
    has_zero_expert,
)
from tokenspeed.runtime.moe.expert_location import (
    ExpertLocationMetadata,
    build_expert_placement,
    compute_logical_to_rank_dispatch_physical_map,
    expert_placement_requested,
    merge_expert_load_records,
)
from tokenspeed.runtime.utils.server_args import ServerArgs


def _placement(ep_rank: int):
    # Two layers, four logical experts on six slots over two ranks. Layer 1
    # gives expert 1 three replicas, two of them on rank 1.
    physical_to_logical = torch.tensor([[0, 1, 2, 3, 0, 2], [3, 2, 1, 0, 1, 1]])
    return ExpertLocationMetadata.from_physical_to_logical_map(
        physical_to_logical, 4, ep_size=2, ep_rank=ep_rank, ep_rank_nodes=(0, 0)
    )


def test_placement_queries_follow_the_physical_map():
    placement = _placement(ep_rank=1)
    assert placement.num_local_physical_experts == 3
    assert placement.local_slot_logical_experts(0, 0) == [0, 1, 2]
    assert placement.local_slot_logical_experts(0, 1) == [3, 0, 2]
    assert placement.local_slot_logical_experts(1, 1) == [0, 1, 1]
    assert placement.logical_to_all_physical(1, 1) == [2, 4, 5]
    # The replica table is the one routing table: int32, trimmed to the
    # widest replica count, never padded to the physical expert count.
    assert placement.logical_to_all_physical_map.dtype == torch.int32
    assert placement.logical_to_all_physical_map.tolist() == [
        [[0, 4, -1], [1, -1, -1], [2, 5, -1], [3, -1, -1]],
        [[3, -1, -1], [2, 4, 5], [1, -1, -1], [0, -1, -1]],
    ]
    assert placement.logical_to_all_physical_map_num_valid.dtype == torch.int32
    assert placement.logical_to_all_physical_map_num_valid.tolist() == [
        [2, 1, 2, 1],
        [1, 3, 1, 1],
    ]
    with pytest.raises(ValueError, match="do not divide"):
        ExpertLocationMetadata.from_physical_to_logical_map(
            torch.tensor([[0, 1, 2, 3, 0]]),
            4,
            ep_size=2,
            ep_rank=0,
            ep_rank_nodes=(0, 0),
        )
    with pytest.raises(ValueError, match="no physical slot"):
        ExpertLocationMetadata.from_physical_to_logical_map(
            torch.tensor([[0, 1, 2, 0]]), 4, ep_size=2, ep_rank=0, ep_rank_nodes=(0, 0)
        )
    with pytest.raises(ValueError, match="ep_rank_nodes"):
        ExpertLocationMetadata.from_physical_to_logical_map(
            torch.tensor([[0, 1, 2, 3]]), 4, ep_size=2, ep_rank=0, ep_rank_nodes=(0,)
        )


def test_static_map_is_built_on_demand_and_prefers_the_local_replica():
    for ep_rank in (0, 1):
        placement = _placement(ep_rank)
        assert placement._rank_dispatch_map is None  # nothing paid for yet
        static = placement.rank_dispatch_map()
        assert static is placement.rank_dispatch_map()  # computed once
        assert static.shape == (2, 4) and static.dtype == torch.int32
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
            logical_to_all,
            num_physical_experts=8,
            ep_rank_nodes=(0, 0, 1, 1),
            ep_rank=r,
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
    assert all((m >= 0).all() for m in maps)


def test_static_map_follows_the_ranks_actual_nodes():
    # The EP ranks' nodes come from the mapping, not from ep_size / nnodes:
    # here EP ranks 0 and 1 share node 0 while rank 2 is alone on node 1, a
    # layout a divisibility rule would reject.
    logical_to_all = torch.tensor([[[0, 5], [1, 2], [3, 4]]])
    maps = [
        compute_logical_to_rank_dispatch_physical_map(
            logical_to_all, num_physical_experts=6, ep_rank_nodes=(0, 0, 1), ep_rank=r
        )
        for r in range(3)
    ]
    # Rank 1 has no copy of expert 0; its node mate rank 0 does (slot 0).
    assert int(maps[1][0, 0]) == 0
    # Rank 0 has no copy of expert 2; node mate rank 1 holds slots 2-3.
    assert int(maps[0][0, 2]) == 3
    # MoE-TP-only: a single EP rank on a multi-node job is fine.
    single = compute_logical_to_rank_dispatch_physical_map(
        torch.tensor([[[0], [1]]]),
        num_physical_experts=2,
        ep_rank_nodes=(1,),
        ep_rank=0,
    )
    assert single.tolist() == [[0, 1]]
    with pytest.raises(ValueError, match="do not divide"):
        compute_logical_to_rank_dispatch_physical_map(
            logical_to_all,
            num_physical_experts=6,
            ep_rank_nodes=(0, 0, 1, 1),
            ep_rank=0,
        )


def test_static_map_is_identical_on_every_rank_and_at_scale():
    # 896 slots over 128 ranks on 16 nodes, 768 experts, a few replicated.
    physical_to_logical = (torch.arange(896) % 768).view(1, 896)
    nodes = tuple(r // 8 for r in range(128))
    full = [
        ExpertLocationMetadata.from_physical_to_logical_map(
            physical_to_logical, 768, ep_size=128, ep_rank=r, ep_rank_nodes=nodes
        ).rank_dispatch_map()
        for r in (0, 77)
    ]
    assert full[0].shape == (1, 768)
    # Ranks agree on where single-replica experts live, and each prefers its
    # own slots for the replicated ones.
    assert torch.equal(full[0][0, 128:], full[1][0, 128:])
    assert full[0][0, :7].tolist() == list(range(7))  # rank 0 owns slots 0..6
    assert all(768 <= p < 896 or p < 128 for p in full[1][0, :128].tolist())


def test_dispatch_reference_alternates_replicas():
    placement = _placement(ep_rank=0)
    dispatch = ExpertDispatch(
        placement.logical_to_all_physical_map[1],
        placement.logical_to_all_physical_map_num_valid[1],
    )
    topk_ids = torch.tensor([[1, 0], [1, 2], [1, 3]], dtype=torch.int32)
    physical = dispatch_topk_ids_reference(topk_ids, dispatch)
    # Expert 1's replicas are physical 2, 4, 5: row r, rank k picks (r + k) % 3.
    assert physical.tolist() == [[2, 3], [4, 1], [5, 0]]
    assert physical.dtype == torch.int32
    with pytest.raises(ValueError, match="int32"):
        ExpertDispatch(dispatch.replicas.long(), dispatch.num_replicas)


def test_load_record_is_per_rank_and_merges_across_ranks(tmp_path):
    records = []
    for ep_rank in (0, 1):
        placement = _placement(ep_rank)
        placement.enable_load_recording()
        assert placement.physical_load.dtype == torch.int64
        # Each rank counted its own tokens' routes (all-to-all EP).
        placement.physical_load.copy_(
            torch.tensor([[1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 1, 1]]) * (ep_rank + 1)
        )
        record = placement.load_record(placement.physical_load)
        assert record["ep_rank"] == ep_rank and record["ep_size"] == 2
        assert record["physical_count"].dtype == torch.int64
        # Logical 0 owns physical 0 and 4 in layer 0; expert 1 owns 2, 4, 5 in layer 1.
        assert record["logical_count"].tolist() == [
            [2 * (ep_rank + 1), 0, 0, 0],
            [0, 3 * (ep_rank + 1), 0, 0],
        ]
        path = tmp_path / f"load-TP{ep_rank}.expert-load.pt"
        torch.save(record, path)
        records.append(path)
        placement.reset_load()
        assert not placement.physical_load.any()

    merged = merge_expert_load_records(records)
    assert merged["ep_ranks"] == [0, 1]
    assert merged["physical_count"].tolist() == [[3, 0, 0, 0, 3, 0], [0, 0, 3, 0, 3, 3]]
    assert merged["logical_count"].tolist() == [[6, 0, 0, 0], [0, 9, 0, 0]]
    assert merged["rank_count"].tolist() == [[3, 3], [3, 6]]
    assert merged["balancedness"].tolist() == pytest.approx([1.0, 0.75])
    # Records of another placement cannot be merged in.
    other = _placement(0)
    other.physical_to_logical_map_cpu[0, 0] = 1
    other.enable_load_recording()
    torch.save(other.load_record(other.physical_load), tmp_path / "x.expert-load.pt")
    with pytest.raises(ValueError, match="different expert placement"):
        merge_expert_load_records(records + [tmp_path / "x.expert-load.pt"])


def test_init_expert_location_merges_a_directory_of_records(tmp_path):
    placement = _placement(0)
    placement.enable_load_recording()
    for ep_rank in (0, 1):
        placement.physical_load.fill_(ep_rank + 1)
        torch.save(
            placement.load_record(placement.physical_load) | {"ep_rank": ep_rank},
            tmp_path / f"p-TP{ep_rank}.expert-load.pt",
        )
    (tmp_path / "p-TP0.trace.json.gz").write_bytes(b"")  # other profile outputs
    seen = {}

    def fake_init_by_eplb(server_args, model_config, logical_count):
        seen["logical_count"] = logical_count
        return "placement"

    with mock.patch.object(
        ExpertLocationMetadata, "init_by_eplb", staticmethod(fake_init_by_eplb)
    ):
        result = expert_location.compute_initial_expert_location_metadata(
            SimpleNamespace(init_expert_location=str(tmp_path)), None
        )
    assert result == "placement"
    # 3 routes per slot summed over the ranks: expert 0 has 2 slots in layer 0.
    assert seen["logical_count"].tolist() == [[6, 3, 6, 3], [3, 9, 3, 3]]
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="holds no"):
        expert_location.compute_initial_expert_location_metadata(
            SimpleNamespace(init_expert_location=str(tmp_path / "empty")), None
        )


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
        p2l, log2phy, ep_size=ep, ep_rank=0, ep_rank_nodes=(0, 0, 0, 0)
    )
    assert placement.num_physical_experts == experts + 8
    # Every logical expert is placed somewhere, and every slot is accounted for.
    assert (placement.logical_to_all_physical_map_num_valid >= 1).all()
    assert (
        placement.logical_to_all_physical_map_num_valid.sum(-1).tolist()
        == [experts + 8] * layers
    )
    assert placement.rank_dispatch_map().shape == (layers, experts)


def test_build_expert_placement_refuses_models_that_do_not_opt_in():
    from tokenspeed.runtime.models.base.causal_lm import BaseCausalLM

    class Plain(BaseCausalLM):
        pass

    class Placed(BaseCausalLM):
        supports_expert_placement = True

        @classmethod
        def get_model_config_for_expert_location(cls, config):
            return expert_location.ModelConfigForExpertLocation(
                num_layers=1, num_logical_experts=4
            )

    args = SimpleNamespace(
        ep_num_redundant_experts=2,
        init_expert_location="trivial",
        expert_distribution_recorder_mode="stat",
        ep_dispatch_algorithm="static",
    )
    model_config = SimpleNamespace(hf_config=None)
    assert not BaseCausalLM.supports_expert_placement
    with mock.patch.object(
        expert_location, "get_model_architecture", return_value=(Plain, "Plain")
    ):
        with pytest.raises(ValueError, match="does not route through"):
            build_expert_placement(args, model_config)
    with (
        mock.patch.object(
            expert_location, "get_model_architecture", return_value=(Placed, "Placed")
        ),
        mock.patch.object(
            expert_location,
            "compute_initial_expert_location_metadata",
            return_value=_placement(0),
        ),
    ):
        placement = build_expert_placement(args, model_config)
    assert placement.physical_load is not None  # stat recording allocated
    # Without a request there is no placement, whatever the model.
    args.ep_num_redundant_experts = 0
    args.expert_distribution_recorder_mode = None
    assert build_expert_placement(args, model_config) is None

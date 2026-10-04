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
from tokenspeed_kernel.ops.moe import ExpertDispatch, dispatch_topk_ids

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.moe import eplb_algorithms, expert_location
from tokenspeed.runtime.moe.expert_location import (
    ExpertLocationMetadata,
    build_expert_placement,
    compute_logical_to_rank_dispatch_physical_map,
    merge_expert_load_records,
)


def _mapping(ep_rank: int = 0, ep_size: int = 2) -> Mapping:
    return Mapping(rank=ep_rank, world_size=ep_size, moe_ep_size=ep_size)


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


def test_dispatch_alternates_replicas_per_row_and_route():
    placement = _placement(ep_rank=0)
    dispatch = ExpertDispatch(
        placement.logical_to_all_physical_map[1],
        placement.logical_to_all_physical_map_num_valid[1],
    )
    topk_ids = torch.tensor([[1, 0], [1, 2], [1, 3]], dtype=torch.int32)
    physical = dispatch_topk_ids(topk_ids, dispatch)
    # Expert 1's replicas are physical 2, 4, 5: row r, rank k picks (r + k) % 3.
    assert physical.tolist() == [[2, 3], [4, 1], [5, 0]]
    assert physical.dtype == torch.int32
    with pytest.raises(ValueError, match="int32"):
        ExpertDispatch(dispatch.replicas.long(), dispatch.num_replicas)


def test_load_record_is_per_rank_and_merges_across_ranks(tmp_path):
    records = []
    for ep_rank in (0, 1):
        placement = _placement(ep_rank)
        placement.enable_load_recording(_mapping())
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
    other.enable_load_recording(_mapping(1))
    torch.save(other.load_record(other.physical_load), tmp_path / "x.expert-load.pt")
    with pytest.raises(ValueError, match="different expert placement"):
        merge_expert_load_records(records + [tmp_path / "x.expert-load.pt"])


def test_init_expert_location_merges_a_directory_of_records(tmp_path):
    placement = _placement(0)
    placement.enable_load_recording(_mapping())
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


def test_init_expert_location_form_is_decided_by_shape_not_by_glob_characters(
    tmp_path,
):
    """Inline JSON, a directory, a file and a glob each take their own path;
    JSON containing ``[``/``?`` is never mistaken for a pattern."""
    placement = _placement(0)
    placement.enable_load_recording(_mapping())
    placement.physical_load.fill_(2)
    record = placement.load_record(placement.physical_load)
    torch.save(record, tmp_path / "p-TP0.expert-load.pt")
    torch.save(record | {"ep_rank": 1}, tmp_path / "p-TP1.expert-load.pt")
    inline = '{"physical_to_logical_map": [[0, 1, 2, 3, 0, 2], [3, 2, 1, 0, 1, 1]]}'
    assert expert_location.init_expert_location_form("trivial") == "trivial"
    assert expert_location.init_expert_location_form(inline) == "json"
    assert expert_location.init_expert_location_form(str(tmp_path)) == "directory"
    assert (
        expert_location.init_expert_location_form(
            str(tmp_path / "p-TP0.expert-load.pt")
        )
        == "file"
    )
    assert (
        expert_location.init_expert_location_form(str(tmp_path / "p-TP?.expert-*"))
        == "glob"
    )

    seen: dict[str, object] = {}

    def fake_init_by_eplb(server_args, model_config, logical_count):
        seen["logical_count"] = logical_count
        return "eplb"

    def fake_init_by_mapping(server_args, model_config, physical_to_logical_map):
        seen["map"] = physical_to_logical_map
        return "mapping"

    def compute(data):
        return expert_location.compute_initial_expert_location_metadata(
            SimpleNamespace(init_expert_location=data), None
        )

    with (
        mock.patch.object(
            ExpertLocationMetadata, "init_by_eplb", staticmethod(fake_init_by_eplb)
        ),
        mock.patch.object(
            ExpertLocationMetadata,
            "init_by_mapping",
            staticmethod(fake_init_by_mapping),
        ),
    ):
        assert compute(inline) == "mapping"
        assert seen["map"] == [[0, 1, 2, 3, 0, 2], [3, 2, 1, 0, 1, 1]]
        # Two records merged: 2 routes per slot per rank, 4 after the merge.
        assert compute(str(tmp_path / "p-TP?.expert-*")) == "eplb"
        assert seen["logical_count"].tolist() == [[8, 4, 8, 4], [4, 12, 4, 4]]
        # One record file alone: its own logical_count, nothing merged.
        assert compute(str(tmp_path / "p-TP1.expert-load.pt")) == "eplb"
        assert seen["logical_count"].tolist() == [[4, 2, 4, 2], [2, 6, 2, 2]]
        # Neither JSON, directory nor file: a glob, which must match.
        with pytest.raises(ValueError, match="matches no expert load record"):
            compute(str(tmp_path / "nothing-here-*"))
        with pytest.raises(ValueError, match="matches no expert load record"):
            compute(str(tmp_path / "not-a-record.txt"))
        (tmp_path / "not-a-record.txt").write_text("x")
        with pytest.raises(ValueError, match="must be a .pt or .json"):
            compute(str(tmp_path / "not-a-record.txt"))


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
        enable_eplb=False,
        mapping=_mapping(),
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
    assert placement.load_rows is not None  # with its live-row mask
    # Without a request there is no placement, whatever the model.
    args.ep_num_redundant_experts = 0
    args.expert_distribution_recorder_mode = None
    assert build_expert_placement(args, model_config) is None


# ----------------------------------------------------------------------
# Mutable placement: fixed-width replica table, in-place row updates that
# reach the router's views, and the counter snapshot.
# ----------------------------------------------------------------------


def test_replica_table_has_the_fixed_width_r_plus_one():
    # Four logical experts on six slots: R = 2, so at most three replicas.
    assert expert_location.replica_table_width(6, 4) == 3
    assert expert_location.replica_table_width(4, 4) == 1
    with pytest.raises(ValueError, match="cannot hold"):
        expert_location.replica_table_width(3, 4)
    # A placement whose widest expert has fewer replicas is padded, not trimmed.
    placement = ExpertLocationMetadata.from_physical_to_logical_map(
        torch.tensor([[0, 1, 2, 3, 0, 1]]),
        4,
        ep_size=2,
        ep_rank=0,
        ep_rank_nodes=(0, 0),
    )
    assert placement.logical_to_all_physical_map.shape == (1, 4, 3)
    assert placement.logical_to_all_physical_map[0].tolist() == [
        [0, 4, -1],
        [1, 5, -1],
        [2, -1, -1],
        [3, -1, -1],
    ]
    # Trivial placement (R = 0): one column.
    trivial = ExpertLocationMetadata.from_physical_to_logical_map(
        torch.tensor([[0, 1, 2, 3]]), 4, ep_size=2, ep_rank=0, ep_rank_nodes=(0, 0)
    )
    assert trivial.logical_to_all_physical_map.shape == (1, 4, 1)
    # The width never follows the placement: a table wider than R + 1 is
    # accepted only when its extra columns are empty.
    with pytest.raises(ValueError, match="more than"):
        expert_location.pad_replica_table(torch.tensor([[[0, 1, 2, 3]]]), 3)
    assert expert_location.pad_replica_table(
        torch.tensor([[[0, 1, -1, -1]]]), 3
    ).tolist() == [[[0, 1, -1]]]


def test_update_layers_rewrites_rows_in_place_and_reaches_the_views():
    placement = _placement(ep_rank=1)
    placement.enable_load_recording(_mapping())
    static_before = placement.rank_dispatch_map()
    # The router's views: the per-layer slices it indexes during a forward.
    replicas_view = placement.logical_to_all_physical_map[1]
    num_valid_view = placement.logical_to_all_physical_map_num_valid[1]
    static_view = placement.rank_dispatch_map()[1]
    device_map = placement.physical_to_logical_map
    host_map = placement.physical_to_logical_map_cpu
    storage = {
        id(placement.logical_to_all_physical_map.untyped_storage()),
        id(placement.logical_to_all_physical_map_num_valid.untyped_storage()),
        id(placement.physical_to_logical_map.untyped_storage()),
        id(placement.physical_to_logical_map_cpu.untyped_storage()),
        id(placement.rank_dispatch_map().untyped_storage()),
    }

    # Layer 1 goes from [3, 2, 1, 0, 1, 1] to [1, 2, 3, 0, 0, 1]: expert 0
    # gains a replica on rank 1, expert 1 loses one.
    placement.update_layers([1], torch.tensor([[1, 2, 3, 0, 0, 1]]))

    assert placement.physical_to_logical_map is device_map
    assert placement.physical_to_logical_map_cpu is host_map
    assert placement.rank_dispatch_map() is static_before
    assert {
        id(placement.logical_to_all_physical_map.untyped_storage()),
        id(placement.logical_to_all_physical_map_num_valid.untyped_storage()),
        id(placement.physical_to_logical_map.untyped_storage()),
        id(placement.physical_to_logical_map_cpu.untyped_storage()),
        id(placement.rank_dispatch_map().untyped_storage()),
    } == storage
    assert placement.local_slot_logical_experts(1, 1) == [0, 0, 1]
    assert placement.local_slot_logical_experts(1, 0) == [1, 2, 3]
    assert device_map[1].tolist() == [1, 2, 3, 0, 0, 1]
    # Layer 0 is untouched; layer 1's views now read the new rows.
    assert placement.local_slot_logical_experts(0, 1) == [3, 0, 2]
    assert replicas_view.tolist() == [[3, 4, -1], [0, 5, -1], [1, -1, -1], [2, -1, -1]]
    assert num_valid_view.tolist() == [2, 2, 1, 1]
    assert placement.logical_to_all_physical(1, 0) == [3, 4]
    # Rank 1 owns slots 3..5 and now prefers its own replica of experts 0 and 1.
    assert static_view[0].item() in (3, 4) and static_view[1].item() == 5
    assert static_view[2].item() == 1 and static_view[3].item() == 2
    assert static_before[0].tolist() == placement.rank_dispatch_map()[0].tolist()
    # The counters are untouched by a table switch.
    assert placement.physical_load.shape == (2, 6)
    with pytest.raises(ValueError, match="no physical slot"):
        placement.update_layers([0], torch.tensor([[0, 1, 2, 0, 0, 2]]))
    with pytest.raises(ValueError, match="expected"):
        placement.update_layers([0, 1], torch.tensor([[0, 1, 2, 3, 0, 2]]))
    with pytest.raises(ValueError, match="outside"):
        placement.update_layers([2], torch.tensor([[0, 1, 2, 3, 0, 2]]))


def test_snapshot_load_reads_then_zeroes_the_counters_with_their_map():
    placement = _placement(ep_rank=0)
    placement.enable_load_recording(_mapping())
    placement.physical_load.copy_(
        torch.tensor([[1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 1, 1]])
    )
    snapshot = placement.snapshot_load()
    assert snapshot.physical_count.dtype == torch.int64
    assert snapshot.physical_count.tolist() == [[1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 1, 1]]
    assert snapshot.logical_count.tolist() == [[2, 0, 0, 0], [0, 3, 0, 0]]
    assert not placement.physical_load.any()
    # The snapshot holds its own copies: a later table switch leaves it intact.
    placement.update_layers([1], torch.tensor([[1, 2, 3, 0, 0, 1]]))
    assert snapshot.physical_to_logical_map[1].tolist() == [3, 2, 1, 0, 1, 1]
    assert expert_location.load_balancedness(
        snapshot.physical_count, 2
    ).tolist() == pytest.approx([1.0, 0.75])
    with pytest.raises(RuntimeError, match="not enabled"):
        _placement(ep_rank=0).snapshot_load()


def test_compute_placement_maps_is_host_only_and_init_by_eplb_wraps_it():
    torch.manual_seed(1)
    load = torch.randint(1, 50, (2, 8))
    phy2log, log2phy = expert_location.compute_placement_maps(
        load,
        num_physical_experts=12,
        ep_size=4,
        num_groups=None,
        num_nodes=1,
        algorithm=eplb_algorithms.EplbAlgorithm.deepseek,
    )
    assert phy2log.shape == (2, 12) and phy2log.dtype == torch.int32
    assert log2phy.shape == (2, 8, 5) and log2phy.dtype == torch.int32
    assert phy2log.device.type == "cpu" and log2phy.device.type == "cpu"
    for layer in range(2):
        assert sorted(set(phy2log[layer].tolist())) == list(range(8))
    with pytest.raises(ValueError, match="logical_count must be"):
        expert_location.compute_placement_maps(
            load.unsqueeze(0),
            num_physical_experts=12,
            ep_size=4,
            num_groups=None,
            num_nodes=1,
            algorithm=eplb_algorithms.EplbAlgorithm.deepseek,
        )

    seen = {}

    def fake_compute(logical_count, **kwargs):
        seen["logical_count"] = logical_count
        seen.update(kwargs)
        return phy2log, log2phy

    args = SimpleNamespace(
        ep_num_redundant_experts=4,
        eplb_algorithm="deepseek",
        device="cpu",
        mapping=SimpleNamespace(
            nnodes=1,
            nprocs_per_node=4,
            moe=SimpleNamespace(ep_size=4, ep_rank=2, ep_group=(0, 1, 2, 3)),
        ),
    )
    geometry = expert_location.ModelConfigForExpertLocation(
        num_layers=2, num_logical_experts=8
    )
    with (
        mock.patch.object(
            expert_location.ModelConfigForExpertLocation,
            "from_model_config",
            staticmethod(lambda model_config: geometry),
        ),
        mock.patch.object(
            expert_location, "compute_placement_maps", side_effect=fake_compute
        ),
    ):
        # Several recording windows are summed on the host before the call.
        placement = ExpertLocationMetadata.init_by_eplb(
            args, None, torch.stack([load, load])
        )
    assert seen["logical_count"].device.type == "cpu"
    assert torch.equal(seen["logical_count"], 2 * load)
    assert seen["ep_size"] == 4 and seen["num_physical_experts"] == 12
    assert placement.ep_rank == 2 and placement.ep_rank_nodes == (0, 0, 0, 0)
    assert torch.equal(placement.physical_to_logical_map, phy2log)

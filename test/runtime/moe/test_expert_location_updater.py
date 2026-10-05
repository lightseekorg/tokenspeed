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

"""The expert location updater: staging, copy order, table switch, gloo P2P.

Weights are CPU tensors whose values encode ``(layer, logical expert)``, so
after an update every slot must hold its new expert's encoding and the
routing tables must read the new rows. The multi-process test applies a plan
across four gloo ranks on two "nodes" with ``batch_isend_irecv``.
"""

from __future__ import annotations

from datetime import timedelta
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.moe import expert_location
from tokenspeed.runtime.moe.expert_location import ExpertLocationMetadata
from tokenspeed.runtime.moe.expert_location_updater import ExpertLocationUpdater
from tokenspeed.runtime.moe.expert_rebalance import plan_slot_moves

NUM_LAYERS = 2


def _encode(layer: int, logical: int, shape, dtype=torch.float32) -> torch.Tensor:
    return torch.full(shape, float(100 * layer + logical), dtype=dtype)


def _weights(placement: ExpertLocationMetadata) -> dict[int, list[torch.Tensor]]:
    """Per layer: a 3-D weight, a 2-D weight and a per-slot scale, slot-major."""
    num_local = placement.num_local_physical_experts
    weights = {}
    for layer in range(placement.num_layers):
        logical = placement.local_slot_logical_experts(layer, placement.ep_rank)
        weights[layer] = [
            torch.stack([_encode(layer, e, (2, 3)) for e in logical]),
            torch.stack([_encode(layer, e, (4,), torch.bfloat16) for e in logical]),
            torch.stack([_encode(layer, e, ()) for e in logical]),
        ]
        assert all(t.shape[0] == num_local for t in weights[layer])
    return weights


def _assert_slots_hold(weights, placement, new_map):
    num_local = placement.num_local_physical_experts
    first = placement.ep_rank * num_local
    for layer, tensors in weights.items():
        for slot in range(num_local):
            logical = int(new_map[layer, first + slot])
            for tensor in tensors:
                assert torch.equal(
                    tensor[slot],
                    _encode(layer, logical, tensor[slot].shape, tensor.dtype),
                ), (layer, slot, logical)
        assert placement.local_slot_logical_experts(layer, placement.ep_rank) == [
            int(v) for v in new_map[layer, first : first + num_local]
        ]


def _placement(old_map, ep_rank, ep_rank_nodes):
    placement = ExpertLocationMetadata.from_physical_to_logical_map(
        old_map,
        int(old_map.max()) + 1,
        ep_size=len(ep_rank_nodes),
        ep_rank=ep_rank,
        ep_rank_nodes=ep_rank_nodes,
    )
    placement.enable_load_recording(
        Mapping(
            rank=ep_rank, world_size=len(ep_rank_nodes), moe_ep_size=len(ep_rank_nodes)
        )
    )
    return placement


def _moves(old_map, new_map, placement):
    return {
        layer: plan_slot_moves(
            old_map[layer],
            new_map[layer],
            ep_rank=placement.ep_rank,
            ep_rank_nodes=placement.ep_rank_nodes,
            num_local=placement.num_local_physical_experts,
        )
        for layer in range(placement.num_layers)
    }


# ----------------------------------------------------------------------
# Single rank: same-GPU cycles, free-riders, tables after weights, staging.
# ----------------------------------------------------------------------


def test_local_moves_keep_cycles_intact_and_switch_tables_after_weights():
    # One rank, four experts on six slots: layer 0 rotates experts 0 -> 2 ->
    # 1 -> 0 through their slots (a cycle) and gives expert 0 two more
    # replicas; layer 1 is untouched.
    old_map = torch.tensor([[0, 1, 2, 3, 0, 1], [0, 1, 2, 3, 0, 1]])
    new_map = torch.tensor([[1, 2, 0, 0, 3, 0], [0, 1, 2, 3, 0, 1]])
    placement = _placement(old_map, 0, (0,))
    weights = _weights(placement)
    order: list = []
    real_switch = placement.switch_layer

    def spy_switch(prepared, index):
        order.append(("tables", (prepared.layer_ids[index],), weights[0][2].tolist()))
        real_switch(prepared, index)

    updater = ExpertLocationUpdater(
        placement,
        weights,
        process_group=mock.Mock(name="pg"),
        peer_ranks=(0,),
        ep_rank=0,
        all_to_all_ep=False,
        num_groups=None,
        num_nodes=1,
    )
    assert updater.specs.num_local_physical_experts == 6
    assert updater.staging_bytes == 6 * (2 * 3 * 4 + 4 * 2 + 4)
    moves = _moves(old_map, new_map, placement)
    assert moves[0].local_copy == ((0, 1), (1, 2), (2, 0), (3, 0), (4, 3), (5, 0))
    assert moves[0].recv == () and moves[0].free_rider == ()
    assert moves[1].is_empty
    with mock.patch.object(
        dist, "batch_isend_irecv", side_effect=AssertionError("no P2P on one rank")
    ):
        with mock.patch.object(placement, "switch_layer", spy_switch):
            updater.apply((0, 1), new_map, moves)
    _assert_slots_hold(weights, placement, new_map)
    # The table switch of layer 0 ran after its slots held the new experts.
    assert order[0][:2] == ("tables", (0,))
    assert order[0][2] == [1.0, 2.0, 0.0, 0.0, 3.0, 0.0]
    assert order[1][:2] == ("tables", (1,))
    assert placement.logical_to_all_physical(0, 0) == [2, 3, 5]
    assert placement.logical_to_all_physical(1, 0) == [0, 4]
    # Load counters are not part of the move.
    assert placement.physical_load.shape == (2, 6)


def test_one_chunk_prepares_its_rows_once_and_switches_per_layer():
    """The host-side derivation (inverse rows and, under all-to-all EP, this
    rank's static dispatch rows) runs once per chunk over the chunk's layers
    only; each layer then switches with device copies after its weights."""
    old_map = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0], [0, 1, 2, 3]])
    new_map = torch.tensor([[1, 0, 2, 3], [3, 2, 0, 1], [0, 1, 2, 3]])
    placement = _placement(old_map, 0, (0, 0))
    static_map = placement.rank_dispatch_map()  # all-to-all EP materializes it
    weights = _weights(placement)
    updater = ExpertLocationUpdater(
        placement,
        weights,
        process_group=mock.Mock(name="pg"),
        peer_ranks=(0, 1),
        ep_rank=0,
        all_to_all_ep=True,
        num_groups=None,
        num_nodes=1,
    )
    moves = _moves(old_map, new_map, placement)
    calls: list = []
    real_compute = expert_location.compute_logical_to_rank_dispatch_physical_map

    def spy_compute(*, logical_to_all_physical_map, **kwargs):
        calls.append(tuple(logical_to_all_physical_map.shape))
        return real_compute(
            logical_to_all_physical_map=logical_to_all_physical_map, **kwargs
        )

    with (
        mock.patch.object(
            dist, "batch_isend_irecv", side_effect=AssertionError("no P2P here")
        ),
        mock.patch.object(
            expert_location,
            "compute_logical_to_rank_dispatch_physical_map",
            spy_compute,
        ),
        mock.patch.object(
            placement, "switch_layer", wraps=placement.switch_layer
        ) as switch,
    ):
        updater.apply((0, 1), new_map[:2], {k: moves[k] for k in (0, 1)})
    # One derivation for the two-layer chunk, over those two layers' rows.
    assert calls == [(2, 4, placement.logical_to_all_physical_map.shape[-1])]
    assert [c.args[1] for c in switch.call_args_list] == [0, 1]
    _assert_slots_hold(weights, placement, new_map)
    # The static map switched in place for the chunk's layers only.
    assert placement.rank_dispatch_map() is static_map
    # Every expert has one replica, so the map is the inverse of each row.
    assert static_map[0].tolist() == [1, 0, 2, 3]
    assert static_map[1].tolist() == [2, 3, 1, 0]
    assert static_map[2].tolist() == [0, 1, 2, 3]


def test_updater_rejects_mismatched_slot_tensors():
    old_map = torch.tensor([[0, 1, 2, 3]])
    placement = _placement(old_map, 0, (0, 0))
    good = {0: [torch.zeros(2, 3)]}
    kwargs = dict(
        process_group=mock.Mock(),
        peer_ranks=(0, 1),
        ep_rank=0,
        all_to_all_ep=False,
        num_groups=None,
        num_nodes=1,
    )
    ExpertLocationUpdater(placement, good, **kwargs)
    with pytest.raises(ValueError, match="one row per local slot"):
        ExpertLocationUpdater(placement, {0: [torch.zeros(3, 3)]}, **kwargs)
    with pytest.raises(ValueError, match="not contiguous"):
        ExpertLocationUpdater(placement, {0: [torch.zeros(3, 2).t()]}, **kwargs)
    with pytest.raises(ValueError, match="no routed expert weights"):
        ExpertLocationUpdater(placement, {}, **kwargs)
    with pytest.raises(ValueError, match="does not match"):
        ExpertLocationUpdater(placement, good, **{**kwargs, "peer_ranks": (0,)})
    two_layers = _placement(torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]), 0, (0, 0))
    with pytest.raises(ValueError, match="differ in layout"):
        ExpertLocationUpdater(
            two_layers, {0: [torch.zeros(2, 3)], 1: [torch.zeros(2, 4)]}, **kwargs
        )
    with pytest.raises(ValueError, match="outside"):
        ExpertLocationUpdater(two_layers, {0: good[0], 5: good[0]}, **kwargs)


# ----------------------------------------------------------------------
# Four gloo ranks on two nodes: P2P receives land in staging, sends pair.
# ----------------------------------------------------------------------

#              rank 0 (node 0) | rank 1 (node 0) | rank 2 (node 1) | rank 3 (node 1)
_OLD = torch.tensor(
    [
        [0, 1, 2] + [3, 4, 5] + [6, 7, 0] + [1, 2, 3],
        [5, 4, 3] + [2, 1, 0] + [7, 6, 5] + [4, 3, 2],
    ]
)
# Layer 0: rank 0 takes 7 from the other node, rank 1 swaps two slots, rank
# 2 replicates 6 locally, rank 3 takes 5 cross-node and free-rides a copy.
# Layer 1: rank 0 takes 0 from its node peer, rank 2 takes 1 cross-node plus
# a free-rider, rank 3 takes 6 from its node peer.
_NEW = torch.tensor(
    [
        [0, 7, 2] + [4, 3, 5] + [6, 6, 0] + [1, 5, 5],
        [5, 4, 0] + [2, 1, 0] + [7, 1, 1] + [4, 3, 6],
    ]
)
_NODES = (0, 0, 1, 1)


def _worker(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=60),
    )
    try:
        group = dist.new_group(list(range(4)), backend="gloo")
        placement = _placement(_OLD, rank, _NODES)
        weights = _weights(placement)
        updater = ExpertLocationUpdater(
            placement,
            weights,
            process_group=group,
            peer_ranks=(0, 1, 2, 3),
            ep_rank=rank,
            all_to_all_ep=True,
            num_groups=None,
            num_nodes=2,
        )
        updater.prewarm()
        moves = _moves(_OLD, _NEW, placement)
        # Two chunks of one layer each, like the controller issues them.
        updater.apply((0,), _NEW[0:1], moves)
        updater.apply((1,), _NEW[1:2], moves)
        _assert_slots_hold(weights, placement, _NEW)
        assert torch.equal(placement.physical_to_logical_map_cpu, _NEW)
        assert torch.equal(placement.physical_to_logical_map, _NEW.to(torch.int64))
        # Every rank's replica table reads the same, from the same rows.
        for layer in range(NUM_LAYERS):
            for logical in range(8):
                expected = [p for p in range(12) if int(_NEW[layer, p]) == logical]
                assert placement.logical_to_all_physical(layer, logical) == expected
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_four_gloo_ranks_apply_the_plan_with_batch_isend_irecv(tmp_path):
    mp.spawn(_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=4, join=True)

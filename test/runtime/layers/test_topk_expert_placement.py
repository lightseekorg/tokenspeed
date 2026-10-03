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

"""Routing onto placed replicas: zero-expert masking, dispatch flavours, load."""

from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.layers.moe.topk import (
    ExpertLocationDispatchInfo,
    TopK,
    TopKConfig,
    map_zero_expert_routes,
    record_expert_load,
    topk_ids_logical_to_physical,
)
from tokenspeed.runtime.moe.expert_location import ExpertLocationMetadata

# E=4 routed experts on P=6 slots over two ranks; expert 0 and 2 have two
# replicas, the second ones in [E, P).
_PHYSICAL_TO_LOGICAL = torch.tensor([[0, 1, 2, 3, 0, 2]])


def _placement(ep_rank: int) -> ExpertLocationMetadata:
    placement = ExpertLocationMetadata.from_physical_to_logical_map(
        _PHYSICAL_TO_LOGICAL, 4, ep_size=2, ep_rank=ep_rank, ep_rank_nodes=(0, 0)
    )
    placement.enable_load_recording()
    return placement


def _info(ep_rank: int, algorithm: str, *, all_to_all_ep: bool):
    placement = _placement(ep_rank)
    info = ExpertLocationDispatchInfo.init_new(
        layer_id=0,
        ep_dispatch_algorithm=algorithm,
        expert_location_metadata=placement,
        all_to_all_ep=all_to_all_ep,
    )
    # Only all-to-all EP under a static algorithm pays for the static map.
    assert (placement._rank_dispatch_map is not None) == (
        all_to_all_ep and algorithm.startswith("static")
    )
    return info


def test_all_to_all_ep_dispatches_to_the_ranks_own_replica():
    ids = torch.tensor([[0, 2, -1], [2, 0, 1]], dtype=torch.int64)
    mapped = {
        rank: map_zero_expert_routes(
            ids, _info(rank, "static_with_zero_expert", all_to_all_ep=True), 4
        )
        for rank in (0, 1)
    }
    # Rank 0 owns slots 0-2 (experts 0, 1, 2), rank 1 owns 3-5 (3, 0, 2).
    assert mapped[0].tolist() == [[0, 2, -1], [2, 0, 1]]
    assert mapped[1].tolist() == [[4, 5, -1], [5, 4, 1]]
    assert mapped[1].dtype == torch.int64
    assert ids.tolist() == [[0, 2, -1], [2, 0, 1]]  # the router's ids are intact


def test_replicated_input_ep_picks_the_same_replica_on_every_rank():
    ids = torch.tensor([[0, 2, -1], [0, 2, 3], [0, -1, 1]], dtype=torch.int32)
    mapped = [
        map_zero_expert_routes(
            ids, _info(rank, "static_with_zero_expert", all_to_all_ep=False), 4
        )
        for rank in (0, 1)
    ]
    assert torch.equal(mapped[0], mapped[1])
    # Replica (row + route rank) % count: expert 0 is on 0 and 4, expert 2 on 2
    # and 5; zero experts stay -1 and single-replica experts stay put.
    assert mapped[0].tolist() == [[0, 5, -1], [4, 2, 3], [0, -1, 1]]
    assert mapped[0].dtype == torch.int32


def test_replicated_input_ep_refuses_random_replica_choice():
    with pytest.raises(ValueError, match="agree on one replica"):
        _info(0, "dynamic_with_zero_expert", all_to_all_ep=False)
    # All-to-all EP may draw at random; the placement needs no static map.
    info = _info(0, "dynamic_with_zero_expert", all_to_all_ep=True)
    mapped = map_zero_expert_routes(torch.tensor([[0, -1], [3, 2]]), info, 4)
    assert mapped[0, 1] == -1 and mapped[1, 0] == 3
    assert int(mapped[0, 0]) in (0, 4) and int(mapped[1, 1]) in (2, 5)


def test_zero_expert_algorithms_keep_legacy_ids_beyond_the_routed_count():
    # A router that numbers zero experts from E (fllm convention) is left
    # alone past E, and -1 never indexes the map.
    info = _info(1, "static_with_zero_expert", all_to_all_ep=True)
    ids = torch.tensor([[0, 4, 5, -1]])
    assert topk_ids_logical_to_physical(ids, info, num_experts=4).tolist() == [
        [4, 4, 5, -1]
    ]


def test_record_expert_load_counts_real_routes_only():
    info = _info(0, "static_with_zero_expert", all_to_all_ep=False)
    ids = map_zero_expert_routes(torch.tensor([[0, 2, -1], [0, 2, 3]]), info, 4)
    record_expert_load(info.physical_load, ids)
    record_expert_load(info.physical_load, ids)
    assert info.physical_load.dtype == torch.int64
    assert info.physical_load.tolist() == [2, 0, 2, 2, 2, 2]
    record_expert_load(None, ids)  # recording off: no-op


def test_topk_config_carries_the_layer_id():
    topk = TopK(top_k=2, layer_id=7, correction_bias=torch.zeros(4))
    assert topk.topk_config.layer_id == 7
    assert TopKConfig(top_k=2).layer_id is None

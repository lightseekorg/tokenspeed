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

"""The EPLB placement algorithm is deterministic and exact on the host.

Online rebalancing derives the placement on one rank and broadcasts it, but
the same inputs must still give the same maps on every run: equal loads tie
in index order (stable sort), and counts are kept in double precision so a
long window cannot be rounded by float32.
"""

from __future__ import annotations

import torch

from tokenspeed.runtime.moe.eplb_algorithms import EplbAlgorithm, deepseek
from tokenspeed.runtime.moe.expert_location import compute_placement_maps


def _maps(load, **kwargs):
    return compute_placement_maps(
        load,
        num_groups=None,
        num_nodes=1,
        algorithm=EplbAlgorithm.deepseek,
        **kwargs,
    )


def test_two_calls_on_the_same_load_give_the_same_maps():
    torch.manual_seed(3)
    load = torch.randint(0, 1000, (4, 64))
    first = _maps(load, num_physical_experts=96, ep_size=8)
    second = _maps(load.clone(), num_physical_experts=96, ep_size=8)
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])
    assert first[0].shape == (4, 96) and first[1].shape == (4, 64, 33)


def test_ties_resolve_in_index_order():
    # Every expert equally loaded: the packing must not depend on sort
    # instability; the hot replicas go to the lowest ids first.
    load = torch.full((2, 8), 10)
    phy2log, _ = _maps(load, num_physical_experts=12, ep_size=4)
    for layer in range(2):
        counts = torch.bincount(phy2log[layer].long(), minlength=8).tolist()
        # Four extra slots: experts 0..3 get the replicas, in order.
        assert counts == [2, 2, 2, 2, 1, 1, 1, 1]
    # Equal weights pack round-robin: the first pack with the least weight.
    packing, _ = deepseek.balanced_packing(torch.full((1, 8), 1.0), 4)
    assert packing.tolist() == [[0, 1, 2, 3, 0, 1, 2, 3]]


def test_every_expert_keeps_a_replica_and_counts_stay_exact_past_two_to_24():
    # Counts past 2^24 collapse to one value in float32 (the spacing at 2^26
    # is 8), which would hand the replicas to experts 0 and 1 by tie order;
    # in double the two hottest experts, 1 and 2, get them.
    big = torch.tensor([[2**26, 2**26 + 3, 2**26 + 2, 2**26 + 1]])
    assert big.float().unique().numel() == 1
    phy2log, _ = _maps(big, num_physical_experts=6, ep_size=2)
    assert torch.bincount(phy2log[0].long(), minlength=4).tolist() == [1, 2, 2, 1]
    torch.manual_seed(7)
    load = torch.randint(0, 50, (3, 16))
    load[:, 5] = 0  # a cold expert still gets exactly one slot
    phy2log, log2phy = _maps(load, num_physical_experts=24, ep_size=4)
    for layer in range(3):
        assert set(phy2log[layer].tolist()) == set(range(16))
    assert ((log2phy >= 0).sum(-1) >= 1).all()
    assert (log2phy[:, 5] >= 0).sum(-1).tolist() == [1, 1, 1]

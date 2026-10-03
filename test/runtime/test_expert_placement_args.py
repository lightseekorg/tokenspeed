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

"""Expert placement flags: the dispatch algorithm vocabulary and ServerArgs checks."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tokenspeed.runtime.moe.dispatch_algorithm import (
    EP_DISPATCH_ALGORITHMS,
    STATIC_EP_DISPATCH_ALGORITHMS,
    has_zero_expert,
)
from tokenspeed.runtime.utils.server_args import (
    ServerArgs,
    expert_placement_requested,
)


def test_placement_is_requested_only_when_it_changes_routing():
    def args(**kw):
        base = dict(
            ep_num_redundant_experts=0,
            init_expert_location="trivial",
            expert_distribution_recorder_mode=None,
        )
        base.update(kw)
        return SimpleNamespace(**base)

    assert not expert_placement_requested(args())
    assert expert_placement_requested(args(ep_num_redundant_experts=8))
    assert expert_placement_requested(args(init_expert_location="/tmp/load.pt"))
    assert expert_placement_requested(args(expert_distribution_recorder_mode="stat"))


def test_dispatch_algorithm_vocabulary_is_defined_once():
    assert STATIC_EP_DISPATCH_ALGORITHMS <= set(EP_DISPATCH_ALGORITHMS)
    assert has_zero_expert("static_with_zero_expert")
    assert has_zero_expert("dynamic_with_zero_expert")
    assert not has_zero_expert("static") and not has_zero_expert("fake")
    with pytest.raises(ValueError, match="unknown"):
        has_zero_expert("nearest")


class TestServerArgsPlacementValidation:
    def test_trivial_serving_needs_no_dispatch_algorithm(self):
        args = ServerArgs(model="x")
        assert args.ep_dispatch_algorithm is None
        assert not expert_placement_requested(args)

    def test_placement_requires_an_explicit_dispatch_algorithm(self):
        ep = dict(attn_tp_size=2, ep_size=2)
        with pytest.raises(ValueError, match="--ep-dispatch-algorithm is required"):
            ServerArgs(model="x", ep_num_redundant_experts=8, **ep)
        with pytest.raises(ValueError, match="--ep-dispatch-algorithm is required"):
            ServerArgs(model="x", init_expert_location="/tmp/load.pt")
        with pytest.raises(ValueError, match="--ep-dispatch-algorithm is required"):
            ServerArgs(model="x", expert_distribution_recorder_mode="stat")
        args = ServerArgs(
            model="x",
            ep_num_redundant_experts=8,
            init_expert_location="/tmp/load.pt",
            ep_dispatch_algorithm="static_with_zero_expert",
            **ep,
        )
        assert args.ep_dispatch_algorithm == "static_with_zero_expert"

    def test_redundant_experts_need_expert_parallelism(self):
        with pytest.raises(ValueError, match="ep_size=1"):
            ServerArgs(
                model="x", ep_num_redundant_experts=8, ep_dispatch_algorithm="static"
            )
        with pytest.raises(ValueError, match="ep_size=1"):
            ServerArgs(
                model="x",
                attn_tp_size=2,
                ep_num_redundant_experts=8,
                ep_dispatch_algorithm="static",
            )
        # Recording load on a MoE-TP-only server is fine: no replicas needed.
        args = ServerArgs(
            model="x",
            attn_tp_size=2,
            expert_distribution_recorder_mode="stat",
            ep_dispatch_algorithm="static",
        )
        assert args.mapping.moe.ep_size == 1

    def test_dispatch_algorithm_without_a_placement_is_refused(self):
        with pytest.raises(ValueError, match="has no effect"):
            ServerArgs(model="x", ep_dispatch_algorithm="static")

    def test_runtime_rebalancing_is_refused(self):
        with pytest.raises(ValueError, match="--enable-eplb"):
            ServerArgs(model="x", enable_eplb=True)

    def test_only_stat_recording_exists(self):
        with pytest.raises(ValueError, match="only 'stat'"):
            ServerArgs(
                model="x",
                expert_distribution_recorder_mode="per_token",
                ep_dispatch_algorithm="static",
            )

    def test_rl_bitwise_refuses_random_replica_choice(self):
        ep = dict(attn_tp_size=2, ep_size=2)
        with pytest.raises(ValueError, match="deterministic expert placement"):
            ServerArgs(
                model="x",
                numerics="rl-bitwise",
                ep_num_redundant_experts=8,
                ep_dispatch_algorithm="dynamic_with_zero_expert",
                **ep,
            )
        args = ServerArgs(
            model="x",
            numerics="rl-bitwise",
            ep_num_redundant_experts=8,
            ep_dispatch_algorithm="static_with_zero_expert",
            **ep,
        )
        assert args.ep_num_redundant_experts == 8

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

import argparse
import dataclasses
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
            enable_eplb=False,
        )
        base.update(kw)
        return SimpleNamespace(**base)

    assert not expert_placement_requested(args())
    assert expert_placement_requested(args(ep_num_redundant_experts=8))
    assert expert_placement_requested(args(init_expert_location="/tmp/load.pt"))
    assert expert_placement_requested(args(expert_distribution_recorder_mode="stat"))
    assert expert_placement_requested(args(enable_eplb=True))


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

    def test_online_rebalancing_spells_out_every_choice(self):
        ep = dict(attn_tp_size=2, ep_size=2)
        full = dict(
            enable_eplb=True,
            expert_distribution_recorder_mode="stat",
            ep_dispatch_algorithm="static_with_zero_expert",
            eplb_rebalance_num_iterations=10000,
            eplb_rebalance_layers_per_chunk=4,
            **ep,
        )
        args = ServerArgs(model="x", **full)
        assert args.enable_eplb and expert_placement_requested(args)
        assert args.ep_num_redundant_experts == 0  # pure permutation is allowed
        # Nothing is auto-set: each missing or wrong choice is named.
        for drop, match in (
            ("expert_distribution_recorder_mode", "recorder-mode stat"),
            ("eplb_rebalance_num_iterations", "num-iterations"),
            ("eplb_rebalance_layers_per_chunk", "layers-per-chunk"),
        ):
            with pytest.raises(ValueError, match=match):
                ServerArgs(model="x", **{**full, drop: None})
        with pytest.raises(ValueError, match="num-iterations"):
            ServerArgs(model="x", **{**full, "eplb_rebalance_num_iterations": 0})
        with pytest.raises(ValueError, match="layers-per-chunk"):
            ServerArgs(model="x", **{**full, "eplb_rebalance_layers_per_chunk": 0})
        for algorithm in ("dynamic_with_zero_expert", "fake", None):
            with pytest.raises(ValueError, match="static replica choice"):
                ServerArgs(model="x", **{**full, "ep_dispatch_algorithm": algorithm})
        with pytest.raises(ValueError, match="ep_size=1"):
            ServerArgs(model="x", **{**full, "ep_size": 1, "attn_tp_size": 1})
        # The knobs mean nothing without the switch.
        with pytest.raises(ValueError, match="no effect without --enable-eplb"):
            ServerArgs(model="x", eplb_rebalance_num_iterations=10, **ep)
        with pytest.raises(ValueError, match="no effect without --enable-eplb"):
            ServerArgs(model="x", eplb_rebalance_layers_per_chunk=1, **ep)
        # Under a bitwise envelope the placement-independent MoE combine is
        # required; the envelope folds it, so the launch is accepted with it.
        args = ServerArgs(model="x", numerics="rl-bitwise", **full)
        assert args.enable_eplb and args.moe_combine_order == "slot"

    def test_recorder_buffer_size_knob_is_gone(self):
        assert "expert_distribution_recorder_buffer_size" not in {
            f.name for f in __import__("dataclasses").fields(ServerArgs)
        }

    def test_only_stat_recording_exists(self):
        with pytest.raises(ValueError, match="only 'stat'"):
            ServerArgs(
                model="x",
                expert_distribution_recorder_mode="per_token",
                ep_dispatch_algorithm="static",
            )
        # The CLI admits only that value, and the metrics switch of the
        # deleted recorder is gone with it.
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        action = next(
            a
            for a in parser._actions
            if "--expert-distribution-recorder-mode" in a.option_strings
        )
        assert list(action.choices) == ["stat"]
        assert not any(
            "--enable-expert-distribution-metrics" in a.option_strings
            for a in parser._actions
        )
        assert "enable_expert_distribution_metrics" not in {
            f.name for f in dataclasses.fields(ServerArgs)
        }

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

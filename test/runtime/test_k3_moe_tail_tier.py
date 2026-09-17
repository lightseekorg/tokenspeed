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

"""Truth table for the K3 small-tail and separate-reduce selector."""

from types import SimpleNamespace

import pytest

from tokenspeed.runtime.models.kimi_k3_comm import (
    K3MoeTailComm,
    K3MoETailTier,
    select_k3_moe_tail_tier,
)


def _select(**overrides):
    args = dict(
        num_tokens=1024,
        graph_phase=False,
        tail_fusion_max_tokens=32,
    )
    args.update(overrides)
    return select_k3_moe_tail_tier(**args)


@pytest.mark.parametrize("m", [1, 8, 16, 32])
def test_small_graph_range_uses_fused_tail(m):
    assert _select(num_tokens=m, graph_phase=True) is K3MoETailTier.TAIL_FUSION


@pytest.mark.parametrize("m", [1, 8, 16, 32, 64, 256, 1024, 8192, 16384])
def test_eager_compatibility_path_uses_separate_reduce(m):
    assert _select(num_tokens=m) is K3MoETailTier.SEPARATE_REDUCE


@pytest.mark.parametrize("m", [0, 64, 256, 1024, 8192, 16384])
def test_outside_small_graph_capacity_uses_separate_reduce(m):
    assert _select(num_tokens=m, graph_phase=True) is K3MoETailTier.SEPARATE_REDUCE


def test_missing_small_tail_uses_separate_reduce():
    assert (
        _select(num_tokens=8, graph_phase=True, tail_fusion_max_tokens=0)
        is K3MoETailTier.SEPARATE_REDUCE
    )


@pytest.mark.parametrize(
    "integrated,m",
    [(False, 8), (False, 256), (False, 8192), (True, 8), (True, 16384)],
)
def test_fallback_plan_reduces_and_projects_routed_in_fork(monkeypatch, integrated, m):
    from tokenspeed.runtime.models import kimi_k3_comm

    monkeypatch.setattr(kimi_k3_comm, "get_is_cuda_graph_phase", lambda: False)
    comm = object.__new__(K3MoeTailComm)
    comm.state = SimpleNamespace(integrated_tail=integrated)
    comm.latent_tail = None

    plan = comm.plan(m)
    assert plan.tier is K3MoETailTier.SEPARATE_REDUCE
    assert plan.routed_in_fork
    assert not plan.defer_finalize
    assert not plan.split_shared_rs
    assert plan.symm_outputs is None


def test_only_four_tail_routes_remain():
    assert set(K3MoETailTier.__members__) == {
        "TAIL_FUSION",
        "MEDIUM_FUSED_RS_UP_AG",
        "FUSED_RS_UP_AG",
        "SEPARATE_REDUCE",
    }

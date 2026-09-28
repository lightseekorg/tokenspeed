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

"""K3 MoE-tail routing and sharded output assembly."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

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
    "deferred,m",
    [(False, 8), (False, 256), (False, 8192), (True, 8), (True, 16384)],
)
def test_fallback_plan_reduces_and_projects_routed_in_fork(monkeypatch, deferred, m):
    from tokenspeed.runtime.models import kimi_k3_comm

    monkeypatch.setattr(kimi_k3_comm, "get_is_cuda_graph_phase", lambda: False)
    comm = object.__new__(K3MoeTailComm)
    comm.state = SimpleNamespace(deferred_tail=deferred)
    comm.latent_tail = None

    plan = comm.plan(m)
    assert plan.tier is K3MoETailTier.SEPARATE_REDUCE
    assert plan.routed_in_fork
    assert not plan.defer_finalize
    assert not plan.split_shared_rs


def test_only_four_tail_routes_remain():
    assert set(K3MoETailTier.__members__) == {
        "TAIL_FUSION",
        "MNNVL_BT",
        "MNNVL_HT",
        "SEPARATE_REDUCE",
    }


@pytest.mark.parametrize(
    "m,tier", [(33, K3MoETailTier.MNNVL_BT), (1025, K3MoETailTier.MNNVL_HT)]
)
def test_deferred_tail_uses_main_sharded_projection_and_shared_reduce(
    monkeypatch, m, tier
):
    from tokenspeed.runtime.models import kimi_k3_comm

    normalized = torch.ones(m, 3, dtype=torch.bfloat16)
    shared = torch.ones(m, 16, dtype=torch.bfloat16)
    residual = torch.full_like(shared, 3)
    reduced = torch.full_like(shared, 21)
    bt = Mock(return_value=normalized)
    ht = Mock(return_value=normalized)
    bt.supports_num_tokens.return_value = True
    ht.supports_num_tokens.return_value = True
    group = tuple(range(8))

    def reduce(value, actual_group):
        assert actual_group == group
        assert torch.all(value[:, :4] == 1)
        assert torch.all(value[:, 4:6] == 10)
        assert torch.all(value[:, 6:] == 1)
        return reduced

    shared_reduce = Mock(side_effect=reduce)
    monkeypatch.setattr(kimi_k3_comm, "all_reduce", shared_reduce)
    comm = object.__new__(K3MoeTailComm)
    comm.state = SimpleNamespace(
        deferred_tail=True, mnnvl_bt_deferred=bt, mnnvl_ht_deferred=ht
    )
    comm._experts_supports_deferred_finalize = True
    comm.routed_norm = SimpleNamespace(weight=torch.ones(3, dtype=torch.bfloat16))
    comm.up_proj = SimpleNamespace(
        shard_slice=(4, 2), weight=torch.full((2, 3), 2, dtype=torch.bfloat16)
    )
    comm.mapping = SimpleNamespace(moe=SimpleNamespace(tp_ep_size=8, tp_ep_group=group))
    plan = comm.plan(m)
    assert plan.tier is tier and plan.defer_finalize
    assert not plan.routed_in_fork
    deferred = (object(), object(), object())
    result = comm.run(plan, deferred, shared, residual, m, 16)

    selected, other = (bt, ht) if tier is K3MoETailTier.MNNVL_BT else (ht, bt)
    selected.assert_called_once_with(*deferred, comm.routed_norm.weight)
    other.assert_not_called()
    shared_reduce.assert_called_once()
    assert result.data_ptr() == reduced.data_ptr()
    torch.testing.assert_close(result, reduced, rtol=0, atol=0)

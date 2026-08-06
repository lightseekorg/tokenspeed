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

"""Truth table for the K3 MoE tail tier selector."""

import pytest

from tokenspeed.runtime.layers.moe.latent import (
    K3MoETailTier,
    select_k3_moe_tail_tier,
)


def _select(**overrides):
    base = dict(
        num_tokens=1024,
        graph_phase=False,
        tail_fusion_max_tokens=16,
        fused_moe_ar=True,
        multimem_ok=True,
        hidden_shardable=True,
    )
    base.update(overrides)
    return select_k3_moe_tail_tier(**base)


@pytest.mark.parametrize("m", [1, 8, 16])
def test_decode_graph_range_uses_fused_tail(m):
    assert _select(num_tokens=m, graph_phase=True) is K3MoETailTier.TAIL_FUSION


def test_fused_tail_needs_graph_phase_and_capacity():
    assert _select(num_tokens=8, graph_phase=False) is not K3MoETailTier.TAIL_FUSION
    assert (
        _select(num_tokens=8, graph_phase=True, tail_fusion_max_tokens=0)
        is not K3MoETailTier.TAIL_FUSION
    )


def test_no_fused_ar_always_separate_reduce():
    for m in (1, 17, 2048, 8192):
        assert (
            _select(num_tokens=m, fused_moe_ar=False) is K3MoETailTier.SEPARATE_REDUCE
        )


@pytest.mark.parametrize("m", [17, 1024, 2047, 2048, 8192])
def test_multimem_stitch_covers_everything_above_decode(m):
    assert _select(num_tokens=m) is K3MoETailTier.MULTIMEM_STITCH


def test_multimem_lower_bound_is_exclusive_of_decode_range():
    assert _select(num_tokens=16) is K3MoETailTier.FUSED_LANE_AR
    assert _select(num_tokens=17) is K3MoETailTier.MULTIMEM_STITCH


@pytest.mark.parametrize(
    "m,expected",
    [
        (2047, K3MoETailTier.FUSED_LANE_AR),
        (2048, K3MoETailTier.NCCL_STITCH),
        (8192, K3MoETailTier.NCCL_STITCH),
    ],
)
def test_nccl_stitch_fallback_without_multimem(m, expected):
    assert _select(num_tokens=m, multimem_ok=False) is expected


def test_unshardable_hidden_stays_on_fused_lane():
    for m in (17, 2048, 8192):
        assert (
            _select(num_tokens=m, hidden_shardable=False) is K3MoETailTier.FUSED_LANE_AR
        )


def test_graph_phase_above_fused_capacity_still_tiers_by_tokens():
    assert _select(num_tokens=32, graph_phase=True) is K3MoETailTier.MULTIMEM_STITCH

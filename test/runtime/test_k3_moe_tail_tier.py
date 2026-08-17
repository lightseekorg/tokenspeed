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

from tokenspeed.runtime.models.kimi_k3_comm import (
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
    )
    base.update(overrides)
    return select_k3_moe_tail_tier(**base)


@pytest.mark.parametrize("m", [1, 8, 16])
def test_decode_graph_range_uses_fused_tail(m):
    assert _select(num_tokens=m, graph_phase=True) is K3MoETailTier.TAIL_FUSION


def test_fused_tail_needs_graph_phase_and_capacity():
    assert _select(num_tokens=8, graph_phase=False) is K3MoETailTier.FUSED_LANE_AR
    assert (
        _select(num_tokens=8, graph_phase=True, tail_fusion_max_tokens=0)
        is K3MoETailTier.FUSED_LANE_AR
    )


def test_no_fused_ar_selects_separate_reduce_outside_the_fused_tail():
    for m in (1, 17, 2048, 8192):
        assert (
            _select(num_tokens=m, fused_moe_ar=False) is K3MoETailTier.SEPARATE_REDUCE
        )


@pytest.mark.parametrize("m", [256, 1024, 2047, 2048, 8192])
def test_multimem_ar_covers_the_measured_window(m):
    assert _select(num_tokens=m) is K3MoETailTier.MULTIMEM_AR


def test_multimem_ar_upper_bound_hands_off_to_the_join():
    assert _select(num_tokens=8192) is K3MoETailTier.MULTIMEM_AR
    assert _select(num_tokens=8193) is K3MoETailTier.FUSED_LANE_AR


def test_fused_tail_wins_even_without_fused_ar():
    assert (
        _select(num_tokens=8, graph_phase=True, fused_moe_ar=False)
        is K3MoETailTier.TAIL_FUSION
    )


def test_graph_phase_without_multimem_lands_on_fused_lane():
    assert (
        _select(num_tokens=512, graph_phase=True, multimem_ok=False)
        is K3MoETailTier.FUSED_LANE_AR
    )


def test_multimem_lower_bound_excludes_decode_bucket_sizes():
    for m in (16, 17, 32, 160, 255):
        assert _select(num_tokens=m) is K3MoETailTier.FUSED_LANE_AR
    assert _select(num_tokens=256) is K3MoETailTier.MULTIMEM_AR


@pytest.mark.parametrize("m", [256, 2047, 2048, 8192])
def test_fused_lane_fallback_without_multimem(m):
    assert _select(num_tokens=m, multimem_ok=False) is K3MoETailTier.FUSED_LANE_AR


def test_graph_phase_above_fused_capacity_still_tiers_by_tokens():
    assert _select(num_tokens=512, graph_phase=True) is K3MoETailTier.MULTIMEM_AR

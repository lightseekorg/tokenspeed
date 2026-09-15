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

from types import SimpleNamespace

import pytest

from tokenspeed.runtime.models import kimi_k3_comm
from tokenspeed.runtime.models.kimi_k3_comm import (
    MNNVL_BT_MAX_TOKENS,
    MNNVL_BT_MIN_TOKENS,
    MNNVL_BT_QUALIFIED_TOKENS,
    MNNVL_HT_MAX_TOKENS,
    MNNVL_HT_MIN_TOKENS,
    MNNVL_HT_QUALIFIED_TOKENS,
    K3MoETailTier,
    select_k3_moe_tail_tier,
)
from tokenspeed.runtime.utils import env
from tokenspeed.runtime.utils.server_args import ServerArgs


def _select(**overrides):
    base = dict(
        num_tokens=1024,
        graph_phase=False,
        tail_fusion_max_tokens=16,
        fused_moe_ar=True,
        multimem_ok=True,
        is_decode=False,
        join_moe_reduce=False,
        mnnvl_bt_deferred_ok=False,
        mnnvl_ht_deferred_ok=False,
        fused_rs_up_ag_ok=False,
        prefill_graph_phase=False,
    )
    base.update(overrides)
    return select_k3_moe_tail_tier(**base)


class _AlwaysSupportedWorkspace:
    def supports_num_tokens(self, _num_tokens):
        return True


def _plan_comm():
    comm = kimi_k3_comm.K3MoeTailComm.__new__(kimi_k3_comm.K3MoeTailComm)
    comm.state = SimpleNamespace(
        mnnvl_bt_deferred=_AlwaysSupportedWorkspace(),
        mnnvl_ht_deferred=_AlwaysSupportedWorkspace(),
        multimem_ar_ok=True,
    )
    comm.execution_plan = SimpleNamespace(
        fused_moe_ar=True,
        join_moe_reduce=False,
    )
    comm.latent_tail = None
    comm._experts_supports_deferred_finalize = True
    comm._shard_up_projection = True
    comm.mapping = SimpleNamespace(moe=SimpleNamespace(tp_ep_size=8))
    comm.routed_hidden = 3584
    comm.hidden_size = 7168
    return comm


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


def test_deferred_mnnvl_protocols_only_preempt_prefill_graphs():
    assert (
        _select(
            num_tokens=512,
            prefill_graph_phase=True,
            mnnvl_bt_deferred_ok=True,
        )
        is K3MoETailTier.MNNVL_BT_DEFERRED
    )
    assert (
        _select(
            num_tokens=4096,
            prefill_graph_phase=True,
            mnnvl_ht_deferred_ok=True,
        )
        is K3MoETailTier.MNNVL_HT_DEFERRED
    )
    assert (
        _select(num_tokens=512, mnnvl_bt_deferred_ok=True) is K3MoETailTier.MULTIMEM_AR
    )
    assert (
        _select(
            num_tokens=4096,
            graph_phase=True,
            prefill_graph_phase=True,
            is_decode=True,
            mnnvl_ht_deferred_ok=True,
        )
        is K3MoETailTier.FUSED_LANE_AR
    )


def test_deferred_mnnvl_ranges_match_the_qualified_complete_tail_windows():
    assert (MNNVL_BT_MIN_TOKENS, MNNVL_BT_MAX_TOKENS) == (256, 1024)
    assert (MNNVL_HT_MIN_TOKENS, MNNVL_HT_MAX_TOKENS) == (1280, 8192)
    assert MNNVL_BT_QUALIFIED_TOKENS == {256, 384, 512, 768, 1024}
    assert MNNVL_HT_QUALIFIED_TOKENS == {1280, 2048, 4096, 6144, 8192}


@pytest.mark.parametrize(
    ("m", "expected"),
    [
        *(
            (m, K3MoETailTier.MNNVL_BT_DEFERRED)
            for m in sorted(MNNVL_BT_QUALIFIED_TOKENS)
        ),
        *(
            (m, K3MoETailTier.MNNVL_HT_DEFERRED)
            for m in sorted(MNNVL_HT_QUALIFIED_TOKENS)
        ),
    ],
)
def test_plan_enables_deferred_finalize_only_at_qualified_buckets(
    monkeypatch, m, expected
):
    monkeypatch.setattr(kimi_k3_comm, "get_is_cuda_graph_phase", lambda: False)
    monkeypatch.setattr(kimi_k3_comm, "get_is_prefill_graph_phase", lambda: True)

    plan = _plan_comm().plan(m, hidden_states=None)

    assert plan.tier is expected
    assert plan.defer_finalize is True


@pytest.mark.parametrize("m", [257, 1023, 1152, 1536])
def test_plan_keeps_exact_bucket_gaps_on_the_established_tier(monkeypatch, m):
    monkeypatch.setattr(kimi_k3_comm, "get_is_cuda_graph_phase", lambda: False)
    monkeypatch.setattr(kimi_k3_comm, "get_is_prefill_graph_phase", lambda: True)

    plan = _plan_comm().plan(m, hidden_states=None)

    assert plan.tier is K3MoETailTier.MULTIMEM_AR
    assert plan.defer_finalize is False


@pytest.mark.parametrize(("m", "is_decode"), [(512, False), (2048, False)])
def test_plan_keeps_eager_prefill_on_the_established_tier(monkeypatch, m, is_decode):
    monkeypatch.setattr(kimi_k3_comm, "get_is_cuda_graph_phase", lambda: False)
    monkeypatch.setattr(kimi_k3_comm, "get_is_prefill_graph_phase", lambda: False)

    plan = _plan_comm().plan(m, hidden_states=None, is_decode=is_decode)

    assert plan.tier is K3MoETailTier.MULTIMEM_AR
    assert plan.defer_finalize is False


@pytest.mark.parametrize("m", [512, 2048])
def test_plan_keeps_decode_on_the_established_tier(monkeypatch, m):
    monkeypatch.setattr(kimi_k3_comm, "get_is_cuda_graph_phase", lambda: True)
    monkeypatch.setattr(kimi_k3_comm, "get_is_prefill_graph_phase", lambda: False)

    plan = _plan_comm().plan(m, hidden_states=None, is_decode=True)

    assert plan.tier is K3MoETailTier.FUSED_LANE_AR
    assert plan.defer_finalize is False


def test_mnnvl_capacity_tracks_the_largest_real_prefill_graph_bucket(monkeypatch):
    monkeypatch.setitem(
        kimi_k3_comm.global_server_args_dict,
        "prefill_graph_max_tokens",
        2048,
    )
    assert kimi_k3_comm._mnnvl_graph_max_tokens() == 2048
    monkeypatch.setitem(
        kimi_k3_comm.global_server_args_dict,
        "prefill_graph_capture_sizes",
        [256, 512],
    )
    assert kimi_k3_comm._mnnvl_graph_max_tokens() == 2048


@pytest.mark.parametrize(
    ("all2all_backend", "explicit_max", "capture_sizes", "expected_max"),
    [
        ("none", None, None, 2048),
        ("none", 8192, [256, 512, 2048], 8192),
        ("deepep", 8192, [256, 512, 2048], 0),
    ],
)
def test_server_args_update_feeds_resolved_mnnvl_graph_capacity(
    monkeypatch,
    all2all_backend,
    explicit_max,
    capture_sizes,
    expected_max,
):
    """The pre-model global update is the selector's source of truth."""

    # This is a ServerArgs unit test, not a launcher-topology test.  Do not
    # inherit the enclosing Slurm step's two-node topology when it runs in the
    # distributed GB300 validation job.
    for name in ("SLURM_STEP_NUM_NODES", "SLURM_NODEID", "SLURM_STEP_NODELIST"):
        monkeypatch.delenv(name, raising=False)
    # Keep the unit test independent of host-specific ephemeral-port policies.
    args = ServerArgs(model="stub", dist_init_addr="127.0.0.1:7654")
    args.all2all_backend = all2all_backend
    args.prefill_graph_max_tokens = explicit_max
    args.prefill_graph_capture_sizes = capture_sizes
    args.chunked_prefill_size = 8192
    args.max_total_tokens = None
    snapshot = dict(env.global_server_args_dict)
    monkeypatch.setattr(env, "pdl_enabled", lambda _: None)
    try:
        env.global_server_args_dict_update(args)
        assert kimi_k3_comm._mnnvl_graph_max_tokens() == expected_max
        assert (
            env.global_server_args_dict["prefill_graph_capture_sizes"] == capture_sizes
        )
    finally:
        env.global_server_args_dict.clear()
        env.global_server_args_dict.update(snapshot)


def test_mnnvl_candidate_is_qualified_only_for_moe_tp8_ep1():
    def mapping(tp_size, ep_size):
        return SimpleNamespace(moe=SimpleNamespace(tp_size=tp_size, ep_size=ep_size))

    assert kimi_k3_comm._mnnvl_tp8_layout(mapping(8, 1))
    assert not kimi_k3_comm._mnnvl_tp8_layout(mapping(1, 8))
    assert not kimi_k3_comm._mnnvl_tp8_layout(mapping(2, 4))


def test_fused_tail_and_bt_keep_priority_over_ht():
    assert (
        _select(
            num_tokens=8,
            graph_phase=True,
            prefill_graph_phase=True,
            mnnvl_bt_deferred_ok=True,
            mnnvl_ht_deferred_ok=True,
        )
        is K3MoETailTier.TAIL_FUSION
    )
    assert (
        _select(
            num_tokens=512,
            prefill_graph_phase=True,
            mnnvl_bt_deferred_ok=True,
            mnnvl_ht_deferred_ok=True,
        )
        is K3MoETailTier.MNNVL_BT_DEFERRED
    )


def test_join_without_a_lane_reaches_the_join_tier():
    """No TRT-LLM lane, but the join itself needs no lane.

    ``fused_moe_ar`` implies a backend-owned lane and is TRT-LLM only, so on
    every other backend it is False and the tail used to fall all the way to
    SEPARATE_REDUCE -- two collectives per MoE layer over the same group.
    ``kimi3_join_reduce_moe`` handles ``lane=None`` with a concatenated
    one-shot or a grouped all-reduce, so the join tier is reachable without
    one.
    """
    assert (
        _select(fused_moe_ar=False, join_moe_reduce=True) is K3MoETailTier.FUSED_LANE_AR
    )


def test_without_join_capability_the_fallback_is_unchanged():
    assert (
        _select(fused_moe_ar=False, join_moe_reduce=False)
        is K3MoETailTier.SEPARATE_REDUCE
    )


def test_join_does_not_preempt_the_fused_tail_or_multimem():
    # The fused decode tail still wins inside its graph-phase window ...
    assert (
        _select(
            num_tokens=8, graph_phase=True, fused_moe_ar=False, join_moe_reduce=True
        )
        is K3MoETailTier.TAIL_FUSION
    )
    # ... and an armed lane still routes through the multimem window.
    assert (
        _select(fused_moe_ar=True, join_moe_reduce=True, multimem_ok=True)
        is K3MoETailTier.MULTIMEM_AR
    )

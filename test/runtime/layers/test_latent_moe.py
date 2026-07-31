from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

from tokenspeed.runtime.layers.moe import latent as latent_module
from tokenspeed.runtime.layers.moe.latent import (
    Kimi3MoEExecutionPlan,
    LatentMoELayer,
)
from tokenspeed.runtime.layers.moe.topk import StandardTopKOutput


def _up(x: torch.Tensor) -> tuple[torch.Tensor, None]:
    return torch.cat((x, torch.zeros_like(x)), dim=-1), None


_TRACE_FNS = {
    "router": lambda x: x[:, :2].float(),
    "down": lambda x: x[:, :2],
    "norm": lambda x: x + 3,
    "up": _up,
    "shared": lambda x: x * 4,
}


class _Trace(nn.Module):
    def __init__(self, events: list[str], name: str) -> None:
        super().__init__()
        self.events, self.name = events, name

    def forward(self, hidden_states: torch.Tensor):
        self.events.append(self.name)
        return _TRACE_FNS[self.name](hidden_states)


class _TopK(nn.Module):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> StandardTopKOutput:
        self.events.append("topk")
        tokens = hidden_states.shape[0]
        weights = torch.ones(tokens, 1, device=hidden_states.device)
        ids = torch.zeros(tokens, 1, dtype=torch.int32, device=hidden_states.device)
        return StandardTopKOutput(weights, ids, router_logits)

    def empty_topk_output(
        self,
        device: torch.device,
        *,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> StandardTopKOutput:
        del hidden_states
        return StandardTopKOutput(
            torch.empty(0, 1, device=device),
            torch.empty(0, 1, dtype=torch.int32, device=device),
            router_logits,
        )


class _Experts(nn.Module):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_output: StandardTopKOutput,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
    ) -> torch.Tensor:
        self.events.append("experts")
        assert topk_output.topk_ids.shape == (hidden_states.shape[0], 1)
        assert num_global_tokens == hidden_states.shape[0]
        assert max_num_tokens_per_gpu == hidden_states.shape[0]
        return hidden_states + 1


def test_kimi3_join_reduce_moe_selects_lane_norm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lane = torch.arange(6, dtype=torch.float32).view(1, 6)
    norm = _Trace([], "norm")
    norm.weight = nn.Parameter(torch.ones(2))
    norm.variance_epsilon = 1e-6
    monkeypatch.setattr(
        latent_module,
        "all_reduce_latent_norm",
        lambda value, *_args, **_kwargs: value + 10,
    )
    monkeypatch.setattr(
        latent_module,
        "all_reduce",
        lambda *_args, **_kwargs: pytest.fail("lane norm must own the reduction"),
    )

    routed, shared = latent_module.kimi3_join_reduce_moe(
        lane[:, :2],
        lane[:, 2:],
        lane=lane,
        routed_hidden=2,
        routed_norm=norm,
        group=(0, 1),
        enable_lane_norm=True,
        max_token_num=8,
    )

    torch.testing.assert_close(routed, lane[:, :2] + 10)
    torch.testing.assert_close(shared, lane[:, 2:] + 10)


def test_kimi3_join_reduce_moe_cats_small_partials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routed_partial = torch.arange(4, dtype=torch.float32).view(2, 2)
    shared_partial = torch.arange(8, dtype=torch.float32).view(2, 4)
    norm = _Trace([], "norm")
    monkeypatch.setattr(
        latent_module,
        "all_reduce",
        lambda value, _group: value + 10,
    )
    monkeypatch.setattr(
        latent_module,
        "all_reduce_two",
        lambda *_args, **_kwargs: pytest.fail(
            "small partials must take the cat + single-reduce path"
        ),
    )

    routed, shared = latent_module.kimi3_join_reduce_moe(
        routed_partial,
        shared_partial,
        lane=None,
        routed_hidden=2,
        routed_norm=norm,
        group=(0, 1),
        enable_lane_norm=True,
        max_token_num=8,
    )

    torch.testing.assert_close(routed, routed_partial + 13)
    torch.testing.assert_close(shared, shared_partial + 10)


def test_kimi3_join_reduce_moe_grouped_reduce_for_large_partials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routed_partial = torch.arange(4, dtype=torch.float32).view(2, 2)
    shared_partial = torch.arange(8, dtype=torch.float32).view(2, 4)
    norm = _Trace([], "norm")
    monkeypatch.setattr(latent_module, "COMM_ONESHOT_MAX_BYTES", 1)
    monkeypatch.setattr(
        latent_module,
        "all_reduce_two",
        lambda first, second, group: (first + 20, second + 20),
    )
    monkeypatch.setattr(
        latent_module,
        "all_reduce",
        lambda *_args, **_kwargs: pytest.fail(
            "large partials must skip the cat + single-reduce path"
        ),
    )

    routed, shared = latent_module.kimi3_join_reduce_moe(
        routed_partial,
        shared_partial,
        lane=None,
        routed_hidden=2,
        routed_norm=norm,
        group=(0, 1),
        enable_lane_norm=True,
        max_token_num=8,
    )

    torch.testing.assert_close(routed, routed_partial + 23)
    torch.testing.assert_close(shared, shared_partial + 20)


def _layer(
    events: list[str],
    experts: nn.Module | None = None,
    **kwargs,
) -> LatentMoELayer:
    return LatentMoELayer(
        router=_Trace(events, "router"),
        topk=_TopK(events),
        routed_down_proj=_Trace(events, "down"),
        experts=experts or _Experts(events),
        routed_up_proj=_Trace(events, "up"),
        **kwargs,
    )


def test_kimi3_moe_execution_policy_is_selected_outside_model() -> None:
    ep_group = tuple(range(8))
    mapping = SimpleNamespace(
        moe=SimpleNamespace(
            tp_size=1,
            ep_size=8,
            ep_group=ep_group,
            tp_ep_size=8,
            tp_ep_group=ep_group,
        )
    )
    backend = SimpleNamespace(
        is_auto=lambda: True,
        is_flashinfer_trtllm=lambda: False,
    )

    with mock.patch.object(
        latent_module,
        "kimi3_native_moe_available",
        return_value=True,
    ):
        plan = Kimi3MoEExecutionPlan.build(
            mapping,
            backend,
            alt_stream=None,
            enforce_eager=False,
        )

    assert plan.use_native
    assert not plan.use_trtllm
    assert plan.use_precomputed_topk
    assert plan.joint_moe_reduce


def test_kimi3_moe_execution_policy_preserves_nvidia_trtllm() -> None:
    mapping = SimpleNamespace(
        moe=SimpleNamespace(
            tp_size=8,
            ep_size=1,
            ep_group=object(),
            tp_ep_size=8,
            tp_ep_group=object(),
        )
    )
    backend = SimpleNamespace(
        is_auto=lambda: True,
        is_flashinfer_trtllm=lambda: False,
    )

    with mock.patch.object(
        latent_module,
        "kimi3_native_moe_available",
        return_value=False,
    ):
        plan = Kimi3MoEExecutionPlan.build(
            mapping,
            backend,
            alt_stream=None,
            enforce_eager=False,
        )

    assert not plan.use_native
    assert plan.use_trtllm
    assert plan.use_precomputed_topk
    assert not plan.overlap_shared_experts
    assert not plan.joint_moe_reduce


def test_kimi3_moe_execution_plan_prepares_latent_fusions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = (0, 1)
    mapping = SimpleNamespace(
        moe=SimpleNamespace(
            has_tp_ep=True,
            tp_ep_group=group,
        )
    )
    plan = Kimi3MoEExecutionPlan(
        use_native=False,
        use_trtllm=True,
        overlap_shared_experts=False,
        joint_moe_reduce=False,
    )
    lane_calls = []
    norm_calls = []
    monkeypatch.setattr(
        latent_module,
        "prepare_all_reduce_lane",
        lambda actual_group, width: lane_calls.append((actual_group, width)) or True,
    )
    monkeypatch.setattr(
        latent_module,
        "prepare_all_reduce_fusion",
        lambda actual_group, width, tokens: norm_calls.append(
            (actual_group, width, tokens)
        )
        or True,
    )

    prepared = plan.prepare_latent_fusion(
        mapping,
        lane_width=10752,
        has_latent_norm=True,
        max_token_num=8,
    )

    assert prepared.fused_moe_ar
    assert prepared.lane_latent_norm_ar
    assert prepared.comm_fusion_max_num_tokens == 8
    assert lane_calls == [(group, 10752)]
    assert norm_calls == [(group, 10752, 8)]


def test_latent_moe_runtime_preserves_widths_and_reduction_order() -> None:
    events: list[str] = []

    def latent_reduce(hidden_states: torch.Tensor) -> torch.Tensor:
        events.append("latent_reduce")
        return hidden_states * 2

    def shared_reduce(hidden_states: torch.Tensor) -> torch.Tensor:
        events.append("shared_reduce")
        return hidden_states + 5

    layer = _layer(
        events,
        routed_norm=_Trace(events, "norm"),
        shared_experts=_Trace(events, "shared"),
        latent_reduce=latent_reduce,
        shared_reduce=shared_reduce,
    )
    hidden_states = torch.arange(12, dtype=torch.float32).view(3, 4)

    actual = layer(hidden_states)

    latent = (hidden_states[:, :2] + 1) * 2 + 3
    routed = torch.cat((latent, torch.zeros_like(latent)), dim=-1)
    expected = routed + hidden_states * 4 + 5
    torch.testing.assert_close(actual, expected)
    routed_events = [event for event in events if not event.startswith("shared")]
    assert routed_events == [
        "router",
        "topk",
        "down",
        "experts",
        "latent_reduce",
        "norm",
        "up",
    ]
    # The shared branch is independent and may execute before or concurrently
    # with routed work; its reduction remains ordered after the branch joins.
    assert events.index("shared") < events.index("shared_reduce")
    assert events[-1] == "shared_reduce"


def test_latent_moe_uses_injected_input_projections() -> None:
    events: list[str] = []

    def input_projections(
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        events.append("projections")
        return (
            _TRACE_FNS["router"](hidden_states),
            _TRACE_FNS["down"](hidden_states),
            _TRACE_FNS["shared"](hidden_states),
        )

    layer = _layer(
        events,
        routed_norm=_Trace(events, "norm"),
        shared_experts=_Trace(events, "shared"),
        input_projections=input_projections,
    )
    hidden_states = torch.arange(12, dtype=torch.float32).view(3, 4)

    actual = layer(hidden_states)

    latent = hidden_states[:, :2] + 1 + 3
    expected = torch.cat((latent, torch.zeros_like(latent)), dim=-1) + hidden_states * 4
    torch.testing.assert_close(actual, expected)
    # One fused projection replaces the router, routed-down, and shared modules.
    assert events.count("projections") == 1
    assert "router" not in events
    assert "down" not in events
    assert "shared" not in events


def test_latent_moe_falls_back_when_input_projections_declines() -> None:
    events: list[str] = []

    layer = _layer(
        events,
        routed_norm=_Trace(events, "norm"),
        shared_experts=_Trace(events, "shared"),
        input_projections=lambda hidden_states: None,
    )
    hidden_states = torch.arange(12, dtype=torch.float32).view(3, 4)

    layer(hidden_states)

    assert {"router", "down", "shared"} <= set(events)


def test_latent_moe_rejects_input_projections_without_shared_experts() -> None:
    with pytest.raises(ValueError, match="input_projections requires shared_experts"):
        _layer([], input_projections=lambda hidden_states: None)


def test_latent_moe_jointly_reduces_shared_and_routed_partials() -> None:
    events: list[str] = []

    def joint_reduce(
        shared: torch.Tensor,
        routed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        events.append("joint_reduce")
        return shared + 5, routed * 2

    layer = _layer(
        events,
        routed_norm=_Trace(events, "norm"),
        shared_experts=_Trace(events, "shared"),
        joint_reduce=joint_reduce,
    )
    hidden_states = torch.arange(12, dtype=torch.float32).view(3, 4)

    actual = layer(hidden_states)

    latent = (hidden_states[:, :2] + 1) * 2 + 3
    routed = torch.cat((latent, torch.zeros_like(latent)), dim=-1)
    expected = routed + hidden_states * 4 + 5
    torch.testing.assert_close(actual, expected)
    assert events.count("joint_reduce") == 1
    assert events.index("experts") < events.index("joint_reduce")
    assert events.index("joint_reduce") < events.index("norm")


def test_latent_moe_can_return_separate_residual_components() -> None:
    events: list[str] = []
    layer = _layer(
        events,
        shared_experts=_Trace(events, "shared"),
        return_separate_outputs=True,
    )
    hidden_states = torch.arange(12, dtype=torch.float32).view(3, 4)

    routed, shared = layer(hidden_states)

    latent = hidden_states[:, :2] + 1
    expected_routed = torch.cat((latent, torch.zeros_like(latent)), dim=-1)
    torch.testing.assert_close(routed, expected_routed)
    torch.testing.assert_close(shared, hidden_states * 4)


def test_latent_moe_rejects_joint_and_individual_reducers() -> None:
    events: list[str] = []

    with pytest.raises(ValueError, match="joint_reduce cannot be combined"):
        _layer(
            events,
            shared_experts=_Trace(events, "shared"),
            latent_reduce=lambda x: x,
            joint_reduce=lambda shared, routed: (shared, routed),
        )


def test_latent_moe_runtime_rejects_wrong_latent_reduction_shape() -> None:
    events: list[str] = []
    layer = _layer(
        events,
        latent_reduce=lambda x: x[:, :1],
    )

    with pytest.raises(ValueError, match="latent_reduce"):
        layer(torch.ones(2, 4))


class _EpExperts(_Experts):
    def __init__(self, events: list[str], ep_size: int, num_experts: int = 8) -> None:
        super().__init__(events)
        self.ep_size = ep_size
        self.num_experts = num_experts
        self.num_local_experts = num_experts // ep_size
        self.ep_group = None


@pytest.mark.parametrize("ep_size", [2, 4, 8])
def test_latent_moe_ep_all_reduces_before_norm(
    monkeypatch: pytest.MonkeyPatch,
    ep_size: int,
) -> None:
    events: list[str] = []
    group = tuple(range(ep_size))

    def fake_all_reduce(value: torch.Tensor, *, group: tuple[int, ...]):
        assert group == tuple(range(ep_size))
        events.append("ep_all_reduce")
        return value * ep_size

    monkeypatch.setattr(latent_module, "all_reduce", fake_all_reduce)
    layer = _layer(
        events,
        _EpExperts(events, ep_size),
        routed_norm=_Trace(events, "norm"),
        expert_parallel_group=group,
    )
    hidden_states = torch.arange(8, dtype=torch.float32).view(2, 4)

    actual = layer(hidden_states)

    latent = (hidden_states[:, :2] + 1) * ep_size + 3
    expected = torch.cat((latent, torch.zeros_like(latent)), dim=-1)
    torch.testing.assert_close(actual, expected)
    assert events.index("experts") < events.index("ep_all_reduce")
    assert events.index("ep_all_reduce") < events.index("norm")


def test_latent_moe_ep_requires_group_or_explicit_reducer() -> None:
    events: list[str] = []
    with pytest.raises(ValueError, match="expert_parallel_group"):
        _layer(events, _EpExperts(events, 2))


def test_latent_moe_infers_ep_group_from_experts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    experts = _EpExperts(events, 2)
    experts.ep_group = (2, 3)
    calls: list[tuple[int, ...]] = []

    def fake_all_reduce(value: torch.Tensor, *, group: tuple[int, ...]):
        calls.append(group)
        return value

    monkeypatch.setattr(latent_module, "all_reduce", fake_all_reduce)
    layer = _layer(events, experts)

    layer(torch.ones(2, 4))

    assert calls == [(2, 3)]


def test_latent_moe_rejects_ep_above_eight() -> None:
    events: list[str] = []
    with pytest.raises(ValueError, match="ep_size in"):
        _layer(
            events,
            _EpExperts(events, 16, num_experts=16),
            latent_reduce=lambda x: x,
        )

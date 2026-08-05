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


class _Router(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states[:, :2].float()


class _Down(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states[:, :2]


class _Norm(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + 3


class _Up(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        return _up(hidden_states)


class _Shared(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * 4


class _Add3Up(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4, 2))

    def forward_add3(
        self,
        routed_latent: torch.Tensor,
        prefix_sum: torch.Tensor,
        shared_output: torch.Tensor,
    ) -> torch.Tensor:
        routed_output, _ = _up(routed_latent)
        return prefix_sum + routed_output + shared_output


class _WeightedNorm(_Norm):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))
        self.variance_epsilon = 1e-6


class _TopK(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> StandardTopKOutput:
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
    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_output: StandardTopKOutput,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
    ) -> torch.Tensor:
        assert topk_output.topk_ids.shape == (hidden_states.shape[0], 1)
        assert num_global_tokens == hidden_states.shape[0]
        assert max_num_tokens_per_gpu == hidden_states.shape[0]
        return hidden_states + 1


def test_kimi3_join_reduce_moe_selects_lane_norm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lane = torch.arange(6, dtype=torch.float32).view(1, 6)
    norm = _Norm()
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
    norm = _Norm()
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
    norm = _Norm()
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
    experts: nn.Module | None = None,
    **kwargs,
) -> LatentMoELayer:
    routed_up_proj = kwargs.pop("routed_up_proj", _Up())
    return LatentMoELayer(
        router=_Router(),
        topk=_TopK(),
        routed_down_proj=_Down(),
        experts=experts or _Experts(),
        routed_up_proj=routed_up_proj,
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
        "native_latent_moe_available",
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
        "native_latent_moe_available",
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
    def latent_reduce(hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * 2

    def shared_reduce(hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + 5

    layer = _layer(
        routed_norm=_Norm(),
        shared_experts=_Shared(),
        latent_reduce=latent_reduce,
        shared_reduce=shared_reduce,
    )
    hidden_states = torch.arange(12, dtype=torch.float32).view(3, 4)

    actual = layer(hidden_states)

    latent = (hidden_states[:, :2] + 1) * 2 + 3
    routed = torch.cat((latent, torch.zeros_like(latent)), dim=-1)
    expected = routed + hidden_states * 4 + 5
    torch.testing.assert_close(actual, expected)


def test_latent_moe_uses_injected_input_projections() -> None:
    hidden_states = torch.arange(12, dtype=torch.float32).view(3, 4)
    input_projections = mock.Mock(
        return_value=(
            hidden_states[:, :2].float(),
            hidden_states[:, :2] + 10,
            hidden_states * 6,
        )
    )

    layer = _layer(
        routed_norm=_Norm(),
        shared_experts=_Shared(),
        input_projections=input_projections,
    )

    actual = layer(hidden_states)

    latent = hidden_states[:, :2] + 10 + 1 + 3
    expected = torch.cat((latent, torch.zeros_like(latent)), dim=-1) + hidden_states * 6
    torch.testing.assert_close(actual, expected)
    input_projections.assert_called_once_with(hidden_states)


def test_latent_moe_falls_back_when_input_projections_declines() -> None:
    input_projections = mock.Mock(return_value=None)

    layer = _layer(
        routed_norm=_Norm(),
        shared_experts=_Shared(),
        input_projections=input_projections,
    )
    hidden_states = torch.arange(12, dtype=torch.float32).view(3, 4)

    actual = layer(hidden_states)

    latent = hidden_states[:, :2] + 1 + 3
    expected = torch.cat((latent, torch.zeros_like(latent)), dim=-1) + hidden_states * 4
    torch.testing.assert_close(actual, expected)
    input_projections.assert_called_once_with(hidden_states)


def test_latent_moe_rejects_input_projections_without_shared_experts() -> None:
    with pytest.raises(ValueError, match="input_projections requires shared_experts"):
        _layer(input_projections=lambda hidden_states: None)


def test_latent_moe_jointly_reduces_shared_and_routed_partials() -> None:
    joint_reduce = mock.Mock(
        side_effect=lambda shared, routed: (shared + 5, routed * 2)
    )

    layer = _layer(
        routed_norm=_Norm(),
        shared_experts=_Shared(),
        joint_reduce=joint_reduce,
    )
    hidden_states = torch.arange(12, dtype=torch.float32).view(3, 4)

    actual = layer(hidden_states)

    latent = (hidden_states[:, :2] + 1) * 2 + 3
    routed = torch.cat((latent, torch.zeros_like(latent)), dim=-1)
    expected = routed + hidden_states * 4 + 5
    torch.testing.assert_close(actual, expected)
    assert joint_reduce.call_count == 1


def test_latent_moe_can_return_separate_residual_components() -> None:
    layer = _layer(
        shared_experts=_Shared(),
        return_separate_outputs=True,
    )
    hidden_states = torch.arange(12, dtype=torch.float32).view(3, 4)

    routed, shared = layer(hidden_states)

    latent = hidden_states[:, :2] + 1
    expected_routed = torch.cat((latent, torch.zeros_like(latent)), dim=-1)
    torch.testing.assert_close(routed, expected_routed)
    torch.testing.assert_close(shared, hidden_states * 4)


def test_latent_moe_fuses_output_projection_addends_without_norm() -> None:
    layer = _layer(
        shared_experts=_Shared(),
        routed_up_proj=_Add3Up(),
    )
    hidden_states = torch.arange(12, dtype=torch.float32).view(3, 4)
    prefix_sum = torch.full_like(hidden_states, 7)

    actual = layer(hidden_states, prefix_sum=prefix_sum)

    routed_latent = hidden_states[:, :2] + 1
    routed_output, _ = _up(routed_latent)
    expected = prefix_sum + routed_output + hidden_states * 4
    torch.testing.assert_close(actual, expected)


def test_latent_moe_fuses_norm_output_projection_addends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[torch.Tensor, ...]] = []

    def fused_norm_projection(
        routed_latent: torch.Tensor,
        norm_weight: torch.Tensor,
        projection_weight: torch.Tensor,
        prefix_sum: torch.Tensor,
        shared_output: torch.Tensor,
        *,
        eps: float,
    ) -> torch.Tensor:
        del norm_weight, projection_weight
        calls.append((routed_latent, prefix_sum, shared_output))
        assert eps == 1e-6
        routed_output, _ = _up(routed_latent + 3)
        return prefix_sum + routed_output + shared_output

    monkeypatch.setattr(
        latent_module.tokenspeed_kernel,
        "rmsnorm_linear_add",
        fused_norm_projection,
    )
    layer = _layer(
        routed_norm=_WeightedNorm(),
        shared_experts=_Shared(),
        routed_up_proj=_Add3Up(),
    )
    hidden_states = torch.arange(4, dtype=torch.float32).view(1, 4)
    prefix_sum = torch.full_like(hidden_states, 7)

    actual = layer(hidden_states, prefix_sum=prefix_sum)

    routed_latent = hidden_states[:, :2] + 4
    routed_output, _ = _up(routed_latent)
    expected = prefix_sum + routed_output + hidden_states * 4
    torch.testing.assert_close(actual, expected)
    assert len(calls) == 1


def test_latent_moe_preserves_multi_token_norm_add3_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_fused_call(*_args, **_kwargs) -> None:
        pytest.fail("single-token fused kernel called for multiple tokens")

    monkeypatch.setattr(
        latent_module.tokenspeed_kernel,
        "rmsnorm_linear_add",
        unexpected_fused_call,
    )
    layer = _layer(
        routed_norm=_WeightedNorm(),
        shared_experts=_Shared(),
        routed_up_proj=_Add3Up(),
    )
    hidden_states = torch.arange(12, dtype=torch.float32).view(3, 4)
    prefix_sum = torch.full_like(hidden_states, 7)

    actual = layer(hidden_states, prefix_sum=prefix_sum)

    routed_latent = hidden_states[:, :2] + 4
    routed_output, _ = _up(routed_latent)
    expected = prefix_sum + routed_output + hidden_states * 4
    torch.testing.assert_close(actual, expected)


def test_latent_moe_prefix_requires_shared_experts() -> None:
    with pytest.raises(ValueError, match="prefix_sum requires shared_experts"):
        _layer()(torch.ones(2, 4), prefix_sum=torch.ones(2, 4))


def test_latent_moe_rejects_joint_and_individual_reducers() -> None:
    with pytest.raises(ValueError, match="joint_reduce cannot be combined"):
        _layer(
            shared_experts=_Shared(),
            latent_reduce=lambda x: x,
            joint_reduce=lambda shared, routed: (shared, routed),
        )


def test_latent_moe_runtime_rejects_wrong_latent_reduction_shape() -> None:
    layer = _layer(
        latent_reduce=lambda x: x[:, :1],
    )

    with pytest.raises(ValueError, match="latent_reduce"):
        layer(torch.ones(2, 4))


class _EpExperts(_Experts):
    def __init__(self, ep_size: int, num_experts: int = 8) -> None:
        super().__init__()
        self.ep_size = ep_size
        self.num_experts = num_experts
        self.num_local_experts = num_experts // ep_size
        self.ep_group = None


@pytest.mark.parametrize("ep_size", [2, 4, 8])
def test_latent_moe_ep_all_reduces_before_norm(
    monkeypatch: pytest.MonkeyPatch,
    ep_size: int,
) -> None:
    group = tuple(range(ep_size))

    def fake_all_reduce(value: torch.Tensor, *, group: tuple[int, ...]):
        assert group == tuple(range(ep_size))
        return value * ep_size

    monkeypatch.setattr(latent_module, "all_reduce", fake_all_reduce)
    layer = _layer(
        _EpExperts(ep_size),
        routed_norm=_Norm(),
        expert_parallel_group=group,
    )
    hidden_states = torch.arange(8, dtype=torch.float32).view(2, 4)

    actual = layer(hidden_states)

    latent = (hidden_states[:, :2] + 1) * ep_size + 3
    expected = torch.cat((latent, torch.zeros_like(latent)), dim=-1)
    torch.testing.assert_close(actual, expected)


def test_latent_moe_ep_requires_group_or_explicit_reducer() -> None:
    with pytest.raises(ValueError, match="expert_parallel_group"):
        _layer(_EpExperts(2))


def test_latent_moe_infers_ep_group_from_experts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experts = _EpExperts(2)
    experts.ep_group = (2, 3)
    calls: list[tuple[int, ...]] = []

    def fake_all_reduce(value: torch.Tensor, *, group: tuple[int, ...]):
        calls.append(group)
        return value

    monkeypatch.setattr(latent_module, "all_reduce", fake_all_reduce)
    layer = _layer(experts)

    layer(torch.ones(2, 4))

    assert calls == [(2, 3)]


def test_latent_moe_rejects_ep_above_eight() -> None:
    with pytest.raises(ValueError, match="ep_size in"):
        _layer(
            _EpExperts(16, num_experts=16),
            latent_reduce=lambda x: x,
        )

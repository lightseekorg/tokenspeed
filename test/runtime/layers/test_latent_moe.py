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
    def forward(
        self,
        hidden_states: torch.Tensor,
        down_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        result = hidden_states * 4
        if down_out is not None:
            down_out.copy_(result)
            return down_out
        return result


class _Add3Up(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4, 2))
        self.fused_norm = False

    def forward_add3(
        self,
        routed_latent: torch.Tensor,
        prefix_sum: torch.Tensor,
        shared_output: torch.Tensor,
        *,
        norm_weight: torch.Tensor | None = None,
        eps: float | None = None,
    ) -> torch.Tensor:
        if norm_weight is not None:
            assert eps == 1e-6
            self.fused_norm = True
            routed_latent = routed_latent + 3
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
    hidden_size = 2

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
        result = hidden_states + 1
        output = getattr(self, "_situ_output_buffer", None)
        if output is not None:
            output.copy_(result)
            return output
        return result


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
        "all_reduce",
        lambda values, group: tuple(value + 20 for value in values),
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


def test_latent_expert_shared_acquires_outputs_and_reduces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hidden_states = torch.empty(2, 3)
    shared_staging = torch.empty(2, 5)
    routed_staging = torch.empty(2, 3)
    reduced_shared = torch.full_like(shared_staging, 7)
    reduced_routed = torch.full_like(routed_staging, 11)

    def fake_acquire(shapes, like, group):
        assert shapes == ((2, 5), (2, 3))
        assert like is hidden_states
        assert group == (0, 1)
        return shared_staging, routed_staging

    def fake_all_reduce(outputs, group):
        assert outputs[0] is shared_staging
        assert outputs[1] is routed_staging
        assert group == (0, 1)
        return reduced_shared, reduced_routed

    def fake_producer(*_args, routed_out, shared_out, **_kwargs):
        assert routed_out is routed_staging
        assert shared_out is shared_staging
        return routed_out, shared_out

    monkeypatch.setattr(latent_module, "acquire_all_reduce_outputs", fake_acquire)
    monkeypatch.setattr(latent_module, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(latent_module, "latent_moe_expert_shared", fake_producer)
    placeholder = torch.empty(1)

    routed, shared = latent_module.latent_moe_expert_shared_all_reduce(
        hidden_states,
        placeholder,
        placeholder,
        placeholder,
        placeholder,
        placeholder,
        placeholder,
        placeholder,
        torch.empty(5, 1),
        activation_clamp=1.0,
        linear_clamp=None,
        expert_start=0,
        w13_interleaved=False,
        group=(0, 1),
    )

    assert routed is reduced_routed
    assert shared is reduced_shared


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
        is_marlin=lambda: False,
    )

    with (
        mock.patch.object(
            latent_module,
            "native_latent_moe_available",
            return_value=False,
        ),
        # The marlin probe is a filesystem check for a locally built .so;
        # pin it so the policy assertion does not depend on build artifacts
        # present in the developer's checkout.
        mock.patch.object(
            latent_module,
            "_marlin_moe_available",
            return_value=False,
        ),
    ):
        plan = Kimi3MoEExecutionPlan.build(
            mapping,
            backend,
            alt_stream=None,
            enforce_eager=False,
        )

    assert not plan.use_native
    assert plan.use_trtllm
    assert not plan.overlap_shared_experts
    assert not plan.joint_moe_reduce


def _plan_mapping():
    return SimpleNamespace(
        moe=SimpleNamespace(
            tp_size=8,
            ep_size=1,
            ep_group=object(),
            tp_ep_size=8,
            tp_ep_group=object(),
        )
    )


def _backend(*, auto=False, trtllm=False, marlin=False):
    return SimpleNamespace(
        is_auto=lambda: auto,
        is_flashinfer_trtllm=lambda: trtllm,
        is_marlin=lambda: marlin,
    )


def test_kimi3_moe_execution_policy_honours_a_forced_marlin_backend() -> None:
    """Asked for Marlin, the plan must say Marlin rather than nothing at all."""
    with mock.patch.object(
        latent_module, "native_latent_moe_available", return_value=False
    ):
        plan = Kimi3MoEExecutionPlan.build(
            _plan_mapping(), _backend(marlin=True), alt_stream=None, enforce_eager=False
        )

    assert plan.use_marlin
    assert not plan.use_trtllm
    assert not plan.use_native


def test_kimi3_moe_execution_policy_takes_marlin_when_its_probe_says_yes() -> None:
    """The probe decides AUTO, and the contraction shard's gate rides on it."""
    with (
        mock.patch.object(
            latent_module, "native_latent_moe_available", return_value=False
        ),
        mock.patch.object(latent_module, "_marlin_moe_available", return_value=True),
    ):
        plan = Kimi3MoEExecutionPlan.build(
            _plan_mapping(), _backend(auto=True), alt_stream=None, enforce_eager=False
        )

    assert plan.use_marlin
    assert not plan.use_trtllm


def test_the_marlin_probe_needs_both_an_arch_and_a_built_library() -> None:
    """Neither term alone answers; the shipped library is a filesystem fact."""
    from tokenspeed_kernel.platform import ArchVersion

    for arch, built, expected in (
        (ArchVersion(10, 0), True, True),
        (ArchVersion(8, 0), True, False),
        (ArchVersion(10, 0), False, False),
    ):
        with (
            mock.patch(
                "tokenspeed_kernel.platform.current_platform",
                return_value=SimpleNamespace(is_nvidia=True, arch_version=arch),
            ),
            mock.patch(
                "tokenspeed_kernel.thirdparty.cuda.marlin_moe.is_marlin_moe_available",
                return_value=built,
            ),
        ):
            assert latent_module._marlin_moe_available() is expected


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
        lambda actual_group, width, tokens: (
            norm_calls.append((actual_group, width, tokens)) or True
        ),
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
    input_projections.assert_called_once_with(hidden_states, None)


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
    input_projections.assert_called_once_with(hidden_states, None)


def test_latent_moe_rejects_input_projections_without_shared_experts() -> None:
    with pytest.raises(ValueError, match="input_projections requires shared_experts"):
        _layer(input_projections=lambda hidden_states, shared_out: None)


def test_latent_moe_acquires_shared_and_routed_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acquisitions = []

    def acquire_outputs(shapes, like, group):
        assert group == (0,)
        outputs = tuple(like.new_empty(shape) for shape in shapes)
        acquisitions.append(outputs)
        return outputs

    def reduce_outputs(outputs, group):
        assert outputs is acquisitions[0]
        assert group == (0,)
        return outputs[0] + 5, outputs[1] * 2

    monkeypatch.setattr(latent_module, "acquire_all_reduce_outputs", acquire_outputs)
    monkeypatch.setattr(latent_module, "all_reduce", reduce_outputs)

    layer = _layer(
        routed_norm=_Norm(),
        shared_experts=_Shared(),
        joint_reduce=True,
        expert_parallel_group=(0,),
    )
    hidden_states = torch.arange(12, dtype=torch.float32).view(3, 4)

    actual = layer(hidden_states)

    latent = (hidden_states[:, :2] + 1) * 2 + 3
    routed = torch.cat((latent, torch.zeros_like(latent)), dim=-1)
    expected = routed + hidden_states * 4 + 5
    torch.testing.assert_close(actual, expected)
    assert len(acquisitions) == 1


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
    monkeypatch.setattr(
        latent_module,
        "current_platform",
        lambda: SimpleNamespace(is_cdna4=True),
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
    assert layer.routed_up_proj.fused_norm


@pytest.mark.parametrize(("tokens", "is_cdna4"), [(1, False), (3, True)])
def test_latent_moe_preserves_unfused_norm_add3_path(
    monkeypatch: pytest.MonkeyPatch,
    tokens: int,
    is_cdna4: bool,
) -> None:
    monkeypatch.setattr(
        latent_module,
        "current_platform",
        lambda: SimpleNamespace(is_cdna4=is_cdna4),
    )
    layer = _layer(
        routed_norm=_WeightedNorm(),
        shared_experts=_Shared(),
        routed_up_proj=_Add3Up(),
    )
    hidden_states = torch.arange(tokens * 4, dtype=torch.float32).view(tokens, 4)
    prefix_sum = torch.full_like(hidden_states, 7)

    actual = layer(hidden_states, prefix_sum=prefix_sum)

    routed_latent = hidden_states[:, :2] + 4
    routed_output, _ = _up(routed_latent)
    expected = prefix_sum + routed_output + hidden_states * 4
    torch.testing.assert_close(actual, expected)
    assert not layer.routed_up_proj.fused_norm


def test_latent_moe_prefix_requires_shared_experts() -> None:
    with pytest.raises(ValueError, match="prefix_sum requires shared_experts"):
        _layer()(torch.ones(2, 4), prefix_sum=torch.ones(2, 4))


def test_latent_moe_rejects_joint_and_individual_reducers() -> None:
    with pytest.raises(ValueError, match="joint_reduce cannot be combined"):
        _layer(
            shared_experts=_Shared(),
            latent_reduce=lambda x: x,
            joint_reduce=True,
            expert_parallel_group=(0,),
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


def _shard_projection(monkeypatch, world: int, rank: int, out: int, k: int):
    """A CPU Kimi3LatentProjection sharded across a fake ``world``-rank group.

    The kernel projection and the gather collective are replaced with
    reference implementations so the shard bookkeeping (loader slice, column
    offsets, gather layout) is what the test exercises.
    """
    monkeypatch.setattr(
        latent_module.tokenspeed_kernel,
        "kimi3_latent_projection",
        lambda x, w, solution=None: x @ w.T,
        raising=False,
    )
    monkeypatch.setattr(
        latent_module.tokenspeed_kernel,
        "kimi3_latent_projection_add3",
        lambda x, w, a, c, norm_weight=None, eps=None: a + x @ w.T + c,
        raising=False,
    )
    return latent_module.Kimi3LatentProjection(
        k,
        out,
        params_dtype=torch.float32,
        shard_group=tuple(range(world)),
        shard_rank=rank,
        shard_size=world,
    )


def test_kimi3_latent_projection_shard_loader_takes_row_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world, out, k = 4, 16, 8
    full = torch.arange(out * k, dtype=torch.float32).view(out, k)
    for rank in range(world):
        proj = _shard_projection(monkeypatch, world, rank, out, k)
        proj.weight_loader(proj.weight, full)
        rows = out // world
        assert proj.weight.shape == (rows, k)
        torch.testing.assert_close(
            proj.weight.data, full[rank * rows : (rank + 1) * rows]
        )
        assert proj.shard_slice == (rank * rows, rows)
        torch.testing.assert_close(proj.project_shard(torch.eye(k)), proj.weight.data.T)


def test_kimi3_latent_projection_shard_forward_add3_matches_replicated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world, out, k, m = 4, 16, 8, 3
    torch.manual_seed(0)
    full = torch.randn(out, k)
    x = torch.randn(m, k)
    a = torch.randn(m, out)
    c = torch.randn(m, out)
    expected = a + x @ full.T + c

    rows = out // world

    def stripe(r):
        return (
            a[:, r * rows : (r + 1) * rows]
            + x @ full[r * rows : (r + 1) * rows].T
            + c[:, r * rows : (r + 1) * rows]
        )

    for rank in range(world):

        def gather_all(output, local, group, _rank=rank):
            # The mock must first CHECK this rank's local block: it is the
            # narrow-offset shard math under test, and a gather that ignores
            # it would pass even with the offsets broken.
            torch.testing.assert_close(local, stripe(_rank))
            for r in range(world):
                output[r] = stripe(r)

        monkeypatch.setattr(latent_module, "all_gather_into_tensor", gather_all)
        # The up projection must not follow the column band onto the backend's
        # own gather; that path is unmeasured for this width and rendezvouses
        # a second workspace.
        monkeypatch.setattr(
            latent_module,
            "all_gather",
            lambda *a, **kw: pytest.fail("the narrowed shard keeps the buffer gather"),
        )
        proj = _shard_projection(monkeypatch, world, rank, out, k)
        proj.weight_loader(proj.weight, full)
        got = proj.forward_add3(x, a, c)
        torch.testing.assert_close(got, expected)


def _plan_for_join(shard_up_projection: bool) -> Kimi3MoEExecutionPlan:
    """prepare_latent_fusion on a native (lane-less) TP x EP layout."""
    mapping = SimpleNamespace(
        moe=SimpleNamespace(has_tp_ep=True, tp_ep_group=(0, 1)),
    )
    plan = Kimi3MoEExecutionPlan(
        use_native=True,
        use_trtllm=False,
        overlap_shared_experts=False,
        joint_moe_reduce=False,
    )
    return plan.prepare_latent_fusion(
        mapping,
        lane_width=10752,
        has_latent_norm=False,
        max_token_num=8,
        shard_up_projection=shard_up_projection,
    )


def test_join_is_available_without_a_trtllm_lane() -> None:
    """No lane, but the partials can still be reduced together.

    prepare_all_reduce_lane is only implemented by the TRT-LLM backend, so a
    native layout never arms fused_moe_ar. kimi3_join_reduce_moe handles
    lane=None with a concatenated one-shot or a grouped all-reduce, so the
    join is still available and the tail issues one collective instead of two.
    """
    prepared = _plan_for_join(shard_up_projection=False)
    assert not prepared.fused_moe_ar
    assert prepared.join_moe_reduce


def test_sharded_up_projection_does_not_advertise_the_join() -> None:
    """A sharded up projection cannot use the join, so it must not claim it.

    _tail_fused_lane_ar_sharded folds the projection between two sequential
    all-reduces rather than calling kimi3_join_reduce_moe: selecting the join
    tier there would save no collective while also dropping routed_in_fork,
    which overlaps the routed reduction with the shared branch.
    """
    prepared = _plan_for_join(shard_up_projection=True)
    assert not prepared.join_moe_reduce


def test_kimi3_latent_projection_rejects_out_of_range_shard_rank() -> None:
    with pytest.raises(ValueError, match="outside the shard size"):
        latent_module.Kimi3LatentProjection(8, 16, shard_rank=2, shard_size=2)


def test_kimi3_latent_projection_rejects_group_size_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match the shard size"):
        latent_module.Kimi3LatentProjection(
            8, 16, shard_group=(0, 1), shard_rank=0, shard_size=4
        )


def test_kimi3_latent_projection_shard_slice_needs_column_parallel() -> None:
    """A projection with no column group has no block to report."""
    proj = latent_module.Kimi3LatentProjection(8, 16, shard_rank=1, shard_size=2)
    with pytest.raises(ValueError, match="column-parallel"):
        proj.shard_slice


def test_kimi3_latent_projection_replicated_forward_never_reduces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A projection with no group must not reduce anything."""
    out, k = 16, 64
    seen: list[tuple[torch.Tensor, object]] = []
    monkeypatch.setattr(
        latent_module.tokenspeed_kernel,
        "kimi3_latent_projection",
        lambda x, w, solution=None: x @ w.T,
        raising=False,
    )
    monkeypatch.setattr(
        latent_module,
        "all_reduce",
        lambda t, group: seen.append((t, group)),
        raising=False,
    )
    torch.manual_seed(1234)
    weight = torch.randn(out, k)
    proj = latent_module.Kimi3LatentProjection(k, out, params_dtype=torch.float32)
    proj.weight_loader(proj.weight, weight)
    hidden = torch.randn(16, k)
    output, _ = proj(hidden)
    assert seen == []
    torch.testing.assert_close(output, hidden @ weight.T)


def test_kimi3_latent_projection_column_parallel_forward_never_reduces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Column-parallel instances must keep gathering, not start reducing."""
    world, out, k = 4, 16, 64
    seen: list[tuple[torch.Tensor, object]] = []
    gathered: list[torch.Tensor] = []
    proj = _shard_projection(monkeypatch, world, 1, out, k)
    monkeypatch.setattr(
        latent_module,
        "all_reduce",
        lambda t, group: seen.append((t, group)),
        raising=False,
    )
    monkeypatch.setattr(
        latent_module.Kimi3LatentProjection,
        "_gather_shards",
        lambda self, local: gathered.append(local) or local,
        raising=False,
    )
    torch.manual_seed(1234)
    proj.weight_loader(proj.weight, torch.randn(out, k))
    proj(torch.randn(16, k))
    assert seen == []
    assert len(gathered) == 1


def test_kimi3_latent_projection_without_a_group_has_no_shard_size() -> None:
    """An ungrouped projection must not carry a shard width it cannot use."""
    proj = latent_module.Kimi3LatentProjection(64, 16, shard_rank=0, shard_size=8)
    assert proj.shard_size == 1
    assert proj.input_size == 64


class _FakeMulticastDown:
    """A multicast op that records what the projection dispatched to it."""

    def __init__(
        self, rank: int, shard_dim: int, latent: int, calls: list, max_m: int = 8
    ) -> None:
        self.rank = rank
        self.shard_dim = shard_dim
        self._latent = latent
        self._calls = calls
        self.max_m = max_m

    def handles(self, num_tokens: int) -> bool:
        return 1 <= num_tokens <= self.max_m

    def __call__(self, hidden_states, block):
        # The caller hands over this rank's rows; the op does not carve them.
        self._calls.append((hidden_states.shape[0], tuple(block.shape)))
        return hidden_states @ block.T


def _multicast_projection(monkeypatch, world, rank, out, k, calls, max_m=8):
    monkeypatch.setattr(
        latent_module.tokenspeed_kernel,
        "kimi3_latent_projection",
        lambda x, w, solution=None: x @ w.T,
        raising=False,
    )
    monkeypatch.setattr(
        latent_module,
        "all_reduce",
        lambda t, group: pytest.fail("the multicast path must not reduce"),
        raising=False,
    )
    return latent_module.Kimi3LatentProjection(
        k,
        out,
        params_dtype=torch.float32,
        shard_rank=rank,
        shard_size=world,
        multicast_down=_FakeMulticastDown(rank, out // world, out, calls, max_m),
    )


def test_the_multicast_path_needs_no_group_of_its_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The op carries its own group, so the projection needs none wired in."""
    out, k = 16, 64
    monkeypatch.setattr(
        latent_module.tokenspeed_kernel,
        "kimi3_latent_projection",
        lambda x, w, solution=None: pytest.fail("the multicast op must be dispatched"),
        raising=False,
    )
    calls: list = []
    proj = latent_module.Kimi3LatentProjection(
        k,
        out,
        params_dtype=torch.float32,
        shard_rank=0,
        shard_size=4,
        multicast_down=_FakeMulticastDown(0, out // 4, out, calls),
    )
    proj(torch.zeros(1, k))
    assert calls == [(1, (out // 4, k))]


@pytest.mark.parametrize("tokens", [1, 8, 9, 512])
def test_kimi3_latent_projection_claimed_batches_take_the_multicast_path(
    monkeypatch: pytest.MonkeyPatch, tokens: int
) -> None:
    """Every rank's block must be dispatched and must sum to the full latent.

    The widths past eight are the ones the fused kernel cannot compile, and
    they reach the op through the same dispatch, so the projection must not
    treat them as a different route.
    """
    world, out, k = 4, 16, 64
    torch.manual_seed(1234)
    weight = torch.randn(out, k)
    hidden = torch.randn(tokens, k)
    blocks, calls = [], []
    for rank in range(world):
        proj = _multicast_projection(monkeypatch, world, rank, out, k, calls, 512)
        proj.weight_loader(proj.weight, weight)
        blocks.append(proj(hidden)[0])
    assert calls == [(tokens, (out // world, k))] * world
    torch.testing.assert_close(torch.cat(blocks, dim=1), hidden @ weight.T)


def test_kimi3_latent_projection_hands_wide_batches_back_to_the_replicated_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Past the width the op claims there is nothing left but the full weight."""
    world, out, k = 4, 16, 64
    calls: list = []
    replicated: list = []
    torch.manual_seed(1234)
    weight = torch.randn(out, k)
    proj = _multicast_projection(monkeypatch, world, 0, out, k, calls, 8)
    proj.weight_loader(proj.weight, weight)
    # After the fixture, whose own stub would otherwise shadow this one.
    monkeypatch.setattr(
        latent_module.tokenspeed_kernel,
        "kimi3_latent_projection",
        lambda x, w, solution=None: replicated.append(x.shape[0]) or x @ w.T,
        raising=False,
    )
    hidden = torch.randn(9, k)
    output, _ = proj(hidden)
    assert calls == []
    assert replicated == [9]
    torch.testing.assert_close(output, hidden @ weight.T)


def test_kimi3_latent_projection_rejects_a_multicast_op_that_disagrees() -> None:
    """The op's rank and block must match the projection it is attached to."""
    with pytest.raises(ValueError, match="does not match the projection"):
        latent_module.Kimi3LatentProjection(
            64,
            16,
            shard_rank=0,
            shard_size=2,
            multicast_down=_FakeMulticastDown(1, 8, 16, []),
        )
    with pytest.raises(ValueError, match="does not cover"):
        latent_module.Kimi3LatentProjection(
            64,
            16,
            shard_rank=0,
            shard_size=2,
            multicast_down=_FakeMulticastDown(0, 4, 16, []),
        )
    with pytest.raises(ValueError, match="cannot also multicast"):
        latent_module.Kimi3LatentProjection(
            64,
            16,
            shard_group=(0, 1),
            shard_rank=0,
            shard_size=2,
            multicast_down=_FakeMulticastDown(0, 8, 16, []),
        )


def _column_projection(monkeypatch, world, rank, out, k, calls=None, max_m=0):
    """A CPU projection that column-splits over replicated storage.

    ``max_m`` above zero also attaches a multicast op, so the three-way
    dispatch can be exercised on one object.
    """
    monkeypatch.setattr(
        latent_module.tokenspeed_kernel,
        "kimi3_latent_projection",
        lambda x, w, solution=None: x @ w.T,
        raising=False,
    )
    monkeypatch.setattr(
        latent_module,
        "all_reduce",
        lambda t, group: pytest.fail("a column shard must gather, not reduce"),
        raising=False,
    )
    return latent_module.Kimi3LatentProjection(
        k,
        out,
        params_dtype=torch.float32,
        column_group=tuple(range(world)),
        shard_rank=rank,
        shard_size=world,
        multicast_down=(
            _FakeMulticastDown(rank, out // world, out, calls, max_m) if max_m else None
        ),
    )


def test_column_group_keeps_the_weight_full_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a mailbox the replica is reachable, so every row is kept.

    This is the configuration that cannot narrow: the shard loses badly to the
    replica at decode widths, so the replica has to stay available, and it
    needs the full width to run. Narrowing is pinned in the mailboxed case.
    """
    world, out, k = 4, 16, 8
    full = torch.arange(out * k, dtype=torch.float32).view(out, k)
    rows = out // world
    for rank in range(world):
        proj = _column_projection(monkeypatch, world, rank, out, k)
        proj.weight_loader(proj.weight, full)
        assert proj.weight.shape == (out, k)
        torch.testing.assert_close(proj.weight.data, full)
        assert proj.shard_slice == (rank * rows, rows)


def _gather_recorder(monkeypatch, world, gathers):
    monkeypatch.setattr(
        latent_module,
        "all_gather",
        lambda local, group, dim=-1: (
            gathers.append(local.shape[0]),
            torch.cat([local] * world, dim=-1),
        )[-1],
    )


def test_a_narrowed_column_projection_never_reaches_the_replica(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a mailbox there is no full width left, so there is no third arm.

    Narrowed storage and an unreachable replica are the same fact, so every
    width the mailbox declines has to go to the shard -- including zero, which
    is below the gate and would otherwise return a block-wide empty tensor.
    """
    world, out, k = 4, 16, 8
    crossover = latent_module.DOWN_MAILBOX_MAX_TOKENS
    torch.manual_seed(0)
    full = torch.randn(out, k)
    calls: list = []
    gathers: list[int] = []
    _gather_recorder(monkeypatch, world, gathers)
    proj = _column_projection(monkeypatch, world, 1, out, k, calls, max_m=8)
    proj.weight_loader(proj.weight, full)
    assert proj.narrowed and proj.weight.shape == (out // world, k)

    proj(torch.randn(8, k))
    assert [c[0] for c in calls] == [8] and gathers == []
    for width in (9, crossover - 1, crossover, 0):
        got, _ = proj(torch.randn(width, k))
        assert got.shape == (width, out)
    # Every width past the mailbox took the shard; none took the replica.
    assert [c[0] for c in calls] == [8]
    assert gathers == [9, crossover - 1, crossover, 0]


def test_without_a_mailbox_no_width_gathers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The column split is taken only where the mailbox is, at every width.

    Without the fabric the gather falls back to a buffer-and-permute over
    NCCL, which under graph replay costs 254us at a full prefill chunk
    against the replicated projection's 226us -- so the split is a loss
    everywhere on such a machine, not only below some crossover. The old
    dispatch gathered above 1280 because that width was measured on hardware
    that has the fabric.
    """
    world, out, k = 4, 16, 8
    torch.manual_seed(0)
    full = torch.randn(out, k)
    monkeypatch.setattr(
        latent_module,
        "all_gather",
        lambda *a, **kw: pytest.fail("no gather without a mailbox"),
    )
    proj = _column_projection(monkeypatch, world, 2, out, k)
    proj.weight_loader(proj.weight, full)
    assert not proj.narrowed and proj.weight.shape == (out, k)

    ceiling = latent_module.DOWN_MAILBOX_MAX_TOKENS
    for tokens in (9, ceiling - 1, ceiling, ceiling + 1):
        x = torch.randn(tokens, k)
        got, _ = proj(x)
        assert got.shape == (tokens, out)
        torch.testing.assert_close(got, x @ full.T)


def test_column_group_and_shard_group_are_exclusive() -> None:
    """One means replicated storage and one means narrowed; not both."""
    with pytest.raises(ValueError, match="already gathers over its own group"):
        latent_module.Kimi3LatentProjection(
            8, 16, shard_group=(0, 1), column_group=(0, 1), shard_rank=0, shard_size=2
        )


def test_column_group_is_checked_against_the_shard_size() -> None:
    with pytest.raises(ValueError, match="does not match the shard size"):
        latent_module.Kimi3LatentProjection(
            8, 16, column_group=(0, 1), shard_rank=0, shard_size=4
        )
    with pytest.raises(ValueError, match="not divisible by the shard size"):
        latent_module.Kimi3LatentProjection(
            8, 12, column_group=(0, 1, 0, 1, 0, 1, 0, 1), shard_rank=0, shard_size=8
        )


def test_forward_add3_on_a_narrowed_column_projection_returns_full_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`shard_group is None` stopped meaning full width when narrowing spread.

    A narrowed column projection has 1/tp of the rows, so the branch that
    projects `self.weight` as if it were the whole thing would return a block
    and call it the output. Nothing calls this on the down projection today,
    which is exactly why it would have gone unnoticed.
    """
    world, out, k, m = 4, 16, 8, 3
    torch.manual_seed(0)
    full = torch.randn(out, k)
    x, a, c = torch.randn(m, k), torch.randn(m, out), torch.randn(m, out)
    rows = out // world
    rank = 2

    def stripe(r):
        return (
            a[:, r * rows : (r + 1) * rows]
            + x @ full[r * rows : (r + 1) * rows].T
            + c[:, r * rows : (r + 1) * rows]
        )

    def gather_all(local, group, dim=-1):
        torch.testing.assert_close(local, stripe(rank))
        return torch.cat([stripe(r) for r in range(world)], dim=-1)

    monkeypatch.setattr(latent_module, "all_gather", gather_all)
    proj = _column_projection(monkeypatch, world, rank, out, k, [], max_m=8)
    proj.weight_loader(proj.weight, full)
    assert proj.narrowed and proj.weight.shape == (rows, k)
    got = proj.forward_add3(x, a, c)
    assert got.shape == (m, out)
    torch.testing.assert_close(got, a + x @ full.T + c)

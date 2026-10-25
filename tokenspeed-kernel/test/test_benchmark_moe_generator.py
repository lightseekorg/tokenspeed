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

from __future__ import annotations

import importlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import tokenspeed_kernel.benchmark.generators.moe as moe_generator
import torch
from tokenspeed_kernel.benchmark.generators.moe import prepare_moe_apply
from tokenspeed_kernel.benchmark.harness import (
    BenchmarkCaseError,
    BenchmarkRequest,
    BenchmarkStatus,
)
from tokenspeed_kernel.ops import moe as moe_ops
from tokenspeed_kernel.platform import Platform, PlatformInfo
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec
from tokenspeed_kernel.selection import (
    SelectedKernel,
    kernel_override,
    resolve_kernel_override,
)


def _use_cpu_allocations(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("empty", "full", "zeros", "ones"):
        allocation = getattr(torch, name)

        def cpu_allocation(*args, _allocation=allocation, **kwargs):
            kwargs.pop("device", None)
            return _allocation(*args, **kwargs)

        monkeypatch.setattr(torch, name, cpu_allocation)


def _beam_request(**overrides) -> BenchmarkRequest:
    fragment = (
        Path(__file__).resolve().parents[1] / "benchmarks/amd/gfx950/beam/moe.json"
    )
    parameters = json.loads(fragment.read_text())["common_parameters"]
    # Keep the same TP4/grouped-route contract with small real tensors.
    parameters.update(
        tokens=2,
        hidden_size=128,
        intermediate_size=256,
        num_experts=16,
        num_local_experts=16,
    )
    parameters.update(overrides)
    return BenchmarkRequest(
        family="moe",
        mode="apply",
        parameters=parameters,
        solution=None,
        registration=None,
        cold_cache=True,
        seed=42,
    )


def test_beam_channel_full_weight_shapes_without_allocating_payload():
    assert moe_generator._fp8_channel_weight_shapes(
        num_local_experts=256,
        hidden_size=5120,
        intermediate_size_per_partition=640,
    ) == {
        "w13": (256, 1280, 5120),
        "w13_scale": (256, 1280),
        "w2": (256, 5120, 640),
        "w2_scale": (256, 5120),
    }


def test_channel_weights_are_real_seeded_per_output_channel_gate_then_up():
    def build():
        return moe_generator._make_fp8_channel_weights(
            num_local_experts=3,
            hidden_size=128,
            intermediate_size_per_partition=32,
            ep_rank=0,
            ep_size=1,
            generator=torch.Generator(device="cpu").manual_seed(7),
        )

    first, repeat = build(), build()
    generator = torch.Generator(device="cpu").manual_seed(7)
    for expert in range(3):
        for name, shape in (("w13", (64, 128)), ("w2", (128, 32))):
            raw = torch.randn(shape, generator=generator) * 0.02
            scale = (raw.abs().amax(-1) / 448).clamp(min=2.0**-126)
            expected = (raw / scale[:, None]).to(torch.float8_e4m3fn)
            codes = getattr(first, f"{name}_weight")
            scales = getattr(first, f"{name}_weight_scale")
            assert codes.dtype == torch.float8_e4m3fn and scales.dtype == torch.float32
            assert codes.is_contiguous() and scales.is_contiguous()
            assert torch.isfinite(codes.float()).all() and torch.count_nonzero(
                codes.float()
            )
            assert (scales > 0).all() and torch.isfinite(scales).all()
            assert torch.equal(
                codes[expert].view(torch.uint8), expected.view(torch.uint8)
            )
            assert torch.equal(scales[expert], scale)
            assert torch.equal(
                codes.view(torch.uint8),
                getattr(repeat, f"{name}_weight").view(torch.uint8),
            )
    assert first.w13_input_layout == "concatenated"
    assert (first.ep_rank, first.ep_size, first.num_local_experts) == (0, 1, 3)


@pytest.mark.parametrize("experts", [16, 32, 64])
def test_beam_grouped_routes_use_biased_selection_but_unbiased_fp32_weights(
    monkeypatch,
    experts,
):
    # Nonzero bias distinguishes selection from weights without dominating
    # the group-score differences that the truncation checks exercise.
    bias = torch.linspace(-0.04, 0.04, experts, dtype=torch.float32)
    monkeypatch.setattr(
        moe_generator, "_correction_bias", lambda experts, *, device: bias
    )
    logits, weights, ids = moe_generator._grouped_routing_tensors(
        tokens=3,
        experts=experts,
        topk=8,
        num_groups=8,
        group_topk=2,
        topk_groups=4,
        routed_scaling_factor=2.5,
        normalize_topk_weights=True,
        generator=torch.Generator(device="cpu").manual_seed(13),
    )
    expected_ids = []
    scores = logits.sigmoid()
    choices = scores + bias
    group_width = experts // 8
    distinguishes_group_sum = False
    for row in choices:
        group_scores = [
            sum(
                sorted(
                    row[group * group_width : (group + 1) * group_width].tolist(),
                    reverse=True,
                )[:2]
            )
            for group in range(8)
        ]
        groups = sorted(range(8), key=lambda group: group_scores[group], reverse=True)[
            :4
        ]
        all_member_scores = [
            sum(row[group * group_width : (group + 1) * group_width].tolist())
            for group in range(8)
        ]
        wrong_groups = sorted(
            range(8), key=lambda group: all_member_scores[group], reverse=True
        )[:4]
        distinguishes_group_sum |= set(groups) != set(wrong_groups)
        selected = sorted(
            (
                expert
                for group in groups
                for expert in range(group * group_width, (group + 1) * group_width)
            ),
            key=lambda expert: float(row[expert]),
            reverse=True,
        )[:8]
        expected_ids.append(selected)
    if experts > 16:
        assert 4 * group_width > 8  # Final expert selection also truncates.
        assert distinguishes_group_sum
    expected_ids = torch.tensor(expected_ids, dtype=torch.int32)
    gathered = scores.gather(1, expected_ids.long())
    expected_weights = gathered / (gathered.sum(-1, keepdim=True) + 1e-20) * 2.5
    assert torch.equal(ids, expected_ids)
    torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)
    assert weights.dtype == torch.float32 and ids.dtype == torch.int32
    assert logits.dtype == torch.float32
    assert torch.all((ids >= 0) & (ids < experts))
    assert all(len(set(row)) == 8 for row in ids.tolist())
    torch.testing.assert_close(weights.sum(-1), torch.full((3,), 2.5))
    wrong = choices.gather(1, ids.long())
    wrong = wrong / wrong.sum(-1, keepdim=True) * 2.5
    assert not torch.allclose(weights, wrong)


def test_beam_channel_ordinary_admission_without_a_timing_override(mi350_platform):
    assert resolve_kernel_override("moe", "apply", None) is None
    moe_generator.load_builtin_kernels()
    original = Platform.get()
    try:
        Platform.override(mi350_platform)
        KernelRegistry.get().clear_cache()
        plan = moe_ops.moe_plan(
            "fp8_channel",
            input_dtype=torch.bfloat16,
            activation="silu",
            requires_deferred_finalize=False,
            routing_mode="precomputed_topk",
            a2a_backend="none",
            ep_size=1,
            ispp=640,
            hidden=5120,
            num_experts=256,
            top_k=8,
            swiglu_form=None,
            activation_clamped=False,
            expert_id_repeats=False,
            fp8_scale_block_shape=None,
            internal_activation_dtype="input",
            with_bias=False,
            fast_math=True,
            combine_order="rank",
            features=frozenset(),
            solution=None,
        )
    finally:
        Platform.override(original)
        KernelRegistry.get().clear_cache()
    assert plan["apply_kernel_name"] == "gluon_fp8_channel_bf16_precomputed_moe_apply"
    assert plan["weight_preprocessor"] is None
    assert plan["supports_precomputed_topk"] and not plan["supports_deferred_finalize"]


def test_beam_preparation_keeps_routing_and_quantization_outside_the_wrong_boundary(
    monkeypatch, mi350_platform
):
    assert resolve_kernel_override("moe", "apply", None) is None
    # The preparation API receives platform metadata, but public MoE planning
    # also consults the global platform. Isolate both selection and its cache.
    monkeypatch.setattr(Platform, "_instance", mi350_platform)
    monkeypatch.setattr(KernelRegistry.get(), "_selection_cache", {})
    monkeypatch.setattr(
        moe_generator,
        "_generator",
        lambda seed: torch.Generator(device="cpu").manual_seed(seed),
    )
    monkeypatch.setattr(
        moe_generator,
        "_randn",
        lambda shape, *, generator, dtype: torch.randn(
            shape, generator=generator, dtype=dtype
        ),
    )
    selected_call = SelectedKernel.__call__
    leaf_calls = []

    def cpu_leaf(selected, *args, **kwargs):
        if selected.name == "triton_fp8_channel_fused_precomputed_moe_apply":
            leaf_calls.append(kwargs)
            return kwargs["x"]
        return selected_call(selected, *args, **kwargs)

    # Only unavailable GPU leaf execution is replaced; public moe_apply,
    # planning, preprocessing and public grouped routing remain real.
    monkeypatch.setattr(SelectedKernel, "__call__", cpu_leaf)
    prepared = prepare_moe_apply(_beam_request(), mi350_platform)
    assert not leaf_calls
    first, second = prepared.invocation.invoke(), prepared.invocation.invoke()
    assert first is second and first.dtype == torch.bfloat16
    assert len(leaf_calls) == 2
    for call in leaf_calls:
        assert call["x"] is first and call["x"].shape == (2, 128)
        assert call["router_logits"].dtype == torch.float32
        assert call["topk_weights"].dtype == torch.float32
        assert call["topk_ids"].dtype == torch.int32
        assert call["w"].w13_weight.dtype == torch.float8_e4m3fn
        assert call["w"].w13_weight_scale.dtype == torch.float32
        assert call["do_finalize"] is True
        assert call["num_tokens_global"] == call["max_num_tokens_per_gpu"] == 2
    assert leaf_calls[0]["topk_ids"] is leaf_calls[1]["topk_ids"]
    assert leaf_calls[0]["topk_weights"] is leaf_calls[1]["topk_weights"]
    assert prepared.validation is None
    assert (
        prepared.parameters["selected_apply_registration"] == prepared.registration.name
    )
    assert prepared.parameters["apply_override"] is None
    assert sum(prepared.parameters["routes_per_expert"]) == 16
    assert prepared.parameters["weight_seed"] == 42 ^ 0x4245414D
    assert (
        prepared.parameters["weight_construction"]
        == "per-output-channel-normal-0.02-v1"
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"activation": "swiglu", "swiglu_limit": 10.0},
        {"internal_activation_dtype": "mxfp4"},
        {"router_logits_dtype": "bfloat16"},
        {"route_distribution": "router"},
        {"route_scope": "local"},
        {"fp8_scale_block_shape": [128, 128]},
        {"tp_size": 3},
        {"ep_rank": 1},
        {"num_groups": 0},
        {"num_groups": 3},
        {"group_topk": 3},
        {"topk_groups": 9},
        {"route_weights_dtype": "bfloat16"},
    ],
)
def test_beam_channel_rejects_unimplemented_contract_before_allocating(
    updates, mi350_platform
):
    with pytest.raises(BenchmarkCaseError):
        prepare_moe_apply(_beam_request(**updates), mi350_platform)


@pytest.mark.parametrize(
    "options",
    [
        {"solution": "triton"},
        {"registration": "triton_fp8_channel_fused_precomputed_moe_apply"},
    ],
)
def test_beam_still_refuses_request_level_exact_selection(options, mi350_platform):
    with pytest.raises(BenchmarkCaseError, match="normal kernel selection"):
        prepare_moe_apply(replace(_beam_request(), **options), mi350_platform)


def test_beam_still_reports_generator_validation_unavailable(mi350_platform):
    with pytest.raises(
        BenchmarkCaseError, match="correctness validation is not implemented"
    ):
        prepare_moe_apply(_beam_request(validation={"runs": 1}), mi350_platform)


@pytest.mark.parametrize(
    "operation", ["grouped_route", "router_select", "route_epilogue", "plan"]
)
def test_beam_rejects_unexplained_preparation_overrides(
    monkeypatch, operation, mi350_platform
):
    monkeypatch.setenv(
        f"TOKENSPEED_KERNEL_OVERRIDE_MOE_{operation.upper()}", "unexplained"
    )
    with pytest.raises(BenchmarkCaseError, match="override"):
        prepare_moe_apply(_beam_request(), mi350_platform)


def test_beam_rejects_a_quantizer_override_that_changes_the_timed_operation(
    monkeypatch, mi350_platform
):
    monkeypatch.setenv("TOKENSPEED_KERNEL_OVERRIDE_QUANTIZATION_FP8", "unexplained")
    with pytest.raises(BenchmarkCaseError, match="quantization.fp8 override"):
        prepare_moe_apply(_beam_request(), mi350_platform)


def test_beam_grouping_counts_are_required_before_allocation(mi350_platform):
    request = _beam_request()
    del request.parameters["group_topk"]
    with pytest.raises(BenchmarkCaseError, match="explicit grouping counts"):
        prepare_moe_apply(request, mi350_platform)


def test_timing_override_precedence_and_restoration_are_not_admission(monkeypatch):
    monkeypatch.delenv("TOKENSPEED_KERNEL_OVERRIDE_MOE_APPLY", raising=False)
    assert resolve_kernel_override("moe", "apply", None) is None
    with kernel_override("moe", "apply", "context-only"):
        assert resolve_kernel_override("moe", "apply", None) == "context-only"
        with monkeypatch.context() as environment:
            environment.setenv("TOKENSPEED_KERNEL_OVERRIDE_MOE_APPLY", "timing-only")
            assert resolve_kernel_override("moe", "apply", None) == "timing-only"
        assert resolve_kernel_override("moe", "apply", None) == "context-only"
    assert resolve_kernel_override("moe", "apply", None) is None


def _kimi_latent_expert_shared_request() -> BenchmarkRequest:
    return BenchmarkRequest(
        family="moe",
        mode="latent_expert_shared",
        parameters={
            "model_profile": "kimi_k3_tp8",
            "tokens": 2,
            "latent_size": 256,
            "intermediate_size": 256,
            "num_experts": 8,
            "num_local_experts": 4,
            "topk": 2,
            "ep_size": 2,
            "ep_rank": 1,
            "shared_size": 16,
            "output_size": 64,
            "input_dtype": "bfloat16",
            "router_logits_dtype": "float32",
            "activation_situ_beta": 4.0,
            "activation_situ_linear_beta": 25.0,
            "routed_scaling_factor": 1.0,
            "normalize_topk_weights": True,
        },
        solution=None,
        registration=None,
        cold_cache=True,
        seed=42,
    )


@pytest.mark.parametrize("ep_size", [1, 4])
def test_fp8_weights_retain_global_and_local_expert_counts(monkeypatch, ep_size):
    _use_cpu_allocations(monkeypatch)
    weights = moe_generator._make_fp8_weights(
        num_experts=16,
        num_local_experts=16 // ep_size,
        hidden_size=128,
        intermediate_size_per_partition=128,
        block_shape=(128, 128),
        weight_dtype=torch.float8_e4m3fn,
        activation="swiglu",
        swiglu_limit=10.0,
        ep_rank=0,
        ep_size=ep_size,
    )
    assert weights.num_experts == 16
    assert weights.num_local_experts == 16 // ep_size
    assert weights.w13_weight.shape[0] == weights.w2_weight.shape[0] == 16 // ep_size
    for weight in (weights.w13_weight, weights.w2_weight):
        original = weight.view(torch.uint8).clone()
        weight.weight_loader(weight, torch.empty(0), weight_copy=None)
        assert weight.device.type == "cpu"
        assert torch.equal(weight.view(torch.uint8), original)


def test_moe_fp8_weight_shapes_match_tp_and_ep_layouts() -> None:
    tp_shapes = moe_generator._fp8_weight_shapes(
        num_local_experts=288,
        hidden_size=4096,
        intermediate_size_per_partition=512,
        block_shape=(128, 128),
    )
    ep_shapes = moe_generator._fp8_weight_shapes(
        num_local_experts=72,
        hidden_size=4096,
        intermediate_size_per_partition=2048,
        block_shape=(128, 128),
    )

    assert tp_shapes == {
        "w13": (288, 1024, 4096),
        "w13_scale": (288, 8, 32),
        "w2": (288, 4096, 512),
        "w2_scale": (288, 32, 4),
    }
    assert ep_shapes == {
        "w13": (72, 4096, 4096),
        "w13_scale": (72, 32, 32),
        "w2": (72, 4096, 2048),
        "w2_scale": (72, 32, 16),
    }


def test_moe_generator_rejects_unimplemented_model_profile() -> None:
    request = BenchmarkRequest(
        family="moe",
        mode="apply",
        parameters={"model_profile": "unimplemented"},
        solution=None,
        registration=None,
        cold_cache=True,
        seed=42,
    )

    with pytest.raises(BenchmarkCaseError, match="Implemented MoE model_profile"):
        prepare_moe_apply(request, None)


def test_topk_generator_selects_and_calls_biased_sqrt_softplus_routing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    spec = KernelSpec(name="unit_topk", family="moe", mode="topk", solution="unit")

    def capture_selection(_request, _platform, _signature, traits):
        captured["traits"] = traits
        return spec

    def capture_topk(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    _use_cpu_allocations(monkeypatch)
    monkeypatch.setattr(moe_generator, "load_builtin_kernels", lambda: None)
    monkeypatch.setattr(moe_generator, "_selected_spec", capture_selection)
    monkeypatch.setattr(
        moe_generator,
        "_generator",
        lambda seed: torch.Generator().manual_seed(seed),
    )
    monkeypatch.setattr(
        moe_generator,
        "_randn",
        lambda shape, *, generator, dtype: torch.randn(
            shape, generator=generator, dtype=dtype
        ),
    )
    monkeypatch.setattr(moe_ops, "moe_topk", capture_topk)
    request = BenchmarkRequest(
        family="moe",
        mode="topk",
        parameters={
            "model_profile": "dsv41_flash_tp4",
            "score_function": "sqrt_softplus",
            "tokens": 6,
            "num_experts": 384,
            "topk": 6,
            "router_logits_dtype": "float32",
            "routed_scaling_factor": 1.5,
            "normalize_topk_weights": True,
        },
        solution=None,
        registration=None,
        cold_cache=True,
        seed=42,
    )

    prepared = moe_generator.prepare_topk(request, None)
    prepared.invocation.invoke()

    assert prepared.registration is spec
    assert captured["traits"] == {
        "tokens": 6,
        "experts": 384,
        "top_k": 6,
        "renormalize": True,
        "routing_kind": "bias",
        "score_function": "sqrt_softplus",
    }
    router_logits, topk, score_function, selection, renormalize, scaling = captured[
        "args"
    ]
    assert router_logits.shape == (6, 384) and router_logits.dtype is torch.float32
    assert (topk, score_function, selection) == (6, "sqrt_softplus", "topk")
    assert renormalize is True and scaling == 1.5
    assert captured["kwargs"]["correction_bias"].shape == (384,)


def test_moe_apply_generator_precomputes_local_ep_routes(
    fresh_registry,
    monkeypatch,
    mi350_platform: PlatformInfo,
) -> None:
    _ = fresh_registry
    spec = KernelSpec(
        name="unit_moe_apply",
        family="moe",
        mode="apply",
        solution="unit",
    )
    KernelRegistry.get().register(spec, lambda **_kwargs: None)
    seen = {}

    monkeypatch.setattr(moe_generator, "load_builtin_kernels", lambda: None)
    monkeypatch.setattr(
        moe_generator,
        "_generator",
        lambda seed: torch.Generator(device="cpu").manual_seed(seed),
    )
    monkeypatch.setattr(
        moe_generator,
        "_randn",
        lambda shape, *, generator, dtype: torch.zeros(shape, dtype=dtype),
    )
    monkeypatch.setattr(
        moe_generator,
        "_zero_fp8",
        lambda shape, _dtype: torch.zeros(shape, dtype=torch.float32),
    )
    monkeypatch.setattr(
        moe_generator,
        "_ones",
        lambda shape: torch.ones(shape, dtype=torch.float32),
    )

    def fake_plan(*_args, **kwargs):
        seen["plan_kwargs"] = kwargs
        return {"apply_kernel_name": "unit_moe_apply"}

    monkeypatch.setattr(moe_ops, "moe_plan", fake_plan)

    def fake_process_weights(plan, weights):
        seen["plan"] = plan
        seen["weights"] = weights

    def fake_topk(
        router_logits,
        top_k,
        score_function,
        selection_method,
        renormalize,
        routed_scaling_factor,
        correction_bias,
        topk_weights_dtype,
    ):
        seen["route_shape"] = tuple(router_logits.shape)
        seen["route_dtype"] = router_logits.dtype
        seen["correction_bias_shape"] = tuple(correction_bias.shape)
        seen["score_function"] = score_function
        seen["selection_method"] = selection_method
        seen["renormalize"] = renormalize
        seen["routed_scaling_factor"] = routed_scaling_factor
        seen["topk_weights_dtype"] = topk_weights_dtype
        ids = torch.arange(top_k, dtype=torch.int32).repeat(router_logits.shape[0], 1)
        weights = torch.ones(router_logits.shape[0], top_k, dtype=topk_weights_dtype)
        return weights, ids

    def fake_apply(plan, hidden_states, weights, router_logits, **kwargs):
        seen["apply"] = {
            "plan": plan,
            "hidden_shape": tuple(hidden_states.shape),
            "weights": weights,
            "router_shape": tuple(router_logits.shape),
            "topk_ids": kwargs["topk_ids"].clone(),
            "topk_weights_dtype": kwargs["topk_weights"].dtype,
            "num_tokens_global": kwargs["num_tokens_global"],
            "max_num_tokens_per_gpu": kwargs["max_num_tokens_per_gpu"],
        }
        return hidden_states

    monkeypatch.setattr(moe_ops, "moe_process_weights", fake_process_weights)
    monkeypatch.setattr(moe_ops, "moe_topk", fake_topk)
    monkeypatch.setattr(moe_ops, "moe_apply", fake_apply)

    prepared = prepare_moe_apply(
        BenchmarkRequest(
            family="moe",
            mode="apply",
            parameters={
                "model_profile": "glm53_flash_tp4",
                "tokens": 3,
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_experts": 16,
                "num_local_experts": 4,
                "topk": 2,
                "tp_size": 2,
                "ep_size": 4,
                "ep_rank": 2,
                "input_dtype": "bfloat16",
                "router_logits_dtype": "bfloat16",
                "weight_dtype": "fp8",
                "activation": "swiglu",
                "swiglu_limit": 10.0,
                "routing_mode": "precomputed_topk",
                "route_scope": "local",
                "route_distribution": "router",
                "token_count_scope": "local",
                "routed_scaling_factor": 2.5,
                "normalize_topk_weights": True,
                "fp8_scale_block_shape": [128, 128],
                "internal_activation_dtype": "input",
            },
            solution=None,
            registration=None,
            cold_cache=True,
            seed=42,
        ),
        mi350_platform,
    )

    prepared.invocation.invoke()

    assert prepared.registration is spec
    assert prepared.parameters["intermediate_size_per_partition"] == 8
    assert prepared.parameters["route_scope"] == "local"
    assert prepared.parameters["router_logits_dtype"] == "bfloat16"
    assert seen["plan_kwargs"]["hidden"] == 8
    assert seen["plan_kwargs"]["swiglu_form"] == "standard"
    assert seen["plan_kwargs"]["activation_clamped"] is True
    assert seen["plan_kwargs"]["expert_id_repeats"] is False
    assert seen["plan_kwargs"]["fast_math"] is False
    assert seen["route_shape"] == (3, 4)
    assert seen["route_dtype"] is torch.bfloat16
    assert seen["correction_bias_shape"] == (4,)
    assert seen["score_function"] == "sigmoid"
    assert seen["selection_method"] == "topk"
    assert seen["renormalize"] is True
    assert seen["routed_scaling_factor"] == 2.5
    assert seen["topk_weights_dtype"] is torch.float32
    assert seen["weights"].ep_rank == 2
    assert seen["weights"].ep_size == 4
    assert seen["weights"].num_local_experts == 4
    assert seen["weights"].num_experts == 16
    assert seen["apply"]["hidden_shape"] == (3, 8)
    assert seen["apply"]["router_shape"] == (3, 4)
    assert seen["apply"]["topk_ids"].dtype is torch.int32
    assert seen["apply"]["topk_weights_dtype"] is torch.float32
    assert seen["apply"]["topk_ids"].tolist() == [[8, 9], [8, 9], [8, 9]]
    assert seen["apply"]["num_tokens_global"] == 12
    assert seen["apply"]["max_num_tokens_per_gpu"] == 3


def test_moe_apply_generator_builds_mxfp4_situ_global_ep_contract(
    fresh_registry,
    monkeypatch,
    mi350_platform: PlatformInfo,
) -> None:
    from tokenspeed_kernel.ops import moe as moe_ops

    _ = fresh_registry
    spec = KernelSpec(
        name="unit_mxfp4_moe_apply",
        family="moe",
        mode="apply",
        solution="unit",
    )
    KernelRegistry.get().register(spec, lambda **_kwargs: None)
    seen = {}
    _use_cpu_allocations(monkeypatch)
    monkeypatch.setattr(moe_generator, "load_builtin_kernels", lambda: None)
    monkeypatch.setattr(
        moe_generator,
        "_generator",
        lambda seed: torch.Generator(device="cpu").manual_seed(seed),
    )
    monkeypatch.setattr(
        moe_generator,
        "_randn",
        lambda shape, *, generator, dtype: torch.zeros(shape, dtype=dtype),
    )

    def fake_routing(**kwargs):
        seen["routing_kwargs"] = kwargs
        return (
            torch.zeros(16, 16, dtype=torch.float32),
            torch.ones(16, 8, dtype=torch.float32),
            torch.arange(8, dtype=torch.int32).repeat(16, 1),
        )

    monkeypatch.setattr(moe_generator, "_routing_tensors", fake_routing)

    def fake_plan(**kwargs):
        seen["plan_kwargs"] = kwargs
        return {"apply_kernel_name": "unit_mxfp4_moe_apply"}

    def fake_process(_plan, weights):
        seen["weight_shapes"] = {
            "w13": tuple(weights.w13_weight.shape),
            "w13_scale": tuple(weights.w13_weight_scale.shape),
            "w2": tuple(weights.w2_weight.shape),
            "w2_scale": tuple(weights.w2_weight_scale.shape),
        }
        seen["situ"] = (
            weights.activation_situ_beta,
            weights.activation_situ_linear_beta,
        )
        weights.w13_weight = torch.zeros(
            (2, 1, 1, 1, 1, 1),
            dtype=torch.uint8,
        )

    def fake_apply(_plan, hidden_states, weights, _router_logits, **kwargs):
        seen["hidden_shape"] = tuple(hidden_states.shape)
        seen["processed_w13_shape"] = tuple(weights.w13_weight.shape)
        seen["num_tokens_global"] = kwargs["num_tokens_global"]
        seen["max_num_tokens_per_gpu"] = kwargs["max_num_tokens_per_gpu"]
        seen["ids"] = kwargs["topk_ids"].clone()
        return hidden_states

    monkeypatch.setattr(moe_ops, "moe_plan", fake_plan)
    monkeypatch.setattr(moe_ops, "moe_process_weights", fake_process)
    monkeypatch.setattr(moe_ops, "moe_apply", fake_apply)
    prepared = prepare_moe_apply(
        BenchmarkRequest(
            family="moe",
            mode="apply",
            parameters={
                "model_profile": "kimi_k3_tp8",
                "tokens": 16,
                "hidden_size": 32,
                "intermediate_size": 32,
                "num_experts": 16,
                "num_local_experts": 2,
                "topk": 8,
                "tp_size": 1,
                "ep_size": 8,
                "ep_rank": 0,
                "input_dtype": "bfloat16",
                "router_logits_dtype": "float32",
                "weight_dtype": "mxfp4",
                "activation": "situ",
                "activation_situ_beta": 4.0,
                "activation_situ_linear_beta": 25.0,
                "routing_mode": "precomputed_topk",
                "route_scope": "global",
                "route_distribution": "router",
                "routed_scaling_factor": 1.0,
                "normalize_topk_weights": True,
                "internal_activation_dtype": "input",
                "token_count_scope": "global",
            },
            solution=None,
            registration=None,
            cold_cache=True,
            seed=42,
        ),
        mi350_platform,
    )

    prepared.invocation.invoke()

    assert prepared.registration is spec
    assert seen["plan_kwargs"]["weight_dtype"] == "mxfp4"
    assert seen["plan_kwargs"]["activation"] == "situ"
    assert seen["plan_kwargs"]["ep_size"] == 8
    assert seen["plan_kwargs"]["hidden"] == 32
    assert seen["plan_kwargs"]["swiglu_form"] is None
    assert seen["plan_kwargs"]["activation_clamped"] is False
    assert seen["plan_kwargs"]["expert_id_repeats"] is False
    assert prepared.parameters["mxfp4_group_size"] == 32
    assert seen["weight_shapes"] == {
        "w13": (2, 64, 16),
        "w13_scale": (2, 64, 1),
        "w2": (2, 32, 16),
        "w2_scale": (2, 32, 1),
    }
    assert seen["situ"] == (4.0, 25.0)
    assert seen["hidden_shape"] == (16, 32)
    assert seen["processed_w13_shape"] == (2, 1, 1, 1, 1, 1)
    assert seen["num_tokens_global"] == 16
    assert seen["max_num_tokens_per_gpu"] == 2
    assert seen["ids"][0].tolist() == list(range(8))
    assert seen["routing_kwargs"]["experts"] == 16
    assert seen["routing_kwargs"]["expert_start"] == 0


def test_latent_input_generator_matches_runtime_packed_projection_contract(
    fresh_registry,
    monkeypatch,
    mi350_platform: PlatformInfo,
) -> None:
    _ = fresh_registry
    spec = KernelSpec(
        name="unit_latent_input",
        family="moe",
        mode="latent_input",
        solution="unit",
    )
    seen = {}
    _use_cpu_allocations(monkeypatch)
    latent_input_ops = importlib.import_module("tokenspeed_kernel.ops.moe")
    monkeypatch.setattr(moe_generator, "load_builtin_kernels", lambda: None)
    monkeypatch.setattr(moe_generator, "_selected_spec", lambda *_args: spec)
    monkeypatch.setattr(
        moe_generator,
        "_generator",
        lambda seed: torch.Generator(device="cpu").manual_seed(seed),
    )
    monkeypatch.setattr(
        moe_generator,
        "_randn",
        lambda shape, *, generator, dtype: torch.zeros(shape, dtype=dtype),
    )

    def fake_latent_input(
        hidden_states, router_weight, routed_weight, shared, **kwargs
    ):
        seen["shapes"] = (
            tuple(hidden_states.shape),
            tuple(router_weight.shape),
            tuple(routed_weight.shape),
            tuple(shared.shape),
        )
        seen["clamps"] = (kwargs["gate_clamp"], kwargs["up_clamp"])
        return hidden_states, hidden_states, hidden_states

    monkeypatch.setattr(
        latent_input_ops,
        "latent_moe_input_projections",
        fake_latent_input,
    )
    prepared = moe_generator.prepare_latent_input(
        BenchmarkRequest(
            family="moe",
            mode="latent_input",
            parameters={
                "model_profile": "kimi_k3_tp8",
                "tokens": 2,
                "hidden_size": 64,
                "num_experts": 8,
                "latent_size": 32,
                "shared_size": 16,
                "input_dtype": "bfloat16",
                "activation_situ_beta": 4.0,
                "activation_situ_linear_beta": 25.0,
            },
            solution=None,
            registration=None,
            cold_cache=True,
            seed=42,
        ),
        mi350_platform,
    )

    prepared.invocation.invoke()

    assert prepared.registration is spec
    # These views share one allocation, but their row counts are not aligned
    # for the fused packed operation, so public dispatch uses its fallback.
    assert prepared.parameters["weights_packed"] is False
    assert seen["shapes"] == ((2, 64), (8, 64), (32, 64), (32, 64))
    assert seen["clamps"] == (4.0, 25.0)


def test_mxfp4_weight_builder_preserves_swiglu_parameters(monkeypatch) -> None:
    _use_cpu_allocations(monkeypatch)

    weights = moe_generator._make_mxfp4_weights(
        num_experts=8,
        num_local_experts=8,
        hidden_size=32,
        intermediate_size_per_partition=32,
        activation="swiglu",
        swiglu_limit=7.0,
        situ_beta=None,
        situ_linear_beta=None,
        ep_rank=0,
        ep_size=1,
    )

    assert weights.activation == "swiglu"
    assert weights.swiglu_arg.alpha == 1.0
    assert weights.swiglu_arg.limit == 7.0


def test_latent_expert_shared_generator_uses_global_ep_routes_and_reset(
    fresh_registry,
    monkeypatch,
    mi350_platform: PlatformInfo,
) -> None:
    from tokenspeed_kernel.ops import moe as moe_ops

    _ = fresh_registry
    spec = KernelSpec(
        name="unit_latent_expert_shared",
        family="moe",
        mode="latent_expert_shared",
        solution="unit",
    )
    seen = {}
    _use_cpu_allocations(monkeypatch)
    latent_decode_ops = importlib.import_module("tokenspeed_kernel.ops.moe")
    monkeypatch.setattr(moe_generator, "load_builtin_kernels", lambda: None)
    monkeypatch.setattr(
        moe_generator,
        "_generator",
        lambda seed: torch.Generator(device="cpu").manual_seed(seed),
    )
    monkeypatch.setattr(
        moe_generator,
        "_randn",
        lambda shape, *, generator, dtype: torch.zeros(shape, dtype=dtype),
    )

    def fake_routing(**kwargs):
        seen["routing_kwargs"] = kwargs
        return (
            torch.zeros(2, 8, dtype=torch.float32),
            torch.ones(2, 2, dtype=torch.float32),
            torch.tensor([[0, 4], [1, 5]], dtype=torch.int32),
        )

    monkeypatch.setattr(moe_generator, "_routing_tensors", fake_routing)

    def fake_plan(**kwargs):
        seen["plan_kwargs"] = kwargs
        plan = {"apply_kernel_name": "unit_a8w4_moe_apply"}
        seen["plan"] = plan
        return plan

    def fake_process(plan, weights):
        seen.setdefault("events", []).append("process")
        seen["processed_plan"] = plan
        weights.w13_weight = torch.zeros(
            (4, 32, 2, 4, 16, 16),
            dtype=torch.uint8,
        )
        weights.w13_weight_scale = torch.zeros(
            (4, 16, 1, 4, 16, 4),
            dtype=torch.uint8,
        )
        weights.w2_weight = torch.zeros(
            (4, 16, 2, 4, 16, 16),
            dtype=torch.uint8,
        )
        weights.w2_weight_scale = torch.zeros(
            (4, 8, 1, 4, 16, 4),
            dtype=torch.uint8,
        )
        seen["processed_weights"] = weights

    def fake_selected(_request, _platform, signature, traits):
        seen.setdefault("events", []).append("select")
        seen["selection_signature"] = signature
        seen["selection_traits"] = traits
        return spec

    monkeypatch.setattr(moe_ops, "moe_plan", fake_plan)
    monkeypatch.setattr(moe_ops, "moe_process_weights", fake_process)
    monkeypatch.setattr(moe_generator, "_selected_spec", fake_selected)

    def fake_joint(*args, **kwargs):
        seen["joint_weight_shapes"] = tuple(tuple(tensor.shape) for tensor in args[1:5])
        seen["ids"] = args[6].clone()
        seen["expert_start"] = kwargs["expert_start"]
        kwargs["routed_out"].fill_(1)
        kwargs["shared_out"].fill_(1)
        seen["outputs"] = (kwargs["routed_out"], kwargs["shared_out"])
        return seen["outputs"]

    monkeypatch.setattr(latent_decode_ops, "latent_moe_expert_shared", fake_joint)
    prepared = moe_generator.prepare_latent_expert_shared(
        _kimi_latent_expert_shared_request(),
        mi350_platform,
    )

    prepared.invocation.invoke()
    prepared.invocation.reset()

    assert prepared.registration is spec
    assert prepared.parameters["route_scope"] == "global"
    assert prepared.parameters["route_distribution"] == "router"
    assert seen["plan_kwargs"]["weight_dtype"] == "mxfp4"
    assert seen["plan_kwargs"]["activation"] == "situ"
    assert seen["plan_kwargs"]["routing_mode"] == "precomputed_topk"
    assert seen["plan_kwargs"]["ep_size"] == 2
    assert seen["plan_kwargs"]["hidden"] == 256
    assert seen["plan_kwargs"]["ispp"] == 256
    assert seen["plan_kwargs"]["internal_activation_dtype"] == "input"
    assert seen["processed_plan"] is seen["plan"]
    assert seen["events"] == ["process", "select"]
    assert seen["selection_signature"].storage_dtype_for("w13_weight") is torch.uint8
    assert seen["selection_traits"]["linear_weights"] is False
    assert seen["selection_traits"]["intermediate_size"] == 256
    assert seen["selection_traits"]["num_local_experts"] == 4
    assert seen["selection_traits"]["inputs_contiguous"] is True
    assert seen["joint_weight_shapes"] == (
        (4, 32, 2, 4, 16, 16),
        (4, 16, 1, 4, 16, 4),
        (4, 16, 2, 4, 16, 16),
        (4, 8, 1, 4, 16, 4),
    )
    assert seen["ids"].tolist() == [[0, 4], [1, 5]]
    assert seen["expert_start"] == 4
    assert seen["routing_kwargs"]["experts"] == 8
    assert seen["routing_kwargs"]["expert_start"] == 0
    assert all(torch.count_nonzero(output) == 0 for output in seen["outputs"])


def test_latent_expert_shared_skips_incompatible_weight_representation() -> None:
    with pytest.raises(BenchmarkCaseError) as error:
        moe_generator._latent_expert_shared_weights(SimpleNamespace())

    assert error.value.status is BenchmarkStatus.NOT_APPLICABLE
    assert "not compatible with the joint" in str(error.value)


def _qwen38_request(mode: str, **parameters: object) -> BenchmarkRequest:
    return BenchmarkRequest(
        family="moe",
        mode=mode,
        parameters={"model_profile": "qwen38_flash_tp2", **parameters},
        solution=None,
        registration=None,
        cold_cache=True,
        seed=42,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_softmax_topk_generator_selection_matches_operation_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tokenspeed_kernel.ops.moe as softmax_topk_ops

    captured: list[tuple] = []

    def select(family, mode, signature, **kwargs):
        captured.append((family, mode, signature, kwargs["traits"]))
        raise RuntimeError("selection captured")

    monkeypatch.setattr(softmax_topk_ops, "select_kernel", select)
    monkeypatch.setattr(moe_generator, "select_kernel", select)
    monkeypatch.setattr(moe_generator, "load_builtin_kernels", lambda: None)

    with pytest.raises(RuntimeError, match="selection captured"):
        moe_ops.moe_topk(
            torch.zeros((64, 512), dtype=torch.bfloat16, device="cuda"),
            10,
            "softmax",
            "topk",
            True,
            1.0,
        )
    request = _qwen38_request(
        "softmax_topk",
        tokens=64,
        num_experts=512,
        topk=10,
        router_logits_dtype="bfloat16",
        normalize_topk_weights=True,
    )
    with pytest.raises(RuntimeError, match="selection captured"):
        moe_generator.prepare_softmax_topk(request, None)

    assert captured[0] == captured[1]


def test_moe_apply_generator_plans_dynamic_mxfp4_activations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def capture_plan(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("plan captured")

    monkeypatch.setattr(moe_generator, "load_builtin_kernels", lambda: None)
    monkeypatch.setattr(moe_ops, "moe_plan", capture_plan)
    request = _qwen38_request(
        "apply",
        tokens=64,
        hidden_size=2560,
        intermediate_size=640,
        num_experts=512,
        num_local_experts=512,
        topk=10,
        tp_size=2,
        ep_size=1,
        ep_rank=0,
        input_dtype="bfloat16",
        router_logits_dtype="bfloat16",
        weight_dtype="mxfp4",
        activation="silu",
        routing_mode="precomputed_topk",
        route_scope="global",
        route_distribution="router",
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
        internal_activation_dtype="mxfp4",
        token_count_scope="global",
    )
    with pytest.raises(RuntimeError, match="plan captured"):
        prepare_moe_apply(request, None)

    assert captured["weight_dtype"] == "mxfp4"
    assert captured["internal_activation_dtype"] == "mxfp4"
    assert captured["ispp"] == 320


def _muse_apply_request(**overrides: object) -> BenchmarkRequest:
    parameters = {
        "model_profile": "muse_spark_tp8",
        "tokens": 4,
        "hidden_size": 4096,
        "intermediate_size": 4096,
        "num_experts": 256,
        "num_local_experts": 256,
        "topk": 8,
        "tp_size": 8,
        "ep_size": 1,
        "ep_rank": 0,
        "input_dtype": "bfloat16",
        "router_logits_dtype": "float32",
        "weight_dtype": "mxfp4",
        "activation": "silu",
        "routing_mode": "precomputed_topk",
        "route_scope": "global",
        "route_distribution": "router",
        "routed_scaling_factor": 1.0,
        "normalize_topk_weights": True,
        "internal_activation_dtype": "mxfp4",
        "token_count_scope": "global",
        "do_finalize": False,
        "prequantized_input": True,
        **overrides,
    }
    return BenchmarkRequest(
        family="moe",
        mode="apply",
        parameters=parameters,
        solution=None,
        registration=None,
        cold_cache=True,
        seed=42,
    )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"do_finalize": 0}, "boolean"),
        ({"prequantized_input": "yes"}, "boolean"),
        (
            {"weight_dtype": "unquant", "internal_activation_dtype": "input"},
            "prequantized input requires MXFP4",
        ),
    ],
)
def test_moe_apply_rejects_invalid_route_output_and_input_contracts(
    overrides, match
) -> None:
    with pytest.raises(BenchmarkCaseError, match=match):
        prepare_moe_apply(_muse_apply_request(**overrides), None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_moe_apply_defers_finalize_over_prequantized_input() -> None:
    from tokenspeed_kernel.platform import current_platform

    platform = current_platform()
    if not platform.is_cdna4:
        pytest.skip("prequantized MXFP4 expert input requires an AMD CDNA4 GPU")

    deferred = prepare_moe_apply(_muse_apply_request(), platform)
    routes, bias, route_map = deferred.invocation.invoke()
    finalized = prepare_moe_apply(
        _muse_apply_request(do_finalize=True, prequantized_input=False), platform
    ).invocation.invoke()

    # Deferred apply leaves one unweighted row per route for the caller to reduce.
    assert routes.shape == (4 * 8, 4096) and bias is None
    assert route_map.numel() == 4 * 8
    assert finalized.shape == (4, 4096)
    assert deferred.parameters["do_finalize"] is False
    assert deferred.parameters["prequantized_input"] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_routed_rmsnorm_sum_reduces_topk_routes_per_token() -> None:
    from tokenspeed_kernel.benchmark.generators.moe import (
        prepare_routed_input_scale_rmsnorm_sum,
    )
    from tokenspeed_kernel.platform import current_platform

    prepared = prepare_routed_input_scale_rmsnorm_sum(
        BenchmarkRequest(
            family="moe",
            mode="routed_input_scale_rmsnorm_sum",
            parameters={
                "model_profile": "muse_spark_tp8",
                "tokens": 3,
                "topk": 8,
                "hidden_size": 256,
                "input_dtype": "bfloat16",
                "eps": 1e-8,
                "weight_offset": 1.0,
            },
            solution=None,
            registration=None,
            cold_cache=True,
            seed=42,
        ),
        current_platform(),
    )
    out = prepared.invocation.invoke()

    assert prepared.registration.mode == "routed_input_scale_rmsnorm_sum"
    assert out.shape == (3, 256) and out.dtype == torch.bfloat16
    # Unit-RMS routes summed with weights that sum to one stay within unit RMS
    # (up to the ~1.1 norm gain).
    assert torch.all(out.float().square().mean(-1).sqrt() < 1.5)

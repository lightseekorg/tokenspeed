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
from tokenspeed_kernel.platform import PlatformInfo
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec


def _use_cpu_allocations(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("empty", "full", "zeros"):
        allocation = getattr(torch, name)

        def cpu_allocation(*args, _allocation=allocation, **kwargs):
            kwargs.pop("device", None)
            return _allocation(*args, **kwargs)

        monkeypatch.setattr(torch, name, cpu_allocation)


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
    latent_input_ops = importlib.import_module("tokenspeed_kernel.ops.moe.latent_input")
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
    latent_decode_ops = importlib.import_module(
        "tokenspeed_kernel.ops.moe.latent_decode"
    )
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

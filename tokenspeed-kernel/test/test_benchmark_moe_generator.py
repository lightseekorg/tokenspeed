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

import pytest
import tokenspeed_kernel.benchmark.generators.moe as moe_generator
import torch
from tokenspeed_kernel.benchmark.generators.moe import prepare_moe_apply
from tokenspeed_kernel.benchmark.harness import BenchmarkCaseError, BenchmarkRequest
from tokenspeed_kernel.ops import moe as moe_ops
from tokenspeed_kernel.platform import PlatformInfo
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec


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

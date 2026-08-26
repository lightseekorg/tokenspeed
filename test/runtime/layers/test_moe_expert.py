from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.layers.moe import expert as expert_module
from tokenspeed.runtime.layers.moe.expert import MoELayer
from tokenspeed.runtime.layers.moe.topk import (
    BypassedTopKOutput,
    StandardTopKOutput,
    TopKConfig,
)


def test_hybrid_moe_dispatches_from_actual_topk_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = MoELayer.__new__(MoELayer)
    torch.nn.Module.__init__(layer)
    layer.plan = {
        "support_routing": True,
        "supports_precomputed_topk": True,
        "supports_deferred_finalize": False,
    }
    calls: list[dict] = []

    def fake_moe_apply(*args, **kwargs):
        calls.append(kwargs)
        return args[1]

    monkeypatch.setattr(expert_module.tokenspeed_kernel, "moe_apply", fake_moe_apply)
    monkeypatch.setattr(expert_module, "pdl_enabled", lambda: False)

    hidden_states = torch.empty((2, 4))
    router_logits = torch.empty((2, 8))
    layer(
        hidden_states,
        BypassedTopKOutput(hidden_states, router_logits, TopKConfig(top_k=2)),
        num_global_tokens=2,
        max_num_tokens_per_gpu=2,
    )
    layer(
        hidden_states,
        StandardTopKOutput(
            torch.empty((2, 2)),
            torch.empty((2, 2), dtype=torch.int32),
            router_logits,
        ),
        num_global_tokens=2,
        max_num_tokens_per_gpu=2,
    )

    assert "topk_weights" not in calls[0]
    assert "topk_ids" not in calls[0]
    assert calls[1]["topk_weights"].shape == (2, 2)
    assert calls[1]["topk_ids"].shape == (2, 2)


@pytest.mark.parametrize(
    "beta,linear_beta",
    [
        (None, 25.0),
        (0.0, 25.0),
        (4.0, 0.0),
    ],
)
def test_moe_layer_rejects_invalid_situ_parameters(
    beta: float | None,
    linear_beta: float | None,
) -> None:
    with pytest.raises(ValueError, match="beta values must be positive"):
        MoELayer(
            top_k=1,
            num_experts=1,
            hidden_size=32,
            intermediate_size=32,
            quant_config=None,
            layer_index=0,
            activation="situ",
            activation_situ_beta=beta,
            activation_situ_linear_beta=linear_beta,
        )


def test_moe_layer_builds_ep8_local_expert_partition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_moe_plan(weight_dtype: str, **kwargs) -> dict:
        captured["plan"] = {"weight_dtype": weight_dtype, **kwargs}
        return {
            "solution": "triton",
            "support_routing": False,
            "supports_deferred_finalize": False,
        }

    def fake_create_layer_weights(spec, *args, **kwargs) -> None:
        captured["spec"] = spec

    auto_backend = type("AutoBackend", (), {"value": "auto"})()
    monkeypatch.setattr(expert_module, "get_moe_backend", lambda: auto_backend)
    monkeypatch.setattr(expert_module.tokenspeed_kernel, "moe_plan", fake_moe_plan)
    monkeypatch.setattr(
        expert_module, "create_layer_weights", fake_create_layer_weights
    )

    layer = MoELayer(
        top_k=16,
        num_experts=896,
        hidden_size=3584,
        intermediate_size=3072,
        quant_config=None,
        layer_index=1,
        ep_rank=7,
        ep_size=8,
        activation="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        routing_mode="precomputed_topk",
    )

    assert layer.num_local_experts == 112
    assert layer.ep_rank == 7
    assert layer.ep_size == 8
    assert layer.activation_situ_beta == 4.0
    assert layer.activation_situ_linear_beta == 25.0
    assert captured["spec"].num_local_experts == 112
    assert captured["plan"]["ep_size"] == 8
    assert captured["plan"]["activation"] == "situ"
    assert captured["plan"]["routing_mode"] == "precomputed_topk"


def test_moe_layer_rejects_uneven_contiguous_ep_partition() -> None:
    with pytest.raises(ValueError, match="must be divisible"):
        MoELayer(
            top_k=2,
            num_experts=10,
            hidden_size=32,
            intermediate_size=32,
            quant_config=None,
            layer_index=0,
            ep_rank=0,
            ep_size=3,
        )


def test_moe_layer_requests_dynamic_mxfp4_activations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    class Mxfp4QuantConfig:
        ignored_layers = None
        exclude_modules = None
        is_w4a8_fp8 = False
        use_dynamic_mxfp4_activations = True

        def moe_weight_dtype(self, prefix: str) -> str:
            return "mxfp4"

    def fake_moe_plan(weight_dtype: str, **kwargs) -> dict:
        captured.update(kwargs)
        return {
            "solution": "triton",
            "support_routing": False,
            "supports_deferred_finalize": False,
        }

    auto_backend = type("AutoBackend", (), {"value": "auto"})()
    monkeypatch.setattr(expert_module, "get_moe_backend", lambda: auto_backend)
    monkeypatch.setattr(expert_module.tokenspeed_kernel, "moe_plan", fake_moe_plan)
    monkeypatch.setattr(
        expert_module, "create_layer_weights", lambda *args, **kwargs: None
    )

    MoELayer(
        top_k=2,
        num_experts=4,
        hidden_size=128,
        intermediate_size=128,
        quant_config=Mxfp4QuantConfig(),
        layer_index=0,
    )

    assert captured["internal_activation_dtype"] == "mxfp4"

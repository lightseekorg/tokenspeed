from __future__ import annotations

import pytest

from tokenspeed.runtime.layers.moe import expert as expert_module
from tokenspeed.runtime.layers.moe.expert import MoELayer


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

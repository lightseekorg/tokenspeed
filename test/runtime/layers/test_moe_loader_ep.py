from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.layers.moe import expert as expert_module
from tokenspeed.runtime.layers.moe.expert import MoELayer
from tokenspeed.runtime.layers.moe.loader import (
    _build_default_expert_plan,
    build_moe_checkpoint_loader,
)
from tokenspeed.runtime.layers.moe.schema import ExpertCheckpointSchema
from tokenspeed.runtime.layers.quantization.modelopt_mixed import ModelOptMixedConfig

_KIMI3_SCHEMA = ExpertCheckpointSchema(
    gate_proj_name="w1",
    up_proj_name="w3",
    down_proj_name="w2",
)


@pytest.mark.parametrize(
    "ep_rank,first_global,last_global",
    [
        pytest.param(0, 0, 111, id="first-rank"),
        pytest.param(3, 336, 447, id="middle-rank"),
        pytest.param(7, 784, 895, id="last-rank"),
    ],
)
def test_kimi_k3_ep8_checkpoint_plan_owns_contiguous_112_experts(
    ep_rank: int,
    first_global: int,
    last_global: int,
) -> None:
    plan = _build_default_expert_plan(
        _KIMI3_SCHEMA,
        num_experts=896,
        ep_rank=ep_rank,
        ep_size=8,
    )

    assert len(plan) == 112 * 3
    assert plan[0].local_expert_id == 0
    assert plan[0].checkpoint_weight_name == f"experts.{first_global}.w1."
    assert plan[-1].local_expert_id == 111
    assert plan[-1].checkpoint_weight_name == f"experts.{last_global}.w2."


def test_checkpoint_plan_rejects_uneven_or_out_of_range_ep() -> None:
    with pytest.raises(ValueError, match="divide evenly"):
        _build_default_expert_plan(
            _KIMI3_SCHEMA,
            num_experts=895,
            ep_rank=0,
            ep_size=8,
        )
    with pytest.raises(ValueError, match="valid EP ranks"):
        _build_default_expert_plan(
            _KIMI3_SCHEMA,
            num_experts=896,
            ep_rank=8,
            ep_size=8,
        )


def test_mtp_fp8_block_scales_load_into_local_ep_expert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_moe_plan(weight_dtype: str, **kwargs) -> dict:
        return {
            "solution": "flashinfer_trtllm",
            "support_routing": False,
            "supports_deferred_finalize": True,
        }

    trtllm_backend = type("TrtllmBackend", (), {"value": "flashinfer_trtllm"})()
    monkeypatch.setattr(expert_module, "get_moe_backend", lambda: trtllm_backend)
    monkeypatch.setattr(expert_module.tokenspeed_kernel, "moe_plan", fake_moe_plan)
    quant_config = ModelOptMixedConfig(
        quantized_layers={
            "mtp.layers.0.mlp.experts": "FP8_BLOCK_SCALES",
        }
    )
    layer = MoELayer(
        top_k=2,
        num_experts=4,
        hidden_size=256,
        intermediate_size=640,
        quant_config=quant_config,
        layer_index=0,
        prefix="mtp.layers.0.mlp",
        ep_rank=1,
        ep_size=2,
    )
    params = {
        f"model.layers.0.mlp.experts.{name}": param
        for name, param in layer.named_parameters()
    }
    loader = build_moe_checkpoint_loader(
        params_dict=params,
        expert_schema=ExpertCheckpointSchema(),
        num_experts=4,
        ep_rank=1,
        ep_size=2,
    )

    gate_scale = torch.full((5, 2), 2.0, dtype=torch.bfloat16)
    up_scale = torch.full((5, 2), 3.0, dtype=torch.bfloat16)
    down_scale = torch.full((2, 5), 4.0, dtype=torch.bfloat16)
    prefix = "model.layers.0.mlp.experts.2"

    assert loader.load(f"{prefix}.gate_proj.weight_scale_inv", gate_scale) == (
        "model.layers.0.mlp.experts.w13_weight_scale_inv"
    )
    loader.load(f"{prefix}.up_proj.weight_scale_inv", up_scale)
    loader.load(f"{prefix}.down_proj.weight_scale_inv", down_scale)

    assert layer.w13_weight_scale_inv.dtype == torch.float32
    torch.testing.assert_close(layer.w13_weight_scale_inv[0, :5], gate_scale.float())
    torch.testing.assert_close(layer.w13_weight_scale_inv[0, 5:], up_scale.float())
    torch.testing.assert_close(layer.w2_weight_scale_inv[0], down_scale.float())
    assert not loader.matches("model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv")
    assert loader.is_expert_checkpoint_weight(
        "model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv"
    )

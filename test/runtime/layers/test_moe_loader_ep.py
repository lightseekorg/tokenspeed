from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.layers.moe import expert as expert_module
from tokenspeed.runtime.layers.moe.expert import MoELayer
from tokenspeed.runtime.layers.moe.loader import (
    FusedExpertWeightPlanEntry,
    MoECheckpointLoader,
    _build_default_expert_plan,
    build_moe_checkpoint_loader,
)
from tokenspeed.runtime.layers.moe.schema import ExpertCheckpointSchema
from tokenspeed.runtime.layers.moe.types import MoELayerSpec
from tokenspeed.runtime.layers.moe.weights.mxfp4 import create_mxfp4_fp8_input_scales
from tokenspeed.runtime.layers.moe.weights.nvfp4 import create_nvfp4_weight_pair
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


@pytest.mark.parametrize("ep_rank", [0, 1])
@pytest.mark.parametrize("quant_kind", ["nvfp4", "mxfp4"])
@pytest.mark.parametrize("fused", [False, True])
def test_activation_scales_use_all_checkpoint_experts(ep_rank, quant_kind, fused):
    layer = torch.nn.Module()
    if quant_kind == "nvfp4":
        spec = MoELayerSpec(
            top_k=2,
            num_experts=4,
            num_local_experts=2,
            hidden_size=32,
            intermediate_size=32,
            activation="situ",
            tp_rank=0,
            tp_size=1,
            ep_rank=ep_rank,
            ep_size=2,
            prefix="model.layers.0.mlp",
            a2a_backend="none",
        )
        create_nvfp4_weight_pair(spec, layer, group_size=16)
    else:
        create_mxfp4_fp8_input_scales(layer, num_local_experts=2)
    prefix = "model.layers.0.mlp.experts"
    params = {f"{prefix}.{name}": value for name, value in layer.named_parameters()}
    fc1 = torch.tensor([[1.0, 2.0], [2.0, 1.0], [8.0, 10.0], [3.0, 4.0]])
    fc2 = torch.tensor([9.0, 1.0, 3.0, 6.0])
    if fused:
        loader = MoECheckpointLoader(
            params_dict=params,
            expert_plan=(),
            global_expert_plan=(),
            fused_plan=(
                FusedExpertWeightPlanEntry(
                    param_name="experts.w13_input_scale",
                    checkpoint_weight_name="experts.fc1_input_scale",
                    shard_id="w13",
                    split_dim=None,
                    split_chunks=None,
                    split_index=None,
                ),
                FusedExpertWeightPlanEntry(
                    param_name="experts.w2_input_scale",
                    checkpoint_weight_name="experts.fc2_input_scale",
                    shard_id="w2",
                    split_dim=None,
                    split_chunks=None,
                    split_index=None,
                ),
            ),
            num_experts=4,
            ep_rank=ep_rank,
            ep_size=2,
            fused_load_style="per_expert",
            transpose_local_tensor_non_bias=False,
        )
        loader.load(f"{prefix}.fc1_input_scale", fc1)
        loader.load(f"{prefix}.fc2_input_scale", fc2)
    else:
        loader = build_moe_checkpoint_loader(
            params_dict=params,
            expert_schema=_KIMI3_SCHEMA,
            num_experts=4,
            ep_rank=ep_rank,
            ep_size=2,
        )
        for expert_id in range(4):
            for projection, value in (
                ("w1", fc1[expert_id, 0]),
                ("w3", fc1[expert_id, 1]),
                ("w2", fc2[expert_id]),
            ):
                name = f"{prefix}.{expert_id}.{projection}.input_scale"
                assert loader.matches(name)
                loader.load(name, value)
        remote_expert = 2 if ep_rank == 0 else 0
        assert not loader.matches(f"{prefix}.{remote_expert}.w1.weight")
        assert not loader.matches(f"{prefix}.{remote_expert}.w1.weight_scale_2")
        if quant_kind == "nvfp4":
            loader.load(f"{prefix}.{ep_rank * 2}.w1.weight_scale_2", torch.tensor(0.25))
            assert layer.w13_weight_scale_2[0, 0] == 0.25
    torch.testing.assert_close(
        layer.w13_input_scale, torch.full_like(layer.w13_input_scale, 10.0)
    )
    torch.testing.assert_close(
        layer.w2_input_scale, torch.full_like(layer.w2_input_scale, 9.0)
    )

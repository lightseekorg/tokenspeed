"""CPU-only coverage for MXFP4 quantization metadata."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import tokenspeed.runtime.layers.quantization.mxfp4 as mxfp4_module
from tokenspeed.runtime.layers.quantization.mxfp4 import (
    Mxfp4Config,
    dequantize_mxfp4_to_bf16,
    preprocess_mxfp4_checkpoint_weights,
)
from tokenspeed.runtime.layers.quantization.utils import should_ignore_quant_layer


def _fp4_e8m0_per_group(*, is_dynamic: bool) -> dict:
    return {
        "dtype": "fp4",
        "is_dynamic": is_dynamic,
        "qscheme": "per_group",
        "group_size": 32,
        "scale_format": "e8m0",
    }


def _amd_quark_mxfp4_config(
    input_tensors: dict,
    *,
    exclude: list[str] | None = None,
) -> dict:
    return {
        "global_quant_config": {
            "input_tensors": input_tensors,
            "output_tensors": None,
            "weight": _fp4_e8m0_per_group(is_dynamic=False),
        },
        "quant_method": "quark",
        "export": {"pack_method": "reorder", "weight_format": "real_quantized"},
        "exclude": exclude or [],
    }


def _mock_platform(monkeypatch, *, is_amd: bool) -> None:
    monkeypatch.setattr(
        mxfp4_module,
        "current_platform",
        lambda: SimpleNamespace(is_amd=is_amd),
    )


def test_amd_quark_dynamic_mxfp4_metadata_selects_mxfp4(monkeypatch) -> None:
    _mock_platform(monkeypatch, is_amd=True)
    config = _amd_quark_mxfp4_config(_fp4_e8m0_per_group(is_dynamic=True))

    assert Mxfp4Config.override_quantization_method(config, None) == "mxfp4"
    assert Mxfp4Config.override_quantization_method(config, "mxfp4") == "mxfp4"
    assert Mxfp4Config.override_quantization_method(config, "nvfp4") is None

    quant_config = Mxfp4Config.from_config(config)
    assert quant_config.is_checkpoint_mxfp4_serialized is True
    assert quant_config.use_dynamic_mxfp4_activations is True
    assert quant_config.is_w4a8_fp8 is False
    assert quant_config.group_size == 32


def test_amd_quark_metadata_is_not_promoted_on_non_amd(monkeypatch) -> None:
    _mock_platform(monkeypatch, is_amd=False)
    config = _amd_quark_mxfp4_config(_fp4_e8m0_per_group(is_dynamic=True))

    assert Mxfp4Config.override_quantization_method(config, None) is None

    quant_config = Mxfp4Config.from_config(config)
    assert quant_config.is_checkpoint_mxfp4_serialized is False
    assert quant_config.use_dynamic_mxfp4_activations is False
    assert quant_config.is_w4a8_fp8 is False


def test_amd_quark_w4a8_fp8_metadata_selects_mxfp4(monkeypatch) -> None:
    _mock_platform(monkeypatch, is_amd=True)
    config = _amd_quark_mxfp4_config({"dtype": "fp8_e4m3"})

    assert Mxfp4Config.override_quantization_method(config, None) == "mxfp4"

    quant_config = Mxfp4Config.from_config(config)
    assert quant_config.is_checkpoint_mxfp4_serialized is True
    assert quant_config.use_dynamic_mxfp4_activations is False
    assert quant_config.is_w4a8_fp8 is True


def test_amd_quark_excludes_match_runtime_layer_names(monkeypatch) -> None:
    _mock_platform(monkeypatch, is_amd=True)
    config = _amd_quark_mxfp4_config(
        _fp4_e8m0_per_group(is_dynamic=True),
        exclude=[
            "*lm_head",
            "language_model.model.layers.0.self_attn.*",
            "re:language_model\\.model\\.layers\\.0\\.mlp\\.gate$",
        ],
    )

    ignored_layers = Mxfp4Config.from_config(config).ignored_layers
    assert should_ignore_quant_layer("lm_head", ignored_layers)
    assert should_ignore_quant_layer(
        "model.layers.0.self_attn.q_proj",
        ignored_layers,
    )
    assert should_ignore_quant_layer(
        "model.layers.0.mlp.gate",
        ignored_layers,
    )
    assert not should_ignore_quant_layer(
        "model.layers.0.mlp.experts.0.gate_proj",
        ignored_layers,
    )


def test_incomplete_amd_quark_metadata_is_not_promoted(monkeypatch) -> None:
    _mock_platform(monkeypatch, is_amd=True)
    config = _amd_quark_mxfp4_config(
        {
            "dtype": "fp4",
            "is_dynamic": False,
            "qscheme": "per_group",
            "group_size": 32,
            "scale_format": "e8m0",
        }
    )
    config["export"] = {"pack_method": "reorder", "weight_format": "real_quantized"}

    assert Mxfp4Config.override_quantization_method(config, None) is None

    quant_config = Mxfp4Config.from_config(config)
    assert quant_config.is_checkpoint_mxfp4_serialized is False
    assert quant_config.use_dynamic_mxfp4_activations is False
    assert quant_config.is_w4a8_fp8 is False


def _fp8_per_channel_w8a8() -> dict:
    """Per-layer FP8 attention override (static per-channel W, per-token A)."""
    return {
        "input_tensors": {
            "dtype": "fp8_e4m3",
            "is_dynamic": True,
            "qscheme": "per_channel",
            "ch_axis": 1,
        },
        "output_tensors": None,
        "weight": {
            "dtype": "fp8_e4m3",
            "is_dynamic": False,
            "qscheme": "per_channel",
            "ch_axis": 0,
        },
    }


def _fp8_attention_override_config(monkeypatch) -> Mxfp4Config:
    _mock_platform(monkeypatch, is_amd=True)
    config = _amd_quark_mxfp4_config(
        _fp4_e8m0_per_group(is_dynamic=True),
        exclude=["language_model.model.layers.0.self_attn.q_conv1d"],
    )
    config["layer_quant_config"] = {"*self_attn*": _fp8_per_channel_w8a8()}
    return Mxfp4Config.from_config(config)


def test_layer_quant_config_routes_fp8_attention(monkeypatch) -> None:
    quant_config = _fp8_attention_override_config(monkeypatch)

    assert quant_config.fp8_config.is_checkpoint_fp8_serialized is True
    route = quant_config.fp8_override_route
    assert route("model.layers.0.self_attn.qkvgb_proj") == "w8a8"
    assert route("language_model.model.layers.3.self_attn.o_proj") == "w8a8"
    # Raw-consumed weights dequantize at load.
    assert route("model.layers.0.self_attn.f_b_proj") == "dequant"
    assert route("model.layers.3.self_attn.kv_b_proj") == "dequant"
    # Excluded and non-attention modules keep the global scheme.
    assert route("model.layers.0.self_attn.q_conv1d") is None
    assert route("model.layers.1.block_sparse_moe.shared_experts.gate_up_proj") is None


def test_rejects_unsupported_layer_quant_config(monkeypatch) -> None:
    _mock_platform(monkeypatch, is_amd=True)
    config = _amd_quark_mxfp4_config(_fp4_e8m0_per_group(is_dynamic=True))
    per_tensor = _fp8_per_channel_w8a8()
    per_tensor["weight"]["qscheme"] = "per_tensor"
    config["layer_quant_config"] = {"*self_attn*": per_tensor}

    with pytest.raises(ValueError, match="Unsupported per-layer quantization scheme"):
        Mxfp4Config.from_config(config)


def test_dequantize_mxfp4_uses_low_nibble_first_and_e8m0_scale() -> None:
    # Codes 1 (0.5) in the low nibble and 0xA (-1.0) in the high nibble,
    # repeated over one 32-element group; scale 2^(128 - 127) = 2.
    packed = torch.full((1, 16), 0xA1, dtype=torch.uint8)
    scales = torch.full((1, 1), 128, dtype=torch.uint8)

    values = dequantize_mxfp4_to_bf16(packed, scales)

    expected = torch.tensor([1.0, -2.0] * 16, dtype=torch.bfloat16).view(1, 32)
    torch.testing.assert_close(values, expected, rtol=0, atol=0)


def test_preprocess_mxfp4_checkpoint_weights_adapts_fp8_and_mxfp4_modules(
    monkeypatch,
) -> None:
    quant_config = _fp8_attention_override_config(monkeypatch)
    fp8 = torch.tensor([[1.0, -2.0], [0.5, 4.0]]).to(torch.float8_e4m3fn)
    scale = torch.tensor([2.0, 0.5])
    mxfp4 = torch.full((2, 16), 0x22, dtype=torch.uint8)  # 1.0 everywhere
    e8m0 = torch.full((2, 1), 127, dtype=torch.uint8)
    stream = [
        ("model.layers.0.self_attn.q_proj.weight", fp8),
        ("model.layers.0.self_attn.q_proj.weight_scale", scale),
        ("model.layers.0.self_attn.f_b_proj.weight_scale", scale),
        ("model.layers.0.self_attn.f_b_proj.weight", fp8),
        ("model.layers.1.block_sparse_moe.shared_experts.up_proj.weight", mxfp4),
        ("model.layers.1.block_sparse_moe.shared_experts.up_proj.weight_scale", e8m0),
        ("model.layers.1.block_sparse_moe.experts.0.w1.weight", mxfp4),
        ("model.layers.0.self_attn.A_log", scale),
    ]

    out = dict(
        preprocess_mxfp4_checkpoint_weights(
            stream,
            quant_config,
            dequantize_mxfp4_module=lambda module: ".shared_experts." in module,
        )
    )

    assert out["model.layers.0.self_attn.q_proj.weight"] is fp8
    assert out["model.layers.0.self_attn.q_proj.weight_scale"].shape == (2, 1)
    torch.testing.assert_close(
        out["model.layers.0.self_attn.f_b_proj.weight"],
        torch.tensor([[2.0, -4.0], [0.25, 2.0]], dtype=torch.bfloat16),
    )
    assert "model.layers.0.self_attn.f_b_proj.weight_scale" not in out
    shared = out["model.layers.1.block_sparse_moe.shared_experts.up_proj.weight"]
    torch.testing.assert_close(shared, torch.ones(2, 32, dtype=torch.bfloat16))
    # Routed experts and unrelated tensors pass through unchanged.
    assert out["model.layers.1.block_sparse_moe.experts.0.w1.weight"] is mxfp4
    assert out["model.layers.0.self_attn.A_log"] is scale


def test_preprocess_mxfp4_checkpoint_weights_without_overrides(monkeypatch) -> None:
    # W4A8 metadata: no per-layer overrides and no dynamic MXFP4 activations.
    _mock_platform(monkeypatch, is_amd=True)
    quant_config = Mxfp4Config.from_config(
        _amd_quark_mxfp4_config({"dtype": "fp8_e4m3"})
    )
    mxfp4 = torch.full((2, 16), 0x22, dtype=torch.uint8)  # 1.0 everywhere
    e8m0 = torch.full((2, 1), 127, dtype=torch.uint8)
    bf16 = torch.ones(2, 32, dtype=torch.bfloat16)
    stream = [
        ("model.layers.1.block_sparse_moe.shared_experts.up_proj.weight", mxfp4),
        ("model.layers.1.block_sparse_moe.shared_experts.up_proj.weight_scale", e8m0),
        ("model.layers.0.mlp.down_proj.weight", bf16),
    ]

    out = dict(
        preprocess_mxfp4_checkpoint_weights(
            stream,
            quant_config,
            dequantize_mxfp4_module=lambda module: ".shared_experts." in module
            or ".mlp." in module,
        )
    )

    shared = out["model.layers.1.block_sparse_moe.shared_experts.up_proj.weight"]
    torch.testing.assert_close(shared, torch.ones(2, 32, dtype=torch.bfloat16))
    # An MLP the checkpoint keeps in BF16 loads as-is.
    assert out["model.layers.0.mlp.down_proj.weight"] is bf16


def test_preprocess_mxfp4_checkpoint_weights_reports_unpaired_dequant_modules(
    monkeypatch,
) -> None:
    quant_config = _fp8_attention_override_config(monkeypatch)
    fp8 = torch.ones(2, 2).to(torch.float8_e4m3fn)
    stream = [("model.layers.0.self_attn.f_b_proj.weight", fp8)]

    with pytest.raises(RuntimeError, match="missing their weight/weight_scale"):
        list(
            preprocess_mxfp4_checkpoint_weights(
                stream, quant_config, dequantize_mxfp4_module=lambda module: False
            )
        )

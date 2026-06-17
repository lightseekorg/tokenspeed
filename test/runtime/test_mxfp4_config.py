"""CPU-only coverage for MXFP4 quantization metadata."""

from __future__ import annotations

from types import SimpleNamespace

import tokenspeed.runtime.layers.quantization.mxfp4 as mxfp4_module
from tokenspeed.runtime.layers.quantization.mxfp4 import Mxfp4Config
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

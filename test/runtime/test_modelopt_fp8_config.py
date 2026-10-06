"""CPU-only detection and parsing of ModelOpt FP8 (per-tensor) checkpoints."""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import pytest

from tokenspeed.runtime.layers.quantization import QUANTIZATION_METHODS
from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config, ModelOptFp8Config
from tokenspeed.runtime.model_loader.weight_utils import get_quant_config

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(
    est_time=10,
    suite="runtime-1gpu",
    disabled_on_runners=["amd-*"],
    disabled_on_runners_reason="ModelOpt FP8 checkpoints are an NVIDIA format",
)

_SECTION = {
    "quant_algo": "FP8",
    "kv_cache_quant_algo": "FP8",
    "exclude_modules": ["lm_head", "model.layers.*.mlp.gate"],
}
# config.json's quantization_config (flat) and hf_quant_config.json (nested).
_FLAT = {"quant_method": "modelopt", **_SECTION}
_NESTED = {"producer": {"name": "modelopt"}, "quantization": dict(_SECTION)}


def _detect(hf_quant_cfg: dict, user_quant: str | None = None) -> str | None:
    # ModelConfig._verify_quantization's loop: the first override wins.
    for method in QUANTIZATION_METHODS.values():
        detected = method.override_quantization_method(hf_quant_cfg, user_quant)
        if detected:
            return detected
    return None


@pytest.mark.parametrize(
    "hf_quant_cfg",
    [
        _FLAT,
        # ModelConfig._parse_quant_hf_config's view of hf_quant_config.json.
        {"quant_method": "modelopt", **_NESTED["quantization"]},
        {"quant_method": "modelopt", **_NESTED},
    ],
)
def test_modelopt_fp8_is_detected(hf_quant_cfg):
    assert "modelopt_fp8" in QUANTIZATION_METHODS
    assert _detect(hf_quant_cfg) == "modelopt_fp8"
    assert _detect(hf_quant_cfg, "fp8") == "modelopt_fp8"


@pytest.mark.parametrize(
    "hf_quant_cfg,expected",
    [
        ({"quant_method": "modelopt", "quant_algo": "NVFP4"}, "nvfp4"),
        ({"quant_method": "modelopt", "quant_algo": "FP8_PB_WO"}, None),
        ({"quant_method": "fp8", "weight_block_size": [128, 128]}, None),
        ({"quant_method": "compressed-tensors", "quant_algo": "FP8"}, None),
    ],
)
def test_other_formats_keep_their_detection(hf_quant_cfg, expected):
    assert _detect(hf_quant_cfg) == expected


@pytest.mark.parametrize("config", [_FLAT, _NESTED])
def test_from_config_is_per_tensor_static(config):
    parsed = ModelOptFp8Config.from_config(config)
    assert isinstance(parsed, Fp8Config)
    assert parsed.get_name() == "modelopt_fp8"
    assert parsed.is_checkpoint_fp8_serialized
    assert parsed.activation_scheme == "static"
    assert parsed.weight_block_size is None
    assert parsed.kv_cache_quant_algo == "FP8"
    assert parsed.exclude_modules == _SECTION["exclude_modules"]
    assert parsed.ignored_layers == _SECTION["exclude_modules"]
    assert parsed.moe_weight_dtype() == "fp8"


def test_from_config_refuses_another_algorithm():
    with pytest.raises(ValueError, match="only supports FP8"):
        ModelOptFp8Config.from_config({**_FLAT, "quant_algo": "NVFP4"})


def test_get_quant_config_reads_hf_quant_config_json(tmp_path):
    (tmp_path / "hf_quant_config.json").write_text(json.dumps(_NESTED))
    model_config = SimpleNamespace(
        quantization="modelopt_fp8",
        hf_config=SimpleNamespace(),
        model_path=str(tmp_path),
        revision=None,
    )
    parsed = get_quant_config(model_config, SimpleNamespace(download_dir=None))
    assert isinstance(parsed, ModelOptFp8Config)
    assert parsed.is_checkpoint_fp8_serialized and parsed.weight_block_size is None
    flat = SimpleNamespace(**vars(model_config))
    flat.hf_config = SimpleNamespace(quantization_config=_FLAT)
    assert get_quant_config(flat, None).exclude_modules == _SECTION["exclude_modules"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

"""CPU-only parsing of ModelOpt NVFP4 configs: the exclude list of each form."""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import pytest

from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.layers.quantization.nvfp4 import Nvfp4Config

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(
    est_time=10,
    suite="runtime-1gpu",
    disabled_on_runners=["amd-*"],
    disabled_on_runners_reason="NVFP4 is an NVIDIA format",
)

_EXCLUDED = ["lm_head", "model.layers.0*", "model.layers.*.mlp.gate"]
_SECTION = {
    "quant_algo": "NVFP4",
    "kv_cache_quant_algo": None,
    "group_size": 16,
    "exclude_modules": _EXCLUDED,
}
# hf_quant_config.json (nested) and config.json's quantization_config (flat,
# ModelOpt names the list "ignore").
_NESTED = {"producer": {"name": "modelopt"}, "quantization": dict(_SECTION)}
_CONFIG_JSON = {
    "quant_method": "modelopt",
    "quant_algo": "NVFP4",
    "group_size": 16,
    "ignore": _EXCLUDED,
}
# config.json's quantization_config in ModelOpt's config_groups export: the
# group size sits in each group, not at the top level.
_FP4_ARGS = {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16}
_CONFIG_GROUPS_JSON = {
    "config_groups": {
        "group_0": {
            "input_activations": dict(_FP4_ARGS),
            "weights": dict(_FP4_ARGS),
            "targets": ["Linear"],
        }
    },
    "ignore": _EXCLUDED,
    "producer": {"name": "modelopt"},
    "quant_algo": "NVFP4",
    "quant_method": "modelopt",
}


def _flat_view(tmp_path) -> dict:
    """ModelConfig._parse_quant_hf_config's flat dict of hf_quant_config.json."""
    (tmp_path / "hf_quant_config.json").write_text(json.dumps(_NESTED))
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(), model_path=str(tmp_path), revision=None
    )
    return ModelConfig._parse_quant_hf_config(model_config)


@pytest.mark.parametrize(
    "form", ["nested", "flat-view", "config-json", "config-json-groups"]
)
def test_every_form_keeps_the_exclude_list(tmp_path, form):
    config = {
        "nested": _NESTED,
        "flat-view": _flat_view(tmp_path),
        "config-json": _CONFIG_JSON,
        "config-json-groups": _CONFIG_GROUPS_JSON,
    }[form]
    parsed = Nvfp4Config.from_config(config)
    assert parsed.exclude_modules == _EXCLUDED
    assert parsed.group_size == 16


def test_the_flat_view_names_the_list_exclude_modules(tmp_path):
    flat = _flat_view(tmp_path)
    assert flat["quant_method"] == "modelopt" and "ignore" not in flat
    assert flat["exclude_modules"] == _EXCLUDED


def test_ignore_is_read_first_when_both_are_given():
    config = {**_CONFIG_JSON, "exclude_modules": ["model.embed_tokens"]}
    assert Nvfp4Config.from_config(config).exclude_modules == _EXCLUDED


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

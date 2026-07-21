"""CPU-only coverage for the ModelOpt MIXED_PRECISION quantization config."""

from __future__ import annotations

import pytest

from tokenspeed.runtime.layers.quantization import QUANTIZATION_METHODS
from tokenspeed.runtime.layers.quantization.modelopt_mixed import ModelOptMixedConfig

_RENAMES = (("language_model.", ""),)


def _mixed_quant_config(extra_layers: dict | None = None) -> dict:
    """Minimal MIXED_PRECISION quantization_config in checkpoint naming."""
    layers = {
        "language_model.model.layers.3.self_attn.q_proj": {"quant_algo": "MXFP8"},
        "language_model.model.layers.3.self_attn.k_proj": {"quant_algo": "MXFP8"},
        "language_model.model.layers.3.self_attn.v_proj": {"quant_algo": "MXFP8"},
        "language_model.model.layers.3.self_attn.o_proj": {"quant_algo": "MXFP8"},
        "language_model.model.layers.3.self_attn.index_q_proj": {"quant_algo": "MXFP8"},
        "language_model.model.layers.3.block_sparse_moe.shared_experts.gate_proj": {
            "quant_algo": "MXFP8"
        },
        "language_model.model.layers.3.block_sparse_moe.shared_experts.up_proj": {
            "quant_algo": "MXFP8"
        },
        "language_model.model.layers.3.block_sparse_moe.experts.0.w1": {
            "quant_algo": "NVFP4",
            "group_size": 16,
        },
        "language_model.model.layers.3.block_sparse_moe.experts.0.w2": {
            "quant_algo": "NVFP4",
            "group_size": 16,
        },
    }
    layers.update(extra_layers or {})
    return {
        "quant_algo": "MIXED_PRECISION",
        "kv_cache_quant_algo": None,
        "quant_method": "modelopt",
        "exclude_modules": [
            "lm_head",
            "language_model.model.layers.3.block_sparse_moe.gate",
        ],
        "quantized_layers": layers,
    }


def _renamed_config(extra_layers: dict | None = None) -> ModelOptMixedConfig:
    config = ModelOptMixedConfig.from_config(_mixed_quant_config(extra_layers))
    config.apply_checkpoint_name_replacements(_RENAMES)
    return config


def test_override_detects_mixed_precision():
    hf_cfg = _mixed_quant_config()
    detected = None
    for method in QUANTIZATION_METHODS.values():
        detected = method.override_quantization_method(hf_cfg, None)
        if detected:
            break
    assert detected == "modelopt_mixed"


def test_override_detects_nested_hf_quant_config():
    nested = {"producer": {"name": "modelopt"}, "quantization": _mixed_quant_config()}
    assert (
        ModelOptMixedConfig.override_quantization_method(nested, None)
        == "modelopt_mixed"
    )
    config = ModelOptMixedConfig.from_config(nested)
    assert config.group_size == 16


def test_from_config_rejects_unknown_algo():
    with pytest.raises(ValueError, match="Unsupported quant_algo"):
        ModelOptMixedConfig.from_config(
            _mixed_quant_config(
                {"language_model.model.layers.3.mlp.up_proj": {"quant_algo": "INT8"}}
            )
        )


def test_from_config_rejects_missing_quantized_layers():
    cfg = _mixed_quant_config()
    cfg["quantized_layers"] = {}
    with pytest.raises(ValueError, match="quantized_layers"):
        ModelOptMixedConfig.from_config(cfg)


def test_resolution_after_renames():
    config = _renamed_config()
    # Fused projections unfuse to their checkpoint members.
    assert config._resolve_quant_algo("model.layers.3.self_attn.qkv_proj") == "MXFP8"
    assert (
        config._resolve_quant_algo(
            "model.layers.3.block_sparse_moe.shared_experts.gate_up_proj"
        )
        == "MXFP8"
    )
    # Construction prefixes keep the flat checkpoint indexer naming.
    assert (
        config._resolve_quant_algo("model.layers.3.self_attn.index_q_proj") == "MXFP8"
    )
    # Parent module resolves through its children.
    assert (
        config._resolve_quant_algo("model.layers.3.block_sparse_moe.experts") == "NVFP4"
    )
    # Unlisted modules resolve to None (unquantized).
    assert config._resolve_quant_algo("model.layers.3.block_sparse_moe.gate") is None
    assert config._resolve_quant_algo("lm_head") is None


def test_fused_members_must_agree():
    config = _renamed_config(
        {"language_model.model.layers.3.self_attn.v_proj": {"quant_algo": "NVFP4"}}
    )
    with pytest.raises(ValueError, match="Mixed quant_algo within fused layer"):
        config._resolve_quant_algo("model.layers.3.self_attn.qkv_proj")


def test_ambiguous_child_scan_raises():
    config = _renamed_config()
    with pytest.raises(ValueError, match="mixed quant_algo"):
        config._resolve_quant_algo("model.layers.3.block_sparse_moe")


def test_moe_weight_dtype_prefers_experts_subtree():
    config = _renamed_config()
    assert config.moe_weight_dtype("model.layers.3.block_sparse_moe.experts") == "nvfp4"
    # A MoE block prefix must not be captured by the MXFP8 shared experts.
    assert config.moe_weight_dtype("model.layers.3.block_sparse_moe") == "nvfp4"
    with pytest.raises(ValueError, match="MoE prefix"):
        config.moe_weight_dtype("model.layers.99.block_sparse_moe")


def test_minimax_m3_quant_rename_table_matches_module_prefixes():
    from tokenspeed.runtime.models.minimax_m3 import MiniMaxM3SparseForCausalLM

    replacements = MiniMaxM3SparseForCausalLM.quant_module_name_replacements
    name = "language_model.model.layers.3.self_attn.index_q_proj"
    for old, new in replacements:
        name = name.replace(old, new)
    # Construction prefixes keep the checkpoint module tree.
    assert name == "model.layers.3.self_attn.index_q_proj"

"""CPU-only coverage for the ModelOpt MIXED_PRECISION quantization config."""

from __future__ import annotations

import pytest
import torch

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


_FP8_PB_WO_LAYERS = {
    # w8a8 (weights flow through quant_method.apply as plain LinearBase calls)
    "language_model.model.layers.4.self_attn.o_proj": {"quant_algo": "FP8_PB_WO"},
    "language_model.model.layers.4.self_attn.kv_b_proj": {"quant_algo": "FP8_PB_WO"},
    # dequant (raw weights consumed by fused/custom bf16 kernels)
    "language_model.model.layers.4.self_attn.q_b_proj": {"quant_algo": "FP8_PB_WO"},
    "language_model.model.layers.4.self_attn.q_proj": {"quant_algo": "FP8_PB_WO"},
    "language_model.model.layers.4.self_attn.f_b_proj": {"quant_algo": "FP8_PB_WO"},
    "language_model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa": {
        "quant_algo": "FP8_PB_WO"
    },
}


def test_from_config_accepts_fp8_pb_wo():
    config = _renamed_config(_FP8_PB_WO_LAYERS)
    assert config._resolve_quant_algo("model.layers.4.self_attn.o_proj") == "FP8_PB_WO"
    assert config.has_fp8_pb_wo
    # FP8_PB_WO aliases to FP8_BLOCK_SCALES: DeepSeek-style w8a8 with
    # ModelOpt's fixed 128x128 block shape and non-ue8m0 float32 scales.
    assert config.fp8_block_scales_config.weight_block_size == [128, 128]
    assert config.fp8_block_scales_config.scale_fmt is None
    assert config.fp8_block_scales_config.activation_scheme == "dynamic"


def test_fp8_pb_wo_route_classification():
    config = _renamed_config(_FP8_PB_WO_LAYERS)
    base = "model.layers.4.self_attn"
    assert config.fp8_pb_wo_route(f"{base}.o_proj") == "w8a8"
    assert config.fp8_pb_wo_route(f"{base}.kv_b_proj") == "w8a8"
    for leaf in ("q_b_proj", "q_proj", "f_b_proj", "fused_qkv_a_proj_with_mqa"):
        assert config.fp8_pb_wo_route(f"{base}.{leaf}") == "dequant"
    # Non-FP8_PB_WO modules never route.
    assert config.fp8_pb_wo_route("model.layers.3.self_attn.o_proj") is None
    assert config.fp8_pb_wo_route("model.layers.99.self_attn.o_proj") is None


def _quantize_per_block_cpu(w):
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    n, k = w.shape
    nb, kb = (n + 127) // 128, (k + 127) // 128
    padded = torch.zeros(nb * 128, kb * 128, dtype=torch.float32)
    padded[:n, :k] = w.float()
    blocks = padded.view(nb, 128, kb, 128)
    scales = (blocks.abs().amax(dim=(1, 3)).clamp(min=1e-12) / fp8_max).contiguous()
    q = (blocks / scales[:, None, :, None]).clamp(-fp8_max, fp8_max)
    return (
        q.view(nb * 128, kb * 128)[:n, :k].to(torch.float8_e4m3fn).contiguous(),
        scales,
    )


def test_preprocess_fp8_pb_wo_weights_stream():
    from tokenspeed.runtime.layers.quantization.modelopt_mixed import (
        preprocess_fp8_pb_wo_weights,
    )

    torch.manual_seed(0)
    config = _renamed_config(_FP8_PB_WO_LAYERS)
    base = "model.layers.4.self_attn"
    n, k = 96, 200  # ragged on both axes
    q_deq, s_deq = _quantize_per_block_cpu(torch.randn(n, k))
    q_w8a8, s_w8a8 = _quantize_per_block_cpu(torch.randn(256, 128))
    scale_4d = s_deq.reshape(s_deq.shape[0], 1, s_deq.shape[1], 1)
    other = torch.randn(8, dtype=torch.bfloat16)
    nvfp4_scale = torch.randn(4)

    stream = [
        # dequant module: scale arrives BEFORE the weight
        (f"{base}.q_proj.weight_scale", scale_4d),
        (f"{base}.q_proj.weight", q_deq),
        # w8a8 module: fp8 weight passes through, scale renamed + squeezed
        (f"{base}.o_proj.weight", q_w8a8),
        (
            f"{base}.o_proj.weight_scale",
            s_w8a8.reshape(s_w8a8.shape[0], 1, s_w8a8.shape[1], 1),
        ),
        # untouched tensors: norms and NVFP4 expert scales keep their names
        ("model.layers.4.input_layernorm.weight", other),
        ("model.layers.3.block_sparse_moe.experts.0.w1.weight_scale", nvfp4_scale),
    ]
    out = dict(preprocess_fp8_pb_wo_weights(iter(stream), config))

    # Dequant path: bitwise-identical to the manual block dequant.
    dq = out[f"{base}.q_proj.weight"].cpu()
    assert dq.dtype == torch.bfloat16
    s_full = s_deq.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)[:, :k]
    expected = (q_deq.float() * s_full).to(torch.bfloat16)
    assert torch.equal(dq, expected)
    assert f"{base}.q_proj.weight_scale" not in out  # scale is consumed

    # w8a8 path: weight untouched, scale renamed to weight_scale_inv (2-D).
    assert out[f"{base}.o_proj.weight"].dtype == torch.float8_e4m3fn
    assert f"{base}.o_proj.weight_scale" not in out
    assert torch.equal(out[f"{base}.o_proj.weight_scale_inv"], s_w8a8)

    # Everything else passes through untouched.
    assert out["model.layers.4.input_layernorm.weight"] is other
    assert (
        out["model.layers.3.block_sparse_moe.experts.0.w1.weight_scale"] is nvfp4_scale
    )


def test_preprocess_fp8_pb_wo_weights_missing_scale_raises():
    from tokenspeed.runtime.layers.quantization.modelopt_mixed import (
        preprocess_fp8_pb_wo_weights,
    )

    config = _renamed_config(_FP8_PB_WO_LAYERS)
    q, _ = _quantize_per_block_cpu(torch.randn(128, 128))
    stream = [("model.layers.4.self_attn.q_proj.weight", q)]
    with pytest.raises(RuntimeError, match="missing their weight/weight_scale"):
        list(preprocess_fp8_pb_wo_weights(iter(stream), config))


def test_preprocess_route_miss_fp8_raises():
    """Unrouted quantized-looking tensors must fail loudly (name drift)."""
    from tokenspeed.runtime.layers.quantization.modelopt_mixed import (
        preprocess_fp8_pb_wo_weights,
    )

    config = _renamed_config(_FP8_PB_WO_LAYERS)
    q, scales = _quantize_per_block_cpu(torch.randn(128, 128))

    # An FP8 weight whose module resolves to no quant_algo -> drift error.
    with pytest.raises(RuntimeError, match="module-name drift"):
        list(
            preprocess_fp8_pb_wo_weights(
                iter([("model.layers.4.self_attn.renamed_proj.weight", q)]), config
            )
        )
    # A weight_scale sidecar with no owning quantized module -> same error.
    with pytest.raises(RuntimeError, match="module-name drift"):
        list(
            preprocess_fp8_pb_wo_weights(
                iter([("model.layers.4.self_attn.renamed_proj.weight_scale", scales)]),
                config,
            )
        )
    # Excluded modules may carry anything; NVFP4 expert scales resolve to
    # NVFP4; plain bf16 weights of unquantized modules are untouched.
    ok_stream = [
        ("lm_head.weight", q),  # excluded
        (
            "model.layers.3.block_sparse_moe.experts.0.w1.weight_scale",
            torch.randn(4),
        ),
        ("model.layers.4.input_layernorm.weight", torch.randn(8)),
    ]
    out = list(preprocess_fp8_pb_wo_weights(iter(ok_stream), config))
    assert [name for name, _ in out] == [name for name, _ in ok_stream]


def test_preprocess_dequant_route_passes_bf16_refit_weights():
    """bf16 refit streams re-send dequantized weights without scale sidecars."""
    from tokenspeed.runtime.layers.quantization.modelopt_mixed import (
        preprocess_fp8_pb_wo_weights,
    )

    config = _renamed_config(_FP8_PB_WO_LAYERS)
    refit = torch.randn(96, 200, dtype=torch.bfloat16)
    out = list(
        preprocess_fp8_pb_wo_weights(
            iter([("model.layers.4.self_attn.q_proj.weight", refit)]), config
        )
    )
    # Passed through unchanged, not buffered waiting for a scale.
    assert out == [("model.layers.4.self_attn.q_proj.weight", refit)]
    assert out[0][1] is refit


def test_preprocess_passthrough_without_fp8_pb_wo():
    from tokenspeed.runtime.layers.quantization.modelopt_mixed import (
        preprocess_fp8_pb_wo_weights,
    )

    stream = [("model.layers.3.self_attn.q_proj.weight", torch.randn(4, 4))]
    # No FP8_PB_WO entries (MiniMax-M3-style config) -> identity.
    assert list(preprocess_fp8_pb_wo_weights(iter(stream), _renamed_config())) == stream
    # Non-mixed configs (or None) -> identity.
    assert list(preprocess_fp8_pb_wo_weights(iter(stream), None)) == stream


def test_moe_weight_dtype_rejects_fp8_pb_wo_experts():
    config = _renamed_config(
        {"language_model.model.layers.5.mlp.experts.0.w1": {"quant_algo": "FP8_PB_WO"}}
    )
    with pytest.raises(ValueError, match="no MoE kernel path"):
        config.moe_weight_dtype("model.layers.5.mlp.experts")


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

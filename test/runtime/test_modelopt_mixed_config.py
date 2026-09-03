"""CPU-only coverage for the ModelOpt MIXED_PRECISION quantization config."""

from __future__ import annotations

import pytest
import tokenspeed_kernel
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


def test_w4a16_routing_rejects_unavailable_backend(monkeypatch):
    monkeypatch.setattr(
        tokenspeed_kernel,
        "has_flashinfer_cute_dsl_nvfp4_a16",
        lambda: False,
    )
    config = _renamed_config(
        {
            "language_model.model.layers.5.mlp.up_proj": {
                "quant_algo": "W4A16_NVFP4",
                "group_size": 16,
            }
        }
    )

    with pytest.raises(RuntimeError, match="SM100/SM103.*FlashInfer"):
        config.get_quant_method(
            torch.nn.Linear(1, 1),
            "model.layers.5.mlp.up_proj",
        )


def test_qwen35_w4a16_and_static_fp8_routing(monkeypatch):
    monkeypatch.setattr(
        tokenspeed_kernel,
        "has_flashinfer_cute_dsl_nvfp4_a16",
        lambda: True,
    )

    from tokenspeed.runtime.layers.dense import (
        Fp8LinearMethod,
        Nvfp4W4A16LinearMethod,
    )

    config = ModelOptMixedConfig.from_config(
        {
            "quant_algo": "MIXED_PRECISION",
            "quant_method": "modelopt",
            "ignore": ["mtp*", "mtp.layers.0*"],
            "quantized_layers": {
                "model.language_model.layers.0.linear_attn.in_proj_qkv": {
                    "quant_algo": "FP8"
                },
                "model.language_model.layers.0.linear_attn.in_proj_z": {
                    "quant_algo": "FP8"
                },
                "model.language_model.layers.0.mlp.gate_proj": {
                    "quant_algo": "W4A16_NVFP4",
                    "group_size": 16,
                },
                "model.language_model.layers.0.mlp.up_proj": {
                    "quant_algo": "W4A16_NVFP4",
                    "group_size": 16,
                },
            },
        }
    )
    # Qwen3_5ForConditionalGeneration declares no quant_module_name_replacements:
    # resolve_model keeps the "model.language_model" scope and attention layers
    # keep "self_attn", so runtime quant-lookup prefixes equal the checkpoint
    # quantized_layers keys verbatim. Apply nothing.

    assert config.exclude_modules == ["mtp*", "mtp.layers.0*"]
    assert config.group_size == 16
    assert config.fp8_static_config.activation_scheme == "static"
    assert config.fp8_static_config.weight_block_size is None
    assert isinstance(
        config.get_quant_method(
            torch.nn.Linear(1, 1),
            "model.language_model.layers.0.mlp.gate_up_proj",
        ),
        Nvfp4W4A16LinearMethod,
    )
    assert isinstance(
        config.get_quant_method(
            torch.nn.Linear(1, 1),
            "model.language_model.layers.0.linear_attn.in_proj_qkvz",
        ),
        Fp8LinearMethod,
    )
    assert config.is_quantized_layer(
        "model.language_model.layers.0.linear_attn.in_proj_qkv"
    )
    assert not config.is_quantized_layer(
        "model.language_model.layers.0.linear_attn.in_proj_b"
    )


# Layer 4 mimics a Kimi-K3 MLA layer, layer 6 a KDA layer (realistic
# checkpoint entry names; routing itself is purely per-leaf).
_FP8_PB_WO_LAYERS = {
    # MLA layer: everything is FP8-resident (w8a8 / verbatim fused reorder)
    "language_model.model.layers.4.self_attn.o_proj": {"quant_algo": "FP8_PB_WO"},
    "language_model.model.layers.4.self_attn.kv_b_proj": {"quant_algo": "FP8_PB_WO"},
    "language_model.model.layers.4.self_attn.q_b_proj": {"quant_algo": "FP8_PB_WO"},
    "language_model.model.layers.4.self_attn.q_a_proj": {"quant_algo": "FP8_PB_WO"},
    "language_model.model.layers.4.self_attn.kv_a_proj_with_mqa": {
        "quant_algo": "FP8_PB_WO"
    },
    "language_model.model.layers.4.self_attn.g_proj": {"quant_algo": "FP8_PB_WO"},
    "language_model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa": {
        "quant_algo": "FP8_PB_WO"
    },
    # KDA layer: FP8-resident except f_b (raw backend GEMV keeps dequant)
    "language_model.model.layers.6.self_attn.q_proj": {"quant_algo": "FP8_PB_WO"},
    "language_model.model.layers.6.self_attn.f_b_proj": {"quant_algo": "FP8_PB_WO"},
    "language_model.model.layers.6.self_attn.g_proj": {"quant_algo": "FP8_PB_WO"},
    "language_model.model.layers.6.self_attn.o_proj": {"quant_algo": "FP8_PB_WO"},
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
    mla = "model.layers.4.self_attn"
    kda = "model.layers.6.self_attn"
    # MLA-side projections stay FP8-resident (w8a8 / verbatim fused reorder).
    for leaf in (
        "o_proj",
        "kv_b_proj",
        "q_b_proj",
        "q_a_proj",
        "kv_a_proj_with_mqa",
        "fused_qkv_a_proj_with_mqa",
        "g_proj",  # w8a8 by leaf name; KDA/MLA are told apart by the
        # model loader (which runtime module exists), not by routing
    ):
        assert config.fp8_pb_wo_route(f"{mla}.{leaf}") == "w8a8", leaf
    # KDA merged-projection segments are FP8-resident too (w8a8 blockscale
    # GEMM in kimi3_qkvfab_projection); only f_b stays dequant (raw GEMV
    # inside the KDA-NaN-sensitive megafuse kernels, kept bf16 by decision).
    for leaf in ("q_proj", "g_proj", "o_proj"):
        assert config.fp8_pb_wo_route(f"{kda}.{leaf}") == "w8a8", leaf
    assert config.fp8_pb_wo_route(f"{kda}.f_b_proj") == "dequant"
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
    kda = "model.layers.6.self_attn"
    mla = "model.layers.4.self_attn"
    n, k = 96, 200  # ragged on both axes
    q_deq, s_deq = _quantize_per_block_cpu(torch.randn(n, k))
    q_w8a8, s_w8a8 = _quantize_per_block_cpu(torch.randn(256, 128))
    scale_4d = s_deq.reshape(s_deq.shape[0], 1, s_deq.shape[1], 1)
    other = torch.randn(8, dtype=torch.bfloat16)
    nvfp4_scale = torch.randn(4)

    stream = [
        # dequant module (KDA f_b): scale arrives BEFORE the weight
        (f"{kda}.f_b_proj.weight_scale", scale_4d),
        (f"{kda}.f_b_proj.weight", q_deq),
        # w8a8 module: fp8 weight passes through, scale renamed + squeezed
        (f"{mla}.o_proj.weight", q_w8a8),
        (
            f"{mla}.o_proj.weight_scale",
            s_w8a8.reshape(s_w8a8.shape[0], 1, s_w8a8.shape[1], 1),
        ),
        # untouched tensors: norms and NVFP4 expert scales keep their names
        ("model.layers.4.input_layernorm.weight", other),
        ("model.layers.3.block_sparse_moe.experts.0.w1.weight_scale", nvfp4_scale),
    ]
    out = dict(preprocess_fp8_pb_wo_weights(iter(stream), config))

    # Dequant path: bitwise-identical to the manual block dequant.
    dq = out[f"{kda}.f_b_proj.weight"].cpu()
    assert dq.dtype == torch.bfloat16
    s_full = s_deq.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)[:, :k]
    expected = (q_deq.float() * s_full).to(torch.bfloat16)
    assert torch.equal(dq, expected)
    assert f"{kda}.f_b_proj.weight_scale" not in out  # scale is consumed

    # w8a8 path: weight untouched, scale renamed to weight_scale_inv (2-D).
    assert out[f"{mla}.o_proj.weight"].dtype == torch.float8_e4m3fn
    assert f"{mla}.o_proj.weight_scale" not in out
    assert torch.equal(out[f"{mla}.o_proj.weight_scale_inv"], s_w8a8)

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
    stream = [("model.layers.6.self_attn.f_b_proj.weight", q)]
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


def test_guard_passes_real_ckpt_expert_subtree_names():
    """NVFP4 experts are declared as PARENT subtree entries in real exports.

    nvidia/Kimi-K3-NVFP4's hf_quant_config.json has no per-expert entries;
    it declares e.g. ``language_model.model.layers.49.block_sparse_moe
    .experts`` (entry names sampled verbatim from the checkpoint), which
    owns every ``...experts.<i>.w{1,2,3}`` tensor. The unrouted-tensor guard
    must honor such subtree entries — regression for a boot crash where the
    guard raised on ``model.layers.49.block_sparse_moe.experts.0.w1
    .weight_scale`` because the test fixture's unrealistic per-expert entry
    naming masked the missing ancestor resolution.
    """
    from tokenspeed.runtime.layers.quantization.modelopt_mixed import (
        preprocess_fp8_pb_wo_weights,
    )

    layers = dict(_FP8_PB_WO_LAYERS)  # keeps has_fp8_pb_wo True
    layers.update(
        {
            "language_model.model.layers.49.block_sparse_moe.experts": {
                "quant_algo": "NVFP4",
                "group_size": 16,
            },
            "language_model.model.layers.49.mlp.experts": {
                "quant_algo": "NVFP4",
                "group_size": 16,
            },
        }
    )
    config = _renamed_config(layers)
    packed_uint8 = torch.zeros(4, 4, dtype=torch.uint8)
    fp8_scales = torch.zeros(4, 4, dtype=torch.uint8).view(torch.float8_e4m3fn)
    stream = [
        (
            "model.layers.49.block_sparse_moe.experts.0.w1.weight_scale",
            fp8_scales,
        ),
        (
            "model.layers.49.block_sparse_moe.experts.383.w2.weight_scale",
            fp8_scales,
        ),
        ("model.layers.49.block_sparse_moe.experts.0.w1.weight", packed_uint8),
    ]
    out = list(preprocess_fp8_pb_wo_weights(iter(stream), config))
    assert [name for name, _ in out] == [name for name, _ in stream]


def test_preprocess_w8a8_route_rejects_bf16_refit_weights():
    """16-bit weights for FP8-resident modules must fail fast, not corrupt.

    A bf16 refit stream hitting a w8a8-routed module would otherwise be
    raw-copied into the FP8 parameter (or the fused buffer at
    checkpoint-canonical offsets) with silently wrong values.
    """
    from tokenspeed.runtime.layers.quantization.modelopt_mixed import (
        preprocess_fp8_pb_wo_weights,
    )

    config = _renamed_config(_FP8_PB_WO_LAYERS)
    refit = torch.randn(96, 200, dtype=torch.bfloat16)
    for module in (
        "model.layers.4.self_attn.q_a_proj",  # fused segment
        "model.layers.4.self_attn.q_b_proj",  # plain LinearBase w8a8
        "model.layers.6.self_attn.q_proj",  # KDA merged segment
    ):
        with pytest.raises(TypeError, match="bf16 refit"):
            list(
                preprocess_fp8_pb_wo_weights(
                    iter([(f"{module}.weight", refit)]), config
                )
            )


def test_guard_fp8_pb_wo_ancestor_does_not_own_children():
    """Only NVFP4/MXFP8 subtree entries legitimize unrouted descendants.

    FP8_PB_WO entries name leaf projections; an unrouted FP8 tensor under an
    FP8_PB_WO ancestor is itself checkpoint/runtime name drift and must keep
    raising.
    """
    from tokenspeed.runtime.layers.quantization.modelopt_mixed import (
        preprocess_fp8_pb_wo_weights,
    )

    config = _renamed_config(_FP8_PB_WO_LAYERS)
    q, _ = _quantize_per_block_cpu(torch.randn(128, 128))
    with pytest.raises(RuntimeError, match="module-name drift"):
        list(
            preprocess_fp8_pb_wo_weights(
                iter([("model.layers.4.self_attn.o_proj.child.weight", q)]),
                config,
            )
        )


def test_fused_qkv_a_fp8_decision():
    """FP8 layout engages from the alias OR the segments; partial raises."""
    from tokenspeed.runtime.models.kimi_k3 import _fused_qkv_a_uses_fp8

    fp8 = {"quant_algo": "FP8_PB_WO"}
    layer = "language_model.model.layers.7.self_attn"
    prefix = "model.layers.7.self_attn"
    # Segment-only config (no fused alias) -> FP8 layout.
    seg_only = _renamed_config(
        {
            f"{layer}.q_a_proj": fp8,
            f"{layer}.kv_a_proj_with_mqa": fp8,
            f"{layer}.g_proj": fp8,
        }
    )
    assert _fused_qkv_a_uses_fp8(seg_only, prefix) is True
    # Alias-only config -> FP8 layout (checkpoint aliases without segments).
    alias_only = _renamed_config({f"{layer}.fused_qkv_a_proj_with_mqa": fp8})
    assert _fused_qkv_a_uses_fp8(alias_only, prefix) is True
    # Partially quantized segments -> loud error, not a later alignment trap.
    partial = _renamed_config({f"{layer}.q_a_proj": fp8})
    with pytest.raises(ValueError, match="partially"):
        _fused_qkv_a_uses_fp8(partial, prefix)
    # Non-mixed configs (bf16/mxfp4) -> bf16 layout.
    assert _fused_qkv_a_uses_fp8(None, prefix) is False
    # Mixed config without any FP8_PB_WO entry for this layer -> bf16 layout.
    assert _fused_qkv_a_uses_fp8(_renamed_config(), prefix) is False


def test_preprocess_dequant_route_passes_bf16_refit_weights():
    """bf16 refit streams re-send dequantized weights without scale sidecars."""
    from tokenspeed.runtime.layers.quantization.modelopt_mixed import (
        preprocess_fp8_pb_wo_weights,
    )

    config = _renamed_config(_FP8_PB_WO_LAYERS)
    refit = torch.randn(96, 200, dtype=torch.bfloat16)
    out = list(
        preprocess_fp8_pb_wo_weights(
            iter([("model.layers.6.self_attn.f_b_proj.weight", refit)]), config
        )
    )
    # Passed through unchanged, not buffered waiting for a scale.
    assert out == [("model.layers.6.self_attn.f_b_proj.weight", refit)]
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

"""CPU coverage for mixed-precision compressed-tensors W4A8.

GLM-5.3 W4A8 stores routed experts as packed INT4 group-128 + dynamic FP8
activations and attention / shared experts as block-FP8. TokenSpeed used to
drop ``input_activations`` whenever the top-level format was not an activation
format, always read ``target_scheme_map["Linear"]`` for MoE kernel selection,
and had no ``get_quant_method`` for compressed-tensors linears.
"""

from __future__ import annotations

import os
import sys
from types import ModuleType

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

# Importing TokenSpeed quantization configs loads ``tokenspeed_kernel`` via
# ``quantization/__init__.py``. That package is optional for these CPU tests.
try:
    import tokenspeed_kernel  # noqa: F401
except ImportError:
    _kernel = ModuleType("tokenspeed_kernel")
    _platform = ModuleType("tokenspeed_kernel.platform")
    _platform.current_platform = lambda: None
    sys.modules["tokenspeed_kernel"] = _kernel
    sys.modules["tokenspeed_kernel.platform"] = _platform

import pytest
import torch

from tokenspeed.runtime.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)

W4A8_TARGET = "re:.*\\.mlp\\.experts\\.\\d+\\.(gate|up|down)_proj$"
FP8_BLOCK_TARGET = "re:.*\\.mlp\\.shared_experts\\.(gate|up|down)_proj$"
ATTN_TARGET = "re:.*\\.self_attn\\.(q_a_proj|o_proj)$"

W4A8_GROUP = {
    "format": "pack-quantized",
    "targets": [W4A8_TARGET],
    "weights": {
        "num_bits": 4,
        "type": "int",
        "symmetric": True,
        "strategy": "group",
        "group_size": 128,
        "dynamic": False,
    },
    "input_activations": {
        "num_bits": 8,
        "type": "float",
        "symmetric": True,
        "strategy": "token",
        "dynamic": True,
    },
}

FP8_BLOCK_GROUP = {
    "format": "float-quantized",
    "targets": [FP8_BLOCK_TARGET, ATTN_TARGET],
    "weights": {
        "num_bits": 8,
        "type": "float",
        "symmetric": True,
        "strategy": "block",
        "block_structure": [128, 128],
        "dynamic": False,
    },
    "input_activations": {
        "num_bits": 8,
        "type": "float",
        "symmetric": True,
        "strategy": "group",
        "group_size": 128,
        "dynamic": True,
    },
}

MXINT4_LINEAR_GROUP = {
    "format": "pack-quantized",
    "targets": ["Linear"],
    "weights": {
        "num_bits": 4,
        "type": "int",
        "symmetric": True,
        "strategy": "group",
        "group_size": 32,
        "dynamic": False,
    },
    "input_activations": None,
}


def _config(*groups, ignore=(), top_level_format="mixed-precision"):
    return {
        "quant_method": "compressed-tensors",
        "format": top_level_format,
        "config_groups": {f"group_{i}": g for i, g in enumerate(groups)},
        "ignore": list(ignore),
    }


def _glm53_w4a8_config(**kwargs):
    """Match the published GLM-5.3 W4A8 card: top-level pack-quantized + Linear."""
    group_0 = dict(
        W4A8_GROUP,
        targets=["Linear", W4A8_TARGET],
    )
    return _config(
        group_0, FP8_BLOCK_GROUP, top_level_format="pack-quantized", **kwargs
    )


def test_input_activations_survive_mixed_precision_top_level():
    quant_config = CompressedTensorsConfig.from_config(
        _config(W4A8_GROUP, FP8_BLOCK_GROUP)
    )
    expert = quant_config.target_scheme_map[W4A8_TARGET]
    assert expert["format"] == "pack-quantized"
    assert expert["input_activations"] is not None
    assert expert["input_activations"].num_bits == 8
    assert expert["weights"].group_size == 128

    fp8 = quant_config.target_scheme_map[FP8_BLOCK_TARGET]
    assert fp8["format"] == "float-quantized"
    assert fp8["input_activations"] is not None
    assert fp8["weights"].block_structure == [128, 128]


def test_input_activations_survive_pack_quantized_top_level():
    quant_config = CompressedTensorsConfig.from_config(_glm53_w4a8_config())
    linear = quant_config.target_scheme_map["Linear"]
    assert linear["input_activations"] is not None
    assert linear["input_activations"].num_bits == 8
    fp8 = quant_config.target_scheme_map[ATTN_TARGET]
    assert fp8["input_activations"] is not None


def test_w4a8_moe_dtype_reads_matched_expert_group_not_linear():
    quant_config = CompressedTensorsConfig.from_config(
        _config(W4A8_GROUP, FP8_BLOCK_GROUP)
    )
    assert "Linear" not in quant_config.target_scheme_map
    assert quant_config.is_w4a8_fp8 is True
    assert quant_config.moe_weight_dtype("model.layers.3.mlp") == "w4a8"
    assert quant_config.moe_group_size("model.layers.3.mlp") == 128
    assert quant_config.weight_block_size == [128, 128]


def test_glm53_card_selects_w4a8_even_when_linear_is_int4():
    quant_config = CompressedTensorsConfig.from_config(_glm53_w4a8_config())
    assert quant_config.is_w4a8_fp8 is True
    assert quant_config.moe_weight_dtype("model.layers.3.mlp") == "w4a8"
    assert quant_config.weight_block_size == [128, 128]


def test_weight_block_size_ignores_non_block_linear_target():
    groups = (dict(W4A8_GROUP, targets=["Linear"]), FP8_BLOCK_GROUP)
    quant_config = CompressedTensorsConfig.from_config(
        _config(*groups, top_level_format="mixed-precision")
    )
    assert "Linear" in quant_config.target_scheme_map
    assert quant_config.weight_block_size == [128, 128]


def test_kimi_int4_group32_still_selects_mxint4():
    quant_config = CompressedTensorsConfig.from_config(
        _config(MXINT4_LINEAR_GROUP, top_level_format="pack-quantized")
    )
    assert quant_config.is_w4a8_fp8 is False
    assert quant_config.moe_weight_dtype("model.layers.3.mlp") == "mxint4"


def test_block_fp8_linears_match_fp8_scheme_not_w4a8():
    quant_config = CompressedTensorsConfig.from_config(_glm53_w4a8_config())
    attn = quant_config._scheme_dict_for_name("model.layers.0.self_attn.q_a_proj")
    assert attn is not None
    assert quant_config._is_fp8_w8a8(attn["weights"], attn["input_activations"])
    assert not quant_config._is_wint4afp8(
        attn["weights"], attn["input_activations"], attn.get("format")
    )

    shared = quant_config._scheme_dict_for_name(
        "model.layers.3.mlp.shared_experts.gate_proj"
    )
    assert shared is not None
    assert quant_config._is_fp8_w8a8(shared["weights"], shared["input_activations"])


def test_dense_w4a8_linear_is_rejected():
    quant_config = CompressedTensorsConfig.from_config(_glm53_w4a8_config())
    expert = quant_config._scheme_dict_for_name(
        "model.layers.3.mlp.experts.0.gate_proj"
    )
    assert expert is not None
    assert quant_config._is_wint4afp8(
        expert["weights"], expert["input_activations"], expert.get("format")
    )
    with pytest.raises(NotImplementedError, match="Dense W4A8"):
        quant_config.get_quant_method(
            torch.nn.Linear(8, 8),
            "model.layers.3.mlp.experts.0.gate_proj",
        )


def test_unpack_int4_dequant_matches_group_scale():
    try:
        from tokenspeed_kernel.ops.moe.flashinfer.w4a8 import (
            dequant_cutlass_int4,
            unpack_int32_uint4b8_to_cutlass_int8,
        )
    except Exception:
        pytest.skip("tokenspeed_kernel W4A8 helpers require a native kernel install")

    group_size = 128
    n, k = 4, 256
    values = torch.arange(n * k, dtype=torch.int32).remainder(16) - 8
    values = values.view(n, k)
    nibbles = (values + 8).view(n, k // 8, 8) & 0x0F
    packed_words = torch.zeros(n, k // 8, dtype=torch.int32)
    for i in range(8):
        packed_words |= nibbles[:, :, i] << (4 * i)
    packed = packed_words.unsqueeze(0)
    scale = torch.full((1, n, k // group_size), 0.5, dtype=torch.bfloat16)

    cutlass = unpack_int32_uint4b8_to_cutlass_int8(packed)[0]
    restored = dequant_cutlass_int4(cutlass, scale[0], torch.float32)
    expected = values.to(torch.float32) * 0.5
    torch.testing.assert_close(restored, expected, atol=0, rtol=0)

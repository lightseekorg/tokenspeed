# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.layers.moe.weights import create_layer_weights
from tokenspeed.runtime.layers.moe.weights.int4 import (
    create_int4_weight_pair,
    flatquant_factors,
    validate_int4_weights_loaded,
)
from tokenspeed.runtime.layers.quantization.compressed_tensors import (
    compressed_tensors as ct,
)


@pytest.mark.parametrize("transposed", (False, True))
@pytest.mark.parametrize("strategy", ("channel", "tensor"))
def test_expert_shards_and_required_flatquant_parameters(transposed, strategy):
    layer = torch.nn.Module()
    spec = SimpleNamespace(
        tp_size=1, tp_rank=0, hidden_size=32, intermediate_size=64, num_local_experts=2
    )
    create_int4_weight_pair(spec, layer, transposed=transposed, scale_strategy=strategy)
    for expert in range(2):
        for shard, name in (("w1", "w13"), ("w3", "w13"), ("w2", "w2")):
            param = getattr(layer, f"{name}_weight")
            shape = list(param.shape[1:])
            if shard != "w2":
                shape[0] //= 2
            value = 10 * expert + {"w1": 1, "w3": 3, "w2": 2}[shard]
            param.weight_loader(
                param, torch.full(shape, value, dtype=torch.int32), shard, expert
            )
            scale = getattr(layer, f"{name}_weight_scale")
            size = 32 if shard == "w2" else 64
            values = torch.full((1 if strategy == "tensor" else size, 1), float(value))
            scale.weight_loader(scale, values, shard, expert)
    half = layer.w13_weight.shape[1] // 2
    assert torch.all(layer.w13_weight[0, :half] == 1)
    assert torch.all(layer.w13_weight[0, half:] == 3)
    assert torch.all(layer.w13_weight[1, :half] == 11)
    assert torch.all(layer.w13_weight_scale[1, 64:] == 13)
    with pytest.raises(ValueError, match="Missing FlatQuant"):
        validate_int4_weights_loaded(layer)
    for module in (
        layer.w1_trans,
        layer.w2_trans,
        layer.act_quantizer_w1,
        layer.act_quantizer_w2,
    ):
        for param in module.parameters():
            param.weight_loader(param, torch.ones_like(param))
    validate_int4_weights_loaded(layer)
    with pytest.raises(ValueError, match="Duplicate"):
        layer.w2_weight.weight_loader(
            layer.w2_weight, torch.zeros_like(layer.w2_weight[0]), "w2", 0
        )
    assert flatquant_factors(512) == (16, 32)


def _config(bits, *, input_symmetric=True, input_dynamic=True):
    return ct.CompressedTensorsConfig.from_config(
        {
            "format": "int-quantized",
            "config_groups": {
                "routed": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": bits,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "channel",
                        "dynamic": False,
                    },
                    "input_activations": {
                        "num_bits": bits,
                        "type": "int",
                        "symmetric": input_symmetric,
                        "strategy": "token",
                        "dynamic": input_dynamic,
                    },
                }
            },
        }
    )


def test_quant_config_int4_int8_and_gpu_boundary(monkeypatch):
    monkeypatch.setattr(ct, "current_platform", lambda: SimpleNamespace(is_npu=True))
    assert _config(4).moe_weight_dtype() == "int4"
    assert _config(8).moe_weight_dtype() == "int8"
    for kwargs in ({"input_symmetric": False}, {"input_dynamic": False}):
        with pytest.raises(ValueError, match="unsupported"):
            _config(4, **kwargs).moe_weight_dtype()
    # MoE-only W4A4 must not override the shared Linear scheme.
    config = _config(8)
    config.target_scheme_map["Moe"] = _config(4).target_scheme_map["Linear"]
    assert config.moe_weight_dtype() == "int4"
    assert config.target_scheme_map["Linear"]["weights"].num_bits == 8
    monkeypatch.setattr(ct, "current_platform", lambda: SimpleNamespace(is_npu=False))
    with pytest.raises(ValueError, match="unsupported"):
        _config(4).moe_weight_dtype()


def test_w4a4_plan_selects_only_the_fused_kernel():
    pytest.importorskip("torch_npu")
    import tokenspeed_kernel

    if not torch.npu.is_available():
        pytest.skip("requires Ascend registration")
    plan = tokenspeed_kernel.moe_plan(
        "int4",
        input_dtype=torch.bfloat16,
        activation="swiglu",
        routing_mode="precomputed_topk",
        a2a_backend="none",
        ep_size=16,
        ispp=1024,
        hidden=512,
        swiglu_form="standard",
        activation_clamped=False,
        expert_id_repeats=True,
        internal_activation_dtype="int4",
        with_bias=False,
        solution="flash_npu_chain",
        fast_math=True,
        combine_order="rank",
        features=frozenset(),
    )
    assert plan["apply_kernel_name"] == "ascend_int4_precomputed_moe_apply"
    assert plan["solution"] == "flash_npu_chain"
    assert plan["supports_precomputed_topk"]
    assert not plan["supports_deferred_finalize"]
    assert not plan["supports_all_to_all_ep"]


def test_shared_weight_factory_keeps_flatquant_separate_from_smoothquant():
    config = _config(4)
    spec = SimpleNamespace(
        gated=True,
        tp_size=1,
        tp_rank=0,
        hidden_size=64,
        intermediate_size=128,
        num_local_experts=2,
    )
    layer = torch.nn.Module()
    create_layer_weights(spec, layer, "int4", config, solution="flash_npu_chain")
    assert layer.w13_weight.shape == (2, 32, 64)
    assert layer.w13_weight_scale.dtype == torch.float32
    assert layer.w1_trans.matrix_left.shape == (8, 8)
    config.config["moe_enable_smooth_quant"] = True
    with pytest.raises(ValueError, match="SmoothQuant"):
        create_layer_weights(
            spec, torch.nn.Module(), "int4", config, solution="flash_npu_chain"
        )

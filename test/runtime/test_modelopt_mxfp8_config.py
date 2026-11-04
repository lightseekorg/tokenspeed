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

"""ModelOpt ``quant_algo: MXFP8`` checkpoints: detection, config, scale names."""

import json
import os
import sys
from types import SimpleNamespace

import pytest
import torch

# CPU-only test scheduled in runtime-1gpu because it imports the full runtime.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.configs.load_config import LoadConfig  # noqa: E402
from tokenspeed.runtime.configs.model_config import ModelConfig  # noqa: E402
from tokenspeed.runtime.layers.dense import UnquantizedLinearMethod  # noqa: E402
from tokenspeed.runtime.layers.dense.fp8 import Fp8LinearMethod  # noqa: E402
from tokenspeed.runtime.layers.linear import ReplicatedLinear  # noqa: E402
from tokenspeed.runtime.layers.quantization.fp8 import (  # noqa: E402
    Fp8Config,
    Mxfp8Config,
    preprocess_modelopt_mxfp8_weights,
)
from tokenspeed.runtime.model_loader.weight_utils import (  # noqa: E402
    get_quant_config,
)

EXCLUDED = [
    "lm_head",
    "model.layers.*.mlp.gate",
    "model.layers.0.self_attn.o_proj",
    "model.layers.2.mlp.experts*",
]


def _model_config(tmp_path, source: str, quantization: str | None = None):
    """A ModelOpt MXFP8 export whose quantization fields live in ``source``."""
    hf_config = SimpleNamespace(text_config=None)
    if source == "hf_quant_config.json":
        quantization_fields = {
            "quant_algo": "MXFP8",
            "kv_cache_quant_algo": None,
            "exclude_modules": EXCLUDED,
        }
        (tmp_path / "hf_quant_config.json").write_text(
            json.dumps(
                {"producer": {"name": "modelopt"}, "quantization": quantization_fields}
            )
        )
    else:
        hf_config.quantization_config = {
            "quant_method": "modelopt",
            "quant_algo": "MXFP8",
            "ignore": EXCLUDED,
        }
    model_config = object.__new__(ModelConfig)
    model_config.hf_config = hf_config
    model_config.model_path = str(tmp_path)
    model_config.revision = None
    model_config.quantization = quantization
    return model_config


@pytest.mark.parametrize("source", ["hf_quant_config.json", "config.json"])
@pytest.mark.parametrize("quantization", [None, "mxfp8"])
def test_modelopt_mxfp8_loads_as_serialized_mxfp8(tmp_path, source, quantization):
    model_config = _model_config(tmp_path, source, quantization)

    model_config._verify_quantization()
    config = get_quant_config(model_config, LoadConfig())

    assert model_config.quantization == "mxfp8"
    assert type(config) is Mxfp8Config
    assert config.is_checkpoint_fp8_serialized
    assert config.activation_scheme == "dynamic"
    assert config.weight_block_size == [1, 32]
    assert config.scale_fmt == "ue8m0"
    assert config.exclude_modules == EXCLUDED


def test_another_user_quantization_still_conflicts(tmp_path):
    model_config = _model_config(tmp_path, "hf_quant_config.json", "fp8")

    with pytest.raises(ValueError, match="does not match"):
        model_config._verify_quantization()


@pytest.mark.parametrize(
    "hf_quant_cfg",
    [
        {"quant_method": "modelopt", "quant_algo": "NVFP4"},
        {"quantization": {"quant_algo": "MIXED_PRECISION"}},
        {"quant_method": "mxfp8", "activation_scheme": "dynamic"},
    ],
)
def test_other_checkpoints_are_not_claimed(hf_quant_cfg):
    assert Mxfp8Config.override_quantization_method(hf_quant_cfg, None) is None


def _modelopt_mxfp8_config() -> Mxfp8Config:
    return Mxfp8Config.from_config(
        {"quantization": {"quant_algo": "MXFP8", "exclude_modules": EXCLUDED}}
    )


def _linear(config: Mxfp8Config, prefix: str) -> ReplicatedLinear:
    return ReplicatedLinear(128, 64, bias=False, quant_config=config, prefix=prefix)


def test_excluded_modules_stay_unquantized():
    config = _modelopt_mxfp8_config()

    excluded = _linear(config, "model.layers.0.self_attn.o_proj")
    gate = _linear(config, "model.layers.3.mlp.gate")
    quantized = _linear(config, "model.layers.1.self_attn.o_proj")

    assert isinstance(excluded.quant_method, UnquantizedLinearMethod)
    assert isinstance(gate.quant_method, UnquantizedLinearMethod)
    assert isinstance(quantized.quant_method, Fp8LinearMethod)
    assert quantized.weight_scale_inv.dtype == torch.uint8
    assert quantized.weight_scale_inv.shape == (64, 128 // 32)


def test_e8m0_weight_scales_load_as_weight_scale_inv():
    config = _modelopt_mxfp8_config()
    layer = _linear(config, "model.layers.1.self_attn.o_proj")
    exponents = torch.randint(110, 140, (64, 4), dtype=torch.int32).to(torch.uint8)
    e8m0 = exponents.view(torch.float8_e8m0fnu)
    nvfp4_scale = torch.ones(64, 8, dtype=torch.float8_e4m3fn)
    per_tensor_scale = torch.ones((), dtype=torch.float32)
    weight = torch.zeros(64, 128, dtype=torch.float8_e4m3fn)

    out = list(
        preprocess_modelopt_mxfp8_weights(
            [
                ("model.layers.1.self_attn.o_proj.weight", weight),
                ("model.layers.1.self_attn.o_proj.weight_scale", e8m0),
                ("model.layers.2.self_attn.o_proj.weight_scale", exponents),
                ("model.layers.3.mlp.experts.0.w1.weight_scale", nvfp4_scale),
                ("model.layers.4.self_attn.o_proj.weight_scale", per_tensor_scale),
            ],
            config,
        )
    )

    names = [name for name, _ in out]
    assert names == [
        "model.layers.1.self_attn.o_proj.weight",
        "model.layers.1.self_attn.o_proj.weight_scale_inv",
        "model.layers.2.self_attn.o_proj.weight_scale_inv",
        "model.layers.3.mlp.experts.0.w1.weight_scale",
        "model.layers.4.self_attn.o_proj.weight_scale",
    ]
    for _, scale in out[1:3]:
        assert scale.dtype == torch.uint8 and torch.equal(scale, exponents)
    assert out[0][1] is weight
    assert out[3][1] is nvfp4_scale
    assert out[4][1] is per_tensor_scale

    param = layer.weight_scale_inv
    param.weight_loader(param, out[1][1])
    assert torch.equal(param, exponents)


def test_other_configs_keep_their_weight_scale_names():
    config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[128, 128],
    )
    stream = [
        (
            "model.layers.0.self_attn.o_proj.weight_scale",
            torch.full((1, 1), 127, dtype=torch.uint8),
        )
    ]

    assert list(preprocess_modelopt_mxfp8_weights(iter(stream), config)) == stream
    assert list(preprocess_modelopt_mxfp8_weights(iter(stream), None)) == stream


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))

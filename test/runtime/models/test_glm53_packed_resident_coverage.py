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

"""Coverage of GLM-5.3-Flash's audited packed block-FP8 projections."""

import pytest
import torch

from tokenspeed.runtime.configs.glm53_flash_config import Glm53FlashTextConfig
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.dense.fp8 import Fp8LinearMethod
from tokenspeed.runtime.layers.linear import LinearBase
from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config
from tokenspeed.runtime.models.glm53_flash import Glm53FlashDecoderLayer
from tokenspeed.runtime.utils.env import global_server_args_dict


@pytest.fixture
def model_dtype():
    previous = torch.get_default_dtype()
    try:
        yield torch.set_default_dtype
    finally:
        torch.set_default_dtype(previous)


def _glm_quantized_tp4(monkeypatch):
    config = Glm53FlashTextConfig()
    mapping = Mapping(rank=0, world_size=4)
    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        ignored_layers=["re:.*self_attn.fused_qkvbfg_a_proj"],
        weight_block_size=[128, 128],
    )
    monkeypatch.setitem(global_server_args_dict, "mapping", mapping)
    monkeypatch.setitem(global_server_args_dict, "attention_backend", "dsa")
    monkeypatch.setitem(global_server_args_dict, "dense_gemm_backend", "triton")
    monkeypatch.setitem(global_server_args_dict, "ep_num_redundant_experts", 0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # Construction needs only the expert-weight layout, not an executable
    # MoE kernel; the host's GPU architecture may not support this model.
    monkeypatch.setattr(
        "tokenspeed_kernel.moe_plan",
        lambda *_args, **_kwargs: {"solution": "triton", "support_routing": False},
    )
    return config, mapping, quant_config


def test_tp4_packed_resident_projection_coverage(monkeypatch, model_dtype) -> None:
    model_dtype(torch.bfloat16)
    config, mapping, quant_config = _glm_quantized_tp4(monkeypatch)

    # The production layout has 45 target layers (three dense, then 42 MoE)
    # and one sparse-attention draft layer at checkpoint prefix 45.
    expected_target = {
        f"model.layers.{layer}.mlp.{name}": shape
        for layer in range(3)
        for name, shape in {
            "gate_up_proj": (6144, 4096),
            "down_proj": (4096, 3072),
        }.items()
    }
    expected_target.update(
        {
            f"model.layers.{layer}.mlp.shared_experts.{name}": shape
            for layer in range(3, 45)
            for name, shape in {
                "gate_up_proj": (1024, 4096),
                "down_proj": (4096, 512),
            }.items()
        }
    )
    attention_shapes = {
        "fused_qkv_a_proj_with_mqa": (2048, 4096),
        "q_b_proj": (4096, 1536),
        "o_proj": (4096, 4096),
    }
    expected_target.update(
        {
            f"model.layers.{layer}.self_attn.{name}": shape
            for layer in range(3, 45, 4)
            for name, shape in attention_shapes.items()
        }
    )
    expected_draft = {
        f"model.layers.45.mlp.shared_experts.{name}": shape
        for name, shape in {
            "gate_up_proj": (1024, 4096),
            "down_proj": (4096, 512),
        }.items()
    }
    expected_draft.update(
        {
            f"model.layers.45.self_attn.{name}": shape
            for name, shape in attention_shapes.items()
        }
    )
    assert len(expected_target) == 123
    assert len(expected_draft) == 5

    marked_target: dict[str, tuple[int, ...]] = {}
    marked_draft: dict[str, tuple[int, ...]] = {}
    excluded_fp8: set[str] = set()
    with torch.device("meta"):
        for layer_id in range(46):
            is_nextn = layer_id == 45
            layer = Glm53FlashDecoderLayer(
                config=config,
                layer_id=0 if is_nextn else layer_id,
                mapping=mapping,
                quant_config=quant_config,
                prefix=f"model.layers.{layer_id}",
                is_nextn=is_nextn,
            )
            for module in layer.modules():
                if not isinstance(module, LinearBase) or not isinstance(
                    module.quant_method, Fp8LinearMethod
                ):
                    continue
                if module.quant_method.packed_resident_requested:
                    marked = marked_draft if is_nextn else marked_target
                    marked[module.prefix] = tuple(module.weight.shape)
                elif module.prefix.endswith(
                    ("self_attn.kv_b_proj", "self_attn.f_b_proj", "indexer.wq_b")
                ):
                    excluded_fp8.add(module.prefix)

    assert marked_target == expected_target
    assert marked_draft == expected_draft
    assert {
        "model.layers.0.self_attn.f_b_proj",
        "model.layers.3.self_attn.kv_b_proj",
        "model.layers.3.self_attn.indexer.wq_b",
        "model.layers.45.self_attn.kv_b_proj",
    } <= excluded_fp8

    # Online quantization replaces checkpoint weight wrappers with ordinary
    # Parameters, so those layers must never enter the packed-resident path.
    online_config = Fp8Config(is_checkpoint_fp8_serialized=False)
    with torch.device("meta"):
        online_layer = Glm53FlashDecoderLayer(
            config=config,
            layer_id=0,
            mapping=mapping,
            quant_config=online_config,
            prefix="model.layers.0",
        )
    assert all(
        not module.quant_method.packed_resident_requested
        for module in online_layer.modules()
        if isinstance(module, LinearBase)
        and isinstance(module.quant_method, Fp8LinearMethod)
    )


def test_fp16_opt_in_does_not_mark_packed_projections(monkeypatch, model_dtype) -> None:
    model_dtype(torch.float16)
    monkeypatch.setenv("TOKENSPEED_EXPERIMENTAL_GLUON_FP8_BLOCKSCALE", "1")
    config, mapping, quant_config = _glm_quantized_tp4(monkeypatch)
    with torch.device("meta"):
        decoder = Glm53FlashDecoderLayer(
            config=config,
            layer_id=3,
            mapping=mapping,
            quant_config=quant_config,
            prefix="model.layers.3",
        )

    assert decoder.input_layernorm.weight.dtype == torch.float16
    eligible = [
        module
        for module in decoder.modules()
        if isinstance(module, LinearBase)
        and isinstance(module.quant_method, Fp8LinearMethod)
        and module.quant_method.block_quant
        and module.quant_method.quant_config.is_checkpoint_fp8_serialized
    ]
    assert eligible
    assert all(not module.quant_method.packed_resident_requested for module in eligible)

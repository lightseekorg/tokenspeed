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

"""Target-only online FP8 projections preserve draft checkpoint quantization."""

import pytest
import torch

from tokenspeed.runtime.configs.qwen3_5_config import Qwen3_5TextConfig
from tokenspeed.runtime.configs.qwen4_exp_config import Qwen4ExpTextConfig
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.dense.fp8 import Fp8LinearMethod
from tokenspeed.runtime.layers.dense.unquant import UnquantizedLinearMethod
from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config
from tokenspeed.runtime.layers.quantization.modelopt_mixed import ModelOptMixedConfig
from tokenspeed.runtime.layers.quantization.nvfp4 import Nvfp4Config
from tokenspeed.runtime.models.qwen3_5 import Qwen3_5AttentionDecoderLayer
from tokenspeed.runtime.models.qwen3_5_nextn import Qwen3_5DraftAttentionDecoderLayer
from tokenspeed.runtime.models.qwen4_exp import Qwen4ExpAttentionDecoderLayer
from tokenspeed.runtime.models.qwen4_exp_nextn import Qwen4ExpDraftAttentionDecoderLayer
from tokenspeed.runtime.utils.env import global_server_args_dict


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("checkpoint", ["bf16", "nvfp4", "fp8", "mixed"])
@pytest.mark.parametrize(
    "config_cls,layer_cls,draft",
    [
        (Qwen3_5TextConfig, Qwen3_5AttentionDecoderLayer, False),
        (Qwen3_5TextConfig, Qwen3_5DraftAttentionDecoderLayer, True),
        (Qwen4ExpTextConfig, Qwen4ExpAttentionDecoderLayer, False),
        (Qwen4ExpTextConfig, Qwen4ExpDraftAttentionDecoderLayer, True),
    ],
)
def test_draft_keeps_checkpoint_projections(
    monkeypatch, enabled, checkpoint, config_cls, layer_cls, draft
):
    monkeypatch.setitem(global_server_args_dict, "enable_fp8_attn_proj", enabled)
    config = config_cls(
        hidden_size=512,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        max_position_embeddings=128,
        layer_types=["full_attention"],
        num_experts=None,
    )
    config.dtype = torch.bfloat16
    mixed = ModelOptMixedConfig(
        quantized_layers={
            f"{root}.mlp.{leaf}": "FP8"
            for root in ("model.layers.0", "mtp.layers.0")
            for leaf in ("gate_proj", "up_proj", "down_proj")
        }
    )
    quant = {
        "mixed": mixed,
        "bf16": None,
        "nvfp4": Nvfp4Config(exclude_modules=["*"]),
        "fp8": Fp8Config(
            is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128]
        ),
    }[checkpoint]
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            layer = layer_cls(
                config,
                Mapping(rank=0, world_size=1),
                0,
                quant_config=quant,
                prefix=(
                    "mtp.layers.0.self_attn" if draft else "model.layers.0.self_attn"
                ),
            )
        expected = (
            Fp8LinearMethod
            if checkpoint == "fp8" or (enabled and not draft)
            else UnquantizedLinearMethod
        )
        assert isinstance(layer.qkv_proj.quant_method, expected)
        assert isinstance(layer.o_proj.quant_method, expected)
        if checkpoint == "mixed":
            assert isinstance(layer.mlp.gate_up_proj.quant_method, Fp8LinearMethod)
        if expected is UnquantizedLinearMethod:
            assert layer.qkv_proj.weight.dtype == torch.bfloat16
            assert layer.o_proj.weight.dtype == torch.bfloat16
    finally:
        torch.set_default_dtype(old_dtype)

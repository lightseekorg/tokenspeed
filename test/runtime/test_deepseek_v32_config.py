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

import json
from types import SimpleNamespace

from tokenspeed.runtime.configs import get_config_class
from tokenspeed.runtime.configs.deepseek_v32_config import DeepseekV32Config
from tokenspeed.runtime.configs.utils import get_config
from tokenspeed.runtime.model_loader.utils import get_model_architecture


def test_deepseek_v32_config_is_registered() -> None:
    assert get_config_class("deepseek_v32") is DeepseekV32Config


def test_deepseek_v32_defaults() -> None:
    config = DeepseekV32Config()

    assert config.model_type == "deepseek_v32"
    assert config.vocab_size == 129280
    assert config.hidden_size == 7168
    assert config.intermediate_size == 18432
    assert config.moe_intermediate_size == 2048
    assert config.num_hidden_layers == 61
    assert config.num_attention_heads == 128
    assert config.num_key_value_heads == 128
    assert config.q_lora_rank == 1536
    assert config.qk_rope_head_dim == 64
    assert config.v_head_dim == 128
    assert config.qk_nope_head_dim == 128
    assert config.n_group == 8
    assert config.topk_group == 4
    assert config.max_position_embeddings == 163840
    assert config.rms_norm_eps == 1e-6
    assert config.n_routed_experts == 256
    assert config.n_shared_experts == 1
    assert config.index_topk == 2048
    assert config.index_head_dim == 128
    assert config.index_n_heads == 64
    assert config.mlp_bias is False


def test_deepseek_v32_derived_fields() -> None:
    config = DeepseekV32Config()

    assert config.qk_head_dim == config.qk_nope_head_dim + config.qk_rope_head_dim
    assert config.qk_head_dim == 192
    assert config.head_dim == config.qk_rope_head_dim
    assert config.head_dim == 64
    # Every layer runs DSA sparse attention; the leading 3 layers use a dense MLP.
    assert config.layer_types == ["deepseek_sparse_attention"] * 61
    assert config.mlp_layer_types == ["dense"] * 3 + ["sparse"] * 58


def test_deepseek_v32_preserves_explicit_dispatch() -> None:
    config = DeepseekV32Config(
        num_hidden_layers=2,
        mlp_layer_types=["dense", "sparse"],
        layer_types=["deepseek_sparse_attention", "deepseek_sparse_attention"],
    )

    assert config.mlp_layer_types == ["dense", "sparse"]
    assert config.layer_types == [
        "deepseek_sparse_attention",
        "deepseek_sparse_attention",
    ]


def test_deepseek_v32_reroutes_num_experts() -> None:
    config = DeepseekV32Config.from_dict({"num_experts": 384})

    assert config.n_routed_experts == 384


def test_registered_deepseek_v32_config_loads_from_raw_json(tmp_path) -> None:
    raw_config = {
        "model_type": "deepseek_v32",
        "architectures": ["DeepseekV32ForCausalLM"],
        "hidden_size": 7168,
        "num_hidden_layers": 4,
        "index_topk": 512,
        "index_n_heads": 8,
        "index_head_dim": 64,
        "checkpoint_extension": "preserved",
    }
    (tmp_path / "config.json").write_text(json.dumps(raw_config), encoding="utf-8")

    config = get_config(str(tmp_path))

    assert isinstance(config, DeepseekV32Config)
    assert config.hidden_size == 7168
    assert config.index_topk == 512
    assert config.index_n_heads == 8
    assert config.index_head_dim == 64
    assert config.checkpoint_extension == "preserved"
    assert config.name_or_path == str(tmp_path)


def test_loaded_deepseek_v32_config_resolves_target_model(tmp_path) -> None:
    raw_config = {
        "model_type": "deepseek_v32",
        "architectures": ["DeepseekV32ForCausalLM"],
        "num_hidden_layers": 2,
    }
    (tmp_path / "config.json").write_text(json.dumps(raw_config), encoding="utf-8")

    config = get_config(str(tmp_path))
    model_class, architecture = get_model_architecture(
        SimpleNamespace(hf_config=config, quantization=None)
    )

    from tokenspeed.runtime.models.deepseek_v32 import DeepseekV32ForCausalLM
    from tokenspeed.runtime.models.glm5 import GlmMoeDsaForCausalLM

    assert isinstance(config, DeepseekV32Config)
    assert architecture == "DeepseekV32ForCausalLM"
    assert model_class is DeepseekV32ForCausalLM
    assert issubclass(model_class, GlmMoeDsaForCausalLM)

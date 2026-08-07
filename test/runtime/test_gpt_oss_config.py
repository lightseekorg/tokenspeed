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

from tokenspeed.runtime.configs import get_config_class
from tokenspeed.runtime.configs.gpt_oss_config import GptOssConfig
from tokenspeed.runtime.configs.utils import get_config


def test_gpt_oss_config_is_registered() -> None:
    assert get_config_class("gpt_oss") is GptOssConfig


def test_gpt_oss_config_defaults() -> None:
    config = GptOssConfig()

    assert config.model_type == "gpt_oss"
    assert config.vocab_size == 201088
    assert config.hidden_size == 2880
    assert config.intermediate_size == 2880
    assert config.num_hidden_layers == 36
    assert config.num_attention_heads == 64
    assert config.num_key_value_heads == 8
    assert config.num_local_experts == 128
    assert config.head_dim == 64
    assert config.sliding_window == 128
    assert config.hidden_act == "silu"
    assert config.swiglu_limit is None
    assert config.max_position_embeddings == 131072
    assert config.initializer_range == 0.02
    assert config.rms_norm_eps == 1e-5
    assert config.use_cache is True
    assert config.tie_word_embeddings is False
    assert config.attention_dropout == 0.0
    assert config.attention_bias is True
    assert config.num_experts_per_tok == 4
    assert config.router_aux_loss_coef == 0.001
    assert config.output_router_logits is False
    assert config.default_theta == 150000.0


def test_gpt_oss_config_num_key_value_heads_defaults_to_num_attention_heads() -> None:
    config = GptOssConfig(num_key_value_heads=None)
    assert config.num_key_value_heads == config.num_attention_heads


def test_gpt_oss_config_head_dim_derived_from_hidden_size() -> None:
    config = GptOssConfig(head_dim=None, hidden_size=4096, num_attention_heads=32)
    assert config.head_dim == 128


def test_gpt_oss_config_layer_types_auto_generation() -> None:
    """Layer types alternate: odd layers (1-indexed) are sliding, even are full."""
    config = GptOssConfig(num_hidden_layers=6)
    assert config.layer_types == [
        "sliding_attention",  # layer 1 (odd)
        "full_attention",  # layer 2 (even)
        "sliding_attention",  # layer 3 (odd)
        "full_attention",  # layer 4 (even)
        "sliding_attention",  # layer 5 (odd)
        "full_attention",  # layer 6 (even)
    ]


def test_gpt_oss_config_layer_types_explicit_override() -> None:
    explicit = ["full_attention", "full_attention", "sliding_attention"]
    config = GptOssConfig(num_hidden_layers=3, layer_types=explicit)
    assert config.layer_types is explicit


def test_gpt_oss_config_rope_parameters_yarn_default() -> None:
    config = GptOssConfig()
    assert config.rope_parameters == {
        "rope_type": "yarn",
        "factor": 32.0,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "truncate": False,
        "original_max_position_embeddings": 4096,
        "rope_theta": 150000.0,
    }


def test_gpt_oss_config_rope_parameters_explicit_override() -> None:
    rope = {"rope_type": "linear", "factor": 2.0}
    config = GptOssConfig(rope_parameters=rope)
    assert config.rope_parameters is rope


def test_gpt_oss_config_attribute_map_num_experts_aliases_num_local_experts() -> None:
    config = GptOssConfig.from_dict({"num_experts": 64})
    assert config.num_local_experts == 64
    assert config.to_dict()["num_local_experts"] == 64


def test_registered_gpt_oss_config_loads_from_raw_json(tmp_path) -> None:
    raw_config = {
        "model_type": "gpt_oss",
        "architectures": ["GptOssForCausalLM"],
        "vocab_size": 256,
        "hidden_size": 512,
        "intermediate_size": 1024,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "num_local_experts": 16,
        "checkpoint_extension": "preserved",
    }
    (tmp_path / "config.json").write_text(json.dumps(raw_config), encoding="utf-8")

    config = get_config(str(tmp_path))

    assert isinstance(config, GptOssConfig)
    assert config.hidden_size == 512
    assert config.num_local_experts == 16
    assert config.checkpoint_extension == "preserved"
    assert config.name_or_path == str(tmp_path)


def test_model_overrides_rebuild_gpt_oss_layer_schedule(tmp_path) -> None:
    raw_config = {
        "model_type": "gpt_oss",
        "architectures": ["GptOssForCausalLM"],
        "num_hidden_layers": 4,
    }
    (tmp_path / "config.json").write_text(json.dumps(raw_config), encoding="utf-8")

    config = get_config(
        str(tmp_path),
        model_override_args={"num_hidden_layers": 6},
    )

    assert config.num_hidden_layers == 6
    assert len(config.layer_types) == 6

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

import pytest

from tokenspeed.runtime.configs import get_config_class
from tokenspeed.runtime.configs.deepseek_v2_config import DeepseekV2Config
from tokenspeed.runtime.configs.utils import get_config


def test_deepseek_v2_config_is_registered() -> None:
    assert get_config_class("deepseek_v2") is DeepseekV2Config


def test_deepseek_v2_defaults_and_derived_fields() -> None:
    config = DeepseekV2Config()

    assert config.model_type == "deepseek_v2"
    assert config.num_key_value_heads == config.num_attention_heads
    assert config.head_dim == config.qk_rope_head_dim
    assert config.n_routed_experts == 64
    assert config.n_shared_experts == 2
    assert config.moe_intermediate_size == 1407


def test_deepseek_v2_preserves_checkpoint_extensions_and_rope_aliases() -> None:
    config = DeepseekV2Config.from_dict(
        {
            "rope_scaling": {"type": "yarn", "factor": 40.0},
            "moe_layer_freq": 1,
            "scoring_func": "softmax",
            "checkpoint_extension": "preserved",
        }
    )

    assert config.rope_parameters["rope_type"] == "yarn"
    assert config.rope_parameters["factor"] == 40.0
    assert config.moe_layer_freq == 1
    assert config.scoring_func == "softmax"
    assert config.checkpoint_extension == "preserved"


def test_deepseek_v2_rejects_invalid_attention_geometry() -> None:
    with pytest.raises(ValueError, match="not a multiple"):
        DeepseekV2Config(hidden_size=65, num_attention_heads=8)


def test_deepseek_v2_eagle_config_loads_from_raw_json(tmp_path) -> None:
    raw_config = {
        "model_type": "deepseek_v2",
        "architectures": ["Eagle3DeepseekV2ForCausalLM"],
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 8,
        "qk_nope_head_dim": 16,
        "qk_rope_head_dim": 8,
        "v_head_dim": 16,
        "checkpoint_extension": "preserved",
    }
    (tmp_path / "config.json").write_text(
        json.dumps(raw_config),
        encoding="utf-8",
    )

    config = get_config(str(tmp_path), is_draft_worker=True)

    assert isinstance(config, DeepseekV2Config)
    assert config.model_type == "deepseek_v2"
    assert config.architectures == ["Eagle3DeepseekV2ForCausalLM"]
    assert config.num_hidden_layers == 1
    assert config.checkpoint_extension == "preserved"
    assert config.name_or_path == str(tmp_path)

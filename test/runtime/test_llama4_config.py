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
from tokenspeed.runtime.configs.llama4_config import Llama4TextConfig
from tokenspeed.runtime.configs.utils import get_config


def test_llama4_text_config_is_registered() -> None:
    assert get_config_class("llama4_text") is Llama4TextConfig
    assert get_config_class("llama4") is None
    assert get_config_class("llama4_vision_model") is None


def test_llama4_text_defaults_and_layer_schedules() -> None:
    config = Llama4TextConfig(num_hidden_layers=4)

    assert config.model_type == "llama4_text"
    assert config.no_rope_layers == [1, 1, 1, 0]
    assert config.layer_types == [
        "chunked_attention",
        "chunked_attention",
        "chunked_attention",
        "full_attention",
    ]
    assert config.moe_layers == [0, 1, 2, 3]
    assert config.head_dim == 128
    assert config.rope_parameters["rope_theta"] == 500000.0


def test_llama4_text_preserves_explicit_schedules() -> None:
    config = Llama4TextConfig(
        num_hidden_layers=2,
        no_rope_layers=[0, 1],
        layer_types=["full_attention", "chunked_attention"],
        moe_layers=[],
    )

    assert config.no_rope_layers == [0, 1]
    assert config.layer_types == ["full_attention", "chunked_attention"]
    assert config.moe_layers == []


def test_llama4_text_eagle_config_loads_from_raw_json(tmp_path) -> None:
    raw_config = {
        "model_type": "llama4_text",
        "architectures": ["LlamaForCausalLMEagle3"],
        "hidden_size": 64,
        "intermediate_size": 96,
        "intermediate_size_mlp": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 8,
        "checkpoint_extension": "preserved",
    }
    (tmp_path / "config.json").write_text(
        json.dumps(raw_config),
        encoding="utf-8",
    )

    config = get_config(str(tmp_path), is_draft_worker=True)

    assert isinstance(config, Llama4TextConfig)
    assert config.model_type == "llama4_text"
    assert config.architectures == ["LlamaForCausalLMEagle3"]
    assert config.intermediate_size_mlp == 128
    assert config.checkpoint_extension == "preserved"
    assert config.name_or_path == str(tmp_path)

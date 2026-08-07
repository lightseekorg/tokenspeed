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
from tokenspeed.runtime.configs.llama_config import LlamaConfig
from tokenspeed.runtime.configs.utils import get_config


def test_llama_config_is_registered() -> None:
    assert get_config_class("llama") is LlamaConfig


def test_llama_config_explicit_init_preserves_config_semantics() -> None:
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=None,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=[2, 3],
        tie_word_embeddings=True,
        rope_scaling={"type": "linear", "factor": 4.0},
        checkpoint_extension="preserved",
    )

    assert config.vocab_size == 128
    assert config.head_dim == 8
    assert config.num_key_value_heads == 8
    assert config.pad_token_id == 0
    assert config.eos_token_id == [2, 3]
    assert config.tie_word_embeddings is True
    assert (
        config.rope_parameters
        == config.rope_scaling
        == {
            "type": "linear",
            "rope_type": "linear",
            "factor": 4.0,
            "rope_theta": 10_000.0,
        }
    )
    assert config.checkpoint_extension == "preserved"


def test_registered_llama_config_loads_from_json(tmp_path) -> None:
    raw_config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "vocab_size": 128,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "checkpoint_extension": "preserved",
    }
    (tmp_path / "config.json").write_text(
        json.dumps(raw_config),
        encoding="utf-8",
    )

    config = get_config(str(tmp_path))

    assert isinstance(config, LlamaConfig)
    assert config.hidden_size == 32
    assert config.head_dim == 8
    assert config.checkpoint_extension == "preserved"
    assert config.name_or_path == str(tmp_path)


def test_model_overrides_recompute_llama_derived_fields(tmp_path) -> None:
    raw_config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
    }
    (tmp_path / "config.json").write_text(
        json.dumps(raw_config),
        encoding="utf-8",
    )

    config = get_config(
        str(tmp_path),
        model_override_args={"hidden_size": 64},
    )

    assert config.hidden_size == 64
    assert config.head_dim == 16


def test_missing_model_type_raises_value_error(
    tmp_path,
) -> None:
    raw_config = {"architectures": ["LlamaForCausalLM"]}
    (tmp_path / "config.json").write_text(
        json.dumps(raw_config),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported model_type"):
        get_config(str(tmp_path))

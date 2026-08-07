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

import torch

from tokenspeed.runtime.configs.deepseek_v3_config import DeepseekV3Config
from tokenspeed.runtime.configs.kimi_k2_config import KimiK2Config
from tokenspeed.runtime.configs.kimi_k25_config import KimiK25Config


def test_deepseek_v3_defaults_and_derived_fields() -> None:
    config = DeepseekV3Config()

    assert config.num_key_value_heads == config.num_attention_heads
    assert config.qk_head_dim == config.qk_nope_head_dim + config.qk_rope_head_dim
    assert config.head_dim == config.qk_rope_head_dim
    assert config.num_nextn_predict_layers == 1
    assert config.num_mtp_layers == 1


def test_deepseek_v3_loads_checkpoint_fields_and_aliases() -> None:
    config = DeepseekV3Config.from_dict(
        {
            "num_local_experts": 384,
            "num_mtp_layers": 2,
            "moe_layer_freq": 1,
            "scoring_func": "sigmoid",
            "topk_method": "noaux_tc",
            "rope_theta": 50000.0,
            "checkpoint_extension": {"enabled": True},
        }
    )

    assert config.n_routed_experts == 384
    assert config.num_nextn_predict_layers == 2
    assert config.moe_layer_freq == 1
    assert config.scoring_func == "sigmoid"
    assert config.topk_method == "noaux_tc"
    assert config.rope_theta == 50000.0
    assert config.to_dict()["rope_theta"] == 50000.0
    assert config.checkpoint_extension == {"enabled": True}
    assert config.to_dict()["checkpoint_extension"] == {"enabled": True}


def test_deepseek_v3_normalizes_rope_parameter_names() -> None:
    old_name = DeepseekV3Config(rope_scaling={"type": "linear", "factor": 4.0})
    new_name = DeepseekV3Config(rope_parameters={"rope_type": "linear", "factor": 8.0})

    assert (
        old_name.rope_parameters
        == old_name.rope_scaling
        == {
            "type": "linear",
            "rope_type": "linear",
            "factor": 4.0,
            "rope_theta": 10_000.0,
        }
    )
    assert (
        new_name.rope_parameters
        == new_name.rope_scaling
        == {
            "rope_type": "linear",
            "factor": 8.0,
            "rope_theta": 10_000.0,
        }
    )


def test_deepseek_v3_normalizes_checkpoint_dtype() -> None:
    config = DeepseekV3Config(dtype="bfloat16")
    legacy_config = DeepseekV3Config(torch_dtype="float16")

    assert config.dtype is torch.bfloat16
    assert config.to_dict()["dtype"] == "bfloat16"
    assert legacy_config.dtype is torch.float16
    assert "torch_dtype" not in legacy_config.to_dict()


def test_kimi_configs_use_local_deepseek_v3_config() -> None:
    kimi_k2 = KimiK2Config(hidden_size=4096)
    kimi_k25 = KimiK25Config(text_config={"hidden_size": 4096})

    assert kimi_k2.model_type == "kimi_k2"
    assert isinstance(kimi_k25.text_config, DeepseekV3Config)
    assert kimi_k25.hidden_size == 4096
    assert kimi_k25.to_dict()["text_config"]["hidden_size"] == 4096


def test_kimi_k2_inherits_deepseek_defaults_and_derived_fields() -> None:
    config = KimiK2Config()

    assert config.hidden_size == 7168
    assert config.n_routed_experts == 256
    assert config.qk_head_dim == 192
    assert config.head_dim == 64
    assert config.num_nextn_predict_layers == 1


def test_kimi_k2_derives_fields_from_overrides() -> None:
    config = KimiK2Config(
        num_attention_heads=16,
        num_key_value_heads=None,
        qk_nope_head_dim=96,
        qk_rope_head_dim=32,
    )

    assert config.num_key_value_heads == 16
    assert config.qk_head_dim == 128
    assert config.head_dim == 32


def test_registered_local_config_loads_from_raw_json(tmp_path) -> None:
    from tokenspeed.runtime.configs.utils import get_config

    raw_config = {
        "model_type": "kimi_k2",
        "architectures": ["KimiK2ForCausalLM"],
        "hidden_size": 4096,
        "checkpoint_extension": "preserved",
    }
    (tmp_path / "config.json").write_text(json.dumps(raw_config), encoding="utf-8")

    config = get_config(str(tmp_path))

    assert isinstance(config, KimiK2Config)
    assert config.hidden_size == 4096
    assert config.checkpoint_extension == "preserved"
    assert config.name_or_path == str(tmp_path)


def test_runtime_override_wins_over_checkpoint_attribute_alias(tmp_path) -> None:
    from tokenspeed.runtime.configs.utils import get_config

    raw_config = {
        "model_type": "deepseek_v3",
        "architectures": ["DeepseekV3ForCausalLM"],
        "num_local_experts": 256,
    }
    (tmp_path / "config.json").write_text(json.dumps(raw_config), encoding="utf-8")

    config = get_config(
        str(tmp_path),
        model_override_args={"n_routed_experts": 8},
    )

    assert config.n_routed_experts == 8


def test_runtime_override_alias_wins_over_checkpoint_canonical(tmp_path) -> None:
    from tokenspeed.runtime.configs.utils import get_config

    raw_config = {
        "model_type": "deepseek_v3",
        "architectures": ["DeepseekV3ForCausalLM"],
        "n_routed_experts": 256,
    }
    (tmp_path / "config.json").write_text(json.dumps(raw_config), encoding="utf-8")

    config = get_config(
        str(tmp_path),
        model_override_args={"num_local_experts": 8},
    )

    assert config.n_routed_experts == 8


def test_runtime_override_wins_over_checkpoint_mtp_alias(tmp_path) -> None:
    from tokenspeed.runtime.configs.utils import get_config

    raw_config = {
        "model_type": "deepseek_v3",
        "architectures": ["DeepseekV3ForCausalLM"],
        "num_mtp_layers": 2,
    }
    (tmp_path / "config.json").write_text(json.dumps(raw_config), encoding="utf-8")

    config = get_config(
        str(tmp_path),
        model_override_args={"num_nextn_predict_layers": 4},
    )

    assert config.num_nextn_predict_layers == 4

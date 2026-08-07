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
from tokenspeed.runtime.configs.longcat_flash_config import LongcatFlashConfig
from tokenspeed.runtime.configs.utils import get_config


def test_longcat_flash_config_is_registered() -> None:
    assert get_config_class("longcat_flash") is LongcatFlashConfig


def test_longcat_flash_config_defaults() -> None:
    config = LongcatFlashConfig()

    assert config.model_type == "longcat_flash"
    assert config.vocab_size == 131072
    assert config.hidden_size == 6144
    assert config.num_hidden_layers == 56
    assert config.num_layers == 28
    assert config.num_attention_heads == 64
    assert config.num_key_value_heads == 64  # defaults to num_attention_heads
    assert config.hidden_act == "silu"
    assert config.max_position_embeddings == 131072
    assert config.initializer_range == 0.02
    assert config.rms_norm_eps == 1e-5
    assert config.use_cache is True
    assert config.tie_word_embeddings is False
    assert config.attention_bias is False
    assert config.attention_dropout == 0.0
    assert config.ffn_hidden_size == 12288
    assert config.q_lora_rank == 1536
    assert config.kv_lora_rank == 512
    assert config.qk_nope_head_dim == 128
    assert config.qk_rope_head_dim == 64
    assert config.head_dim == 64
    assert config.v_head_dim == 128
    assert config.moe_topk == 12
    assert config.n_routed_experts == 512
    assert config.zero_expert_num == 256
    assert config.expert_ffn_hidden_size == 2048
    assert config.routed_scaling_factor == 6.0
    assert config.default_theta == 10000000.0
    assert config.pad_token_id is None
    assert config.bos_token_id == 1
    assert config.eos_token_id == 2


def test_longcat_flash_config_qk_head_dim_derived() -> None:
    """When ``qk_head_dim`` is None, it defaults to qk_nope + qk_rope."""
    config = LongcatFlashConfig(
        qk_head_dim=None, qk_nope_head_dim=96, qk_rope_head_dim=32
    )
    assert config.qk_head_dim == 128


def test_longcat_flash_config_explicit_qk_head_dim_preserved() -> None:
    config = LongcatFlashConfig(qk_head_dim=256)
    assert config.qk_head_dim == 256


def test_longcat_flash_config_num_key_value_heads_defaults_to_num_attention_heads() -> (
    None
):
    config = LongcatFlashConfig(num_key_value_heads=None, num_attention_heads=32)
    assert config.num_key_value_heads == 32


def test_longcat_flash_config_attribute_map() -> None:
    """Check the three ``attribute_map`` aliases for checkpoint compatibility."""
    config = LongcatFlashConfig.from_dict(
        {
            "num_local_experts": 256,
            "num_experts_per_tok": 8,
            "intermediate_size": 4096,
        }
    )
    assert config.n_routed_experts == 256
    assert config.moe_topk == 8
    assert config.ffn_hidden_size == 4096

    d = config.to_dict()
    assert d["n_routed_experts"] == 256
    assert d["moe_topk"] == 8
    assert d["ffn_hidden_size"] == 4096


def test_longcat_flash_config_explicit_init_overrides_all() -> None:
    config = LongcatFlashConfig(
        vocab_size=256,
        hidden_size=512,
        num_hidden_layers=4,
        num_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        hidden_act="gelu",
        max_position_embeddings=4096,
        initializer_range=0.01,
        rms_norm_eps=1e-4,
        use_cache=False,
        tie_word_embeddings=True,
        attention_bias=True,
        attention_dropout=0.1,
        ffn_hidden_size=2048,
        q_lora_rank=256,
        kv_lora_rank=128,
        qk_nope_head_dim=64,
        qk_rope_head_dim=32,
        head_dim=32,
        v_head_dim=64,
        qk_head_dim=96,
        moe_topk=4,
        n_routed_experts=128,
        zero_expert_num=64,
        expert_ffn_hidden_size=1024,
        routed_scaling_factor=3.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=[2, 3],
    )

    assert config.vocab_size == 256
    assert config.hidden_size == 512
    assert config.num_hidden_layers == 4
    assert config.num_layers == 2
    assert config.num_attention_heads == 8
    assert config.num_key_value_heads == 4
    assert config.hidden_act == "gelu"
    assert config.max_position_embeddings == 4096
    assert config.initializer_range == 0.01
    assert config.rms_norm_eps == 1e-4
    assert config.use_cache is False
    assert config.tie_word_embeddings is True
    assert config.attention_bias is True
    assert config.attention_dropout == 0.1
    assert config.ffn_hidden_size == 2048
    assert config.q_lora_rank == 256
    assert config.kv_lora_rank == 128
    assert config.qk_nope_head_dim == 64
    assert config.qk_rope_head_dim == 32
    assert config.head_dim == 32
    assert config.v_head_dim == 64
    assert config.qk_head_dim == 96
    assert config.moe_topk == 4
    assert config.n_routed_experts == 128
    assert config.zero_expert_num == 64
    assert config.expert_ffn_hidden_size == 1024
    assert config.routed_scaling_factor == 3.0
    assert config.pad_token_id == 0
    assert config.bos_token_id == 1
    assert config.eos_token_id == [2, 3]


def test_registered_longcat_flash_config_loads_from_raw_json(tmp_path) -> None:
    raw_config = {
        "model_type": "longcat_flash",
        "architectures": ["LongcatFlashForCausalLM"],
        "vocab_size": 256,
        "hidden_size": 512,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "checkpoint_extension": "preserved",
    }
    (tmp_path / "config.json").write_text(json.dumps(raw_config), encoding="utf-8")

    config = get_config(str(tmp_path))

    assert isinstance(config, LongcatFlashConfig)
    assert config.hidden_size == 512
    assert config.num_attention_heads == 8
    assert config.checkpoint_extension == "preserved"
    assert config.name_or_path == str(tmp_path)

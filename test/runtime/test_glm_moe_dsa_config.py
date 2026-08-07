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
from tokenspeed.runtime.configs.glm_moe_dsa_config import GlmMoeDsaConfig
from tokenspeed.runtime.configs.utils import get_config


def test_glm_moe_dsa_config_is_registered() -> None:
    assert get_config_class("glm_moe_dsa") is GlmMoeDsaConfig


def test_glm_moe_dsa_defaults() -> None:
    config = GlmMoeDsaConfig()

    assert config.model_type == "glm_moe_dsa"
    assert config.vocab_size == 154880
    assert config.hidden_size == 6144
    assert config.intermediate_size == 12288
    assert config.moe_intermediate_size == 2048
    assert config.num_hidden_layers == 78
    assert config.num_attention_heads == 64
    assert config.num_key_value_heads == 64
    assert config.q_lora_rank == 2048
    assert config.qk_rope_head_dim == 64
    assert config.v_head_dim == 256
    assert config.qk_nope_head_dim == 192
    assert config.n_group == 1
    assert config.topk_group == 1
    assert config.max_position_embeddings == 202752
    assert config.rms_norm_eps == 1e-5
    assert config.n_routed_experts == 256
    assert config.n_shared_experts == 1
    assert config.index_topk == 2048
    assert config.index_head_dim == 128
    assert config.index_n_heads == 32
    assert config.mlp_bias is False


def test_glm_moe_dsa_derived_fields() -> None:
    config = GlmMoeDsaConfig()

    assert config.qk_head_dim == config.qk_nope_head_dim + config.qk_rope_head_dim
    assert config.qk_head_dim == 256
    assert config.head_dim == config.qk_rope_head_dim
    assert config.head_dim == 64
    # Every layer runs DSA sparse attention; the leading 3 layers use a dense MLP.
    assert config.layer_types == ["deepseek_sparse_attention"] * 78
    assert config.mlp_layer_types == ["dense"] * 3 + ["sparse"] * 75
    # Default schedule (freq=1) never shares an indexer, so every layer is "full".
    assert config.indexer_types == ["full"] * 78


def test_glm_moe_dsa_indexer_types_from_pattern() -> None:
    config = GlmMoeDsaConfig(num_hidden_layers=4, index_topk_pattern="FSSF")

    assert config.indexer_types == ["full", "shared", "shared", "full"]


def test_glm_moe_dsa_indexer_types_from_freq_offset() -> None:
    config = GlmMoeDsaConfig(
        num_hidden_layers=4,
        index_topk_freq=2,
        index_skip_topk_offset=2,
    )

    assert config.indexer_types == ["full", "full", "shared", "full"]


def test_glm_moe_dsa_preserves_explicit_indexer_types() -> None:
    config = GlmMoeDsaConfig(
        num_hidden_layers=2,
        indexer_types=["full", "shared"],
    )

    assert config.indexer_types == ["full", "shared"]


def test_glm_moe_dsa_normalizes_legacy_explicit_indexer_types() -> None:
    config = GlmMoeDsaConfig(
        num_hidden_layers=2,
        indexer_types=["F", "S"],
    )

    assert config.indexer_types == ["full", "shared"]


@pytest.mark.parametrize(
    ("indexer_types", "match"),
    [
        (["full", "invalid"], "entries must be one of"),
        (["full"], "number of `indexer_types` entries"),
        (["shared", "full"], "first `indexer_types` entry"),
    ],
)
def test_glm_moe_dsa_rejects_invalid_indexer_types(
    indexer_types: list[str], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        GlmMoeDsaConfig(num_hidden_layers=2, indexer_types=indexer_types)


@pytest.mark.parametrize("freq", [0, -1])
def test_glm_moe_dsa_rejects_nonpositive_index_topk_freq(freq: int) -> None:
    with pytest.raises(ValueError, match="`index_topk_freq` must be positive"):
        GlmMoeDsaConfig(index_topk_freq=freq)


@pytest.mark.parametrize("offset", [0, -1])
def test_glm_moe_dsa_rejects_nonpositive_active_topk_offset(offset: int) -> None:
    with pytest.raises(ValueError, match="`index_skip_topk_offset` must be positive"):
        GlmMoeDsaConfig(index_topk_freq=2, index_skip_topk_offset=offset)


def test_glm_moe_dsa_indexer_types_serialization_round_trip() -> None:
    config = GlmMoeDsaConfig(num_hidden_layers=4, index_topk_pattern="FSSF")

    serialized = config.to_dict()
    restored = GlmMoeDsaConfig.from_dict(serialized)

    assert "index_topk_pattern" not in serialized
    assert restored.indexer_types == ["full", "shared", "shared", "full"]


def test_glm_moe_dsa_reroutes_num_experts() -> None:
    config = GlmMoeDsaConfig.from_dict({"num_experts": 384})

    assert config.n_routed_experts == 384


def test_registered_glm_moe_dsa_config_loads_from_raw_json(tmp_path) -> None:
    raw_config = {
        "model_type": "glm_moe_dsa",
        "architectures": ["GlmMoeDsaForCausalLM"],
        "hidden_size": 6144,
        "num_hidden_layers": 4,
        "index_topk": 512,
        "index_n_heads": 8,
        "index_head_dim": 64,
        "checkpoint_extension": "preserved",
    }
    (tmp_path / "config.json").write_text(json.dumps(raw_config), encoding="utf-8")

    config = get_config(str(tmp_path))

    assert isinstance(config, GlmMoeDsaConfig)
    assert config.hidden_size == 6144
    assert config.index_topk == 512
    assert config.index_n_heads == 8
    assert config.index_head_dim == 64
    assert config.checkpoint_extension == "preserved"
    assert config.name_or_path == str(tmp_path)

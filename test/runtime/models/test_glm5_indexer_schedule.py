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

import pytest
import torch

from tokenspeed.runtime.configs.deepseek_v32_config import DeepseekV32Config
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.models.glm5 import (
    GlmDsaIndexer,
    GlmMoeDsaAttention,
    _glm_dsa_skip_indexer_topk,
)
from tokenspeed.runtime.utils.env import global_server_args_dict


def test_deepseek_v32_without_indexer_schedule_runs_each_indexer() -> None:
    config = DeepseekV32Config(num_hidden_layers=2)

    assert getattr(config, "indexer_types", None) is None
    assert not _glm_dsa_skip_indexer_topk(config, 0)
    assert not _glm_dsa_skip_indexer_topk(config, 1)


def test_deepseek_v32_constructs_dsa_attention_without_indexer_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = DeepseekV32Config(
        num_hidden_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        max_position_embeddings=128,
        index_topk=4,
        index_n_heads=2,
        index_head_dim=8,
    )
    mapping = Mapping(rank=0, world_size=1)
    monkeypatch.setitem(global_server_args_dict, "attention_backend", "mla")

    with torch.device("cpu"):
        attention = GlmMoeDsaAttention(
            config=config,
            mapping=mapping,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            max_position_embeddings=config.max_position_embeddings,
            layer_id=0,
            prefix="model.layers.0.self_attn",
        )

    assert attention.skip_indexer_topk is False
    assert isinstance(attention.indexer, GlmDsaIndexer)
    assert next(attention.parameters()).device.type == "cpu"


@pytest.mark.parametrize(
    ("indexer_types", "layer_id", "expected"),
    [
        (["full", "shared"], 0, False),
        (["full", "shared"], 1, True),
        (["F", "S"], 1, True),
    ],
)
def test_indexer_schedule_controls_topk_reuse(
    indexer_types: list[str], layer_id: int, expected: bool
) -> None:
    config = DeepseekV32Config(num_hidden_layers=2, indexer_types=indexer_types)

    assert _glm_dsa_skip_indexer_topk(config, layer_id) is expected


def test_missing_layer_id_does_not_skip_indexer() -> None:
    assert not _glm_dsa_skip_indexer_topk(DeepseekV32Config(), None)


@pytest.mark.parametrize("layer_id", [-1, 2])
def test_indexer_schedule_rejects_out_of_range_layer_id(layer_id: int) -> None:
    config = DeepseekV32Config(
        num_hidden_layers=2,
        indexer_types=["full", "shared"],
    )

    with pytest.raises(ValueError, match="outside `indexer_types` length"):
        _glm_dsa_skip_indexer_topk(config, layer_id)

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

from tokenspeed.runtime.configs.qwen3_config import Qwen3Config
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.models.base.weight_update import weight_update_session
from tokenspeed.runtime.models.qwen3 import Qwen3ForCausalLM


def make_config(tied: bool) -> Qwen3Config:
    return Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        tie_word_embeddings=tied,
    )


def make_model(tied: bool, mapping: Mapping | None = None) -> Qwen3ForCausalLM:
    return Qwen3ForCausalLM(make_config(tied), mapping or Mapping(rank=0))


def checkpoint() -> dict[str, torch.Tensor]:
    shapes = {
        "model.embed_tokens.weight": (64, 16),
        "model.norm.weight": (16,),
        "model.layers.0.input_layernorm.weight": (16,),
        "model.layers.0.post_attention_layernorm.weight": (16,),
        "model.layers.0.self_attn.q_proj.weight": (16, 16),
        "model.layers.0.self_attn.k_proj.weight": (8, 16),
        "model.layers.0.self_attn.v_proj.weight": (8, 16),
        "model.layers.0.self_attn.o_proj.weight": (16, 16),
        "model.layers.0.self_attn.q_norm.weight": (8,),
        "model.layers.0.self_attn.k_norm.weight": (8,),
        "model.layers.0.mlp.gate_proj.weight": (32, 16),
        "model.layers.0.mlp.up_proj.weight": (32, 16),
        "model.layers.0.mlp.down_proj.weight": (16, 32),
    }
    return {
        name: torch.full(shape, float(index))
        for index, (name, shape) in enumerate(shapes.items(), start=1)
    }


@pytest.mark.parametrize("tied", [True, False])
def test_complete_qwen3_update_preserves_fused_parameter_storage(tied):
    model = make_model(tied)
    source = checkpoint()
    if tied:
        assert model.weight_discard_redundant_checkpoint_names() == {"lm_head.weight"}
        assert model.lm_head.weight is model.model.embed_tokens.weight
    else:
        assert not model.weight_discard_redundant_checkpoint_names()
        source["lm_head.weight"] = torch.ones(64, 16)
    assert model.required_weight_update_names() == set(source)
    assert not model.supports_weight_discard  # This fixture uses CPU.
    pointers = [param.data_ptr() for param in model.parameters()]
    with weight_update_session([model], require_complete=True, cpu_group=None):
        model.load_weights(source.items())
    assert [param.data_ptr() for param in model.parameters()] == pointers
    layer = model.model.layers[0]
    expected_qkv = torch.cat(
        [source[f"model.layers.0.self_attn.{name}_proj.weight"] for name in "qkv"]
    )
    torch.testing.assert_close(layer.self_attn.qkv_proj.weight, expected_qkv)
    expected_gate_up = torch.cat(
        [source[f"model.layers.0.mlp.{name}_proj.weight"] for name in ("gate", "up")]
    )
    torch.testing.assert_close(layer.mlp.gate_up_proj.weight, expected_gate_up)


@pytest.mark.parametrize("missing", ["self_attn.k_proj.weight", "mlp.up_proj.weight"])
def test_qwen3_update_requires_all_fused_sources(missing):
    model = make_model(tied=True)
    source = checkpoint()
    del source[f"model.layers.0.{missing}"]
    with pytest.raises(ValueError, match=rf"missing checkpoint weights: .*{missing}"):
        with weight_update_session([model], require_complete=True, cpu_group=None):
            model.load_weights(source.items())


def test_qwen3_weight_discard_audits_rope_and_rejects_unknown_buffers():
    model = make_model(tied=True)
    rope = model.model.layers[0].self_attn.rotary_emb.cos_sin_cache
    assert model.weight_discard_preserved_tensors() == (rope,)
    model.register_buffer("unknown", torch.ones(1), persistent=False)
    with pytest.raises(ValueError, match="unaudited buffers"):
        model.weight_discard_preserved_tensors()


@pytest.mark.parametrize("rank", [0, 1])
def test_qwen3_complete_update_slices_full_checkpoint_on_every_tp_rank(rank):
    model = make_model(tied=False, mapping=Mapping(rank=rank, world_size=2))
    source = checkpoint()
    for name, tensor in source.items():
        source[name] = torch.arange(tensor.numel(), dtype=tensor.dtype).view_as(tensor)
    source["lm_head.weight"] = source["model.embed_tokens.weight"] + 1
    layer = model.model.layers[0]
    with weight_update_session(
        [model],
        require_complete=True,
        cpu_group=None,
        source_device=torch.device("cpu"),
    ):
        model.load_weights(source.items())

    attn = "model.layers.0.self_attn."
    mlp = "model.layers.0.mlp."
    torch.testing.assert_close(
        layer.self_attn.qkv_proj.weight,
        torch.cat(
            (
                source[attn + "q_proj.weight"].chunk(2, dim=0)[rank],
                source[attn + "k_proj.weight"],
                source[attn + "v_proj.weight"],
            )
        ),
    )
    torch.testing.assert_close(
        layer.self_attn.o_proj.weight,
        source[attn + "o_proj.weight"].chunk(2, dim=1)[rank],
    )
    torch.testing.assert_close(
        layer.mlp.gate_up_proj.weight,
        torch.cat(
            [
                source[mlp + part + "_proj.weight"].chunk(2, dim=0)[rank]
                for part in ("gate", "up")
            ]
        ),
    )
    torch.testing.assert_close(
        layer.mlp.down_proj.weight,
        source[mlp + "down_proj.weight"].chunk(2, dim=1)[rank],
    )
    for name, module in (
        ("model.embed_tokens.weight", model.model.embed_tokens),
        ("lm_head.weight", model.lm_head),
    ):
        indices = module.shard_indices
        local = source[name][
            indices.org_vocab_start_index : indices.org_vocab_end_index
        ]
        torch.testing.assert_close(module.weight[: len(local)], local)
        assert not torch.count_nonzero(module.weight[len(local) :])


def test_qwen3_complete_update_rejects_tp_local_q_proj_even_on_rank_zero():
    model = make_model(tied=True, mapping=Mapping(rank=0, world_size=2))
    source = checkpoint()
    name = "model.layers.0.self_attn.q_proj.weight"
    source[name] = source[name].chunk(2)[0]
    with pytest.raises(
        ValueError, match="expected checkpoint shape.*before TP slicing"
    ):
        with weight_update_session(
            [model],
            require_complete=True,
            cpu_group=None,
            source_device=torch.device("cpu"),
        ):
            model.load_weights(source.items())


def test_qwen3_sdk_update_uses_the_tied_embedding_as_the_only_writer():
    model = make_model(tied=True)
    source = checkpoint()
    source["lm_head.weight"] = torch.full((64, 16), -99.0)
    with weight_update_session(
        [model],
        require_complete=True,
        cpu_group=None,
        source_device=torch.device("cpu"),
    ):
        model.load_weights(source.items())
    assert model.lm_head.weight is model.model.embed_tokens.weight
    torch.testing.assert_close(
        model.lm_head.weight, source["model.embed_tokens.weight"]
    )

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
import unittest
from pathlib import Path
import tempfile

import torch

from tokenspeed.runtime.configs.qwen2_5_moe_config import Qwen2_5MoeConfig
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.utils.env import global_server_args_dict
from tokenspeed.runtime.utils.hf_transformers_utils import _CONFIG_REGISTRY, get_config


def _tiny_qwen2_5_moe_config() -> Qwen2_5MoeConfig:
    return Qwen2_5MoeConfig(
        architectures=["Qwen2MoeForCausalLM"],
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=128,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        decoder_sparse_step=1,
    )


def _single_rank_mapping() -> Mapping:
    mapping = Mapping(rank=0, world_size=1)
    global_server_args_dict["mapping"] = mapping
    return mapping


class TestQwen2_5MoeConfig(unittest.TestCase):
    def test_config_registry(self):
        self.assertEqual(Qwen2_5MoeConfig.model_type, "qwen2_moe")
        self.assertIs(_CONFIG_REGISTRY["qwen2_moe"], Qwen2_5MoeConfig)

    def test_get_config_loads_qwen2_5_moe_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "architectures": ["Qwen2MoeForCausalLM"],
                        "hidden_act": "silu",
                        "hidden_size": 2048,
                        "intermediate_size": 6144,
                        "max_position_embeddings": 40960,
                        "model_type": "qwen2_moe",
                        "moe_intermediate_size": 768,
                        "norm_topk_prob": True,
                        "num_attention_heads": 32,
                        "num_experts": 64,
                        "num_experts_per_tok": 8,
                        "num_hidden_layers": 24,
                        "num_key_value_heads": 8,
                        "rope_theta": 1000000.0,
                        "tie_word_embeddings": False,
                        "vocab_size": 152064,
                    }
                )
            )

            config = get_config(tmpdir, trust_remote_code=False)

        self.assertIsInstance(config, Qwen2_5MoeConfig)
        self.assertEqual(config.architectures, ["Qwen2MoeForCausalLM"])
        self.assertEqual(config.num_experts, 64)
        self.assertEqual(config.num_experts_per_tok, 8)
        self.assertEqual(config.moe_intermediate_size, 768)

    def test_model_registry_resolves_qwen2_5_moe(self):
        from tokenspeed.runtime.models.qwen2_5_moe import Qwen2MoeForCausalLM
        from tokenspeed.runtime.models.registry import ModelRegistry

        cls, arch = ModelRegistry.resolve_model_cls(["Qwen2MoeForCausalLM"])
        self.assertIs(cls, Qwen2MoeForCausalLM)
        self.assertEqual(arch, "Qwen2MoeForCausalLM")

    def test_constructs_sparse_moe_layers(self):
        from tokenspeed.runtime.models.qwen2_5_moe import (
            Qwen2MoeForCausalLM,
            Qwen2_5MoeSparseMoeBlock,
        )

        model = Qwen2MoeForCausalLM(
            _tiny_qwen2_5_moe_config(),
            mapping=_single_rank_mapping(),
        )

        self.assertIsInstance(model.model.layers[0].mlp, Qwen2_5MoeSparseMoeBlock)
        self.assertIsInstance(model.model.layers[1].mlp, Qwen2_5MoeSparseMoeBlock)

    def test_moe_layer_detection(self):
        from tokenspeed.runtime.models.qwen2_5_moe import _is_moe_layer

        config = _tiny_qwen2_5_moe_config()
        self.assertTrue(_is_moe_layer(0, config))
        self.assertTrue(_is_moe_layer(1, config))

        config2 = Qwen2_5MoeConfig(
            architectures=["Qwen2MoeForCausalLM"],
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=8,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=8,
            decoder_sparse_step=2,
        )
        self.assertFalse(_is_moe_layer(0, config2))
        self.assertTrue(_is_moe_layer(1, config2))
        self.assertFalse(_is_moe_layer(2, config2))
        self.assertTrue(_is_moe_layer(3, config2))

    def test_moe_layer_detection_with_mlp_only_layers(self):
        from tokenspeed.runtime.models.qwen2_5_moe import _is_moe_layer

        config = Qwen2_5MoeConfig(
            architectures=["Qwen2MoeForCausalLM"],
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=8,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=8,
            decoder_sparse_step=2,
            mlp_only_layers=[0],
        )
        self.assertFalse(_is_moe_layer(0, config))
        self.assertTrue(_is_moe_layer(1, config))
        self.assertFalse(_is_moe_layer(2, config))
        self.assertTrue(_is_moe_layer(3, config))

    def test_loads_unfused_expert_weights(self):
        from tokenspeed.runtime.models.qwen2_5_moe import Qwen2MoeForCausalLM

        model = Qwen2MoeForCausalLM(
            _tiny_qwen2_5_moe_config(),
            mapping=_single_rank_mapping(),
        )
        weights = []
        for expert_id in range(4):
            weights.extend(
                [
                    (
                        f"model.layers.0.mlp.experts.{expert_id}.gate_proj.weight",
                        torch.full((8, 16), 1.0 + expert_id),
                    ),
                    (
                        f"model.layers.0.mlp.experts.{expert_id}.up_proj.weight",
                        torch.full((8, 16), 11.0 + expert_id),
                    ),
                    (
                        f"model.layers.0.mlp.experts.{expert_id}.down_proj.weight",
                        torch.full((16, 8), 21.0 + expert_id),
                    ),
                ]
            )

        model.load_weights(weights)

        params = dict(model.named_parameters())
        w13 = params["model.layers.0.mlp.experts.w13_weight"]
        w2 = params["model.layers.0.mlp.experts.w2_weight"]
        self.assertEqual(w13[0, :8].mean().item(), 1.0)
        self.assertEqual(w13[0, 8:].mean().item(), 11.0)
        self.assertEqual(w2[0].mean().item(), 21.0)

    def test_skips_nonlocal_expert_weights_under_expert_parallelism(self):
        from tokenspeed.runtime.models.qwen2_5_moe import Qwen2MoeForCausalLM

        ep_mapping = Mapping(rank=0, world_size=4, moe_ep_size=4)
        global_server_args_dict["mapping"] = ep_mapping
        model = Qwen2MoeForCausalLM(
            _tiny_qwen2_5_moe_config(),
            mapping=ep_mapping,
        )
        model.load_weights(
            [
                (
                    "model.layers.0.mlp.experts.0.down_proj.weight",
                    torch.full((16, 8), 21.0),
                ),
                (
                    "model.layers.0.mlp.experts.3.down_proj.weight",
                    torch.full((16, 8), 24.0),
                ),
            ]
        )

        params = dict(model.named_parameters())
        w2 = params["model.layers.0.mlp.experts.w2_weight"]
        self.assertEqual(w2.shape[0], 1)
        self.assertEqual(w2[0].mean().item(), 21.0)

    def test_model_has_correct_num_parameters(self):
        from tokenspeed.runtime.models.qwen2_5_moe import Qwen2MoeForCausalLM

        model = Qwen2MoeForCausalLM(
            _tiny_qwen2_5_moe_config(),
            mapping=_single_rank_mapping(),
        )
        total_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(total_params, 0)

    def test_config_defaults(self):
        config = Qwen2_5MoeConfig()
        self.assertEqual(config.num_experts, 60)
        self.assertEqual(config.num_experts_per_tok, 8)
        self.assertEqual(config.decoder_sparse_step, 1)
        self.assertEqual(config.moe_intermediate_size, 768)
        self.assertEqual(config.shared_expert_intermediate_size, 0)
        self.assertTrue(config.norm_topk_prob)
        self.assertEqual(config.mlp_only_layers, [])
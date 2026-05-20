import json
import unittest

import torch
from torch import nn

from tokenspeed.runtime.configs.qwen3_moe_config import Qwen3MoeConfig
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.utils.env import global_server_args_dict
from tokenspeed.runtime.utils.hf_transformers_utils import _CONFIG_REGISTRY, get_config


def _tiny_qwen3_moe_config() -> Qwen3MoeConfig:
    return Qwen3MoeConfig(
        architectures=["Qwen3MoeForCausalLM"],
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
    )


def _single_rank_mapping() -> Mapping:
    mapping = Mapping(rank=0, world_size=1)
    global_server_args_dict["mapping"] = mapping
    return mapping


def _ep_rank_mapping(rank: int) -> Mapping:
    mapping = Mapping(rank=rank, world_size=4, moe_ep_size=4)
    global_server_args_dict["mapping"] = mapping
    return mapping


class TestQwen3MoeConfig(unittest.TestCase):
    def test_config_registry(self):
        self.assertEqual(Qwen3MoeConfig.model_type, "qwen3_moe")
        self.assertIs(_CONFIG_REGISTRY["qwen3_moe"], Qwen3MoeConfig)

    def test_get_config_loads_qwen3_30b_a3b_shape(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "architectures": ["Qwen3MoeForCausalLM"],
                        "attention_bias": False,
                        "head_dim": 128,
                        "hidden_act": "silu",
                        "hidden_size": 2048,
                        "intermediate_size": 6144,
                        "max_position_embeddings": 40960,
                        "max_window_layers": 48,
                        "mlp_only_layers": [],
                        "model_type": "qwen3_moe",
                        "moe_intermediate_size": 768,
                        "norm_topk_prob": True,
                        "num_attention_heads": 32,
                        "num_experts": 128,
                        "num_experts_per_tok": 8,
                        "num_hidden_layers": 48,
                        "num_key_value_heads": 4,
                        "rope_theta": 1000000.0,
                        "sliding_window": None,
                        "tie_word_embeddings": False,
                        "use_sliding_window": False,
                        "vocab_size": 151936,
                    }
                )
            )

            config = get_config(tmpdir, trust_remote_code=False)

        self.assertIsInstance(config, Qwen3MoeConfig)
        self.assertEqual(config.architectures, ["Qwen3MoeForCausalLM"])
        self.assertEqual(config.num_experts, 128)
        self.assertEqual(config.num_experts_per_tok, 8)
        self.assertEqual(config.moe_intermediate_size, 768)

    def test_model_registry_resolves_qwen3_moe(self):
        from tokenspeed.runtime.models.qwen3_moe import Qwen3MoeForCausalLM
        from tokenspeed.runtime.models.registry import ModelRegistry

        cls, arch = ModelRegistry.resolve_model_cls(["Qwen3MoeForCausalLM"])
        self.assertIs(cls, Qwen3MoeForCausalLM)
        self.assertEqual(arch, "Qwen3MoeForCausalLM")

    def test_constructs_sparse_moe_layers_with_comm_manager(self):
        from tokenspeed.runtime.models.qwen3_5_moe import Qwen3_5MoeSparseMoeBlock
        from tokenspeed.runtime.models.qwen3_moe import Qwen3MoeForCausalLM

        model = Qwen3MoeForCausalLM(
            _tiny_qwen3_moe_config(),
            mapping=_single_rank_mapping(),
        )

        self.assertIsInstance(model.model.layers[0].mlp, Qwen3_5MoeSparseMoeBlock)
        self.assertTrue(model.model.layers[0].comm_manager.is_moe)
        self.assertFalse(model.model.layers[0].comm_manager.prev_is_moe)
        self.assertTrue(model.model.layers[1].comm_manager.prev_is_moe)

    def test_loads_unfused_expert_weights_into_moe_layer(self):
        from tokenspeed.runtime.models.qwen3_moe import Qwen3MoeForCausalLM

        model = Qwen3MoeForCausalLM(
            _tiny_qwen3_moe_config(),
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
        from tokenspeed.runtime.models.qwen3_moe import Qwen3MoeForCausalLM

        model = Qwen3MoeForCausalLM(
            _tiny_qwen3_moe_config(),
            mapping=_ep_rank_mapping(rank=0),
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

    def test_idle_forward_skips_final_norm(self):
        from tokenspeed.runtime.models.qwen3_moe import Qwen3MoeForCausalLM

        class FailingCommManager:
            def final_norm(self, *args, **kwargs):
                raise AssertionError("final_norm should not run for idle forwards")

        class IdleLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.comm_manager = FailingCommManager()

            def forward(
                self,
                positions,
                hidden_states,
                ctx,
                out_cache_loc,
                residual,
                cos_sin=None,
            ):
                return hidden_states, residual

        model = Qwen3MoeForCausalLM(
            _tiny_qwen3_moe_config(),
            mapping=_single_rank_mapping(),
        )
        model.model.layers = nn.ModuleList([IdleLayer()])
        ctx = ForwardContext(
            attn_backend=None,
            token_to_kv_pool=None,
            bs=0,
            num_extends=0,
            input_num_tokens=0,
            forward_mode=ForwardMode.IDLE,
        )

        hidden_states, residual = model.model(
            input_ids=torch.empty(0, dtype=torch.long),
            positions=torch.empty(0, dtype=torch.long),
            ctx=ctx,
            out_cache_loc=torch.empty(0, dtype=torch.long),
        )

        self.assertEqual(hidden_states.shape, (0, 16))
        self.assertIsNone(residual)


if __name__ == "__main__":
    unittest.main()

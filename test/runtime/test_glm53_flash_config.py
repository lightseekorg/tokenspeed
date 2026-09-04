"""GLM-5.3-Flash config protocol tests (cheap, no GPU)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tokenspeed.runtime.configs.glm53_flash_config import (
    Glm53FlashConfig,
    Glm53FlashTextConfig,
    Glm53FlashVisionConfig,
)
from tokenspeed.runtime.configs.model_config import (
    AttentionArch,
    configure_glm_attention,
)
from tokenspeed.runtime.layers.attention.configs.linear_attn import (
    LinearAttnConfig,
)
from tokenspeed.runtime.layers.attention.registry import _LINEAR_ATTN_CLS
from tokenspeed.runtime.utils.hf_transformers_utils import get_config

_NUM_LAYERS = 45
_NUM_KDA = 34
_NUM_DSA = 11


def _linear_attn_config(
    *,
    num_layers: int = _NUM_LAYERS,
    num_heads: int = 64,
    head_dim: int = 128,
) -> dict:
    full_attn_layers = list(range(3, num_layers, 4))
    full_attn_layer_set = set(full_attn_layers)
    return {
        "num_heads": num_heads,
        "head_dim": head_dim,
        "short_conv_kernel_size": 4,
        "gate_lower_bound": -5.0,
        "kda_layers": [
            layer for layer in range(num_layers) if layer not in full_attn_layer_set
        ],
        "full_attn_layers": full_attn_layers,
    }


class Glm53FlashConfigTests(unittest.TestCase):
    def test_runtime_attention_config_preserves_kpool_geometry(self):
        text_config = Glm53FlashTextConfig(
            index_kpool=4,
            linear_attn_config=_linear_attn_config(),
        )
        model_config = SimpleNamespace(
            hf_text_config=text_config,
            hf_config=SimpleNamespace(),
        )

        configure_glm_attention(model_config)

        self.assertEqual(model_config.attention_arch, AttentionArch.DSA)
        self.assertEqual(model_config.index_kpool, 4)

    def test_top_level_materializes_text_and_vision(self):
        config = Glm53FlashConfig(
            architectures=["Glm53FlashForConditionalGeneration"],
            text_config={
                "num_hidden_layers": 4,
                "layer_types": [
                    "linear_attention",
                    "linear_attention",
                    "linear_attention",
                    "deepseek_sparse_attention",
                ],
                "linear_attn_config": _linear_attn_config(num_layers=4),
                "index_kpool": 4,
                "qk_rope_head_dim": 0,
            },
            vision_config={"projection_intermediate_size": 10240},
        )

        self.assertIsInstance(config.text_config, Glm53FlashTextConfig)
        self.assertIsInstance(config.vision_config, Glm53FlashVisionConfig)
        self.assertEqual(config.model_type, "glm53_flash")
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.vocab_size, 154880)
        self.assertEqual(config.text_config.qk_head_dim, 256)
        self.assertEqual(config.vision_config.out_hidden_size, config.hidden_size)
        self.assertEqual(config.vision_config.swiglu_limit, 10.0)

    def test_legacy_checkpoint_metadata_loads_canonical_config(self):
        raw_config = {
            "model_type": "glm5_next",
            "architectures": ["Glm5NextForConditionalGeneration"],
            "text_config": {"model_type": "glm5_next_text"},
            "vision_config": {"model_type": "glm5_next_vision"},
        }
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, "config.json").write_text(
                json.dumps(raw_config), encoding="utf-8"
            )
            config = get_config(model_dir, trust_remote_code=False)
            draft_config = get_config(
                model_dir,
                trust_remote_code=False,
                is_draft_worker=True,
                speculative_algorithm="MTP",
            )

        self.assertIsInstance(config, Glm53FlashConfig)
        self.assertIsInstance(config.text_config, Glm53FlashTextConfig)
        self.assertIsInstance(config.vision_config, Glm53FlashVisionConfig)
        self.assertEqual(config.model_type, "glm53_flash")
        self.assertEqual(config.text_config.model_type, "glm53_flash_text")
        self.assertEqual(config.vision_config.model_type, "glm53_flash_vision")
        self.assertEqual(config.architectures, ["Glm53FlashForConditionalGeneration"])
        self.assertEqual(
            draft_config.architectures,
            ["Glm53FlashForConditionalGenerationNextN"],
        )

    def test_vision_swiglu_limit_precedence(self):
        text_config = {
            "swiglu_limit": 12.0,
            "linear_attn_config": _linear_attn_config(),
        }

        explicit = Glm53FlashConfig(
            text_config=text_config,
            vision_config={"swiglu_limit": 7.0},
        )
        self.assertEqual(explicit.vision_config.swiglu_limit, 7.0)

        inherited = Glm53FlashConfig(text_config=text_config, vision_config={})
        self.assertEqual(inherited.vision_config.swiglu_limit, 12.0)

        with self.assertRaisesRegex(ValueError, "vision_config requires swiglu_limit"):
            Glm53FlashConfig(
                text_config=text_config,
                vision_config={"swiglu_limit": None},
            )

    def test_checkpoint_layer_schedules_match_model_contract(self):
        config = Glm53FlashTextConfig(linear_attn_config=_linear_attn_config())

        self.assertEqual(len(config.linear_layer_ids), _NUM_KDA)
        self.assertEqual(len(config.full_attention_layer_ids), _NUM_DSA)
        self.assertEqual(config.full_attention_layer_ids, list(range(3, 45, 4)))
        self.assertEqual(
            config.layer_types[:4],
            [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "deepseek_sparse_attention",
            ],
        )
        self.assertEqual(
            config.layers_block_type[:4],
            ["linear_attention", "linear_attention", "linear_attention", "attention"],
        )
        self.assertEqual(
            config.paged_cache_layer_types[:4],
            [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
        )
        self.assertEqual(config.mlp_layer_types[:3], ["dense"] * 3)
        self.assertEqual(config.mlp_layer_types[3:], ["sparse"] * 42)
        self.assertEqual(config.indexer_types, ["full"] * _NUM_LAYERS)

    def test_nested_linear_attention_config_normalizes_legacy_fields(self):
        config = Glm53FlashTextConfig(
            num_hidden_layers=4,
            layer_types=None,
            linear_attn_config={
                "num_heads": 8,
                "head_dim": 64,
                "short_conv_kernel_size": 6,
                "gate_lower_bound": -3.0,
                "kda_layers": [0, 2],
                "full_attn_layers": [1, 3],
            },
        )

        self.assertEqual(config.linear_num_heads, 8)
        self.assertEqual(config.linear_head_dim, 64)
        self.assertEqual(config.linear_conv_kernel_dim, 6)
        self.assertEqual(config.linear_lower_bound, -3.0)
        self.assertEqual(config.gate_lower_bound, -3.0)
        self.assertEqual(config.linear_layer_ids, [0, 2])
        self.assertEqual(config.full_attention_layer_ids, [1, 3])
        self.assertEqual(
            config.layer_types,
            [
                "linear_attention",
                "deepseek_sparse_attention",
                "linear_attention",
                "deepseek_sparse_attention",
            ],
        )

    def test_legacy_linear_attention_fields_materialize_nested_config(self):
        config = Glm53FlashTextConfig(
            num_hidden_layers=4,
            layer_types=[
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "deepseek_sparse_attention",
            ],
            linear_num_heads=8,
            linear_head_dim=64,
            linear_conv_kernel_dim=6,
            linear_lower_bound=-3.0,
        )

        self.assertEqual(
            config.linear_attn_config,
            {
                "num_heads": 8,
                "head_dim": 64,
                "short_conv_kernel_size": 6,
                "gate_lower_bound": -3.0,
                "kda_layers": [0, 1, 2],
                "full_attn_layers": [3],
            },
        )

    def test_composite_attention_registry_assigns_kda_only_to_target(self):
        text_config = Glm53FlashTextConfig(
            num_hidden_layers=4,
            layer_types=[
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "deepseek_sparse_attention",
            ],
            linear_attn_config=_linear_attn_config(num_layers=4),
        )
        self.assertIs(
            _LINEAR_ATTN_CLS["Glm53FlashForConditionalGeneration"],
            LinearAttnConfig,
        )
        self.assertNotIn("Glm53FlashForConditionalGenerationNextN", _LINEAR_ATTN_CLS)
        self.assertFalse(hasattr(text_config, "mamba2_cache_params"))

    def test_nested_linear_attn_config_remains_serializable(self):
        config = Glm53FlashConfig(
            text_config={"linear_attn_config": _linear_attn_config()},
            vision_config={},
        )
        serialized = config.to_diff_dict()
        self.assertEqual(
            serialized["text_config"]["linear_attn_config"],
            _linear_attn_config(),
        )
        self.assertIn("linear_attn_config", repr(config.text_config))


if __name__ == "__main__":
    unittest.main()

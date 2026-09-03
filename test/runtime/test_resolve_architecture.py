"""Branch coverage for the architecture-resolution helpers in
``configs.utils``: ``resolve_architecture`` (None-safe read) and
``get_hf_text_config`` (nested text-config unwrapping)."""

# ruff: noqa: E402

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.configs import Qwen3_5MoeConfig  # noqa: E402
from tokenspeed.runtime.configs.base_config import (  # noqa: E402
    BaseConfig,
    get_rope_parameters,
)
from tokenspeed.runtime.configs.qwen3_5_config import Qwen3_5MoeTextConfig  # noqa: E402
from tokenspeed.runtime.configs.utils import (  # noqa: E402
    get_hf_text_config,
    resolve_architecture,
)


class ResolveArchitectureTests(unittest.TestCase):
    def test_qwen3_5_moe_default_construction_returns_class_name(self) -> None:
        config = Qwen3_5MoeConfig()
        self.assertIsNone(config.architectures)
        self.assertEqual(resolve_architecture(config), "Qwen3_5MoeConfig")

    def test_qwen3_5_moe_with_explicit_architecture_returns_it(self) -> None:
        config = Qwen3_5MoeConfig(architectures=["Qwen3_5MoeForConditionalGeneration"])
        self.assertEqual(
            resolve_architecture(config),
            "Qwen3_5MoeForConditionalGeneration",
        )

    def test_qwen3_5_moe_with_empty_list_falls_back(self) -> None:
        config = Qwen3_5MoeConfig(architectures=[])
        self.assertEqual(resolve_architecture(config), "Qwen3_5MoeConfig")

    def test_missing_architectures_attribute_returns_class_name(self) -> None:
        class _Stub:
            pass

        self.assertEqual(resolve_architecture(_Stub()), "_Stub")


class Qwen3_5ConfigTests(unittest.TestCase):
    def test_nested_moe_text_config_unwraps_to_attention_config(self) -> None:
        nested = Qwen3_5MoeConfig()
        config = Qwen3_5MoeConfig(text_config=nested)

        self.assertIs(config.text_config, nested.text_config)
        self.assertIs(get_hf_text_config(config), config.text_config)
        self.assertTrue(hasattr(config.text_config, "num_attention_heads"))
        self.assertEqual(
            config.num_attention_heads,
            config.text_config.num_attention_heads,
        )

    def test_mrope_extensions_survive_rope_validation_and_round_trip(
        self,
    ) -> None:
        rope_parameters = {
            "rope_type": "default",
            "mrope_section": [16, 24, 24],
            "mrope_interleaved": True,
        }

        config = Qwen3_5MoeTextConfig(rope_parameters=rope_parameters)

        self.assertEqual(config.rope_parameters["rope_type"], "default")
        self.assertIn("mrope_section", config.rope_parameters)
        self.assertIn("mrope_interleaved", config.rope_parameters)
        self.assertEqual(get_rope_parameters(config)["mrope_section"], [16, 24, 24])

        round_tripped = Qwen3_5MoeTextConfig(**config.to_dict())
        self.assertIn("mrope_section", round_tripped.rope_parameters)
        self.assertIn("mrope_interleaved", round_tripped.rope_parameters)


class ConfigDtypeTests(unittest.TestCase):
    def test_llava_dtype_override_sets_float16(self) -> None:
        config = BaseConfig(architectures=["LlavaForCausalLM"])
        self.assertIs(get_hf_text_config(config), config)
        self.assertIs(config.dtype, torch.float16)


if __name__ == "__main__":
    unittest.main(verbosity=2)

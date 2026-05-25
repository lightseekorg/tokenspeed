"""Branch coverage for the architecture-resolution helpers in
``hf_transformers_utils``: ``resolve_architecture`` (None-safe read) and
``_materialize_architectures`` (pin the raw config.json value back onto
the live config when ``from_pretrained`` lost it)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.configs import Qwen3_5MoeConfig  # noqa: E402
from tokenspeed.runtime.utils.hf_transformers_utils import (  # noqa: E402
    _materialize_architectures,
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


class MaterializeArchitecturesTests(unittest.TestCase):
    def test_pins_when_live_config_lost_architectures(self) -> None:
        config = Qwen3_5MoeConfig()
        self.assertIsNone(config.architectures)
        _materialize_architectures(
            config, {"architectures": ["Qwen3_5MoeForConditionalGeneration"]}
        )
        self.assertEqual(config.architectures, ["Qwen3_5MoeForConditionalGeneration"])

    def test_no_op_when_live_config_already_has_architectures(self) -> None:
        config = Qwen3_5MoeConfig(architectures=["Original"])
        _materialize_architectures(config, {"architectures": ["WouldOverride"]})
        self.assertEqual(config.architectures, ["Original"])

    def test_rejects_non_list_value(self) -> None:
        # Malformed config.json with a bare string would otherwise be
        # silently split into characters by ``list("Foo")``.
        config = Qwen3_5MoeConfig()
        _materialize_architectures(config, {"architectures": "Foo"})
        self.assertIsNone(config.architectures)

    def test_rejects_list_with_non_string_items(self) -> None:
        config = Qwen3_5MoeConfig()
        _materialize_architectures(config, {"architectures": [{"name": "Foo"}]})
        self.assertIsNone(config.architectures)

    def test_no_op_when_raw_config_has_no_architectures_key(self) -> None:
        config = Qwen3_5MoeConfig()
        _materialize_architectures(config, {})
        self.assertIsNone(config.architectures)

    def test_pinned_list_is_a_copy_not_a_shared_reference(self) -> None:
        # Subsequent in-place rewrites (e.g. the draft-worker
        # ``architectures[0] += "NextN"`` step) must not leak back into
        # the caller's raw_config dict.
        config = Qwen3_5MoeConfig()
        raw = {"architectures": ["Qwen3_5MoeForConditionalGeneration"]}
        _materialize_architectures(config, raw)
        config.architectures[0] += "NextN"
        self.assertEqual(raw["architectures"], ["Qwen3_5MoeForConditionalGeneration"])
        self.assertEqual(
            config.architectures, ["Qwen3_5MoeForConditionalGenerationNextN"]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

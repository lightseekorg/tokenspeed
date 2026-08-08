"""Branch coverage for Hugging Face config loading and repair helpers."""

# ruff: noqa: E402

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import PretrainedConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.configs import Qwen3_5MoeConfig  # noqa: E402
from tokenspeed.runtime.configs.model_config import get_hf_text_config  # noqa: E402
from tokenspeed.runtime.configs.qwen3_5_config import Qwen3_5MoeTextConfig  # noqa: E402
from tokenspeed.runtime.configs.utils import get_rope_parameters  # noqa: E402
from tokenspeed.runtime.utils.hf_transformers_utils import (
    _materialize_architectures,
    _materialize_qwen3_5_text_config,
    get_config,
)
from tokenspeed.runtime.utils.hf_transformers_utils import (  # noqa: E402
    get_hf_text_config as get_runtime_hf_text_config,
)
from tokenspeed.runtime.utils.hf_transformers_utils import (
    resolve_architecture,
)

_QWEN3_5_RAW_TEXT_CONFIG = {
    "hidden_size": 4096,
    "max_position_embeddings": 262144,
    "num_attention_heads": 32,
    "num_hidden_layers": 60,
    "full_attention_interval": 4,
}
_QWEN3_5_RAW_CONFIG = {
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "model_type": "qwen3_5_moe",
    "text_config": _QWEN3_5_RAW_TEXT_CONFIG,
}


def _write_config(directory: str, raw_config: dict) -> None:
    with open(os.path.join(directory, "config.json"), "w") as file:
        json.dump(raw_config, file)


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

    def test_mrope_extensions_do_not_leak_to_transformers_rope_validation(
        self,
    ) -> None:
        rope_parameters = {
            "rope_type": "default",
            "mrope_section": [16, 24, 24],
            "mrope_interleaved": True,
        }

        with self.assertNoLogs("transformers", level="WARNING"):
            config = Qwen3_5MoeTextConfig(rope_parameters=rope_parameters)

        self.assertEqual(config.rope_parameters["rope_type"], "default")
        self.assertNotIn("mrope_section", config.rope_parameters)
        self.assertNotIn("mrope_interleaved", config.rope_parameters)
        self.assertEqual(get_rope_parameters(config), rope_parameters)

    def test_default_config_serialization_still_round_trips(self) -> None:
        config = Qwen3_5MoeConfig()
        self.assertIsInstance(config.to_diff_dict(), dict)

        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            restored = Qwen3_5MoeConfig.from_pretrained(directory)

        self.assertEqual(restored.text_config.hidden_size, 2048)
        self.assertEqual(restored.text_config.max_position_embeddings, 32768)


class MaterializeQwen3_5TextConfigTests(unittest.TestCase):
    def test_restores_exact_defaulted_text_config_from_raw_payload(self) -> None:
        config = Qwen3_5MoeConfig()
        self.assertEqual(config.text_config.hidden_size, 2048)
        self.assertFalse(hasattr(config.text_config, "full_attention_interval"))

        _materialize_qwen3_5_text_config(config, _QWEN3_5_RAW_CONFIG)

        self.assertEqual(config.text_config.hidden_size, 4096)
        self.assertEqual(config.text_config.max_position_embeddings, 262144)
        self.assertEqual(config.text_config.full_attention_interval, 4)
        self.assertEqual(len(config.text_config.layers_block_type), 60)

    def test_rejects_missing_nested_text_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "nested text_config object"):
            _materialize_qwen3_5_text_config(
                Qwen3_5MoeConfig(),
                {"model_type": "qwen3_5_moe"},
            )

    def test_rejects_missing_full_attention_interval(self) -> None:
        raw_config = {
            **_QWEN3_5_RAW_CONFIG,
            "text_config": {"hidden_size": 4096},
        }
        with self.assertRaisesRegex(ValueError, "full_attention_interval"):
            _materialize_qwen3_5_text_config(Qwen3_5MoeConfig(), raw_config)

    def test_reports_materialization_mismatches(self) -> None:
        # Protect against future config constructors dropping or normalizing
        # checkpoint-defining kwargs while rebuilding the nested payload.
        config = Qwen3_5MoeConfig()
        with (
            patch.object(
                config,
                "_ensure_text_config",
                return_value=Qwen3_5MoeTextConfig(full_attention_interval=4),
            ),
            self.assertRaisesRegex(
                ValueError,
                "hidden_size: raw=4096, materialized=2048",
            ),
        ):
            _materialize_qwen3_5_text_config(config, _QWEN3_5_RAW_CONFIG)


class GetConfigSnapshotTests(unittest.TestCase):
    def test_registered_config_uses_one_revision_pinned_snapshot(self) -> None:
        defaulted_config = Qwen3_5MoeConfig()
        with tempfile.TemporaryDirectory() as directory:
            _write_config(directory, _QWEN3_5_RAW_CONFIG)
            with (
                patch(
                    "tokenspeed.runtime.utils.hf_transformers_utils.snapshot_download",
                    return_value=directory,
                ) as download,
                patch.object(
                    Qwen3_5MoeConfig,
                    "from_pretrained",
                    return_value=defaulted_config,
                ) as from_pretrained,
            ):
                config = get_config(
                    "nvidia/Qwen3.5-397B-A17B-NVFP4",
                    trust_remote_code=False,
                    revision="checkpoint-revision",
                )

        download.assert_called_once_with(
            "nvidia/Qwen3.5-397B-A17B-NVFP4",
            revision="checkpoint-revision",
            ignore_patterns=["*.pt", "*.safetensors", "*.bin"],
        )
        from_pretrained.assert_called_once_with(directory)
        self.assertEqual(config._name_or_path, "nvidia/Qwen3.5-397B-A17B-NVFP4")
        self.assertEqual(config.text_config.hidden_size, 4096)
        self.assertEqual(config.text_config.full_attention_interval, 4)

    def test_text_only_qwen3_5_config_skips_wrapper_materialization(self) -> None:
        raw_config = {
            "architectures": ["Qwen3_5MoeForCausalLM"],
            "model_type": "qwen3_5_moe_text",
            **_QWEN3_5_RAW_TEXT_CONFIG,
        }
        with tempfile.TemporaryDirectory() as directory:
            _write_config(directory, raw_config)
            config = get_config(directory, trust_remote_code=False)

        self.assertIsInstance(config, Qwen3_5MoeTextConfig)
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.full_attention_interval, 4)

    def test_model_overrides_apply_after_raw_config_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _write_config(directory, _QWEN3_5_RAW_CONFIG)
            config = get_config(
                directory,
                trust_remote_code=False,
                model_override_args={"max_position_embeddings": 8192},
            )

        self.assertEqual(config.text_config.max_position_embeddings, 8192)

    def test_draft_worker_can_override_interval_after_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _write_config(directory, _QWEN3_5_RAW_CONFIG)
            config = get_config(
                directory,
                trust_remote_code=False,
                is_draft_worker=True,
            )

        self.assertEqual(
            config.architectures,
            ["Qwen3_5MoeForConditionalGenerationNextN"],
        )
        config.text_config.full_attention_interval = 1
        self.assertEqual(
            config.text_config.full_attention_layer_ids,
            list(range(config.text_config.num_hidden_layers)),
        )

    def test_auto_config_branch_loads_resolved_local_snapshot(self) -> None:
        raw_config = {
            "architectures": ["UnregisteredForCausalLM"],
            "model_type": "unregistered",
        }
        loaded_config = PretrainedConfig(architectures=["UnregisteredForCausalLM"])
        with tempfile.TemporaryDirectory() as directory:
            _write_config(directory, raw_config)
            with patch(
                "tokenspeed.runtime.utils.hf_transformers_utils.AutoConfig.from_pretrained",
                return_value=loaded_config,
            ) as from_pretrained:
                config = get_config(
                    directory,
                    trust_remote_code=True,
                    custom_argument="value",
                )

        from_pretrained.assert_called_once_with(
            directory,
            trust_remote_code=True,
            custom_argument="value",
        )
        self.assertIs(config, loaded_config)


class ConfigDtypeTests(unittest.TestCase):
    def test_llava_dtype_override_does_not_use_deprecated_field(self) -> None:
        helpers = (get_hf_text_config, get_runtime_hf_text_config)

        with self.assertNoLogs("transformers", level="WARNING"):
            for helper in helpers:
                config = PretrainedConfig(architectures=["LlavaForCausalLM"])
                self.assertIs(helper(config), config)
                self.assertIs(config.dtype, torch.float16)


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

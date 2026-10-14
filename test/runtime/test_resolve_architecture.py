"""Branch coverage for the architecture-resolution helpers in
``hf_transformers_utils``: ``resolve_architecture`` (None-safe read) and
``_materialize_architectures`` (pin the raw config.json value back onto
the live config when ``from_pretrained`` lost it)."""

# ruff: noqa: E402

import logging
import os
import sys
from unittest import mock

import pytest
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
)
from tokenspeed.runtime.utils.hf_transformers_utils import (  # noqa: E402
    get_hf_text_config as get_runtime_hf_text_config,
)
from tokenspeed.runtime.utils.hf_transformers_utils import (
    resolve_architecture,
)


class TestResolveArchitectureTests:
    def test_qwen3_5_moe_default_construction_returns_class_name(self) -> None:
        config = Qwen3_5MoeConfig()
        assert (config.architectures) is None
        assert (resolve_architecture(config)) == ("Qwen3_5MoeConfig")

    def test_qwen3_5_moe_with_explicit_architecture_returns_it(self) -> None:
        config = Qwen3_5MoeConfig(architectures=["Qwen3_5MoeForConditionalGeneration"])
        assert (resolve_architecture(config)) == ("Qwen3_5MoeForConditionalGeneration")

    def test_qwen3_5_moe_with_empty_list_falls_back(self) -> None:
        config = Qwen3_5MoeConfig(architectures=[])
        assert (resolve_architecture(config)) == ("Qwen3_5MoeConfig")

    def test_missing_architectures_attribute_returns_class_name(self) -> None:
        class _Stub:
            pass

        assert (resolve_architecture(_Stub())) == ("_Stub")


class TestQwen3_5ConfigTests:
    def test_nested_moe_text_config_unwraps_to_attention_config(self) -> None:
        nested = Qwen3_5MoeConfig()
        config = Qwen3_5MoeConfig(text_config=nested)

        assert (config.text_config) is (nested.text_config)
        assert (get_hf_text_config(config)) is (config.text_config)
        assert hasattr(config.text_config, "num_attention_heads")
        assert (config.num_attention_heads) == (config.text_config.num_attention_heads)

    def test_mrope_extensions_survive_rope_validation_and_round_trip(
        self,
        caplog,
    ) -> None:
        caplog.clear()
        rope_parameters = {
            "rope_type": "default",
            "mrope_section": [16, 24, 24],
            "mrope_interleaved": True,
        }

        logger = logging.getLogger("transformers")
        with (
            caplog.at_level(logging.WARNING, logger="transformers"),
            mock.patch.object(logger, "handlers", [caplog.handler]),
            mock.patch.object(logger, "propagate", False),
        ):
            config = Qwen3_5MoeTextConfig(rope_parameters=rope_parameters)
        assert not [
            record
            for record in caplog.records
            if record.name == "transformers" or record.name.startswith("transformers.")
        ]

        assert (config.rope_parameters["rope_type"]) == ("default")
        assert ("mrope_section") in (config.rope_parameters)
        assert ("mrope_interleaved") in (config.rope_parameters)
        assert (get_rope_parameters(config)["mrope_section"]) == ([16, 24, 24])

        round_tripped = Qwen3_5MoeTextConfig(**config.to_dict())
        assert ("mrope_section") in (round_tripped.rope_parameters)
        assert ("mrope_interleaved") in (round_tripped.rope_parameters)


class TestConfigDtypeTests:
    def test_llava_dtype_override_does_not_use_deprecated_field(self, caplog) -> None:
        helpers = (get_hf_text_config, get_runtime_hf_text_config)
        caplog.clear()

        logger = logging.getLogger("transformers")
        with (
            caplog.at_level(logging.WARNING, logger="transformers"),
            mock.patch.object(logger, "handlers", [caplog.handler]),
            mock.patch.object(logger, "propagate", False),
        ):
            for helper in helpers:
                config = PretrainedConfig(architectures=["LlavaForCausalLM"])
                assert (helper(config)) is (config)
                assert (config.dtype) is (torch.float16)
        assert not [
            record
            for record in caplog.records
            if record.name == "transformers" or record.name.startswith("transformers.")
        ]


class TestMaterializeArchitecturesTests:
    def test_pins_when_live_config_lost_architectures(self) -> None:
        config = Qwen3_5MoeConfig()
        assert (config.architectures) is None
        _materialize_architectures(
            config, {"architectures": ["Qwen3_5MoeForConditionalGeneration"]}
        )
        assert (config.architectures) == (["Qwen3_5MoeForConditionalGeneration"])

    def test_no_op_when_live_config_already_has_architectures(self) -> None:
        config = Qwen3_5MoeConfig(architectures=["Original"])
        _materialize_architectures(config, {"architectures": ["WouldOverride"]})
        assert (config.architectures) == (["Original"])

    def test_rejects_non_list_value(self) -> None:
        # Malformed config.json with a bare string would otherwise be
        # silently split into characters by ``list("Foo")``.
        config = Qwen3_5MoeConfig()
        _materialize_architectures(config, {"architectures": "Foo"})
        assert (config.architectures) is None

    def test_rejects_list_with_non_string_items(self) -> None:
        config = Qwen3_5MoeConfig()
        _materialize_architectures(config, {"architectures": [{"name": "Foo"}]})
        assert (config.architectures) is None

    def test_no_op_when_raw_config_has_no_architectures_key(self) -> None:
        config = Qwen3_5MoeConfig()
        _materialize_architectures(config, {})
        assert (config.architectures) is None

    def test_pinned_list_is_a_copy_not_a_shared_reference(self) -> None:
        # Subsequent in-place rewrites (e.g. the draft-worker
        # ``architectures[0] += "NextN"`` step) must not leak back into
        # the caller's raw_config dict.
        config = Qwen3_5MoeConfig()
        raw = {"architectures": ["Qwen3_5MoeForConditionalGeneration"]}
        _materialize_architectures(config, raw)
        config.architectures[0] += "NextN"
        assert (raw["architectures"]) == (["Qwen3_5MoeForConditionalGeneration"])
        assert (config.architectures) == (["Qwen3_5MoeForConditionalGenerationNextN"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))

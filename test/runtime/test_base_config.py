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
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tokenspeed.runtime.configs.base_config import BaseConfig
from tokenspeed.runtime.configs.deepseek_v3_config import DeepseekV3Config
from tokenspeed.runtime.configs.deepseek_v4_config import (
    DeepseekV4Config,
    get_deepseek_v4_compress_ratio,
)
from tokenspeed.runtime.configs.llama_config import LlamaConfig
from tokenspeed.runtime.configs.longcat_flash_config import LongcatFlashConfig
from tokenspeed.runtime.configs.minimax_m3_config import (
    MiniMaxM3Config,
    MiniMaxM3VLTextConfig,
)
from tokenspeed.runtime.configs.qwen2_config import Qwen2Config
from tokenspeed.runtime.configs.qwen3_5_config import Qwen3_5TextConfig
from tokenspeed.runtime.configs.qwen3_asr_config import Qwen3ASRConfig
from tokenspeed.runtime.configs.qwen3_config import Qwen3Config
from tokenspeed.runtime.configs.utils import get_config, get_context_length


def test_configs_do_not_export_config_protocol() -> None:
    import tokenspeed.runtime.configs as configs

    assert "ConfigLike" not in configs.__all__
    with pytest.raises(AttributeError):
        getattr(configs, "ConfigLike")


def test_base_config_preserves_checkpoint_fields_and_aliases() -> None:
    config = DeepseekV3Config.from_dict(
        {
            "num_local_experts": 384,
            "num_mtp_layers": 2,
            "rope_scaling": {"type": "linear", "factor": 4.0},
            "dtype": "bfloat16",
            "checkpoint_extension": {"enabled": True},
        }
    )

    assert config.n_routed_experts == 384
    assert config.num_nextn_predict_layers == 2
    assert (
        config.rope_parameters
        == config.rope_scaling
        == {
            "type": "linear",
            "rope_type": "linear",
            "factor": 4.0,
            "rope_theta": 10_000.0,
        }
    )
    assert config.dtype is torch.bfloat16
    assert config.to_dict()["dtype"] == "bfloat16"
    assert config.to_dict()["checkpoint_extension"] == {"enabled": True}


def test_base_config_does_not_shadow_classvars() -> None:
    """ClassVar names such as ``model_type`` must not become instance fields."""
    config = DeepseekV3Config.from_dict(
        {"model_type": "not_deepseek_v3", "hidden_size": 2048}
    )

    assert "model_type" not in config.__dict__
    assert config.model_type == "deepseek_v3"
    assert type(config).model_type == "deepseek_v3"


def test_composite_subconfig_does_not_shadow_model_type() -> None:
    """Nested text/vision configs must not leak their ``model_type`` into __dict__."""
    config = MiniMaxM3Config.from_dict(
        {
            "model_type": "minimax_m3_vl",
            "text_config": {"model_type": "minimax_m3_vl_text", "hidden_size": 6144},
            "vision_config": {"model_type": "minimax_m3", "hidden_size": 1280},
        }
    )

    assert "model_type" not in config.__dict__
    assert "model_type" not in config.text_config.__dict__
    assert "model_type" not in config.vision_config.__dict__
    assert config.text_config.model_type == "minimax_m3_vl_text"
    assert config.vision_config.model_type == "minimax_m3"


def test_to_dict_excludes_private_runtime_state() -> None:
    """Private (``_``-prefixed) attributes must not be serialized."""
    config = DeepseekV3Config.from_dict(
        {"model_type": "deepseek_v3", "hidden_size": 2048, "name_or_path": "/tmp/foo"}
    )
    output = config.to_dict()
    assert "_name_or_path" not in output
    assert config.name_or_path == "/tmp/foo"

    qwen = Qwen3_5TextConfig(
        rope_parameters={"rope_theta": 10000.0, "mrope_section": [16, 24, 24]}
    )
    assert "mrope_section" in qwen.to_dict()["rope_parameters"]


def test_base_config_loads_nested_local_config(tmp_path) -> None:
    raw_config = {
        "model_type": "qwen3_asr",
        "architectures": ["Qwen3ASRForConditionalGeneration"],
        "thinker_config": {
            "text_config": {
                "model_type": "qwen3",
                "hidden_size": 2048,
            }
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(raw_config), encoding="utf-8")

    config = Qwen3ASRConfig.from_pretrained(tmp_path)

    assert config.name_or_path == str(tmp_path)
    assert config.thinker_config.text_config.hidden_size == 2048
    assert config.to_dict()["thinker_config"]["text_config"]["hidden_size"] == 2048


def test_base_config_standardizes_and_validates_rope() -> None:
    config = Qwen2Config(rope_scaling={"type": "linear", "factor": 2.0})

    assert config.rope_parameters["rope_type"] == "linear"
    assert config.rope_parameters["rope_theta"] == 10_000.0

    with pytest.raises(KeyError, match="factor"):
        Qwen2Config(rope_scaling={"type": "linear"})


def test_base_config_applies_default_theta_without_rope_fields() -> None:
    """A config declaring only ``default_theta`` still builds a default RoPE dict."""
    text_config = MiniMaxM3VLTextConfig()
    assert text_config.rope_parameters["rope_type"] == "default"
    assert text_config.rope_parameters["rope_theta"] == 5_000_000.0

    longcat = LongcatFlashConfig()
    assert longcat.rope_parameters["rope_type"] == "default"
    assert longcat.rope_parameters["rope_theta"] == 10_000_000.0


def test_deepseek_v4_standardizes_flat_rope_and_validates_once() -> None:
    class CountingDeepseekV4Config(DeepseekV4Config):
        validate_rope_calls = 0

        def validate_rope(self) -> None:
            self.validate_rope_calls += 1
            super().validate_rope()

    config = CountingDeepseekV4Config(
        rope_scaling={
            "type": "yarn",
            "factor": 16.0,
            "original_max_position_embeddings": 65_536,
        }
    )

    assert config.validate_rope_calls == 1
    assert config.rope_parameters["main"] == {
        "rope_type": "default",
        "rope_theta": 10_000.0,
        "partial_rotary_factor": 0.125,
    }
    assert config.rope_parameters["compress"] == {
        "type": "yarn",
        "rope_type": "yarn",
        "factor": 16.0,
        "original_max_position_embeddings": 65_536,
        "rope_theta": 160_000.0,
        "partial_rotary_factor": 0.125,
        "attn_factor": 1.0,
    }


def test_deepseek_v4_preserves_missing_yarn_original_context_length() -> None:
    rope_parameters = {
        "main": {"type": "default"},
        "compress": {"type": "yarn", "factor": 16.0},
    }

    config = DeepseekV4Config(rope_parameters=rope_parameters)

    assert rope_parameters == {
        "main": {"type": "default"},
        "compress": {"type": "yarn", "factor": 16.0},
    }
    assert config.rope_parameters["main"] == {
        "type": "default",
        "rope_type": "default",
        "rope_theta": 10_000.0,
        "partial_rotary_factor": 0.125,
    }
    assert config.rope_parameters["compress"] == {
        "type": "yarn",
        "rope_type": "yarn",
        "factor": 16.0,
        "rope_theta": 160_000.0,
        "partial_rotary_factor": 0.125,
        "attn_factor": 1.0,
    }


def test_deepseek_v4_layered_rope_does_not_rescale_global_context() -> None:
    """A branch-local YaRN ``factor`` must not scale the global context length.

    V4 keys ``rope_parameters`` by rope-type label, so ``get_context_length``
    sees a layered dict whose ``factor`` sits on the ``compress`` branch. That
    factor scales compressed attention only; the global usable length remains
    ``max_position_embeddings``.
    """
    config = DeepseekV4Config(
        rope_parameters={
            "main": {"type": "default"},
            "compress": {"type": "yarn", "factor": 16.0},
        }
    )

    assert get_context_length(config) == config.max_position_embeddings


@pytest.mark.parametrize(
    "compress_rope_parameters",
    [
        {
            "type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
        },
        {
            "type": "longrope",
            "short_factor": [1.0],
            "long_factor": [1.0],
        },
    ],
)
def test_deepseek_v4_llama3_and_longrope_require_original_context_length(
    compress_rope_parameters: dict,
) -> None:
    with pytest.raises(KeyError, match="original_max_position_embeddings"):
        DeepseekV4Config(
            rope_parameters={
                "main": {"type": "default"},
                "compress": compress_rope_parameters,
            }
        )


def test_deepseek_v4_rejects_partial_nested_rope_parameters() -> None:
    with pytest.raises(ValueError, match="both `main` and `compress`"):
        DeepseekV4Config(rope_parameters={"main": {"rope_type": "default"}})


def test_deepseek_v4_rejects_unknown_rope_type() -> None:
    with pytest.raises(ValueError, match="Unsupported DeepSeek V4 rope type"):
        DeepseekV4Config(
            rope_parameters={
                "main": {"rope_type": "default"},
                "compress": {"rope_type": "unknown"},
            }
        )


def test_deepseek_v4_accepts_runtime_yarn_alias() -> None:
    config = DeepseekV4Config(
        rope_parameters={
            "main": {"rope_type": "default"},
            "compress": {"rope_type": "deepseek_yarn", "factor": 16.0},
        }
    )

    assert config.rope_parameters["compress"]["rope_type"] == "deepseek_yarn"
    assert config.rope_parameters["compress"]["attn_factor"] == 1.0


def test_deepseek_v4_truncates_oversized_layer_schedule() -> None:
    """Schedules longer than ``num_hidden_layers`` truncate (transformers parity)."""
    config = DeepseekV4Config(
        num_hidden_layers=2,
        layer_types=[
            "sliding_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
        ],
    )
    assert config.layer_types == [
        "sliding_attention",
        "compressed_sparse_attention",
    ]


def test_deepseek_v4_rejects_mismatched_layer_schedule_lengths() -> None:
    with pytest.raises(ValueError, match=r"len\(mlp_layer_types\)"):
        DeepseekV4Config(
            num_hidden_layers=2,
            mlp_layer_types=["hash_moe"],
        )

    with pytest.raises(ValueError, match=r"len\(layer_types\)"):
        DeepseekV4Config(
            num_hidden_layers=2,
            compress_ratios=[0],
        )


def test_deepseek_v4_rejects_unsupported_legacy_compress_ratio() -> None:
    with pytest.raises(ValueError, match="must be 0, 1, 4, or 128"):
        DeepseekV4Config(
            num_hidden_layers=1,
            compress_ratios=[8],
        )


def test_deepseek_v4_truncates_mtp_padded_compress_ratios() -> None:
    """MTP/DSpark draft entries past ``num_hidden_layers`` are dropped."""
    config = DeepseekV4Config(
        num_hidden_layers=2,
        num_hash_layers=1,
        compress_ratios=[0, 4, 0, 128],
    )
    assert config.layer_types == ["sliding_attention", "compressed_sparse_attention"]
    assert config.mlp_layer_types == ["hash_moe", "moe"]


def test_deepseek_v4_accepts_legacy_compress_ratio_one() -> None:
    config = DeepseekV4Config(num_hidden_layers=1, compress_ratios=[1])
    assert config.layer_types == ["sliding_attention"]


def test_deepseek_v4_draft_layer_resolves_to_sliding_window() -> None:
    config = DeepseekV4Config(num_hidden_layers=2, compress_ratios=[0, 4])
    assert get_deepseek_v4_compress_ratio(config, 0) == 1  # sliding_attention
    assert get_deepseek_v4_compress_ratio(config, 1) == 4  # compressed_sparse_attention
    assert get_deepseek_v4_compress_ratio(config, 2) == 1  # MTP/DSpark draft layer


def test_deepseek_v4_copies_compress_rates() -> None:
    compress_rates = {
        "compressed_sparse_attention": 4,
        "heavily_compressed_attention": 128,
    }

    config = DeepseekV4Config(
        compress_rates=compress_rates,
    )

    assert config.compress_rates is not compress_rates
    config.compress_rates["compressed_sparse_attention"] = 128
    assert compress_rates["compressed_sparse_attention"] == 4
    assert config.compress_rates["compressed_sparse_attention"] == 128


@pytest.mark.parametrize(
    "rope_scaling",
    [
        {
            "type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
        },
        {
            "type": "longrope",
            "short_factor": [1.0],
            "long_factor": [1.0],
        },
    ],
)
def test_llama3_and_longrope_require_an_explicit_original_context_length(
    rope_scaling: dict,
) -> None:
    """Never infer a rotary training length from its extended maximum."""
    with pytest.raises(KeyError, match="original_max_position_embeddings"):
        LlamaConfig(max_position_embeddings=131_072, rope_scaling=rope_scaling)


def test_yarn_rope_without_original_max_keeps_context_scaling() -> None:
    """A YaRN checkpoint that omits the baseline must keep the ``factor``.

    Omitting ``original_max_position_embeddings`` declares
    ``max_position_embeddings`` to be the *pre-extension* length, so the usable
    context is ``factor`` times it. Standardization must not fill the key in --
    doing so makes the config indistinguishable from the already-extended
    spelling below and silently divides the context length by ``factor``.
    """
    config = Qwen3Config.from_dict(
        {
            "max_position_embeddings": 32_768,
            "rope_scaling": {"type": "yarn", "factor": 4.0},
        }
    )

    assert "original_max_position_embeddings" not in config.rope_parameters
    assert get_context_length(config) == 131_072

    # The absence must also survive config serialization; a private provenance
    # marker would be lost during this round trip and recreate the bug.
    reloaded = Qwen3Config.from_dict(config.to_dict())
    assert "original_max_position_embeddings" not in reloaded.rope_parameters
    assert get_context_length(reloaded) == 131_072


def test_yarn_runtime_override_keeps_context_scaling(tmp_path) -> None:
    """The CLI override path must retain YaRN's missing-field convention."""
    raw_config = {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "max_position_embeddings": 32_768,
    }
    (tmp_path / "config.json").write_text(
        json.dumps(raw_config),
        encoding="utf-8",
    )

    config = get_config(
        str(tmp_path),
        model_override_args={"rope_parameters": {"type": "yarn", "factor": 4.0}},
    )

    assert "original_max_position_embeddings" not in config.rope_parameters
    assert get_context_length(config) == 131_072


@pytest.mark.parametrize("override_value", [None, {}])
def test_runtime_rope_parameters_override_can_clear_checkpoint_scaling(
    tmp_path, override_value: dict | None
) -> None:
    """Explicitly empty canonical overrides must replace checkpoint scaling."""
    raw_config = {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "max_position_embeddings": 32_768,
        "rope_parameters": {"rope_type": "yarn", "factor": 4.0},
    }
    (tmp_path / "config.json").write_text(
        json.dumps(raw_config),
        encoding="utf-8",
    )

    config = get_config(
        str(tmp_path),
        model_override_args={"rope_parameters": override_value},
    )

    assert config.rope_parameters == {
        "rope_type": "default",
        "rope_theta": 10_000.0,
    }
    assert get_context_length(config) == 32_768


def test_runtime_legacy_rope_scaling_override_legacy_checkpoint(tmp_path) -> None:
    """A legacy override converts and replaces a legacy checkpoint field."""
    raw_config = {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "max_position_embeddings": 32_768,
        "rope_scaling": {"type": "yarn", "factor": 4.0},
    }
    (tmp_path / "config.json").write_text(
        json.dumps(raw_config),
        encoding="utf-8",
    )

    config = get_config(
        str(tmp_path),
        model_override_args={"rope_scaling": {"type": "yarn", "factor": 8.0}},
    )

    assert config.rope_parameters["rope_type"] == "yarn"
    assert config.rope_parameters["factor"] == 8.0
    assert get_context_length(config) == 32_768 * 8


def test_runtime_legacy_rope_scaling_override_canonical_checkpoint(tmp_path) -> None:
    """A legacy override converts and replaces a canonical checkpoint field."""
    raw_config = {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "max_position_embeddings": 32_768,
        "rope_parameters": {"rope_type": "yarn", "factor": 4.0},
    }
    (tmp_path / "config.json").write_text(
        json.dumps(raw_config),
        encoding="utf-8",
    )

    config = get_config(
        str(tmp_path),
        model_override_args={"rope_scaling": {"type": "yarn", "factor": 8.0}},
    )

    assert config.rope_parameters["rope_type"] == "yarn"
    assert config.rope_parameters["factor"] == 8.0
    assert get_context_length(config) == 32_768 * 8


@pytest.mark.parametrize("override_value", [None, {}])
def test_runtime_legacy_rope_scaling_override_can_clear_checkpoint_scaling(
    tmp_path, override_value: dict | None
) -> None:
    """An explicit legacy null/empty override removes checkpoint scaling."""
    raw_config = {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "max_position_embeddings": 32_768,
        "rope_parameters": {"rope_type": "yarn", "factor": 4.0},
    }
    (tmp_path / "config.json").write_text(
        json.dumps(raw_config),
        encoding="utf-8",
    )

    config = get_config(
        str(tmp_path),
        model_override_args={"rope_scaling": override_value},
    )

    assert config.rope_parameters == {
        "rope_type": "default",
        "rope_theta": 10_000.0,
    }
    assert get_context_length(config) == 32_768


def test_runtime_override_both_rope_spellings_prefers_canonical(tmp_path) -> None:
    """When an override supplies both spellings, canonical ``rope_parameters`` wins."""
    raw_config = {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "max_position_embeddings": 32_768,
    }
    (tmp_path / "config.json").write_text(
        json.dumps(raw_config),
        encoding="utf-8",
    )

    config = get_config(
        str(tmp_path),
        model_override_args={
            "rope_parameters": {"rope_type": "default"},
            "rope_scaling": {"type": "yarn", "factor": 8.0},
        },
    )

    assert config.rope_parameters["rope_type"] == "default"
    assert "factor" not in config.rope_parameters
    assert get_context_length(config) == 32_768


def test_runtime_rope_parameters_remain_without_scaling_override(tmp_path) -> None:
    """An absent legacy override must retain canonical checkpoint scaling."""
    raw_config = {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "max_position_embeddings": 32_768,
        "rope_parameters": {"rope_type": "yarn", "factor": 4.0},
    }
    (tmp_path / "config.json").write_text(
        json.dumps(raw_config),
        encoding="utf-8",
    )

    config = get_config(str(tmp_path))

    assert config.rope_parameters["rope_type"] == "yarn"
    assert config.rope_parameters["factor"] == 4.0
    assert get_context_length(config) == 131_072


def test_runtime_rope_parameters_override_legacy_checkpoint_scaling(tmp_path) -> None:
    """Canonical user overrides must replace legacy checkpoint scaling."""
    raw_config = {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "max_position_embeddings": 32_768,
        "rope_scaling": {"type": "yarn", "factor": 4.0},
    }
    (tmp_path / "config.json").write_text(
        json.dumps(raw_config),
        encoding="utf-8",
    )

    override = {"rope_parameters": {"rope_type": "default"}}
    config = get_config(
        str(tmp_path),
        model_override_args=override,
    )

    assert config.rope_parameters == {
        "rope_type": "default",
        "rope_theta": 10_000.0,
    }
    assert get_context_length(config) == 32_768
    assert override == {"rope_parameters": {"rope_type": "default"}}


def test_checkpoint_legacy_null_rope_scaling_keeps_canonical_rope_parameters() -> None:
    """A checkpoint shipping both canonical and a stale null legacy field must
    keep the canonical scaling rather than treat the null as an explicit clear."""
    config = DeepseekV3Config.from_dict(
        {
            "rope_parameters": {"rope_type": "yarn", "factor": 4.0},
            "rope_scaling": None,
        }
    )

    assert config.rope_parameters["rope_type"] == "yarn"
    assert config.rope_parameters["factor"] == 4.0


def test_get_config_legacy_null_rope_scaling_keeps_canonical(tmp_path) -> None:
    """The no-override ``get_config`` path must preserve canonical scaling when
    the checkpoint also carries a legacy ``rope_scaling: null``."""
    raw_config = {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "max_position_embeddings": 32_768,
        "rope_parameters": {"rope_type": "yarn", "factor": 4.0},
        "rope_scaling": None,
    }
    (tmp_path / "config.json").write_text(json.dumps(raw_config), encoding="utf-8")

    config = get_config(str(tmp_path))

    assert config.rope_parameters["rope_type"] == "yarn"
    assert config.rope_parameters["factor"] == 4.0
    assert get_context_length(config) == 131_072


def test_yarn_rope_with_original_max_does_not_rescale_context() -> None:
    """The other spelling: ``max_position_embeddings`` is already extended."""
    config = Qwen2Config(
        max_position_embeddings=131_072,
        rope_scaling={
            "type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32_768,
        },
    )

    assert config.rope_parameters["original_max_position_embeddings"] == 32_768
    assert get_context_length(config) == 131_072


@pytest.mark.parametrize(
    "max_position_embeddings, rope_scaling",
    [
        # Baseline omitted: max_position_embeddings is the pre-extension length.
        (32_768, {"type": "yarn", "factor": 4.0}),
        # Baseline stated: max_position_embeddings is already extended.
        (
            131_072,
            {
                "type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 32_768,
            },
        ),
    ],
)
@pytest.mark.parametrize("runtime_rope_type", ["yarn", "deepseek_yarn"])
def test_yarn_context_length_matches_rope_extrapolation_baseline(
    max_position_embeddings: int,
    rope_scaling: dict,
    runtime_rope_type: str,
    monkeypatch,
) -> None:
    """``get_context_length`` must agree with the YaRN cache length."""
    from tokenspeed.runtime.layers import rotary_embedding

    config = Qwen2Config(
        max_position_embeddings=max_position_embeddings,
        rope_scaling=rope_scaling,
    )

    constructed: dict[str, int | float] = {}

    def capture_yarn_constructor(*args, **kwargs):
        del kwargs
        constructed["original_max_position"] = args[2]
        constructed["scaling_factor"] = args[5]
        return object()

    monkeypatch.setattr(
        rotary_embedding,
        "YaRNScalingRotaryEmbedding",
        capture_yarn_constructor,
    )
    monkeypatch.setattr(
        rotary_embedding,
        "DeepseekScalingRotaryEmbedding",
        capture_yarn_constructor,
    )
    monkeypatch.setattr(rotary_embedding, "_ROPE_DICT", {})
    runtime_rope_scaling = {
        **config.rope_parameters,
        "rope_type": runtime_rope_type,
    }
    rotary_embedding.get_rope(
        head_size=8,
        rotary_dim=8,
        max_position=config.max_position_embeddings,
        base=10_000,
        rope_scaling=runtime_rope_scaling,
    )

    rope_supports = int(
        constructed["original_max_position"] * constructed["scaling_factor"]
    )

    assert get_context_length(config) == rope_supports


def test_base_config_normalizes_rope_after_checkpoint_fields() -> None:
    config = LlamaConfig(
        max_position_embeddings=8192,
        original_max_position_embeddings=4096,
        rope_scaling={
            "type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
        },
    )

    assert config.original_max_position_embeddings == 4096
    assert config.rope_parameters["original_max_position_embeddings"] == 4096


def test_base_config_normalizes_layered_rope_after_layer_context() -> None:
    config = Qwen3_5TextConfig(
        num_hidden_layers=4,
        full_attention_interval=2,
        rope_parameters={
            "full_attention": {
                "rope_type": "default",
                "rope_theta": 10_000.0,
            },
            "linear_attention": {
                "rope_type": "default",
                "rope_theta": 10_000.0,
            },
        },
    )

    assert config.layer_types == [
        "linear_attention",
        "full_attention",
        "linear_attention",
        "full_attention",
    ]
    assert set(config.rope_parameters) == {"full_attention", "linear_attention"}
    assert all(
        parameters["partial_rotary_factor"] == 0.25
        for parameters in config.rope_parameters.values()
    )


def test_base_config_consumes_legacy_rope_alias_before_setattr() -> None:
    class ReadOnlyRopeConfig(BaseConfig):
        def __init__(self, **kwargs) -> None:
            self.rope_parameters = None
            super().__init__(**kwargs)

        @property
        def rope_scaling(self):
            return self.rope_parameters

    config = ReadOnlyRopeConfig(
        rope_scaling={"type": "linear", "factor": 2.0},
    )

    assert config.rope_parameters == {
        "type": "linear",
        "rope_type": "linear",
        "factor": 2.0,
        "rope_theta": 10_000.0,
    }


def test_base_config_does_not_copy_forwarded_rope_fields() -> None:
    class WrapperConfig(BaseConfig):
        def __init__(self) -> None:
            self.text_config = LlamaConfig()
            super().__init__()

        def __getattr__(self, name: str):
            return getattr(self.text_config, name)

    config = WrapperConfig()

    assert "rope_parameters" not in config.__dict__
    assert config.rope_parameters is config.text_config.rope_parameters


def test_field_annotated_config_is_a_dataclass_without_decorator() -> None:
    """Field-annotated configs become kw-only dataclasses via ``__init_subclass__``."""

    class AnnotatedConfig(BaseConfig):
        foo: int = 1
        bar: str = "x"

    assert "__init__" in AnnotatedConfig.__dict__
    config = AnnotatedConfig(foo=10, checkpoint_extension="kept")

    assert config.foo == 10
    assert config.bar == "x"
    assert config.checkpoint_extension == "kept"


def test_rename_subclass_without_fields_inherits_parent_constructor() -> None:
    """A subclass that only renames a parent keeps the parent's ``__init__``."""

    class VisionConfig(BaseConfig):
        def __init__(self, depth: int = 27, **kwargs) -> None:
            super().__init__(**kwargs)
            self.depth = depth

    class RenamedVisionConfig(VisionConfig):
        model_type = "renamed_vision"

    assert "__init__" not in RenamedVisionConfig.__dict__
    config = RenamedVisionConfig()
    assert config.depth == 27


def test_local_config_import_does_not_import_transformers() -> None:
    code = """
import sys
from tokenspeed.runtime.configs import (
    DeepseekV3Config,
    DeepseekV4Config,
    InklingMMConfig,
    KimiK25Config,
    KimiK3Config,
    KimiK3DSparkConfig,
    LlamaConfig,
    MiniMaxM3Config,
    Qwen2Config,
    Qwen3ASRConfig,
    Qwen3Config,
    Qwen3MoeConfig,
    Qwen3_5Config,
    Qwen3_5MoeConfig,
)
classes = (
    DeepseekV3Config,
    DeepseekV4Config,
    InklingMMConfig,
    KimiK25Config,
    KimiK3Config,
    KimiK3DSparkConfig,
    LlamaConfig,
    MiniMaxM3Config,
    Qwen2Config,
    Qwen3ASRConfig,
    Qwen3Config,
    Qwen3MoeConfig,
    Qwen3_5Config,
    Qwen3_5MoeConfig,
)
assert all(config_class.model_type for config_class in classes)
assert "transformers" not in sys.modules
"""
    python_root = Path(__file__).resolve().parents[2] / "python"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(python_root), env.get("PYTHONPATH")) if part
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=env)

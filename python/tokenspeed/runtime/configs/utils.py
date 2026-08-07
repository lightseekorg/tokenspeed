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

"""Configuration helpers and HuggingFace checkpoint loaders."""

import copy
import json
import logging
import os
from typing import Any

import huggingface_hub
import torch
from huggingface_hub import hf_hub_download, snapshot_download

from tokenspeed.runtime.configs import get_config_class
from tokenspeed.runtime.configs.base_config import BaseConfig
from tokenspeed.runtime.utils import lru_cache_frozenset


def resolve_architecture(config: BaseConfig) -> str:
    """Return ``config.architectures[0]`` or the config class name.

    ``config.architectures`` can be ``None`` on configs that forward
    attribute access to a nested ``text_config`` (e.g. ``Qwen3_5MoeConfig``).
    Callers should use this helper instead of indexing the list directly.
    """
    archs = getattr(config, "architectures", None)
    if archs:
        return archs[0]
    return type(config).__name__


def get_hf_text_config(config: BaseConfig):
    """Get the "sub" config relevant to llm for multi modal models.
    No op for pure text models.
    """
    class_name = resolve_architecture(config)
    if class_name.startswith("Llava") and class_name.endswith("ForCausalLM"):
        # We support non-hf version of llava models, so we do not want to
        # read the wrong values from the unused default text_config.
        # We set `dtype` of config to `torch.float16` for the weights, as
        # `torch.float16` is default used for image features in
        # `python/tokenspeed/runtime/models/llava.py`.
        config.dtype = torch.float16
        return config

    text_config = None
    if hasattr(config, "text_config"):
        # The code operates under the assumption that text_config should have
        # `num_attention_heads` (among others). Check here to fail early
        # if transformers config doesn't align with this assumption.
        if not hasattr(config.text_config, "num_attention_heads"):
            raise AttributeError("text_config must define num_attention_heads.")
        text_config = config.text_config
    if hasattr(config, "language_config"):
        text_config = config.language_config
    if hasattr(config, "thinker_config"):
        # Qwen Omni wrappers keep the language model below thinker_config.
        thinker_config = config.thinker_config
        if hasattr(thinker_config, "text_config"):
            thinker_config.text_config.dtype = thinker_config.dtype
            text_config = thinker_config.text_config
        else:
            text_config = thinker_config

    if text_config is None:
        return config

    if hasattr(config, "quantization_config") and not hasattr(
        text_config, "quantization_config"
    ):
        quantization_config = config.quantization_config
        for key in ["ignore", "ignored_layers", "modules_to_not_convert"]:
            if key in quantization_config and isinstance(
                quantization_config[key], list
            ):
                quantization_config[key] = [
                    (
                        x.replace("language_model.", "")
                        if x.startswith("language_model.")
                        else x
                    )
                    for x in quantization_config[key]
                ]
        text_config.quantization_config = quantization_config

    return text_config


def _text_config_sub_class(
    config_class: type[BaseConfig], key: str
) -> type[BaseConfig] | None:
    """Return the sub-config class registered under ``key``, or None."""
    sub = getattr(config_class, "sub_configs", {})
    cls = sub.get(key)
    return cls if isinstance(cls, type) else None


def _resolve_text_config(
    config_class: type[BaseConfig],
    config_values: dict[str, Any],
) -> tuple[dict[str, Any], type[BaseConfig]]:
    """Resolve ``(dict, class)`` that ``get_hf_text_config`` would select.

    Mirrors the selection order in ``get_hf_text_config``
    (``thinker_config.text_config`` > ``thinker_config`` > ``language_config``
    > ``text_config`` > self), but walks the ``sub_configs`` class tree in
    lockstep so the caller also gets the class that actually parses the
    returned dict. Returning both from a single walk is what keeps the dict
    and its ``attribute_map`` from drifting apart (e.g. a sibling talker text
    config's aliases leaking onto the thinker's dict).
    """
    thinker = config_values.get("thinker_config")
    if isinstance(thinker, dict):
        thinker_cls = _text_config_sub_class(config_class, "thinker_config")
        text = thinker.get("text_config")
        if isinstance(text, dict):
            text_cls = (
                _text_config_sub_class(thinker_cls, "text_config")
                if thinker_cls is not None
                else None
            )
            return text, text_cls or thinker_cls or config_class
        return thinker, thinker_cls or config_class

    language = config_values.get("language_config")
    if isinstance(language, dict):
        language_cls = _text_config_sub_class(config_class, "language_config")
        return language, language_cls or config_class

    text = config_values.get("text_config")
    if isinstance(text, dict):
        text_cls = _text_config_sub_class(config_class, "text_config")
        return text, text_cls or config_class

    return config_values, config_class


_GLM53_FLASH_ARCHITECTURE_ALIASES = {
    "Glm5NextForConditionalGeneration": "Glm53FlashForConditionalGeneration",
    "Glm5NextForConditionalGenerationNextN": (
        "Glm53FlashForConditionalGenerationNextN"
    ),
}


def _apply_glm53_flash_architecture_aliases(
    config_values: dict[str, Any],
) -> None:
    """Collapse legacy ``glm5_next`` architecture names at the config boundary.

    Checkpoints saved before the ``glm5_next`` -> ``glm53_flash`` rename
    declare ``architectures: ["Glm5NextForConditionalGeneration"]``. Rewrite
    those to the current name so the model loader dispatches to the
    ``Glm53Flash*`` entry classes instead of failing to resolve the model.

    Args:
        config_values: The effective raw config dict, mutated in place.
    """
    architectures = config_values.get("architectures")
    if not isinstance(architectures, list):
        return
    config_values["architectures"] = [
        _GLM53_FLASH_ARCHITECTURE_ALIASES.get(arch, arch) for arch in architectures
    ]


def _apply_dflash_aliases(text_values: dict[str, Any]) -> None:
    """Translate a DFLASH/DSpark draft's SWA fields before construction.

    ``Qwen3DSparkModel`` checkpoints declare ``model_type: "qwen3"`` and so
    parse as ``Qwen3Config``, which nulls ``sliding_window`` unless
    ``use_sliding_window`` is set -- a flag these checkpoints never write,
    since they carry the window in ``dflash_config``. Apply the aliases to the
    raw text config so Qwen3 derives ``layer_types`` and validates the final
    sliding-window state in one pass.

    Args:
        text_values: The effective raw text config (the dict selected by
            ``_resolve_text_config``), mutated in place.
    """
    dflash_config = text_values.get("dflash_config")
    if not isinstance(dflash_config, dict):
        return

    explicit_use_sliding_window = "use_sliding_window" in text_values
    sliding_window = text_values.get("sliding_window")
    if explicit_use_sliding_window:
        use_sliding_window = bool(text_values["use_sliding_window"])
    else:
        use_sliding_window = sliding_window is not None or bool(
            dflash_config.get("use_swa")
        )

    if not use_sliding_window:
        return

    text_values["use_sliding_window"] = True
    if sliding_window is not None:
        text_values["sliding_window"] = int(sliding_window)
        return

    dflash_window = dflash_config.get("swa_window_size")
    if dflash_window is not None:
        text_values["sliding_window"] = int(dflash_window)
    elif not explicit_use_sliding_window:
        raise ValueError(
            "`dflash_config.swa_window_size` must be provided when "
            "`dflash_config.use_swa` is enabled"
        )


def _apply_rope_override(
    text_values: dict[str, Any], overrides: dict[str, Any]
) -> None:
    """Reconcile a RoPE override with the checkpoint's RoPE spelling.

    A checkpoint may ship its rope configuration under the legacy
    ``rope_scaling`` key or the canonical ``rope_parameters`` key, and an
    override may use either spelling too. Normalize the override onto the
    canonical field so ``__post_init__`` sees a single authoritative value,
    covering all four checkpoint/override combinations:

    * ``rope_parameters`` override -- written canonical, and the checkpoint's
      stale ``rope_scaling`` is dropped so it cannot shadow the override.
    * ``rope_scaling`` override -- the checkpoint's ``rope_parameters`` is
      dropped and the legacy dict written in its place; ``__post_init__``
      converts it with the same model-specific rules as a legacy checkpoint.
    * Both override keys -- the canonical ``rope_parameters`` wins and the
      legacy spelling is discarded, mirroring ``__post_init__`` precedence.

    Args:
        text_values: The effective raw text config, mutated in place.
        overrides: The override fields; ``rope_scaling``/``rope_parameters``
            are consumed here, remaining fields are left for the caller.
    """
    has_parameters = "rope_parameters" in overrides
    has_scaling = "rope_scaling" in overrides
    override_parameters = overrides.pop("rope_parameters", None)
    override_scaling = overrides.pop("rope_scaling", None)

    if has_parameters:
        text_values["rope_parameters"] = override_parameters
        text_values.pop("rope_scaling", None)
    elif has_scaling:
        text_values.pop("rope_parameters", None)
        text_values["rope_scaling"] = override_scaling


def _apply_alias_override(
    text_values: dict[str, Any],
    overrides: dict[str, Any],
    attribute_map: dict[str, str],
) -> None:
    """Reconcile ``attribute_map`` aliases so an override always wins.

    ``attribute_map`` maps a legacy checkpoint name (key) onto a canonical field
    name (value). A checkpoint may write a field under the legacy alias while an
    override uses the canonical name, or vice versa. After the override is
    merged both keys coexist, and ``__post_init__``'s setattr loop applies the
    leftover alias last, silently shadowing the override. Normalize each pair
    onto the canonical field so the override is the single authoritative value
    regardless of which spelling either side used.

    Args:
        text_values: The effective raw text config, mutated in place.
        overrides: The override fields; alias and canonical spellings are
            consumed here, remaining fields are left for the caller.
        attribute_map: ``alias -> canonical`` pairs of the config that parses
            ``text_values``.
    """
    for alias, canonical in attribute_map.items():
        if canonical in overrides:
            text_values[canonical] = overrides.pop(canonical)
            text_values.pop(alias, None)
        elif alias in overrides:
            text_values[canonical] = overrides.pop(alias)
            text_values.pop(alias, None)


def get_config(
    model: str,
    revision: str | None = None,
    model_override_args: dict | None = None,
    is_draft_worker: bool | None = False,
    speculative_algorithm: str | None = None,
    override_config_file: str | None = None,
):
    # Mirrors transformers' ``_configuration_file`` (default ``config.json``):
    # the config filename within the model directory/repo.
    config_file = override_config_file or "config.json"
    if os.path.isdir(model):
        model_path = model
    else:
        from tokenspeed.runtime.model_loader.weight_utils import get_lock

        with get_lock(model):
            model_path = snapshot_download(
                model,
                revision=revision,
                ignore_patterns=["*.pt", "*.safetensors", "*.bin"],
            )

    try:
        with open(os.path.join(model_path, config_file)) as file:
            raw_config = json.load(file)
    except FileNotFoundError:
        raise RuntimeError(f"Config file not found in {model}. Please check the path.")
    except json.JSONDecodeError:
        raise RuntimeError(
            f"Failed to decode JSON from config file in {model}. Please ensure the file is valid JSON."
        )

    model_type = raw_config.get("model_type")
    config_class = get_config_class(model_type)
    if config_class is None:
        raise ValueError(
            f"Unsupported model_type '{model_type}' for model '{model}'. "
            "Only model types with a config class registered in "
            "tokenspeed.runtime.configs are supported."
        )

    config_values = copy.deepcopy(raw_config)
    text_values, text_config_class = _resolve_text_config(config_class, config_values)
    if model_override_args:
        # An override may spell a field with either its canonical name or a
        # legacy ``attribute_map`` alias, independent of the checkpoint's
        # spelling. Normalize both RoPE and attribute-map aliases onto the
        # canonical field first so the override always wins. The alias map is
        # taken from the config class that actually parses ``text_values``,
        # not from sibling sub-configs.
        overrides = dict(model_override_args)
        _apply_rope_override(text_values, overrides)
        _apply_alias_override(
            text_values,
            overrides,
            getattr(text_config_class, "attribute_map", {}),
        )
        text_values.update(overrides)

    _apply_dflash_aliases(text_values)
    _apply_glm53_flash_architecture_aliases(config_values)

    # Construct from the overridden raw fields so __post_init__ recomputes
    # dependent values such as head_dim and per-layer dispatch schedules.
    config = config_class.from_dict(config_values)
    config.name_or_path = model

    # extract 'text_config'
    text_config = get_hf_text_config(config)

    # quantization config will copy to text_config
    if hasattr(text_config, "quantization_config"):
        if "modules_to_not_convert" in text_config.quantization_config:
            text_config.quantization_config["ignored_layers"] = (
                text_config.quantization_config["modules_to_not_convert"]
            )
            del text_config.quantization_config["modules_to_not_convert"]

    # If the draft head ships in the same checkpoint as the base model,
    # rewrite the architecture in place so the model loader dispatches
    # to the *NextN / *Eagle3 entry class instead of the base one.
    # ``architectures`` may be None when the on-disk config.json lacks the
    # field, so the truthiness checks below stay.
    if (
        is_draft_worker
        and config.architectures
        and config.architectures[0].startswith("Qwen3DSparkModel")
    ):
        config.architectures[0] = "DSparkDraftModel"

    if (
        is_draft_worker
        and config.architectures
        and config.architectures[0]
        in ("Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM")
    ):
        # The Qwen4-Exp MTP head ships in the same checkpoint as the base
        # model. Rewrite both the multimodal and text-only spellings to the
        # single-layer draft entry class; the generic ``+= "NextN"`` below
        # cannot produce the multimodal ``Qwen4ExpForCausalLMNextN`` name.
        # Layer pruning (``num_hidden_layers=1`` / ``layer_types`` /
        # ``ple_layer_ids``) lives in ``Qwen4ExpForCausalLMNextN.__init__``,
        # which deep-copies the text config.
        config.architectures[0] = "Qwen4ExpForCausalLMNextN"

    if (
        is_draft_worker
        and config.architectures
        and "NextN" not in config.architectures[0]
        and "MTP" not in config.architectures[0]
        and "Eagle" not in config.architectures[0]
        and "DFlash" not in config.architectures[0]
        and "DSpark" not in config.architectures[0]
    ):
        if (
            speculative_algorithm == "DSPARK"
            and config.architectures[0] == "DeepseekV4ForCausalLM"
        ):
            config.architectures[0] = "DeepseekV4ForCausalLMDSpark"
        else:
            config.architectures[0] += "NextN"

    if text_config.architectures == ["LlamaForCausalLMNextN"]:
        text_config.num_hidden_layers = 1

    if resolve_architecture(config) in [
        "KimiK25ForConditionalGeneration",
        "KimiK25Config",
        "KimiK3ForConditionalGeneration",
        "KimiK3ForConditionalGenerationNextN",
        "KimiK3Config",
        "Glm53FlashForConditionalGeneration",
        "Glm53FlashForConditionalGenerationNextN",
        "Glm53FlashConfig",
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen3_5MoeForConditionalGenerationNextN",
        "Qwen3_5MoeConfig",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5ForConditionalGenerationNextN",
        "Qwen4ExpForConditionalGeneration",
        "Qwen4ExpForCausalLM",
        "Qwen4ExpForCausalLMNextN",
        "InklingForConditionalGeneration",
        "InklingForConditionalGenerationNextN",
        "InklingMMConfig",
        "Qwen3OmniMoeForConditionalGeneration",
        "Qwen3OmniMoeConfig",
        "Qwen3ASRForConditionalGeneration",
        "Qwen3ASRConfig",
        "MiniMaxM3SparseForConditionalGeneration",
    ]:
        if config is text_config:
            return config
        config.text_config = text_config
        return config

    return text_config


@lru_cache_frozenset(maxsize=32)
def get_generation_config(
    model: str,
    revision: str | None = None,
):
    """Load ``generation_config.json`` as a plain dict, or ``None`` if absent.

    ``GenerationConfig.from_pretrained`` was dropped together with the other
    transformers config interfaces; only ``eos_token_id`` is read downstream,
    so a raw JSON load is sufficient.
    """
    try:
        if os.path.isdir(model):
            generation_config_path = os.path.join(model, "generation_config.json")
        else:
            generation_config_path = hf_hub_download(
                model,
                "generation_config.json",
                revision=revision,
            )

        with open(generation_config_path, encoding="utf-8") as file:
            return json.load(file)
    except (
        OSError,
        huggingface_hub.utils.EntryNotFoundError,
        huggingface_hub.utils.LocalEntryNotFoundError,
    ):
        logging.debug("model doesn't have generation_config.json")
        return None


# Models don't use the same configuration key for determining the maximum
# context length.  Store them here so we can sanely check them.
#  The ordering here is important. Some models have two of these and we
# have a preference for which value gets used.
CONTEXT_LENGTH_KEYS = [
    "max_sequence_length",
    "seq_length",
    "max_seq_len",
    "model_max_length",
    "max_position_embeddings",
]


def get_context_length(config):
    """Get the context length of a model from a Hugging Face model config.

    For YaRN, a missing ``original_max_position_embeddings`` means
    ``max_position_embeddings`` is the pre-extension length and still needs to
    be multiplied by ``factor``. Config normalization must preserve that
    absence; an explicitly present field means the maximum is already scaled.
    """
    text_config = config
    rope_scaling = getattr(text_config, "rope_scaling", None)
    rope_scaling_factor = 1
    if isinstance(rope_scaling, dict) and "factor" in rope_scaling:
        rope_scaling_factor = rope_scaling.get("factor", 1)
        if "original_max_position_embeddings" in rope_scaling:
            rope_scaling_factor = 1
        if rope_scaling.get("rope_type", None) == "llama3":
            rope_scaling_factor = 1

    for key in CONTEXT_LENGTH_KEYS:
        val = getattr(text_config, key, None)
        if val is not None:
            return int(rope_scaling_factor * val)
    return 2048

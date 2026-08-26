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

"""Utilities for Huggingface Transformers."""

import contextlib
import copy
import importlib.util
import json
import logging
import os
import re
import warnings
from collections.abc import Callable
from typing import Any

import torch
from huggingface_hub import snapshot_download
from transformers import (
    AutoConfig,
    AutoTokenizer,
    GenerationConfig,
    PretrainedConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.utils import cached_file

from tokenspeed.runtime.configs import (
    DeepseekV4Config,
    InklingMMConfig,
    InklingModelConfig,
    KimiK2Config,
    KimiK3Config,
    KimiK3DSparkConfig,
    KimiK25Config,
    MiniMaxM2Config,
    MiniMaxM3Config,
    Qwen2Config,
    Qwen3_5Config,
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
    Qwen3ASRConfig,
    Qwen3Config,
    Qwen3MoeConfig,
    Qwen4ExpConfig,
    Qwen4ExpTextConfig,
)
from tokenspeed.runtime.configs.glm53_flash_config import (
    GLM53_FLASH_MODEL_TYPE,
    GLM53_FLASH_TEXT_MODEL_TYPE,
    GLM53_FLASH_VISION_MODEL_TYPE,
    LEGACY_GLM53_FLASH_MODEL_TYPE,
    Glm53FlashConfig,
)
from tokenspeed.runtime.utils import lru_cache_frozenset

_HF_COMMIT_HASH_RE = re.compile(r"[0-9a-f]{40}")
logger = logging.getLogger(__name__)

_CONFIG_REGISTRY: dict[str, type[PretrainedConfig]] = {
    Qwen2Config.model_type: Qwen2Config,
    Qwen3Config.model_type: Qwen3Config,
    Qwen3MoeConfig.model_type: Qwen3MoeConfig,
    Qwen3ASRConfig.model_type: Qwen3ASRConfig,
    DeepseekV4Config.model_type: DeepseekV4Config,
    Qwen3_5Config.model_type: Qwen3_5Config,
    Qwen3_5MoeConfig.model_type: Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig.model_type: Qwen3_5MoeTextConfig,
    Qwen4ExpConfig.model_type: Qwen4ExpConfig,
    Qwen4ExpTextConfig.model_type: Qwen4ExpTextConfig,
    MiniMaxM2Config.model_type: MiniMaxM2Config,
    MiniMaxM3Config.model_type: MiniMaxM3Config,
    KimiK2Config.model_type: KimiK2Config,
    KimiK25Config.model_type: KimiK25Config,
    KimiK3Config.model_type: KimiK3Config,
    KimiK3DSparkConfig.model_type: KimiK3DSparkConfig,
    InklingModelConfig.model_type: InklingModelConfig,
    InklingMMConfig.model_type: InklingMMConfig,
    Glm53FlashConfig.model_type: Glm53FlashConfig,
    LEGACY_GLM53_FLASH_MODEL_TYPE: Glm53FlashConfig,
}

_GLM53_FLASH_ARCHITECTURE_ALIASES = {
    "Glm5NextForConditionalGeneration": "Glm53FlashForConditionalGeneration",
    "Glm5NextForConditionalGenerationNextN": (
        "Glm53FlashForConditionalGenerationNextN"
    ),
}


def _snapshot_commit_hash(snapshot_path: str) -> str | None:
    """Extract an immutable commit from a standard HF snapshot path.

    Args:
        snapshot_path: Local snapshot directory returned by ``snapshot_download``.

    Returns:
        The 40-character lowercase commit hash, or ``None`` for a nonstandard
        cache layout.
    """
    candidate = os.path.basename(os.path.normpath(snapshot_path))
    return candidate if _HF_COMMIT_HASH_RE.fullmatch(candidate) else None


_DEEPSEEK_V4_ENCODING_MODULE_NAME = "_tokenspeed_deepseek_v4_encoding"

for name, cls in _CONFIG_REGISTRY.items():
    with contextlib.suppress(ValueError):
        AutoConfig.register(name, cls)


def resolve_architecture(config: PretrainedConfig) -> str:
    """Return ``config.architectures[0]`` or the config class name.

    ``config.architectures`` can be ``None`` on configs that forward
    attribute access to a nested ``text_config`` (e.g. ``Qwen3_5MoeConfig``).
    Callers should use this helper instead of indexing the list directly.
    """
    archs = getattr(config, "architectures", None)
    if archs:
        return archs[0]
    return type(config).__name__


def get_hf_text_config(config: PretrainedConfig):
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


def _materialize_architectures(config: PretrainedConfig, raw_config: dict) -> None:
    """Ensure ``config.architectures`` resolves to a real ``list[str]``.

    HuggingFace's ``from_pretrained`` sometimes returns a config whose
    ``.architectures`` attribute resolves to ``None`` via ``__getattr__``
    forwarding to a nested text_config (observed on ``Qwen3_5MoeConfig``;
    likely to repeat on any wrapper class with the same pattern). The
    on-disk ``config.json`` is the source of truth, so pin its value
    onto ``config.__dict__`` when the live config has lost it. Bypasses
    ``__setattr__`` deliberately — that's the only way around the
    ``__getattr__`` redirect.

    Silently no-ops when the raw value is missing, empty, or not a
    ``list[str]``; downstream code already handles the absence via
    ``resolve_architecture``.
    """
    if getattr(config, "architectures", None):
        return
    raw_archs = raw_config.get("architectures")
    if not (
        isinstance(raw_archs, list)
        and raw_archs
        and all(isinstance(a, str) for a in raw_archs)
    ):
        return
    config.__dict__["architectures"] = list(raw_archs)


def _normalize_glm53_flash_metadata(config: PretrainedConfig) -> None:
    """Collapse legacy checkpoint names at the config-loading boundary."""
    architectures = getattr(config, "architectures", None) or []
    if not (
        getattr(config, "model_type", None)
        in {GLM53_FLASH_MODEL_TYPE, LEGACY_GLM53_FLASH_MODEL_TYPE}
        or any(arch in _GLM53_FLASH_ARCHITECTURE_ALIASES for arch in architectures)
    ):
        return

    config.model_type = GLM53_FLASH_MODEL_TYPE
    config.__dict__["architectures"] = [
        _GLM53_FLASH_ARCHITECTURE_ALIASES.get(arch, arch) for arch in architectures
    ]
    for nested_name, model_type in (
        ("text_config", GLM53_FLASH_TEXT_MODEL_TYPE),
        ("vision_config", GLM53_FLASH_VISION_MODEL_TYPE),
    ):
        nested = getattr(config, nested_name, None)
        if nested is not None:
            nested.model_type = model_type


def _restore_raw_glm_dsa_fields(config: PretrainedConfig, raw_config: dict) -> None:
    if raw_config.get("architectures") != ["GlmMoeDsaForCausalLM"]:
        return

    # Transformers may rewrite these GLM DSA dimensions; config.json is authoritative.
    for key in (
        "qk_head_dim",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "kv_lora_rank",
        "q_lora_rank",
        "index_topk",
        "index_head_dim",
        "index_n_heads",
        "index_topk_freq",
        "index_skip_topk_offset",
        "index_topk_pattern",
        "indexer_types",
        "indexer_rope_interleave",
        "index_share_for_mtp_iteration",
    ):
        if key in raw_config:
            setattr(config, key, raw_config[key])


def _restore_raw_dflash_fields(config: PretrainedConfig, raw_config: dict) -> None:
    """Re-assert a DFLASH/DSpark draft's sliding window.

    ``Qwen3DSparkModel`` parses as ``Qwen3Config``, which nulls
    ``sliding_window`` unless ``use_sliding_window`` is set -- a flag these
    checkpoints never write, since they carry the window in ``dflash_config``.
    """
    dflash_config = raw_config.get("dflash_config")
    if not isinstance(dflash_config, dict):
        return
    if getattr(config, "sliding_window", None) is not None:
        return

    sliding_window = raw_config.get("sliding_window")
    if sliding_window is None and dflash_config.get("use_swa"):
        sliding_window = dflash_config.get("swa_window_size")
    if sliding_window is None:
        return

    config.sliding_window = int(sliding_window)
    if hasattr(config, "use_sliding_window"):
        # transformers gates the window on this flag; keep the two consistent.
        config.use_sliding_window = True


def get_config(
    model: str,
    trust_remote_code: bool,
    revision: str | None = None,
    model_override_args: dict | None = None,
    is_draft_worker: bool | None = False,
    speculative_algorithm: str | None = None,
    **kwargs,
):
    """Load one config from one revision-pinned metadata snapshot.

    Remote repositories are resolved under the TokenSpeed cross-process lock.
    Ordinary configs parse from that immutable local snapshot. Custom config
    code parses from the original repo at the snapshot's immutable commit,
    still under the lock. This avoids an unlocked second Hub/cache lookup
    racing across tensor-parallel worker processes.
    """

    def load_raw_config(model_path: str) -> dict[str, Any]:
        try:
            with open(os.path.join(model_path, "config.json")) as file:
                return json.load(file)
        except FileNotFoundError:
            raise RuntimeError(
                f"Config file not found in {model}. Please check the path."
            )
        except json.JSONDecodeError:
            raise RuntimeError(
                f"Failed to decode JSON from config file in {model}. "
                "Please ensure the file is valid JSON."
            )

    config = None
    raw_config = None
    is_remote_model = not os.path.isdir(model)
    if is_remote_model:
        from tokenspeed.runtime.model_loader.weight_utils import get_lock

        with get_lock(model):
            model_path = snapshot_download(
                model,
                revision=revision,
                ignore_patterns=["*.pt", "*.safetensors", "*.bin"],
            )
            raw_config = load_raw_config(model_path)

            # Local snapshot parsing is mandatory for ordinary configs: a
            # second remote config lookup can silently construct defaults (as
            # in GB200 run 31909404825). Custom AutoConfig code is the sole
            # exception because Transformers 5.12 resolves its relative
            # imports incorrectly from symlink-backed local snapshots. Keep
            # that remote-code load revision-pinned and inside the same lock.
            if (
                raw_config.get("model_type", "llama") not in _CONFIG_REGISTRY
                and trust_remote_code
            ):
                snapshot_revision = _snapshot_commit_hash(model_path)
                if snapshot_revision is not None:
                    # Keep the lock while Transformers copies executable code
                    # into transformers_modules. Remote code must not call
                    # get_config/get_tokenizer recursively for this repo: the
                    # host-local FileLock is intentionally non-reentrant.
                    config = AutoConfig.from_pretrained(
                        model,
                        trust_remote_code=True,
                        revision=snapshot_revision,
                        **kwargs,
                    )
                else:
                    logger.warning(
                        "Cannot derive an immutable Hugging Face commit from %s; "
                        "parsing custom config code from the local snapshot. "
                        "Remote-code sibling imports may fail in this layout.",
                        model_path,
                    )
    else:
        model_path = model

    if raw_config is None:
        raw_config = load_raw_config(model_path)

    if config is None:
        if raw_config.get("model_type", "llama") in _CONFIG_REGISTRY:
            config_class = _CONFIG_REGISTRY[raw_config["model_type"]]
            config = config_class.from_pretrained(model_path)
        else:
            config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=trust_remote_code, **kwargs
            )

    # Keep user-facing diagnostics and downstream cache keys stable even though
    # parsing is deliberately pinned to the immutable local snapshot.
    config._name_or_path = model

    _materialize_architectures(config, raw_config)
    _normalize_glm53_flash_metadata(config)
    _restore_raw_glm_dsa_fields(config, raw_config)
    _restore_raw_dflash_fields(config, raw_config)

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
    # ``architectures`` is guaranteed non-None here when the on-disk
    # config.json declared it (see the source-of-truth pin above);
    # the truthiness check stays for configs that genuinely lack the
    # field.
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
        config.architectures[0] = "Qwen4ExpForCausalLMNextN"
        text_config.num_hidden_layers = 1
        text_config.layer_types = ["full_attention"]
        text_config.ple_layer_ids = []

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
        elif config.architectures[0] == "MiniMaxM2ForCausalLM":
            config.architectures[0] = "LlamaForCausalLMEagle3"
        else:
            config.architectures[0] += "NextN"

    if text_config.architectures == ["LlamaForCausalLMNextN"]:
        text_config.num_hidden_layers = 1

    if model_override_args:
        text_config.update(model_override_args)

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
    trust_remote_code: bool,
    revision: str | None = None,
    **kwargs,
):
    """Load generation metadata from one revision-pinned local snapshot."""
    try:
        model_path = model
        if not os.path.isdir(model):
            from tokenspeed.runtime.model_loader.weight_utils import get_lock

            with get_lock(model):
                model_path = snapshot_download(
                    model,
                    revision=revision,
                    ignore_patterns=["*.pt", "*.safetensors", "*.bin"],
                )
        return GenerationConfig.from_pretrained(
            model_path, trust_remote_code=trust_remote_code, **kwargs
        )
    except OSError:
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
    """Get the context length of a model from a huggingface model configs."""
    text_config = config
    rope_scaling = getattr(text_config, "rope_scaling", None)
    if rope_scaling:
        rope_scaling_factor = rope_scaling.get("factor", 1)
        if "original_max_position_embeddings" in rope_scaling:
            rope_scaling_factor = 1
        if rope_scaling.get("rope_type", None) == "llama3":
            rope_scaling_factor = 1
    else:
        rope_scaling_factor = 1

    for key in CONTEXT_LENGTH_KEYS:
        val = getattr(text_config, key, None)
        if val is not None:
            return int(rope_scaling_factor * val)
    return 2048


# A fast LLaMA tokenizer with the pre-processed `tokenizer.json` file.
_FAST_LLAMA_TOKENIZER = "hf-internal-testing/llama-tokenizer"


# Architectures for which ``tokenizer.json`` encodes the exact pre-tokenizer
# / normalizer the model was trained with, and whose AutoTokenizer defaults
# diverge from that. Kimi-K2.5 ships a custom ``TikTokenTokenizer`` via
# ``trust_remote_code`` that AutoTokenizer already handles correctly, so this
# verbatim tokenizer path must stay architecture-gated.
_VERBATIM_TOKENIZER_ARCHITECTURES: frozenset = frozenset(
    {
        "MiniMaxM2ForCausalLM",
    }
)
_DEEPSEEK_V4_TOKENIZER_ARCHITECTURES: frozenset = frozenset(
    {
        "DeepseekV4ForCausalLM",
    }
)


def prefers_verbatim_fast_tokenizer(architectures: list[str] | None) -> bool:
    """True if the model's architectures warrant bypassing AutoTokenizer and
    loading ``PreTrainedTokenizerFast`` from ``tokenizer.json`` verbatim.
    """
    if not architectures:
        return False
    return any(arch in _VERBATIM_TOKENIZER_ARCHITECTURES for arch in architectures)


def prefers_deepseek_v4_tokenizer(architectures: list[str] | None) -> bool:
    if not architectures:
        return False
    return any(arch in _DEEPSEEK_V4_TOKENIZER_ARCHITECTURES for arch in architectures)


def _find_deepseek_v4_encoding_file(
    tokenizer_name: str,
    tokenizer_revision: str | None,
) -> str:
    if os.path.isdir(tokenizer_name):
        encoding_path = os.path.join(tokenizer_name, "encoding", "encoding_dsv4.py")
        if os.path.exists(encoding_path):
            return encoding_path
        raise RuntimeError(
            "DeepSeek V4 tokenizer mode requires "
            f"`encoding/encoding_dsv4.py` in {tokenizer_name}."
        )

    try:
        encoding_path = cached_file(
            tokenizer_name,
            "encoding/encoding_dsv4.py",
            revision=tokenizer_revision,
            _raise_exceptions_for_gated_repo=False,
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=False,
        )
    except TypeError:
        encoding_path = cached_file(
            tokenizer_name,
            "encoding/encoding_dsv4.py",
            revision=tokenizer_revision,
        )

    if not encoding_path:
        raise RuntimeError(
            "DeepSeek V4 tokenizer mode requires "
            "`encoding/encoding_dsv4.py` from the model repository."
        )
    return encoding_path


def _load_deepseek_v4_encode_messages(
    tokenizer_name: str,
    tokenizer_revision: str | None,
) -> Callable[..., str]:
    encoding_path = _find_deepseek_v4_encoding_file(tokenizer_name, tokenizer_revision)
    spec = importlib.util.spec_from_file_location(
        _DEEPSEEK_V4_ENCODING_MODULE_NAME, encoding_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load DeepSeek V4 encoding from {encoding_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    encode_messages = getattr(module, "encode_messages", None)
    if encode_messages is None:
        raise RuntimeError(f"{encoding_path} does not define encode_messages")
    return encode_messages


def _wrap_deepseek_v4_tokenizer(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    encode_messages: Callable[..., str],
) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    """Attach DeepSeek V4's model-provided chat encoder to a HF tokenizer.

    This loads the official encoder from the checkpoint instead of vendoring it
    in TokenSpeed.
    """

    dsv4_tokenizer = copy.copy(tokenizer)
    added_vocab = tokenizer.get_added_vocab()
    added_vocab_size = len(added_vocab)
    tokenizer_vocab_size = tokenizer.vocab_size

    class _DeepseekV4Tokenizer(tokenizer.__class__):  # type: ignore
        def apply_chat_template(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            **kwargs,
        ):
            thinking = kwargs.get("thinking", False) or kwargs.get(
                "enable_thinking", False
            )
            conversation = kwargs.get("conversation", messages)
            conversation = conversation.copy()
            if tools:
                conversation.insert(0, {"role": "system", "tools": tools})

            reasoning_effort = kwargs.get("reasoning_effort")
            if reasoning_effort not in ("max", "high"):
                reasoning_effort = None

            prompt = encode_messages(
                conversation,
                thinking_mode="thinking" if thinking else "chat",
                drop_thinking=kwargs.get("drop_thinking", True),
                reasoning_effort=reasoning_effort,
            )

            if not kwargs.get("tokenize", True):
                return prompt

            return_dict = kwargs.get("return_dict", False)
            forwarded_keys = (
                "truncation",
                "max_length",
                "padding",
                "return_tensors",
                "return_attention_mask",
                "return_token_type_ids",
                "return_special_tokens_mask",
                "return_offsets_mapping",
                "return_length",
            )
            forwarded = {k: kwargs[k] for k in forwarded_keys if k in kwargs}
            encoding = self(prompt, add_special_tokens=False, **forwarded)
            if return_dict:
                return encoding
            return encoding["input_ids"]

        def num_special_tokens_to_add(self) -> int:
            return len(self.encode(""))

        def __len__(self) -> int:
            return tokenizer_vocab_size + added_vocab_size

        def get_added_vocab(self) -> dict[str, int]:
            return added_vocab.copy()

    _DeepseekV4Tokenizer.__name__ = f"DSV4{tokenizer.__class__.__name__}"
    dsv4_tokenizer.__class__ = _DeepseekV4Tokenizer
    return dsv4_tokenizer


def get_tokenizer(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    tokenizer_revision: str | None = None,
    revision: str | None = None,
    architectures: list[str] | None = None,
    **kwargs,
) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    """Gets a tokenizer for the given model name via Huggingface.

    Remote tokenizers are downloaded once under the TokenSpeed cross-process
    lock. Ordinary tokenizers parse from that local snapshot. Custom tokenizer
    code parses from the original repo at the snapshot's immutable commit,
    while still holding the lock, so Transformers can resolve sibling imports.

    ``architectures`` is the model's ``config.architectures`` list (caller
    should pass it when available). It gates whether we bypass AutoTokenizer
    and load ``PreTrainedTokenizerFast`` from ``tokenizer.json`` verbatim —
    needed for a small set of models (e.g. MiniMax-M2) whose AutoTokenizer
    defaults diverge from training. Models with custom tokenizer classes
    loaded via ``trust_remote_code`` (e.g. Kimi-K2.5's ``TikTokenTokenizer``)
    must NOT go through the verbatim path; leaving ``architectures`` as None
    (the default) keeps the safe AutoTokenizer-only behavior.

    ``revision`` is the production-facing alias for ``tokenizer_revision``.
    When both are provided they must name the same snapshot.
    """
    if tokenizer_revision is not None and revision is not None:
        if tokenizer_revision != revision:
            raise ValueError(
                f"tokenizer_revision ({tokenizer_revision!r}) and revision "
                f"({revision!r}) must match when both are set."
            )
    elif tokenizer_revision is None:
        tokenizer_revision = revision

    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError("Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False

    tokenizer_path = tokenizer_name
    tokenizer = None

    def load_tokenizer(
        auto_tokenizer_target: str,
        auto_tokenizer_revision: str | None = None,
    ) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        fast_tokenizer = None
        if (
            tokenizer_mode != "slow"
            and kwargs.get("use_fast", True)
            and prefers_verbatim_fast_tokenizer(architectures)
        ):
            try:
                fast_tokenizer = PreTrainedTokenizerFast.from_pretrained(
                    tokenizer_path,
                    *args,
                    clean_up_tokenization_spaces=False,
                )
            except Exception:
                fast_tokenizer = None

        auto_tokenizer_kwargs = dict(kwargs)
        if auto_tokenizer_revision is not None:
            auto_tokenizer_kwargs["revision"] = auto_tokenizer_revision

        try:
            loaded_tokenizer = AutoTokenizer.from_pretrained(
                auto_tokenizer_target,
                *args,
                trust_remote_code=trust_remote_code,
                clean_up_tokenization_spaces=False,
                **auto_tokenizer_kwargs,
            )
        except TypeError as e:
            # The LLaMA tokenizer causes a protobuf error in some environments.
            err_msg = (
                "Failed to load the tokenizer. If you are using a LLaMA V1 model "
                f"consider using '{_FAST_LLAMA_TOKENIZER}' instead of the "
                "original tokenizer."
            )
            raise RuntimeError(err_msg) from e
        except ValueError as e:
            # If the error pertains to the tokenizer class not existing or not
            # currently being imported, suggest using --trust-remote-code.
            if not trust_remote_code and (
                "does not exist or is not currently imported." in str(e)
                or "requires you to execute the tokenizer file" in str(e)
            ):
                err_msg = (
                    "Failed to load the tokenizer. If the tokenizer is a custom "
                    "tokenizer not yet available in the HuggingFace transformers "
                    "library, consider setting `trust_remote_code=True` in LLM "
                    "or using the `--trust-remote-code` flag in the CLI."
                )
                raise RuntimeError(err_msg) from e
            raise

        # Swap in the fast tokenizer, carrying over chat_template from
        # tokenizer_config.json if tokenizer.json doesn't have one.
        if fast_tokenizer is not None and fast_tokenizer is not loaded_tokenizer:
            if getattr(loaded_tokenizer, "chat_template", None) and not getattr(
                fast_tokenizer, "chat_template", None
            ):
                fast_tokenizer.chat_template = loaded_tokenizer.chat_template
            loaded_tokenizer = fast_tokenizer

        if not isinstance(loaded_tokenizer, PreTrainedTokenizerFast):
            warnings.warn(
                "Using a slow tokenizer. This might cause a significant "
                "slowdown. Consider using a fast tokenizer instead."
            )

        if tokenizer_mode == "auto" and prefers_deepseek_v4_tokenizer(architectures):
            loaded_tokenizer = _wrap_deepseek_v4_tokenizer(
                loaded_tokenizer,
                _load_deepseek_v4_encode_messages(tokenizer_path, tokenizer_revision),
            )
        return loaded_tokenizer

    if not os.path.isdir(tokenizer_name):
        from tokenspeed.runtime.model_loader.weight_utils import get_lock

        with get_lock(tokenizer_name):
            tokenizer_path = snapshot_download(
                tokenizer_name,
                revision=tokenizer_revision,
                ignore_patterns=["*.pt", "*.safetensors", "*.bin"],
            )
            snapshot_revision = _snapshot_commit_hash(tokenizer_path)
            if trust_remote_code and snapshot_revision is not None:
                tokenizer = load_tokenizer(tokenizer_name, snapshot_revision)
            elif trust_remote_code:
                logger.warning(
                    "Cannot derive an immutable Hugging Face commit from %s; "
                    "parsing custom tokenizer code from the local snapshot. "
                    "Remote-code sibling imports may fail in this layout.",
                    tokenizer_path,
                )

    if tokenizer is None:
        tokenizer = load_tokenizer(tokenizer_path)

    tokenizer.name_or_path = tokenizer_name
    if isinstance(getattr(tokenizer, "init_kwargs", None), dict):
        tokenizer.init_kwargs["name_or_path"] = tokenizer_name
    attach_additional_stop_token_ids(tokenizer)
    return tokenizer


def attach_additional_stop_token_ids(tokenizer):
    # Special handling for stop token <|eom_id|> generated by llama 3 tool use.
    if "<|eom_id|>" in tokenizer.get_added_vocab():
        tokenizer.additional_stop_token_ids = set(
            [tokenizer.get_added_vocab()["<|eom_id|>"]]
        )
    else:
        tokenizer.additional_stop_token_ids = None

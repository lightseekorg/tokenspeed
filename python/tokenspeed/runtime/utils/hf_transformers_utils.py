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
import warnings
from collections.abc import Callable
from typing import Any

import torch
from huggingface_hub import snapshot_download
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PretrainedConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.utils import cached_file

from tokenspeed.runtime.configs import (
    DeepseekV4Config,
    KimiK2Config,
    MiniMaxM2Config,
    Qwen2Config,
    Qwen3_5Config,
    Qwen3_5MoeConfig,
    Qwen3Config,
)
from tokenspeed.runtime.utils import lru_cache_frozenset

_CONFIG_REGISTRY: dict[str, type[PretrainedConfig]] = {
    Qwen2Config.model_type: Qwen2Config,
    Qwen3Config.model_type: Qwen3Config,
    DeepseekV4Config.model_type: DeepseekV4Config,
    Qwen3_5Config.model_type: Qwen3_5Config,
    Qwen3_5MoeConfig.model_type: Qwen3_5MoeConfig,
    MiniMaxM2Config.model_type: MiniMaxM2Config,
    KimiK2Config.model_type: KimiK2Config,
}

_DEEPSEEK_V4_ENCODING_MODULE_NAME = "_tokenspeed_deepseek_v4_encoding"

for name, cls in _CONFIG_REGISTRY.items():
    with contextlib.suppress(ValueError):
        AutoConfig.register(name, cls)


def get_hf_text_config(config: PretrainedConfig):
    """Get the "sub" config relevant to llm for multi modal models.
    No op for pure text models.
    """
    if config.architectures is not None:
        class_name = config.architectures[0]
        if class_name.startswith("Llava") and class_name.endswith("ForCausalLM"):
            # We support non-hf version of llava models, so we do not want to
            # read the wrong values from the unused default text_config.
            #  We set `torch_dtype` of config to `torch.float16` for the weights, as
            # `torch.float16` is default used for image features in `python/tokenspeed/runtime/models/llava.py`.
            setattr(config, "torch_dtype", torch.float16)
            return config

    text_config = None
    if hasattr(config, "text_config"):
        # The code operates under the assumption that text_config should have
        # `num_attention_heads` (among others). Assert here to fail early
        # if transformers config doesn't align with this assumption.
        assert hasattr(config.text_config, "num_attention_heads")
        text_config = config.text_config
    if hasattr(config, "language_config"):
        text_config = config.language_config
    if hasattr(config, "thinker_config"):
        # qwen2.5 omni
        thinker_config = config.thinker_config
        if hasattr(thinker_config, "text_config"):
            setattr(
                thinker_config.text_config,
                "torch_dtype",
                getattr(thinker_config, "torch_dtype", None),
            )
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


def get_config(
    model: str,
    trust_remote_code: bool,
    revision: str | None = None,
    model_override_args: dict | None = None,
    is_draft_worker: bool | None = False,
    **kwargs,
):
    if os.path.isdir(model):
        model_path = model
    else:
        model_path = snapshot_download(
            model, ignore_patterns=["*.pt", "*.safetensors", "*.bin"]
        )

    try:
        with open(os.path.join(model_path, "config.json")) as file:
            config = json.load(file)
    except FileNotFoundError:
        raise RuntimeError(f"Config file not found in {model}. Please check the path.")
    except json.JSONDecodeError:
        raise RuntimeError(
            f"Failed to decode JSON from config file in {model}. Please ensure the file is valid JSON."
        )
    if config.get("model_type", "llama") in _CONFIG_REGISTRY:
        config_class = _CONFIG_REGISTRY[config["model_type"]]
        config = config_class.from_pretrained(model, revision=revision)
        setattr(config, "_name_or_path", model)
    else:
        try:
            config = AutoConfig.from_pretrained(
                model, trust_remote_code=trust_remote_code, revision=revision, **kwargs
            )
        except ValueError as e:
            raise e

    # extract 'text_config'
    text_config = get_hf_text_config(config)

    # quantization config will copy to text_config
    if hasattr(text_config, "quantization_config"):
        if "modules_to_not_convert" in text_config.quantization_config:
            text_config.quantization_config["ignored_layers"] = (
                text_config.quantization_config["modules_to_not_convert"]
            )
            del text_config.quantization_config["modules_to_not_convert"]

    # Adapt to the case where the draft head and base model are placed together.
    # Check if config.architectures already points at a draft implementation.
    if (
        is_draft_worker
        and "NextN" not in config.architectures[0]
        and "Eagle" not in config.architectures[0]
    ):
        if config.architectures[0] == "MiniMaxM2ForCausalLM":
            config.architectures[0] = "LlamaForCausalLMEagle3"
        else:
            config.architectures[0] += "NextN"

    if text_config.architectures == ["LlamaForCausalLMNextN"]:
        text_config.num_hidden_layers = 1

    if model_override_args:
        text_config.update(model_override_args)

    # update text_config for qwen3.5
    # config.architectures may be None if Qwen3_5MoeConfig.__getattr__ redirects to
    # text_config (which has architectures=None) in certain subprocess environments.
    # Fall back to checking the config class name when architectures is unavailable.
    _arch = config.architectures[0] if config.architectures else type(config).__name__
    if _arch in [
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen3_5MoeForConditionalGenerationNextN",
        "Qwen3_5MoeConfig",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5ForConditionalGenerationNextN",
    ]:
        config.text_config = text_config
        # Ensure architectures is set in the instance __dict__ so downstream code
        # doesn't get redirected via __getattr__ to text_config.architectures
        # (which may be ['Qwen2ForCausalLM'] or None).
        if not config.architectures:
            config.__dict__["architectures"] = ["Qwen3_5MoeForConditionalGeneration"]
        return config

    return text_config


@lru_cache_frozenset(maxsize=32)
def get_generation_config(
    model: str,
    trust_remote_code: bool,
    revision: str | None = None,
    **kwargs,
):
    try:
        return GenerationConfig.from_pretrained(
            model, trust_remote_code=trust_remote_code, revision=revision, **kwargs
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
    architectures: list[str] | None = None,
    **kwargs,
) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    """Gets a tokenizer for the given model name via Huggingface.

    ``architectures`` is the model's ``config.architectures`` list (caller
    should pass it when available). It gates whether we bypass AutoTokenizer
    and load ``PreTrainedTokenizerFast`` from ``tokenizer.json`` verbatim —
    needed for a small set of models (e.g. MiniMax-M2) whose AutoTokenizer
    defaults diverge from training. Models with custom tokenizer classes
    loaded via ``trust_remote_code`` (e.g. Kimi-K2.5's ``TikTokenTokenizer``)
    must NOT go through the verbatim path; leaving ``architectures`` as None
    (the default) keeps the safe AutoTokenizer-only behavior.
    """
    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError("Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False

    fast_tokenizer = None
    if (
        tokenizer_mode != "slow"
        and kwargs.get("use_fast", True)
        and prefers_verbatim_fast_tokenizer(architectures)
    ):
        try:
            fast_tokenizer = PreTrainedTokenizerFast.from_pretrained(
                tokenizer_name,
                *args,
                revision=tokenizer_revision,
                clean_up_tokenization_spaces=False,
            )
        except Exception:
            fast_tokenizer = None

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            *args,
            trust_remote_code=trust_remote_code,
            tokenizer_revision=tokenizer_revision,
            clean_up_tokenization_spaces=False,
            **kwargs,
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
        # currently being imported, suggest using the --trust-remote-code flag.
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
        else:
            raise e

    # Swap in the fast tokenizer, carrying over chat_template from
    # tokenizer_config.json if tokenizer.json doesn't have one.
    if fast_tokenizer is not None and fast_tokenizer is not tokenizer:
        if getattr(tokenizer, "chat_template", None) and not getattr(
            fast_tokenizer, "chat_template", None
        ):
            fast_tokenizer.chat_template = tokenizer.chat_template
        tokenizer = fast_tokenizer

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        warnings.warn(
            "Using a slow tokenizer. This might cause a significant "
            "slowdown. Consider using a fast tokenizer instead."
        )

    if tokenizer_mode == "auto" and prefers_deepseek_v4_tokenizer(architectures):
        tokenizer = _wrap_deepseek_v4_tokenizer(
            tokenizer,
            _load_deepseek_v4_encode_messages(tokenizer_name, tokenizer_revision),
        )

    attach_additional_stop_token_ids(tokenizer)
    return tokenizer


def get_processor(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    tokenizer_revision: str | None = None,
    **kwargs,
):
    processor = AutoProcessor.from_pretrained(
        tokenizer_name,
        *args,
        trust_remote_code=trust_remote_code,
        tokenizer_revision=tokenizer_revision,
        **kwargs,
    )

    attach_additional_stop_token_ids(processor.tokenizer)
    return processor


def attach_additional_stop_token_ids(tokenizer):
    # Special handling for stop token <|eom_id|> generated by llama 3 tool use.
    if "<|eom_id|>" in tokenizer.get_added_vocab():
        tokenizer.additional_stop_token_ids = set(
            [tokenizer.get_added_vocab()["<|eom_id|>"]]
        )
    else:
        tokenizer.additional_stop_token_ids = None

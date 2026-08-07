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

"""Tokenizer loading via HuggingFace transformers."""

import copy
import importlib.util
import logging
import os
import re
import warnings
from collections.abc import Callable
from typing import Any

from huggingface_hub import snapshot_download
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.utils import cached_file

_HF_COMMIT_HASH_RE = re.compile(r"[0-9a-f]{40}")
logger = logging.getLogger(__name__)

_DEEPSEEK_V4_ENCODING_MODULE_NAME = "_tokenspeed_deepseek_v4_encoding"

# A fast LLaMA tokenizer with the pre-processed `tokenizer.json` file.
_FAST_LLAMA_TOKENIZER = "hf-internal-testing/llama-tokenizer"


_DEEPSEEK_V4_TOKENIZER_ARCHITECTURES: frozenset = frozenset(
    {
        "DeepseekV4ForCausalLM",
    }
)


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
    should pass it when available). It gates whether the loaded tokenizer is
    wrapped with DeepSeek V4's model-provided chat encoder.

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

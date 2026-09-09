# MIT License
#
# Copyright (c) 2026 TokenSpeed contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import importlib.util
from types import MethodType
from typing import Any

from transformers import AutoTokenizer
from transformers.utils.hub import cached_file


def load_tokenizer(model: str, revision: str):
    """Load a revision-pinned tokenizer with the checkpoint's V4 chat encoding."""
    tokenizer = AutoTokenizer.from_pretrained(
        model,
        revision=revision,
        trust_remote_code=True,
        clean_up_tokenization_spaces=False,
    )
    encoding_path = cached_file(
        model,
        "encoding/encoding_dsv4.py",
        revision=revision,
    )
    spec = importlib.util.spec_from_file_location("deepseek_v4_encoding", encoding_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load DeepSeek V4 encoding from {encoding_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        **kwargs,
    ):
        thinking = kwargs.get("thinking", False) or kwargs.get("enable_thinking", False)
        conversation = kwargs.get("conversation", messages).copy()
        if tools:
            conversation.insert(0, {"role": "system", "tools": tools})

        reasoning_effort = kwargs.get("reasoning_effort")
        if reasoning_effort not in ("max", "high"):
            reasoning_effort = None

        prompt = module.encode_messages(
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
        forwarded = {key: kwargs[key] for key in forwarded_keys if key in kwargs}
        encoding = self(prompt, add_special_tokens=False, **forwarded)
        if return_dict:
            return encoding
        return encoding["input_ids"]

    tokenizer.apply_chat_template = MethodType(apply_chat_template, tokenizer)
    return tokenizer

# Adapted from meituan-longcat/SGLang-FluentLLM.
# This file has been modified for this repository.
# This file may incorporate material from ModelTC/lightllm,
# vllm-project/vllm, and sgl-project/sglang, as identified in
# python/THIRDPARTYNOTICES.
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

import inspect
import logging
from typing import Any, Union, get_args, get_origin

from tokenspeed.runtime.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    LogProbs,
)

logger = logging.getLogger(__name__)


def to_openai_style_logprobs(
    input_token_logprobs=None,
    output_token_logprobs=None,
    input_top_logprobs=None,
    output_top_logprobs=None,
):
    ret_logprobs = LogProbs()

    def append_token_logprobs(token_logprobs):
        for logprob, _, token_text in token_logprobs:
            ret_logprobs.tokens.append(token_text)
            ret_logprobs.token_logprobs.append(logprob)

            # Not supported yet
            ret_logprobs.text_offset.append(-1)

    def append_top_logprobs(top_logprobs):
        for tokens in top_logprobs:
            if tokens is not None:
                ret_logprobs.top_logprobs.append(
                    {token[2]: token[0] for token in tokens}
                )
            else:
                ret_logprobs.top_logprobs.append(None)

    if input_token_logprobs is not None:
        append_token_logprobs(input_token_logprobs)
    if output_token_logprobs is not None:
        append_token_logprobs(output_token_logprobs)
    if input_top_logprobs is not None:
        append_top_logprobs(input_top_logprobs)
    if output_top_logprobs is not None:
        append_top_logprobs(output_top_logprobs)

    return ret_logprobs


def process_hidden_states_from_ret(
    ret_item: dict[str, Any],
    request: ChatCompletionRequest | CompletionRequest,
) -> list | None:
    """Process hidden states from a ret item in non-streaming response.

    Args:
        ret_item: Response item containing meta_info
        request: The original request object

    Returns:
        Processed hidden states for the last token, or None
    """
    if not request.return_hidden_states:
        return None

    hidden_states = ret_item["meta_info"].get("hidden_states", None)
    if hidden_states is not None:
        hidden_states = hidden_states[-1] if len(hidden_states) > 1 else []
    return hidden_states


def recursive_type_check(value, expected_type, param_name="参数"):
    origin = get_origin(expected_type)
    args = get_args(expected_type)

    # Optional[X] == Union[X, NoneType]
    if origin is Union:
        # None value only allowed in Union
        if value is None:
            if type(None) in args:
                return
            else:
                raise ValueError(f"{param_name} does not allow None")
        # Value just needs to match any type in Union
        for typ in args:
            try:
                recursive_type_check(value, typ, param_name)
                return  # Just need one type to pass
            except ValueError:
                continue
        # Only throw error if none pass
        allowed_types = [t.__name__ if hasattr(t, "__name__") else str(t) for t in args]
        raise ValueError(
            f"{param_name} type error, expected {'|'.join(allowed_types)}, got {type(value).__name__}"
        )

    # Non-generic, direct judgment
    if origin is None:
        if not isinstance(value, expected_type):
            raise ValueError(
                f"{param_name} type error, expected {expected_type.__name__}, got {type(value).__name__}"
            )
        return

    # List[T]
    if origin is list:
        if not isinstance(value, list):
            raise ValueError(
                f"{param_name} type error, expected list, got {type(value).__name__}"
            )
        if args:
            for i, item in enumerate(value):
                try:
                    recursive_type_check(item, args[0], f"{param_name}[{i}]")
                except ValueError as e:
                    raise e
        return

    # Tuple[T1, T2, ...]
    if origin is tuple:
        if not isinstance(value, tuple):
            raise ValueError(
                f"{param_name} type error, expected tuple, got {type(value).__name__}"
            )
        if args:
            if len(args) == 2 and args[1] is Ellipsis:
                # Tuple[T, ...]
                for i, item in enumerate(value):
                    recursive_type_check(item, args[0], f"{param_name}[{i}]")
            else:
                if len(value) != len(args):
                    raise ValueError(
                        f"{param_name} length error, expected {len(args)}, got {len(value)}"
                    )
                for i, (item, typ) in enumerate(zip(value, args)):
                    recursive_type_check(item, typ, f"{param_name}[{i}]")
        return

    # Dict[K, V]
    if origin is dict:
        if not isinstance(value, dict):
            raise ValueError(
                f"{param_name} type error, expected dict, got {type(value).__name__}"
            )
        if args and len(args) == 2:
            key_type, val_type = args
            for k, v in value.items():
                recursive_type_check(k, key_type, f"{param_name}.key")
                recursive_type_check(v, val_type, f"{param_name}[{repr(k)}]")
        return

    # Set[T]
    if origin is set:
        if not isinstance(value, set):
            raise ValueError(
                f"{param_name} type error, expected set, got {type(value).__name__}"
            )
        if args:
            for i, item in enumerate(value):
                recursive_type_check(item, args[0], f"{param_name}[{i}]")
        return


def validate_sampling_params(params: dict, cls):
    sig = inspect.signature(cls.__init__)
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        expected_type = param.annotation
        has_default = param.default != inspect._empty
        value = params.get(name)
        if value is None and not has_default:
            raise ValueError(f"Sampling parameter '{name}' is missing")
        if value is not None and expected_type is not inspect._empty:
            recursive_type_check(
                value, expected_type, f"sampling parameter '{name=}'|'{value=}'"
            )

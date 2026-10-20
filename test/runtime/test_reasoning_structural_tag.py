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

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from transformers import AutoTokenizer

from tokenspeed.runtime.engine.input_processor import InputProcessor
from tokenspeed.runtime.engine.io_struct import GenerateReqInput
from tokenspeed.runtime.grammar.reasoning_structural_tag import (
    prompt_ends_in_reasoning,
    reasoning_block_markers,
    structural_tag_for_reasoning_json_schema,
)

xgrammar = pytest.importorskip("xgrammar")

_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}
_MESSAGES = [
    {"role": "user", "content": "What is the capital of France? Reply in JSON."}
]
_ANSWER = '{"answer": "Paris"}'
_WRONG_ANSWER = '{"answer": 1}'


@pytest.mark.parametrize("parser", ["deepseek_v31", "deepseek_v41"])
def test_deepseek_reasoning_parser_wraps_json_schema_after_thinking(parser):
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }

    structural_tag = structural_tag_for_reasoning_json_schema(
        parser, schema, prompt_opens_reasoning=True
    )

    assert structural_tag is not None
    payload = json.loads(structural_tag)
    elements = payload["format"]["elements"]
    assert payload["type"] == "structural_tag"
    assert payload["format"]["type"] == "sequence"
    assert elements[0]["type"] == "tag"
    assert elements[0]["begin"] in ("", "<think>")
    assert elements[0]["end"] == "</think>"
    assert elements[-1]["type"] == "json_schema"
    assert elements[-1]["json_schema"] == schema


def test_qwen3_reasoning_parser_uses_xgrammar_0_2_builtin_name():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }

    structural_tag = structural_tag_for_reasoning_json_schema(
        "qwen3", schema, prompt_opens_reasoning=True
    )

    assert structural_tag is not None
    payload = json.loads(structural_tag)
    elements = payload["format"]["elements"]
    assert payload["format"]["type"] == "sequence"
    assert elements[-1]["type"] == "json_schema"
    assert elements[-1]["json_schema"] == schema


def test_inkling_reasoning_parser_wraps_json_schema_in_content_text_block():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }

    structural_tag = structural_tag_for_reasoning_json_schema("inkling", schema)

    assert structural_tag is not None
    payload = json.loads(structural_tag)
    elements = payload["format"]["elements"]
    assert payload["type"] == "structural_tag"
    assert payload["format"]["type"] == "sequence"
    assert elements[0]["type"] == "star"
    assert elements[0]["content"]["type"] == "tag"
    assert elements[0]["content"]["begin"] == "<|content_thinking|>"
    assert elements[0]["content"]["content"]["type"] == "any_text"
    assert elements[0]["content"]["end"] == "<|end_message|>"
    assert elements[1]["type"] == "tag"
    assert elements[1]["begin"] == "<|content_text|>"
    assert elements[1]["content"]["type"] == "json_schema"
    assert elements[1]["content"]["json_schema"] == schema
    assert elements[1]["end"] == "<|end_message|>"


_OPEN, _CLOSE = 7, 8


@pytest.mark.parametrize(
    "prompt_ids, expected",
    [
        ([1, 2, _OPEN], True),
        ([1, _OPEN, 3, 4], True),
        ([1, _OPEN, 3, _CLOSE, 5], False),
        ([1, _OPEN, _CLOSE, 4, _OPEN, 6], True),
        ([1, 2, 3], False),
        ([], False),
    ],
)
def test_prompt_ends_in_reasoning(prompt_ids, expected):
    assert prompt_ends_in_reasoning(prompt_ids, _OPEN, _CLOSE) is expected


@pytest.mark.parametrize(
    "parser",
    [
        "qwen3",
        "qwen3_thinking",
        "deepseek_r1",
        "deepseek_v31",
        "deepseek_v41",
        "glm45",
        "kimi",
        "kimi_k25",
        "minimax",
    ],
)
def test_reasoning_block_markers(parser):
    assert reasoning_block_markers(parser) == ("<think>", "</think>")


def test_wrap_needs_the_prompt():
    # The gRPC servicer calls the wrap before the prompt is known. It must
    # keep json_schema, which InputProcessor then wraps with the prompt.
    assert structural_tag_for_reasoning_json_schema("qwen3", _SCHEMA) is None
    assert structural_tag_for_reasoning_json_schema("inkling", _SCHEMA) is not None


def _accepts(structural_tag: str, reply: str) -> bool:
    compiler = xgrammar.GrammarCompiler(xgrammar.TokenizerInfo([]), cache_enabled=False)
    matcher = xgrammar.GrammarMatcher(
        compiler.compile_structural_tag(structural_tag),
        terminate_without_stop_token=True,
    )
    return matcher.accept_string(reply) and matcher.is_terminated()


def _wrap_json_schema(tokenizer, reasoning_parser: str, prompt_ids: list[int]) -> str:
    engine = SimpleNamespace(
        context_len=len(prompt_ids) + 64,
        max_req_input_len=len(prompt_ids) + 63,
        is_generation=True,
        tokenizer=tokenizer,
        logger=SimpleNamespace(warning=lambda *a, **k: None),
        server_args=SimpleNamespace(
            reasoning_parser=reasoning_parser, enable_prefix_caching=False
        ),
        model_config=SimpleNamespace(vocab_size=len(tokenizer)),
    )
    obj = GenerateReqInput(
        input_ids=prompt_ids, sampling_params={"json_schema": json.dumps(_SCHEMA)}
    )
    obj.normalize_batch_and_arguments()
    out = asyncio.run(InputProcessor(engine).tokenize_one_request(obj))
    assert out.sampling_params.json_schema is None
    return out.sampling_params.structural_tag


# (model, reasoning parser, chat template options, accepted replies, refused
# replies). The Qwen3 tag puts "\n\n" after </think>; the DeepSeek tags do not.
_CASES = [
    # Hybrid Qwen3, thinking on: the prompt ends with "<|im_start|>assistant\n"
    # and the model opens <think> itself.
    (
        "Qwen/Qwen3-30B-A3B",
        "qwen3",
        {"enable_thinking": True},
        ["<think>\nParis.\n</think>\n\n" + _ANSWER, _ANSWER],
        ["Paris.\n</think>\n\n" + _ANSWER, _WRONG_ANSWER],
    ),
    # Hybrid Qwen3, thinking off: the prompt ends with "<think>\n\n</think>\n\n".
    (
        "Qwen/Qwen3-30B-A3B",
        "qwen3",
        {"enable_thinking": False},
        [_ANSWER],
        [_WRONG_ANSWER],
    ),
    # Thinking-only Qwen3: the prompt ends with "<think>\n".
    (
        "Qwen/Qwen3-4B-Thinking-2507",
        "qwen3_thinking",
        {},
        ["Paris.\n</think>\n\n" + _ANSWER],
        [_ANSWER],
    ),
    # DeepSeek-R1-0528: the prompt ends with "<｜Assistant｜>".
    (
        "deepseek-ai/DeepSeek-R1-0528",
        "deepseek_r1",
        {},
        ["<think>\nParis.\n</think>" + _ANSWER, _ANSWER],
        ["Paris.\n</think>" + _ANSWER, _WRONG_ANSWER],
    ),
    # DeepSeek-V3.1, thinking on: the prompt ends with "<think>".
    (
        "deepseek-ai/DeepSeek-V3.1",
        "deepseek_v31",
        {"thinking": True},
        ["Paris.\n</think>" + _ANSWER],
        [_ANSWER],
    ),
    # DeepSeek-V3.1, thinking off: the prompt ends with "</think>".
    (
        "deepseek-ai/DeepSeek-V3.1",
        "deepseek_v31",
        {"thinking": False},
        [_ANSWER],
        [_WRONG_ANSWER],
    ),
]


@pytest.mark.parametrize("model, parser, options, accepted, refused", _CASES)
def test_json_schema_wrap_follows_the_prompt(model, parser, options, accepted, refused):
    tokenizer = AutoTokenizer.from_pretrained(model)
    prompt = tokenizer.apply_chat_template(
        _MESSAGES, tokenize=False, add_generation_prompt=True, **options
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    structural_tag = _wrap_json_schema(tokenizer, parser, prompt_ids)

    for reply in accepted:
        assert _accepts(structural_tag, reply), reply
    for reply in refused:
        assert not _accepts(structural_tag, reply), reply


def test_json_schema_wrap_assumes_open_reasoning_without_marker_tokens():
    # "<think>" is several GPT-2 tokens, so the prompt ids cannot show where
    # the block opens; the wrap assumes the prompt opened it, as before.
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    prompt_ids = tokenizer.encode("Q: capital of France?\nA: <think>\n")

    structural_tag = _wrap_json_schema(tokenizer, "qwen3", prompt_ids)

    assert _accepts(structural_tag, "Paris.\n</think>\n\n" + _ANSWER)
    assert not _accepts(structural_tag, _ANSWER)

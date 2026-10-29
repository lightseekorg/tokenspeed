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

"""The structural tags the engine defers a request's JSON schema with: the
templates ``reasoning_grammar`` gives the host, with the schema in the slot
the host fills and the template it picks for the prompt."""

from __future__ import annotations

import json

import pytest
from transformers import AutoTokenizer

from tokenspeed.runtime.grammar.reasoning_structural_tag import (
    ReasoningGrammar,
    reasoning_block_markers,
    reasoning_grammar,
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


class _NoTokenizer:
    """A tokenizer with no marker tokens: ``reasoning_grammar``'s markers are
    then None."""

    def convert_tokens_to_ids(self, token):
        return None


def _host_tag(grammar: ReasoningGrammar, prompt_ids: list[int]) -> str:
    """The tag the host admits a request with ``_SCHEMA`` under (the
    ingress's ``ReasoningGrammar::defer_schema``): the continuing one when
    the last block marker in the prompt is the opener, or without markers."""
    opened = True
    if grammar.markers is not None:
        opened = False
        for token_id in reversed(prompt_ids):
            if token_id in grammar.markers:
                opened = token_id == grammar.markers[0]
                break
    tag = grammar.continues if opened else grammar.may_open
    return tag.replace(grammar.schema_slot, json.dumps(_SCHEMA), 1)


@pytest.mark.parametrize("parser", ["deepseek_v31", "deepseek_v41"])
def test_deepseek_reasoning_parser_wraps_json_schema_after_thinking(parser):
    grammar = reasoning_grammar(parser, _NoTokenizer())

    payload = json.loads(_host_tag(grammar, []))
    elements = payload["format"]["elements"]
    assert payload["type"] == "structural_tag"
    assert payload["format"]["type"] == "sequence"
    assert elements[0]["type"] == "tag"
    assert elements[0]["begin"] in ("", "<think>")
    assert elements[0]["end"] == "</think>"
    assert elements[-1]["type"] == "json_schema"
    assert elements[-1]["json_schema"] == _SCHEMA


def test_qwen3_reasoning_parser_uses_xgrammar_0_2_builtin_name():
    grammar = reasoning_grammar("qwen3", _NoTokenizer())

    payload = json.loads(_host_tag(grammar, []))
    elements = payload["format"]["elements"]
    assert payload["format"]["type"] == "sequence"
    assert elements[-1]["type"] == "json_schema"
    assert elements[-1]["json_schema"] == _SCHEMA


def test_inkling_reasoning_parser_wraps_json_schema_in_content_text_block():
    grammar = reasoning_grammar("inkling", _NoTokenizer())

    # One layout whatever the prompt: the prompt never opens a block.
    assert grammar.continues == grammar.may_open
    payload = json.loads(_host_tag(grammar, []))
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
    assert elements[1]["content"]["json_schema"] == _SCHEMA
    assert elements[1]["end"] == "<|end_message|>"


def test_a_parser_without_a_template_defers_nothing():
    assert reasoning_grammar("passthrough", _NoTokenizer()) is None


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


def _accepts(structural_tag: str, reply: str) -> bool:
    compiler = xgrammar.GrammarCompiler(xgrammar.TokenizerInfo([]), cache_enabled=False)
    matcher = xgrammar.GrammarMatcher(
        compiler.compile_structural_tag(structural_tag),
        terminate_without_stop_token=True,
    )
    return matcher.accept_string(reply) and matcher.is_terminated()


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

    grammar = reasoning_grammar(parser, tokenizer)
    assert grammar.markers == tuple(
        tokenizer.convert_tokens_to_ids(["<think>", "</think>"])
    )
    structural_tag = _host_tag(grammar, prompt_ids)

    for reply in accepted:
        assert _accepts(structural_tag, reply), reply
    for reply in refused:
        assert not _accepts(structural_tag, reply), reply


def test_json_schema_wrap_assumes_open_reasoning_without_marker_tokens():
    # "<think>" is several GPT-2 tokens, so the prompt ids cannot show where
    # the block opens; the wrap assumes the prompt opened it, as before.
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    prompt_ids = tokenizer.encode("Q: capital of France?\nA: <think>\n")

    grammar = reasoning_grammar("qwen3", tokenizer)
    assert grammar.markers is None
    structural_tag = _host_tag(grammar, prompt_ids)

    assert _accepts(structural_tag, "Paris.\n</think>\n\n" + _ANSWER)
    assert not _accepts(structural_tag, _ANSWER)

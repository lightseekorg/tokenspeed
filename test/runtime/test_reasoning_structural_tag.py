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

import json
import os
import sys

import pytest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.grammar.reasoning_structural_tag import (
    hy_v4_think_tokens,
    structural_tag_for_hy_v4_response,
    structural_tag_for_reasoning_json_schema,
)

pytest.importorskip("xgrammar")


@pytest.mark.parametrize("parser", ["deepseek_v31", "deepseek_v41"])
def test_deepseek_reasoning_parser_wraps_json_schema_after_thinking(parser):
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }

    structural_tag = structural_tag_for_reasoning_json_schema(parser, schema)

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

    structural_tag = structural_tag_for_reasoning_json_schema("qwen3", schema)

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


@pytest.mark.parametrize("suffix", [":0a1b2c3d", ""])
def test_hy_v4_think_tokens_follow_the_release_suffix(suffix):
    vocab = {"<think:other>": 3, f"<think{suffix}>": 7, f"</think{suffix}>": 8}
    assert hy_v4_think_tokens(vocab) == (7, f"</think{suffix}>")
    assert hy_v4_think_tokens({"<think:x>": 1, "</think:y>": 2}) is None


def test_hy_v4_defers_json_schema_past_reasoning():
    from xgrammar.structural_tag import JSONSchemaFormat

    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    think_tokens = (7, "</think:0a1b2c3d>")
    content = JSONSchemaFormat(json_schema=schema)

    structural_tag = structural_tag_for_hy_v4_response(content, [1, 2, 7], think_tokens)

    elements = json.loads(structural_tag)["format"]["elements"]
    assert elements[0]["type"] == "tag"
    assert (elements[0]["begin"], elements[0]["end"]) == ("", "</think:0a1b2c3d>")
    assert elements[1]["type"] == "json_schema"
    assert elements[1]["json_schema"] == schema
    # ``no_think`` prompts end with the closed block: constrain from token 0.
    assert structural_tag_for_hy_v4_response(content, [1, 7, 8], think_tokens) is None
    assert structural_tag_for_hy_v4_response(content, [], think_tokens) is None

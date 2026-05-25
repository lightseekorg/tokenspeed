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

import pytest

from tokenspeed.runtime.grammar.reasoning_structural_tag import (
    structural_tag_for_reasoning_json_schema,
)

pytest.importorskip("xgrammar")


def test_deepseek_v31_reasoning_parser_wraps_json_schema_after_thinking():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }

    structural_tag = structural_tag_for_reasoning_json_schema("deepseek_v31", schema)

    assert structural_tag is not None
    payload = json.loads(structural_tag)
    elements = payload["format"]["elements"]
    assert payload["type"] == "structural_tag"
    assert payload["format"]["type"] == "sequence"
    assert elements[0]["type"] == "tag"
    assert elements[0]["begin"] == "<think>"
    assert elements[0]["end"] == "</think>"
    assert elements[-1]["type"] == "json_schema"
    assert elements[-1]["json_schema"] == schema

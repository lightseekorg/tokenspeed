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

"""The rendered-skeleton splice of ``structural_tag_for_reasoning_response``
reproduces, byte for byte, what building the pydantic tree per request and
mutating it produced — the refactor that made the response slot a marked
const-string spliced per request must not change the tag the host's grammar
compiler sees."""

from __future__ import annotations

import pytest

from tokenspeed.runtime.grammar.reasoning_structural_tag import (
    _REASONING_PARSER_TO_XGRAMMAR_MODEL,
    structural_tag_for_reasoning_response,
)

xgrammar = pytest.importorskip("xgrammar")
from xgrammar.structural_tag import (  # noqa: E402
    AnyTextFormat,
    ConstStringFormat,
    JSONSchemaFormat,
    RegexFormat,
    SequenceFormat,
    StarFormat,
    StructuralTag,
    TagFormat,
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "address": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "zip": {"type": "string"}},
            "required": ["city"],
        },
    },
    "required": ["name", "age"],
}
# Non-ASCII keys and values, so a splice that re-escaped the content's JSON
# instead of passing it through would diverge from pydantic's raw-UTF-8 dump.
_SCHEMA_UNICODE = {
    "type": "object",
    "properties": {"回复": {"type": "string", "description": "café ☕ \U0001f680"}},
    "required": ["回复"],
}


def _reference_mapped_tag(model, reasoning, content_format):
    """The pre-skeleton implementation: fetch the template, mutate the
    response slot to the content format, dump the whole tag."""
    tag = xgrammar.get_builtin_structural_tag(model=model, reasoning=reasoning)
    fmt = tag.format
    substituted = False
    if fmt.type == "tags_with_separator":
        analysis_tag = None
        final_tag = None
        for t in fmt.tags:
            if t.begin == "<|channel|>analysis<|message|>":
                analysis_tag = t
            elif t.begin == "<|channel|>final<|message|>":
                final_tag = t
        if analysis_tag is not None and final_tag is not None:
            final_tag.content = content_format
            tag.format = SequenceFormat(
                elements=[
                    analysis_tag,
                    ConstStringFormat(value=fmt.separator),
                    final_tag,
                ]
            )
            substituted = True
    elif fmt.type == "sequence":
        last_idx = len(fmt.elements) - 1
        if (
            last_idx >= 0
            and getattr(fmt.elements[last_idx], "type", None) == "any_text"
        ):
            fmt.elements[last_idx] = content_format
            substituted = True
    if not substituted:
        return None
    return tag.model_dump_json(by_alias=True)


def _reference_inkling_tag(content_format):
    reasoning_block = TagFormat(
        begin="<|content_thinking|>",
        content=AnyTextFormat(),
        end="<|end_message|>",
    )
    response_block = TagFormat(
        begin="<|content_text|>",
        content=content_format,
        end="<|end_message|>",
    )
    tag = StructuralTag(
        format=SequenceFormat(
            elements=[StarFormat(content=reasoning_block), response_block]
        )
    )
    return tag.model_dump_json(by_alias=True)


_CONTENT_FORMATS = [
    JSONSchemaFormat(json_schema=_SCHEMA),
    JSONSchemaFormat(json_schema=_SCHEMA_UNICODE),
    JSONSchemaFormat(json_schema=True),
    RegexFormat(pattern=r"\d{4}-\d{2}-\d{2}"),
]


@pytest.mark.parametrize(
    "reasoning_parser", sorted(_REASONING_PARSER_TO_XGRAMMAR_MODEL)
)
@pytest.mark.parametrize("prompt_opens_reasoning", [False, True])
@pytest.mark.parametrize(
    "content_format", _CONTENT_FORMATS, ids=range(len(_CONTENT_FORMATS))
)
def test_splice_matches_the_mutated_tree_dump(
    reasoning_parser, prompt_opens_reasoning, content_format
):
    model = _REASONING_PARSER_TO_XGRAMMAR_MODEL[reasoning_parser]
    reasoning = "enabled" if prompt_opens_reasoning else "auto"
    got = structural_tag_for_reasoning_response(
        reasoning_parser, content_format, prompt_opens_reasoning=prompt_opens_reasoning
    )
    want = _reference_mapped_tag(model, reasoning, content_format)
    assert got == want
    # The content's own serialization passes through unchanged.
    assert content_format.model_dump_json(by_alias=True) in got


@pytest.mark.parametrize(
    "content_format", _CONTENT_FORMATS, ids=range(len(_CONTENT_FORMATS))
)
def test_splice_matches_the_inkling_tree_dump(content_format):
    got = structural_tag_for_reasoning_response(
        "inkling", content_format, prompt_opens_reasoning=False
    )
    assert got == _reference_inkling_tag(content_format)
    assert content_format.model_dump_json(by_alias=True) in got


def test_unmapped_parser_leaves_the_constraint_unchanged():
    assert (
        structural_tag_for_reasoning_response(
            "no_such_parser",
            JSONSchemaFormat(json_schema=_SCHEMA),
            prompt_opens_reasoning=False,
        )
        is None
    )


def test_content_carrying_the_marker_string_still_splices_once():
    # A schema that embeds the slot marker's own text must ride through as
    # data: the splice replaces the skeleton's slot, never the content.
    content = JSONSchemaFormat(
        json_schema={
            "type": "object",
            "properties": {"m": {"const": "__tokenspeed_content_slot__"}},
        }
    )
    got = structural_tag_for_reasoning_response(
        "qwen3", content, prompt_opens_reasoning=False
    )
    want = _reference_mapped_tag("qwen_3", "auto", content)
    assert got == want
    assert got.count("__tokenspeed_content_slot__") == 1

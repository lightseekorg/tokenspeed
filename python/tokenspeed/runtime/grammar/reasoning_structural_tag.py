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

"""Compose xgrammar structural tags that defer constraint enforcement
past the model's reasoning channel.

Given a tokenspeed ``--reasoning-parser`` name and any xgrammar content
format (``JSONSchemaFormat``, ``RegexFormat``, ``GrammarFormat``, …),
``structural_tag_for_reasoning_response()`` produces an xgrammar
structural-tag JSON string that:

1. Uses xgrammar's ``get_builtin_structural_tag`` template for the
   matching model family (so the channel layout matches the model's
   chat-template expectations exactly).
2. Substitutes the response-channel content (default ``any_text``) with
   the supplied content format, so the constraint only kicks in inside
   that channel — reasoning content stays free-form.
3. Picks the template's reasoning mode: a chat template may open the
   reasoning block in the prompt (``...<think>\\n``) or leave it to the
   model, depending on the model and on options such as
   ``enable_thinking``.

This avoids the channel-preamble corruption that activating any grammar
at token 0 would cause for reasoning models (the model emits
``<|channel|>analysis<|message|>...`` etc. before the response, and a
naïve grammar would reject those tokens).

The engine applies it to a request's ``json_schema`` at admission (the
host's ingress), whatever frontend sent the request: ``reasoning_grammar``
gives the host the tags for both modes, with a slot for the schema, and the
reasoning block's marker ids, from which the host tells the mode by the
prompt.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

# smg reasoning_parser name → xgrammar builtin model template. Keys
# must match ``smg::get_available_reasoning_parsers()`` exactly. Unmapped
# parsers fall through to the unwrapped json_schema constraint.
# gpt-oss / harmony is intentionally absent: smg's Harmony Responses path
# wraps the schema in a structural tag itself, so the engine never sees
# json_schema for that family.
_REASONING_PARSER_TO_XGRAMMAR_MODEL: dict[str, str] = {
    "deepseek_r1": "deepseek_r1",
    "deepseek_v31": "deepseek_v3_2",
    "deepseek_v41": "deepseek_v3_2",
    "kimi": "kimi",
    "kimi_k25": "kimi",
    "minimax": "minimax",
    "qwen3": "qwen_3",
    "qwen3_thinking": "qwen_3",
    "glm45": "glm_4_7",
}


def _structural_tag_for_inkling(content_format: Any) -> str:
    """Build Inkling's reasoning/content structural tag.

    The prompt already ends with ``<|message_model|>``. Generation should then
    contain zero or more reasoning blocks followed by the constrained answer
    inside the text content block.
    """
    from xgrammar.structural_tag import (
        AnyTextFormat,
        SequenceFormat,
        StarFormat,
        StructuralTag,
        TagFormat,
    )

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
            elements=[
                StarFormat(content=reasoning_block),
                response_block,
            ]
        )
    )
    return tag.model_dump_json(by_alias=True)


def reasoning_block_markers(reasoning_parser: str) -> tuple[str, str] | None:
    """Return the strings that open and close the reasoning block.

    For example ``("<think>", "</think>")``. They are read from xgrammar's
    builtin template in ``"auto"`` mode, where the block is optional and
    keeps its opener. Returns ``None`` when the parser has no mapping or the
    template has another layout.
    """
    model = _REASONING_PARSER_TO_XGRAMMAR_MODEL.get(reasoning_parser)
    if model is None:
        return None

    import xgrammar

    fmt = xgrammar.get_builtin_structural_tag(model=model, reasoning="auto").format
    # Layout: [optional(reasoning block), response], where the block is a
    # tag, or a sequence that starts with the tag.
    if fmt.type != "sequence" or not fmt.elements:
        return None
    if fmt.elements[0].type != "optional":
        return None
    block = fmt.elements[0].content
    if block.type == "sequence" and block.elements:
        block = block.elements[0]
    if block.type != "tag" or not block.begin or not block.end:
        return None
    return block.begin, block.end


def structural_tag_for_reasoning_response(
    reasoning_parser: str,
    content_format: Any,
    *,
    prompt_opens_reasoning: bool,
) -> str | None:
    """Build a JSON-serialized xgrammar structural tag for ``reasoning + content``.

    ``content_format`` is any xgrammar content Format instance —
    ``JSONSchemaFormat``, ``RegexFormat``, ``GrammarFormat``, etc. It
    replaces the response channel's default ``any_text``.

    ``prompt_opens_reasoning`` says whether the prompt ends inside an open
    reasoning block. If it does, the reply continues that block (xgrammar's
    ``"enabled"`` mode). If not, the reply may open the block itself or
    answer directly (``"auto"``).

    Returns ``None`` if ``reasoning_parser`` has no xgrammar mapping —
    the caller should leave the user's constraint unchanged in that case.
    """
    if reasoning_parser == "inkling":
        return _structural_tag_for_inkling(content_format)

    model = _REASONING_PARSER_TO_XGRAMMAR_MODEL.get(reasoning_parser)
    if model is None:
        return None

    import xgrammar

    tag = xgrammar.get_builtin_structural_tag(
        model=model, reasoning="enabled" if prompt_opens_reasoning else "auto"
    )
    fmt = tag.format

    # Track whether the substitution actually landed. If it didn't
    # (template shape changed in a future xgrammar release, response
    # slot moved, etc.) we MUST NOT return the tag with the user's
    # constraint silently dropped — return None so the caller falls
    # back to the unwrapped constraint and the user gets the
    # constraint they asked for (at the cost of channel-preamble
    # corruption on reasoning models).
    substituted = False

    if fmt.type == "tags_with_separator":
        # harmony: the builtin template uses ``tags_with_separator``
        # which allows the model to repeat ``analysis sep final``
        # cycles indefinitely — and after the first JSON it does,
        # producing trailing ``<|end|><|start|>assistant…`` plus a
        # second JSON that the reasoning parser concatenates,
        # breaking ``json.loads``. ``stop_after_first=True`` is
        # *worse*: it ends after the FIRST tag (analysis only), so
        # ``content`` ends up empty.
        #
        # Instead we replace the format with a fixed sequence of
        # exactly one analysis tag, the literal separator, and one
        # final tag carrying the user's content. The matcher
        # forbids any further channel re-open after the final tag's
        # ``<|end|>``, so the model stops cleanly.
        from xgrammar.structural_tag import (
            ConstStringFormat,
            SequenceFormat,
            TagFormat,
        )

        analysis_tag: TagFormat | None = None
        final_tag: TagFormat | None = None
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
        # qwen / deepseek_r1 / glm47 / kimi / minimax: the layout is
        # ``[reasoning_tag, response_any_text]``. Anchor on the LAST
        # element (the response) so the reasoning tag is never replaced
        # — even if xgrammar later inserts pre-reasoning preamble
        # elements.
        last_idx = len(fmt.elements) - 1
        if (
            last_idx >= 0
            and getattr(fmt.elements[last_idx], "type", None) == "any_text"
        ):
            fmt.elements[last_idx] = content_format
            substituted = True

    if not substituted:
        # Either the format type isn't one we know how to handle, or
        # the response slot wasn't where we expected. Bail out so the
        # user's constraint isn't silently dropped.
        return None

    return tag.model_dump_json(by_alias=True)


# The JSON schema a template's answer carries until the host puts a
# request's schema there: an object no request sends.
_SCHEMA_SLOT = {"__tokenspeed_reasoning_schema__": True}


@dataclasses.dataclass(frozen=True)
class ReasoningGrammar:
    """How the engine lets a request's JSON schema wait for the reasoning
    block (``--reasoning-parser``): structural tags whose answer is
    ``schema_slot``, which the host replaces with the request's schema.

    Attributes:
        continues: The tag for a reply that continues a reasoning block the
            prompt opened.
        may_open: The tag for a reply that may open the block itself or
            answer directly.
        schema_slot: The JSON text in both tags that the schema replaces.
        markers: The block's opener and closer token ids, which tell
            whether a prompt ends inside the block: it does when the last of
            them in the prompt is the opener. None when they are not single
            tokens; every prompt then counts as having opened the block.
    """

    continues: str
    may_open: str
    schema_slot: str
    markers: tuple[int, int] | None


def _reasoning_marker_ids(reasoning_parser: str, tokenizer) -> tuple[int, int] | None:
    markers = reasoning_block_markers(reasoning_parser)
    if markers is None:
        return None
    ids = [tokenizer.convert_tokens_to_ids(marker) for marker in markers]
    if not all(isinstance(i, int) for i in ids):
        return None
    if list(tokenizer.convert_ids_to_tokens(ids)) != list(markers):
        return None
    return ids[0], ids[1]


def reasoning_grammar(reasoning_parser: str, tokenizer) -> ReasoningGrammar | None:
    """The engine's ``ReasoningGrammar`` for ``reasoning_parser``, or None
    when xgrammar has no template for it: a JSON schema then constrains the
    whole reply."""
    from xgrammar.structural_tag import JSONSchemaFormat

    tags = [
        structural_tag_for_reasoning_response(
            reasoning_parser,
            JSONSchemaFormat(json_schema=_SCHEMA_SLOT),
            prompt_opens_reasoning=opens,
        )
        for opens in (True, False)
    ]
    if None in tags:
        return None
    schema_slot = json.dumps(_SCHEMA_SLOT, separators=(",", ":"))
    for tag in tags:
        if tag.count(schema_slot) != 1:
            raise RuntimeError(
                f"reasoning-parser={reasoning_parser}: the structural tag "
                f"carries the schema slot {tag.count(schema_slot)} times"
            )
    return ReasoningGrammar(
        continues=tags[0],
        may_open=tags[1],
        schema_slot=schema_slot,
        markers=_reasoning_marker_ids(reasoning_parser, tokenizer),
    )

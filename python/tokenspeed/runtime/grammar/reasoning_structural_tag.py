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

This avoids the channel-preamble corruption that activating any grammar
at token 0 would cause for reasoning models (the model emits
``<|channel|>analysis<|message|>...`` etc. before the response, and a
naïve grammar would reject those tokens).

Convenience wrappers per content type are provided
(``..._json_schema``, ``..._regex``, ``..._grammar``); call sites that
already have an xgrammar Format instance can call the base function.
"""

from __future__ import annotations

from typing import Any

# tokenspeed reasoning_parser → xgrammar builtin model template.
# Parsers not listed here fall back to the unwrapped constraint, which
# is likely to corrupt the reasoning preamble. Add new mappings as the
# reasoning_parser registry grows.
_REASONING_PARSER_TO_XGRAMMAR_MODEL: dict[str, str] = {
    "gpt-oss": "harmony",
    "deepseek-r1": "deepseek_r1",
    "deepseekv31": "deepseek_r1",
    "deepseek-v3": "deepseek_r1",
    "kimi": "kimi",
    "kimi_k2": "kimi",
    "minimax": "minimax",
    "minimax-append-think": "minimax",
    "qwen3": "qwen",
    "qwen3-thinking": "qwen",
}


def structural_tag_for_reasoning_response(
    reasoning_parser: str, content_format: Any
) -> str | None:
    """Build a JSON-serialized xgrammar structural tag for ``reasoning + content``.

    ``content_format`` is any xgrammar content Format instance —
    ``JSONSchemaFormat``, ``RegexFormat``, ``GrammarFormat``, etc. It
    replaces the response channel's default ``any_text``.

    Returns ``None`` if ``reasoning_parser`` has no xgrammar mapping —
    the caller should leave the user's constraint unchanged in that case.
    """
    model = _REASONING_PARSER_TO_XGRAMMAR_MODEL.get(reasoning_parser)
    if model is None:
        return None

    import xgrammar

    tag = xgrammar.get_builtin_structural_tag(model=model, reasoning=True)
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


def structural_tag_for_reasoning_json_schema(
    reasoning_parser: str, user_schema: Any
) -> str | None:
    """Convenience wrapper: response channel constrained to a JSON schema."""
    from xgrammar.structural_tag import JSONSchemaFormat

    return structural_tag_for_reasoning_response(
        reasoning_parser, JSONSchemaFormat(json_schema=user_schema)
    )


def structural_tag_for_reasoning_regex(
    reasoning_parser: str, pattern: str
) -> str | None:
    """Convenience wrapper: response channel constrained to a regex pattern."""
    from xgrammar.structural_tag import RegexFormat

    return structural_tag_for_reasoning_response(
        reasoning_parser, RegexFormat(pattern=pattern)
    )


def structural_tag_for_reasoning_grammar(
    reasoning_parser: str, ebnf_grammar: str
) -> str | None:
    """Convenience wrapper: response channel constrained to an EBNF grammar."""
    from xgrammar.structural_tag import GrammarFormat

    return structural_tag_for_reasoning_response(
        reasoning_parser, GrammarFormat(grammar=ebnf_grammar)
    )

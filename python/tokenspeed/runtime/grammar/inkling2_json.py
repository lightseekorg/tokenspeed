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

"""Inkling2's native JSON answers: the structural tag of its reasoning and
answer blocks, where a prompt may end before an answer (``MessageBoundary``),
and how its vocabulary frames the answer in protocol tokens for the host's
grammar compiler (``JsonFraming``)."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

_INKLING_CONTROL_TOKENS = (
    "<|endoftext|>",
    "<|message_user|>",
    "<|message_model|>",
    "<|message_system|>",
    "<|message_tool|>",
    "<|content_text|>",
    "<|content_image|>",
    "<|content_model_end_sampling|>",
    "<|content_thinking|>",
    "<|content_audio_input|>",
    "<|content_tool_error|>",
    "<|content_xml|>",
    "<|content_invoke_tool_json|>",
    "<|content_invoke_tool_text|>",
    "<|content_invoke_tool|>",
    "<|model_trigger_generation|>",
    "<|end_message|>",
    "<|audio_end|>",
)


# Inkling2's native message delimiters as structural tags spell them.
_INKLING2_ANSWER_HEADER = "<|message_model|><|content_text|>"
_INKLING2_THINKING_HEADER = "<|message_model|><|content_thinking|>"
_INKLING2_END = "<|end_message|>"


def inkling2_structural_tag(
    content_format: Any, *, continue_final_message: bool
) -> str:
    """Build Inkling2's reasoning/content structural tag.

    Inkling2 writes ``<|message_model|>`` before each block itself; its
    reasoning excludes the control tokens and its JSON answer bounds
    whitespace at 16 unless the format sets a bound.
    ``continue_final_message`` continues an empty answer header the prompt
    ends with.
    """
    from xgrammar.structural_tag import (
        AnyTextFormat,
        JSONSchemaFormat,
        SequenceFormat,
        StarFormat,
        StructuralTag,
        TagFormat,
    )

    if (
        isinstance(content_format, JSONSchemaFormat)
        and content_format.style == "json"
        and "max_whitespace_cnt" not in content_format.model_fields_set
    ):
        content_format = content_format.model_copy(update={"max_whitespace_cnt": 16})

    reasoning_block = TagFormat(
        begin=_INKLING2_THINKING_HEADER,
        content=AnyTextFormat(excludes=list(_INKLING_CONTROL_TOKENS)),
        end=_INKLING2_END,
    )
    response_block = TagFormat(
        begin="" if continue_final_message else _INKLING2_ANSWER_HEADER,
        content=content_format,
        end=_INKLING2_END,
    )
    elements = [] if continue_final_message else [StarFormat(content=reasoning_block)]
    elements.append(response_block)
    tag = StructuralTag(format=SequenceFormat(elements=elements))
    return tag.model_dump_json(by_alias=True)


@dataclasses.dataclass(frozen=True)
class JsonFraming:
    """How a vocabulary frames a JSON answer in protocol tokens (Inkling2),
    for the host's grammar compiler: a structural tag ending in such an
    answer matches its block delimiters as the protocol tokens, and its
    matchers keep protocol tokens out of the answer's JSON strings.

    Attributes:
        answer_begin: The answer block's opener as structural tags spell it.
        answer_end: Every block's closer as structural tags spell it.
        delimiters: Each delimiter spelling (the empty one included) and the
            token ids it stands for.
        text_token: The token that opens an answer's payload.
        end_token: The token that closes a message.
        reserved_tokens: Protocol tokens a JSON string may not hold.
        string_steps: Per token, the JSON string state after its bytes from
            each start state (``ts_grammar::string_steps_of``).
    """

    answer_begin: str
    answer_end: str
    delimiters: tuple[tuple[str, tuple[int, ...]], ...]
    text_token: int
    end_token: int
    reserved_tokens: tuple[int, ...]
    string_steps: bytes


def _string_steps(token: bytes) -> int:
    """The JSON string state after ``token`` from each of the four start
    states ``2 * quoted + escaped``, two bits each."""
    packed = 0
    for start in range(4):
        quoted, escaped = bool(start & 2), bool(start & 1)
        for byte in token:
            if escaped:
                escaped = False
            elif quoted and byte == 0x5C:
                escaped = True
            elif byte == 0x22:
                quoted = not quoted
        packed |= (int(quoted) << 1 | int(escaped)) << (2 * start)
    return packed


# A token without quotes or backslashes: it ends an escape and keeps the rest.
_PLAIN_STEPS = _string_steps(b"a")


def json_framing(
    tokenizer: PreTrainedTokenizerBase, decoded_vocab: list[bytes]
) -> JsonFraming:
    """The framing of Inkling2's native JSON answers in ``tokenizer``'s
    vocabulary, which has their protocol tokens (``inkling2_boundary``).

    ``decoded_vocab`` holds each token's bytes as xgrammar decodes them
    (``TokenizerInfo.decoded_vocab``); it sets the vocabulary's size.
    """
    vocab = tokenizer.get_vocab()
    reserved = {vocab[token] for token in _INKLING_CONTROL_TOKENS if token in vocab}
    # Native JSON still reserves legacy audio IDs that H200k calls unused.
    for name, token in (("<|content_audio_input|>", 200020), ("<|audio_end|>", 200043)):
        if name not in vocab and vocab.get(f"<|unused_{token}|>") == token:
            reserved.add(token)
    steps = bytearray(len(decoded_vocab))
    for token, text in enumerate(decoded_vocab):
        plain = text and b'"' not in text and b"\\" not in text
        steps[token] = _PLAIN_STEPS if plain else _string_steps(text)
    delimiters = tuple(
        (text, tuple(tokenizer.encode(text, add_special_tokens=False)))
        for text in (
            "",
            _INKLING2_ANSWER_HEADER,
            _INKLING2_THINKING_HEADER,
            _INKLING2_END,
        )
    )
    return JsonFraming(
        answer_begin=_INKLING2_ANSWER_HEADER,
        answer_end=_INKLING2_END,
        delimiters=delimiters,
        text_token=vocab["<|content_text|>"],
        end_token=vocab[_INKLING2_END],
        reserved_tokens=tuple(sorted(reserved)),
        string_steps=bytes(steps),
    )


def inkling2_json_framing(
    reasoning_parser: str | None, tokenizer: PreTrainedTokenizerBase, vocab_size: int
) -> JsonFraming | None:
    """``json_framing`` over the grammar compiler's vocabulary of ``vocab_size``
    tokens for ``--reasoning-parser inkling2``; None for any other parser."""
    if reasoning_parser != "inkling2":
        return None
    from xgrammar import TokenizerInfo

    vocabulary = TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
    return json_framing(tokenizer, vocabulary.decoded_vocab)


@dataclasses.dataclass(frozen=True)
class MessageBoundary:
    """Where a prompt may end before a JSON answer when the model writes
    every block's header itself; any other ending is the request's error.

    Attributes:
        open_answer: An empty answer header, which the reply continues with
            the payload.
        closed: The endings of a closed message, after which the reply
            writes its own blocks.
    """

    open_answer: tuple[int, ...]
    closed: tuple[tuple[int, ...], ...]


def inkling2_boundary(tokenizer: PreTrainedTokenizerBase) -> MessageBoundary:
    """Where a prompt may end before an Inkling2 JSON answer in
    ``tokenizer``'s vocabulary; raises ValueError when it lacks the protocol
    tokens."""
    names = (
        "<|message_model|>",
        "<|content_text|>",
        _INKLING2_END,
        "<|content_model_end_sampling|>",
    )
    ids = tokenizer.convert_tokens_to_ids(list(names))
    if not all(isinstance(token, int) for token in ids) or list(
        tokenizer.convert_ids_to_tokens(ids)
    ) != list(names):
        raise ValueError(
            "reasoning-parser=inkling2: the tokenizer lacks the JSON protocol "
            "boundary tokens"
        )
    message, text, end, stop = ids
    return MessageBoundary(open_answer=(message, text), closed=((end,), (end, stop)))

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

"""What the engine host gets for Inkling2's native JSON answers: the
templates and prompt boundary of ``reasoning_grammar`` and the vocabulary's
framing (``json_framing``), which the host's grammar compiler and matchers
enforce (``ts-control`` and ``ts-grammar`` test that side)."""

from __future__ import annotations

import json

import pytest
from tokenizers import Tokenizer, decoders, models
from transformers import PreTrainedTokenizerFast

from tokenspeed.runtime.grammar.inkling2_json import (
    MessageBoundary,
    inkling2_json_framing,
    json_framing,
)
from tokenspeed.runtime.grammar.reasoning_structural_tag import reasoning_grammar

xgrammar = pytest.importorskip("xgrammar")

_CONTROL = (
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
    "<|end_message|>",
    "<|audio_end|>",
)
_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def _tokenizer(special: dict[str, int] | None = None, size: int = 128):
    """A byte-level vocabulary of ``size`` tokens with ``_CONTROL`` (or
    ``special`` at fixed ids) as special tokens."""
    vocab = {chr(i): i for i in range(128)}
    vocab.update({f"v{i}": i for i in range(128, size)})
    for name, token in (special or {}).items():
        del vocab[f"v{token}"]
        vocab[name] = token
    raw = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    raw.decoder = decoders.Fuse()
    return PreTrainedTokenizerFast(
        tokenizer_object=raw,
        additional_special_tokens=list(special) if special else list(_CONTROL),
        eos_token="<|content_model_end_sampling|>",
    )


@pytest.fixture(scope="module")
def tokenizer():
    return _tokenizer()


def _decoded(tokenizer) -> list[bytes]:
    return xgrammar.TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=len(tokenizer)
    ).decoded_vocab


def _ids(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def test_inkling2_answers_follow_the_prompt_s_message_boundary(tokenizer):
    grammar = reasoning_grammar("inkling2", tokenizer)

    message, text, end, stop = (
        tokenizer.convert_tokens_to_ids(name)
        for name in (
            "<|message_model|>",
            "<|content_text|>",
            "<|end_message|>",
            "<|content_model_end_sampling|>",
        )
    )
    assert grammar.markers is None
    assert grammar.boundary == MessageBoundary(
        open_answer=(message, text), closed=((end,), (end, stop))
    )
    # After an empty answer header the reply is the payload and its end.
    continues = json.loads(grammar.continues)["format"]["elements"]
    assert [element["type"] for element in continues] == ["tag"]
    assert continues[0]["begin"] == ""
    assert continues[0]["end"] == "<|end_message|>"
    # After a closed message the reply writes its own blocks' headers.
    may_open = json.loads(grammar.may_open)["format"]["elements"]
    assert may_open[0]["type"] == "star"
    thinking = may_open[0]["content"]
    assert thinking["begin"] == "<|message_model|><|content_thinking|>"
    assert set(thinking["content"]["excludes"]) >= set(_CONTROL)
    assert may_open[1]["begin"] == "<|message_model|><|content_text|>"
    for tag in (grammar.continues, grammar.may_open):
        assert tag.count(grammar.schema_slot) == 1
        answer = json.loads(tag)["format"]["elements"][-1]["content"]
        # Native JSON bounds its whitespace unless the schema says otherwise.
        assert answer["max_whitespace_cnt"] == 16


def test_inkling2_needs_its_boundary_tokens():
    with pytest.raises(ValueError, match="boundary tokens"):
        reasoning_grammar(
            "inkling2", _tokenizer(special={"<|end_message|>": 130}, size=140)
        )


def test_framing_reserves_the_protocol_tokens(tokenizer):
    framing = json_framing(tokenizer, _decoded(tokenizer))

    vocab = tokenizer.get_vocab()
    assert framing.text_token == vocab["<|content_text|>"]
    assert framing.end_token == vocab["<|end_message|>"]
    assert framing.reserved_tokens == tuple(sorted(vocab[name] for name in _CONTROL))
    delimiters = dict(framing.delimiters)
    assert delimiters[""] == ()
    assert delimiters[framing.answer_begin] == (
        vocab["<|message_model|>"],
        vocab["<|content_text|>"],
    )
    assert delimiters["<|message_model|><|content_thinking|>"] == (
        vocab["<|message_model|>"],
        vocab["<|content_thinking|>"],
    )
    assert delimiters[framing.answer_end] == (vocab["<|end_message|>"],)
    assert len(framing.string_steps) == len(tokenizer)


@pytest.mark.parametrize("parser", ["inkling2", "inkling", "qwen3", None])
def test_only_the_inkling2_parser_frames_json_answers(tokenizer, parser):
    framing = inkling2_json_framing(parser, tokenizer, len(tokenizer))

    if parser == "inkling2":
        assert framing == json_framing(tokenizer, _decoded(tokenizer))
    else:
        assert framing is None


def test_string_steps_follow_quotes_and_escapes(tokenizer):
    steps = json_framing(tokenizer, _decoded(tokenizer)).string_steps

    def after(token: str, quoted: bool, escaped: bool) -> tuple[bool, bool]:
        state = steps[ord(token)] >> (2 * (2 * quoted + escaped)) & 3
        return bool(state & 2), bool(state & 1)

    assert after('"', False, False) == (True, False)
    assert after('"', True, False) == (False, False)
    assert after('"', True, True) == (True, False)
    assert after("\\", True, False) == (True, True)
    assert after("\\", False, False) == (False, False)
    assert after("a", True, True) == (True, False)
    assert after("a", True, False) == (True, False)


@pytest.mark.parametrize("unused", [True, False])
def test_unused_legacy_audio_slots_stay_reserved(unused):
    # A native-sized vocabulary whose legacy audio ids are unused slots.
    special = {
        "<|message_model|>": 200001,
        "<|content_text|>": 200004,
        "<|content_model_end_sampling|>": 200006,
        "<|end_message|>": 200010,
        ("<|unused_200020|>" if unused else "<|content_audio_input|>"): 200020,
        "<|begin_of_text|>": 200028,
        ("<|unused_200043|>" if unused else "<|audio_end|>"): 200043,
    }
    tokenizer = _tokenizer(special=special, size=200058)

    framing = json_framing(tokenizer, [b"x"] * 200058)

    assert 200020 in framing.reserved_tokens
    assert 200043 in framing.reserved_tokens
    # Other special tokens remain JSON text.
    assert 200028 not in framing.reserved_tokens

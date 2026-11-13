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

"""CPU grammar checks for legacy and complete-message response framing."""

import pytest
import xgrammar
from xgrammar.structural_tag import JSONSchemaFormat

from tokenspeed.runtime.grammar.reasoning_structural_tag import (
    structural_tag_for_reasoning_response,
)


@pytest.fixture(params=[("inkling", ""), ("inkling2", "<|message_model|>")])
def grammar(request):
    parser, prefix = request.param
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    tag = structural_tag_for_reasoning_response(
        parser, JSONSchemaFormat(json_schema=schema), prompt_opens_reasoning=False
    )
    tokenizer = xgrammar.TokenizerInfo(
        [bytes([i]) for i in range(128)] + [b"<eos>"], stop_token_ids=[128]
    )
    compiler = xgrammar.GrammarCompiler(tokenizer, max_threads=1)
    return compiler.compile_structural_tag(tag), prefix


@pytest.mark.parametrize("reasoning_blocks", [0, 1, 2])
def test_json_grammar_accepts_complete_answer(grammar, reasoning_blocks):
    compiled, prefix = grammar
    reply = (
        f"{prefix}<|content_thinking|>Compute four.<|end_message|>" * reasoning_blocks
        + f'{prefix}<|content_text|>{{"answer":4}}<|end_message|>'
    )
    matcher = xgrammar.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert matcher.accept_string(reply)
    assert matcher.is_terminated()


@pytest.mark.parametrize(
    "answer",
    ['{"answer":"four"}', '{"answer":4,"extra":0}', "{}"],
)
def test_json_grammar_rejects_invalid_payload(grammar, answer):
    compiled, prefix = grammar
    matcher = xgrammar.GrammarMatcher(compiled)
    assert not matcher.accept_string(f"{prefix}<|content_text|>{answer}<|end_message|>")


def test_json_grammar_rejects_wrong_header(grammar):
    compiled, prefix = grammar
    wrong_prefix = "" if prefix else "<|message_model|>"
    matcher = xgrammar.GrammarMatcher(compiled)
    assert not matcher.accept_string(
        f'{wrong_prefix}<|content_text|>{{"answer":4}}<|end_message|>'
    )

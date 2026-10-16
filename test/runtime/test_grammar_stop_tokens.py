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

"""The xgrammar backend ends a grammar on the ids the engine stops on.

A model's generation config can list several end-of-sequence ids. The engine
stops on each of them, so the matcher takes each as the end of a grammar that
may end there, and refuses it anywhere else. Without them, the matcher ends a
grammar only on the tokenizer's own EOS and reads any other id as text.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from transformers import AutoTokenizer

pytest.importorskip("xgrammar")

from tokenspeed.runtime.grammar.base_grammar_backend import (  # noqa: E402
    create_grammar_backend,
)
from tokenspeed.runtime.grammar.xgrammar_backend import (  # noqa: E402
    XGrammarGrammarBackend,
)

# "hi", then an optional "!": the grammar may end after "hi" and after "!".
_FORMAT = {
    "type": "sequence",
    "elements": [
        {"type": "const_string", "value": "hi"},
        {"type": "optional", "content": {"type": "const_string", "value": "!"}},
    ],
}


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("gpt2")


def _grammar(backend):
    key = json.dumps({"type": "structural_tag", "format": _FORMAT})
    return backend.init_value_impl(("structural_tag", key), False)


def _ids(tokenizer):
    hi = tokenizer.encode("hi")
    (bang,) = tokenizer.encode("!")
    return hi, bang, tokenizer.eos_token_id


def test_each_stop_id_ends_the_grammar_where_it_may_end(tokenizer):
    hi, bang, eos = _ids(tokenizer)
    # "!" stands for a second end-of-sequence id of a generation config.
    backend = XGrammarGrammarBackend(
        tokenizer, vocab_size=len(tokenizer), stop_token_ids={eos, bang}
    )
    grammar = _grammar(backend)
    assert not grammar.try_accept_token(bang), "the grammar cannot end before hi"
    for token in hi:
        grammar.accept_token(token)
    assert grammar.try_accept_token(bang)
    assert grammar.is_terminated()

    grammar = _grammar(backend)
    for token in hi:
        grammar.accept_token(token)
    assert grammar.try_accept_token(eos)
    assert grammar.is_terminated()


def test_the_tokenizers_own_stop_ids_still_end_the_grammar():
    # A config whose ids leave out the tokenizer's EOS and its extra stop ids:
    # the engine stops on those too, so the grammar still ends on them.
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    hi, bang, eos = _ids(tokenizer)
    (question,) = tokenizer.encode("?")
    tokenizer.additional_stop_token_ids = {question}
    backend = XGrammarGrammarBackend(
        tokenizer, vocab_size=len(tokenizer), stop_token_ids={bang}
    )
    for stop in (bang, eos, question):
        grammar = _grammar(backend)
        for token in hi:
            grammar.accept_token(token)
        assert grammar.try_accept_token(stop), stop
        assert grammar.is_terminated(), stop


def test_without_stop_ids_only_the_tokenizers_eos_ends_the_grammar(tokenizer):
    hi, bang, eos = _ids(tokenizer)
    backend = XGrammarGrammarBackend(tokenizer, vocab_size=len(tokenizer))
    grammar = _grammar(backend)
    for token in hi:
        grammar.accept_token(token)
    assert grammar.try_accept_token(bang), "read as the text of the optional '!'"
    assert not grammar.is_terminated()
    assert grammar.try_accept_token(eos)
    assert grammar.is_terminated()


def test_the_factory_passes_the_stop_ids_on(tokenizer):
    hi, bang, eos = _ids(tokenizer)
    args = SimpleNamespace(grammar_backend="xgrammar", disable_any_whitespace=False)
    backend = create_grammar_backend(
        args, tokenizer, len(tokenizer), stop_token_ids={eos, bang}
    )
    grammar = _grammar(backend)
    for token in hi:
        grammar.accept_token(token)
    assert grammar.try_accept_token(bang)
    assert grammar.is_terminated()
    # No ids, or an empty set (a config that lists none): the tokenizer's EOS.
    for stop_token_ids in (None, set()):
        backend = create_grammar_backend(
            args, tokenizer, len(tokenizer), stop_token_ids=stop_token_ids
        )
        grammar = _grammar(backend)
        for token in hi:
            grammar.accept_token(token)
        assert grammar.try_accept_token(bang)
        assert not grammar.is_terminated()

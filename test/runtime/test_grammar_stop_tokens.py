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

"""The vocabulary the grammar compiler works over (``grammar_tokenizer_info``):
a grammar may end on every id the engine stops on."""

import os
import sys

import pytest

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

xgrammar = pytest.importorskip("xgrammar")
transformers = pytest.importorskip("transformers")

from tokenspeed.runtime.engine.rust_host import grammar_tokenizer_info  # noqa: E402


@pytest.fixture
def tokenizer():
    return transformers.AutoTokenizer.from_pretrained("gpt2")


def _stop_ids(tokenizer, eos_token_ids) -> set[int]:
    info = xgrammar.TokenizerInfo.deserialize_json(
        grammar_tokenizer_info(tokenizer, len(tokenizer), eos_token_ids)
    )
    return set(info.stop_token_ids)


def test_the_models_stop_ids_and_the_tokenizers_own_end_a_grammar(tokenizer):
    (bang,) = tokenizer.encode("!")
    (question,) = tokenizer.encode("?")
    tokenizer.additional_stop_token_ids = {question}
    # "!" stands for a second end-of-sequence id of a generation config that
    # leaves out the tokenizer's EOS: the engine stops on all three.
    assert _stop_ids(tokenizer, [bang]) == {bang, tokenizer.eos_token_id, question}


@pytest.mark.parametrize("eos_token_ids", [None, []])
def test_without_model_stop_ids_the_tokenizers_eos_ends_a_grammar(
    tokenizer, eos_token_ids
):
    assert _stop_ids(tokenizer, eos_token_ids) == {tokenizer.eos_token_id}

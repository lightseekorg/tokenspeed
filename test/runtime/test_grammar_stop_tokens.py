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

import logging
import os
import sys

import pytest

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

xgrammar = pytest.importorskip("xgrammar")
transformers = pytest.importorskip("transformers")

from tokenspeed.runtime.engine.rust_host import (  # noqa: E402
    grammar_tokenizer_info,
    resolve_stop_token_ids,
)
from tokenspeed.runtime.sampling.sampling_params import SamplingParams  # noqa: E402


@pytest.fixture
def tokenizer():
    from tokenizers import Tokenizer, models

    vocab = {"<unk>": 0, "<eos>": 1, "!": 2, "?": 3, "a": 4}
    vocab.update({f"token{i}": i for i in range(5, 40)})
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel(vocab, unk_token="<unk>")),
        unk_token="<unk>",
        eos_token="<eos>",
    )


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


def test_reduced_output_vocabulary_compiles_and_bounds_grammar_masks(tokenizer):
    # Cross a 32-bit mask-word boundary and leave real tokenizer ids in the
    # input-only tail. Exercise the host adapter, serialization and real matcher.
    output_size = 33
    info = xgrammar.TokenizerInfo.deserialize_json(
        grammar_tokenizer_info(tokenizer, output_size, {tokenizer.eos_token_id})
    )
    assert info.vocab_size == output_size
    matcher = xgrammar.GrammarMatcher(
        xgrammar.GrammarCompiler(info).compile_regex("a+")
    )
    mask = xgrammar.allocate_token_bitmask(1, output_size)
    assert mask.shape == (1, 2)
    matcher.fill_next_token_bitmask(mask)
    allowed = {
        token_id
        for token_id in range(mask.shape[1] * 32)
        if (int(mask[0, token_id // 32]) >> (token_id % 32)) & 1
    }
    (a,) = tokenizer.encode("a")
    assert allowed == {a}
    assert not matcher.accept_token(35)
    assert matcher.accept_token(a)
    matcher.fill_next_token_bitmask(mask)
    assert int(mask[0, tokenizer.eos_token_id // 32]) & (
        1 << (tokenizer.eos_token_id % 32)
    )
    assert matcher.accept_token(tokenizer.eos_token_id)
    assert matcher.is_terminated()


@pytest.mark.parametrize("model_eos", [{1, 33}, {33}])
def test_unreachable_stop_ids_warn_without_dropping_reachable_eos(
    tokenizer, caplog, model_eos
):
    tokenizer.additional_stop_token_ids = {34}
    with caplog.at_level(logging.WARNING):
        ids = resolve_stop_token_ids(
            SamplingParams(stop_token_ids={-1}),
            model_eos,
            tokenizer,
            output_vocab_size=33,
        )
    assert ids == {-1, 1, 33, 34}
    assert "EOS/stop token ids [-1, 33, 34]" in caplog.text
    assert "[0, 33)" in caplog.text


def test_all_stop_ids_outside_output_vocabulary_warn_without_rejecting(
    tokenizer, caplog
):
    tokenizer.eos_token = "token39"
    with caplog.at_level(logging.WARNING):
        ids = resolve_stop_token_ids(
            SamplingParams(), {33}, tokenizer, output_vocab_size=33
        )
    assert ids == {33, 39}
    assert "cannot be generated" in caplog.text
    assert "token limit" in caplog.text


@pytest.mark.parametrize("model_eos", [None, {0, 32}])
def test_reachable_stop_ids_do_not_warn(tokenizer, caplog, model_eos):
    with caplog.at_level(logging.WARNING):
        ids = resolve_stop_token_ids(
            SamplingParams(), model_eos, tokenizer, output_vocab_size=33
        )
    assert ids == {1} | (model_eos or set())
    assert not caplog.records


def test_ignore_eos_disables_even_unreachable_stop_ids(tokenizer, caplog):
    with caplog.at_level(logging.WARNING):
        ids = resolve_stop_token_ids(
            SamplingParams(ignore_eos=True, stop_token_ids={35}),
            {33},
            tokenizer,
            output_vocab_size=33,
        )
    assert not ids
    assert not caplog.records

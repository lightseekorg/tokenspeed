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

"""Bit-pattern regression for the configured full-vocabulary logprob form."""

import pytest
import torch
from tokenspeed_kernel.ops.sampling import FullVocabTopK


@pytest.mark.parametrize(
    "form,zero_bits", [("direct", 0), ("negative-cross-entropy", -2147483648)]
)
def test_full_vocab_preserves_zero_sign(form, zero_bits):
    logits = torch.full((2, 8), -200.0, dtype=torch.float32)
    logits[0, 0] = 0
    logits[1, 3] = 0
    scores = FullVocabTopK(k=2, vocab_block=8, form=form, solution="torch").prepare(
        logits
    )
    targets = torch.tensor([0, 3])
    packed, sampled = scores.pack(targets, prediction_rows=None)
    assert torch.equal(
        sampled.view(torch.int32), torch.full((2,), zero_bits, dtype=torch.int32)
    )
    assert torch.equal(
        scores.logprobs[:, 0].contiguous().view(torch.int32), sampled.view(torch.int32)
    )
    # The packed head is a bit reinterpretation; it must preserve signed zero too.
    assert torch.equal(packed[:, 4], sampled.view(torch.int32))


@pytest.mark.parametrize("form", ["direct", "negative-cross-entropy"])
def test_tail_matches_raw_logit_reference_with_common_offset(form):
    vocab, k = 32768, 128
    scorer = FullVocabTopK(k=k, vocab_block=32768, form=form, solution="torch")
    logits = torch.full((1, vocab), 1e8, dtype=torch.float32)
    scores = scorer.prepare(logits)
    maximum = logits.max(dim=-1).values
    expected = torch.logsumexp(logits.scatter(1, scores.ids, -torch.inf), dim=-1)
    expected = expected - (maximum + scores.log_z)
    assert torch.equal(scores.log_tail.view(torch.int32), expected.view(torch.int32))
    packed, _ = scores.pack(torch.tensor([0]), prediction_rows=None)
    assert torch.equal(packed[:, 1], expected.view(torch.int32))
    assert torch.all(logits == 1e8)


def test_full_vocab_packs_tree_prediction_rows_in_accepted_order():
    logits = torch.arange(48, dtype=torch.float32).reshape(6, 8) % 11
    scores = FullVocabTopK(k=2, vocab_block=8, form="direct", solution="torch").prepare(
        logits
    )
    # Two three-node trees with distinct accepted paths and identity padding.
    prediction_rows = torch.tensor([0, 2, 2, 3, 5, 5])
    tokens = torch.tensor([3, 4, -1, 6, 7, -1])
    packed, sampled = scores.pack(tokens, prediction_rows=prediction_rows)
    reference = torch.log_softmax(logits.double(), dim=-1)[prediction_rows]
    expected = reference.gather(1, tokens.clamp(0, 7)[:, None]).squeeze(1)
    torch.testing.assert_close(sampled.double(), expected, atol=1e-6, rtol=0)
    assert torch.equal(packed[:, 0], tokens.to(torch.int32))
    assert torch.equal(packed[:, 2:4], scores.ids[prediction_rows].to(torch.int32))
    assert torch.equal(packed[:, 1], scores.log_tail[prediction_rows].view(torch.int32))


@pytest.mark.parametrize("form", ["direct", "negative-cross-entropy"])
def test_shared_scores_and_tree_gathers_preserve_selected_bits(form):
    from tokenspeed_kernel.ops.sampling import blockwise_logprobs

    logits = torch.arange(96, dtype=torch.float32).reshape(6, 16) % 13
    scorer = FullVocabTopK(k=4, vocab_block=8, form=form, solution="torch")
    scores = scorer.prepare(logits)
    tokens = torch.tensor([0, 1, 2, 3, 4, 5])
    expected = blockwise_logprobs(
        logits, tokens, vocab_block=8, form=form, solution="torch"
    )
    actual = scores.sampled_logprobs(tokens, prediction_rows=None)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    for part in (slice(0, 2), slice(2, None)):
        value = scores.rows(part).sampled_logprobs(tokens[part], prediction_rows=None)
        assert torch.equal(value.view(torch.int32), expected[part].view(torch.int32))
    path = torch.tensor([0, 2, 2, 3, 5, 5])
    expected = blockwise_logprobs(
        logits[path], tokens, vocab_block=8, form=form, solution="torch"
    )
    packed, actual = scores.pack(tokens, prediction_rows=path)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    assert torch.equal(packed[:, 0], tokens.int())


@pytest.mark.parametrize("form", ["direct", "negative-cross-entropy"])
def test_packing_without_materialized_head_preserves_explicit_scores(form):
    from tokenspeed_kernel.ops.sampling.torch import torch_full_vocab_pack

    logits = torch.arange(96, dtype=torch.float32).reshape(6, 16) % 13
    logits[0].zero_()
    scorer = FullVocabTopK(k=4, vocab_block=8, form=form, solution="torch")
    scores = scorer.prepare(logits)
    tokens = torch.tensor([0, 2, -1, 3, 4, -1])
    path = torch.tensor([0, 2, 2, 3, 5, 5])
    expected = torch_full_vocab_pack(
        scores.shifted,
        scores.log_z,
        scores.ids,
        scores.logprobs,
        scores.log_tail,
        tokens,
        prediction_rows=path,
        form=form,
    )
    actual, sampled = scores.rows(slice(None)).pack(tokens, prediction_rows=path)
    assert torch.equal(actual, expected[:, :-1])
    assert torch.equal(sampled.view(torch.int32), expected[:, -1])

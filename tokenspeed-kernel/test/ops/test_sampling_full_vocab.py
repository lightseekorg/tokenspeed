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

import math

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
def test_tail_is_invariant_to_a_large_common_logit_offset(form):
    vocab, k = 32768, 128
    scorer = FullVocabTopK(k=k, vocab_block=32768, form=form, solution="torch")
    baseline = scorer.prepare(torch.zeros((1, vocab), dtype=torch.float32))
    logits = torch.full((1, vocab), 1e8, dtype=torch.float32)
    shifted = scorer.prepare(logits)
    torch.testing.assert_close(
        shifted.log_tail,
        torch.tensor([math.log1p(-k / vocab)], dtype=torch.float32),
        atol=1e-6,
        rtol=0,
    )
    assert torch.equal(
        shifted.log_tail.view(torch.int32), baseline.log_tail.view(torch.int32)
    )
    packed, sampled = shifted.pack(torch.tensor([0]), prediction_rows=None)
    baseline_packed, baseline_sampled = baseline.pack(
        torch.tensor([0]), prediction_rows=None
    )
    assert torch.equal(packed, baseline_packed)
    assert torch.equal(sampled.view(torch.int32), baseline_sampled.view(torch.int32))
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

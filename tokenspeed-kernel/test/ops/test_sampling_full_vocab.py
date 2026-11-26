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

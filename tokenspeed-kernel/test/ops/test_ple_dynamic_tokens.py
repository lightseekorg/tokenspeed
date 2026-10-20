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

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.ple import ple_ngram_ids
from tokenspeed_kernel.ops.ple.triton import _ngram_ids_kernel
from utils import assert_no_triton_compile


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_ngram_reuses_kernel_for_variable_token_counts():
    device = "cuda"
    multipliers = torch.tensor([12345678901, 31415926535], device=device)
    vocab_sizes = torch.tensor([1013], device=device)
    offsets = torch.zeros(1, dtype=torch.int64, device=device)

    def run(length, batch_size, index_mode):
        lengths = [length] * batch_size
        if index_mode == "ragged":
            lengths[-1] += 3
        lens = torch.tensor(lengths, device=device)
        starts = lens.cumsum(0) - lens
        req = torch.repeat_interleave(torch.arange(batch_size, device=device), lens)
        col = torch.arange(sum(lengths), device=device) - starts[req]
        ids = torch.arange(sum(lengths), device=device) + 1
        initial = torch.full((batch_size, 1), 5, dtype=torch.int64, device=device)
        actual, _ = ple_ngram_ids(
            ids,
            initial,
            req,
            col,
            lens,
            starts,
            multipliers,
            vocab_sizes,
            offsets,
            ngram_size=2,
            heads_per_ngram=1,
            eos_token_id=0,
            uniform_length=0 if index_mode == "ragged" else length,
            mod_reciprocals=None,
            need_tail=False,
        )
        return actual

    run(1476, 1, "single")
    with assert_no_triton_compile(_ngram_ids_kernel):
        for index_mode in ("ragged", "uniform", "single"):
            for length, batch_size in ((1499, 3), (1620, 4), (2064, 16)):
                actual = run(
                    length, 1 if index_mode == "single" else batch_size, index_mode
                )
                assert actual.shape == (
                    (length if index_mode == "single" else length * batch_size)
                    + (3 if index_mode == "ragged" else 0),
                    1,
                )

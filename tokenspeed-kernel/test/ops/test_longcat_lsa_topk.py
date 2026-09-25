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

"""LongCat structural initial/local candidate selection."""

import pytest
import torch
from tokenspeed_kernel.ops.attention.dsa._triton.topk import (
    mark_forced_initial_local_logits,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_forced_initial_and_local_candidates_survive_topk() -> None:
    topk = 2048
    causal_lens = torch.tensor([8, 2049, 4096], device="cuda", dtype=torch.int32)
    logits = torch.randn(3, 4096, device="cuda")
    columns = torch.arange(4096, device="cuda")
    logits.masked_fill_(columns.unsqueeze(0) >= causal_lens.unsqueeze(1), float("-inf"))

    mark_forced_initial_local_logits(
        logits,
        causal_lens,
        initial_tokens=16,
        local_tokens=1024,
    )

    selected = torch.topk(logits, topk).indices.cpu()
    for row, causal_len in enumerate(causal_lens.cpu().tolist()):
        selected_ids = set(selected[row, : min(causal_len, topk)].tolist())
        forced_ids = set(range(min(16, causal_len)))
        forced_ids.update(range(max(16, causal_len - 1024), causal_len))
        assert forced_ids <= selected_ids

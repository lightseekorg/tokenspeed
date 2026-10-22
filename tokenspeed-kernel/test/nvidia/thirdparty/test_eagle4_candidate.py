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

"""EAGLE-4 candidate head kernels: bit for bit the torch ops they replace."""

import pytest
import torch
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.thirdparty.cuda.eagle4_candidate import (
    eagle4_candidate_tail,
    eagle4_shard_topk,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform().is_hopper_plus,
    reason="needs an NVIDIA SM90+ GPU (thread-block clusters)",
)


def _scores(rows: int, cols: int, levels: int, seed: int) -> torch.Tensor:
    """bf16 scores with many ties (``levels`` distinct values)."""
    gen = torch.Generator().manual_seed(seed)
    return (torch.randint(0, levels, (rows, cols), generator=gen).float() / levels).to(
        "cuda", torch.bfloat16
    )


@pytest.mark.parametrize("cols", [31040, 62080, 124160])
@pytest.mark.parametrize("levels", [64, 60000])
def test_shard_topk_matches_torch(cols: int, levels: int) -> None:
    rows = 4
    scores = _scores(rows, cols, levels, rows * cols + levels)
    ids = torch.empty((rows, 128), dtype=torch.int64, device="cuda")
    eagle4_shard_topk(scores, 128, 1000, ids)
    expected = torch.topk(scores, 128, dim=-1, sorted=False).indices
    torch.testing.assert_close(ids, expected + 1000, rtol=0, atol=0)


@pytest.mark.parametrize("nc", [32, 128, 256])
@pytest.mark.parametrize("ranks", [1, 4])
@pytest.mark.parametrize("top_k", [0, 1, 8])
@pytest.mark.parametrize("levels", [16, 60000])
def test_candidate_tail_matches_torch(
    nc: int, ranks: int, top_k: int, levels: int
) -> None:
    rows = 6
    gen = torch.Generator().manual_seed(ranks * 100 + top_k + levels)
    logits = _scores(ranks * rows, nc, levels, ranks + top_k) * 8
    tokens = torch.randint(0, 248320, (ranks * rows, nc), generator=gen).cuda()
    views = (logits.view(ranks, rows, nc), tokens.view(ranks, rows, nc))
    logits_out = torch.empty((rows, nc), dtype=torch.float32, device="cuda")
    # Unread at R == 1, where the head passes a strided view of its rank's ids.
    width = nc if ranks > 1 else 2 * nc
    tokens_out = torch.empty((rows, width), dtype=torch.int64, device="cuda")[:, :nc]
    probs = torch.empty((rows, max(top_k, 1)), dtype=torch.float32, device="cuda")
    top_tokens = torch.empty((rows, max(top_k, 1)), dtype=torch.int64, device="cuda")
    eagle4_candidate_tail(
        *views, top_k, logits_out, tokens_out, probs if top_k else None,
        top_tokens if top_k else None,
    )  # fmt: skip

    # The rank-concatenated [M, R * NC] view the all-gather along the last dim makes.
    pool_logits = views[0].permute(1, 0, 2).reshape(rows, ranks * nc)
    pool_tokens = views[1].permute(1, 0, 2).reshape(rows, ranks * nc)
    if ranks > 1:
        ref_logits, picked = torch.topk(pool_logits, nc, dim=-1, sorted=False)
        ref_tokens = torch.gather(pool_tokens, -1, picked)
        torch.testing.assert_close(tokens_out, ref_tokens, rtol=0, atol=0)
    else:
        ref_logits, ref_tokens = pool_logits, pool_tokens
    ref_logits = ref_logits.float()
    torch.testing.assert_close(logits_out, ref_logits, rtol=0, atol=0)
    if top_k:
        top = torch.topk(torch.softmax(ref_logits, -1), top_k, dim=-1)
        torch.testing.assert_close(probs, top.values, rtol=0, atol=0)
        torch.testing.assert_close(
            top_tokens, torch.gather(ref_tokens, -1, top.indices), rtol=0, atol=0
        )

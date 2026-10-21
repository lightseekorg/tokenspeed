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

"""The greedy backend's GPU chain verify matches its CPU reference."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from tokenspeed.runtime.sampling.backends.greedy import _verify_chain_greedy

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")


def _verify(candidates: torch.Tensor, target: torch.Tensor, count_bonus: bool):
    bs, n = candidates.shape
    predicts = torch.full((bs * n,), -7, dtype=torch.int32, device=candidates.device)
    # Stale entries: every path writes the whole row.
    accept_index = torch.full((bs, n), 55, dtype=torch.int32, device=candidates.device)
    accept_num = torch.zeros(bs, dtype=torch.int32, device=candidates.device)
    _verify_chain_greedy(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_num,
        candidates=candidates,
        target_predict=target,
        batch_size=bs,
        num_draft_tokens=n,
        count_bonus=count_bonus,
    )
    return predicts.view(bs, n).cpu(), accept_index.cpu(), accept_num.cpu()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("count_bonus", [False, True])
@pytest.mark.parametrize("n", [1, 4])
def test_gpu_chain_verify_matches_cpu(n, count_bonus):
    generator = torch.Generator().manual_seed(0)
    bs = 64
    target = torch.randint(0, 3, (bs, n), generator=generator, dtype=torch.int64)
    # Candidates match the previous target with probability 2/3, so every
    # accepted length from 0 to n - 1 occurs.
    candidates = torch.randint(0, 3, (bs, n), generator=generator, dtype=torch.int32)
    candidates[:, 1:] = torch.where(
        torch.rand(bs, n - 1, generator=generator) < 0.67,
        target[:, :-1].to(torch.int32),
        candidates[:, 1:],
    )

    want_predicts, want_index, want_num = _verify(candidates, target, count_bonus)
    got_predicts, got_index, got_num = _verify(
        candidates.cuda(), target.cuda(), count_bonus
    )
    # Accepted draft tokens, then the bonus one.
    accepted = want_num - int(count_bonus)

    assert torch.equal(got_num, want_num)
    assert torch.equal(got_index, want_index)
    # Only the accepted prefix and the bonus token are outputs.
    kept = torch.arange(n)[None, :] <= accepted[:, None]
    assert torch.equal(got_predicts[kept], want_predicts[kept])
    if n > 1:
        assert set(accepted.tolist()) == set(range(n))

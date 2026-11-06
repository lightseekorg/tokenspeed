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

"""Vocab-parallel top-k: the shard selection and the post-gather merge."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.numerics.torch.sampling import (
    torch_vocab_shard_topk,
    torch_vocab_topk_merge,
)
from tokenspeed_kernel.ops.sampling import vocab_shard_topk, vocab_topk_merge
from tokenspeed_kernel.ops.sampling.triton.vocab_topk import (
    _vocab_shard_topk_kernel,
    _vocab_topk_merge_kernel,
)
from utils import assert_no_triton_compile

# One TP8 shard of a 202048-token vocabulary, and rank 3's first id.
_SHARD = 25256
_VOCAB_START = 3 * _SHARD


def _logits(rows: int, seed: int, device: str, levels: int | None) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    logits = torch.randn(rows, _SHARD, generator=gen) * 4
    if levels is not None:
        # Few distinct values: the k-th largest is tied across many columns.
        logits = torch.round(logits * levels / 4) / levels
    return logits.to(device, torch.bfloat16)


@pytest.mark.parametrize("k", [16, 10])
@pytest.mark.parametrize("levels", [None, 2])
@pytest.mark.parametrize("scale", [None, 0.1767766952966369])
def test_shard_topk_keeps_the_best_values_lowest_column_first(
    levels, scale, k, device: str
) -> None:
    """The shard's chunks select in parallel and merge; ties straddling the
    k-th slot (few distinct values) keep the lowest columns."""
    rows = 14
    logits = _logits(rows, 7, device, levels)
    out = torch.full((rows, 2 * k), -1.0, device=device)

    vocab_shard_topk(logits, k, scale, _VOCAB_START, out, softcap=None, vocab_mask=None)

    expected = torch.empty_like(out)
    torch_vocab_shard_topk(
        logits, k, scale, _VOCAB_START, expected, softcap=None, vocab_mask=None
    )
    assert torch.equal(out, expected)


def test_shard_topk_handles_a_shard_narrower_than_its_block(device: str) -> None:
    # 1000 columns pad to a 1024-wide block; the padding never wins.
    logits = -torch.rand(3, 1000, device=device).to(torch.bfloat16) - 1
    out = torch.empty(3, 8, device=device)
    vocab_shard_topk(logits, 4, None, 0, out, softcap=None, vocab_mask=None)
    expected = torch.empty_like(out)
    torch_vocab_shard_topk(logits, 4, None, 0, expected, softcap=None, vocab_mask=None)
    assert torch.equal(out, expected)


@pytest.mark.parametrize("k", [16, 10])
@pytest.mark.parametrize("ranks", [1, 8])
@pytest.mark.parametrize("softcap", [None, 20.0])
def test_merge_sorts_every_rank_candidates_lowest_slot_first(
    ranks, softcap, k, device: str
) -> None:
    rows = 7
    gen = torch.Generator().manual_seed(ranks)
    # Rows of rank blocks side by side, as a column all-gather lands them;
    # bf16-rounded values tie across ranks.
    values = (torch.randn(rows, ranks, k, generator=gen) * 2).bfloat16().float()
    ids = torch.randint(0, 202048, (rows, ranks, k), generator=gen).float()
    gathered = torch.cat((values, ids), dim=-1).reshape(rows, ranks * 2 * k)
    gathered = gathered.to(device)
    pairs = gathered.view(rows, ranks, 2 * k).transpose(0, 1)
    ids_out = torch.empty(rows, k, dtype=torch.int64, device=device)
    values_out = torch.empty(rows, k, device=device)

    vocab_topk_merge(pairs, k, softcap, ids_out, values_out)

    expected_ids = torch.empty_like(ids_out)
    expected_values = torch.empty_like(values_out)
    torch_vocab_topk_merge(pairs, k, softcap, expected_ids, expected_values)
    assert torch.equal(ids_out, expected_ids)
    if softcap is None:
        assert torch.equal(values_out, expected_values)
    else:
        # Triton's softcap goes through sigmoid, torch's through tanh: a few
        # fp32 steps apart.
        torch.testing.assert_close(values_out, expected_values, rtol=1e-5, atol=1e-5)


def test_row_counts_and_shard_offsets_never_recompile(device: str) -> None:
    k = 16
    logits = _logits(24, 1, device, None)
    out = torch.empty(24, 2 * k, device=device)
    pairs = torch.randn(8, 24, 2 * k, device=device)
    ids_out = torch.empty(24, k, dtype=torch.int64, device=device)
    values_out = torch.empty(24, k, device=device)
    # Warm both integer classes of every runtime scalar.
    for rows, start in ((16, 0), (7, 3 * _SHARD)):
        vocab_shard_topk(
            logits[:rows], k, 0.5, start, out[:rows], softcap=20.0, vocab_mask=None
        )
        vocab_topk_merge(pairs[:, :rows], k, 20.0, ids_out[:rows], values_out[:rows])
    with assert_no_triton_compile(_vocab_shard_topk_kernel, _vocab_topk_merge_kernel):
        for rows, start in ((1, 7 * _SHARD), (14, _SHARD), (24, 0)):
            vocab_shard_topk(
                logits[:rows],
                k,
                0.1767766952966369,
                start,
                out[:rows],
                softcap=30.0,
                vocab_mask=None,
            )
            vocab_topk_merge(
                pairs[:, :rows], k, 30.0, ids_out[:rows], values_out[:rows]
            )

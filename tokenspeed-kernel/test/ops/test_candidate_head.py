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

"""EAGLE-4 candidate head kernels: the shard top-k, the gather-dot and the
tail after the candidate all-gather."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.sampling.triton.candidate_head import (
    _candidate_tail_kernel,
    _gather_dot_kernel,
    _shard_topk_emit_kernel,
    _shard_topk_hist_kernel,
    _shard_topk_select_kernel,
    gather_dot,
    triton_eagle4_candidate_tail,
    triton_eagle4_shard_topk,
)
from utils import assert_no_triton_compile

# The vocab id of the weight shard's first row (rank 1 of a TP4 248320 vocab).
_ID_OFFSET = 62080


def _gather_dot_problem(rows: int, seed: int, device: str):
    """An EAGLE-4 candidate head shape: 128 candidates of a 5120-wide head."""
    gen = torch.Generator().manual_seed(seed)
    weight = torch.randn(4096, 5120, generator=gen).to(device, torch.bfloat16)
    hidden = torch.randn(rows, 5120, generator=gen).to(device, torch.bfloat16)
    ids = torch.randint(0, 4096, (rows, 128), generator=gen).to(device) + _ID_OFFSET
    return weight, ids, hidden


@pytest.mark.parametrize("rows", [1, 32, 64])
def test_gather_dot_matches_reference(rows: int, device: str) -> None:
    weight, ids, hidden = _gather_dot_problem(rows, rows, device)
    out = gather_dot(weight, ids, hidden, _ID_OFFSET)
    ref = torch.einsum("bkh,bh->bk", weight[ids - _ID_OFFSET].float(), hidden.float())
    # Computed in fp32: one bf16 rounding of the fp32 dot.
    torch.testing.assert_close(out.float(), ref, rtol=2**-8, atol=1e-2)


def test_gather_dot_small_row_tiles_agree_and_compile_once(device: str) -> None:
    """Every tile config up to 32 rows reduces along H alike: the same rows
    score bit for bit whether launched with 2 (BLOCK_K 2) or 8 (BLOCK_K 1).
    The row count picks one of three static tile configs, never a compile."""
    weight, ids, hidden = _gather_dot_problem(64, 5, device)
    wide = gather_dot(weight, ids[:8], hidden[:8], _ID_OFFSET)
    narrow = gather_dot(weight, ids[:2], hidden[:2], _ID_OFFSET)
    torch.testing.assert_close(narrow, wide[:2], rtol=0, atol=0)
    gather_dot(weight, ids, hidden, _ID_OFFSET)  # The many-row config.
    with assert_no_triton_compile(_gather_dot_kernel):
        for rows in (1, 3, 17, 32, 33, 64):
            gather_dot(weight, ids[:rows], hidden[:rows], _ID_OFFSET)
        gather_dot(weight, ids - _ID_OFFSET, hidden, 0)  # Rank 0's offset.


def _scores(rows: int, cols: int, levels: int, seed: int, device: str) -> torch.Tensor:
    """bf16 scores with many ties (``levels`` distinct values)."""
    gen = torch.Generator().manual_seed(seed)
    values = torch.randint(0, levels, (rows, cols), generator=gen).float() / levels
    return values.to(device, torch.bfloat16)


def _select_reference(values: torch.Tensor, k: int) -> torch.Tensor:
    """Positions of each row's ``k`` largest values in CUDA ``torch.topk(...,
    sorted=False)`` order: those above the k-th largest by position, then the
    first k-th largest ties by position."""
    cols = values.shape[-1]
    kth = torch.topk(values, k, dim=-1).values[:, -1:]
    gt, eq = values > kth, values == kth
    take = gt | (eq & (eq.cumsum(-1) <= k - gt.sum(-1, keepdim=True)))
    pos = torch.arange(cols, device=values.device).expand_as(values)
    order = torch.where(take, torch.where(gt, pos, cols + pos), 3 * cols)
    return order.sort(-1).indices[:, :k]


# Shard vocabs of a 248320 vocab at TP 8, 4, 2 and 1, and a ragged small one.
@pytest.mark.parametrize("cols", [31040, 62080, 124160, 248320, 5000])
@pytest.mark.parametrize("levels", [64, 60000])
def test_shard_topk_matches_reference(cols: int, levels: int, device: str) -> None:
    rows = 7
    scores = _scores(rows, cols, levels, cols + levels, device)
    ids = torch.empty((rows, 128), dtype=torch.int64, device=device)
    triton_eagle4_shard_topk(scores, 128, 1000, ids)
    expected = _select_reference(scores.float(), 128) + 1000
    torch.testing.assert_close(ids, expected, rtol=0, atol=0)


def test_shard_topk_writes_strided_rows_and_compiles_once(device: str) -> None:
    """Ids land in a row-strided view, and the row count never compiles."""
    scores = _scores(64, 62080, 64, 3, device)
    packed = torch.zeros((64, 160), dtype=torch.int64, device=device)
    triton_eagle4_shard_topk(scores, 128, 0, packed[:, 16:144])
    expected = _select_reference(scores.float(), 128)
    torch.testing.assert_close(packed[:, 16:144], expected, rtol=0, atol=0)
    assert not packed[:, :16].any() and not packed[:, 144:].any()
    kernels = (_shard_topk_hist_kernel, _shard_topk_select_kernel)
    with assert_no_triton_compile(*kernels, _shard_topk_emit_kernel):
        for rows in (1, 3, 17, 32, 64):
            triton_eagle4_shard_topk(scores[:rows], 128, 62080, packed[:rows, 16:144])


def _tail_problem(ranks: int, rows: int, nc: int, levels: int, seed: int, device: str):
    logits = _scores(ranks * rows, nc, levels, seed, device) * 8
    gen = torch.Generator().manual_seed(seed + 1)
    tokens = torch.randint(0, 248320, (ranks * rows, nc), generator=gen).to(device)
    return logits.view(ranks, rows, nc), tokens.view(ranks, rows, nc)


@pytest.mark.parametrize("nc", [32, 128, 256])
@pytest.mark.parametrize("ranks", [1, 2, 4])
@pytest.mark.parametrize("top_k", [0, 1, 8, 32])
@pytest.mark.parametrize("levels", [16, 60000])
def test_candidate_tail_matches_reference(
    nc: int, ranks: int, top_k: int, levels: int, device: str
) -> None:
    if ranks * nc > 1024:
        pytest.skip("the tail covers ranks * nc <= 1024")
    rows = 6
    cand_logits, cand_tokens = _tail_problem(
        ranks, rows, nc, levels, nc + top_k, device
    )
    logits = torch.empty((rows, nc), device=device)
    tokens = torch.empty((rows, nc), dtype=torch.int64, device=device)
    probs = torch.empty((rows, top_k), device=device) if top_k else None
    top_tokens = (
        torch.empty((rows, top_k), dtype=torch.int64, device=device) if top_k else None
    )
    triton_eagle4_candidate_tail(
        cand_logits, cand_tokens, top_k, logits, tokens, probs, top_tokens
    )

    # The rank-concatenated [M, R * NC] view the all-gather along the last dim makes.
    pool_logits = cand_logits.permute(1, 0, 2).reshape(rows, ranks * nc)
    pool_tokens = cand_tokens.permute(1, 0, 2).reshape(rows, ranks * nc)
    if ranks > 1:
        picked = _select_reference(pool_logits.float(), nc)
        ref_logits = pool_logits.gather(-1, picked).float()
        ref_tokens = pool_tokens.gather(-1, picked)
        torch.testing.assert_close(tokens, ref_tokens, rtol=0, atol=0)
    else:
        ref_logits, ref_tokens = pool_logits.float(), pool_tokens
    torch.testing.assert_close(logits, ref_logits, rtol=0, atol=0)
    if top_k:
        # Tied probabilities in slot order.
        ref_probs, order = torch.softmax(ref_logits, -1).sort(
            dim=-1, descending=True, stable=True
        )
        torch.testing.assert_close(probs, ref_probs[:, :top_k], rtol=0, atol=0)
        torch.testing.assert_close(
            top_tokens, ref_tokens.gather(-1, order[:, :top_k]), rtol=0, atol=0
        )


def test_candidate_tail_takes_strided_views_and_compiles_once(device: str) -> None:
    """The tail reads rank blocks of a packed all-gather buffer in place, and
    neither the row count nor the strides compile."""
    ranks, nc, top_k = 2, 128, 8
    # [R, M, 2 * NC]: logits in the first NC columns, as a packed exchange.
    packed, cand_tokens = _tail_problem(ranks, 64, 2 * nc, 60000, 9, device)
    cand_logits = packed[..., :nc]
    cand_tokens = cand_tokens[..., nc:]
    logits = torch.empty((64, nc), device=device)
    tokens = torch.empty((64, nc), dtype=torch.int64, device=device)
    probs = torch.empty((64, top_k), device=device)
    top_tokens = torch.empty((64, top_k), dtype=torch.int64, device=device)
    triton_eagle4_candidate_tail(
        cand_logits, cand_tokens, top_k, logits, tokens, probs, top_tokens
    )
    picked = _select_reference(
        cand_logits.permute(1, 0, 2).reshape(64, ranks * nc).float(), nc
    )
    ref_tokens = cand_tokens.permute(1, 0, 2).reshape(64, ranks * nc).gather(-1, picked)
    torch.testing.assert_close(tokens, ref_tokens, rtol=0, atol=0)
    with assert_no_triton_compile(_candidate_tail_kernel):
        for rows in (1, 3, 17, 33):
            triton_eagle4_candidate_tail(
                cand_logits[:, :rows],
                cand_tokens[:, :rows],
                top_k,
                logits[:rows],
                tokens[:rows],
                probs[:rows],
                top_tokens[:rows],
            )
        # A narrower packed exchange: other rank and row strides.
        narrow = packed[:, :32, : nc + 16].contiguous()[..., :nc]
        triton_eagle4_candidate_tail(
            narrow,
            cand_tokens[:, :32],
            top_k,
            logits[:32],
            tokens[:32],
            probs[:32],
            top_tokens[:32],
        )

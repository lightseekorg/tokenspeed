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

"""Exact top-p nucleus thresholds against a sort-based reference, and the
Gumbel-Max over the nucleus they define."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.sampling.triton import (
    gumbel_sample_from_pools_in_nucleus,
    gumbel_scratch_shape,
    top_p_threshold_scratch_shapes,
    top_p_thresholds_from_pools,
    topp_threshold,
)

VOCAB = 151936


def _reference_kept(logits: torch.Tensor, temperature: float, top_p: float):
    """Rank by scaled logit, ties by lower id; keep while the mass before a
    token is below top_p of the total (fp64)."""
    s = logits.float() / temperature
    if top_p >= 1.0:
        return torch.ones_like(s, dtype=torch.bool)
    order = torch.sort(-s, dim=-1, stable=True).indices
    ranked = torch.gather(s, 1, order)
    w = torch.exp(ranked - ranked[:, :1]).double()
    before = torch.cumsum(w, -1) - w
    kept_ranked = before < top_p * w.sum(-1, keepdim=True)
    kept = torch.zeros_like(kept_ranked)
    kept.scatter_(1, order, kept_ranked)
    return kept


class _Thresholds:
    """Pools and scratch for a batch of rows, one request per row."""

    def __init__(self, rows: int, vocab: int, device: str):
        self.pool = torch.arange(rows, dtype=torch.int32, device=device)
        self.scratch = {
            name: torch.empty(shape, dtype=dtype, device=device)
            for name, (shape, dtype) in top_p_threshold_scratch_shapes(
                rows, vocab
            ).items()
        }
        self.threshold = torch.empty(rows, dtype=torch.float32, device=device)
        self.limit = torch.empty(rows, dtype=torch.int32, device=device)
        self.rare = torch.empty(rows, dtype=torch.int32, device=device)

    def kept(self, logits, temperature: torch.Tensor, top_p: torch.Tensor):
        rows, vocab = logits.shape
        threshold, limit = top_p_thresholds_from_pools(
            logits,
            self.pool[:rows],
            temperature,
            top_p,
            self.scratch,
            self.threshold[:rows],
            self.limit[:rows],
            self.rare[:rows],
            num_tokens_per_req=1,
        )
        s = logits.float() / temperature[:, None]
        ids = torch.arange(vocab, device=logits.device)
        return (s > threshold[:, None]) | (
            (s == threshold[:, None]) & (ids[None, :] <= limit[:, None])
        )


def _rows(kind: str, rows: int, device: str) -> torch.Tensor:
    generator = torch.Generator().manual_seed(17)
    if kind == "peaked":
        return (torch.randn(rows, VOCAB, generator=generator) * 4).to(device)
    if kind == "flat":
        # Nuclei of tens of thousands of tokens: past each block's candidates.
        return (torch.randn(rows, VOCAB, generator=generator) * 0.5).to(device)
    if kind == "bf16_ties":
        logits = torch.randn(rows, VOCAB, generator=generator) * 2
        return logits.bfloat16().float().to(device)
    if kind == "tied_tail":
        # A few high tokens, then every other token at one value: the rare
        # path's single-value bracket, resolved by token id.
        logits = torch.zeros(rows, VOCAB)
        logits[:, :5] = 3.0
        return logits.to(device)
    raise ValueError(kind)


@pytest.mark.parametrize("kind", ["peaked", "flat", "bf16_ties", "tied_tail"])
@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_thresholds_keep_the_reference_nucleus(kind, temperature, device) -> None:
    rows = 8
    logits = _rows(kind, rows, device)
    thresholds = _Thresholds(rows, VOCAB, device)
    for top_p in (0.05, 0.3, 0.9, 0.99):
        temp = torch.full((rows,), temperature, device=device)
        got = thresholds.kept(logits, temp, torch.full((rows,), top_p, device=device))
        want = _reference_kept(logits, temperature, top_p)
        # fp32 sums in the kernel against fp64 in the reference: only tokens
        # whose mass before them is within rounding of the target may differ.
        assert int((got != want).sum(-1).max()) <= 2, f"top_p {top_p}"
        assert bool(got.any(-1).all())


def test_rows_at_top_p_one_keep_every_token(device) -> None:
    rows = 4
    logits = _rows("peaked", rows, device)
    thresholds = _Thresholds(rows, VOCAB, device)
    top_p = torch.tensor([1.0, 0.5, 1.0, 0.9], device=device)
    got = thresholds.kept(logits, torch.ones(rows, device=device), top_p)
    assert bool(got[0].all()) and bool(got[2].all())
    assert not bool(got[1].all()) and not bool(got[3].all())
    assert thresholds.threshold[[0, 2]].isneginf().all()


def test_only_rows_past_the_candidates_take_the_rare_path(device) -> None:
    rows = 4
    logits = torch.cat([_rows("peaked", 2, device), _rows("flat", 2, device)])
    thresholds = _Thresholds(rows, VOCAB, device)
    top_p = torch.full((rows,), 0.95, device=device)
    got = thresholds.kept(logits, torch.ones(rows, device=device), top_p)
    assert thresholds.rare.tolist() == [0, 0, 1, 1]
    want = _reference_kept(logits, 1.0, 0.95)
    assert int((got != want).sum(-1).max()) <= 2


def test_rare_rows_past_the_histogram_slots_take_the_per_row_search(device) -> None:
    rows = topp_threshold._MAX_RARE_ROWS + 3
    vocab = 8192
    generator = torch.Generator().manual_seed(5)
    logits = (torch.randn(rows, vocab, generator=generator) * 0.3).to(device)
    thresholds = _Thresholds(rows, vocab, device)
    top_p = torch.full((rows,), 0.9, device=device)
    got = thresholds.kept(logits, torch.ones(rows, device=device), top_p)
    rare = thresholds.rare.tolist()
    assert rare.count(topp_threshold._RARE_HISTOGRAM) == topp_threshold._MAX_RARE_ROWS
    assert rare.count(topp_threshold._RARE_SEARCH) == 3
    want = _reference_kept(logits, 1.0, 0.9)
    assert int((got != want).sum(-1).max()) <= 2


def test_rare_path_thresholds_repeat_bitwise(device) -> None:
    """The histograms add in integers, so the order programs add in does not
    change the result."""
    rows = 4
    logits = _rows("flat", rows, device)
    thresholds = _Thresholds(rows, VOCAB, device)
    top_p = torch.full((rows,), 0.7, device=device)
    temperature = torch.ones(rows, device=device)
    results = []
    for _ in range(3):
        thresholds.kept(logits, temperature, top_p)
        results.append((thresholds.threshold.clone(), thresholds.limit.clone()))
    assert (thresholds.rare == topp_threshold._RARE_HISTOGRAM).all()
    for threshold, limit in results[1:]:
        assert torch.equal(threshold, results[0][0])
        assert torch.equal(limit, results[0][1])


def test_thresholds_do_not_recompile_across_row_counts(device) -> None:
    from utils import assert_no_triton_compile

    logits = _rows("peaked", 24, device)
    thresholds = _Thresholds(24, VOCAB, device)
    top_p = torch.full((24,), 0.9, device=device)
    temperature = torch.ones(24, device=device)
    thresholds.kept(logits[:3], temperature[:3], top_p[:3])
    thresholds.kept(logits[:16], temperature[:16], top_p[:16])
    with assert_no_triton_compile(
        topp_threshold._top_p_candidates_kernel,
        topp_threshold._top_p_select_kernel,
        topp_threshold._top_p_rare_level_kernel,
        topp_threshold._top_p_threshold_exact_kernel,
    ):
        for rows in (1, 5, 8, 13, 24):
            thresholds.kept(logits[:rows], temperature[:rows], top_p[:rows])


def _nucleus_probs(row: torch.Tensor, fraction: float):
    """A top-p placed mid-way through the mass of the token at ``fraction`` of
    the row's mass, so fp32 rounding cannot move the boundary; returns the
    top-p and the exact nucleus distribution (fp64)."""
    probs = torch.softmax(row.double(), dim=0)
    order = torch.sort(-row, stable=True).indices
    ranked = probs[order]
    before = torch.cumsum(ranked, 0) - ranked
    last = int((before < fraction).sum()) - 1
    top_p = float(before[last] + 0.5 * ranked[last])
    kept = torch.zeros_like(probs)
    kept[order[: last + 1]] = ranked[: last + 1]
    return top_p, kept / kept.sum()


def _chi2_p_value(counts: torch.Tensor, probs: torch.Tensor) -> float:
    """Goodness of fit, pooling bins expected below 5 draws."""
    expected = probs * counts.sum()
    small = expected < 5
    observed = torch.cat([counts[~small], counts[small].sum()[None]])
    expected = torch.cat([expected[~small], expected[small].sum()[None]])
    keep = expected > 0
    observed, expected = observed[keep], expected[keep]
    stat = ((observed - expected) ** 2 / expected).sum()
    dof = torch.tensor((observed.numel() - 1) / 2.0, dtype=torch.float64)
    return float(torch.special.gammaincc(dof, stat / 2))


@pytest.mark.parametrize(
    "kind,scale,fraction,rare",
    [
        # Most mass on a few tokens: the candidates hold the nucleus.
        ("peaked", 3.0, 0.6, 0),
        # A low top-p over a flat-ish head, where a sampler that retries a
        # bounded number of times and then takes the argmax over-draws it.
        ("low_top_p", 1.0, 0.12, 0),
        # A nucleus past every block's candidates: the histogram rare path.
        ("flat", 0.3, 0.5, 1),
    ],
)
def test_sampling_in_the_nucleus_follows_the_top_p_distribution(
    kind, scale, fraction, rare, device
) -> None:
    rows, vocab, draws = 64, 8192, 400
    generator = torch.Generator().manual_seed(23)
    row = torch.randn(vocab, generator=generator, dtype=torch.float64) * scale
    row = row.float()
    top_p, probs = _nucleus_probs(row, fraction)
    # One row per sample, all reading the same logits.
    logits = row.to(device).expand(rows, vocab)
    thresholds = _Thresholds(rows, vocab, device)
    temperature = torch.ones(rows, device=device)
    top_p_pool = torch.full((rows,), top_p, device=device)
    seed = torch.arange(rows, dtype=torch.int64, device=device) * 7919 + 3
    offsets = torch.zeros(rows, dtype=torch.int64, device=device)
    scratch_shape = gumbel_scratch_shape(rows, vocab)
    local_ids = torch.empty(scratch_shape, dtype=torch.int32, device=device)
    local_scores = torch.empty(scratch_shape, dtype=torch.float32, device=device)
    out = torch.empty(rows, dtype=torch.int32, device=device)
    counts = torch.zeros(vocab, dtype=torch.float64, device=device)
    for draw in range(draws):
        offsets.fill_(draw)
        nucleus = top_p_thresholds_from_pools(
            logits,
            thresholds.pool,
            temperature,
            top_p_pool,
            thresholds.scratch,
            thresholds.threshold,
            thresholds.limit,
            thresholds.rare,
            num_tokens_per_req=1,
        )
        sampled = gumbel_sample_from_pools_in_nucleus(
            logits,
            thresholds.pool,
            temperature,
            seed,
            offsets,
            local_ids,
            local_scores,
            out,
            nucleus,
            num_tokens_per_req=1,
        )
        counts += torch.bincount(sampled.long(), minlength=vocab).double()
    assert int(thresholds.rare.max()) == rare
    counts = counts.cpu()
    assert counts[probs == 0].sum() == 0, "a draw outside the nucleus"
    assert _chi2_p_value(counts, probs) > 1.0e-3

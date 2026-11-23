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

"""Target-only chain verification over each row's top-k logits (``topk_chain_verify``)."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.numerics.torch.sampling import (
    topk_chain_rows,
    torch_topk_chain_verify,
)
from tokenspeed_kernel.ops.sampling import topk_chain_verify, vocab_shard_topk
from tokenspeed_kernel.ops.sampling.triton.topk_chain_verify import (
    _topk_chain_verify_kernel,
)
from tokenspeed_kernel.ops.sampling.triton.vocab_topk import (
    _vocab_shard_topk_kernel,
    _vocab_topk_merge_kernel,
)
from utils import assert_no_triton_compile

K = 32
N = 4
# Coins lie in [2**-24, 1 - 2**-24], as the serving coin streams draw them.
_COIN_LO = 2.0**-24
_COIN_HI = 1.0 - 2.0**-24


def _logits(rows: int, vocab: int, seed: int, device: str) -> torch.Tensor:
    """bf16 rows with a few strong tokens each, so top-k and top-p both bite."""
    gen = torch.Generator().manual_seed(seed)
    logits = torch.randn(rows, vocab, generator=gen) * 2
    hot = torch.randint(0, vocab, (rows, 8), generator=gen)
    logits.scatter_add_(1, hot, torch.rand(rows, 8, generator=gen) * 12)
    return logits.to(device, torch.bfloat16)


def _params(bs: int, n: int, seed: int, device: str, top_k: int | None = None):
    """Per-request (temperature, top-k, top-p), repeated over the request's rows."""
    gen = torch.Generator().manual_seed(seed)
    temperatures = torch.tensor([0.6, 1.0, 1.3])[
        torch.randint(0, 3, (bs,), generator=gen)
    ]
    top_ks = torch.tensor([1, 5, 20, K])[torch.randint(0, 4, (bs,), generator=gen)]
    if top_k is not None:
        top_ks.fill_(top_k)
    top_ps = torch.tensor([0.8, 0.95, 1.0])[torch.randint(0, 3, (bs,), generator=gen)]
    return (
        temperatures.repeat_interleave(n).to(device),
        top_ks.to(torch.int32).repeat_interleave(n).to(device),
        top_ps.repeat_interleave(n).to(device),
    )


def _chains(logits: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    """Drafts mostly among the previous row's three best tokens, sometimes anywhere."""
    rows, vocab = logits.shape
    gen = torch.Generator().manual_seed(seed)
    best = torch.topk(logits.float().cpu(), 3, dim=1).indices.view(rows // n, n, 3)
    pick = torch.randint(0, 3, (rows // n, n, 1), generator=gen)
    chains = torch.gather(best, 2, pick).squeeze(2).roll(1, dims=1)
    anywhere = torch.randint(0, vocab, chains.shape, generator=gen)
    chains = torch.where(
        torch.rand(chains.shape, generator=gen) < 0.2, anywhere, chains
    )
    return chains.to(logits.device, torch.int32)


def _coins(shape: tuple[int, ...], seed: int, device: str) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(shape, generator=gen).clamp_(_COIN_LO, _COIN_HI).to(device)


def _select(logits: torch.Tensor) -> torch.Tensor:
    topk = torch.empty(logits.shape[0], 2 * K, device=logits.device)
    vocab_shard_topk(
        logits,
        K,
        None,
        0,
        topk,
        softcap=None,
        softcap_form=None,
        vocab_mask=None,
        values_out=None,
        halo_out=None,
    )
    return topk


def _verify(verify, topk, chains, coins, final_coins, temperatures, top_ks, top_ps):
    bs, n = chains.shape
    device = topk.device
    predicts = torch.full((bs * n,), -7, dtype=torch.int32, device=device)
    accept_index = torch.full((bs, n), -7, dtype=torch.int32, device=device)
    accepted = torch.full((bs,), -7, dtype=torch.int32, device=device)
    verify(
        predicts,
        accept_index,
        accepted,
        chains,
        coins,
        final_coins,
        topk,
        K,
        temperatures,
        top_ks,
        top_ps,
        threshold_single=1.0,
        threshold_acc=1.0,
    )
    return predicts, accept_index, accepted


def _emitted(predicts: torch.Tensor, accepted: torch.Tensor, n: int) -> list[list[int]]:
    """Each request's committed tokens: the accepted drafts and the token after them."""
    tokens = predicts.view(-1, n).tolist()
    return [row[: int(a) + 1] for row, a in zip(tokens, accepted.tolist())]


def _margin(rows, chains, coins, final_coins, n: int, b: int) -> float:
    """Smallest distance of request ``b``'s float64 decisions from their thresholds."""
    margins = []
    accepted = n - 1
    for i in range(1, n):
        prob = rows[b * n + i - 1].get(int(chains[b, i]), 0.0)
        margins.append(abs(float(coins[b, i - 1]) - prob))
        if float(coins[b, i - 1]) > prob and prob < 1.0:
            accepted = i - 1
            break
    rejected = int(chains[b, accepted + 1]) if accepted < n - 1 else None
    mass = {t: p for t, p in rows[b * n + accepted].items() if t != rejected}
    total = sum(mass.values())
    running = 0.0
    for token in sorted(mass):
        running += mass[token]
        margins.append(abs(running - float(final_coins[b]) * total) / total)
    return min(margins)


def test_matches_the_float64_reference_but_at_rounding_boundaries(device: str) -> None:
    """Mixed temperatures, top-k (1 to K) and top-p over 2048 requests: tokens,
    accepted lengths and indices equal the float64 rule except where a decision
    lies within rounding of its threshold (the kernel works in fp32)."""
    bs, vocab = 2048, 50000
    logits = _logits(bs * N, vocab, 1, device)
    temperatures, top_ks, top_ps = _params(bs, N, 2, device)
    chains = _chains(logits, N, 3)
    coins = _coins((bs, N), 4, device)
    final_coins = _coins((bs,), 5, device)
    topk = _select(logits)

    got = _verify(
        topk_chain_verify,
        topk,
        chains,
        coins,
        final_coins,
        temperatures,
        top_ks,
        top_ps,
    )
    want = _verify(
        torch_topk_chain_verify,
        topk,
        chains,
        coins,
        final_coins,
        temperatures,
        top_ks,
        top_ps,
    )

    rows = topk_chain_rows(topk, K, temperatures, top_ks, top_ps)
    got_tokens, want_tokens = _emitted(got[0], got[2], N), _emitted(want[0], want[2], N)
    mismatched = [b for b in range(bs) if got_tokens[b] != want_tokens[b]]
    margins = [
        _margin(rows, chains.cpu(), coins.cpu(), final_coins.cpu(), N, b)
        for b in mismatched
    ]
    print(
        f"{len(mismatched)} of {bs} requests differ; float64 decision margins {margins}"
    )
    assert len(mismatched) <= bs // 1000
    assert all(margin < 1e-5 for margin in margins)
    agree = torch.tensor([b not in mismatched for b in range(bs)], device=device)
    assert torch.equal(got[2][agree], want[2][agree])
    assert torch.equal(got[1][agree], want[1][agree])
    # Some chains accept everything, some reject the first draft.
    assert {0, N - 1} <= set(got[2].tolist())


def _chi_square_p(counts: torch.Tensor, expected: torch.Tensor) -> float:
    """Pearson's goodness-of-fit p-value, merging categories expected below 5 times."""
    large = expected >= 5
    observed = torch.cat([counts[large], counts[~large].sum().view(1)])
    wanted = torch.cat([expected[large], expected[~large].sum().view(1)])
    if wanted[-1] == 0:
        observed, wanted = observed[:-1], wanted[:-1]
    statistic = ((observed - wanted) ** 2 / wanted).sum()
    dof = observed.numel() - 1
    return float(torch.special.gammaincc(torch.tensor(dof / 2.0), statistic / 2.0))


@pytest.mark.parametrize("draft_rank", [0, 3, None])
def test_first_token_follows_the_filtered_target(draft_rank, device: str) -> None:
    """Speculative sampling keeps the target law: over 200000 coin draws of one
    request (one draft, T 0.8, top-k 20, top-p 0.9), the first committed token
    is distributed as the filtered target, whether the draft is its best token,
    a lower kept token, or outside the kept set. Pearson chi-square, p >= 1e-4."""
    draws, vocab = 200000, 4096
    # Both rows of every request are one logits row (stride 0): the bonus row never shows in the first token.
    rows = _logits(1, vocab, 11, device).expand(draws * 2, vocab)
    temperatures = torch.full((draws * 2,), 0.8, device=device)
    top_ks = torch.full((draws * 2,), 20, dtype=torch.int32, device=device)
    top_ps = torch.full((draws * 2,), 0.9, device=device)
    topk = _select(rows)
    target = topk_chain_rows(topk[:1], K, temperatures[:1], top_ks[:1], top_ps[:1])[0]
    kept = sorted(target, key=target.get, reverse=True)
    outside = next(t for t in range(vocab) if t not in target)
    draft = outside if draft_rank is None else kept[draft_rank]
    chains = (
        torch.tensor([[0, draft]], dtype=torch.int32, device=device)
        .expand(draws, 2)
        .contiguous()
    )
    coins = _coins((draws, 2), 12, device)
    final_coins = _coins((draws,), 13, device)

    predicts, _, _ = _verify(
        topk_chain_verify,
        topk,
        chains,
        coins,
        final_coins,
        temperatures,
        top_ks,
        top_ps,
    )

    first = predicts.view(draws, 2)[:, 0].long().cpu()
    assert set(first.unique().tolist()) <= set(target)
    counts = torch.bincount(first, minlength=vocab).double()[kept]
    expected = torch.tensor([target[t] for t in kept], dtype=torch.float64) * draws
    p_value = _chi_square_p(counts, expected)
    print(
        f"draft rank {draft_rank}: {len(kept)} kept tokens, {draws} draws, p = {p_value:.3g}"
    )
    assert p_value >= 1e-4


def _one_request(logits, top_k, top_p, chain, coins, final_coin, temperature=1.0):
    device = logits.device
    n = logits.shape[0]
    return _verify(
        topk_chain_verify,
        _select(logits),
        torch.tensor([chain], dtype=torch.int32, device=device),
        torch.tensor([coins], device=device),
        torch.tensor([final_coin], device=device),
        torch.full((n,), temperature, device=device),
        torch.full((n,), top_k, dtype=torch.int32, device=device),
        torch.full((n,), top_p, device=device),
    )


def _peaked(device: str, best: list[int], vocab: int = 4096) -> torch.Tensor:
    """Rows whose token ``best[r]`` dominates every other by 15 logits."""
    logits = torch.zeros(len(best), vocab)
    logits[torch.arange(len(best)), torch.tensor(best)] = 15.0
    return logits.to(device, torch.bfloat16)


def test_top_p_down_to_one_token_accepts_it_at_every_coin(device: str) -> None:
    """A top-p cut to one token gives it probability exactly 1: drafts equal to
    it accept even at the largest coin, and the bonus is the last row's token."""
    best = [5, 9, 2, 7]
    predicts, accept_index, accepted = _one_request(
        _peaked(device, best), 20, 0.5, [1, 5, 9, 2], [_COIN_HI] * N, _COIN_HI
    )
    assert accepted.tolist() == [N - 1]
    assert predicts.tolist() == [5, 9, 2, 7]
    assert accept_index.tolist() == [[0, 1, 2, 3]]


def test_first_draft_outside_the_kept_set_stops_the_chain(device: str) -> None:
    """A draft the target never keeps has probability 0 and is rejected at
    any positive coin; the next token comes from the rest of row 0."""
    best = [5, 9, 2, 7]
    predicts, accept_index, accepted = _one_request(
        _peaked(device, best), 20, 0.5, [1, 4000, 9, 2], [_COIN_LO] * N, 0.5
    )
    assert accepted.tolist() == [0]
    assert predicts[0].item() == 5
    assert accept_index.tolist() == [[0, -1, -1, -1]]


def test_rejecting_the_only_kept_token_samples_among_the_rest(device: str) -> None:
    """Top-k 1 rejects any other draft; the residual is then the best token."""
    best = [5, 9, 2, 7]
    predicts, _, accepted = _one_request(
        _peaked(device, best), 1, 1.0, [1, 5, 3, 2], [0.5] * N, _COIN_HI
    )
    assert accepted.tolist() == [1]
    assert predicts[:2].tolist() == [5, 9]


def test_masked_tokens_are_never_drawn(device: str) -> None:
    """Rows with three finite logits (a grammar mask) and top-k K: every
    committed token is one of the three, at every coin."""
    draws, vocab = 4096, 4096
    allowed = [17, 900, 3001]
    row = torch.full((N, vocab), float("-inf"))
    row[:, allowed] = torch.tensor([1.0, 0.5, 0.25])
    logits = row.to(device, torch.bfloat16).repeat(draws, 1)
    chains = torch.full((draws, N), 5, dtype=torch.int32, device=device)
    chains[:, 1] = 900
    temperatures, top_ks, top_ps = (
        torch.ones(draws * N, device=device),
        torch.full((draws * N,), K, dtype=torch.int32, device=device),
        torch.ones(draws * N, device=device),
    )
    predicts, _, accepted = _verify(
        topk_chain_verify,
        _select(logits),
        chains,
        _coins((draws, N), 21, device),
        _coins((draws,), 22, device),
        temperatures,
        top_ks,
        top_ps,
    )
    tokens = {t for row in _emitted(predicts, accepted, N) for t in row}
    assert tokens == set(allowed)


def test_ties_at_the_kth_value_keep_the_lowest_ids(device: str) -> None:
    """Sixty tokens share the K-th logit: top-k K keeps the lowest ids among
    them, as the dense top-k does, so the higher ones are never drawn."""
    draws, vocab = 4096, 4096
    row = torch.zeros(N, vocab)
    row[:, 100:110] = 2.0
    row[:, 200:260] = 1.0
    logits = row.to(device, torch.bfloat16).repeat(draws, 1)
    chains = torch.full((draws, N), 4000, dtype=torch.int32, device=device)
    temperatures, top_ks, top_ps = (
        torch.ones(draws * N, device=device),
        torch.full((draws * N,), K, dtype=torch.int32, device=device),
        torch.ones(draws * N, device=device),
    )
    predicts, _, accepted = _verify(
        topk_chain_verify,
        _select(logits),
        chains,
        _coins((draws, N), 31, device),
        _coins((draws,), 32, device),
        temperatures,
        top_ks,
        top_ps,
    )
    assert accepted.max().item() == 0
    tokens = set(predicts.view(draws, N)[:, 0].tolist())
    assert tokens == set(range(100, 110)) | set(range(200, 200 + K - 10))


def test_top_k_one_is_greedy_verification(device: str) -> None:
    """Top-k 1 accepts exactly the drafts equal to the previous row's argmax and
    then emits that row's argmax, whatever the coins."""
    bs, vocab = 256, 50000
    logits = _logits(bs * N, vocab, 41, device)
    temperatures, top_ks, top_ps = _params(bs, N, 42, device, top_k=1)
    chains = _chains(logits, N, 43)
    argmax = logits.float().cpu().argmax(dim=1).view(bs, N)
    predicts, _, accepted = _verify(
        topk_chain_verify,
        _select(logits),
        chains,
        _coins((bs, N), 44, device),
        _coins((bs,), 45, device),
        temperatures,
        top_ks,
        top_ps,
    )
    chains = chains.cpu()
    for b, tokens in enumerate(_emitted(predicts, accepted, N)):
        matched = 0
        while matched < N - 1 and int(chains[b, matched + 1]) == int(
            argmax[b, matched]
        ):
            matched += 1
        assert tokens == argmax[b, : matched + 1].tolist()


def test_same_inputs_give_the_same_tokens(device: str) -> None:
    bs, vocab = 512, 50000
    logits = _logits(bs * N, vocab, 51, device)
    params = _params(bs, N, 52, device)
    chains = _chains(logits, N, 53)
    coins, final_coins = _coins((bs, N), 54, device), _coins((bs,), 55, device)
    first = _verify(
        topk_chain_verify, _select(logits), chains, coins, final_coins, *params
    )
    second = _verify(
        topk_chain_verify, _select(logits), chains, coins, final_coins, *params
    )
    for a, b in zip(first, second):
        assert torch.equal(a, b)


@pytest.mark.parametrize("bs,position", [(1, 0), (3, 1), (8, 7), (32, 13)])
def test_a_request_selects_and_verifies_the_same_at_any_batch_size(
    bs, position, device: str
) -> None:
    """One request (its rows, knobs, chain and coins) placed at different batch
    sizes and positions selects bit-identical candidates and commits the same
    tokens, accepted length and relative accept indices."""
    vocab = 50000
    probe_logits = _logits(N, vocab, 61, device)
    probe_params = _params(1, N, 62, device, top_k=20)
    probe_chain = _chains(probe_logits, N, 63)
    probe_coins, probe_final = _coins((1, N), 64, device), _coins((1,), 65, device)

    logits = _logits(bs * N, vocab, 70 + bs, device)
    params = list(_params(bs, N, 80 + bs, device))
    chains = _chains(logits, N, 90 + bs)
    coins, final_coins = _coins((bs, N), 100 + bs, device), _coins(
        (bs,), 110 + bs, device
    )
    rows = slice(position * N, (position + 1) * N)
    logits[rows] = probe_logits
    for got, want in zip(params, probe_params):
        got[rows] = want
    chains[position] = probe_chain[0]
    coins[position] = probe_coins[0]
    final_coins[position] = probe_final[0]

    probe_topk = _select(probe_logits)
    alone = _verify(
        topk_chain_verify,
        probe_topk,
        probe_chain,
        probe_coins,
        probe_final,
        *probe_params,
    )
    topk = _select(logits)
    batched = _verify(topk_chain_verify, topk, chains, coins, final_coins, *params)

    assert torch.equal(topk[rows], probe_topk)
    accepted = int(alone[2][0])
    assert batched[2][position].item() == accepted
    assert torch.equal(batched[0][rows][: accepted + 1], alone[0][: accepted + 1])
    relative = torch.where(
        batched[1][position] >= 0, batched[1][position] - position * N, -1
    )
    assert torch.equal(relative, alone[1][0])


def test_batch_size_and_strides_do_not_recompile(device: str) -> None:
    vocab = 50000

    def run(bs: int, coin_width: int) -> None:
        logits = _logits(bs * N, vocab, bs, device)
        coins = _coins((bs, coin_width), bs, device)[:, :N]
        _verify(
            topk_chain_verify,
            _select(logits),
            _chains(logits, N, bs),
            coins,
            _coins((bs,), bs + 1, device),
            *_params(bs, N, bs, device),
        )

    run(16, N)
    run(5, N + 1)
    with assert_no_triton_compile(
        _topk_chain_verify_kernel, _vocab_shard_topk_kernel, _vocab_topk_merge_kernel
    ):
        for bs, coin_width in ((1, N), (7, N + 3), (33, 16), (128, N)):
            run(bs, coin_width)

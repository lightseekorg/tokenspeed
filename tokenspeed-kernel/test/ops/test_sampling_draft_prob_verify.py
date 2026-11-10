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

"""The Triton sampling backend's dense draft-distribution verify: the row
softmax, each verify row's target distribution p', the chain rule against
recorded per-vocab draft distributions, and the position-keyed tree coins."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.sampling import softmax
from tokenspeed_kernel.ops.sampling.triton import (
    TOP_K_DISABLED,
    chain_verify_draft_probs,
)
from tokenspeed_kernel.ops.sampling.triton import draft_prob_verify as dpv
from tokenspeed_kernel.ops.sampling.triton import softmax as triton_softmax_module
from tokenspeed_kernel.ops.sampling.triton import (
    target_probs_from_pools,
    tree_verify_coins,
)
from tokenspeed_kernel.ops.sampling.triton.rejection import COIN_OFFSET_SALT
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

_BLOCK = 1024
_CHI2_MIN_P_VALUE = 1.0e-3
_THRESHOLD = 2.0
_SENTINEL = _THRESHOLD + 1.0


@triton.jit
def _coin_kernel(seed_ptr, key_ptr, lane_ptr, out_ptr, COIN_SALT: tl.constexpr):
    """The documented coin: ``rand(randint(seed, SALT + key), lane)``."""
    i = tl.program_id(0)
    seed = tl.load(seed_ptr + i).to(tl.int64)
    key = tl.load(key_ptr + i).to(tl.int64)
    lane = tl.load(lane_ptr + i)
    tl.store(out_ptr + i, tl.rand(tl.randint(seed, COIN_SALT + key), lane))


def _coins(seed: torch.Tensor, key: torch.Tensor, lane: torch.Tensor) -> torch.Tensor:
    out = torch.empty(seed.shape, dtype=torch.float32, device=seed.device)
    _coin_kernel[(seed.numel(),)](
        seed.to(torch.int64).contiguous(),
        key.to(torch.int64).contiguous(),
        lane.to(torch.int32).contiguous(),
        out,
        COIN_SALT=COIN_OFFSET_SALT,
    )
    return out


def _route_probs(
    logits: torch.Tensor, temperature: float, top_k: int, top_p: float, pad: int
) -> torch.Tensor:
    """A row's p' from its definition, in fp64: the softmax, or over the
    ``min(top_k, pad)`` best logits (lowest id first among equal logits) the
    tokens whose top-k mass above them is below ``top_p``, renormalized."""
    scaled = logits.double() / max(temperature, 1.0e-20)
    if top_k == TOP_K_DISABLED:
        return torch.softmax(scaled, dim=0)
    order = torch.sort(logits, descending=True, stable=True).indices
    order = order[: min(max(top_k, 1), pad)]
    probs = torch.softmax(scaled[order], dim=0)
    kept = order[torch.cumsum(probs, 0) - probs < top_p]
    out = torch.zeros_like(scaled)
    out[kept] = torch.softmax(scaled[kept], dim=0)
    return out


def _assert_follows(tokens: torch.Tensor, probs: torch.Tensor, label: str) -> None:
    """Chi-squared goodness of fit, pooling bins expected below 5 draws."""
    counts = torch.bincount(tokens.long().cpu(), minlength=probs.numel()).double()
    total = counts.sum()
    probs = probs.double().cpu()
    support = probs > 0
    assert counts[~support].sum() == 0, f"{label}: draws outside the support"
    expected = probs[support] / probs[support].sum() * total
    observed = counts[support]
    small = expected < 5
    if bool(small.any()):
        expected = torch.cat([expected[~small], expected[small].sum()[None]])
        observed = torch.cat([observed[~small], observed[small].sum()[None]])
    dof = expected.numel() - 1
    if dof == 0:
        return
    stat = ((observed - expected) ** 2 / expected).sum()
    half_dof = torch.tensor(dof / 2.0, dtype=torch.float64)
    p_value = float(torch.special.gammaincc(half_dof, stat / 2))
    assert (
        p_value > _CHI2_MIN_P_VALUE
    ), f"{label}: chi2 {stat:.1f} dof {dof} p {p_value:.2e}"


# ---------------------------------------------------------------------------
# Softmax and p'.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_softmax_matches_torch(dtype: torch.dtype, device: str) -> None:
    generator = torch.Generator().manual_seed(0)
    rows, vocab = 7, 5000
    logits = (torch.randn(rows, vocab, generator=generator) * 4).to(dtype)
    temperature = torch.tensor([1.0, 0.5, 1.7, 0.0, 1.0, 0.25, 2.0])
    inv = torch.where(temperature == 0, 0.0, 1.0 / temperature)
    want = torch.softmax(logits.double() * inv.double()[:, None], dim=-1)
    logits, temperature = logits.to(device), temperature.to(device)
    # The family op (the CUDA softmax on NVIDIA) and the Triton build.
    for got in (
        softmax(logits, temperature=temperature),
        triton_softmax_module.triton_softmax(logits, temperature=temperature, out=None),
    ):
        assert got.dtype == torch.float32
        torch.testing.assert_close(got.double().cpu(), want, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("route", ["no_filter", "top_k", "top_k_top_p"])
@pytest.mark.parametrize("pad", [64, 128])
def test_target_probs_match_the_route_distribution(
    route: str, pad: int, device: str
) -> None:
    generator = torch.Generator().manual_seed(1)
    bs, n, vocab, pool = 3, 4, 3000, 6
    logits = torch.randn(bs * n, vocab, generator=generator) * 3
    # Ties straddling the cut: the lowest id is kept first.
    logits[0, 10:20] = 9.0
    temperature = torch.tensor([1.0, 0.7, 1.0, 1.3, 0.9, 0.5])
    if route == "no_filter":
        top_k = torch.full((pool,), TOP_K_DISABLED, dtype=torch.int32)
        top_p = torch.ones(pool)
    else:
        top_k = torch.tensor([1, 5, 20, 1, 300, 64], dtype=torch.int32)
        top_p = (
            torch.ones(pool)
            if route == "top_k"
            else torch.tensor([1.0, 0.9, 0.5, 1.0, 0.95, 0.3])
        )
    req = torch.tensor([4, 1, 5], dtype=torch.int32)
    blocks = triton.cdiv(vocab, _BLOCK)
    ids = torch.empty(bs * n, blocks * 128, dtype=torch.int32, device=device)
    values = torch.empty(bs * n, blocks * 128, dtype=torch.float32, device=device)
    out = torch.full((bs * n, vocab), float("nan"), device=device)
    target_probs_from_pools(
        logits.to(device),
        req.to(device),
        temperature.to(device),
        top_k.to(device),
        top_p.to(device),
        ids,
        values,
        out,
        top_k_filter=route != "no_filter",
        block_size=_BLOCK,
        top_k_pad=pad,
        num_tokens_per_req=n,
    )
    for row in range(bs * n):
        slot = int(req[row // n])
        want = _route_probs(
            logits[row], float(temperature[slot]), int(top_k[slot]),
            float(top_p[slot]), pad,
        )  # fmt: skip
        got = out[row].double().cpu()
        assert torch.equal(got > 0, want > 0), f"row {row}: support"
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------------------
# The chain rule.
# ---------------------------------------------------------------------------


def _chain_problem(bs: int, n: int, vocab: int, pool: int, generator):
    """Requests at pool slots ``1..bs`` with random p' rows and q steps."""
    p = torch.softmax(torch.randn(bs * n, vocab, generator=generator) * 2, -1)
    q = torch.full((pool, n, vocab), _SENTINEL)
    q[:, -1] = 0.0
    q[:, :-1] = torch.softmax(torch.randn(pool, n - 1, vocab, generator=generator), -1)
    candidates = torch.randint(
        0, vocab, (bs, n), generator=generator, dtype=torch.int32
    )
    req = torch.arange(1, bs + 1, dtype=torch.int32)
    seed = torch.randint(0, 2**40, (pool,), generator=generator)
    offsets = torch.randint(0, 2**20, (pool,), generator=generator)
    return p, q, candidates, req, seed, offsets


def _run_chain(p, q, candidates, req, seed, offsets, device):
    bs, n = candidates.shape
    predicts = torch.full((bs * n,), -1, dtype=torch.int32, device=device)
    accept_index = torch.full((bs, n), -1, dtype=torch.int32, device=device)
    accept_length = torch.full((bs,), -1, dtype=torch.int32, device=device)
    chain_verify_draft_probs(
        p.to(device), q.to(device), candidates.to(device), req.to(device),
        seed.to(device), offsets.to(device), predicts, accept_index,
        accept_length, reject_threshold=_THRESHOLD,
    )  # fmt: skip
    return predicts.cpu(), accept_index.cpu(), accept_length.cpu()


def test_chain_rule_matches_the_reference_given_the_kernel_coins(device: str) -> None:
    """Every accept decision is ``coin * q < p`` with the documented coin, the
    walk stops at the first rejection, and the emitted token is the
    inverse-CDF draw of ``relu(p - q)`` (``p`` after the last row) at the
    residual coin."""
    generator = torch.Generator().manual_seed(2)
    bs, n, vocab, pool = 64, 5, 300, 70
    p, q, candidates, req, seed, offsets = _chain_problem(bs, n, vocab, pool, generator)
    # Make most drafts likely so the walks run deep: draw them from q.
    for b in range(bs):
        slot = int(req[b])
        for i in range(1, n):
            candidates[b, i] = int(
                torch.multinomial(q[slot, i - 1], 1, generator=generator)
            )
    # A few rows without a recorded proposal (the sentinel) reject.
    q[req[:4].long(), 1] = _SENTINEL
    predicts, accept_index, accept_length = _run_chain(
        p, q, candidates, req, seed, offsets, device
    )
    slots = req.long()
    keys = offsets[slots][:, None] + torch.arange(n)[None, :]
    accept_coins = _coins(
        seed[slots][:, None].expand(bs, n).to(device), keys.to(device),
        torch.zeros(bs, n, dtype=torch.int32, device=device),
    ).cpu()  # fmt: skip
    residual_coins = _coins(
        seed[slots][:, None].expand(bs, n).to(device), keys.to(device),
        torch.ones(bs, n, dtype=torch.int32, device=device),
    ).cpu()  # fmt: skip
    deep = 0
    for b in range(bs):
        slot = int(slots[b])
        length = 0
        for i in range(1, n):
            x = int(candidates[b, i])
            qx = float(q[slot, i - 1, x])
            px = float(p[b * n + i - 1, x])
            accept = (
                qx <= _THRESHOLD
                and float(
                    torch.tensor(float(accept_coins[b, i - 1])) * torch.tensor(qx)
                )
                < px
            )
            if not accept:
                break
            length += 1
        deep += length >= 2
        assert int(accept_length[b]) == length, f"request {b}"
        assert accept_index[b, : length + 1].tolist() == [
            b * n + j for j in range(length + 1)
        ]
        assert (accept_index[b, length + 1 :] == -1).all()
        assert (
            predicts[b * n : b * n + length].tolist()
            == candidates[b, 1 : length + 1].tolist()
        )
        # The emitted token brackets its coin's share of the residual mass.
        row = p[b * n + length].double()
        if length < n - 1:
            sub = q[slot, length].double()
            sub = torch.where(sub > _THRESHOLD, 0.0, sub)
            row = (row - sub).clamp_min(0.0)
        token = int(predicts[b * n + length])
        assert row[token] > 0, f"request {b}: emitted outside the residual"
        target = float(residual_coins[b, length]) * float(row.sum())
        before = float(row[:token].sum())
        assert before - 1e-5 <= target <= before + float(row[token]) + 1e-5
    assert deep > 0


def test_chain_rule_one_hot_outcomes(device: str) -> None:
    """Without randomness in play: q equal to a one-hot p' accepts, a draft
    outside p' rejects and the residual is p' itself, a sentinel step
    rejects and the residual is p', and the bonus after the last row is
    p'."""
    bs, n, vocab, pool = 3, 3, 16, 4
    p = torch.zeros(bs * n, vocab)
    q = torch.full((pool, n, vocab), _SENTINEL)
    q[:, -1] = 0.0
    candidates = torch.zeros(bs, n, dtype=torch.int32)
    for b in range(bs):
        for r in range(n):
            p[b * n + r, (3 * b + r + 1) % vocab] = 1.0
    req = torch.tensor([1, 2, 3], dtype=torch.int32)
    # Request 0: every draft is p's mode and q its one-hot -> all accepted.
    for i in range(1, n):
        token = (0 + i) % vocab
        candidates[0, i] = token
        q[1, i - 1] = 0.0
        q[1, i - 1, token] = 1.0
    # Request 1: the first draft has p' = 0 -> rejected; residual = p'.
    candidates[1, 1] = 15
    q[2, 0] = 0.0
    q[2, 0, 15] = 1.0
    # Request 2: no recorded proposal (sentinel) -> rejected; residual = p'.
    candidates[2, 1] = int(p[2 * n].argmax())
    seed = torch.arange(pool)
    offsets = torch.zeros(pool, dtype=torch.int64)
    predicts, accept_index, accept_length = _run_chain(
        p, q, candidates, req, seed, offsets, device
    )
    assert accept_length.tolist() == [n - 1, 0, 0]
    assert predicts[:n].tolist() == [1, 2, 3]
    assert int(predicts[n]) == int(p[n].argmax())
    assert int(predicts[2 * n]) == int(p[2 * n].argmax())
    assert accept_index[0].tolist() == [0, 1, 2]
    assert accept_index[1].tolist() == [n, -1, -1]


@pytest.mark.parametrize("sharp_q", [False, True])
def test_chain_rule_is_lossless(sharp_q: bool, device: str) -> None:
    """Drafts drawn from q: the first emitted token follows p' row 0 and,
    given an accepted first draft, the second follows row 1."""
    generator = torch.Generator().manual_seed(3)
    requests, n, vocab = 60_000, 3, 24
    p_rows = torch.softmax(torch.randn(n, vocab, generator=generator) * 1.5, -1)
    p_rows[0, :4] = 0.0  # p' cut some tokens q still proposes
    p_rows[0] /= p_rows[0].sum()
    scale = 4.0 if sharp_q else 1.0
    q_rows = torch.softmax(torch.randn(n - 1, vocab, generator=generator) * scale, -1)
    q = torch.zeros(requests + 1, n, vocab)
    q[:, : n - 1] = q_rows
    p = p_rows.repeat(requests, 1)
    candidates = torch.zeros(requests, n, dtype=torch.int32)
    for i in range(1, n):
        candidates[:, i] = torch.multinomial(
            q_rows[i - 1], requests, replacement=True, generator=generator
        ).to(torch.int32)
    req = torch.arange(1, requests + 1, dtype=torch.int32)
    seed = torch.randint(0, 2**40, (requests + 1,), generator=generator)
    offsets = torch.randint(0, 2**20, (requests + 1,), generator=generator)
    predicts, _, accept_length = _run_chain(
        p, q, candidates, req, seed, offsets, device
    )
    first = predicts.view(requests, n)[:, 0]
    _assert_follows(first, p_rows[0], "first token")
    accepted = accept_length >= 1
    _assert_follows(predicts.view(requests, n)[accepted, 1], p_rows[1], "second token")
    # The first draft is accepted with probability sum(min(p', q)).
    want = float(torch.minimum(p_rows[0], q_rows[0]).sum())
    got = float(accepted.double().mean())
    assert abs(got - want) < 4 * (want * (1 - want) / requests) ** 0.5 + 1e-3


# ---------------------------------------------------------------------------
# Tree coins.
# ---------------------------------------------------------------------------


def test_tree_coins_are_keyed_by_position(device: str) -> None:
    """Node ``i`` at depth ``d`` draws lane ``i`` of row ``offset + max(d - 1,
    0)``, the bonus lane ``N`` of row ``offset``: a request one position
    ahead sees the same coin one depth earlier."""
    bs, n = 2, 6
    seed = torch.tensor([0, 11, 11], dtype=torch.int64, device=device)
    offsets = torch.tensor([0, 40, 41], dtype=torch.int64, device=device)
    req = torch.tensor([1, 2], dtype=torch.int32, device=device)
    depths = torch.tensor(
        [[0, 1, 2, 3, 2, 1], [0, 0, 1, 2, 1, 0]], dtype=torch.int32, device=device
    )
    coins = torch.empty(bs, n, device=device)
    final = torch.empty(bs, device=device)
    tree_verify_coins(req, seed[:3], offsets, depths, coins, final)
    for b in range(bs):
        slot = int(req[b])
        rows = offsets[slot] + (depths[b].long() - 1).clamp_min(0)
        lanes = torch.arange(n, dtype=torch.int32, device=device)
        want = _coins(seed[slot].expand(n), rows, lanes)
        assert torch.equal(coins[b], want)
        want_final = _coins(
            seed[slot].view(1), offsets[slot].view(1),
            torch.full((1,), n, dtype=torch.int32, device=device),
        )  # fmt: skip
        assert torch.equal(final[b : b + 1], want_final)
    # Request 1 (offset 41) at depth d matches request 0 (offset 40) at d + 1.
    assert torch.equal(coins[1, 2], coins[0, 2])  # both row 41, lane 2
    assert ((coins >= 0) & (coins < 1)).all()


# ---------------------------------------------------------------------------
# Compile-cache keys and 64-bit offsets.
# ---------------------------------------------------------------------------


def test_kernels_do_not_recompile_across_batches(device: str) -> None:
    vocab, pool = 3000, 40
    generator = torch.Generator().manual_seed(4)
    temperature = torch.ones(pool, device=device)
    top_k = torch.full((pool,), 20, dtype=torch.int32, device=device)
    top_p = torch.full((pool,), 0.9, device=device)
    seed = torch.arange(pool, device=device)
    offsets = torch.zeros(pool, dtype=torch.int64, device=device)
    blocks = triton.cdiv(vocab, _BLOCK)
    q = torch.softmax(torch.randn(pool, 8, vocab, generator=generator), -1).to(device)

    def run(bs: int, n: int) -> None:
        rows = bs * n
        logits = torch.randn(rows, vocab, generator=generator).to(device)
        req = torch.arange(1, bs + 1, dtype=torch.int32, device=device)
        ids = torch.empty(rows, blocks * 128, dtype=torch.int32, device=device)
        values = torch.empty(rows, blocks * 128, device=device)
        probs = torch.empty(rows, vocab, device=device)
        for top_k_filter in (False, True):
            target_probs_from_pools(
                logits, req, temperature, top_k, top_p, ids, values, probs,
                top_k_filter=top_k_filter, block_size=_BLOCK, top_k_pad=128,
                num_tokens_per_req=n,
            )  # fmt: skip
        softmax(logits, temperature=torch.ones(rows, device=device))
        candidates = torch.zeros(bs, n, dtype=torch.int32, device=device)
        predicts = torch.empty(rows, dtype=torch.int32, device=device)
        index = torch.empty(bs, n, dtype=torch.int32, device=device)
        length = torch.empty(bs, dtype=torch.int32, device=device)
        chain_verify_draft_probs(
            probs, q[:, :n].contiguous(), candidates, req, seed, offsets,
            predicts, index, length, reject_threshold=_THRESHOLD,
        )  # fmt: skip
        depths = torch.zeros(bs, n, dtype=torch.int32, device=device)
        tree_verify_coins(
            req, seed, offsets, depths, torch.empty(bs, n, device=device),
            torch.empty(bs, device=device),
        )  # fmt: skip

    run(16, 8)
    with assert_no_triton_compile(
        triton_softmax_module._softmax_rows_kernel,
        dpv._target_probs_top_k_kernel,
        dpv._chain_verify_draft_probs_kernel,
        dpv._tree_verify_coins_kernel,
    ):
        for bs, n in ((1, 8), (3, 5), (17, 7), (32, 6)):
            run(bs, n)


def test_offsets_past_int32(device: str) -> None:
    """Rows of p' and pool slots of q whose element offsets pass 2**31."""
    vocab, n = 64, 2
    stride = 2**31 + 2 * vocab
    if torch.cuda.mem_get_info()[0] < (stride + 4 * vocab) * 4 + 2**30:
        pytest.skip("needs 9 GB of free device memory")
    storage = torch.zeros(stride + 4 * vocab, device=device)
    # p' rows 0 and 1 at elements 0 and stride; q pools 0 and 1 at vocab and stride + vocab.
    p = storage.as_strided((n, vocab), (stride, 1), 0)
    q = storage.as_strided((2, n, vocab), (stride, vocab, 1), vocab)
    p[0, 5] = 1.0
    q[1, 0, 5] = 1.0
    p[1, 9] = 1.0
    req = torch.tensor([1], dtype=torch.int32, device=device)
    seed = torch.zeros(2, dtype=torch.int64, device=device)
    offsets = torch.zeros(2, dtype=torch.int64, device=device)
    candidates = torch.tensor([[0, 5]], dtype=torch.int32, device=device)
    predicts = torch.empty(n, dtype=torch.int32, device=device)
    index = torch.empty(1, n, dtype=torch.int32, device=device)
    length = torch.empty(1, dtype=torch.int32, device=device)
    chain_verify_draft_probs(
        p, q, candidates, req, seed, offsets, predicts, index, length,
        reject_threshold=_THRESHOLD,
    )  # fmt: skip
    # Accepted against q at pool 1, the bonus drawn from p' row 1.
    assert predicts.tolist() == [5, 9] and length.tolist() == [1]

    # The softmax and the top-k p' write row 1 at element ``stride``.
    logits = torch.zeros(n, vocab, device=device)
    logits[1, 7] = 30.0
    temperature = torch.ones(2, device=device)
    top_k = torch.full((2,), 4, dtype=torch.int32, device=device)
    top_p = torch.ones(2, device=device)
    blocks = triton.cdiv(vocab, _BLOCK)
    ids = torch.empty(n, blocks * 64, dtype=torch.int32, device=device)
    values = torch.empty(n, blocks * 64, device=device)
    for top_k_filter in (False, True):
        p.fill_(-1.0)
        target_probs_from_pools(
            logits, req, temperature, top_k, top_p, ids, values, p,
            top_k_filter=top_k_filter, block_size=_BLOCK, top_k_pad=64,
            num_tokens_per_req=n,
        )  # fmt: skip
        assert float(p[1, 7]) > 0.99 and float(p[1].sum()) == pytest.approx(1.0)

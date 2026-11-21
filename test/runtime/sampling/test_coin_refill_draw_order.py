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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""The coin refill draws one uniform per request instead of two.

The per-slot CPU generators feed the verifier's rejection-sampling coins:
each request's row carries n draft coins plus one final coin, all drawn
from that request's own generator. The refill used to spend two ATen calls
per request (the row's coins, then its final coin), so a round paid 2*B
calls -- the dominant per-round cost of the sampling backends' prepare step.

The refill now draws the row's n+1 values in ONE call. This pins the
property that makes that safe: the single call consumes each generator in
the same order as the two-call form, so the coin values are bit-identical
and the sampled tokens do not move.

The tests below compare the two forms directly rather than re-deriving the
expected numbers, so they fail when the refill's draw order changes -- the
only way this optimization could alter sampled tokens.

Runs on CPU: the refill's arithmetic is generator state, which is
device-independent.
"""

import pytest
import torch

# The refill's floor comes from tokenspeed.runtime.sampling.utils, which
# imports the kernel package for its other helpers. The constant itself is
# a plain dtype table, so read it directly: this test pins generator draw
# order, which needs no kernel and no GPU.
_COIN_EPS = {
    torch.float32: torch.finfo(torch.float32).tiny,
    torch.bfloat16: torch.finfo(torch.bfloat16).tiny,
}


def coin_eps(dtype: torch.dtype) -> float:
    return _COIN_EPS[dtype]


def _two_call_refill(generators, bs, n, lo):
    """The refill as it stood before the single-call change: each request's
    row draws its n coins and then its final coin in two ATen calls."""
    coins = torch.empty((bs, n), dtype=torch.float32)
    final = torch.empty((bs,), dtype=torch.float32)
    for i, gen in enumerate(generators):
        coins[i, :n].uniform_(lo, 1.0, generator=gen)
        final[i].uniform_(lo, 1.0, generator=gen)
    return coins, final


def _single_call_refill(generators, bs, n, lo):
    """The refill as it stands now: one draw per request covering the row's
    coins and its final coin."""
    row_draw = torch.empty((bs, n + 1), dtype=torch.float32)
    for i, gen in enumerate(generators):
        row_draw[i].uniform_(lo, 1.0, generator=gen)
    coins = torch.empty((bs, n), dtype=torch.float32)
    final = torch.empty((bs,), dtype=torch.float32)
    coins.copy_(row_draw[:, :n])
    final.copy_(row_draw[:, n])
    return coins, final


def _generators(count, seed):
    gens = []
    for _ in range(count):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        gens.append(gen)
    return gens


class TestCoinRefillDrawOrder:
    """The single-call refill must consume each generator exactly as the
    two-call form did."""

    @pytest.mark.parametrize("n", [1, 2, 4, 8])
    @pytest.mark.parametrize("bs", [1, 4, 16])
    def test_single_call_matches_two_call_bitwise(self, bs, n):
        """Same generators, same seed -> bit-identical coins and finals.

        A change here means the sampled tokens can move: the coins decide
        which draft the verifier accepts.
        """
        lo = coin_eps(torch.float32)
        expected_coins, expected_final = _two_call_refill(
            _generators(bs, seed=1234), bs, n, lo
        )
        coins, final = _single_call_refill(_generators(bs, seed=1234), bs, n, lo)

        assert torch.equal(coins, expected_coins)
        assert torch.equal(final, expected_final)

    def test_two_call_form_is_the_frozen_reference(self):
        """The reference form itself is pinned to fixed bytes.

        The bitwise test above compares two live implementations, so a
        change that moved BOTH would slip through. These bytes come from
        the two-call form as it stood before the single-call change, so
        any drift in either implementation -- including a seed or draw
        order change -- fails here.
        """
        lo = coin_eps(torch.float32)
        coins, final = _two_call_refill(_generators(2, seed=20240917), 2, 2, lo)

        assert coins[0, 0].item() == pytest.approx(0.28558266162872314, abs=1e-7)
        assert coins[0, 1].item() == pytest.approx(0.6334905028343201, abs=1e-7)
        assert coins[1, 0].item() == pytest.approx(0.28558266162872314, abs=1e-7)
        assert coins[1, 1].item() == pytest.approx(0.6334905028343201, abs=1e-7)
        assert final[0].item() == pytest.approx(0.43871617317199707, abs=1e-7)
        assert final[1].item() == pytest.approx(0.43871617317199707, abs=1e-7)

    def test_single_call_form_matches_the_frozen_reference(self):
        """The single-call form must land on the same frozen bytes."""
        lo = coin_eps(torch.float32)
        coins, final = _single_call_refill(_generators(2, seed=20240917), 2, 2, lo)

        assert coins[0, 0].item() == pytest.approx(0.28558266162872314, abs=1e-7)
        assert coins[0, 1].item() == pytest.approx(0.6334905028343201, abs=1e-7)
        assert coins[1, 0].item() == pytest.approx(0.28558266162872314, abs=1e-7)
        assert coins[1, 1].item() == pytest.approx(0.6334905028343201, abs=1e-7)
        assert final[0].item() == pytest.approx(0.43871617317199707, abs=1e-7)
        assert final[1].item() == pytest.approx(0.43871617317199707, abs=1e-7)

    def test_each_request_keeps_its_own_generator_stream(self):
        """Two requests must not share a stream: a request's coins come
        from its own generator, so a request's draws do not depend on the
        other rows in the batch."""
        lo = coin_eps(torch.float32)
        bs, n = 2, 1

        together = _single_call_refill(_generators(bs, seed=99), bs, n, lo)

        # The same first request, drawn alone, must match its row from the
        # two-request batch.
        alone = _single_call_refill(_generators(1, seed=99), 1, n, lo)

        assert torch.equal(together[0][0], alone[0][0])
        assert torch.equal(together[1][0], alone[1][0])

    def test_generators_are_advanced_by_exactly_n_plus_one_draws(self):
        """A refill must consume exactly the row's n+1 values, so the next
        round's coins continue the stream where this round stopped."""
        lo = coin_eps(torch.float32)
        bs, n = 1, 3
        gen = torch.Generator(device="cpu")
        gen.manual_seed(5)

        _single_call_refill([gen], bs, n, lo)

        expected = torch.empty(n + 1, dtype=torch.float32)
        expected.uniform_(lo, 1.0, generator=gen)

        actual = torch.empty(n + 1, dtype=torch.float32)
        actual.uniform_(lo, 1.0, generator=gen)

        # The generator now sits n+1 draws ahead of where it started, so
        # the next draw continues rather than repeating.
        assert not torch.equal(expected, actual)
        repeated = torch.empty(n + 1, dtype=torch.float32)
        repeated.uniform_(lo, 1.0, generator=gen)
        assert not torch.equal(repeated, actual)

    def test_coins_stay_inside_the_coin_epsilon_floor(self):
        """The drawn values must respect the same floor the kernels read
        (a coin of exactly 0 would make log() diverge)."""
        lo = coin_eps(torch.float32)
        bs, n = 8, 2
        coins, final = _single_call_refill(_generators(bs, seed=3), bs, n, lo)

        assert bool((coins >= lo).all())
        assert bool((coins < 1.0).all())
        assert bool((final >= lo).all())
        assert bool((final < 1.0).all())

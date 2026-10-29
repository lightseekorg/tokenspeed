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

"""Sampled DFlash2 drafts and their rejection-sampling verify.

CPU tests enumerate the torch reference exactly in fp64; GPU tests check the
Triton kernels against the reference given the kernels' own noise, against
the plain (target-only) path where no randomness is involved, and for
losslessness with a chi-squared test over many seeds.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.numerics.torch.sampling import (
    torch_dflash2_sample_path,
    torch_verify_rows,
)
from tokenspeed_kernel.ops.sampling.triton import (
    COIN_OFFSET_SALT,
    DRAFT_SAMPLE_OFFSET_SALT,
    dflash2_sample_path,
    gumbel_sample_from_pools,
    gumbel_sample_top_k_top_p_from_pools,
    gumbel_scratch_shape,
    verify_chain_row_decisions,
    verify_chain_target_sampled,
    verify_rows_no_filter_from_pools,
    verify_rows_no_filter_scratch_shape,
    verify_rows_top_k_top_p_from_pools,
)

# Sentinel matching tokenspeed.runtime.sampling.sampling_params._TOP_K_DISABLED.
_TOP_K_DISABLED = 1 << 30
_TOP_K_BLOCK = 1024
_TOP_K_PAD = 128
_CHI2_MIN_P_VALUE = 1.0e-3


@triton.jit
def _noise_kernel(
    seed_ptr,
    offset_ptr,
    ids_ptr,
    gumbel_ptr,
    coin_ptr,
    num_ids,
    COIN_SALT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """The documented noise streams: Gumbel at ``ids`` and the accept coin."""
    row = tl.program_id(0)
    seed = tl.load(seed_ptr + row).to(tl.int64)
    offset = tl.load(offset_ptr + row).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < num_ids
    ids = tl.load(ids_ptr + row * num_ids + cols, mask=mask, other=0)
    uniform = tl.maximum(tl.rand(tl.randint(seed, offset), ids), 1.0e-7)
    tl.store(gumbel_ptr + row * num_ids + cols, -tl.log(-tl.log(uniform)), mask=mask)
    coin = tl.rand(tl.randint(seed, COIN_SALT + offset), tl.full((), 0, tl.int32))
    tl.store(coin_ptr + row, coin)


def _noise(
    seeds: torch.Tensor, offsets: torch.Tensor, ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(gumbel [rows, M], coin [rows])`` of each row's stream at ``ids``."""
    rows, num_ids = ids.shape
    gumbel = torch.empty(rows, num_ids, dtype=torch.float32, device=ids.device)
    coin = torch.empty(rows, dtype=torch.float32, device=ids.device)
    _noise_kernel[(rows,)](
        seeds.to(torch.int64).contiguous(),
        offsets.to(torch.int64).contiguous(),
        ids.to(torch.int32).contiguous(),
        gumbel,
        coin,
        num_ids,
        COIN_SALT=COIN_OFFSET_SALT,
        BLOCK=triton.next_power_of_2(num_ids),
    )
    return gumbel, coin


# ---------------------------------------------------------------------------
# CPU: exact enumeration of the reference rule in fp64.
# ---------------------------------------------------------------------------


def _target_probs(
    logits: torch.Tensor, temperature: float, top_k: int, top_p: float
) -> torch.Tensor:
    """The route's target distribution, from its definition."""
    scaled = logits / temperature
    if top_k == _TOP_K_DISABLED:
        return torch.softmax(scaled, dim=0)
    order = torch.sort(logits, descending=True, stable=True).indices[:top_k]
    probs = torch.softmax(scaled[order], dim=0)
    kept = order[torch.cumsum(probs, 0) - probs < top_p]
    out = torch.zeros_like(scaled)
    out[kept] = torch.softmax(scaled[kept], dim=0)
    return out


def _emitted_distribution(logits, temperature, top_k, top_p, q, q_ids) -> torch.Tensor:
    """Exact distribution of the token a row emits when its draft ~ q.

    Acceptance of each candidate is the measure of the coins the reference
    accepts (found by bisection); the residual draw's distribution is the
    softmax of the reference's own residual scores (Gumbel-max).
    """
    vocab = logits.numel()
    k = q.numel()

    def run(lane: int, coin: float):
        return torch_verify_rows(
            logits[None],
            torch.tensor([temperature], dtype=torch.float64),
            torch.tensor([top_k]),
            torch.tensor([top_p], dtype=torch.float64),
            q[None],
            q_ids[None],
            torch.tensor([True]),
            torch.tensor([lane]),
            torch.tensor([int(q_ids[lane])]),
            torch.tensor([coin], dtype=torch.float64),
            torch.zeros(1, vocab, dtype=torch.float64),
            top_k_pad=None,
        )

    emitted = torch.zeros(vocab, dtype=torch.float64)
    for lane in range(k):
        weight = float(q[lane])
        if weight == 0.0:
            continue
        lo, hi = 0.0, 1.0
        if bool(run(lane, hi)[0][0]):
            lo = 1.0
        elif not bool(run(lane, lo)[0][0]):
            hi = 0.0
        else:
            for _ in range(64):
                mid = (lo + hi) / 2
                if bool(run(lane, mid)[0][0]):
                    lo = mid
                else:
                    hi = mid
        accept = lo
        emitted[int(q_ids[lane])] += weight * accept
        if accept < 1.0:
            _, token, residual = run(lane, 1.0)
            residual = residual[0]
            if torch.isinf(residual).all():
                # No residual mass: the reference emits the target's mode.
                emitted[int(token[0])] += weight * (1.0 - accept)
            else:
                emitted += weight * (1.0 - accept) * torch.softmax(residual, dim=0)
    return emitted


@pytest.mark.parametrize(
    "case", ["no_filter", "top_k", "top_k_top_p", "q_equals_p", "no_overlap"]
)
def test_reference_rule_emits_the_target_distribution(case: str) -> None:
    generator = torch.Generator().manual_seed(7)
    vocab, num_candidates, temperature = 12, 5, 0.7
    logits = torch.randn(vocab, generator=generator, dtype=torch.float64) * 1.5
    top_k, top_p = {
        "no_filter": (_TOP_K_DISABLED, 1.0),
        "top_k": (6, 1.0),
        "top_k_top_p": (8, 0.6),
        "q_equals_p": (4, 1.0),
        "no_overlap": (3, 1.0),
    }[case]
    target = _target_probs(logits, temperature, top_k, top_p)
    order = torch.sort(logits, descending=True, stable=True).indices
    if case == "q_equals_p":
        # q is exactly p on the kept tokens, plus a zero-probability lane.
        kept = target.nonzero().flatten()
        q_ids = torch.cat([kept, order[-1:]])
        q = torch.cat([target[kept], torch.zeros(1, dtype=torch.float64)])
    elif case == "no_overlap":
        # Every candidate is outside the kept set: always rejected.
        q_ids = order[-num_candidates:]
        q = torch.rand(num_candidates, generator=generator, dtype=torch.float64)
        q /= q.sum()
    else:
        # Candidates straddle the kept set: some in, some out.
        q_ids = torch.cat([order[1:4], order[-2:]])
        q = torch.rand(num_candidates, generator=generator, dtype=torch.float64)
        q /= q.sum()
    emitted = _emitted_distribution(logits, temperature, top_k, top_p, q, q_ids)
    torch.testing.assert_close(emitted, target, atol=1e-12, rtol=0)


# ---------------------------------------------------------------------------
# GPU: the DFlash2 walk.
# ---------------------------------------------------------------------------


def _lattice(batch: int, steps: int, top_k: int, device: str, seed: int = 0):
    """A random block-drafter lattice."""
    vocab = 4 * top_k + 7
    generator = torch.Generator(device="cpu").manual_seed(seed)
    candidate_ids = (
        torch.rand(batch, steps, vocab, generator=generator).topk(top_k).indices
    ).to(device=device, dtype=torch.int32)
    scores = torch.randn(batch, steps, top_k, top_k, generator=generator).to(device)
    # Step 0 has a single predecessor, so the checkpoint emits equal rows.
    scores[:, 0] = scores[:, 0, :1].expand(-1, top_k, -1)
    anchors = torch.randint(vocab, (batch,), generator=generator).to(device)
    return candidate_ids, scores, anchors


class _WalkPools:
    """Pool-indexed sampling parameters of a batch, in a shuffled pool order."""

    def __init__(self, temperature, top_k, top_p, device: str, seed: int = 0):
        bs = len(temperature)
        pool_rows = bs + 3
        generator = torch.Generator().manual_seed(seed)
        self.req_pool_indices = (
            torch.randperm(pool_rows - 1, generator=generator)[:bs] + 1
        ).to(device=device, dtype=torch.int32)
        index = self.req_pool_indices.long()
        self.temperature = torch.ones(pool_rows, device=device)
        self.temperature[index] = torch.tensor(temperature, device=device)
        self.top_k = torch.ones(pool_rows, dtype=torch.int32, device=device)
        self.top_k[index] = torch.tensor(top_k, dtype=torch.int32, device=device)
        self.top_p = torch.ones(pool_rows, device=device)
        self.top_p[index] = torch.tensor(top_p, device=device)
        self.seed = torch.arange(pool_rows, dtype=torch.int64, device=device) * 7919
        self.seed += 11
        self.draft_offsets = (
            torch.arange(bs, dtype=torch.int64, device=device) * 8 + 1000
        )


def _walk(candidate_ids, scores, anchors, pools: _WalkPools):
    """Run the walk; returns ``(tokens, q, lanes [bs, L], has_proposal [bs])``
    after checking its chain tables against the draft-q layout."""
    bs, steps, top_k = candidate_ids.shape
    device = candidate_ids.device
    tokens = torch.full((bs, steps + 1), -5, dtype=torch.int32, device=device)
    q = torch.full((bs, steps, top_k), -1.0, device=device)
    q_ids = torch.full((bs, steps, top_k), -5, dtype=torch.int32, device=device)
    pos = torch.full((bs, steps + 1), -5, dtype=torch.int32, device=device)
    slot = torch.full((bs, steps + 1), -5, dtype=torch.int32, device=device)
    dflash2_sample_path(
        candidate_ids,
        scores,
        anchors,
        pools.req_pool_indices,
        pools.temperature,
        pools.top_k,
        pools.top_p,
        pools.seed,
        pools.draft_offsets,
        tokens,
        q,
        q_ids,
        pos,
        slot,
    )
    # Node 0 is the anchor; node l + 1 reads q row l, or every node is -1.
    has_proposal = (slot[:, 1] >= 0).to(torch.int32)
    chain = torch.arange(-1, steps, dtype=torch.int32, device=device)
    expected_slot = torch.where(has_proposal[:, None] != 0, chain[None, :], -1)
    assert torch.equal(slot, expected_slot.to(torch.int32))
    assert torch.equal(pos[:, 0], torch.zeros_like(pos[:, 0]))
    assert torch.equal(q_ids, candidate_ids.to(torch.int32))
    return tokens, q, pos[:, 1:], has_proposal


def _walk_reference(candidate_ids, scores, anchors, pools: _WalkPools):
    """The torch reference fed the documented draft noise stream."""
    bs, steps, top_k = candidate_ids.shape
    index = pools.req_pool_indices.long()
    step = torch.arange(steps, device=scores.device)
    offsets = DRAFT_SAMPLE_OFFSET_SALT + pools.draft_offsets[:, None] + step[None, :]
    seeds = pools.seed[index][:, None].expand(bs, steps)
    gumbel, _ = _noise(
        seeds.reshape(-1), offsets.reshape(-1), candidate_ids.reshape(bs * steps, -1)
    )
    return torch_dflash2_sample_path(
        candidate_ids,
        scores,
        anchors,
        pools.temperature[index],
        pools.top_k[index],
        pools.top_p[index],
        gumbel.view(bs, steps, top_k),
    )


def _old_greedy_walk(candidate_ids, scores, anchors) -> list[list[int]]:
    """The walk before sampling: the best edge out of the previous pick."""
    out = []
    for request in range(candidate_ids.shape[0]):
        previous, row = 0, [int(anchors[request])]
        for step in range(candidate_ids.shape[1]):
            previous = int(torch.argmax(scores[request, step, previous]))
            row.append(int(candidate_ids[request, step, previous]))
        out.append(row)
    return out


# Greedy; top-k; no filter; top-p without top-k (falls back); top-k and top-p.
_MIXED_WALK = dict(
    temperature=[1.0, 0.8, 1.3, 0.9, 0.5, 1.0],
    top_k=[1, 64, _TOP_K_DISABLED, _TOP_K_DISABLED, 20, 1],
    top_p=[1.0, 1.0, 1.0, 0.9, 0.9, 1.0],
)


@pytest.mark.parametrize(("steps", "top_k"), ((1, 4), (7, 16), (5, 6)))
def test_dflash2_sample_path_matches_the_reference(
    steps: int, top_k: int, device: str
) -> None:
    candidate_ids, scores, anchors = _lattice(6, steps, top_k, device, seed=steps)
    pools = _WalkPools(**_MIXED_WALK, device=device)

    tokens, q, lanes, has_proposal = _walk(candidate_ids, scores, anchors, pools)
    ref_tokens, ref_q, ref_lanes, ref_has = _walk_reference(
        candidate_ids, scores, anchors, pools
    )

    assert tokens.tolist() == ref_tokens.tolist()
    assert lanes.tolist() == ref_lanes.tolist()
    assert has_proposal.tolist() == [0, 1, 1, 0, 1, 0]
    assert has_proposal.tolist() == ref_has.tolist()
    torch.testing.assert_close(q, ref_q, rtol=1e-6, atol=1e-7)


def test_dflash2_sample_path_greedy_rows_match_the_old_walk(device: str) -> None:
    candidate_ids, scores, anchors = _lattice(5, 7, 16, device, seed=3)
    # Greedy by top-k 1, and the top-p-only fallback.
    pools = _WalkPools(
        temperature=[1.0, 1.0, 0.7, 1.0, 1.0],
        top_k=[1, 1, _TOP_K_DISABLED, 1, _TOP_K_DISABLED],
        top_p=[1.0, 1.0, 0.95, 1.0, 0.5],
        device=device,
    )

    tokens, q, lanes, has_proposal = _walk(candidate_ids, scores, anchors, pools)

    assert tokens.tolist() == _old_greedy_walk(candidate_ids, scores, anchors)
    assert has_proposal.tolist() == [0] * 5
    expected_q = torch.nn.functional.one_hot(lanes.long(), 16).float()
    assert torch.equal(q, expected_q)


def test_dflash2_sample_path_follows_the_selected_predecessor(device: str) -> None:
    """Step 1's row is chosen by step 0's pick, not by its own best edge."""
    candidate_ids = torch.tensor([[[10, 11], [20, 21]]], device=device)
    scores = torch.zeros(1, 2, 2, 2, device=device)
    scores[0, 0, :, 1] = 3.0
    scores[0, 1, 0, 1] = 9.0
    scores[0, 1, 1, 0] = 4.0
    anchors = torch.tensor([7], device=device)
    pools = _WalkPools([1.0], [1], [1.0], device=device)

    tokens, _, lanes, _ = _walk(candidate_ids, scores, anchors, pools)

    assert tokens[0].tolist() == [7, 11, 20]
    assert lanes[0].tolist() == [1, 0]


def test_dflash2_sample_path_ignores_the_lane_order(device: str) -> None:
    """Noise is keyed by token id: permuting the lanes keeps every token."""
    steps, top_k = 7, 16
    candidate_ids, scores, anchors = _lattice(6, steps, top_k, device, seed=5)
    pools = _WalkPools(**_MIXED_WALK, device=device)
    generator = torch.Generator().manual_seed(9)
    perms = [
        torch.randperm(top_k, generator=generator).to(device) for _ in range(steps)
    ]
    permuted_ids = torch.stack(
        [candidate_ids[:, step, perms[step]] for step in range(steps)], dim=1
    )
    # Permuted lane j of step l is original lane perms[l][j]; the score rows
    # follow step l - 1's permutation (step 0's rows are all equal).
    permuted_scores = torch.stack(
        [
            scores[:, step][:, perms[step - 1] if step else perms[0]][:, :, perms[step]]
            for step in range(steps)
        ],
        dim=1,
    )

    tokens, q, lanes, has_proposal = _walk(candidate_ids, scores, anchors, pools)
    p_tokens, p_q, p_lanes, p_has = _walk(permuted_ids, permuted_scores, anchors, pools)

    assert p_tokens.tolist() == tokens.tolist()
    assert p_has.tolist() == has_proposal.tolist()
    for step in range(steps):
        # Permuted lane j is original lane perms[step][j].
        assert torch.equal(perms[step][p_lanes[:, step].long()], lanes[:, step].long())
        torch.testing.assert_close(p_q[:, step], q[:, step][:, perms[step]])


def test_dflash2_sample_path_non_finite_rows_propose_nothing(device: str) -> None:
    candidate_ids, scores, anchors = _lattice(3, 4, 8, device, seed=11)
    scores[0, 2, :, :] = float("-inf")
    scores[1, 1, :, 3] = float("inf")
    scores[2, 3, :, 5] = float("nan")
    pools = _WalkPools([1.0] * 3, [_TOP_K_DISABLED] * 3, [1.0] * 3, device=device)

    tokens, q, lanes, has_proposal = _walk(candidate_ids, scores, anchors, pools)

    assert has_proposal.tolist() == [0, 0, 0]
    assert lanes.min() >= 0 and lanes.max() < 8
    gathered = torch.gather(candidate_ids, 2, lanes.long()[:, :, None])[:, :, 0]
    assert torch.equal(tokens[:, 1:], gathered)
    # The non-finite steps fall back to the best lane, one-hot q.
    assert int(lanes[0, 2]) == 0 and float(q[0, 2, 0]) == 1.0
    assert int(lanes[1, 1]) == 3 and float(q[1, 1, 3]) == 1.0
    assert float(q[2, 3].sum()) == 1.0


# ---------------------------------------------------------------------------
# GPU: per-row verify and the chain pick.
# ---------------------------------------------------------------------------


class _VerifyBatch:
    """A verify window, its pool-indexed sampling parameters and draft q.

    The q tables follow the chain layout of ``DraftTreeQ``: node 0 is the
    anchor, node ``i`` reads q row ``i - 1`` at lane ``pos``; a request
    without q has slot ``-1`` on every node.
    """

    def __init__(
        self,
        logits: torch.Tensor,
        temperature: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
        q: torch.Tensor,
        q_ids: torch.Tensor,
        has_q: torch.Tensor,
        lanes: torch.Tensor,
        *,
        drafts: torch.Tensor | None = None,
        seed: int = 0,
    ) -> None:
        device = logits.device
        bs, steps, num_candidates = q.shape
        self.bs, self.n = bs, steps + 1
        self.logits = logits
        pool_rows = bs + 2
        generator = torch.Generator().manual_seed(seed)
        self.req_pool_indices = (
            torch.randperm(pool_rows - 1, generator=generator)[:bs] + 1
        ).to(device=device, dtype=torch.int32)
        index = self.req_pool_indices.long()

        def pool(values, dtype, fill):
            out = torch.full((pool_rows,), fill, dtype=dtype, device=device)
            out[index] = values.to(device=device, dtype=dtype)
            return out

        self.temperature = pool(temperature, torch.float32, 1.0)
        self.top_k = pool(top_k, torch.int32, 1)
        self.top_p = pool(top_p, torch.float32, 1.0)
        self.seed = torch.arange(pool_rows, dtype=torch.int64, device=device) * 7919
        self.seed += 13
        self.offsets = torch.arange(pool_rows, dtype=torch.int64, device=device) * 5
        self.offsets += 300
        self.q = torch.zeros(pool_rows, steps, num_candidates, device=device)
        self.q[index] = q.to(device=device, dtype=torch.float32)
        self.q_ids = torch.zeros(
            pool_rows, steps, num_candidates, dtype=torch.int32, device=device
        )
        self.q_ids[index] = q_ids.to(device=device, dtype=torch.int32)
        chain = torch.arange(-1, steps, dtype=torch.int32, device=device)
        self.slot = torch.full(
            (pool_rows, self.n), -1, dtype=torch.int32, device=device
        )
        self.slot[index] = torch.where(
            has_q.to(device)[:, None], chain[None, :], -1
        ).to(torch.int32)
        self.pos = torch.zeros(pool_rows, self.n, dtype=torch.int32, device=device)
        self.pos[index, 1:] = lanes.to(device=device, dtype=torch.int32)
        if drafts is None:
            drafts = torch.gather(
                q_ids.to(device), 2, lanes.to(device).long()[..., None]
            )
            drafts = drafts[..., 0]
        self.candidates = torch.zeros(bs, self.n, dtype=torch.int32, device=device)
        self.candidates[:, 1:] = drafts.to(device=device, dtype=torch.int32)

    @property
    def rows(self) -> int:
        return self.bs * self.n

    def verify_rows(self, route: str) -> tuple[torch.Tensor, torch.Tensor]:
        rows, vocab = self.logits.shape
        device = self.logits.device
        row_accept = torch.full((rows,), -7, dtype=torch.int32, device=device)
        row_token = torch.full((rows,), -7, dtype=torch.int32, device=device)
        draft_q = (self.candidates, self.q, self.q_ids, self.slot, self.pos)
        if route == "no_filter":
            scores_shape, ids_shape = verify_rows_no_filter_scratch_shape(rows, vocab)
            verify_rows_no_filter_from_pools(
                self.logits,
                self.req_pool_indices,
                self.temperature,
                self.seed,
                self.offsets,
                *draft_q,
                torch.empty(scores_shape, dtype=torch.float32, device=device),
                torch.empty(ids_shape, dtype=torch.int32, device=device),
                row_accept,
                row_token,
                num_tokens_per_req=self.n,
            )
        else:
            verify_rows_top_k_top_p_from_pools(
                self.logits,
                self.req_pool_indices,
                self.temperature,
                self.top_k,
                self.top_p,
                self.seed,
                self.offsets,
                *_top_k_scratch(rows, vocab, device),
                *draft_q,
                row_accept,
                row_token,
                block_size=_TOP_K_BLOCK,
                top_k_pad=_TOP_K_PAD,
                num_tokens_per_req=self.n,
            )
        return row_accept, row_token

    def plain_targets(self, route: str) -> torch.Tensor:
        """Today's target draws: the sampler the flag-off verify runs."""
        rows, vocab = self.logits.shape
        device = self.logits.device
        out = torch.empty(rows, dtype=torch.int32, device=device)
        if route == "no_filter":
            shape = gumbel_scratch_shape(rows, vocab)
            return gumbel_sample_from_pools(
                self.logits,
                self.req_pool_indices,
                self.temperature,
                self.seed,
                self.offsets,
                torch.empty(shape, dtype=torch.int32, device=device),
                torch.empty(shape, dtype=torch.float32, device=device),
                out,
                num_tokens_per_req=self.n,
            )
        return gumbel_sample_top_k_top_p_from_pools(
            self.logits,
            self.req_pool_indices,
            self.temperature,
            self.top_k,
            self.top_p,
            self.seed,
            self.offsets,
            *_top_k_scratch(rows, vocab, device),
            out,
            block_size=_TOP_K_BLOCK,
            top_k_pad=_TOP_K_PAD,
            num_tokens_per_req=self.n,
        )

    def _chain_outputs(self):
        device = self.logits.device
        return (
            torch.full((self.rows,), 777, dtype=torch.int32, device=device),
            torch.full((self.bs, self.n), -1, dtype=torch.int32, device=device),
            torch.full((self.bs,), 99, dtype=torch.int32, device=device),
        )

    def chain(self, row_accept, row_token):
        predicts, accept_index, accept_length = self._chain_outputs()
        verify_chain_row_decisions(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_length,
            candidates=self.candidates,
            row_accept=row_accept,
            row_token=row_token,
        )
        return predicts, accept_index, accept_length

    def old_chain(self, target):
        predicts, accept_index, accept_length = self._chain_outputs()
        verify_chain_target_sampled(
            predicts,
            accept_index,
            accept_length,
            self.candidates,
            target,
            write_rejected_targets=False,
        )
        return predicts, accept_index, accept_length

    def reference(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``torch_verify_rows`` per row, fed the documented noise streams."""
        rows, vocab = self.logits.shape
        device = self.logits.device
        request = torch.arange(rows, device=device) // self.n
        spec_pos = torch.arange(rows, device=device) % self.n
        pool = self.req_pool_indices.long()[request]
        draft_row = spec_pos < self.n - 1
        node = (spec_pos + 1).clamp(max=self.n - 1)
        q_row = spec_pos.clamp(max=self.n - 2)
        gumbels, coins = _noise(
            self.seed[pool],
            self.offsets[pool] + spec_pos,
            torch.arange(vocab, device=device).expand(rows, vocab),
        )
        accept, token, _ = torch_verify_rows(
            self.logits,
            self.temperature[pool],
            self.top_k[pool],
            self.top_p[pool],
            self.q[pool, q_row],
            self.q_ids[pool, q_row],
            draft_row & (self.slot[pool, node] >= 0),
            self.pos[pool, node],
            torch.where(draft_row, self.candidates[request, node], -1),
            coins,
            gumbels,
            top_k_pad=_TOP_K_PAD,
        )
        return accept.to(device), token.to(device)


def _top_k_scratch(rows: int, vocab: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    width = triton.cdiv(vocab, _TOP_K_BLOCK) * _TOP_K_PAD
    return (
        torch.empty(rows, width, dtype=torch.int32, device=device),
        torch.empty(rows, width, dtype=torch.float32, device=device),
    )


def _random_batch(
    route: str,
    bs: int,
    n: int,
    vocab: int,
    device: str,
    *,
    seed: int,
    has_q: torch.Tensor | None = None,
) -> _VerifyBatch:
    """Random logits; each row's q over 16 candidates near its top tokens."""
    num_candidates = 16
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(bs * n, vocab, generator=generator) * 2.0
    if route == "no_filter":
        top_k = torch.full((bs,), _TOP_K_DISABLED)
        top_p = torch.ones(bs)
    else:
        top_k = torch.tensor([1, 5, 40, 64, 200, 20, 8, 3])[torch.arange(bs) % 8]
        top_p = torch.tensor([1.0, 0.8, 1.0, 0.5, 1.0, 0.95, 0.3, 1.0])[
            torch.arange(bs) % 8
        ]
    temperature = 0.6 + 0.8 * torch.rand(bs, generator=generator)
    draft_logits = logits.view(bs, n, vocab)[:, :-1]
    draft_logits = draft_logits + torch.randn(draft_logits.shape, generator=generator)
    q_ids = draft_logits.topk(num_candidates, dim=-1).indices
    q = torch.softmax(torch.randn(bs, n - 1, num_candidates, generator=generator), -1)
    lanes = torch.multinomial(q.view(-1, num_candidates), 1, generator=generator)
    lanes = lanes.view(bs, n - 1)
    if has_q is None:
        has_q = torch.arange(bs) % 4 != 3
    return _VerifyBatch(
        logits.to(device),
        temperature,
        top_k,
        top_p,
        q,
        q_ids,
        has_q,
        lanes,
        seed=seed,
    )


@pytest.mark.parametrize("route", ["no_filter", "top_k"])
def test_verify_rows_match_the_reference_given_the_kernel_noise(
    route: str, device: str
) -> None:
    batch = _random_batch(route, 16, 4, 2500, device, seed=21)

    row_accept, row_token = batch.verify_rows(route)
    ref_accept, ref_token = batch.reference()

    assert row_accept.tolist() == ref_accept.tolist()
    assert row_token.tolist() == ref_token.tolist()
    # The batch exercises both outcomes.
    assert 0 < int(row_accept.sum()) < batch.bs * (batch.n - 1)


@pytest.mark.parametrize("route", ["no_filter", "top_k"])
@pytest.mark.parametrize("mismatched_lane", [False, True])
def test_rows_without_q_match_the_plain_path_bitwise(
    route: str, mismatched_lane: bool, device: str
) -> None:
    """Rows verified by the target-only rule: no slot, a greedy row, or a
    recorded lane that does not hold the draft."""
    bs, n = 16, 4
    has_q = torch.full((bs,), mismatched_lane)
    if route == "top_k":
        # Greedy rows (top-k 1) ignore a recorded q.
        has_q[torch.arange(bs) % 8 == 0] = True
    batch = _random_batch(route, bs, n, 2500, device, seed=5, has_q=has_q)
    target = batch.plain_targets(route).clone()
    # Request b's drafts follow the target for its first b % n rows, then miss.
    vocab = batch.logits.shape[1]
    for b in range(bs):
        pool = int(batch.req_pool_indices[b])
        for r in range(n - 1):
            hit = int(target[b * n + r])
            draft = hit if r < b % n else (hit + 1) % vocab
            batch.candidates[b, r + 1] = draft
            if mismatched_lane:
                # Point the row at a lane holding some other token.
                other = (batch.q_ids[pool, r] != draft).nonzero()[0, 0]
                batch.pos[pool, r + 1] = other

    row_accept, row_token = batch.verify_rows(route)
    new = batch.chain(row_accept, row_token)
    old = batch.old_chain(target)

    assert torch.equal(row_token, target)
    draft_row = (torch.arange(batch.rows, device=device) % n) < n - 1
    no_draft = torch.full((bs, 1), -1, dtype=torch.int32, device=device)
    drafts = torch.cat([batch.candidates[:, 1:], no_draft], dim=1).reshape(-1)
    assert torch.equal(row_accept != 0, draft_row & (target == drafts))
    for got, want in zip(new, old):
        assert torch.equal(got, want)
    assert sorted(set(new[2].tolist())) == list(range(n))


def test_one_hot_q_outcomes_without_randomness(device: str) -> None:
    """Decisions a one-hot q makes regardless of the noise."""
    bs, n, vocab = 4, 2, 2500
    generator = torch.Generator().manual_seed(17)
    logits = torch.randn(bs * n, vocab, generator=generator)
    # No-filter rows: request 2 drafts a -inf token, request 3 a token holding
    # all the mass.
    logits[2 * n, 7] = float("-inf")
    logits[3 * n] = -1.0e4
    logits[3 * n, 9] = 1.0e4
    order = logits[0::n].argsort(dim=1, descending=True)
    # One-hot q at lane 0; the other lanes hold each row's least likely tokens.
    q_ids = torch.cat([order[:, :1], order[:, -15:]], dim=1)[:, None, :]
    q = torch.zeros(bs, 1, 16)
    q[..., 0] = 1.0
    lanes = torch.zeros(bs, 1, dtype=torch.int64)
    # Top-k rows: request 0 drafts a token outside its top-5 (never kept),
    # request 1 the mode with top-p so small only the mode is kept.
    q_ids[0, 0, 0] = order[0, 50]
    q_ids[2, 0, 0] = 7
    top_k = torch.tensor([5, 64, _TOP_K_DISABLED, _TOP_K_DISABLED])
    top_p = torch.tensor([1.0, 1.0e-6, 1.0, 1.0])
    temperature = torch.tensor([0.8, 1.1, 0.9, 1.0])
    has_q = torch.ones(bs, dtype=torch.bool)
    args = (temperature, top_k, top_p, q, q_ids, has_q, lanes)

    top_k_batch = _VerifyBatch(logits[: 2 * n].to(device), *[a[:2] for a in args])
    accept, token = top_k_batch.verify_rows("top_k")
    target = top_k_batch.plain_targets("top_k")
    assert accept.tolist() == [0, 0, 1, 0]
    assert int(token[0]) == int(target[0])

    no_filter_batch = _VerifyBatch(logits[2 * n :].to(device), *[a[2:] for a in args])
    accept, token = no_filter_batch.verify_rows("no_filter")
    target = no_filter_batch.plain_targets("no_filter")
    assert accept.tolist() == [0, 0, 1, 0]
    assert int(token[0]) == int(target[0])


@pytest.mark.parametrize("route", ["no_filter", "top_k"])
def test_empty_residual_falls_back_to_the_mode(route: str, device: str) -> None:
    """q above p on every token leaves no residual: a rejection emits the mode."""
    bs, n, vocab = 256, 2, 2500
    logits = torch.full((bs * n, vocab), -5.0)
    kept = torch.tensor([40, 7, 1300])
    logits[:, kept] = torch.tensor([2.0, 1.5, 1.0])
    if route == "no_filter":
        # Every other token has p = 0.
        logits[:, :] = float("-inf")
        logits[:, kept] = torch.tensor([2.0, 1.5, 1.0])
        top_k = torch.full((bs,), _TOP_K_DISABLED)
    else:
        top_k = torch.full((bs,), 3)
    p = torch.softmax(torch.tensor([2.0, 1.5, 1.0]), 0)
    # Twice p: accepts half the time; the residual max(p - q, 0) is empty.
    q = torch.zeros(bs, 1, 16)
    q[..., :3] = 2 * p
    q_ids = torch.cat([kept, torch.arange(2000, 2013)]).repeat(bs, 1, 1)
    lanes = torch.randint(3, (bs, 1), generator=torch.Generator().manual_seed(4))
    batch = _VerifyBatch(
        logits.to(device),
        torch.ones(bs),
        top_k,
        torch.ones(bs),
        q,
        q_ids,
        torch.ones(bs, dtype=torch.bool),
        lanes,
    )

    row_accept, row_token = batch.verify_rows(route)

    rejected = row_accept[0::n] == 0
    assert 0 < int(rejected.sum()) < bs
    assert torch.all(row_token[0::n][rejected] == 40)


def _chain_oracle(candidates, row_accept, row_token):
    bs, n = len(candidates), len(candidates[0])
    predicts = [777] * (bs * n)
    accept_index = [[-1] * n for _ in range(bs)]
    lengths = []
    for b in range(bs):
        base, accepted = b * n, 0
        accept_index[b][0] = base
        while accepted < n - 1 and row_accept[base + accepted]:
            predicts[base + accepted] = candidates[b][accepted + 1]
            accepted += 1
            accept_index[b][accepted] = base + accepted
        predicts[base + accepted] = row_token[base + accepted]
        lengths.append(accepted)
    return predicts, accept_index, lengths


@pytest.mark.parametrize("n", [1, 2, 8])
def test_chain_row_decisions_emit_the_accepted_prefix(n: int, device: str) -> None:
    bs = 24
    generator = torch.Generator().manual_seed(n)
    candidates = torch.randint(1000, (bs, n), generator=generator, dtype=torch.int32)
    row_accept = (torch.rand(bs * n, generator=generator) < 0.7).to(torch.int32)
    row_token = torch.randint(1000, (bs * n,), generator=generator, dtype=torch.int32)
    predicts = torch.full((bs * n,), 777, dtype=torch.int32, device=device)
    accept_index = torch.full((bs, n), -1, dtype=torch.int32, device=device)
    accept_length = torch.full((bs,), 99, dtype=torch.int32, device=device)

    verify_chain_row_decisions(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_length,
        candidates=candidates.to(device),
        row_accept=row_accept.to(device),
        row_token=row_token.to(device),
    )

    expected = _chain_oracle(
        candidates.tolist(), row_accept.tolist(), row_token.tolist()
    )
    assert predicts.tolist() == expected[0]
    assert accept_index.tolist() == expected[1]
    assert accept_length.tolist() == expected[2]


def test_chain_row_decisions_reproduce_exact_match_verify(device: str) -> None:
    bs, n = 16, 6
    generator = torch.Generator().manual_seed(3)
    target = torch.randint(4, (bs * n,), generator=generator, dtype=torch.int32)
    candidates = torch.randint(4, (bs, n), generator=generator, dtype=torch.int32)
    no_draft = torch.full((bs, 1), -1, dtype=torch.int32)
    drafts = torch.cat([candidates[:, 1:], no_draft], 1).reshape(-1)
    row_accept = (target == drafts).to(torch.int32)
    batch_outputs = []
    for decisions in (True, False):
        predicts = torch.full((bs * n,), 777, dtype=torch.int32, device=device)
        accept_index = torch.full((bs, n), -1, dtype=torch.int32, device=device)
        accept_length = torch.full((bs,), 99, dtype=torch.int32, device=device)
        if decisions:
            verify_chain_row_decisions(
                predicts=predicts,
                accept_index=accept_index,
                accept_token_num=accept_length,
                candidates=candidates.to(device),
                row_accept=row_accept.to(device),
                row_token=target.to(device),
            )
        else:
            verify_chain_target_sampled(
                predicts,
                accept_index,
                accept_length,
                candidates.to(device),
                target.to(device),
                write_rejected_targets=False,
            )
        batch_outputs.append((predicts, accept_index, accept_length))
    for got, want in zip(*batch_outputs):
        assert torch.equal(got, want)


# ---------------------------------------------------------------------------
# GPU: losslessness. The first emitted token follows the target distribution
# of row 0; given that row 0 accepted, the second follows row 1's.
# ---------------------------------------------------------------------------


def _assert_follows(tokens: torch.Tensor, probs: torch.Tensor, label: str) -> None:
    """Chi-squared goodness of fit, pooling bins expected below 5 draws."""
    counts = torch.bincount(tokens.long().cpu(), minlength=probs.numel()).double()
    total = counts.sum()
    support = probs > 0
    assert counts[~support].sum() == 0, f"{label}: draws outside the support"
    expected = probs[support] * total
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


# name -> (route, groups of (label, requests, top_k, top_p, has_q)).
_CHI2_CASES = {
    "no_filter": ("no_filter", [("sampled", 100_000, _TOP_K_DISABLED, 1.0, True)]),
    "top_k": ("top_k", [("sampled", 100_000, 10, 1.0, True)]),
    "top_k_top_p": ("top_k", [("sampled", 100_000, 12, 0.75, True)]),
    "q_equals_p": ("top_k", [("sampled", 100_000, 6, 1.0, True)]),
    "mixed_top_k": (
        "top_k",
        [
            ("greedy", 2_000, 1, 1.0, True),
            ("sampled", 100_000, 10, 0.9, True),
            ("no_q", 100_000, 10, 0.9, False),
        ],
    ),
    "mixed_no_filter": (
        "no_filter",
        [
            ("sampled", 100_000, _TOP_K_DISABLED, 1.0, True),
            ("no_q", 100_000, _TOP_K_DISABLED, 1.0, False),
        ],
    ),
}


@pytest.mark.parametrize("case", list(_CHI2_CASES))
def test_rejection_verify_is_lossless(case: str, device: str) -> None:
    route, groups = _CHI2_CASES[case]
    vocab, num_candidates, n, temperature = 40, 16, 3, 0.9
    generator = torch.Generator().manual_seed(1234)
    # One logits row per window position, shared by every request.
    window_logits = torch.randn(n, vocab, generator=generator) * 1.5
    order = window_logits.argsort(dim=1, descending=True)

    def group_q(top_k: int, top_p: float) -> tuple[torch.Tensor, torch.Tensor]:
        q_ids = torch.empty(n - 1, num_candidates, dtype=torch.int64)
        q = torch.empty(n - 1, num_candidates)
        for r in range(n - 1):
            if case == "q_equals_p":
                target = _target_probs(
                    window_logits[r].double(), temperature, top_k, top_p
                )
                kept = target.nonzero().flatten()
                rest = [t for t in order[r].tolist() if t not in set(kept.tolist())]
                q_ids[r] = torch.cat([kept, torch.tensor(rest)])[:num_candidates]
                q[r] = 0.0
                q[r, : kept.numel()] = target[kept].float()
            else:
                # Candidates mix top tokens and the tail.
                q_ids[r] = torch.cat([order[r, 2:12], order[r, -6:]])
                q[r] = torch.softmax(
                    torch.randn(num_candidates, generator=generator) * 1.2, 0
                )
        return q, q_ids

    parts = []
    for label, count, top_k, top_p, has_q in groups:
        q, q_ids = group_q(top_k, top_p)
        lanes = torch.stack(
            [
                torch.multinomial(q[r], count, replacement=True, generator=generator)
                for r in range(n - 1)
            ],
            dim=1,
        )
        parts.append(
            dict(
                label=label,
                count=count,
                top_k=top_k,
                top_p=top_p,
                q=q.expand(count, -1, -1),
                q_ids=q_ids.expand(count, -1, -1),
                has_q=torch.full((count,), has_q),
                lanes=lanes,
            )
        )
    bs = sum(part["count"] for part in parts)
    batch = _VerifyBatch(
        window_logits.repeat(bs, 1).to(device),
        torch.full((bs,), temperature),
        torch.cat([torch.full((p["count"],), p["top_k"]) for p in parts]),
        torch.cat([torch.full((p["count"],), p["top_p"]) for p in parts]),
        torch.cat([p["q"] for p in parts]),
        torch.cat([p["q_ids"] for p in parts]),
        torch.cat([p["has_q"] for p in parts]),
        torch.cat([p["lanes"] for p in parts]),
        seed=99,
    )

    predicts, _, accept_length = batch.chain(*batch.verify_rows(route))
    predicts = predicts.view(bs, n).cpu()
    accept_length = accept_length.cpu()

    start = 0
    for part in parts:
        rows = slice(start, start + part["count"])
        start += part["count"]
        targets = [
            _target_probs(
                window_logits[r].double(), temperature, part["top_k"], part["top_p"]
            )
            for r in range(2)
        ]
        first = predicts[rows, 0]
        second = predicts[rows, 1][accept_length[rows] >= 1]
        label = f"{case}/{part['label']}"
        if part["top_k"] == 1:
            assert torch.all(first == int(targets[0].argmax()))
            assert torch.all(second == int(targets[1].argmax()))
            continue
        assert second.numel() > 2_000, f"{label}: too few accepted drafts"
        _assert_follows(first, targets[0], f"{label} position 1")
        _assert_follows(second, targets[1], f"{label} position 2")

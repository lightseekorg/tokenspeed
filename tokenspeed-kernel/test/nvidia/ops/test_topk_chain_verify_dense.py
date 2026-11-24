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

"""The sparse chain verify against the dense route it replaces, same coins.

Dense: full-vocabulary softmax, fused top-k / top-p renorm, and the
target-only chain kernel. Sparse: ``vocab_shard_topk`` and
``topk_chain_verify``. Both see the same logits, chains and coins; a token may
differ only where a decision lies within rounding of its threshold.
"""

import math

import pytest
import torch
from tokenspeed_kernel.ops.sampling import topk_chain_verify, vocab_shard_topk
from tokenspeed_kernel.ops.sampling.cuda import (
    chain_speculative_sampling_target_only,
    fused_topk_topp_prepare,
    fused_topk_topp_renorm,
)
from tokenspeed_kernel.ops.sampling.flashinfer import softmax
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform().is_nvidia, reason="CUDA sampling kernels require NVIDIA"
)

VOCAB = 248320
K = 32
N = 4
_COIN_LO = 2.0**-24
_COIN_HI = 1.0 - 2.0**-24


def _batch(bs: int, seed: int, top_k: int | None = None):
    gen = torch.Generator().manual_seed(seed)
    logits = torch.randn(bs * N, VOCAB, generator=gen) * 2
    hot = torch.randint(0, VOCAB, (bs * N, 8), generator=gen)
    logits.scatter_add_(1, hot, torch.rand(bs * N, 8, generator=gen) * 12)
    logits = logits.to("cuda", torch.bfloat16)
    pick = lambda values: torch.tensor(values)[
        torch.randint(0, len(values), (bs,), generator=gen)
    ]
    temperatures = pick([0.6, 1.0, 1.3]).repeat_interleave(N).cuda()
    top_ks = pick([1, 5, 20, K]) if top_k is None else torch.full((bs,), top_k)
    top_ks = top_ks.to(torch.int32).repeat_interleave(N).cuda()
    top_ps = pick([0.8, 0.95, 1.0]).repeat_interleave(N).cuda()
    best = torch.topk(logits.float(), 3, dim=1).indices.view(bs, N, 3).cpu()
    chains = torch.gather(best, 2, torch.randint(0, 3, (bs, N, 1), generator=gen))
    chains = chains.squeeze(2).roll(1, dims=1)
    anywhere = torch.randint(0, VOCAB, chains.shape, generator=gen)
    chains = torch.where(
        torch.rand(chains.shape, generator=gen) < 0.2, anywhere, chains
    )
    coins = torch.rand(bs, N, generator=gen).clamp_(_COIN_LO, _COIN_HI)
    final_coins = torch.rand(bs, generator=gen).clamp_(_COIN_LO, _COIN_HI)
    return (
        logits,
        temperatures,
        top_ks,
        top_ps,
        chains.to("cuda", torch.int32),
        coins.cuda(),
        final_coins.cuda(),
    )


def _outputs(bs: int):
    return (
        torch.full((bs * N,), -7, dtype=torch.int32, device="cuda"),
        torch.full((bs, N), -1, dtype=torch.int32, device="cuda"),
        torch.full((bs,), -7, dtype=torch.int32, device="cuda"),
    )


def _dense(
    logits,
    temperatures,
    top_ks,
    top_ps,
    chains,
    coins,
    final_coins,
    thresholds=(1.0, 1.0),
):
    bs = chains.shape[0]
    fused_topk_topp_prepare(logits.device)
    probs = softmax(logits, temperature=temperatures)
    probs = fused_topk_topp_renorm(probs, top_ks, top_ps, in_place_skip=False)
    predicts, accept_index, accepted = _outputs(bs)
    chain_speculative_sampling_target_only(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accepted,
        candidates=chains,
        uniform_samples=coins,
        uniform_samples_for_final_sampling=final_coins,
        target_probs=probs.view(bs, N, VOCAB),
        draft_probs=None,
        threshold_single=thresholds[0],
        threshold_acc=thresholds[1],
        deterministic=True,
        use_draft_prob=False,
        reject_draft_prob_threshold=2.0,
    )
    return (predicts, accept_index, accepted), probs


def _sparse(
    logits,
    temperatures,
    top_ks,
    top_ps,
    chains,
    coins,
    final_coins,
    thresholds=(1.0, 1.0),
):
    bs = chains.shape[0]
    topk = torch.empty(bs * N, 2 * K, device="cuda")
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
    predicts, accept_index, accepted = _outputs(bs)
    topk_chain_verify(
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
        threshold_single=thresholds[0],
        threshold_acc=thresholds[1],
    )
    return predicts, accept_index, accepted


def _emitted(outputs, b: int) -> list[int]:
    predicts, _, accepted = outputs
    return predicts[b * N : b * N + int(accepted[b]) + 1].tolist()


def _ulps(a: float, b: float) -> float:
    """|a - b| in units of the last place of the larger magnitude (fp32)."""
    scale = max(abs(a), abs(b), 2.0**-126)
    return abs(a - b) / 2.0 ** (math.floor(math.log2(scale)) - 23)


def _boundary_ulps(
    dense, probs, chains, coins, final_coins, sparse, b: int, thresholds
) -> tuple[str, float]:
    """Where request ``b``'s routes first part, and how far the dense decision
    there lies from its threshold, in fp32 ulps."""
    want, got = _emitted(dense, b), _emitted(sparse, b)
    j = next(i for i in range(min(len(want), len(got))) if want[i] != got[i])
    row = probs[b * N + j].double()
    draft = int(chains[b, j + 1]) if j + 1 < N else None
    if (len(want) > j + 1) != (len(got) > j + 1):
        prob = float(row[draft])
        single, acc = thresholds
        return "accept", min(
            _ulps(float(coins[b, j]), prob / max(acc, 1e-9)), _ulps(prob, single)
        )
    rejected = draft
    mass = row.clone()
    if rejected is not None:
        mass[rejected] = 0.0
    cdf = torch.cumsum(mass, 0)
    threshold = float(final_coins[b]) * float(cdf[-1])
    low, high = sorted((want[j], got[j]))
    return "sample", min(_ulps(threshold, float(c)) for c in cdf[low:high].tolist())


@pytest.mark.parametrize(
    "thresholds,batches", [((1.0, 1.0), 16), ((0.4, 0.2), 4), ((1.0, 0.0), 4)]
)
def test_sparse_route_commits_the_dense_tokens_but_at_rounding_boundaries(
    thresholds, batches
) -> None:
    """256 requests per batch (mixed temperature, top-k 1 to 32, top-p) at the
    production vocabulary, with the serving thresholds and two others: report
    the mismatch fraction and, for each mismatch, the dense decision's
    distance from its threshold in fp32 ulps; every distance must be a
    rounding-scale one (at most 64 ulps)."""
    requests, mismatches, decisions = 0, [], 0
    for seed in range(batches):
        batch = _batch(256, seed)
        chains, coins, final_coins = (t.cpu() for t in batch[4:])
        dense, probs = _dense(*batch, thresholds)
        sparse = _sparse(*batch, thresholds)
        for b in range(256):
            if _emitted(dense, b) != _emitted(sparse, b):
                mismatches.append(
                    _boundary_ulps(
                        dense, probs, chains, coins, final_coins, sparse, b, thresholds
                    )
                )
        requests += 256
        decisions += int((dense[2] + 1).clamp(max=N - 1).sum()) + 256
    print(
        f"{len(mismatches)} of {requests} requests ({decisions} decisions) differ: {mismatches}"
    )
    assert len(mismatches) <= requests // 1000
    assert all(distance <= 64 for _, distance in mismatches)


def test_top_k_one_rows_commit_the_dense_tokens() -> None:
    batch = _batch(256, 99, top_k=1)
    dense, _ = _dense(*batch)
    sparse = _sparse(*batch)
    assert all(_emitted(dense, b) == _emitted(sparse, b) for b in range(256))
    assert torch.equal(dense[2], sparse[2])
    assert torch.equal(dense[1], sparse[1])

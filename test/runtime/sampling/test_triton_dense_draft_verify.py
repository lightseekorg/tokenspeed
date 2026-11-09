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

"""Draft-distribution verify on the triton sampling backend under
--enable-speculative-sampling: a chain against its recorded per-vocab draft
distributions (``SamplingBatchInfo.draft_probs``) and a sampled draft tree
against its draft q (``DraftTreeQ``) both emit exactly the target
distribution p' of the batch's route, and greedy requests stay deterministic.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from ci_system.ci_register import register_cuda_ci

from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
from tokenspeed.runtime.sampling.backends.base import SamplingBackendConfig
from tokenspeed.runtime.sampling.backends.triton import TritonSamplingBackend
from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
from tokenspeed.runtime.sampling.sampling_params import SamplingParams
from tokenspeed.runtime.sampling.tree_verify import DraftTreeQ, TreeVerifyBatch

register_cuda_ci(est_time=30, suite="runtime-1gpu")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

VOCAB, BS, THRESHOLD = 48, 128, 2.0
_CHI2_MIN_P_VALUE = 1.0e-3
ROUTES = {
    "no_filter": dict(temperature=1.2, top_k=-1, top_p=1.0),
    "top_k_top_p": dict(temperature=0.8, top_k=12, top_p=0.85),
}


def _backend(
    n: int, params: list[dict], tag: str, *, tree: bool
) -> TritonSamplingBackend:
    # As SamplingBackendConfig.from_server_args splits the flag: a chain
    # verifies recorded draft probs, a tree its q table.
    backend = TritonSamplingBackend(
        SamplingBackendConfig(
            synthetic_acceptance_length=None,
            enable_speculative_sampling=not tree,
            sampled_draft_trees=tree,
            sampling_stream="batch",
            logprob_order="torch",
            max_bs=BS,
            max_draft_tokens_per_req=n,
            max_req_pool_size=BS,
            vocab_size=VOCAB,
            device="cuda",
            spec_reject_draft_prob_threshold=THRESHOLD,
        )
    )
    sampling_params = []
    for i, kwargs in enumerate(params):
        sp = SamplingParams(**kwargs)
        sp.resolve_seed(f"{tag}_{i}")
        sp.normalize(None)
        sampling_params.append(sp)
    backend.prepare_step(
        request_ids=[f"{tag}_{i}" for i in range(len(params))],
        request_pool_indices=list(range(1, len(params) + 1)),
        sampling_params_list=sampling_params,
        num_tokens_per_req=n,
    )
    return backend


def _info(position: int, draft_probs: torch.Tensor | None) -> SamplingBatchInfo:
    """Every request at ``position``: a new position draws fresh coins."""
    return SamplingBatchInfo(
        req_pool_indices=torch.arange(1, BS + 1, device="cuda"),
        valid_cache_lengths=torch.full(
            (BS + 1,), position, dtype=torch.int32, device="cuda"
        ),
        draft_probs=draft_probs,
        vocab_size=VOCAB,
        device="cuda",
    )


def _route_probs(logits: torch.Tensor, params: dict) -> torch.Tensor:
    """p' from its definition: the softmax at the temperature, or over the
    top-k logits the tokens whose top-k mass above them is below top_p,
    renormalized."""
    scaled = logits.double() / params["temperature"]
    if params["top_k"] < 0:
        return torch.softmax(scaled, -1)
    order = torch.sort(logits, descending=True, stable=True).indices[: params["top_k"]]
    probs = torch.softmax(scaled[order], -1)
    kept = order[torch.cumsum(probs, 0) - probs < params["top_p"]]
    out = torch.zeros_like(scaled)
    out[kept] = torch.softmax(scaled[kept], -1)
    return out


def _assert_follows(tokens: torch.Tensor, probs: torch.Tensor, label: str) -> None:
    """Chi-squared goodness of fit, pooling bins expected below 5 draws."""
    counts = torch.bincount(tokens.long().cpu(), minlength=probs.numel()).double()
    probs = probs.double().cpu()
    support = probs > 0
    assert counts[~support].sum() == 0, f"{label}: draws outside the support"
    expected = probs[support] / probs[support].sum() * counts.sum()
    observed = counts[support]
    small = expected < 5
    if bool(small.any()):
        expected = torch.cat([expected[~small], expected[small].sum()[None]])
        observed = torch.cat([observed[~small], observed[small].sum()[None]])
    dof = expected.numel() - 1
    stat = ((observed - expected) ** 2 / expected).sum()
    p_value = float(
        torch.special.gammaincc(torch.tensor(dof / 2.0, dtype=torch.float64), stat / 2)
    )
    assert (
        p_value > _CHI2_MIN_P_VALUE
    ), f"{label}: chi2 {stat:.1f} dof {dof} p {p_value:.2e}"


@pytest.mark.parametrize("route", list(ROUTES))
def test_chain_with_recorded_draft_probs_emits_the_target_distribution(route):
    """Drafts drawn from the recorded q: the first emitted token follows p'
    of row 0, the next (after an accepted first draft) p' of row 1, and the
    first draft is accepted with probability sum(min(p', q))."""
    n, rounds = 3, 160
    generator = torch.Generator().manual_seed(0)
    params = ROUTES[route]
    backend = _backend(n, [params] * BS, f"chain_{route}", tree=False)
    row_logits = torch.randn(n, VOCAB, generator=generator) * 2
    p_rows = torch.stack([_route_probs(row_logits[r], params) for r in range(n)])
    # q proposes tokens p' cuts too, so rejections and residuals both occur.
    q_rows = torch.softmax(torch.randn(n - 1, VOCAB, generator=generator), -1)
    draft_probs = torch.zeros(BS + 1, n, VOCAB)
    draft_probs[:, : n - 1] = q_rows
    draft_probs[0] = THRESHOLD + 1.0  # the padding slot carries the sentinel
    draft_probs = draft_probs.cuda()
    logits = row_logits.repeat(BS, 1).cuda()
    first, second, accepted = [], [], 0
    for step in range(rounds):
        candidates = torch.zeros(BS, n, dtype=torch.int64)
        for i in range(1, n):
            candidates[:, i] = torch.multinomial(
                q_rows[i - 1], BS, replacement=True, generator=generator
            )
        predict, accept_length = backend.verify(
            LogitsProcessorOutput(next_token_logits=logits.clone()),
            _info(16 * step, draft_probs),
            candidates.cuda(),
            tree=None,
        )
        predict = predict.view(BS, n).cpu()
        lengths = accept_length.cpu()
        assert ((lengths >= 1) & (lengths <= n)).all()
        for b in range(BS):
            assert predict[b, : lengths[b] - 1].tolist() == (
                candidates[b, 1 : int(lengths[b])].tolist()
            )
        first.append(predict[:, 0])
        second.append(predict[lengths >= 2, 1])
        accepted += int((lengths >= 2).sum())
    _assert_follows(torch.cat(first), p_rows[0], "first token")
    _assert_follows(torch.cat(second), p_rows[1], "second token")
    want = float(torch.minimum(p_rows[0], q_rows[0].double()).sum())
    total = rounds * BS
    assert abs(accepted / total - want) < 4 * (want * (1 - want) / total) ** 0.5


def test_greedy_requests_accept_the_argmax_chain():
    """Greedy requests in a top-k batch accept drafts while they match the
    argmax and then emit the argmax, whatever the draft q."""
    n = 4
    generator = torch.Generator().manual_seed(1)
    params = [
        dict(temperature=0.0, top_k=-1, top_p=1.0) if i % 2 else ROUTES["top_k_top_p"]
        for i in range(BS)
    ]
    backend = _backend(n, params, "greedy", tree=False)
    logits = torch.randn(BS * n, VOCAB, generator=generator)
    best = logits.argmax(-1).view(BS, n)
    candidates = torch.randint(0, VOCAB, (BS, n), generator=generator)
    # Greedy-matching prefixes of random lengths.
    for b in range(BS):
        keep = b % n
        candidates[b, 1 : keep + 1] = best[b, :keep]
    q = torch.softmax(torch.randn(BS + 1, n, VOCAB, generator=generator), -1)
    q[:, -1] = 0.0
    predict, accept_length = backend.verify(
        LogitsProcessorOutput(next_token_logits=logits.cuda()),
        _info(7, q.cuda()),
        candidates.cuda(),
        tree=None,
    )
    predict = predict.view(BS, n).cpu()
    for b in range(1, BS, 2):
        length = 1
        while length < n and int(candidates[b, length]) == int(best[b, length - 1]):
            length += 1
        assert int(accept_length[b]) == length, f"request {b}"
        assert predict[b, :length].tolist() == best[b, :length].tolist()


@pytest.mark.parametrize("route", list(ROUTES))
def test_sampled_tree_emits_the_target_distribution(route):
    """A root with K children drawn without replacement from its draft q over
    NC candidates: the first emitted token follows p' of the root row."""
    k, nc, rounds = 3, 8, 160
    n = k + 1
    generator = torch.Generator().manual_seed(2)
    params = ROUTES[route]
    backend = _backend(n, [params] * BS, f"tree_{route}", tree=True)
    root_logits = torch.randn(VOCAB, generator=generator) * 2
    p_root = _route_probs(root_logits, params)
    # Candidates: the root row's best tokens and some p' cuts.
    ids = torch.cat(
        [
            torch.sort(root_logits, descending=True).indices[:5],
            torch.tensor([45, 46, 47]),
        ]
    ).to(torch.int32)
    assert ids.unique().numel() == nc
    q_root = torch.softmax(torch.randn(nc, generator=generator), -1)
    logits = torch.randn(BS * n, VOCAB, generator=generator)
    logits.view(BS, n, VOCAB)[:, 0] = root_logits
    logits = logits.cuda()
    parents = torch.tensor([-1] + [0] * k, dtype=torch.int32).repeat(BS, 1).cuda()
    depths = torch.tensor([0] + [1] * k, dtype=torch.int32).repeat(BS, 1).cuda()
    rows = BS + 1
    q = torch.zeros(rows, n, nc)
    q[:, 0] = q_root
    q_ids = ids.repeat(rows, n, 1)
    slot = torch.tensor([-1] + [0] * k, dtype=torch.int32).repeat(rows, 1)
    first = []
    for step in range(rounds):
        pos = torch.zeros(rows, n, dtype=torch.int32)
        for r in range(rows):
            pos[r, 1:] = torch.multinomial(
                q_root, k, replacement=False, generator=generator
            )
        candidates = torch.zeros(BS, n, dtype=torch.int32)
        candidates[:, 1:] = ids[pos[1:, 1:].long()]
        draft_q = DraftTreeQ(
            q=q.cuda(), q_ids=q_ids.cuda(), slot=slot.cuda(), pos=pos.cuda()
        )
        predict, accept_length = backend.verify(
            LogitsProcessorOutput(next_token_logits=logits.clone()),
            _info(16 * step, None),
            candidates.cuda(),
            tree=TreeVerifyBatch(parents=parents, depths=depths, draft_q=draft_q),
        )
        lengths = accept_length.cpu()
        assert ((lengths >= 1) & (lengths <= 2)).all()
        first.append(predict.view(BS, n)[:, 0].cpu())
    _assert_follows(torch.cat(first), p_root, "first token")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

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

"""Draft-tree verify on the flashinfer backend (``TreeRejectionSampler``):
greedy and uncut rows skip their probability work without changing a
decision, for chosen children and for children drawn from the draft q
(``DraftTreeQ``, read at each request's pool slot). test_eagle_harness.py
checks that sampled trees emit exactly the target distribution."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from ci_system.ci_register import register_cuda_ci
from tokenspeed_kernel.ops.sampling.cuda import fused_topk_topp_renorm
from tokenspeed_kernel.ops.sampling.flashinfer import softmax
from tokenspeed_kernel.thirdparty.cuda.eagle_harness import (
    eagle_harness_residual_sample,
    eagle_harness_tree_reject_sample,
)

from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
from tokenspeed.runtime.sampling.backends.base import SamplingBackendConfig
from tokenspeed.runtime.sampling.backends.flashinfer import FlashInferSamplingBackend
from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
from tokenspeed.runtime.sampling.sampling_params import SamplingParams
from tokenspeed.runtime.sampling.tree_verify import DraftTreeQ, TreeVerifyBatch

register_cuda_ci(
    est_time=20,
    suite="runtime-1gpu",
    disabled_on_runners=["amd-*"],
    disabled_on_runners_reason="the eagle_harness CUDA kernels are NVIDIA-only",
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _backend(bs: int, n: int, vocab: int) -> FlashInferSamplingBackend:
    return FlashInferSamplingBackend(
        SamplingBackendConfig(
            synthetic_acceptance_length=None,
            max_bs=bs,
            max_draft_tokens_per_req=n,
            max_req_pool_size=bs + 1,
            vocab_size=vocab,
            device="cuda",
            enable_speculative_sampling=False,
            sampling_stream="batch",
            logprob_order="torch",
        )
    )


# Greedy (temperature 0 and top-k 1), uncut, top-k + top-p, top-p only and top-k only.
SKIP_PARAMS = [
    dict(temperature=0.0, top_k=-1, top_p=1.0),
    dict(temperature=1.0, top_k=-1, top_p=1.0),
    dict(temperature=0.8, top_k=20, top_p=0.95),
    dict(temperature=0.7, top_k=1, top_p=1.0),
    dict(temperature=0.7, top_k=-1, top_p=0.9),
    dict(temperature=1.2, top_k=5, top_p=1.0),
]
SKIP_BS, SKIP_N, SKIP_NC = 8, 16, 32


def _skip_problem(vocab: int, sampled_tree: bool, seed: int):
    """Random logits and random trees whose tokens often hit their parent
    row's argmax; a sampled tree draws each sampled request's children from a
    q slot per parent (greedy requests stay deterministic, as the drafter
    leaves them)."""
    dev = "cuda"
    gen = torch.Generator().manual_seed(seed)
    bs, n, nc = SKIP_BS, SKIP_N, SKIP_NC
    params = [SKIP_PARAMS[i % len(SKIP_PARAMS)] for i in range(bs)]
    greedy = [p["temperature"] == 0.0 or p["top_k"] == 1 for p in params]
    logits = torch.randn((bs * n, vocab), generator=gen) * 3
    best = logits.argmax(-1).view(bs, n)
    parents = torch.full((bs, n), -1, dtype=torch.int32)
    tokens = torch.randint(0, vocab, (bs, n), generator=gen, dtype=torch.int32)
    rows = bs + 1
    q = torch.softmax(torch.randn((rows, n, nc), generator=gen) * 2, -1)
    q_ids = torch.randint(0, vocab, (rows, n, nc), generator=gen, dtype=torch.int32)
    slot = torch.full((rows, n), -1, dtype=torch.int32)
    pos = torch.zeros((rows, n), dtype=torch.int32)
    for b in range(bs):
        used = int(torch.randint(n // 2, n + 1, (1,), generator=gen))
        free = [torch.randperm(nc, generator=gen).tolist() for _ in range(n)]
        for i in range(1, used):
            parent = int(torch.randint(0, i, (1,), generator=gen))
            parents[b, i] = parent
            if float(torch.rand((1,), generator=gen)) < 0.6:
                tokens[b, i] = best[b, parent]
            if sampled_tree and not greedy[b]:
                slot[b + 1, i] = parent
                pos[b + 1, i] = free[parent].pop()
                q_ids[b + 1, parent, pos[b + 1, i]] = tokens[b, i]
        tokens[b, used:] = tokens[b, 0]
    draft_q = (
        DraftTreeQ(q=q.to(dev), q_ids=q_ids.to(dev), slot=slot.to(dev), pos=pos.to(dev))
        if sampled_tree
        else None
    )
    return params, logits.to(dev), tokens.to(dev), parents.to(dev), draft_q


def _full_probability_verify(backend, logits, tokens, parents, draft_q, req):
    """The verify with every row's probabilities computed: the softmax cut
    and renormalized out of place (uncut rows keep the softmax values),
    torch.argmax of those rows, and the rejection kernels on every row."""
    bs, n = tokens.shape
    pools = backend.speculative_sampling_pools()
    temperatures = pools.temperature[req].repeat_interleave(n)
    top_ks = pools.top_k[req].repeat_interleave(n)
    top_ps = pools.top_p[req].repeat_interleave(n)
    probs = softmax(logits, temperature=temperatures)
    cut = fused_topk_topp_renorm(probs, top_ks, top_ps, in_place_skip=False)
    uncut = (top_ks >= logits.shape[1]) & (top_ps >= 1.0)
    target = torch.where(uncut[:, None], probs, cut)
    argmax = target.argmax(dim=-1)
    never_greedy = torch.zeros_like(top_ks)
    q = draft_q if draft_q is not None else backend._tree_sampler.no_draft_q
    path = torch.empty((bs, n), dtype=torch.int32, device="cuda")
    predict = torch.full((bs * n,), -1, dtype=torch.int32, device="cuda")
    accept_length = torch.empty(bs, dtype=torch.int32, device="cuda")
    last_row = torch.empty(bs, dtype=torch.int64, device="cuda")
    eagle_harness_tree_reject_sample(
        target, tokens, parents, req if draft_q is not None else None, q.slot,
        q.pos, q.q, q.q_ids, backend._coins_buf[:bs, :n], never_greedy, argmax,
        path, predict, accept_length, last_row,
    )  # fmt: skip
    eagle_harness_residual_sample(
        target, last_row, backend._final_coins_buf[:bs], never_greedy, argmax,
        accept_length, predict, n,
    )  # fmt: skip
    return path, predict, accept_length, last_row


@pytest.mark.parametrize("sampled_tree", [False, True])
def test_verify_tree_skips_greedy_and_uncut_rows_without_changing_decisions(
    sampled_tree: bool,
) -> None:
    """Greedy rows (no probabilities, the logits' argmax) and uncut rows
    (no renorm) decide every child, path and bonus exactly as the full
    probability path does, across mixed batches and random trees."""
    bs, n, vocab = SKIP_BS, SKIP_N, 32768
    backend = _backend(bs, n, vocab)
    pool = list(range(1, bs + 1))
    req = torch.tensor(pool, dtype=torch.int64, device="cuda")
    accepted = {True: 0, False: 0}
    for seed in range(3):
        params, logits, tokens, parents, draft_q = _skip_problem(
            vocab, sampled_tree, seed
        )
        sampling_params = []
        for i, kwargs in enumerate(params):
            sp = SamplingParams(**kwargs)
            sp.resolve_seed(f"skip_{seed}_{i}")
            sp.normalize(None)
            sampling_params.append(sp)
        backend.prepare_step(
            request_ids=[f"skip_{seed}_{i}" for i in range(bs)],
            request_pool_indices=pool,
            sampling_params_list=sampling_params,
            num_tokens_per_req=n,
        )
        expected = _full_probability_verify(
            backend, logits, tokens, parents, draft_q, req
        )
        tree = TreeVerifyBatch(
            parents=parents,
            depths=torch.zeros_like(parents),
            draft_q=draft_q,
        )
        info = SamplingBatchInfo(
            req_pool_indices=req,
            valid_cache_lengths=torch.zeros(bs + 1, dtype=torch.int32, device="cuda"),
            vocab_size=vocab,
            device="cuda",
        )
        predict, accept_length = backend.verify(
            LogitsProcessorOutput(next_token_logits=logits.clone()),
            info,
            tokens,
            tree=tree,
        )
        path = backend.accepted_path(bs, n)
        want_path, want_predict, want_length, want_last = expected
        assert torch.equal(accept_length, want_length)
        assert torch.equal(path, want_path)
        assert torch.equal(backend._tree_sampler.last_row[:bs], want_last)
        for b in range(bs):
            length = int(want_length[b])
            assert torch.equal(
                predict[b * n : b * n + length], want_predict[b * n : b * n + length]
            )
            accepted[params[b]["temperature"] == 0.0 or params[b]["top_k"] == 1] += (
                length - 1
            )
    # Both kinds of rows accept children, so the comparison covers acceptance.
    assert accepted[True] > 0 and accepted[False] > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

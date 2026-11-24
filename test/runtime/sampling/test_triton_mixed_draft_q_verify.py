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

"""Triton verify of the DFlash2 sparse draft q on a batch that mixes filter
kinds (the generic route): each request verifies as it would in a batch of its
own kind, so a mixed batch keeps the rejection rule instead of falling back to
the target-only rule for every request."""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.layers.logits_processor import (  # noqa: E402
    LogitsProcessorOutput,
)
from tokenspeed.runtime.sampling.backends.base import (  # noqa: E402
    CUDA_GRAPH_VARIANT_DEFAULT,
    SamplingBackendConfig,
)
from tokenspeed.runtime.sampling.backends.triton import (  # noqa: E402
    TritonSamplingBackend,
)
from tokenspeed.runtime.sampling.sampling_batch_info import (  # noqa: E402
    SamplingBatchInfo,
)
from tokenspeed.runtime.sampling.sampling_params import SamplingParams  # noqa: E402
from tokenspeed.runtime.sampling.tree_verify import DraftTreeQ  # noqa: E402

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

VOCAB = 4096
N = 8  # verify width: anchor + 7 drafts
K = 16  # recorded q lanes per draft step
POOL = 8
POOL_INDICES = [2, 5, 7, 3]
RIDS = ["no-filter", "top-k", "greedy", "top-p"]
# (top_k, top_p) per request; -1 disables top-k. Together: the generic route.
# The top-k request keeps 2 tokens, so drafts of a row's third best token are
# rejected under its rule but not under the no-filter rule.
MIXED = [(-1, 1.0), (2, 0.9), (1, 1.0), (-1, 0.8)]
BS = len(MIXED)


def _backend() -> TritonSamplingBackend:
    return TritonSamplingBackend(
        SamplingBackendConfig(
            synthetic_acceptance_length=None,
            enable_speculative_sampling=True,
            sampling_stream="batch",
            logprob_order="torch",
            max_bs=BS,
            max_draft_tokens_per_req=N,
            max_req_pool_size=POOL,
            vocab_size=VOCAB,
            device="cuda",
        )
    )


def _prepare(backend: TritonSamplingBackend, filters) -> torch.Tensor:
    """Admit the batch with each request's (top_k, top_p); seeds follow the
    request id, so a request draws the same noise in any batch."""
    params = []
    for rid, (top_k, top_p) in zip(RIDS, filters):
        sp = SamplingParams(temperature=1.0, top_k=top_k, top_p=top_p)
        sp.resolve_seed(rid)
        sp.normalize(None)
        params.append(sp)
    backend.prepare_step(
        admission_ids=list(range(1, BS + 1)),
        request_pool_indices=POOL_INDICES,
        sampling_params_list=params,
        num_tokens_per_req=N,
    )
    return torch.tensor(POOL_INDICES, dtype=torch.int64, device="cuda")


def _info(req_pool_indices, draft_q) -> SamplingBatchInfo:
    return SamplingBatchInfo(
        req_pool_indices=req_pool_indices,
        valid_cache_lengths=torch.arange(
            1000, 1000 + POOL + 1, dtype=torch.int32, device="cuda"
        ),
        draft_q=draft_q,
        vocab_size=VOCAB,
        device="cuda",
    )


def _logits() -> torch.Tensor:
    gen = torch.Generator().manual_seed(11)
    return (torch.randn(BS * N, VOCAB, generator=gen) * 4).cuda()


def _draft_q(logits: torch.Tensor) -> tuple[DraftTreeQ, torch.Tensor]:
    """A recorded chain per request over each row's K best tokens, and the
    window whose drafts the chain proposed."""
    gen = torch.Generator().manual_seed(3)
    q = DraftTreeQ.zeros(POOL + 1, N - 1, K, N, "cuda")
    best = torch.topk(logits, K, dim=-1).indices
    candidates = torch.zeros(BS, N, dtype=torch.int32, device="cuda")
    for b, pool in enumerate(POOL_INDICES):
        weights = torch.rand(N - 1, K, generator=gen) ** 4
        q.q[pool] = (weights / weights.sum(-1, keepdim=True)).cuda()
        # Row r verifies draft r + 1 against q row r.
        q.q_ids[pool] = best[b * N : (b + 1) * N - 1].to(torch.int32)
        q.slot[pool] = torch.arange(-1, N - 1, dtype=torch.int32, device="cuda")
        lanes = torch.randint(0, 3, (N - 1,), generator=gen).cuda()
        q.pos[pool, 1:] = lanes.to(torch.int32)
        candidates[b, 1:] = q.q_ids[pool, torch.arange(N - 1), lanes]
    return q, candidates


def _verify(filters, logits, draft_q, candidates) -> list[list[int]]:
    """Each request's emitted tokens (accepted drafts plus the final token)."""
    backend = _backend()
    req_pool_indices = _prepare(backend, filters)
    predict, accept_length = backend.verify(
        LogitsProcessorOutput(next_token_logits=logits.clone()),
        _info(req_pool_indices, draft_q),
        candidates,
        tree=None,
    )
    predict = predict.view(BS, N).tolist()
    return [predict[b][:n] for b, n in enumerate(accept_length.tolist())]


@requires_cuda
def test_each_request_of_a_mixed_batch_verifies_as_its_own_kind() -> None:
    logits = _logits()
    q, candidates = _draft_q(logits)
    mixed = _verify(MIXED, logits, q, candidates)

    # The no-filter, top-k and greedy requests: what a batch of only that
    # kind (the no-filter or top-k route, with the rejection rule) emits.
    uniform = [_verify([MIXED[b]] * BS, logits, q, candidates) for b in range(3)]
    for b in range(3):
        assert mixed[b] == uniform[b][b], RIDS[b]
    # The data tells the kinds apart: under the no-filter rule the top-k and
    # greedy requests would emit something else.
    assert uniform[0][1] != uniform[1][1]
    assert uniform[0][2] != uniform[2][2]
    # The top-p-only request has no exact filtered distribution: the
    # target-only rule, as when the batch carries no q at all.
    assert mixed[3] == _verify(MIXED, logits, None, candidates)[3]
    # The windows were neither all accepted nor all rejected.
    lengths = [len(tokens) for tokens in mixed]
    assert min(lengths) < N and max(lengths) > 1


@requires_cuda
def test_a_mixed_batch_verify_replays_from_a_cuda_graph() -> None:
    """Serving's order: capture the default variant (generic route, top-k pad
    128), then replay after a step that admitted the mixed batch (pad 64)."""
    logits = _logits()
    q, candidates = _draft_q(logits)
    eager = _verify(MIXED, logits, q, candidates)

    backend = _backend()
    req_pool_indices = _prepare(backend, MIXED)
    info = _info(req_pool_indices, q)
    output = LogitsProcessorOutput(next_token_logits=logits.clone())
    backend.prepare_capture_variant(
        bs=BS, num_tokens_per_req=N, variant=CUDA_GRAPH_VARIANT_DEFAULT
    )
    # Warm the kernels outside the capture, as serving's startup does.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        backend.verify(output, info, candidates, tree=None)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        predict, accept_length = backend.verify(output, info, candidates, tree=None)

    _prepare(backend, MIXED)
    predict.zero_()
    accept_length.zero_()
    graph.replay()
    torch.cuda.synchronize()

    predict = predict.view(BS, N).tolist()
    replayed = [predict[b][:n] for b, n in enumerate(accept_length.tolist())]
    assert replayed == eager

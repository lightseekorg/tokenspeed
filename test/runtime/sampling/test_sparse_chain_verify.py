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

"""The FlashInfer backend's sparse chain verify: routing, graph variants, and tokens.

A batch whose requests all keep at most the route's width of tokens (top-k)
verifies its target-only chains on each row's top-k candidates, under the
``topk_sparse`` sampler graph variant; other batches keep the dense route.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")

from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
from tokenspeed.runtime.sampling.backends.base import SamplingBackendConfig
from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
from tokenspeed.runtime.sampling.sampling_params import SamplingParams
from tokenspeed.runtime.utils.env import global_server_args_dict

requires_flashinfer_kernels = pytest.mark.skipif(
    not torch.cuda.is_available() or bool(torch.version.hip),
    reason="flashinfer sampling kernels require CUDA (not ROCm)",
)
pytestmark = requires_flashinfer_kernels

VOCAB = 50000
POOL = 40
MAX_BS = 34
N = 4
DEFAULT, SPARSE = "default", "topk_sparse"


def _config(**overrides) -> SamplingBackendConfig:
    fields = dict(
        synthetic_acceptance_length=None,
        enable_speculative_sampling=False,
        sampling_stream="per-request",
        logprob_order="torch",
        max_bs=MAX_BS,
        max_draft_tokens_per_req=N,
        max_req_pool_size=POOL,
        vocab_size=VOCAB,
        device="cuda",
    )
    fields.update(overrides)
    return SamplingBackendConfig(**fields)


def _backend(**overrides):
    from tokenspeed.runtime.sampling.backends.flashinfer import (
        FlashInferSamplingBackend,
    )

    return FlashInferSamplingBackend(_config(**overrides))


def _params(
    top_k: int, *, seed: int = 0, temperature: float = 1.0, top_p: float = 0.95
):
    params = SamplingParams(
        temperature=temperature, top_k=top_k, top_p=top_p, seed=seed
    )
    params.verify(VOCAB)
    return params


def _step(
    backend, admissions: list[int], params: list[SamplingParams], n: int = N
) -> str:
    backend.prepare_step(
        admission_ids=admissions,
        request_pool_indices=[a % POOL + 1 for a in admissions],
        sampling_params_list=params,
        num_tokens_per_req=n,
    )
    return backend.cuda_graph_replay_variant(n)


def test_the_batch_top_k_picks_the_route_and_its_graph():
    backend = _backend()
    assert backend.cuda_graph_capture_variants(N) == (DEFAULT, SPARSE)
    assert backend.cuda_graph_capture_variants(1) == (DEFAULT,)

    assert _step(backend, [1, 2], [_params(20), _params(32)]) == SPARSE
    assert _step(backend, [3], [_params(-1)]) == DEFAULT
    assert _step(backend, [4, 5], [_params(20), _params(33)]) == DEFAULT
    # Greedy requests normalize to top-k 1.
    assert _step(backend, [6], [_params(-1, temperature=0.0)]) == SPARSE
    assert _step(backend, [7], [_params(20)], n=1) == DEFAULT

    # Capture sets the route per variant; the next step of the same batch restores its own.
    assert _step(backend, [6], [_params(-1, temperature=0.0)]) == SPARSE
    backend.prepare_capture_variant(bs=2, num_tokens_per_req=N, variant=DEFAULT)
    assert backend.cuda_graph_replay_variant(N) == DEFAULT
    assert _step(backend, [6], [_params(-1, temperature=0.0)]) == SPARSE
    assert _step(backend, [3], [_params(-1)]) == DEFAULT
    backend.prepare_capture_variant(bs=2, num_tokens_per_req=N, variant=SPARSE)
    assert backend.cuda_graph_replay_variant(N) == SPARSE
    assert _step(backend, [3], [_params(-1)]) == DEFAULT
    with pytest.raises(ValueError, match="Unsupported CUDA graph variant"):
        backend.prepare_capture_variant(bs=2, num_tokens_per_req=N, variant="other")


@pytest.mark.parametrize(
    "server", ["speculative_sampling", "draft_tree", "flashinfer_full"]
)
def test_verifies_the_route_does_not_cover_capture_only_the_dense_graph(server):
    """Recorded draft probabilities, draft trees and the full backend's
    penalties keep every batch on whole probability rows."""
    if server == "flashinfer_full":
        from tokenspeed.runtime.sampling.backends.flashinfer_full import (
            FlashInferFullSamplingBackend,
        )

        backend = FlashInferFullSamplingBackend(_config())
    else:
        backend = _backend(
            enable_speculative_sampling=server == "speculative_sampling",
            draft_tree=server == "draft_tree",
        )
    assert backend.cuda_graph_capture_variants(N) == (DEFAULT,)
    assert _step(backend, [1], [_params(20)]) == DEFAULT


def _logits(rows: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    logits = torch.randn(rows, VOCAB, generator=gen) * 2
    hot = torch.randint(0, VOCAB, (rows, 8), generator=gen)
    logits.scatter_add_(1, hot, torch.rand(rows, 8, generator=gen) * 12)
    return logits.to("cuda", torch.bfloat16)


def _chains(logits: torch.Tensor, seed: int) -> torch.Tensor:
    """Drafts mostly the previous row's best token, else its second or third or anywhere."""
    gen = torch.Generator().manual_seed(seed)
    bs = logits.shape[0] // N
    best = torch.topk(logits.float().cpu(), 3, dim=1).indices.view(bs, N, 3)
    pick = torch.randint(1, 3, (bs, N, 1), generator=gen)
    pick = torch.where(torch.rand(bs, N, 1, generator=gen) < 0.8, 0, pick)
    chains = torch.gather(best, 2, pick).squeeze(2).roll(1, dims=1)
    anywhere = torch.randint(0, VOCAB, chains.shape, generator=gen)
    chains = torch.where(
        torch.rand(chains.shape, generator=gen) < 0.2, anywhere, chains
    )
    return chains.to("cuda", torch.int32)


def _mixed_params(bs: int, seed: int) -> list[SamplingParams]:
    gen = torch.Generator().manual_seed(seed)
    choice = lambda values: values[
        int(torch.randint(0, len(values), (1,), generator=gen))
    ]
    return [
        _params(
            choice([1, 5, 20, 32]),
            seed=1000 + i,
            temperature=choice([0.6, 1.0, 1.3]),
            top_p=choice([0.8, 0.95, 1.0]),
        )
        for i in range(bs)
    ]


def _verify(backend, logits, chains, pool, cache_lengths):
    lengths = torch.zeros(POOL + 1, dtype=torch.int32, device="cuda")
    lengths[torch.tensor(pool)] = torch.tensor(
        cache_lengths, dtype=torch.int32, device="cuda"
    )
    info = SamplingBatchInfo(
        req_pool_indices=torch.tensor(pool, device="cuda"),
        valid_cache_lengths=lengths,
        device="cuda",
    )
    predict, accept_length = backend.verify(
        LogitsProcessorOutput(next_token_logits=logits), info, chains, tree=None
    )
    tokens = predict.view(-1, N).tolist()
    return [row[:length] for row, length in zip(tokens, accept_length.tolist())]


def test_both_routes_commit_the_same_tokens():
    """Same logits, chains and seeds: the sparse and dense routes commit the
    same tokens for every request (mixed temperature, top-k 1 to 32, top-p)."""
    bs = 32
    backend = _backend()
    params = _mixed_params(bs, 1)
    admissions = list(range(1, bs + 1))
    pool = [a % POOL + 1 for a in admissions]
    assert _step(backend, admissions, params) == SPARSE
    logits = _logits(bs * N, 2)
    chains = _chains(logits, 3)
    cache_lengths = [100 + 7 * i for i in range(bs)]

    sparse = _verify(backend, logits, chains, pool, cache_lengths)
    backend.prepare_capture_variant(bs=bs, num_tokens_per_req=N, variant=DEFAULT)
    dense = _verify(backend, logits, chains, pool, cache_lengths)

    assert sparse == dense
    lengths = {len(tokens) for tokens in sparse}
    assert {1, N} <= lengths


@pytest.mark.parametrize("top_k", [1, 20])
def test_a_request_commits_the_same_tokens_at_any_batch_size_under_rl_bitwise(
    top_k, monkeypatch
):
    """rl-bitwise takes the sparse route too: one request alone and at position
    5 of 8 selects, accepts and samples identically."""
    monkeypatch.setitem(global_server_args_dict, "numerics", "rl-bitwise")
    probe = _params(top_k, seed=77, temperature=0.8, top_p=0.9)
    probe_logits = _logits(N, 11)
    probe_chain = _chains(probe_logits, 12)

    alone = _backend()
    assert _step(alone, [1], [probe]) == SPARSE
    want = _verify(alone, probe_logits, probe_chain, [2], [321])

    bs, position = 8, 5
    batched = _backend()
    params = _mixed_params(bs, 13)
    params[position] = probe
    admissions = list(range(10, 10 + bs))
    pool = [a % POOL + 1 for a in admissions]
    assert _step(batched, admissions, params) == SPARSE
    logits = _logits(bs * N, 14)
    logits[position * N : (position + 1) * N] = probe_logits
    chains = _chains(logits, 15)
    chains[position] = probe_chain[0]
    cache_lengths = [50 + i for i in range(bs)]
    cache_lengths[position] = 321
    got = _verify(batched, logits, chains, pool, cache_lengths)

    assert got[position] == want[0]

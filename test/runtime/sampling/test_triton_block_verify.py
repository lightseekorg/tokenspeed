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

"""Block verification on the triton sampler.

Under ``--speculative-rejection-sample-method block`` the triton verifier
gathers each request's recorded draft q by pool slot, builds the target law
from the request's temperature / top-k / top-p, draws the prefix and residual
coins from the request's seed and position, and runs the block verifier.
CPU: the coin kernel is replaced by a stand-in keyed the same way and the
block verifier runs its torch oracle; the GPU tests run the same verify on
device, eagerly and from a captured graph.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

# Mostly CPU; the device tests need a GPU.
register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed_kernel.numerics.torch.sampling import (  # noqa: E402
    torch_topk_topp_renorm,
)
from tokenspeed_kernel.ops.sampling.block_verify import (  # noqa: E402
    torch_block_verify,
)
from tokenspeed_kernel.ops.sampling.triton import TOP_K_DISABLED  # noqa: E402

from tokenspeed.runtime.layers.logits_processor import (  # noqa: E402
    LogitsProcessorOutput,
)
from tokenspeed.runtime.sampling.backends import triton as tb  # noqa: E402
from tokenspeed.runtime.sampling.backends.base import (  # noqa: E402
    SamplingBackendConfig,
)
from tokenspeed.runtime.sampling.sampling_batch_info import (  # noqa: E402
    SamplingBatchInfo,
)
from tokenspeed.runtime.sampling.sampling_params import SamplingParams  # noqa: E402
from tokenspeed.runtime.utils.env import global_server_args_dict  # noqa: E402

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")

POOL, VOCAB, MAX_BS, N = 8, 64, 4, 4
THRESHOLD = 1.5
SENTINEL = THRESHOLD + 1.0
SLOTS = [2, 5, 7]

ROUTES = {
    "no_filter": tb._SAMPLE_ROUTE_GUMBEL_NO_FILTER,
    "top_k": tb._SAMPLE_ROUTE_GUMBEL_TOP_K,
    "top_k_top_p": tb._SAMPLE_ROUTE_GUMBEL_TOP_K_TOP_P,
    "top_p": tb._SAMPLE_ROUTE_GUMBEL_TOP_P,
    "generic": tb._SAMPLE_ROUTE_GUMBEL_GENERIC,
}


@pytest.mark.parametrize(
    "route, top_k, top_p",
    [
        ("no_filter", [TOP_K_DISABLED] * 3, [1.0] * 3),
        ("top_k", [5, 1, 20], [1.0] * 3),
        ("top_k_top_p", [5, 1, 20], [0.9, 1.0, 0.5]),
        ("top_p", [TOP_K_DISABLED] * 3, [0.9, 1.0, 0.3]),
        ("generic", [TOP_K_DISABLED, 1, 7], [0.95, 1.0, 0.6]),
    ],
)
def test_target_law_is_the_filtered_softmax(route, top_k, top_p):
    gen = torch.Generator().manual_seed(0)
    logits = torch.randn(3, VOCAB, generator=gen) * 2
    temperature = torch.tensor([0.7, 1.0, 1.3])
    top_k = torch.tensor(top_k, dtype=torch.int32)
    top_p = torch.tensor(top_p)
    got = tb.block_target_probs(
        logits, temperature, top_k, top_p, sample_route=ROUTES[route], top_k_pad=32
    )
    expected = torch_topk_topp_renorm(
        torch.softmax(logits / temperature[:, None], dim=-1), top_k, top_p
    )
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("route", ["top_k", "generic"])
def test_greedy_rows_are_one_hot_at_the_lowest_index_argmax(route):
    # The token a greedy draft proposes, so exact greedy verify survives ties.
    logits = torch.zeros(2, VOCAB)
    logits[:, [9, 3]] = 5.0
    got = tb.block_target_probs(
        logits,
        torch.ones(2),
        torch.tensor([1, 1], dtype=torch.int32),
        torch.ones(2),
        sample_route=ROUTES[route],
        top_k_pad=32,
    )
    expected = torch.zeros(2, VOCAB)
    expected[:, 3] = 1.0
    assert torch.equal(got, expected)


def _config(device: str) -> SamplingBackendConfig:
    return SamplingBackendConfig(
        synthetic_acceptance_length=None,
        enable_speculative_sampling=True,
        sampling_stream="batch",
        logprob_order="torch",
        max_bs=MAX_BS,
        max_draft_tokens_per_req=N,
        max_req_pool_size=POOL,
        vocab_size=VOCAB,
        device=device,
        spec_reject_draft_prob_threshold=THRESHOLD,
    )


def _backend(monkeypatch, device: str = "cpu") -> tb.TritonSamplingBackend:
    # The rejection method is read from the server args, like ``numerics``.
    monkeypatch.setitem(
        global_server_args_dict, "speculative_rejection_sample_method", "block"
    )
    if device != "cpu":
        return tb.TritonSamplingBackend(_config(device))
    # The backend sizes a scratch buffer by the SM count; there is no GPU here.
    with patch.object(
        torch.cuda,
        "get_device_properties",
        return_value=SimpleNamespace(multi_processor_count=4),
    ):
        return tb.TritonSamplingBackend(_config(device))


def _keyed_coins(req_pool_indices, seed_pool, offsets_pool, coins, final_coins):
    """CPU stand-in for the coin kernel, keyed by seed and position alone."""
    for row, slot in enumerate(req_pool_indices.tolist()):
        key = int(seed_pool[slot]) * 1_000_003 + int(offsets_pool[slot])
        gen = torch.Generator().manual_seed(key)
        coins[row] = torch.rand(coins.shape[1], generator=gen)
        final_coins[row] = torch.rand((), generator=gen)


def _prepare(backend: tb.TritonSamplingBackend, device: str) -> torch.Tensor:
    """A sampled request (top-p), a greedy one and a temperature-only one."""
    params = [
        SamplingParams(temperature=0.8, top_p=0.9),
        SamplingParams(temperature=0.0),
        SamplingParams(temperature=1.2),
    ]
    rids = [f"r{i}" for i in range(len(params))]
    for rid, sp in zip(rids, params):
        sp.resolve_seed(rid)
        sp.normalize(None)
    backend.prepare_step(
        request_ids=rids,
        request_pool_indices=SLOTS,
        sampling_params_list=params,
        num_tokens_per_req=N,
    )
    return torch.tensor(SLOTS, dtype=torch.int64, device=device)


def _inputs(device: str):
    """Target logits, recorded draft q and candidates drawn from that q."""
    gen = torch.Generator().manual_seed(11)
    bs = len(SLOTS)
    logits = torch.randn(bs * N, VOCAB, generator=gen) * 2
    draft_probs = torch.full((POOL + 1, N, VOCAB), SENTINEL)
    draft_probs[:, -1] = 0.0
    candidates = torch.zeros(bs, N, dtype=torch.int64)
    candidates[:, 0] = torch.randint(VOCAB, (bs,), generator=gen)
    for row, slot in enumerate(SLOTS):
        # Draft logits near the target's, so long prefixes get accepted.
        draft_logits = logits[row * N : row * N + N - 1] + 0.5 * torch.randn(
            N - 1, VOCAB, generator=gen
        )
        q = torch.softmax(draft_logits, dim=-1)
        if row == 1:
            # The greedy request records one-hot q at its argmax drafts.
            q = torch.nn.functional.one_hot(q.argmax(-1), VOCAB).float()
        draft_probs[slot, :-1] = q
        candidates[row, 1:] = torch.multinomial(q, 1, generator=gen).view(-1)
    valid_cache_lengths = torch.arange(100, 100 + POOL + 1, dtype=torch.int32)
    return (
        logits.to(device),
        draft_probs.to(device),
        candidates.to(device),
        valid_cache_lengths.to(device),
    )


def _info(req_pool_indices, valid_cache_lengths, draft_probs) -> SamplingBatchInfo:
    return SamplingBatchInfo(
        req_pool_indices=req_pool_indices,
        valid_cache_lengths=valid_cache_lengths,
        draft_probs=draft_probs,
        vocab_size=VOCAB,
        device=req_pool_indices.device,
    )


def _expected(backend, logits, draft_probs, candidates, coins, final_coins):
    """The block verifier's oracle over the backend's pools, on the host."""
    bs = candidates.shape[0]
    rows = torch.tensor(SLOTS).repeat_interleave(N)
    pools = (backend._temperature_pool, backend._top_k_pool, backend._top_p_pool)
    target = tb.block_target_probs(
        logits,
        *(pool[rows.to(pool.device)] for pool in pools),
        sample_route=backend._sample_route,
        top_k_pad=backend._top_k_top_p_pad,
    )
    predicts = torch.full((bs * N,), -1, dtype=torch.int32)
    accept_index = torch.empty((bs, N), dtype=torch.int32)
    accept_length = torch.empty((bs,), dtype=torch.int32)
    torch_block_verify(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_length,
        candidates=candidates.cpu(),
        uniform_samples=coins.cpu(),
        uniform_samples_for_final_sampling=final_coins.cpu(),
        target_probs=target.cpu().view(bs, N, VOCAB).contiguous(),
        draft_probs=draft_probs.cpu()[SLOTS].contiguous(),
        reject_draft_prob_threshold=THRESHOLD,
    )
    return predicts, accept_index, accept_length + 1


def test_verify_runs_the_block_verifier_over_the_recorded_q(monkeypatch):
    monkeypatch.setattr(tb, "block_verify_coins_from_pools", _keyed_coins)
    backend = _backend(monkeypatch)
    req_pool_indices = _prepare(backend, "cpu")
    assert backend._sample_route == tb._SAMPLE_ROUTE_GUMBEL_GENERIC
    logits, draft_probs, candidates, valid_cache_lengths = _inputs("cpu")

    predict, accept_length = backend.verify(
        LogitsProcessorOutput(next_token_logits=logits.clone()),
        _info(req_pool_indices, valid_cache_lengths, draft_probs),
        candidates,
        tree=None,
    )

    bs = len(SLOTS)
    # The gather lands in the persistent buffer the captured graph records.
    assert torch.equal(backend._draft_probs_gather_buf[:bs], draft_probs[SLOTS])
    # Coins keyed by each request's seed and first verify position.
    offsets = torch.zeros(POOL + 1, dtype=torch.int64)
    offsets[SLOTS] = valid_cache_lengths[SLOTS].long() + 1
    coins, final_coins = torch.empty(bs, N), torch.empty(bs)
    _keyed_coins(req_pool_indices, backend._seed_pool, offsets, coins, final_coins)
    assert torch.equal(backend._block_coins[:bs], coins)
    assert torch.equal(backend._block_final_coins[:bs], final_coins)

    want_predict, want_index, want_length = _expected(
        backend, logits, draft_probs, candidates, coins, final_coins
    )
    accept_index = backend._accept_index_buf[: bs * N].view(bs, N)
    assert torch.equal(accept_length, want_length)
    assert torch.equal(accept_index, want_index)
    live = want_index >= 0
    assert torch.equal(predict.view(bs, N)[live], want_predict.view(bs, N)[live])
    # The greedy request keeps every draft that matches the target's argmax.
    greedy_drafts = candidates[1, 1:]
    target_argmax = logits[N : 2 * N - 1].argmax(-1)
    matched = int((greedy_drafts == target_argmax).long().cumprod(0).sum())
    assert int(accept_length[1]) == matched + 1


class _TargetOnly(Exception):
    pass


def test_without_recorded_q_the_target_only_rule_runs(monkeypatch):
    backend = _backend(monkeypatch)
    req_pool_indices = _prepare(backend, "cpu")
    logits, _, candidates, valid_cache_lengths = _inputs("cpu")

    def target_only(*args, **kwargs):
        raise _TargetOnly

    monkeypatch.setattr(backend, "_sample_verify_targets", target_only)
    with pytest.raises(_TargetOnly):
        backend.verify(
            LogitsProcessorOutput(next_token_logits=logits),
            _info(req_pool_indices, valid_cache_lengths, None),
            candidates,
            tree=None,
        )


def test_standard_rejection_allocates_no_block_buffers(monkeypatch):
    monkeypatch.setitem(
        global_server_args_dict, "speculative_rejection_sample_method", "standard"
    )
    with patch.object(
        torch.cuda,
        "get_device_properties",
        return_value=SimpleNamespace(multi_processor_count=4),
    ):
        backend = tb.TritonSamplingBackend(_config("cpu"))
    assert not backend._block_verify
    assert backend._draft_probs_gather_buf is None and backend._block_coins is None


@requires_gpu
def test_coins_are_keyed_by_seed_and_position():
    seed_pool = torch.tensor([0, 11, 22, 33], dtype=torch.int64, device="cuda")
    offsets_pool = torch.tensor([0, 500, 600, 500], dtype=torch.int64, device="cuda")

    def coins_of(slots):
        slots = torch.tensor(slots, dtype=torch.int32, device="cuda")
        coins = torch.empty(len(slots), N, device="cuda")
        final = torch.empty(len(slots), device="cuda")
        tb.block_verify_coins_from_pools(slots, seed_pool, offsets_pool, coins, final)
        return coins.cpu(), final.cpu()

    coins, final = coins_of([1, 2, 3])
    assert ((coins >= 0) & (coins < 1)).all() and ((final >= 0) & (final < 1)).all()
    # A request's coins do not depend on its batch row or its co-batch.
    alone, alone_final = coins_of([2])
    assert torch.equal(alone[0], coins[1]) and torch.equal(alone_final[0], final[1])
    # Another seed at the same position draws other coins.
    assert not torch.equal(coins[0], coins[2])
    # The residual coin is its own stream, not prefix coin 0.
    assert not torch.equal(final, coins[:, 0])


@requires_gpu
def test_device_verify_matches_the_host_oracle(monkeypatch):
    backend = _backend(monkeypatch, device="cuda")
    req_pool_indices = _prepare(backend, "cuda")
    logits, draft_probs, candidates, valid_cache_lengths = _inputs("cuda")

    predict, accept_length = backend.verify(
        LogitsProcessorOutput(next_token_logits=logits.clone()),
        _info(req_pool_indices, valid_cache_lengths, draft_probs),
        candidates,
        tree=None,
    )

    bs = len(SLOTS)
    want_predict, want_index, want_length = _expected(
        backend,
        logits,
        draft_probs,
        candidates,
        backend._block_coins[:bs].clone(),
        backend._block_final_coins[:bs].clone(),
    )
    accept_index = backend._accept_index_buf[: bs * N].view(bs, N).cpu()
    assert torch.equal(accept_length.cpu(), want_length)
    assert torch.equal(accept_index, want_index)
    live = want_index >= 0
    assert torch.equal(predict.view(bs, N).cpu()[live], want_predict.view(bs, N)[live])


@requires_gpu
def test_graph_replay_matches_eager(monkeypatch):
    backend = _backend(monkeypatch, device="cuda")
    req_pool_indices = _prepare(backend, "cuda")
    logits, draft_probs, candidates, valid_cache_lengths = _inputs("cuda")
    static_logits = logits.clone()
    info = _info(req_pool_indices, valid_cache_lengths, draft_probs)

    def step():
        return backend.verify(
            LogitsProcessorOutput(next_token_logits=static_logits),
            info,
            candidates,
            tree=None,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        predict, accept_length = step()

    bs = len(SLOTS)
    for scale in (1.0, 0.5):
        fresh = logits * scale
        static_logits.copy_(fresh)
        graph.replay()
        got = (predict.clone(), accept_length.clone())
        static_logits.copy_(fresh)
        want = step()
        assert torch.equal(got[1], want[1])
        live = backend._accept_index_buf[: bs * N].view(bs, N) >= 0
        assert torch.equal(got[0].view(bs, N)[live], want[0].view(bs, N)[live])

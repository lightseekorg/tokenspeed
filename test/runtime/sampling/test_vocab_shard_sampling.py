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

"""The triton sampler on vocab-sharded logits equals it on gathered rows.

Each of eight simulated ranks hands the sampler its raw LM-head shard; the
top-k routes select per shard (logit scale, softcap and grammar mask fused)
and gather only the candidates. Tokens, accept decisions and accept lengths
must be bitwise those of the same backend on the whole rows the logits
processor produces (gather, scale + softcap, then the in-place mask).
"""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=20, suite="runtime-1gpu")

from tokenspeed_kernel.ops.sampling import vocab_shard_topk  # noqa: E402
from tokenspeed_kernel.ops.sampling.triton.topk_topp import (  # noqa: E402
    _top_k_top_p_gumbel_stage2_kernel,
)
from tokenspeed_kernel.ops.sampling.triton.vocab_topk import (  # noqa: E402
    _vocab_shard_topk_kernel,
    _vocab_topk_merge_kernel,
)

from tokenspeed.runtime.grammar.base_grammar_backend import (  # noqa: E402
    get_apply_vocab_mask_func,
)
from tokenspeed.runtime.layers import sampled_vocab_shard as shard_module  # noqa: E402
from tokenspeed.runtime.layers.logits_processor import (  # noqa: E402
    LogitsProcessorOutput,
    fused_softcap_generic,
)
from tokenspeed.runtime.layers.sampled_vocab_shard import (  # noqa: E402
    SampledVocabShard,
)
from tokenspeed.runtime.sampling.backends.base import (  # noqa: E402
    SamplingBackendConfig,
)
from tokenspeed.runtime.sampling.backends.triton import (  # noqa: E402
    TritonSamplingBackend,
)
from tokenspeed.runtime.sampling.backends.triton_full import (  # noqa: E402
    TritonFullSamplingBackend,
)
from tokenspeed.runtime.sampling.sampling_batch_info import (  # noqa: E402
    SamplingBatchInfo,
)
from tokenspeed.runtime.sampling.sampling_params import SamplingParams  # noqa: E402
from tokenspeed.runtime.sampling.tree_verify import DraftTreeQ  # noqa: E402

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

RANKS = 8
SHARD = 2600
VOCAB = RANKS * SHARD
SCALE = 0.17677669529663687
SOFTCAP = 20.0
BS = 3
N = 8
POOL = 8


def _config() -> SamplingBackendConfig:
    return SamplingBackendConfig(
        synthetic_acceptance_length=None,
        enable_speculative_sampling=True,
        sampling_stream="batch",
        logprob_order="torch",
        max_bs=BS,
        max_draft_tokens_per_req=N,
        max_req_pool_size=POOL,
        vocab_size=VOCAB,
        device="cuda",
        tp_group=tuple(range(RANKS)),
        # One process stands in for the group: nothing to broadcast.
        enable_tp_sync=False,
    )


def _backend() -> TritonSamplingBackend:
    return TritonSamplingBackend(_config())


def _prepare(backend, top_k, top_p, num_tokens_per_req, rid="r") -> torch.Tensor:
    rids = [f"{rid}{i}" for i in range(BS)]
    params = []
    for rid, k, p in zip(rids, top_k, top_p):
        sp = SamplingParams(temperature=1.0, top_k=k, top_p=p)
        sp.resolve_seed(rid)
        sp.normalize(None)
        params.append(sp)
    pool_indices = [2, 5, 7]
    backend.prepare_step(
        request_ids=rids,
        request_pool_indices=pool_indices,
        sampling_params_list=params,
        num_tokens_per_req=num_tokens_per_req,
    )
    return torch.tensor(pool_indices, dtype=torch.int64, device="cuda")


def _info(req_pool_indices, mask, draft_q) -> SamplingBatchInfo:
    return SamplingBatchInfo(
        req_pool_indices=req_pool_indices,
        valid_cache_lengths=torch.arange(
            1000, 1000 + POOL + 1, dtype=torch.int32, device="cuda"
        ),
        draft_q=draft_q,
        vocab_mask=mask,
        apply_vocab_mask=get_apply_vocab_mask_func("xgrammar"),
        vocab_size=VOCAB,
        device="cuda",
    )


class _Group:
    """Eight ranks' raw shards; rank ``rank`` is the one sampling."""

    def __init__(self, rows: int, seed: int, masked: bool):
        gen = torch.Generator().manual_seed(seed)
        raw = torch.randn(rows, VOCAB, generator=gen) * 3
        # One dominant token per row, so drafts of it are mostly accepted.
        peaks = torch.randint(0, VOCAB, (rows,), generator=gen)
        raw[torch.arange(rows), peaks] += 80
        self.raw = raw.to("cuda", torch.bfloat16)
        self.mask = None
        allowed = torch.ones(rows, VOCAB, dtype=torch.bool)
        if masked:
            allowed = torch.rand(rows, VOCAB, generator=gen) < 0.5
            allowed[torch.arange(rows), peaks] = True
            bits = allowed.view(rows, -1, 32).to(torch.int64) << torch.arange(32)
            bits = bits.sum(-1)
            self.mask = torch.where(bits >= 2**31, bits - 2**32, bits).to(
                "cuda", torch.int32
            )
        # The logits processor's whole rows: gathered, then scaled and capped.
        self.whole = self.raw.clone()
        fused_softcap_generic(self.whole, SOFTCAP, SCALE)
        # What the sampler draws from once the mask applies.
        self.allowed_whole = self.whole.float().masked_fill(
            ~allowed.cuda(), float("-inf")
        )

    def shard(self, rank: int) -> SampledVocabShard:
        return SampledVocabShard(
            logits=self.raw[:, rank * SHARD : (rank + 1) * SHARD],
            vocab_start=rank * SHARD,
            logit_scale=SCALE,
            softcap=SOFTCAP,
            tp_size=RANKS,
            tp_group=tuple(range(RANKS)),
            gather_full=lambda logits: self.whole.clone(),
        )

    def all_gather_single(self, landing, staged, group):
        """Every rank's packed pairs, rank-major, as the group would send."""
        rows, width = staged.shape
        for peer in range(RANKS):
            vocab_shard_topk(
                self.raw[:rows, peer * SHARD : (peer + 1) * SHARD],
                width // 2,
                SCALE,
                peer * SHARD,
                landing[peer * rows : (peer + 1) * rows],
                softcap=SOFTCAP,
                vocab_mask=None if self.mask is None else self.mask[:rows],
            )


def _draft_q(group: _Group, req_pool_indices: torch.Tensor) -> DraftTreeQ:
    """q over each step's 16 best whole-row tokens; request 0 records none."""
    gen = torch.Generator().manual_seed(3)
    q = DraftTreeQ.zeros(POOL + 1, N - 1, 16, N, "cuda")
    best = torch.topk(group.allowed_whole, 16, dim=-1).indices
    for b, pool in enumerate(req_pool_indices.tolist()):
        if b == 0:
            continue
        weights = torch.rand(N - 1, 16, generator=gen) ** 4
        q.q[pool] = (weights / weights.sum(-1, keepdim=True)).cuda()
        # Row r verifies draft r + 1 against q row r.
        q.q_ids[pool] = best[b * N : (b + 1) * N - 1].to(torch.int32)
        q.slot[pool] = torch.arange(-1, N - 1, dtype=torch.int32, device="cuda")
        # Request 1 drafts each row's most likely token, so its window runs
        # long; request 2 drafts lower ones too.
        lanes = torch.randint(0, 3, (N - 1,), generator=gen) * (b - 1)
        q.pos[pool, 1:] = lanes.to("cuda", torch.int32)
    return q


def _candidates(q: DraftTreeQ, req_pool_indices: torch.Tensor) -> torch.Tensor:
    candidates = torch.zeros(BS, N, dtype=torch.int32, device="cuda")
    for b, pool in enumerate(req_pool_indices.tolist()):
        lanes = q.pos[pool, 1:].long()
        candidates[b, 1:] = q.q_ids[pool, torch.arange(N - 1), lanes]
    return candidates


@requires_cuda
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("with_q", [False, True])
@pytest.mark.parametrize(
    "top_k,top_p", [([64, 64, 64], [1.0] * 3), ([1, 30, 128], [1.0, 0.9, 0.8])]
)
def test_verify_on_shards_matches_whole_rows(
    monkeypatch, masked, with_q, top_k, top_p
) -> None:
    rows = BS * N
    group = _Group(rows, 11, masked)
    monkeypatch.setattr(shard_module, "all_gather_single", group.all_gather_single)
    results = []
    for logits_output in (
        LogitsProcessorOutput(next_token_logits=group.whole.clone()),
        LogitsProcessorOutput(next_token_logits=None, vocab_shard=group.shard(5)),
    ):
        backend = _backend()
        req_pool_indices = _prepare(backend, top_k, top_p, N)
        q = _draft_q(group, req_pool_indices) if with_q else None
        candidates = _candidates(_draft_q(group, req_pool_indices), req_pool_indices)
        predict, accept_length = backend.verify(
            logits_output,
            _info(req_pool_indices, group.mask, q),
            candidates,
            tree=None,
        )
        results.append((predict.clone(), accept_length.clone()))
    assert torch.equal(results[1][0], results[0][0])
    assert torch.equal(results[1][1], results[0][1])
    # The windows were neither all accepted nor all rejected.
    lengths = results[0][1].tolist()
    assert min(lengths) < N and max(lengths) > 1


@requires_cuda
@pytest.mark.parametrize("masked", [False, True])
def test_sample_on_shards_matches_whole_rows(monkeypatch, masked) -> None:
    group = _Group(BS, 21, masked)
    monkeypatch.setattr(shard_module, "all_gather_single", group.all_gather_single)
    results = []
    for logits_output in (
        LogitsProcessorOutput(next_token_logits=group.whole.clone()),
        LogitsProcessorOutput(next_token_logits=None, vocab_shard=group.shard(2)),
    ):
        backend = _backend()
        req_pool_indices = _prepare(backend, [64, 7, 128], [1.0, 0.9, 1.0], 1)
        sampled, _ = backend.sample(
            logits_output, _info(req_pool_indices, group.mask, None)
        )
        results.append(sampled.clone())
    assert torch.equal(results[1], results[0])


@requires_cuda
def test_other_routes_gather_whole_rows(monkeypatch) -> None:
    """Without a finite top-k a route reads every logit, so the shard gathers."""
    group = _Group(BS, 31, False)
    gathered = []

    def gather_full(logits):
        gathered.append(logits.shape)
        return group.whole.clone()

    shard = dataclasses.replace(group.shard(0), gather_full=gather_full)
    backend = _backend()
    req_pool_indices = _prepare(backend, [-1] * BS, [1.0] * BS, 1)
    sampled, _ = backend.sample(
        LogitsProcessorOutput(next_token_logits=None, vocab_shard=shard),
        _info(req_pool_indices, None, None),
    )
    assert gathered == [(BS, SHARD)]
    assert bool(((sampled >= 0) & (sampled < VOCAB)).all())


@requires_cuda
def test_full_backend_keeps_whole_rows_under_tp() -> None:
    # Its sample() and verify() read next_token_logits; shard selection would
    # leave the processor's whole rows unset.
    assert TritonSamplingBackend(_config()).vocab_shard_top_k() is not None
    assert TritonFullSamplingBackend(_config()).vocab_shard_top_k() is None


def _kernel_test_utils():
    helper = Path(__file__).resolve().parents[3] / "tokenspeed-kernel/test/utils.py"
    spec = importlib.util.spec_from_file_location("kernel_test_utils", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@requires_cuda
@pytest.mark.parametrize("with_q,shard", [(False, 3000), (True, 900)])
def test_warmup_compiles_every_candidate_count(monkeypatch, with_q, shard) -> None:
    """After the startup warmup, sample and verify compile nothing at either
    candidate count, with or without a grammar mask.

    Graph capture records 128 candidates per row, but an eager step whose
    top_k all fit in 64 selects 64. Each case's head (four ranks, no scale or
    softcap, its own shard width) differs from every other test's head, so
    only that case compiles its selection kernels.
    """
    ranks = 4
    vocab = ranks * shard
    raw = (
        torch.randn(BS * N, vocab, generator=torch.Generator().manual_seed(41)) * 3
    ).to("cuda", torch.bfloat16)

    def shard_of(rows: int) -> SampledVocabShard:
        return SampledVocabShard(
            logits=raw[:rows, shard : 2 * shard],
            vocab_start=shard,
            logit_scale=None,
            softcap=None,
            tp_size=ranks,
            tp_group=tuple(range(ranks)),
            gather_full=lambda logits: raw[:rows].clone(),
        )

    def all_gather_single(landing, staged, group):
        rows, width = staged.shape
        for peer in range(ranks):
            vocab_shard_topk(
                raw[:rows, peer * shard : (peer + 1) * shard],
                width // 2,
                None,
                peer * shard,
                landing[peer * rows : (peer + 1) * rows],
                softcap=None,
                vocab_mask=None,
            )

    monkeypatch.setattr(shard_module, "all_gather_single", all_gather_single)
    backend = TritonSamplingBackend(
        dataclasses.replace(_config(), vocab_size=vocab, tp_group=tuple(range(ranks)))
    )
    q = DraftTreeQ.zeros(POOL + 1, N - 1, 16, N, "cuda") if with_q else None
    backend.warmup_vocab_shard_sampling(
        shard_of(N),
        torch.zeros((1, N), dtype=torch.int32, device="cuda"),
        draft_q=q,
        with_vocab_mask=True,
        draft_probs=None,
    )
    # Every token allowed.
    mask = torch.full(
        (BS * N, (vocab + 31) // 32), -1, dtype=torch.int32, device="cuda"
    )
    kernels = (
        _vocab_shard_topk_kernel,
        _vocab_topk_merge_kernel,
        _top_k_top_p_gumbel_stage2_kernel,
    )
    with _kernel_test_utils().assert_no_triton_compile(*kernels):
        for step, top_k in enumerate(([5, 64, 1], [100, 3, 128])):
            for masked in (False, True):
                # New requests each time, so the backend picks the count again.
                rid = f"s{step}m{int(masked)}r"
                req_pool_indices = _prepare(backend, top_k, [0.9, 1.0, 1.0], 1, rid)
                info = _info(req_pool_indices, mask[:BS] if masked else None, None)
                info.vocab_size = vocab
                sampled, _ = backend.sample(
                    LogitsProcessorOutput(
                        next_token_logits=None, vocab_shard=shard_of(BS)
                    ),
                    info,
                )
                assert bool(((sampled >= 0) & (sampled < vocab)).all())
                req_pool_indices = _prepare(backend, top_k, [0.9, 1.0, 1.0], N, rid)
                info = _info(req_pool_indices, mask if masked else None, q)
                info.vocab_size = vocab
                _, accept_length = backend.verify(
                    LogitsProcessorOutput(
                        next_token_logits=None, vocab_shard=shard_of(BS * N)
                    ),
                    info,
                    torch.zeros((BS, N), dtype=torch.int32, device="cuda"),
                    tree=None,
                )
                assert bool(((accept_length >= 1) & (accept_length <= N)).all())

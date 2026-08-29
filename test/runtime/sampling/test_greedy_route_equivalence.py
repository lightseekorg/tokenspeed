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

"""Greedy-route equivalence gate for the unified sampling path.

Greedy requests normalize to ``top_k=1`` (``SamplingParams.__post_init__``),
and the CUDA graph always captures the pool-indexed sampling route, so the
pool route with ``top_k=1`` IS the greedy path. These tests pin that
equivalence numerically — sample() and verify() through the pool kernels
must match an argmax reference exactly — which is what licenses removing
the eager-only ``is_all_greedy`` argmax branch.

Tie rows are asserted separately: the pool kernels and argmax may break a
tie differently (both answers are correct greedy outputs), so exact-match
rows use tie-free logits and tie rows only require membership in the
argmax-equivalent set.
"""

from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.sampling.backends.base import SamplingBackendConfig
from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
from tokenspeed.runtime.sampling.sampling_params import SamplingParams

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

VOCAB = 1024
POOL = 8
MAX_BS = 4
MAX_N = 4


def _make_config() -> SamplingBackendConfig:
    return SamplingBackendConfig(
        max_bs=MAX_BS,
        max_draft_tokens_per_req=MAX_N,
        max_req_pool_size=POOL,
        vocab_size=VOCAB,
        device="cuda",
    )


def _greedy_sp(rid: str) -> SamplingParams:
    # temperature=0 normalizes to top_k=1 + temperature=1.0 — the pool
    # route's greedy spelling.
    sp = SamplingParams(temperature=0.0, top_k=-1, top_p=1.0)
    sp.resolve_seed(rid)
    sp.normalize(None)
    return sp


def _prepare_backend(backend, bs: int, num_tokens_per_req: int = 1):
    rids = [f"greedy_{i}" for i in range(bs)]
    pool_indices = list(range(bs))
    backend.prepare_step(
        request_ids=rids,
        request_pool_indices=pool_indices,
        sampling_params_list=[_greedy_sp(r) for r in rids],
        num_tokens_per_req=num_tokens_per_req,
    )
    return torch.tensor(pool_indices, dtype=torch.int64, device="cuda")


def _tie_free_logits(rows: int) -> torch.Tensor:
    logits = torch.randn(rows, VOCAB, device="cuda", dtype=torch.float32)
    # Distinct per-element jitter keeps every row's max unique so the
    # pool kernels and argmax cannot disagree via tie-breaking.
    logits += torch.arange(VOCAB, device="cuda", dtype=torch.float32) * 1e-4
    return logits


def _sampling_info(req_pool_indices: torch.Tensor) -> SamplingBatchInfo:
    return SamplingBatchInfo(
        req_pool_indices=req_pool_indices,
        vocab_size=VOCAB,
        device="cuda",
    )


def _logits_output(logits: torch.Tensor):
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput

    return LogitsProcessorOutput(next_token_logits=logits)


def _backends():
    from tokenspeed.runtime.sampling.backends.flashinfer import (
        FlashInferSamplingBackend,
    )
    from tokenspeed.runtime.sampling.backends.triton import TritonSamplingBackend

    return {
        "triton": TritonSamplingBackend,
        "flashinfer": FlashInferSamplingBackend,
    }


@requires_cuda
@pytest.mark.parametrize("backend_name", ["triton", "flashinfer"])
def test_sample_top_k1_equals_argmax(backend_name):
    torch.manual_seed(7)
    backend = _backends()[backend_name](_make_config())
    bs = MAX_BS
    req_pool_indices = _prepare_backend(backend, bs)

    logits = _tie_free_logits(bs)
    expected = torch.argmax(logits, dim=-1).to(torch.int32)

    sampled, lengths = backend.sample(
        _logits_output(logits.clone()), _sampling_info(req_pool_indices)
    )

    torch.testing.assert_close(sampled.cpu(), expected.cpu())
    torch.testing.assert_close(lengths.cpu(), torch.ones(bs, dtype=torch.int32))

    if backend_name == "flashinfer":
        # sample() must land its outputs in the packed region so the
        # single-D2H fast path fires on the eager path too.
        packed = backend.get_packed_output_d2h(sampled, lengths)
        assert packed is not None, "sample() outputs must alias _output_pack_buf"
        torch.testing.assert_close(packed[0], expected.cpu())


@requires_cuda
@pytest.mark.parametrize("backend_name", ["triton", "flashinfer"])
def test_sample_top_k1_tie_rows_pick_a_max(backend_name):
    torch.manual_seed(11)
    backend = _backends()[backend_name](_make_config())
    bs = MAX_BS
    req_pool_indices = _prepare_backend(backend, bs)

    logits = torch.full((bs, VOCAB), -10.0, device="cuda", dtype=torch.float32)
    tie_cols = torch.tensor([3, 17, 256, VOCAB - 1], device="cuda")
    logits[:, tie_cols] = 5.0  # manufactured 4-way tie per row

    sampled, _ = backend.sample(
        _logits_output(logits.clone()), _sampling_info(req_pool_indices)
    )

    assert torch.isin(sampled, tie_cols.to(torch.int32)).all(), (
        f"{backend_name} greedy tie row escaped the argmax-equivalent set: "
        f"{sampled.cpu().tolist()}"
    )


def _chain_greedy_reference(
    candidates: torch.Tensor, target_argmax: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequential greedy verify: row-i target token t_j replaces candidate
    j+1; accept while the next candidate equals the current target token."""
    bs, n = candidates.shape
    predict = torch.zeros(bs, n, dtype=torch.int32)
    accept_length = torch.zeros(bs, dtype=torch.int32)
    for b in range(bs):
        acc = 0
        for j in range(n):
            predict[b, j] = target_argmax[b, j]
            if j + 1 < n and candidates[b, j + 1] == target_argmax[b, j]:
                acc += 1
            else:
                break
        accept_length[b] = acc + 1  # backends report accepted+1 (bonus token)
    return predict, accept_length


@requires_cuda
@pytest.mark.parametrize("backend_name", ["triton", "flashinfer"])
def test_verify_top_k1_equals_greedy_chain(backend_name):
    torch.manual_seed(13)
    backend = _backends()[backend_name](_make_config())
    bs, n = MAX_BS, MAX_N
    req_pool_indices = _prepare_backend(backend, bs, num_tokens_per_req=n)

    logits = _tie_free_logits(bs * n)
    target_argmax = torch.argmax(logits, dim=-1).view(bs, n).to(torch.int32)

    # Mix of full-accept, partial-accept and zero-accept chains.
    candidates = torch.randint(0, VOCAB, (bs, n), device="cuda", dtype=torch.int32)
    candidates[0, 1:] = target_argmax[0, :-1]  # accept all
    candidates[1, 1] = target_argmax[1, 0]  # accept one
    # rows 2..: random → likely zero accepts

    predict, accept_length = backend.verify(
        _logits_output(logits.clone()),
        _sampling_info(req_pool_indices),
        candidates,
    )

    ref_predict, ref_accept = _chain_greedy_reference(
        candidates.cpu(), target_argmax.cpu()
    )

    torch.testing.assert_close(accept_length.cpu(), ref_accept)
    predict = predict.view(bs, n).cpu()
    for b in range(bs):
        acc = int(ref_accept[b])
        torch.testing.assert_close(predict[b, :acc], ref_predict[b, :acc])

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

"""verify() at N == 1 IS non-speculative decode sampling.

The unified sampling rule (prefill rows sample, decode rows verify) treats
non-speculative serving as the one-column candidate window: no candidate to
accept, one target token sampled, ``accept_length == 1``. These tests pin
the equivalence that licenses routing non-spec decode through verify():

* triton: verify's target draw is the SAME gumbel pool kernel with the same
  seed/offset pools sample() uses — bitwise identical tokens, greedy and
  stochastic alike;
* flashinfer: greedy is bitwise identical (top_k=1 renorm collapses to a
  single atom, the coin is irrelevant). Stochastic draws from the same
  renormalized distribution through the per-slot coin stream instead of the
  philox stream — per-request deterministic, same distribution, different
  draws — so it is asserted for validity and accept shape, not token
  equality.
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

VOCAB = 512
POOL = 8
MAX_BS = 4


def _make_config() -> SamplingBackendConfig:
    return SamplingBackendConfig(
        max_bs=MAX_BS,
        max_draft_tokens_per_req=1,
        max_req_pool_size=POOL,
        vocab_size=VOCAB,
        device="cuda",
    )


def _sp(rid: str, greedy: bool) -> SamplingParams:
    sp = (
        SamplingParams(temperature=0.0, top_k=-1, top_p=1.0)
        if greedy
        else SamplingParams(temperature=0.8, top_k=40, top_p=0.95)
    )
    sp.resolve_seed(rid)
    sp.normalize(None)
    return sp


def _prepare(backend, bs: int, greedy: bool):
    rids = [f"n1_{i}" for i in range(bs)]
    backend.prepare_step(
        request_ids=rids,
        request_pool_indices=list(range(bs)),
        sampling_params_list=[_sp(r, greedy) for r in rids],
        num_tokens_per_req=1,
    )
    return torch.tensor(list(range(bs)), dtype=torch.int64, device="cuda")


def _info(req_pool_indices, offsets=None) -> SamplingBatchInfo:
    return SamplingBatchInfo(
        req_pool_indices=req_pool_indices,
        valid_cache_lengths=offsets,
        vocab_size=VOCAB,
        device="cuda",
    )


def _logits_output(logits):
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput

    return LogitsProcessorOutput(next_token_logits=logits)


def _backend(name: str):
    if name == "triton":
        from tokenspeed.runtime.sampling.backends.triton import TritonSamplingBackend

        return TritonSamplingBackend(_make_config())
    from tokenspeed.runtime.sampling.backends.flashinfer import (
        FlashInferSamplingBackend,
    )

    return FlashInferSamplingBackend(_make_config())


@requires_cuda
@pytest.mark.parametrize("backend_name", ["triton", "flashinfer"])
@pytest.mark.parametrize("greedy", [True, False], ids=["greedy", "stochastic"])
def test_verify_n1_matches_sample(backend_name, greedy):
    if backend_name == "flashinfer" and not greedy:
        pytest.skip("flashinfer stochastic verify draws a different (valid) stream")
    torch.manual_seed(31)
    bs = MAX_BS
    # Pool-indexed philox offsets: both calls must consume identical ones.
    offsets = torch.arange(100, 100 + POOL + 1, dtype=torch.int32, device="cuda")

    backend = _backend(backend_name)
    req = _prepare(backend, bs, greedy)
    logits = torch.randn(bs, VOCAB, device="cuda", dtype=torch.float32)
    sampled, ones = backend.sample(_logits_output(logits.clone()), _info(req, offsets))
    sampled = sampled.clone()

    backend2 = _backend(backend_name)
    req2 = _prepare(backend2, bs, greedy)
    candidates = torch.randint(0, VOCAB, (bs, 1), dtype=torch.int32, device="cuda")
    predict, accept = backend2.verify(
        _logits_output(logits.clone()), _info(req2, offsets), candidates
    )

    torch.testing.assert_close(predict.view(bs).cpu(), sampled.view(bs).cpu())
    torch.testing.assert_close(accept.cpu(), torch.ones(bs, dtype=torch.int32))
    torch.testing.assert_close(ones.cpu(), torch.ones(bs, dtype=torch.int32))


@requires_cuda
def test_verify_n1_flashinfer_stochastic_is_valid():
    torch.manual_seed(37)
    bs = MAX_BS
    backend = _backend("flashinfer")
    req = _prepare(backend, bs, greedy=False)
    logits = torch.randn(bs, VOCAB, device="cuda", dtype=torch.float32)
    candidates = torch.randint(0, VOCAB, (bs, 1), dtype=torch.int32, device="cuda")

    predict, accept = backend.verify(
        _logits_output(logits.clone()), _info(req), candidates
    )
    predict = predict.view(bs)
    assert ((predict >= 0) & (predict < VOCAB)).all()
    torch.testing.assert_close(accept.cpu(), torch.ones(bs, dtype=torch.int32))
    # Per-request determinism: the same slots and coins reproduce the draw.
    backend2 = _backend("flashinfer")
    req2 = _prepare(backend2, bs, greedy=False)
    predict2, _ = backend2.verify(
        _logits_output(logits.clone()), _info(req2), candidates
    )
    torch.testing.assert_close(predict.cpu(), predict2.view(bs).cpu())


@requires_cuda
@pytest.mark.parametrize("backend_name", ["triton", "flashinfer"])
def test_verify_n1_ignores_candidate_content(backend_name):
    """The one-column window has no acceptable candidate: its content must
    not influence the output token."""
    torch.manual_seed(41)
    bs = MAX_BS
    logits = torch.randn(bs, VOCAB, device="cuda", dtype=torch.float32)

    outs = []
    for fill in (0, VOCAB - 1):
        backend = _backend(backend_name)
        req = _prepare(backend, bs, greedy=True)
        candidates = torch.full((bs, 1), fill, dtype=torch.int32, device="cuda")
        predict, accept = backend.verify(
            _logits_output(logits.clone()), _info(req), candidates
        )
        torch.testing.assert_close(accept.cpu(), torch.ones(bs, dtype=torch.int32))
        outs.append(predict.view(bs).clone())
    torch.testing.assert_close(outs[0].cpu(), outs[1].cpu())

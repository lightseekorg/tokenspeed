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

"""Verify wiring for draft-prob rejection sampling.

With ``SamplingBatchInfo.draft_probs`` set, the FlashInfer backends gather the
recorded distributions by pool index into their persistent buffer and run the
chain kernel's draft-prob rule; without it the target-only call is unchanged.
CPU: the sampling kernels are replaced by torch stand-ins that record their
arguments.
"""

from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
from tokenspeed.runtime.sampling.backends import flashinfer as fi
from tokenspeed.runtime.sampling.backends import flashinfer_full as ff
from tokenspeed.runtime.sampling.backends.base import SamplingBackendConfig
from tokenspeed.runtime.sampling.backends.greedy import GreedySamplingBackend
from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo

POOL, VOCAB, MAX_BS, N = 6, 16, 4, 3


def _config(enable: bool) -> SamplingBackendConfig:
    return SamplingBackendConfig(
        enable_speculative_sampling=enable,
        sampling_stream="batch",
        logprob_order="torch",
        max_bs=MAX_BS,
        max_draft_tokens_per_req=N,
        max_req_pool_size=POOL,
        vocab_size=VOCAB,
        device="cpu",
        spec_reject_draft_prob_threshold=1.5,
    )


def _fake_kernels(monkeypatch, module, seen: dict) -> None:
    def gather(
        index, *, temperature, top_k, top_p, min_p=None, seed=None, offsets=None, n=1
    ):
        idx = index.repeat_interleave(n)
        return (
            temperature[idx],
            top_k[idx],
            top_p[idx],
            None if min_p is None else min_p[idx],
            None,
            None,
        )

    def chain(**kwargs):
        seen.update(kwargs)
        kwargs["accept_token_num"].zero_()

    monkeypatch.setattr(module, "_FUSED_TOPK_TOPP_AVAILABLE", False)
    monkeypatch.setattr(module, "gather_and_expand_scalars", gather)
    monkeypatch.setattr(
        module, "softmax", lambda logits, temperature: torch.softmax(logits.float(), -1)
    )
    monkeypatch.setattr(module, "top_k_renorm_prob", lambda probs, top_ks: probs)
    monkeypatch.setattr(
        module, "top_p_renorm_prob", lambda probs, top_ps, is_deterministic: probs
    )
    monkeypatch.setattr(module, "chain_speculative_sampling_target_only", chain)
    if module is ff:
        monkeypatch.setattr(module, "min_p_renorm_prob", lambda probs, min_ps: probs)


def _info(draft_probs: torch.Tensor | None) -> SamplingBatchInfo:
    return SamplingBatchInfo(
        req_pool_indices=torch.tensor([4, 1], dtype=torch.int64),
        valid_cache_lengths=torch.zeros(POOL + 1, dtype=torch.int32),
        draft_probs=draft_probs,
        vocab_size=VOCAB,
        device="cpu",
    )


def _recorded_draft_probs() -> torch.Tensor:
    probs = torch.full((POOL + 1, N, VOCAB), 2.5)
    probs[:, -1] = 0.0
    probs[4, :-1] = torch.softmax(torch.randn(N - 1, VOCAB), -1)
    probs[1, :-1] = torch.softmax(torch.randn(N - 1, VOCAB), -1)
    return probs


@pytest.mark.parametrize("module", [fi, ff], ids=["flashinfer", "flashinfer_full"])
def test_verify_gathers_recorded_rows_and_selects_the_draft_prob_rule(
    monkeypatch, module
):
    seen: dict = {}
    _fake_kernels(monkeypatch, module, seen)
    backend_cls = (
        fi.FlashInferSamplingBackend
        if module is fi
        else ff.FlashInferFullSamplingBackend
    )
    backend = backend_cls(_config(enable=True))
    assert backend._draft_probs_gather_buf.shape == (MAX_BS, N, VOCAB)

    torch.manual_seed(0)
    recorded = _recorded_draft_probs()
    logits = torch.randn(2 * N, VOCAB)
    candidates = torch.randint(0, VOCAB, (2, N), dtype=torch.int64)
    backend.verify(
        LogitsProcessorOutput(next_token_logits=logits),
        _info(recorded),
        candidates,
        tree=None,
    )

    assert seen["use_draft_prob"] is True
    assert seen["reject_draft_prob_threshold"] == 1.5
    draft = seen["draft_probs"]
    assert draft.shape == (2, N, VOCAB)
    assert torch.equal(draft, recorded[[4, 1]])
    # The gather lands in the persistent buffer the captured graph records.
    assert draft.data_ptr() == backend._draft_probs_gather_buf.data_ptr()
    assert seen["target_probs"].shape == (2, N, VOCAB)

    # Without recorded distributions the target-only call is unchanged.
    seen.clear()
    backend.verify(
        LogitsProcessorOutput(next_token_logits=logits),
        _info(None),
        candidates,
        tree=None,
    )
    assert seen["use_draft_prob"] is False and seen["draft_probs"] is None


def test_verify_refuses_draft_probs_on_a_backend_built_without_the_flag(monkeypatch):
    seen: dict = {}
    _fake_kernels(monkeypatch, fi, seen)
    backend = fi.FlashInferSamplingBackend(_config(enable=False))
    assert backend._draft_probs_gather_buf is None
    logits = torch.randn(2 * N, VOCAB)
    candidates = torch.zeros(2, N, dtype=torch.int64)
    with pytest.raises(RuntimeError, match="without enable_speculative_sampling"):
        backend.verify(
            LogitsProcessorOutput(next_token_logits=logits),
            _info(_recorded_draft_probs()),
            candidates,
            tree=None,
        )
    assert "use_draft_prob" not in seen


def test_gather_refuses_a_chain_width_the_buffer_was_not_sized_for(monkeypatch):
    backend = fi.FlashInferSamplingBackend(_config(enable=True))
    with pytest.raises(RuntimeError, match="geometry"):
        backend._gather_draft_probs(
            torch.zeros(POOL + 1, N + 1, VOCAB), torch.tensor([0, 1]), 2, N + 1
        )


def test_speculative_sampling_pools_are_the_verifier_pools():
    backend = fi.FlashInferSamplingBackend(_config(enable=True))
    pools = backend.speculative_sampling_pools()
    assert pools.temperature is backend._temperature_pool
    assert pools.top_k is backend._top_k_pool
    assert pools.seed is backend._seed_pool
    with pytest.raises(NotImplementedError, match="no draft-prob verify"):
        GreedySamplingBackend(_config(enable=False)).speculative_sampling_pools()


def test_sampling_batch_info_slices_keep_the_pool_indexed_draft_probs():
    recorded = _recorded_draft_probs()
    info = SamplingBatchInfo(
        req_pool_indices=torch.arange(4), draft_probs=recorded, device="cpu"
    )
    tail = info[2:]
    assert tail.draft_probs is recorded
    assert tail.req_pool_indices.tolist() == [2, 3]

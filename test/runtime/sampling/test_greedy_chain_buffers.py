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

"""Greedy verification overwrites stale accepted paths in persistent buffers."""

import pytest
import torch

from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
from tokenspeed.runtime.sampling.backends import greedy
from tokenspeed.runtime.sampling.backends.base import SamplingBackendConfig
from tokenspeed.runtime.sampling.backends.greedy import GreedySamplingBackend
from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA/ROCm"
            ),
        ),
    ],
)
@pytest.mark.parametrize("bs", [1, 4, 32])
@pytest.mark.parametrize("width", [1, 6, 9])
def test_verify_refreshes_complete_accepted_path(device, bs, width):
    backend = GreedySamplingBackend(
        SamplingBackendConfig(
            synthetic_acceptance_length=None,
            enable_speculative_sampling=False,
            sampling_stream="batch",
            logprob_order="torch",
            max_bs=bs,
            max_draft_tokens_per_req=width,
            max_req_pool_size=bs + 1,
            vocab_size=64,
            device=device,
        )
    )
    target = (
        torch.arange(bs * width, device=device, dtype=torch.int32).view(bs, width) % 64
    )
    candidates = target.clone()
    candidates[:, 1:] = target[:, :-1]
    info = SamplingBatchInfo(vocab_size=64, device=device)
    logits = LogitsProcessorOutput(
        next_token_logits=torch.empty(bs * width, 64, device=device),
        # Match argmax output and the CUDA verifier's target-ID contract.
        next_token_ids=target.flatten().to(torch.int64),
    )
    for accepted in (width - 1, 0, (width - 1) // 2):
        candidates[:, 1:] = target[:, :-1]
        if accepted < width - 1:
            candidates[:, accepted + 1] = -1
        tokens, lengths = backend.verify(logits, info, candidates, tree=None)
        torch.testing.assert_close(
            lengths,
            torch.full((bs,), accepted + 1, device=device, dtype=torch.int32),
            rtol=0,
            atol=0,
        )
        expected = torch.arange(bs * width, device=device, dtype=torch.int32).view(
            bs, width
        )
        expected[:, accepted + 1 :] = -1
        torch.testing.assert_close(
            backend.accepted_path(bs, width), expected, rtol=0, atol=0
        )
        torch.testing.assert_close(
            tokens.view(bs, width)[:, : accepted + 1],
            target[:, : accepted + 1],
            rtol=0,
            atol=0,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm")
@pytest.mark.parametrize(
    "candidate_shape,target_shape", [((2, 5), (2, 6)), ((2, 6), (1, 6))]
)
def test_verify_reports_mismatched_shapes(monkeypatch, candidate_shape, target_shape):
    monkeypatch.setattr(greedy, "_verify_chain_greedy_cuda", greedy.error_fn)
    candidates = torch.empty(candidate_shape, device="cuda", dtype=torch.int32)
    target = torch.empty(target_shape, device="cuda", dtype=torch.int32)
    outputs = (
        torch.full((12,), -1, device="cuda", dtype=torch.int32),
        torch.full((2, 6), -1, device="cuda", dtype=torch.int32),
        torch.full((2,), -1, device="cuda", dtype=torch.int32),
    )
    with pytest.raises(ValueError) as error:
        greedy._verify_chain_greedy(*outputs, candidates, target, 2, 6)
    assert "expects candidates and target_predict shapes (2, 6)" in str(error.value)
    assert f"got candidates {candidate_shape}" in str(error.value)
    assert f"target_predict {target_shape}" in str(error.value)
    for output in outputs:
        assert (output == -1).all()

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

"""Ascend output scores follow the token agreed by tensor parallelism.

The actual NPU sampler and probability arithmetic run. Only the communication
step is replaced, to reproduce ranks choosing different tokens before sync.
"""

import pytest
import torch

pytest.importorskip("torch_npu", reason="Ascend runtime is not installed")
pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="requires Ascend NPU"
)


def test_verify_scores_the_final_tp_token_once(monkeypatch):
    from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
    from tokenspeed.runtime.sampling.backends.ascend_full import (
        AscendFullSamplingBackend,
    )
    from tokenspeed.runtime.sampling.backends.base import SamplingBackendConfig
    from tokenspeed.runtime.sampling.diagnostics import sample_with_diagnostics
    from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
    from tokenspeed.runtime.sampling.sampling_params import SamplingParams

    local_tokens = []

    def agree_on_another_rank(self, *tensors):
        del self
        local_tokens.append(tensors[0].clone())
        tensors[0].copy_((tensors[0] + 1).remainder(4))

    monkeypatch.setattr(
        AscendFullSamplingBackend, "maybe_broadcast", agree_on_another_rank
    )
    backend = AscendFullSamplingBackend(
        SamplingBackendConfig(
            synthetic_acceptance_length=None,
            enable_speculative_sampling=False,
            sampling_stream="batch",
            logprob_order="torch",
            enable_output_logprobs=True,
            enable_output_cumulative_probs=True,
            max_bs=1,
            max_draft_tokens_per_req=1,
            max_req_pool_size=2,
            vocab_size=16,
            device="npu",
        )
    )
    backend.prepare_step(
        ["ascend-cumulative-tp"],
        [1],
        [SamplingParams(temperature=1.0, top_k=-1, top_p=1.0, seed=73)],
        num_tokens_per_req=1,
    )
    raw = torch.full((1, 16), -torch.inf)
    raw[0, :4] = torch.tensor([0.4, 0.3, 0.2, 0.1]).log()
    output = LogitsProcessorOutput(next_token_logits=raw.to("npu"))
    info = SamplingBatchInfo(req_pool_indices=torch.tensor([1], device="npu"))
    candidates = torch.zeros((1, 1), dtype=torch.int32, device="npu")

    def sample():
        return backend.verify(output, info, candidates, tree=None)

    tokens, lengths = sample_with_diagnostics(
        output,
        info,
        ForwardOutputLayout(0, 0, 1, 1),
        sample,
        enabled=backend.config.enable_output_cumulative_probs,
        finalize=None,
    )
    final_ids = tokens.cpu().long()
    assert lengths.cpu().tolist() == [1]
    assert not torch.equal(local_tokens[0].cpu().long(), final_ids)
    expected_lp = raw.double().log_softmax(-1).gather(1, final_ids[:, None]).squeeze(-1)
    torch.testing.assert_close(
        output.next_token_logprobs.cpu().double(),
        expected_lp,
        rtol=1e-6,
        atol=1e-7,
    )
    probabilities = raw.double().softmax(-1)
    order = torch.argsort(raw, descending=True, stable=True)
    ranked = probabilities.gather(1, order)
    left = ranked.cumsum(1) - ranked
    rank = (order == final_ids[:, None]).long().argmax(1)
    expected_c = left.gather(1, rank[:, None]).squeeze(-1)
    torch.testing.assert_close(
        output.next_token_cumulative_probs.cpu().double(),
        expected_c,
        rtol=1e-6,
        atol=1e-7,
    )

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

"""Ascend finite top-k normalization and graph replay on real NPU kernels."""

import pytest
import torch

pytest.importorskip("torch_npu", reason="Ascend runtime is not installed")
pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="requires Ascend NPU"
)


@pytest.mark.parametrize("top_p", [1.0, 0.9])
def test_finite_top_k_cap_and_disabled_survive_graph_replay(top_p):
    from tokenspeed.runtime.sampling.backends.ascend_full import (
        AscendFullSamplingBackend,
    )
    from tokenspeed.runtime.sampling.backends.base import SamplingBackendConfig
    from tokenspeed.runtime.sampling.sampling_params import SamplingParams

    requested = [1, 127, 128, 4095, 4096, 4097, 163840, 1 << 40, -1]
    expected = [1, 127, 128, 4095, 4096, 4096, 4096, 4096, 1 << 30]
    vocab = 163840
    rows = len(requested)
    backend = AscendFullSamplingBackend(
        SamplingBackendConfig(
            synthetic_acceptance_length=None,
            enable_speculative_sampling=False,
            sampling_stream="batch",
            logprob_order="torch",
            max_bs=rows,
            max_req_pool_size=rows,
            vocab_size=vocab,
            device="npu",
            enable_tp_sync=False,
        )
    )
    indices = torch.arange(1, rows + 1, device="npu")
    cpu_logits = torch.linspace(-2, 2, vocab).expand(rows, -1).contiguous()
    logits = cpu_logits.to("npu")

    def prepare(ks, generation):
        params = [
            SamplingParams(temperature=1, top_k=k, top_p=top_p, seed=73) for k in ks
        ]
        backend.prepare_step(
            [f"top-k-{generation}-{row}" for row in range(rows)],
            list(range(1, rows + 1)),
            params,
            num_tokens_per_req=1,
        )
        # The backend changes its own pool, not the caller's parameters.
        assert [p.top_k for p in params] == [k if k != -1 else 1 << 30 for k in ks]

    def probabilities():
        scalars = backend._gather_sampling_scalars(indices)
        return backend._sampling_probs(logits, *scalars)

    def check(probs, ks):
        actual = probs.cpu()
        assert backend._top_k_pool[1:].cpu().tolist() == ks
        for row, k in enumerate(ks):
            count = min(k, vocab)
            retained = torch.softmax(cpu_logits[row, -count:], dim=-1)
            if top_p < 1:
                descending = retained.flip(0)
                remove = descending.cumsum(0) - descending > top_p
                retained = descending.masked_fill(remove, 0).flip(0)
                retained /= retained.sum()
            reference = torch.zeros(vocab)
            reference[-count:] = retained
            torch.testing.assert_close(actual[row], reference, atol=2e-6, rtol=2e-5)
            assert (actual[row] > 0).sum() == (reference > 0).sum()

    prepare(requested, 0)
    check(probabilities(), expected)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = probabilities()
    graph.replay()
    check(captured, expected)

    # Change the same pool slots after capture. No host scalar is baked into
    # the graph: disabled, greedy and over-limit rows all change on replay.
    prepare(requested[::-1], 1)
    graph.replay()
    check(captured, expected[::-1])


@pytest.mark.parametrize("verify", [False, True])
def test_sampling_with_nan_detection_is_graph_safe(verify):
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
    from tokenspeed.runtime.sampling.backends.ascend_full import (
        AscendFullSamplingBackend,
    )
    from tokenspeed.runtime.sampling.backends.base import SamplingBackendConfig
    from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
    from tokenspeed.runtime.sampling.sampling_params import SamplingParams

    vocab = 8192
    backend = AscendFullSamplingBackend(
        SamplingBackendConfig(
            synthetic_acceptance_length=None,
            enable_speculative_sampling=False,
            sampling_stream="batch",
            logprob_order="torch",
            max_bs=1,
            max_req_pool_size=1,
            vocab_size=vocab,
            device="npu",
            enable_nan_detection=True,
            enable_output_logprobs=True,
            enable_tp_sync=False,
        )
    )
    backend.prepare_step(
        ["graph-safe"],
        [1],
        [SamplingParams(temperature=1, top_k=4097, seed=73)],
        num_tokens_per_req=1,
    )
    info = SamplingBatchInfo(req_pool_indices=torch.tensor([1], device="npu"))
    output = LogitsProcessorOutput(
        next_token_logits=torch.linspace(-2, 2, vocab, device="npu").reshape(1, -1)
    )
    candidates = torch.zeros((1, 1), dtype=torch.int32, device="npu")

    def run():
        if verify:
            return backend.verify(output, info, candidates, tree=None)
        return backend.sample(output, info)

    run()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        tokens, lengths = run()
    graph.replay()
    assert vocab - 4096 <= tokens.cpu().item() < vocab
    assert lengths.cpu().tolist() == [1]
    assert torch.isfinite(output.next_token_logprobs.cpu()).all()

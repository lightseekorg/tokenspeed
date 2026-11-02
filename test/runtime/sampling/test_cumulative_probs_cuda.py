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

"""Real CUDA coverage for optional sampled-token cumulative probabilities."""

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="runtime-1gpu")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

VOCAB = 32768
POOL = 8


@pytest.fixture(params=["flashinfer", "flashinfer_full"])
def backend_pair(request):
    from tokenspeed.runtime.sampling.backends.base import SamplingBackendConfig
    from tokenspeed.runtime.sampling.backends.flashinfer import (
        FlashInferSamplingBackend,
    )
    from tokenspeed.runtime.sampling.backends.flashinfer_full import (
        FlashInferFullSamplingBackend,
    )

    cls = (
        FlashInferSamplingBackend
        if request.param == "flashinfer"
        else FlashInferFullSamplingBackend
    )

    def create(*, stream, logprob_order):
        config = SamplingBackendConfig(
            synthetic_acceptance_length=None,
            enable_speculative_sampling=False,
            sampling_stream=stream,
            logprob_order=logprob_order,
            enable_output_logprobs=True,
            max_bs=4,
            max_draft_tokens_per_req=1,
            max_req_pool_size=POOL,
            vocab_size=VOCAB,
            device="cuda",
        )
        return cls(config), cls(replace(config, enable_output_cumulative_probs=True))

    return create


def _prepare(backend):
    from tokenspeed.runtime.sampling.sampling_params import SamplingParams

    params = [
        SamplingParams(temperature=1.0, top_k=-1, top_p=1.0, seed=seed)
        for seed in (73, 74)
    ]
    backend.prepare_step(
        ["cumulative-a", "cumulative-b"], [3, 1], params, num_tokens_per_req=1
    )


def _draw(backend, logits, info, *, decode):
    from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
    from tokenspeed.runtime.sampling.diagnostics import sample_with_diagnostics

    output = LogitsProcessorOutput(next_token_logits=logits)
    rows = logits.shape[0]
    if decode:
        candidates = torch.zeros((rows, 1), dtype=torch.int32, device="cuda")
        layout = ForwardOutputLayout(0, 0, rows, 1)

        def sample():
            return backend.verify(output, info, candidates, tree=None)

    else:
        layout = ForwardOutputLayout(rows, rows, 0, 1)

        def sample():
            return backend.sample(output, info)

    tokens, lengths = sample_with_diagnostics(
        output,
        info,
        layout,
        sample,
        enabled=backend.config.enable_output_cumulative_probs,
        finalize=None,
    )
    return (
        tokens.clone(),
        lengths.clone(),
        output.next_token_logprobs.clone(),
        (
            None
            if output.next_token_cumulative_probs is None
            else output.next_token_cumulative_probs.clone()
        ),
    )


def _reference(logits, tokens):
    probabilities = logits.double().softmax(-1)
    order = torch.argsort(logits, descending=True, stable=True, dim=-1)
    sorted_probabilities = probabilities.gather(-1, order)
    left = sorted_probabilities.cumsum(-1) - sorted_probabilities
    rank = (order == tokens.long().unsqueeze(-1)).long().argmax(-1)
    return left.gather(-1, rank.unsqueeze(-1)).squeeze(-1)


@pytest.mark.parametrize(
    "stream,logprob_order,numerics",
    [
        pytest.param(
            "batch",
            "torch",
            "auto",
            marks=pytest.mark.skipif(
                torch.version.hip is not None,
                reason="FlashInfer batch sampling requires an NVIDIA CUDA GPU",
            ),
        ),
        ("per-request", "torch", "auto"),
        ("per-request", "megatron", "rl-bitwise"),
    ],
)
@pytest.mark.parametrize("decode", [False, True], ids=["sample", "verify_n1"])
@pytest.mark.parametrize("use_graph", [False, True], ids=["eager", "cuda_graph"])
def test_diagnostic_preserves_draws_and_refreshes_cumulative_mass(
    backend_pair, stream, logprob_order, numerics, decode, use_graph
):
    from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
    from tokenspeed.runtime.utils.env import global_server_args_dict

    previous = global_server_args_dict.get("numerics")
    global_server_args_dict["numerics"] = numerics
    try:
        baseline, diagnostic = backend_pair(stream=stream, logprob_order=logprob_order)
        generator = torch.Generator(device="cuda").manual_seed(83)
        logits = torch.randn(4, VOCAB, generator=generator, device="cuda")
        slots = torch.tensor([3, 1, 0, 0], device="cuda")
        frontier = torch.zeros(POOL + 1, dtype=torch.int32, device="cuda")
        info = SamplingBatchInfo(
            req_pool_indices=slots,
            valid_cache_lengths=frontier,
            extend_lengths=(
                None if decode else torch.ones(4, dtype=torch.int32, device="cuda")
            ),
        )
        _prepare(baseline)
        _prepare(diagnostic)
        if use_graph:
            stream_obj = torch.cuda.Stream()
            stream_obj.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream_obj):
                for _ in range(3):
                    _draw(diagnostic, logits, info, decode=decode)
            torch.cuda.current_stream().wait_stream(stream_obj)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = _draw(diagnostic, logits, info, decode=decode)
        cumulative_outputs = []
        for step in range(4):
            logits.copy_(
                torch.randn(4, VOCAB, generator=generator, device="cuda") * (step + 1)
            )
            # Exercise token-ID tie order without making the whole row greedy.
            logits[0, :16] = 1.5
            frontier.fill_(100 + step)
            _prepare(baseline)
            _prepare(diagnostic)
            expected = _draw(baseline, logits, info, decode=decode)
            if use_graph:
                graph.replay()
                actual = captured
            else:
                actual = _draw(diagnostic, logits, info, decode=decode)
            assert expected[3] is None
            for got, reference in zip(actual[:3], expected[:3], strict=True):
                assert torch.equal(got.view(torch.int32), reference.view(torch.int32))
            assert actual[3] is not None
            torch.testing.assert_close(
                actual[3][:2].double(),
                _reference(logits[:2], actual[0][:2]),
                rtol=3e-6,
                atol=3e-7,
            )
            cumulative_outputs.append(actual[3][:2].clone())
        assert any(
            not torch.equal(cumulative_outputs[0], value)
            for value in cumulative_outputs[1:]
        )
    finally:
        if previous is None:
            global_server_args_dict.pop("numerics", None)
        else:
            global_server_args_dict["numerics"] = previous

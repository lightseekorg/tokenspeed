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

"""Shared output metadata across CUDA sampling backends.

C is scored against the same pre-penalty logits as output logprobs. Greedy
and biased sampling are included to verify metadata, not quantile uniformity.
"""

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, suite="runtime-1gpu")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires an NVIDIA CUDA GPU",
)

VOCAB = 32768
POOL = 8


@pytest.fixture(
    params=["flashinfer", "flashinfer_full", "triton", "triton_full", "greedy"]
)
def backend_pair(request):
    from tokenspeed.runtime.sampling.backends.base import SamplingBackendConfig
    from tokenspeed.runtime.sampling.backends.flashinfer import (
        FlashInferSamplingBackend,
    )
    from tokenspeed.runtime.sampling.backends.flashinfer_full import (
        FlashInferFullSamplingBackend,
    )
    from tokenspeed.runtime.sampling.backends.greedy import GreedySamplingBackend
    from tokenspeed.runtime.sampling.backends.triton import TritonSamplingBackend
    from tokenspeed.runtime.sampling.backends.triton_full import (
        TritonFullSamplingBackend,
    )

    implementations = {
        "flashinfer": FlashInferSamplingBackend,
        "flashinfer_full": FlashInferFullSamplingBackend,
        "triton": TritonSamplingBackend,
        "triton_full": TritonFullSamplingBackend,
        "greedy": GreedySamplingBackend,
    }
    implementation = implementations[request.param]

    def create(*, logprob_order):
        config = SamplingBackendConfig(
            synthetic_acceptance_length=None,
            enable_speculative_sampling=False,
            sampling_stream="per-request",
            logprob_order=logprob_order,
            enable_output_logprobs=True,
            max_bs=4,
            max_draft_tokens_per_req=1,
            max_req_pool_size=POOL,
            vocab_size=VOCAB,
            device="cuda",
        )
        return implementation(config), implementation(
            replace(config, enable_output_cumulative_probs=True)
        )

    return request.param, create


def _prepare(backend, *, step, with_bias):
    from tokenspeed.runtime.sampling.sampling_params import SamplingParams

    params = [
        SamplingParams(
            temperature=1.0,
            top_k=-1,
            top_p=1.0,
            seed=seed,
            logit_bias={7: 50.0} if with_bias else None,
        )
        for seed in (73, 74)
    ]
    backend.prepare_step(
        [f"shared-cumulative-a-{step}", f"shared-cumulative-b-{step}"],
        [3, 1],
        params,
        num_tokens_per_req=1,
    )


def _draw(backend, logits, info, *, decode):
    from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
    from tokenspeed.runtime.sampling.diagnostics import sample_with_diagnostics

    # Full backends may update their sampler input in place; every draw
    # starts from the same model output before those updates.
    output = LogitsProcessorOutput(next_token_logits=logits.clone())
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
    ranked = probabilities.gather(-1, order)
    left = ranked.cumsum(-1) - ranked
    rank = (order == tokens.long()[:, None]).long().argmax(-1)
    return left.gather(-1, rank[:, None]).squeeze(-1)


@pytest.mark.parametrize("logprob_order", ["torch", "megatron"])
@pytest.mark.parametrize("decode", [False, True], ids=["sample", "verify_n1"])
@pytest.mark.parametrize("use_graph", [False, True], ids=["eager", "cuda_graph"])
def test_shared_metadata_preserves_tokens_and_logprobs(
    backend_pair, logprob_order, decode, use_graph
):
    from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
    from tokenspeed.runtime.utils.env import global_server_args_dict

    previous = global_server_args_dict.get("numerics")
    global_server_args_dict["numerics"] = (
        "rl-bitwise" if logprob_order == "megatron" else "auto"
    )
    try:
        backend_name, create = backend_pair
        baseline, diagnostic = create(logprob_order=logprob_order)
        with_bias = backend_name in {"flashinfer_full", "triton_full"}
        generator = torch.Generator(device="cuda").manual_seed(104)
        logits = torch.randn(4, VOCAB, generator=generator, device="cuda")
        frontier = torch.zeros(POOL + 1, dtype=torch.int32, device="cuda")
        info = SamplingBatchInfo(
            req_pool_indices=torch.tensor([3, 1, 0, 0], device="cuda"),
            valid_cache_lengths=frontier,
            extend_lengths=(
                None if decode else torch.ones(4, dtype=torch.int32, device="cuda")
            ),
        )
        if with_bias:
            from tokenspeed.runtime.grammar.base_grammar_backend import (
                get_apply_vocab_mask_func,
            )

            # Real grammar application stays captured while the mask changes.
            # The forbidden lane has a strong bias in the full sampler, so the
            # reference must be masked and still precede that bias.
            info.vocab_mask = torch.full(
                (4, VOCAB // 32), -1, dtype=torch.int32, device="cuda"
            )
            info.apply_vocab_mask = get_apply_vocab_mask_func("xgrammar")
        _prepare(baseline, step=-1, with_bias=with_bias)
        _prepare(diagnostic, step=-1, with_bias=with_bias)
        if use_graph:
            capture_stream = torch.cuda.Stream()
            capture_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(capture_stream):
                for _ in range(3):
                    _draw(diagnostic, logits, info, decode=decode)
            torch.cuda.current_stream().wait_stream(capture_stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = _draw(diagnostic, logits, info, decode=decode)
        for step in range(3):
            logits.copy_(
                torch.randn(4, VOCAB, generator=generator, device="cuda") * (step + 1)
            )
            logits[0, :16] = 1.5
            original = logits.clone()
            if with_bias:
                info.vocab_mask.fill_(-1)
                if step % 2:
                    info.vocab_mask[:, 0].bitwise_and_(~(1 << 7))
                    original[:, 7] = -torch.inf
            frontier.fill_(20 + step)
            _prepare(baseline, step=step, with_bias=with_bias)
            _prepare(diagnostic, step=step, with_bias=with_bias)
            expected = _draw(baseline, logits, info, decode=decode)
            if use_graph:
                graph.replay()
                actual = captured
            else:
                actual = _draw(diagnostic, logits, info, decode=decode)
            for got, reference in zip(actual[:3], expected[:3], strict=True):
                assert torch.equal(
                    got[:2].view(torch.int32), reference[:2].view(torch.int32)
                )
            assert expected[3] is None and actual[3] is not None
            torch.testing.assert_close(
                actual[3][:2].double(),
                _reference(original[:2], actual[0][:2]),
                rtol=2e-6,
                atol=2e-7,
            )
            if with_bias:
                if step % 2:
                    assert (actual[0][:2] != 7).all()
                else:
                    assert actual[0][:2].tolist() == [7, 7]
    finally:
        if previous is None:
            global_server_args_dict.pop("numerics", None)
        else:
            global_server_args_dict["numerics"] = previous

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

"""Cumulative probability metadata survives execution and configuration gates."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.execution.forward_step import live_graph_outputs


@pytest.mark.parametrize("live_bs", [0, 1, 3])
def test_graph_replay_slices_cumulative_probabilities_with_the_sampled_tokens(
    live_bs,
):
    # A padded batch of four decodes, two verify columns each.
    width = 2
    tokens = torch.arange(8, dtype=torch.int32)
    lengths = torch.full((4,), width, dtype=torch.int32)
    logprobs = -torch.arange(8, dtype=torch.float32)
    cumulative = torch.linspace(0.0, 0.875, 8)
    live = live_graph_outputs(
        (tokens, lengths, logprobs, cumulative, None), live_bs, width
    )
    rows = live_bs * width
    assert torch.equal(live[0], tokens[:rows])
    assert torch.equal(live[1], lengths[:live_bs])
    assert torch.equal(live[2], logprobs[:rows])
    assert torch.equal(live[3], cumulative[:rows])
    assert live[4] is None
    # Each replay refreshes the captured storage the slices view.
    cumulative.copy_(torch.linspace(0.125, 1.0, 8))
    assert torch.equal(live[3], torch.linspace(0.125, 1.0, 8)[:rows])


def test_graph_replay_without_the_diagnostic_returns_none_for_it():
    tokens = torch.arange(4, dtype=torch.int32)
    lengths = torch.ones(4, dtype=torch.int32)
    live = live_graph_outputs((tokens, lengths, None, None, None), 2, 1)
    assert live[2] is None and live[3] is None and live[4] is None


def test_a_captured_decode_with_prompt_or_top_k_logprobs_is_refused():
    from tokenspeed.runtime.execution.types import ForwardLogprobs

    metadata = ForwardLogprobs(output_top_logprobs_val=torch.tensor([[-0.5]]))
    with pytest.raises(RuntimeError, match="prompt or top-k"):
        live_graph_outputs((torch.zeros(1), torch.ones(1), None, None, metadata), 1, 1)


def _args(**overrides):
    from tokenspeed.runtime.utils.server_args import ServerArgs

    options = dict(
        model="x",
        sampling_backend="flashinfer",
        enable_output_logprobs=True,
        enable_output_cumulative_probs=True,
    )
    options.update(overrides)
    return ServerArgs(**options)


@pytest.mark.parametrize(
    "backend", ["flashinfer", "flashinfer_full", "triton", "triton_full", "greedy"]
)
@pytest.mark.parametrize("stream", ["batch", "per-request"])
def test_cumulative_diagnostics_accept_supported_tp8_sampling(backend, stream):
    args = _args(sampling_backend=backend, sampling_stream=stream, world_size=8)
    assert args.mapping.attn.tp_size == 8
    assert args.enable_output_logprobs and args.enable_output_cumulative_probs


@pytest.mark.parametrize("algorithm", ["DUMMY", "MTP"])
@pytest.mark.parametrize(
    "backend", ["flashinfer", "flashinfer_full", "triton", "triton_full", "greedy"]
)
def test_cumulative_diagnostics_accept_real_chain_verification(algorithm, backend):
    args = _args(
        speculative_algorithm=algorithm,
        speculative_num_draft_tokens=4,
        sampling_backend=backend,
    )
    assert args.speculative_algorithm == algorithm
    assert args.speculative_num_draft_tokens == 4
    assert args.synthetic_acceptance_length is None


def test_cumulative_diagnostics_refuse_simulated_acceptance(monkeypatch):
    monkeypatch.setenv("TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN", "2")
    with pytest.raises(ValueError, match="TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN"):
        _args(speculative_algorithm="DUMMY", speculative_num_draft_tokens=4)


@pytest.mark.parametrize(
    "options,reason",
    [
        ({"enable_output_logprobs": False}, "requires --enable-output-logprobs"),
        (
            {
                "speculative_algorithm": "MTP",
                "speculative_draft_model_path": "draft",
                "speculative_eagle_topk": 2,
                "speculative_num_draft_tokens": 4,
            },
            "supports chain verification only",
        ),
        (
            {
                "speculative_algorithm": "EAGLE4",
                "speculative_exec_mode": 0,
                "speculative_data_path": "unused-cost-data",
                "speculative_compute_intensity": "chain_intensity",
            },
            "supports chain verification only",
        ),
        ({"data_parallel_size": 2}, "requires DP1"),
        ({"dp_sampling": True, "world_size": 8}, "requires DP1"),
        ({"disaggregation_mode": "decode"}, "aggregated serving"),
        (
            {"pipeline_parallel_size": 2, "disaggregation_mode": "prefill"},
            "one pipeline stage",
        ),
    ],
)
def test_cumulative_diagnostics_refuse_unsupported_execution(options, reason):
    with pytest.raises(ValueError, match=reason):
        _args(**options)


def test_disabled_diagnostics_do_not_restrict_other_sampling_backends():
    args = _args(
        sampling_backend="triton",
        enable_output_logprobs=False,
        enable_output_cumulative_probs=False,
    )
    assert not args.enable_output_logprobs and not args.enable_output_cumulative_probs


def test_a_backend_that_does_not_declare_aligned_logprobs_is_refused_at_startup():
    from tokenspeed.runtime.sampling.backends.base import (
        SamplingBackend,
        SamplingBackendConfig,
    )

    class ExtensionBackend(SamplingBackend):
        def sample(self, logits_output, sampling_info):
            raise NotImplementedError

        def verify(self, logits_output, sampling_info, candidates, tree):
            raise NotImplementedError

    def config(enabled):
        return SamplingBackendConfig(
            synthetic_acceptance_length=None,
            enable_speculative_sampling=False,
            enable_block_rejection=False,
            sampling_stream="batch",
            logprob_order="torch",
            enable_output_logprobs=True,
            enable_output_cumulative_probs=enabled,
            max_draft_tokens_per_req=1,
            device="cpu",
        )

    ExtensionBackend(config(False))
    with pytest.raises(ValueError, match="_SUPPORTS_OUTPUT_CUMULATIVE_PROBS"):
        ExtensionBackend(config(True))

    class DeclaringBackend(ExtensionBackend):
        _SUPPORTS_OUTPUT_CUMULATIVE_PROBS = True

    DeclaringBackend(config(True))

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

from tokenspeed.runtime.execution.types import ForwardLogprobs


def test_cumulative_probabilities_survive_logits_metadata_and_host_copy():
    from tokenspeed.runtime.layers.logits_processor_output import LogitsProcessorOutput

    output = LogitsProcessorOutput(next_token_logits=torch.zeros(2, 4))
    assert ForwardLogprobs.of(output) is None
    output.next_token_cumulative_probs = torch.tensor([0.0, 0.875])
    output.input_token_logprobs = torch.tensor([-1.0, -2.0])
    metadata = ForwardLogprobs.of(output)
    assert metadata is not None
    host = metadata.to_host()
    torch.testing.assert_close(
        host.output_token_cumulative_probs, torch.tensor([0.0, 0.875])
    )
    torch.testing.assert_close(host.input_token_logprobs, torch.tensor([-1.0, -2.0]))


@pytest.mark.parametrize("live_rows", [0, 1, 3])
def test_decode_metadata_trims_padding_and_reads_refreshed_capture_storage(live_rows):
    captured = torch.tensor([0.0, 0.25, 0.875, -99.0])
    metadata = ForwardLogprobs(
        output_token_cumulative_probs=captured,
        input_token_logprobs=torch.tensor([-3.0]),
        output_top_logprobs_val=torch.tensor([[-0.5]]),
    )
    live = metadata.decode_slice(live_rows)
    assert live is not None
    assert live.input_token_logprobs is None
    assert live.output_top_logprobs_val is None
    torch.testing.assert_close(live.output_token_cumulative_probs, captured[:live_rows])
    captured.copy_(torch.tensor([0.125, 0.5, 0.75, -98.0]))
    torch.testing.assert_close(
        live.output_token_cumulative_probs,
        torch.tensor([0.125, 0.5, 0.75])[:live_rows],
    )


def test_decode_without_diagnostics_has_no_metadata():
    assert ForwardLogprobs().decode_slice(2) is None
    assert (
        ForwardLogprobs(input_token_logprobs=torch.tensor([-1.0])).decode_slice(2)
        is None
    )


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


def test_cumulative_metadata_requires_logprobs():
    from tokenspeed.runtime.sampling.backends.base import SamplingBackendConfig

    with pytest.raises(ValueError, match="cumulative probabilities require"):
        SamplingBackendConfig(
            synthetic_acceptance_length=None,
            enable_speculative_sampling=False,
            sampling_stream="batch",
            logprob_order="torch",
            enable_output_logprobs=False,
            enable_output_cumulative_probs=True,
            max_draft_tokens_per_req=4,
        )

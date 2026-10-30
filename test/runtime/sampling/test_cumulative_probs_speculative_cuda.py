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

"""Cumulative target probability through real multi-token chain verification.

Controlled acceptance coins cover every kept width. The kernels, probability
filters, logprob arithmetic and CUDA Graph replay are the production ones.
"""

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="runtime-1gpu")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="FlashInfer chain verification requires an NVIDIA CUDA GPU",
)

VOCAB = 32768
WIDTH = 4
LIVE_ROWS = 4
BATCH_ROWS = 6
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

    backend_type = (
        FlashInferSamplingBackend
        if request.param == "flashinfer"
        else FlashInferFullSamplingBackend
    )

    def create(*, stream, logprob_order, recorded_proposals):
        config = SamplingBackendConfig(
            synthetic_acceptance_length=None,
            enable_speculative_sampling=recorded_proposals,
            enable_block_rejection=False,
            sampling_stream=stream,
            logprob_order=logprob_order,
            enable_output_logprobs=True,
            max_bs=BATCH_ROWS,
            max_draft_tokens_per_req=WIDTH,
            max_req_pool_size=POOL,
            vocab_size=VOCAB,
            device="cuda",
        )
        return backend_type(config), backend_type(
            replace(config, enable_output_cumulative_probs=True)
        )

    return create


def _prepare_coins(backend, kept_lengths):
    from tokenspeed.runtime.sampling.sampling_params import SamplingParams

    params = [
        SamplingParams(temperature=1.0, top_k=-1, top_p=1.0, seed=100 + row)
        for row in range(LIVE_ROWS)
    ]
    backend.prepare_step(
        [f"cumulative-chain-{row}" for row in range(LIVE_ROWS)],
        list(range(1, LIVE_ROWS + 1)),
        params,
        num_tokens_per_req=WIDTH,
    )
    # Every proposed token has target probability 0.25. For a one-hot
    # proposal q, both verifier rules accept coin=0.1 and reject coin=0.9.
    # These are ordinary tensor inputs, updated outside capture as in serving.
    columns = torch.arange(WIDTH, device="cuda")
    coins = torch.where(columns < kept_lengths[:, None] - 1, 0.1, 0.9)
    backend._coins_buf.fill_(0.9)
    backend._coins_buf[:LIVE_ROWS].copy_(coins)
    # The replacement is the last supported token in vocabulary order, whose
    # target C differs from its residual C after the proposal is rejected.
    backend._final_coins_buf.fill_(0.97)


def _set_target_and_proposals(logits, candidates, recorded, *, step):
    rows = torch.arange(BATCH_ROWS * WIDTH, device="cuda")
    first_id = 20 + rows * 11 + step * 5
    support = first_id[:, None] + torch.arange(4, device="cuda")
    values = torch.tensor([0.45, 0.25, 0.25, 0.05], device="cuda").log()
    logits.fill_(-80.0)
    logits.scatter_(1, support, values.expand_as(support))
    candidates[:, 0] = 7
    # Alternating exact-tie proposals exercise both token-ID positions at p=.25.
    proposal_rank = 1 + rows.remainder(2)
    proposals = first_id + proposal_rank
    candidates[:, 1:].copy_(proposals.view(BATCH_ROWS, WIDTH)[:, :-1])
    if recorded is not None:
        recorded.zero_()
        for row in range(LIVE_ROWS):
            recorded[
                row + 1,
                torch.arange(WIDTH - 1, device="cuda"),
                candidates[row, 1:].long(),
            ] = 1.0
    return support[:, -1].view(BATCH_ROWS, WIDTH)


def _draw(backend, logits, info, candidates):
    from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
    from tokenspeed.runtime.sampling.diagnostics import sample_with_diagnostics

    output = LogitsProcessorOutput(next_token_logits=logits)
    layout = ForwardOutputLayout(0, 0, candidates.shape[0], candidates.shape[1])

    def sample():
        return backend.verify(output, info, candidates, tree=None)

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
        backend.accepted_path(BATCH_ROWS, WIDTH).clone(),
    )


def _reference_target_mass(logits, tokens):
    probabilities = logits.double().softmax(-1)
    order = torch.argsort(logits, descending=True, stable=True, dim=-1)
    ranked = probabilities.gather(-1, order)
    left = ranked.cumsum(-1) - ranked
    ranks = (order == tokens.long()[:, None]).long().argmax(-1)
    return left.gather(-1, ranks[:, None]).squeeze(-1)


@pytest.mark.parametrize(
    "recorded_proposals", [False, True], ids=["target_only", "one_hot_q"]
)
@pytest.mark.parametrize(
    "stream,logprob_order,numerics",
    [
        ("batch", "torch", "auto"),
        ("per-request", "torch", "auto"),
        ("per-request", "megatron", "rl-bitwise"),
    ],
)
@pytest.mark.parametrize("use_graph", [False, True], ids=["eager", "cuda_graph"])
def test_cumulative_mass_scores_accepted_and_replacement_tokens_against_target(
    backend_pair, recorded_proposals, stream, logprob_order, numerics, use_graph
):
    from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
    from tokenspeed.runtime.utils.env import global_server_args_dict

    previous = global_server_args_dict.get("numerics")
    global_server_args_dict["numerics"] = numerics
    try:
        baseline, diagnostic = backend_pair(
            stream=stream,
            logprob_order=logprob_order,
            recorded_proposals=recorded_proposals,
        )
        logits = torch.empty(BATCH_ROWS * WIDTH, VOCAB, device="cuda")
        candidates = torch.empty(BATCH_ROWS, WIDTH, dtype=torch.int32, device="cuda")
        recorded = (
            torch.zeros(POOL + 1, WIDTH, VOCAB, device="cuda")
            if recorded_proposals
            else None
        )
        slots = torch.tensor([1, 2, 3, 4, 0, 0], device="cuda")
        info = SamplingBatchInfo(
            req_pool_indices=slots,
            valid_cache_lengths=torch.zeros(POOL + 1, dtype=torch.int32, device="cuda"),
            draft_probs=recorded,
        )
        kept_lengths = torch.arange(1, WIDTH + 1, device="cuda")
        _set_target_and_proposals(logits, candidates, recorded, step=0)
        _prepare_coins(baseline, kept_lengths)
        _prepare_coins(diagnostic, kept_lengths)
        if use_graph:
            stream_obj = torch.cuda.Stream()
            stream_obj.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream_obj):
                for _ in range(3):
                    _draw(diagnostic, logits, info, candidates)
            torch.cuda.current_stream().wait_stream(stream_obj)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = _draw(diagnostic, logits, info, candidates)

        for step in range(3):
            kept_lengths = torch.arange(1, WIDTH + 1, device="cuda").roll(step)
            bonus_ids = _set_target_and_proposals(
                logits, candidates, recorded, step=step + 1
            )
            original_logits = logits.clone()
            original_recorded = None if recorded is None else recorded.clone()
            _prepare_coins(baseline, kept_lengths)
            _prepare_coins(diagnostic, kept_lengths)
            expected = _draw(baseline, logits, info, candidates)
            if use_graph:
                graph.replay()
                actual = captured
            else:
                actual = _draw(diagnostic, logits, info, candidates)

            torch.testing.assert_close(
                actual[1][:LIVE_ROWS].long(), kept_lengths, rtol=0, atol=0
            )
            assert torch.equal(actual[1][:LIVE_ROWS], expected[1][:LIVE_ROWS])
            positions = torch.arange(WIDTH, device="cuda")
            valid = positions < kept_lengths[:, None]
            # Suffix values and padding requests have no emitted token and are
            # deliberately excluded from every token/probability comparison.
            tokens = actual[0].view(BATCH_ROWS, WIDTH)[:LIVE_ROWS]
            expected_tokens = expected[0].view(BATCH_ROWS, WIDTH)[:LIVE_ROWS]
            assert torch.equal(tokens[valid], expected_tokens[valid])
            actual_lps = actual[2].view(BATCH_ROWS, WIDTH)[:LIVE_ROWS]
            expected_lps = expected[2].view(BATCH_ROWS, WIDTH)[:LIVE_ROWS]
            assert torch.equal(
                actual_lps[valid].view(torch.int32),
                expected_lps[valid].view(torch.int32),
            )
            assert expected[3] is None and actual[3] is not None

            accepted = positions < kept_lengths[:, None] - 1
            assert torch.equal(
                tokens[:, :-1][accepted[:, :-1]],
                candidates[:LIVE_ROWS, 1:][accepted[:, :-1]],
            )
            last = kept_lengths - 1
            emitted_bonus = tokens.gather(1, last[:, None]).squeeze(-1)
            assert torch.equal(
                emitted_bonus,
                bonus_ids[:LIVE_ROWS].gather(1, last[:, None]).squeeze(-1).int(),
            )
            path = actual[4][:LIVE_ROWS]
            flat_rows = torch.arange(LIVE_ROWS * WIDTH, device="cuda").view(
                LIVE_ROWS, WIDTH
            )
            assert torch.equal(path[valid].long(), flat_rows[valid])
            assert (path[~valid] == -1).all()

            cumulative = actual[3].view(BATCH_ROWS, WIDTH)[:LIVE_ROWS]
            target_logits = original_logits.view(BATCH_ROWS, WIDTH, VOCAB)[:LIVE_ROWS][
                valid
            ]
            target_reference = _reference_target_mass(target_logits, tokens[valid])
            torch.testing.assert_close(
                cumulative[valid].double(), target_reference, rtol=3e-6, atol=3e-7
            )
            target_lp = (
                target_logits.double()
                .log_softmax(-1)
                .gather(1, tokens[valid].long()[:, None])
                .squeeze(-1)
            )
            torch.testing.assert_close(
                actual_lps[valid].double(), target_lp, rtol=3e-6, atol=3e-7
            )
            # Rejected-row replacements have target C=.95, versus residual
            # C=(.45+.25)/.75 after the one-hot proposal is removed.
            rejected_rows = kept_lengths < WIDTH
            replacement_c = cumulative.gather(1, last[:, None]).squeeze(-1)
            torch.testing.assert_close(
                replacement_c[rejected_rows],
                torch.full_like(replacement_c[rejected_rows], 0.95),
                rtol=3e-6,
                atol=3e-7,
            )
            assert (
                torch.abs(replacement_c[rejected_rows] - (0.70 / 0.75)) > 0.01
            ).all()
            assert torch.equal(logits, original_logits)
            if recorded is not None:
                assert torch.equal(recorded, original_recorded)
    finally:
        if previous is None:
            global_server_args_dict.pop("numerics", None)
        else:
            global_server_args_dict["numerics"] = previous

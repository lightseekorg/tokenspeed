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

"""Compact scores retain request identity through the head and verification."""

import os
import sys

import pytest
import torch
from transformers import PretrainedConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from tokenspeed.runtime.execution.nan_guard import NanGuard
from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
from tokenspeed.runtime.execution.verify_layout import VerifyQueryLayout
from tokenspeed.runtime.layers.logits_processor_output import LogitsProcessorOutput
from tokenspeed.runtime.models.base.causal_lm import BaseCausalLM
from tokenspeed.runtime.sampling.backends.base import SamplingBackendConfig
from tokenspeed.runtime.sampling.backends.greedy import GreedySamplingBackend
from tokenspeed.runtime.sampling.diagnostics import (
    gather_token_cumulative_probs,
    sample_with_diagnostics,
)
from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo

DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    ),
]


class _SmallLanguageModel(BaseCausalLM):
    def resolve_model(self, config, mapping, quant_config, prefix):
        return torch.nn.Identity()

    def resolve_lm_head(self, config, quant_config, prefix):
        return torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)


def _context(layout, requests):
    return ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=requests,
        num_extends=0,
        input_num_tokens=requests * layout.width,
        forward_mode=ForwardMode.DECODE,
        output_layout=ForwardOutputLayout(0, 0, requests, layout.width),
        verify_queries=layout,
        logits_queries=layout,
        capture_hidden_mode=CaptureHiddenMode.FULL,
        gather_ids=torch.arange(
            requests * layout.width, device=layout.query_offsets.device
        ),
    )


def _backend(device, *, logprobs, cumulative):
    return GreedySamplingBackend(
        SamplingBackendConfig(
            synthetic_acceptance_length=None,
            enable_speculative_sampling=False,
            sampling_stream="batch",
            logprob_order="torch",
            max_bs=4,
            max_draft_tokens_per_req=4,
            max_req_pool_size=5,
            vocab_size=16,
            device=device,
            enable_output_logprobs=logprobs,
            enable_output_cumulative_probs=cumulative,
        )
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("compact_head", [False, True])
def test_model_exit_projects_only_its_score_rows_and_restores_draft_taps(
    device, compact_head
):
    torch.manual_seed(71)
    model = _SmallLanguageModel(
        PretrainedConfig(hidden_size=4, vocab_size=16), Mapping(rank=0)
    ).to(device)
    model.logits_processor.prepare_communication(model.lm_head)
    layout = VerifyQueryLayout(4, 8, 4, device)
    rates = torch.tensor([[0.99] * 3, [0.6] * 3, [0.1] * 3, [0.2] * 3], device=device)
    hidden = torch.randn(16, 4, device=device)
    for requests, ready in [(3, True), (1, False), (4, True)]:
        layout.refresh(
            rates,
            torch.full((4,), ready, device=device),
            torch.arange(4, device=device) < requests,
        )
        compact = layout.pack(hidden, token_dim=0)
        compact[~layout.packed_valid] = float("nan")
        taps = [compact * 2, compact + 7]
        ctx = _context(layout, requests)
        if not compact_head:
            ctx.logits_queries = None
        output = model.exit_logits(
            torch.zeros(requests * 4, device=device), compact, taps, ctx
        )
        projected = (
            layout.mask_compact_rows(compact)
            if compact_head
            else layout.expand(compact, token_dim=0, request_count=requests)
        )
        assert output.next_token_logits.shape == (projected.shape[0], 16)
        assert torch.isfinite(output.next_token_logits).all()
        torch.testing.assert_close(
            output.next_token_logits, projected @ model.lm_head.weight.T, atol=0, rtol=0
        )
        expected_taps = torch.cat((hidden * 2, hidden + 7), dim=-1)[
            : requests * 4
        ].clone()
        for request, length in enumerate(layout.query_lengths[:requests].tolist()):
            expected_taps[request * 4 + length : (request + 1) * 4].zero_()
        torch.testing.assert_close(output.hidden_states, expected_taps, atol=0, rtol=0)
        assert (output.query_layout is layout) == compact_head


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("prefused_ids", [False, True])
@pytest.mark.parametrize("mismatch", [None, 1, 3])
@pytest.mark.parametrize("logprobs", [False, True])
def test_compact_greedy_matches_dense_prefixes(
    device, prefused_ids, mismatch, logprobs
):
    width, batch, vocab = 4, 3, 16
    layout = VerifyQueryLayout(4, 7, width, device)
    layout.refresh(
        torch.tensor([[0.99] * 3, [0.2] * 3, [0.5] * 3, [0.1] * 3], device=device),
        torch.ones(4, dtype=torch.bool, device=device),
        torch.tensor([True, False, True, False], device=device),
    )
    targets = (
        torch.arange(batch * width, device=device).view(batch, width) * 3 + 7
    ) % vocab
    scores = torch.zeros(batch * width, vocab, device=device)
    scores.scatter_(1, targets.flatten()[:, None], 8)
    candidates = targets.to(torch.int32).clone()
    candidates[:, 1:] = targets[:, :-1]
    if mismatch is not None:
        candidates[:, mismatch] = (candidates[:, mismatch] + 1) % vocab
    info = SamplingBatchInfo(vocab_size=vocab, device=device)
    dense = _backend(device, logprobs=logprobs, cumulative=False)
    dense_output = LogitsProcessorOutput(next_token_logits=scores.clone())
    expected_tokens, expected_lengths = dense.verify(
        dense_output, info, candidates, tree=None
    )
    expected_lengths = torch.minimum(expected_lengths, layout.query_lengths[:batch])
    compact_output = LogitsProcessorOutput(
        next_token_logits=layout.pack(scores, token_dim=0),
        next_token_ids=(
            layout.pack(targets.flatten(), token_dim=0) if prefused_ids else None
        ),
        query_layout=layout,
    )
    compact = _backend(device, logprobs=logprobs, cumulative=False)
    tokens, lengths = compact.verify(compact_output, info, candidates, tree=None)
    torch.testing.assert_close(lengths, expected_lengths, atol=0, rtol=0)
    for request, length in enumerate(lengths.tolist()):
        start = request * width
        torch.testing.assert_close(
            tokens[start : start + length], expected_tokens[start : start + length]
        )
        expected_path = torch.full((width,), -1, device=device, dtype=torch.int32)
        expected_path[:length] = torch.arange(start, start + length, device=device)
        torch.testing.assert_close(
            compact.accepted_path(batch, width)[request], expected_path
        )
        if logprobs:
            torch.testing.assert_close(
                compact_output.next_token_logprobs[start : start + length],
                dense_output.next_token_logprobs[start : start + length],
                atol=0,
                rtol=0,
            )


@pytest.mark.parametrize("device", DEVICES)
def test_compact_score_diagnostics_ignore_filler_and_keep_request_slots(device):
    layout = VerifyQueryLayout(4, 8, 4, device)
    layout.refresh(
        torch.ones(4, 3, device=device),
        torch.zeros(4, dtype=torch.bool, device=device),
        torch.tensor([True, False, True, False], device=device),
    )
    ctx = _context(layout, 3)
    scores = torch.randn(8, 16, device=device)
    corrupted = scores.clone()
    corrupted[1, 0] = float("nan")
    corrupted[-1, 0] = float("nan")
    guard = NanGuard(4, device)
    guard.reset(3)
    guard.audit_logits(
        LogitsProcessorOutput(next_token_logits=corrupted, query_layout=layout), ctx
    )
    torch.testing.assert_close(
        guard.flags[:3], torch.tensor([0, 0, 1], dtype=torch.int32, device=device)
    )
    assert torch.isfinite(corrupted).all()
    output = LogitsProcessorOutput(next_token_logits=scores, query_layout=layout)
    backend = _backend(device, logprobs=True, cumulative=True)
    info = SamplingBatchInfo(vocab_size=16, device=device)
    candidates = torch.zeros(3, 4, dtype=torch.int32, device=device)
    tokens, lengths = sample_with_diagnostics(
        output,
        info,
        ctx.output_layout,
        lambda: backend.verify(output, info, candidates, tree=None),
        enabled=True,
        finalize=None,
    )
    expected = layout.expand(
        gather_token_cumulative_probs(scores, layout.pack(tokens, token_dim=0)),
        token_dim=0,
        request_count=3,
    )
    torch.testing.assert_close(output.next_token_cumulative_probs, expected)
    torch.testing.assert_close(
        lengths, torch.tensor([1, 0, 1], dtype=torch.int32, device=device)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay")
def test_compact_head_and_verify_graph_replays_changed_prefixes():
    torch.manual_seed(79)
    model = _SmallLanguageModel(
        PretrainedConfig(hidden_size=4, vocab_size=16), Mapping(rank=0)
    ).cuda()
    model.logits_processor.prepare_communication(model.lm_head)
    layout = VerifyQueryLayout(4, 8, 4, "cuda")
    rates = torch.tensor([[0.99] * 3, [0.6] * 3, [0.1] * 3, [0.2] * 3], device="cuda")
    ready = torch.ones(4, dtype=torch.bool, device="cuda")
    active = torch.tensor([True, True, True, False], device="cuda")
    layout.refresh(rates, ready, active)
    ctx = _context(layout, 4)
    hidden = torch.randn(16, 4, device="cuda")
    candidates = torch.zeros(4, 4, dtype=torch.int32, device="cuda")
    info = SamplingBatchInfo(vocab_size=16, device="cuda")
    backend = _backend("cuda", logprobs=True, cumulative=False)

    def run():
        compact = layout.pack(hidden, token_dim=0)
        output = model.exit_logits(candidates.flatten(), compact, [compact * 2], ctx)
        return backend.verify(output, info, candidates, tree=None)

    for _ in range(2):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        tokens, lengths = run()
    for count in (3, 1, 4, 0, 2):
        layout.refresh(
            rates.roll(count, 0), ready, torch.arange(4, device="cuda") < count
        )
        expected_tokens, expected_lengths = (t.clone() for t in run())
        graph.replay()
        torch.testing.assert_close(tokens, expected_tokens, atol=0, rtol=0)
        torch.testing.assert_close(lengths, expected_lengths, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA grammar masks")
def test_compact_verify_applies_each_original_columns_grammar_before_argmax():
    pytest.importorskip("xgrammar")
    from tokenspeed.runtime.grammar.base_grammar_backend import (
        get_apply_vocab_mask_func,
    )

    layout = VerifyQueryLayout(4, 8, 4, "cuda")
    layout.refresh(
        torch.ones(4, 3, device="cuda"),
        torch.ones(4, dtype=torch.bool, device="cuda"),
        torch.tensor([True, True, True, False], device="cuda"),
    )
    allowed = (torch.arange(12, device="cuda") * 2 + 3) % 16
    mask = (1 << allowed).to(torch.int32)[:, None]
    info = SamplingBatchInfo(
        vocab_size=16,
        device="cuda",
        vocab_mask=mask,
        apply_vocab_mask=get_apply_vocab_mask_func("xgrammar"),
    )
    logits = torch.arange(16, device="cuda").float().expand(8, 16).clone()
    output = LogitsProcessorOutput(
        next_token_logits=logits,
        next_token_ids=torch.full((8,), 15, dtype=torch.int64, device="cuda"),
        query_layout=layout,
    )
    candidates = torch.zeros(3, 4, dtype=torch.int32, device="cuda")
    candidates[:, 1:] = allowed.view(3, 4)[:, :-1]
    tokens, lengths = _backend("cuda", logprobs=False, cumulative=False).verify(
        output, info, candidates, tree=None
    )
    torch.testing.assert_close(lengths, layout.query_lengths[:3], atol=0, rtol=0)
    for request, length in enumerate(lengths.tolist()):
        torch.testing.assert_close(
            tokens.view(3, 4)[request, :length],
            allowed.view(3, 4)[request, :length].int(),
        )

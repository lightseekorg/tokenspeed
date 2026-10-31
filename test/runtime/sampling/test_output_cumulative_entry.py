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

"""CPU sampling implementations exercise the production diagnostic boundary."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
from tokenspeed.runtime.layers.logits_processor_output import LogitsProcessorOutput
from tokenspeed.runtime.sampling.diagnostics import sample_with_diagnostics
from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo


class TorchCategoricalSampler:
    """A CPU categorical sampler reporting raw scores and reusing output storage."""

    def __init__(self, output, lengths, *, seed, allowed, bias, record_logprobs):
        self.output = output
        self.allowed = allowed
        self.bias = bias
        self.record_logprobs = record_logprobs
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.tokens = torch.empty(output.next_token_logits.shape[0], dtype=torch.int32)
        self.lengths = lengths.clone()
        self.logprobs = torch.empty(output.next_token_logits.shape[0])
        self.result = (self.tokens, self.lengths)

    def __call__(self):
        logits = self.output.next_token_logits
        if self.allowed is not None:
            logits.masked_fill_(~self.allowed, -torch.inf)
        raw = logits.clone()
        if self.bias is not None:
            logits.add_(self.bias)
        self.tokens.copy_(
            torch.multinomial(logits.softmax(-1), 1, generator=self.generator)
            .squeeze(1)
            .to(torch.int32)
        )
        if self.record_logprobs:
            self.logprobs.copy_(
                raw.log_softmax(-1, dtype=torch.float32)
                .gather(1, self.tokens.long()[:, None])
                .squeeze(1)
            )
            self.output.next_token_logprobs = self.logprobs
        return self.result


class ArgmaxResultSampler:
    """Consume precomputed global IDs when no full-vocabulary scores are needed."""

    def __init__(self, output, lengths):
        self.tokens = output.next_token_ids
        self.lengths = lengths.clone()
        self.result = (self.tokens, self.lengths)

    def __call__(self):
        return self.result


class TruncatedLogprobsSampler(TorchCategoricalSampler):
    def __call__(self):
        result = super().__call__()
        self.output.next_token_logprobs = self.logprobs[:-1]
        return result


class TruncatedTokensSampler(TorchCategoricalSampler):
    def __call__(self):
        tokens, lengths = super().__call__()
        return tokens[:-1], lengths


class PackedGrammar:
    """Apply the ordinary int32 vocabulary bitmask with real CPU tensor math."""

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def __call__(self, logits, vocab_mask):
        columns = torch.arange(self.vocab_size)
        words = vocab_mask[:, columns // 32].long()
        allowed = ((words >> (columns % 32)) & 1).bool()
        logits.masked_fill_(~allowed, -torch.inf)


def reference_mass(logits, tokens):
    probabilities = logits.double().softmax(-1)
    expected = []
    for row, selected in enumerate(tokens.tolist()):
        scores = logits[row].tolist()
        order = sorted(range(len(scores)), key=lambda token: (-scores[token], token))
        expected.append(probabilities[row, order[: order.index(selected)]].sum())
    return torch.stack(expected)


def test_an_extension_sampler_gets_metadata_without_a_diagnostic_hook():
    logits = torch.tensor([[0.55, 0.25, 0.15, 0.05]]).log().repeat(8, 1)
    output = LogitsProcessorOutput(next_token_logits=logits.clone())
    sampler = TorchCategoricalSampler(
        output,
        torch.ones(8, dtype=torch.int32),
        seed=731,
        allowed=None,
        bias=None,
        record_logprobs=True,
    )
    result = sample_with_diagnostics(
        output,
        SamplingBatchInfo(device="cpu"),
        ForwardOutputLayout(8, 8, 0, 1),
        sampler,
        enabled=True,
    )
    assert result is sampler.result
    assert output.next_token_logprobs is sampler.logprobs
    torch.testing.assert_close(
        output.next_token_cumulative_probs.double(),
        reference_mass(logits, result[0]),
        rtol=3e-7,
        atol=1e-7,
    )


def test_disabled_entry_keeps_argmax_only_sampling_and_clears_stale_metadata():
    # Global IDs need not index these local/sharded scores. The disabled
    # boundary must delegate without requiring probabilities or full logits.
    output = LogitsProcessorOutput(
        next_token_logits=torch.tensor([[0.2, 0.5]]),
        next_token_ids=torch.tensor([7], dtype=torch.int32),
        next_token_cumulative_probs=torch.tensor([0.75]),
    )
    sampler = ArgmaxResultSampler(output, torch.ones(1, dtype=torch.int32))
    result = sample_with_diagnostics(
        output,
        SamplingBatchInfo(device="cpu"),
        ForwardOutputLayout(1, 1, 0, 1),
        sampler,
        enabled=False,
    )
    assert result is sampler.result
    assert result[0].tolist() == [7]
    assert output.next_token_cumulative_probs is None
    assert output.next_token_logprobs is None


def test_flag_preserves_real_sampling_logprob_bits_and_rng_stream():
    logits = torch.randn(4, 17, generator=torch.Generator().manual_seed(21))
    outputs, samplers, results = [], [], []
    for enabled in (False, True):
        output = LogitsProcessorOutput(next_token_logits=logits.clone())
        sampler = TorchCategoricalSampler(
            output,
            torch.ones(4, dtype=torch.int32),
            seed=41,
            allowed=None,
            bias=None,
            record_logprobs=True,
        )
        results.append(
            sample_with_diagnostics(
                output,
                SamplingBatchInfo(device="cpu"),
                ForwardOutputLayout(4, 4, 0, 1),
                sampler,
                enabled=enabled,
            )
        )
        outputs.append(output)
        samplers.append(sampler)
    assert torch.equal(results[0][0], results[1][0])
    assert torch.equal(results[0][1], results[1][1])
    assert torch.equal(
        outputs[0].next_token_logprobs.view(torch.int32),
        outputs[1].next_token_logprobs.view(torch.int32),
    )
    assert torch.equal(
        torch.rand(8, generator=samplers[0].generator),
        torch.rand(8, generator=samplers[1].generator),
    )
    assert outputs[0].next_token_cumulative_probs is None
    assert outputs[1].next_token_cumulative_probs is not None


@pytest.mark.parametrize("with_grammar", [False, True])
def test_entry_scores_before_inplace_bias_and_after_grammar(with_grammar):
    logits = torch.tensor([[0.55, 0.25, 0.15, 0.05]]).log()
    allowed = torch.tensor([[False, True, True, True]]) if with_grammar else None
    expected_scores = logits.clone()
    if allowed is not None:
        expected_scores.masked_fill_(~allowed, -torch.inf)
    info = SamplingBatchInfo(
        device="cpu",
        vocab_size=4,
        vocab_mask=torch.tensor([[14]], dtype=torch.int32) if with_grammar else None,
        apply_vocab_mask=PackedGrammar(4) if with_grammar else None,
    )
    output = LogitsProcessorOutput(next_token_logits=logits.clone())
    sampler = TorchCategoricalSampler(
        output,
        torch.ones(1, dtype=torch.int32),
        seed=19,
        allowed=allowed,
        bias=torch.tensor([0.0, 0.0, 0.0, 80.0]),
        record_logprobs=True,
    )
    tokens, _ = sample_with_diagnostics(
        output,
        info,
        ForwardOutputLayout(1, 1, 0, 1),
        sampler,
        enabled=True,
    )
    assert tokens.tolist() == [3]
    assert output.next_token_logits.argmax(1).tolist() == [3]
    expected_c = 0.4 / 0.45 if with_grammar else 0.95
    torch.testing.assert_close(
        output.next_token_cumulative_probs,
        torch.tensor([expected_c]),
        rtol=3e-7,
        atol=1e-7,
    )
    expected_lp = (
        expected_scores.log_softmax(-1).gather(1, tokens.long()[:, None]).squeeze(1)
    )
    assert torch.equal(output.next_token_logprobs, expected_lp)


@pytest.mark.parametrize("prefill_outputs", [0, 1])
def test_mixed_open_prefills_and_chain_rows_use_their_own_grammar_masks(
    prefill_outputs,
):
    layout = ForwardOutputLayout(2, prefill_outputs, 2, 3)
    rows = layout.num_output_tokens
    logits = torch.randn(rows, 4, generator=torch.Generator().manual_seed(43))
    # Completing prefill uses mask row 0; open prefill rows have empty masks
    # and no logits. Decode requests use mask rows 6..11, in chain order.
    decode_words = [13, 11, 7, 3, 5, 9]
    words = torch.tensor([14, 1, 2, 0, 0, 0] + decode_words, dtype=torch.int32)[:, None]
    packed_words = ([14] if prefill_outputs else []) + decode_words
    allowed = torch.tensor(
        [[(word >> token) & 1 for token in range(4)] for word in packed_words],
        dtype=torch.bool,
    )
    info = SamplingBatchInfo(
        device="cpu",
        vocab_size=4,
        vocab_mask=words,
        apply_vocab_mask=PackedGrammar(4),
        req_pool_indices=torch.tensor([1, 2, 3, 4]),
    )
    output = LogitsProcessorOutput(next_token_logits=logits.clone())
    lengths = torch.tensor([1 if prefill_outputs else 0, 0, 3, 3], dtype=torch.int32)
    sampler = TorchCategoricalSampler(
        output,
        lengths,
        seed=55,
        allowed=allowed,
        bias=None,
        record_logprobs=True,
    )
    tokens, actual_lengths = sample_with_diagnostics(
        output, info, layout, sampler, enabled=True
    )
    assert torch.equal(actual_lengths, lengths)
    expected_scores = logits.masked_fill(~allowed, -torch.inf)
    torch.testing.assert_close(
        output.next_token_cumulative_probs.double(),
        reference_mass(expected_scores, tokens),
        rtol=3e-7,
        atol=1e-7,
    )
    expected_lp = (
        expected_scores.log_softmax(-1).gather(1, tokens.long()[:, None]).squeeze(1)
    )
    assert torch.equal(output.next_token_logprobs, expected_lp)


def test_zero_emission_needs_no_scores_or_grammar_applier():
    output = LogitsProcessorOutput(
        next_token_logits=torch.empty(0, 4),
        next_token_ids=torch.empty(0, dtype=torch.int32),
        next_token_cumulative_probs=torch.tensor([0.9]),
    )
    sampler = ArgmaxResultSampler(output, torch.zeros(2, dtype=torch.int32))
    info = SamplingBatchInfo(
        device="cpu", vocab_mask=torch.zeros(6, 1, dtype=torch.int32)
    )
    result = sample_with_diagnostics(
        output,
        info,
        ForwardOutputLayout(2, 0, 0, 3),
        sampler,
        enabled=True,
    )
    assert result is sampler.result
    assert result[0].numel() == 0
    assert result[1].tolist() == [0, 0]
    assert output.next_token_logprobs is None
    assert output.next_token_cumulative_probs is None


@pytest.mark.parametrize("failure", ["missing_scores", "short_scores", "short_tokens"])
def test_requested_diagnostic_rejects_incomplete_sampler_outputs(failure):
    output = LogitsProcessorOutput(next_token_logits=torch.zeros(2, 4))
    cls = {
        "missing_scores": TorchCategoricalSampler,
        "short_scores": TruncatedLogprobsSampler,
        "short_tokens": TruncatedTokensSampler,
    }[failure]
    sampler = cls(
        output,
        torch.ones(2, dtype=torch.int32),
        seed=29,
        allowed=None,
        bias=None,
        record_logprobs=failure != "missing_scores",
    )
    reason = "output layout" if failure == "short_tokens" else "aligned output logprobs"
    with pytest.raises(ValueError, match=reason):
        sample_with_diagnostics(
            output,
            SamplingBatchInfo(device="cpu"),
            ForwardOutputLayout(2, 2, 0, 1),
            sampler,
            enabled=True,
        )


def test_mask_without_an_applier_is_a_configuration_error():
    output = LogitsProcessorOutput(next_token_logits=torch.zeros(1, 4))
    sampler = TorchCategoricalSampler(
        output,
        torch.ones(1, dtype=torch.int32),
        seed=1,
        allowed=None,
        bias=None,
        record_logprobs=True,
    )
    with pytest.raises(ValueError, match="apply_vocab_mask"):
        sample_with_diagnostics(
            output,
            SamplingBatchInfo(device="cpu", vocab_mask=torch.tensor([[15]])),
            ForwardOutputLayout(1, 1, 0, 1),
            sampler,
            enabled=True,
        )

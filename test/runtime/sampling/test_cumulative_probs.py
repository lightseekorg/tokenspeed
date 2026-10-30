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

"""Real CPU arithmetic for the portable cumulative-probability diagnostic."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ci_system.ci_register import register_cuda_ci

# CPU arithmetic, scheduled with the sampling regressions.
register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.sampling.diagnostics import gather_token_cumulative_probs


def reference_cumulative_probs(logits, tokens):
    """Independent FP64 softmax and explicit sorted-prefix reference."""
    probabilities = logits.double().softmax(-1)
    result = []
    for row, selected in enumerate(tokens.tolist()):
        scores = logits[row].tolist()
        order = sorted(range(len(scores)), key=lambda token: (-scores[token], token))
        before = order[: order.index(selected)]
        result.append(probabilities[row, before].sum())
    return torch.stack(result)


def test_cumulative_mass_excludes_the_selected_token():
    probabilities = torch.tensor([0.55, 0.25, 0.15, 0.05])
    logits = probabilities.log().repeat(4, 1)
    tokens = torch.arange(4)
    torch.testing.assert_close(
        gather_token_cumulative_probs(logits, tokens),
        torch.tensor([0.0, 0.55, 0.80, 0.95]),
        rtol=3e-7,
        atol=1e-7,
    )


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_exact_ties_use_token_id_and_masked_tokens_contribute_no_mass(dtype):
    logits = torch.tensor([[0.0, 1.0, 1.0, -torch.inf, 0.0]]).repeat(5, 1)
    tokens = torch.tensor([1, 2, 0, 4, 3], dtype=dtype)
    actual = gather_token_cumulative_probs(logits, tokens)
    torch.testing.assert_close(
        actual.double(),
        reference_cumulative_probs(logits, tokens),
        rtol=3e-7,
        atol=1e-7,
    )
    assert actual[0].item() == 0
    assert actual[-1].item() == 1


def test_finite_tail_cannot_round_above_one():
    # This row previously produced C=1.0000002384 when reconstructed from
    # the rounded selected-token logprob (-16.0343666).
    logits = torch.randn(1, 256, generator=torch.Generator().manual_seed(0))
    logits[0, 255] = -10
    tokens = torch.tensor([255])
    actual = gather_token_cumulative_probs(logits, tokens)
    assert 0 <= actual.item() <= 1
    torch.testing.assert_close(
        actual.double(), reference_cumulative_probs(logits, tokens), rtol=0, atol=2e-7
    )


def test_extreme_finite_logits_do_not_cancel_against_selected_logprob():
    logits = torch.tensor(
        [[0.0, -1.0, -1e20], [1e20, -1e20, 0.0], [-1e20, -1e20, -1e20]]
    )
    tokens = torch.tensor([2, 0, 1])
    actual = gather_token_cumulative_probs(logits, tokens)
    torch.testing.assert_close(
        actual, torch.tensor([1.0, 0.0, 1 / 3]), rtol=0, atol=3e-8
    )


def test_negative_infinity_selected_token_has_endpoint_one():
    logits = torch.tensor([[0.0, -1.0, -torch.inf, -torch.inf]]).repeat(2, 1)
    actual = gather_token_cumulative_probs(logits, torch.tensor([2, 3]))
    torch.testing.assert_close(actual, torch.ones(2), rtol=0, atol=0)


def test_invalid_distributions_remain_nan_without_poisoning_valid_rows():
    logits = torch.tensor(
        [
            [-torch.inf, -torch.inf, -torch.inf],
            [0.0, torch.nan, -1.0],
            [torch.inf, 0.0, -1.0],
            [torch.nan, torch.nan, torch.nan],
            [0.0, -torch.inf, -1.0],
        ]
    )
    actual = gather_token_cumulative_probs(logits, torch.tensor([0, 2, 1, 0, 0]))
    assert torch.isnan(actual[:4]).all()
    assert actual[4].item() == 0


def test_common_logit_offset_does_not_change_cumulative_mass():
    logits = torch.tensor([[0.0, 0.5, -0.5, 1.0], [0.0, 0.0, 0.0, 0.0]])
    tokens = torch.tensor([1, 2])
    reference = gather_token_cumulative_probs(logits, tokens)
    for offset in (-131072.0, 131072.0):
        torch.testing.assert_close(
            gather_token_cumulative_probs(logits + offset, tokens),
            reference,
            rtol=0,
            atol=0,
        )
    torch.testing.assert_close(
        gather_token_cumulative_probs(torch.full((1, 4), float(2**30)), tokens[1:]),
        torch.tensor([0.5]),
        rtol=0,
        atol=0,
    )


def test_finite_float64_scores_above_float32_range_stay_normalizable():
    logits = torch.tensor(
        [[1e100, 1e100, 1e100, 1e100], [1e100, -1e100, 0.0, 1e100]],
        dtype=torch.float64,
    )
    tokens = torch.tensor([2, 3])
    actual = gather_token_cumulative_probs(logits, tokens)
    torch.testing.assert_close(actual, torch.tensor([0.5, 0.5]), rtol=0, atol=0)
    assert torch.isfinite(actual).all()


def test_float64_near_ties_keep_the_original_score_order():
    logits = torch.tensor([[1.0, 1.0 + 1e-10, 0.0]], dtype=torch.float64).repeat(2, 1)
    tokens = torch.tensor([0, 1])
    actual = gather_token_cumulative_probs(logits, tokens)
    torch.testing.assert_close(
        actual.double(),
        reference_cumulative_probs(logits, tokens),
        rtol=3e-7,
        atol=1e-7,
    )
    assert actual[0].item() > 0.4
    assert actual[1].item() == 0


def test_cumulative_diagnostic_does_not_advance_cpu_randomness():
    logits = torch.tensor([[0.0, 1.0, -1.0, 0.5]])
    tokens = torch.tensor([3])
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator(device="cpu").manual_seed(731).get_state())
        before = torch.get_rng_state().clone()
        expected_next_values = torch.rand(12)
        torch.set_rng_state(before)
        gather_token_cumulative_probs(logits, tokens)
        assert torch.equal(torch.get_rng_state(), before)
        assert torch.equal(torch.rand(12), expected_next_values)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_floating_inputs_match_normalized_float64_reference(dtype):
    logits = (torch.randn(6, 257, generator=torch.Generator().manual_seed(23)) * 3).to(
        dtype
    )
    tokens = torch.tensor([230, 17, 1, 256, 80, 45])
    actual = gather_token_cumulative_probs(logits, tokens)
    torch.testing.assert_close(
        actual.double(),
        reference_cumulative_probs(logits, tokens),
        rtol=3e-7,
        atol=1e-7,
    )
    assert (actual >= 0).all() and (actual <= 1).all()


def test_rows_are_independent_for_strided_inputs_and_permutations():
    logits = torch.randn(8, 38, generator=torch.Generator().manual_seed(24))[::2, 1::2]
    tokens = torch.tensor([2, 0, 7, 0, 15, 0, 18, 0])[::2]
    expected = reference_cumulative_probs(logits, tokens)
    actual = gather_token_cumulative_probs(logits, tokens)
    torch.testing.assert_close(actual.double(), expected, rtol=3e-7, atol=1e-7)
    for row in range(4):
        torch.testing.assert_close(
            gather_token_cumulative_probs(logits[row : row + 1], tokens[row : row + 1]),
            actual[row : row + 1],
            rtol=3e-7,
            atol=1e-7,
        )
    order = torch.tensor([2, 0, 3, 1])
    torch.testing.assert_close(
        gather_token_cumulative_probs(logits[order], tokens[order]),
        actual[order],
        rtol=3e-7,
        atol=1e-7,
    )


def test_diagnostic_does_not_modify_scores_tokens_or_existing_logprobs():
    logits = torch.randn(4, 19, generator=torch.Generator().manual_seed(25))
    tokens = torch.tensor([2, 7, 15, 18])
    original_logits, original_tokens = logits.clone(), tokens.clone()
    logprobs = logits.log_softmax(-1, dtype=torch.float32).gather(1, tokens[:, None])
    gather_token_cumulative_probs(logits, tokens)
    rescored = logits.log_softmax(-1, dtype=torch.float32).gather(1, tokens[:, None])
    assert torch.equal(logits, original_logits)
    assert torch.equal(tokens, original_tokens)
    assert torch.equal(rescored, logprobs)


def test_empty_batch_preserves_the_empty_result_shape():
    actual = gather_token_cumulative_probs(
        torch.empty(0, 17), torch.empty(0, dtype=torch.int64)
    )
    assert actual.shape == (0,)
    assert actual.dtype == torch.float32


@pytest.mark.parametrize(
    "logits,tokens,message",
    [
        (torch.empty(4), torch.tensor([0]), "logits must be"),
        (torch.empty(1, 0), torch.tensor([0]), "nonempty vocabulary"),
        (torch.ones(1, 4, dtype=torch.int32), torch.tensor([0]), "floating-point"),
        (torch.ones(2, 4), torch.tensor([0]), "one entry"),
        (torch.ones(1, 4), torch.tensor([0.0]), "int32 or int64"),
    ],
)
def test_invalid_input_geometry_and_dtypes_are_rejected(logits, tokens, message):
    with pytest.raises(ValueError, match=message):
        gather_token_cumulative_probs(logits, tokens)

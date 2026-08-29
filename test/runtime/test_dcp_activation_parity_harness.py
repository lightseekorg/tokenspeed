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

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


_HARNESS_PATH = Path(__file__).parents[1] / "manual" / "dcp_activation_parity.py"
_SPEC = importlib.util.spec_from_file_location("dcp_activation_parity", _HARNESS_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_HARNESS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HARNESS)

_LOGIT_HARNESS_PATH = Path(__file__).parents[1] / "manual" / "dcp_logit_parity.py"
_LOGIT_SPEC = importlib.util.spec_from_file_location(
    "dcp_logit_parity", _LOGIT_HARNESS_PATH
)
assert _LOGIT_SPEC is not None and _LOGIT_SPEC.loader is not None
_LOGIT_HARNESS = importlib.util.module_from_spec(_LOGIT_SPEC)
_LOGIT_SPEC.loader.exec_module(_LOGIT_HARNESS)


def _args(**overrides):
    values = {
        "require_hidden_states": False,
        "hidden_atol": 3e-2,
        "logprob_atol": 2e-2,
        "rtol": 0.0,
        "allow_output_id_divergence": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _result(ids: list[int], logprobs: list[float], *, cached_tokens: int = 0):
    return {
        "output_ids": ids,
        "meta_info": {
            "cached_tokens": cached_tokens,
            "output_token_logprobs": [
                [logprob, float(token_id)] for logprob, token_id in zip(logprobs, ids)
            ],
        },
    }


def test_compare_rejects_result_count_mismatch() -> None:
    with pytest.raises(AssertionError, match="request pass count differs"):
        _HARNESS._compare([_result([1], [-0.1])], [], _args())


def test_compare_does_not_compare_logprobs_after_token_divergence() -> None:
    actual = [
        _result([1, 9, 8], [-0.1, -100.0, -100.0]),
        _result([1], [-0.1]),
    ]
    expected = [
        _result([1, 2, 3], [-0.1, 100.0, 100.0]),
        _result([1], [-0.1]),
    ]

    with pytest.raises(AssertionError) as error:
        _HARNESS._compare(actual, expected, _args())

    message = str(error.value)
    assert "cold output token IDs first differ at 1" in message
    assert "matching_history_sampled_logprobs" not in message


def test_compare_can_accept_measured_tp_output_id_noise() -> None:
    actual = [
        _result([1, 9], [-0.1, -100.0]),
        _result([3], [-0.2]),
    ]
    expected = [
        _result([1, 2], [-0.1, 100.0]),
        _result([3], [-0.2]),
    ]

    _HARNESS._compare(
        actual,
        expected,
        _args(allow_output_id_divergence=True),
    )


def test_compare_keeps_matching_history_logprob_check() -> None:
    actual = [
        _result([1, 9], [-0.5, -100.0]),
        _result([1], [-0.1]),
    ]
    expected = [
        _result([1, 2], [-0.1, 100.0]),
        _result([1], [-0.1]),
    ]

    with pytest.raises(AssertionError) as error:
        _HARNESS._compare(actual, expected, _args())

    assert "cold.matching_history_sampled_logprobs[0] differs" in str(error.value)


def test_matching_history_rejects_logprob_token_mismatch() -> None:
    result = _result([1], [-0.1])
    result["meta_info"]["output_token_logprobs"][0][1] = 2.0

    with pytest.raises(AssertionError, match="sampled-logprob token id"):
        _HARNESS._matching_history_logprobs(result, [1], 1)


def test_replay_suffix_avoids_first_cold_output_token() -> None:
    initial = _HARNESS._choose_replay_suffix(4096, 160000, [])
    adjusted = _HARNESS._choose_replay_suffix(4096, 160000, [initial])

    assert adjusted != initial
    assert 0 < adjusted < 160000


def test_cache_evidence_requires_cold_miss_and_full_prefix_hit() -> None:
    cold = _result([1], [-0.1], cached_tokens=0)
    replay = _result([1], [-0.1], cached_tokens=4096)

    _HARNESS._validate_cache_evidence(cold, replay, 4096)

    with pytest.raises(AssertionError, match="cold request reported"):
        _HARNESS._validate_cache_evidence(
            _result([1], [-0.1], cached_tokens=256), replay, 4096
        )
    with pytest.raises(AssertionError, match="prefix-cache hit"):
        _HARNESS._validate_cache_evidence(
            cold, _result([1], [-0.1], cached_tokens=0), 4096
        )


def test_reference_metadata_requires_same_source_and_request_shape() -> None:
    args = SimpleNamespace(
        source_id="commit+diff-a",
        checkpoint="checkpoint-a",
        model="model-a",
        prompt_tokens=4096,
        decode_tokens=16,
        vocab_size=160000,
        prefix_granularity=256,
        fixed_context_prefill_probes=False,
        allow_output_id_divergence=False,
    )
    metadata = {
        "schema_version": 2,
        "source_id": "commit+diff-b",
        "checkpoint": "checkpoint-a",
        "model": "model-a",
        "prompt_tokens": 4096,
        "decode_tokens": 16,
        "vocab_size": 160000,
        "prefix_granularity": 256,
        "fixed_context_prefill_probes": False,
        "allow_output_id_divergence": False,
    }

    with pytest.raises(AssertionError, match="source_id"):
        _HARNESS._validate_reference_metadata(metadata, args)


def test_full_logit_harness_requires_identical_decode_context() -> None:
    expected = {
        "forward_mode": "DECODE",
        "input_ids": torch.tensor([11]),
        "positions": torch.tensor([4096]),
    }
    actual = {
        "forward_mode": "DECODE",
        "input_ids": torch.tensor([11]),
        "positions": torch.tensor([4096]),
    }
    assert _LOGIT_HARNESS._same_context(actual, expected)

    actual["input_ids"] = torch.tensor([12])
    assert not _LOGIT_HARNESS._same_context(actual, expected)


def _logit_step(logits, *, token=11, position=4096, mode="DECODE"):
    return {
        "forward_mode": mode,
        "input_ids": torch.tensor([token]),
        "positions": torch.tensor([position]),
        "logits": torch.tensor([logits], dtype=torch.float32),
    }


def test_full_logit_compare_checks_tolerance_and_argmax() -> None:
    expected = [_logit_step([2.0, 1.0, -1.0])]
    within_tolerance = [_logit_step([2.01, 1.0, -1.0])]
    _LOGIT_HARNESS._compare_steps(
        within_tolerance,
        expected,
        atol=0.02,
        require_argmax=True,
        topk_report=2,
    )

    with pytest.raises(AssertionError, match="logits exceed"):
        _LOGIT_HARNESS._compare_steps(
            [_logit_step([2.1, 1.0, -1.0])],
            expected,
            atol=0.02,
            topk_report=2,
        )
    with pytest.raises(AssertionError, match="next-token argmax differs"):
        _LOGIT_HARNESS._compare_steps(
            [_logit_step([1.0, 2.0, -1.0])],
            expected,
            atol=10.0,
            require_argmax=True,
            topk_report=2,
        )


def test_full_logit_compare_rejects_shape_context_and_step_count() -> None:
    expected = [_logit_step([2.0, 1.0, -1.0])]
    with pytest.raises(AssertionError, match="forward-step count differs"):
        _LOGIT_HARNESS._compare_steps([], expected, atol=0.1)
    with pytest.raises(AssertionError, match="input context"):
        _LOGIT_HARNESS._compare_steps(
            [_logit_step([2.0, 1.0, -1.0], token=12)], expected, atol=0.1
        )
    with pytest.raises(AssertionError, match="logit shape differs"):
        _LOGIT_HARNESS._compare_steps([_logit_step([2.0, 1.0])], expected, atol=0.1)


def test_full_logit_window_selects_largest_extend_and_decode_chain() -> None:
    steps = [
        _logit_step([1.0], token=1, position=0, mode="EXTEND"),
        _logit_step([1.0], token=2, position=1),
        {
            **_logit_step([1.0], token=3, position=2, mode="EXTEND"),
            "input_ids": torch.tensor([3, 4, 5]),
            "positions": torch.tensor([2, 3, 4]),
        },
        _logit_step([1.0], token=6, position=5),
        _logit_step([1.0], token=7, position=6, mode="EXTEND"),
    ]
    selected = _LOGIT_HARNESS._largest_extend_window(steps)
    assert len(selected) == 2
    assert selected[0] is steps[2]
    assert selected[1] is steps[3]

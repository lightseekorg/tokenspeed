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

"""The native frontend keeps diagnostics aligned while joining output chunks."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.entrypoints.wire_driver import Completion


def _chunk(ids, logprobs, cumulative_probs, reason):
    return {
        "output_ids": ids,
        "output_token_logprobs": logprobs,
        "output_token_cumulative_probs": cumulative_probs,
        "finish_reason": reason,
        "finish_details": None if reason is None else {"type": reason},
        "prompt_tokens": 3,
        "completion_tokens": 3,
        "cached_tokens": 0,
        "input_token_logprobs": None,
        "spec_verify_ct": None,
        "accept_draft_tokens": None,
    }


def test_completion_joins_diagnostics_with_their_tokens_and_logprobs():
    completion = Completion()
    completion.add(_chunk([4, 9], [-0.2, -2.0], [0.8, 0.99], None))
    completion.add(_chunk([2], [-0.4], [0.9], "stop"))
    assert completion.output_ids == [4, 9, 2]
    assert completion.output_token_logprobs == [-0.2, -2.0, -0.4]
    assert completion.output_token_cumulative_probs == [0.8, 0.99, 0.9]
    assert completion.finish_reason == "stop"


def test_empty_finish_chunk_does_not_repeat_diagnostics():
    completion = Completion()
    completion.add(_chunk([4], [-0.2], [0.8], None))
    completion.add(_chunk([], [], [], "length"))
    assert completion.output_ids == [4]
    assert completion.output_token_logprobs == [-0.2]
    assert completion.output_token_cumulative_probs == [0.8]
    assert completion.finish_reason == "length"


def test_completion_without_diagnostics_keeps_the_column_empty():
    completion = Completion()
    completion.add(_chunk([4], [-0.2], [], "length"))
    assert completion.output_token_cumulative_probs == []


@pytest.mark.parametrize("logprobs,probs", [([-0.2], [0.8, 0.9]), ([], [0.8])])
def test_completion_rejects_misaligned_diagnostics_before_adding_tokens(
    logprobs, probs
):
    completion = Completion()
    with pytest.raises(ValueError, match="must match"):
        completion.add(_chunk([4], logprobs, probs, "length"))
    assert completion.output_ids == []
    assert completion.output_token_logprobs == []
    assert completion.output_token_cumulative_probs == []

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

"""Request-level validation of ``SamplingParams.verify``."""

import math
import os
import sys

import pytest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.sampling.sampling_params import SamplingParams  # noqa: E402

VOCAB_SIZE = 32


@pytest.mark.parametrize("repetition_penalty", [0.5, 1.0, 2.0])
def test_repetition_penalty_in_range_is_accepted(repetition_penalty):
    SamplingParams(repetition_penalty=repetition_penalty).verify(VOCAB_SIZE)


# The penalty divides positive logits of already-seen tokens, so 0 would
# turn them into inf.
@pytest.mark.parametrize("repetition_penalty", [0.0, -0.5, 2.5])
def test_repetition_penalty_out_of_range_is_rejected(repetition_penalty):
    with pytest.raises(ValueError, match=r"repetition_penalty must be in \(0, 2\]"):
        SamplingParams(repetition_penalty=repetition_penalty).verify(VOCAB_SIZE)


def test_nan_temperature_is_rejected():
    with pytest.raises(ValueError, match="temperature must be non-negative"):
        SamplingParams(temperature=math.nan).verify(VOCAB_SIZE)


@pytest.mark.parametrize("temperature", [0.0, 0.7, 1.0])
def test_finite_temperature_is_accepted(temperature):
    SamplingParams(temperature=temperature).verify(VOCAB_SIZE)


# The output loop takes ``output_length % stream_interval``.
@pytest.mark.parametrize("stream_interval", [0, -1])
def test_stream_interval_below_one_is_rejected(stream_interval):
    with pytest.raises(ValueError, match="stream_interval must be at least 1"):
        SamplingParams(stream_interval=stream_interval).verify(VOCAB_SIZE)


@pytest.mark.parametrize("stream_interval", [None, 1, 20])
def test_stream_interval_is_accepted(stream_interval):
    SamplingParams(stream_interval=stream_interval).verify(VOCAB_SIZE)

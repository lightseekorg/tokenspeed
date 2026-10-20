# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""``SamplingParams.verify`` range checks.

Each check has to match its own message: an inclusive bound that the message
excludes is how a bad value reaches the penalty kernels and corrupts a
distribution. ``repetition_penalty`` divides positive logits by the penalty, so
``0.0`` would turn every repeated token into ``+inf`` at sample time instead of
being rejected at the request boundary.
"""

import pytest

from tokenspeed.runtime.sampling.sampling_params import SamplingParams


def test_repetition_penalty_rejects_zero():
    params = SamplingParams(repetition_penalty=0.0)
    with pytest.raises(ValueError, match=r"repetition_penalty must be in \(0, 2\]"):
        params.verify(vocab_size=32)


def test_repetition_penalty_rejects_negative():
    params = SamplingParams(repetition_penalty=-0.5)
    with pytest.raises(ValueError, match=r"repetition_penalty must be in \(0, 2\]"):
        params.verify(vocab_size=32)


def test_repetition_penalty_accepts_positive_and_upper_bound():
    SamplingParams(repetition_penalty=1.0).verify(vocab_size=32)
    SamplingParams(repetition_penalty=2.0).verify(vocab_size=32)


def test_repetition_penalty_rejects_above_upper_bound():
    params = SamplingParams(repetition_penalty=2.5)
    with pytest.raises(ValueError, match=r"repetition_penalty must be in \(0, 2\]"):
        params.verify(vocab_size=32)


def test_min_p_stays_inclusive_at_zero():
    # Guards the neighbouring check against a blanket flip of every bound in
    # this file: min_p=0 is valid and must keep passing.
    SamplingParams(min_p=0.0).verify(vocab_size=32)
    SamplingParams(min_p=1.0).verify(vocab_size=32)

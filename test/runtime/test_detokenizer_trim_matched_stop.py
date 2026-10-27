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

"""Tests for ``trim_matched_stop``."""

import os
import sys

import pytest

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.engine.detokenizer import trim_matched_stop  # noqa: E402


@pytest.mark.parametrize("stop_token_id", [0, 7])
def test_matched_stop_token_is_dropped(stop_token_id):
    # Id 0 is a valid stop token (for example a model whose EOS id is 0).
    reason = {"type": "stop", "matched": stop_token_id}
    assert trim_matched_stop([11, 12, stop_token_id], reason, False) == [11, 12]


def test_matched_stop_string_truncates_output():
    reason = {"type": "stop", "matched": "STOP"}
    assert trim_matched_stop("answer STOP tail", reason, False) == "answer "


@pytest.mark.parametrize("matched", [None, "", []])
def test_empty_match_leaves_output_unchanged(matched):
    reason = {"type": "stop", "matched": matched}
    assert trim_matched_stop("answer", reason, False) == "answer"
    assert trim_matched_stop([11, 12], reason, False) == [11, 12]


def test_no_stop_trim_and_unfinished_leave_output_unchanged():
    assert trim_matched_stop([11, 0], {"type": "stop", "matched": 0}, True) == [11, 0]
    assert trim_matched_stop([11, 0], None, False) == [11, 0]

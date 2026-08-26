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

"""Metadata-side ``cu_extend_seq_lens_cpu`` construction for linear attention.

The tuple must equal the contents of ``query_start_loc`` (a wrong hint
silently corrupts the CuteDSL host chunk plan). Both are built together by
``init_forward_metadata`` once per extend batch — mirroring MHA's
``cu_extend_seq_lens_cpu`` — so the builder raises on absence or length
misalignment instead of silently degrading to the wrapper's boundary
re-read.
"""

from __future__ import annotations

import os
import sys

_TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TEST_DIR)
sys.path.insert(0, os.path.dirname(_TEST_DIR))

import pytest
import torch
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")

from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (
    _build_cu_extend_seq_lens_cpu,
)


def test_prefix_sum_matches_query_start_loc_contents() -> None:
    lens = torch.tensor([3, 5, 2], dtype=torch.int32)
    assert _build_cu_extend_seq_lens_cpu(lens, expected_len=4) == (0, 3, 8, 10)


def test_absent_lens_raise() -> None:
    with pytest.raises(RuntimeError, match="host extend lengths"):
        _build_cu_extend_seq_lens_cpu(None, expected_len=4)


def test_length_misalignment_raises() -> None:
    lens = torch.tensor([3, 5], dtype=torch.int32)
    # query_start_loc has 4 entries but only 2 lens -> 3 bounds: mismatch.
    with pytest.raises(RuntimeError, match="disagree with query_start_loc"):
        _build_cu_extend_seq_lens_cpu(lens, expected_len=4)
    # And the over-long case.
    with pytest.raises(RuntimeError, match="disagree with query_start_loc"):
        _build_cu_extend_seq_lens_cpu(lens, expected_len=2)


def test_single_sequence() -> None:
    lens = torch.tensor([129104], dtype=torch.int32)
    assert _build_cu_extend_seq_lens_cpu(lens, expected_len=2) == (0, 129104)

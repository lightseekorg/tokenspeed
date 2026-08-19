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

"""Runtime-side ``cu_seqlens_cpu`` hint construction for the KDA backend.

The hint tuple must equal the contents of ``query_start_loc`` (a wrong hint
silently corrupts the CuteDSL host chunk plan), so the builder returns
``None`` on any absence or length misalignment rather than guessing.
"""

from __future__ import annotations

import torch

from tokenspeed.runtime.layers.attention.backends.hybrid_kda import (
    _cu_seqlens_cpu_hint,
)


def test_prefix_sum_matches_query_start_loc_contents() -> None:
    lens = torch.tensor([3, 5, 2], dtype=torch.int32)
    assert _cu_seqlens_cpu_hint(lens, expected_len=4) == (0, 3, 8, 10)


def test_absent_lens_yield_none() -> None:
    assert _cu_seqlens_cpu_hint(None, expected_len=4) is None


def test_length_misalignment_yields_none() -> None:
    lens = torch.tensor([3, 5], dtype=torch.int32)
    # query_start_loc has 4 entries but only 2 lens -> 3 bounds: mismatch.
    assert _cu_seqlens_cpu_hint(lens, expected_len=4) is None
    # And the over-long case.
    assert _cu_seqlens_cpu_hint(lens, expected_len=2) is None


def test_single_sequence() -> None:
    lens = torch.tensor([129104], dtype=torch.int32)
    assert _cu_seqlens_cpu_hint(lens, expected_len=2) == (0, 129104)

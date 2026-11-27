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

"""Retained verification preserves original QKV and projection storage."""

from __future__ import annotations

import pytest
import torch
from kda_retained_reference import make_case, reference, verify

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


@pytest.mark.parametrize("batch,width", [(1, 1), (4, 4), (8, 1)])
@pytest.mark.parametrize("strided", [False, True])
def test_retained_capture_matches_original_rows(batch, width, strided):
    case = make_case(batch, width, strided=strided, ragged=True)
    original = case["raw"].clone()
    expected = reference(case)
    output = verify(case)
    torch.testing.assert_close(case["records"].original_qkv, original, atol=0, rtol=0)
    torch.testing.assert_close(case["raw"], original, atol=0, rtol=0)
    torch.testing.assert_close(output[0].float(), expected[0], atol=2e-4, rtol=1e-2)


def test_retained_capture_rejects_overlapping_projection_storage():
    from dataclasses import replace

    case = make_case(2, 4)
    case["records"] = replace(case["records"], original_qkv=case["raw"])
    before = case["raw"].clone()
    with pytest.raises(ValueError, match="overlap|separate"):
        verify(case)
    torch.testing.assert_close(case["raw"], before, atol=0, rtol=0)


def test_retained_verify_can_leave_original_qkv_with_caller():
    from dataclasses import replace

    case = make_case(2, 4)
    case["records"] = replace(case["records"], original_qkv=None)
    expected = reference(case)
    output = verify(case)
    torch.testing.assert_close(output[0].float(), expected[0], atol=2e-4, rtol=1e-2)

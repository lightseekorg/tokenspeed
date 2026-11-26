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

"""Retained multi-token verify against independent sequential PyTorch arithmetic."""

import pytest
import torch
from kda_retained_reference import check_prefix, commit, make_case, reference, verify

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

# K3 TP8 per-rank KDA geometry.


@pytest.mark.parametrize("batch,width", [(1, 2), (3, 4), (8, 4)])
@pytest.mark.parametrize("lower_bound", [None, -1.5])
@pytest.mark.parametrize("cross_page", [False, True])
def test_retained_mtp_matches_sequential_reference(
    batch, width, lower_bound, cross_page
):
    case = make_case(batch, width, lower_bound=lower_bound, cross_page=cross_page)
    expected = reference(case)
    actual = verify(case)
    torch.testing.assert_close(actual[0].float(), expected[0], atol=2e-4, rtol=1e-2)
    accepted = torch.arange(batch, device="cuda", dtype=torch.int32) % (width + 1)
    commit(case, accepted, ended_slots=())
    check_prefix(case, accepted, expected)

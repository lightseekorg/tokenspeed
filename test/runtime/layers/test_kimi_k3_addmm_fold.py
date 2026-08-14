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

import pytest
import torch


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("m", [0, 1, 5])
def test_narrowed_staging_addmm_matches_unfolded_expression(
    dtype: torch.dtype, m: int
) -> None:
    torch.manual_seed(7)
    hidden, start, width, latent = 11, 3, 5, 4
    staging = torch.randn(m, hidden, dtype=dtype)
    prefix = torch.randn(m, hidden * 2, dtype=dtype)[:, ::2]
    routed = torch.randn(m, latent, dtype=dtype)
    weight = torch.randn(width, latent, dtype=dtype)
    expected = staging.clone()
    expected[:, start : start + width] = (
        expected[:, start : start + width]
        + prefix[:, start : start + width]
        + routed @ weight.t()
    )

    actual = staging.clone()
    target = actual[:, start : start + width]
    target += prefix[:, start : start + width]
    target.addmm_(routed, weight.t())

    torch.testing.assert_close(actual, expected)

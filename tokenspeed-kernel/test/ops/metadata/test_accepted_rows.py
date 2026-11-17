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

"""Gathering each request's last accepted row of a verify batch."""

import pytest
import torch
from tokenspeed_kernel.ops.metadata import gather_last_accepted_rows
from tokenspeed_kernel.ops.metadata.accepted_rows import (
    _gather_last_accepted_rows_kernel,
)
from utils import assert_no_triton_compile


def _expected(rows, accept_lengths, rows_per_request):
    """Request r's row r * rows_per_request - 1 + accepted, the accepted count
    clamped to the request's rows and the row clamped into the batch."""
    picks = []
    for request, accepted in enumerate(accept_lengths.tolist()):
        accepted = min(max(accepted, 0), rows_per_request)
        row = request * rows_per_request - 1 + accepted
        picks.append(min(max(row, 0), rows.shape[0] - 1))
    return rows[picks]


def _inputs(requests, rows_per_request, width, dtype, device, seed):
    torch.manual_seed(seed)
    rows = torch.randn(requests * rows_per_request, width, device=device).to(dtype)
    # In range, none, all, past the request's rows and negative.
    accepts = torch.randint(
        -2, rows_per_request + 3, (requests,), device=device, dtype=torch.int32
    )
    accepts[0] = 0
    if requests > 1:
        accepts[-1] = rows_per_request
    return rows, accepts


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    ("requests", "rows_per_request", "width"),
    ((1, 8, 4608), (3, 4, 1000), (16, 8, 64), (64, 1, 2304)),
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_gather_picks_each_requests_last_accepted_row(
    device, requests, rows_per_request, width, dtype
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    rows, accepts = _inputs(requests, rows_per_request, width, dtype, device, requests)
    expected = _expected(rows, accepts, rows_per_request)

    assert torch.equal(
        gather_last_accepted_rows(rows, accepts, rows_per_request=rows_per_request),
        expected,
    )
    out = torch.full_like(expected, 7.0)
    got = gather_last_accepted_rows(
        rows, accepts.long(), rows_per_request=rows_per_request, out=out
    )
    assert got is out and torch.equal(out, expected)


def test_gather_reads_strided_rows_and_clamps_into_a_short_batch():
    """Rows may be a column slice of a wider tensor; a batch holding fewer rows
    than its requests own (a trailing request's rows missing) clamps into it."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    wide = torch.randn(30, 96, device="cuda")
    rows = wide[:, 16:80]
    accepts = torch.tensor([4, 8, 8, 8], device="cuda", dtype=torch.int32)
    expected = _expected(rows, accepts, 8)
    assert torch.equal(
        gather_last_accepted_rows(rows, accepts, rows_per_request=8), expected
    )


def test_gather_compiles_once_across_batch_sizes():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    rows, accepts = _inputs(2, 8, 4608, torch.bfloat16, "cuda", 0)
    gather_last_accepted_rows(rows, accepts, rows_per_request=8)
    with assert_no_triton_compile(_gather_last_accepted_rows_kernel):
        for requests in (1, 3, 16, 37):
            rows, accepts = _inputs(requests, 8, 4608, torch.bfloat16, "cuda", requests)
            gather_last_accepted_rows(rows, accepts, rows_per_request=8)


def test_gather_rejects_layouts_it_cannot_index():
    rows = torch.randn(16, 8)
    accepts = torch.ones(2, dtype=torch.int32)
    with pytest.raises(ValueError, match="dense columns"):
        gather_last_accepted_rows(rows.t(), accepts, rows_per_request=8)
    with pytest.raises(ValueError, match="int32/int64"):
        gather_last_accepted_rows(rows, accepts.float(), rows_per_request=8)
    with pytest.raises(ValueError, match="positive"):
        gather_last_accepted_rows(rows, accepts, rows_per_request=0)
    with pytest.raises(ValueError, match="out must be"):
        gather_last_accepted_rows(
            rows,
            accepts,
            rows_per_request=8,
            out=torch.empty(2, 8, dtype=torch.float64),
        )

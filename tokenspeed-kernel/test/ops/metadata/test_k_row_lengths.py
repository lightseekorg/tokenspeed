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

"""The k-row expansion must equal the chain of tensor ops it replaces."""

import pytest
import torch
from tokenspeed_kernel.ops.metadata import expand_k_row_lengths, expand_k_row_requests
from tokenspeed_kernel.ops.metadata.k_row_lengths import (
    _expand_k_row_lengths_kernel,
    _expand_k_row_requests_kernel,
)
from utils import assert_no_triton_compile

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


def _reference(context_lengths, verify_lengths, k):
    """The per-token rows the DSA leaf built from tensor ops."""
    context = context_lengths.unsqueeze(1).expand(-1, k).reshape(-1)
    offsets = torch.arange(k, device=verify_lengths.device, dtype=torch.int32) - (k - 1)
    causal = verify_lengths.to(torch.int32).unsqueeze(1).add(offsets).clamp_min(0)
    return context, causal.reshape(-1)


@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("requests", [1, 3, 64, 300])
@pytest.mark.parametrize("k", [1, 2, 8])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_rows_match_the_tensor_op_chain(device, requests, k, dtype):
    g = torch.Generator().manual_seed(requests * 10 + k)
    context = torch.randint(0, 5000, (requests,), generator=g).to(dtype).to(device)
    # Short requests clamp their early causal rows at zero.
    verify = torch.randint(0, 12, (requests,), generator=g).to(dtype).to(device)
    context_out = torch.full((requests * k,), -9, dtype=dtype, device=device)
    causal_out = torch.full((requests * k,), -9, dtype=torch.int32, device=device)
    expand_k_row_lengths(
        context, verify, k=k, context_out=context_out, causal_out=causal_out
    )
    want_context, want_causal = _reference(context, verify, k)
    assert torch.equal(context_out, want_context)
    assert torch.equal(causal_out, want_causal)


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_context_rows_alone_leave_no_causal_rows(device):
    context = torch.tensor([7, 3], dtype=torch.int32, device=device)
    rows = torch.zeros(8, dtype=torch.int32, device=device)
    expand_k_row_lengths(context, None, k=4, context_out=rows, causal_out=None)
    assert rows.tolist() == [7, 7, 7, 7, 3, 3, 3, 3]


def test_request_counts_reuse_one_compiled_kernel():
    def expand(requests):
        context = torch.arange(requests, dtype=torch.int32, device="cuda") + 9
        rows = torch.empty(requests * 8, dtype=torch.int32, device="cuda")
        causal = torch.empty(requests * 8, dtype=torch.int32, device="cuda")
        expand_k_row_lengths(context, context, k=8, context_out=rows, causal_out=causal)

    expand(2)
    expand(32)
    with assert_no_triton_compile(_expand_k_row_lengths_kernel):
        for requests in (1, 3, 16, 17, 64):
            expand(requests)


def test_mismatched_shapes_are_rejected():
    context = torch.tensor([7, 3], dtype=torch.int32, device="cuda")
    rows = torch.zeros(8, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="verify lengths"):
        expand_k_row_lengths(context, None, k=4, context_out=rows, causal_out=rows)
    with pytest.raises(ValueError, match="8 contiguous"):
        expand_k_row_lengths(context, None, k=4, context_out=rows[:6], causal_out=None)
    with pytest.raises(ValueError, match="causal rows int32"):
        expand_k_row_lengths(
            context, context, k=4, context_out=rows, causal_out=rows.long()
        )


def _requests_reference(lengths, rows, k):
    """The KVP decode indexer's rows as the GLM DSA top-k built them."""
    requests = torch.arange(rows, device=lengths.device, dtype=torch.int32) // k
    ends = (
        lengths[requests.long()]
        - (k - 1)
        + torch.arange(rows, device=lengths.device) % k
    )
    return requests, ends.to(torch.int32)


@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("requests", [1, 3, 64, 300])
@pytest.mark.parametrize("k", [1, 2, 9])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_request_rows_match_the_tensor_op_chain(device, requests, k, dtype):
    g = torch.Generator().manual_seed(requests * 10 + k)
    # Short requests keep non-positive ends: nothing is clamped.
    lengths = torch.randint(0, 40, (requests,), generator=g)
    lengths[0] = 0
    lengths = lengths.to(dtype).to(device)
    for rows in (requests * k, requests * k - k // 2):
        requests_out = torch.full((rows,), -9, dtype=torch.int32, device=device)
        ends_out = torch.full((rows,), -9, dtype=torch.int32, device=device)
        expand_k_row_requests(
            lengths, k=k, requests_out=requests_out, ends_out=ends_out
        )
        want_requests, want_ends = _requests_reference(lengths, rows, k)
        assert torch.equal(requests_out, want_requests)
        assert torch.equal(ends_out, want_ends)
    assert bool((ends_out <= 0).any())


def test_request_rows_reuse_one_compiled_kernel():
    def expand(requests, k):
        lengths = torch.arange(requests, dtype=torch.int32, device="cuda") + 9
        rows = torch.empty(requests * k, dtype=torch.int32, device="cuda")
        ends = torch.empty(requests * k, dtype=torch.int32, device="cuda")
        expand_k_row_requests(lengths, k=k, requests_out=rows, ends_out=ends)
        want_rows, want_ends = _requests_reference(lengths, requests * k, k)
        assert torch.equal(rows, want_rows) and torch.equal(ends, want_ends)

    expand(2, 9)
    # Batch sizes and widths (a verify's, a draft step's) share one binary.
    with assert_no_triton_compile(_expand_k_row_requests_kernel):
        for requests, k in ((1, 9), (3, 1), (16, 4), (17, 9), (64, 1), (5, 16)):
            expand(requests, k)


def test_request_rows_reject_mismatched_shapes():
    lengths = torch.tensor([7, 3], dtype=torch.int32, device="cuda")
    rows = torch.zeros(8, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="exceed 2 requests"):
        expand_k_row_requests(
            lengths, k=3, requests_out=rows, ends_out=torch.zeros_like(rows)
        )
    with pytest.raises(ValueError, match="8 contiguous int32"):
        expand_k_row_requests(lengths, k=4, requests_out=rows, ends_out=rows.long())
    with pytest.raises(ValueError, match="dense int32/int64"):
        expand_k_row_requests(
            lengths.float(), k=4, requests_out=rows, ends_out=torch.zeros_like(rows)
        )

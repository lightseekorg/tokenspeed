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

"""DSA packed index scales remain relative to their cache field's storage."""

import pytest
import torch
from tokenspeed_kernel.ops.attention.dsa import deep_gemm as dsa_deep_gemm
from tokenspeed_kernel.ops.quantization import quantize_fp8_with_scale


@pytest.mark.parametrize("offset_bytes", [256, 64 * 132 * 65])
def test_prefill_topk_slab_view_matches_independent_storage(offset_bytes):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("requires Hopper or newer")
    kernel = getattr(dsa_deep_gemm, "deep_gemm_dsa_prefill_topk", None)
    if kernel is None:
        pytest.skip("DeepGEMM is unavailable")
    torch.manual_seed(41)
    rows, heads, dim, page, queries = 4160, 32, 128, 64, 7
    q = torch.randn((queries, heads, dim), device="cuda", dtype=torch.bfloat16)
    weights = torch.rand((queries, heads), device="cuda", dtype=torch.float32)
    keys = torch.randn((rows, dim), device="cuda", dtype=torch.bfloat16)
    fp8, scales = quantize_fp8_with_scale(
        keys, granularity="token_group", group_size=128, scale_encoding="float32"
    )
    packed = torch.cat(
        (
            fp8.view(torch.uint8).reshape(-1, page * dim),
            scales.contiguous().view(torch.uint8).reshape(-1, page * 4),
        ),
        dim=1,
    ).reshape(rows, dim + 4)
    # Another plane precedes this field in a monolithic arena. Reading its
    # zeros as this field's scales destroys scores without an illegal access.
    slab = torch.zeros(offset_bytes + packed.numel(), device="cuda", dtype=torch.uint8)
    field = slab[offset_bytes:].view_as(packed)
    field.copy_(packed)
    slots = torch.arange(4103, device="cuda", dtype=torch.int32)
    starts = torch.zeros(queries, device="cuda", dtype=torch.int32)
    ends = torch.arange(4097, 4104, device="cuda", dtype=torch.int32)
    kwargs = dict(
        topk=2048,
        softmax_scale=dim**-0.5,
        page_size=page,
        batch_invariant=True,
        candidate_lens_cpu=ends.cpu(),
    )
    expected, lengths = kernel(
        q, weights, slots, starts, ends, index_k_cache=packed, **kwargs
    )
    actual, actual_lengths = kernel(
        q, weights, slots, starts, ends, index_k_cache=field, **kwargs
    )
    assert torch.equal(actual_lengths, lengths)
    assert torch.equal(actual, expected)

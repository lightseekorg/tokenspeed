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

"""Triton MoE combine: route rows read through ``route_rows``."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.moe._triton.common import _combine


@pytest.mark.parametrize("round_weighted_routes", [False, True])
def test_combine_through_route_rows_matches_canonical_rows(
    device, round_weighted_routes
):
    num_tokens, top_k, hidden, num_experts = 7, 3, 300, 4
    generator = torch.Generator(device=device).manual_seed(2857)
    topk_ids = torch.randint(
        0, num_experts, (num_tokens, top_k), device=device, generator=generator
    )
    topk_ids[1, 2] = -1
    topk_ids[4, 0] = num_experts
    invalid = ((topk_ids < 0) | (topk_ids >= num_experts)).flatten()
    topk_weights = torch.rand(num_tokens, top_k, device=device, generator=generator)
    canonical = torch.randn(
        num_tokens * top_k, hidden, device=device, generator=generator
    ).bfloat16()
    canonical[invalid] = float("nan")

    num_routes = num_tokens * top_k
    route_rows = torch.randperm(num_routes, device=device, generator=generator).int()
    permuted = torch.full_like(canonical, float("nan"))
    permuted[route_rows.long()] = canonical
    route_rows[invalid] = 2**30

    expected = torch.empty(num_tokens, hidden, device=device, dtype=torch.bfloat16)
    actual = torch.empty_like(expected)
    for route_output, rows, output in (
        (canonical, None, expected),
        (permuted, route_rows, actual),
    ):
        _combine(
            route_output,
            topk_weights,
            output,
            topk_ids,
            num_experts,
            round_weighted_routes=round_weighted_routes,
            output_scale=1.0,
            route_rows=rows,
        )

    assert not expected.isnan().any()
    torch.testing.assert_close(
        actual.view(torch.int16), expected.view(torch.int16), atol=0, rtol=0
    )

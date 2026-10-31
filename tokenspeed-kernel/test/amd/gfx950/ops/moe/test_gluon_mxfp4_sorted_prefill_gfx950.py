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

"""Deterministic route order of the gfx950 MXFP4 package sort."""

from __future__ import annotations

import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip(
        "AMD CDNA4 is required for the Gluon MXFP4 sorted prefill tests",
        allow_module_level=True,
    )

from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4 import moe_sorting  # noqa: E402

HIDDEN = 1024
TOP_K = 8


@pytest.mark.parametrize(
    "rows, experts, start, block, minimum, runtime_rows",
    [
        (8192, 256, 0, 64, False, False),  # expert TP prefill: 256-route chunks
        (8192, 32, 64, 64, False, False),  # expert parallel: 1024-route chunks
        (2049, 256, 0, 128, False, False),
        (150, 32, 32, 32, False, True),
        (100, 32, 0, 32, True, False),  # one fused chunk
        (193, 256, 0, 16, False, False),
    ],
)
def test_deterministic_sort_is_the_stable_expert_order(
    rows, experts, start, block, minimum, runtime_rows
) -> None:
    g = torch.Generator(device="cuda").manual_seed(rows)
    ids = torch.rand(rows, 256, device="cuda", generator=g).topk(TOP_K, dim=-1)
    ids = ids.indices.to(torch.int32).contiguous()
    weights = torch.rand(rows, TOP_K, device="cuda", generator=g)

    def sort():
        return moe_sorting.gluon_moe_sorting(
            ids,
            weights,
            experts,
            HIDDEN,
            torch.bfloat16,
            block,
            use_minimum_route_chunks=minimum,
            runtime_rows=runtime_rows,
            deterministic_order=True,
            expert_start=start,
        )

    sorted_ids, sorted_weights, expert_ids, valid, _ = sort()
    # Reference: each local expert's routes in flat top-k order, padded to blocks.
    flat = ids.flatten().cpu() - start
    expected = []
    expected_weights = []
    expected_blocks = []
    for expert in range(experts):
        routes = torch.nonzero(flat == expert).flatten()
        if routes.numel() == 0:
            continue
        padded = (routes.numel() + block - 1) // block * block
        packed = (routes % TOP_K) << 24 | (routes // TOP_K)
        expected += packed.tolist() + [TOP_K << 24 | rows] * (padded - routes.numel())
        expected_weights += weights.flatten().cpu()[routes].tolist()
        expected_weights += [None] * (padded - routes.numel())
        expected_blocks += [expert] * (padded // block)
    assert int(valid[0]) == len(expected)
    assert sorted_ids[: len(expected)].cpu().tolist() == expected
    assert expert_ids[: len(expected_blocks)].cpu().tolist() == expected_blocks
    actual_weights = sorted_weights[: len(expected)].cpu().tolist()
    assert [a for a, e in zip(actual_weights, expected_weights) if e is not None] == [
        e for e in expected_weights if e is not None
    ]
    again = sort()[0]
    assert torch.equal(again[: len(expected)], sorted_ids[: len(expected)])

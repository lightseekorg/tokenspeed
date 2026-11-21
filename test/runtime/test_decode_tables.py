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

"""Compare the fused owner refresh against the existing table/write-slot path."""

import pytest
import torch

from tokenspeed.runtime.layers.attention.backends.paged.group_tables import (
    GroupTableSpec,
    GroupTableStacks,
)


def make_stacks(device, full_atom_pages):
    return GroupTableStacks(
        [
            GroupTableSpec("full", 256, 64, 80, full_atom_pages),
            GroupTableSpec("swa", 64, 64, 33, 1),
            GroupTableSpec("flat_single", 64, 64, 33, 3),
        ],
        max_bs=17,
        max_tokens_per_req=8,
        max_extend_tokens=0,
        device=device,
    )


def refresh_reference(stacks, bs, actual, raw, lengths, width, targets):
    stacks.fill(bs, actual, raw)
    stacks.compute_decode_locations(bs, lengths, width)
    if targets:
        floored = {
            floor: lengths[:bs] if floor <= 1 else lengths[:bs].clamp_min(floor)
            for _, _, floor in targets.values()
        }
        torch._foreach_copy_(
            [buf for pair in targets.values() for buf in pair[:2]],
            [
                src
                for gid, (_, _, floor) in targets.items()
                for src in (floored[floor], stacks.table(gid, bs))
            ],
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("width", [1, 4, 8])
@pytest.mark.parametrize("actual", [0, 3, 17])
@pytest.mark.parametrize("full_atom_pages", [4, 8])
def test_decode_refresh(device, width, actual, full_atom_pages):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    fused, ref = make_stacks(device, full_atom_pages), make_stacks(
        device, full_atom_pages
    )
    bs = 17
    # Strided and independent allocations, holes, and over-capacity lengths.
    raw = (
        {
            gid: (
                torch.arange(bs * 82, dtype=torch.int32, device=device).reshape(bs, 82)
                % 19
                - 2
            )[:, ::2]
            for gid in fused.group_ids
        }
        if actual
        else {}
    )
    lengths = torch.tensor(
        [
            0,
            1,
            3,
            63,
            64,
            65,
            127,
            128,
            129,
            1000,
            2048,
            2112,
            3000,
            4096,
            5120,
            5121,
            9000,
        ],
        dtype=torch.int32,
        device=device,
    )

    def outputs():
        return {
            gid: (
                torch.full((bs * 2,), -99, dtype=torch.int32, device=device)[::2],
                torch.full(
                    (bs, fused.table(gid, bs).shape[1] * 2),
                    -99,
                    dtype=torch.int32,
                    device=device,
                )[:, ::2],
                4,
            )
            for gid in fused.group_ids
        }

    dst, expected = outputs(), outputs()
    refresh_reference(ref, bs, actual, raw, lengths, width, expected)
    fused.fill_decode(bs, actual, raw, lengths, width, targets=dst)
    torch.testing.assert_close(fused.tables, ref.tables, rtol=0, atol=0)
    torch.testing.assert_close(fused.decode_locs, ref.decode_locs, rtol=0, atol=0)
    for gid in dst:
        for a, b in zip(dst[gid][:2], expected[gid][:2]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    if device == "cuda":
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fused.fill_decode(bs, actual, raw, lengths, width, targets=dst)
        torch.cuda.current_stream().wait_stream(stream)
        lengths.add_(17)
        refresh_reference(ref, bs, actual, raw, lengths, width, expected)
        graph.replay()
        torch.testing.assert_close(fused.tables, ref.tables, rtol=0, atol=0)
        torch.testing.assert_close(fused.decode_locs, ref.decode_locs, rtol=0, atol=0)
        for gid in dst:
            for a, b in zip(dst[gid][:2], expected[gid][:2]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

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

"""Admission contract for capacity-planned KDA execution."""

import pytest
import torch
from tokenspeed_kernel.ops.attention.kda import KdaPrefillCapacity


@pytest.mark.parametrize("lengths", ([1, 511], [511, 1], [1, 1], [256, 256]))
def test_capacity_admits_unequal_live_partitions(lengths):
    capacity = KdaPrefillCapacity(512, 2)
    bounds = torch.tensor([0, lengths[0], sum(lengths)])
    capacity.validate(bounds, 512)
    torch.testing.assert_close(capacity.boundaries_cpu(), torch.tensor([0, 512, 1024]))
    torch.testing.assert_close(bounds, torch.tensor([0, lengths[0], sum(lengths)]))


@pytest.mark.parametrize(
    "bounds,tokens",
    (
        ([0, 1, 513], 512),
        ([0, 0, 1], 512),
        ([1, 2, 3], 512),
        ([0, 2], 512),
        ([0, 1, 2], 256),
        ([0, 2, 1], 512),
    ),
)
def test_capacity_rejects_invalid_admission(bounds, tokens):
    with pytest.raises(ValueError):
        KdaPrefillCapacity(512, 2).validate(torch.tensor(bounds), tokens)


@pytest.mark.parametrize("tokens,sequences", ((0, 1), (1, 0), (-1, 2)))
def test_capacity_rejects_invalid_geometry(tokens, sequences):
    with pytest.raises(ValueError):
        KdaPrefillCapacity(tokens, sequences)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_conv_capacity_refresh_masks_history_loads():
    from tokenspeed_kernel.ops.attention.gdn.triton import (
        CausalConv1dPrefillMetadata,
        refresh_causal_conv1d_capacity_metadata,
    )

    maps = CausalConv1dPrefillMetadata(
        batch_indices=torch.empty(19, dtype=torch.int32, device="cuda"),
        chunk_offsets=torch.empty(19, dtype=torch.int32, device="cuda"),
        block_m=8,
    )
    # Zero length tests the conv map builder only, not KDA/native scan admission.
    for lengths in ([1, 1, 1, 125], [125, 1, 1, 1], [0, 8, 9, 16]):
        bounds = torch.tensor(
            [0] + torch.tensor(lengths).cumsum(0).tolist(), device="cuda"
        )
        refresh_causal_conv1d_capacity_metadata(bounds, maps, 128)
        expected = torch.full((19,), -1, dtype=torch.int32)
        offsets = torch.zeros(19, dtype=torch.int32)
        cursor = 0
        for row, length in enumerate(lengths):
            count = (length + 7) // 8
            expected[cursor : cursor + count] = row
            offsets[cursor : cursor + count] = torch.arange(count)
            cursor += count
        torch.testing.assert_close(maps.batch_indices.cpu(), expected)
        torch.testing.assert_close(maps.chunk_offsets.cpu(), offsets)

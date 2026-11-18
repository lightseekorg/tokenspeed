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

"""Runtime-sized metadata must neither recompile nor wrap large offsets."""

import pytest
import torch
from tokenspeed_kernel.ops.metadata.decode_tables import (
    _decode_tables_kernel,
    prepare_decode_tables,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_dynamic_shapes_do_not_compile():
    from utils import assert_no_triton_compile

    tables = torch.empty((1, 17, 80), dtype=torch.int32, device="cuda")
    locations = torch.empty((1, 17 * 8), dtype=torch.int32, device="cuda")
    lengths = torch.full((17,), 65, dtype=torch.int32, device="cuda")

    def refresh(bs, columns, width, offset):
        raw = torch.full((17, columns + offset), 7, dtype=torch.int32, device="cuda")[
            :, offset:
        ]
        prepare_decode_tables(
            (raw,),
            tables,
            locations,
            lengths[:bs],
            actual_bs=bs,
            tokens_per_req=width,
            ratios=(4,),
            atom_pages_per_block=(8,),
            pages=(80,),
            page_sizes=(64,),
            targets=(None,),
        )

    refresh(1, 17, 1, 0)
    with assert_no_triton_compile(_decode_tables_kernel):
        for args in ((3, 19, 4, 1), (16, 32, 8, 0), (17, 33, 3, 3)):
            refresh(*args)
    torch.testing.assert_close(
        locations[0, :3],
        torch.tensor([3646, 3647, 3648], device="cuda", dtype=torch.int32),
        atol=0,
        rtol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_source_offsets_beyond_int32():
    # Reserve the span but touch only its ends: the second row is > 2**31
    # int32 elements from the base, not merely > 2**31 bytes.
    if torch.cuda.mem_get_info()[0] < 9 * (1 << 30):
        pytest.skip("needs 9 GiB free addressable device memory")
    stride = (1 << 31) + 16
    storage = torch.empty(stride + 4, dtype=torch.int32, device="cuda")
    raw = storage.as_strided((2, 4), (stride, 1))
    raw.fill_(7)
    tables = torch.empty((1, 2, 16), dtype=torch.int32, device="cuda")
    locations = torch.empty((1, 8), dtype=torch.int32, device="cuda")
    lengths = torch.full((2,), 65, dtype=torch.int32, device="cuda")
    prepare_decode_tables(
        (raw,),
        tables,
        locations,
        lengths,
        actual_bs=2,
        tokens_per_req=4,
        ratios=(4,),
        atom_pages_per_block=(8,),
        pages=(16,),
        page_sizes=(64,),
        targets=(None,),
    )
    expected = torch.tensor(
        [56, 57, 58, 59] * 4, dtype=torch.int32, device="cuda"
    ).expand(2, -1)
    torch.testing.assert_close(tables[0], expected, atol=0, rtol=0)
    torch.testing.assert_close(
        locations[0],
        torch.tensor([3645, 3646, 3647, 3648] * 2, dtype=torch.int32, device="cuda"),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("actual_bs", [0, 2, 3])
@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("atoms", [(2, 1, 4), (8, 3, 16)])
def test_cpu_decode_buffers(actual_bs, width, atoms):
    ratios, pages, sizes = (2, 1, 4), (5, 7, 3), (4, 2, 2)
    raw = torch.tensor([[-1, 3, 0, 4], [5, 0, -1, 7], [2, 3, 4, 5]], dtype=torch.int32)
    source = raw.repeat_interleave(2, dim=1)[:, ::2]
    tables = torch.full((3, 4, 7), -99, dtype=torch.int32)
    locations = torch.full((3, 16), -99, dtype=torch.int32)
    lengths = torch.tensor([1, 9, 100], dtype=torch.int32)
    targets = tuple(
        (
            (
                torch.empty(6, dtype=torch.int32)[::2],
                torch.empty(3, capacity * 2, dtype=torch.int32)[:, ::2],
                floor,
            )
            if g != 1
            else None
        )
        for g, (capacity, floor) in enumerate(zip(pages, (4, 1, 1)))
    )
    expected_tables, expected_locations = tables.clone(), locations.clone()
    for g in range(3):
        for row in range(3):
            for col in range(7):
                block = col // ratios[g]
                live = row < actual_bs and col < pages[g] and block < raw.shape[1]
                value = int(raw[row, block]) if live else 0
                expected_tables[g, row, col] = (
                    value * atoms[g] + col % ratios[g] if value > 0 else 0
                )
            for step in range(width):
                pos = max(int(lengths[row]) - width + step, 0)
                page_index = pos // sizes[g]
                page = int(expected_tables[g, row, page_index]) if page_index < 7 else 0
                expected_locations[g, row * width + step] = (
                    page * sizes[g] + pos % sizes[g] if page > 0 else 0
                )
    prepare_decode_tables(
        (source,) * 3,
        tables,
        locations,
        lengths,
        actual_bs=actual_bs,
        tokens_per_req=width,
        ratios=ratios,
        atom_pages_per_block=atoms,
        pages=pages,
        page_sizes=sizes,
        targets=targets,
    )
    torch.testing.assert_close(tables, expected_tables, atol=0, rtol=0)
    torch.testing.assert_close(locations, expected_locations, atol=0, rtol=0)
    for g, target in enumerate(targets):
        if target is not None:
            expected_lengths = (
                lengths.clamp_min(target[2]) if target[2] > 1 else lengths
            )
            torch.testing.assert_close(target[0], expected_lengths, atol=0, rtol=0)
            torch.testing.assert_close(
                target[1], expected_tables[g, :3, : pages[g]], atol=0, rtol=0
            )

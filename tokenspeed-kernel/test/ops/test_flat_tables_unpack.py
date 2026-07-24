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

"""flat_tables_unpack == per-group copy + tail fill + row padding."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.kvcache.triton import (
    flat_decode_locs,
    flat_tables_unpack,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU."
)


@pytest.mark.parametrize("actual_bs,bs", [(13, 13), (13, 16), (1, 1)])
def test_unpack_matches_per_group(actual_bs, bs):
    torch.manual_seed(0)
    dev = "cuda"
    widths = [32, 64, 64, 64, 1024, 1024, 8]
    g, max_bs = len(widths), 32
    wmax = max(widths)
    # packed source: back-to-back (rows x cols) per group
    spans, flat = [], []
    for w in widths:
        vals = torch.randint(1, 5000, (actual_bs, w), dtype=torch.int32)
        spans.append((len(flat and [0]) and 0, w))  # placeholder
        spans[-1] = (sum(len(x) for x in flat), w)
        flat.append(vals.reshape(-1))
    src = torch.cat([v for v in flat]).to(dev)
    meta = torch.tensor([[off, w] for (off, w) in spans], dtype=torch.int32, device=dev)
    dst = torch.full((g, max_bs, wmax), 7, dtype=torch.int32, device=dev)
    flat_tables_unpack(src, meta, dst, bs, actual_bs=actual_bs, tail_pad=-1)

    for i, (off, w) in enumerate(spans):
        rows = src[off : off + actual_bs * w].view(actual_bs, w)
        assert torch.equal(dst[i, :actual_bs, :w], rows), i
        if w < wmax:
            assert (dst[i, :actual_bs, w:] == -1).all(), i
        if actual_bs < bs:
            assert (dst[i, actual_bs:bs] == 0).all(), i
        # rows beyond bs untouched
        assert (dst[i, bs:] == 7).all(), i


@pytest.mark.parametrize("actual_bs,bs", [(5, 8), (1, 1)])
def test_ragged_unpack_copies_tables_and_bases_into_exact_storage(actual_bs, bs):
    torch.manual_seed(1)
    dev = "cuda"
    source_widths = [3, 9, 2]
    destination_widths = [5, 9, 4]
    max_bs = 8
    source_parts = []
    meta_rows = []
    source_offset = 0
    destination_offset = 0
    references = []
    for source_width, destination_width in zip(
        source_widths, destination_widths, strict=True
    ):
        table = torch.randint(1, 5000, (actual_bs, source_width), dtype=torch.int32)
        bases = torch.randint(0, 30, (actual_bs,), dtype=torch.int32)
        table_source_offset = source_offset
        source_parts.append(table.reshape(-1))
        source_offset += table.numel()
        base_source_offset = source_offset
        source_parts.append(bases)
        source_offset += bases.numel()

        table_destination_offset = destination_offset
        destination_offset += max_bs * destination_width
        base_destination_offset = destination_offset
        destination_offset += max_bs
        meta_rows.append(
            [
                table_source_offset,
                source_width,
                table_destination_offset,
                destination_width,
                base_source_offset,
                base_destination_offset,
            ]
        )
        references.append(
            (
                table,
                bases,
                table_destination_offset,
                base_destination_offset,
                destination_width,
            )
        )

    source = torch.cat(source_parts).to(dev)
    meta = torch.tensor(meta_rows, dtype=torch.int32, device=dev)
    destination = torch.full((destination_offset,), 7, dtype=torch.int32, device=dev)
    flat_tables_unpack(
        source,
        meta,
        destination,
        bs,
        actual_bs=actual_bs,
        tail_pad=-1,
        max_cols=max(destination_widths),
    )

    for table, bases, table_offset, base_offset, destination_width in references:
        table_view = destination[
            table_offset : table_offset + max_bs * destination_width
        ].view(max_bs, destination_width)
        base_view = destination[base_offset : base_offset + max_bs]
        assert torch.equal(table_view[:actual_bs, : table.shape[1]], table.to(dev))
        assert (table_view[:actual_bs, table.shape[1] :] == -1).all()
        assert torch.equal(base_view[:actual_bs], bases.to(dev))
        assert (table_view[actual_bs:bs] == 0).all()
        assert (base_view[actual_bs:bs] == 0).all()
        assert (table_view[bs:] == 7).all()
        assert (base_view[bs:] == 7).all()


@pytest.mark.parametrize("n", [1, 3, 4])
def test_locs_multiquery(n):
    """tokens_per_req > 1 (spec verify): token t of request b lands at
    position seq[b] - n + t (clamped at 0 for graph-padded seq=1 rows),
    token-major per request, honoring the destination's row stride. n=1 is
    plain decode and calls through the default-arg form (pos = seq - 1)."""
    dev = "cuda"
    torch.manual_seed(2)
    g_att, max_bs, wmax, bs = 3, 16, 64, 5
    stack = torch.randint(1, 999, (g_att, max_bs, wmax), dtype=torch.int32, device=dev)
    ps = torch.tensor([256, 128, 128], dtype=torch.int32, device=dev)
    seq = torch.randint(1, 8000, (bs,), dtype=torch.int32, device=dev)
    seq[-1] = 1  # graph-padded row: positions clamp to 0
    out = torch.full((g_att, max_bs * n), -7, dtype=torch.int32, device=dev)
    if n == 1:
        flat_decode_locs(stack, ps, seq, out, bs)
    else:
        flat_decode_locs(stack, ps, seq, out, bs, tokens_per_req=n)
    steps = torch.arange(n, device=dev, dtype=torch.int64)
    pos = (seq.to(torch.int64).unsqueeze(1) - n + steps).clamp_min(0)  # [bs, n]
    for i in range(g_att):
        p = int(ps[i])
        pages = stack[i, :bs].gather(1, pos // p)
        ref = (pages * p + (pos % p).to(torch.int32)).reshape(-1).to(torch.int32)
        assert torch.equal(out[i, : bs * n], ref), i
        assert (out[i, bs * n :] == -7).all(), i


@pytest.mark.parametrize("n", [1, 4])
def test_locs_negative_page_routes_to_dummy(n):
    """A -1 table entry (column-tail pad / hole) must yield a slot in dummy
    page 0, never a negative loc: a stale seq_lens can point the gather past
    the bridge row's filled width (the 2026-07 DFLASH draft KV-scatter IMA)."""
    dev = "cuda"
    g_att, max_bs, wmax, bs = 2, 8, 16, 3
    stack = torch.full((g_att, max_bs, wmax), -1, dtype=torch.int32, device=dev)
    stack[:, :, :2] = 5  # only the first 2 columns are real pages
    ps = torch.tensor([256, 128], dtype=torch.int32, device=dev)
    # request 1's length reaches column 4 -> gathers the -1 tail pad
    seq = torch.tensor([100, 1100, 30], dtype=torch.int32, device=dev)
    out = torch.full((g_att, max_bs * n), -7, dtype=torch.int32, device=dev)
    flat_decode_locs(stack, ps, seq, out, bs, tokens_per_req=n)
    live = out[:, : bs * n]
    assert (live >= 0).all(), live
    for i in range(g_att):
        p = int(ps[i])
        pos = (
            seq.to(torch.int64).unsqueeze(1) - n + torch.arange(n, device=dev)
        ).clamp_min(0)
        hit_pad = stack[i, :bs].gather(1, pos // p) < 0
        # pad hits land inside dummy page 0 at the position's in-page offset
        expect = (pos % p).to(torch.int32)
        got = out[i, : bs * n].view(bs, n)
        assert torch.equal(got[hit_pad], expect[hit_pad]), i

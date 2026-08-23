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

"""The fused MLA write-location kernel must match the torch chain exactly.

These are cache write addresses: an off-by-one silently corrupts another
request's KV, so the comparison is integer equality, not a tolerance.
"""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from tokenspeed_kernel.ops.attention.triton.mla_write_locations import (  # noqa: E402
    mla_write_locations,
)


def _reference(seq_lens, table, *, page_size, q_len_per_req, batch_size):
    """The elementwise chain the kernel replaces, kept verbatim."""
    last = (seq_lens[:batch_size].to(torch.int64) - 1).clamp_min(0)
    if q_len_per_req == 1:
        positions = last.unsqueeze(1)
    else:
        steps = torch.arange(
            1 - q_len_per_req, 1, device=seq_lens.device, dtype=torch.int64
        )
        positions = (last.unsqueeze(1) + steps).clamp_min(0)
    page_indices = torch.div(positions, page_size, rounding_mode="floor")
    pages = table[:batch_size].gather(1, page_indices)
    return (
        pages.clamp_min(0).to(torch.int64) * page_size + (positions % page_size)
    ).reshape(-1)


def _inputs(batch_size, num_pages, page_size, seed=0, holes=False):
    g = torch.Generator(device="cuda").manual_seed(seed)
    high = num_pages * page_size
    seq_lens = torch.randint(
        1, high, (batch_size,), device="cuda", dtype=torch.int32, generator=g
    )
    table = torch.randint(
        1, 4096, (batch_size, num_pages), device="cuda", dtype=torch.int32, generator=g
    )
    if holes:
        table[:, ::3] = -1
    return seq_lens, table


@pytest.mark.parametrize(
    ("batch_size", "q_len_per_req"),
    [(1, 1), (1, 4), (2, 2), (3, 4), (5, 1), (8, 2), (32, 4)],
)
@pytest.mark.parametrize("page_size", [1, 64])
def test_matches_the_torch_chain(batch_size, q_len_per_req, page_size):
    seq_lens, table = _inputs(batch_size, 16, page_size, seed=batch_size)
    got = mla_write_locations(
        seq_lens,
        table,
        page_size=page_size,
        q_len_per_req=q_len_per_req,
        batch_size=batch_size,
    )
    want = _reference(
        seq_lens,
        table,
        page_size=page_size,
        q_len_per_req=q_len_per_req,
        batch_size=batch_size,
    )
    assert torch.equal(got, want), (batch_size, q_len_per_req, page_size)


@pytest.mark.parametrize("q_len_per_req", [1, 4])
def test_table_holes_clamp_to_the_null_page(q_len_per_req):
    """Negative table entries are holes; both paths must resolve them the same."""
    seq_lens, table = _inputs(4, 16, 64, seed=7, holes=True)
    got = mla_write_locations(
        seq_lens, table, page_size=64, q_len_per_req=q_len_per_req, batch_size=4
    )
    want = _reference(
        seq_lens, table, page_size=64, q_len_per_req=q_len_per_req, batch_size=4
    )
    assert torch.equal(got, want)


def test_short_sequences_clamp_at_zero():
    """A request shorter than the speculative window still writes in range."""
    table = torch.randint(1, 100, (3, 16), device="cuda", dtype=torch.int32)
    seq_lens = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32)
    got = mla_write_locations(
        seq_lens, table, page_size=64, q_len_per_req=4, batch_size=3
    )
    want = _reference(seq_lens, table, page_size=64, q_len_per_req=4, batch_size=3)
    assert torch.equal(got, want)


def test_writes_in_place_for_graph_replay():
    """Replay needs the recorded buffer, so out must keep its data_ptr."""
    seq_lens, table = _inputs(4, 16, 64, seed=3)
    out = torch.zeros(64, dtype=torch.int64, device="cuda")
    ptr = out.data_ptr()
    got = mla_write_locations(
        seq_lens, table, page_size=64, q_len_per_req=4, batch_size=4, out=out
    )
    assert got.data_ptr() == ptr
    assert torch.equal(
        got, _reference(seq_lens, table, page_size=64, q_len_per_req=4, batch_size=4)
    )
    # Rows past the live batch must be left alone, not zero-filled or scribbled.
    assert torch.equal(out[16:], torch.zeros(48, dtype=torch.int64, device="cuda"))


def test_rows_past_the_batch_are_not_read():
    """A stale row beyond batch_size must not change the answer."""
    seq_lens, table = _inputs(8, 16, 64, seed=11)
    small = mla_write_locations(
        seq_lens, table, page_size=64, q_len_per_req=4, batch_size=3
    ).clone()
    seq_lens[3:] = 999999
    table[3:] = -5
    again = mla_write_locations(
        seq_lens, table, page_size=64, q_len_per_req=4, batch_size=3
    )
    assert torch.equal(small, again)


def test_strided_inputs_are_rejected_not_misread():
    """The kernel walks these with unit-stride arithmetic; a strided view would
    silently address the wrong elements, so it has to be refused."""
    seq_lens, table = _inputs(4, 16, 64, seed=5)
    strided_lens = torch.zeros(8, device="cuda", dtype=torch.int32)[::2]
    with pytest.raises(ValueError, match="unit-stride"):
        mla_write_locations(
            strided_lens, table, page_size=64, q_len_per_req=1, batch_size=4
        )
    wide = torch.zeros((4, 32), device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="unit-stride"):
        mla_write_locations(
            seq_lens, wide[:, ::2], page_size=64, q_len_per_req=1, batch_size=4
        )
    backing = torch.zeros(64, dtype=torch.int64, device="cuda")
    with pytest.raises(ValueError, match="unit-stride"):
        mla_write_locations(
            seq_lens,
            table,
            page_size=64,
            q_len_per_req=1,
            batch_size=4,
            out=backing[::2],
        )


def test_output_aliasing_the_table_is_rejected():
    """Read and write of one allocation in a single launch has no ordering."""
    table = torch.zeros((4, 16), device="cuda", dtype=torch.int64)
    with pytest.raises(ValueError, match="alias"):
        mla_write_locations(
            table[:, 0].contiguous(),
            table,
            page_size=64,
            q_len_per_req=1,
            batch_size=4,
            out=table.view(-1),
        )


@pytest.mark.parametrize("batch_size", [128, 256])
def test_more_locations_than_one_program_block(batch_size):
    """A production batch writes more locations than a block covers, so the
    grid's later programs have to carry their own range."""
    seq_lens, table = _inputs(batch_size, 16, 64, seed=batch_size)
    got = mla_write_locations(
        seq_lens, table, page_size=64, q_len_per_req=4, batch_size=batch_size
    )
    want = _reference(
        seq_lens, table, page_size=64, q_len_per_req=4, batch_size=batch_size
    )
    assert torch.equal(got, want)

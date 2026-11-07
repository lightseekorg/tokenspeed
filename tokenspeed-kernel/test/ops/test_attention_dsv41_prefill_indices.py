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

"""Sparse prefill attention indices match the separate PyTorch construction."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.attention import dsv41
from tokenspeed_kernel.ops.attention.dsv41.triton import _prefill_indices_kernel
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")


def _reference(swa, logical, slots, rows, base):
    selected, live = logical[rows], slots[rows]
    history = torch.where(
        (selected >= 0) & (live >= 0),
        selected.to(torch.int64) + base[:, None].to(torch.int64),
        -1,
    )
    return torch.cat((swa.to(torch.int64), history), dim=-1).to(torch.int32)


def _inputs(tokens, queries, width, seed):
    torch.manual_seed(seed)
    # A wider parent makes the row stride differ from the row width.
    logical = torch.randint(-1, 4096, (tokens, width + 16), device="cuda")
    logical = logical.to(torch.int32)[:, :width]
    slots = torch.randint(-3, 1 << 20, (tokens, width), device="cuda")
    swa = torch.randint(-1, 9000, (queries, 128), device="cuda", dtype=torch.int32)
    base = torch.randint(0, 1 << 22, (queries,), device="cuda", dtype=torch.int32)
    return swa, logical, slots, base


@pytest.mark.parametrize("queries", [1, 7, 8, 13, 300])
@pytest.mark.parametrize("width", [512, 96])
def test_prefill_indices_slice_rows(queries, width):
    swa, logical, slots, base = _inputs(queries + 11, queries, width, queries)
    rows = slice(5, 5 + queries)
    got = dsv41.prefill_indices(swa, logical, slots, rows, base)
    assert got.dtype == torch.int32 and got.is_contiguous()
    torch.testing.assert_close(
        got, _reference(swa, logical, slots, rows, base), rtol=0, atol=0
    )


@pytest.mark.parametrize("queries", [1, 9, 257])
def test_prefill_indices_reordered_rows(queries):
    swa, logical, slots, base = _inputs(400, queries, 512, 7)
    rows = torch.randperm(400, device="cuda")[:queries]
    got = dsv41.prefill_indices(swa, logical, slots, rows, base)
    torch.testing.assert_close(
        got, _reference(swa, logical, slots, rows, base), rtol=0, atol=0
    )


def test_prefill_indices_cpu_matches_cuda():
    swa, logical, slots, base = _inputs(50, 20, 512, 3)
    rows = torch.arange(30, 10, -1, device="cuda")
    want = dsv41.prefill_indices(swa, logical, slots, rows, base)
    got = dsv41.prefill_indices(
        swa.cpu(), logical.cpu(), slots.cpu(), rows.cpu(), base.cpu()
    )
    torch.testing.assert_close(got, want.cpu(), rtol=0, atol=0)


def test_prefill_indices_empty():
    swa, logical, slots, base = _inputs(4, 0, 512, 1)
    got = dsv41.prefill_indices(swa, logical, slots, slice(0, 0), base)
    assert got.shape == (0, 640)


def test_prefill_indices_rows_beyond_int32_offsets():
    # Row * stride of the last logical rows exceeds 2**31 elements; only those
    # rows are written, the rest stays uninitialized and is never read.
    tokens = (1 << 31) // 512 + 8
    if torch.cuda.get_device_properties(0).total_memory < 3 * tokens * 512 * 4:
        pytest.skip("needs two 8.6 GB integer tables")
    logical = torch.empty((tokens, 512), device="cuda", dtype=torch.int32)
    slots = torch.empty_like(logical)
    rows = torch.tensor([tokens - 1, tokens - 2], device="cuda")
    torch.manual_seed(5)
    logical[rows] = torch.randint(-1, 4096, (2, 512), device="cuda").to(torch.int32)
    slots[rows] = torch.randint(-1, 9, (2, 512), device="cuda").to(torch.int32)
    swa = torch.randint(-1, 99, (2, 128), device="cuda", dtype=torch.int32)
    base = torch.tensor([7, 70_000], device="cuda", dtype=torch.int32)
    got = dsv41.prefill_indices(swa, logical, slots, rows, base)
    torch.testing.assert_close(
        got, _reference(swa, logical, slots, rows, base), rtol=0, atol=0
    )
    del logical, slots
    torch.cuda.empty_cache()


def test_prefill_indices_no_recompile_across_steps():
    # Query counts, row starts and strides change every step; neither may
    # specialize the kernel once both row forms have compiled.
    for has_rows in (False, True):
        swa, logical, slots, base = _inputs(64, 16, 512, 0)
        rows = torch.arange(16, device="cuda") if has_rows else slice(0, 16)
        dsv41.prefill_indices(swa, logical, slots, rows, base)
    with assert_no_triton_compile(_prefill_indices_kernel):
        for queries, start in ((1, 3), (17, 0), (33, 31), (250, 5)):
            swa, logical, slots, base = _inputs(start + queries + 2, queries, 512, 1)
            dsv41.prefill_indices(
                swa, logical, slots, slice(start, start + queries), base
            )
            dsv41.prefill_indices(
                swa,
                logical,
                slots,
                torch.arange(start, start + queries, device="cuda"),
                base,
            )

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

"""Block-end lengths reach every request row and entry in one launch."""

import itertools
from test.utils import assert_no_triton_compile

import pytest
import torch
from tokenspeed_kernel.ops.metadata import broadcast_block_lengths
from tokenspeed_kernel.ops.metadata.block_lengths import (
    _broadcast_block_lengths_kernel,
)

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


def _write(lengths, low, high, expansion, dtype, causal=False):
    rows = torch.full((lengths.numel(),), -7, dtype=dtype, device=lengths.device)
    entries = torch.full(
        (lengths.numel(), expansion), -7, dtype=dtype, device=lengths.device
    )
    broadcast_block_lengths(
        lengths, low, high, causal=causal, rows_out=rows, entries_out=entries
    )
    return rows, entries


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("expansion", [1, 8, 13])
def test_cuda_matches_the_clamp_and_copy(expansion, dtype):
    lengths = torch.tensor([0, 3, 8, 9, 500, 4097, 4096, 1], dtype=dtype)
    expected = lengths.clamp(8, 4096)
    for device in ("cpu", "cuda"):
        rows, entries = _write(lengths.to(device), 8, 4096, expansion, dtype)
        assert torch.equal(rows.cpu(), expected), device
        assert torch.equal(entries.cpu(), expected[:, None].expand(-1, expansion))


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("expansion", [1, 8, 13])
def test_a_causal_block_backs_each_entry_off_by_its_distance_to_the_end(
    expansion, dtype
):
    # The low bound is the expansion, as for a block drafter: every entry keeps a token.
    lengths = torch.tensor([0, 3, 13, 14, 500, 4097, 4096, 1], dtype=dtype)
    expected = lengths.clamp(13, 4096)
    backoff = torch.arange(expansion - 1, -1, -1, dtype=dtype)
    for device in ("cpu", "cuda"):
        rows, entries = _write(lengths.to(device), 13, 4096, expansion, dtype, True)
        assert torch.equal(rows.cpu(), expected), device
        assert torch.equal(entries.cpu(), expected[:, None] - backoff), device


def test_batches_and_bounds_reuse_one_kernel():
    lengths = torch.arange(300, dtype=torch.int32, device="cuda") * 7
    _write(lengths, 8, 1000, 8, torch.int32)
    with assert_no_triton_compile(_broadcast_block_lengths_kernel):
        for batch, (low, high) in itertools.product(
            (1, 3, 128, 129, 300), ((8, 1000), (2, 64))
        ):
            rows, entries = _write(lengths[:batch], low, high, 8, torch.int32)
            expected = lengths[:batch].clamp(low, high)
            assert torch.equal(rows, expected)
            assert torch.equal(entries, expected[:, None].expand(-1, 8))


def test_misshaped_outputs_are_refused():
    lengths = torch.ones(4, dtype=torch.int32)
    with pytest.raises(ValueError, match="entries_out must hold"):
        broadcast_block_lengths(
            lengths,
            1,
            2,
            causal=False,
            rows_out=torch.empty(4, dtype=torch.int32),
            entries_out=torch.empty(4, 8, dtype=torch.int64),
        )
    with pytest.raises(ValueError, match="low 3 exceeds high 2"):
        _write(lengths, 3, 2, 8, torch.int32)
    with pytest.raises(ValueError, match="lengths must be dense"):
        _write(torch.ones(4, 2, dtype=torch.int32)[:, 0], 1, 2, 8, torch.int32)
    with pytest.raises(ValueError, match="colocated"):
        broadcast_block_lengths(
            lengths.cuda(),
            1,
            2,
            causal=False,
            rows_out=torch.empty(4, dtype=torch.int32),
            entries_out=torch.empty(4, 8, dtype=torch.int32),
        )

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

"""EAGLE-4 candidate head's gather-dot kernel."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.sampling.triton.candidate_head import (
    _gather_dot_kernel,
    gather_dot,
)
from utils import assert_no_triton_compile

# The vocab id of the weight shard's first row (rank 1 of a TP4 248320 vocab).
_ID_OFFSET = 62080


def _gather_dot_problem(rows: int, seed: int, device: str):
    """An EAGLE-4 candidate head shape: 128 candidates of a 5120-wide head."""
    gen = torch.Generator().manual_seed(seed)
    weight = torch.randn(4096, 5120, generator=gen).to(device, torch.bfloat16)
    hidden = torch.randn(rows, 5120, generator=gen).to(device, torch.bfloat16)
    ids = torch.randint(0, 4096, (rows, 128), generator=gen).to(device) + _ID_OFFSET
    return weight, ids, hidden


@pytest.mark.parametrize("rows", [1, 32, 64])
def test_gather_dot_matches_reference(rows: int, device: str) -> None:
    weight, ids, hidden = _gather_dot_problem(rows, rows, device)
    out = gather_dot(weight, ids, hidden, _ID_OFFSET)
    ref = torch.einsum("bkh,bh->bk", weight[ids - _ID_OFFSET].float(), hidden.float())
    # Computed in fp32: one bf16 rounding of the fp32 dot.
    torch.testing.assert_close(out.float(), ref, rtol=2**-8, atol=1e-2)


def test_gather_dot_small_row_tiles_agree_and_compile_once(device: str) -> None:
    """Every tile config up to 32 rows reduces along H alike: the same rows
    score bit for bit whether launched with 2 (BLOCK_K 2) or 8 (BLOCK_K 1).
    The row count picks one of three static tile configs, never a compile."""
    weight, ids, hidden = _gather_dot_problem(64, 5, device)
    wide = gather_dot(weight, ids[:8], hidden[:8], _ID_OFFSET)
    narrow = gather_dot(weight, ids[:2], hidden[:2], _ID_OFFSET)
    torch.testing.assert_close(narrow, wide[:2], rtol=0, atol=0)
    gather_dot(weight, ids, hidden, _ID_OFFSET)  # The many-row config.
    with assert_no_triton_compile(_gather_dot_kernel):
        for rows in (1, 3, 17, 32, 33, 64):
            gather_dot(weight, ids[:rows], hidden[:rows], _ID_OFFSET)
        gather_dot(weight, ids - _ID_OFFSET, hidden, 0)  # Rank 0's offset.

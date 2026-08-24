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

"""The fused payload capture must land exactly where three copy_ calls did.

This stages what a commit replays, so a misplaced row silently replays another
request's projection: the comparison is bitwise, and the buffer outside the
copied region has to be untouched.
"""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from tokenspeed_kernel.ops.attention.triton.capture_payload import (  # noqa: E402
    capture_replay_payload,
)

WIDTHS = (2304, 128, 6)


def _stacks(layers, rows, widths, pad=0, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    src = [
        torch.randn(rows + 3, w, device="cuda", dtype=torch.bfloat16, generator=g)
        for w in widths
    ]
    dst = [
        torch.randn(
            layers, rows, w + pad, device="cuda", dtype=torch.bfloat16, generator=g
        )
        for w in widths
    ]
    return src, dst


@pytest.mark.parametrize("rows", [1, 8, 128, 1024])
@pytest.mark.parametrize("pad", [0, 16])
def test_matches_three_copies(rows, pad):
    """Wide windows also cover the grid: 1024 rows spans several programs for
    the 2304-wide payload while the 6-wide one masks all but the first."""
    src, dst = _stacks(4, rows, WIDTHS, pad=pad, seed=rows)
    want = [d.clone() for d in dst]
    for s, w, width in zip(src, want, WIDTHS):
        w[2][:rows, :width].copy_(s[:rows])

    capture_replay_payload(
        tuple(s[:rows] for s in src),
        tuple(d[2][:rows, :width] for d, width in zip(dst, WIDTHS)),
        rows,
    )
    for got, expected in zip(dst, want):
        assert torch.equal(got, expected)


def test_nothing_outside_the_written_rows_moves():
    """The row belongs to one layer inside a stack every layer shares, so a
    stray write lands in another layer's replay payload."""
    rows, capacity = 8, 24
    src, dst = _stacks(4, capacity, WIDTHS, pad=16, seed=5)
    before = [d.clone() for d in dst]
    capture_replay_payload(
        tuple(s[:rows] for s in src),
        tuple(d[1][:rows, :width] for d, width in zip(dst, WIDTHS)),
        rows,
    )
    for got, was, width in zip(dst, before, WIDTHS):
        for layer in (0, 2, 3):
            assert torch.equal(got[layer], was[layer])
        assert torch.equal(got[1][:, width:], was[1][:, width:])
        assert torch.equal(got[1][rows:], was[1][rows:])


def test_a_source_row_inside_a_wider_projection():
    """The projection hands column slices, so the source row stride exceeds
    its width; the kernel must follow that stride, not assume packing."""
    rows = 8
    wide = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16)
    src = (wide[:, :2304], wide[:, 2304:2432], wide[:, 2432:2438])
    dst = [torch.zeros(rows, w, device="cuda", dtype=torch.bfloat16) for w in WIDTHS]
    want = [s.clone() for s in src]
    capture_replay_payload(src, tuple(dst), rows)
    for got, expected in zip(dst, want):
        assert torch.equal(got, expected)


def test_layouts_the_kernel_cannot_address_are_rejected():
    """Each of these silently read or wrote the wrong elements before the
    wrapper checked for it."""
    rows = 8
    src, _ = _stacks(1, rows, WIDTHS, seed=2)
    dst = [torch.empty(rows, w, device="cuda", dtype=torch.bfloat16) for w in WIDTHS]

    wide = torch.randn(rows, 8192, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="dense inner"):
        capture_replay_payload(
            (wide[:, ::2][:, :2304], wide[:, :128], wide[:, :6]), tuple(dst), rows
        )

    small = [
        torch.empty(rows, w - 1, device="cuda", dtype=torch.bfloat16) for w in WIDTHS
    ]
    with pytest.raises(ValueError, match="cannot hold"):
        capture_replay_payload(tuple(s[:rows] for s in src), tuple(small), rows)

    short = [
        torch.randn(rows - 1, w, device="cuda", dtype=torch.bfloat16) for w in WIDTHS
    ]
    with pytest.raises(ValueError, match="need 8"):
        capture_replay_payload(tuple(short), tuple(dst), rows)

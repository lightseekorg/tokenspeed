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

"""The one-launch stopped-row masking equals the torch op chain."""

import pytest
import torch
from tokenspeed_kernel.ops.metadata import mask_stopped_state_tables_
from tokenspeed_kernel.ops.metadata.state_tables import (
    _mask_stopped_state_tables_kernel,
)
from utils import assert_no_triton_compile

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


@pytest.mark.parametrize("widths", [(2,), (2, 2, 2), (2, 70), (2, 2, 2, 2)])
def test_stopped_rows_become_padding(widths):
    g = torch.Generator().manual_seed(0)
    batch, pool = 5, 9
    stopped = torch.rand(pool, generator=g) < 0.5
    stopped[3] = True
    stopped[0] = False
    req_pool_indices = torch.tensor([3, 0, 7, 1, 8], dtype=torch.int64)
    # Strided rows, as views over one packed table upload.
    packed = torch.randint(0, 1000, (batch, sum(widths) + 3), generator=g).int()
    packed = packed.cuda()
    tables, start = [], 0
    for w in widths:
        tables.append(packed[:, start : start + w])
        start += w
    want = [
        torch.where(stopped[req_pool_indices][:, None], -1, t.cpu()) for t in tables
    ]

    rest = packed[:, start:].clone()

    mask_stopped_state_tables_(tables, stopped.cuda(), req_pool_indices.cuda())

    for t, w in zip(tables, want):
        assert torch.equal(t.cpu(), w)
    # The masking writes only the tables' columns of the packed upload.
    assert torch.equal(packed[:, start:], rest)


def test_no_tables():
    stopped = torch.zeros(1, dtype=torch.bool)
    mask_stopped_state_tables_([], stopped, torch.zeros(0, dtype=torch.int64))


def _mask_packed(batch, widths, extra):
    """Mask state tables that are views of one packed upload of ``extra``
    more columns; return whether the result is the torch reference."""
    g = torch.Generator().manual_seed(batch)
    stopped = torch.rand(9, generator=g) < 0.5
    req_pool_indices = torch.randint(0, 9, (batch,), generator=g)
    packed = torch.randint(0, 1000, (batch, extra + sum(widths)), generator=g).int()
    packed = packed.cuda()
    tables, start = [], extra
    for w in widths:
        tables.append(packed[:, start : start + w])
        start += w
    want = [
        torch.where(stopped[req_pool_indices][:, None], -1, t.cpu()) for t in tables
    ]
    mask_stopped_state_tables_(tables, stopped.cuda(), req_pool_indices.cuda())
    return all(torch.equal(t.cpu(), w) for t, w in zip(tables, want))


def test_per_step_table_geometry_does_not_recompile():
    # The packed upload's width (the row stride), the tables' start columns
    # and the batch change from step to step while serving.
    assert _mask_packed(3, (2, 2), 5)
    with assert_no_triton_compile(_mask_stopped_state_tables_kernel):
        for batch, widths, extra in (
            (1, (2, 2), 12),
            (16, (2, 2), 44),
            (7, (1, 16), 3),
            (5, (32, 2), 30),
            (300, (2, 70), 61),
        ):
            assert _mask_packed(batch, widths, extra)


def test_more_than_four_groups_are_refused():
    tables = [torch.zeros(1, 2, dtype=torch.int32, device="cuda")] * 5
    stopped = torch.zeros(1, dtype=torch.bool, device="cuda")
    with pytest.raises(ValueError, match="up to 4 state groups"):
        mask_stopped_state_tables_(
            tables, stopped, torch.zeros(1, dtype=torch.int64, device="cuda")
        )

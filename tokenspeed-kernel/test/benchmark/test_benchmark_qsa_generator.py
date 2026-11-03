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

from __future__ import annotations

import tokenspeed_kernel.benchmark.generators.qsa as qsa_generator
import torch


def test_selected_slots_follow_each_rows_prefix() -> None:
    positions = torch.tensor([3, 5, 9, 9])
    request_index = torch.tensor([0, 0, 1, 1])
    slots = qsa_generator.qsa_selected_slots(
        positions,
        request_index,
        region_slots=16,
        width=6,
        generator=torch.Generator().manual_seed(0),
    )

    assert slots.shape == (4, 6) and slots.dtype == torch.int32
    for row, (position, request) in enumerate(zip(positions, request_index)):
        valid = slots[row][slots[row] >= 0]
        start = 2 + int(request) * 16
        assert valid.numel() == min(int(position), 6)
        assert torch.all(slots[row][valid.numel() :] == -1)
        assert torch.all(valid[1:] > valid[:-1])
        assert torch.all((valid >= start) & (valid < start + int(position)))
    # A row whose prefix fits the width selects every slot of it.
    assert slots[0].tolist() == [2, 3, 4, -1, -1, -1]

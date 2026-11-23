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

import pytest

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.moe.source_rows import MoESourceRows


@pytest.mark.parametrize(
    "padded,live,expected",
    [
        (16, 1, [1] + [0] * 15),
        (16, 3, [2, 1] + [0] * 14),
        (16, 0, [0] * 16),
        (11, 5, [2, 2, 1] + [0] * 13),
    ],
)
def test_decode_metadata_uses_captured_rs_offsets(padded, live, expected):
    for rank in (0, 8):
        mapping = Mapping(
            rank=rank,
            world_size=16,
            attn_tp_size=8,
            attn_dp_size=2,
            moe_tp_size=1,
            moe_ep_size=16,
        )
        rows = MoESourceRows(mapping, device="cpu", max_rows_per_rank=64)
        address = rows.source_token_counts.data_ptr()
        for counts, wanted in (
            ([live] * 8 + [0] * 8, expected),
            ([0] * 8 + [live] * 8, expected[8:] + expected[:8]),
        ):
            rows.upload_scattered(padded, counts)
            assert rows.source_token_counts.tolist() == wanted
            assert rows.source_token_counts.data_ptr() == address

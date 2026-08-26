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

from tokenspeed.runtime.layers.attention.chunk import (
    build_dcp_compact_reconstruction_plan,
)


def test_compact_dcp_prefix_plan_reconstructs_request_major_rows() -> None:
    starts = [3, 10, 17]
    lengths = [9, 5, 14]
    dcp_size = 4
    plans = [
        build_dcp_compact_reconstruction_plan(starts, lengths, dcp_size, rank)
        for rank in range(dcp_size)
    ]
    padded = plans[0][1]
    assert all(plan[1] == padded for plan in plans)
    assert sum(
        plans[rank][0][req] for rank in range(dcp_size) for req in range(3)
    ) == sum(lengths)

    gathered = [-1] * (dcp_size * padded)
    for rank, (counts, _, _) in enumerate(plans):
        cursor = rank * padded
        for start, length, count in zip(starts, lengths, counts):
            owned = [
                pos for pos in range(start, start + length) if pos % dcp_size == rank
            ]
            assert len(owned) == count
            gathered[cursor : cursor + count] = owned
            cursor += count

    reconstruction = plans[0][2]
    restored = [gathered[index] for index in reconstruction]
    expected = [
        pos
        for start, length in zip(starts, lengths)
        for pos in range(start, start + length)
    ]
    assert restored == expected

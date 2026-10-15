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

import sys
from pathlib import Path

import pytest
import torch

from tokenspeed.runtime.layers.attention import query_partition as implementation
from tokenspeed.runtime.layers.attention.query_partition import InterleavedQueryPlan

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")


@pytest.mark.parametrize("total", [0, 1, 3, 9, 1025, 2051])
@pytest.mark.parametrize("size", [1, 2, 8])
def test_request_spans_preserve_global_row_ownership(total, size):
    boundaries = [0, total // 3, total // 2, total]
    owned = []
    for rank in range(size):
        plan = InterleavedQueryPlan(total_rows=total, rank=rank, size=size)
        rows = list(range(total))[plan.local_slice]
        assert len(rows) == plan.local_rows
        owned.extend(rows)
        reconstructed = []
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
            low, high = plan.local_bounds(start, stop)
            assert rows[low:high] == [r for r in range(start, stop) if r % size == rank]
            reconstructed.extend(rows[low:high])
        assert reconstructed == rows
    assert sorted(owned) == list(range(total))


@pytest.mark.parametrize("total", [0, 1, 3, 9, 1025])
@pytest.mark.parametrize("columns", [3, 512])
def test_gather_restores_rows_and_lengths_including_empty_ranks(
    total, columns, monkeypatch
):
    size = 8
    expected = torch.arange(total * columns, dtype=torch.int32).reshape(total, columns)
    lengths = torch.arange(total, dtype=torch.int32) % (columns + 1)
    capacity = (total + size - 1) // size
    wire_columns = ((columns + 4) // 4) * 4
    for rank in range(size):
        plan = InterleavedQueryPlan(total_rows=total, rank=rank, size=size)
        calls = []

        def gather(output, value, group, backend):
            assert group == tuple(range(size))
            assert backend is None
            assert value.shape[1] * value.element_size() % 16 == 0
            torch.testing.assert_close(
                value[: plan.local_rows, :columns], expected[plan.local_slice]
            )
            torch.testing.assert_close(
                value[: plan.local_rows, columns], lengths[plan.local_slice]
            )
            calls.append(True)
            for peer in range(size):
                part = InterleavedQueryPlan(total_rows=total, rank=peer, size=size)
                padded = torch.full((capacity, wire_columns), -1, dtype=torch.int32)
                padded[: part.local_rows, :columns] = expected[part.local_slice]
                padded[: part.local_rows, columns] = lengths[part.local_slice]
                output[peer * capacity : (peer + 1) * capacity].copy_(padded)

        monkeypatch.setattr(implementation, "all_gather_single", gather)
        rows, lens = plan.gather_selection(
            expected[plan.local_slice], lengths[plan.local_slice], tuple(range(size))
        )
        torch.testing.assert_close(rows, expected, atol=0, rtol=0)
        torch.testing.assert_close(lens, lengths, atol=0, rtol=0)
        assert len(calls) == int(total > 0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

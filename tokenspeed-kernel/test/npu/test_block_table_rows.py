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

"""Exact NPU parity for the shared Triton compact-row expansion."""

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
if not torch_npu.npu.is_available():
    pytest.skip("NPU unavailable", allow_module_level=True)

from tokenspeed_kernel.numerics.torch.metadata import torch_expand_block_table_rows
from tokenspeed_kernel.ops.metadata import expand_block_table_rows


@pytest.mark.parametrize("value_offset", [0, 1, 2, 3])
@pytest.mark.parametrize(
    "shapes",
    [
        [],
        [(0, 0, 0)],
        [(5, 0, 0)],
        [(5, 5, 5)],
        [(5, 2, 2)],
        [(5, 0, 5)],
        [(3, 0, 0), (3, 3, 3), (3, 1, 1)],
        [(1023, 1000, 1023), (1024, 0, 1024), (1025, 1000, 1025)],
        [(200000, 199997, 200000), (3, 1, 2)],
        [
            (width, width - span, width)
            for width, span in [(2080, 1), (1040, 2), (2080, 3), (66560, 1)] * 16
        ],
    ],
)
def test_npu_expansion_matches_reference(shapes, value_offset):
    rows = []
    offset = source = 0
    for width, start, end in shapes:
        rows.append([offset, width, start, end, source])
        offset += width
        source += end - start
    rows_cpu = torch.tensor(rows, dtype=torch.int32).reshape(-1, 5)
    pool = torch.arange(1, source + value_offset + 1, dtype=torch.int32)
    values_cpu = pool[value_offset:]
    expected = torch.empty(offset, dtype=torch.int32)
    torch_expand_block_table_rows(rows_cpu, values_cpu, expected)
    # Preserve the pointer's alignment class when moving to the device.
    values = pool.npu()[value_offset:]
    out = torch.full((offset,), 0x5A5A5A5A, dtype=torch.int32, device="npu")
    expand_block_table_rows(rows_cpu.npu(), values, out)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def test_npu_expansion_orders_async_uploads_on_current_stream():
    stream = torch.npu.Stream()
    outputs = []
    expected = []
    with torch.npu.stream(stream):
        for step in range(32):
            width = 2049 + step
            rows = torch.tensor(
                [[0, width, width - 3, width, 0], [width, 5, 1, 4, 3]],
                dtype=torch.int32,
                pin_memory=True,
            )
            values = torch.arange(
                step * 6 + 1, step * 6 + 7, dtype=torch.int32, pin_memory=True
            )
            reference = torch.empty(width + 5, dtype=torch.int32)
            torch_expand_block_table_rows(rows, values, reference)
            out = torch.empty(width + 5, dtype=torch.int32, device="npu")
            expand_block_table_rows(
                rows.to("npu", non_blocking=True),
                values.to("npu", non_blocking=True),
                out,
            )
            # The consumer must follow the expansion on this same stream.
            outputs.append(out + step)
            expected.append(reference + step)
        done = stream.record_event()
    done.synchronize()
    for actual, reference in zip(outputs, expected, strict=True):
        torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)

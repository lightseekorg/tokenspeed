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

"""Runtime prefill/decode positions use the platform Triton compiler."""

import unittest

import torch
import torch_npu  # noqa: F401

from tokenspeed.runtime.execution.cache_loc_kernel import fused_decode_input_prep
from tokenspeed.runtime.execution.forward_batch_info import compute_position_triton


class PositionTest(unittest.TestCase):
    def test_prefill_offsets_prefixes_and_output_tail(self):
        lengths = torch.tensor([2, 513, 0, 3], dtype=torch.int32, device="npu")
        prefixes = torch.tensor([7, 1000, 4, 9], dtype=torch.int32, device="npu")
        for has_prefix in (False, True):
            out = torch.full((523,), -77, dtype=torch.int64, device="npu")
            got, starts = compute_position_triton(
                prefixes if has_prefix else prefixes[:0], lengths, 518, out=out
            )
            expected = torch.cat(
                [
                    torch.arange(n, dtype=torch.int64) + (p if has_prefix else 0)
                    for n, p in zip([2, 513, 0, 3], [7, 1000, 4, 9])
                ]
            )
            torch.testing.assert_close(got[:518].cpu(), expected)
            self.assertEqual(starts.cpu().tolist(), [0, 2, 515, 515])
            self.assertEqual(out[518:].cpu().tolist(), [-77] * 5)

    def test_full_batch_short_rows_do_not_overwrite_neighbors(self):
        for batch in (8, 32, 64):
            ids = torch.randperm(batch, dtype=torch.int32, device="npu")
            cache = torch.arange(batch, dtype=torch.int64, device="npu") * 103
            lens = torch.ones(batch, dtype=torch.int32, device="npu")
            expected = cache[ids.long()].cpu()
            for repeat in range(10):
                positions = torch.full((batch,), -999, dtype=torch.int64, device="npu")
                lengths = torch.full((batch,), -999, dtype=torch.int32, device="npu")
                fused_decode_input_prep(positions, lengths, ids, cache, 1)
                torch.testing.assert_close(positions.cpu(), expected)
                torch.testing.assert_close(lengths.cpu().long(), expected + 1)
                out, starts = compute_position_triton(cache, lens, batch)
                torch.testing.assert_close(out.cpu(), cache.cpu())
                torch.testing.assert_close(starts.cpu().long(), torch.arange(batch))

    def test_decode_pool_permutation_and_width(self):
        indices = torch.tensor([3, 1, 4], dtype=torch.int32, device="npu")
        cache = torch.tensor([0, 1001, 7, 23, 9000], dtype=torch.int64, device="npu")
        for width in (1, 3, 129):
            positions = torch.empty(3 * width, dtype=torch.int64, device="npu")
            lengths = torch.empty(3, dtype=torch.int32, device="npu")
            fused_decode_input_prep(positions, lengths, indices, cache, width)
            expected = torch.cat([torch.arange(width) + n for n in (23, 1001, 9000)])
            torch.testing.assert_close(positions.cpu(), expected)
            self.assertEqual(
                lengths.cpu().tolist(), [23 + width, 1001 + width, 9000 + width]
            )


if __name__ == "__main__":
    unittest.main()
